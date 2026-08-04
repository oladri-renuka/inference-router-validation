# Validation Plan: Inference Router A/B Testing

## Objective

Determine whether predicted-cost routing provides measurable latency improvements over round-robin load balancing using **real inference data** and **rigorous statistical testing**.

## Methodology

### Phase 1: Predictor Accuracy Validation

**Goal**: Measure how well the Ridge regression model predicts output token counts.

**Process**:
1. Load 500 real prompts from Alpaca dataset
2. Run each through Mistral 7B inference (get actual output tokens)
3. Train Ridge regression predictor on 80% of data
4. Evaluate on held-out 20% test set
5. Calculate R² and MAE

**Success Criteria** (original plan):
- R² > 0.7 (predictor explains 70%+ of variance)
- MAE < 100 tokens (predictions within ~100 tokens)

**Actual Results**:
- ❌ R² = 0.114 (explains only 11% of variance — **FAILED**)
- ⚠️ MAE = 83.1 tokens (met, but weak R² undermines confidence)

### Phase 2: A/B Test Setup

**Controlled Variables**:
- Same 500 prompts for both strategies
- Same vLLM configuration (Mistral 7B)
- Same batch size and inference parameters
- Same worker pool (3 workers)

**Independent Variable**:
- Routing strategy (smart vs round-robin)

**Measured Variable**:
- Request latency (p50, p95, p99)

### Phase 3: A/B Test Execution

#### Strategy 1: Smart Routing
```
FOR each of 250 prompts:
  1. Extract prompt features
  2. Predict output length (Ridge model)
  3. IF predicted > 500 tokens:
       Route to LEAST_LOADED worker
     ELSE:
       Route via ROUND_ROBIN
  4. Send to worker, measure latency
  5. Record result
```

#### Strategy 2: Round-Robin (Baseline)
```
FOR each of 250 prompts:
  1. Route via simple ROUND_ROBIN
  2. Send to worker, measure latency
  3. Record result
```

**Why 250 requests per strategy?**
- Large enough for stable statistics
- Small enough to run in ~1 hour
- Good signal-to-noise ratio with 100-1500ms latency range

### Phase 4: Statistical Analysis

#### Primary Metric: Mean Latency
**Note**: While the original plan focused on P95, the actual significance test (t-test) was run on mean latency, not percentiles. P95/P99 are reported but not statistically validated.

**Rationale for mean**: 
- T-tests are parametric and assume normality
- Percentile-based comparisons need resampling (bootstrap)
- Mean is what got tested, so that's what gets the p-value

#### Hypothesis Test: Independent T-Test on Mean
**Null hypothesis (H0)**: Mean latencies are equal

**Test**:
```
t = (mean_smart - mean_baseline) / sqrt(var_smart/n_smart + var_baseline/n_baseline)
p-value = P(T > |t|) where T ~ t(n1+n2-2)
```

**Decision**:
- If p-value < 0.05: Reject null, conclude meaningful difference
- If p-value ≥ 0.05: Fail to reject null, no proven difference

#### Effect Size
- Report absolute difference in milliseconds
- Report confidence interval (e.g., "P95 improved by 50ms [30ms, 70ms]")

### Phase 5: Threshold Analysis

**Goal**: Find optimal routing threshold (currently hardcoded at 500 tokens)

**Process**:
1. Test thresholds: 200, 300, 500, 1000 tokens
2. For each threshold, measure smart-routing P95 latency
3. Compare to round-robin baseline
4. Select threshold with best P95 improvement

### Phase 6: Results & Interpretation

#### If Smart Routing Helps (p < 0.05) — **ACTUAL RESULTS**
```
Interpretation: We have statistical evidence that smart routing
reduces MEAN latency. The improvement is real, not due to chance.

Actual results: Mean improved 501ms (17% faster), p=0.0152 ✓
But P95 unchanged (+0.2%, p not validated, noise only)

Action: Load-balancing provides average-case benefit, not tail-case.
Not suitable for SLAs requiring tail-latency guarantees.
```

#### If No Difference (p ≥ 0.05)
```
Interpretation: We cannot prove smart routing helps with this
workload/model/threshold.

Possible reasons:
1. Predictor accuracy too low
2. Threshold not optimal
3. Workload doesn't have enough contention
4. Overhead cancels out benefits

Action: Debug predictor, optimize threshold, increase load
```

## Key Assumptions

1. **Mistral 7B output length correlates with compute cost**
   - Verified by: inference time ∝ output tokens in vLLM

2. **500-token threshold is reasonable**
   - Will be validated by threshold analysis phase

3. **Prompt distribution matters**
   - Using Alpaca (realistic instructions)
   - Not using only short or only long prompts

4. **Latency variation is meaningful**
   - Some requests should be faster than others
   - Otherwise routing strategy doesn't matter

## Potential Limitations

1. **Synthetic vs Real Inference**
   - Mistral 7B is real model
   - But runs on single machine, not distributed cluster
   - Scaling characteristics may differ in production

2. **Latency Variance**
   - Mistral has high variance (depends on output length, temperature)
   - May need larger sample sizes for stable estimates

3. **Model Specificity**
   - Results may differ with different LLMs
   - Optimal threshold likely model-specific

4. **Workload Distribution**
   - Alpaca is instruction-following
   - May not represent customer workload
   - Would need customer data for production validation

## Timeline

| Phase | Duration | Notes |
|-------|----------|-------|
| Setup (downloads) | 10-15 min | vLLM + Mistral 7B |
| Predictor training | 5 min | Ridge regression |
| Smart routing test | 30-40 min | 250 requests |
| Round-robin baseline | 30-40 min | 250 requests |
| Analysis | 10 min | Calculate stats, plots |
| **Total** | **1.5-2.5 hrs** | On RTX 4090 |

## Success Criteria

✅ **Complete**: All phases run without errors
✅ **Reproducible**: Can rerun and get similar results
✅ **Honest**: Report results regardless of outcome
✅ **Statistically Valid**: Use proper CI and hypothesis testing
✅ **Actionable**: Clear recommendation on whether to deploy

## Deliverables

1. `results/validation_results.json`
   - Predictor accuracy (R², MAE, features)
   - Smart routing latencies (all values)
   - Round-robin latencies (all values)
   - Statistical test results
   - Optimal threshold

2. `results/validation_report.txt`
   - Human-readable summary
   - Interpretation of results
   - Limitations and caveats

3. `results/plots.png`
   - Latency distribution plots (both strategies)
   - Threshold analysis chart
   - Feature importance bar chart

4. This notebook as reference implementation
