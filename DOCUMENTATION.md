# Inference Router Validation — Technical Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Component Reference](#component-reference)
4. [Statistical Methodology](#statistical-methodology)
5. [Configuration](#configuration)
6. [Execution](#execution)
7. [Results Interpretation](#results-interpretation)
8. [Limitations](#limitations)
9. [Troubleshooting](#troubleshooting)

---

## Overview

This project implements and validates a predicted-cost routing strategy for LLM inference load balancing. The routing algorithm uses Ridge regression to predict output token counts from prompts and routes requests predicted to exceed a threshold to the least-loaded worker, otherwise using round-robin assignment.

**Validation approach**: Controlled A/B testing on Mistral 7B inference with 500 Alpaca dataset prompts. 250 requests per strategy with independent t-tests and 95% confidence intervals.

**Results** (August 2026):
- Mean latency: 17% improvement (501ms, p=0.0152)
- P95 latency: 0.2% improvement (14ms, not statistically tested)
- Predictor R²: 0.114 (11% variance explained)

---

## Architecture

```
Validation Flow
===============

1. Load Dataset (500 Alpaca prompts)
2. Run Inference (Mistral 7B, collect output tokens + latency)
3. Split Data (80% train, 20% test)
4. Train Predictor (Ridge regression)
5. Evaluate Predictor (R², MAE on test set)
6. Run A/B Tests
   a. Smart Routing (250 requests, threshold-based)
   b. Round-Robin (250 requests, baseline)
7. Collect Latencies (per request)
8. Statistical Analysis
   a. Mean comparison (t-test, CI)
   b. Percentile summary (P50, P95, P99)
9. Save Results (JSON + visualization)
```

Component interfaces:

```
validation_notebook.ipynb orchestrates:
  - datasets.load_alpaca() -> List[prompt]
  - inference.generate(prompt) -> (text, tokens, latency_ms)
  - predictor.train(prompts, lengths)
  - predictor.predict(features) -> predicted_tokens
  - load_balancer.route_request(predicted_tokens, strategy) -> worker_id
  - statistics.calculate_statistics(smart_latencies, rr_latencies) -> Dict
```

---

## Component Reference

### datasets.py

Purpose: Load instruction-following prompts.

```python
def load_alpaca(num_prompts: int = 500) -> List[str]
def load_dataset(name: str, num_prompts: int) -> List[str]
```

Dataset characteristics (Alpaca):
- 52,000 total prompts from instruction-following dataset
- Diverse task types: classification, writing, analysis
- Prompt length: 20-2000 tokens typical
- Source: HuggingFace datasets

---

### inference.py

Purpose: Run Mistral 7B and measure latency.

```python
class VLLMInference:
    def __init__(self, model_id: str, **vllm_config)
    def generate(prompt: str) -> Tuple[str, int]
    # Returns: (output_text, num_output_tokens)
    # Latency tracked internally in milliseconds
```

vLLM configuration from config.yaml:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| gpu_memory_utilization | 0.9 | Allocate 90% of VRAM |
| batch_size | 1 | Single request per batch |
| max_model_len | 512 | Max input+output tokens |
| kv_cache_dtype | "fp8" | 8-bit KV cache quantization |
| cpu_offload_gb | 16 | Offload intermediates to CPU |
| enforce_eager | true | Disable PagedAttention optimization |

Tuning guidance:
- If OOM: reduce gpu_memory_utilization (0.9 -> 0.7 -> 0.5)
- If timeouts: increase vLLM startup timeout
- If high variance: reduce max_model_len or increase batch_size

---

### predictor.py

Purpose: Predict output token count before inference.

**PromptFeatureExtractor**: Extracts 8 features from prompt text

| Feature | Definition |
|---------|-----------|
| prompt_length | len(prompt) / 10000 |
| word_count | len(prompt.split()) / 2000 |
| vocabulary_entropy | Shannon entropy of word frequencies |
| question_density | count('?') / word_count |
| code_markers | count('```') + count('{') + count('}') |
| imperative_verbs | count of action verbs |
| punctuation_density | punctuation_count / word_count |
| uppercase_density | uppercase_chars / total_chars |

**OutputLengthPredictor**: Ridge regression model

```python
class OutputLengthPredictor:
    def train(prompts: List[str], output_lengths: List[int])
    def predict_single(features: np.ndarray) -> float
    def score(X: np.ndarray, y: np.ndarray) -> float  # R²
    def mae(X: np.ndarray, y: np.ndarray) -> float
```

Model details:
- Algorithm: Ridge regression (L2-regularized linear regression)
- Regularization: alpha = 1.0 (fixed)
- Train/test split: 80/20
- Rationale: Handles correlated features, stable on small data

Actual performance (August 2026):
- R² = 0.114 (11% variance explained)
- MAE = 83.1 tokens
- Assessment: Weak predictive signal; 89% variance unexplained

---

### load_balancer.py

Purpose: Assign requests to workers based on routing strategy.

```python
class SmartLoadBalancer:
    def route_request(prompt_id: int, predicted_tokens: float,
                      strategy: RoutingStrategy,
                      threshold: float = 500.0) -> int
    def record_completion(worker_id: int, latency: float,
                         success: bool = True)
```

Routing strategies:

1. **ROUND_ROBIN**: Cycle through workers sequentially
   - Fair distribution, ignores load and request properties

2. **LEAST_LOADED**: Route to worker with lowest load score
   - Load score = 0.7 * (active_requests / 10.0) + 0.3 * (avg_latency / 5000.0)
   - Reacts to current state

3. **PREDICTED_COST**: Threshold-based routing
   - If predicted_tokens > threshold: route to LEAST_LOADED
   - Else: route via ROUND_ROBIN

Worker state tracking:

```python
@dataclass
class WorkerMetrics:
    worker_id: int
    current_requests: int = 0      # Active jobs
    total_requests: int = 0        # Lifetime count
    total_latency: float = 0.0     # Sum of latencies (ms)
    error_count: int = 0
    
    @property
    def avg_latency(self) -> float:
        return total_latency / max(total_requests, 1)
    
    @property
    def load_score(self) -> float:
        return min(
            (current_requests / 10.0) * 0.7 + (avg_latency / 5000.0) * 0.3,
            1.0
        )
```

Note: Simulation uses 3 workers on single GPU. Scaling to distributed cluster is not validated.

---

### statistics.py

Purpose: Compute significance tests and confidence intervals.

```python
def confidence_interval(data: List[float], confidence: float = 0.95) 
    -> Tuple[float, float]:
    # 95% CI on mean using t-distribution
    # Accounts for sample size via degrees of freedom

def ttest_independent(group1: List[float], group2: List[float]) 
    -> Dict[str, float]:
    # H0: mean(group1) = mean(group2)
    # Returns: t_statistic, p_value, mean_diff, significant (1 if p<0.05)

def calculate_statistics(smart_latencies: List[float],
                        rr_latencies: List[float]) 
    -> Dict:
    # Comprehensive analysis: descriptive stats, percentiles,
    # t-test results, confidence intervals, comparison metrics
```

Test assumptions:
- Independence: Yes (each request is independent)
- Equal variances: Approximately (smart std=2272, RR std=2317)
- Normality: Approximately (right-skewed but large n helps via CLT)

Limitation: t-test operates on means. For percentile (P95, P99) comparisons, bootstrap resampling is required.

---

## Statistical Methodology

### Hypothesis Test

H0 (null): Mean latency is equal between smart routing and round-robin
H1 (alternative): Mean latencies differ

### Test Procedure

Independent two-sample t-test:

```
t = (mean_smart - mean_rr) / SE
SE = sqrt(s²_smart/n_smart + s²_rr/n_rr)

p_value = P(T > |t|) where T ~ t-distribution(n_smart + n_rr - 2 df)

Decision: Reject H0 if p_value < 0.05 (significance level α = 0.05)
```

### August 2026 Results

```
Smart routing:   n=250, mean=2446.6ms, std=2272.2ms
Round-robin:     n=250, mean=2947.7ms, std=2317.4ms
Difference:      -501.0ms

t-statistic:     -2.436
p-value:         0.0152
95% CI (smart):  [2163ms, 2730ms]

Conclusion: Reject H0. Mean latency difference is statistically 
significant at alpha=0.05.
```

### Percentile Analysis

P50, P95, P99 are reported for descriptive summary but are not statistically tested.

- P95 estimated from 12-13 samples (top 5% of n=250)
- 14ms difference (6056 vs 6070) is within noise
- Significance test for percentiles requires bootstrap resampling (not performed)

### Confidence Interval Interpretation

95% CI [2163ms, 2730ms] describes the range where the true mean latency for smart routing is expected to fall with 95% confidence. This is not the distribution of individual requests.

---

## Configuration

### Model

```yaml
model:
  name: "mistralai/Mistral-7B-Instruct-v0.1"
  max_tokens: 256
```

Selection rationale: 7B parameter size fits 24GB VRAM, instruction-tuned for Alpaca dataset, open weights ensure reproducibility.

### Dataset

```yaml
dataset:
  name: "alpaca"
  num_prompts: 500
```

500 prompts chosen for: sufficient sample size (n=250 per strategy), manageable runtime (2-3 hours), reasonable cloud cost ($1.50-2.00).

### vLLM

```yaml
vllm:
  gpu_memory_utilization: 0.9
  batch_size: 1
  max_model_len: 512
  kv_cache_dtype: "fp8"
  cpu_offload_gb: 16
  enforce_eager: true
```

### Predictor

```yaml
predictor:
  ridge_alpha: 1.0
```

L2 regularization strength (fixed). Higher alpha increases regularization (simpler model).

### A/B Test

```yaml
a_b_test:
  requests_per_strategy: 250

smart_routing:
  cost_threshold: 500
```

Threshold (500 tokens) is arbitrary. Optimal threshold depends on output-length distribution and load distribution.

---

## Execution

### Prerequisites

- GPU: 24GB+ VRAM (RTX 4090, A100, A5000, or equivalent)
- Python 3.10+
- Runtime: 2-3 hours
- Internet: For model and dataset downloads

### Local

```bash
git clone https://github.com/oladri-renuka/inference-router-validation.git
cd inference-router-validation
pip install -r requirements.txt
jupyter notebook validation_notebook.ipynb
```

Run notebook cells in order:
1. Import and configure
2. Load Alpaca dataset
3. Initialize vLLM
4. Collect inference data (500 prompts)
5. Train predictor
6. Run A/B tests
7. Calculate statistics
8. Generate plots

### RunPod

```bash
# Create RTX 4090 or A5000 instance
git clone https://github.com/oladri-renuka/inference-router-validation.git
cd inference-router-validation
pip install -r requirements.txt
jupyter notebook --ip=0.0.0.0 validation_notebook.ipynb
# Open browser at provided URL

# After completion
git add results/
git commit -m "validation results"
git push origin main
```

---

## Results Interpretation

### Mean Difference is Significant (p < 0.05)

Observation:
```
Smart mean:   2447ms
RR mean:      2948ms
Difference:   501ms (17% improvement)
p-value:      0.0152
```

Interpretation: The observed difference is unlikely under the null hypothesis. Statistical evidence that smart routing reduces mean latency.

Scope: Applies to Mistral 7B, Alpaca dataset, simulated 3-worker environment.

Does NOT prove:
- Generalization to other models (Llama, GPT, Phi)
- Generalization to other datasets or workloads
- Performance in real distributed cluster
- Production readiness

Action: Treat as promising but preliminary. Validation on additional models and datasets recommended.

### P95 is Unchanged

Observation:
```
Smart P95:   6056ms
RR P95:      6070ms
Difference:  14ms (0.2%)
```

Interpretation: Tail latency (P95) shows minimal change. 14ms difference is:
- Not statistically tested (no bootstrap resampling)
- Estimated from 12-13 samples (noisy)
- Likely within measurement variance

Implication: Routing addresses average-case latency, not worst-case. Not suitable for strict tail-latency SLAs.

Suitable for: Cost optimization, average response time improvement, best-effort services.

### No Significant Difference (p ≥ 0.05)

Observation:
```
Smart mean:   2650ms
RR mean:      2800ms
Difference:   150ms (5%)
p-value:      0.23
```

Interpretation: Cannot reject null hypothesis. 5% difference may be noise.

Likely causes:
1. Predictor too weak for threshold-based routing
2. Threshold not optimal
3. Insufficient load contention
4. Sample size insufficient

Remediation:
- Improve predictor (better features, more data)
- Optimize threshold via sensitivity analysis
- Test on larger worker pool
- Increase sample size

---

## Limitations

### Single Model, Single Dataset

Scope: Validation on Mistral 7B + Alpaca only.

Generalization risk: Results may not apply to other models or datasets.

Mitigation: Repeat validation on diverse models and datasets.

### Simulated Load Balancing

Setup: 3 workers simulated on single GPU.

Does not reflect:
- Real distributed cluster (network latency, GPU isolation)
- Scaling to 100s of workers
- Multi-tenant effects
- GPU resource contention

Scope limitation: Single-GPU routing simulation only.

### Weak Predictor

Predictor accuracy: R² = 0.114 (11% variance explained)

Implication: For the 500-token threshold to work, errors must not concentrate near the boundary. Missing validation: No confusion matrix analysis at 500-token threshold.

### P95/P99 Not Statistically Tested

Limitation: Percentiles reported but not tested for significance.

Statistical issue: t-test operates on means; percentile comparisons require bootstrap resampling.

Consequence: Claiming "P95 improved significantly" based on this p-value is incorrect.

### Random Seed Not Documented

Issue: No fixed seed in config or notebook.

Effect: Running validation twice produces slightly different results.

Current state: Not reproducible to bit-level.

### Sample Size for Tail Estimates

P95 estimation: From ~12-13 samples (top 5% of n=250).

Consequence: High variance in percentile estimates. 14ms difference is likely noise.

Improvement: Increase n to 1000+ per strategy, or use bootstrap CI for percentiles.

---

## Troubleshooting

### CUDA Out of Memory

Error: `RuntimeError: out_of_memory: CUDA out of memory. Tried to allocate X.XX GB`

Solutions (in order):
1. Reduce gpu_memory_utilization in config.yaml (0.9 -> 0.7 -> 0.5)
2. Reduce max_model_len (512 -> 256)
3. Reduce batch_size (already 1, cannot go lower)
4. Switch to smaller model (Phi-2 instead of Mistral 7B)

Verify: `nvidia-smi` to check GPU memory usage

### vLLM Initialization Timeout

Error: `ConnectionError: Failed to connect to vLLM engine`

Solutions:
1. Increase timeout in notebook
2. Check GPU availability: `nvidia-smi`
3. Restart Jupyter kernel

### Predictor R² Too Low

Observation: R² < 0.10

Investigation:
```python
# Check feature-to-target correlation
for i, feat_name in enumerate(feature_names):
    corr = np.corrcoef(features[:, i], output_lengths)[0, 1]
    print(f"{feat_name}: r = {corr:.3f}")
```

Possible causes:
- Features don't correlate with output length for this dataset
- Model temperature too high (inherent randomness)
- Output length determined by factors not in features

Mitigation: Accept weak predictor; report results honestly.

### Results Don't Reproduce

Observation: Different p-value, effect size from previous run.

Cause: No fixed random seed. Variance due to:
- Different random sampling in statistics
- Model version differences (vLLM updates)
- GPU state differences

Solution: Add `np.random.seed(42)` to notebook top cell.

### Cannot Push to GitHub

Error: `fatal: repository not found` or authentication failure

Solutions:
1. Verify remote: `git remote -v`
2. Add remote if missing: `git remote add origin https://...`
3. Configure credentials: `git config user.email` and `git config user.name`
4. Check SSH key or personal access token

---

**Last updated**: August 4, 2026
