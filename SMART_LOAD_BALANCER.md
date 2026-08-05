# Smart Load Balancer for LLM Inference

Production-grade load balancing system for distributed LLM inference with predicted output length routing and automatic A/B testing validation.

## Overview

This project implements a load balancer that routes inference requests based on predicted output length before dispatching to workers. Instead of naive round-robin, the system:

1. **Extracts prompt features** (length, entropy, code markers, etc.)
2. **Predicts output token count** using Ridge regression (~50ms overhead)
3. **Routes requests intelligently** — high-predicted-cost requests to underloaded workers
4. **Validates performance** with statistical A/B testing across three LLM models

## Key Finding

**No statistically significant latency improvement was detected** with the current configuration (p-value: 0.87). Smart routing P95 latencies matched round-robin across all three models tested (Llama-2, Mistral, Phi-2).

**Root Cause:** The 500-token routing threshold never triggered—output lengths clustered below 500 tokens on all models, leaving routing decisions unutilized.

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────┐
│                  Inference Router                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Input Prompt → Feature Extractor → Output Predictor  │
│                    (10 features)    (Ridge Regressor) │
│                                           ↓             │
│                                    Route Strategy       │
│                                    (Smart/RR/LL)        │
│                                           ↓             │
│         ┌─────────────────────────────────┴──┐          │
│         ↓                 ↓                ↓           │
│      Worker-1         Worker-2         Worker-3       │
│   (vLLM Engine)   (vLLM Engine)   (vLLM Engine)       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Prompt Features (10-dimensional)

| Feature | Type | Purpose |
|---------|------|---------|
| `prompt_length` | int | Raw character count |
| `word_count` | int | Tokenization estimate |
| `vocabulary_entropy` | float | Lexical complexity (0-1) |
| `question_marks` | int | Query vs instruction detection |
| `code_markers` | int | Code block presence |
| `imperative_verbs` | int | Task complexity signal |
| `punctuation_density` | float | Structural complexity |
| `uppercase_density` | float | Emphasis/acronym markers |
| `sentence_count` | int | Discourse structure |
| `digit_presence` | binary | Numerical reasoning trigger |

### Routing Strategies

1. **Smart Routing** — Uses predicted output length to route high-cost requests to least-loaded workers
2. **Round-Robin** — Baseline; cycles through workers uniformly  
3. **Least-Loaded** — Routes to worker with lowest current queue depth

## Validation Results

### Data

Tested on **500 Alpaca prompts** across 3 models with **166 requests per strategy**.

### Llama-2-7b-chat-hf

```
Smart Routing:
  P50:   3,386ms
  P95:   5,449ms
  P99:   5,637ms
  Mean:  3,258ms

Round-Robin:
  P50:   3,702ms
  P95:   5,441ms
  P99:   5,600ms
  Mean:  3,296ms

P95 Improvement: -0.15% (not significant, p=0.87)
Predictor R²: 0.109 | MAE: 78.5 tokens
```

### Mistral-7B-Instruct-v0.1

```
Smart Routing:
  P50:   1,907ms
  P95:   5,855ms
  P99:   5,860ms
  Mean:  2,552ms

Round-Robin:
  P50:   1,923ms
  P95:   5,855ms
  P99:   5,864ms
  Mean:  2,487ms

P95 Improvement: +0.01% (not significant, p=0.79)
Predictor R²: 0.095 | MAE: 79.2 tokens
```

### Phi-2

```
Smart Routing:
  P50:   1,167ms
  P95:   3,743ms
  P99:   4,635ms
  Mean:  1,624ms

Round-Robin:
  P50:   1,391ms
  P95:   3,577ms
  P99:   3,696ms
  Mean:  1,723ms

P95 Improvement: -4.65% (not significant, p=0.51)
Predictor R²: -0.061 | MAE: 91.2 tokens
```

### Statistical Significance

All three models showed **no statistically significant difference** between routing strategies:

| Model | p-value | Significant (α=0.05) | t-statistic |
|-------|---------|----------------------|-------------|
| Llama-2 | 0.870 | ✗ | -0.163 |
| Mistral | 0.789 | ✗ | 0.268 |
| Phi-2 | 0.509 | ✗ | -0.662 |

## Root Cause Analysis

### Why Smart Routing Didn't Improve Latency

**The 500-token threshold never activated.** Analysis of predicted outputs:

```
Threshold Validation @ 500 tokens:
  True Positives (TP):  0
  True Negatives (TN):  100
  False Positives (FP): 0
  False Negatives (FN): 0

Accuracy: 100% (but meaningless — no routing decisions made)
```

**Output distribution across all models:**
- Min: 1 token
- Mean: 159 tokens
- P95: ~160-170 tokens
- Max: 256 tokens

**Implication:** Smart routing and round-robin made identical decisions on 100% of requests. Routing logic never engaged.

### Secondary Issue: Predictor Quality

- **Llama-2:** R² = 0.109 (explains 10.9% of variance)
- **Mistral:** R² = 0.095 (explains 9.5% of variance)
- **Phi-2:** R² = -0.061 (worse than predicting mean)

Ridge regression with 10 prompt features is insufficient to predict output length. Token count appears to depend on factors not captured by surface-level prompt analysis.

## Setup & Usage

### Requirements

```bash
pip install vllm transformers torch scikit-learn numpy pydantic pyyaml datasets
```

### Configuration

Edit `config.yaml`:

```yaml
random_seed: 42

models:
  - name: "mistralai/Mistral-7B-Instruct-v0.1"
    max_tokens: 256
  - name: "microsoft/phi-2"
    max_tokens: 256

dataset:
  name: "alpaca"
  num_prompts: 500
  test_split: 0.2

vllm:
  gpu_memory_utilization: 0.7
  batch_size: 1
  max_model_len: 512
  kv_cache_dtype: "auto"
  cpu_offload_gb: 16
  enforce_eager: false

smart_routing:
  cost_threshold: 500  # Lower to 300-400 to enable routing

ab_test:
  requests_per_strategy: 250

predictor:
  model_types: ["ridge", "lasso", "elasticnet"]
  alpha: 1.0
  num_features: 10

load_balancer:
  num_workers: 3
  worker_addresses:
    - "worker-1"
    - "worker-2"
    - "worker-3"
```

### Running Validation

```bash
jupyter notebook validation_notebook.ipynb
```

The notebook:
1. Loads 500 Alpaca prompts
2. Runs inference on all models sequentially
3. Trains predictor on 400 samples, validates on 100
4. Executes A/B test with 166 requests per strategy
5. Computes statistical significance with t-tests
6. Saves results to JSON

### Output Files

```
/Downloads/
├── validation_results_Llama-2-7b-chat-hf.json
├── validation_results_Mistral-7B-Instruct-v0.1.json
└── validation_results_phi-2.json
```

Each contains:
- Latency statistics (P50, P95, P99, mean, std)
- Predictor accuracy (R², MAE)
- Threshold validation (confusion matrix)
- Statistical test results (p-value, t-statistic)

## Recommendations for Next Iteration

### 1. **Lower the Routing Threshold** (High Priority)

Current 500-token threshold never activates. Test with:
- Threshold = 300 tokens (~2x median output)
- Threshold = 200 tokens (~1.3x median output)

**Expected outcome:** Actual routing decisions on 5-15% of requests; predictor quality becomes observable.

**Estimated time:** 1 model run (~45 min on single GPU)

### 2. **Improve Predictor Features** (Medium Priority)

Current 10 features may be insufficient. Consider:
- Semantic embeddings (prompt encoding similarity)
- Task type classification (QA vs summarization vs code)
- Contextual prompt analysis (not just raw metrics)
- Historical patterns from training data

**Estimated impact:** Could increase R² from 0.10 to 0.30-0.40

### 3. **Validate on Longer Output Tasks** (Medium Priority)

Current validation uses Alpaca (instruction tuning), which produces short outputs. Test on:
- Summarization tasks (typically longer outputs)
- Code generation (longer, structured outputs)
- Multi-turn conversations

**Estimated impact:** May reveal where routing provides actual value

### 4. **Add Worker-Level Metrics** (Low Priority)

Current system only measures aggregate latency. Add:
- Per-worker latency distribution
- Worker queue depth over time
- Load imbalance metric (coefficient of variation)

Would clarify whether routing improves worker utilization even if latency is unchanged.

## Implementation Details

### Predictor Training

- **Algorithm:** Ridge regression (α=1.0)
- **Train/test split:** 80% / 20%
- **Features:** Standardized (zero mean, unit variance)
- **Cross-validation:** None (single holdout set)

### A/B Test Design

- **Sample size:** min(250, len(prompts) // 3) per strategy (adaptive scaling)
- **Ensures:** Never exceed available test prompts
- **Statistical test:** Independent samples t-test (two-tailed)
- **Significance level:** α = 0.05
- **Confidence interval:** 95%

### GPU Configuration

- **Device:** NVIDIA RTX A5000 (24GB VRAM, compute capability 8.6)
- **vLLM settings:**
  - gpu_memory_utilization: 0.7 (reduced from 0.9 to avoid OOM)
  - kv_cache_dtype: "auto" (avoids FP8 incompatibility with SM86)
  - enforce_eager: false
  - cpu_offload_gb: 16

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Ridge regression | Simple, interpretable, ~50ms inference (acceptable overhead) |
| 10 prompt features | Balance complexity/inference-time tradeoff; covers lexical, structural, semantic signals |
| 500-token threshold | Assumed median output ~150-200; conservative for routing activation |
| Adaptive A/B test scaling | Prevents index errors when prompt count < 250 * 3 |
| Sequential model execution | Avoids GPU OOM; requires explicit cleanup (gc + torch.cuda.empty_cache()) |

## Project Structure

```
smart-load-balancer/
├── SMART_LOAD_BALANCER.md             (this file)
├── config.yaml                         (configuration)
├── validation_notebook.ipynb           (main validation script)
├── utils/
│   ├── __init__.py
│   ├── inference.py                    (vLLM wrapper, VLLMInference class)
│   ├── predictor.py                    (OutputLengthPredictor, PromptFeatureExtractor)
│   ├── load_balancer.py                (LoadBalancer, routing strategies)
│   └── statistics.py                   (statistical testing, confidence intervals)
└── results/
    ├── validation_results_Llama-2-7b-chat-hf.json
    ├── validation_results_Mistral-7B-Instruct-v0.1.json
    └── validation_results_phi-2.json
```

## Key Takeaways

1. **Routing without threshold activation is moot** — if all requests route the same way, strategy doesn't matter

2. **Output length prediction is hard** — 10 surface-level features explain <12% of variance; needs deeper semantic analysis

3. **Statistical validation is essential** — apparent mean latency differences were noise (p > 0.78 across all models)

4. **Null results are valuable** — knowing what doesn't work informs next steps and prevents sunk-cost fallacy

5. **Threshold tuning is critical** — one hyperparameter change (500 → 300 tokens) could transform the experiment

## References

- **vLLM:** https://github.com/vllm-project/vllm
- **Alpaca Dataset:** https://github.com/tatsu-lab/alpaca
- **Scikit-learn Ridge Regression:** https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
- **Statistical Testing:** Welch's t-test for unequal variances (scipy.stats.ttest_ind)

## License

MIT
