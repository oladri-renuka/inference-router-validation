# Smart Load Balancer for LLM Inference

A production-grade load balancing system for distributed LLM inference that routes requests based on predicted output length before dispatching to workers.

## Overview

This project implements intelligent request routing for LLM inference workloads. Instead of round-robin load balancing, the system predicts output token count from prompt features and routes accordingly: high-predicted-cost requests go to underloaded workers; low-cost requests distribute uniformly.

**Status**: Validation complete. No statistically significant latency improvement detected with current configuration (p > 0.50 across all models tested).

## Validation Results

| Metric | Llama-2 | Mistral | Phi-2 |
|--------|---------|---------|--------|
| **P95 Improvement** | -0.15% | +0.01% | -4.65% |
| **p-value** | 0.870 | 0.789 | 0.509 |
| **Significant?** | No | No | No |
| **Predictor R²** | 0.109 | 0.095 | -0.061 |

### Key Finding

Smart routing showed no measurable latency benefit (p > 0.50 across three models, 166 requests per strategy). **Root cause**: The 500-token routing threshold never activated—outputs clustered below 250 tokens, making smart routing and round-robin functionally identical.

## How It Works

```
1. Extract prompt features (length, entropy, code markers, etc.)
2. Predict output token count via Ridge regression
3. Route high-cost requests (> threshold) to least-loaded worker
4. Route low-cost requests via round-robin
5. Measure latency and token count
```

## Architecture

### Components

| Component | File | Purpose |
|-----------|------|---------|
| **PromptFeatureExtractor** | `utils/predictor.py` | Extract 10 features from prompt |
| **OutputLengthPredictor** | `utils/predictor.py` | Ridge regression for token prediction |
| **LoadBalancer** | `utils/load_balancer.py` | Route requests via smart/round-robin/least-loaded |
| **VLLMInference** | `utils/inference.py` | vLLM wrapper with GPU memory management |
| **StatisticalTest** | `utils/statistics.py` | Compute latency distributions and t-tests |

### Prompt Features

- `prompt_length` — Character count
- `word_count` — Token estimate
- `vocabulary_entropy` — Lexical complexity
- `question_marks` — Query signal
- `code_markers` — Code detection
- `imperative_verbs` — Task type
- `punctuation_density` — Structure signal
- `uppercase_density` — Emphasis markers
- `sentence_count` — Discourse structure
- `digit_presence` — Numerical reasoning

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
jupyter notebook validation_notebook.ipynb
```

Runs full validation pipeline:
1. Load 500 Alpaca prompts
2. Inference on 3 models (Llama-2, Mistral, Phi-2)
3. Train predictor on 400 samples, validate on 100
4. A/B test 166 requests per strategy
5. Compute statistics and p-values

## Configuration

Edit `config.yaml`:

```yaml
models:
  - name: "mistralai/Mistral-7B-Instruct-v0.1"
    max_tokens: 256
  - name: "microsoft/phi-2"
    max_tokens: 256

dataset:
  name: "alpaca"
  num_prompts: 500

vllm:
  gpu_memory_utilization: 0.7
  batch_size: 1
  max_model_len: 512
  kv_cache_dtype: "auto"

smart_routing:
  cost_threshold: 500  # Adjust for activation rate

predictor:
  model_types: ["ridge", "lasso", "elasticnet"]
  alpha: 1.0
```

## Why It Didn't Work

### Threshold Too High
- Output tokens: 1-256 (median ~160)
- Routing threshold: 500 tokens
- Activation rate: 0%
- Result: Routing logic never engaged

### Weak Predictor
- Ridge regression R²: 0.09-0.11
- MAE: ~80 tokens
- Explanation: Only 10% of output variance captured by surface features

### No Load Contention
- All workers equally available
- Load balancing most valuable under queue pressure
- Simulation with 3 idle workers cannot reveal benefits


## Project Structure

```
smart-load-balancer/
├── README.md                          (this file)
├── config.yaml                        (configuration)
├── validation_notebook.ipynb          (validation pipeline)
├── requirements.txt
├── utils/
│   ├── __init__.py
│   ├── inference.py                   (vLLM wrapper)
│   ├── predictor.py                   (feature extraction, prediction)
│   ├── load_balancer.py               (routing strategies)
│   └── statistics.py                  (statistical testing)
├── SMART_LOAD_BALANCER.md             (detailed analysis)
└── IMPLEMENTATION_GUIDE.md            (technical deep-dive)
```

## Documentation

- **SMART_LOAD_BALANCER.md** — Full validation analysis with root cause breakdown
- **IMPLEMENTATION_GUIDE.md** — Code structure, extension points, debugging guide

## Technical Details

### A/B Test Design
- Sample size: 166 requests per strategy (adaptive scaling)
- Statistical test: Welch's t-test (unequal variances)
- Significance level: α = 0.05
- Confidence intervals: 95%

### GPU Configuration
- Device: NVIDIA RTX A5000 (24GB VRAM)
- vLLM: gpu_memory_utilization=0.7, kv_cache_dtype="auto"
- Sequential model loading with explicit cleanup (gc + torch.cuda.empty_cache())

### Predictor Training
- Algorithm: Ridge regression (α=1.0)
- Train/test split: 80/20 on 500 prompts
- Features: Standardized (zero mean, unit variance)

## References

- **vLLM**: https://github.com/vllm-project/vllm
- **Alpaca Dataset**: https://github.com/tatsu-lab/alpaca
- **Ridge Regression**: Scikit-learn documentation
- **Statistical Testing**: Welch's t-test for unequal variances

## License

MIT
