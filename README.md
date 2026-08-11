# Inference Router Validation

A production-grade system for validating predictive load balancing strategies in distributed LLM inference. Implements intelligent request routing based on predicted output length, with rigorous A/B testing and statistical analysis.

## Overview

This project demonstrates end-to-end validation methodology for inference optimization techniques. The system:

1. **Predicts output token counts** using Ridge regression on prompt features
2. **Routes requests intelligently** based on predicted computational cost
3. **Measures performance** against baseline with statistical rigor
4. **Validates across models** for generalization (Llama-2, Mistral, Phi-2)

## Architecture

```
Prompt Input
    ↓
Feature Extraction (10 dimensions)
    ├─ prompt_length, word_count, vocabulary_entropy
    ├─ question_marks, code_markers, imperative_verbs
    ├─ punctuation_density, uppercase_density
    ├─ sentence_count, digit_presence
    ↓
Output Length Predictor (Ridge Regression)
    ├─ R² ≈ 0.10 (test set)
    ├─ MAE ≈ 80 tokens
    ↓
Load Balancer
    ├─ Smart Routing (predicted-cost aware)
    ├─ Round-Robin (baseline)
    ├─ Least-Loaded (queue-aware)
    ↓
Worker Pool (vLLM Inference)
    ├─ Llama-2-7b-chat-hf
    ├─ Mistral-7B-Instruct-v0.1
    ├─ microsoft/phi-2
    ↓
Statistical Analysis
    ├─ Latency distributions (P50, P95, P99)
    ├─ Predictor accuracy (R², MAE)
    ├─ Threshold validation (confusion matrix)
    ├─ A/B testing (t-tests, confidence intervals)
```

## Components

### `utils/predictor.py`
- **PromptFeatureExtractor**: Extracts 10 linguistic features from prompts
- **OutputLengthPredictor**: Ridge regression model with configurable regularization
- Threshold-based binary classification for routing decisions

### `utils/load_balancer.py`
- **LoadBalancer**: Routes requests via multiple strategies
  - Smart routing: High-cost → least-loaded; low-cost → round-robin
  - Round-robin: Uniform distribution baseline
  - Least-loaded: Queue-depth aware routing
- Worker health tracking and metrics collection

### `utils/inference.py`
- **VLLMInference**: vLLM wrapper with GPU memory management
- Batch inference with per-request latency measurement
- Sequential model loading and cleanup for multi-model validation

### `utils/statistics.py`
- Statistical significance testing (Welch's t-test)
- Confidence interval computation
- Percentile analysis (P50, P95, P99)

## Validation Pipeline

```bash
jupyter notebook validation_notebook.ipynb
```

**Full workflow** (2-4 hours on RTX A5000):

1. Load 500 Alpaca prompts (instruction-tuning dataset)
2. Run inference on all 3 models sequentially
3. Extract features and train predictors
4. Perform A/B testing (166 requests per strategy per model)
5. Compute statistics and significance tests
6. Save results as JSON

## Configuration

```yaml
# config.yaml
models:
  - name: "mistralai/Mistral-7B-Instruct-v0.1"
  - name: "microsoft/phi-2"

dataset:
  name: "alpaca"
  num_prompts: 500

vllm:
  gpu_memory_utilization: 0.7
  kv_cache_dtype: "auto"
  max_model_len: 512

smart_routing:
  cost_threshold: 500  # Route requests > 500 tokens

predictor:
  model_types: ["ridge"]
  alpha: 1.0

ab_test:
  requests_per_strategy: 250
```

## Installation

```bash
pip install -r requirements.txt
```

**Requirements**:
- GPU: 24GB VRAM minimum (RTX A5000, A100, H100)
- Python 3.10+
- CUDA 12.0+

## Results

Validation results saved to `results/` directory:

- `validation_results_Llama-2-7b-chat-hf.json`
- `validation_results_Mistral-7B-Instruct-v0.1.json`
- `validation_results_phi-2.json`

Each file contains:
- Latency statistics (mean, std, percentiles)
- Predictor accuracy (R², MAE)
- A/B test results (t-statistic, p-value, confidence intervals)
- Threshold validation (confusion matrix at 500 tokens)

## Design Decisions

### Ridge Regression for Prediction
- Simple, interpretable, ~50ms inference overhead
- Baseline for output length prediction
- Allows A/B testing of routing strategy independently from predictor quality

### 500-Token Threshold
- Empirically chosen based on output distribution
- Tunable parameter in config.yaml
- Enables binary classification for routing decisions

### Sequential Model Validation
- Tests generalization across model architectures
- Detects model-specific routing behavior
- Manages GPU memory constraints (sequential loading/cleanup)

### Welch's t-test for Statistics
- Handles unequal variances in latency distributions
- Standard choice for A/B testing in systems
- Provides p-values and confidence intervals

## Project Structure

```
inference-router-validation/
├── README.md                          (this file)
├── config.yaml                        (configuration)
├── requirements.txt                   (dependencies)
├── validation_notebook.ipynb          (main validation pipeline)
├── utils/
│   ├── __init__.py
│   ├── inference.py                   (vLLM inference wrapper)
│   ├── predictor.py                   (feature extraction + Ridge regression)
│   ├── load_balancer.py               (routing strategies)
│   └── statistics.py                  (A/B testing + statistics)
├── results/                           (validation outputs)
│   ├── validation_results_Llama-2-7b-chat-hf.json
│   ├── validation_results_Mistral-7B-Instruct-v0.1.json
│   └── validation_results_phi-2.json
└── .gitignore
```

## Technical Details

### Feature Standardization
Ridge regression requires zero-mean, unit-variance features. StandardScaler fit on training data; same transformation applied to test data for consistency.

### Adaptive A/B Test Scaling
Sample size: `min(config_requests_per_strategy, len(prompts) // 3)`
Prevents index errors when prompt count varies.

### GPU Memory Management
- Sequential model execution (load → infer → cleanup)
- `gc.collect()` + `torch.cuda.empty_cache()` between models
- gpu_memory_utilization=0.7 (conservative; tunable in config)

### Threshold Validation
Confusion matrix at 500-token threshold:
- Binary classification: output >= threshold vs. < threshold
- Evaluates routing activation rate
- Provides precision/recall for threshold choice

## Performance Characteristics

| Component | Time | Notes |
|-----------|------|-------|
| Feature extraction | <1ms | Per prompt |
| Ridge prediction | 5-10ms | Batch of 166 |
| vLLM inference | 4-6s | Per prompt on GPU |
| Statistics computation | 100-200ms | Full A/B test |

**Bottleneck**: vLLM inference (99% of total time). Predictor overhead is negligible.

## Deployment Considerations

### Strengths
- Production-grade statistical methodology
- Multi-model validation (generalization check)
- Complete end-to-end pipeline
- Configurable for different models/datasets

### Limitations
- Single-run validation (no independent replication)
- Simulated environment (3 workers on single GPU)
- Conservative vLLM configuration (eager mode)
- Alpaca dataset (instruction-tuning biased)

## References

- **vLLM**: https://github.com/vllm-project/vllm
- **Alpaca Dataset**: https://github.com/tatsu-lab/alpaca
- **Scikit-learn**: Ridge regression, StandardScaler
- **Scipy.stats**: Welch's t-test, confidence intervals

## License

MIT
