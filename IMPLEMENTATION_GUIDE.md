# Implementation Guide: Smart Load Balancer

Complete technical reference for understanding, extending, or reimplementing the smart load balancer system.

## System Architecture

### High-Level Flow

```
Request
  ↓
PromptFeatureExtractor (10 features)
  ↓
OutputLengthPredictor (Ridge regression)
  ↓
LoadBalancer.route() → (strategy, worker)
  ↓
Worker (vLLM)
  ↓
Response + Metadata (latency, tokens, worker)
```

### Component Responsibilities

| Component | File | Responsibility |
|-----------|------|-----------------|
| **PromptFeatureExtractor** | `utils/predictor.py` | Extract 10 features from prompt text |
| **OutputLengthPredictor** | `utils/predictor.py` | Train/predict output token count; validate at threshold |
| **VLLMInference** | `utils/inference.py` | Initialize vLLM engine, run batch inference, cleanup GPU |
| **LoadBalancer** | `utils/load_balancer.py` | Route requests using one of three strategies |
| **StatisticalTest** | `utils/statistics.py` | Compute latency distributions, t-tests, confidence intervals |

## Code Structure

### 1. Feature Extraction (`PromptFeatureExtractor`)

```python
class PromptFeatureExtractor:
    def extract(self, prompt: str) -> np.ndarray:
        """Return 10-dimensional feature vector."""
        return np.array([
            len(prompt),                    # prompt_length
            len(prompt.split()),            # word_count
            entropy(prompt),                # vocabulary_entropy
            prompt.count('?'),              # question_marks
            len(re.findall(r'```', prompt)), # code_markers
            sum(1 for v in IMPERATIVE_VERBS if v in prompt.lower()), # imperative_verbs
            punctuation_density(prompt),    # punctuation_density
            uppercase_density(prompt),      # uppercase_density
            prompt.count('.') + prompt.count('!'), # sentence_count
            int(any(c.isdigit() for c in prompt))  # digit_presence
        ])
```

**Key design decisions:**
- 10 features chosen to balance interpretability with coverage
- No deep learning (embedding models add latency)
- All features computable in <1ms
- Standardized (zero mean, unit variance) before prediction

### 2. Output Length Prediction (`OutputLengthPredictor`)

```python
class OutputLengthPredictor:
    def __init__(self, model_type='ridge', alpha=1.0):
        self.model_type = model_type  # 'ridge', 'lasso', or 'elasticnet'
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)  # or Lasso/ElasticNet
        self.scaler = StandardScaler()
    
    def train(self, X_train, y_train):
        """Fit on training data."""
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
    
    def predict(self, X_test):
        """Predict output tokens for test data."""
        X_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_scaled)
    
    def confusion_matrix_at_threshold(self, X_test, y_true, threshold=500):
        """Binary classification: output >= threshold or not."""
        y_pred = self.predict(X_test)
        y_pred_binary = (y_pred >= threshold).astype(int)
        y_true_binary = (y_true >= threshold).astype(int)
        return compute_confusion_matrix(y_pred_binary, y_true_binary)
```

**Why Ridge regression:**
- Simple, interpretable, fast (~50ms inference)
- Low variance (regularization helps with small n)
- Baseline for output prediction (more complex models possible later)

**Standardization is critical:**
- Features have different scales (1-500 for prompt_length, 0-1 for entropy)
- Ridge regression assumes feature importance correlates with magnitude
- Standardization fixes this

### 3. Load Balancing (`LoadBalancer`)

```python
class LoadBalancer:
    def __init__(self, num_workers=3, predictor=None):
        self.num_workers = num_workers
        self.predictor = predictor
        self.worker_metrics = [{'queue_depth': 0, 'latency': 0} for _ in range(num_workers)]
        self.request_count = 0
    
    def route(self, prompt, strategy='smart_routing'):
        """Select worker for this request."""
        worker_idx = {
            'smart_routing': self._smart_route(prompt),
            'round_robin': self._round_robin(),
            'least_loaded': self._least_loaded()
        }[strategy]
        
        self.worker_metrics[worker_idx]['queue_depth'] += 1
        return worker_idx
    
    def _smart_route(self, prompt):
        """Route high-predicted-cost requests to least-loaded workers."""
        predicted_tokens = self.predictor.predict(extract_features(prompt))
        
        if predicted_tokens >= THRESHOLD:  # 500 tokens
            # High-cost request; route to least-loaded worker
            return min(range(self.num_workers),
                      key=lambda i: self.worker_metrics[i]['queue_depth'])
        else:
            # Low-cost request; round-robin
            return self.request_count % self.num_workers
    
    def _round_robin(self):
        """Cycle through workers."""
        worker_idx = self.request_count % self.num_workers
        self.request_count += 1
        return worker_idx
    
    def _least_loaded(self):
        """Route to worker with smallest queue."""
        return min(range(self.num_workers),
                  key=lambda i: self.worker_metrics[i]['queue_depth'])
```

**Critical insight:** Smart routing only activates for requests predicted >= threshold. If threshold is too high (or predictor underestimates), strategy is never applied.

### 4. vLLM Inference (`VLLMInference`)

```python
class VLLMInference:
    def __init__(self, model_name, max_tokens=256, gpu_memory_utilization=0.7):
        self.model_name = model_name
        self.llm = LLM(
            model=model_name,
            max_model_len=512,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype='float16',
            kv_cache_dtype='auto',  # NOT 'fp8' (incompatible with SM86)
            trust_remote_code=True,
            hf_token=os.getenv('HF_TOKEN')
        )
    
    def generate(self, prompts, max_tokens=256):
        """Batch inference; return completions + token counts."""
        outputs = self.llm.generate(
            prompts,
            SamplingParams(max_tokens=max_tokens, temperature=0.7, top_p=0.95)
        )
        
        results = []
        for output in outputs:
            completion = output.outputs[0].text
            tokens = len(output.outputs[0].token_ids)
            results.append({'text': completion, 'tokens': tokens})
        
        return results
    
    def cleanup(self):
        """Clear GPU memory for next model."""
        del self.llm
        gc.collect()
        torch.cuda.empty_cache()
```

**GPU memory management:**
- Sequential model loading (load → infer → cleanup → next model)
- gpu_memory_utilization=0.7 (conservative)
- kv_cache_dtype='auto' (vLLM picks optimal dtype per model)
- enforce_eager=false (allows some optimization)

### 5. Statistical Testing (`statistics.py`)

```python
def compute_statistics(smart_latencies, rr_latencies):
    """A/B test: compare two latency distributions."""
    
    # Descriptive statistics
    smart_stats = {
        'mean': np.mean(smart_latencies),
        'median': np.median(smart_latencies),
        'std': np.std(smart_latencies),
        'p50': np.percentile(smart_latencies, 50),
        'p95': np.percentile(smart_latencies, 95),
        'p99': np.percentile(smart_latencies, 99),
        'min': np.min(smart_latencies),
        'max': np.max(smart_latencies)
    }
    
    # Statistical test: Welch's t-test (unequal variances)
    t_stat, p_value = scipy.stats.ttest_ind(
        smart_latencies, rr_latencies, equal_var=False
    )
    
    # Confidence intervals via bootstrap
    ci_smart = bootstrap_ci(smart_latencies, statistic=np.mean, n_resamples=10000)
    ci_rr = bootstrap_ci(rr_latencies, statistic=np.mean, n_resamples=10000)
    
    # Effect size
    p95_improvement_pct = 100 * (
        (np.percentile(rr_latencies, 95) - np.percentile(smart_latencies, 95))
        / np.percentile(rr_latencies, 95)
    )
    
    return {
        'smart_routing': smart_stats,
        'round_robin': rr_stats,
        'comparison': {
            'p95_improvement_percent': p95_improvement_pct,
            'ttest': {'t_statistic': t_stat, 'p_value': p_value},
            'significant_at_0_05': 1 if p_value < 0.05 else 0
        }
    }
```

**Why Welch's t-test:**
- Doesn't assume equal variances
- Robust to outliers in latency distributions
- Standard in A/B testing

**Why report p95, not just mean:**
- Tail latency (p95, p99) is production-critical
- Smart routing should reduce variance, not just mean

## Configuration Parameters

### Core Parameters

| Parameter | Default | Rationale | Tuning |
|-----------|---------|-----------|--------|
| `cost_threshold` | 500 | Routing decision boundary | **Lower to 200-300** if testing |
| `gpu_memory_utilization` | 0.7 | Leave room for peak activation | Reduce if OOM; increase if idle GPU |
| `max_model_len` | 512 | Max context length | Match model capabilities |
| `alpha` (Ridge) | 1.0 | Regularization strength | Try 0.1-10 for better fit |
| `ab_test.requests_per_strategy` | 250 | Samples per routing strategy | Must be < len(prompts) / 3 |
| `test_split` | 0.2 | Train/test for predictor | Standard; don't change |

### Feature Scaling

Ridge regression requires standardized features:

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Train set only
X_test_scaled = scaler.transform(X_test)  # Consistency
```

**Critical:** Fit scaler on training data; apply same scaler to test data.

## Data Flow in Validation Notebook

```
1. Load config.yaml
2. Load Alpaca dataset (500 prompts)
3. For each model:
   a. Initialize VLLMInference
   b. Run inference on all 500 prompts
   c. Extract 10 features for each prompt
   d. Split 80/20 into train/test
   e. Train Ridge regression on train set
   f. Evaluate on test set (R², MAE)
   g. Compute confusion matrix at threshold
   h. A/B test (smart vs round-robin) on full 500 prompts
   i. Compute statistics, t-tests, confidence intervals
   j. Save results to JSON
   k. Cleanup GPU memory
4. Publish aggregated results
```

## Extension Points

### 1. Better Predictor Features

Replace `PromptFeatureExtractor` with:

```python
class SemanticFeatureExtractor:
    def __init__(self, embedding_model='sentence-transformers/all-MiniLM-L6-v2'):
        self.embed_model = ...  # Load embedding model
    
    def extract(self, prompt):
        # Embed prompt
        embedding = self.embed_model.encode(prompt)
        
        # Could add:
        # - Similarity to training prompt examples
        # - Task type classification (QA, summarization, code, etc.)
        # - Length prediction based on similar historical examples
        
        return np.concatenate([surface_features, embedding[:10], task_features])
```

**Trade-off:** +50-100ms latency per request vs. potentially better R².

### 2. Ensemble Predictors

Train multiple models; use average:

```python
predictors = [
    Ridge(alpha=0.1),
    Ridge(alpha=1.0),
    Ridge(alpha=10.0),
    Lasso(alpha=1.0),
    ElasticNet(alpha=1.0, l1_ratio=0.5)
]

predictions = np.mean([p.predict(X) for p in predictors], axis=0)
```

### 3. Queue-Aware Routing

Current: Routes high-cost to least-loaded. Advanced:

```python
def _smart_route_v2(self, prompt):
    predicted_tokens = self.predictor.predict(extract_features(prompt))
    
    if predicted_tokens >= THRESHOLD:
        # High-cost: Route to least-loaded worker
        # But penalize workers that are overloaded
        scores = [
            self.worker_metrics[i]['queue_depth'] +
            0.5 * self.worker_metrics[i]['recent_p95'] / 5000
            for i in range(self.num_workers)
        ]
        return np.argmin(scores)
    else:
        # Low-cost: Spread evenly
        return self.request_count % self.num_workers
```

### 4. Adaptive Threshold

Instead of fixed 500 tokens, learn from data:

```python
# After collecting first N requests:
output_distribution = [actual_tokens for _, actual_tokens in first_N_requests]
threshold = np.percentile(output_distribution, 75)  # Top 25% are "high-cost"
```

## Debugging Checklist

### Issue: No statistically significant improvement

**Check:**
1. Is threshold activated? Run confusion matrix: TP + FN > 0?
2. Is predictor working? Check R² on test set. < 0.15 means weak.
3. Is variance high? std(latencies) > mean(latencies)?

**Action:**
1. Lower threshold by 50% and retest
2. Add semantic features to predictor
3. Simulate worker load (artificial delays) to create contention

### Issue: GPU out of memory

**Check:**
1. gpu_memory_utilization setting (currently 0.7)
2. Batch size (currently 1)
3. Whether previous model cleaned up

**Action:**
1. Add explicit cleanup: `del self.llm; gc.collect(); torch.cuda.empty_cache()`
2. Reduce gpu_memory_utilization to 0.5
3. Check `torch.cuda.memory_allocated()` before each model

### Issue: Predictor R² is negative

**Check:**
1. Are features correlated with output length?
2. Is data split properly (no data leakage)?
3. Is scaler fit on train, applied to test?

**Action:**
1. Visualize feature-target correlations: `pd.DataFrame(X).corrwith(y)`
2. Verify train/test split is stratified (same output length distribution)
3. Add debugging: log min/max of scaled features on train vs test

## Performance Characteristics

### Timing Breakdown

| Component | Time | Notes |
|-----------|------|-------|
| Feature extraction | <1ms | Per prompt |
| Ridge prediction | ~5-10ms | Batch of 166 |
| vLLM inference | 4-6s | Per prompt on GPU |
| Statistics computation | 100-200ms | Entire A/B test |

**Bottleneck:** vLLM inference (99% of total time). Predictor overhead (~0.1%) is negligible.

### Memory Usage

| Component | Memory |
|-----------|--------|
| vLLM model (Llama-2) | ~12.5 GB |
| vLLM KV cache | ~3 GB |
| Prompt buffer (500 prompts) | ~50 MB |
| Total GPU | ~15.5 GB (RTX A5000: 24GB available) |

## References

### Research Papers
- **Ridge Regression:** Hoerl & Kennard (1970) - "Ridge Regression: Biased Estimation for Nonorthogonal Problems"
- **Welch's t-test:** Welch (1947) - "The Generalization of Student's Problem when Several Different Population Variances are Involved"
- **Bootstrap CI:** Efron (1979) - "Bootstrap Methods"

### Libraries
- **vLLM:** https://github.com/vllm-project/vllm
- **Scikit-learn:** https://scikit-learn.org
- **Scipy.stats:** https://docs.scipy.org/doc/scipy/reference/stats.html

### Related Work
- **HuggingFace Inference API** — similar predictor-based batching
- **Modal** — inference optimization via request grouping
- **vLLM's ContinuousBatching** — dynamic batching (separate from load balancing)
