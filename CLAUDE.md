# Smart Load Balancer for LLM Inference — Project Instructions

## Project Overview

Build and benchmark a predictive load balancer for LLM inference that routes requests based on predicted output length, not just round-robin.

**Target Companies**: Meta, OpenAI, Nvidia, Scale AI  
**Estimated Time**: 1.5 weeks  
**Stack**: Python, FastAPI, asyncio, scikit-learn, httpx, locust  
**GPU Required**: No (CPU simulation)

## Key Success Criteria

1. **Predictor Accuracy**: Ridge regression predicts output length within 30% error on test set
2. **Latency Improvement**: Smart routing achieves 15-25% p95 latency reduction vs round-robin
3. **Throughput**: 10-15% throughput improvement under mixed traffic
4. **Code Quality**: Full test coverage for predictor and load balancer
5. **Documentation**: Clear benchmarking results showing real impact

## Repository Structure

```
smart-load-balancer/
├── src/
│   ├── predictor.py           # Ridge regression model + feature extraction
│   ├── load_balancer.py       # Routing engine, health monitoring
│   ├── server.py              # FastAPI load balancer server
│   └── worker.py              # Simulated inference worker
├── benchmarks/
│   ├── locustfile.py          # Load test scenarios (60/30/10 traffic mix)
│   └── run_benchmarks.py      # Benchmark harness
├── tests/
│   ├── test_predictor.py      # Unit tests for prediction
│   └── test_load_balancer.py  # Unit tests for routing
├── requirements.txt
├── README.md
└── CLAUDE.md (this file)
```

## Component Details

### 1. Predictor (`src/predictor.py`)

**PromptFeatureExtractor**:
- 8 features extracted from prompts:
  1. Normalized prompt length
  2. Word count
  3. Vocabulary entropy
  4. Question mark density
  5. Code block indicators
  6. Imperative verb count
  7. Punctuation density
  8. Uppercase density

**OutputLengthPredictor**:
- Ridge regression model (alpha=1.0)
- Training on 5 example prompts × 20 repetitions
- Predict method returns single prediction
- Predict_batch returns array of predictions

Key: All predictions must be > 0 (use max(prediction, 1.0))

### 2. Load Balancer (`src/load_balancer.py`)

**RoutingStrategies**:
- `ROUND_ROBIN`: Cycle through workers
- `LEAST_LOADED`: Route to worker with lowest load score
- `PREDICTED_COST`: 
  - High-cost (>500 tokens) → least-loaded
  - Low-cost (<500 tokens) → round-robin

**WorkerHealth**:
- Tracks: current_requests, total_requests, total_latency, error_count
- Health checks run async every 5 seconds
- load_score combines request count (70%) and latency (30%)

**InferenceRequest**:
- Tracks: request_id, prompt, predicted_output_length, submitted_at, routed_to_worker, completed_at
- Compute latency as completed_at - submitted_at

### 3. Server (`src/server.py`)

**Endpoints**:
- `GET /health` → server status
- `POST /infer` → submit inference request
- `GET /metrics` → load balancer metrics (p50/p95/p99, throughput, worker health)
- `GET /workers` → worker status

**Startup**:
- Initialize predictor with training data
- Initialize load balancer with 3 workers (127.0.0.1:8001/8002/8003)
- Start async health checks

### 4. Worker (`src/worker.py`)

**Endpoints**:
- `GET /health` → worker status
- `POST /infer` → simulate inference (sleep proportional to output length)

**Simulation**:
- Output length = random(base_length * 0.8, base_length * 1.2)
- Processing time = tokens × 10ms + gaussian(0, 50ms)
- Actually sleep to simulate compute

### 5. Benchmarks (`benchmarks/locustfile.py` + `run_benchmarks.py`)

**Traffic Mix**:
- 60% short requests (~100 tokens)
- 30% medium requests (~400 tokens)
- 10% long requests (~800 tokens)

**Run Settings**:
- Duration: 120 seconds
- Concurrent users: 50
- Hatch rate: 10 users/sec

**Metrics**:
- Collect: p50, p95, p99 latency
- Report: throughput (req/s), worker load distribution

## Development Workflow

### Phase 1: Core Implementation (completed)
- [x] Predictor with 8-feature extraction
- [x] Ridge regression training/prediction
- [x] Load balancer routing strategies
- [x] Health monitoring system
- [x] FastAPI server
- [x] Worker simulator
- [x] Unit tests

### Phase 2: Testing
- [ ] Run unit tests: `python tests/test_predictor.py` and `python tests/test_load_balancer.py`
- [ ] Verify all tests pass
- [ ] Manual integration testing:
  - Start load balancer + 3 workers
  - Send test requests to `/infer`
  - Verify `/metrics` endpoint works

### Phase 3: Benchmarking
- [ ] Install locust: `pip install locust` (in requirements.txt)
- [ ] Run benchmarks: `python benchmarks/run_benchmarks.py`
- [ ] Collect metrics for both strategies:
  - Smart routing (predicted cost)
  - Round-robin baseline
- [ ] Compare: p95 latency, throughput, worker load balance

### Phase 4: Documentation
- [ ] Write benchmark results section
- [ ] Document findings (% improvement)
- [ ] Clean up any debug code
- [ ] Final commit

## Testing Checklist

Unit Tests:
- [ ] test_predictor.py passes all 4 tests
- [ ] test_load_balancer.py passes all 6 tests

Integration Tests:
- [ ] Load balancer starts without errors
- [ ] Workers connect and pass health checks
- [ ] POST /infer returns valid response
- [ ] GET /metrics returns expected format
- [ ] Round-robin distributes evenly
- [ ] Least-loaded routes to lowest-load worker
- [ ] Predicted cost routes correctly

Benchmark:
- [ ] 50 concurrent users for 120 seconds
- [ ] Collect full results with p50/p95/p99
- [ ] Compare against round-robin baseline

## Key Invariants

1. **Predictor output**: Always > 0
2. **Load score**: Between 0 and 1
3. **Health checks**: Every 5 seconds, non-blocking
4. **Routing**: Always returns a healthy worker or raises RuntimeError
5. **Metrics**: p50 ≤ p95 ≤ p99 ≤ max latency

## Common Issues & Fixes

**Issue**: Predictor not trained
→ Fix: Call predictor.train() with prompts and lengths before predict()

**Issue**: Workers not responding
→ Fix: Verify workers are running on correct ports (8001-8003)

**Issue**: Health checks failing
→ Fix: Ensure workers have /health endpoint responding with 200

**Issue**: Benchmark hangs
→ Fix: Check locust is installed; may need to adjust wait_time in locustfile.py

## References

- scikit-learn Ridge regression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
- FastAPI async: https://fastapi.tiangolo.com/async-sql-databases/
- Locust load testing: https://locust.io/
- Load balancing strategies: https://www.nginx.com/resources/glossary/load-balancing/

## Notes

- No GPU required; CPU simulation is realistic for benchmarking routing logic
- Feature extraction is deterministic (same prompt → same features)
- Ridge regression is chosen for speed (<50ms per prediction)
- Async/await allows handling many concurrent requests efficiently
- Metrics are collected in-memory (not persisted between runs)
