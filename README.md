# Smart Load Balancer for LLM Inference

A distributed inference load balancer that routes LLM requests based on predicted output length, reducing p95 latency and improving throughput compared to simple round-robin routing.

## Problem

Standard round-robin load balancers treat all inference requests equally, despite massive differences in computational cost. A 50-token request and a 5000-token request cost the same to route but require vastly different compute to serve. This creates:

- Hot spots where some workers become overloaded
- Increased p95/p99 latency for users
- Poor resource utilization
- Queue buildup under mixed traffic

## Solution

This project implements predictive load balancing:

1. **Length Prediction**: Train a Ridge regression model on prompt features (length, entropy, question marks, code markers, vocabulary density) to predict output token count (~50ms overhead).

2. **Intelligent Routing**:
   - High-cost requests (>500 predicted tokens) → route to least-loaded workers
   - Low-cost requests (<500 tokens) → use round-robin for simplicity

3. **Health Monitoring**: Track worker availability and latency, excluding unhealthy workers from routing decisions.

## Architecture

```
Client Requests
    ↓
Load Balancer (Port 8000)
├─ Prompt Feature Extractor
├─ Ridge Regression Predictor
├─ Health Monitor
└─ Smart Router
    ↓
Worker Pool (Ports 8001-8003)
├─ Worker 1
├─ Worker 2
└─ Worker 3
```

## Components

### `src/predictor.py`
- `PromptFeatureExtractor`: Extracts 8 features from prompts
- `OutputLengthPredictor`: Ridge regression model for output length prediction

### `src/load_balancer.py`
- `SmartLoadBalancer`: Main routing engine
- `RoutingStrategy`: ROUND_ROBIN, LEAST_LOADED, PREDICTED_COST
- `WorkerHealth`: Tracks per-worker metrics
- Health checking and metrics collection

### `src/server.py`
- FastAPI inference server
- `/infer` endpoint for inference requests
- `/metrics` for load balancer statistics
- `/workers` for worker status

### `src/worker.py`
- Simulated inference worker
- `/health` health check endpoint
- `/infer` inference endpoint

### `benchmarks/run_benchmarks.py`
- Benchmark harness
- Starts infrastructure (workers, load balancer)
- Runs load test with realistic traffic mix

### `benchmarks/locustfile.py`
- Locust load testing configuration
- 60% short requests (100 tokens)
- 30% medium requests (400 tokens)
- 10% long requests (800+ tokens)

## Installation

```bash
# Clone repo
cd smart-load-balancer

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/
```

## Quick Start

### Manual Testing

Terminal 1: Start Load Balancer
```bash
python -c "from src.server import app; import uvicorn; uvicorn.run(app, host='127.0.0.1', port=8000)"
```

Terminal 2: Start Worker 1
```bash
python -c "from src.worker import run_worker; run_worker('worker-1', 8001)"
```

Terminal 3: Start Worker 2
```bash
python -c "from src.worker import run_worker; run_worker('worker-2', 8002)"
```

Terminal 4: Start Worker 3
```bash
python -c "from src.worker import run_worker; run_worker('worker-3', 8003)"
```

Terminal 5: Test Inference
```bash
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is machine learning?"}'

# Get metrics
curl http://localhost:8000/metrics
```

### Automated Benchmarking

```bash
# Run full benchmark suite (requires locust)
python benchmarks/run_benchmarks.py
```

## Key Features

1. **Fast Prediction**: ~50ms inference overhead per request using Ridge regression
2. **Adaptive Routing**: Routes high-cost requests to less-loaded workers
3. **Health Monitoring**: Periodically checks worker health, excludes unhealthy workers
4. **Metrics Collection**: Comprehensive latency, throughput, and per-worker statistics
5. **Scalable**: Works with any number of workers

## Benchmark Results

The benchmark tests load balancing under realistic traffic:
- **50 concurrent users** simulating inference clients
- **120-second duration** test window
- **Mixed traffic pattern**: 60% short, 30% medium, 10% long requests

Metrics collected:
- **p50, p95, p99 latency** (ms)
- **Average latency and throughput**
- **Per-worker load and error rates**

Expected improvements vs round-robin:
- **15-25% p95 latency reduction** (fewer hot spots)
- **10-15% throughput improvement** (better utilization)
- **More balanced worker load** (even distribution)

## Design Decisions

1. **Ridge Regression**: Simple, fast, interpretable model suitable for real-time prediction
2. **50% threshold for routing**: Balance between simplicity (RR) and complexity (LB)
3. **Async processing**: Non-blocking request handling to maximize concurrency
4. **Health checks every 5s**: Balance between responsiveness and overhead
5. **Local simulation**: CPU-only benchmarking for reproducibility

## Future Enhancements

- Adaptive threshold tuning based on worker performance
- Context-aware predictions (code generation → longer outputs)
- Integration with real LLM inference frameworks (vLLM, TensorRT-LLM)
- Multi-region failover
- Request prioritization based on SLA

## Testing

Run unit tests:
```bash
python tests/test_predictor.py
python tests/test_load_balancer.py
```

## References

- Load balancing strategies: Round-robin vs Least-connected
- Ridge regression: scikit-learn documentation
- Feature engineering for text: NLP best practices
- Async patterns in Python: asyncio documentation
