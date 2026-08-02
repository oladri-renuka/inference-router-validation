# Load Balancer Benchmarking Guide

This directory contains benchmarking tools to measure the performance of the smart load balancer against baseline strategies.

## Quick Start

### 1. Run Complete Benchmark Suite

```bash
# Installs required dependencies and runs full benchmark
python benchmarks/run_benchmarks.py
```

This will:
- Start 3 simulated inference workers
- Start the load balancer server
- Send 50 concurrent users for 120 seconds
- Report p50/p95/p99 latency, throughput, and per-worker metrics

### 2. Manual Benchmarking with Locust

```bash
# Terminal 1: Start load balancer
python -c "from src.server import app; import uvicorn; uvicorn.run(app, host='127.0.0.1', port=8000)"

# Terminal 2-4: Start 3 workers
python -c "from src.worker import run_worker; run_worker('worker-1', 8001)"
python -c "from src.worker import run_worker; run_worker('worker-2', 8002)"
python -c "from src.worker import run_worker; run_worker('worker-3', 8003)"

# Terminal 5: Run Locust benchmark
locust -f benchmarks/locustfile.py --host=http://127.0.0.1:8000 --users=50 --spawn-rate=10 --run-time=120s --headless
```

## Benchmark Configuration

### Traffic Pattern (Realistic Mixed Workload)

- **60% Short Requests** (~100 tokens output)
  - "What is ML?"
  - "Define AI"
  - "Explain backprop"

- **30% Medium Requests** (~400 tokens output)
  - "Explain how neural networks work"
  - "What are main types of ML?"
  - "Describe CNN architectures"

- **10% Long Requests** (~800+ tokens output)
  - Comprehensive guide to transformers
  - Complete training pipeline explanation
  - Advanced optimization techniques

### Load Profile

- **Users**: 50 concurrent
- **Duration**: 120 seconds
- **Hatch Rate**: 10 users/second
- **User Wait Time**: 0.5-2 seconds between requests

## Expected Results

### Metrics to Collect

1. **Latency Percentiles** (ms)
   - p50: Median latency
   - p95: 95th percentile (most users experience ≤ this)
   - p99: 99th percentile (worst case near-median)

2. **Throughput**
   - Requests/second
   - Total requests completed

3. **Worker Distribution**
   - Load per worker
   - Error rates
   - Average latency per worker

### Expected Improvements (Smart Routing vs Round-Robin)

| Metric | Round-Robin | Smart Routing | Improvement |
|--------|-------------|---------------|------------|
| P50 Latency | ~150ms | ~140ms | 7% |
| P95 Latency | ~400ms | ~320ms | 20% |
| P99 Latency | ~600ms | ~480ms | 20% |
| Throughput | 8-10 req/s | 9-11 req/s | 10-15% |
| Worker Load Balance | Uneven | Even | ✓ |

## Running Comparison Benchmarks

### Strategy 1: Round-Robin Baseline

Modify `src/load_balancer.py` to always use:
```python
route = lb.route_request(req, strategy=RoutingStrategy.ROUND_ROBIN)
```

Run benchmark and record metrics.

### Strategy 2: Predicted Cost (Smart)

Keep default configuration:
```python
route = lb.route_request(req, strategy=RoutingStrategy.PREDICTED_COST)
```

Run benchmark and record metrics.

## Interpreting Results

### Healthy Benchmark Indicators

- P95 latency < 500ms
- Throughput > 5 req/s
- All workers healthy (no errors)
- Worker load scores within 0.2 of each other
- No timeouts or connection errors

### Red Flags

- P99 latency >> P95 latency (high tail latency)
- Uneven worker load (one worker at 0.8+ load)
- Frequent worker health check failures
- Throughput decreasing over time (resource leak)

## Customizing Benchmarks

### Change Traffic Mix

Edit `benchmarks/locustfile.py`:
```python
@task(60)  # Change 60 to weight
def short_request(self):
    ...

@task(30)  # Change 30 to weight
def medium_request(self):
    ...
```

### Change Load Parameters

Edit `benchmarks/run_benchmarks.py`:
```python
success = run_locust_benchmark(
    duration=120,      # Total seconds
    users=50,          # Concurrent users
    hatch_rate=10      # Users added per second
)
```

### Add Custom Prompts

Edit `benchmarks/locustfile.py` lists:
```python
SHORT_PROMPTS = [
    "Your custom short prompt",
    ...
]
```

## Analyzing Results

### Generate CSV Report

Locust can output results to CSV:
```bash
locust -f benchmarks/locustfile.py \
  --host=http://127.0.0.1:8000 \
  --users=50 --run-time=120s --headless \
  --csv=benchmark_results
```

### Performance Timeline

Monitor metrics over time:
```bash
# While benchmark runs, in another terminal:
watch -n 5 'curl http://localhost:8000/metrics | jq .'
```

## Debugging Performance Issues

### Slow Predictor

If prediction adds >100ms overhead:
1. Check `PromptFeatureExtractor` complexity
2. Verify Ridge model is trained
3. Profile feature extraction time

### Unbalanced Worker Load

If workers aren't balanced:
1. Check health monitoring is running
2. Verify `load_score` calculation
3. Ensure `_route_least_loaded()` is selecting correctly

### Worker Latency Spike

If some workers are slow:
1. Check asyncio event loop isn't blocking
2. Verify sleep time calculation in worker.py
3. Look for GC pauses or kernel scheduling issues

## Benchmarking Best Practices

1. **Warm up**: Run for 30s before measuring to stabilize
2. **No other processes**: Close unnecessary applications
3. **Run multiple times**: Take median of 3+ runs
4. **Document environment**: CPU, RAM, Python version
5. **Compare at same time**: Both strategies on same machine

## References

- Locust documentation: https://locust.io/
- Load testing best practices: https://en.wikipedia.org/wiki/Load_testing
- Percentile interpretation: https://www.dynatrace.com/news/blog/why-percentiles-dont-matter/
