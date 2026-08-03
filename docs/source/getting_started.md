# Getting Started with Smart Load Balancer

## Installation (5 minutes)

```bash
# Clone and enter directory
cd smart-load-balancer

# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
```

## Verify Installation (2 minutes)

```bash
# Run unit tests
python tests/test_predictor.py
python tests/test_load_balancer.py
python tests/test_integration.py

# Expected output: All tests pass ✓
```

## First Run: Simple Test (5 minutes)

### Option 1: Quick Integration Test (No servers needed)

```bash
python tests/test_integration.py
```

This tests the complete flow (predict → route → measure) without running actual servers.

### Option 2: Manual End-to-End Test (Start servers)

**Terminal 1: Start Load Balancer**
```bash
python -c "
from src.server import app
import uvicorn
uvicorn.run(app, host='127.0.0.1', port=8000, log_level='info')
"
```

**Terminal 2-4: Start Workers**
```bash
# Worker 1
python -c "from src.worker import run_worker; run_worker('worker-1', 8001)"

# Worker 2
python -c "from src.worker import run_worker; run_worker('worker-2', 8002)"

# Worker 3
python -c "from src.worker import run_worker; run_worker('worker-3', 8003)"
```

**Terminal 5: Test Inference**
```bash
# Send a request
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is machine learning?"}'

# Expected response:
# {
#   "request_id": "...",
#   "output": "Generated response...",
#   "worker_id": "127.0.0.1:8001",
#   "predicted_length": 123.45,
#   "actual_length": 125,
#   "latency_ms": 234.5
# }
```

**Check Metrics**
```bash
# Get load balancer metrics
curl http://localhost:8000/metrics | jq

# Get worker status
curl http://localhost:8000/workers | jq
```

## Run Benchmarks (10 minutes)

### Automated Benchmark
```bash
# Starts infrastructure + runs 120-second load test
python benchmarks/run_benchmarks.py

# Expected output:
# Started worker-1 on port 8001
# Started worker-2 on port 8002
# Started worker-3 on port 8003
# Started load balancer on port 8000
# Running benchmark: 120s, 50 concurrent users
# [... Locust output ...]
# Final Load Balancer Metrics:
#   Total Requests: 456
#   P50 Latency: 145.23ms
#   P95 Latency: 320.45ms
#   P99 Latency: 485.67ms
#   Avg Latency: 178.90ms
#   Throughput: 3.80 req/s
```

### Manual Benchmark with Locust
```bash
# Requires locust: pip install locust
locust -f benchmarks/locustfile.py \
  --host=http://127.0.0.1:8000 \
  --users=50 --spawn-rate=10 --run-time=120s \
  --headless
```

## Next Steps

### 1. Understand the Code
- Read `ARCHITECTURE.md` for system design
- Review `src/predictor.py` for prediction logic
- Check `src/load_balancer.py` for routing strategies

### 2. Run Comparison Benchmarks
- See `benchmarks/README.md` for methodology
- Compare round-robin vs smart routing
- Document latency/throughput improvements

### 3. Customize for Your Use Case
- Edit `src/server.py` to use real inference endpoint
- Modify traffic mix in `benchmarks/locustfile.py`
- Adjust prediction threshold in `src/load_balancer.py`

### 4. Production Deployment
- Replace worker simulator with real inference service
- Add persistent metrics logging (Prometheus/InfluxDB)
- Implement request queuing for overload scenarios
- Add circuit breaker for failing workers

## Troubleshooting

### "No module named 'fastapi'"
Solution: `pip install -r requirements.txt`

### "Port 8000 already in use"
Solution: Kill existing process or use different port
```bash
lsof -i :8000  # Find process
kill -9 <PID>  # Kill it
```

### Workers not connecting
Solution: Verify workers started and health checks are working
```bash
curl http://127.0.0.1:8001/health  # Should return 200
curl http://127.0.0.1:8002/health
curl http://127.0.0.1:8003/health
```

### Benchmark hangs
Solution: Check Locust timeout settings or restart services

## Project Structure Overview

```
smart-load-balancer/
├── src/                     # Core implementation
│   ├── predictor.py        # Output length prediction (Ridge regression)
│   ├── load_balancer.py    # Routing engine & health monitoring
│   ├── server.py           # FastAPI load balancer server
│   └── worker.py           # Simulated inference worker
├── tests/                   # Unit & integration tests
│   ├── test_predictor.py
│   ├── test_load_balancer.py
│   └── test_integration.py
├── benchmarks/             # Load testing & benchmarking
│   ├── locustfile.py       # Traffic patterns
│   ├── run_benchmarks.py   # Benchmark harness
│   └── README.md           # Benchmarking guide
├── README.md               # Project overview
├── ARCHITECTURE.md         # System design & architecture
├── GETTING_STARTED.md      # This file
├── requirements.txt        # Python dependencies
└── .gitignore
```

## Key Concepts

### Predicted Cost Routing
- **High-cost** requests (predicted >500 tokens) go to **least-loaded** workers
- **Low-cost** requests use **round-robin** for simplicity
- Balances between perfect optimization and operational simplicity

### Feature Extraction
8 features predict output length:
1. Prompt length
2. Word count
3. Vocabulary entropy
4. Question mark density
5. Code block indicators
6. Imperative verb count
7. Punctuation density
8. Uppercase density

### Health Monitoring
- Async health checks every 5 seconds
- Unhealthy workers excluded from routing
- Tracks: current load, total requests, error count, latency

## Performance Characteristics

- **Prediction latency**: ~50ms (Ridge regression on 8 features)
- **Routing latency**: <5ms (O(n) worker scan)
- **Health check overhead**: <1% (once per 5 seconds)
- **Memory**: <50MB for full system

## Questions?

- See `README.md` for project overview
- Check `ARCHITECTURE.md` for implementation details
- Review `benchmarks/README.md` for measurement guide
- Look at test files for usage examples

---

**Ready to benchmark?** → `python benchmarks/run_benchmarks.py`
