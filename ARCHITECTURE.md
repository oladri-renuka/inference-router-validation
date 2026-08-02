# Architecture & Design

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Load Balancer (Port 8000)                │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           Request Handler (FastAPI)                   │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │  POST /infer       - Submit inference request         │   │
│  │  GET  /health      - Health check                     │   │
│  │  GET  /metrics     - Load balancer metrics            │   │
│  │  GET  /workers     - Worker status                    │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                    │
│                          ↓                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │      Prediction Pipeline (50ms overhead)             │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │  1. PromptFeatureExtractor (8 features)              │   │
│  │  2. Ridge Regression Model                           │   │
│  │  3. Predict output length                            │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                    │
│                          ↓                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         Smart Router (Routing Decision)              │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │  IF predicted_length > 500:                          │   │
│  │    → Route to LEAST_LOADED worker                    │   │
│  │  ELSE:                                               │   │
│  │    → Route via ROUND_ROBIN                           │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                    │
│                          ↓                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │      Health Monitor (async, every 5s)                │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │  • Check worker health endpoints                     │   │
│  │  • Track: latency, errors, load                      │   │
│  │  • Exclude unhealthy workers from routing            │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                    │
│                          ↓                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │      Request Processor (async/await)                 │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │  • Send to selected worker                           │   │
│  │  • Track latency                                     │   │
│  │  • Collect metrics                                   │   │
│  │  • Return response                                   │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
          │                    │                    │
          ↓                    ↓                    ↓
      Worker 1            Worker 2            Worker 3
    (Port 8001)          (Port 8002)          (Port 8003)
   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
   │   /health    │   │   /health    │   │   /health    │
   │   /infer     │   │   /infer     │   │   /infer     │
   └──────────────┘   └──────────────┘   └──────────────┘
```

## Component Details

### 1. PromptFeatureExtractor

**Purpose**: Convert prompts into numeric features

**Features (8 total)**:
| Feature | Calculation | Range | Purpose |
|---------|-------------|-------|---------|
| Prompt Length | len(prompt) / 10000 | [0, 1] | Longer → more output |
| Word Count | words / 2000 | [0, 1] | Vocabulary size indicator |
| Vocabulary Entropy | unique_words / total_words | [0, 1] | Repetition indicator |
| Question Marks | count / 5 | [0, 1] | Question type detection |
| Code Markers | {`}[]` / 20 | [0, 1] | Code generation signal |
| Imperative Verbs | {generate,write,explain} | [0, 1] | Instruction type |
| Punctuation Density | punct_count / len | [0, 1] | Writing style |
| Uppercase Density | uppercase / len | [0, 1] | Acronym indicator |

### 2. OutputLengthPredictor

**Model**: Ridge Regression (alpha=1.0)

**Training**:
```
Input: List of (prompt, output_length) pairs
Process: Extract 8 features, fit Ridge model
Output: Trained model ready for prediction
```

**Prediction**:
```
Input: Prompt text
Process: Extract 8 features → predict(features)
Output: Predicted output length (tokens)
```

**Characteristics**:
- O(1) inference time (~5-50ms)
- Deterministic (same prompt → same prediction)
- No randomness
- Small model (< 1MB)

### 3. SmartLoadBalancer

**Routing Strategies**:

#### Round-Robin
```
Route to: workers[index % len(workers)]
Update: index += 1
Pros: Simple, fair distribution
Cons: Ignores request complexity
```

#### Least-Loaded
```
Route to: worker with min(load_score)
Load Score: 0.7*current_requests + 0.3*avg_latency
Pros: Balances load
Cons: Overhead of calculating all scores
```

#### Predicted Cost (Default)
```
IF predicted_length > 500:
  Route via: LEAST_LOADED
ELSE:
  Route via: ROUND_ROBIN

Pros: Hybrid - optimization where it matters most
Cons: Threshold is manual (500 tokens)
```

### 4. WorkerHealth

**Tracked Metrics**:
```python
class WorkerHealth:
    worker_id: str
    healthy: bool                 # Last health check result
    current_requests: int         # Active requests now
    total_requests: int           # Lifetime counter
    total_latency: float          # Sum of all latencies
    last_heartbeat: float         # Timestamp of last check
    error_count: int              # Failed requests

    @property
    def avg_latency(self) -> float:
        return total_latency / max(total_requests, 1)

    @property
    def load_score(self) -> float:
        return min(
            (current_requests / 10.0) * 0.7 +
            (avg_latency / 5000.0) * 0.3,
            1.0
        )
```

### 5. InferenceRequest

**Lifecycle**:
```
1. Created: request_id, prompt, predicted_output_length, submitted_at
2. Routed: routed_to_worker assigned
3. Processed: completed_at set after worker responds
4. Metrics: latency = completed_at - submitted_at
```

## Data Flow

```
User Request
    │
    └─→ POST /infer {prompt}
        │
        ├─→ Extract Features (8 features)
        │
        ├─→ Ridge Model.predict(features)
        │   → predicted_length (float)
        │
        ├─→ Create InferenceRequest
        │   request_id, prompt, predicted_length, submitted_at
        │
        ├─→ Route Request
        │   IF predicted_length > 500:
        │     → Route to least-loaded worker
        │   ELSE:
        │     → Round-robin
        │
        ├─→ Get healthy workers
        │   Filter workers with health.healthy == True
        │
        ├─→ Send to Worker
        │   POST http://{worker}/infer {prompt}
        │   Track: routed_to_worker, time
        │
        ├─→ Receive Response
        │   output, tokens_generated, processing_time_ms
        │
        ├─→ Record Metrics
        │   completed_at = now()
        │   latency = completed_at - submitted_at
        │   worker_health.total_latency += latency
        │   completed_requests.append(request)
        │
        └─→ Return Response
            {request_id, output, worker_id, predicted_length,
             actual_length, latency_ms}
```

## Async Concurrency

**Request Handling**:
```python
async def infer(request):
    # All I/O is non-blocking
    predicted_length = predictor.predict(prompt)  # <50ms
    response = await lb.process_request(req, worker)  # Awaitable
    return response
```

**Health Checking**:
```python
# Runs in background, doesn't block requests
async def health_check_worker(worker_id):
    while True:
        try:
            await client.get(f"http://{worker_id}/health")
            # Update health status
        except:
            worker.healthy = False
        await asyncio.sleep(5)  # Non-blocking sleep
```

## Metrics Collection

**Real-Time Metrics** (`GET /metrics`):
```json
{
  "total_requests": 456,
  "p50_latency": 145.23,
  "p95_latency": 320.45,
  "p99_latency": 485.67,
  "avg_latency": 178.90,
  "throughput_rps": 3.80,
  "worker_health": {
    "worker-1": {
      "healthy": true,
      "current_load": 2,
      "avg_latency": 175.50,
      "error_count": 0
    },
    ...
  }
}
```

**Percentile Calculation**:
```python
latencies = [r.latency for r in completed_requests if r.latency]
p50 = percentile(latencies, 50)  # Median
p95 = percentile(latencies, 95)  # 95th percentile
p99 = percentile(latencies, 99)  # 99th percentile
```

## Performance Characteristics

| Component | Time | Notes |
|-----------|------|-------|
| Feature extraction | 5-10ms | 8 string operations |
| Ridge prediction | 40-50ms | Matrix multiplication |
| Routing decision | <1ms | O(n) worker check |
| Health check | ~100ms | Network I/O (async) |
| Total overhead | 50-60ms | Before reaching worker |

## Failure Scenarios

### Worker Unavailable
```
→ Detected by health check (5-10s delay)
→ Worker marked unhealthy
→ Routed requests get RuntimeError
→ Client sees 503 Service Unavailable
→ Recovery: Worker comes back online
```

### Predictor Not Trained
```
→ predict() returns default 100.0
→ All requests treated as low-cost
→ Load balancer uses round-robin
→ No crash, graceful degradation
```

### All Workers Down
```
→ get_healthy_workers() returns []
→ route_request() raises RuntimeError
→ Inference returns 503 error
→ Metric collection continues
```

## Scaling Considerations

**Horizontal Scaling** (add workers):
- No central bottleneck
- Load balancer scales with number of workers
- Each worker is independent

**Vertical Scaling** (more load):
- Async design handles many concurrent users
- Feature extraction is cheap
- Health checks don't interfere with requests

**Prediction Model Scaling**:
- Ridge model is O(1) prediction
- No hyperparameter tuning needed
- Retraining done offline
