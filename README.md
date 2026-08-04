# Smart Load Balancer for LLM Inference

Intelligent request routing for LLM inference based on predicted output length.

**Validation Results** (August 2026): Mean latency improved 17% (501ms, p=0.0152). P95 tail latency unchanged (14ms improvement, not statistically validated).

**[Documentation →](DOCUMENTATION.md) | [Validation Plan →](VALIDATION_PLAN.md) | [Full Docs →](docs/source/index.md)**

## Quick Start

```bash
pip install -r requirements.txt
python benchmarks/simple_benchmark.py
```

## Validation Results

- Mean latency: 17% improvement (2947ms → 2447ms, p=0.0152)
- P95 latency: 0.2% improvement (not statistically tested)
- Predictor: R²=0.114 (weak predictive signal)
- Sample size: 250 requests per strategy
- Dataset: 500 Alpaca prompts on Mistral 7B

## What It Does

Routes high-complexity LLM requests to underutilized workers by predicting output length before dispatch. Strategy: if predicted_tokens > threshold, route to least-loaded worker; otherwise use round-robin.

## Key Components

- **Prediction** - Ridge regression predicts output tokens from prompts
- **Load Balancer** - Routes requests intelligently
- **Health Monitor** - Async worker health checks
- **Metrics** - Real-time latency & throughput tracking

## Documentation

Full documentation available in [`docs/source/`](docs/source/):

- [Getting Started](docs/source/getting_started.md) - Setup & first run
- [Architecture](docs/source/architecture.md) - System design
- [Benchmarks](docs/source/benchmarks.md) - Performance results
- [Benchmarking Guide](docs/source/benchmarking_guide.md) - Run your own tests

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT - see [LICENSE](LICENSE) file

---

**[→ Start with documentation](docs/source/index.md)**
