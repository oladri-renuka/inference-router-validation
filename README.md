# Inference Router Validation

Experimental validation of predicted-cost routing for LLM inference: does predicting output token count before inference allow better load balancing than round-robin?

## Status

**Experimental validation only.** Single unseeded run (August 2026), not independently replicated. Not production-ready code.

## Hypothesis

Ridge regression model predicts output token counts from prompts. Requests predicted to exceed a threshold (500 tokens) routed to least-loaded worker; others use round-robin. Hypothesis: this improves latency compared to round-robin baseline.

## Results

**Mean latency**: 17% improvement (2947ms → 2447ms, p=0.0152) on this run.

**P95 tail latency**: 0.2% improvement (6070ms → 6056ms), not statistically tested. Operationally unchanged.

**Predictor accuracy**: R²=0.114. Weak predictive signal; 89% of output-length variance unexplained.

**Threshold validation**: Missing. Whether the 500-token threshold actually separates long-output requests is unvalidated.

**Reproducibility**: Single point estimate from unseeded run. Second independent run needed to verify these results are stable.

## What This Means

- Average-case latency can improve via predicted-cost routing (one data point)
- Tail latency (P95/P99) does not improve in this validation
- Predictor is too weak to claim routing strategy works as designed
- Results may not replicate on second run
- Unsuitable for strict SLAs or production deployment without further validation

## Key Limitations

1. **Single run**: No independent replication
2. **Weak predictor**: R²=0.114 (threshold effectiveness unvalidated)
3. **P95 unchanged**: Not suitable for tail-latency SLAs
4. **Simulated environment**: Single GPU, 3 workers (not distributed cluster)
5. **Absolute latencies**: From conservative vLLM config (enforce_eager=true)
6. **Single model/dataset**: Mistral 7B + Alpaca only

## Documentation

**Start here**: [DOCUMENTATION.md](DOCUMENTATION.md) — Technical reference on components, methodology, results interpretation, and all limitations.

**Methodology**: [VALIDATION_PLAN.md](VALIDATION_PLAN.md) — What was tested, success criteria, what succeeded and failed.

**Older docs**: [`docs/source/`](docs/source/) — Previous architectural documentation (may be outdated).

## Implementation

The codebase contains implementations of:

- Ridge regression predictor for output length
- Load balancer with round-robin, least-loaded, and predicted-cost strategies
- Statistics module for A/B testing with t-tests and confidence intervals
- Validation notebook that orchestrates the full experiment

See DOCUMENTATION.md for component details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT - see [LICENSE](LICENSE) file

---

**[→ Start with documentation](docs/source/index.md)**
