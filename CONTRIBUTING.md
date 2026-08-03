# Contributing

Contributions are welcome! Here's how to get started.

## Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/smart-load-balancer.git
cd smart-load-balancer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies with dev requirements
pip install -r requirements.txt
pip install pytest pytest-cov  # For testing
```

## Running Tests

```bash
# Run all tests
python tests/test_predictor.py
python tests/test_load_balancer.py
python tests/test_integration.py

# Or with pytest
pytest tests/

# Run specific test
pytest tests/test_predictor.py::test_feature_extraction
```

## Running Benchmarks

```bash
# Simple benchmark
python benchmarks/simple_benchmark.py

# Comparison benchmark
python benchmarks/compare_strategies.py

# Full benchmark with Locust
locust -f benchmarks/locustfile.py --host=http://127.0.0.1:8000 --headless
```

## Code Style

- Follow PEP 8
- Use meaningful variable names
- Add docstrings to functions
- Keep functions small and focused

## Making Changes

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make your changes
3. Run tests to ensure nothing breaks: `pytest tests/`
4. Commit with clear message: `git commit -m "Add feature: description"`
5. Push to your fork: `git push origin feature/my-feature`
6. Create a Pull Request

## Commit Messages

- Use clear, descriptive messages
- Reference issues if applicable: `Fixes #123`
- Start with imperative mood: "Add", "Fix", "Update" (not "Added", "Fixed")

## Testing Requirements

- All new code must have tests
- Tests must pass before PR is merged
- Aim for >90% code coverage

## Documentation

- Update relevant docs in `docs/source/`
- Add examples if adding new functionality
- Update BENCHMARK_RESULTS.md if changing performance-related code

## Questions?

- Check the [documentation](docs/source/index.md)
- Review [architecture](docs/source/architecture.md) for system design
- Open an issue to discuss larger changes

## Code of Conduct

- Be respectful
- Give credit
- Help others
- Report issues constructively

Thanks for contributing!
