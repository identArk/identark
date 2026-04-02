# Contributing to identark-sdk

Thank you for considering a contribution. This document explains how to get involved.

---

## Before you start

For anything beyond a small bug fix, **please open an issue first**. Describe what you want to change and why. This saves everyone time — we can agree on the approach before you invest in writing code.

---

## Development setup

```bash
git clone https://github.com/identark/sdk.git
cd identark-sdk
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

---

## Running tests

```bash
# Unit tests — no network or LLM calls required
pytest tests/unit/ -v

# With coverage
pytest tests/unit/ --cov=identark --cov-report=term-missing

# Type checking
mypy identark/

# Linting
ruff check identark/ tests/
```

All CI checks must pass before a PR is merged.

---

## What we need

| Area | What helps |
|---|---|
| Gateway implementations | Adapters for AWS Bedrock, Azure OpenAI, Vertex AI |
| Framework integrations | LangChain, LlamaIndex, CrewAI, AutoGen, Haystack adapters |
| Bug reports | Reproducible issues with clear steps, expected vs actual behaviour |
| Documentation | Corrections, clarifications, additional examples |
| Test coverage | Edge cases and error scenarios |
| Performance | Profiling and optimisation of hot paths |

---

## Pull request guidelines

1. Open an issue first for any significant change
2. Write or update tests for changed behaviour — coverage must not decrease
3. Run the full test suite locally before submitting
4. Follow the existing code style — ruff enforces this via pre-commit
5. Write a clear PR description: motivation, approach, testing notes
6. Reference the related issue number in the PR title

---

## Code style

- **Python 3.10+** with full type annotations
- **ruff** for linting and formatting (pre-commit enforces this)
- **mypy strict** — all public APIs must be fully typed
- Docstrings on all public classes and methods
- No bare `except:` — always catch specific exceptions

---

## Commit message format

```
type: short description

Longer explanation if needed. Reference issue numbers: #123
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
