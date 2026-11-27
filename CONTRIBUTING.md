# Contributing to identark

Thank you for considering a contribution. This document explains how to get involved.

---

## Finding something to work on

Issues labelled [`good first issue`](https://github.com/identArk/identark/labels/good%20first%20issue)
say what to change and which files to change. **Comment on the issue to claim it** before you
start, so two people don't write the same patch.

For anything beyond a small bug fix, **please open an issue first**. Describe what you want to
change and why. This saves everyone time — we can agree on the approach before you invest in
writing code.

**What to expect from us:** a first response within **3 working days**. If we are slower than
that, say so on the thread. It means we missed it, not that your change was rejected.

---

## Development setup

**Supported Python: 3.10, 3.11 and 3.12.** These are the versions CI tests. The package declares
`requires-python = ">=3.10"`, but 3.13 is not yet tested and some optional dependencies may not
install on it — if you are on 3.13 and the install fails, that is the likely cause, and dropping
to 3.12 is the quickest fix.

```bash
git clone https://github.com/identArk/identark.git
cd identark
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

`[dev]` installs every optional provider and framework (OpenAI, Anthropic, Mistral, Gemini,
LangChain, LlamaIndex, LangGraph) plus the test tooling. It is a large install and the one most
likely to fail on an unusual platform.

**If you are only changing documentation or an example**, you do not need all of that:

```bash
pip install -e . && pip install pytest pytest-asyncio ruff
```

That is enough to run the unit tests and the linter.

---

## Running tests

These are the exact commands CI runs. If they pass locally, CI should agree.

```bash
# Unit tests — no network, no LLM calls, no control plane
pytest tests/unit/ -v

# With coverage
pytest tests/unit/ --cov=identark --cov-report=term-missing

# Type checking
mypy identark/

# Linting
ruff check identark/ tests/
```

### A warning about plain `pytest`

**Running `pytest` with no path also runs `tests/integration/`, and those tests currently fail —
around 40 of them — unless the mock control plane in `tests/integration/mock_server.py` is
running.** They fail on an unmodified `main` too, so if you see them, you have not broken
anything.

CI only runs `tests/unit/`, which is why it stays green. Use `pytest tests/unit/` for day-to-day
work, and see `tests/integration/README.md` if you want to run the integration suite properly.

This is a wart, not a design decision. [#19](https://github.com/identArk/identark/issues/19)
tracks making those tests *skip* rather than fail when their dependency is absent, and it is
labelled `good first issue` if you would like to fix it.

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

Note that the Python and TypeScript SDKs are kept at **parity**: a change to one is normally
mirrored in [identark-js](https://github.com/identArk/identark-js). You do not have to write both
halves — say in your pull request which half you have done.

Both SDKs have **zero runtime dependencies** by design. A pull request that adds one will be asked
to remove it; optional extras for a specific provider or framework are fine.

---

## Pull request guidelines

1. Open an issue first for any significant change.
2. Write or update tests for changed behaviour — coverage must not decrease.
3. Run `pytest tests/unit/`, `mypy identark/` and `ruff check identark/ tests/` before submitting.
4. Follow the existing code style — ruff enforces this via pre-commit.
5. Write a clear PR description: motivation, approach, testing notes.
6. Reference the related issue number in the PR title.
7. Never include a secret, token, customer record or signed URL — not in code, tests, fixtures or
   logs.

Pull requests from forks need a maintainer to approve the CI run before it starts, so there may be
a short wait before you see checks.

---

## Commit sign-off

Every commit must be signed off under the
[Developer Certificate of Origin](https://developercertificate.org/):

```bash
git commit -s -m "fix: concise description"
```

This adds a `Signed-off-by:` line, which must match your commit author. It certifies that you
wrote the change, or have the right to submit it under the MIT licence. To sign off commits you
have already made, run `git rebase --signoff main` and force-push your branch.

---

## Code style

- **Python 3.10+** with full type annotations
- **ruff** for linting and formatting (pre-commit enforces this)
- **mypy strict** — all public APIs must be fully typed
- Docstrings on all public classes and methods
- No bare `except:` — always catch specific exceptions

A note on `ruff format`: running it across the whole repository currently rewrites some
pre-existing files. Format only the files you touched, so your diff stays reviewable.

---

## Commit message format

```
type: short description

Longer explanation if needed. Reference issue numbers: #123
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`

---

## Code of conduct

Participation is covered by the [IdentArk Code of
Conduct](https://github.com/identArk/.github/blob/main/CODE_OF_CONDUCT.md).

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
