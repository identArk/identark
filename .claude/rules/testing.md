---
paths:
  - "tests/**/*.py"
---

# Testing Conventions

## MockGateway Usage

Always use `credseal.testing.MockGateway` when testing code that calls the gateway:

```python
from credseal.testing import MockGateway

async def test_my_agent_tool():
    gateway = MockGateway()
    gateway.register_response(
        tool="openai.chat",
        response={"choices": [{"message": {"content": "Hello"}}]}
    )
    result = await my_agent_function(gateway=gateway)
    assert result == "Hello"
    assert gateway.call_count("openai.chat") == 1
```

## Fixtures

Standard fixtures are in `tests/conftest.py`:
- `mock_gateway` — a pre-configured MockGateway instance
- `sample_agent_config` — a valid AgentConfig with test credentials
- `sample_org_id` — a fixed UUID for multi-tenant tests

## Test Naming

- `test_<function>_<scenario>` — e.g., `test_execute_raises_on_invalid_agent`
- `test_<function>_success` for the happy path
- `test_<function>_when_<condition>` for edge cases

## What Makes a Good Test

- Tests one thing. If you need to assert more than 3 things, split the test.
- Uses `pytest.raises` with `match=` to verify exception messages, not just types.
- Does NOT mock internal implementation details — only external I/O (HTTP, DB, vault).
- Integration tests are marked `@pytest.mark.integration` and skipped in CI by default.
- Never use `time.sleep()`. Use `pytest-asyncio` and proper async patterns.
