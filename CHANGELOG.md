# Changelog

All notable changes to `credseal-sdk` will be documented here.

This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] — 2026-02-28

### Added
- `AgentGateway` — the core Protocol interface defining how agents communicate with the outside world
- `DirectGateway` — local development implementation with OpenAI and Anthropic support, and any OpenAI-compatible endpoint (Ollama, Groq, etc.)
- `ControlPlaneGateway` — production implementation that routes all requests through the CredSeal control plane; agents hold zero secrets
- `MockGateway` — test implementation with response queueing and full call recording for assertions
- Full exception hierarchy rooted at `CredSealError` — `GatewayError`, `ControlPlaneError`, `AuthenticationError`, `CostCapExceededError`, `SessionNotFoundError`, `NetworkError`, `LLMError`, `RateLimitError`, `ContentPolicyError`, `FileError`, `PathNotAllowedError`, `PresignedURLExpiredError`, `ConfigurationError`
- Data models: `Message`, `Role`, `LLMResponse`, `PresignedURL`, `TokenUsage`, `ToolCall`, `Function`
- Built-in cost tracking on every `invoke_llm` call
- Automatic retry with exponential backoff in `ControlPlaneGateway`
- `py.typed` marker — full mypy strict mode compatibility
- Complete type annotations throughout
- Unit test suite with 30+ test cases, zero network calls required
- GitHub Actions CI — tests across Python 3.10, 3.11, 3.12 with automatic PyPI publish on tag

---

[1.0.0]: https://github.com/Goldokpa/Sandcastle/releases/tag/v1.0.0
