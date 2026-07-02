# Changelog

All notable changes to `identark` will be documented here.

This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.2.0] — 2026-04-02

### Changed
- **Rebrand: CredSeal → IdentArk.** Package renamed `credseal-sdk` → `identark`; all classes, docs, and URLs updated (`CredSealChatModel` → `IdentArkChatModel`, etc.)
- Modernized type syntax to Python 3.10+ (`X | Y` unions)

### Added
- `validation` module for input validation
- CI/CD enhancements, Sentry error monitoring, `key_prefix` migration
- n8n adapter (later moved to dedicated repo `identark/n8n-nodes-identark`)
- Benchmarks and community docs (`COMMUNITY.md`, `CONTRIBUTING.md`)

### Fixed
- mypy and ruff errors across tests and integrations
- Gemini workspace tests now use `tmp_path` in CI

---

## [1.1.0] — 2026-03-18

### Added
- **LangChain integration** — `ChatModel` adapter
- **LlamaIndex integration** — LLM adapter
- **CrewAI integration** — adapter
- Mistral EU + Ollama local support (UK Sovereign AI alignment)
- PyPI trusted publishing workflow

---

## [1.0.0] — 2026-02-28

### Added
- `AgentGateway` — the core Protocol interface defining how agents communicate with the outside world
- `DirectGateway` — local development implementation with OpenAI and Anthropic support, and any OpenAI-compatible endpoint (Ollama, Groq, etc.)
- `ControlPlaneGateway` — production implementation that routes all requests through the IdentArk control plane; agents hold zero secrets
- `MockGateway` — test implementation with response queueing and full call recording for assertions
- Full exception hierarchy rooted at `IdentArkError` — `GatewayError`, `ControlPlaneError`, `AuthenticationError`, `CostCapExceededError`, `SessionNotFoundError`, `NetworkError`, `LLMError`, `RateLimitError`, `ContentPolicyError`, `FileError`, `PathNotAllowedError`, `PresignedURLExpiredError`, `ConfigurationError`
- Data models: `Message`, `Role`, `LLMResponse`, `PresignedURL`, `TokenUsage`, `ToolCall`, `Function`
- Built-in cost tracking on every `invoke_llm` call
- Automatic retry with exponential backoff in `ControlPlaneGateway`
- `py.typed` marker — full mypy strict mode compatibility
- Complete type annotations throughout
- Unit test suite with 30+ test cases, zero network calls required
- GitHub Actions CI — tests across Python 3.10, 3.11, 3.12 with automatic PyPI publish on tag

---

[1.2.0]: https://github.com/identark/identark/releases/tag/v1.2.0
[1.1.0]: https://github.com/identark/identark/releases/tag/v1.1.0
[1.0.0]: https://github.com/identark/identark/releases/tag/v1.0.0
