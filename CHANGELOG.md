# Changelog

All notable changes to `identark` will be documented here.

This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Changed
- **Gemini Flash pricing.** `gemini-3.8-flash` and `gemini-3.6-flash` now use the
  2027-01-01 paid-tier rates of $1.50 input / $7.50 output per 1M tokens
  (previously $0.75 / $3.75), so default-model cost estimates do not undercount
  after the price increase.

---

## [1.4.0] — 2026-11-23

### Added
- **Gemini 4 pricing.** `gemini-4-carbon` and `gemini-4-argon` ($4.00 input /
  $20.00 output per 1M tokens, Google's rate from 1 Dec 2026), `gemini-4-neon`
  ($3.00 / $15.00) and `gemini-4-hydrogen` ($1.50 / $7.50) now have their own
  entries in the Gemini integration and the bundled price table. Until 30 Nov
  2026 Google charges $2.00 / $10.00 for Carbon and Argon, so their estimates
  run high until then. Unknown Gemini models are still estimated at the most
  expensive known rate, which is now $4.00 / $20.00 (was $2.00 / $12.00).
- **Typed export errors.** A refused export raises `CredentialNotExportableError`
  (the credential is not exportable) or `CredentialResolveScopeError` (the key
  lacks `credentials:resolve`, or is bound to an agent). Both subclass
  `CredentialExportError` → `ControlPlaneError`, with `path` and
  `required_scope`. Matches `resolveCredential` in the TypeScript SDK.
- `ControlPlaneError.required_scope` carries the missing scope from a 403
  `insufficient_scope`, and error codes inside the control plane's
  `{"detail": {...}}` envelope are now parsed.

### Deprecated
- **`ControlPlaneGateway.resolve_credential`** now emits a `DeprecationWarning`.
  Credentials brokered by the control plane are non-exportable by default: the
  gateway uses them server-side and they are not returned to callers. Export now
  needs an API key holding the explicit `credentials:resolve` scope (not
  included in any preset or implied by `org:admin`, and never available to
  agent-bound keys) **and** a credential an org owner has marked exportable. The
  method keeps its signature for local-development tooling and will be removed
  in a future major release.

---

## [1.3.0] — 2026-11-16

### Added
- **Governed actions.** `ControlPlaneGateway.request_capability(capability,
  provider, payload, idempotency_key)` asks the control plane to act, for
  example a payment, a payment link or a refund, while the provider key stays
  in IdentArk; `get_capability_execution(execution_id)` reads its status and any
  result (such as a checkout link). `CapabilityProvider` covers `stripe`,
  `paystack`, `github` and `aws`. Grants, budgets and human approval apply
  before anything executes, and a reused idempotency key never acts twice.
  Matches `requestCapability` in the TypeScript SDK 1.0.4.
- **Managed tools client.** `ManagedToolsClient` lists managed tools and
  creates, waits on, resumes and cancels executions without the caller ever
  receiving the provider credential.
- **Structured credential sessions.** `ControlPlaneGateway.resolve_credential`
  returns a `CredentialSession` whose `fields` carry structured kinds (for
  example `database`), alongside `kind` and `category`; single-value kinds keep
  `value` as before.
- **Zero-retention history mode.** `ControlPlaneGateway(history="client")` (or
  `IDENTARK_HISTORY_MODE=client`) keeps the conversation in the gateway and
  sends it in full each call with `store_history: false`, so the control plane
  never stores prompts or responses. Assistant tool calls are kept so tool
  loops replay correctly. New `history_mode`, `history` and `clear_history()`.
  Requires a control plane that supports `store_history` (backend PR "session
  history retention").

### Fixed
- `invoke_llm_stream` read `content` from SSE events, but the control plane
  streams `delta`, so streamed text came back empty. Both are now accepted, and
  an `{"error": ...}` event raises `ControlPlaneError` instead of ending silently.
- **Gemini default model and pricing.** `GeminiGateway` now defaults to
  `gemini-3.8-flash` instead of the retired `gemini-1.5-pro`, and both pricing
  tables list current Flash, Flash-Lite and Pro models instead of retired ones.
  Unknown Gemini models are estimated at the most expensive known rate rather than
  the Flash rate (#10).

---

## [1.2.1] — 2026-07-02

### Fixed
- **License metadata corrected to MIT.** The published 1.2.0 wheel carried a
  contradictory AGPL classifier; the SDK is MIT everywhere now (badge,
  classifier, `__license__`, README).
- CI: single PyPI publisher — removed duplicate publish jobs that raced on the
  same tag and could turn releases red.

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

[1.4.0]: https://github.com/identark/identark/releases/tag/v1.4.0
[1.3.0]: https://github.com/identark/identark/releases/tag/v1.3.0
[1.2.1]: https://github.com/identark/identark/releases/tag/v1.2.1
[1.2.0]: https://github.com/identark/identark/releases/tag/v1.2.0
[1.1.0]: https://github.com/identark/identark/releases/tag/v1.1.0
[1.0.0]: https://github.com/identark/identark/releases/tag/v1.0.0