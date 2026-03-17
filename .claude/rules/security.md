---
paths:
  - "credseal/**/*.py"
  - "cloud/**/*.py"
---

# Security Rules — Credential Isolation

These rules apply to ALL Python files in the credseal and cloud layers.

## Credential Handling

- Credentials MUST be typed as `pydantic.SecretStr`, never plain `str`
- Never pass credential values as positional arguments — use keyword arguments only
- Never include credential values in f-strings, log statements, or exception messages
- When serialising models to dict/JSON, always use `model.model_dump(exclude={"credential"})`
- If you see a raw string that looks like an API key being passed around, STOP and refactor it
  through the gateway pattern before continuing

## Vault Integration

- All reads from the credential vault go through `credseal.cloud.vault.VaultClient`
- VaultClient methods are async. Never call them synchronously.
- Vault paths follow the pattern: `secret/orgs/{org_id}/agents/{agent_id}/tools/{tool_name}`
- Vault tokens must be rotated. Never hardcode vault tokens in config files.

## Audit Logging

- Every call to `gateway.execute()` MUST produce an `AuditLogEntry`
- AuditLogEntry fields: `execution_id`, `agent_id`, `org_id`, `tool`, `timestamp`,
  `success`, `latency_ms`, `error_code` (if failed)
- NEVER include `params` or `result` in the audit log — only metadata
- Audit logs are append-only. There is no update or delete path.
