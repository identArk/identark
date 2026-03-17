---
paths:
  - "cloud/**/*.py"
---

# Cloud API — FastAPI Conventions

## Endpoint Design

- All endpoints live under `/v1/` prefix
- Use kebab-case for URL paths: `/v1/gateway/execute`, `/v1/agents/register`
- Always include `org_id` scope on every endpoint (injected from JWT, not from request body)
- Standard response envelope:
  ```json
  { "data": {}, "execution_id": "uuid", "timestamp": "iso8601" }
  ```
- Error response envelope:
  ```json
  { "error": { "code": "AGENT_NOT_FOUND", "message": "human readable" }, "execution_id": "uuid" }
  ```

## Database (SQLAlchemy async)

- All DB operations are async using `AsyncSession`
- Every query MUST filter by `org_id` — no cross-tenant queries allowed
- Use Supabase Row Level Security as a second enforcement layer
- Migrations via Alembic. Never modify schema outside of a migration file.
- Table naming: snake_case plural — `audit_logs`, `agent_registrations`, `organisations`

## Multi-Tenancy Checklist

Before shipping any new endpoint, verify:
- [ ] `org_id` is extracted from JWT, not from request body
- [ ] All DB queries include `.where(Model.org_id == org_id)`
- [ ] No cross-org data can be inferred from error messages or response timing
- [ ] Rate limiting is scoped per `org_id`, not per IP

## Billing (Stripe)

- Increment usage counter on every successful `gateway/execute` call
- Use Stripe Meters API for usage-based billing
- Meter event name: `gateway_execution`
- Free tier limit: 500 executions/month per org. Enforce at gateway layer, not just UI.
