# IdentArk Docs (Mintlify)

Public documentation and API reference for the IdentArk control plane.

## Structure

```
docs-site/
├── docs.json                     # Mintlify config (nav, theme, tabs)
├── introduction.mdx              # Landing
├── quickstart.mdx
├── concepts.mdx
├── authentication.mdx
├── sdks/{python,typescript}.mdx
├── guides/                       # production, mcp-hitl, acs, limits-and-errors, security
├── api-reference/
│   ├── introduction.mdx
│   └── openapi.json              # Generated from the FastAPI app — see below
└── images/                       # logo + favicon
```

## Develop locally

```bash
npm i -g mint     # Mintlify CLI
cd docs-site
mint dev          # http://localhost:3000
```

The **API Reference → Endpoints** group is auto-generated from
`api-reference/openapi.json`; no per-endpoint MDX to maintain.

## Regenerate the OpenAPI spec

The spec is the FastAPI app's own schema, post-processed to add the production server
and a bearer-auth scheme. From the repo root:

```bash
cloud/.venv/bin/python - <<'PY'
import os, json
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://u:p@localhost/db")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("SECRET_KEY", "dev-secret")
os.environ.setdefault("FIREBASE_API_KEY", "x")
import sys; sys.path.insert(0, "cloud")
from app.main import app
spec = app.openapi()
spec["servers"] = [{"url": "https://api.identark.io", "description": "Production"}]
spec["components"].setdefault("securitySchemes", {})["bearerAuth"] = {
    "type": "http", "scheme": "bearer",
}
spec["security"] = [{"bearerAuth": []}]
json.dump(spec, open("docs-site/api-reference/openapi.json", "w"), indent=2)
print("paths:", len(spec["paths"]))
PY
```

> Keep `docs.json`'s navigation in sync only when you add new **guide** pages —
> endpoint pages come from the spec automatically.

## Deploy

Connect this repo to Mintlify (mintlify.com) and set the docs root to `docs-site/`.
Pushes to the default branch publish automatically.
