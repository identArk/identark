"""
identark.pricing
~~~~~~~~~~~~~~~~~
LLM token pricing tables with support for external overrides.

Pricing can be customized via:
1. Environment variable: IDENTARK_PRICING_URL (fetched on first use)
2. Local file: ~/.identark/pricing.json
3. Programmatic override: set_pricing_table()

Falls back to bundled defaults if no override is found.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger("identark.pricing")


class ModelPricing(TypedDict):
    input: float   # USD per 1M tokens
    output: float  # USD per 1M tokens


# Bundled defaults — updated periodically with SDK releases
_BUNDLED_PRICING: dict[str, ModelPricing] = {
    # OpenAI
    "gpt-4o":            {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":       {"input": 0.15,  "output": 0.60},
    "gpt-4-turbo":       {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo":     {"input": 0.50,  "output": 1.50},
    "o1":                {"input": 15.00, "output": 60.00},
    "o1-mini":           {"input": 3.00,  "output": 12.00},
    # Anthropic
    "claude-3-5-sonnet-20241022": {"input": 3.00,  "output": 15.00},
    "claude-3-5-haiku-20241022":  {"input": 0.80,  "output": 4.00},
    "claude-3-opus-20240229":     {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-20250514":   {"input": 3.00,  "output": 15.00},
    "claude-opus-4-20250514":     {"input": 15.00, "output": 75.00},
    # Mistral (EU)
    "mistral-large-latest":  {"input": 2.00, "output": 6.00},
    "mistral-small-latest":  {"input": 0.20, "output": 0.60},
    "open-mistral-nemo":     {"input": 0.15, "output": 0.15},
    "codestral-latest":      {"input": 0.20, "output": 0.60},
    # Google Gemini
    "gemini-1.5-pro":        {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash":      {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash-exp":  {"input": 0.10, "output": 0.40},
}

# Active pricing table — starts as bundled, can be overridden
_pricing_table: dict[str, ModelPricing] = _BUNDLED_PRICING.copy()
_initialized: bool = False


def _load_local_overrides() -> dict[str, ModelPricing] | None:
    """Load pricing from ~/.identark/pricing.json if it exists."""
    config_path = Path.home() / ".identark" / "pricing.json"
    if not config_path.exists():
        return None
    try:
        with config_path.open() as f:
            data: dict[str, ModelPricing] = json.load(f)
            logger.info("Loaded pricing overrides from %s", config_path)
            return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load %s: %s", config_path, exc)
        return None


def _fetch_remote_pricing(url: str) -> dict[str, ModelPricing] | None:
    """Fetch pricing from a remote URL (sync, best-effort)."""
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=5) as resp:
            data: dict[str, ModelPricing] = json.loads(resp.read().decode())
            logger.info("Fetched pricing from %s", url)
            return data
    except Exception as exc:
        logger.warning("Failed to fetch pricing from %s: %s", url, exc)
        return None


def _initialize() -> None:
    """Initialize pricing table from overrides if available."""
    global _pricing_table, _initialized
    if _initialized:
        return

    # Priority: env URL > local file > bundled
    if url := os.environ.get("IDENTARK_PRICING_URL"):
        if remote := _fetch_remote_pricing(url):
            _pricing_table = {**_BUNDLED_PRICING, **remote}
    elif local := _load_local_overrides():
        _pricing_table = {**_BUNDLED_PRICING, **local}

    _initialized = True


def get_pricing(model: str) -> ModelPricing | None:
    """Get pricing for a model, or None if unknown."""
    _initialize()
    return _pricing_table.get(model)


def set_pricing_table(table: dict[str, ModelPricing]) -> None:
    """Programmatically override the pricing table."""
    global _pricing_table, _initialized
    _pricing_table = {**_BUNDLED_PRICING, **table}
    _initialized = True


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    provider: str = "openai",
) -> float:
    """
    Estimate cost in USD for a given model and token counts.

    Returns 0.0 for local providers (Ollama, self-hosted models).
    Returns a conservative estimate for unknown models.
    """
    if provider == "local":
        return 0.0

    pricing = get_pricing(model)
    if pricing is None:
        # Unknown model — conservative fallback ($10/1M tokens)
        logger.debug("Unknown model %s, using fallback pricing", model)
        return (input_tokens + output_tokens) * 0.000_010

    return (
        input_tokens * pricing["input"] + output_tokens * pricing["output"]
    ) / 1_000_000


def list_known_models() -> list[str]:
    """Return list of models with known pricing."""
    _initialize()
    return list(_pricing_table.keys())
