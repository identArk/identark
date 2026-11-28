"""Gemini Flash pricing reflects the 2027-01-01 rate increase.

Google doubles the paid-tier, standard, text rates for gemini-3.8-flash and
gemini-3.6-flash from 1 Jan 2027. Cost estimates must use the new rates so a
default-model estimate is not half the real cost.
"""

from identark.integrations.gemini import _GEMINI_PRICING, _estimate_gemini_cost
from identark.pricing import _BUNDLED_PRICING

FLASH_MODELS = ("gemini-3.8-flash", "gemini-3.6-flash")
EXPECTED = {"input": 1.50, "output": 7.50}


def test_gemini_flash_2027_rates_in_pricing_tables():
    for model in FLASH_MODELS:
        assert _GEMINI_PRICING[model] == EXPECTED
        assert _BUNDLED_PRICING[model] == EXPECTED


def test_gemini_flash_2027_cost_estimate():
    # 1M input + 1M output tokens == input rate + output rate.
    for model in FLASH_MODELS:
        assert _estimate_gemini_cost(model, 1_000_000, 1_000_000) == 9.00