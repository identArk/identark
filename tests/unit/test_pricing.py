"""Tests for identark.pricing module."""

from identark.pricing import estimate_cost, get_pricing, list_known_models, set_pricing_table


class TestPricing:
    """Test pricing estimation functions."""

    def test_known_model_pricing(self) -> None:
        """Known models return expected pricing structure."""
        pricing = get_pricing("gpt-4o")
        assert pricing is not None
        assert "input" in pricing
        assert "output" in pricing
        assert pricing["input"] == 2.50
        assert pricing["output"] == 10.00

    def test_unknown_model_returns_none(self) -> None:
        """Unknown models return None."""
        pricing = get_pricing("unknown-model-xyz")
        assert pricing is None

    def test_estimate_cost_known_model(self) -> None:
        """Cost estimation works for known models."""
        # gpt-4o: $2.50/1M input, $10.00/1M output
        cost = estimate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
        expected = (1000 * 2.50 + 500 * 10.00) / 1_000_000
        assert abs(cost - expected) < 0.000001

    def test_estimate_cost_local_provider(self) -> None:
        """Local provider always returns zero cost."""
        cost = estimate_cost(
            "any-model",
            input_tokens=1000000,
            output_tokens=1000000,
            provider="local",
        )
        assert cost == 0.0

    def test_estimate_cost_unknown_model_fallback(self) -> None:
        """Unknown models use fallback pricing."""
        cost = estimate_cost("unknown-model", input_tokens=1000, output_tokens=500)
        # Fallback is $10/1M tokens
        expected = (1000 + 500) * 0.000_010
        assert abs(cost - expected) < 0.000001

    def test_list_known_models(self) -> None:
        """List of known models includes expected entries."""
        models = list_known_models()
        assert "gpt-4o" in models
        assert "gpt-4o-mini" in models
        assert "claude-3-5-sonnet-20241022" in models
        assert "mistral-large-latest" in models

    def test_set_pricing_table_override(self) -> None:
        """Custom pricing table overrides defaults."""
        set_pricing_table({"custom-model": {"input": 1.0, "output": 2.0}})
        pricing = get_pricing("custom-model")
        assert pricing is not None
        assert pricing["input"] == 1.0
        assert pricing["output"] == 2.0

        # Original models still available
        assert get_pricing("gpt-4o") is not None

    def test_anthropic_pricing(self) -> None:
        """Anthropic models have correct pricing."""
        pricing = get_pricing("claude-3-5-sonnet-20241022")
        assert pricing is not None
        assert pricing["input"] == 3.00
        assert pricing["output"] == 15.00

    def test_mistral_pricing(self) -> None:
        """Mistral models have correct pricing."""
        pricing = get_pricing("mistral-large-latest")
        assert pricing is not None
        assert pricing["input"] == 2.00
        assert pricing["output"] == 6.00
