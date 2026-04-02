"""Unit tests for identark.exceptions."""

from identark.exceptions import (
    AuthenticationError,
    ConfigurationError,
    ControlPlaneError,
    CostCapExceededError,
    IdentArkError,
    FileError,
    GatewayError,
    LLMError,
    NetworkError,
    PathNotAllowedError,
    PresignedURLExpiredError,
    RateLimitError,
    SessionNotFoundError,
)


class TestExceptionHierarchy:
    def test_gateway_error_is_identark_error(self):
        assert issubclass(GatewayError, IdentArkError)

    def test_control_plane_error_is_gateway_error(self):
        assert issubclass(ControlPlaneError, GatewayError)

    def test_authentication_error_is_control_plane_error(self):
        assert issubclass(AuthenticationError, ControlPlaneError)

    def test_cost_cap_exceeded_is_control_plane_error(self):
        assert issubclass(CostCapExceededError, ControlPlaneError)

    def test_session_not_found_is_control_plane_error(self):
        assert issubclass(SessionNotFoundError, ControlPlaneError)

    def test_network_error_is_gateway_error(self):
        assert issubclass(NetworkError, GatewayError)

    def test_llm_error_is_identark_error(self):
        assert issubclass(LLMError, IdentArkError)

    def test_rate_limit_is_llm_error(self):
        assert issubclass(RateLimitError, LLMError)

    def test_file_error_is_identark_error(self):
        assert issubclass(FileError, IdentArkError)

    def test_path_not_allowed_is_file_error(self):
        assert issubclass(PathNotAllowedError, FileError)

    def test_configuration_error_is_identark_error(self):
        assert issubclass(ConfigurationError, IdentArkError)


class TestExceptionAttributes:
    def test_cost_cap_exceeded_attributes(self):
        exc = CostCapExceededError("Cap hit", cap_usd=1.0, consumed_usd=1.05, session_id="s1")
        assert exc.cap_usd == 1.0
        assert exc.consumed_usd == 1.05
        assert exc.session_id == "s1"
        assert exc.status_code == 402

    def test_authentication_error_attributes(self):
        exc = AuthenticationError("Bad token", session_id="sess_123", reason="expired")
        assert exc.session_id == "sess_123"
        assert exc.reason == "expired"
        assert exc.status_code == 401

    def test_rate_limit_attributes(self):
        exc = RateLimitError("Too many requests", retry_after_seconds=30, provider="openai")
        assert exc.retry_after_seconds == 30
        assert exc.provider == "openai"

    def test_network_error_attributes(self):
        exc = NetworkError("Timeout", attempts=3, last_status_code=503)
        assert exc.attempts == 3
        assert exc.last_status_code == 503

    def test_path_not_allowed_message(self):
        exc = PathNotAllowedError("/etc/passwd")
        assert "/etc/passwd" in str(exc)
        assert "/workspace/" in str(exc)
        assert exc.attempted_path == "/etc/passwd"

    def test_session_not_found_message(self):
        exc = SessionNotFoundError("sess_abc")
        assert "sess_abc" in str(exc)
        assert exc.session_id == "sess_abc"
        assert exc.status_code == 404

    def test_presigned_url_expired_message(self):
        exc = PresignedURLExpiredError("/workspace/file.txt", "2026-01-01T00:00:00Z")
        assert "/workspace/file.txt" in str(exc)
        assert exc.file_path == "/workspace/file.txt"
        assert exc.expired_at == "2026-01-01T00:00:00Z"


class TestCatchingBroadly:
    def test_catch_all_with_identark_error(self):
        exceptions = [
            AuthenticationError("test"),
            CostCapExceededError("test"),
            RateLimitError("test"),
            PathNotAllowedError("/bad/path"),
            NetworkError("test"),
            ConfigurationError("test"),
        ]
        for exc in exceptions:
            assert isinstance(exc, IdentArkError)
