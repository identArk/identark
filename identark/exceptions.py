"""
identark.exceptions
~~~~~~~~~~~~~~~~~~~~~
All exceptions raised by the SDK. Rooted at ``IdentArkError`` so
callers can catch broadly or specifically.

Hierarchy
---------
IdentArkError
├── GatewayError
│   ├── ControlPlaneError
│   │   ├── AuthenticationError
│   │   ├── CostCapExceededError
│   │   ├── SessionNotFoundError
│   │   └── CredentialExportError
│   │       ├── CredentialNotExportableError
│   │       └── CredentialResolveScopeError
│   └── NetworkError
├── LLMError
│   ├── RateLimitError
│   ├── ContentPolicyError
│   └── ProviderError
├── FileError
│   ├── PathNotAllowedError
│   └── PresignedURLExpiredError
└── ConfigurationError
"""

from __future__ import annotations


class IdentArkError(Exception):
    """Base class for all SDK exceptions."""


# ── Gateway ───────────────────────────────────────────────────────────────────


class GatewayError(IdentArkError):
    """Raised when a gateway communication operation fails."""


class ControlPlaneError(GatewayError):
    """
    The control plane returned an error response.

    Attributes:
        status_code:    HTTP status code from the control plane.
        error_code:     Machine-readable error code string.
        message:        Human-readable error description.
        required_scope: For a 403 ``insufficient_scope``, the scope the key lacks.
    """

    def __init__(
        self,
        message: str,
        status_code: int = 0,
        error_code: str = "unknown",
        required_scope: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.message = message
        self.required_scope = required_scope


class AuthenticationError(ControlPlaneError):
    """
    Session token is invalid or has expired.

    Attributes:
        session_id: The session ID associated with the failed token.
        reason:     A short description of why authentication failed.
    """

    def __init__(self, message: str, session_id: str = "", reason: str = "") -> None:
        super().__init__(message, status_code=401, error_code="authentication_failed")
        self.session_id = session_id
        self.reason = reason


class CostCapExceededError(ControlPlaneError):
    """
    The session has reached its configured cost cap.

    Attributes:
        cap_usd:       The configured cost ceiling in USD.
        consumed_usd:  How much has been spent in this session.
        session_id:    The session that hit the cap.
    """

    def __init__(
        self,
        message: str,
        cap_usd: float = 0.0,
        consumed_usd: float = 0.0,
        session_id: str = "",
    ) -> None:
        super().__init__(message, status_code=402, error_code="cost_cap_exceeded")
        self.cap_usd = cap_usd
        self.consumed_usd = consumed_usd
        self.session_id = session_id


class SessionNotFoundError(ControlPlaneError):
    """Session ID does not exist or has already been terminated."""

    def __init__(self, session_id: str) -> None:
        super().__init__(
            f"Session '{session_id}' not found or already terminated.",
            status_code=404,
            error_code="session_not_found",
        )
        self.session_id = session_id


class CredentialExportError(ControlPlaneError):
    """
    The control plane refused to export a credential's plaintext (HTTP 403).

    Brokered credentials are non-exportable by default: the gateway uses them
    server-side, and nothing reads them back. Catch this base class to handle
    both refusal reasons; catch a subclass to tell them apart.

    Attributes:
        path:           The credential path that was requested.
        required_scope: The scope the control plane asked for, if any.
    """

    def __init__(
        self,
        message: str,
        error_code: str,
        path: str = "",
        required_scope: str | None = None,
    ) -> None:
        super().__init__(
            message, status_code=403, error_code=error_code, required_scope=required_scope
        )
        self.path = path


class CredentialNotExportableError(CredentialExportError):
    """The credential is not exportable.

    The gateway can still use it server-side. An org owner can mark it
    exportable from the dashboard if it is genuinely needed for local
    development.
    """


class CredentialResolveScopeError(CredentialExportError):
    """The caller may not export credentials.

    Either the API key lacks the ``credentials:resolve`` scope (``org:admin``
    and presets do not include it — it must be granted by name), or the key is
    bound to an agent, which can never export credentials.
    """


class NetworkError(GatewayError):
    """
    All retry attempts to the control plane were exhausted.

    Attributes:
        attempts:         Number of attempts made before giving up.
        last_status_code: HTTP status from the final attempt, if any.
    """

    def __init__(
        self,
        message: str,
        attempts: int = 0,
        last_status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_status_code = last_status_code


# ── LLM ──────────────────────────────────────────────────────────────────────


class LLMError(IdentArkError):
    """Raised when the LLM provider returns an error."""


class RateLimitError(LLMError):
    """
    The LLM provider's rate limit has been hit.

    Attributes:
        retry_after_seconds: How long to wait before retrying.
        provider:            The provider that issued the rate limit.
    """

    def __init__(
        self,
        message: str,
        retry_after_seconds: int = 60,
        provider: str = "unknown",
    ) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds
        self.provider = provider


class ContentPolicyError(LLMError):
    """The request was blocked by the provider's content policy."""


class ProviderError(LLMError):
    """An unclassified error returned by the LLM provider."""


# ── File ──────────────────────────────────────────────────────────────────────


class FileError(IdentArkError):
    """Raised when a file operation fails."""


class PathNotAllowedError(FileError):
    """
    The requested file path is outside the allowed workspace.

    Attributes:
        attempted_path: The path the agent attempted to access.
        allowed_prefix: The only permitted path prefix.
    """

    def __init__(self, attempted_path: str, allowed_prefix: str = "/workspace/") -> None:
        super().__init__(
            f"Path '{attempted_path}' is not allowed. "
            f"File paths must start with '{allowed_prefix}'."
        )
        self.attempted_path = attempted_path
        self.allowed_prefix = allowed_prefix


class PresignedURLExpiredError(FileError):
    """
    The presigned URL was used after its expiry timestamp.

    Attributes:
        file_path:  The workspace path the URL was for.
        expired_at: ISO 8601 expiry timestamp from the original URL.
    """

    def __init__(self, file_path: str, expired_at: str) -> None:
        super().__init__(
            f"Presigned URL for '{file_path}' expired at {expired_at}. "
            "Request a new URL via gateway.request_file_url()."
        )
        self.file_path = file_path
        self.expired_at = expired_at


# ── Config ────────────────────────────────────────────────────────────────────


class ConfigurationError(IdentArkError):
    """The gateway was misconfigured at initialisation time."""
