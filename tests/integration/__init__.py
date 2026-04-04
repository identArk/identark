"""
Integration tests for the IdentArk SDK.

These tests verify the SDK's interaction with the control plane API and
DirectGateway functionality. They use a mock control plane server to simulate
the real API without requiring network access.

To run integration tests:

    pytest tests/integration/ -v

To run with verbose mocking output:

    pytest tests/integration/ -v -s

Environment Variables:
    IDENTARK_INTEGRATION=1  — Enable integration tests (auto-enabled by fixtures)
"""
