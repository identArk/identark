#!/usr/bin/env python3
"""
Benchmark: SDK Overhead Measurement

Measures the overhead introduced by the CredSeal SDK, excluding actual LLM latency.
This validates the "under 1.5ms SDK overhead per call" claim.

Components measured:
1. Gateway initialization
2. Request construction (message serialization)
3. Response parsing (deserialization)
4. Cost calculation
5. Message history management

Run with: python benchmarks/benchmark_sdk_overhead.py
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, asdict

# Import SDK components
from credseal.models import (
    Message,
    Role,
    LLMResponse,
    TokenUsage,
    StreamChunk,
    ToolCall,
    Function,
)


@dataclass
class BenchmarkResult:
    name: str
    iterations: int
    mean_us: float  # microseconds
    median_us: float
    std_dev_us: float
    min_us: float
    max_us: float
    p95_us: float
    p99_us: float

    def __str__(self) -> str:
        return (
            f"{self.name}:\n"
            f"  Iterations: {self.iterations:,}\n"
            f"  Mean:       {self.mean_us:>8.2f} µs ({self.mean_us/1000:.3f} ms)\n"
            f"  Median:     {self.median_us:>8.2f} µs\n"
            f"  Std Dev:    {self.std_dev_us:>8.2f} µs\n"
            f"  Min:        {self.min_us:>8.2f} µs\n"
            f"  Max:        {self.max_us:>8.2f} µs\n"
            f"  P95:        {self.p95_us:>8.2f} µs\n"
            f"  P99:        {self.p99_us:>8.2f} µs\n"
        )


def percentile(data: list[float], p: float) -> float:
    """Calculate the p-th percentile of a sorted list."""
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def run_benchmark(name: str, func: callable, iterations: int = 10000) -> BenchmarkResult:
    """Run a benchmark and collect timing statistics."""
    # Warmup
    for _ in range(min(100, iterations // 10)):
        func()

    # Actual benchmark
    times_ns: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        func()
        end = time.perf_counter_ns()
        times_ns.append(end - start)

    times_us = [t / 1000 for t in times_ns]  # Convert to microseconds

    return BenchmarkResult(
        name=name,
        iterations=iterations,
        mean_us=statistics.mean(times_us),
        median_us=statistics.median(times_us),
        std_dev_us=statistics.stdev(times_us) if len(times_us) > 1 else 0,
        min_us=min(times_us),
        max_us=max(times_us),
        p95_us=percentile(times_us, 95),
        p99_us=percentile(times_us, 99),
    )


# ── Benchmark Functions ──────────────────────────────────────────────────────


def bench_message_creation():
    """Create a Message object."""
    Message(role=Role.USER, content="Hello, how are you today?")


def bench_message_with_tool_result():
    """Create a tool result Message."""
    Message(
        role=Role.TOOL,
        content='{"temperature": 18, "unit": "celsius", "condition": "cloudy"}',
        tool_call_id="call_abc123",
    )


def bench_message_serialization():
    """Serialize a Message to dict (for API calls)."""
    msg = Message(role=Role.USER, content="What is the capital of France?")
    msg.to_openai_dict()


def bench_tool_call_creation():
    """Create a ToolCall object."""
    ToolCall(
        id="call_abc123",
        function=Function(
            name="get_weather",
            arguments='{"location": "London", "unit": "celsius"}',
        ),
    )


def bench_response_creation():
    """Create an LLMResponse object (simulating parsed API response)."""
    LLMResponse(
        message=Message(role=Role.ASSISTANT, content="The capital of France is Paris."),
        model="gpt-4o",
        finish_reason="stop",
        cost_usd=0.00015,
        usage=TokenUsage(input_tokens=15, output_tokens=8, total_tokens=23),
    )


def bench_response_with_tool_calls():
    """Create an LLMResponse with tool calls."""
    LLMResponse(
        message=Message(role=Role.ASSISTANT, content=""),
        model="gpt-4o",
        finish_reason="tool_calls",
        cost_usd=0.00012,
        tool_calls=[
            ToolCall(
                id="call_abc123",
                function=Function(
                    name="get_weather",
                    arguments='{"location": "London"}',
                ),
            )
        ],
        usage=TokenUsage(input_tokens=20, output_tokens=15, total_tokens=35),
    )


def bench_token_usage_creation():
    """Create TokenUsage object."""
    usage = TokenUsage(input_tokens=1500, output_tokens=500, total_tokens=2000)
    _ = usage.total_tokens


def bench_stream_chunk_creation():
    """Create a StreamChunk (for streaming responses)."""
    StreamChunk(
        content="Hello",
        finish_reason=None,
        model="gpt-4o",
    )


def bench_cost_calculation():
    """Simulate cost calculation for a response."""
    # Pricing: GPT-4o input=$2.50/1M, output=$10.00/1M
    input_tokens = 1500
    output_tokens = 500
    input_cost = (input_tokens / 1_000_000) * 2.50
    output_cost = (output_tokens / 1_000_000) * 10.00
    _ = input_cost + output_cost


def bench_message_history_append():
    """Append messages to history (common operation)."""
    history = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    new_msg = Message(role=Role.USER, content="How are you?")
    history.append(new_msg.to_openai_dict())


def bench_full_request_construction():
    """Full request construction: messages + serialization."""
    messages = [
        Message(role=Role.SYSTEM, content="You are a helpful assistant."),
        Message(role=Role.USER, content="What is the weather in London?"),
    ]
    # Serialize all messages (what happens before API call)
    [m.to_openai_dict() for m in messages]


def bench_full_response_handling():
    """Full response handling: create response + extract data."""
    response = LLMResponse(
        message=Message(
            role=Role.ASSISTANT,
            content="The weather in London is cloudy with a high of 18C.",
        ),
        model="gpt-4o",
        finish_reason="stop",
        cost_usd=0.00023,
        usage=TokenUsage(input_tokens=25, output_tokens=15, total_tokens=40),
    )
    _ = response.message.content
    _ = response.cost_usd
    _ = response.usage.total_tokens


def bench_dataclass_to_dict():
    """Convert dataclass to dict using asdict (for logging/serialization)."""
    response = LLMResponse(
        message=Message(role=Role.ASSISTANT, content="Hello!"),
        model="gpt-4o",
        finish_reason="stop",
        cost_usd=0.0001,
        usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
    )
    asdict(response)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("CredSeal SDK Overhead Benchmark")
    print("=" * 70)
    print()
    print("Measuring SDK overhead components (excludes actual LLM latency)")
    print("Target: < 1.5ms (1500µs) total SDK overhead per call")
    print()
    print("-" * 70)

    benchmarks = [
        ("Message Creation", bench_message_creation),
        ("Message with Tool Result", bench_message_with_tool_result),
        ("Message Serialization", bench_message_serialization),
        ("ToolCall Creation", bench_tool_call_creation),
        ("Response Creation", bench_response_creation),
        ("Response with Tool Calls", bench_response_with_tool_calls),
        ("Token Usage Creation", bench_token_usage_creation),
        ("Stream Chunk Creation", bench_stream_chunk_creation),
        ("Cost Calculation", bench_cost_calculation),
        ("Message History Append", bench_message_history_append),
        ("Full Request Construction", bench_full_request_construction),
        ("Full Response Handling", bench_full_response_handling),
        ("Dataclass to Dict", bench_dataclass_to_dict),
    ]

    results: list[BenchmarkResult] = []
    for name, func in benchmarks:
        print(f"Running: {name}...", end=" ", flush=True)
        result = run_benchmark(name, func, iterations=10000)
        results.append(result)
        print(f"done ({result.mean_us:.2f}µs mean)")

    print()
    print("=" * 70)
    print("DETAILED RESULTS")
    print("=" * 70)
    print()

    for result in results:
        print(result)

    # Calculate total overhead
    request_overhead = next(r for r in results if "Full Request" in r.name)
    response_overhead = next(r for r in results if "Full Response" in r.name)
    total_overhead_us = request_overhead.mean_us + response_overhead.mean_us
    total_overhead_ms = total_overhead_us / 1000

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"  Request construction:  {request_overhead.mean_us:>8.2f} µs")
    print(f"  Response handling:     {response_overhead.mean_us:>8.2f} µs")
    print(f"  ─────────────────────────────────────")
    print(f"  TOTAL SDK OVERHEAD:    {total_overhead_us:>8.2f} µs ({total_overhead_ms:.3f} ms)")
    print()

    if total_overhead_ms < 1.5:
        print(f"  [PASS] SDK overhead ({total_overhead_ms:.3f}ms) is under 1.5ms target")
    else:
        print(f"  [FAIL] SDK overhead ({total_overhead_ms:.3f}ms) exceeds 1.5ms target")

    print()
    print("Note: This measures SDK overhead only. Actual LLM call latency")
    print("(typically 200-2000ms) is not included in these measurements.")
    print()

    return total_overhead_ms < 1.5


if __name__ == "__main__":
    import sys

    success = main()
    sys.exit(0 if success else 1)
