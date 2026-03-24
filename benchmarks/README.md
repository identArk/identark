# CredSeal SDK Benchmarks

This directory contains performance benchmarks for the CredSeal SDK.

## SDK Overhead Benchmark

Validates the claim: **"under 1.5ms SDK overhead per call"**

### What It Measures

The benchmark measures time spent in SDK code, **excluding** actual LLM API latency:

| Component | Description |
|-----------|-------------|
| Message Creation | Time to instantiate a `Message` object |
| Message Serialization | Time to convert messages to dict for API calls |
| Response Parsing | Time to parse LLM response into `LLMResponse` |
| Cost Calculation | Time to compute token costs |
| History Management | Time to append messages to conversation history |

### Running the Benchmark

```bash
cd credseal-sdk
python benchmarks/benchmark_sdk_overhead.py
```

### Typical Results

On Apple Silicon (M-series):

```
SUMMARY
======================================================================

  Request construction:      1.31 µs
  Response handling:         1.45 µs
  ─────────────────────────────────────
  TOTAL SDK OVERHEAD:        2.76 µs (0.003 ms)

  [PASS] SDK overhead (0.003ms) is under 1.5ms target
```

Detailed component breakdown:

| Component | Mean Time |
|-----------|-----------|
| Message Creation | 0.57 µs |
| Message Serialization | 0.65 µs |
| Response Creation | 1.40 µs |
| Token Usage Creation | 0.45 µs |
| Stream Chunk Creation | 0.53 µs |
| Cost Calculation | 0.24 µs |

### Interpretation

- **SDK overhead** is the time spent in CredSeal code (serialization, parsing, validation)
- **LLM latency** is the time waiting for the LLM provider (200-2000ms typically)
- The SDK adds **< 0.1ms** overhead to each call, well under our 1.5ms target
- Total round-trip time = SDK overhead + network latency + LLM inference time

### Why This Matters

For production AI agents making thousands of LLM calls:
- 1000 calls × 0.1ms overhead = 100ms total SDK overhead
- 1000 calls × 1.5ms overhead = 1.5 seconds total SDK overhead

The SDK is designed to add negligible latency to your agent workflows.

## Running All Benchmarks

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all benchmarks
python -m pytest benchmarks/ -v
```

## Adding New Benchmarks

1. Create a new file: `benchmarks/benchmark_<name>.py`
2. Use the `run_benchmark()` helper from the overhead benchmark
3. Document what you're measuring and why it matters
