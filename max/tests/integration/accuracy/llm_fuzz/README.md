# llm-fuzz

Unified fuzz and correctness testing for LLM inference endpoints. Tests two
things: crash resilience (can we break the server?) and output correctness (does
it follow the OpenAI API contract?).

## Quick start

The fuzz binary is built and run via Bazel. All examples below define a
`$LLM_FUZZ` shorthand for the long invocation:

```bash
LLM_FUZZ="./bazelw run //max/tests/integration/accuracy:llm-fuzz --"

# Fuzz test a local server
$LLM_FUZZ --url http://localhost:8000 --model your-model

# Correctness validation
$LLM_FUZZ --url http://localhost:8000 --model your-model --validation-only

# Model-specific tests for Kimi K2.5
$LLM_FUZZ --url http://localhost:8200 --model kimi-k2.5 --model-profile kimi-k2.5

# List everything
$LLM_FUZZ --list
```

Bazel manages all dependencies; no extra setup is needed.

## Usage

```bash
# Run all fuzz scenarios against a local vLLM/TGI/Dynamo server
$LLM_FUZZ --url http://localhost:8000 --model nvidia/DeepSeek-V3.1-NVFP4

# Run against OpenAI
$LLM_FUZZ --url https://api.openai.com --api-key $OPENAI_API_KEY --model gpt-4o-mini

# Run specific scenarios
$LLM_FUZZ --url http://localhost:8000 --model my-model \
    --scenarios malformed_payloads,streaming_attacks,kv_cache_pressure

# Filter by tags
$LLM_FUZZ --url http://localhost:8000 --model my-model --tags crash,deepseek

# Fuzz-only (skip validation scenarios)
$LLM_FUZZ --url http://localhost:8000 --model my-model --fuzz-only

# Validation-only (skip fuzz scenarios)
$LLM_FUZZ --url http://localhost:8000 --model my-model --validation-only

# Model-specific testing (includes generic + model-specific scenarios)
$LLM_FUZZ --url http://localhost:8200 --model kimi-k2.5 --model-profile kimi-k2.5
$LLM_FUZZ --url http://localhost:8100 --model glm-5.1 --model-profile glm-5.1

# Verbose output (show all details, not just failures)
$LLM_FUZZ --url http://localhost:8000 --model my-model -v

# Export results
$LLM_FUZZ --url http://localhost:8000 --model my-model \
    --export-json results.json --export-csv results.csv

# Flakiness detection (run each scenario 3 times)
$LLM_FUZZ --url http://localhost:8000 --model my-model --repeat 3

# Circuit breaker (stop after 3 consecutive server failures)
$LLM_FUZZ --url http://localhost:8000 --model my-model --circuit-breaker 3

# Compare against a previous run (regression detection)
$LLM_FUZZ --url http://localhost:8000 --model my-model \
    --compare logs/run-20260413-120000.jsonl

# Disable automatic run logging
$LLM_FUZZ --url http://localhost:8000 --model my-model --no-log

# Write the run log to a specific path (useful for CI pipelines that consume the log)
$LLM_FUZZ --url http://localhost:8000 --model my-model --log-file /tmp/fuzz.jsonl

# Endurance mode (long-running soak test for memory leaks)
$LLM_FUZZ --url http://localhost:8000 --model my-model \
    --endurance --endurance-duration 15 --endurance-intensity high

# Zombie Horde mode (all scenarios fired in random parallel waves)
$LLM_FUZZ --url http://localhost:8000 --model my-model \
    --scenarios zombie_horde

# Version
$LLM_FUZZ -V
```

## Model configuration

Scenarios that test context window boundaries and KV cache pressure adapt their
test sizes to the model's actual parameters. The tool tries to fetch
`config.json` from HuggingFace using the `--model` value as a repo ID.

```bash
# Auto-detect: --model is used as HF repo ID to fetch max_position_embeddings
$LLM_FUZZ --url http://localhost:8000 --model deepseek-ai/DeepSeek-R1

# Manual override (takes priority over HF config)
$LLM_FUZZ --url http://localhost:8000 --model my-model \
    --max-context-length 163840 --max-num-tokens 32768

# Disable HF fetch (use defaults: context=4096, max_num_tokens=4096)
$LLM_FUZZ --url http://localhost:8000 --model my-model --no-hf-fetch
```

Options:

- `--max-context-length TOKENS` -- Override max context window
- `--max-num-tokens TOKENS` -- Override engine max_num_tokens
- `--no-hf-fetch` -- Disable HuggingFace config fetch

If the model is not found on HuggingFace, the tool silently falls back to
conservative defaults (4096 tokens).

## Scenarios

### Fuzz scenarios (adversarial, crash-focused)

| Scenario                           | Tests | What it attacks                                                                        |
|------------------------------------|-------|----------------------------------------------------------------------------------------|
| `malformed_payloads`               | ~40   | Missing fields, broken JSON, wrong types, truncated bodies, binary junk                |
| `content_edge_cases`               | ~60   | Unicode surrogates, control chars, zero-width joiners, emoji clusters, extreme lengths |
| `parameter_abuse`                  | ~50   | NaN/Infinity, negative max_tokens, temperature extremes, conflicting params            |
| `concurrency_attacks`              | ~8    | Thundering herd (100 identical), burst patterns, ramp 1-100                            |
| `streaming_attacks`                | ~14   | Cancel after 0/1/3 chunks, cancel storms, random cancel points                         |
| `structured_output`                | ~15   | JSON mode state corruption (CVE), invalid schemas, rapid alternation                   |
| `tool_calling`                     | ~25   | Malformed tools, empty args (vLLM #19419), streaming divergence                        |
| `protocol_abuse`                   | ~20   | Wrong Content-Type, huge headers, slowloris, encoding attacks                          |
| `resource_exhaustion`              | ~12   | Context boundary probing, logprobs amplification, concurrent large contexts            |
| `state_interaction`                | ~18   | Cross-request state leaks, seed consistency, model switching                           |
| `kv_cache_pressure`                | ~42   | Prefix cache pollution, page boundary probing, concurrent KV fill                      |
| `thinking_tokens`                  | ~11   | DeepSeek `<think>` abuse: unbounded reasoning, mid-think cancellation                  |
| `pipeline_stall`                   | ~8    | Prefill/decode imbalance, streaming cancel during KV transfer                          |
| `connection_exhaustion`            | ~10   | Half-open connections, TCP resets, slowloris, pipelining                               |
| `endurance_soak`                   | ~5    | Sustained load, error rate windows, latency degradation detection                      |
| `endpoint_abuse`                   | ~30   | /metrics flooding, path traversal, method abuse, CORS probing                          |
| `openai_spec_compliance`           | ~27   | Response structure, finish_reason, usage math, streaming chunks                        |
| `output_correctness`               | ~13   | Concurrent prompts with quality validation, cross-contamination detection              |
| `openrouter_tests`                 | ~59   | OpenRouter provider validation (49 text-only tests)                                    |
| `tool_schema_validation`           | ~24   | K2VV-style tool call schema validation                                                 |
| `json_schema_compliance`           | ~100  | 100-run cache-busted guided decoding validation                                        |
| `tool_arguments_json_stability`    | 32x   | Repeated forced tool calls, arguments must parse as JSON                               |
| `streaming_reasoning_monotonicity` | 30x   | Reasoning/content monotonicity in streaming                                            |
| `zombie_horde`                     | all   | Chaos mode, all scenarios in random parallel waves                                     |

### Validation scenarios (correctness, OpenAI SDK-based)

These require the `openai` Python package, which Bazel provides automatically.

| Scenario                | Tests | What it validates                                                                   |
|-------------------------|-------|-------------------------------------------------------------------------------------|
| `tc_streaming_protocol` | 8     | First-chunk fields (id, type, name), finish_reason semantics, no post-finish deltas |
| `tc_basics`             | 11    | Tool choice modes (auto/required/named), parallel TCs, multi-turn, many tools       |
| `tc_advanced`           | 7     | Streaming argument accumulation, SO/TC switching, concurrency soak                  |
| `so_basics`             | 11    | Schema types, enums, anyOf/allOf, $ref, nesting, streaming                          |
| `so_advanced`           | 11    | Circular refs, large schemas, concurrent isolation, 50-property objects             |
| `concurrent_stress`     | 5     | Mixed concurrent requests, type isolation, usage tracking                           |
| `production_resilience` | 10    | Truncation, empty input, token counting, long conversations                         |

### Model-specific scenarios

Run with `--model-profile kimi-k2.5` or `--model-profile glm-5.1`.

| Scenario          | Model     | Tests | What it validates                                  |
|-------------------|-----------|-------|----------------------------------------------------|
| `kimi_battle`     | Kimi K2.5 | 15    | xgrammar edge cases, parallel TCs, structural tags |
| `kimi_3am`        | Kimi K2.5 | 12    | Production edge cases, soak tests, precision       |
| `kimi_production` | Kimi K2.5 | 10    | Long context, error recovery, token counting       |
| `kimi_k2vv`       | Kimi K2.5 | 2K    | K2 Vendor Verifier benchmark (see below)           |
| `glm_battle`      | GLM-5.1   | 12    | Schema compilation, tool calling, streaming        |
| `glm_3am`         | GLM-5.1   | 10    | Edge cases, soak tests, concurrent stress          |

### K2 Vendor Verifier (K2VV)

MoonshotAI's
[official benchmark](https://github.com/MoonshotAI/K2-Vendor-Verifier) for
validating Kimi K2 deployments. Measures two things:

- **ToolCall-Trigger F1**: does the deployment trigger tool calls in the same
  situations as the official Moonshot API? (F1 of
  `finish_reason == "tool_calls"` vs reference)
- **Schema Accuracy**: when tool calls fire, do the JSON arguments validate
  against the tool's parameter schema?

The dataset (~13MB) is downloaded from MoonshotAI's CDN on each run. Use
`--k2vv-mode` to control sample count:

```bash
# Quick mode (default): 500 randomly sampled requests
$LLM_FUZZ --url http://localhost:8200 --model kimi-k2.5 \
    --model-profile kimi-k2.5 --scenarios kimi_k2vv

# Full mode: all 2,000 requests
$LLM_FUZZ --url http://localhost:8200 --model kimi-k2.5 \
    --model-profile kimi-k2.5 --scenarios kimi_k2vv --k2vv-mode full
```

Thresholds (from K2VV README): F1 >= 73% (thinking mode), schema accuracy >=
95%. vLLM currently scores 87% schema accuracy and 76% F1 on the official
benchmark.

## Run logging

Every run automatically saves a structured JSONL log to
`logs/run-YYYYMMDD-HHMMSS.jsonl`. Override the path with `--log-file PATH`. Each
line is a JSON event:

```jsonl
{"event": "run_start", "timestamp": "...", "url": "...", "model": "...", "scenario_count": 36}
{"event": "scenario_start", "timestamp": "...", "scenario_name": "malformed_payloads"}
{"event": "test_result", "timestamp": "...", "scenario_name": "...", "test_name": "...", "verdict": "PASS", ...}
{"event": "scenario_end", "timestamp": "...", "scenario_name": "...", "elapsed_ms": 1234, "pass_count": 38, "fail_count": 2}
{"event": "run_end", "timestamp": "...", "total_tests": 450, "pass_count": 420, "fail_count": 15, ...}
```

These logs enable post-run analysis (`jq '.[] | select(.verdict == "FAIL")'`)
and regression comparison. Disable with `--no-log`, or write to a specific path
with `--log-file PATH`.

## Regression comparison

Compare results against a previous run to detect regressions:

```bash
# Run, then later compare against the saved log
$LLM_FUZZ --url http://localhost:8000 --model my-model \
    --compare logs/run-20260413-120000.jsonl
```

Accepts both `.json` (export format) and `.jsonl` (run log format) baselines.
Reports:

- New failures (PASS in baseline, FAIL now)
- Recovered tests (FAIL in baseline, PASS now)
- New/removed tests
- Unchanged count

## Architecture

```text
fuzz.py              CLI entry point, progress bar, circuit breaker, scenario filtering
client.py            Low-level async HTTP client (stdlib only)
validator_client.py  OpenAI SDK client for validation scenarios
helpers.py           Shared utilities (collect_stream, make_tool, validators)
reporting.py         Terminal output, JSON/CSV export, baseline comparison
run_log.py           Structured JSONL run logging
model_config.py      HuggingFace config fetch, model profiles
__main__.py          Module entry point (python -m fuzz)
scenarios/
  __init__.py        Base class, auto-discovery registry (recursive), CircuitBreaker
  s01-s24_*.py       Fuzz scenarios (adversarial, crash-focused)
  validation/        Correctness validation (any OpenAI-compatible endpoint)
    v01-v07_*.py     Tool calling, structured output, concurrency, production
  models/            Model-specific test suites
    kimi_*.py        Kimi K2.5 battle, 3am edge cases, production readiness
    glm_*.py         GLM-5.1 battle, 3am edge cases
```

## Verdicts

Each test produces one of four verdicts:

- **PASS** -- Server handled it gracefully (proper error code or valid response)
- **FAIL** -- Server crashed (5xx), hung (timeout), or produced corrupt output
- **INTERESTING** -- Unexpected behavior worth investigating
- **ERROR** -- The test client itself errored (network issue, our bug)

Every scenario ends with a health check to verify the server survived.

## Circuit breaker

The circuit breaker stops execution after N consecutive server failures
(default: 5). This prevents wasting time testing a dead server.

```bash
# Stop after 3 consecutive failures
$LLM_FUZZ --url http://localhost:8000 --model my-model --circuit-breaker 3

# Disable circuit breaker
$LLM_FUZZ --url http://localhost:8000 --model my-model --circuit-breaker 0
```

## Endurance mode

For detecting memory leaks, KV cache fragmentation, and latency degradation:

```bash
$LLM_FUZZ --url http://localhost:8000 --model my-model \
    --endurance --endurance-duration 15 --endurance-intensity medium
```

Options:

- `--endurance-duration MINUTES` -- Test duration (default: 5)
- `--endurance-intensity {low,medium,high}` -- Request rate: low=5/s,
  medium=20/s, high=100/s

Fails if error rate degrades or p99 latency grows 3x from baseline.

## Adding custom scenarios

### Fuzz scenario (raw HTTP)

```python
# scenarios/s99_my_custom.py
from scenarios import BaseScenario, register_scenario, Verdict

@register_scenario
class MyCustomScenario(BaseScenario):
    name = "my_custom"
    description = "Description of what this tests"
    tags = ["custom", "crash"]

    async def run(self, client, config):
        results = []
        resp = await client.post_json({"model": config.model, "messages": [...]})
        results.append(self.make_result("my_custom", "test_name", Verdict.PASS))
        return results
```

### Validation scenario (OpenAI SDK)

```python
# scenarios/validation/v99_my_validation.py
import asyncio
from scenarios import BaseScenario, register_scenario, Verdict
from helpers import make_tool, collect_stream

@register_scenario
class MyValidation(BaseScenario):
    name = "my_validation"
    description = "Description"
    tags = ["validation", "custom"]
    requires_validator = True
    scenario_type = "validation"

    async def run(self, client, config):
        results = []
        validator = config.validator
        loop = asyncio.get_event_loop()

        def _test():
            tools = [make_tool("get_weather", {"type": "object", "properties": {"city": {"type": "string"}}})]
            return validator.tc_chat_stream(
                [{"role": "user", "content": "Weather in Paris"}],
                tools, tool_choice="required",
            )

        try:
            result = await loop.run_in_executor(None, _test)
            # Validate result...
            results.append(self.make_result(self.name, "test_name", Verdict.PASS))
        except Exception as e:
            results.append(self.make_result(self.name, "test_name", Verdict.ERROR, error=str(e)))
        return results
```

### Model-specific scenario

Same as validation, but add `model_filter`:

```python
@register_scenario
class MyModelTest(BaseScenario):
    name = "my_model_test"
    tags = ["validation", "model:my-model"]
    requires_validator = True
    scenario_type = "validation"
    model_filter = "my-model"  # Only runs with --model-profile my-model
```

Scenarios are auto-discovered from `scenarios/`, `scenarios/validation/`, and
`scenarios/models/`.

## Known bugs this tool detects

Real, documented bugs in popular inference servers:

- **vLLM #4070**: `response_format: json_object` corrupts global engine state
- **vLLM #17248**: Invalid `guided_json` schema crashes the API server
- **vLLM #19419**: Empty tool call arguments cause `JSONDecodeError` crash
- **vLLM #27641**: Streaming produces different tool call output than
  non-streaming
- **vLLM #16340**: First streaming tool call chunk missing `"type":"function"`
  field
- **vLLM #10325**: `prompt_logprobs=1` increases memory 5x
- **vLLM #35191**: Prefix cache hit rate degrades to 0% over hours
- **vllm-mlx**: Disconnected streaming clients lock the server
- **vLLM #12886**: Error response format doesn't match OpenAI API spec
