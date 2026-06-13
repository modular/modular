# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""
Scenario: Token runaway detection

Detects runaway token generation: a server that returns finish_reason="length"
on a trivially-short prompt is stuck in a generation loop (or has a grammar
constraint that prevents EOS). Reports FAIL whenever finish_reason is "length"
on a prompt that should need ≤100 output tokens.

All requests use max_tokens=2048. Any response consuming 2048 tokens on a
simple question is runaway. The scenario checks:

1. json_schema response_format with a minimal schema (should produce ~30 tokens).
2. tool_choice="required" forced call (should produce ~50 tokens of tool args).
3. json_schema + implicit tool call mix (Gemma sometimes mishandles this).
4. json_object response_format with a one-field object request.
5. A baseline plain-text request to confirm the server itself is healthy.

Motivated by: nvidia/Gemma-4-31B-IT-NVFP4 reports of 60k-200k output tokens
hitting the server max-length limit every time, with structured output and
tool calling suspected as the trigger.

Request timeout: 60 seconds per call (at 50 tok/s, max_tokens=2048 ≈ 40s;
60s gives headroom without hanging for 20 min on a real runaway).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig

# Hard cap. Anything hitting this on a trivial prompt is runaway.
_MAX_TOKENS = 2048
# Per-request HTTP timeout in seconds.
_REQUEST_TIMEOUT = 60.0

# A minimal tool definition for forced-call tests.
_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                },
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}

# A minimal JSON schema that should produce ~30 tokens.
_STATUS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "status_response",
        "description": "Simple status check",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["ok", "error"],
                },
                "message": {"type": "string"},
            },
            "required": ["status", "message"],
            "additionalProperties": False,
        },
    },
}


def _extract_finish_reason(resp_body: str) -> tuple[str | None, int | None]:
    """Return (finish_reason, completion_tokens) from a response body."""
    try:
        data = json.loads(resp_body)
    except (json.JSONDecodeError, TypeError):
        return None, None
    try:
        finish_reason = data["choices"][0]["finish_reason"]
    except (KeyError, IndexError, TypeError):
        finish_reason = None
    try:
        completion_tokens = data["usage"]["completion_tokens"]
    except (KeyError, TypeError):
        completion_tokens = None
    return finish_reason, completion_tokens


def _runaway_verdict(
    finish_reason: str | None,
    completion_tokens: int | None,
    test_name: str,
) -> tuple[Verdict, str]:
    """Classify a response as runaway or healthy."""
    if finish_reason == "length":
        tok_info = (
            f"completion_tokens={completion_tokens}"
            if completion_tokens is not None
            else "completion_tokens=unknown"
        )
        return (
            Verdict.FAIL,
            f"RUNAWAY: finish_reason='length' on trivial prompt "
            f"({tok_info}). Model is stuck in a generation loop or "
            f"grammar constraint is preventing EOS. test={test_name}",
        )
    if finish_reason is None:
        return (
            Verdict.INTERESTING,
            f"finish_reason missing in response. test={test_name}",
        )
    return (
        Verdict.PASS,
        f"finish_reason='{finish_reason}' completion_tokens={completion_tokens}",
    )


@register_scenario
class TokenRunaway(BaseScenario):
    name = "token_runaway"
    description = (
        "Detects runaway token generation (finish_reason=length on trivial "
        "prompts) with structured output and tool calling"
    )
    tags = ["structured", "tool_calling", "runaway", "correctness"]

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []
        model = config.model

        # ------------------------------------------------------------------ #
        # 1. Baseline plain-text — if this runaways, the server is broken
        #    for all requests, not just structured output.
        # ------------------------------------------------------------------ #
        baseline_payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with exactly one word: 'hello'.",
                }
            ],
            "max_tokens": _MAX_TOKENS,
        }
        resp = await client.post_json(
            baseline_payload, timeout=_REQUEST_TIMEOUT
        )
        if resp.error or resp.status != 200:
            results.append(
                self.make_result(
                    self.name,
                    "baseline_plain_text",
                    Verdict.FAIL,
                    status_code=resp.status,
                    detail=f"API error: {resp.error or resp.body[:400]}",
                    request_body=json.dumps(baseline_payload),
                    response_body=resp.body[:800],
                )
            )
        else:
            finish_reason, completion_tokens = _extract_finish_reason(resp.body)
            verdict, detail = _runaway_verdict(
                finish_reason, completion_tokens, "baseline_plain_text"
            )
            results.append(
                self.make_result(
                    self.name,
                    "baseline_plain_text",
                    verdict,
                    status_code=resp.status,
                    detail=detail,
                    request_body=json.dumps(baseline_payload),
                    response_body=resp.body[:800],
                )
            )

        # ------------------------------------------------------------------ #
        # 2. json_schema response_format — minimal status schema.
        #    Should produce ~30 tokens. finish_reason="length" = runaway.
        # ------------------------------------------------------------------ #
        json_schema_payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "Is the system operational? Respond using the required JSON format.",
                }
            ],
            "response_format": _STATUS_SCHEMA,
            "max_tokens": _MAX_TOKENS,
        }
        resp = await client.post_json(
            json_schema_payload, timeout=_REQUEST_TIMEOUT
        )
        if resp.error or resp.status != 200:
            results.append(
                self.make_result(
                    self.name,
                    "json_schema_status",
                    Verdict.FAIL,
                    status_code=resp.status,
                    detail=f"API error: {resp.error or resp.body[:400]}",
                    request_body=json.dumps(json_schema_payload),
                    response_body=resp.body[:800],
                )
            )
        else:
            finish_reason, completion_tokens = _extract_finish_reason(resp.body)
            verdict, detail = _runaway_verdict(
                finish_reason, completion_tokens, "json_schema_status"
            )
            # Also check that content is actually JSON
            try:
                data = json.loads(resp.body)
                content = data["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                if verdict == Verdict.PASS and not isinstance(parsed, dict):
                    verdict = Verdict.INTERESTING
                    detail += " | content not a JSON object"
                elif verdict == Verdict.PASS:
                    detail += f" | content_keys={list(parsed.keys())}"
            except Exception as e:
                if verdict == Verdict.PASS:
                    verdict = Verdict.INTERESTING
                    detail += f" | could not parse content as JSON: {e}"
            results.append(
                self.make_result(
                    self.name,
                    "json_schema_status",
                    verdict,
                    status_code=resp.status,
                    detail=detail,
                    request_body=json.dumps(json_schema_payload),
                    response_body=resp.body[:800],
                )
            )

        # ------------------------------------------------------------------ #
        # 3. tool_choice="required" forced call — no max_tokens restriction
        #    beyond our 2048 cap.  Should produce ~50 tokens of tool args.
        # ------------------------------------------------------------------ #
        tool_required_payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "What's the weather in Paris?",
                }
            ],
            "tools": [_WEATHER_TOOL],
            "tool_choice": "required",
            "max_tokens": _MAX_TOKENS,
        }
        resp = await client.post_json(
            tool_required_payload, timeout=_REQUEST_TIMEOUT
        )
        if resp.error or resp.status != 200:
            results.append(
                self.make_result(
                    self.name,
                    "tool_choice_required",
                    Verdict.FAIL,
                    status_code=resp.status,
                    detail=f"API error: {resp.error or resp.body[:400]}",
                    request_body=json.dumps(tool_required_payload),
                    response_body=resp.body[:800],
                )
            )
        else:
            finish_reason, completion_tokens = _extract_finish_reason(resp.body)
            verdict, detail = _runaway_verdict(
                finish_reason, completion_tokens, "tool_choice_required"
            )
            results.append(
                self.make_result(
                    self.name,
                    "tool_choice_required",
                    verdict,
                    status_code=resp.status,
                    detail=detail,
                    request_body=json.dumps(tool_required_payload),
                    response_body=resp.body[:800],
                )
            )

        # ------------------------------------------------------------------ #
        # 4. json_schema + tools in the same request (Gemma sometimes
        #    mishandles this combination — the grammar for the tool call
        #    and json_schema can interact unexpectedly).
        # ------------------------------------------------------------------ #
        combo_payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Check the weather in London and report "
                        "the status in the required JSON format."
                    ),
                }
            ],
            "tools": [_WEATHER_TOOL],
            "response_format": _STATUS_SCHEMA,
            "max_tokens": _MAX_TOKENS,
        }
        resp = await client.post_json(combo_payload, timeout=_REQUEST_TIMEOUT)
        if resp.error or resp.status != 200:
            # A 400 here is acceptable (conflicting constraints); a 500 is not.
            if resp.status == 400:
                results.append(
                    self.make_result(
                        self.name,
                        "json_schema_plus_tools",
                        Verdict.PASS,
                        status_code=resp.status,
                        detail="400 — server correctly rejected conflicting constraints",
                        request_body=json.dumps(combo_payload),
                        response_body=resp.body[:400],
                    )
                )
            else:
                results.append(
                    self.make_result(
                        self.name,
                        "json_schema_plus_tools",
                        Verdict.FAIL,
                        status_code=resp.status,
                        detail=f"API error: {resp.error or resp.body[:400]}",
                        request_body=json.dumps(combo_payload),
                        response_body=resp.body[:800],
                    )
                )
        else:
            finish_reason, completion_tokens = _extract_finish_reason(resp.body)
            verdict, detail = _runaway_verdict(
                finish_reason, completion_tokens, "json_schema_plus_tools"
            )
            results.append(
                self.make_result(
                    self.name,
                    "json_schema_plus_tools",
                    verdict,
                    status_code=resp.status,
                    detail=detail,
                    request_body=json.dumps(combo_payload),
                    response_body=resp.body[:800],
                )
            )

        # ------------------------------------------------------------------ #
        # 5. json_object response_format — simpler than json_schema.
        #    "Return a JSON object with a single key 'answer'" should produce
        #    ~20 tokens.
        # ------------------------------------------------------------------ #
        json_object_payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Return a JSON object with a single key 'answer' "
                        "whose value is the string 'yes'."
                    ),
                }
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": _MAX_TOKENS,
        }
        resp = await client.post_json(
            json_object_payload, timeout=_REQUEST_TIMEOUT
        )
        if resp.error or resp.status != 200:
            results.append(
                self.make_result(
                    self.name,
                    "json_object_format",
                    Verdict.FAIL,
                    status_code=resp.status,
                    detail=f"API error: {resp.error or resp.body[:400]}",
                    request_body=json.dumps(json_object_payload),
                    response_body=resp.body[:800],
                )
            )
        else:
            finish_reason, completion_tokens = _extract_finish_reason(resp.body)
            verdict, detail = _runaway_verdict(
                finish_reason, completion_tokens, "json_object_format"
            )
            results.append(
                self.make_result(
                    self.name,
                    "json_object_format",
                    verdict,
                    status_code=resp.status,
                    detail=detail,
                    request_body=json.dumps(json_object_payload),
                    response_body=resp.body[:800],
                )
            )

        # ------------------------------------------------------------------ #
        # 6. json_schema with missing top-level "type" field.
        #    {"properties": {"x": {}}} is valid JSON Schema (type omission =
        #    accept any type), but the grammar backend may produce a broken
        #    token mask that prevents EOS, causing a tight repetition loop
        #    until max_model_len is hit.
        #    Repro rate: ~1/5 requests enter the loop; all produce garbage.
        #    Reference: Gemma-4-31B runaway issue 2026-06-13.
        # ------------------------------------------------------------------ #
        missing_type_payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "Return a JSON object with key name and value test",
                }
            ],
            "max_tokens": _MAX_TOKENS,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "bad",
                    "schema": {"properties": {"x": {}}},
                },
            },
        }
        resp = await client.post_json(
            missing_type_payload, timeout=_REQUEST_TIMEOUT
        )
        if resp.error or resp.status not in (200, 400):
            results.append(
                self.make_result(
                    self.name,
                    "json_schema_missing_type",
                    Verdict.FAIL,
                    status_code=resp.status,
                    detail=f"Unexpected error: {resp.error or resp.body[:400]}",
                    request_body=json.dumps(missing_type_payload),
                    response_body=resp.body[:800],
                )
            )
        elif resp.status == 400:
            # Acceptable: server rejected a schema it can't handle gracefully.
            results.append(
                self.make_result(
                    self.name,
                    "json_schema_missing_type",
                    Verdict.PASS,
                    status_code=resp.status,
                    detail="400 — server correctly rejected type-less schema",
                    request_body=json.dumps(missing_type_payload),
                    response_body=resp.body[:400],
                )
            )
        else:
            finish_reason, completion_tokens = _extract_finish_reason(resp.body)
            verdict, detail = _runaway_verdict(
                finish_reason, completion_tokens, "json_schema_missing_type"
            )
            # Even on "stop", check whether the content is a valid JSON
            # *object* (not just any JSON value).
            # Garbage like '"]```jsons{ "' parses as a JSON string, so we
            # must require isinstance(parsed, dict) to catch the broken-grammar
            # pattern that starts with ']``.
            if verdict == Verdict.PASS:
                try:
                    data = json.loads(resp.body)
                    content = data["choices"][0]["message"]["content"]
                    parsed = json.loads(content)
                    if not isinstance(parsed, dict):
                        raise ValueError(
                            f"content parsed as {type(parsed).__name__!r}, "
                            f"not a JSON object: {content!r}"
                        )
                except Exception as e:
                    verdict = Verdict.FAIL
                    detail = (
                        f"finish_reason='{finish_reason}' but content is not "
                        f"a valid JSON object (grammar constraint broken): {e}"
                    )
            results.append(
                self.make_result(
                    self.name,
                    "json_schema_missing_type",
                    verdict,
                    status_code=resp.status,
                    detail=detail,
                    request_body=json.dumps(missing_type_payload),
                    response_body=resp.body[:800],
                )
            )

        # ------------------------------------------------------------------ #
        # 7. Health check — make sure the server survived the scenario.
        # ------------------------------------------------------------------ #
        health = await client.health_check()
        results.append(
            self.make_result(
                self.name,
                "post_runaway_health_check",
                Verdict.PASS if health.status == 200 else Verdict.FAIL,
                status_code=health.status,
                detail=(
                    "Server healthy after runaway scenario"
                    if health.status == 200
                    else (health.error or f"HTTP {health.status}")
                ),
            )
        )

        return results
