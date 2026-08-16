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
"""GLM-5.1 3am edge cases -- production failures.

The tests that break at 3am when nobody is watching. Each test targets a
specific production failure mode: parallel tool call cross-talk under
concurrent load, large tool results in conversation history, many-tools
selection, rapid SO/TC mode switching, empty inputs, sequential soak tests,
nested JSON in tool arguments, streaming usage with tool calls, and
minimal-input stress.
"""

from __future__ import annotations

import asyncio
import json
import random
from typing import TYPE_CHECKING, Any

from helpers import (
    budget_exhausted,
    collect_stream,
    make_tool,
    stream_budget_exhausted,
)

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig
    from validator_client import ValidatorClient

# ---------------------------------------------------------------------------
# Shared tool definitions
# ---------------------------------------------------------------------------

_SEARCH_TOOL = make_tool(
    "search",
    {
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": ["q"],
        "additionalProperties": False,
    },
    description="Search for information",
)

_CALC_TOOL = make_tool(
    "calc",
    {
        "type": "object",
        "properties": {"expr": {"type": "string"}},
        "required": ["expr"],
        "additionalProperties": False,
    },
    description="Evaluate a math expression",
)

_FETCH_TOOL = make_tool(
    "fetch",
    {
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
        "additionalProperties": False,
    },
    description="Fetch data from a URL",
)


@register_scenario
class Glm3am(BaseScenario):
    """GLM-5.1 3am edge cases -- production failure modes."""

    name = "glm_3am"
    description = "GLM-5.1 3am edge cases -- production failures"
    tags = ["validation", "model:glm-5.1", "edge_cases"]
    requires_validator = True
    scenario_type = "validation"
    model_filter = "glm-5.1"

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []
        v = config.validator
        if not v:
            results.append(
                self.make_result(
                    self.name,
                    "setup",
                    Verdict.ERROR,
                    detail="No validator client available",
                )
            )
            return results

        loop = asyncio.get_running_loop()

        for test_fn in [
            self._test_parallel_tc_concurrent_4,
            self._test_large_tool_result_8k,
            self._test_50_tools_auto,
            self._test_so_tc_rapid_switch,
            self._test_empty_user_message,
            self._test_sequential_so_soak_10,
            self._test_sequential_tc_soak_10,
            self._test_special_json_nested_strings,
            self._test_streaming_usage_with_tc,
            self._test_single_char_prompt,
        ]:
            try:
                sub_results = await loop.run_in_executor(None, test_fn, v)
                results.extend(sub_results)
            except Exception as e:
                test_name = test_fn.__name__.removeprefix("_test_")
                results.append(
                    self.make_result(
                        self.name,
                        test_name,
                        Verdict.ERROR,
                        error=str(e),
                    )
                )

        return results

    # ------------------------------------------------------------------
    # 1. parallel_tc_concurrent_4 -- 4 concurrent TC requests, no cross-talk
    # ------------------------------------------------------------------

    def _test_parallel_tc_concurrent_4(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """4 concurrent parallel-TC requests -- verify no index cross-talk."""
        test_name = "parallel_tc_concurrent_4"
        tools = [_SEARCH_TOOL, _CALC_TOOL]

        def tc_call(idx: int) -> dict[str, Any]:
            result = v.tc_chat_stream(
                [
                    {
                        "role": "user",
                        "content": (
                            f"Request {idx}: search for 'test_{idx}' AND calculate '{idx}+{idx}'."
                        ),
                    }
                ],
                tools,
                tool_choice="auto",
                max_tokens=1024,
            )
            return result

        args_list = [(i,) for i in range(4)]
        concurrent_results = v.concurrent_run(tc_call, args_list, max_workers=4)

        errors: list[str] = []
        for idx, result, err in concurrent_results:
            if err is not None:
                errors.append(f"request[{idx}]: {err}")
                continue
            if stream_budget_exhausted(result):
                continue
            # Each tool call's arguments must be valid JSON
            for i, tc in enumerate(result["tool_calls"]):
                try:
                    args = json.loads(tc["arguments"])
                    if not isinstance(args, dict):
                        errors.append(f"request[{idx}] tc[{i}]: args not dict")
                except (json.JSONDecodeError, TypeError) as e:
                    errors.append(f"request[{idx}] tc[{i}]: invalid JSON: {e}")

            # Check for duplicate IDs within a single request
            ids = [tc["id"] for tc in result["tool_calls"] if tc["id"]]
            if len(ids) != len(set(ids)):
                errors.append(f"request[{idx}]: duplicate tool call IDs: {ids}")

        if errors:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"{len(errors)} error(s): {'; '.join(errors[:5])}",
                )
            ]
        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.PASS,
                detail="All 4 concurrent parallel-TC requests had valid, independent results",
            )
        ]

    # ------------------------------------------------------------------
    # 2. large_tool_result_8k -- 8K tool result in history
    # ------------------------------------------------------------------

    def _test_large_tool_result_8k(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """8K character tool result in conversation history."""
        test_name = "large_tool_result_8k"
        tool = _SEARCH_TOOL

        # Build a ~8K char tool result
        big_result = json.dumps(
            {
                "results": [
                    {
                        "title": f"Result {i}",
                        "content": "x" * 200,
                        "score": round(random.random(), 4),
                    }
                    for i in range(30)
                ],
            }
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Search for machine learning."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_big",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": json.dumps({"q": "machine learning"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_big",
                "content": big_result,
            },
            {"role": "user", "content": "Now search for deep learning."},
        ]

        resp = v.tc_chat(
            messages,
            [tool],
            tool_choice="required",
            max_tokens=512,
        )
        if budget_exhausted(resp):
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.PASS,
                    detail="budget exhausted",
                )
            ]

        tcs = resp.choices[0].message.tool_calls
        if not tcs:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail="No tool call after 8K tool result in history",
                )
            ]

        try:
            args = json.loads(tcs[0].function.arguments)
        except (json.JSONDecodeError, TypeError) as e:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"Tool call args invalid after large result: {e}",
                )
            ]

        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.PASS,
                detail=f"Tool call after 8K result: {tcs[0].function.name}({args})",
            )
        ]

    # ------------------------------------------------------------------
    # 3. 50_tools_auto -- 50 tools selection
    # ------------------------------------------------------------------

    def _test_50_tools_auto(self, v: ValidatorClient) -> list[ScenarioResult]:
        """50 tools defined, model picks one -- grammar compilation stress."""
        test_name = "50_tools_auto"
        tools = [
            make_tool(
                f"tool_{i}",
                {
                    "type": "object",
                    "properties": {f"arg_{i}": {"type": "string"}},
                    "required": [f"arg_{i}"],
                    "additionalProperties": False,
                },
                description=f"Tool number {i}",
            )
            for i in range(50)
        ]

        try:
            resp = v.tc_chat(
                [
                    {
                        "role": "user",
                        "content": "Call tool_25 with arg_25='hello'.",
                    }
                ],
                tools,
                tool_choice="required",
                max_tokens=1024,
            )
        except Exception as e:
            err_str = str(e).lower()
            if "400" in err_str or "422" in err_str:
                return [
                    self.make_result(
                        self.name,
                        test_name,
                        Verdict.PASS,
                        detail=f"Server rejected 50 tools gracefully: {e}",
                    )
                ]
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"50-tools request crashed: {e}",
                )
            ]

        if budget_exhausted(resp):
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.PASS,
                    detail="budget exhausted",
                )
            ]

        tcs = resp.choices[0].message.tool_calls
        if not tcs:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail="No tool call with 50 tools defined",
                )
            ]

        try:
            args = json.loads(tcs[0].function.arguments)
        except (json.JSONDecodeError, TypeError) as e:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"Tool call args invalid with 50 tools: {e}",
                )
            ]

        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.PASS,
                detail=f"50-tools selection: called {tcs[0].function.name!r} with {args}",
            )
        ]

    # ------------------------------------------------------------------
    # 4. so_tc_rapid_switch -- mode switching
    # ------------------------------------------------------------------

    def _test_so_tc_rapid_switch(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """Rapid SO -> TC -> SO -> TC -> SO alternation (5 requests)."""
        test_name = "so_tc_rapid_switch"
        schema = {
            "type": "object",
            "properties": {"v": {"type": "integer"}},
            "required": ["v"],
            "additionalProperties": False,
        }
        tool = make_tool(
            "ping",
            {
                "type": "object",
                "properties": {"host": {"type": "string"}},
                "required": ["host"],
                "additionalProperties": False,
            },
            description="Ping a host",
        )

        errors: list[str] = []
        for i in range(5):
            if i % 2 == 0:
                # Structured output
                try:
                    resp = v.so_chat(
                        [{"role": "user", "content": f"Return v={i}."}],
                        schema,
                        max_tokens=256,
                    )
                    if not budget_exhausted(resp):
                        content = resp.choices[0].message.content or ""
                        data = json.loads(content)
                        if "v" not in data:
                            errors.append(f"round {i} (SO): missing 'v' key")
                except Exception as e:
                    errors.append(f"round {i} (SO): {e}")
            else:
                # Tool calling
                try:
                    resp = v.tc_chat(
                        [{"role": "user", "content": f"Ping host_{i}."}],
                        [tool],
                        tool_choice="required",
                        max_tokens=256,
                    )
                    if not budget_exhausted(resp):
                        tcs = resp.choices[0].message.tool_calls
                        if not tcs:
                            errors.append(f"round {i} (TC): no tool calls")
                except Exception as e:
                    errors.append(f"round {i} (TC): {e}")

        if errors:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"Mode switch errors: {'; '.join(errors)}",
                )
            ]
        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.PASS,
                detail="5 rapid SO/TC alternations completed successfully",
            )
        ]

    # ------------------------------------------------------------------
    # 5. empty_user_message
    # ------------------------------------------------------------------

    def _test_empty_user_message(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """Empty user message content -- should not crash."""
        test_name = "empty_user_message"
        try:
            resp = v.chat(
                [{"role": "user", "content": ""}],
                max_tokens=50,
            )
            fr = resp.choices[0].finish_reason
            if fr in ("stop", "length"):
                return [
                    self.make_result(
                        self.name,
                        test_name,
                        Verdict.PASS,
                        detail=f"Empty message handled, finish_reason={fr!r}",
                    )
                ]
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.INTERESTING,
                    detail=f"Empty message: unexpected finish_reason={fr!r}",
                )
            ]
        except Exception as e:
            err_str = str(e).lower()
            if "400" in err_str or "422" in err_str or "bad request" in err_str:
                return [
                    self.make_result(
                        self.name,
                        test_name,
                        Verdict.PASS,
                        detail=f"Server rejected empty message gracefully: {e}",
                    )
                ]
            if "500" in err_str or "502" in err_str or "503" in err_str:
                return [
                    self.make_result(
                        self.name,
                        test_name,
                        Verdict.FAIL,
                        detail=f"Server crashed on empty message: {e}",
                    )
                ]
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.ERROR,
                    error=f"Unexpected error on empty message: {e}",
                )
            ]

    # ------------------------------------------------------------------
    # 6. sequential_so_soak_10 -- 10 sequential SO requests
    # ------------------------------------------------------------------

    def _test_sequential_so_soak_10(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """10 sequential structured output requests -- soak test for grammar cache."""
        test_name = "sequential_so_soak_10"
        schemas = [
            {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                "required": ["x", "y"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "items": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["items"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {"flag": {"type": "boolean"}},
                "required": ["flag"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {"score": {"type": "number"}},
                "required": ["score"],
                "additionalProperties": False,
            },
        ]

        errors: list[str] = []
        for i in range(10):
            schema = schemas[i % len(schemas)]
            props = schema["properties"]
            assert isinstance(props, dict)
            expected_keys = set(props.keys())
            try:
                resp = v.so_chat(
                    [
                        {
                            "role": "user",
                            "content": f"Generate for schema round {i}.",
                        }
                    ],
                    schema,
                    max_tokens=256,
                )
                if budget_exhausted(resp):
                    continue
                content = resp.choices[0].message.content or ""
                data = json.loads(content)
                actual_keys = set(data.keys())
                if actual_keys != expected_keys:
                    errors.append(
                        f"round {i}: expected {expected_keys}, got {actual_keys}"
                    )
            except Exception as e:
                errors.append(f"round {i}: {e}")

        if errors:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"SO soak errors: {'; '.join(errors[:5])}",
                )
            ]
        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.PASS,
                detail="All 10 sequential SO requests returned correct schemas",
            )
        ]

    # ------------------------------------------------------------------
    # 7. sequential_tc_soak_10 -- 10 sequential TC requests
    # ------------------------------------------------------------------

    def _test_sequential_tc_soak_10(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """10 sequential tool calling requests -- soak test for TC path."""
        test_name = "sequential_tc_soak_10"
        tools = [
            make_tool(
                f"soak_fn_{i}",
                {
                    "type": "object",
                    "properties": {f"param_{i}": {"type": "string"}},
                    "required": [f"param_{i}"],
                    "additionalProperties": False,
                },
                description=f"Soak test function {i}",
            )
            for i in range(5)
        ]

        errors: list[str] = []
        for i in range(10):
            tool_idx = i % len(tools)
            try:
                resp = v.tc_chat(
                    [
                        {
                            "role": "user",
                            "content": f"Call soak_fn_{tool_idx} with param_{tool_idx}='round_{i}'.",
                        }
                    ],
                    [tools[tool_idx]],
                    tool_choice="required",
                    max_tokens=512,
                )
                if budget_exhausted(resp):
                    continue
                tcs = resp.choices[0].message.tool_calls
                if not tcs:
                    errors.append(f"round {i}: no tool calls")
                    continue
                args = json.loads(tcs[0].function.arguments)
                if not isinstance(args, dict):
                    errors.append(f"round {i}: args not dict")
            except Exception as e:
                errors.append(f"round {i}: {e}")

        if errors:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"TC soak errors: {'; '.join(errors[:5])}",
                )
            ]
        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.PASS,
                detail="All 10 sequential TC requests returned valid tool calls",
            )
        ]

    # ------------------------------------------------------------------
    # 8. special_json_nested_strings -- nested JSON in tool args
    # ------------------------------------------------------------------

    def _test_special_json_nested_strings(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """Tool that should produce nested JSON strings in arguments."""
        test_name = "special_json_nested_strings"
        tool = make_tool(
            "store_config",
            {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["key", "value"],
                "additionalProperties": False,
            },
            description="Store a configuration key-value pair (value may be JSON string)",
        )

        resp = v.tc_chat(
            [
                {
                    "role": "user",
                    "content": (
                        "Store config key='settings' with value being the JSON string "
                        '\'{"debug": true, "level": 5}\' (the value should be the literal string).'
                    ),
                }
            ],
            [tool],
            tool_choice="required",
            max_tokens=512,
        )
        if budget_exhausted(resp):
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.PASS,
                    detail="budget exhausted",
                )
            ]

        tcs = resp.choices[0].message.tool_calls
        if not tcs:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail="No tool call for nested JSON strings test",
                )
            ]

        try:
            args = json.loads(tcs[0].function.arguments)
        except (json.JSONDecodeError, TypeError) as e:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"Tool call args not valid JSON (nested string escaping issue?): {e}",
                    response_body=tcs[0].function.arguments,
                )
            ]

        if "key" not in args or "value" not in args:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"Missing expected keys, got {list(args.keys())}",
                )
            ]

        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.PASS,
                detail=f"Nested JSON string handled: key={args['key']!r}, value_len={len(args['value'])}",
            )
        ]

    # ------------------------------------------------------------------
    # 9. streaming_usage_with_tc -- usage tracking with TC
    # ------------------------------------------------------------------

    def _test_streaming_usage_with_tc(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """Streaming tool call with stream_options.include_usage -- verify usage chunk."""
        test_name = "streaming_usage_with_tc"
        tool = _SEARCH_TOOL

        stream = v.tc_chat(
            [{"role": "user", "content": "Search for 'test'."}],
            [tool],
            tool_choice="required",
            max_tokens=512,
            stream=True,
            stream_options={"include_usage": True},
        )
        result = collect_stream(stream)

        usage = result.get("usage")
        if usage is None:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail="No usage chunk in streaming TC response with include_usage=True",
                )
            ]

        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0

        issues: list[str] = []
        if prompt_tokens <= 0:
            issues.append(f"prompt_tokens={prompt_tokens} (expected > 0)")
        if completion_tokens <= 0:
            issues.append(
                f"completion_tokens={completion_tokens} (expected > 0)"
            )

        # Also verify tool calls were produced
        if not result["tool_calls"]:
            issues.append("no tool calls in streaming response")

        if issues:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"Streaming TC usage issues: {'; '.join(issues)}",
                )
            ]
        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.PASS,
                detail=(
                    f"Streaming TC with usage: prompt={prompt_tokens}, "
                    f"completion={completion_tokens}, tool_calls={len(result['tool_calls'])}"
                ),
            )
        ]

    # ------------------------------------------------------------------
    # 10. single_char_prompt -- single character input
    # ------------------------------------------------------------------

    def _test_single_char_prompt(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """Single character input -- minimal prompt stress test."""
        test_name = "single_char_prompt"
        resp = v.chat(
            [{"role": "user", "content": "x"}],
            max_tokens=128,
        )
        msg = resp.choices[0].message
        content = msg.content or ""
        reasoning = (
            getattr(msg, "reasoning_content", None)
            or getattr(msg, "reasoning", None)
            or ""
        )
        fr = resp.choices[0].finish_reason

        if fr in ("stop", "length") and (content or reasoning):
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.PASS,
                    detail=f"Single char handled: {len(content)} chars, finish_reason={fr!r}",
                )
            ]
        if fr in ("stop", "length"):
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.INTERESTING,
                    detail=f"Empty response to single char, finish_reason={fr!r}",
                )
            ]
        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.FAIL,
                detail=f"Unexpected finish_reason={fr!r} for single char prompt",
            )
        ]
