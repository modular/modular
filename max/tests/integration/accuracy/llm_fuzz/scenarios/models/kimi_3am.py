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
"""Kimi K2.5 3am edge cases -- production failures that break at 3am."""

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


@register_scenario
class Kimi3am(BaseScenario):
    name = "kimi_3am"
    description = (
        "Kimi K2.5 3am edge cases -- production failures that break at 3am"
    )
    tags = ["validation", "model:kimi-k2.5", "edge_cases"]
    requires_validator = True
    scenario_type = "validation"
    model_filter = "kimi-k2.5"

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

        # ------------------------------------------------------------------ 1
        # parallel_tc_concurrent_4
        # 4 concurrent requests each producing 2 parallel TCs -- no cross-talk
        # ------------------------------------------------------------------
        try:
            tools = [
                make_tool(
                    "search",
                    {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                        "required": ["q"],
                        "additionalProperties": False,
                    },
                ),
                make_tool(
                    "calc",
                    {
                        "type": "object",
                        "properties": {"expr": {"type": "string"}},
                        "required": ["expr"],
                        "additionalProperties": False,
                    },
                ),
            ]

            def _run_parallel_tc(request_idx: int) -> tuple[str, str | None]:
                stream = v.tc_chat(
                    [
                        {
                            "role": "user",
                            "content": f"Request {request_idx}: search 'test' AND calculate '1+1'. Use both tools.",
                        }
                    ],
                    tools,
                    tool_choice="auto",
                    max_tokens=1024,
                    stream=True,
                )
                r = collect_stream(stream)
                if stream_budget_exhausted(r):
                    return "budget", None
                for tc in r["tool_calls"]:
                    try:
                        json.loads(tc["arguments"])
                    except json.JSONDecodeError as e:
                        return "fail", f"Req[{request_idx}] bad JSON: {e}"
                ids = [tc["id"] for tc in r["tool_calls"] if tc.get("id")]
                if len(ids) != len(set(ids)):
                    return "fail", f"Req[{request_idx}] duplicate IDs: {ids}"
                return "ok", None

            concurrent_results = await loop.run_in_executor(
                None,
                lambda: v.concurrent_run(
                    lambda args: _run_parallel_tc(args[0]),
                    [((i,),) for i in range(4)],
                    max_workers=4,
                ),
            )
            errors = []
            for idx, result_val, err in concurrent_results:
                if err:
                    errors.append(f"Req[{idx}] error: {err}")
                elif result_val:
                    status, detail = result_val
                    if status == "fail":
                        errors.append(detail or f"Req[{idx}] failed")
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            results.append(
                self.make_result(
                    self.name,
                    "parallel_tc_concurrent_4",
                    verdict,
                    detail="; ".join(errors) or "OK",
                )
            )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name,
                    "parallel_tc_concurrent_4",
                    Verdict.ERROR,
                    error=str(e),
                )
            )

        # ------------------------------------------------------------------ 2
        # large_tool_result_8k
        # 8K char tool result in history -- common in RAG pipelines
        # ------------------------------------------------------------------
        try:
            search_tool = make_tool(
                "search",
                {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            )
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
            )  # ~8K chars
            messages = [
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
                                "arguments": '{"query": "machine learning"}',
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
            resp = await loop.run_in_executor(
                None,
                lambda: v.tc_chat(
                    messages,
                    [search_tool],
                    tool_choice="required",
                    max_tokens=512,
                ),
            )
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "large_tool_result_8k",
                        Verdict.INTERESTING,
                        detail="Budget exhausted",
                    )
                )
            else:
                tcs = resp.choices[0].message.tool_calls
                errors = []
                if not tcs:
                    errors.append("No tool call after 8K result")
                else:
                    try:
                        json.loads(tcs[0].function.arguments)
                    except json.JSONDecodeError as e:
                        errors.append(f"Invalid JSON args: {e}")
                verdict = Verdict.PASS if not errors else Verdict.FAIL
                results.append(
                    self.make_result(
                        self.name,
                        "large_tool_result_8k",
                        verdict,
                        detail="; ".join(errors) or "OK",
                    )
                )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name,
                    "large_tool_result_8k",
                    Verdict.ERROR,
                    error=str(e),
                )
            )

        # ------------------------------------------------------------------ 3
        # 50_tools_auto
        # 50 tools defined, ask matching one specific tool
        # ------------------------------------------------------------------
        try:
            fifty_tools = [
                make_tool(
                    f"tool_{i}",
                    {
                        "type": "object",
                        "properties": {f"arg_{i}": {"type": "string"}},
                        "required": [f"arg_{i}"],
                        "additionalProperties": False,
                    },
                )
                for i in range(50)
            ]
            resp = await loop.run_in_executor(
                None,
                lambda: v.tc_chat(
                    [
                        {
                            "role": "user",
                            "content": "Call tool_23 with arg_23='test'.",
                        }
                    ],
                    fifty_tools,
                    tool_choice="auto",
                    max_tokens=1024,
                ),
            )
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "50_tools_auto",
                        Verdict.INTERESTING,
                        detail="Budget exhausted",
                    )
                )
            else:
                tcs = resp.choices[0].message.tool_calls
                errors = []
                if not tcs:
                    errors.append("No tool call with 50 tools defined")
                else:
                    try:
                        json.loads(tcs[0].function.arguments)
                    except json.JSONDecodeError as e:
                        errors.append(f"Invalid JSON args: {e}")
                    valid_names = {f"tool_{i}" for i in range(50)}
                    if tcs[0].function.name not in valid_names:
                        errors.append(
                            f"Selected '{tcs[0].function.name}' not in defined tools"
                        )
                verdict = Verdict.PASS if not errors else Verdict.FAIL
                results.append(
                    self.make_result(
                        self.name,
                        "50_tools_auto",
                        verdict,
                        detail="; ".join(errors) or "OK",
                    )
                )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name, "50_tools_auto", Verdict.ERROR, error=str(e)
                )
            )

        # ------------------------------------------------------------------ 4
        # so_tc_rapid_switch
        # SO -> TC -> SO -> TC -> SO rapid alternation (5 requests)
        # ------------------------------------------------------------------
        try:
            so_schema = {
                "type": "object",
                "properties": {"v": {"type": "integer"}},
                "required": ["v"],
                "additionalProperties": False,
            }
            tc_tool = make_tool(
                "ping",
                {
                    "type": "object",
                    "properties": {"host": {"type": "string"}},
                    "required": ["host"],
                    "additionalProperties": False,
                },
            )
            errors = []
            for i in range(5):
                if i % 2 == 0:

                    def _so_call(idx: int = i) -> Any:
                        return v.so_chat(
                            [
                                {
                                    "role": "user",
                                    "content": f"Return v={idx}.",
                                }
                            ],
                            so_schema,
                            max_tokens=256,
                        )

                    resp = await loop.run_in_executor(None, _so_call)
                    if not budget_exhausted(resp):
                        data = json.loads(resp.choices[0].message.content)
                        if "v" not in data:
                            errors.append(f"SO[{i}] missing 'v'")
                else:

                    def _tc_call(idx: int = i) -> Any:
                        return v.tc_chat(
                            [
                                {
                                    "role": "user",
                                    "content": f"Ping host_{idx}.",
                                }
                            ],
                            [tc_tool],
                            tool_choice="required",
                            max_tokens=256,
                        )

                    resp = await loop.run_in_executor(None, _tc_call)
                    if not budget_exhausted(resp):
                        tcs = resp.choices[0].message.tool_calls
                        if not tcs:
                            errors.append(f"TC[{i}] no tool calls")
                        else:
                            try:
                                json.loads(tcs[0].function.arguments)
                            except json.JSONDecodeError as e:
                                errors.append(f"TC[{i}] bad JSON: {e}")
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            results.append(
                self.make_result(
                    self.name,
                    "so_tc_rapid_switch",
                    verdict,
                    detail="; ".join(errors) or "OK",
                )
            )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name, "so_tc_rapid_switch", Verdict.ERROR, error=str(e)
                )
            )

        # ------------------------------------------------------------------ 5
        # empty_user_message
        # Empty string user content -- should not crash
        # ------------------------------------------------------------------
        try:
            resp = await loop.run_in_executor(
                None,
                lambda: v.chat(
                    [{"role": "user", "content": ""}],
                    max_tokens=50,
                ),
            )
            errors = []
            if resp.choices[0].finish_reason not in ("stop", "length"):
                errors.append(
                    f"Unexpected finish_reason: {resp.choices[0].finish_reason}"
                )
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            results.append(
                self.make_result(
                    self.name,
                    "empty_user_message",
                    verdict,
                    detail="; ".join(errors) or "OK",
                )
            )
        except Exception as e:
            # A 400 error is acceptable for empty content -- server didn't crash
            error_str = str(e).lower()
            if (
                "400" in error_str
                or "invalid" in error_str
                or "empty" in error_str
            ):
                results.append(
                    self.make_result(
                        self.name,
                        "empty_user_message",
                        Verdict.PASS,
                        detail=f"Server rejected gracefully: {e}",
                    )
                )
            else:
                results.append(
                    self.make_result(
                        self.name,
                        "empty_user_message",
                        Verdict.ERROR,
                        error=str(e),
                    )
                )

        # ------------------------------------------------------------------ 6
        # reasoning_only_output
        # Prompt that might produce only reasoning, no content
        # ------------------------------------------------------------------
        try:
            resp = await loop.run_in_executor(
                None,
                lambda: v.chat(
                    [{"role": "user", "content": "Hi"}],
                    max_tokens=100,
                ),
            )
            msg = resp.choices[0].message
            reasoning = getattr(msg, "reasoning_content", None) or getattr(
                msg, "reasoning", None
            )
            errors = []
            if not msg.content and not reasoning:
                errors.append("Both content and reasoning empty")
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            results.append(
                self.make_result(
                    self.name,
                    "reasoning_only_output",
                    verdict,
                    detail="; ".join(errors) or "OK",
                )
            )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name,
                    "reasoning_only_output",
                    Verdict.ERROR,
                    error=str(e),
                )
            )

        # ------------------------------------------------------------------ 7
        # single_char_prompt
        # Single character input with structured output
        # ------------------------------------------------------------------
        try:
            schema = {
                "type": "object",
                "properties": {"r": {"type": "string"}},
                "required": ["r"],
                "additionalProperties": False,
            }
            resp = await loop.run_in_executor(
                None,
                lambda: v.so_chat(
                    [{"role": "user", "content": "?"}],
                    schema,
                    max_tokens=256,
                ),
            )
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "single_char_prompt",
                        Verdict.INTERESTING,
                        detail="Budget exhausted",
                    )
                )
            else:
                data = json.loads(resp.choices[0].message.content)
                errors = []
                if not isinstance(data.get("r"), str):
                    errors.append(f"r not str: {data.get('r')!r}")
                verdict = Verdict.PASS if not errors else Verdict.FAIL
                results.append(
                    self.make_result(
                        self.name,
                        "single_char_prompt",
                        verdict,
                        detail="; ".join(errors) or "OK",
                    )
                )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name, "single_char_prompt", Verdict.ERROR, error=str(e)
                )
            )

        # ------------------------------------------------------------------ 8
        # streaming_usage_with_tc
        # TC streaming with usage tracking
        # ------------------------------------------------------------------
        try:
            tool = make_tool(
                "usage_tc",
                {
                    "type": "object",
                    "properties": {"input": {"type": "string"}},
                    "required": ["input"],
                    "additionalProperties": False,
                },
            )
            stream = await loop.run_in_executor(
                None,
                lambda: v.tc_chat(
                    [{"role": "user", "content": "Use the tool."}],
                    [tool],
                    tool_choice="required",
                    max_tokens=512,
                    stream=True,
                    stream_options={"include_usage": True},
                ),
            )
            result = collect_stream(stream)
            errors = []
            usage = result.get("usage")
            if usage is None:
                errors.append("Missing usage in final chunk")
            else:
                if (
                    not hasattr(usage, "prompt_tokens")
                    or usage.prompt_tokens <= 0
                ):
                    errors.append(
                        f"Bad prompt_tokens: {getattr(usage, 'prompt_tokens', None)}"
                    )
                if (
                    not hasattr(usage, "completion_tokens")
                    or usage.completion_tokens <= 0
                ):
                    errors.append(
                        f"Bad completion_tokens: {getattr(usage, 'completion_tokens', None)}"
                    )
                if (
                    not hasattr(usage, "total_tokens")
                    or usage.total_tokens <= 0
                ):
                    errors.append(
                        f"Bad total_tokens: {getattr(usage, 'total_tokens', None)}"
                    )
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            results.append(
                self.make_result(
                    self.name,
                    "streaming_usage_with_tc",
                    verdict,
                    detail="; ".join(errors) or "OK",
                )
            )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name,
                    "streaming_usage_with_tc",
                    Verdict.ERROR,
                    error=str(e),
                )
            )

        # ------------------------------------------------------------------ 9
        # special_json_nested_strings
        # JSON args with nested quoted strings
        # ------------------------------------------------------------------
        try:
            tool = make_tool(
                "execute",
                {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                    "additionalProperties": False,
                },
            )
            resp = await loop.run_in_executor(
                None,
                lambda: v.tc_chat(
                    [
                        {
                            "role": "user",
                            "content": 'Call execute with code=\'print(json.dumps({"key": "value"}))\'.',
                        }
                    ],
                    [tool],
                    tool_choice="required",
                    max_tokens=1024,
                ),
            )
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "special_json_nested_strings",
                        Verdict.INTERESTING,
                        detail="Budget exhausted",
                    )
                )
            else:
                tcs = resp.choices[0].message.tool_calls
                errors = []
                if not tcs:
                    errors.append("No tool calls returned")
                else:
                    try:
                        args = json.loads(tcs[0].function.arguments)
                        if not isinstance(args.get("code"), str):
                            errors.append(
                                f"code not str: {type(args.get('code'))}"
                            )
                        elif len(args["code"]) == 0:
                            errors.append("code is empty")
                    except json.JSONDecodeError as e:
                        errors.append(f"Nested JSON string broke parsing: {e}")
                verdict = Verdict.PASS if not errors else Verdict.FAIL
                results.append(
                    self.make_result(
                        self.name,
                        "special_json_nested_strings",
                        verdict,
                        detail="; ".join(errors) or "OK",
                    )
                )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name,
                    "special_json_nested_strings",
                    Verdict.ERROR,
                    error=str(e),
                )
            )

        # ------------------------------------------------------------------ 10
        # number_precision
        # JSON args with float precision (3.14159265)
        # ------------------------------------------------------------------
        try:
            tool = make_tool(
                "set_value",
                {
                    "type": "object",
                    "properties": {"value": {"type": "number"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            )
            resp = await loop.run_in_executor(
                None,
                lambda: v.tc_chat(
                    [
                        {
                            "role": "user",
                            "content": "Call set_value with value=3.14159265.",
                        }
                    ],
                    [tool],
                    tool_choice="required",
                    max_tokens=512,
                ),
            )
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "number_precision",
                        Verdict.INTERESTING,
                        detail="Budget exhausted",
                    )
                )
            else:
                tcs = resp.choices[0].message.tool_calls
                errors = []
                if not tcs:
                    errors.append("No tool calls returned")
                else:
                    try:
                        args = json.loads(tcs[0].function.arguments)
                        val = args.get("value")
                        if not isinstance(val, (int, float)):
                            errors.append(f"value not number: {type(val)}")
                        elif (
                            isinstance(val, float)
                            and abs(val - 3.14159265) > 0.01
                        ):
                            errors.append(
                                f"Precision loss: got {val}, expected ~3.14159265"
                            )
                    except json.JSONDecodeError as e:
                        errors.append(f"Invalid JSON args: {e}")
                verdict = Verdict.PASS if not errors else Verdict.FAIL
                results.append(
                    self.make_result(
                        self.name,
                        "number_precision",
                        verdict,
                        detail="; ".join(errors) or "OK",
                    )
                )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name, "number_precision", Verdict.ERROR, error=str(e)
                )
            )

        # ------------------------------------------------------------------ 11
        # sequential_so_soak_10
        # 10 sequential SO requests with different schemas
        # ------------------------------------------------------------------
        try:
            schemas = [
                {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {"count": {"type": "integer"}},
                    "required": ["count"],
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
                    "properties": {
                        "items": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["items"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {"score": {"type": "number"}},
                    "required": ["score"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "enum": ["ok", "error"]}
                    },
                    "required": ["status"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {
                        "a": {"type": "string"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {
                        "nested": {
                            "type": "object",
                            "properties": {"x": {"type": "string"}},
                            "required": ["x"],
                            "additionalProperties": False,
                        }
                    },
                    "required": ["nested"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {
                        "tags": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["tags"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {
                        "value": {
                            "anyOf": [{"type": "string"}, {"type": "integer"}]
                        }
                    },
                    "required": ["value"],
                    "additionalProperties": False,
                },
            ]
            prompts = [
                "Return name='Alice'.",
                "Return count=42.",
                "Return flag=true.",
                "Return items=['a','b','c'].",
                "Return score=9.5.",
                "Return status='ok'.",
                "Return a='hello', b=7.",
                "Return nested.x='deep'.",
                "Return tags=['python','rust'].",
                "Return value=99.",
            ]
            errors = []
            for i in range(10):
                schema_i = schemas[i]
                prompt_i = prompts[i]

                def _so_soak_call(s: Any = schema_i, p: Any = prompt_i) -> Any:
                    return v.so_chat(
                        [{"role": "user", "content": p}],
                        s,
                        max_tokens=512,
                    )

                resp = await loop.run_in_executor(None, _so_soak_call)
                if not budget_exhausted(resp):
                    try:
                        data = json.loads(resp.choices[0].message.content)
                        props = schemas[i]["properties"]
                        assert isinstance(props, dict)
                        expected_key = list(props.keys())[0]
                        if expected_key not in data:
                            errors.append(f"SO[{i}] missing '{expected_key}'")
                    except json.JSONDecodeError as e:
                        errors.append(f"SO[{i}] invalid JSON: {e}")
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            results.append(
                self.make_result(
                    self.name,
                    "sequential_so_soak_10",
                    verdict,
                    detail="; ".join(errors) or "OK",
                )
            )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name,
                    "sequential_so_soak_10",
                    Verdict.ERROR,
                    error=str(e),
                )
            )

        # ------------------------------------------------------------------ 12
        # sequential_tc_soak_10
        # 10 sequential TC requests with different tools
        # ------------------------------------------------------------------
        try:
            soak_tools = [
                make_tool(
                    f"soak_{i}",
                    {
                        "type": "object",
                        "properties": {f"param_{i}": {"type": "string"}},
                        "required": [f"param_{i}"],
                        "additionalProperties": False,
                    },
                )
                for i in range(10)
            ]
            errors = []
            for i in range(10):
                tool_i = soak_tools[i]

                def _tc_soak_call(t: Any = tool_i, idx: int = i) -> Any:
                    return v.tc_chat(
                        [
                            {
                                "role": "user",
                                "content": f"Call soak_{idx} with param_{idx}='value_{idx}'.",
                            }
                        ],
                        [t],
                        tool_choice="required",
                        max_tokens=512,
                    )

                resp = await loop.run_in_executor(None, _tc_soak_call)
                if not budget_exhausted(resp):
                    tcs = resp.choices[0].message.tool_calls
                    if not tcs:
                        errors.append(f"TC[{i}] no tool calls")
                    else:
                        try:
                            args = json.loads(tcs[0].function.arguments)
                            if not isinstance(args, dict):
                                errors.append(f"TC[{i}] args not dict")
                        except json.JSONDecodeError as e:
                            errors.append(f"TC[{i}] invalid JSON: {e}")
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            results.append(
                self.make_result(
                    self.name,
                    "sequential_tc_soak_10",
                    verdict,
                    detail="; ".join(errors) or "OK",
                )
            )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name,
                    "sequential_tc_soak_10",
                    Verdict.ERROR,
                    error=str(e),
                )
            )

        return results
