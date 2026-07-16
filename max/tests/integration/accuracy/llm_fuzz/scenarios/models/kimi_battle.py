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
"""Kimi K2.5 battle tests -- streaming protocol, anyOf/allOf/$ref, parallel TCs, structural tags."""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from helpers import (
    budget_exhausted,
    check_streaming_protocol,
    make_tool,
    stream_budget_exhausted,
    validate_json_args,
)

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig


@register_scenario
class KimiBattle(BaseScenario):
    name = "kimi_battle"
    description = (
        "Kimi K2.5 battle tests -- streaming protocol, anyOf/allOf/$ref, "
        "parallel TCs, structural tags"
    )
    tags = ["validation", "model:kimi-k2.5", "battle"]
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
        # anyof_nullable_string
        # Kimi/xgrammar crash category: anyOf with nullable string
        # ------------------------------------------------------------------
        try:
            schema = {
                "type": "object",
                "properties": {
                    "value": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                },
                "required": ["value"],
                "additionalProperties": False,
            }
            resp = await loop.run_in_executor(
                None,
                lambda: v.so_chat(
                    [{"role": "user", "content": "Return value='hello'."}],
                    schema,
                    max_tokens=512,
                ),
            )
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "anyof_nullable_string",
                        Verdict.INTERESTING,
                        detail="Budget exhausted",
                    )
                )
            else:
                data = json.loads(resp.choices[0].message.content)
                errors: list[str] = []
                if "value" not in data:
                    errors.append("Missing 'value' key")
                elif not isinstance(data["value"], (str, type(None))):
                    errors.append(
                        f"Expected str|null, got {type(data['value']).__name__}"
                    )
                verdict = Verdict.PASS if not errors else Verdict.FAIL
                results.append(
                    self.make_result(
                        self.name,
                        "anyof_nullable_string",
                        verdict,
                        detail="; ".join(errors) or "OK",
                    )
                )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name,
                    "anyof_nullable_string",
                    Verdict.ERROR,
                    error=str(e),
                )
            )

        # ------------------------------------------------------------------ 2
        # anyof_type_array
        # Array-form type: {"type": ["string", "integer"]}
        # ------------------------------------------------------------------
        try:
            schema = {
                "type": "object",
                "properties": {
                    "data": {"type": ["string", "integer"]},
                },
                "required": ["data"],
                "additionalProperties": False,
            }
            resp = await loop.run_in_executor(
                None,
                lambda: v.so_chat(
                    [{"role": "user", "content": "Return data=42."}],
                    schema,
                    max_tokens=512,
                ),
            )
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "anyof_type_array",
                        Verdict.INTERESTING,
                        detail="Budget exhausted",
                    )
                )
            else:
                data = json.loads(resp.choices[0].message.content)
                errors = []
                if "data" not in data:
                    errors.append("Missing 'data' key")
                elif not isinstance(data["data"], (str, int)):
                    errors.append(
                        f"Expected str|int, got {type(data['data']).__name__}"
                    )
                verdict = Verdict.PASS if not errors else Verdict.FAIL
                results.append(
                    self.make_result(
                        self.name,
                        "anyof_type_array",
                        verdict,
                        detail="; ".join(errors) or "OK",
                    )
                )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name, "anyof_type_array", Verdict.ERROR, error=str(e)
                )
            )

        # ------------------------------------------------------------------ 3
        # anyof_object_variants
        # anyOf with two different object shapes
        # ------------------------------------------------------------------
        try:
            schema = {
                "type": "object",
                "properties": {
                    "item": {
                        "anyOf": [
                            {
                                "type": "object",
                                "properties": {
                                    "kind": {
                                        "type": "string",
                                        "enum": ["text"],
                                    },
                                    "body": {"type": "string"},
                                },
                                "required": ["kind", "body"],
                                "additionalProperties": False,
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "kind": {
                                        "type": "string",
                                        "enum": ["number"],
                                    },
                                    "value": {"type": "integer"},
                                },
                                "required": ["kind", "value"],
                                "additionalProperties": False,
                            },
                        ],
                    },
                },
                "required": ["item"],
                "additionalProperties": False,
            }
            resp = await loop.run_in_executor(
                None,
                lambda: v.so_chat(
                    [
                        {
                            "role": "user",
                            "content": "Return a text item with body='hello'.",
                        }
                    ],
                    schema,
                    max_tokens=512,
                ),
            )
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "anyof_object_variants",
                        Verdict.INTERESTING,
                        detail="Budget exhausted",
                    )
                )
            else:
                data = json.loads(resp.choices[0].message.content)
                errors = []
                item = data.get("item", {})
                if "kind" not in item:
                    errors.append("Missing 'kind' in item")
                elif item["kind"] == "text" and "body" not in item:
                    errors.append("kind=text but missing 'body'")
                elif item["kind"] == "number" and "value" not in item:
                    errors.append("kind=number but missing 'value'")
                elif item["kind"] not in ("text", "number"):
                    errors.append(f"Unexpected kind: {item['kind']}")
                verdict = Verdict.PASS if not errors else Verdict.FAIL
                results.append(
                    self.make_result(
                        self.name,
                        "anyof_object_variants",
                        verdict,
                        detail="; ".join(errors) or "OK",
                    )
                )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name,
                    "anyof_object_variants",
                    Verdict.ERROR,
                    error=str(e),
                )
            )

        # ------------------------------------------------------------------ 4
        # allof_basic_merge
        # Two schemas merged via allOf -- both sets of properties present
        # ------------------------------------------------------------------
        try:
            schema = {
                "type": "object",
                "allOf": [
                    {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    },
                    {
                        "type": "object",
                        "properties": {"age": {"type": "integer"}},
                        "required": ["age"],
                    },
                ],
                "additionalProperties": False,
            }
            resp = await loop.run_in_executor(
                None,
                lambda: v.so_chat(
                    [
                        {
                            "role": "user",
                            "content": "Return name='Alice', age=30.",
                        }
                    ],
                    schema,
                    max_tokens=512,
                ),
            )
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "allof_basic_merge",
                        Verdict.INTERESTING,
                        detail="Budget exhausted",
                    )
                )
            else:
                data = json.loads(resp.choices[0].message.content)
                errors = []
                if not isinstance(data.get("name"), str):
                    errors.append(f"name not str: {data.get('name')!r}")
                if not isinstance(data.get("age"), int):
                    errors.append(f"age not int: {data.get('age')!r}")
                verdict = Verdict.PASS if not errors else Verdict.FAIL
                results.append(
                    self.make_result(
                        self.name,
                        "allof_basic_merge",
                        verdict,
                        detail="; ".join(errors) or "OK",
                    )
                )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name, "allof_basic_merge", Verdict.ERROR, error=str(e)
                )
            )

        # ------------------------------------------------------------------ 5
        # ref_simple_def
        # Simple $defs/$ref reference
        # ------------------------------------------------------------------
        try:
            schema = {
                "type": "object",
                "properties": {
                    "user": {"$ref": "#/$defs/User"},
                },
                "required": ["user"],
                "additionalProperties": False,
                "$defs": {
                    "User": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "email": {"type": "string"},
                        },
                        "required": ["name", "email"],
                        "additionalProperties": False,
                    },
                },
            }
            resp = await loop.run_in_executor(
                None,
                lambda: v.so_chat(
                    [
                        {
                            "role": "user",
                            "content": "Return user name='Bob', email='bob@example.com'.",
                        }
                    ],
                    schema,
                    max_tokens=512,
                ),
            )
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "ref_simple_def",
                        Verdict.INTERESTING,
                        detail="Budget exhausted",
                    )
                )
            else:
                data = json.loads(resp.choices[0].message.content)
                errors = []
                user = data.get("user", {})
                if not isinstance(user.get("name"), str):
                    errors.append(f"user.name not str: {user.get('name')!r}")
                if not isinstance(user.get("email"), str):
                    errors.append(f"user.email not str: {user.get('email')!r}")
                verdict = Verdict.PASS if not errors else Verdict.FAIL
                results.append(
                    self.make_result(
                        self.name,
                        "ref_simple_def",
                        verdict,
                        detail="; ".join(errors) or "OK",
                    )
                )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name, "ref_simple_def", Verdict.ERROR, error=str(e)
                )
            )

        # ------------------------------------------------------------------ 6
        # ref_multiple_defs
        # Multiple $defs referenced in properties
        # ------------------------------------------------------------------
        try:
            schema = {
                "type": "object",
                "properties": {
                    "source": {"$ref": "#/$defs/Endpoint"},
                    "target": {"$ref": "#/$defs/Endpoint"},
                    "config": {"$ref": "#/$defs/Config"},
                },
                "required": ["source", "target", "config"],
                "additionalProperties": False,
                "$defs": {
                    "Endpoint": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                            "port": {"type": "integer"},
                        },
                        "required": ["url", "port"],
                        "additionalProperties": False,
                    },
                    "Config": {
                        "type": "object",
                        "properties": {
                            "timeout": {"type": "integer"},
                            "retries": {"type": "integer"},
                        },
                        "required": ["timeout", "retries"],
                        "additionalProperties": False,
                    },
                },
            }
            resp = await loop.run_in_executor(
                None,
                lambda: v.so_chat(
                    [
                        {
                            "role": "user",
                            "content": "source url=http://a.com port=80, target url=http://b.com port=443, config timeout=30 retries=3.",
                        }
                    ],
                    schema,
                    max_tokens=1024,
                ),
            )
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "ref_multiple_defs",
                        Verdict.INTERESTING,
                        detail="Budget exhausted",
                    )
                )
            else:
                data = json.loads(resp.choices[0].message.content)
                errors = []
                for key in ("source", "target"):
                    ep = data.get(key, {})
                    if not isinstance(ep.get("url"), str):
                        errors.append(f"{key}.url not str")
                    if not isinstance(ep.get("port"), int):
                        errors.append(f"{key}.port not int")
                cfg = data.get("config", {})
                if not isinstance(cfg.get("timeout"), int):
                    errors.append("config.timeout not int")
                if not isinstance(cfg.get("retries"), int):
                    errors.append("config.retries not int")
                verdict = Verdict.PASS if not errors else Verdict.FAIL
                results.append(
                    self.make_result(
                        self.name,
                        "ref_multiple_defs",
                        verdict,
                        detail="; ".join(errors) or "OK",
                    )
                )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name, "ref_multiple_defs", Verdict.ERROR, error=str(e)
                )
            )

        # ------------------------------------------------------------------ 7
        # parallel_tc_3
        # Request producing 3 parallel tool calls -- verify indices 0,1,2
        # ------------------------------------------------------------------
        try:
            tools = [
                make_tool(
                    "get_weather",
                    {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                ),
                make_tool(
                    "get_stock",
                    {
                        "type": "object",
                        "properties": {"ticker": {"type": "string"}},
                        "required": ["ticker"],
                        "additionalProperties": False,
                    },
                ),
                make_tool(
                    "get_time",
                    {
                        "type": "object",
                        "properties": {"timezone": {"type": "string"}},
                        "required": ["timezone"],
                        "additionalProperties": False,
                    },
                ),
            ]
            result = await loop.run_in_executor(
                None,
                lambda: v.tc_chat_stream(
                    [
                        {
                            "role": "user",
                            "content": "Weather in Paris, stock price of AAPL, and current time in UTC. Use all three tools.",
                        }
                    ],
                    tools,
                    tool_choice="auto",
                    max_tokens=1024,
                ),
            )
            if stream_budget_exhausted(result):
                results.append(
                    self.make_result(
                        self.name,
                        "parallel_tc_3",
                        Verdict.INTERESTING,
                        detail="Budget exhausted",
                    )
                )
            else:
                errors = []
                tcs = result["tool_calls"]
                if len(tcs) < 3:
                    errors.append(f"Expected 3 tool calls, got {len(tcs)}")
                else:
                    indices = sorted(result["first_tc_chunks"].keys())
                    if indices != list(range(len(indices))):
                        errors.append(f"Non-sequential indices: {indices}")
                    ids = [tc["id"] for tc in tcs]
                    if len(ids) != len(set(ids)):
                        errors.append(f"Duplicate IDs: {ids}")
                    _, parse_errors = validate_json_args(tcs)
                    errors.extend(parse_errors)
                verdict = Verdict.PASS if not errors else Verdict.FAIL
                results.append(
                    self.make_result(
                        self.name,
                        "parallel_tc_3",
                        verdict,
                        detail="; ".join(errors) or "OK",
                    )
                )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name, "parallel_tc_3", Verdict.ERROR, error=str(e)
                )
            )

        # ------------------------------------------------------------------ 8
        # parallel_tc_10
        # 10 parallel tool calls -- verify all present
        # ------------------------------------------------------------------
        try:
            tools_10 = [
                make_tool(
                    f"action_{i}",
                    {
                        "type": "object",
                        "properties": {f"arg_{i}": {"type": "string"}},
                        "required": [f"arg_{i}"],
                        "additionalProperties": False,
                    },
                )
                for i in range(10)
            ]
            result = await loop.run_in_executor(
                None,
                lambda: v.tc_chat_stream(
                    [
                        {
                            "role": "user",
                            "content": (
                                "Call ALL ten tools: "
                                + ", ".join(
                                    f"action_{i} with arg_{i}='val_{i}'"
                                    for i in range(10)
                                )
                                + ". Use every single tool."
                            ),
                        }
                    ],
                    tools_10,
                    tool_choice="auto",
                    max_tokens=4096,
                ),
            )
            if stream_budget_exhausted(result):
                results.append(
                    self.make_result(
                        self.name,
                        "parallel_tc_10",
                        Verdict.INTERESTING,
                        detail="Budget exhausted",
                    )
                )
            else:
                errors = []
                tcs = result["tool_calls"]
                if len(tcs) < 10:
                    errors.append(f"Expected 10 tool calls, got {len(tcs)}")
                ids = [tc["id"] for tc in tcs if tc.get("id")]
                if len(ids) != len(set(ids)):
                    errors.append(f"Duplicate IDs among {len(ids)} calls")
                _, parse_errors = validate_json_args(tcs)
                errors.extend(parse_errors)
                verdict = Verdict.PASS if not errors else Verdict.FAIL
                results.append(
                    self.make_result(
                        self.name,
                        "parallel_tc_10",
                        verdict,
                        detail="; ".join(errors) or "OK",
                    )
                )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name, "parallel_tc_10", Verdict.ERROR, error=str(e)
                )
            )

        # ------------------------------------------------------------------ 9
        # enum_single
        # Single-value enum -- grammar must constrain to exactly one value
        # ------------------------------------------------------------------
        try:
            schema = {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["active"]},
                },
                "required": ["status"],
                "additionalProperties": False,
            }
            resp = await loop.run_in_executor(
                None,
                lambda: v.so_chat(
                    [{"role": "user", "content": "Return the status."}],
                    schema,
                    max_tokens=256,
                ),
            )
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "enum_single",
                        Verdict.INTERESTING,
                        detail="Budget exhausted",
                    )
                )
            else:
                data = json.loads(resp.choices[0].message.content)
                errors = []
                if data.get("status") != "active":
                    errors.append(
                        f"Expected 'active', got {data.get('status')!r}"
                    )
                verdict = Verdict.PASS if not errors else Verdict.FAIL
                results.append(
                    self.make_result(
                        self.name,
                        "enum_single",
                        verdict,
                        detail="; ".join(errors) or "OK",
                    )
                )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name, "enum_single", Verdict.ERROR, error=str(e)
                )
            )

        # ------------------------------------------------------------------ 10
        # enum_100_values
        # Large 100-value enum -- grammar compilation stress test
        # ------------------------------------------------------------------
        try:
            enum_values = [f"option_{i:03d}" for i in range(100)]
            target = random.choice(enum_values)
            schema = {
                "type": "object",
                "properties": {
                    "selected": {"type": "string", "enum": enum_values},
                },
                "required": ["selected"],
                "additionalProperties": False,
            }
            resp = await loop.run_in_executor(
                None,
                lambda: v.so_chat(
                    [{"role": "user", "content": f"Select '{target}'."}],
                    schema,
                    max_tokens=256,
                ),
            )
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "enum_100_values",
                        Verdict.INTERESTING,
                        detail="Budget exhausted",
                    )
                )
            else:
                data = json.loads(resp.choices[0].message.content)
                errors = []
                if data.get("selected") not in enum_values:
                    errors.append(
                        f"Value {data.get('selected')!r} not in 100-value enum"
                    )
                verdict = Verdict.PASS if not errors else Verdict.FAIL
                results.append(
                    self.make_result(
                        self.name,
                        "enum_100_values",
                        verdict,
                        detail="; ".join(errors) or "OK",
                    )
                )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name, "enum_100_values", Verdict.ERROR, error=str(e)
                )
            )

        # ------------------------------------------------------------------ 11
        # multi_turn_tool_history
        # 3-step tool calling conversation
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
            messages = [
                {"role": "user", "content": "Search for 'python'."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_step1",
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": '{"query": "python"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_step1",
                    "content": '{"results": ["Python docs", "Learn Python"]}',
                },
                {"role": "user", "content": "Now search for 'rust'."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_step2",
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": '{"query": "rust"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_step2",
                    "content": '{"results": ["Rust docs", "Learn Rust"]}',
                },
                {"role": "user", "content": "Search for 'golang' now."},
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
                        "multi_turn_tool_history",
                        Verdict.INTERESTING,
                        detail="Budget exhausted",
                    )
                )
            else:
                tcs = resp.choices[0].message.tool_calls
                errors = []
                if not tcs:
                    errors.append("No tool call on 3rd turn")
                else:
                    try:
                        args = json.loads(tcs[0].function.arguments)
                        if not isinstance(args.get("query"), str):
                            errors.append(f"query not str: {args}")
                    except json.JSONDecodeError as e:
                        errors.append(f"Invalid JSON args: {e}")
                verdict = Verdict.PASS if not errors else Verdict.FAIL
                results.append(
                    self.make_result(
                        self.name,
                        "multi_turn_tool_history",
                        verdict,
                        detail="; ".join(errors) or "OK",
                    )
                )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name,
                    "multi_turn_tool_history",
                    Verdict.ERROR,
                    error=str(e),
                )
            )

        # ------------------------------------------------------------------ 12
        # max_tokens_graceful
        # max_tokens=20 with tool call -- finish_reason should be "length"
        # ------------------------------------------------------------------
        try:
            tool = make_tool(
                "complex_tool",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "filters": {"type": "string"},
                        "options": {"type": "string"},
                    },
                    "required": ["query", "filters", "options"],
                    "additionalProperties": False,
                },
            )
            result = await loop.run_in_executor(
                None,
                lambda: v.tc_chat_stream(
                    [
                        {
                            "role": "user",
                            "content": "Search with complex filters and many options.",
                        }
                    ],
                    [tool],
                    tool_choice="required",
                    max_tokens=20,
                ),
            )
            errors = []
            # With only 20 tokens, the model should hit the limit
            if result["finish_reason"] == "length":
                pass  # expected
            elif result["finish_reason"] == "tool_calls":
                # Model managed within budget -- fine, verify JSON
                _, parse_errors = validate_json_args(result["tool_calls"])
                errors.extend(parse_errors)
            elif result["finish_reason"] is None:
                errors.append("No finish_reason returned")
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            results.append(
                self.make_result(
                    self.name,
                    "max_tokens_graceful",
                    verdict,
                    detail=f"finish_reason={result['finish_reason']}; "
                    + ("; ".join(errors) or "OK"),
                )
            )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name,
                    "max_tokens_graceful",
                    Verdict.ERROR,
                    error=str(e),
                )
            )

        # ------------------------------------------------------------------ 13
        # concurrent_mixed_10
        # 10 concurrent requests (5 SO + 5 TC) -- verify isolation
        # ------------------------------------------------------------------
        try:
            so_schema = {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
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

            def run_so(idx: int) -> tuple[str, str | None]:
                resp = v.so_chat(
                    [{"role": "user", "content": f"Return value={idx}."}],
                    so_schema,
                    max_tokens=256,
                )
                if budget_exhausted(resp):
                    return "budget", None
                data = json.loads(resp.choices[0].message.content)
                if not isinstance(data.get("value"), int):
                    return (
                        "fail",
                        f"SO[{idx}] value not int: {data.get('value')!r}",
                    )
                return "ok", None

            def run_tc(idx: int) -> tuple[str, str | None]:
                resp = v.tc_chat(
                    [{"role": "user", "content": f"Ping host_{idx}."}],
                    [tc_tool],
                    tool_choice="required",
                    max_tokens=256,
                )
                if budget_exhausted(resp):
                    return "budget", None
                tcs = resp.choices[0].message.tool_calls
                if not tcs:
                    return "fail", f"TC[{idx}] no tool calls"
                try:
                    json.loads(tcs[0].function.arguments)
                except json.JSONDecodeError as e:
                    return "fail", f"TC[{idx}] bad JSON: {e}"
                return "ok", None

            args_list: list[tuple[Any, ...]] = []
            for i in range(5):
                args_list.append((run_so, i))
            for i in range(5):
                args_list.append((run_tc, i))

            def _run_mixed(fn: Callable[[int], Any], idx: int) -> Any:
                return fn(idx)

            concurrent_results = await loop.run_in_executor(
                None,
                lambda: v.concurrent_run(
                    lambda args: args[0](args[1]),
                    [((run_so, i),) for i in range(5)]
                    + [((run_tc, i),) for i in range(5)],
                    max_workers=10,
                ),
            )
            errors = []
            for idx, result_val, err in concurrent_results:
                if err:
                    errors.append(f"Request[{idx}] error: {err}")
                elif result_val:
                    status, detail = result_val
                    if status == "fail":
                        errors.append(detail or f"Request[{idx}] failed")
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            results.append(
                self.make_result(
                    self.name,
                    "concurrent_mixed_10",
                    verdict,
                    detail="; ".join(errors[:5]) or "OK",
                )
            )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name,
                    "concurrent_mixed_10",
                    Verdict.ERROR,
                    error=str(e),
                )
            )

        # ------------------------------------------------------------------ 14
        # streaming_arg_chunks
        # Streaming TC -- accumulated arguments form valid JSON
        # ------------------------------------------------------------------
        try:
            tool = make_tool(
                "analyze",
                {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "language": {"type": "string"},
                        "detailed": {"type": "boolean"},
                    },
                    "required": ["text", "language", "detailed"],
                    "additionalProperties": False,
                },
            )
            result = await loop.run_in_executor(
                None,
                lambda: v.tc_chat_stream(
                    [
                        {
                            "role": "user",
                            "content": "Analyze the text 'Hello world' in English with detailed=true.",
                        }
                    ],
                    [tool],
                    tool_choice="required",
                    max_tokens=1024,
                ),
            )
            if stream_budget_exhausted(result):
                results.append(
                    self.make_result(
                        self.name,
                        "streaming_arg_chunks",
                        Verdict.INTERESTING,
                        detail="Budget exhausted",
                    )
                )
            else:
                errors = []
                if not result["tool_calls"]:
                    errors.append("No tool calls in stream")
                else:
                    tc = result["tool_calls"][0]
                    try:
                        args = json.loads(tc["arguments"])
                        if not isinstance(args, dict):
                            errors.append(
                                f"Accumulated args not dict: {type(args)}"
                            )
                    except json.JSONDecodeError as e:
                        errors.append(
                            f"Accumulated args not valid JSON: {e} -- raw: {tc['arguments'][:200]}"
                        )
                    # Check streaming protocol compliance
                    protocol_issues = check_streaming_protocol(
                        result["first_tc_chunks"]
                    )
                    errors.extend(protocol_issues)
                verdict = Verdict.PASS if not errors else Verdict.FAIL
                results.append(
                    self.make_result(
                        self.name,
                        "streaming_arg_chunks",
                        verdict,
                        detail="; ".join(errors) or "OK",
                    )
                )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name,
                    "streaming_arg_chunks",
                    Verdict.ERROR,
                    error=str(e),
                )
            )

        # ------------------------------------------------------------------ 15
        # special_chars_in_args
        # Tool args containing newlines, quotes, unicode
        # ------------------------------------------------------------------
        try:
            tool = make_tool(
                "process_text",
                {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
            )
            # Ask for text with special characters
            resp = await loop.run_in_executor(
                None,
                lambda: v.tc_chat(
                    [
                        {
                            "role": "user",
                            "content": (
                                "Call process_text with text containing a newline, a quote, "
                                'and a unicode character. For example: "line1\\nline2 said \\"hello\\" with emoji \\u2764"'
                            ),
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
                        "special_chars_in_args",
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
                        if not isinstance(args.get("text"), str):
                            errors.append(
                                f"text not str: {type(args.get('text'))}"
                            )
                        elif len(args["text"]) == 0:
                            errors.append("text is empty")
                    except json.JSONDecodeError as e:
                        errors.append(f"Invalid JSON with special chars: {e}")
                verdict = Verdict.PASS if not errors else Verdict.FAIL
                results.append(
                    self.make_result(
                        self.name,
                        "special_chars_in_args",
                        verdict,
                        detail="; ".join(errors) or "OK",
                    )
                )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name,
                    "special_chars_in_args",
                    Verdict.ERROR,
                    error=str(e),
                )
            )

        return results
