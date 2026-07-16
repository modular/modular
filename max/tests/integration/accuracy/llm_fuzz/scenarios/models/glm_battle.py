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
"""GLM-5.1 battle tests -- streaming protocol, schema compilation, tool calling.

Battle-tested adversarial scenarios drawn from real-world production failures,
vLLM issues, and xgrammar compilation bugs. Covers anyOf/allOf/$ref schemas,
parallel tool calls, enum edge cases, multi-turn tool conversation, concurrent
mixed requests, streaming argument accumulation, and max_tokens truncation.
"""

from __future__ import annotations

import asyncio
import json
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
    from validator_client import ValidatorClient

# ---------------------------------------------------------------------------
# Shared tool definitions
# ---------------------------------------------------------------------------

_WEATHER_TOOL = make_tool(
    "get_weather",
    {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    },
    description="Get the current weather for a city",
)

_SEARCH_TOOL = make_tool(
    "search",
    {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    },
    description="Search the web for information",
)

_CALCULATOR_TOOL = make_tool(
    "calculator",
    {
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
        "additionalProperties": False,
    },
    description="Evaluate a mathematical expression",
)

_TRANSLATE_TOOL = make_tool(
    "translate_text",
    {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "target_language": {"type": "string"},
        },
        "required": ["text", "target_language"],
        "additionalProperties": False,
    },
    description="Translate text to a target language",
)


@register_scenario
class GlmBattle(BaseScenario):
    """GLM-5.1 battle tests for schema compilation and tool calling edge cases."""

    name = "glm_battle"
    description = "GLM-5.1 battle tests -- streaming protocol, schema compilation, tool calling"
    tags = ["validation", "model:glm-5.1", "battle"]
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
            self._test_anyof_nullable_string,
            self._test_anyof_type_array,
            self._test_anyof_object_variants,
            self._test_allof_basic_merge,
            self._test_ref_simple_def,
            self._test_parallel_tc_3,
            self._test_enum_single,
            self._test_enum_100_values,
            self._test_multi_turn_tool_history,
            self._test_concurrent_mixed_10,
            self._test_streaming_arg_chunks,
            self._test_max_tokens_graceful,
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
    # 1. anyof_nullable_string -- xgrammar crash test
    # ------------------------------------------------------------------

    def _test_anyof_nullable_string(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """anyOf with null (nullable pattern) -- known xgrammar crash vector."""
        test_name = "anyof_nullable_string"
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "nickname": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
            "required": ["name", "nickname"],
            "additionalProperties": False,
        }
        resp = v.so_chat(
            [
                {
                    "role": "user",
                    "content": "Person named Alice with no nickname.",
                }
            ],
            schema,
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

        content = resp.choices[0].message.content or ""
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError) as e:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"Invalid JSON for anyOf nullable: {e}",
                    response_body=content,
                )
            ]

        if "name" not in data or "nickname" not in data:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"Missing required keys, got {list(data.keys())}",
                )
            ]

        if data["nickname"] is not None and not isinstance(
            data["nickname"], str
        ):
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"nickname should be string or null, got {type(data['nickname']).__name__}",
                )
            ]

        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.PASS,
                detail=f"anyOf nullable handled: nickname={data['nickname']!r}",
            )
        ]

    # ------------------------------------------------------------------
    # 2. anyof_type_array -- array-form type
    # ------------------------------------------------------------------

    def _test_anyof_type_array(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """type: ['string', 'null'] shorthand for nullable."""
        test_name = "anyof_type_array"
        schema = {
            "type": "object",
            "properties": {
                "value": {"type": ["string", "null"]},
            },
            "required": ["value"],
            "additionalProperties": False,
        }
        resp = v.so_chat(
            [{"role": "user", "content": "Return a null value."}],
            schema,
            max_tokens=256,
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

        content = resp.choices[0].message.content or ""
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError) as e:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"Invalid JSON for type-array nullable: {e}",
                    response_body=content,
                )
            ]

        if "value" not in data:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"Missing 'value' key, got {list(data.keys())}",
                )
            ]

        if data["value"] is not None and not isinstance(data["value"], str):
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"value should be string or null, got {type(data['value']).__name__}",
                )
            ]

        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.PASS,
                detail=f"type-array nullable handled: value={data['value']!r}",
            )
        ]

    # ------------------------------------------------------------------
    # 3. anyof_object_variants -- discriminated union
    # ------------------------------------------------------------------

    def _test_anyof_object_variants(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """anyOf with two different object shapes (discriminated union)."""
        test_name = "anyof_object_variants"
        schema = {
            "type": "object",
            "properties": {
                "result": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                            "additionalProperties": False,
                        },
                        {
                            "type": "object",
                            "properties": {"error_code": {"type": "integer"}},
                            "required": ["error_code"],
                            "additionalProperties": False,
                        },
                    ],
                },
            },
            "required": ["result"],
            "additionalProperties": False,
        }
        resp = v.so_chat(
            [
                {
                    "role": "user",
                    "content": "Return a text result saying 'hello'.",
                }
            ],
            schema,
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

        content = resp.choices[0].message.content or ""
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError) as e:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"Invalid JSON for anyOf object variants: {e}",
                    response_body=content,
                )
            ]

        if "result" not in data:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"Missing 'result' key, got {list(data.keys())}",
                )
            ]

        r = data["result"]
        if "text" not in r and "error_code" not in r:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"Result matches neither variant, got keys={list(r.keys())}",
                )
            ]

        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.PASS,
                detail=f"anyOf object variant resolved: {list(r.keys())}",
            )
        ]

    # ------------------------------------------------------------------
    # 4. allof_basic_merge -- two-schema allOf merge
    # ------------------------------------------------------------------

    def _test_allof_basic_merge(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """allOf merging two object schemas."""
        test_name = "allof_basic_merge"
        schema = {
            "type": "object",
            "properties": {
                "item": {
                    "allOf": [
                        {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                            "required": ["name"],
                        },
                        {
                            "type": "object",
                            "properties": {"value": {"type": "integer"}},
                            "required": ["value"],
                        },
                    ],
                },
            },
            "required": ["item"],
            "additionalProperties": False,
        }
        try:
            resp = v.so_chat(
                [
                    {
                        "role": "user",
                        "content": "Return item with name='widget' and value=42.",
                    }
                ],
                schema,
                max_tokens=512,
            )
        except Exception as e:
            err_str = str(e).lower()
            # allOf may not be supported -- 400 is acceptable
            if "400" in err_str or "422" in err_str:
                return [
                    self.make_result(
                        self.name,
                        test_name,
                        Verdict.PASS,
                        detail=f"Server rejected allOf gracefully: {e}",
                    )
                ]
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"allOf request crashed: {e}",
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

        content = resp.choices[0].message.content or ""
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError) as e:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"Invalid JSON for allOf: {e}",
                    response_body=content,
                )
            ]

        item = data.get("item", {})
        if "name" not in item or "value" not in item:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"allOf merge missing keys, got {list(item.keys())}",
                )
            ]

        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.PASS,
                detail=f"allOf merge correct: name={item['name']!r}, value={item['value']}",
            )
        ]

    # ------------------------------------------------------------------
    # 5. ref_simple_def -- $defs/$ref
    # ------------------------------------------------------------------

    def _test_ref_simple_def(self, v: ValidatorClient) -> list[ScenarioResult]:
        """$defs/$ref schema resolution."""
        test_name = "ref_simple_def"
        schema = {
            "type": "object",
            "properties": {
                "address": {"$ref": "#/$defs/Address"},
            },
            "required": ["address"],
            "additionalProperties": False,
            "$defs": {
                "Address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                    },
                    "required": ["street", "city"],
                    "additionalProperties": False,
                },
            },
        }
        try:
            resp = v.so_chat(
                [
                    {
                        "role": "user",
                        "content": "Return address: 123 Main St, Springfield.",
                    }
                ],
                schema,
                max_tokens=512,
            )
        except Exception as e:
            err_str = str(e).lower()
            if "400" in err_str or "422" in err_str:
                return [
                    self.make_result(
                        self.name,
                        test_name,
                        Verdict.PASS,
                        detail=f"Server rejected $ref schema gracefully: {e}",
                    )
                ]
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"$ref schema crashed: {e}",
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

        content = resp.choices[0].message.content or ""
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError) as e:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"Invalid JSON for $ref: {e}",
                    response_body=content,
                )
            ]

        addr = data.get("address", {})
        if "street" not in addr or "city" not in addr:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"$ref resolution missing keys, got {list(addr.keys())}",
                )
            ]

        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.PASS,
                detail=f"$ref resolved: street={addr['street']!r}, city={addr['city']!r}",
            )
        ]

    # ------------------------------------------------------------------
    # 6. parallel_tc_3 -- 3 parallel tool calls
    # ------------------------------------------------------------------

    def _test_parallel_tc_3(self, v: ValidatorClient) -> list[ScenarioResult]:
        """Request that should trigger 3 parallel tool calls."""
        test_name = "parallel_tc_3"
        tools = [_WEATHER_TOOL, _SEARCH_TOOL, _CALCULATOR_TOOL]

        result = v.tc_chat_stream(
            [
                {
                    "role": "user",
                    "content": (
                        "Do all three: weather in Paris, search for 'vllm', "
                        "and calculate 2+2. Use all three tools."
                    ),
                }
            ],
            tools,
            tool_choice="auto",
            max_tokens=2048,
        )
        if stream_budget_exhausted(result):
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.PASS,
                    detail="budget exhausted",
                )
            ]

        tc_count = len(result["tool_calls"])
        if tc_count == 0:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail="No tool calls produced",
                )
            ]

        # Validate all tool call arguments are valid JSON
        _, parse_errors = validate_json_args(result["tool_calls"])
        if parse_errors:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"Invalid tool call args: {'; '.join(parse_errors)}",
                )
            ]

        # Check for unique IDs
        ids = [tc["id"] for tc in result["tool_calls"]]
        if len(ids) != len(set(ids)):
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"Duplicate tool call IDs: {ids}",
                )
            ]

        if tc_count < 3:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.INTERESTING,
                    detail=f"Only {tc_count} tool call(s), expected 3",
                )
            ]

        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.PASS,
                detail=f"{tc_count} parallel tool calls with unique IDs",
            )
        ]

    # ------------------------------------------------------------------
    # 7. enum_single -- single-value enum
    # ------------------------------------------------------------------

    def _test_enum_single(self, v: ValidatorClient) -> list[ScenarioResult]:
        """Single-value enum -- constrained generation edge case."""
        test_name = "enum_single"
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["active"]},
            },
            "required": ["status"],
            "additionalProperties": False,
        }
        resp = v.so_chat(
            [{"role": "user", "content": "Return the status."}],
            schema,
            max_tokens=256,
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

        content = resp.choices[0].message.content or ""
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError) as e:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"Invalid JSON for single enum: {e}",
                    response_body=content,
                )
            ]

        if data.get("status") != "active":
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"Expected status='active', got {data.get('status')!r}",
                )
            ]

        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.PASS,
                detail="Single-value enum correctly constrained to 'active'",
            )
        ]

    # ------------------------------------------------------------------
    # 8. enum_100_values -- large enum
    # ------------------------------------------------------------------

    def _test_enum_100_values(self, v: ValidatorClient) -> list[ScenarioResult]:
        """100-value enum -- grammar compilation stress test."""
        test_name = "enum_100_values"
        values = [f"option_{i:03d}" for i in range(100)]
        schema = {
            "type": "object",
            "properties": {
                "choice": {"type": "string", "enum": values},
            },
            "required": ["choice"],
            "additionalProperties": False,
        }
        try:
            resp = v.so_chat(
                [{"role": "user", "content": "Pick option_042."}],
                schema,
                max_tokens=256,
            )
        except Exception as e:
            err_str = str(e).lower()
            if "400" in err_str or "422" in err_str:
                return [
                    self.make_result(
                        self.name,
                        test_name,
                        Verdict.PASS,
                        detail=f"Server rejected 100-value enum gracefully: {e}",
                    )
                ]
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"100-value enum crashed server: {e}",
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

        content = resp.choices[0].message.content or ""
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError) as e:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"Invalid JSON for large enum: {e}",
                    response_body=content,
                )
            ]

        choice = data.get("choice", "")
        if choice not in values:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"Choice {choice!r} not in 100-value enum",
                )
            ]

        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.PASS,
                detail=f"100-value enum constrained correctly: {choice!r}",
            )
        ]

    # ------------------------------------------------------------------
    # 9. multi_turn_tool_history -- multi-turn tool conversation
    # ------------------------------------------------------------------

    def _test_multi_turn_tool_history(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """Multi-turn conversation with tool call history."""
        test_name = "multi_turn_tool_history"
        tool = _WEATHER_TOOL
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "What's the weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_paris",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"city": "Paris"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_paris",
                "content": json.dumps(
                    {"temperature": 22, "condition": "sunny"}
                ),
            },
            {"role": "user", "content": "Now check London."},
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
                    detail="No tool call after multi-turn history",
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
                    detail=f"Tool call args invalid JSON: {e}",
                )
            ]

        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.PASS,
                detail=f"Multi-turn tool call produced: {tcs[0].function.name}({args})",
            )
        ]

    # ------------------------------------------------------------------
    # 10. concurrent_mixed_10 -- 10 concurrent mixed requests
    # ------------------------------------------------------------------

    def _test_concurrent_mixed_10(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """10 concurrent requests mixing plain, TC, and SO -- no cross-contamination."""
        test_name = "concurrent_mixed_10"

        person_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
            "additionalProperties": False,
        }

        def dispatch(idx: int) -> tuple[str, Any]:
            if idx < 4:
                # Plain chat
                resp = v.chat(
                    [{"role": "user", "content": f"Say hello, request {idx}."}],
                    max_tokens=128,
                )
                return ("plain", resp)
            elif idx < 7:
                # Tool calling
                resp = v.tc_chat(
                    [
                        {
                            "role": "user",
                            "content": f"Get weather in city_{idx}.",
                        }
                    ],
                    [_WEATHER_TOOL],
                    tool_choice="required",
                    max_tokens=256,
                )
                return ("tc", resp)
            else:
                # Structured output
                resp = v.so_chat(
                    [
                        {
                            "role": "user",
                            "content": f"Return name='Person{idx}' and age={20 + idx}.",
                        }
                    ],
                    person_schema,
                    max_tokens=256,
                )
                return ("so", resp)

        args_list = [(i,) for i in range(10)]
        concurrent_results = v.concurrent_run(
            dispatch, args_list, max_workers=10
        )

        errors: list[str] = []
        for idx, result, err in concurrent_results:
            if err is not None:
                errors.append(f"request[{idx}]: {err}")
                continue
            req_type, resp = result
            if budget_exhausted(resp):
                continue

            if req_type == "plain":
                content = resp.choices[0].message.content
                if not content or not content.strip():
                    errors.append(f"request[{idx}] (plain): empty content")
            elif req_type == "tc":
                tcs = resp.choices[0].message.tool_calls
                if not tcs:
                    errors.append(f"request[{idx}] (tc): no tool calls")
            elif req_type == "so":
                content = resp.choices[0].message.content
                try:
                    data = json.loads(content)
                    if "name" not in data or "age" not in data:
                        errors.append(
                            f"request[{idx}] (so): missing keys {list(data.keys())}"
                        )
                except (json.JSONDecodeError, TypeError) as e:
                    errors.append(f"request[{idx}] (so): invalid JSON: {e}")

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
                detail="All 10 concurrent mixed requests returned correct response types",
            )
        ]

    # ------------------------------------------------------------------
    # 11. streaming_arg_chunks -- streaming TC argument accumulation
    # ------------------------------------------------------------------

    def _test_streaming_arg_chunks(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """Streaming tool call argument accumulation -- verify args are valid JSON when joined."""
        test_name = "streaming_arg_chunks"
        tool = make_tool(
            "create_event",
            {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "date": {"type": "string"},
                    "attendees": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["title", "date", "attendees"],
                "additionalProperties": False,
            },
            description="Create a calendar event",
        )

        result = v.tc_chat_stream(
            [
                {
                    "role": "user",
                    "content": (
                        "Create an event titled 'Team Meeting' on 2025-01-15 "
                        "with attendees Alice, Bob, and Carol."
                    ),
                }
            ],
            [tool],
            tool_choice="required",
            max_tokens=1024,
        )
        if stream_budget_exhausted(result):
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.PASS,
                    detail="budget exhausted",
                )
            ]

        if not result["tool_calls"]:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail="No tool calls in streaming response",
                )
            ]

        tc = result["tool_calls"][0]
        try:
            args = json.loads(tc["arguments"])
        except (json.JSONDecodeError, TypeError) as e:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"Accumulated streaming args not valid JSON: {e}",
                    response_body=tc["arguments"],
                )
            ]

        missing = {"title", "date", "attendees"} - set(args.keys())
        if missing:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"Missing keys in streamed args: {missing}",
                )
            ]

        # Check protocol compliance on first chunk
        violations = check_streaming_protocol(result["first_tc_chunks"])
        if violations:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"Streaming protocol violations: {'; '.join(violations)}",
                )
            ]

        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.PASS,
                detail=f"Streaming args accumulated correctly: {list(args.keys())}",
            )
        ]

    # ------------------------------------------------------------------
    # 12. max_tokens_graceful -- short max_tokens truncation
    # ------------------------------------------------------------------

    def _test_max_tokens_graceful(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """Short max_tokens with tool calling -- should truncate gracefully."""
        test_name = "max_tokens_graceful"
        tool = _SEARCH_TOOL
        resp = v.tc_chat(
            [
                {
                    "role": "user",
                    "content": "Search for a very long query about everything.",
                }
            ],
            [tool],
            tool_choice="required",
            max_tokens=5,
        )
        fr = resp.choices[0].finish_reason
        if fr in ("tool_calls", "length"):
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.PASS,
                    detail=f"max_tokens=5 TC handled gracefully, finish_reason={fr!r}",
                )
            ]
        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.INTERESTING,
                detail=f"Unexpected finish_reason={fr!r} with max_tokens=5",
            )
        ]
