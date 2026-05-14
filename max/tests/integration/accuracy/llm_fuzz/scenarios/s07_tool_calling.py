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
Scenarios: Tool calling attacks
Target: Crashes from malformed tool definitions, empty args, huge schemas.
Known issues: vLLM #19419 (empty args crash), #27641 (streaming divergence).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig


@register_scenario
class ToolCallingAttacks(BaseScenario):
    name = "tool_calling"
    description = "Malformed tools, empty args, huge definitions, streaming vs non-streaming divergence"
    tags = ["tools", "function_calling", "crash"]

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results = []
        model = config.model

        def with_tools(
            tools: Any,
            content: str = "What's the weather in Paris?",
            **extra: Any,
        ) -> dict[str, Any]:
            p = {
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "tools": tools,
                "max_tokens": 200,
            }
            p.update(extra)
            return p

        valid_tool = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name",
                        },
                    },
                    "required": ["location"],
                },
            },
        }

        # ----- 1. Valid tool call (baseline) -----
        resp = await client.post_json(with_tools([valid_tool]))
        results.append(
            self.make_result(
                self.name,
                "valid_tool_baseline",
                Verdict.PASS if resp.status == 200 else Verdict.FAIL,
                status_code=resp.status,
            )
        )

        # ----- 2. Malformed tool definitions -----
        # Each entry: (tools, expected) where expected is:
        #   "reject"  → 400 = PASS, 200 = FAIL
        #   "accept"  → 200 = PASS, 400 = INTERESTING
        #   "either"  → 200 or 400 = PASS
        # 500 is always FAIL (server crash).
        malformed_tools = {
            # Empty array: normalized to no-tools, 200 is correct
            "empty_tools_array": ([], "accept"),
            # Missing type: Pydantic defaults to "function", 200 is correct
            "tool_missing_type": (
                [{"function": valid_tool["function"]}],
                "accept",
            ),
            # Missing function: structurally invalid, must reject
            "tool_missing_function": ([{"type": "function"}], "reject"),
            # Wrong type value: must reject
            "tool_wrong_type": (
                [{"type": "invalid", "function": valid_tool["function"]}],
                "reject",
            ),
            # Missing name: must reject (required field)
            "tool_missing_name": (
                [
                    {
                        "type": "function",
                        "function": {"description": "test", "parameters": {}},
                    }
                ],
                "reject",
            ),
            # Empty name: must reject (breaks grammar/ID generation)
            "tool_empty_name": (
                [
                    {
                        "type": "function",
                        "function": {"name": "", "parameters": {}},
                    }
                ],
                "reject",
            ),
            # Whitespace in name: must reject (breaks token parsing)
            "tool_name_special_chars": (
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "fn-with spaces & symbols!",
                            "parameters": {},
                        },
                    }
                ],
                "reject",
            ),
            # Null parameters: normalized to empty schema, 200 is correct
            "tool_null_parameters": (
                [
                    {
                        "type": "function",
                        "function": {"name": "test", "parameters": None},
                    }
                ],
                "accept",
            ),
            # Parameters as string: must reject
            "tool_parameters_not_object": (
                [
                    {
                        "type": "function",
                        "function": {"name": "test", "parameters": "string"},
                    }
                ],
                "reject",
            ),
            # Non-dict tool elements: must reject
            "tool_is_string": (["not a tool object"], "reject"),
            "tool_is_number": ([42], "reject"),
            "tool_is_null": ([None], "reject"),
            # Duplicate names: accepted (OpenAI accepts too)
            "duplicate_tool_names": ([valid_tool, valid_tool], "accept"),
        }

        for name, (tools, expected) in malformed_tools.items():
            resp = await client.post_json(with_tools(tools))
            if resp.error == "TIMEOUT":
                v = Verdict.FAIL
            elif resp.status >= 500:
                v = Verdict.FAIL
            elif expected == "reject":
                if 400 <= resp.status < 500:
                    v = Verdict.PASS
                elif resp.status == 200:
                    v = Verdict.FAIL  # should have been rejected
                else:
                    v = Verdict.INTERESTING
            elif expected == "accept":
                if resp.status == 200:
                    v = Verdict.PASS
                elif 400 <= resp.status < 500:
                    v = Verdict.INTERESTING  # stricter than required
                else:
                    v = Verdict.INTERESTING
            else:  # "either"
                v = (
                    Verdict.PASS
                    if resp.status in (200, 400)
                    else Verdict.INTERESTING
                )
            results.append(
                self.make_result(
                    self.name,
                    f"malformed_{name}",
                    v,
                    status_code=resp.status,
                    detail=f"Status {resp.status}"
                    + (f" error: {resp.error}" if resp.error else ""),
                )
            )

        # ----- 3. Huge tool definitions -----
        huge_tool = {
            "type": "function",
            "function": {
                "name": "mega_function",
                "description": "x" * 50000,
                "parameters": {
                    "type": "object",
                    "properties": {
                        f"param_{i}": {
                            "type": "string",
                            "description": f"Param {i}",
                        }
                        for i in range(500)
                    },
                },
            },
        }
        resp = await client.post_json(with_tools([huge_tool]))
        results.append(
            self.make_result(
                self.name,
                "huge_tool_definition",
                Verdict.FAIL
                if resp.status >= 500 or resp.error == "TIMEOUT"
                else Verdict.PASS,
                status_code=resp.status,
                detail=f"500 params, 50k desc: status {resp.status}",
            )
        )

        # ----- 4. Many tools -----
        many_tools = [
            {
                "type": "function",
                "function": {
                    "name": f"function_{i}",
                    "description": f"Function number {i}",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": "string"}},
                    },
                },
            }
            for i in range(200)
        ]
        resp = await client.post_json(with_tools(many_tools))
        results.append(
            self.make_result(
                self.name,
                "200_tools",
                Verdict.FAIL
                if resp.status >= 500 or resp.error == "TIMEOUT"
                else Verdict.PASS,
                status_code=resp.status,
            )
        )

        # ----- 5. Tool call with empty arguments (vLLM #19419) -----
        # Simulate tool response with empty args
        tool_response_payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": "Call the test function"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {"name": "test_fn", "arguments": ""},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_abc123",
                    "content": "result",
                },
            ],
            "tools": [valid_tool],
            "max_tokens": 50,
        }
        resp = await client.post_json(
            tool_response_payload, timeout=config.timeout * 0.5
        )
        results.append(
            self.make_result(
                self.name,
                "empty_tool_arguments",
                Verdict.FAIL if resp.status >= 500 else Verdict.PASS,
                status_code=resp.status,
                detail="Tests vLLM #19419 pattern",
            )
        )

        # ----- 6. Malformed tool_calls in assistant message -----
        bad_tool_calls = {
            "null_arguments": {"name": "test", "arguments": None},
            "invalid_json_arguments": {
                "name": "test",
                "arguments": "{invalid json",
            },
            "arguments_is_number": {"name": "test", "arguments": "42"},
            "missing_name": {"arguments": "{}"},
            "empty_function": {},
        }

        for tc_name, fn_data in bad_tool_calls.items():
            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_test",
                                "type": "function",
                                "function": fn_data,
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_test",
                        "content": "ok",
                    },
                ],
                "max_tokens": 50,
            }
            resp = await client.post_json(payload)
            results.append(
                self.make_result(
                    self.name,
                    f"bad_tool_call_{tc_name}",
                    Verdict.FAIL if resp.status >= 500 else Verdict.PASS,
                    status_code=resp.status,
                )
            )

        # ----- 7. Streaming vs non-streaming tool call divergence -----
        # vLLM #27641: streaming produces different tool call output
        tool_payload = with_tools(
            [valid_tool], content="Get me the weather in Tokyo"
        )
        resp_sync = await client.post_json(tool_payload)
        resp_stream = await client.post_streaming(tool_payload)

        if resp_sync.status == 200 and resp_stream.status == 200:
            try:
                sync_data = json.loads(resp_sync.body)
                sync_calls = (
                    sync_data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("tool_calls", [])
                )

                # Parse streaming chunks for tool calls
                stream_has_tools = False
                for c in resp_stream.chunks or []:
                    if c == "[DONE]":
                        continue
                    try:
                        chunk_data = json.loads(c)
                        delta = chunk_data.get("choices", [{}])[0].get(
                            "delta", {}
                        )
                        if delta.get("tool_calls"):
                            stream_has_tools = True
                            break
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

                if sync_calls and not stream_has_tools:
                    verdict = Verdict.INTERESTING
                    detail = "Sync has tool calls, streaming doesn't"
                elif not sync_calls and stream_has_tools:
                    verdict = Verdict.INTERESTING
                    detail = "Streaming has tool calls, sync doesn't"
                else:
                    verdict = Verdict.PASS
                    detail = "Consistent tool calling behavior"
            except (json.JSONDecodeError, KeyError, IndexError):
                verdict = Verdict.PASS
                detail = "Could not compare (parse error)"
        else:
            verdict = Verdict.PASS
            detail = f"Sync={resp_sync.status}, Stream={resp_stream.status}"

        results.append(
            self.make_result(
                self.name,
                "streaming_vs_sync_tool_divergence",
                verdict,
                detail=detail,
            )
        )

        # ----- 8. tool_choice forcing -----
        force_payloads = {
            "tool_choice_none": with_tools([valid_tool], tool_choice="none"),
            "tool_choice_auto": with_tools([valid_tool], tool_choice="auto"),
            "tool_choice_required": with_tools(
                [valid_tool], tool_choice="required"
            ),
            "tool_choice_specific": with_tools(
                [valid_tool],
                tool_choice={
                    "type": "function",
                    "function": {"name": "get_weather"},
                },
            ),
            "tool_choice_nonexistent": with_tools(
                [valid_tool],
                tool_choice={
                    "type": "function",
                    "function": {"name": "nonexistent_fn"},
                },
            ),
            "tool_choice_invalid": with_tools(
                [valid_tool], tool_choice="invalid_value"
            ),
            "tool_choice_empty_obj": with_tools([valid_tool], tool_choice={}),
        }

        for name, payload in force_payloads.items():
            resp = await client.post_json(payload, timeout=config.timeout * 0.5)
            results.append(
                self.make_result(
                    self.name,
                    name,
                    Verdict.FAIL if resp.status >= 500 else Verdict.PASS,
                    status_code=resp.status,
                )
            )

        # ----- 9. parallel_tool_calls parameter -----
        resp = await client.post_json(
            with_tools([valid_tool], parallel_tool_calls=True)
        )
        results.append(
            self.make_result(
                self.name,
                "parallel_tool_calls_true",
                Verdict.FAIL if resp.status >= 500 else Verdict.PASS,
                status_code=resp.status,
            )
        )

        resp = await client.post_json(
            with_tools([valid_tool], parallel_tool_calls=False)
        )
        results.append(
            self.make_result(
                self.name,
                "parallel_tool_calls_false",
                Verdict.FAIL if resp.status >= 500 else Verdict.PASS,
                status_code=resp.status,
            )
        )

        return results
