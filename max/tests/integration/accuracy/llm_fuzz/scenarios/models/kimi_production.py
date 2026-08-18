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
"""Kimi K2.5 production readiness -- long context, error recovery, rate limiting, token counting.

Tests the specific production failure modes and edge cases encountered with
Kimi K2.5 deployments: multi-turn context accumulation, graceful error recovery,
burst concurrency, token count consistency between streaming and non-streaming,
max_tokens boundary behavior, and mixed SO/TC conversations.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from helpers import (
    budget_exhausted,
    collect_stream,
    make_tool,
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
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    },
    description="Search the web for information",
)

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


@register_scenario
class KimiProduction(BaseScenario):
    """Kimi K2.5 production readiness validation."""

    name = "kimi_production"
    description = (
        "Kimi K2.5 production readiness -- long context, error recovery, "
        "rate limiting, token counting"
    )
    tags = ["validation", "model:kimi-k2.5", "production"]
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

        for test_fn in [
            self._test_long_context_10_turn,
            self._test_error_recovery_after_400,
            self._test_concurrent_burst_10,
            self._test_token_count_consistency,
            self._test_max_tokens_edge_1,
            self._test_tools_empty_array,
            self._test_mixed_so_tc_conversation,
            self._test_streaming_stability_100_chunks,
            self._test_model_variant_check,
            self._test_backwards_compat_null_params,
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
    # 1. long_context_10_turn
    # ------------------------------------------------------------------

    def _test_long_context_10_turn(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """10-turn conversation building up context -- verify coherent response at the end."""
        test_name = "long_context_10_turn"
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Keep answers concise.",
            },
        ]
        prompts = [
            "What is the capital of France?",
            "What language do they speak there?",
            "Name a famous landmark in that city.",
            "When was it built?",
            "How tall is it?",
            "Who designed it?",
            "What other famous structures did that person design?",
            "Which country is the most famous one located in?",
            "What is the population of that country?",
            "Summarize everything we discussed in 2 sentences.",
        ]

        last_content = ""
        for i, prompt in enumerate(prompts):
            messages.append({"role": "user", "content": prompt})
            try:
                resp = v.chat(messages, max_tokens=256)
            except Exception as e:
                return [
                    self.make_result(
                        self.name,
                        test_name,
                        Verdict.FAIL,
                        detail=f"Turn {i + 1} failed: {e}",
                    )
                ]

            content = resp.choices[0].message.content or ""
            reasoning = (
                getattr(resp.choices[0].message, "reasoning_content", None)
                or getattr(resp.choices[0].message, "reasoning", None)
                or ""
            )
            if not content and not reasoning:
                return [
                    self.make_result(
                        self.name,
                        test_name,
                        Verdict.FAIL,
                        detail=f"Turn {i + 1}: empty response",
                    )
                ]

            messages.append({"role": "assistant", "content": content})
            last_content = content

        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.PASS,
                detail=f"All 10 turns produced responses. Last: {last_content[:100]!r}",
            )
        ]

    # ------------------------------------------------------------------
    # 2. error_recovery_after_400
    # ------------------------------------------------------------------

    def _test_error_recovery_after_400(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """Send invalid request (bad tool_choice), then valid request -- verify recovery."""
        test_name = "error_recovery_after_400"
        tool = _SEARCH_TOOL

        # Send a bad request: tool_choice references a nonexistent function
        try:
            v.tc_chat(
                [{"role": "user", "content": "Search for something."}],
                [tool],
                tool_choice={
                    "type": "function",
                    "function": {"name": "nonexistent_tool"},
                },
                max_tokens=64,
            )
            # If it didn't raise, that's unexpected but not a failure of recovery
        except Exception:
            pass  # Expected -- bad tool_choice should error

        # Now send a valid request -- server must recover
        try:
            resp = v.tc_chat(
                [{"role": "user", "content": "Search for Python tutorials."}],
                [tool],
                tool_choice="required",
                max_tokens=512,
            )
        except Exception as e:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"Valid request failed after bad request: {e}",
                )
            ]

        if budget_exhausted(resp):
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.PASS,
                    detail="Recovery succeeded (budget exhausted on valid request)",
                )
            ]

        tcs = resp.choices[0].message.tool_calls
        if not tcs:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail="Valid request after error produced no tool calls",
                )
            ]

        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.PASS,
                detail=f"Server recovered: valid request produced tool call {tcs[0].function.name!r}",
            )
        ]

    # ------------------------------------------------------------------
    # 3. concurrent_burst_10
    # ------------------------------------------------------------------

    def _test_concurrent_burst_10(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """10 concurrent requests -- verify all return valid responses."""
        test_name = "concurrent_burst_10"

        def burst_call(idx: int) -> Any:
            return v.chat(
                [
                    {
                        "role": "user",
                        "content": f"Request {idx}: what is {idx} + {idx}?",
                    }
                ],
                max_tokens=128,
            )

        args_list = [(i,) for i in range(10)]
        concurrent_results = v.concurrent_run(
            burst_call, args_list, max_workers=10
        )

        errors: list[str] = []
        for idx, result, err in concurrent_results:
            if err is not None:
                errors.append(f"request[{idx}]: {err}")
                continue
            resp = result
            msg = resp.choices[0].message
            content = msg.content or ""
            reasoning = (
                getattr(msg, "reasoning_content", None)
                or getattr(msg, "reasoning", None)
                or ""
            )
            if not content and not reasoning:
                errors.append(f"request[{idx}]: empty response")

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
                detail="All 10 concurrent requests returned valid responses",
            )
        ]

    # ------------------------------------------------------------------
    # 4. token_count_consistency
    # ------------------------------------------------------------------

    def _test_token_count_consistency(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """Non-streaming vs streaming token counts should be close (within 10%)."""
        test_name = "token_count_consistency"
        messages = [
            {
                "role": "user",
                "content": "Explain what a neural network is in 3 sentences.",
            }
        ]

        # Non-streaming
        ns_resp = v.chat(messages, max_tokens=256)
        ns_usage = ns_resp.usage
        if ns_usage is None:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail="Non-streaming response has no usage field",
                )
            ]

        # Streaming with usage
        stream = v.chat(
            messages,
            max_tokens=256,
            stream=True,
            stream_options={"include_usage": True},
        )
        s_result = collect_stream(stream)
        s_usage = s_result.get("usage")
        if s_usage is None:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail="Streaming response has no usage chunk despite include_usage=True",
                )
            ]

        ns_prompt = ns_usage.prompt_tokens
        s_prompt = getattr(s_usage, "prompt_tokens", 0) or 0

        # Prompt tokens should be identical (same input)
        if ns_prompt > 0 and s_prompt > 0 and ns_prompt != s_prompt:
            diff_pct = (
                abs(ns_prompt - s_prompt) / max(ns_prompt, s_prompt) * 100
            )
            if diff_pct > 10:
                return [
                    self.make_result(
                        self.name,
                        test_name,
                        Verdict.FAIL,
                        detail=(
                            f"Prompt token mismatch > 10%: "
                            f"non-stream={ns_prompt}, stream={s_prompt} ({diff_pct:.1f}%)"
                        ),
                    )
                ]

        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.PASS,
                detail=f"Token counts consistent: non-stream prompt={ns_prompt}, stream prompt={s_prompt}",
            )
        ]

    # ------------------------------------------------------------------
    # 5. max_tokens_edge_1
    # ------------------------------------------------------------------

    def _test_max_tokens_edge_1(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """max_tokens=1 -- should return at least 1 token with finish_reason='length'."""
        test_name = "max_tokens_edge_1"
        resp = v.chat(
            [
                {
                    "role": "user",
                    "content": "Tell me a long story about dragons.",
                }
            ],
            max_tokens=1,
        )
        fr = resp.choices[0].finish_reason
        if fr == "length":
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.PASS,
                    detail="finish_reason='length' as expected with max_tokens=1",
                )
            ]
        if fr == "stop":
            # Some models might complete in 1 token (unlikely but possible)
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.INTERESTING,
                    detail="finish_reason='stop' with max_tokens=1 (model completed in 1 token?)",
                )
            ]
        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.FAIL,
                detail=f"Expected finish_reason='length', got {fr!r}",
            )
        ]

    # ------------------------------------------------------------------
    # 6. tools_empty_array
    # ------------------------------------------------------------------

    def _test_tools_empty_array(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """tools=[] should work as if no tools (not crash)."""
        test_name = "tools_empty_array"
        try:
            resp = v.chat(
                [{"role": "user", "content": "Say hello."}],
                max_tokens=128,
                tools=[],
            )
            content = resp.choices[0].message.content or ""
            reasoning = (
                getattr(resp.choices[0].message, "reasoning_content", None)
                or getattr(resp.choices[0].message, "reasoning", None)
                or ""
            )
            if content or reasoning:
                return [
                    self.make_result(
                        self.name,
                        test_name,
                        Verdict.PASS,
                        detail=f"Empty tools array handled correctly ({len(content)} chars)",
                    )
                ]
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.INTERESTING,
                    detail="Empty response with tools=[]",
                )
            ]
        except Exception as e:
            err_str = str(e).lower()
            # 400/422 is acceptable -- server rejected empty tools gracefully
            if "400" in err_str or "422" in err_str or "bad request" in err_str:
                return [
                    self.make_result(
                        self.name,
                        test_name,
                        Verdict.PASS,
                        detail=f"Server rejected empty tools array gracefully: {e}",
                    )
                ]
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"Server crashed on tools=[]: {e}",
                )
            ]

    # ------------------------------------------------------------------
    # 7. mixed_so_tc_conversation
    # ------------------------------------------------------------------

    def _test_mixed_so_tc_conversation(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """SO -> TC -> plain in same conversation context -- mode switching."""
        test_name = "mixed_so_tc_conversation"
        errors: list[str] = []

        # Step 1: Structured output request
        schema = {
            "type": "object",
            "properties": {"topic": {"type": "string"}},
            "required": ["topic"],
            "additionalProperties": False,
        }
        try:
            so_resp = v.so_chat(
                [{"role": "user", "content": "Pick a science topic."}],
                schema,
                max_tokens=256,
            )
            if not budget_exhausted(so_resp):
                content = so_resp.choices[0].message.content or ""
                try:
                    data = json.loads(content)
                    if "topic" not in data:
                        errors.append("SO response missing 'topic' key")
                except (json.JSONDecodeError, TypeError) as e:
                    errors.append(f"SO response invalid JSON: {e}")
        except Exception as e:
            errors.append(f"SO request failed: {e}")

        # Step 2: Tool calling request
        try:
            tc_resp = v.tc_chat(
                [{"role": "user", "content": "Search for quantum physics."}],
                [_SEARCH_TOOL],
                tool_choice="required",
                max_tokens=512,
            )
            if not budget_exhausted(tc_resp):
                tcs = tc_resp.choices[0].message.tool_calls
                if not tcs:
                    errors.append("TC request produced no tool calls")
        except Exception as e:
            errors.append(f"TC request failed: {e}")

        # Step 3: Plain chat request
        try:
            plain_resp = v.chat(
                [{"role": "user", "content": "Just say hello."}],
                max_tokens=128,
            )
            content = plain_resp.choices[0].message.content or ""
            reasoning = (
                getattr(
                    plain_resp.choices[0].message, "reasoning_content", None
                )
                or getattr(plain_resp.choices[0].message, "reasoning", None)
                or ""
            )
            if not content and not reasoning:
                errors.append("Plain chat produced empty response")
        except Exception as e:
            errors.append(f"Plain chat failed: {e}")

        if errors:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"Mode switching errors: {'; '.join(errors)}",
                )
            ]
        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.PASS,
                detail="SO -> TC -> plain mode switching completed successfully",
            )
        ]

    # ------------------------------------------------------------------
    # 8. streaming_stability_100_chunks
    # ------------------------------------------------------------------

    def _test_streaming_stability_100_chunks(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """Request that generates many chunks -- verify all parse correctly."""
        test_name = "streaming_stability_100_chunks"

        stream = v.chat(
            [
                {
                    "role": "user",
                    "content": (
                        "Write a detailed list of 50 interesting facts about space, "
                        "one per line, numbered 1 through 50."
                    ),
                }
            ],
            max_tokens=4096,
            stream=True,
        )
        result = collect_stream(stream)

        chunk_count = result["chunk_count"]
        content = result["content"]
        fr = result["finish_reason"]

        if chunk_count == 0:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail="No chunks received from streaming response",
                )
            ]

        if fr not in ("stop", "length"):
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"Unexpected finish_reason={fr!r} after {chunk_count} chunks",
                )
            ]

        # Verify we got substantial output
        if chunk_count < 10:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.INTERESTING,
                    detail=f"Only {chunk_count} chunks (expected many more for long response)",
                )
            ]

        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.PASS,
                detail=f"{chunk_count} chunks parsed, {len(content)} chars, finish_reason={fr!r}",
            )
        ]

    # ------------------------------------------------------------------
    # 9. model_variant_check
    # ------------------------------------------------------------------

    def _test_model_variant_check(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """Verify models.list() returns at least one model ID."""
        test_name = "model_variant_check"
        try:
            detected = v.detect_model()
            if detected is None:
                return [
                    self.make_result(
                        self.name,
                        test_name,
                        Verdict.FAIL,
                        detail="models.list() returned no models",
                    )
                ]
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.PASS,
                    detail=f"Model detected: {detected!r}",
                )
            ]
        except Exception as e:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"models.list() failed: {e}",
                )
            ]

    # ------------------------------------------------------------------
    # 10. backwards_compat_null_params
    # ------------------------------------------------------------------

    def _test_backwards_compat_null_params(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """Tool with parameters: null -- should not crash."""
        test_name = "backwards_compat_null_params"
        # Build a tool definition with null parameters (some older clients send this)
        tool_with_null_params = {
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "Get the current time",
                "parameters": None,
            },
        }
        try:
            resp = v.chat(
                [
                    {
                        "role": "user",
                        "content": "What time is it? Use the get_time tool.",
                    }
                ],
                max_tokens=512,
                tools=[tool_with_null_params],
                tool_choice="auto",
            )
            msg = resp.choices[0].message
            content = msg.content or ""
            has_tc = bool(msg.tool_calls)
            reasoning = (
                getattr(msg, "reasoning_content", None)
                or getattr(msg, "reasoning", None)
                or ""
            )
            if content or has_tc or reasoning:
                return [
                    self.make_result(
                        self.name,
                        test_name,
                        Verdict.PASS,
                        detail=f"Null parameters handled (content={len(content)} chars, tool_calls={has_tc})",
                    )
                ]
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.INTERESTING,
                    detail="Empty response with null-parameters tool",
                )
            ]
        except Exception as e:
            err_str = str(e).lower()
            # 400/422 is acceptable -- server gracefully rejected null params
            if "400" in err_str or "422" in err_str or "bad request" in err_str:
                return [
                    self.make_result(
                        self.name,
                        test_name,
                        Verdict.PASS,
                        detail=f"Server rejected null parameters gracefully: {e}",
                    )
                ]
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"Server crashed on null parameters: {e}",
                )
            ]
