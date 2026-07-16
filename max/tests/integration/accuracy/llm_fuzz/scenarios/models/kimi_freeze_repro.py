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
"""Kimi K2.5 freeze reproduction.

Targets the production hang observed on the Kimi K2.5 release
container on 2026-05-23 where all four shadow MAX nodes wedged after
spewing repeated ``Failed to convert tools to TypeScript style:
Invalid JSON Schema object`` warnings. The offending schema was a
``oneOf`` whose first branch was a bare ``{"const": "end"}`` literal
-- a construct Kimi's bundled HF tokenizer
(``tool_declaration_ts.py``) does not recognise.

PR #87007 added ``_sanitize_kimi_tool_schemas`` at the chat-template
boundary so the tokenizer never sees the rejected constructs. This
scenario exercises the same shape under concurrent load plus the
adjacent failure mode from the production logs (an OpenAI custom-tool
payload that fails request validation) to confirm the engine continues
to make forward progress instead of wedging.

The s07 ``tool_schema_with_oneof_and_const`` test only covers the
single-request happy path -- it would have passed even while the
production engine was hanging, because the hang only appeared under
sustained traffic. Pair this scenario with ``--enable-structured-output``
so the bitmask-path code in ``overlap_text_generation`` is wired in.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario
from scenarios._kimi_fixtures import PRODUCTION_ONEOF_TOOL

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig


# Payload shape from the production validation error: an OpenAI
# "custom tool" (``type=custom``, ``format={type: grammar, ...}``)
# sent against a deployment that only speaks the legacy function-tool
# schema. The server should reject this with 4xx and stay healthy.
CUSTOM_GRAMMAR_TOOL: dict[str, Any] = {
    "type": "custom",
    "name": "apply_patch",
    "description": (
        "Use the apply_patch tool to apply a patch. Wrap the patch in JSON."
    ),
    "format": {
        "type": "grammar",
        "definition": (
            'start: patch\npatch: "BEGIN" CONTENT "END"\nCONTENT: /[^E]+/\n'
        ),
        "syntax": "lark",
    },
}


# Kimi K2.5 emits an explicit ``<|tool_calls_section_end|>`` marker
# at the end of any tool-call response. The Kimi tool parser only
# populates ``message.tool_calls`` when it sees the closing marker;
# an in-progress section that gets cut off by ``max_tokens`` leaves
# the raw begin/end markers in ``content`` and an empty
# ``tool_calls``. Budget enough headroom so the closing marker always
# fits, plus space for any reasoning preamble the model emits before
# the first tool call.
_TOOL_CALL_MAX_TOKENS = 512


def _basic_payload(model: str, idx: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    f"[req {idx}] Place 'apple-{idx}' at the end of the list."
                ),
            }
        ],
        "tools": [PRODUCTION_ONEOF_TOOL],
        "tool_choice": "required",
        "max_tokens": _TOOL_CALL_MAX_TOKENS,
    }


@register_scenario
class KimiFreezeRepro(BaseScenario):
    name = "kimi_freeze_repro"
    description = (
        "Kimi K2.5 freeze reproduction: production oneOf/const tool "
        "schemas under concurrent load, mixed with the custom-grammar "
        "tool payload that triggered the adjacent validation error in "
        "the production logs"
    )
    tags = ["model:kimi-k2.5", "tool_calling", "hang", "freeze"]
    model_filter = "kimi-k2.5"

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []
        model = config.model

        results.append(await self._baseline(client, model))
        results.append(await self._concurrent_load(client, model))
        results.append(await self._custom_grammar_validation(client, model))
        results.append(await self._mixed_valid_invalid(client, model))
        results.append(await self._structured_output_plus_oneof(client, model))
        results.append(await self._post_load_health(client))

        return results

    # ------------------------------------------------------------------
    # 1. Single-request baseline -- if this fails, the sanitizer is gone
    # ------------------------------------------------------------------
    async def _baseline(self, client: FuzzClient, model: str) -> ScenarioResult:
        resp = await client.post_json(_basic_payload(model, 0))
        if resp.status >= 500 or resp.error == "TIMEOUT":
            return self.make_result(
                self.name,
                "oneof_const_baseline",
                Verdict.FAIL,
                status_code=resp.status,
                detail=(
                    f"server error: status={resp.status} error={resp.error!r}"
                ),
            )
        if resp.status >= 400:
            # If the validator (the sanitizer's caller) rejects the
            # schema outright, surface that -- it is not a crash but
            # it would still wedge tool calling in production.
            return self.make_result(
                self.name,
                "oneof_const_baseline",
                Verdict.INTERESTING,
                status_code=resp.status,
                detail=f"client error rejecting valid schema: {resp.body[:200]!r}",
            )
        try:
            body = json.loads(resp.body)
            tcs = (
                body.get("choices", [{}])[0]
                .get("message", {})
                .get("tool_calls")
                or []
            )
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            return self.make_result(
                self.name,
                "oneof_const_baseline",
                Verdict.FAIL,
                status_code=resp.status,
                detail=f"unparseable response: {e}",
            )
        if not tcs:
            try:
                msg = body.get("choices", [{}])[0].get("message", {}) or {}
                content_preview = (msg.get("content") or "")[:200]
                finish_reason = body.get("choices", [{}])[0].get(
                    "finish_reason"
                )
            except Exception:
                content_preview = ""
                finish_reason = None
            return self.make_result(
                self.name,
                "oneof_const_baseline",
                Verdict.FAIL,
                status_code=resp.status,
                detail=(
                    "tool_choice='required' produced no tool_calls. "
                    "Either the Kimi HF tokenizer dropped the tool "
                    "declarations (sanitizer regression) or the "
                    "constrained-decoding path is not enforcing the "
                    "tool grammar. "
                    f"finish_reason={finish_reason!r} "
                    f"content[:200]={content_preview!r}"
                ),
            )
        return self.make_result(
            self.name,
            "oneof_const_baseline",
            Verdict.PASS,
            status_code=resp.status,
        )

    # ------------------------------------------------------------------
    # 2. Concurrent load -- the actual freeze trigger
    # ------------------------------------------------------------------
    async def _concurrent_load(
        self, client: FuzzClient, model: str
    ) -> ScenarioResult:
        n_requests = 24
        max_concurrent = 8
        payloads = [_basic_payload(model, i) for i in range(n_requests)]
        responses = await client.concurrent_requests(
            payloads, max_concurrent=max_concurrent
        )
        timeouts = sum(1 for r in responses if r.error == "TIMEOUT")
        server_errors = sum(1 for r in responses if r.status >= 500)
        client_errors = sum(1 for r in responses if 400 <= r.status < 500)
        ok = sum(1 for r in responses if r.status == 200)
        # Any timeout is the freeze signature -- the engine stops
        # responding mid-batch. Any 5xx is a crash. Both fail loudly.
        if timeouts > 0 or server_errors > 0:
            return self.make_result(
                self.name,
                "oneof_const_concurrent_24",
                Verdict.FAIL,
                detail=(
                    f"{ok}/{n_requests} ok, {server_errors} 5xx, "
                    f"{timeouts} timeouts -- freeze/crash under load"
                ),
            )
        if client_errors > 0:
            # A structurally valid request that comes back 4xx (e.g.
            # 422 because a future sanitizer regression starts
            # rejecting the schema rather than silently dropping tool
            # declarations) isn't a crash, but it isn't the contract
            # this scenario is meant to verify either. Surface so a
            # regression doesn't slip through as PASS.
            return self.make_result(
                self.name,
                "oneof_const_concurrent_24",
                Verdict.INTERESTING,
                detail=(
                    f"{ok}/{n_requests} ok, {client_errors} 4xx -- "
                    "valid oneOf/const tool requests rejected by the "
                    "server"
                ),
            )
        return self.make_result(
            self.name,
            "oneof_const_concurrent_24",
            Verdict.PASS,
            detail=f"{ok}/{n_requests} ok",
        )

    # ------------------------------------------------------------------
    # 3. Custom-grammar tool format -- adjacent production failure
    # ------------------------------------------------------------------
    async def _custom_grammar_validation(
        self, client: FuzzClient, model: str
    ) -> ScenarioResult:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [CUSTOM_GRAMMAR_TOOL],
            "max_tokens": 32,
        }
        resp = await client.post_json(payload)
        # The server must reject the payload (we never advertised custom
        # tool support) without crashing. The interesting failure mode is
        # status 200 (silent acceptance of unsupported feature) or 5xx.
        if resp.status >= 500 or resp.error == "TIMEOUT":
            return self.make_result(
                self.name,
                "custom_grammar_tool_validation",
                Verdict.FAIL,
                status_code=resp.status,
                detail=(
                    f"server failed on validation error: status={resp.status} "
                    f"error={resp.error!r}"
                ),
            )
        if 400 <= resp.status < 500:
            return self.make_result(
                self.name,
                "custom_grammar_tool_validation",
                Verdict.PASS,
                status_code=resp.status,
            )
        return self.make_result(
            self.name,
            "custom_grammar_tool_validation",
            Verdict.INTERESTING,
            status_code=resp.status,
            detail=(
                "server accepted an unsupported custom-tool payload "
                "(expected 4xx)"
            ),
        )

    # ------------------------------------------------------------------
    # 4. Mixed valid + invalid -- the production sequence
    # ------------------------------------------------------------------
    async def _mixed_valid_invalid(
        self, client: FuzzClient, model: str
    ) -> ScenarioResult:
        # 18 valid oneOf/const requests interleaved with 6 invalid
        # custom-grammar payloads -- approximates the production
        # sequence where a single validation error preceded the cascade
        # of TypeScript warnings and the eventual freeze.
        payloads: list[dict[str, Any]] = []
        for i in range(24):
            if i % 4 == 0:
                payloads.append(
                    {
                        "model": model,
                        "messages": [
                            {"role": "user", "content": f"[req {i}] bad"}
                        ],
                        "tools": [CUSTOM_GRAMMAR_TOOL],
                        "max_tokens": 16,
                    }
                )
            else:
                payloads.append(_basic_payload(model, i))
        responses = await client.concurrent_requests(payloads, max_concurrent=8)
        timeouts = sum(1 for r in responses if r.error == "TIMEOUT")
        server_errors = sum(1 for r in responses if r.status >= 500)
        # Count successful tool calls among the valid-shaped requests
        # (every index not divisible by 4), and count 4xx rejections
        # among the invalid-shaped requests (every index divisible by
        # 4). ``_custom_grammar_validation`` flags silent acceptance
        # of the unsupported ``type=custom`` payload as INTERESTING;
        # mirror that contract here so the two tests agree on what
        # "correct" looks like.
        valid_ok = 0
        invalid_4xx = 0
        for i, r in enumerate(responses):
            if i % 4 == 0:
                if 400 <= r.status < 500:
                    invalid_4xx += 1
            elif r.status == 200:
                valid_ok += 1
        n_valid = sum(1 for i in range(24) if i % 4 != 0)
        n_invalid = 24 - n_valid
        if timeouts > 0 or server_errors > 0:
            return self.make_result(
                self.name,
                "mixed_valid_and_invalid",
                Verdict.FAIL,
                detail=(
                    f"{valid_ok}/{n_valid} valid ok, "
                    f"{server_errors} 5xx, {timeouts} timeouts -- "
                    "engine wedged when validation errors interleave "
                    "with valid traffic"
                ),
            )
        if valid_ok < n_valid or invalid_4xx < n_invalid:
            return self.make_result(
                self.name,
                "mixed_valid_and_invalid",
                Verdict.INTERESTING,
                detail=(
                    f"{valid_ok}/{n_valid} valid ok, "
                    f"{invalid_4xx}/{n_invalid} invalid 4xx -- "
                    "engine survived but either valid requests "
                    "failed or invalid requests were silently "
                    "accepted"
                ),
            )
        return self.make_result(
            self.name,
            "mixed_valid_and_invalid",
            Verdict.PASS,
            detail=(
                f"{valid_ok}/{n_valid} valid ok, "
                f"{invalid_4xx}/{n_invalid} invalid 4xx"
            ),
        )

    # ------------------------------------------------------------------
    # 5. Structured output + oneOf/const tool schemas
    # ------------------------------------------------------------------
    async def _structured_output_plus_oneof(
        self, client: FuzzClient, model: str
    ) -> ScenarioResult:
        # Production used --enable-structured-output. The oneOf in
        # ``PRODUCTION_ONEOF_TOOL`` is itself enough to drive the FSM /
        # bitmask path -- the tool-call grammar derived from the schema
        # forces grammar-constrained decoding regardless of
        # ``response_format``. Concurrently issuing 6 of them exercises
        # the bitmask path in ``overlap_text_generation`` alongside the
        # tokenizer's tool-declaration pipeline; hang regressions can
        # sit in either layer.
        #
        # The 6 payloads are salted with a per-request ``apple-{i}``
        # so each one tokenises and prefills distinctly -- without the
        # salt, only the first request runs the FSM/bitmask path
        # end-to-end; the other 5 hit prefix cache and exercise
        # essentially nothing.
        def _schema_payload(idx: int) -> dict[str, Any]:
            item = f"apple-{idx}"
            return {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": (f"Place '{item}' at the end of the list."),
                    }
                ],
                "tools": [PRODUCTION_ONEOF_TOOL],
                "tool_choice": "required",
                "max_tokens": _TOOL_CALL_MAX_TOKENS,
            }

        # Issue 6 of these concurrently; the structured-output path
        # serialises through the FSM and is where the dogfood logs
        # showed the assert-level escape.
        payloads = [_schema_payload(i) for i in range(6)]
        responses = await client.concurrent_requests(payloads, max_concurrent=6)
        timeouts = sum(1 for r in responses if r.error == "TIMEOUT")
        server_errors = sum(1 for r in responses if r.status >= 500)
        if timeouts > 0 or server_errors > 0:
            return self.make_result(
                self.name,
                "structured_output_plus_oneof_const",
                Verdict.FAIL,
                detail=(
                    f"{server_errors} 5xx, {timeouts} timeouts -- "
                    "FSM/bitmask path wedged with oneOf/const tools "
                    "in scope"
                ),
            )
        return self.make_result(
            self.name,
            "structured_output_plus_oneof_const",
            Verdict.PASS,
        )

    # ------------------------------------------------------------------
    # 6. Post-load health probe -- proves the engine is still alive
    # ------------------------------------------------------------------
    async def _post_load_health(self, client: FuzzClient) -> ScenarioResult:
        # If the engine is wedged but the HTTP layer still answers
        # /v1/chat/completions for trivial prompts, the request will
        # time out (no batch ever runs). The effective timeout is
        # ``client.config.timeout`` (set by ``--timeout``, default
        # 30s); ``client.health_check`` catches the inner asyncio
        # timeout and surfaces it as ``RawResponse(status=0,
        # error='TIMEOUT')`` rather than raising. Mirror the
        # ``error == 'TIMEOUT'`` check used by the five other
        # health-check call sites in this directory rather than
        # nesting a second ``wait_for`` cap that only takes effect
        # when ``--timeout`` is bumped past it.
        resp = await client.health_check()
        if resp.error == "TIMEOUT":
            return self.make_result(
                self.name,
                "post_load_health",
                Verdict.FAIL,
                detail=("health probe timed out -- engine wedged post-load"),
            )
        if resp.status != 200:
            return self.make_result(
                self.name,
                "post_load_health",
                Verdict.FAIL,
                status_code=resp.status,
                detail=(
                    f"health probe failed: status={resp.status} "
                    f"error={resp.error!r}"
                ),
            )
        return self.make_result(
            self.name,
            "post_load_health",
            Verdict.PASS,
            status_code=resp.status,
        )
