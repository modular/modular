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
Scenario: multi-turn needle-in-a-haystack recall over a reused KV prefix.

Several conversations run side by side. Each buries a high-entropy code in
a unique multi-page haystack, asks for it cold, then re-asks it on every
follow-up turn, where the server should serve the whole earlier prompt from
the prefix cache. Most haystacks are a few pages; a couple are sized to half
the model's window so production-length prompts co-batch with short ones.
Three checks:

1. ``prefix_cache_hits``: every follow-up turn reports at least the previous
   prompt's full KV pages as ``usage.prompt_tokens_details.cached_tokens``.
   No reuse anywhere is a FAIL; a shortfall on some turns is INTERESTING,
   since a multi-replica deployment can legitimately miss, unless
   ``--needle-strict-cache-hits`` says the deployment is expected to serve
   every reload (a KV connector behind a deliberately small GPU pool).
2. ``needle_recall``: a cache-served turn that loses the needle is replayed
   fresh after a prefix-cache reset. Recall on the replay means reusing the
   prefix changed the answer (FAIL); a miss on the replay too is the model.
3. ``cross_talk``: no response carries another conversation's needle (FAIL).

Sized by ``--needle-conversations``, ``--needle-long-conversations``,
``--needle-turns`` and ``--needle-context-tokens``; the defaults finish in
minutes on any model.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario
from scenarios._needle import (
    RECALL_QUESTION,
    CacheHit,
    TurnReply,
    build_haystack,
    classify_cache_hit,
    expected_cached_tokens,
    find_leaked_needles,
    first_turn_content,
    generate_needle,
    haystack_token_target,
    long_haystack_token_target,
    needle_recalled,
    parse_reply,
)

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig

_SEED = 7331
# Large enough for a reasoning model to think before it echoes the code; a
# budget spent entirely on reasoning is reported, never scored as a miss.
_MAX_TOKENS = 512
# A half-window prefill co-batched with several others on a slow model
# outlasts the tool's interactive default timeout by a wide margin.
_REQUEST_TIMEOUT = 300.0
# MAX's default KV page size. A deployment with larger pages reports a
# smaller cached count on the same prompt and lands in PARTIAL, not FAIL.
_KV_PAGE_TOKENS = 128
_SNIPPET = 160


@dataclass
class _Conversation:
    index: int
    needle: str
    messages: list[dict[str, str]]
    replies: list[TurnReply] = field(default_factory=list)
    """One entry per turn sent, cold turn first."""
    prompts: list[list[dict[str, str]]] = field(default_factory=list)
    """The message list each turn was sent with, for fresh replays."""
    alive: bool = True

    def hit(self, turn: int) -> CacheHit:
        prev = self.replies[turn - 1].prompt_tokens
        return classify_cache_hit(self.replies[turn], prev, _KV_PAGE_TOKENS)


def _payload(model: str, messages: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "seed": _SEED,
        "max_tokens": _MAX_TOKENS,
    }


def _haystack_sizes(
    rng: random.Random, config: RunConfig, target: int, long_target: int
) -> list[int]:
    """One haystack size per conversation, short ones first.

    Short sizes spread over a 2x range so the needle straddles different page
    boundaries across conversations; the long ones sit at their target.
    """
    short = [
        rng.randint(max(1, target // 2), target)
        for _ in range(config.needle_conversations)
    ]
    return short + [long_target] * config.needle_long_conversations


def _build_conversations(
    rng: random.Random, sizes: list[int]
) -> list[_Conversation]:
    convs = []
    for index, size in enumerate(sizes):
        needle = generate_needle(rng)
        haystack = build_haystack(rng, needle, size, rng.random())
        convs.append(
            _Conversation(
                index,
                needle,
                [{"role": "user", "content": first_turn_content(haystack)}],
            )
        )
    return convs


def _snippet(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _SNIPPET else text[:_SNIPPET] + "..."


@dataclass
class _RecallTally:
    """Recall outcomes sorted by what can be blamed for a miss."""

    cold_scored: int = 0
    cold_missed: int = 0
    served: int = 0
    """Follow-up turns that reused some prefix."""
    empty: int = 0
    """Completions with no visible answer, usually a reasoning budget."""
    fresh_misses: list[str] = field(default_factory=list)
    """Misses on follow-up turns that reused nothing: the model's own."""
    suspects: list[tuple[_Conversation, int]] = field(default_factory=list)
    """Misses on turns that reused a prefix, or whose reuse is unknown."""
    model_misses: list[str] = field(default_factory=list)
    """Suspects a fresh replay also missed."""

    def caveats(self) -> list[str]:
        parts = []
        if self.cold_missed:
            parts.append(
                f"{self.cold_missed}/{self.cold_scored} cold turns missed"
            )
        if self.model_misses:
            parts.append(
                f"{len(self.model_misses)} cache-served miss(es) also missed "
                f"on a fresh replay ({self.model_misses[0]})"
            )
        if self.fresh_misses:
            parts.append(
                f"{len(self.fresh_misses)} miss(es) on turns served without "
                f"any cached prefix ({self.fresh_misses[0]})"
            )
        if self.empty:
            parts.append(
                f"{self.empty} empty completion(s), budget spent on reasoning"
            )
        return parts


def _tally_recall(convs: list[_Conversation]) -> _RecallTally:
    tally = _RecallTally()
    for conv in convs:
        for turn, reply in enumerate(conv.replies):
            if not reply.ok:
                continue
            if not reply.has_answer:
                tally.empty += 1
                continue
            recalled = needle_recalled(reply.visible, conv.needle)
            if turn == 0:
                tally.cold_scored += 1
                tally.cold_missed += not recalled
                continue
            kind = conv.hit(turn)
            if kind in (CacheHit.FULL, CacheHit.PARTIAL):
                tally.served += 1
            if recalled:
                continue
            if kind is CacheHit.MISS:
                tally.fresh_misses.append(f"conv {conv.index} turn {turn}")
            else:
                tally.suspects.append((conv, turn))
    return tally


def _attribute_miss(
    conv: _Conversation, turn: int, control: TurnReply
) -> str | None:
    """Explains a cache-served miss when its fresh replay recalled the needle.

    None when the replay missed too, which makes the miss the model's.
    """
    if not (control.ok and control.has_answer):
        return None
    if not needle_recalled(control.visible, conv.needle):
        return None
    reply = conv.replies[turn]
    return (
        f"conv {conv.index} turn {turn} (cached {reply.cached_tokens} of "
        f"{reply.prompt_tokens} prompt tokens): cache-served answer "
        f"{_snippet(reply.visible)!r} vs fresh replay "
        f"{_snippet(control.visible)!r} (replay cached {control.cached_tokens})"
    )


@register_scenario
class NeedlePrefixCacheScenario(BaseScenario):
    name = "needle_prefix_cache"
    description = (
        "Multi-turn needle recall over a reused KV prefix: cached-token "
        "accounting, recall on cache-served turns, cross-conversation leaks"
    )
    tags = ["kv_cache", "prefix_cache", "correctness", "concurrency"]
    scenario_type = "validation"

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        window = config.model_config.max_position_embeddings
        target = config.needle_context_tokens or haystack_token_target(window)
        long_target = long_haystack_token_target(window)
        rng = random.Random(_SEED)
        convs = _build_conversations(
            rng, _haystack_sizes(rng, config, target, long_target)
        )
        sizing = f"haystack target ~{target} tokens"
        if config.needle_long_conversations:
            sizing += (
                f" plus {config.needle_long_conversations} long at "
                f"~{long_target}"
            )
        timeout = max(config.timeout, _REQUEST_TIMEOUT)

        await self._reset_cache(client)
        for turn in range(config.needle_turns + 1):
            active = [c for c in convs if c.alive]
            if not active:
                break
            await self._advance(client, config, active, turn, timeout)

        results = [
            self._cache_hits(
                convs, sizing, strict=config.needle_strict_cache_hits
            ),
            await self._recall(client, config, convs, timeout),
            self._cross_talk(convs),
        ]
        failed = self._request_failures(convs)
        if failed is not None:
            results.insert(0, failed)
        return results

    @staticmethod
    async def _reset_cache(client: FuzzClient) -> None:
        """Best-effort ``POST /reset_prefix_cache``, as ``fuzz.py`` does between
        repeats; a short settle lets the enqueued reset apply."""
        try:
            await client.post_to_path("/reset_prefix_cache", {})
            await asyncio.sleep(0.1)
        except Exception:
            pass

    @staticmethod
    async def _advance(
        client: FuzzClient,
        config: RunConfig,
        active: list[_Conversation],
        turn: int,
        timeout: float,
    ) -> None:
        """Sends turn ``turn`` of every live conversation as one wave."""
        for conv in active:
            if turn > 0:
                conv.messages.append(
                    {"role": "user", "content": RECALL_QUESTION}
                )
            conv.prompts.append(list(conv.messages))
        responses = await client.concurrent_requests(
            [_payload(config.model, c.messages) for c in active],
            timeout=timeout,
        )
        for conv, resp in zip(active, responses, strict=True):
            reply = parse_reply(resp.status, resp.body, resp.error)
            conv.replies.append(reply)
            if reply.ok and reply.has_answer:
                conv.messages.append(
                    {"role": "assistant", "content": reply.content}
                )
            else:
                # Nothing to continue the history from.
                conv.alive = False

    def _request_failures(
        self, convs: list[_Conversation]
    ) -> ScenarioResult | None:
        bad = [
            (c.index, turn, r)
            for c in convs
            for turn, r in enumerate(c.replies)
            if not r.ok
        ]
        if not bad:
            return None
        # A 5xx or an unparseable 200 is the server's; a 4xx on a prompt this
        # size is worth a look but is a graceful refusal.
        if any(r.status == 0 for _, _, r in bad):
            verdict = Verdict.ERROR
        elif any(r.status >= 500 or r.status == 200 for _, _, r in bad):
            verdict = Verdict.FAIL
        else:
            verdict = Verdict.INTERESTING
        index, turn, first = bad[0]
        return self.make_result(
            self.name,
            "requests",
            verdict,
            status_code=first.status,
            detail=(
                f"{len(bad)} turn request(s) did not return a usable 200; "
                f"first: conv {index} turn {turn}: {first.error}"
            ),
        )

    def _cache_hits(
        self, convs: list[_Conversation], sizing: str, *, strict: bool
    ) -> ScenarioResult:
        hits = dict.fromkeys(CacheHit, 0)
        shortfalls: list[str] = []
        fractions: list[float] = []
        for conv in convs:
            for turn in range(1, len(conv.replies)):
                reply = conv.replies[turn]
                if not reply.ok:
                    continue
                kind = conv.hit(turn)
                hits[kind] += 1
                if kind is CacheHit.UNKNOWN:
                    continue
                assert reply.cached_tokens is not None
                assert reply.prompt_tokens is not None
                fractions.append(reply.cached_tokens / reply.prompt_tokens)
                if kind is not CacheHit.FULL:
                    prev = conv.replies[turn - 1].prompt_tokens
                    assert prev is not None
                    shortfalls.append(
                        f"conv {conv.index} turn {turn}: cached "
                        f"{reply.cached_tokens} of {reply.prompt_tokens} "
                        f"prompt tokens, expected >= "
                        f"{expected_cached_tokens(prev, _KV_PAGE_TOKENS)}"
                    )

        cold = [
            c.replies[0].prompt_tokens
            for c in convs
            if c.replies and c.replies[0].prompt_tokens
        ]
        if cold:
            sizing += f", cold prompts {min(cold)}-{max(cold)} tokens"
        scored = sum(hits.values())
        counted = scored - hits[CacheHit.UNKNOWN]
        if scored == 0:
            return self.make_result(
                self.name,
                "prefix_cache_hits",
                Verdict.INTERESTING,
                detail=f"no follow-up turn completed; {sizing}",
            )
        if counted == 0:
            return self.make_result(
                self.name,
                "prefix_cache_hits",
                Verdict.INTERESTING,
                detail=(
                    "no response reported usable usage.prompt_tokens and "
                    "usage.prompt_tokens_details.cached_tokens, so prefix "
                    f"reuse cannot be checked; {sizing}"
                ),
            )
        reused = hits[CacheHit.FULL] + hits[CacheHit.PARTIAL]
        if reused == 0:
            return self.make_result(
                self.name,
                "prefix_cache_hits",
                Verdict.FAIL,
                detail=(
                    f"none of {counted} follow-up turns reused any prefix "
                    "(cached_tokens=0 on every one): prefix caching is "
                    f"disabled or not finding the previous turn; {sizing}"
                ),
            )
        summary = (
            f"{hits[CacheHit.FULL]}/{counted} follow-up turns reused the "
            f"previous prompt in full (cached/prompt "
            f"{min(fractions):.0%}-{max(fractions):.0%}); {sizing}"
        )
        if shortfalls:
            shortfall = (
                f"{summary}; {hits[CacheHit.PARTIAL]} partial, "
                f"{hits[CacheHit.MISS]} miss, e.g. {shortfalls[0]}"
            )
            if strict:
                return self.make_result(
                    self.name,
                    "prefix_cache_hits",
                    Verdict.FAIL,
                    detail=(
                        f"{shortfall} [strict: this deployment is expected "
                        "to serve every reload, so a miss is a reload the "
                        "cache tiers did not deliver]"
                    ),
                )
            return self.make_result(
                self.name,
                "prefix_cache_hits",
                Verdict.INTERESTING,
                detail=(
                    f"{shortfall} [a replica without the prefix or an "
                    "eviction explains a miss; a partial hit needs a look]"
                ),
            )
        return self.make_result(
            self.name, "prefix_cache_hits", Verdict.PASS, detail=summary
        )

    async def _recall(
        self,
        client: FuzzClient,
        config: RunConfig,
        convs: list[_Conversation],
        timeout: float,
    ) -> ScenarioResult:
        tally = _tally_recall(convs)
        confirmed: list[str] = []
        bodies: list[str] = []
        for conv, turn in tally.suspects:
            control = await self._replay_fresh(
                client, config, conv.prompts[turn], timeout
            )
            explanation = _attribute_miss(conv, turn, control)
            if explanation is None:
                tally.model_misses.append(
                    f"conv {conv.index} turn {turn}: fresh replay missed too"
                )
            else:
                confirmed.append(explanation)
                bodies.append(
                    f"needle {conv.needle}; served: {conv.replies[turn].visible}"
                )

        if confirmed:
            return self.make_result(
                self.name,
                "needle_recall",
                Verdict.FAIL,
                detail=(
                    f"{len(confirmed)} cache-served turn(s) lost the needle "
                    "that a fresh replay of the same prompt recalled: reusing "
                    f"the prefix changed the answer. {confirmed[0]}"
                ),
                response_body="\n".join(bodies[:3]),
            )
        if tally.cold_scored == 0:
            return self.make_result(
                self.name,
                "needle_recall",
                Verdict.INTERESTING,
                detail=(
                    f"no cold turn produced an answer to score ({tally.empty} "
                    "empty completions); the model spends the whole budget "
                    "reasoning or cannot do the task"
                ),
            )
        summary = (
            f"{tally.cold_scored} conversations, {tally.served} cache-served "
            "recall turns, none lost the needle"
        )
        caveats = tally.caveats()
        if caveats:
            return self.make_result(
                self.name,
                "needle_recall",
                Verdict.INTERESTING,
                detail=f"{summary}; not the cache: " + "; ".join(caveats),
            )
        return self.make_result(
            self.name, "needle_recall", Verdict.PASS, detail=summary
        )

    async def _replay_fresh(
        self,
        client: FuzzClient,
        config: RunConfig,
        messages: list[dict[str, str]],
        timeout: float,
    ) -> TurnReply:
        await self._reset_cache(client)
        resp = await client.post_json(
            _payload(config.model, messages), timeout=timeout
        )
        return parse_reply(resp.status, resp.body, resp.error)

    def _cross_talk(self, convs: list[_Conversation]) -> ScenarioResult:
        pool = [c.needle for c in convs]
        leaks = []
        checked = 0
        for conv in convs:
            for turn, reply in enumerate(conv.replies):
                if not reply.ok:
                    continue
                checked += 1
                leaked = find_leaked_needles(
                    f"{reply.visible}\n{reply.reasoning}", conv.needle, pool
                )
                if leaked:
                    leaks.append(f"conv {conv.index} turn {turn}: {leaked}")
        if leaks:
            return self.make_result(
                self.name,
                "cross_talk",
                Verdict.FAIL,
                detail=(
                    f"{len(leaks)} response(s) carried another conversation's "
                    f"needle, e.g. {leaks[0]}: KV or request mixing across "
                    "sequences"
                ),
            )
        return self.make_result(
            self.name,
            "cross_talk",
            Verdict.PASS,
            detail=f"{checked} responses, no foreign needle",
        )
