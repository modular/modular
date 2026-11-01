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
"""Pure logic for the multi-turn needle-in-a-haystack prefix-cache check.

Needle and haystack generation, chat-completion parsing, recall matching and
the cache-hit expectation. No network and no scenario imports, so the module
is unit-testable against synthetic responses.
"""

from __future__ import annotations

import json
import math
import random
import re
import string
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

# Uppercase letters and digits minus the glyphs a model transcribes
# inconsistently (I, L, O, 0, 1), so a mismatch is a recall miss and not a
# transcription slip.
_NEEDLE_LETTERS = "ABCDEFGHJKMNPQRSTUVWXYZ"
_NEEDLE_DIGITS = "23456789"

_FILLER_VOCAB = (
    "north",
    "south",
    "river",
    "valley",
    "signal",
    "beacon",
    "cargo",
    "lantern",
    "harbor",
    "meadow",
    "cipher",
    "orbit",
    "granite",
    "willow",
    "ember",
    "thunder",
    "marble",
    "compass",
    "velvet",
    "cobalt",
    "canyon",
    "glacier",
    "prairie",
    "summit",
    "anchor",
    "drifting",
    "quartz",
    "saffron",
    "timber",
    "juniper",
    "falcon",
    "otter",
    "badger",
    "heron",
    "walrus",
    "mantis",
    "pigeon",
    "serpent",
    "bison",
    "amber",
    "crimson",
    "indigo",
    "scarlet",
    "auburn",
    "hazel",
    "copper",
    "silver",
    "golden",
    "bronze",
)

_WORDS_PER_LINE = 12
# A line is "[16-char id] entry N: <12 words>". The id tokenizes into many
# short pieces; Qwen2.5 spends about 37 tokens per line. Rounding up keeps
# the haystack under the target rather than over the server's max length.
_EST_TOKENS_PER_LINE = 40

RECALL_QUESTION = (
    "What is the secret access code stated in the log above? "
    "Reply with ONLY the code, nothing else."
)


def generate_needle(rng: random.Random, length: int = 8) -> str:
    """Returns a high-entropy code containing at least one letter and one digit.

    A bare number could be comma-grouped by the model (``12345678`` to
    ``12,345,678``); mixing in a letter rules that out.
    """
    if length < 2:
        raise ValueError(f"needle length must be >= 2, got {length}")
    alphabet = _NEEDLE_LETTERS + _NEEDLE_DIGITS
    chars = [rng.choice(_NEEDLE_LETTERS), rng.choice(_NEEDLE_DIGITS)]
    chars += [rng.choice(alphabet) for _ in range(length - 2)]
    rng.shuffle(chars)
    return "".join(chars)


def build_haystack(
    rng: random.Random,
    needle: str,
    target_tokens: int,
    position_fraction: float,
) -> str:
    """Builds a unique log-shaped context with the needle on one line.

    Every line carries a per-conversation random id, so no two conversations
    share a cacheable block and any prefix hit is the conversation's own.
    """
    if not 0.0 <= position_fraction <= 1.0:
        raise ValueError(
            f"position_fraction must be in [0, 1], got {position_fraction}"
        )
    n_lines = max(1, target_tokens // _EST_TOKENS_PER_LINE)
    conv_id = "".join(
        rng.choice(string.ascii_uppercase + string.digits) for _ in range(16)
    )
    lines = [
        f"[{conv_id}] entry {i}: "
        + " ".join(rng.choice(_FILLER_VOCAB) for _ in range(_WORDS_PER_LINE))
        for i in range(n_lines)
    ]
    pos = min(len(lines), int(position_fraction * len(lines)))
    lines.insert(
        pos, f"The secret access code is {needle}. Remember this code exactly."
    )
    header = (
        "Read the following log carefully and remember the secret access "
        "code embedded in it.\n\n"
    )
    return header + "\n".join(lines)


def first_turn_content(haystack: str) -> str:
    return f"{haystack}\n\n{RECALL_QUESTION}"


# Opening or closing think tag, tolerant of attributes and of ``<thinking>``.
_THINK_TAG_RE = re.compile(r"<\s*(/?)\s*think[^>]*>", re.IGNORECASE)


def split_think_block(content: str) -> tuple[str, str]:
    """Splits folded ``<think>...</think>`` reasoning out of message content.

    A server that does not split reasoning into its own field folds it into
    ``content``, where the needle appears while the model re-reads the prompt.
    Scoring the raw content would then count a recall the model never made,
    so only the text outside think blocks is visible. Nesting is tracked; an
    unterminated block runs to the end, and a closing tag with no opener
    closes a block that started where the visible run began (a chat template
    that pre-fills ``<think>`` leaves only the closing half in the output).
    """
    visible: list[str] = []
    thinking: list[str] = []
    depth = 0
    resume = 0
    thought_start = 0
    for tag in _THINK_TAG_RE.finditer(content):
        if tag.group(1):
            if depth == 0:
                thinking.append(content[resume : tag.start()])
                resume = thought_start = tag.end()
                continue
            depth -= 1
            if depth == 0:
                thinking.append(content[thought_start : tag.start()])
                resume = tag.end()
        else:
            if depth == 0:
                visible.append(content[resume : tag.start()])
                thought_start = tag.end()
            depth += 1
    if depth > 0:
        thinking.append(content[thought_start:])
    else:
        visible.append(content[resume:])
    return "".join(visible), "\n".join(thinking)


def normalize(text: str) -> str:
    """Strips non-alphanumerics and uppercases so ``K7 X9-Q2`` matches ``K7X9Q2``."""
    return re.sub(r"[^A-Za-z0-9]", "", text).upper()


def needle_recalled(visible: str, needle: str) -> bool:
    return normalize(needle) in normalize(visible)


def find_leaked_needles(
    text: str, own_needle: str, needle_pool: Iterable[str]
) -> list[str]:
    """Returns other conversations' needles that appear in ``text``."""
    norm_text = normalize(text)
    own = normalize(own_needle)
    return sorted(
        needle
        for needle in {normalize(n) for n in needle_pool}
        if needle != own and needle in norm_text
    )


def _as_int(value: object) -> int | None:
    """Returns ``value`` as an int, or None for a missing or malformed field.

    ``bool`` is an ``int`` subclass and a non-finite float parses from
    ``NaN``/``1e9999``; neither is a token count.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return int(value)


def _as_dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


@dataclass
class TurnReply:
    """One chat-completion response reduced to what the check scores."""

    status: int
    content: str = ""
    """Raw ``message.content``, appended to the history as the assistant turn."""
    visible: str = ""
    """``content`` with folded think blocks removed; the text that is scored."""
    reasoning: str = ""
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.error is None

    @property
    def has_answer(self) -> bool:
        return bool(self.visible.strip())

    @property
    def reports_usage(self) -> bool:
        return (
            self.prompt_tokens is not None
            and self.prompt_tokens > 0
            and self.cached_tokens is not None
        )


def parse_reply(status: int, body: str, error: str | None = None) -> TurnReply:
    """Parses a raw chat-completion response into a :class:`TurnReply`.

    A 200 whose body is missing or mistypes a field is recorded with an
    ``error`` rather than raised: one odd response must not lose the run.
    """
    if status != 200:
        return TurnReply(status, error=error or f"HTTP {status}")
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        return TurnReply(status, error=f"malformed body: {exc}")
    choices = _as_dict(data).get("choices")
    if not isinstance(choices, list) or not choices:
        return TurnReply(status, error="response has no choices")
    choice = _as_dict(choices[0])
    message = _as_dict(choice.get("message"))
    content = message.get("content")
    content = content if isinstance(content, str) else ""
    visible, folded = split_think_block(content)
    reasoning = ""
    # MAX emits reasoning under ``reasoning``; ``reasoning_content`` is the
    # deprecated alias other servers still use.
    for name in ("reasoning", "reasoning_content"):
        value = message.get(name)
        if isinstance(value, str) and value:
            reasoning = value
            break
    finish_reason = choice.get("finish_reason")
    usage = _as_dict(_as_dict(data).get("usage"))
    details = _as_dict(usage.get("prompt_tokens_details"))
    return TurnReply(
        status,
        content=content,
        visible=visible,
        reasoning=reasoning or folded,
        finish_reason=finish_reason if isinstance(finish_reason, str) else None,
        prompt_tokens=_as_int(usage.get("prompt_tokens")),
        cached_tokens=_as_int(details.get("cached_tokens")),
    )


class CacheHit(Enum):
    """How much of a follow-up turn's prompt the server reused."""

    FULL = "full"
    """At least the previous turn's whole prompt, to the page boundary."""
    PARTIAL = "partial"
    """Something was reused, but less than the previous prompt."""
    MISS = "miss"
    """Nothing was reused."""
    UNKNOWN = "unknown"
    """The response did not report usable token counts."""


def expected_cached_tokens(prev_prompt_tokens: int, block_size: int) -> int:
    """The least a follow-up turn should report as cached.

    The previous turn's prompt is a prefix of this one, and the server commits
    a KV page once it is full, so every page the previous prompt filled is
    reusable. The trailing partial page is not, which is why the bound rounds
    down rather than demanding the whole previous prompt.
    """
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    return (prev_prompt_tokens // block_size) * block_size


def classify_cache_hit(
    reply: TurnReply, prev_prompt_tokens: int | None, block_size: int
) -> CacheHit:
    if not reply.reports_usage or prev_prompt_tokens is None:
        return CacheHit.UNKNOWN
    assert reply.cached_tokens is not None
    if reply.cached_tokens <= 0:
        return CacheHit.MISS
    if reply.cached_tokens >= expected_cached_tokens(
        prev_prompt_tokens, block_size
    ):
        return CacheHit.FULL
    return CacheHit.PARTIAL


def haystack_token_target(max_position_embeddings: int) -> int:
    """Picks a haystack size that leaves room for the follow-up turns.

    A quarter of the window, capped so the default run stays quick, and never
    below a few KV pages so the needle can land in more than one of them.
    """
    return max(256, min(2048, max_position_embeddings // 4))


def long_haystack_token_target(max_position_embeddings: int) -> int:
    """Picks a haystack size for the long-context conversations.

    Half the window, so a production-shaped prompt of tens of thousands of
    tokens is exercised while the follow-up turns still fit, capped where a
    million-token window would otherwise make one prefill the whole run.
    """
    return max(256, min(131072, max_position_embeddings // 2))
