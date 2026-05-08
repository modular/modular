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

"""Build multiturn :class:`ChatSamples` from a pool of user texts and CLI distributions.

Used when ``--fit-distributions`` is set so real datasets (instruct-coder,
agentic-code, etc.) match ``--random-*`` and ``--delay-between-chat-turns``
the same way as :class:`RandomBenchmarkDataset` multiturn mode.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import numpy as np
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .distribution import BaseDistribution, DistributionParameter
from .types import (
    ChatSamples,
    ChatSession,
    SessionMessage,
    SharedContext,
    estimate_num_tokens,
)

logger = logging.getLogger(__name__)

# Match RandomBenchmarkDataset: headroom for template / special-token overhead.
MAX_CONTEXT_USAGE_RATIO = 0.95


def resolve_constant_delay_ms(
    param: DistributionParameter | None,
) -> float | None:
    """Resolve a CLI delay parameter to one non-negative delay in milliseconds.

    Used when per-turn distribution sampling is disabled: sample once (constants
    return themselves).

    Args:
        param: Raw ``delay_between_chat_turns`` from config, or None.

    Returns:
        Delay in ms, or None when unset.
    """
    if param is None:
        return None
    dist = BaseDistribution.from_distribution_parameter(param)
    if dist is None:
        return None
    return max(float(dist.sample_value()), 0.0)


def parse_two_part_distribution(
    param: DistributionParameter,
    label: str,
) -> tuple[DistributionParameter, DistributionParameter]:
    """Split ``first;rest`` specs the same way as :class:`RandomBenchmarkDataset`."""
    if isinstance(param, str):
        parts = param.split(";")
        if len(parts) > 2:
            raise ValueError(
                f"{label}: at most two segments separated by ';' "
                f"(first turn vs remaining turns), got {len(parts)}"
            )
        first = parts[0].strip()
        second = parts[1].strip() if len(parts) == 2 else first
        return first, second
    return param, param


def _replacement_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    return next(
        tid
        for candidate in [" ", chr(0x0120), chr(0x2581)]
        if (tid := tokenizer.convert_tokens_to_ids(candidate))
        not in (None, tokenizer.unk_token_id)
    )


def _sanitize_token_ids(
    tokenizer: PreTrainedTokenizerBase, ids: list[int]
) -> list[int]:
    special_ids = set(tokenizer.all_special_ids)
    replacement = _replacement_token_id(tokenizer)
    return [(replacement if (tid in special_ids) else tid) for tid in ids]


def _repeat_truncate_token_ids(
    token_ids: list[int], target_len: int
) -> list[int]:
    if target_len <= 0:
        return []
    assert token_ids, "Cannot scale an empty token sequence"
    prompt_len = len(token_ids)
    if prompt_len >= target_len:
        return token_ids[:target_len]
    ratio = (target_len + prompt_len - 1) // prompt_len
    return (token_ids * ratio)[:target_len]


def _text_scaled_to_token_length(
    tokenizer: PreTrainedTokenizerBase,
    base_text: str,
    target_len: int,
    min_len: int = 4,
) -> tuple[str, int]:
    target_len = max(target_len, min_len)
    ids = tokenizer.encode(base_text, add_special_tokens=False)
    ids = _sanitize_token_ids(tokenizer, ids)
    if not ids:
        rep = _replacement_token_id(tokenizer)
        ids = [rep] * min_len
    scaled_ids = _repeat_truncate_token_ids(ids, target_len)
    text = tokenizer.decode(scaled_ids, skip_special_tokens=False)
    n_tokens = estimate_num_tokens(tokenizer, text)
    return text, n_tokens


def _split_sys_user_token_targets(
    target_total: int,
    sys_prompt_ratio: float,
    min_user_len: int,
) -> tuple[int, int]:
    target_total = max(target_total, min_user_len)
    if sys_prompt_ratio <= 0:
        return 0, target_total
    sys_len = int(np.floor(target_total * sys_prompt_ratio))
    user_len = target_total - sys_len
    if user_len < min_user_len:
        user_len = min_user_len
        sys_len = max(0, target_total - user_len)
    return sys_len, user_len


def _system_prompt_template(variant: int) -> str:
    return (
        "You are a helpful coding assistant. Follow the user's instructions "
        f"carefully (assistant profile {variant})."
    )


@dataclass(frozen=True)
class ScaledUserMessage:
    """User message scaled to a per-turn token budget, with optional sys prefix.

    `content` is the full text sent to the model
    (`sys_prefix.text + "\\n\\n" + user_body_scaled` when `sys_prefix` is set).
    `sys_prefix` describes the leading sys-prefix slice, or `None` when
    `sys_prompt_ratio <= 0`.
    """

    content: str
    num_tokens: int
    sys_prefix: SharedContext | None


def build_scaled_user_message(
    tokenizer: PreTrainedTokenizerBase,
    user_body: str,
    target_input_tokens: int,
    sys_prompt_ratio: float,
    sys_variant: int,
    min_input_len: int,
) -> ScaledUserMessage:
    """Optional synthetic system prefix + body scaled to the target token budget."""
    sys_len, user_len = _split_sys_user_token_targets(
        target_input_tokens, sys_prompt_ratio, min_input_len
    )
    parts: list[str] = []
    sys_prefix: SharedContext | None = None
    if sys_len > 0:
        sys_text, sys_tokens = _text_scaled_to_token_length(
            tokenizer, _system_prompt_template(sys_variant), sys_len, min_len=1
        )
        parts.append(sys_text)
        sys_prefix = SharedContext(text=sys_text, num_tokens=sys_tokens)
    user_text, _ = _text_scaled_to_token_length(
        tokenizer, user_body, user_len, min_len=min_input_len
    )
    parts.append(user_text)
    combined = "\n\n".join(parts)
    return ScaledUserMessage(
        content=combined,
        num_tokens=estimate_num_tokens(tokenizer, combined),
        sys_prefix=sys_prefix,
    )


def _register_longest_sys_prompt(
    warmup_dict: dict[int, SharedContext],
    variant_idx: int,
    sys_prefix: SharedContext,
) -> None:
    """Update warmup_dict[variant_idx] when sys_prefix.num_tokens exceeds the prior entry."""
    prev = warmup_dict.get(variant_idx)
    prev_tokens = prev.num_tokens if prev else 0
    if prev_tokens < sys_prefix.num_tokens:
        warmup_dict[variant_idx] = sys_prefix


def build_chat_samples_from_user_text_pool(
    tokenizer: PreTrainedTokenizerBase,
    user_text_pool: list[str],
    num_sessions: int,
    num_turns: DistributionParameter,
    input_len: DistributionParameter,
    output_len: DistributionParameter,
    delay_between_turns_dist: DistributionParameter | None,
    sys_prompt_ratio: float,
    max_num_unique_sys_prompt: int,
    min_input_len: int = 4,
    min_output_len: int = 1,
    *,
    shuffle_pool: bool = False,
    log_prefix: str = "multiturn-fit",
) -> ChatSamples:
    """Assemble :class:`ChatSamples` by grouping pooled user strings into sessions.

    For each session, samples a turn count and per-turn input/output targets from
    the given distributions (with optional ``';'`` split for first vs remaining
    turns). Each user message is produced by token-space repeat/truncation of a
    pooled string, optionally prefixed with a scaled synthetic system block.

    Args:
        tokenizer: Tokenizer used for counting and scaling.
        user_text_pool: Candidate user message bodies (one per underlying turn).
        num_sessions: Target number of chat sessions.
        num_turns: Distribution for turns per session (>= 1 after rounding).
        input_len: Per-turn user-side token target distribution(s).
        output_len: Per-turn assistant ``max_tokens`` distribution(s).
        delay_between_turns_dist: Optional per-assistant-message delay (ms).
        sys_prompt_ratio: Fraction of user message token budget for system prefix.
        max_num_unique_sys_prompt: Cycle count for system-prefix variants.
        min_input_len: Floor for sampled input lengths.
        min_output_len: Floor for sampled output lengths.
        shuffle_pool: Whether to shuffle ``user_text_pool`` in place before use.
        log_prefix: Logger prefix for warnings.

    Returns:
        Generated chat sessions (may be fewer than ``num_sessions`` if the pool
        is exhausted).
    """
    first_in, rest_in = parse_two_part_distribution(input_len, "input_len")
    first_out, rest_out = parse_two_part_distribution(output_len, "output_len")

    num_turns_dist = BaseDistribution.from_distribution_parameter(num_turns)
    in_first_dist = BaseDistribution.from_distribution_parameter(first_in)
    in_rest_dist = BaseDistribution.from_distribution_parameter(rest_in)
    out_first_dist = BaseDistribution.from_distribution_parameter(first_out)
    out_rest_dist = BaseDistribution.from_distribution_parameter(rest_out)
    delay_dist = BaseDistribution.from_distribution_parameter(
        delay_between_turns_dist
    )

    assert num_turns_dist is not None
    assert in_first_dist is not None
    assert in_rest_dist is not None
    assert out_first_dist is not None
    assert out_rest_dist is not None

    model_max_length = min(tokenizer.model_max_length, np.iinfo(np.int64).max)
    max_context_length = int(model_max_length * MAX_CONTEXT_USAGE_RATIO)

    filtered: list[str] = []
    for text in user_text_pool:
        if estimate_num_tokens(tokenizer, text) >= min_input_len:
            filtered.append(text)

    if shuffle_pool:
        random.shuffle(filtered)

    if not filtered:
        logger.warning(
            "%s: no valid user texts in pool after filtering (min_input_len=%s).",
            log_prefix,
            min_input_len,
        )
        return ChatSamples(chat_sessions=[])

    num_turns_per_session = [
        max(round(num_turns_dist.sample_value()), 1)
        for _ in range(num_sessions)
    ]
    total_turns_needed = sum(num_turns_per_session)
    if total_turns_needed > len(filtered):
        logger.warning(
            "%s: need %d user turns but only %d valid rows; some sessions shorter.",
            log_prefix,
            total_turns_needed,
            len(filtered),
        )

    sessions: list[ChatSession] = []
    warmup_dict: dict[int, SharedContext] = {}
    idx = 0
    max_variant = max(1, max_num_unique_sys_prompt)

    for session_id in range(num_sessions):
        if idx >= len(filtered):
            break

        n_turns = num_turns_per_session[session_id]
        messages: list[SessionMessage] = []
        context_tokens = 0

        for turn_i in range(n_turns):
            if idx >= len(filtered):
                break

            prompt_text = filtered[idx]

            if turn_i == 0:
                in_dist, out_dist = in_first_dist, out_first_dist
            else:
                in_dist, out_dist = in_rest_dist, out_rest_dist

            target_in = max(round(in_dist.sample_value()), min_input_len)
            target_out = max(round(out_dist.sample_value()), min_output_len)

            sys_variant = (session_id + turn_i) % max_variant
            msg = build_scaled_user_message(
                tokenizer,
                prompt_text,
                target_in,
                sys_prompt_ratio,
                sys_variant,
                min_input_len,
            )

            if (
                context_tokens + msg.num_tokens + target_out
                > max_context_length
            ):
                logger.info(
                    "%s session %s: stopping at turn %s (planned %s): "
                    "context %s + turn %s+%s exceeds max %s",
                    log_prefix,
                    session_id,
                    turn_i,
                    n_turns,
                    context_tokens,
                    msg.num_tokens,
                    target_out,
                    max_context_length,
                )
                break

            # Register after the context-length check; that bound check also
            # implicitly ensures sys_prefix stays within max_context_length.
            if msg.sys_prefix is not None:
                _register_longest_sys_prompt(
                    warmup_dict, sys_variant, msg.sys_prefix
                )

            idx += 1

            delay_ms = (
                max(float(delay_dist.sample_value()), 0.0)
                if delay_dist
                else None
            )

            messages.append(
                SessionMessage(
                    source="user",
                    content=msg.content,
                    num_tokens=msg.num_tokens,
                )
            )
            messages.append(
                SessionMessage(
                    source="assistant",
                    content="",
                    num_tokens=target_out,
                    delay_until_next_message=delay_ms,
                )
            )
            context_tokens += msg.num_tokens + target_out

        if len(messages) >= 2:
            sessions.append(ChatSession(session_id, messages))

    if len(sessions) < num_sessions:
        logger.warning(
            "%s: only %d sessions formed, requested %d.",
            log_prefix,
            len(sessions),
            num_sessions,
        )

    return ChatSamples(
        chat_sessions=sessions, shared_contexts=list(warmup_dict.values())
    )
