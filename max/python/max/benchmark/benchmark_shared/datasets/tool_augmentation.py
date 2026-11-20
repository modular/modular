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

"""Dataset-agnostic tool-definition mixing for benchmark workloads.

Attaches an OpenAI ``tools`` list to a fraction of already-sampled requests or
chat sessions, so a workload can carry the share of tool-calling traffic
production sees -- the quantity ``maxserve.tool_call.requests`` reports.

Selection is per session, unlike ``response_format_augmentation``'s per turn.
Chat templates such as MiniMax M3's render tool definitions into the system
block, ahead of the conversation, so a session that toggled tools between turns
would change its prompt from that block onward and miss the prefix cache on
every toggle. An agent sends the same tools on every turn it takes.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Mapping, Sequence
from typing import Any

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .chat_judge import ChatJudgeChatSamples
from .types import ChatSamples, RequestSamples, Samples

logger = logging.getLogger(__name__)


def augment_samples_with_tools(
    samples: Samples,
    *,
    tools: Sequence[Mapping[str, Any]],
    fraction: float,
    tokenizer: PreTrainedTokenizerBase | None,
    seed: int | None = None,
) -> None:
    """Attach ``tools`` to a fraction of already-sampled requests/sessions.

    The rendered tool definitions are carved out of the prompt they join rather
    than added on top of it: the first user turn of a selected session, or a
    selected single-turn string prompt, is trimmed by the block's token cost
    while its drawn length stays put. A prompt shorter than the block is sent
    as the block alone, and its recorded length grows to the block's. An
    input-length distribution fitted to production traffic already includes
    the tools its clients send, so adding the block on top would inflate
    processed tokens per minute. Single-turn prompts already split into chat
    messages are sent untrimmed.

    A single-turn request whose dataset already attached its own tools keeps
    them, and is neither selected nor trimmed.

    ``ignore_eos`` is left as drawn. With the default ``tool_choice`` of auto,
    MAX Serve enforces its tool grammar only between a call's start and end
    tags, so generating past a finished call runs unconstrained instead of
    tripping the matcher the way a satisfied ``response_format`` does.

    Args:
        samples: Already-sampled requests or chat sessions, mutated in place.
        tools: The OpenAI ``tools`` list to attach.
        fraction: Fraction (0.0-1.0) of requests (single-turn) or chat sessions
            (multi-turn) to attach it to. A selected session carries it on
            every user turn.
        tokenizer: Sizes the rendered tool block for the prompt carve-out.
            Without one the carve-out is skipped, and the prompt grows by the
            block.
        seed: Seed for this function's own RNG, so the selection is
            reproducible without displacing the draws any later augmentation
            makes.

    Raises:
        ValueError: If ``fraction`` is outside [0, 1] or ``tools`` is empty.
        TypeError: If ``samples`` is neither ``RequestSamples`` nor
            ``ChatSamples``.
    """
    if not (0.0 <= fraction <= 1.0):
        raise ValueError(f"tools_fraction must be in [0, 1], got {fraction}")
    if not tools:
        raise ValueError("tools must hold at least one tool definition")
    if fraction == 0:
        logger.info(
            "Tool mixing: fraction is 0, leaving the workload tool-free"
        )
        return

    tool_list = [dict(tool) for tool in tools]
    block_tokens = _tool_block_tokens(tokenizer, tool_list)
    # Salted so a run mixing tools and response_format under one seed does not
    # draw both selections from the same stream, which would correlate them.
    rng = random.Random(None if seed is None else f"tools-{seed}")

    max_context = tokenizer.model_max_length if tokenizer is not None else None
    if isinstance(samples, RequestSamples):
        selected = 0
        grown = 0
        over_context = 0
        kept_dataset_tools = 0
        untrimmed = 0
        for request in samples.requests:
            if request.tools:
                kept_dataset_tools += 1
                continue
            if rng.random() >= fraction:
                continue
            request.tools = tool_list
            selected += 1
            if tokenizer is None:
                continue
            if isinstance(request.prompt_formatted, str):
                request.prompt_formatted = _trim_tail(
                    tokenizer, request.prompt_formatted, block_tokens
                )
                if request.prompt_len < block_tokens:
                    request.prompt_len = block_tokens
                    grown += 1
                    if (
                        max_context is not None
                        and request.prompt_len + (request.output_len or 0)
                        > max_context
                    ):
                        over_context += 1
            else:
                untrimmed += 1
        logger.info(
            "Tool mixing: attached %d tools (~%d tokens) to %d/%d requests",
            len(tool_list),
            block_tokens,
            selected,
            len(samples.requests),
        )
        if kept_dataset_tools:
            logger.warning(
                "Tool mixing: %d requests already carry their dataset's tools;"
                " kept those instead of --tools",
                kept_dataset_tools,
            )
        if untrimmed:
            logger.warning(
                "Tool mixing: %d selected prompts are chat messages, which are"
                " not trimmed, so each grows by the ~%d-token tool block",
                untrimmed,
                block_tokens,
            )
        _warn_grown(grown, over_context, block_tokens, "requests")
    elif isinstance(samples, ChatJudgeChatSamples):
        # chat_judge_session_driver builds its own requests and never reads the
        # per-turn field, so attaching here would be reported but never sent.
        logger.warning(
            "Tool mixing: skipping chat-judge workload; its driver does not"
            " send tools."
        )
    elif isinstance(samples, ChatSamples):
        selected_sessions = 0
        tool_turns = 0
        total_turns = 0
        grown = 0
        over_context = 0
        for session in samples.chat_sessions:
            user_turns = [m for m in session.messages if m.source == "user"]
            total_turns += len(user_turns)
            if not user_turns or rng.random() >= fraction:
                continue
            for message in user_turns:
                message.tools = tool_list
            if tokenizer is not None:
                first = user_turns[0]
                first.content = _trim_tail(
                    tokenizer, first.content, block_tokens
                )
                if first.num_tokens < block_tokens:
                    first.num_tokens = block_tokens
                    grown += 1
                    if (
                        max_context is not None
                        and sum(m.num_tokens for m in session.messages)
                        > max_context
                    ):
                        over_context += 1
            selected_sessions += 1
            tool_turns += len(user_turns)
        # Sessions are selected but production reports requests; turn counts
        # vary by session, so the two shares can differ.
        logger.info(
            "Tool mixing: attached %d tools (~%d tokens) to %d/%d chat"
            " sessions, %d/%d user turns (%.1f%% of requests)",
            len(tool_list),
            block_tokens,
            selected_sessions,
            len(samples.chat_sessions),
            tool_turns,
            total_turns,
            100.0 * tool_turns / total_turns if total_turns else 0.0,
        )
        _warn_grown(grown, over_context, block_tokens, "sessions")
    else:
        raise TypeError(f"Unsupported samples type: {type(samples)}")


def _tool_block_tokens(
    tokenizer: PreTrainedTokenizerBase | None, tools: list[dict[str, Any]]
) -> int:
    """Tokens the chat template spends rendering ``tools``.

    Measured as the difference between a probe conversation rendered with and
    without them, so it includes the template's own framing. Falls back to the
    tools' JSON when the template fails or renders nothing for them.
    """
    if tokenizer is None:
        return 0
    probe = [{"role": "user", "content": "x"}]
    try:
        with_tools = tokenizer.apply_chat_template(
            probe, tools=tools, tokenize=False, add_generation_prompt=True
        )
        without_tools = tokenizer.apply_chat_template(
            probe, tokenize=False, add_generation_prompt=True
        )
    # A tokenizer without a chat template raises ValueError, and a template can
    # raise jinja errors on tool shapes it does not expect.
    except Exception as e:
        logger.warning(
            "Tool mixing: chat template could not render tools (%s); sizing"
            " the block from its JSON instead",
            e,
        )
    else:
        assert isinstance(with_tools, str) and isinstance(without_tools, str)
        delta = len(
            tokenizer.encode(with_tools, add_special_tokens=False)
        ) - len(tokenizer.encode(without_tools, add_special_tokens=False))
        if delta > 0:
            return delta
    return len(tokenizer.encode(json.dumps(tools), add_special_tokens=False))


def _warn_grown(
    grown: int, over_context: int, block_tokens: int, unit: str
) -> None:
    """Reports prompts that the tool block outgrew, once per workload."""
    if not grown:
        return
    logger.warning(
        "Tool mixing: %d %s open with a prompt shorter than the ~%d-token tool"
        " block, so each is sent as the block alone, longer than drawn and"
        " without its shared prefix",
        grown,
        unit,
        block_tokens,
    )
    if over_context:
        # The drivers still stop at the context limit; this says how often.
        logger.warning(
            "Tool mixing: %d of those %s now exceed the model's max context"
            " and will end early",
            over_context,
            unit,
        )


def _trim_tail(
    tokenizer: PreTrainedTokenizerBase, text: str, num_tokens: int
) -> str:
    """Drops ``num_tokens`` from the end of ``text``.

    The end, because a fitted first turn opens with the system prompt that
    sessions share. A block longer than the turn's body still trims into that
    prompt, and the session then misses the shared prefix.
    """
    if num_tokens <= 0:
        return text
    ids = tokenizer.encode(text, add_special_tokens=False)
    if num_tokens >= len(ids):
        return ""
    return tokenizer.decode(ids[:-num_tokens])
