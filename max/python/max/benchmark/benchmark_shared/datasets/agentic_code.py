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

from __future__ import annotations

import logging
import random
from collections.abc import Sequence
from typing import Any

import msgspec
from huggingface_hub import hf_hub_download
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .huggingface import HuggingFaceBenchmarkDataset
from .types import (
    ChatMessage,
    ChatSamples,
    ChatSession,
    RequestSamples,
    SampledRequest,
)

logger = logging.getLogger(__name__)


class MessageContentPart(msgspec.Struct):
    type: str
    text: str = ""


class Message(msgspec.Struct):
    role: str
    content: str | list[MessageContentPart] | None = None


class Turn(msgspec.Struct):
    messages: list[Message] | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: Any = None
    status_code: int = 200


class Session(msgspec.Struct):
    turns: list[Turn]


class AgenticCodeData(msgspec.Struct):
    sessions: list[Session]


class AgenticCodeBenchmarkDataset(HuggingFaceBenchmarkDataset):
    """Benchmark dataset from novita/agentic_code_dataset_22.

    22 real Claude Code sessions converted to OpenAI chat format.
    Each sample is a full turn with the complete message history,
    using pre-recorded input/output token counts from the original sessions.
    """

    def fetch(self) -> None:
        self.dataset_path = hf_hub_download(
            repo_id="novita/agentic_code_dataset_22",
            filename="e22_sessions_openai.json",
            repo_type="dataset",
        )

    def sample_requests(
        self,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase,
        output_lengths: Sequence[int] | None = None,
        shuffle: bool = True,
        enable_tool_calls: bool = True,
        **kwargs,
    ) -> RequestSamples:
        assert self.dataset_path is not None, (
            "dataset_path must be set; call fetch() first"
        )

        with open(self.dataset_path, "rb") as f:
            data = msgspec.json.decode(f.read(), type=AgenticCodeData)

        # Flatten all turns across all sessions into
        # (messages, input_tokens, output_tokens)
        turns: list[tuple[list[Message], int, int]] = []
        for session in data.sessions:
            for turn in session.turns:
                messages = turn.messages
                input_tokens = turn.input_tokens
                output_tokens = turn.output_tokens
                # Skip turns without messages or token counts
                if (
                    not messages
                    or input_tokens is None
                    or output_tokens is None
                ):
                    continue
                # Skip turns with errors
                if turn.error or turn.status_code != 200:
                    continue
                # Optionally skip turns that include tool calls
                if not enable_tool_calls:
                    allowed_roles = {"system", "user"}
                    if any(m.role not in allowed_roles for m in messages):
                        continue
                turns.append((messages, input_tokens, output_tokens))

        if output_lengths is not None and len(output_lengths) < len(turns):
            raise ValueError(
                f"output_lengths has {len(output_lengths)} entries but "
                f"there are {len(turns)} valid turns; "
                "output_lengths must be at least as long as the number of valid turns"
            )

        if shuffle:
            if output_lengths is not None:
                raise NotImplementedError(
                    "Shuffling with pinned output_lengths is not supported"
                )
            random.shuffle(turns)

        sampled: list[SampledRequest] = []
        for i, (messages, input_tokens, output_tokens) in enumerate(turns):
            if len(sampled) >= num_requests:
                break
            out_len = (
                output_tokens if output_lengths is None else output_lengths[i]
            )
            sampled.append(
                SampledRequest(
                    prompt_formatted=msgspec.to_builtins(messages),
                    prompt_len=input_tokens,
                    output_len=out_len,
                    encoded_images=[],
                    ignore_eos=(output_lengths is not None),
                )
            )

        if len(sampled) < num_requests:
            logger.warning(
                "agentic-code: only %d valid turns available, "
                "requested %d. Returning fewer requests.",
                len(sampled),
                num_requests,
            )

        return RequestSamples(requests=sampled)

    def gen_multiturn_sessions(
        self,
        num_sessions: int,
        max_turns_per_session: int | None = None,
        shuffle: bool = True,
    ) -> ChatSamples:
        """Generate multiturn ChatSessions from the agentic-code dataset.

        Each session corresponds to one recorded Claude Code session. Turns are
        replayed sequentially: the benchmark sends each user message to the live
        model, receives a real response, and appends it before the next turn.

        Args:
            num_sessions: Number of sessions to sample.
            max_turns_per_session: Cap on turns per session. Each turn produces
                one user+assistant round trip. Defaults to None (all turns).
            shuffle: Whether to shuffle session order before sampling.

        Returns:
            ChatSamples with one ChatSession per selected session.
        """
        assert self.dataset_path is not None, (
            "dataset_path must be set; call fetch() first"
        )

        with open(self.dataset_path, "rb") as f:
            data = msgspec.json.decode(f.read(), type=AgenticCodeData)

        # Build all valid sessions, then shuffle and slice.
        all_messages: list[list[ChatMessage]] = []
        for session in data.sessions:
            messages: list[ChatMessage] = []
            turns_added = 0
            for turn in session.turns:
                if (
                    max_turns_per_session is not None
                    and turns_added >= max_turns_per_session
                ):
                    break

                input_tokens = turn.input_tokens
                output_tokens = turn.output_tokens
                if input_tokens is None or output_tokens is None:
                    continue
                if turn.error or turn.status_code != 200:
                    continue

                # Extract the last user message from this turn's message list
                turn_messages = turn.messages or []
                user_content = next(
                    (
                        m.content
                        for m in reversed(turn_messages)
                        if m.role == "user"
                    ),
                    None,
                )
                if user_content is None:
                    continue

                # Flatten content to a plain string (handles str or list-of-parts)
                if isinstance(user_content, list):
                    user_text = " ".join(
                        part.text
                        for part in user_content
                        if part.type == "text"
                    )
                else:
                    user_text = user_content

                if not user_text.strip():
                    continue

                messages.append(
                    ChatMessage(
                        source="user",
                        content=user_text,
                        num_tokens=input_tokens,
                    )
                )
                messages.append(
                    ChatMessage(
                        source="assistant",
                        content="",  # filled by live model response
                        num_tokens=output_tokens,
                    )
                )
                turns_added += 1

            if len(messages) >= 2:
                all_messages.append(messages)

        if shuffle:
            random.shuffle(all_messages)

        chat_sessions: list[ChatSession] = [
            ChatSession(session_id, messages)
            for session_id, messages in enumerate(all_messages[:num_sessions])
        ]

        if len(chat_sessions) < num_sessions:
            logger.warning(
                "agentic-code: only %d valid sessions available, "
                "requested %d. Returning fewer sessions.",
                len(chat_sessions),
                num_sessions,
            )

        return ChatSamples(chat_sessions=chat_sessions)
