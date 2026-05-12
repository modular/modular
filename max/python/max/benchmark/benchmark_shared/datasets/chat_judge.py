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

import json
import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .local import LocalBenchmarkDataset
from .types import (
    ChatSamples,
    ChatSession,
    MessageSource,
    RequestSamples,
    SessionMessage,
    estimate_num_tokens,
)

logger = logging.getLogger(__name__)


@dataclass
class ChatJudgeChatSamples(ChatSamples):
    """Marker subclass that routes to the chat-judge driver path.

    Self-contained prompts: each turn already inlines its full context
    in the user message, so the driver sends ``[system?, user]`` per
    turn without accumulating assistant responses across turns. A
    turn-0 ``source="system"`` message, if present, is prepended to
    every user turn of the session.
    """


class ChatJudgeBenchmarkDataset(LocalBenchmarkDataset):
    """LLM-as-judge multi-turn workload backed by a JSONL session file.

    Models a per-turn judging / classification scenario (e.g. content
    moderation, safety scoring, RAG grading): each turn supplies a piece
    of content to evaluate with all relevant prior context already
    inlined as text in the user message. Self-contained prompts: each
    turn carries its full context, so the driver sends ``[system?, user]``
    per turn without accumulating assistant responses across turns.

    Format (per line, one session):
        {
          "session_id": "...",
          "turns": [
            {"text": "...", "role": "system"},   // turn 0: system prompt (optional)
            {"text": "..."},                      // turn 1: user
            {"text": "..."},                      // turn 2: user
            ...
          ]
        }

    The system prompt is per-session: each session may declare its own
    by setting ``role: "system"`` on turn 0, or omit it entirely.
    """

    def sample_requests(
        self,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase,
        output_lengths: Sequence[int] | None = None,
        shuffle: bool = True,
        **kwargs,
    ) -> RequestSamples:
        raise NotImplementedError(
            "ChatJudgeBenchmarkDataset is multi-turn only; use "
            "gen_chat_sessions() with --num-chat-sessions."
        )

    def gen_chat_sessions(
        self,
        num_sessions: int,
        tokenizer: PreTrainedTokenizerBase,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> ChatJudgeChatSamples:
        assert self.dataset_path is not None, (
            "dataset_path must be set; call fetch() first"
        )

        with open(self.dataset_path) as f:
            raw_sessions = [json.loads(line) for line in f if line.strip()]

        chat_sessions: list[ChatSession] = []
        for line_idx, raw in enumerate(raw_sessions):
            turns = raw.get("turns") or []
            if not turns:
                continue

            session_messages: list[SessionMessage] = []
            for turn_idx, turn in enumerate(turns):
                role = turn.get("role")
                if role == "system" and turn_idx != 0:
                    raise ValueError(
                        f"chat-judge line {line_idx} turn {turn_idx}: "
                        "system messages are only allowed at turn 0; "
                        "the driver prepends turn 0 (if system) to every "
                        "user turn and ignores anything else tagged system."
                    )
                source: MessageSource = "system" if role == "system" else "user"
                text = turn.get("text", "")
                session_messages.append(
                    SessionMessage(
                        source=source,
                        content=text,
                        num_tokens=estimate_num_tokens(tokenizer, text),
                    )
                )

            # Skip sessions with no user turns to drive.
            if not any(m.source == "user" for m in session_messages):
                continue

            chat_sessions.append(
                ChatSession(id=line_idx, messages=session_messages)
            )

        if not chat_sessions:
            raise ValueError(
                f"chat-judge: no sessions found in {self.dataset_path}"
            )

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(chat_sessions)

        if num_sessions < len(chat_sessions):
            chat_sessions = chat_sessions[:num_sessions]
        elif len(chat_sessions) < num_sessions:
            logger.warning(
                "chat-judge: requested %d sessions but only %d available in %s",
                num_sessions,
                len(chat_sessions),
                self.dataset_path,
            )

        return ChatJudgeChatSamples(chat_sessions=chat_sessions)
