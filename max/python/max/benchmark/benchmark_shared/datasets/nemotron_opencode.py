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
from collections.abc import Iterable, Sequence
from typing import Any

from datasets import load_dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ._tokenizer_pool import TokenizerPool
from .distribution import DistributionParameter
from .huggingface import HuggingFaceBenchmarkDataset
from .multiturn_distribution_fit import build_chat_samples_from_user_text_pool
from .types import (
    ChatMessage,
    ChatSamples,
    ChatSession,
    RequestSamples,
    SampledRequest,
    SessionMessage,
    TextContentBlock,
    estimate_num_tokens,
)

logger = logging.getLogger(__name__)


NEMOTRON_OPENCODE_REPO_ID = "nvidia/Nemotron-SFT-OpenCode-v1"

# Each subset is a separate ``<subset>/data.jsonl`` LFS blob (3-5 GB). The
# default is the smallest one so first-time runs do not block on downloading
# the entire 30 GB corpus when only a few thousand prompts are needed.
NEMOTRON_OPENCODE_SUBSETS: tuple[str, ...] = (
    "general",
    "bash_only_tool",
    "bash_only_tool_skills",
    "question_tool",
    "agent_skills",
    "agent_skills_question_tool",
)
DEFAULT_NEMOTRON_OPENCODE_SUBSET = "agent_skills_question_tool"

# Roles that are safe to forward as-is when ``enable_tool_calls=False``.
_CHAT_ONLY_ROLES = frozenset({"system", "user", "assistant"})

# Cap on rows we will stream before giving up trying to build a sample. The
# dataset has hundreds of thousands of rows; this is just defensive headroom
# so we do not loop forever if every row is filtered out by user options.
_MAX_STREAM_ROWS_PER_REQUEST = 50


class NemotronOpenCodeBenchmarkDataset(HuggingFaceBenchmarkDataset):
    """Benchmark dataset backed by ``nvidia/Nemotron-SFT-OpenCode-v1``.

    The corpus contains ~459K synthetic agentic-coding traces generated with
    Qwen3-Coder-480B inside the OpenCode CLI harness. Each row is a single
    multi-turn conversation laid out as:

    - ``messages``: list of ``{"role", "content"}`` entries covering the full
      system/user/assistant/tool exchange.
    - ``tools``: list of OpenAI-style tool schemas the model was permitted to
      call during the trace.
    - ``enabled_tools``, ``agent_prompt``, ``skills_path``, ``uuid``,
      ``question_category``, ``complexity_level``, ``hf_split``: trace
      metadata.

    The data lives in six per-subset ``<subset>/data.jsonl`` LFS blobs
    (3-5 GB each). To avoid downloading 30 GB before the first prompt is
    issued, this dataset uses ``datasets.load_dataset(..., streaming=True)``
    and pulls only enough rows to satisfy the requested sample count.

    Coverage gap vs. existing datasets:

    - ``instruct-coder`` is pure single-turn instruction/edit and has no tool
      messages.
    - ``agentic-code`` replays 22 recorded Claude Code sessions; it includes
      tool/assistant messages but ships only a few dozen distinct prompts.

    This dataset is ~3 orders of magnitude larger and exposes a much wider
    spread of agentic-coding prompt shapes. The tool definitions on each row
    are preserved in :attr:`last_loaded_tool_schemas` for inspection, but
    they are *not* forwarded to the server: the benchmark request pipeline
    does not currently send a ``tools=[...]`` field on chat completions
    payloads. Wiring that through (``SampledRequest.tools`` →
    ``RequestFuncInput.tools`` → ``OpenAIChatCompletionsRequestDriver``) is
    tracked separately. Until then, this dataset behaves as a richer agentic
    prompt corpus, not a true tool-call protocol exerciser.
    """

    #: Subset name (one of :data:`NEMOTRON_OPENCODE_SUBSETS`). Override before
    #: calling :meth:`sample_requests` to pull from a different split.
    subset: str = DEFAULT_NEMOTRON_OPENCODE_SUBSET

    #: Tool schemas captured from the most recent ``sample_requests`` /
    #: ``gen_multiturn_sessions`` call. One entry per emitted request/session.
    #: Empty until the first call. Exposed for follow-up work that wires
    #: ``tools`` through the request payload.
    last_loaded_tool_schemas: list[list[dict[str, Any]]]

    def __init__(self) -> None:
        self.last_loaded_tool_schemas = []

    def fetch(self) -> None:
        # Streaming download happens lazily inside ``_iter_rows``; there is no
        # single file to materialise up front and we deliberately avoid eager
        # downloads since each subset is multiple GB.
        #
        # The streaming loader cannot be pointed at a local JSONL via
        # ``--dataset-path`` today: each row is several KB of nested JSON and
        # the row->SampledRequest pipeline reads the schema with HF column
        # names, so a hand-curated local file would have to match the upstream
        # layout exactly. Surface that explicitly rather than silently
        # ignoring the flag.
        if self.dataset_path is not None:
            raise ValueError(
                "nemotron-opencode does not support --dataset-path; the "
                "loader streams "
                f"{NEMOTRON_OPENCODE_REPO_ID} via "
                "datasets.load_dataset(streaming=True). Drop --dataset-path "
                "and let the dataset stream from HuggingFace."
            )

    # ------------------------------------------------------------------ utils

    def _iter_rows(self) -> Iterable[dict[str, Any]]:
        """Stream rows from the configured subset.

        Yields raw dataset dicts in source order (the HF stream does not
        shuffle). Callers are responsible for breaking out of the iterator
        once they have enough samples.
        """
        if self.subset not in NEMOTRON_OPENCODE_SUBSETS:
            raise ValueError(
                f"Unknown Nemotron-OpenCode subset {self.subset!r}; expected "
                f"one of {sorted(NEMOTRON_OPENCODE_SUBSETS)}"
            )
        # NOTE: ``self.dataset_name`` is the registry key (e.g.
        # "nemotron-opencode"), not the HuggingFace repo id, so we
        # deliberately use the module constant here.
        stream = load_dataset(
            NEMOTRON_OPENCODE_REPO_ID,
            data_files=f"{self.subset}/data.jsonl",
            split="train",
            streaming=True,
        )
        yield from stream

    @staticmethod
    def _message_text(content: Any) -> str:
        """Flatten a message ``content`` field to a plain string."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                elif isinstance(part, str):
                    parts.append(part)
            return "".join(parts)
        return str(content)

    @classmethod
    def _to_chat_messages(
        cls,
        messages: Sequence[dict[str, Any]],
    ) -> list[ChatMessage]:
        """Convert raw dataset messages into ``ChatMessage`` objects.

        Tool/assistant messages with non-string content are flattened into a
        single ``TextContentBlock`` so they round-trip through the chat
        completions driver, which expects ``str`` or ``list[TextContentBlock]``.
        """
        chat_messages: list[ChatMessage] = []
        for msg in messages:
            role = msg.get("role")
            if not isinstance(role, str):
                continue
            text = cls._message_text(msg.get("content"))
            chat_messages.append(
                ChatMessage(role=role, content=[TextContentBlock(text=text)])
            )
        return chat_messages

    @staticmethod
    def _split_prompt_and_completion(
        messages: Sequence[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], str] | None:
        """Split a conversation into (prompt-messages, last-assistant-text).

        Returns ``None`` if the trace does not end on a non-empty assistant
        reply (e.g. truncated traces or traces that only contain tool calls).
        """
        last_assistant_idx: int | None = None
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg.get("role") != "assistant":
                continue
            text = NemotronOpenCodeBenchmarkDataset._message_text(
                msg.get("content")
            )
            if text.strip():
                last_assistant_idx = i
                break
        if last_assistant_idx is None or last_assistant_idx == 0:
            return None
        prompt_messages = list(messages[:last_assistant_idx])
        completion = NemotronOpenCodeBenchmarkDataset._message_text(
            messages[last_assistant_idx].get("content")
        )
        return prompt_messages, completion

    # ------------------------------------------------------------ single turn

    def sample_requests(
        self,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase,
        output_lengths: Sequence[int] | None = None,
        shuffle: bool = True,
        enable_tool_calls: bool = True,
        min_prompt_len: int = 4,
        min_output_len: int = 4,
        **kwargs,
    ) -> RequestSamples:
        """Sample single-turn requests by collapsing each row.

        Each row in the source is treated as one sample: the message history
        up to (but excluding) the final assistant message is sent as the
        prompt, and the final assistant message's tokenised length is used as
        the reference ``output_len``.

        Args:
            num_requests: Number of samples to emit.
            tokenizer: Tokenizer for computing token lengths.
            output_lengths: Optional pinned per-request output lengths. When
                set, ``ignore_eos`` is ``True`` and the dataset's reference
                completion length is ignored.
            shuffle: Whether to shuffle the buffered candidates before
                sampling. Note that the HF stream itself is not random; this
                only shuffles within an oversampled buffer.
            enable_tool_calls: When ``False``, rows whose prompt history
                contains a ``tool`` or non-``user/assistant/system`` role are
                skipped. The corpus is heavily tool-flavoured so disabling
                this discards most rows.
            min_prompt_len: Floor on tokenised prompt length.
            min_output_len: Floor on tokenised completion length (ignored when
                ``output_lengths`` is provided).
        """
        if num_requests <= 0:
            self.last_loaded_tool_schemas = []
            return RequestSamples(requests=[])

        # Oversample so ``shuffle=True`` is not a no-op; cap defensively so we
        # do not iterate the full multi-GB stream when only a handful of rows
        # are needed.
        candidate_count = min(
            max(num_requests * 4, num_requests + 32),
            num_requests + _MAX_STREAM_ROWS_PER_REQUEST * num_requests,
        )

        candidates: list[tuple[list[dict[str, Any]], str, list[dict[str, Any]]]]
        candidates = []
        for row in self._iter_rows():
            if len(candidates) >= candidate_count:
                break
            messages = row.get("messages")
            if not isinstance(messages, list):
                continue
            if not enable_tool_calls and any(
                m.get("role") not in _CHAT_ONLY_ROLES for m in messages
            ):
                continue
            split = self._split_prompt_and_completion(messages)
            if split is None:
                continue
            prompt_messages, completion = split
            tools_raw = row.get("tools") or []
            if not isinstance(tools_raw, list):
                tools_raw = []
            candidates.append((prompt_messages, completion, tools_raw))

        if shuffle:
            if output_lengths is not None:
                raise NotImplementedError(
                    "Shuffling with pinned output_lengths is not supported"
                )
            random.shuffle(candidates)

        sampled: list[SampledRequest] = []
        sampled_tool_schemas: list[list[dict[str, Any]]] = []
        for prompt_messages, completion, tools_raw in candidates:
            if len(sampled) >= num_requests:
                break
            chat_messages = self._to_chat_messages(prompt_messages)
            if not chat_messages:
                continue
            # Concatenated text token count is an approximation; the exact
            # wire length depends on the chat template, which is model-
            # specific. This is consistent with how other multi-message
            # datasets in this package estimate prompt_len.
            prompt_text = "\n".join(
                m.text for m in _iter_text_blocks(chat_messages)
            )
            prompt_len = estimate_num_tokens(tokenizer, prompt_text)
            if prompt_len < min_prompt_len:
                continue
            out_idx = len(sampled)
            if output_lengths is not None:
                output_len = output_lengths[out_idx]
            else:
                output_len = estimate_num_tokens(tokenizer, completion)
                if output_len < min_output_len:
                    continue
            sampled.append(
                SampledRequest(
                    prompt_formatted=chat_messages,
                    prompt_len=prompt_len,
                    output_len=output_len,
                    encoded_images=[],
                    # ``output_len`` is always set on this path, so pin
                    # ``ignore_eos=True`` to match sharegpt / arxiv: decode the
                    # full target length instead of letting an early EOS skew
                    # throughput vs. instruct-coder baselines.
                    ignore_eos=True,
                )
            )
            sampled_tool_schemas.append(tools_raw)

        self.last_loaded_tool_schemas = sampled_tool_schemas

        if len(sampled) < num_requests:
            logger.warning(
                "nemotron-opencode: only %d valid rows yielded after streaming "
                "%d candidates from subset %r; requested %d.",
                len(sampled),
                len(candidates),
                self.subset,
                num_requests,
            )
        return RequestSamples(requests=sampled)

    # --------------------------------------------------------------- multiturn

    def gen_multiturn_sessions(
        self,
        num_sessions: int,
        tokenizer: PreTrainedTokenizerBase,
        max_turns_per_session: int | None = None,
        shuffle: bool = True,
        *,
        pool: TokenizerPool | None = None,
        fit_length_distributions: bool = False,
        num_turns: DistributionParameter | None = None,
        input_len: DistributionParameter | None = None,
        output_len: DistributionParameter | None = None,
        delay_between_turns_dist: DistributionParameter | None = None,
        sys_prompt_ratio: float = 0.0,
        max_num_unique_sys_prompt: int = 1,
        min_input_len: int = 4,
        min_output_len: int = 1,
        enable_tool_calls: bool = True,
    ) -> ChatSamples:
        """Build :class:`ChatSamples` from streamed rows.

        Each row becomes one :class:`ChatSession`: consecutive user/assistant
        pairs from the recorded conversation become measured turns. Tool and
        system messages are dropped (the chat-session abstraction only models
        user/assistant alternation). Set ``enable_tool_calls=False`` to also
        skip any row whose conversation contains a non-user/assistant/system
        message before flattening.

        When ``fit_length_distributions=True``, behaves like the
        ``instruct-coder`` / ``agentic-code`` "fit" path: user texts are
        pooled and synthetic sessions are built to match
        ``num_turns`` / ``input_len`` / ``output_len`` distributions.
        """
        if fit_length_distributions:
            assert num_turns is not None, "num_turns required when fitting"
            assert input_len is not None, "input_len required when fitting"
            assert output_len is not None, "output_len required when fitting"
            if pool is None:
                raise ValueError(
                    "pool is required for nemotron-opencode "
                    "fit-distributions multiturn"
                )
            user_texts = self._collect_user_turn_texts(
                enable_tool_calls=enable_tool_calls,
                num_sessions=num_sessions,
            )
            if shuffle:
                random.shuffle(user_texts)
            return build_chat_samples_from_user_text_pool(
                pool=pool,
                user_text_pool=user_texts,
                num_sessions=num_sessions,
                num_turns=num_turns,
                input_len=input_len,
                output_len=output_len,
                delay_between_turns_dist=delay_between_turns_dist,
                sys_prompt_ratio=sys_prompt_ratio,
                max_num_unique_sys_prompt=max_num_unique_sys_prompt,
                min_input_len=min_input_len,
                min_output_len=min_output_len,
                shuffle_pool=False,
                log_prefix="nemotron-opencode",
            )

        # Oversample so shuffling is meaningful but bounded.
        candidate_count = min(
            max(num_sessions * 2, num_sessions + 8),
            num_sessions + _MAX_STREAM_ROWS_PER_REQUEST * num_sessions,
        )

        candidates: list[tuple[list[SessionMessage], list[dict[str, Any]]]] = []
        for row in self._iter_rows():
            if len(candidates) >= candidate_count:
                break
            messages = row.get("messages")
            if not isinstance(messages, list):
                continue
            if not enable_tool_calls and any(
                m.get("role") not in _CHAT_ONLY_ROLES for m in messages
            ):
                continue
            session_messages = self._build_session_messages(
                messages,
                tokenizer=tokenizer,
                max_turns_per_session=max_turns_per_session,
                min_input_len=min_input_len,
                min_output_len=min_output_len,
            )
            if not session_messages:
                continue
            tools_raw = row.get("tools") or []
            if not isinstance(tools_raw, list):
                tools_raw = []
            candidates.append((session_messages, tools_raw))

        if shuffle:
            random.shuffle(candidates)

        chat_sessions: list[ChatSession] = []
        session_tool_schemas: list[list[dict[str, Any]]] = []
        for idx, (session_messages, tools_raw) in enumerate(candidates):
            if len(chat_sessions) >= num_sessions:
                break
            chat_sessions.append(ChatSession(idx, session_messages))
            session_tool_schemas.append(tools_raw)

        self.last_loaded_tool_schemas = session_tool_schemas

        if len(chat_sessions) < num_sessions:
            logger.warning(
                "nemotron-opencode: only %d valid sessions yielded from "
                "subset %r; requested %d.",
                len(chat_sessions),
                self.subset,
                num_sessions,
            )
        return ChatSamples(chat_sessions=chat_sessions)

    def _build_session_messages(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tokenizer: PreTrainedTokenizerBase,
        max_turns_per_session: int | None,
        min_input_len: int,
        min_output_len: int,
    ) -> list[SessionMessage]:
        """Project a recorded conversation onto alternating user/assistant turns."""
        session_messages: list[SessionMessage] = []
        pending_user: str | None = None
        turns = 0
        for msg in messages:
            role = msg.get("role")
            text = self._message_text(msg.get("content"))
            if not text.strip():
                continue
            if role == "user":
                # Coalesce consecutive user-side messages (e.g. tool outputs
                # surfaced back to the user) into one turn.
                pending_user = (
                    text if pending_user is None else f"{pending_user}\n{text}"
                )
            elif role == "assistant":
                if pending_user is None:
                    continue
                user_tokens = estimate_num_tokens(tokenizer, pending_user)
                assistant_tokens = estimate_num_tokens(tokenizer, text)
                if (
                    user_tokens < min_input_len
                    or assistant_tokens < min_output_len
                ):
                    pending_user = None
                    continue
                session_messages.append(
                    SessionMessage(
                        source="user",
                        content=pending_user,
                        num_tokens=user_tokens,
                    )
                )
                session_messages.append(
                    SessionMessage(
                        source="assistant",
                        content="",
                        num_tokens=assistant_tokens,
                    )
                )
                pending_user = None
                turns += 1
                if (
                    max_turns_per_session is not None
                    and turns >= max_turns_per_session
                ):
                    break
            # system / tool / other roles are dropped: they belong to the
            # transcript, not the user/assistant alternation that ChatSession
            # models.
        return session_messages

    def _collect_user_turn_texts(
        self,
        *,
        enable_tool_calls: bool,
        num_sessions: int,
    ) -> list[str]:
        """Flatten user-side messages across rows into a pool of texts."""
        target = max(num_sessions * 8, num_sessions + 32)
        texts: list[str] = []
        for row in self._iter_rows():
            if len(texts) >= target:
                break
            messages = row.get("messages")
            if not isinstance(messages, list):
                continue
            if not enable_tool_calls and any(
                m.get("role") not in _CHAT_ONLY_ROLES for m in messages
            ):
                continue
            for msg in messages:
                if msg.get("role") != "user":
                    continue
                text = self._message_text(msg.get("content"))
                if text.strip():
                    texts.append(text)
        return texts


def _iter_text_blocks(
    chat_messages: Sequence[ChatMessage],
) -> Iterable[TextContentBlock]:
    for msg in chat_messages:
        if isinstance(msg.content, list):
            for part in msg.content:
                if isinstance(part, TextContentBlock):
                    yield part
        elif isinstance(msg.content, str):
            yield TextContentBlock(text=msg.content)
