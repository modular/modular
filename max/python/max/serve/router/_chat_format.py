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

"""The chat prompt format for decisions.

A question becomes one user message under the model's chat template, and the
answer letter is read at the position where the assistant turn opens. Any
instruction-tuned model with a chat template works without further setup.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar, Protocol, runtime_checkable

from max.pipelines.context.exceptions import InputError
from max.pipelines.modeling.types import (
    LabelScoringTokenizer,
    TextGenerationRequestMessage,
)
from max.serve.pipelines._label_scoring import LabelScoringOutput
from max.serve.router._decisions_prompt import (
    QuestionScores,
    QuestionView,
    default_labels,
    label_mass,
    option_probabilities,
    render_question,
    render_text,
    resolve_label_token_ids,
    resolve_scores,
)
from max.serve.router._decisions_scoring import (
    DecisionRequestError,
    EncodedQuestion,
    EncodedRow,
)
from max.serve.schemas._chat_template import without_thinking
from max.serve.schemas._decisions import DecisionText

CHAT_FORMAT_VERSION = 1
"""Version of the chat prompt wording. Any change needs a new version."""

CHAT_TEMPERATURE = 1.0
"""Default softmax temperature of the chat format."""


@runtime_checkable
class _ReasoningDelimiters(Protocol):
    """A tokenizer that resolved its reasoning-span delimiters to ids.

    Only reasoning models have these; for the rest there is no span to leave
    open, so the check below does not apply.
    """

    @property
    def reasoning_start_token_id(self) -> int: ...

    @property
    def reasoning_end_token_id(self) -> int: ...


def _check_reasoning_closed(
    tokenizer: LabelScoringTokenizer, prompt_ids: Sequence[int]
) -> None:
    """Refuses a prompt that leaves a reasoning block open at the answer slot."""
    if not isinstance(tokenizer, _ReasoningDelimiters):
        return
    start_id = tokenizer.reasoning_start_token_id
    end_id = tokenizer.reasoning_end_token_id
    ids = list(prompt_ids)
    last_start = max(
        (i for i, t in enumerate(ids) if t == start_id), default=-1
    )
    last_end = max((i for i, t in enumerate(ids) if t == end_id), default=-1)
    if last_start > last_end:
        raise ValueError(
            "the chat template leaves a reasoning block open at the answer "
            "position, so this model is not supported with these "
            "chat_template_kwargs"
        )


class ChatFormat:
    """Renders each question as one chat turn and scores a single row."""

    version: ClassVar[int] = CHAT_FORMAT_VERSION

    async def encode(
        self,
        tokenizer: LabelScoringTokenizer,
        state: DecisionText,
        items: Sequence[tuple[str, QuestionView]],
        chat_template_kwargs: dict[str, Any],
    ) -> list[EncodedQuestion]:
        """Renders every question with the chat template.

        Args:
            tokenizer: The serving tokenizer.
            state: The text or JSON value the questions are about.
            items: Each question's id and view.
            chat_template_kwargs: Caller overrides of the chat template
                options; thinking stays off.

        Raises:
            DecisionRequestError: If thinking is turned on, or a question's
                prompt leaves reasoning open, does not fit the context length,
                or has an answer label that is not one distinct token. The
                message names the question.
        """
        try:
            options = without_thinking(chat_template_kwargs)
        except ValueError as e:
            raise DecisionRequestError(str(e)) from e
        text = render_text(state)
        return [
            await _encode_question(tokenizer, text, question_id, view, options)
            for question_id, view in items
        ]

    def scores(
        self,
        item: EncodedQuestion,
        results: Sequence[LabelScoringOutput],
        temperature: float | None,
    ) -> QuestionScores:
        """Option probabilities and label mass for one scored question.

        Args:
            item: The encoded question.
            results: One scoring result per row of ``item``.
            temperature: A request-wide softmax temperature, replacing the
                format's own.
        """
        if len(results) != len(item.rows):
            raise ValueError(
                f"{len(results)} results for {len(item.rows)} rows"
            )
        logprobs = results[0].label_log_probabilities
        used = CHAT_TEMPERATURE if temperature is None else temperature
        return resolve_scores(
            item.view,
            option_probabilities(logprobs, used),
            label_mass(logprobs),
            item.question_id,
        )


async def _encode_question(
    tokenizer: LabelScoringTokenizer,
    text: str,
    question_id: str,
    view: QuestionView,
    chat_template_options: dict[str, Any],
) -> EncodedQuestion:
    """Renders one question with the chat template and resolves its labels."""
    labels = default_labels(view)
    content = render_question(text, view, labels)

    async def encode(value: str) -> list[int]:
        encoded = await tokenizer.encode(value, add_special_tokens=False)
        return [int(token) for token in encoded]

    try:
        prompt_text = tokenizer.apply_chat_template(
            [TextGenerationRequestMessage(role="user", content=content)],
            tools=None,
            **chat_template_options,
        )
        prompt_ids = await encode(prompt_text)
        _check_reasoning_closed(tokenizer, prompt_ids)
        max_length = tokenizer.max_length
        if max_length and len(prompt_ids) + 1 > max_length:
            raise ValueError(
                f"the prompt has {len(prompt_ids)} tokens, which does not "
                f"fit the context length of {max_length} tokens"
            )
        label_ids = await resolve_label_token_ids(
            encode, prompt_text, prompt_ids, labels
        )
    except (ValueError, InputError) as e:
        raise DecisionRequestError(f"question {question_id!r}: {e}") from e
    return EncodedQuestion(
        question_id, view, [EncodedRow(prompt_ids, label_ids)]
    )
