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

"""The decider prompt format for decisions.

Decider models (such as Mapika/decider-2b) are trained on a plain state-first
layout with no chat template: the state, then one question and its options,
then the answer letter. The state and each question are tokenized apart and
spliced as token ids, a score question is scored once per level, and the
softmax temperature comes from the checkpoint's ``decider_config.json``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any, ClassVar

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import (
    EntryNotFoundError,
    HFValidationError,
    RepositoryNotFoundError,
)
from max.pipelines.context.exceptions import InputError
from max.pipelines.modeling.types import LabelScoringTokenizer
from max.serve.pipelines._label_scoring import LabelScoringOutput
from max.serve.router._decider_prompt import (
    LETTERS,
    DeciderSettings,
    answer_type,
    combine_rows,
    parse_decider_config,
    plan_rows,
    render_row,
    render_state,
    state_prefix,
)
from max.serve.router._decisions_prompt import (
    QuestionScores,
    QuestionView,
    label_mass,
    option_probabilities,
    resolve_scores,
)
from max.serve.router._decisions_scoring import (
    DecisionRequestError,
    EncodedQuestion,
    EncodedRow,
)
from max.serve.schemas._decisions import DecisionText

DECIDER_FORMAT_VERSION = 2
"""Version of the decider prompt wording. Any change needs a new version."""

DECIDER_CONFIG_FILE = "decider_config.json"


def find_decider_settings(model_path: str) -> DeciderSettings | None:
    """Reads the checkpoint's decider settings, if it ships any.

    Args:
        model_path: A local checkpoint directory or a Hub repo id.

    Returns:
        The parsed ``decider_config.json``, or ``None`` for a checkpoint
        without one (an ordinary chat model).

    Raises:
        ValueError: If the file exists but is not a valid decider config.
    """
    try:
        if os.path.isdir(model_path):
            path = os.path.join(model_path, DECIDER_CONFIG_FILE)
        else:
            path = hf_hub_download(model_path, DECIDER_CONFIG_FILE)
        with open(path) as config_file:
            config: dict[str, Any] = json.load(config_file)
    except (
        EntryNotFoundError,
        RepositoryNotFoundError,
        HFValidationError,
        FileNotFoundError,
    ):
        return None
    return parse_decider_config(config)


class DeciderFormat:
    """Renders questions in the decider layout and combines per-level rows."""

    version: ClassVar[int] = DECIDER_FORMAT_VERSION

    def __init__(self, settings: DeciderSettings) -> None:
        self._settings = settings

    async def encode(
        self,
        tokenizer: LabelScoringTokenizer,
        state: DecisionText,
        items: Sequence[tuple[str, QuestionView]],
        chat_template_kwargs: dict[str, Any],
    ) -> list[EncodedQuestion]:
        """Renders questions in the decider layout with exact token ids.

        The state and each question are tokenized apart (as the model was
        trained), so the prompt is token ids rather than text.

        Args:
            tokenizer: The serving tokenizer.
            state: The text or JSON value the questions are about.
            items: Each question's id and view.
            chat_template_kwargs: Must be empty, since no chat template is
                rendered.

        Raises:
            DecisionRequestError: If chat template kwargs are given, the
                tokenizer cannot give the option letters one token each, or a
                question does not fit the context length. The message names
                the question.
        """
        if chat_template_kwargs:
            raise DecisionRequestError(
                "chat_template_kwargs is not used with a decider model, "
                "which renders no chat template"
            )
        settings = self._settings

        async def encode(text: str) -> list[int]:
            return [
                int(token)
                for token in await tokenizer.encode(
                    text, add_special_tokens=False
                )
            ]

        letter_ids = [await encode(letter) for letter in LETTERS]
        if any(len(ids) != 1 for ids in letter_ids) or len(
            {ids[0] for ids in letter_ids}
        ) != len(LETTERS):
            raise DecisionRequestError(
                "the option letters are not distinct single tokens for this "
                "tokenizer, so this model is not supported"
            )
        label_ids = [ids[0] for ids in letter_ids]
        state_ids = (await encode(state_prefix(render_state(state))))[
            : settings.max_state_tokens
        ]
        max_length = tokenizer.max_length
        encoded: list[EncodedQuestion] = []
        for question_id, view in items:
            try:
                rows = []
                for row in plan_rows(view, settings.isolated_levels):
                    prompt_ids = state_ids + await encode(render_row(row))
                    if max_length and len(prompt_ids) + 1 > max_length:
                        raise ValueError(
                            f"the prompt has {len(prompt_ids)} tokens, which "
                            f"does not fit the context length of "
                            f"{max_length} tokens"
                        )
                    rows.append(
                        EncodedRow(prompt_ids, label_ids[: len(row.options)])
                    )
            except (ValueError, InputError) as e:
                raise DecisionRequestError(
                    f"question {question_id!r}: {e}"
                ) from e
            encoded.append(EncodedQuestion(question_id, view, rows))
        return encoded

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
                checkpoint's per-answer-type one.
        """
        if len(results) != len(item.rows):
            raise ValueError(
                f"{len(results)} results for {len(item.rows)} rows"
            )
        settings = self._settings
        logprobs = [result.label_log_probabilities for result in results]
        used = (
            settings.temperature_for(answer_type(item.view))
            if temperature is None
            else temperature
        )
        row_probabilities = [
            option_probabilities(row, used) for row in logprobs
        ]
        return resolve_scores(
            item.view,
            combine_rows(
                item.view, row_probabilities, settings.isolated_levels
            ),
            min(label_mass(row) for row in logprobs),
            item.question_id,
        )
