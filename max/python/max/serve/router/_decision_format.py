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

"""Prompt formats for decisions: chat or decider.

A format decides how a question becomes scoring rows
(:meth:`DecisionFormat.encode`) and how row scores become option
probabilities (:meth:`DecisionFormat.scores`). The server picks one at
startup from the checkpoint: a model that ships ``decider_config.json`` is a
decider, any other is served with its chat template.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from max.pipelines.modeling.types import LabelScoringTokenizer
from max.serve.pipelines._label_scoring import LabelScoringOutput
from max.serve.router._chat_format import ChatFormat
from max.serve.router._decider_format import (
    DeciderFormat,
    find_decider_settings,
)
from max.serve.router._decisions_prompt import QuestionScores, QuestionView
from max.serve.router._decisions_scoring import EncodedQuestion
from max.serve.schemas._decisions import DecisionText


class DecisionFormat(Protocol):
    """How a server renders decision questions and reads their scores."""

    @property
    def version(self) -> int:
        """Version of the prompt wording, reported to callers."""
        ...

    async def encode(
        self,
        tokenizer: LabelScoringTokenizer,
        state: DecisionText,
        items: Sequence[tuple[str, QuestionView]],
        chat_template_kwargs: dict[str, Any],
    ) -> list[EncodedQuestion]:
        """Renders every question to scoring rows.

        Raises:
            DecisionRequestError: If a question cannot be rendered.
        """
        ...

    def scores(
        self,
        item: EncodedQuestion,
        results: Sequence[LabelScoringOutput],
        temperature: float | None,
    ) -> QuestionScores:
        """Option probabilities and label mass for one scored question."""
        ...


def detect_decision_format(model_path: str) -> DecisionFormat:
    """Picks the format for a checkpoint.

    Args:
        model_path: A local checkpoint directory or a Hub repo id.

    Returns:
        The decider format if the checkpoint ships ``decider_config.json``,
        else the chat format.
    """
    settings = find_decider_settings(model_path)
    if settings is None:
        return ChatFormat()
    return DeciderFormat(settings)
