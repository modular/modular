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

"""Request and response schemas for ``POST /v1/decisions``.

The wire format follows SGLang's ``/v1/decisions`` so existing clients work
unchanged: typed questions go in, and one probability per candidate comes out,
read from the model's next-token scores at the answer position. No text is
generated.
"""

from __future__ import annotations

import string
import unicodedata
from collections.abc import Iterable
from typing import Annotated, Any, Literal

from max.serve.schemas.openai import CompletionUsage
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# TODO(PRDT-1041): Each label must be one token after the prompt, which caps a
# choice at 26 options and a score at 10 levels. Longer lists need
# multi-character labels, validated by ``resolve_label_token_ids``.
CHOICE_LABELS = string.ascii_uppercase
"""Choice options are labeled ``A`` to ``Z`` in order."""

SCORE_LABELS = string.digits
"""Score levels are labeled ``0`` to ``9`` in order."""

MAX_CHOICE_OPTIONS = len(CHOICE_LABELS)
MAX_SCORE_LEVELS = len(SCORE_LABELS)

QuestionKind = Literal["choice", "score", "yes_no"]

MAX_QUESTIONS = 64
"""Questions per request. Each question is admitted as its own scoring
request (several for a decider ``score`` question), so this bounds the work a
single request body can queue."""

# Objects and arrays are rendered into the prompt as compact JSON. Other JSON
# scalars (numbers, booleans, null) are not accepted.
DecisionText = str | dict[str, Any] | list[Any]


def is_blank_decision_text(value: Any) -> bool:
    """Returns whether ``value`` is empty text, or an empty object or array."""
    return not (value.strip() if isinstance(value, str) else value)


def _nonblank_decision_text(value: DecisionText) -> DecisionText:
    if is_blank_decision_text(value):
        raise ValueError("must not be blank")
    return value


RequiredDecisionText = Annotated[
    DecisionText, AfterValidator(_nonblank_decision_text)
]
_QuestionId = Annotated[str, AfterValidator(_nonblank_decision_text)]


def check_option_names(names: Iterable[str]) -> None:
    """Refuses option names that make the rendered option lines ambiguous.

    Raises:
        ValueError: If a name is blank, contains a control or line-break
            character, or repeats another name (ignoring case and padding).
    """
    seen: set[str] = set()
    for name in names:
        key = name.strip().casefold()
        if not key:
            raise ValueError("option names must be nonempty")
        # Each option is rendered as one prompt line.
        if any(unicodedata.category(c) in ("Cc", "Zl", "Zp") for c in name):
            raise ValueError(
                f"option name {name!r} must not contain control or line break "
                "characters"
            )
        if key in seen:
            raise ValueError(f"option name {name!r} repeats another option")
        seen.add(key)


class DecisionOption(BaseModel):
    """One candidate of a ``choice`` question."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: DecisionText | None = None


class DecisionChoiceQuestion(BaseModel):
    """Pick one of 2 to 26 named options, labeled ``A`` to ``Z`` in order."""

    model_config = ConfigDict(extra="forbid")

    id: _QuestionId
    type: Literal["choice"]
    question: RequiredDecisionText
    options: list[DecisionOption] = Field(
        min_length=2, max_length=MAX_CHOICE_OPTIONS
    )

    @model_validator(mode="after")
    def _option_names_distinct(self) -> DecisionChoiceQuestion:
        try:
            check_option_names(option.name for option in self.options)
        except ValueError as e:
            raise ValueError(f"question {self.id!r}: {e}") from None
        return self


class DecisionScoreQuestion(BaseModel):
    """Rate on 2 to 10 described levels, labeled ``0`` to ``9`` in order."""

    model_config = ConfigDict(extra="forbid")

    id: _QuestionId
    type: Literal["score"]
    question: RequiredDecisionText
    levels: list[RequiredDecisionText] = Field(
        min_length=2, max_length=MAX_SCORE_LEVELS
    )


class DecisionYesNoQuestion(BaseModel):
    """A yes/no question, answered by the labels ``yes`` and ``no``."""

    model_config = ConfigDict(extra="forbid")

    id: _QuestionId
    type: Literal["yes_no"]
    question: RequiredDecisionText
    yes: DecisionText | None = None
    no: DecisionText | None = None


DecisionQuestion = Annotated[
    DecisionChoiceQuestion | DecisionScoreQuestion | DecisionYesNoQuestion,
    Field(discriminator="type"),
]


class DecisionRequest(BaseModel):
    """Body of ``POST /v1/decisions``."""

    model_config = ConfigDict(extra="forbid")

    input: RequiredDecisionText
    """The text (a string, object or array) the questions are about."""
    questions: list[DecisionQuestion] = Field(
        min_length=1, max_length=MAX_QUESTIONS
    )
    temperature: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    """Scales option probabilities only, not ``label_mass``. Unset uses the
    model's own temperature: 1.0 for chat prompts, the calibrated values of
    ``decider_config.json`` for decider models."""
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Applied over the defaults, with the template's thinking toggle off."""
    prompt_format_version: int | None = None
    """Pins the server-owned prompt wording. A different version is refused."""
    return_prompt_token_ids: bool = False
    model: str = ""

    @field_validator("questions")
    @classmethod
    def _question_ids_distinct(
        cls, questions: list[DecisionQuestion]
    ) -> list[DecisionQuestion]:
        seen: set[str] = set()
        for question in questions:
            if question.id in seen:
                raise ValueError(
                    f"question id {question.id!r} repeats another question"
                )
            seen.add(question.id)
        return questions


class DecisionAnswer(BaseModel):
    """The answer to one question."""

    type: QuestionKind
    probabilities: dict[str, float]
    """Probability of each candidate, summing to 1 over the candidates."""
    label_mass: float
    """Full-vocabulary probability of all answer labels at the answer
    position. Near 1 when the model answers with a label; low when it wants to
    say something else."""
    choice: str | None = None
    """The most probable option name (``choice`` questions)."""
    score: float | None = None
    """Expected level under ``probabilities`` (``score`` questions)."""
    prompt_token_ids: list[int] | None = None
    """The scored prompt, when the request sets ``return_prompt_token_ids``."""
    label_token_ids: list[int] | None = None
    """The token id of each candidate in ``probabilities``, in order, when the
    request sets ``return_prompt_token_ids``."""


class DecisionResponse(BaseModel):
    """Response of ``POST /v1/decisions``."""

    object: Literal["decisions"] = "decisions"
    model: str
    prompt_format_version: int
    answers: dict[str, DecisionAnswer]
    usage: CompletionUsage
    """Nothing is generated, so ``completion_tokens`` is 0. Prefix-cache hits
    are ``prompt_tokens_details.cached_tokens``."""
