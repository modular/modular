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

"""Request and response schemas for ``POST /v1/systemone``.

The wire format follows the System One decision API (also served by SGLang):
a state, a map of ``noul`` (yes/no), ``choice`` and ``score`` questions keyed
by caller-chosen ids, and one answer per question id. Questions are answered
by the same scoring path as ``/v1/decisions``.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from max.serve.schemas._decisions import (
    MAX_CHOICE_OPTIONS,
    MAX_QUESTIONS,
    MAX_SCORE_LEVELS,
    DecisionText,
    RequiredDecisionText,
    check_option_names,
    is_blank_decision_text,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    field_validator,
    model_validator,
)

_DECISIONS_ONLY_FIELDS = (
    "temperature",
    "prompt_format_version",
    "return_prompt_token_ids",
)


class _Question(BaseModel):
    # A misspelled key would otherwise silently answer a different question.
    model_config = ConfigDict(extra="forbid")

    instructions: DecisionText | None = None
    """What to decide, in the caller's words."""


class SystemOneNoulCriteria(BaseModel):
    """What makes a ``noul`` question true or false."""

    model_config = ConfigDict(extra="forbid")

    true: DecisionText | None = None
    false: DecisionText | None = None


class SystemOneNoulQuestion(_Question):
    """A yes/no question. The answer is the probability of yes."""

    type: Literal["noul"]
    criteria: SystemOneNoulCriteria | None = None

    @model_validator(mode="after")
    def _asks_something(self) -> SystemOneNoulQuestion:
        criteria = self.criteria or SystemOneNoulCriteria()
        if all(
            is_blank_decision_text(value)
            for value in (self.instructions, criteria.true, criteria.false)
        ):
            raise ValueError(
                "a noul question needs instructions or a true or false "
                "description to decide on"
            )
        return self


class SystemOneChoiceQuestion(_Question):
    """Pick one of 1 to 26 options. ``criteria`` maps option name to detail."""

    type: Literal["choice"]
    criteria: dict[str, DecisionText | None] = Field(min_length=1)

    @field_validator("criteria")
    @classmethod
    def _option_names_distinct(
        cls, criteria: dict[str, DecisionText | None]
    ) -> dict[str, DecisionText | None]:
        # Worded so a client can tell a capacity limit from a malformed request.
        if len(criteria) > MAX_CHOICE_OPTIONS:
            raise ValueError(
                f"at most {MAX_CHOICE_OPTIONS} options per choice question, "
                f"got {len(criteria)}"
            )
        check_option_names(criteria)
        return criteria


class SystemOneScoreQuestion(_Question):
    """Rate on 1 to 10 described levels, scored from 0 in order."""

    type: Literal["score"]
    criteria: list[RequiredDecisionText] = Field(min_length=1)

    @field_validator("criteria")
    @classmethod
    def _levels_fit(cls, criteria: list[DecisionText]) -> list[DecisionText]:
        # Worded so a client can tell a capacity limit from a malformed request.
        if len(criteria) > MAX_SCORE_LEVELS:
            raise ValueError(
                f"a score takes 2 to {MAX_SCORE_LEVELS} levels, "
                f"got {len(criteria)}"
            )
        return criteria


SystemOneQuestion = Annotated[
    SystemOneNoulQuestion | SystemOneChoiceQuestion | SystemOneScoreQuestion,
    Field(discriminator="type"),
]


class SystemOneRequest(BaseModel):
    """Body of ``POST /v1/systemone``. Unknown top-level fields are ignored."""

    state: DecisionText
    """The text (a string, object or array) the questions are about.

    May be empty: a caller can put everything in the question instructions."""
    model: str
    questions: dict[str, SystemOneQuestion] = Field(
        min_length=1, max_length=MAX_QUESTIONS
    )
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Applied over the defaults, with the template's thinking toggle off."""

    @model_validator(mode="before")
    @classmethod
    def _refuse_decisions_fields(cls, data: Any) -> Any:
        # Honoring or ignoring these would change the answers, so they are
        # refused by name.
        if isinstance(data, dict):
            for field in _DECISIONS_ONLY_FIELDS:
                if data.get(field) is not None:
                    raise ValueError(
                        f"{field} is not part of this API, use /v1/decisions "
                        "for it"
                    )
        return data

    @field_validator("questions")
    @classmethod
    def _question_ids_nonblank(
        cls, questions: dict[str, SystemOneQuestion]
    ) -> dict[str, SystemOneQuestion]:
        for question_id in questions:
            if not question_id.strip():
                raise ValueError("question ids must not be blank")
        return questions


class SystemOneNoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float
    """Probability of yes."""
    x_label_mass: float
    """Full-vocabulary probability of the answer labels (an extension)."""


class SystemOneChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    confidence: float
    """How far the top option stands above a uniform guess, from 0 to 1."""
    probabilities: dict[str, float]
    x_label_mass: float


class SystemOneScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    """Expected level under ``probabilities``."""
    confidence: float
    """One minus the spread around the top level relative to uniform."""
    legend: dict[str, Any]
    """Each level index mapped to its description from the request."""
    probabilities: dict[str, float]
    x_label_mass: float


class SystemOneUsage(BaseModel):
    input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt = 0


class SystemOneResponse(BaseModel):
    """Response of ``POST /v1/systemone``."""

    model: str
    answers: dict[
        str, SystemOneNoulAnswer | SystemOneChoiceAnswer | SystemOneScoreAnswer
    ]
    usage: SystemOneUsage
