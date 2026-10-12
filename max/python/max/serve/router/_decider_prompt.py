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

"""The plain state-first prompt layout of decider models, as pure functions.

Decider models (for example ``Mapika/decider-2b``) are trained to read a
state and one question in a fixed plain-text layout and to answer with the
letter of an option::

    Context:
    <state>

    Question: <question>
    Options:
    (A) <option>
    (B) <option>
    Answer: (

The answer is the letter distribution at the final ``(``. A score question is
read with isolated levels: each level becomes its own yes/no question that
does not show the level's number or its neighbours, and the per-level
"fits" probabilities are normalized into a distribution over levels.

This module reproduces the reference implementation's text exactly (it is
what the models were trained on), so every string here is load bearing.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from max.serve.router._decisions_prompt import QuestionView
from max.serve.schemas._decisions import DecisionText

AnswerType = Literal["choice", "noul", "score"]

LETTERS = "ABCDEFGHIJ"
"""Option labels. More options need the reference's two-letter labels."""

MAX_DECIDER_OPTIONS = len(LETTERS)

PLAIN_LAYOUT = "plain"

DEFAULT_MAX_STATE_TOKENS = 32768

# A yes/no question with no instructions is carried by its true/false text.
NOUL_WITHOUT_INSTRUCTIONS = "Which answer fits the context?"

_ISOLATED_LEVEL = (
    "{question}\nProposed answer: {level}\nDoes the proposed answer fit?"
)

# Index of "yes" in an isolated level row, whose options are ["no", "yes"].
_YES = 1

# Index in a question view of the yes/no names ["yes", "no"].
_VIEW_YES, _VIEW_NO = 0, 1

# The minimum summed "fits" probability, so an all-zero row stays defined.
_MIN_FIT_MASS = 1e-9

_ARRAY_INDEX_MIN_LENGTH = 8
_LEVEL_NUMBER = re.compile(r"^\s*-?\d+\s*:\s*")


@dataclass(frozen=True)
class DeciderSettings:
    """What ``decider_config.json`` says about serving a decider model."""

    temperature: float
    """Divides the label logits for every answer type without its own value."""
    temperature_by_type: Mapping[AnswerType, float]
    max_state_tokens: int
    """The state is cut to this many tokens before the question is added."""
    isolated_levels: bool

    def temperature_for(self, answer_type: AnswerType) -> float:
        """The temperature for one answer type."""
        return self.temperature_by_type.get(answer_type, self.temperature)


def _positive(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{where} must be a number above 0, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{where} must be a number above 0, got {value!r}")
    return number


def parse_decider_config(config: Mapping[str, Any]) -> DeciderSettings:
    """Reads the serving settings out of a ``decider_config.json`` value.

    Raises:
        ValueError: If the layout is not the plain one (chat-layout models are
            not supported), or a temperature is not a positive number.
    """
    layout = config.get("layout", PLAIN_LAYOUT)
    if layout != PLAIN_LAYOUT or config.get("chat_template"):
        raise ValueError(
            f"decider_config.json layout {layout!r} is not supported; only "
            f"the {PLAIN_LAYOUT!r} layout is"
        )
    by_type_config = config.get("temperature_by_type") or {}
    if not isinstance(by_type_config, Mapping):
        raise ValueError("temperature_by_type must be a map")
    by_type: dict[AnswerType, float] = {}
    for key, value in by_type_config.items():
        if key not in ("choice", "noul", "score"):
            raise ValueError(
                f"temperature_by_type has the unknown key {key!r}; the keys "
                'are "choice", "noul" and "score"'
            )
        by_type[key] = _positive(value, f"temperature_by_type[{key!r}]")
    return DeciderSettings(
        temperature=_positive(config.get("temperature", 1.0), "temperature"),
        temperature_by_type=by_type,
        max_state_tokens=int(
            config.get("max_state_tokens", DEFAULT_MAX_STATE_TOKENS)
        ),
        isolated_levels=bool(config.get("isolated_levels", True)),
    )


def answer_type(view: QuestionView) -> AnswerType:
    """The temperature class of a question."""
    return "noul" if view.kind == "yes_no" else view.kind


def _text(value: DecisionText) -> str:
    """Strings as they are, other JSON values as ``json.dumps`` writes them."""
    return (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False)
    )


def annotate_indices(value: Any) -> Any:
    """Writes each element's position into long arrays.

    A path such as ``records[47].text`` otherwise makes the model count 47
    elements; with the index in the text it is a lookup.
    """
    if isinstance(value, list):
        if len(value) >= _ARRAY_INDEX_MIN_LENGTH:
            return [
                {"_index": i, **annotate_indices(item)}
                if isinstance(item, dict)
                else {"_index": i, "value": annotate_indices(item)}
                for i, item in enumerate(value)
            ]
        return [annotate_indices(item) for item in value]
    if isinstance(value, dict):
        return {key: annotate_indices(item) for key, item in value.items()}
    return value


def render_state(state: DecisionText) -> str:
    """The state as prompt text: strings verbatim, JSON with indexed arrays."""
    if isinstance(state, str):
        return state
    return json.dumps(annotate_indices(state), ensure_ascii=False)


@dataclass(frozen=True)
class DeciderRow:
    """One scored prompt: a question and the options its letters name."""

    question: str
    options: list[str]


def _strip_level_number(text: str) -> str:
    """``"2: somewhat"`` to ``"somewhat"``; an isolated level shows no number."""
    return _LEVEL_NUMBER.sub("", text)


def _is_unset(value: DecisionText | None) -> bool:
    """Whether the reference layout treats a description as absent."""
    return value is None or value == ""


def _question_text(view: QuestionView) -> str:
    if view.question is not None and not _is_unset(view.question):
        return _text(view.question)
    if view.kind == "yes_no" and not all(map(_is_unset, view.details)):
        return NOUL_WITHOUT_INSTRUCTIONS
    raise ValueError("the question has no instructions")


def _described(label: str, detail: DecisionText | None) -> str:
    if detail is None or _is_unset(detail):
        return label
    return f"{label}: {_text(detail)}"


def plan_rows(view: QuestionView, isolated_levels: bool) -> list[DeciderRow]:
    """The prompts that answer a question, in scoring order.

    A choice or yes/no question is one row. A score question is one yes/no row
    per level when levels are isolated, else one row listing every level.

    Raises:
        ValueError: If the question has no instructions, or has more options
            than there are letters.
    """
    question = _question_text(view)
    if view.kind == "choice":
        if len(view.names) > MAX_DECIDER_OPTIONS:
            raise ValueError(
                f"decider models read at most {MAX_DECIDER_OPTIONS} options "
                f"per choice question, got {len(view.names)}"
            )
        options = [
            _described(name, detail)
            for name, detail in zip(view.names, view.details, strict=True)
        ]
        return [DeciderRow(question, options)]
    if view.kind == "yes_no":
        yes, no = view.details[_VIEW_YES], view.details[_VIEW_NO]
        return [
            DeciderRow(question, [_described("no", no), _described("yes", yes)])
        ]
    assert all(level is not None for level in view.details), (
        "score levels are never absent"
    )
    levels = [_text(level) for level in view.details if level is not None]
    if len(levels) > MAX_DECIDER_OPTIONS:
        raise ValueError(
            f"a score takes 2 to {MAX_DECIDER_OPTIONS} levels in the decider "
            f"format, got {len(levels)}"
        )
    if isolated_levels:
        return [
            DeciderRow(
                _ISOLATED_LEVEL.format(
                    question=question, level=_strip_level_number(level)
                ),
                ["no", "yes"],
            )
            for level in levels
        ]
    return [
        DeciderRow(
            question, [f"{i}: {level}" for i, level in enumerate(levels)]
        )
    ]


def render_row(row: DeciderRow) -> str:
    """The text that follows the state, ending at the answer slot."""
    options = "".join(
        f"\n({LETTERS[i]}) {option}" for i, option in enumerate(row.options)
    )
    return f"\n\nQuestion: {row.question}\nOptions:{options}\nAnswer: ("


def state_prefix(state_text: str) -> str:
    """The state with its header; tokenized apart from the question."""
    return "Context:\n" + state_text


def combine_rows(
    view: QuestionView,
    row_probabilities: Sequence[Sequence[float]],
    isolated_levels: bool,
) -> list[float]:
    """Turns per-row letter probabilities into probabilities per question option.

    Returns:
        One probability per name of ``view``, in its order.
    """
    if view.kind == "choice":
        return list(row_probabilities[0])
    if view.kind == "yes_no":
        no, yes = row_probabilities[0]
        return [yes, no]
    if not isolated_levels:
        return list(row_probabilities[0])
    fits = [row[_YES] for row in row_probabilities]
    total = max(math.fsum(fits), _MIN_FIT_MASS)
    return [fit / total for fit in fits]
