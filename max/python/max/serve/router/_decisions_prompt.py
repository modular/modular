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

"""Prompt rendering, answer labels, and probability math for ``/v1/decisions``.

Everything here is pure (no server or model state) so it can be unit tested
directly. The only I/O is the tokenizer callable handed to
:func:`resolve_label_token_ids`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from max.serve.schemas._decisions import (
    CHOICE_LABELS,
    DecisionAnswer,
    DecisionChoiceQuestion,
    DecisionQuestion,
    DecisionScoreQuestion,
    DecisionText,
    QuestionKind,
    is_blank_decision_text,
)

EncodeFn = Callable[[str], Awaitable[Sequence[int]]]
"""Tokenizes text without adding special tokens."""


@dataclass(frozen=True)
class QuestionView:
    """A question as the renderer and scorer see it."""

    kind: QuestionKind
    question: DecisionText | None
    """The question text, or ``None`` when the request gives no wording."""
    names: list[str]
    """Option names, level indices, or ``yes`` and ``no``, in candidate order."""
    details: list[DecisionText | None]
    """Option descriptions, level texts, or the yes and no descriptions."""


def render_text(value: DecisionText | None) -> str:
    """Renders a text-or-JSON value as prompt text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def question_view(question: DecisionQuestion) -> QuestionView:
    """Normalizes a typed question into candidate names and descriptions."""
    if question.type == "choice":
        assert isinstance(question, DecisionChoiceQuestion)
        return QuestionView(
            kind="choice",
            question=question.question,
            names=[option.name for option in question.options],
            details=[option.description for option in question.options],
        )
    if question.type == "score":
        assert isinstance(question, DecisionScoreQuestion)
        return QuestionView(
            kind="score",
            question=question.question,
            names=[str(level) for level in range(len(question.levels))],
            details=list(question.levels),
        )
    return QuestionView(
        kind="yes_no",
        question=question.question,
        names=["yes", "no"],
        details=[question.yes, question.no],
    )


def default_labels(view: QuestionView) -> list[str]:
    """Single-token answer labels in candidate order.

    ``A`` to ``Z`` for choices, level indices for scores, ``yes`` and ``no``
    for yes/no questions.
    """
    if view.kind == "choice":
        return list(CHOICE_LABELS[: len(view.names)])
    return list(view.names)


def render_question(
    text: str, view: QuestionView, labels: Sequence[str]
) -> str:
    """Renders the user message for one question (prompt format version 1).

    Args:
        text: The rendered input the question is about.
        view: The question.
        labels: Answer labels, in candidate order.

    Returns:
        The input, a blank line, then the question, its candidates, and a
        closing instruction naming the form of the answer.
    """
    question_text = (
        ""
        if is_blank_decision_text(view.question)
        else render_text(view.question)
    )
    if view.kind == "choice":
        lines = [f"Question: {question_text}"] if question_text else []
        for label, name, description in zip(
            labels, view.names, view.details, strict=True
        ):
            detail = render_text(description)
            lines.append(
                f"{label}: {name} - {detail}" if detail else f"{label}: {name}"
            )
        lines.append("Answer with the letter of one option only.")
    elif view.kind == "score":
        lines = [f"Question: {question_text}"] if question_text else []
        lines += [
            f"{label}: {render_text(level)}"
            for label, level in zip(labels, view.details, strict=True)
        ]
        lines.append("Answer with the number of one level only.")
    else:
        lines = [
            f"Is the following true? {question_text}"
            if question_text
            else "Is the following true?"
        ]
        for label, description in zip(labels, view.details, strict=True):
            detail = render_text(description)
            if detail:
                lines.append(f"{label}: {detail}")
        lines.append("Answer with yes or no only.")
    return "\n".join([text, "", *lines])


async def resolve_label_token_ids(
    encode: EncodeFn,
    prompt_text: str,
    prompt_ids: Sequence[int],
    labels: Sequence[str],
) -> list[int]:
    """Checks that each label adds exactly one distinct token after the prompt.

    A label is a valid candidate only if tokenizing ``prompt_text + label``
    yields ``prompt_ids`` followed by a single new token, so the label really
    is one next-token candidate at the answer position.

    Args:
        encode: Tokenizes text without special tokens.
        prompt_text: The rendered chat prompt.
        prompt_ids: Token ids of ``prompt_text``.
        labels: Answer labels, in candidate order.

    Returns:
        The token id of each label, in order.

    Raises:
        ValueError: If a label is not exactly one distinct token.
    """
    label_ids: list[int] = []
    for label in labels:
        ids = list(await encode(prompt_text + label))
        if (
            len(ids) != len(prompt_ids) + 1
            or ids[:-1] != list(prompt_ids)
            or ids[-1] in label_ids
        ):
            raise ValueError(
                f"the answer label {label!r} is not one distinct token after "
                "the chat prompt for this tokenizer, so this model is not "
                "supported"
            )
        label_ids.append(ids[-1])
    return label_ids


def option_probabilities(
    label_log_probabilities: Sequence[float], temperature: float
) -> list[float]:
    """Softmax of the label log-probabilities divided by ``temperature``.

    The full-vocabulary normalizer is a constant across candidates, so it
    cancels: the result sums to 1 over the candidates.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if not label_log_probabilities:
        raise ValueError("need at least one label log-probability")
    scaled = [logprob / temperature for logprob in label_log_probabilities]
    peak = max(scaled)
    weights = [math.exp(value - peak) for value in scaled]
    total = math.fsum(weights)
    return [weight / total for weight in weights]


def label_mass(label_log_probabilities: Sequence[float]) -> float:
    """Full-vocabulary probability of all answer labels at the answer slot."""
    return math.fsum(math.exp(logprob) for logprob in label_log_probabilities)


@dataclass(frozen=True)
class QuestionScores:
    """What one scored question resolved to, shared by both answer shapes."""

    probabilities: list[float]
    """One per option name of the question, in order."""
    label_mass: float
    """Full-vocabulary probability of the answer labels; for a question of
    several rows, the lowest of its rows."""
    choice: str | None = None
    """The most probable option name (``choice`` questions)."""
    score: float | None = None
    """Expected level under ``probabilities`` (``score`` questions)."""


def resolve_scores(
    view: QuestionView,
    probabilities: Sequence[float],
    mass: float,
    question_id: str,
) -> QuestionScores:
    """Derives the choice or expected score of a question from probabilities.

    Args:
        view: The question.
        probabilities: One per option name of the question, in order.
        mass: The question's label mass.
        question_id: The question's id, for the error message.

    Returns:
        The probabilities with the choice (``choice`` questions) or expected
        level (``score`` questions) filled in.

    Raises:
        ValueError: If the number of probabilities differs from the number of
            option names.
        RuntimeError: If any value is not finite, which is a server fault.
    """
    if len(probabilities) != len(view.names):
        raise ValueError(
            f"{len(probabilities)} probabilities for {len(view.names)} "
            "candidates"
        )
    if not all(math.isfinite(value) for value in [*probabilities, mass]):
        raise RuntimeError(f"question {question_id!r} scored non-finite values")
    choice = None
    score = None
    if view.kind == "choice":
        choice = view.names[probabilities.index(max(probabilities))]
    elif view.kind == "score":
        score = math.fsum(
            level * probability
            for level, probability in enumerate(probabilities)
        )
    return QuestionScores(list(probabilities), mass, choice, score)


def build_answer(
    view: QuestionView,
    scores: QuestionScores,
    *,
    prompt_token_ids: Sequence[int] | None = None,
    label_token_ids: Sequence[int] | None = None,
) -> DecisionAnswer:
    """Builds the ``/v1/decisions`` answer for one scored question."""
    return DecisionAnswer(
        type=view.kind,
        probabilities=dict(zip(view.names, scores.probabilities, strict=True)),
        label_mass=scores.label_mass,
        choice=scores.choice,
        score=scores.score,
        prompt_token_ids=list(prompt_token_ids)
        if prompt_token_ids is not None
        else None,
        label_token_ids=list(label_token_ids)
        if label_token_ids is not None
        else None,
    )
