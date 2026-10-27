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
"""OpenRouter's multiple-choice answer extraction, ported from TypeScript.

A port of ``src/benchmarks/scorers/mcq/extract.ts`` and ``scorer.ts`` from
OpenRouter's benchmark harness at :data:`HARNESS_COMMIT`. The harness's
extractor started as a port of openbench's, but they have since drifted: the
harness accepts ``Answer: $C``, and its multilingual patterns take A-J rather
than A-D. A benchmark with more than four options, such as VGI-Bench, needs the
harness's version.

The patterns are JavaScript regular expressions, which differ from Python's in
ways that change what matches, so each is translated rather than copied:

* ``\\s`` is JavaScript's whitespace set, which has U+FEFF and lacks
  U+001C-U+001F and U+0085, unlike ``str.isspace``.
* The English patterns are ASCII-only for ``\\b`` and case folding, as
  JavaScript is without the ``u`` flag.
* ``^`` and ``$`` under the ``m`` flag also break at ``\\r``, U+2028 and U+2029.
* JavaScript never case-folds a non-ASCII character onto an ASCII one, so the
  four characters Python folds that way are hidden from the multilingual
  patterns.

Stdlib only, so it unit-tests under bazel and imports into the harness venv.
"""

# The patterns and tables match full-width letters and colons and en dashes on
# purpose, copied character for character from the harness.
# ruff: noqa: RUF001

from __future__ import annotations

import re
from dataclasses import dataclass

#: The harness commit this module was ported from.
HARNESS_COMMIT = "ccab6a7988c181f396331ae46474a8e7e18eaf80"

_ASCII_UPPERCASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# JavaScript's WhiteSpace and LineTerminator code points.
_JS_WS = (
    "[\t\n\v\f\r \u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff]"
)
_JS_LINE_START = "(?:\\A|(?<=[\n\r\u2028\u2029]))"
_JS_LINE_END = "(?=[\n\r\u2028\u2029]|\\Z)"

# Characters Python's Unicode case folding equates with an ASCII letter and
# JavaScript's non-unicode folding does not: dotted and dotless I, long s and
# the Kelvin sign.
_NON_ASCII_FOLDS = str.maketrans(dict.fromkeys("\u0130\u0131\u017f\u212a", 0))


def _js(source: str) -> str:
    return source.replace(r"\s", _JS_WS)


_I = re.IGNORECASE | re.ASCII

# In priority order, as MCQ_PATTERNS in extract.ts. The second and last are
# the two that anchor on lines.
_MCQ_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        _js(
            r"(?:\*{1,2}|_{1,2})Answer[s]?\s*[:\-–]?(?:\*{1,2}|_{1,2})\s*\$?\s*([A-Z])\b"
        ),
        _I,
    ),
    re.compile(
        _JS_LINE_START
        + _js(
            r"\s*(?:\*{1,2}|_{1,2})?Answer:?(?:\*{1,2}|_{1,2})?\s*:?\s*(?:\*{1,2}|_{1,2})?\$?\s*([A-Z])(?:\*{1,2}|_{1,2})?\s*"
        ),
        _I,
    ),
    re.compile(_js(r"\bAnswer[s]?\b\s*[:\-–]?\s*\$?\s*\(\s*([A-Z])\s*\)"), _I),
    re.compile(_js(r"\bAnswer[s]?\b\s*[:\-–]?\s*\$?\s*([A-Z])\b"), _I),
    re.compile(_js(r"\b(?:Option|Choice)\b\s*[:\-–]?\s*([A-Z])\b"), _I),
    re.compile(r"\\boxed\{[^}]*?([A-Z])[^}]*\}", re.ASCII),
    re.compile(
        r"\\boxed\{[^}]*?\\textbf\{[^}]*?([A-Z])[^}]*\}[^}]*\}", re.ASCII
    ),
    re.compile(r"\\boxed\{[^}]*?\\text\{[^}]*?([A-Z])[^}]*\}[^}]*\}", re.ASCII),
    re.compile(
        _js(r"(?<![A-Za-z0-9])[(\[]\s*([A-Z])\s*[)\]](?![A-Za-z0-9])"),
        re.ASCII,
    ),
    re.compile(
        r"(?<![A-Za-z0-9])(?:\*{1,2}|_{1,2})([A-Z])(?:\*{1,2}|_{1,2})(?![A-Za-z0-9])",
        re.ASCII,
    ),
    re.compile(r"\\textbf\{[^}]*?([A-Z])[^}]*\}", re.ASCII),
    re.compile(
        r"(?<![A-Za-z0-9])(?:\*{1,2}|_{1,2})\s*([A-Z])\)[^*_\n]+?(?:\*{1,2}|_{1,2})(?![A-Za-z0-9])".replace(
            r"\s", _JS_WS
        ),
        re.ASCII,
    ),
    re.compile(
        _JS_LINE_START
        + _js(
            r"\s*(?:\*{1,2}|_{1,2})?([A-Z])(?:\*{1,2}|_{1,2})?\s*[.)\-–:]?\s*"
        )
        + _JS_LINE_END,
        re.ASCII,
    ),
)

#: MULTILINGUAL_ANSWER_REGEXES in extract.ts, in order. Duplicates are kept:
#: each one is its own priority slot.
MULTILINGUAL_ANSWER_REGEXES: tuple[str, ...] = (
    r"Answer\s*:",
    "Answer\\s*:\u200b\u200b\u200b\u200b\u200b\u200b",
    r"উত্তর\s*:",
    r"उत्तर\s*:",
    r"উত্তরঃ",
    r"উত্তর\s*:",
    r"Antwort\s*:",
    r"답변\s*:",
    r"정답\s*:",
    r"답\s*:",
    r"答案\s*：",
    r"答案\s*:",
    r"答\s*：",
    r"答\s*:",
    r"答复\s*：",
    r"答曰\s*：",
    r"الإجابة:",
    r"الجواب:",
    r"إجابة:",
    r"الإجابة النهائية:",
    r"الإجابة الصحيحة:",
    r"الإجابة الصحيحة هي:",
    r"الإجابة هي:",
    r"الجواب النهائي:",
    r"Respuesta\s*:",
    r"Risposta\s*:",
    r"答え\s*:",
    r"答え\s*：",
    r"回答\s*:",
    r"回答\s*：",
    r"解答\s*:",
    r"Jawaban\s*:",
    r"Javob\s*:",
    r"Жавоб\s*:",
    r"Cevap\s*:",
    r"Джевап\s*:",
    r"Җавап\s*:",
    r"Жауап\s*:",
    r"Jawap\s*:",
    r"Juwap\s*:",
    r"جاۋاب\:",
    r"Cavab\s*:",
    r"Réponse\s*:",
    r"Resposta\s*:",
    r"Jibu\s*:",
    r"Idahun\s*:",
)

_MULTILINGUAL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(
        _js(answer + r"[ \t]*([A-J]|[أ-د]|[অ]|[ব]|[ড]|[ঢ]|[Ａ-Ｊ])"),
        re.IGNORECASE,
    )
    for answer in MULTILINGUAL_ANSWER_REGEXES
)

_MD_LATEX_TOKENS = (
    "**",
    "$\\boxed{",
    "}$",
    "\\$",
    "$\\text{",
    "$",
    "\\mathrm{",
    "\\{",
    "\\text",
    "\\(",
    "\\mathbf{",
    "{",
    "\\boxed",
)

_MCQ_LETTER_MAP = (
    ("أ", " A"),
    ("ب", " B"),
    ("ج", " C"),
    ("د", " D"),
    ("অ", " A"),
    ("ব", " B"),
    ("ড", " C"),
    ("ঢ", " D"),
    ("Ａ", " A"),
    ("Ｂ", " B"),
    ("Ｃ", " C"),
    ("Ｄ", " D"),
    ("Ｅ", " E"),
    ("Ｆ", " F"),
    ("Ｇ", " G"),
    ("Ｈ", " H"),
    ("Ｉ", " I"),
    ("Ｊ", " J"),
)


def strip_md_latex(response: str) -> str:
    """Removes the Markdown and LaTeX wrappers ``stripMdLatex`` removes."""
    for token in _MD_LATEX_TOKENS:
        response = response.replace(token, "")
    return response


def normalize_mcq_answer(extracted: str) -> str:
    """Maps Arabic, Bengali and full-width letters to Latin, as the harness."""
    for source, latin in _MCQ_LETTER_MAP:
        extracted = extracted.replace(source, latin)
    return extracted.strip()


def extract_mcq_answer(text: str | None) -> str | None:
    """Returns the answer letter the harness would extract, or ``None``.

    Every pattern is tried; the highest-priority match wins, the longer match
    breaking a tie, as in ``extractMcqAnswer``.

    Args:
        text: The completion's text, without its reasoning.

    Returns:
        One uppercase ASCII letter, or ``None`` when no pattern matches.
    """
    if not text:
        return None
    cleaned = strip_md_latex(text).translate(_NON_ASCII_FOLDS)
    candidates: list[tuple[int, int, str]] = []
    for priority, pattern in enumerate(_MCQ_PATTERNS):
        match = pattern.search(text)
        if match and match.group(1):
            letter = match.group(1).upper()
            if len(letter) == 1 and letter in _ASCII_UPPERCASE:
                candidates.append((priority, -len(match.group(0)), letter))
    for index, pattern in enumerate(_MULTILINGUAL_PATTERNS):
        match = pattern.search(cleaned)
        if match and match.group(1):
            letter = normalize_mcq_answer(match.group(1)).upper()
            if len(letter) == 1 and letter in _ASCII_UPPERCASE:
                candidates.append(
                    (len(_MCQ_PATTERNS) + index, -len(match.group(0)), letter)
                )
    if not candidates:
        return None
    return min(candidates)[2]


@dataclass(frozen=True)
class McqScore:
    """One response scored the way the harness's ``mcqScorer`` scores it."""

    correct: bool
    answer: str | None
    explanation: str


def score_mcq(completion: str | None, target: str) -> McqScore:
    """Scores a completion against the target letter, as ``mcqScorer``.

    Args:
        completion: The response text, or ``None`` when there was no output.
        target: The correct letter.

    Returns:
        The verdict, the extracted letter and the harness's explanation text.
    """
    extracted = extract_mcq_answer(completion)
    target_answer = target.strip().upper()
    return McqScore(
        correct=extracted is not None and extracted == target_answer,
        answer=extracted,
        explanation=(
            f"Extracted '{extracted}' from response, target was "
            f"'{target_answer}'"
            if extracted
            else "No answer found"
        ),
    )
