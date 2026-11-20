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
"""Audits explicit final answers without changing canonical GPQA scores."""

import argparse
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict

# The pinned prompt requests `Answer: $LETTER`; the canonical English scorer
# can miss that final line and select an option mentioned in the explanation.
FINAL_ANSWER = re.compile(
    r"(?:^|\n)[ \t]*Answer[ \t]*:[ \t]*\$?([ABCD])\$?[.!]?[ \t]*$",
    re.IGNORECASE,
)


class Disagreement(TypedDict):
    sample_id: int | str
    epoch: int
    canonical_answer: str | None
    canonical_grade: str
    explicit_final_answer: str
    target: str


class AnswerAudit(TypedDict):
    scope: str
    source_sha256: str
    epoch_filter: int | None
    selected_samples: int
    scored_samples: int
    explicit_final_lines: int
    disagreements: list[Disagreement]
    correct_final_marked_incorrect: int
    incorrect_final_marked_correct: int


def _mapping(value: object) -> Mapping[str, object]:
    assert isinstance(value, dict)
    assert all(isinstance(key, str) for key in value)
    return value


def _final_text(output: Mapping[str, object]) -> str:
    choices = output.get("choices") or []
    assert isinstance(choices, list)
    if not choices:
        return ""
    message = _mapping(_mapping(choices[0]).get("message") or {})
    content = message.get("content")
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    assert isinstance(content, list)
    pieces = []
    for part in content:
        block = _mapping(part)
        if block.get("type") == "text":
            text = block.get("text")
            assert isinstance(text, str)
            pieces.append(text)
    return "".join(pieces)


def audit_log(raw: bytes, epoch: int | None = None) -> AnswerAudit:
    """Finds disagreements using only an explicit last answer line.

    Structured reasoning blocks and answers embedded in an explanation are
    excluded. The audit neither replaces the scorer nor estimates accuracy
    from partial logs.
    """
    document = _mapping(json.loads(raw))
    samples = document.get("samples") or []
    assert isinstance(samples, list)
    disagreements: list[Disagreement] = []
    seen: set[tuple[int | str, int]] = set()
    selected = scored = explicit = 0
    for item in samples:
        sample = _mapping(item)
        sample_id, sample_epoch = sample.get("id"), sample.get("epoch")
        assert isinstance(sample_id, (int, str))
        assert isinstance(sample_epoch, int)
        key = (sample_id, sample_epoch)
        if key in seen:
            raise ValueError(f"Duplicate sample: {key}")
        seen.add(key)
        if epoch is not None and sample_epoch != epoch:
            continue
        selected += 1
        scores = _mapping(sample.get("scores") or {})
        score = _mapping(scores.get("mcq_scorer") or {})
        grade, target = score.get("value"), sample.get("target")
        if sample.get("error") or grade not in ("C", "I"):
            continue
        assert isinstance(grade, str) and isinstance(target, str)
        scored += 1
        text = _final_text(_mapping(sample.get("output") or {}))
        match = FINAL_ANSWER.search(text.rstrip())
        if not match:
            continue
        explicit += 1
        answer = match.group(1).upper()
        canonical = score.get("answer")
        assert canonical is None or isinstance(canonical, str)
        if answer != canonical:
            disagreements.append(
                Disagreement(
                    sample_id=sample_id,
                    epoch=sample_epoch,
                    canonical_answer=canonical,
                    canonical_grade=grade,
                    explicit_final_answer=answer,
                    target=target,
                )
            )
    return AnswerAudit(
        scope="Final-answer format diagnostic; canonical grades unchanged. "
        "No corrected accuracy or equivalence claim.",
        source_sha256=hashlib.sha256(raw).hexdigest(),
        epoch_filter=epoch,
        selected_samples=selected,
        scored_samples=scored,
        explicit_final_lines=explicit,
        disagreements=disagreements,
        correct_final_marked_incorrect=sum(
            row["canonical_grade"] == "I"
            and row["explicit_final_answer"] == row["target"]
            for row in disagreements
        ),
        incorrect_final_marked_correct=sum(
            row["canonical_grade"] == "C"
            and row["explicit_final_answer"] != row["target"]
            for row in disagreements
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epoch", type=int)
    args = parser.parse_args()
    report = audit_log(args.log.read_bytes(), args.epoch)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
