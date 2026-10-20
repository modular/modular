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
"""Checks answer-format diagnostics against scorer failures and controls."""

import hashlib
import json

import exacto_gpqa_answer_audit as audit
import pytest


@pytest.mark.parametrize(
    ("text", "canonical", "grade", "missed", "credited", "explicit"),
    [
        ("Option B is wrong.\nAnswer: $A$", "B", "I", 1, 0, 1),
        ("Option C is wrong.\nAnswer: $A", "C", "I", 1, 0, 1),
        ("Answer: A", "A", "C", 0, 0, 1),
        ("Answer: $B$", "A", "C", 0, 1, 1),
        ("Answer: $B$", "C", "I", 0, 0, 1),
        ("Answer: $A\nThe answer is still uncertain.", None, "I", 0, 0, 0),
        ("Answer: $LETTER", None, "I", 0, 0, 0),
        ("Answer: A or B", None, "I", 0, 0, 0),
    ],
)
def test_final_answer_diagnostics(
    text: str,
    canonical: str | None,
    grade: str,
    missed: int,
    credited: int,
    explicit: int,
) -> None:
    raw = json.dumps(
        {
            "samples": [
                {
                    "id": 1,
                    "epoch": 1,
                    "target": "A",
                    "scores": {
                        "mcq_scorer": {"answer": canonical, "value": grade}
                    },
                    "output": {"choices": [{"message": {"content": text}}]},
                }
            ]
        }
    ).encode()
    report = audit.audit_log(raw)
    assert report["scored_samples"] == 1
    assert report["explicit_final_lines"] == explicit
    assert report["correct_final_marked_incorrect"] == missed
    assert report["incorrect_final_marked_correct"] == credited
    assert report["source_sha256"] == hashlib.sha256(raw).hexdigest()


def test_reasoning_blocks_are_not_final_answers() -> None:
    raw = b"""{"samples":[{"id":1,"epoch":1,"target":"A",
      "scores":{"mcq_scorer":{"answer":null,"value":"I"}},
      "output":{"choices":[{"message":{"content":[
        {"type":"reasoning","reasoning":"Answer: $A$"},
        {"type":"text","text":"No final answer."}]}}]}}]}"""
    report = audit.audit_log(raw)
    assert report["explicit_final_lines"] == 0
    assert report["disagreements"] == []


def test_missing_or_errored_samples_are_not_rescued() -> None:
    raw = b"""{"samples":[{"id":1,"epoch":1,"error":{"message":"failed"},
      "target":"A","scores":{"mcq_scorer":{"answer":null,"value":"I"}},
      "output":{"choices":[{"message":{"content":"Answer: $A"}}]}},
      {"id":2,"epoch":1}]}"""
    report = audit.audit_log(raw)
    assert report["selected_samples"] == 2
    assert report["scored_samples"] == 0
    assert report["correct_final_marked_incorrect"] == 0


def test_epoch_filter_and_duplicate_guard() -> None:
    raw = b'{"samples":[{"id":1,"epoch":1},{"id":1,"epoch":2}]}'
    assert audit.audit_log(raw, epoch=1)["selected_samples"] == 1
    duplicate = b'{"samples":[{"id":1,"epoch":1},{"id":1,"epoch":1}]}'
    with pytest.raises(ValueError, match="Duplicate sample"):
        audit.audit_log(duplicate)
