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
"""Tests for the shared AIME eval binary.

The driver is shared between AIME25 and AIME26; these tests cover the pieces
that make that shape safe: the pure grader, the dataset-id parameterisation on
:func:`aime_eval.load_aime_dataset`, and the ``train``/``test`` split
fallback.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import aime_eval
import pytest
from click.testing import CliRunner


@pytest.mark.parametrize(
    ("content", "expected", "want_correct", "want_predicted"),
    [
        # Canonical \\boxed{} answer wins over trailing recap integers.
        (
            r"Working... the answer is \boxed{42}. Total was 7+35=42.",
            42,
            True,
            "42",
        ),
        # Zero-padded literals compare integer-valued.
        (r"\boxed{042}", "42", True, "042"),
        # No \\boxed — fall back to the last standalone integer.
        ("The answer is 123", 123, True, "123"),
        # Wrong answer still scores wrong.
        (r"\boxed{7}", 42, False, "7"),
        # Nothing parseable yields an empty prediction and incorrect verdict.
        ("I don't know.", 42, False, ""),
        # ``None`` content (truncated-mid-reasoning backend) must not raise.
        (None, 42, False, ""),
        # <think> block is stripped before extraction, so a stray integer
        # inside the thought cannot shadow the boxed answer below it.
        (
            r"<think>Let me try 7</think>\boxed{42}",
            42,
            True,
            "42",
        ),
    ],
)
def test_grade(
    content: str | None,
    expected: object,
    want_correct: bool,
    want_predicted: str,
) -> None:
    correct, predicted = aime_eval.grade(content, expected)
    assert (correct, predicted) == (want_correct, want_predicted)


def test_default_dataset_is_aime25() -> None:
    """The default is kept at ``math-ai/aime25`` so existing callers continue
    to run AIME25 unchanged after the dataset-id parameterisation."""
    assert aime_eval.DEFAULT_DATASET == "math-ai/aime25"


@pytest.mark.parametrize(
    "dataset_id, want_prefix, want_out_dir",
    [
        ("math-ai/aime25", "AIME25", "/tmp/aime25-results"),
        ("math-ai/aime26", "AIME26", "/tmp/aime26-results"),
        ("aime26", "AIME26", "/tmp/aime26-results"),
        ("org/aime-2027", "AIME_2027", "/tmp/aime-2027-results"),
    ],
)
def test_dataset_defaults(
    dataset_id: str, want_prefix: str, want_out_dir: str
) -> None:
    assert aime_eval.dataset_defaults(dataset_id) == (
        want_prefix,
        want_out_dir,
    )


def _run_main(args: list[str]) -> tuple[str, str]:
    """Invokes the CLI with I/O stubbed and returns the (out_dir, prefix)
    that reached :func:`eval_common.write_outputs`."""
    written: list[tuple[str, str]] = []

    def fake_write_outputs(
        out_dir: str, results: object, summary: object, metric_prefix: str
    ) -> None:
        written.append((out_dir, metric_prefix))

    with (
        patch.object(aime_eval, "make_client"),
        patch.object(aime_eval, "load_aime_dataset", return_value=[]),
        patch.object(aime_eval, "run_eval", return_value=([], {})),
        patch.object(aime_eval, "write_outputs", fake_write_outputs),
    ):
        result = CliRunner().invoke(
            aime_eval.main,
            ["--base-url", "http://x", "--model", "m", *args],
            catch_exceptions=False,
        )
    assert result.exit_code == 0, result.output
    assert len(written) == 1
    return written[0]


def test_main_defaults_to_aime25_outputs() -> None:
    assert _run_main([]) == ("/tmp/aime25-results", "AIME25")


def test_main_derives_outputs_from_dataset() -> None:
    """``--dataset`` alone must not write AIME26 into AIME25's out-dir or
    export it under AIME25_* keys."""
    assert _run_main(["--dataset", "math-ai/aime26"]) == (
        "/tmp/aime26-results",
        "AIME26",
    )


def test_main_explicit_flags_override_derived_defaults() -> None:
    assert _run_main(
        [
            "--dataset",
            "math-ai/aime26",
            "--metric-prefix",
            "CUSTOM",
            "--out-dir",
            "/tmp/custom",
        ]
    ) == ("/tmp/custom", "CUSTOM")


def test_load_aime_dataset_forwards_the_requested_id() -> None:
    """``--dataset math-ai/aime26`` must reach the HF loader, not fall back
    to a hard-coded ``math-ai/aime25``.
    """
    recorded: list[tuple[str, str]] = []

    def fake_load(dataset_id: str, split: str) -> list[dict[str, Any]]:
        recorded.append((dataset_id, split))
        return [{"problem": "p", "answer": 1}]

    with patch.object(aime_eval, "load_dataset", side_effect=fake_load):
        rows = aime_eval.load_aime_dataset("math-ai/aime26")
    assert rows == [{"problem": "p", "answer": 1}]
    assert recorded == [("math-ai/aime26", "test")]


def test_load_aime_dataset_falls_back_to_train_split() -> None:
    """A sibling that only publishes ``train`` is loaded without the caller
    needing to know which split it ships.
    """
    recorded: list[tuple[str, str]] = []

    def fake_load(dataset_id: str, split: str) -> list[dict[str, Any]]:
        recorded.append((dataset_id, split))
        if split == "test":
            raise ValueError("no test split")
        return [{"problem": "p", "answer": 2}]

    with patch.object(aime_eval, "load_dataset", side_effect=fake_load):
        rows = aime_eval.load_aime_dataset("math-ai/aime26")
    assert rows == [{"problem": "p", "answer": 2}]
    assert recorded == [
        ("math-ai/aime26", "test"),
        ("math-ai/aime26", "train"),
    ]


def test_build_messages_omits_root_preamble_when_empty() -> None:
    msgs = aime_eval.build_messages("Compute 2+2.", "", "You are helpful.")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == "You are helpful."
    assert "Compute 2+2." in msgs[1]["content"]
    assert msgs[1]["content"].startswith(
        r"Return your final response within \boxed{}. "
    )


def test_build_messages_includes_root_preamble_when_set() -> None:
    msgs = aime_eval.build_messages(
        "Compute 2+2.", "I am MiniMax-M3.", "You are helpful."
    )
    assert [m["role"] for m in msgs] == ["root", "system", "user"]
    assert msgs[0]["content"] == "I am MiniMax-M3."
