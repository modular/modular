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
"""Tests for the shared eval scaffolding."""

import json
from pathlib import Path
from typing import Any

import eval_common
import pytest

# An endpoint that rejects every request: the score is 0.0 not because the model
# was wrong but because nothing was measured.
ALL_ERRORED = {"accuracy": 0.0, "correct": 0, "total": 120, "errors": 120}

# A real AIME25 run whose only failures were gateway timeouts on the two
# longest-generating problems. Well inside the budget; must still score.
GATEWAY_TIMEOUTS = {
    "accuracy": 0.9021,
    "correct": 433,
    "total": 480,
    "errors": 3,
}


def test_error_budget_rejects_an_all_errored_run() -> None:
    with pytest.raises(SystemExit):
        eval_common.enforce_error_budget(ALL_ERRORED)


def test_error_budget_allows_incidental_errors() -> None:
    eval_common.enforce_error_budget(GATEWAY_TIMEOUTS)


@pytest.mark.parametrize(
    ("errors", "rejected"),
    [(9, False), (10, False), (11, True)],
)
def test_error_budget_boundary(errors: int, rejected: bool) -> None:
    summary = {"total": 100, "errors": errors}
    if rejected:
        with pytest.raises(SystemExit):
            eval_common.enforce_error_budget(summary)
    else:
        eval_common.enforce_error_budget(summary)


def test_error_budget_reads_the_judged_eval_key() -> None:
    """AA-Omniscience names its error count ``errored``, not ``errors``."""
    with pytest.raises(SystemExit):
        eval_common.enforce_error_budget({"total": 600, "errored": 600})


@pytest.mark.parametrize(
    "summary",
    [
        {"total": 600},  # no error count reported
        {"accuracy": 0.4},  # no total either (SciCode-style)
        {"total": 0, "errors": 0},  # nothing submitted
    ],
)
def test_error_budget_skips_summaries_it_cannot_judge(
    summary: dict[str, Any],
) -> None:
    eval_common.enforce_error_budget(summary)


def test_dump_score_writes_before_rejecting(tmp_path: Path) -> None:
    """A rejected run still leaves score.json behind to diagnose from."""
    out = tmp_path / "results"
    with pytest.raises(SystemExit):
        eval_common.dump_score(str(out), ALL_ERRORED)
    written = json.loads((out / "score.json").read_text())
    assert written == ALL_ERRORED


def test_gated_dataset_denial_fails_the_step() -> None:
    """A lane that cannot load its dataset must not report success."""

    def denied() -> list[dict[str, Any]]:
        raise RuntimeError("403 Client Error: gated repo, enable access")

    with pytest.raises(SystemExit) as excinfo:
        eval_common.load_gated(denied, label="GPQA", dataset_id="some/dataset")
    assert excinfo.value.code != 0


def test_gated_loader_passes_rows_through() -> None:
    rows = [{"q": 1}]
    assert (
        eval_common.load_gated(
            lambda: rows, label="GPQA", dataset_id="some/dataset"
        )
        is rows
    )


def test_gated_loader_propagates_unrelated_errors() -> None:
    """Only access denials are translated; real bugs keep their traceback."""

    def broken() -> list[dict[str, Any]]:
        raise ValueError("malformed row")

    with pytest.raises(ValueError):
        eval_common.load_gated(broken, label="GPQA", dataset_id="some/dataset")
