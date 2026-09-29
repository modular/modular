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
"""Unit tests for the MMMU-Pro logit-shift gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from calibration.gpqa_gate import score_results
from calibration.mmmu_gate import (
    N_STD,
    expand_row_ids,
    load_mmmu_hist,
    merge_config_results,
    score_mmmu_spec,
    split_config_ids,
    subset_ids,
)
from calibration.mmmu_logit_gate import run_gate


def test_hist_and_subsets() -> None:
    hist = load_mmmu_hist()
    assert len(hist.q_acc) == 3460
    assert len(hist.q_trunc) == 3460
    assert len(hist.noisy_ids) == 57
    assert len(hist.plus_12_ids) == 69
    assert len(hist.ever2_ids) == 294
    assert set(hist.noisy_ids) <= set(hist.plus_12_ids)
    assert subset_ids(hist, "plus_12") == hist.plus_12_ids
    std, vis = split_config_ids(hist.plus_12_ids)
    assert len(std) == 45
    assert len(vis) == 24
    assert all(0 <= i < N_STD for i in std + vis)


def test_expand_row_ids_matches_repeat_order() -> None:
    assert expand_row_ids([3, 17], 3) == [3, 17, 3, 17, 3, 17]


def test_plus_12_designed_matches_calibration() -> None:
    scored = score_mmmu_spec("plus_12")
    assert scored.n_repeats == 104
    assert scored.cost == 7176
    assert scored.acc_cutoff == 5580
    assert scored.stop_cutoff == 4500
    assert scored.acc_h == pytest.approx(0.765)
    assert scored.stop_h == pytest.approx(0.979421965317919)


def test_stop_only_recounts_repeats() -> None:
    both = score_mmmu_spec("plus_12")
    stop_only = score_mmmu_spec("plus_12", want_acc=False)
    assert stop_only.n_repeats == 10
    assert stop_only.n_repeats < both.n_repeats
    assert stop_only.acc_cutoff is None
    assert stop_only.cost == 690


def test_need_a_metric() -> None:
    with pytest.raises(ValueError, match="need stop and/or accuracy"):
        score_mmmu_spec("plus_12", want_stop=False, want_acc=False)


def _write_mmmu_pair(
    tmp_path: Path,
    *,
    std_ids: list[int],
    vis_ids: list[int],
    n_repeats: int,
    n_trunc: int,
    n_wrong: int,
) -> Path:
    """Writes config jsonl shaped like ``mmmu_pro_eval`` and merges it."""
    rows: list[tuple[int, dict[str, object]]] = []
    for offset, ids in ((0, std_ids), (N_STD, vis_ids)):
        for _rep in range(n_repeats):
            for local in ids:
                rows.append(
                    (
                        offset + local,
                        {
                            "prompt_index": local,
                            "predicted": "A",
                            "ground_truth": "A",
                            "finish_reason": "stop",
                            "completion_tokens": 10,
                        },
                    )
                )
    for i in range(n_trunc):
        rows[i][1]["finish_reason"] = "length"
    for i in range(n_wrong):
        rows[i][1]["predicted"] = "B"
    std_path = tmp_path / "standard" / "results.jsonl"
    vis_path = tmp_path / "vision" / "results.jsonl"
    std_path.parent.mkdir(parents=True)
    vis_path.parent.mkdir(parents=True)
    std_lines = [json.dumps(row) for gid, row in rows if gid < N_STD]
    vis_lines = [json.dumps(row) for gid, row in rows if gid >= N_STD]
    std_path.write_text("\n".join(std_lines) + "\n")
    vis_path.write_text("\n".join(vis_lines) + "\n")
    return merge_config_results(std_path, vis_path, tmp_path / "results.jsonl")


def test_merge_and_score_results(tmp_path: Path) -> None:
    scored = score_mmmu_spec("plus_12", n_repeats=1, prompt_ids=[14, 40, 1892])
    results = _write_mmmu_pair(
        tmp_path,
        std_ids=[14, 40],
        vis_ids=[162],
        n_repeats=1,
        n_trunc=0,
        n_wrong=0,
    )
    live = score_results(results, scored)
    assert live.status == "pass"
    assert live.n_rows == 3


def test_smoke_overrides_repeats_and_ids(tmp_path: Path) -> None:
    results = _write_mmmu_pair(
        tmp_path / "eval",
        std_ids=[14],
        vis_ids=[162],
        n_repeats=1,
        n_trunc=0,
        n_wrong=0,
    )
    work = tmp_path / "smoke"
    run_gate(
        work_dir=work,
        subset="plus_12",
        results_jsonl=results,
        n_repeats=1,
        prompt_ids=[14, 1892],
        include_catalog=False,
    )
    selected = json.loads((work / "selected.json").read_text())
    assert selected["prompt_ids"] == [14, 1892]
    assert selected["n_repeats"] == 1
    assert selected["n_standard"] == 1
    assert selected["n_vision"] == 1
    report = (work / "REPORT.md").read_text()
    assert "# MMMU-Pro logit-gate report" in report
    assert "✅ pass" in report


def test_compare_only_report(tmp_path: Path) -> None:
    work = tmp_path / "gate"
    run_gate(
        work_dir=work,
        subset="plus_12",
        compare_only=True,
        include_catalog=False,
    )
    report = (work / "REPORT.md").read_text()
    assert "## Selected" in report
    assert "## Verdict" in report
    assert "`per_prompt:plus_12`" in report
    selected = json.loads((work / "selected.json").read_text())
    assert selected["n_repeats"] == 104
