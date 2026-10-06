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
"""Markdown report for the MMMU-Pro logit-shift gate."""

from __future__ import annotations

from pathlib import Path

from calibration.gpqa_gate import (
    DEFAULT_CAP,
    DEFAULT_DELTA_ACC,
    DEFAULT_DELTA_STOP,
    LiveVerdict,
    ScoredGate,
    rate_from_k,
)
from calibration.mmmu_gate import (
    N_STD,
    SUBSETS,
    MmmuHist,
    load_mmmu_hist,
    score_mmmu_spec,
)


def score_mmmu_catalog(
    hist: MmmuHist | None = None,
    *,
    alpha: float = DEFAULT_CAP,
    beta: float = DEFAULT_CAP,
    delta_stop: float = DEFAULT_DELTA_STOP,
    delta_acc: float = DEFAULT_DELTA_ACC,
    want_stop: bool = True,
    want_acc: bool = True,
) -> list[ScoredGate]:
    """Sizes every named MMMU-Pro subset."""
    data = hist or load_mmmu_hist()
    return [
        score_mmmu_spec(
            subset,
            data,
            alpha=alpha,
            beta=beta,
            delta_stop=delta_stop,
            delta_acc=delta_acc,
            want_stop=want_stop,
            want_acc=want_acc,
        )
        for subset in SUBSETS
    ]


def scored_to_dict(cfg: ScoredGate) -> dict[str, object]:
    """Serializes a scored spec for ``selected.json`` / ``catalog.json``."""
    spec = cfg.spec
    return {
        "name": spec.name,
        "hist_mode": spec.hist_mode,
        "subset": spec.subset,
        "prompt_ids": list(spec.prompt_ids),
        "n_subset": len(spec.prompt_ids),
        "n_repeats": cfg.n_repeats,
        "cost": cfg.cost,
        "alpha": spec.alpha,
        "beta": spec.beta,
        "delta_stop": spec.delta_stop,
        "delta_acc": spec.delta_acc,
        "want_stop": spec.want_stop,
        "want_acc": spec.want_acc,
        "stop_cutoff": cfg.stop_cutoff,
        "acc_cutoff": cfg.acc_cutoff,
        "stop_fp": cfg.stop_fp,
        "stop_fn": cfg.stop_fn,
        "acc_fp": cfg.acc_fp,
        "acc_fn": cfg.acc_fn,
        "stop_snr": cfg.stop_snr,
        "acc_snr": cfg.acc_snr,
        "base_stop": cfg.base_stop,
        "stop_h": cfg.stop_h,
        "stop_r": cfg.stop_r,
        "base_acc": cfg.base_acc,
        "acc_h": cfg.acc_h,
        "acc_r": cfg.acc_r,
        "n_standard": sum(1 for i in spec.prompt_ids if i < N_STD),
        "n_vision": sum(1 for i in spec.prompt_ids if i >= N_STD),
    }


def pct(rate: float | None) -> str:
    return "-" if rate is None else f"{100.0 * rate:.2f}%"


def _mark(status: str | None) -> str:
    if status == "pass":
        return "✅ pass"
    if status == "fail":
        return "❌ fail"
    if status == "error":
        return "⚠️ error"
    return "skipped"


def _k(value: int | None) -> str:
    return "-" if value is None else str(value)


def _md_table(
    headers: list[str],
    rows: list[list[str]],
    *,
    right: set[int] | None = None,
) -> list[str]:
    aligns = [
        "---:" if right and i in right else "---" for i in range(len(headers))
    ]
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
        *[("| " + " | ".join(row) + " |") for row in rows],
    ]


def write_report(
    work_dir: Path,
    rows: list[ScoredGate],
    selected: ScoredGate | None,
    live: LiveVerdict | None,
    *,
    model: str | None = None,
    base_url: str | None = None,
) -> Path:
    """Writes the job-summary markdown to ``work_dir/REPORT.md``."""
    acc_h = acc_r = stop_h = stop_r = acc_pass = stop_pass = None
    if selected is not None:
        acc_h, acc_r = selected.acc_h, selected.acc_r
        stop_h, stop_r = selected.stop_h, selected.stop_r
        acc_pass = rate_from_k(selected.acc_cutoff, selected.cost)
        stop_pass = rate_from_k(selected.stop_cutoff, selected.cost)
    if live is not None and live.status != "error":
        if live.acc_rate_cutoff is not None:
            acc_pass = live.acc_rate_cutoff
        if live.stop_rate_cutoff is not None:
            stop_pass = live.stop_rate_cutoff
    if live is None:
        headline = "compare-only"
        rationale = None
        acc_status = stop_status = None
        acc_k = selected.acc_cutoff if selected is not None else None
        stop_k = selected.stop_cutoff if selected is not None else None
    else:
        headline = _mark(live.status)
        rationale = live.rationale
        acc_status, stop_status = live.acc_status, live.stop_status
        acc_k = live.acc_cutoff
        stop_k = live.stop_cutoff
    endpoint = base_url or (live.base_url if live is not None else None)
    model_id = model or (live.model if live is not None else None)
    n_ids = len(selected.spec.prompt_ids) if selected is not None else 0
    v1_acc = v1_stop = None
    if selected is not None and n_ids:
        hist = load_mmmu_hist()
        v1_acc = sum(hist.q_acc[i] for i in selected.spec.prompt_ids) / n_ids
        v1_stop = (
            1.0 - sum(hist.q_trunc[i] for i in selected.spec.prompt_ids) / n_ids
        )
    if rows:
        catalog = rows
    elif selected is not None:
        catalog = [selected]
    else:
        catalog = score_mmmu_catalog()
    lines = [
        "# MMMU-Pro logit-gate report",
        "",
        f"- endpoint: `{endpoint or '(not set)'}`",
        f"- model: `{model_id or '(not set)'}`",
        "",
    ]
    if selected is not None:
        lines.extend(
            [
                "## Selected",
                "",
                f"- `{selected.spec.name}` n={n_ids} "
                f"m={selected.n_repeats} cost={selected.cost}",
                f"- stop k={_k(selected.stop_cutoff)} "
                f"acc k={_k(selected.acc_cutoff)}",
                "",
            ]
        )
    lines.extend(
        [
            "## Verdict",
            "",
            f"**{headline}**  (job fails if either enabled metric fails)",
            "",
            *([rationale, ""] if rationale else []),
            *_md_table(
                ["Metric", "Observed", "Pass threshold", "k", "S", "Verdict"],
                [
                    [
                        "Accuracy",
                        pct(live.acc_rate if live else None),
                        f"≥ {pct(acc_pass)}",
                        _k(acc_k),
                        "-" if live is None else str(live.n_wrong),
                        _mark(acc_status),
                    ],
                    [
                        "Stop ratio",
                        pct(live.stop_rate if live else None),
                        f"≥ {pct(stop_pass)}",
                        _k(stop_k),
                        "-" if live is None else str(live.n_trunc),
                        _mark(stop_status),
                    ],
                ],
                right={1, 2, 3, 4},
            ),
            "",
            "## Subset worlds",
            "",
            *_md_table(
                ["World", "Acc", "Stop ratio"],
                [
                    ["Healthy (H)", pct(acc_h), pct(stop_h)],
                    ["V1-train subset", pct(v1_acc), pct(v1_stop)],
                    ["Pass threshold", pct(acc_pass), pct(stop_pass)],
                    ["Regressed (R)", pct(acc_r), pct(stop_r)],
                ],
                right={1, 2},
            ),
            "",
            "## Configurations",
            "",
            *_md_table(
                ["Config", "n", "m", "Cost", "k_stop", "k_acc"],
                [
                    [
                        f"`{cfg.spec.name}`",
                        str(len(cfg.spec.prompt_ids)),
                        str(cfg.n_repeats),
                        f"**{cfg.cost}**",
                        _k(cfg.stop_cutoff),
                        _k(cfg.acc_cutoff),
                    ]
                    for cfg in catalog
                ],
                right={1, 2, 3, 4, 5},
            ),
            "",
        ]
    )
    path = work_dir / "REPORT.md"
    path.write_text("\n".join(lines) + "\n")
    return path
