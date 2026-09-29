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
"""MMMU-Pro logit-shift gate: histogram, subsets, and result shaping."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from calibration.gpqa_gate import (
    DEFAULT_CAP,
    DEFAULT_DELTA_ACC,
    DEFAULT_DELTA_STOP,
    GateSpec,
    ScoredGate,
    apply_shift,
    evaluate_repeats,
    shift_for_mean,
    snr,
    solve_repeats,
)

N_STD = 1730
N_PROMPTS = 3460
TRUNC_TOKENS = 65534
SUBSETS = ("noisy", "plus_12", "ever2")
_HIST = Path(__file__).resolve().parent / "data" / "mmmu_pro_gate_hist.json"


@dataclass(frozen=True)
class MmmuHist:
    """V1-train per-prompt rates plus the designed subset id lists."""

    q_acc: list[float]
    q_trunc: list[float]
    noisy_ids: list[int]
    plus_12_ids: list[int]
    ever2_ids: list[int]
    park_acc: float
    park_stop: float


def load_mmmu_hist(path: Path | None = None) -> MmmuHist:
    """Loads the committed MMMU-Pro gate histogram."""
    payload = json.loads((path or _HIST).read_text())
    q_acc = [float(x) for x in payload["q_acc"]]
    q_trunc = [float(x) for x in payload["q_trunc"]]
    assert len(q_acc) == N_PROMPTS
    assert len(q_trunc) == N_PROMPTS
    return MmmuHist(
        q_acc=q_acc,
        q_trunc=q_trunc,
        noisy_ids=[int(i) for i in payload["noisy_ids"]],
        plus_12_ids=[int(i) for i in payload["plus_12_ids"]],
        ever2_ids=[int(i) for i in payload["ever2_ids"]],
        park_acc=float(payload["park_acc"]),
        park_stop=float(payload["park_stop"]),
    )


def subset_ids(hist: MmmuHist, subset: str) -> list[int]:
    """Returns the designed prompt ids for ``subset``."""
    if subset == "noisy":
        return list(hist.noisy_ids)
    if subset == "plus_12":
        return list(hist.plus_12_ids)
    if subset == "ever2":
        return list(hist.ever2_ids)
    raise KeyError(f"unknown subset {subset!r}")


def split_config_ids(prompt_ids: list[int]) -> tuple[list[int], list[int]]:
    """Splits global ids into standard and vision-local row indexes.

    Global ids ``0..1729`` are standard; ``1730..3459`` are vision with
    local index ``id - 1730``.
    """
    std = [i for i in prompt_ids if i < N_STD]
    vis = [i - N_STD for i in prompt_ids if i >= N_STD]
    if any(i < 0 or i >= N_STD for i in std + vis):
        raise ValueError(f"prompt ids must be in 0..{N_PROMPTS - 1}")
    return std, vis


def expand_row_ids(ids: list[int], n_repeats: int) -> list[int]:
    """Repeats ``ids`` ``n_repeats`` times, matching :func:`expand_repeats`."""
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")
    return [i for _ in range(n_repeats) for i in ids]


def is_trunc(row: dict[str, object]) -> bool:
    """Returns whether one MMMU-Pro row counted as a truncation."""
    if row.get("finish_reason") == "length":
        return True
    if row.get("finish_reason") == "stop":
        return False
    tokens = row.get("completion_tokens") or 0
    assert isinstance(tokens, int)
    return tokens >= TRUNC_TOKENS


def normalize_mmmu_row(
    row: dict[str, object], offset: int
) -> dict[str, object]:
    """Maps one config row onto the GPQA live-verdict schema."""
    idx = row["prompt_index"]
    assert isinstance(idx, int)
    out = dict(row)
    out["prompt_index"] = offset + idx
    if "error" in row:
        return out
    out["correct"] = row.get("predicted") == row.get("ground_truth")
    if is_trunc(row):
        out["finish_reason"] = "length"
    return out


def merge_config_results(
    standard_path: Path, vision_path: Path, dest: Path
) -> Path:
    """Writes a single jsonl that :func:`score_results` can consume."""
    lines: list[str] = []
    for path, offset in ((standard_path, 0), (vision_path, N_STD)):
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            lines.append(
                json.dumps(normalize_mmmu_row(json.loads(line), offset))
            )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + ("\n" if lines else ""))
    return dest


def _coins(
    hist: MmmuHist, ids: list[int], delta_acc: float, delta_stop: float
) -> dict[str, list[float]]:
    """Builds H/R coins the same way the offline MMMU sizer does.

    H coins are raw V1-train rates. R coins shift the full V1-train hist
    to ``park - delta`` and then take the subset.
    """
    q_acc, q_trunc = hist.q_acc, hist.q_trunc
    acc_r = apply_shift(q_acc, shift_for_mean(q_acc, hist.park_acc - delta_acc))
    tr_r = apply_shift(
        q_trunc, shift_for_mean(q_trunc, 1.0 - (hist.park_stop - delta_stop))
    )
    return {
        "ah": [1.0 - q_acc[i] for i in ids],
        "ar": [1.0 - acc_r[i] for i in ids],
        "th": [q_trunc[i] for i in ids],
        "tr": [tr_r[i] for i in ids],
    }


def _solve_metric(
    qs_h: list[float],
    qs_r: list[float],
    n_repeats: int | None,
    alpha: float,
    beta: float,
    msg: str,
) -> tuple[int, int, float, float]:
    if n_repeats is None:
        solved = solve_repeats(qs_h, qs_r, alpha, beta)
        if solved is None:
            raise ValueError(msg)
        return (
            solved.n_repeats,
            solved.cutoff,
            solved.false_positive,
            solved.false_negative,
        )
    found = evaluate_repeats(qs_h, qs_r, n_repeats, alpha)
    if found is None:
        raise ValueError(msg)
    return n_repeats, found[0], found[1], found[2]


def score_mmmu_spec(
    subset: str,
    hist: MmmuHist | None = None,
    *,
    alpha: float = DEFAULT_CAP,
    beta: float = DEFAULT_CAP,
    delta_stop: float = DEFAULT_DELTA_STOP,
    delta_acc: float = DEFAULT_DELTA_ACC,
    want_stop: bool = True,
    want_acc: bool = True,
    n_repeats: int | None = None,
    prompt_ids: list[int] | None = None,
) -> ScoredGate:
    """Sizes one MMMU-Pro subset and returns a :class:`ScoredGate`."""
    if subset not in SUBSETS:
        raise KeyError(f"unknown subset {subset!r}")
    if not want_stop and not want_acc:
        raise ValueError("need stop and/or accuracy")
    if n_repeats is not None and n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")
    data = hist or load_mmmu_hist()
    ids = (
        list(prompt_ids) if prompt_ids is not None else subset_ids(data, subset)
    )
    if not ids:
        raise ValueError("need at least one prompt id")
    if any(i < 0 or i >= N_PROMPTS for i in ids):
        raise ValueError(f"prompt ids must be in 0..{N_PROMPTS - 1}")
    coins = _coins(data, ids, delta_acc, delta_stop)
    stop_k = acc_k = None
    stop_fp = stop_fn = acc_fp = acc_fn = 0.0
    needed: list[int] = []
    if want_stop:
        m_stop, stop_k, stop_fp, stop_fn = _solve_metric(
            coins["th"],
            coins["tr"],
            n_repeats,
            alpha,
            beta,
            f"{subset}: stop infeasible",
        )
        needed.append(m_stop)
    if want_acc:
        m_acc, acc_k, acc_fp, acc_fn = _solve_metric(
            coins["ah"],
            coins["ar"],
            n_repeats,
            alpha,
            beta,
            f"{subset}: acc infeasible",
        )
        needed.append(m_acc)
    m = n_repeats if n_repeats is not None else max(needed)
    if want_stop and n_repeats is None and m != needed[0]:
        _, stop_k, stop_fp, stop_fn = _solve_metric(
            coins["th"],
            coins["tr"],
            m,
            alpha,
            beta,
            f"{subset}: stop infeasible",
        )
    if want_acc and n_repeats is None and m != needed[-1]:
        _, acc_k, acc_fp, acc_fn = _solve_metric(
            coins["ah"],
            coins["ar"],
            m,
            alpha,
            beta,
            f"{subset}: acc infeasible",
        )
    spec = GateSpec(
        name=f"per_prompt:{subset}",
        hist_mode="per_prompt",
        subset=subset,
        prompt_ids=ids,
        alpha=alpha,
        beta=beta,
        delta_stop=delta_stop,
        delta_acc=delta_acc,
        want_stop=want_stop,
        want_acc=want_acc,
    )
    return ScoredGate(
        spec,
        m,
        stop_k,
        acc_k,
        stop_fp,
        stop_fn,
        acc_fp,
        acc_fn,
        m * len(ids),
        snr(coins["th"], coins["tr"]) if want_stop else 0.0,
        snr(coins["ah"], coins["ar"]) if want_acc else 0.0,
        data.park_stop,
        data.park_stop,
        data.park_stop - delta_stop,
        data.park_acc,
        data.park_acc,
        data.park_acc - delta_acc,
    )
