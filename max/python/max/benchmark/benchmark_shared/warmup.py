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

"""Warmup population sampling for multi-turn benchmarks.

Provides length-biased session selection to seed the server's KV cache before
the measured benchmark window, avoiding cold-start bias in steady-state metrics.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from max.benchmark.benchmark_shared.datasets import ChatSession

logger = logging.getLogger(__name__)


def systematic_probability_proportional_to_size(
    weights: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pick ``k`` distinct indices into ``weights`` with inclusion probability
    ``min(1, k * weights[i] / sum(weights))``.

    Algorithm (systematic PPS with iterated-cap handling):

    1. Lay items end-to-end on a number line of length ``W = sum(weights)``,
       item ``i`` occupying an interval of length ``weights[i]``.
    2. Place ``k`` equally-spaced ticks at ``u, u + W/k, ..., u + (k-1)*W/k``
       with a single shared offset ``u ~ Uniform(0, W/k)``. Each tick lands
       in exactly one item's interval; that item is selected.
    3. If any item has ``weights[i] >= W/k``, its interval is at least as wide
       as the tick spacing and would be hit by 2+ ticks (a duplicate).
       Pre-include such items, remove them from the pool, recompute the
       threshold over the residual weight and residual tick count, and
       repeat. Then run systematic PPS over the residual.

    Why this removes the depletion bias of ``rng.choice(replace=False, p=...)``:
    sequential PPSWOR (numpy's no-replacement weighted sampling) over-includes
    small items because once a heavy item is drawn the remaining weights
    renormalize. Systematic PPS, by contrast, gives ``pi_i = k*weights[i]/W``
    *exactly* for any uncapped item: its interval is shorter than the tick
    spacing, so at most one tick can land in it, with probability equal to
    the interval length divided by the spacing. Plugging in:
    ``E[mean] = (1/k) * sum_i weights[i] * pi_i = sum_i weights[i]^2 / W``
    = the size-biased mean. No depletion, no convergence rate — equality
    by construction.

    Caps still leak some bias because pre-included items contribute
    ``T_i * 1`` instead of the ideal ``T_i * (k * T_i / W)``. ``cap_count == 0``
    is the condition for analytically zero bias; callers should warn the user
    when caps occur.
    """
    residual = np.asarray(weights, dtype=np.float64).copy()
    n = len(residual)
    chosen: list[int] = []

    # Iterated cap: any item with weight >= W_remaining / k_remaining is
    # wider than the tick spacing and would be hit by multiple ticks.
    # Pre-include it (its ideal pi >= 1 anyway) and recompute the threshold
    # over the residual.
    while True:
        remaining_k = k - len(chosen)
        if remaining_k <= 0:
            break
        residual_sum = float(residual.sum())
        if residual_sum <= 0.0:
            break
        cap = residual_sum / remaining_k
        over = np.where(residual >= cap)[0]
        if len(over) == 0:
            break
        chosen.extend(int(i) for i in over)
        residual[over] = 0.0

    # Run systematic PPS over the residual with the remaining ticks.
    remaining_k = k - len(chosen)
    if remaining_k > 0:
        cum = np.cumsum(residual)
        residual_sum = float(cum[-1])
        if residual_sum > 0.0:
            step = residual_sum / remaining_k
            offset = float(rng.uniform(0.0, step))
            ticks = offset + np.arange(remaining_k) * step
            picks = np.searchsorted(cum, ticks)
            # Guard the float-edge case where a tick equals the total sum.
            picks = np.minimum(picks, n - 1)
            chosen.extend(int(i) for i in picks)

    return np.asarray(chosen[:k], dtype=np.int64)


@dataclass
class _WarmupSamplingReport:
    """Statistics about a warmup pick, logged at the start of a run."""

    # Size of the warmup-candidate sub-pool. Equals ``factor * warmup_count``;
    # bigger pools leave more cap headroom.
    warmup_pool: int
    # Size of the unbiased main sub-pool — the actual benchmark sessions.
    # Untouched by the warmup pick so it preserves natural P(T).
    main_pool: int
    # Number of in-flight warmup slots (M) seeded at t=0.
    warmup_count: int
    # The configured ``warmup_oversample_factor`` for this run.
    factor: int
    # Closed-form size-biased mean: E[T_live] = sum(T**2)/sum(T).
    target_mean: float
    # Realized mean of the sampled warmup picks.
    realized_mean: float
    # Conservative stdev of ``realized_mean`` around ``target_mean``,
    # treating each pick as an independent length-biased draw:
    #   sb_var = sum(T**3)/sum(T) - target_mean**2
    #   stdev  = sqrt(sb_var / K)
    # Systematic PPS picks are negatively correlated, so the true stdev is
    # smaller (typically ~half). Used to report ``|realized - target|`` as
    # a unitless ratio in stdev units; under this conservative bound,
    # anything below ~1 is unambiguously noise.
    realized_mean_stdev: float
    # Candidates whose ideal proportional inclusion probability would exceed 1
    # (T_i > W/K). Systematic PPS pre-includes these with pi=1, which is the
    # best a no-replacement scheme can do but introduces a small residual
    # bias relative to ``target_mean``.
    cap_count: int
    # ``main_pool_target + warmup_oversample_factor * warmup_count``.
    ideal_total: int
    # ``len(chat_sessions)``. If less than ``ideal_total``, both pools shrank.
    available_total: int


def pick_warmup_population(
    chat_sessions: Sequence[ChatSession],
    warmup_count: int,
    *,
    warmup_to_steady_state: bool,
    warmup_oversample_factor: int,
    main_pool_target: int,
    rng: np.random.Generator,
) -> tuple[list[ChatSession], _WarmupSamplingReport | None]:
    """Build the runner's task list with warmup picks at the head.

    When ``warmup_to_steady_state=True`` and ``warmup_count > 0``, the
    helper picks ``warmup_count`` warmup sessions from the leading
    ``factor * warmup_count`` candidates (clamped to what's available)
    and assigns each a random ``prefix_turns`` in ``[0, T-1)``. The
    remaining trailing slice is the main benchmark sessions (untouched,
    preserves natural P(T)). For ``factor >= 2`` the picks are length-biased
    via :func:`systematic_probability_proportional_to_size`, which gives
    inclusion probability ``min(1, K * T_i / sum(T))`` exactly — no
    depletion bias. For ``factor < 2`` we don't have headroom for a
    weighted draw so the picks are uniform; the report's target/realized
    split lets users see the residual bias.

    Returns ``(reordered_sessions, report)``. ``report`` is ``None``
    only when warmup is off (``warmup_to_steady_state=False`` or
    ``warmup_count == 0``).
    """
    n_total = len(chat_sessions)
    warmup_count = max(0, warmup_count)
    main_pool_target = max(0, main_pool_target)

    if not warmup_to_steady_state or warmup_count == 0:
        return list(chat_sessions), None

    if n_total < warmup_count:
        raise ValueError(
            f"Dataset produced {n_total} sessions but warmup needs at least"
            f" {warmup_count}. Increase the dataset size or lower"
            " --max-concurrency / --max-concurrent-conversations."
        )

    # When the dataset under-produces, shrink both pools by the same
    # fraction so warmup doesn't absorb the whole shortfall. Truncating
    # biases the warmup pool down rather than the main pool.
    ideal_candidate_pool = warmup_oversample_factor * warmup_count
    ideal_total = main_pool_target + ideal_candidate_pool
    if n_total >= ideal_total:
        candidate_pool = ideal_candidate_pool
        main_count = main_pool_target
    else:
        shrink = n_total / ideal_total
        candidate_pool = int(ideal_candidate_pool * shrink)
        main_count = n_total - candidate_pool
    # factor=0 leaves candidate_pool=0; floor it so the picker has M slots.
    if candidate_pool < warmup_count:
        candidate_pool = warmup_count
        main_count = min(main_pool_target, n_total - candidate_pool)

    actual_warmup_count = min(warmup_count, candidate_pool)
    candidates = chat_sessions[:candidate_pool]
    main_sessions = list(
        chat_sessions[candidate_pool : candidate_pool + main_count]
    )

    turn_counts = np.array(
        [max(1, s.num_turns) for s in candidates], dtype=np.int64
    )
    # Length-biased only when factor>=2 AND we have headroom to pick from.
    # Otherwise fall back to a plain uniform pick — at factor<2 we don't
    # have enough candidates above ``actual_warmup_count`` to do a
    # meaningful weighted draw, so the report's target/realized split tells
    # the user about the residual bias.
    use_length_bias = (
        warmup_oversample_factor >= 2 and candidate_pool > actual_warmup_count
    )
    if use_length_bias:
        warmup_idx = systematic_probability_proportional_to_size(
            turn_counts, actual_warmup_count, rng
        )
    else:
        warmup_idx = rng.choice(
            candidate_pool, size=actual_warmup_count, replace=False
        )

    warmup_sessions: list[ChatSession] = []
    for i in warmup_idx:
        s = candidates[int(i)]
        total_turns = max(1, s.num_turns)
        prefix_turns = (
            int(rng.integers(0, total_turns)) if total_turns > 1 else 0
        )
        warmup_sessions.append(
            dataclasses.replace(s, prefix_turns=prefix_turns)
        )

    # ``target_mean`` is the size-biased mean of the *full* dataset (warmup
    # candidates + main pool), so it reflects steady-state for the workload
    # as a whole — not just the candidate slice the picker drew from.
    full_turn_counts = np.array(
        [max(1, s.num_turns) for s in chat_sessions],
        dtype=np.int64,
    )
    full_sum = float(full_turn_counts.sum())
    target_mean = float((full_turn_counts**2).sum() / full_sum)
    realized_mean = float(turn_counts[list(warmup_idx)].mean())

    # Per-draw stdev on ``realized_mean`` under a with-replacement bound.
    # Variance of a single length-biased pick is
    #   E[T^2 | size-bias] - sb_mean^2 = sum(T^3)/sum(T) - target_mean^2;
    # K independent picks scale Var by 1/K. Systematic PPS picks are
    # negatively correlated, so this overestimates by ~2x — that's in the
    # safe direction: if ``|realized - target| / stdev`` is below ~1 even
    # under this conservative bound, it's unambiguously noise.
    sb_var = (full_turn_counts**3).sum() / full_sum - target_mean**2
    sb_var = max(0.0, float(sb_var))
    realized_mean_stdev = float(np.sqrt(sb_var / max(1, actual_warmup_count)))

    cap_threshold = float(turn_counts.sum()) / actual_warmup_count
    cap_count = int((turn_counts > cap_threshold).sum())

    report = _WarmupSamplingReport(
        warmup_pool=candidate_pool,
        main_pool=len(main_sessions),
        warmup_count=actual_warmup_count,
        factor=warmup_oversample_factor,
        target_mean=target_mean,
        realized_mean=realized_mean,
        realized_mean_stdev=realized_mean_stdev,
        cap_count=cap_count,
        ideal_total=ideal_total,
        available_total=n_total,
    )
    return warmup_sessions + main_sessions, report


def log_warmup_sampling_report(report: _WarmupSamplingReport) -> None:
    """Emit the per-run [warmup-sampling] log + cap-triggered warning."""

    def _pct(value: float, ref: float) -> str:
        if ref == 0:
            return "n/a"
        return f"{100.0 * (value - ref) / ref:+.1f}%"

    if report.realized_mean_stdev > 0:
        stdev_ratio = (
            abs(report.realized_mean - report.target_mean)
            / report.realized_mean_stdev
        )
        stdev_str = f"{stdev_ratio:.2f} stdev from target"
    else:
        stdev_str = "stdev n/a"

    logger.info(
        "[warmup-sampling] warmup_pool=%d main_pool=%d M=%d factor=%d\n"
        "  target mean from samples:                      %.2f\n"
        "  realized warmup mean (one draw):               %.2f  (%s, %s)\n"
        "  always-picked sessions (too long for pool):    %d / %d",
        report.warmup_pool,
        report.main_pool,
        report.warmup_count,
        report.factor,
        report.target_mean,
        report.realized_mean,
        _pct(report.realized_mean, report.target_mean),
        stdev_str,
        report.cap_count,
        report.warmup_pool,
    )

    if report.available_total < report.ideal_total and report.ideal_total > 0:
        deficit = report.ideal_total - report.available_total
        pct = 100.0 * deficit / report.ideal_total
        logger.warning(
            "Dataset under-produced for warmup: %d sessions short of ideal"
            " %d (%.1f%% deficit, available=%d). Shrunk both pools"
            " proportionally to warmup_pool=%d, main_pool=%d.",
            deficit,
            report.ideal_total,
            pct,
            report.available_total,
            report.warmup_pool,
            report.main_pool,
        )

    if report.cap_count > 0:
        logger.warning(
            "Could not warmup to steady state: %d session(s) are too long for "
            "a candidate pool of %d, so they get picked every time and bias "
            "the warmup. Increase --warmup-oversample-factor (currently %d) "
            "to enlarge the pool.",
            report.cap_count,
            report.warmup_pool,
            report.factor,
        )
