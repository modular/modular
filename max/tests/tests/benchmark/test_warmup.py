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
"""Unit tests for benchmark_shared.warmup."""

from __future__ import annotations

import numpy as np
import pytest
from max.benchmark.benchmark_shared.datasets.types import (
    ChatSession,
    SessionMessage,
)
from max.benchmark.benchmark_shared.warmup import (
    _WarmupSamplingReport,
    log_warmup_sampling_report,
    pick_warmup_population,
    systematic_probability_proportional_to_size,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_session(session_id: int, num_turns: int) -> ChatSession:
    msgs: list[SessionMessage] = []
    for _ in range(num_turns):
        msgs.append(SessionMessage(source="user", content="u", num_tokens=1))
        msgs.append(
            SessionMessage(source="assistant", content="a", num_tokens=1)
        )
    return ChatSession(id=session_id, messages=msgs)


def _fixed_pool(turn_counts: list[int]) -> list[ChatSession]:
    return [_fake_session(i, t) for i, t in enumerate(turn_counts)]


def _make_report(
    target_mean: float = 20.0,
    realized_mean: float = 20.0,
    realized_mean_stdev: float = 1.0,
    cap_count: int = 0,
    ideal_total: int = 228,
    available_total: int = 228,
) -> _WarmupSamplingReport:
    return _WarmupSamplingReport(
        warmup_pool=128,
        main_pool=100,
        warmup_count=16,
        factor=8,
        target_mean=target_mean,
        realized_mean=realized_mean,
        realized_mean_stdev=realized_mean_stdev,
        cap_count=cap_count,
        ideal_total=ideal_total,
        available_total=available_total,
    )


# ---------------------------------------------------------------------------
# pick_warmup_population
# ---------------------------------------------------------------------------


def test_pick_warmup_off_returns_unchanged() -> None:
    pool = _fixed_pool([3, 5, 7, 11])
    sessions, report = pick_warmup_population(
        pool,
        warmup_count=2,
        warmup_to_steady_state=False,
        warmup_oversample_factor=8,
        main_pool_target=4,
        rng=np.random.default_rng(0),
    )
    assert report is None
    assert sessions == pool
    assert all(s.prefix_turns == 0 for s in sessions)


def test_pick_warmup_zero_count_returns_unchanged() -> None:
    pool = _fixed_pool([3, 5, 7])
    sessions, report = pick_warmup_population(
        pool,
        warmup_count=0,
        warmup_to_steady_state=True,
        warmup_oversample_factor=8,
        main_pool_target=3,
        rng=np.random.default_rng(0),
    )
    assert report is None
    assert sessions == pool


def test_pick_warmup_factor_one_uniform_emits_report() -> None:
    # factor=1 has no length-bias headroom; falls through to uniform pick.
    # Report still emitted so the user can see the bias.
    pool = _fixed_pool([3, 5, 7, 11, 13])
    sessions, report = pick_warmup_population(
        pool,
        warmup_count=2,
        warmup_to_steady_state=True,
        warmup_oversample_factor=1,
        main_pool_target=3,
        rng=np.random.default_rng(0),
    )
    assert report is not None
    assert report.factor == 1
    # Warmup picks at the head, prefix_turns assigned (0 for T=1).
    assert len(sessions) == len(pool)
    assert all(s.prefix_turns >= 0 for s in sessions[:2])
    # Main sessions (last 3) untouched: prefix_turns=0.
    assert all(s.prefix_turns == 0 for s in sessions[2:])


def test_pick_warmup_factor_zero_uniform_emits_report() -> None:
    # factor=0 behaves the same as factor=1 — uniform pick.
    pool = _fixed_pool([3, 5, 7, 11, 13])
    sessions, report = pick_warmup_population(
        pool,
        warmup_count=2,
        warmup_to_steady_state=True,
        warmup_oversample_factor=0,
        main_pool_target=5,
        rng=np.random.default_rng(0),
    )
    assert report is not None
    assert report.factor == 0
    assert len(sessions) == len(pool)


def test_pick_warmup_length_biased_basic() -> None:
    rng = np.random.default_rng(123)
    # Pool of 144 sessions: 16 main sessions + 8*16 = 128 candidates.
    main = 16
    factor = 8
    M = 16
    turn_pool = list(rng.integers(1, 50, size=main + factor * M))
    pool = _fixed_pool([int(t) for t in turn_pool])
    sessions, report = pick_warmup_population(
        pool,
        warmup_count=M,
        warmup_to_steady_state=True,
        warmup_oversample_factor=factor,
        main_pool_target=main,
        rng=rng,
    )
    assert report is not None
    assert report.factor == factor
    assert report.warmup_pool == factor * M
    assert report.main_pool == main
    assert report.warmup_count == M
    # All warmup picks at head; main sessions at tail with prefix_turns=0.
    assert len(sessions) == M + main
    warmup_picks = sessions[:M]
    main_sessions = sessions[M:]
    assert all(s.prefix_turns >= 0 for s in warmup_picks)
    assert any(s.prefix_turns > 0 for s in warmup_picks)
    assert all(s.prefix_turns == 0 for s in main_sessions)


def test_pick_warmup_length_biased_matches_size_biased_target() -> None:
    """Systematic PPS gives E[realized_mean] = target_mean exactly when no
    items cap. Average across many draws should be within ~1% of target."""
    main = 32
    M = 32
    factor = 8
    realized_means: list[float] = []
    target_mean: float | None = None
    for seed in range(50):
        rng = np.random.default_rng(seed)
        turn_pool = list(rng.integers(1, 60, size=main + factor * M))
        pool = _fixed_pool([int(t) for t in turn_pool])
        _, report = pick_warmup_population(
            pool,
            warmup_count=M,
            warmup_to_steady_state=True,
            warmup_oversample_factor=factor,
            main_pool_target=main,
            rng=rng,
        )
        assert report is not None
        realized_means.append(report.realized_mean)
        target_mean = report.target_mean
    assert target_mean is not None
    avg_realized = float(np.mean(realized_means))
    # 50 draws of K=32 averages out to ~1600 picks of T; the analytical bias
    # is 0 (no caps) and per-draw stddev averages out fast.
    assert abs(avg_realized - target_mean) / target_mean < 0.02


def test_pick_warmup_sufficient_dataset_no_shrink() -> None:
    main = 20
    M = 4
    factor = 8
    pool = _fixed_pool([5] * (main + factor * M))
    _, report = pick_warmup_population(
        pool,
        warmup_count=M,
        warmup_to_steady_state=True,
        warmup_oversample_factor=factor,
        main_pool_target=main,
        rng=np.random.default_rng(0),
    )
    assert report is not None
    assert report.warmup_pool == factor * M
    assert report.main_pool == main
    assert report.ideal_total == main + factor * M
    assert report.available_total == main + factor * M


def test_pick_warmup_mild_underproduction_shrinks_both_pools() -> None:
    main = 20
    M = 4
    factor = 8
    ideal_total = main + factor * M  # 52
    n_total = 39  # ~75% of ideal
    pool = _fixed_pool([5] * n_total)
    _, report = pick_warmup_population(
        pool,
        warmup_count=M,
        warmup_to_steady_state=True,
        warmup_oversample_factor=factor,
        main_pool_target=main,
        rng=np.random.default_rng(0),
    )
    assert report is not None
    assert report.ideal_total == ideal_total
    assert report.available_total == n_total
    # Both pools shrunk: warmup still oversampled (> M), main below target.
    assert report.warmup_pool < factor * M
    assert report.warmup_pool > M
    assert report.main_pool < main
    assert report.main_pool > 0
    # Both pools shrank by roughly the same fraction.
    warmup_frac = report.warmup_pool / (factor * M)
    main_frac = report.main_pool / main
    assert abs(warmup_frac - main_frac) < 0.1


def test_pick_warmup_raises_when_dataset_smaller_than_warmup_count() -> None:
    pool = _fixed_pool([5, 5, 5])
    with pytest.raises(ValueError, match="warmup needs at least"):
        pick_warmup_population(
            pool,
            warmup_count=4,
            warmup_to_steady_state=True,
            warmup_oversample_factor=8,
            main_pool_target=20,
            rng=np.random.default_rng(0),
        )


def test_pick_warmup_severe_underproduction_floors_warmup() -> None:
    main = 20
    M = 4
    factor = 8
    n_total = 5  # shrink drops candidate_pool below M, floor kicks in
    pool = _fixed_pool([5] * n_total)
    _, report = pick_warmup_population(
        pool,
        warmup_count=M,
        warmup_to_steady_state=True,
        warmup_oversample_factor=factor,
        main_pool_target=main,
        rng=np.random.default_rng(0),
    )
    assert report is not None
    assert report.warmup_pool == M
    assert report.main_pool == n_total - M
    assert report.ideal_total == main + factor * M
    assert report.available_total == n_total


def test_pick_warmup_realized_mean_stdev_brackets_observed_spread() -> None:
    """The reported per-draw stdev should upper-bound the actual spread of
    realized_mean across seeds (with-replacement bound; systematic PPS
    variance is at most this)."""
    main = 32
    M = 32
    factor = 8
    realized_means: list[float] = []
    stdev: float | None = None
    for seed in range(60):
        rng = np.random.default_rng(seed)
        turn_pool = list(rng.integers(1, 60, size=main + factor * M))
        pool = _fixed_pool([int(t) for t in turn_pool])
        _, report = pick_warmup_population(
            pool,
            warmup_count=M,
            warmup_to_steady_state=True,
            warmup_oversample_factor=factor,
            main_pool_target=main,
            rng=rng,
        )
        assert report is not None
        realized_means.append(report.realized_mean)
        stdev = report.realized_mean_stdev
    assert stdev is not None
    assert stdev > 0
    observed_std = float(np.std(realized_means, ddof=1))
    # Reported stdev is a with-replacement upper bound; observed should be
    # at most ~1.3x stdev even with sampling noise on the std estimate.
    assert observed_std <= 1.3 * stdev


# ---------------------------------------------------------------------------
# log_warmup_sampling_report
# ---------------------------------------------------------------------------


def test_log_warmup_sampling_report_no_caps_no_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    report = _make_report(cap_count=0)
    with caplog.at_level("WARNING"):
        log_warmup_sampling_report(report)
    assert not any(
        "Could not warmup to steady state" in r.message for r in caplog.records
    )


def test_log_warmup_sampling_report_caps_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    report = _make_report(cap_count=3)
    with caplog.at_level("WARNING"):
        log_warmup_sampling_report(report)
    assert any(
        "Could not warmup to steady state" in r.message for r in caplog.records
    )


def test_log_warmup_sampling_report_warns_on_underproduction(
    caplog: pytest.LogCaptureFixture,
) -> None:
    report = _make_report(ideal_total=200, available_total=160)
    with caplog.at_level("WARNING"):
        log_warmup_sampling_report(report)
    msgs = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert any("Dataset under-produced for warmup" in m for m in msgs)
    assert any("20.0% deficit" in m for m in msgs)


def test_log_warmup_sampling_report_no_warning_when_target_met(
    caplog: pytest.LogCaptureFixture,
) -> None:
    report = _make_report()
    with caplog.at_level("WARNING"):
        log_warmup_sampling_report(report)
    assert not any(
        "Dataset under-produced for warmup" in r.message for r in caplog.records
    )


# ---------------------------------------------------------------------------
# systematic_probability_proportional_to_size
# ---------------------------------------------------------------------------


def test_systematic_pps_no_duplicates() -> None:
    rng = np.random.default_rng(0)
    weights = rng.integers(1, 50, size=100).astype(np.float64)
    for _ in range(500):
        idx = systematic_probability_proportional_to_size(weights, 20, rng)
        assert len(idx) == 20
        assert len(set(idx.tolist())) == 20


def test_systematic_pps_inclusion_probability_matches_weight() -> None:
    """Empirical P(item i picked) ≈ K * w_i / W when no item caps."""
    rng = np.random.default_rng(1)
    weights = np.array([1, 2, 3, 5, 8, 13, 21], dtype=np.float64)
    K = 3
    # K * max(w) / W = 3 * 21 / 53 = 1.19 > 1 → cap. Lower K to avoid cap
    # for the matching-probability check.
    K = 2
    cap_thresh = weights.sum() / K
    assert (weights < cap_thresh).all()
    n_trials = 20000
    counts = np.zeros(len(weights), dtype=np.int64)
    for _ in range(n_trials):
        idx = systematic_probability_proportional_to_size(weights, K, rng)
        counts[idx] += 1
    expected = K * weights / weights.sum()
    actual = counts / n_trials
    assert np.allclose(actual, expected, atol=0.02)


def test_systematic_pps_iterated_cap_pre_includes() -> None:
    """An item with weight >= W/K is always included."""
    rng = np.random.default_rng(2)
    weights = np.array([1, 1, 1, 1, 100], dtype=np.float64)
    K = 2
    # weights[4] = 100 >> W/K = 104/2 = 52.
    for _ in range(200):
        idx = systematic_probability_proportional_to_size(weights, K, rng)
        assert 4 in idx.tolist()


def test_systematic_pps_mean_matches_size_biased() -> None:
    """E[mean of picks] = sum(T^2) / sum(T) when no item caps."""
    rng = np.random.default_rng(3)
    T = rng.integers(1, 50, size=200).astype(np.float64)
    K = 8
    cap_thresh = T.sum() / K
    assert (T < cap_thresh).all()
    sb_mean = float((T * T).sum() / T.sum())
    realized = []
    for _ in range(500):
        idx = systematic_probability_proportional_to_size(T, K, rng)
        realized.append(float(T[idx].mean()))
    avg = float(np.mean(realized))
    assert abs(avg - sb_mean) / sb_mean < 0.01
