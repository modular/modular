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
    _nested_size_biased_pick,
    _WarmupSamplingReport,
    log_warmup_sampling_report,
    pick_warmup_population,
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
        warmup_multiturn=False,
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
        warmup_multiturn=True,
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
        warmup_multiturn=True,
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
        warmup_multiturn=True,
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
        warmup_multiturn=True,
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
    """With no items capping, the average realized_mean across many draws is
    close to target_mean."""
    main = 32
    M = 32
    factor = 8
    realized_means: list[float] = []
    target_mean: float | None = None
    for seed in range(30):
        rng = np.random.default_rng(seed)
        turn_pool = list(rng.integers(1, 60, size=main + factor * M))
        pool = _fixed_pool([int(t) for t in turn_pool])
        _, report = pick_warmup_population(
            pool,
            warmup_count=M,
            warmup_multiturn=True,
            warmup_oversample_factor=factor,
            main_pool_target=main,
            rng=rng,
        )
        assert report is not None
        realized_means.append(report.realized_mean)
        target_mean = report.target_mean
    assert target_mean is not None
    avg_realized = float(np.mean(realized_means))
    # 30 draws of K=32 average out to ~960 picks of T, so the per-draw
    # stddev averages out and what is left is the pick's small bias.
    assert abs(avg_realized - target_mean) / target_mean < 0.03


def test_pick_warmup_sufficient_dataset_no_shrink() -> None:
    main = 20
    M = 4
    factor = 8
    pool = _fixed_pool([5] * (main + factor * M))
    _, report = pick_warmup_population(
        pool,
        warmup_count=M,
        warmup_multiturn=True,
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
        warmup_multiturn=True,
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
            warmup_multiturn=True,
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
        warmup_multiturn=True,
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
    realized_mean across seeds (with-replacement bound; picks without
    replacement vary less)."""
    main = 32
    M = 32
    factor = 8
    realized_means: list[float] = []
    stdev: float | None = None
    for seed in range(30):
        rng = np.random.default_rng(seed)
        turn_pool = list(rng.integers(1, 60, size=main + factor * M))
        pool = _fixed_pool([int(t) for t in turn_pool])
        _, report = pick_warmup_population(
            pool,
            warmup_count=M,
            warmup_multiturn=True,
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
        "Could not warm up multi-turn sessions evenly" in r.message
        for r in caplog.records
    )


def test_log_warmup_sampling_report_caps_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    report = _make_report(cap_count=3)
    with caplog.at_level("WARNING"):
        log_warmup_sampling_report(report)
    assert any(
        "Could not warm up multi-turn sessions evenly" in r.message
        for r in caplog.records
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
# _nested_size_biased_pick
# ---------------------------------------------------------------------------


def _streams(n: int, seed: int) -> list[np.random.Generator]:
    return [np.random.default_rng([seed, i]) for i in range(n)]


def test_nested_pick_no_duplicates() -> None:
    weights = np.random.default_rng(0).integers(1, 50, size=160)
    for seed in range(50):
        idx = _nested_size_biased_pick(
            weights.astype(np.float64), 20, _streams(160, seed)
        )
        assert len(idx) == 20
        assert len(set(idx)) == 20


def test_nested_pick_mean_matches_size_biased() -> None:
    """E[mean of picks] is close to sum(T^2) / sum(T) when no item caps."""
    T = np.random.default_rng(3).integers(1, 50, size=8 * 16)
    T = T.astype(np.float64)
    sb_mean = float((T * T).sum() / T.sum())
    realized = [
        float(T[_nested_size_biased_pick(T, 16, _streams(len(T), seed))].mean())
        for seed in range(300)
    ]
    assert abs(float(np.mean(realized)) - sb_mean) / sb_mean < 0.03


def test_nested_pick_keeps_smaller_picks() -> None:
    """The picks for k are the first picks for any larger k."""
    factor = 8
    T = np.random.default_rng(4).integers(1, 50, size=factor * 32)
    T = T.astype(np.float64)
    larger = _nested_size_biased_pick(T, 32, _streams(len(T), 7))
    for k in (1, 5, 16, 31):
        n = factor * k
        assert _nested_size_biased_pick(T[:n], k, _streams(n, 7)) == larger[:k]


def test_pick_warmup_keeps_main_pool_and_smaller_picks() -> None:
    """A larger warmup count keeps the main pool and every smaller pick.

    The picks keep their start turns too, so the workload a sweep sends
    differs across concurrency levels only by the sessions added.
    """
    main = 40
    factor = 8
    turns = np.random.default_rng(5).integers(1, 60, size=main + factor * 24)
    pool = _fixed_pool([int(t) for t in turns])

    def pick(
        warmup_count: int,
    ) -> tuple[list[tuple[int | None, int]], list[ChatSession]]:
        sessions, _ = pick_warmup_population(
            pool[: main + factor * warmup_count],
            warmup_count,
            warmup_multiturn=True,
            warmup_oversample_factor=factor,
            main_pool_target=main,
            rng=np.random.default_rng(42),
        )
        picks = [(s.id, s.prefix_turns) for s in sessions[:warmup_count]]
        return picks, sessions[warmup_count:]

    larger_picks, larger_main = pick(24)
    for warmup_count in (4, 16):
        picks, main_sessions = pick(warmup_count)
        assert picks == larger_picks[:warmup_count]
        assert main_sessions == larger_main
