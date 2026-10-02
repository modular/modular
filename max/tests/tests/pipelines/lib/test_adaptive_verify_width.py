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
"""The throughput-driven verify width: which width it picks, and when."""

from __future__ import annotations

import logging
import random
from collections.abc import Callable, Sequence

import pytest
from max.pipelines.speculative.adaptive_width import AdaptiveVerifyWidth

CANDIDATES = [1, 3, 5, 7]


def _geometric(rate: float) -> list[float]:
    """Shares accepting at least 1..7 drafts, each draft landing at ``rate``."""
    return [rate**k for k in range(1, 8)]


def _run(
    policy: AdaptiveVerifyWidth,
    steps: int,
    shares: Sequence[float],
    step_time: Callable[[int], float | None],
    batch_size: int = 16,
) -> list[int]:
    """Runs decode steps at the widths ``policy`` picks; returns those widths.

    A step verifying ``w`` drafts reports acceptance for its first ``w``
    positions only, as a real one does.
    """
    widths = []
    for _ in range(steps):
        width = policy.next_step_width(batch_size)
        policy.record_step(
            batch_size,
            width,
            [round(batch_size * share) for share in shares[:width]],
            batch_size,
            step_time(width),
        )
        widths.append(width)
    return widths


def test_cold_start_verifies_the_widest_candidate() -> None:
    assert AdaptiveVerifyWidth(CANDIDATES).next_step_width(16) == 7


@pytest.mark.parametrize("rate", [0.3, 0.6, 0.9, 1.0])
def test_flat_cost_settles_on_the_widest_width(rate: float) -> None:
    """Narrowing only pays for itself when it makes the step cheaper."""
    policy = AdaptiveVerifyWidth(CANDIDATES)
    _run(policy, 300, _geometric(rate), lambda width: 1.0)
    assert set(_run(policy, 100, _geometric(rate), lambda width: 1.0)) == {7}


def test_a_nearly_free_narrowing_keeps_the_widest_width() -> None:
    """Width 4 costing 2% less than 7 does not repay the drafts it drops."""
    shares = [0.85, 0.72, 0.6, 0.53, 0.46, 0.38, 0.32]

    def step_time(width: int) -> float:
        return 0.0395 + 0.0008 * (width - 4) / 3

    policy = AdaptiveVerifyWidth([4, 5, 6, 7])
    _run(policy, 300, shares, step_time)
    assert set(_run(policy, 100, shares, step_time)) == {7}


def test_cost_rising_with_width_narrows() -> None:
    policy = AdaptiveVerifyWidth(CANDIDATES)
    _run(policy, 300, _geometric(0.5), lambda width: 1 + 0.5 * width)
    widths = _run(policy, 100, _geometric(0.5), lambda width: 1 + 0.5 * width)
    # Two of these steps are probes at the widest width.
    assert widths.count(1) == 98


def test_untimed_steps_keep_the_widest_width() -> None:
    policy = AdaptiveVerifyWidth(CANDIDATES)
    assert set(_run(policy, 300, _geometric(0.1), lambda width: None)) == {7}


@pytest.mark.parametrize(
    ("probe_interval", "recovered"), [(50, True), (10**9, False)]
)
def test_the_probe_widens_once_acceptance_recovers(
    probe_interval: int, recovered: bool
) -> None:
    """At width 1 no step measures how later drafts fare relative to earlier
    ones, so without a probe the stale extrapolation keeps the width at 1."""
    policy = AdaptiveVerifyWidth([1, 7], probe_interval=probe_interval)

    def step_time(width: int) -> float:
        return 1 + 0.1 * width

    _run(policy, 100, [0.1] + [0.0] * 6, step_time)
    assert policy.next_step_width(16) == 1
    widths = _run(policy, 400, [1.0] * 7, step_time)
    assert (widths[-10:] == [7] * 10) is recovered


@pytest.mark.parametrize(("margin", "most_changes"), [(0.0, None), (0.02, 3)])
def test_the_margin_absorbs_noisy_timings(
    caplog: pytest.LogCaptureFixture, margin: float, most_changes: int | None
) -> None:
    """Widths 4 and 7 decode equally fast; only noise separates them."""
    rng = random.Random(0)
    policy = AdaptiveVerifyWidth([4, 7], margin=margin)
    shares = [1.0] * 4 + [0.5] * 3
    cost = {4: 1.0, 7: 1.3}
    with caplog.at_level(logging.INFO, logger="max.pipelines"):
        _run(
            policy,
            1000,
            shares,
            lambda width: cost[width] * (1 + rng.uniform(-0.03, 0.03)),
        )
    changes = sum("Verify width" in r.getMessage() for r in caplog.records)
    if most_changes is None:
        assert changes > 10
    else:
        assert changes <= most_changes


def test_batch_buckets_choose_independently() -> None:
    """Small batches are memory-bound: a wider step costs them nothing."""
    policy = AdaptiveVerifyWidth(CANDIDATES)
    shares = _geometric(0.5)
    small: list[int] = []
    large: list[int] = []
    for _ in range(300):
        small += _run(policy, 1, shares, lambda width: 1.0, batch_size=2)
        large += _run(
            policy, 1, shares, lambda width: 1 + 0.5 * width, batch_size=32
        )
    assert set(small[-50:]) == {7}
    assert large[-50:].count(1) >= 45
    # Batch size 3 shares the 2-3 bucket.
    assert policy.next_step_width(3) == 7


def test_unmeasured_positions_extrapolate_the_measured_decay() -> None:
    policy = AdaptiveVerifyWidth([3, 7], ema_alpha=1.0)
    # Nothing measured yet: no decay.
    assert policy._acceptance() == [1.0] * 7
    policy.record_step(16, 3, [16, 8, 4], 20, 1.0)
    assert policy._acceptance() == pytest.approx(
        [0.8, 0.4, 0.2, 0.1, 0.05, 0.025, 0.0125]
    )
