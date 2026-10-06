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
"""Chooses the verify width that decodes the most tokens per second.

Acceptance depends on the traffic, not the batch size: code drafts well,
open-ended prose does not. Step cost depends on the batch size, so it is
measured per power-of-two batch bucket.
"""

from __future__ import annotations

import logging
from bisect import bisect_left
from collections.abc import Sequence
from itertools import accumulate

logger = logging.getLogger("max.pipelines")

__all__ = ["AdaptiveVerifyWidth", "parse_adaptive_widths"]


def parse_adaptive_widths(value: str, max_width: int | None) -> list[int]:
    """Returns the verify widths ``value`` names, ascending and distinct.

    Args:
        value: A comma-separated list such as ``"1,3,5"``, or ``"all"`` for
            every width from 1 to ``max_width``.
        max_width: The drafts a step carries, which no width may exceed.
            ``None`` while it is still unknown: only the syntax is checked,
            and ``"all"`` names no widths yet.

    Raises:
        ValueError: If a field is not an integer or is outside
            ``1..max_width``.
    """
    if value.strip() == "all":
        return [] if max_width is None else list(range(1, max_width + 1))
    widths: set[int] = set()
    for field in value.split(","):
        try:
            widths.add(int(field))
        except ValueError:
            raise ValueError(
                f"Could not read {field.strip()!r} as a verify width. Use a "
                'comma-separated list such as "1,3,5", or "all".'
            ) from None
    # 0 is legal in a batch-size schedule, but a step that verifies nothing
    # reports no acceptance, so the width could never widen again.
    out_of_range = sorted(
        w for w in widths if w < 1 or (max_width is not None and w > max_width)
    )
    if out_of_range:
        limit = "num_speculative_tokens" if max_width is None else max_width
        raise ValueError(
            f"Verify widths {out_of_range} are outside 1..{limit}: a step "
            "verifies at least one draft and at most the drafts it carries."
        )
    return sorted(widths)


class AdaptiveVerifyWidth:
    """Picks the verify width with the best accepted tokens per step time.

    A width's rate is ``(1 + p[1] + ... + p[w]) / time(w)``, where ``p[k]`` is
    the share of sequences accepting at least ``k`` drafts. A step verifying
    ``w`` drafts cannot measure ``p[k]`` past ``w``, so those positions are
    extrapolated, and a periodic probe at the widest width re-measures them.
    """

    def __init__(
        self,
        candidates: Sequence[int],
        *,
        ema_alpha: float = 0.2,
        update_interval: int = 5,
        probe_interval: int = 50,
        margin: float = 0.02,
        min_batch_size: int = 1,
    ) -> None:
        """Builds a controller over ``candidates``.

        Args:
            candidates: The widths this may choose, ascending and distinct.
            ema_alpha: Weight of the newest observation in each average.
            update_interval: Observations of a bucket between width
                choices. A draft position not measured within this many
                observations is extrapolated.
            probe_interval: Steps between probes at the widest width.
            margin: Relative gain a new width needs over the current one.
            min_batch_size: Smallest batch size whose graphs include the
                narrower widths. Smaller buckets keep the widest.
        """
        self._candidates = list(candidates)
        self._alpha = ema_alpha
        self._update_interval = max(1, update_interval)
        self._probe_interval = max(1, probe_interval)
        self._margin = margin
        self._min_batch_size = min_batch_size
        widest = self._candidates[-1]
        self._accepted = [0.0] * widest
        # Observations each position stays measured for; 0 extrapolates it.
        self._fresh_for = [0] * widest
        # Unmeasured, no decay: extrapolation then favors the widest width.
        self._ratio = 1.0
        self._probe_countdown = self._probe_interval
        self._widths: dict[int, int] = {}
        self._step_times: dict[int, dict[int, float]] = {}
        self._bucket_countdowns: dict[int, int] = {}

    def next_step_width(self, batch_size: int) -> int:
        """Returns the width to verify for the next decode step.

        Call once per step: every ``probe_interval``-th call is a probe.
        """
        self._probe_countdown -= 1
        if self._probe_countdown == 0:
            self._probe_countdown = self._probe_interval
            return self._candidates[-1]
        return self._widths.get(_bucket(batch_size), self._candidates[-1])

    def record_step(
        self,
        batch_size: int,
        width: int,
        accepted_per_position: Sequence[int],
        num_verifications: int,
        step_time_s: float | None,
    ) -> None:
        """Folds one verified decode step into the averages.

        Args:
            batch_size: The batch size the step's width was chosen for.
            width: The drafts the step verified.
            accepted_per_position: Sequences accepting each draft position.
            num_verifications: Sequences the step verified; at least one.
            step_time_s: The step's execution time, or ``None`` when it is
                unknown. Acceptance trains either way.
        """
        self._fresh_for = [max(0, left - 1) for left in self._fresh_for]
        prior = self._acceptance()
        for k in range(width):
            share = accepted_per_position[k] / num_verifications
            self._accepted[k] = (
                self._alpha * share + (1 - self._alpha) * prior[k]
            )
            self._fresh_for[k] = self._update_interval
        reached = sum(accepted_per_position[: width - 1])
        if reached:
            ratio = sum(accepted_per_position[1:width]) / reached
            self._ratio = self._alpha * ratio + (1 - self._alpha) * self._ratio

        bucket = _bucket(batch_size)
        if step_time_s is not None:
            times = self._step_times.setdefault(bucket, {})
            times[width] = (
                step_time_s
                if width not in times
                else self._alpha * step_time_s
                + (1 - self._alpha) * times[width]
            )
        left = self._bucket_countdowns.get(bucket, self._update_interval) - 1
        if left == 0:
            left = self._update_interval
            self._choose_width(bucket)
        self._bucket_countdowns[bucket] = left

    def _acceptance(self) -> list[float]:
        """Returns ``p[1..widest]``, extrapolating stale positions."""
        ratio = min(self._ratio, 1.0)
        shares: list[float] = []
        previous = 1.0
        for accepted, fresh_for in zip(
            self._accepted, self._fresh_for, strict=True
        ):
            previous = accepted if fresh_for else previous * ratio
            shares.append(previous)
        return shares

    def _step_time(self, bucket: int, width: int) -> float:
        """Returns the expected step time at ``width`` in ``bucket``.

        With nothing timed every width costs the same, which keeps the
        widest. Below the narrowest timed width, the time is scaled by the
        tokens verified per sequence. No step is cheaper than that, so the
        width gets tried once and then timed.
        """
        times = self._step_times.get(bucket)
        if not times:
            return 1.0
        timed = sorted(times)
        i = bisect_left(timed, width)
        if i == len(timed):
            return times[timed[-1]]
        above = timed[i]
        if i == 0:
            return times[above] * (width + 1) / (above + 1)
        below = timed[i - 1]
        return times[below] + (times[above] - times[below]) * (
            width - below
        ) / (above - below)

    def _choose_width(self, bucket: int) -> None:
        """Switches ``bucket`` to the width with the best tokens per second."""
        # A bucket entirely below the floor can only run the widest width.
        if (2 << bucket) - 1 < self._min_batch_size:
            return
        tokens = list(accumulate(self._acceptance(), initial=1.0))
        rates = {
            width: tokens[width] / self._step_time(bucket, width)
            for width in self._candidates
        }
        current = self._widths.get(bucket, self._candidates[-1])
        best = max(rates, key=rates.__getitem__)
        if rates[best] <= rates[current] * (1 + self._margin):
            return
        logger.info(
            "Verify width %d -> %d for decode batch sizes %d-%d: %.3g -> "
            "%.3g tokens per sequence per second.",
            current,
            best,
            1 << bucket,
            (2 << bucket) - 1,
            rates[current],
            rates[best],
        )
        self._widths[bucket] = best


def _bucket(batch_size: int) -> int:
    """Returns the power-of-two bucket: 1, 2-3, 4-7, 8-15, and so on."""
    return batch_size.bit_length() - 1
