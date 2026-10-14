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

"""Provides the steps of a serving benchmark run and its sweep iterations.

Seed resolution for a run, and stats trimming, summary reporting with
validation, and result retention for each ``(max_concurrency, request_rate)``
iteration of its sweep.
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np
from max.benchmark.benchmark_shared.config import ServingBenchmarkConfig
from max.benchmark.benchmark_shared.lora_benchmark_manager import (
    LoRABenchmarkManager,
)
from max.benchmark.benchmark_shared.metrics import BenchmarkResult
from max.benchmark.benchmark_shared.serving_result_output import (
    print_benchmark_summary,
)

logger = logging.getLogger(__name__)


def resolve_skip_counts(
    *,
    orig_skip_first: int | None,
    orig_skip_last: int | None,
    request_rate: float,
    max_concurrency: int | None,
    ignore_first_turn_stats: bool,
    warmup_to_steady_state: bool,
) -> tuple[int, int]:
    """Resolves the head and tail request counts to exclude from stats.

    Unset counts are derived from the load shape: an unpaced run at a
    concurrency above 1 trims that many requests from each end, where its
    ramp-up and ramp-down run below full concurrency.

    Args:
        orig_skip_first: The requested head count, or ``None`` to derive it.
        orig_skip_last: The requested tail count, or ``None`` to derive it.
        request_rate: The run's request rate; ``inf`` for unpaced load.
        max_concurrency: The run's concurrency limit, or ``None`` for none.
        ignore_first_turn_stats: Whether each session's first turn is
            already dropped.
        warmup_to_steady_state: Whether sessions start at randomized turns.

    Returns:
        ``(skip_first, skip_last)`` request counts to exclude from stats.
    """
    skip_first = orig_skip_first
    skip_last = orig_skip_last

    if request_rate != float("inf"):
        # Finite rate → steady drip with no ramp-up / ramp-down artifacts,
        # so skip nothing (PERF-878).
        if skip_first is None:
            skip_first = 0
        if skip_last is None:
            skip_last = 0
    elif max_concurrency is not None and max_concurrency > 1:
        if skip_first is None:
            skip_first = max_concurrency
            logger.info(
                f"Auto-setting skip_first_n_requests={skip_first}"
                f" (max_concurrency={max_concurrency})"
            )
        if skip_last is None:
            skip_last = max_concurrency
            logger.info(
                f"Auto-setting skip_last_n_requests={skip_last}"
                f" (max_concurrency={max_concurrency})"
            )
    # max_concurrency=1 → sequential requests, no ramp-up to trim.
    # max_concurrency=None → no cap; default to 0.
    # Both leave auto values unset → fall through to 0 below.
    if skip_first is None:
        skip_first = 0
    if skip_last is None:
        skip_last = 0

    if ignore_first_turn_stats and skip_first and not warmup_to_steady_state:
        # Without --warmup-to-steady-state, sessions all start at turn 0,
        # so --ignore-first-turn-stats already drops the same head requests
        # that --skip-first-n-requests would target. Combining them just
        # trims deeper into the run than the user asked for.
        # With --warmup-to-steady-state, sessions begin at randomized turn
        # offsets, so the two features filter different requests and we
        # want them to compose.
        logger.warning(
            "--ignore-first-turn-stats and --skip-first-n-requests both set"
            " without --warmup-to-steady-state. --ignore-first-turn-stats"
            " already drops every session's first turn, so"
            " --skip-first-n-requests would trim deeper than expected."
            " Ignoring --skip-first-n-requests."
        )
        skip_first = 0

    return skip_first, skip_last


def resolve_seed(args: ServingBenchmarkConfig) -> None:
    """Draws and records a random seed when one was not pinned.

    ``args.seed`` defaults to a fixed value so scheduled and repeated runs are
    reproducible. A caller can opt into a fresh draw with ``--seed none`` (which
    sets ``args.seed`` to ``None``); this resolves that draw to a concrete value
    *before* the workload is sampled, so the run stays reproducible from the
    recorded seed.

    Args:
        args: The benchmark configuration, whose ``seed`` is resolved in
            place.
    """
    if args.seed is None:
        args.seed = int(np.random.default_rng().integers(0, 10000))
        logger.info("Drew a fresh random seed=%d", args.seed)
    else:
        logger.info("Using seed=%d", args.seed)


def seed_for_concurrency(seed: int, max_concurrency: int | None) -> int:
    """Derives a per-concurrency-level seed from the base seed.

    Each concurrency level gets its own request sample instead of replaying
    the same requests as every other level, while staying reproducible from
    the single base seed.

    Args:
        seed: The run's base seed.
        max_concurrency: The concurrency level to seed.

    Returns:
        The level's seed.
    """
    digest = hashlib.sha256(f"{seed}:{max_concurrency}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def report_and_validate(
    result: BenchmarkResult,
    *,
    request_rate: float,
    max_concurrency: int | None,
    collect_gpu_stats: bool,
    collect_cpu_stats: bool,
    lora_manager: LoRABenchmarkManager | None = None,
) -> bool:
    """Prints an iteration's summary and validates its metrics.

    Args:
        result: The iteration's result.
        request_rate: The iteration's request rate.
        max_concurrency: The iteration's concurrency limit.
        collect_gpu_stats: Whether to print the GPU statistics section.
        collect_cpu_stats: Whether to print the CPU statistics section.
        lora_manager: The LoRA manager, when the run benchmarks adapters.

    Returns:
        Whether the result passed validation.
    """
    print_benchmark_summary(
        metrics=result,
        request_rate=request_rate,
        max_concurrency=max_concurrency,
        # TODO: no caller measures the achieved request rate,
        # so a paced run's summary always reports 0.0 for it.
        achieved_request_rate=0.0,
        collect_gpu_stats=collect_gpu_stats,
        collect_cpu_stats=collect_cpu_stats,
        lora_manager=lora_manager,
    )

    ok, validation_errors = result.validate_metrics()
    if not ok:
        for err in validation_errors:
            logger.error(f"Benchmark result validation failed: {err}")
        logger.info("finished benchmark run: Failed.")
        return False
    logger.info("finished benchmark run: Success.")
    return True


def retain_result(
    result: BenchmarkResult, *, passed: bool, always_save_result: bool
) -> BenchmarkResult | None:
    """Returns the result to persist for an iteration, if any.

    A result that failed validation is dropped unless the caller opted into
    keeping it.

    Args:
        result: The iteration's result.
        passed: Whether the result passed validation.
        always_save_result: Whether to keep the result even when it failed.

    Returns:
        ``result`` when it passed or ``always_save_result`` is set, else
        ``None``.
    """
    return result if passed or always_save_result else None
