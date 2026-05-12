#!/usr/bin/env python3
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

"""Run a suite of sweep serving benchmarks.

A thin orchestration layer on top of :func:`benchmark_serving.main_with_parsed_args`
that adds a log directory, a CSV writer, and an optional caller-supplied
:class:`SweepUploader` for per-iteration result-JSON upload.  The actual
range iteration, workload YAML loading, ``num_iters`` / median selection,
and prefix-cache flushing are all handled inside ``benchmark_serving``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

import yaml
from max.benchmark.benchmark_serving import (
    BenchmarkRunResult,
    save_result_json,
)
from max.benchmark.benchmark_serving import (
    main_with_parsed_args as benchmark_serving_main,
)
from max.benchmark.benchmark_serving import (
    parse_args as _parse_serving_args,
)
from max.benchmark.benchmark_shared.config import ServingBenchmarkConfig
from max.benchmark.sweep_benchmark_serving_result_utils import (
    LLMBenchmarkResult,
    LLMBenchmarkResultWriter,
    SweepServingBenchmarkResult,
    SweepUploader,
    TextToImageBenchmarkResult,
    TextToImageBenchmarkResultWriter,
    validate_sweep_serving_percentiles,
)

DESCRIPTION = "Run a suite of sweep serving benchmarks."

logger = logging.getLogger("sweep-benchmark-serving")


def _build_sweep_result(
    result: BenchmarkRunResult,
    percentiles: list[int],
    *,
    is_pixel_gen: bool = False,
) -> SweepServingBenchmarkResult:
    """Convert a :class:`BenchmarkRunResult` to the CSV-writable form."""
    if result.result is None:
        if is_pixel_gen:
            return TextToImageBenchmarkResult.zeros(percentiles)
        return LLMBenchmarkResult.zeros(percentiles)
    metrics = result.result.metrics
    if metrics.task_type == "pixel":
        return TextToImageBenchmarkResult.from_metrics(metrics, percentiles)
    return LLMBenchmarkResult.from_metrics(metrics, percentiles)


def parse_args(
    args: Sequence[str] | None = None,
    *,
    app_name: str = "sweep-benchmark-serving",
    description: str = DESCRIPTION,
) -> ServingBenchmarkConfig:
    """Parse command line arguments into a ServingBenchmarkConfig."""
    return _parse_serving_args(
        args,
        app_name=app_name,
        description=description,
    )


def main(
    args: Sequence[str] | None = None,
    *,
    uploader: SweepUploader | None = None,
    app_name: str = "sweep-benchmark-serving",
    description: str = DESCRIPTION,
) -> None:
    """CLI entry point.

    Args:
        args: CLI argv (defaults to ``sys.argv[1:]``).
        uploader: Optional :class:`SweepUploader` invoked with the
            per-iteration result JSON path whenever
            ``--upload-results`` is set and the run is not a dry run.
            Intentionally not wired by default so this module has no
            hard dependency on any specific result-ingestion backend.
        app_name: Name shown in ``--help`` output.
        description: Description shown in ``--help`` output.
    """
    try:
        config = parse_args(args, app_name=app_name, description=description)
    except SystemExit as e:
        if e.code == 0:
            return
        raise

    if not config.model:
        raise SystemExit("error: the following arguments are required: --model")
    # TODO(PAQ-2397): upload_results doesn't have to require workload_config.
    if config.upload_results and not config.workload_config:
        raise SystemExit(
            "error: --workload-config is required when --upload-results is set"
        )

    run_sweep(config, uploader=uploader)


def run_sweep(
    config: ServingBenchmarkConfig,
    *,
    uploader: SweepUploader | None = None,
    report_result: Callable[[BenchmarkRunResult], None] | None = None,
) -> list[BenchmarkRunResult]:
    """Set up CSV + upload infrastructure and delegate benchmarking to the library.

    The actual range iteration, workload YAML loading, num_iters / median
    selection, and prefix-cache flushing are all handled by
    :func:`benchmark_serving.main_with_parsed_args`.  This function provides
    the thin orchestration layer on top: log directory, CSV writer, and
    optional uploader invocation.

    Args:
        config: Parsed :class:`ServingBenchmarkConfig`.
        uploader: Optional :class:`SweepUploader` invoked with the
            per-iteration result JSON path.  Only consulted when
            ``config.upload_results`` is True and the run is not a dry
            run.
        report_result: Optional callback invoked once per sweep iteration as
            soon as that iteration's :class:`BenchmarkRunResult` is
            produced. Used by the unified ``benchmark_serving`` binary to
            stream rows into ``utils/benchmarking/results_publication``
            during the run rather than after — preserves "live progress"
            visibility and ensures partial results survive a mid-sweep
            crash.

    Returns:
        The per-iteration :class:`BenchmarkRunResult` list produced by
        :func:`benchmark_serving.main_with_parsed_args`, so legacy callers
        that haven't migrated to ``report_result`` can still iterate the
        whole sweep at once.
    """
    if config.upload_results and config.cluster_information_path is None:
        logger.warning("Warning: uploading results without cluster information")

    # ---- Log directory ----
    # Skip auto-creating a sweep-serving-* directory under --dry-run when
    # the user didn't explicitly ask for one. Tests that pass --log-dir
    # still get a real on-disk CSV.
    if config.log_dir:
        log_dir = Path(config.log_dir)
        os.makedirs(log_dir, exist_ok=True)
        config.log_dir = str(log_dir)
        print(f"Saving logs to: {log_dir}")
    elif not config.dry_run:
        timestamp = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
        log_dir = Path(f"sweep-serving-{timestamp}")
        os.makedirs(log_dir, exist_ok=True)
        config.log_dir = str(log_dir)
        print(f"Saving logs to: {log_dir}")
    else:
        log_dir = None

    # ---- Percentiles ----
    percentiles = [
        int(x.strip()) for x in config.latency_percentiles.split(",")
    ]
    validate_sweep_serving_percentiles(percentiles)

    # Peek at workload YAML for task type (needed to choose the CSV writer
    # class before the benchmark runs).
    is_pixel_gen = False
    if config.workload_config:
        with open(config.workload_config) as f:
            wl = yaml.safe_load(f)
        is_pixel_gen = wl.get("benchmark-task") in (
            "text-to-image",
            "image-to-image",
            "text-to-video",
        )

    upload_active = uploader is not None and config.upload_results

    # ---- CSV output + upload ----
    # Stream per-iteration: ``benchmark_serving_main`` yields one result per
    # ``(max_concurrency, request_rate)`` step, and we drive CSV writes,
    # upload side effects, and the optional ``report_result`` callback as
    # each yields. Don't materialize the iterator here — batching at end
    # would defeat the streaming guarantees of
    # ``utils/benchmarking/results_publication`` (live BigQuery rows during
    # the run + partial results surviving a mid-sweep crash).
    results: list[BenchmarkRunResult] = []

    # No log directory configured (dry-run without --log-dir): skip CSV
    # and per-result JSON output but still drive the benchmarks.
    if log_dir is None:
        for result in benchmark_serving_main(config):
            results.append(result)
            if report_result is not None:
                report_result(result)
        return results

    results_csv_path = log_dir / "results.csv"
    writer_cls = (
        TextToImageBenchmarkResultWriter
        if is_pixel_gen
        else LLMBenchmarkResultWriter
    )
    result_writer = writer_cls(
        results_csv_path,
        percentiles=percentiles,
        collect_gpu_stats=config.collect_gpu_stats,
        uploader=uploader if upload_active else None,
    )

    with result_writer:
        for result in benchmark_serving_main(config):
            results.append(result)
            # Save per-concurrency JSON with full metrics.
            json_path: str | None = None
            if result.result is not None:
                assert config.model is not None
                json_path = str(
                    log_dir / f"results-{result.max_concurrency}-median.json"
                )
                save_result_json(
                    json_path,
                    config,
                    result.result,
                    benchmark_task=config.benchmark_task,
                    model_id=config.model,
                    tokenizer_id=config.tokenizer or config.model,
                    request_rate=result.request_rate,
                    record_max_concurrency=result.max_concurrency,
                )

            sweep_result = _build_sweep_result(
                result, percentiles, is_pixel_gen=is_pixel_gen
            )
            if upload_active:
                sweep_result.result_filename = json_path

            result_writer.write_row(
                max_concurrency=result.max_concurrency,
                request_rate=result.request_rate,
                num_prompts=result.num_prompts,
                result=sweep_result,
            )

            # Stream the row to the results-publication reporter, if the
            # caller wired one in. Fires here — after CSV/upload side
            # effects, before the next iteration begins — so a crash
            # mid-sweep still leaves rows 1..N-1 published downstream.
            if report_result is not None:
                report_result(result)

    result_file_path = results_csv_path.resolve()
    logger.info(
        f"All concurrency sweep results have been written to: {result_file_path}"
    )
    return results


if __name__ == "__main__":
    main()
