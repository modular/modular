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

"""Run a suite of sweep serving benchmarks."""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from collections.abc import Callable, Generator, Iterable, Sequence
from datetime import datetime
from pathlib import Path

from max.benchmark.benchmark_serving import BenchmarkRunResult
from max.benchmark.benchmark_serving import (
    main_with_parsed_args as benchmark_serving_main,
)
from max.benchmark.benchmark_serving import (
    parse_args as _parse_serving_args,
)
from max.benchmark.benchmark_shared.config import (
    ServingBenchmarkConfig,
)
from max.benchmark.benchmark_shared.model_csv import CsvStreamWriter
from max.benchmark.benchmark_shared.serving_result_output import (
    save_output_lengths,
    save_result_json,
)
from max.benchmark.sweep_benchmark_serving_result_utils import (
    SweepUploader,
    validate_sweep_serving_percentiles,
)

DESCRIPTION = "Run a suite of sweep serving benchmarks."

logger = logging.getLogger("sweep-benchmark-serving")


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

    if config.upload_results and not config.workload_config:
        logger.warning(
            "--workload-config is not set while --upload-results is set. "
            "Run results will be recorded, but will not include any workload name, "
            "and may not be picked up by dashboards."
        )

    run_sweep(config, uploader=uploader)


def run_sweep(
    config: ServingBenchmarkConfig,
    *,
    uploader: SweepUploader | None = None,
    report_result: Callable[[BenchmarkRunResult], None] | None = None,
    server_liveness: Callable[[], bool] | None = None,
) -> list[BenchmarkRunResult]:
    """Runs a sweep and persists it.

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
        server_liveness: Optional predicate forwarded to
            :func:`benchmark_serving.main_with_parsed_args`. When the
            orchestrator launched the server it passes a process-liveness
            check so the server-ready wait aborts promptly on a crashed
            bring-up instead of polling until the timeout.

    Returns:
        The per-iteration :class:`BenchmarkRunResult` list returned by
        :func:`persist_sweep`, so legacy callers that haven't migrated to
        ``report_result`` can still iterate the whole sweep at once.
    """
    return persist_sweep(
        config,
        benchmark_serving_main(config, server_liveness=server_liveness),
        uploader=uploader,
        report_result=report_result,
    )


def persist_sweep(
    config: ServingBenchmarkConfig,
    iterations: Iterable[BenchmarkRunResult],
    *,
    uploader: SweepUploader | None = None,
    report_result: Callable[[BenchmarkRunResult], None] | None = None,
) -> list[BenchmarkRunResult]:
    """Persists and reports each iteration of a sweep as it is produced.

    For each iteration with a result, writes ``config.result_filename``, the
    output lengths, a per-concurrency result JSON, a CSV row, and the
    optional upload, then invokes ``report_result``. An iteration that
    failed validation ends the process with exit code 1 after it is
    persisted and reported. ``iterations`` is closed whenever the sweep
    stops early, so a generator's ``with`` blocks unwind before the exit.

    Args:
        config: The sweep's configuration. A missing ``log_dir`` is created
            and recorded on it before the first iteration is drawn, so a
            lazy ``iterations`` can write under it.
        iterations: The sweep's iterations, typically a generator, so each
            is persisted as soon as it completes.
        uploader: Invoked with each iteration's result JSON path when
            ``config.upload_results`` is set.
        report_result: Invoked with each iteration's result as soon as it is
            produced.

    Returns:
        Every iteration's result.

    Raises:
        SystemExit: If an iteration did not pass validation.
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

    # ---- CSV output + upload ----
    # Stream per-iteration: each iteration is persisted, uploaded, and
    # reported as it is drawn. Don't materialize the iterator here — batching
    # at end would defeat the streaming guarantees of
    # ``utils/benchmarking/results_publication`` (live BigQuery rows during
    # the run + partial results surviving a mid-sweep crash).
    results: list[BenchmarkRunResult] = []
    with contextlib.ExitStack() as stack:
        if isinstance(iterations, Generator):
            stack.enter_context(contextlib.closing(iterations))
        # results.csv streams one row per iteration (all columns) as that
        # iteration's JSON blob is written, so a crash mid-sweep still leaves
        # the completed rows on disk. The per-concurrency JSON blobs remain
        # the source of truth; the authoritative all-columns superset can be
        # regenerated from them via results_to_csv if a later iteration
        # surfaces new columns. Without a log directory (a dry run without
        # --log-dir) there is no CSV, per-concurrency JSON, or upload.
        csv_writer: CsvStreamWriter | None = None
        if log_dir is not None:
            csv_writer = stack.enter_context(
                CsvStreamWriter(log_dir / "results.csv")
            )
        for run in iterations:
            results.append(run)
            save_iteration_result(config, run)
            if (
                log_dir is not None
                and csv_writer is not None
                and run.result is not None
            ):
                json_path = (
                    log_dir / f"results-{run.max_concurrency}-median.json"
                )
                save_result_json(
                    str(json_path),
                    run.labels,
                    run.result,
                    request_rate=run.request_rate,
                    record_max_concurrency=run.max_concurrency,
                )
                csv_writer.write_result(json_path)
                if config.upload_results and uploader is not None:
                    uploader.upload(str(json_path))

            # Stream the row to the results-publication reporter, if the caller
            # wired one in — so a crash mid-sweep still leaves rows 1..N-1
            # published downstream.
            if report_result is not None:
                report_result(run)
            if run.status != "pass":
                sys.exit(1)

    if log_dir is not None:
        result_file_path = (log_dir / "results.csv").resolve()
        logger.info(
            f"All concurrency sweep results have been written to: {result_file_path}"
        )
    return results


def save_iteration_result(
    config: ServingBenchmarkConfig, run: BenchmarkRunResult
) -> None:
    """Writes an iteration's ``config.result_filename`` and output lengths.

    Writes nothing for an iteration without a result.

    Args:
        config: The sweep's configuration, naming the files to write.
        run: The iteration to persist.
    """
    if run.result is None:
        return
    save_result_json(
        config.result_filename,
        run.labels,
        run.result,
        request_rate=run.request_rate,
        record_max_concurrency=run.max_concurrency,
    )
    save_output_lengths(
        config,
        run.result,
        run.labels.benchmark_task,
        iteration_config=(run.max_concurrency, run.request_rate),
    )


if __name__ == "__main__":
    main()
