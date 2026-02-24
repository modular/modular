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
"""Precompile-pipelines orchestrator for pipeline verification.

Reads the PIPELINES dict from verify_pipelines.py and PIPELINE_ORACLES from
create_pipelines.py to pre-compile each pipeline via the Python registry API,
running compilations in parallel.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing.sharedctypes import Synchronized

import click
from create_pipelines import PIPELINE_ORACLES
from max.pipelines.lib.config_enums import SupportedEncoding
from verify_pipelines import (
    PIPELINES,
    DeviceKind,
    TagFilter,
    TagFilterParamType,
)

logger = logging.getLogger(__name__)


@dataclass
class PrecompileJob:
    pipeline_name: str
    oracle_key: str
    encoding: SupportedEncoding
    devices: str
    target: str | None
    model_path: str


def pin_worker_to_cpus(
    counter: Synchronized, cpus_per_worker: int, num_workers: int
) -> None:
    """Pin this worker process to a dedicated CPU slice."""
    with counter.get_lock():
        wid = counter.value
        counter.value += 1
    # Modulo so recycled workers (max_tasks_per_child=1) reuse the same
    # CPU slots instead of going out of bounds.
    slot = wid % num_workers
    start = slot * cpus_per_worker
    end = start + cpus_per_worker
    try:
        os.sched_setaffinity(0, range(start, end))
    except (AttributeError, OSError) as e:
        logger.warning("CPU affinity pinning failed for worker %d: %s", wid, e)


def run_precompile_inprocess(job: PrecompileJob) -> tuple[bool, str, float]:
    """Compile a model in-process. Returns (success, output, elapsed_secs)."""
    import io
    import time

    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(
        logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    )
    # Replace inherited handlers so all logs go to our buffer, not real stderr.
    old_handlers = logging.root.handlers[:]
    logging.root.handlers = [handler]
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = buf

    t0 = time.monotonic()
    try:
        if job.target:
            from max.driver import (
                calculate_virtual_device_count_from_cli,
                set_virtual_device_api,
                set_virtual_device_count,
                set_virtual_device_target_arch,
            )
            from max.serve.config import parse_api_and_target_arch

            api, target_arch = parse_api_and_target_arch(job.target)
            set_virtual_device_api(api)
            set_virtual_device_target_arch(target_arch)
            devices_arg: str | list[int] = job.devices
            if job.devices.startswith("gpu:"):
                devices_arg = [int(x) for x in job.devices[4:].split(",")]
            set_virtual_device_count(
                calculate_virtual_device_count_from_cli(devices_arg)
            )

        from max.pipelines.lib.device_specs import (
            device_specs_from_normalized_device_handle,
            normalize_device_specs_input,
        )

        device_specs = device_specs_from_normalized_device_handle(
            normalize_device_specs_input(job.devices)
        )

        oracle = PIPELINE_ORACLES[job.oracle_key]
        oracle.create_max_pipeline(
            encoding=job.encoding,
            device_specs=device_specs,
        )
        return True, buf.getvalue(), time.monotonic() - t0

    except Exception:
        return (
            False,
            buf.getvalue() + traceback.format_exc(),
            time.monotonic() - t0,
        )
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        logging.root.handlers = old_handlers


def collect_precompile_jobs(
    *,
    devices: str,
    target: str | None,
    tag_filter: TagFilter,
    name_filter: str | None,
) -> list[PrecompileJob]:
    """Walk PIPELINES and build a PrecompileJob for each match."""
    jobs: list[PrecompileJob] = []
    seen: set[tuple[str, str, str | None]] = set()
    requested_devices = {DeviceKind(devices.split(":")[0].strip())}

    for pipeline_name, pipeline_def in PIPELINES.items():
        if not requested_devices & set(pipeline_def.compatible_with):
            continue
        if not tag_filter.satisfied_by(pipeline_def.tags):
            continue
        if name_filter and not any(
            f.strip().casefold() in pipeline_name.casefold()
            for f in name_filter.split(",")
            if f.strip()
        ):
            continue

        encoding = pipeline_def.encoding
        oracle = PIPELINE_ORACLES[pipeline_def.pipeline]
        model_path: str = oracle.model_path  # type: ignore[attr-defined]

        dedup_key = (model_path, devices, encoding)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        jobs.append(
            PrecompileJob(
                pipeline_name=pipeline_name,
                oracle_key=pipeline_def.pipeline,
                encoding=encoding,
                devices=devices,
                target=target,
                model_path=model_path,
            )
        )

    return jobs


def compile_models_in_parallel(
    jobs: list[PrecompileJob], cores_per_worker: int
) -> list[str]:
    """Execute precompile jobs in parallel, returning failed model paths."""
    total_cpus = os.cpu_count() or 1
    cpus_per_worker = min(cores_per_worker, total_cpus)
    workers = max(1, total_cpus // cpus_per_worker)
    multiprocessing.set_start_method("spawn", force=True)
    ctx = multiprocessing.get_context("spawn")
    counter = ctx.Value("i", 0)

    logger.info(
        "precompile: %d workers x %d cores each (%d of %d total cores)",
        workers,
        cpus_per_worker,
        workers * cpus_per_worker,
        total_cpus,
    )

    failed_models: list[str] = []
    # max_tasks_per_child is a valid Python 3.11+ parameter but typeshed
    # stubs don't model it in the overload that accepts initializer/initargs.
    with ProcessPoolExecutor(  # type: ignore[call-overload]
        max_workers=workers,
        mp_context=ctx,
        max_tasks_per_child=1,
        initializer=pin_worker_to_cpus,
        initargs=(counter, cpus_per_worker, workers),
    ) as executor:
        futures = {
            executor.submit(run_precompile_inprocess, job): job for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            success, output, elapsed = future.result()
            status = "OK" if success else "FAILED"
            title = f"{status} {job.model_path} ({elapsed:.0f}s)"
            print(f"::group::{title}", flush=True)
            if output.strip():
                print(output, flush=True)
            print("::endgroup::", flush=True)
            if not success:
                failed_models.append(job.model_path)

    return failed_models


@click.command()
@click.option(
    "--target",
    default=None,
    help="Compilation target (e.g. cuda:sm_90a).",
)
@click.option(
    "--tag-filter",
    "tag_filter",
    type=TagFilterParamType(),
    default=None,
    help="Tag filter expression (e.g. -manual,-no-h100).",
)
@click.option(
    "--devices",
    required=True,
    help="Device spec (e.g. cpu, gpu, gpu:0,1).",
)
@click.option(
    "--cores-per-worker",
    type=int,
    default=32,
    help="CPU cores allocated per parallel worker.",
)
@click.option(
    "--filter",
    "name_filter",
    default=None,
    help="Comma-separated pipeline name substrings.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print jobs without executing.",
)
def main(
    target: str | None,
    tag_filter: TagFilter | None,
    devices: str,
    cores_per_worker: int,
    name_filter: str | None,
    dry_run: bool,
) -> None:
    logging.basicConfig(level=logging.INFO)

    if tag_filter is None:
        tag_filter = TagFilter()

    jobs = collect_precompile_jobs(
        devices=devices,
        target=target,
        tag_filter=tag_filter,
        name_filter=name_filter,
    )

    logger.info(
        "precompile: %d models selected (target=%s, devices=%s)",
        len(jobs),
        target,
        devices,
    )

    if dry_run:
        for job in jobs:
            print(
                f"[DRY RUN] {job.oracle_key}"
                f" encoding={job.encoding}"
                f" devices={job.devices}"
                f" target={job.target}"
            )
        return

    logger.info("Starting parallel compilation of %d models", len(jobs))
    failed_models = compile_models_in_parallel(jobs, cores_per_worker)

    ok = len(jobs) - len(failed_models)
    logger.info("=" * 44)
    logger.info("precompile: %d/%d succeeded", ok, len(jobs))
    logger.info("=" * 44)

    if failed_models:
        sys.exit(1)


if __name__ == "__main__":
    if directory := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(directory)
    main()
