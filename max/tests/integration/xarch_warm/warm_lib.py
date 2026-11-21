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
"""Shared producer loop for the split eager-interpreter warm.

The warm is split into Bazel actions, one per (op family, device class), so the
~20-min CPU sweep runs once per host arch instead of once per GPU lane (see
``BUILD.bazel``). Each action writes a :class:`~max.engine.MefStore` directory
holding one artifact per (family, device slot); consumers union the
directories (see ``gc_compile.warm_store``).

Kept deliberately import-light: it must NOT import ``max._interpreter_ops`` at
module top. Those modules freeze their device set from ``accelerator_count()``
at import, so the producers must set the virtual-device parameters FIRST; the
deferred import below therefore runs inside the function, after the caller has
set them.
"""

from __future__ import annotations

import importlib
import multiprocessing
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from max.driver import Device
from max.engine import InferenceSession, MefStore

# How many (family, device) jobs run at once. Each job is a graph-compiler run
# of its own that uses ~2.5 cores and ~2.5 GiB, so the default cap keeps a
# 32-core lane from oversubscribing. ``MODULAR_WARM_JOBS`` overrides it (set it
# to 1 to run the jobs sequentially).
_MAX_JOBS_ENV = "MODULAR_WARM_JOBS"
_DEFAULT_MAX_JOBS = 8


def _max_workers() -> int:
    override = os.environ.get(_MAX_JOBS_ENV)
    if override:
        return max(1, int(override))
    return max(1, min(os.cpu_count() or 1, _DEFAULT_MAX_JOBS))


def _virtual_device_params() -> tuple[int, str, str, str]:
    """Returns this process's virtual-device configuration.

    A worker process starts fresh, so it has to re-apply what the producer set
    before importing the op families, whose device set is frozen at import.
    """
    from max import driver

    return (
        driver.get_virtual_device_count(),
        driver.get_virtual_device_api(),
        driver.get_virtual_device_target_arch(),
        driver.get_virtual_cpu_target(),
    )


def _apply_virtual_device_params(params: tuple[int, str, str, str]) -> None:
    from max import driver

    count, api, arch, cpu_target = params
    if cpu_target:
        driver.set_virtual_cpu_target(cpu_target)
    if count:
        driver.set_virtual_device_api(api)
        driver.set_virtual_device_target_arch(arch)
        driver.set_virtual_device_count(count)


@dataclass(frozen=True)
class _MEFJob:
    """One (family, device) MEF to compile."""

    # The producer's virtual-device parameters, re-applied by the worker.
    virtual_device_params: tuple[int, str, str, str]
    family: str
    device_label: str
    device_id: int
    mef_path: str


def _device(interpreter_ops: ModuleType, label: str, device_id: int) -> Device:
    for device in interpreter_ops.gc_compile.DISCOVERED_DEVICES:
        if device.label == label and device.id == device_id:
            return device
    raise RuntimeError(f"no device {label}:{device_id} in this process")


def _compile_job(job: _MEFJob) -> tuple[str, str]:
    """Compiles and exports one job's MEF. Runs in a worker process.

    Returns:
        The job's MEF path, and the scratch derived path it was compiled in.
        The caller removes that path once the worker is gone: the compiler's
        threads outlive this function and recreate the cache while it runs, so
        a worker that deleted its own directory on the way out left it behind.
    """
    _apply_virtual_device_params(job.virtual_device_params)
    # A scratch cache per job: the workers must not share the compiler's cache
    # directory, and an action's store is its only output anyway.
    derived_path = tempfile.mkdtemp(prefix="warm-")
    os.environ["MODULAR_DERIVED_PATH"] = derived_path
    interpreter_ops = importlib.import_module("max._interpreter_ops")
    family = next(
        f for f in interpreter_ops.GC_FAMILIES if f.name == job.family
    )
    device = _device(interpreter_ops, job.device_label, job.device_id)
    module = family.build_module_for_device(device)
    if not module.top_level_graph_names():
        raise ValueError(
            f"{job.family} sweeps {device.label}:{device.id} but builds no"
            " graph for it; exclude it in sweep_devices()"
        )
    InferenceSession(devices=[device]).compile(module).export_mef(
        Path(job.mef_path)
    )
    return job.mef_path, derived_path


def export_slots(
    store: MefStore,
    *,
    include_cpu: bool,
    include_accelerators: bool,
    only_family: str | None = None,
) -> None:
    """Compiles and exports one artifact per (op family, kept device).

    For each op family, and each device that family sweeps, filtered to the
    requested device classes, build that slot's per-device module, compile it
    on a single-device session, and export it into *store* under the name
    adoption looks it up by.

    Per-slot artifacts (rather than one module spanning every device) keep the
    warm device-count-independent, and let ``warm_cpu`` and
    ``warm_accelerators`` each compile only their own device class. The slots are
    independent, so they compile in parallel worker processes.

    Args:
        store: The exporting store the artifacts go into.
        include_cpu: Warm the CPU slot.
        include_accelerators: Warm the accelerator slots.
        only_family: Warm only this GC family (a ``GC_FAMILIES`` name); ``None``
            warms all.

    Raises:
        ValueError: If ``only_family`` names no family, or a family sweeps a
            device it builds nothing for. Adoption loads an artifact for every
            device a family sweeps, so such a slot would fail every consumer.
    """
    # Deferred import: freezes each family's device set from the (already-set)
    # virtual-device parameters, then reads the shared GC_FAMILIES registry.
    interpreter_ops = importlib.import_module("max._interpreter_ops")
    gc_compile = interpreter_ops.gc_compile

    families = interpreter_ops.GC_FAMILIES
    if only_family is not None:
        known = {f.name for f in families}
        if only_family not in known:
            raise ValueError(
                f"unknown GC family {only_family!r}; known: {sorted(known)}"
            )
        families = tuple(f for f in families if f.name == only_family)

    virtual_device_params = _virtual_device_params()
    jobs: list[_MEFJob] = []
    for family in families:
        for device in family.sweep_devices():
            is_cpu = device.label == "cpu"
            if is_cpu and not include_cpu:
                continue
            if not is_cpu and not include_accelerators:
                continue
            name = gc_compile.warm_artifact_name(family.name, device)
            jobs.append(
                _MEFJob(
                    virtual_device_params=virtual_device_params,
                    family=family.name,
                    device_label=device.label,
                    device_id=device.id,
                    mef_path=str(store.claim_export_artifact(name)),
                )
            )

    workers = min(len(jobs), _max_workers())
    scratch: list[str] = []
    try:
        if workers <= 1:
            for job in jobs:
                _, derived_path = _compile_job(job)
                scratch.append(derived_path)
        else:
            # Spawn, not fork: the producer has already started the compiler's
            # threads, and a child that forks them deadlocks.
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=context
            ) as pool:
                futures = [pool.submit(_compile_job, job) for job in jobs]
                for future in as_completed(futures):
                    _, derived_path = future.result()
                    scratch.append(derived_path)
    finally:
        # Only here, once the workers are gone, does removing the scratch caches
        # stick: their compiler threads write to them after a job returns.
        for derived_path in scratch:
            shutil.rmtree(derived_path, ignore_errors=True)
    store.write_manifest()
