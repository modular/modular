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
"""Sustained-load energy and throughput for benchmark regions.

Compute-bound kernels on a power-capped GPU run at whatever clock the board
power limit allows, and the limit takes tens of milliseconds to engage. A
short timing window measures the burst clock; a window of seconds measures
the steady state, where throughput is set by energy per op. This module
measures the steady state: it replays a workload for a few seconds and reads
the driver's cumulative energy counter at its update ticks. It also times
short bursts and single replays that start from idle, before the averaged
power limit engages, and lists the kernels a replay launches.
"""

from __future__ import annotations

import ctypes
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import torch

_NVML_SUCCESS = 0


class _Nvml:
    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        if self.lib.nvmlInit_v2() != _NVML_SUCCESS:
            raise OSError("nvmlInit failed")
        self.handle = ctypes.c_void_p()
        # NVML ignores CUDA_VISIBLE_DEVICES, so select the GPU by UUID.
        uuid = f"GPU-{torch.cuda.get_device_properties(0).uuid}".encode()
        if (
            self.lib.nvmlDeviceGetHandleByUUID(uuid, ctypes.byref(self.handle))
            != _NVML_SUCCESS
        ):
            raise OSError("nvmlDeviceGetHandleByUUID failed")

    def energy_mj(self) -> int:
        value = ctypes.c_ulonglong()
        if (
            self.lib.nvmlDeviceGetTotalEnergyConsumption(
                self.handle, ctypes.byref(value)
            )
            != _NVML_SUCCESS
        ):
            raise OSError("nvmlDeviceGetTotalEnergyConsumption failed")
        return value.value


_nvml: _Nvml | None = None


def _get_nvml() -> _Nvml | None:
    global _nvml
    if _nvml is None:
        try:
            _nvml = _Nvml()
        except (OSError, AttributeError):
            return None
    return _nvml


@dataclass
class Sustained:
    us_per_op: float
    mj_per_op: float
    mean_w: float
    seconds: float


class _TickRecorder(threading.Thread):
    """Polls the energy counter off the launch thread; records each update.

    An NVML query can block for milliseconds, which would drain the launch
    queue if the launching thread made it. ctypes releases the GIL for the
    call, so polling here does not stall launches.
    """

    def __init__(self, nvml: _Nvml) -> None:
        super().__init__(daemon=True)
        self.nvml = nvml
        self.ticks: list[tuple[float, int]] = []
        self.stop = threading.Event()

    def run(self) -> None:
        last = self.nvml.energy_mj()
        while not self.stop.is_set():
            if (value := self.nvml.energy_mj()) != last:
                self.ticks.append((time.perf_counter(), value))
                last = value
            time.sleep(0.001)


def time_bursts(
    replay: Callable[[], object],
    ops_per_replay: int,
    replays: int,
    repeats: int,
    sleep_s: float = 0.0,
    label: str = "",
) -> list[float]:
    """Returns microseconds per op for each of `repeats` bursts.

    A burst is `replays` back-to-back replays, timed on the host with a device
    synchronize at both ends. `sleep_s` of idle GPU before each burst lets it
    start below the power limit. A non-empty `label` names an NVTX range around
    the bursts, so a profile can attribute kernels to the case.
    """
    per_op = []
    torch.cuda.synchronize()
    if label:
        torch.cuda.nvtx.range_push(label)
    for _ in range(repeats):
        torch.cuda.synchronize()
        time.sleep(sleep_s)
        t0 = time.perf_counter()
        for _ in range(replays):
            replay()
        torch.cuda.synchronize()
        per_op.append(
            (time.perf_counter() - t0) * 1e6 / (replays * ops_per_replay)
        )
    if label:
        torch.cuda.nvtx.range_pop()
    return per_op


def replay_kernels(replay: Callable[[], object]) -> list[str]:
    """Returns the distinct kernel names one replay launches.

    An autotuner can pick a different kernel in each process, so a benchmark
    records what actually ran next to its time.
    """
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        replay()
        torch.cuda.synchronize()
    return sorted(
        {
            e.name
            for e in prof.events()
            if e.device_type == torch.autograd.DeviceType.CUDA
        }
    )


def cold_bursts(
    replay: Callable[[], object],
    ops_per_replay: int,
    samples: int,
    idle_s: float = 0.25,
) -> list[float]:
    """Times `samples` single replays, each after `idle_s` of idle GPU.

    One replay of a few milliseconds that starts from idle finishes before the
    averaged power limit engages; a fast current limit can still slow it.
    Returns microseconds per op for each sample, from device events.
    """
    per_op = []
    for _ in range(samples):
        torch.cuda.synchronize()
        time.sleep(idle_s)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        # Holds the stream while the host enqueues the replay, so the events
        # time the replay's device work and not its launch.
        torch.cuda._sleep(1_000_000)
        start.record()
        replay()
        end.record()
        end.synchronize()
        per_op.append(start.elapsed_time(end) * 1e3 / ops_per_replay)
    return per_op


def sustained(
    replay: Callable[[], object],
    ops_per_replay: int,
    seconds: float,
    queue_s: float = 0.05,
) -> Sustained | None:
    """Replays for about `seconds` after a warm-up of the same length.

    Returns the steady-state time and energy per op, or None when the energy
    counter is unavailable. The host keeps about `queue_s` of work queued so
    the GPU never idles; mean power comes from the energy-counter ticks inside
    the window.
    """
    nvml = _get_nvml()
    if nvml is None or seconds <= 0:
        return None

    torch.cuda.synchronize()
    t = time.perf_counter()
    replay()
    torch.cuda.synchronize()
    depth = max(2, math.ceil(queue_s / (time.perf_counter() - t)))

    def run_for(duration: float) -> tuple[int, float]:
        pending: deque[torch.cuda.Event] = deque()
        replays = 0
        start = time.perf_counter()
        while time.perf_counter() - start < duration:
            replay()
            event = torch.cuda.Event()
            event.record()
            pending.append(event)
            replays += 1
            while len(pending) > depth:
                pending.popleft().synchronize()
        torch.cuda.synchronize()
        return replays, time.perf_counter() - start

    run_for(seconds)
    recorder = _TickRecorder(nvml)
    recorder.start()
    replays, elapsed = run_for(seconds)
    recorder.stop.set()
    recorder.join()
    # Drop the first tick: it can straddle the start of the window.
    ticks = recorder.ticks[1:]
    if len(ticks) < 2:
        return None
    (t0, e0), (t1, e1) = ticks[0], ticks[-1]
    mean_w = (e1 - e0) / 1e3 / (t1 - t0)
    us_per_op = elapsed * 1e6 / (replays * ops_per_replay)
    return Sustained(
        us_per_op=us_per_op,
        mj_per_op=mean_w * us_per_op / 1e3,
        mean_w=mean_w,
        seconds=elapsed,
    )
