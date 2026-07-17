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
"""Dynolog on-demand profiling harness (MXTOOLS-190).

Long-running MAX process used to end-to-end verify the libkineto <-> Dynolog
daemon handshake that docs/internal/Profiling.md flags as NOT yet verified.

What it does:
  1. Brings libkineto + its Dynolog IPC listener up via ONE start()/stop()
     cycle. The first start() runs libkineto_init(), which (with
     KINETO_USE_DAEMON=1 in the env) connects libkineto IpcFabricConfigClient
     to the dynolog daemon and registers this PID. The segfault-fix
     (init-once) keeps the profiler proxy + IPC/config-poll thread alive after
     stop(), so the process sits "registered-but-idle".
  2. Idles in a loop of real CUDA add kernels so an on-demand
     `dyno gputrace --pids <pid>` request has live GPU activity to capture,
     while MAX itself holds NO trace. The dynolog-driven trace is written by
     libkineto to the --log-file dyno passes, independent of MAX.

The idle loop runs in one of two modes (HARNESS_MODE env var):
  - "eager" (default): model.execute() per iteration — eager kernel
    launches, the path verified for MXTOOLS-190.
  - "replay": the model is captured into a CUDA graph BEFORE libkineto is
    ever initialized, and the loop replays that graph exec. This is the
    MXTOOLS-266 scenario: an on-demand trace must capture kernels launched
    from a graph exec that was instantiated while CUPTI was cold (verified
    to work on CUDA 13.1 — CUPTI no longer requires subscribing before
    cuGraphInstantiate). The loop also reports a median per-iteration
    latency at each heartbeat so disarm-overhead reversion is observable.

Run order (see run script):
  - start dynolog daemon FIRST (IPC client connects at init)
  - KINETO_USE_DAEMON=1 run this harness
  - dyno gputrace --pids <pid> --log-file <out>
  - touch the stop file to exit
"""

import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from max._core.profiler import kineto_can_record, kineto_is_enabled
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

PID_FILE = Path(os.environ.get("HARNESS_PID_FILE", "/tmp/dyno_harness.pid"))
STOP_FILE = Path(os.environ.get("HARNESS_STOP_FILE", "/tmp/dyno_harness.stop"))
INIT_TRACE = os.environ.get(
    "HARNESS_INIT_TRACE", "/tmp/harness_init_trace.json"
)
MODE = os.environ.get("HARNESS_MODE", "eager")
# Fail fast on a typo'd mode: "replay" is the documented re-validation repro
# for CUDA toolkit bumps (docs/internal/Profiling.md), and silently falling
# back to eager would make that re-validation vacuously pass.
if MODE not in ("eager", "replay"):
    sys.exit(
        f"[harness] unknown HARNESS_MODE: {MODE!r} (want 'eager' or 'replay')"
    )
GRAPH_KEY = 1


def build_tiny_add_graph() -> Graph:
    t = TensorType(
        dtype=DType.float32,
        shape=["batch", "channels"],
        device=DeviceRef.GPU(0),
    )
    with Graph("dyno_harness_add", input_types=(t, t)) as g:
        g.output(ops.add(g.inputs[0], g.inputs[1]))
    return g


def main() -> None:
    pid = os.getpid()
    print(
        f"[harness] pid={pid} KINETO_USE_DAEMON={os.environ.get('KINETO_USE_DAEMON')}",
        flush=True,
    )
    dev = Accelerator()
    session = InferenceSession(devices=[dev])
    if not kineto_can_record():
        print("[harness] FATAL kineto_can_record()=False", flush=True)
        sys.exit(2)

    session.debug.profiling_output_path = INIT_TRACE
    model = session.load(build_tiny_add_graph())
    # Inputs must be device-resident: Model.execute wraps a numpy array as a
    # *host* Buffer and does not auto-transfer, so move them to GPU(0) once.
    a = Buffer.from_numpy(np.ones((4, 8), dtype=np.float32)).to(dev)
    b = Buffer.from_numpy(np.full((4, 8), 2.0, dtype=np.float32)).to(dev)

    # In replay mode, instantiate the CUDA graph BEFORE libkineto ever runs,
    # so the on-demand trace must capture kernels from a pre-existing exec.
    captured_outputs = None
    if MODE == "replay":
        captured_outputs = model.capture(GRAPH_KEY, a, b)
        model.replay(GRAPH_KEY, a, b)
        print("[harness] CUDA graph captured with CUPTI cold", flush=True)

    # (1) Bring libkineto + Dynolog IPC listener up, then go idle.
    session.profiling.start()
    model.execute(a, b)
    session.profiling.stop()
    session.profiling.wait_for_trace()
    print(
        f"[harness] libkineto up; init trace={INIT_TRACE} exists={Path(INIT_TRACE).exists()}"
        f" kineto_is_enabled={kineto_is_enabled()} (expect False)",
        flush=True,
    )

    PID_FILE.write_text(str(pid))
    if STOP_FILE.exists():
        STOP_FILE.unlink()

    print(
        f"[harness] IDLE. trigger: dyno gputrace --pids {pid} --log-file /tmp/dyno_trace.json",
        flush=True,
    )
    print(f"[harness] touch {STOP_FILE} to exit", flush=True)

    i = 0
    window: list[float] = []
    while not STOP_FILE.exists():
        t0 = time.perf_counter()
        # The range is a no-op while no trace is live; during a dyno-driven
        # window it must appear as a named user_annotation span (MXTOOLS-266).
        with session.profiling.range("harness-iter"):
            if MODE == "replay":
                model.replay(GRAPH_KEY, a, b)
            else:
                model.execute(a, b)
        window.append(time.perf_counter() - t0)
        i += 1
        if i % 400 == 0:
            median_us = statistics.median(window) * 1e6
            print(
                f"[harness] heartbeat iter={i} mode={MODE}"
                f" median_iter={median_us:.1f}us"
                f" kineto_enabled={kineto_is_enabled()}",
                flush=True,
            )
            window.clear()
        time.sleep(0.005)
    print(f"[harness] stop after {i} iters", flush=True)
    del captured_outputs


if __name__ == "__main__":
    main()
