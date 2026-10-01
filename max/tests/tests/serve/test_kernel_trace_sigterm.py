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
"""Tests that a kernel-level model worker runs its atexit hooks on SIGTERM."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Mirrors the model worker: configure_kernel_tracing runs inside uvloop.run on
# the main thread, and the API process then stops the worker with SIGTERM.
_WORKER = """
import asyncio, atexit, sys
import uvloop
from max.serve.config import Settings
from max.serve.telemetry.common import configure_kernel_tracing

async def main():
    configure_kernel_tracing(Settings())
    atexit.register(lambda: open(sys.argv[1], "w").close())
    print("ready", flush=True)
    await asyncio.sleep(60)

uvloop.run(main())
"""

# Blocks in a C exit hook, where the trace is written, after finalization has
# reset Python signal handlers. The marker is registered before
# configure_kernel_tracing, so it prints after SIGTERM is ignored.
_EXITING_WORKER = """
import atexit, ctypes
from max.serve.config import Settings
from max.serve.telemetry.common import configure_kernel_tracing

libc = ctypes.CDLL(None)
# glibc doesn't export atexit, but it and macOS both export __cxa_atexit.
cxa_atexit = libc["__cxa_atexit"]
cxa_atexit.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
assert cxa_atexit(ctypes.cast(libc.getchar, ctypes.c_void_p), None, None) == 0
atexit.register(print, "exiting", flush=True)
configure_kernel_tracing(Settings())
"""


@pytest.mark.parametrize(
    "level, exits_through_atexit",
    [("off", False), ("op", False), ("kernel", True)],
)
def test_sigterm_runs_atexit_only_at_kernel_level(
    tmp_path: Path, level: str, exits_through_atexit: bool
) -> None:
    # The profiler plugin writes the libkineto trace from an atexit hook.
    marker = tmp_path / "atexit-ran"
    env = {**os.environ, "MAX_SERVE_KERNEL_TRACE_LEVEL": level}
    with subprocess.Popen(
        [sys.executable, "-c", _WORKER, str(marker)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as worker:
        assert worker.stdout is not None and worker.stderr is not None
        ready = worker.stdout.readline()
        assert ready == "ready\n", worker.stderr.read()
        worker.send_signal(signal.SIGTERM)
        _, stderr = worker.communicate(timeout=60)
    if exits_through_atexit:
        assert worker.returncode == 128 + signal.SIGTERM, stderr
    else:
        assert worker.returncode == -signal.SIGTERM, stderr
    assert marker.exists() == exits_through_atexit


@pytest.mark.parametrize(
    "level, survives_sigterm",
    [("off", False), ("op", False), ("kernel", True)],
)
def test_sigterm_during_exit_only_interrupts_below_kernel_level(
    level: str, survives_sigterm: bool
) -> None:
    # At kernel level, a SIGTERM that arrives during exit must not cut the
    # trace write short.
    env = {**os.environ, "MAX_SERVE_KERNEL_TRACE_LEVEL": level}
    with subprocess.Popen(
        [sys.executable, "-c", _EXITING_WORKER],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as worker:
        assert worker.stdout is not None and worker.stderr is not None
        exiting = worker.stdout.readline()
        assert exiting == "exiting\n", worker.stderr.read()
        # Gives the child time to reach the C exit hook.
        time.sleep(1)
        worker.send_signal(signal.SIGTERM)
        # Closing stdin releases the C exit hook.
        _, stderr = worker.communicate(timeout=60)
    if survives_sigterm:
        assert worker.returncode == 0, stderr
    else:
        assert worker.returncode == -signal.SIGTERM, stderr
