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
"""Tests how a model worker handles SIGTERM."""

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
# With tracing on, it queues a span whose export hangs.
_WORKER = """
import asyncio, atexit, signal, sys, time
import uvloop
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from max.serve.config import Settings
from max.serve.telemetry._tracing import configure_kernel_tracing

class Exporter(SpanExporter):
    def export(self, spans):
        print(spans[0].name, flush=True)
        time.sleep(10)

async def main():
    if sys.argv[2]:
        trace.set_tracer_provider(provider := TracerProvider())
        provider.add_span_processor(BatchSpanProcessor(Exporter(), schedule_delay_millis=60000))
    configure_kernel_tracing(Settings())
    trace.get_tracer("test").start_span("span").end()
    atexit.register(lambda: open(sys.argv[1], "w").close())
    print("ready", signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL, flush=True)
    await asyncio.sleep(60)

uvloop.run(main())
"""

# Blocks in a C exit hook, where the trace is written, after finalization has
# reset Python signal handlers. The marker is registered before
# configure_kernel_tracing, so it prints after SIGTERM is ignored.
_EXITING_WORKER = """
import atexit, ctypes
from max.serve.config import Settings
from max.serve.telemetry._tracing import configure_kernel_tracing

libc = ctypes.CDLL(None)
# glibc doesn't export atexit, but it and macOS both export __cxa_atexit.
cxa_atexit = libc["__cxa_atexit"]
cxa_atexit.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
assert cxa_atexit(ctypes.cast(libc.getchar, ctypes.c_void_p), None, None) == 0
atexit.register(print, "exiting", flush=True)
configure_kernel_tracing(Settings())
"""


@pytest.mark.parametrize(
    "level, tracing",
    [("off", False), ("op", False), ("kernel", False), ("off", True)],
)
def test_sigterm_runs_atexit_only_at_kernel_level(
    tmp_path: Path, level: str, tracing: bool
) -> None:
    exits_through_atexit = level == "kernel"
    handles_sigterm = tracing or exits_through_atexit
    # The profiler plugin writes the libkineto trace from an atexit hook.
    marker = tmp_path / "atexit-ran"
    env = {**os.environ, "MAX_SERVE_KERNEL_TRACE_LEVEL": level}
    with subprocess.Popen(
        [sys.executable, "-c", _WORKER, str(marker), "on" if tracing else ""],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as worker:
        assert worker.stdout is not None and worker.stderr is not None
        ready = worker.stdout.readline()
        worker.send_signal(signal.SIGTERM)
        # The API process SIGKILLs the worker 5 s after SIGTERM.
        stdout, stderr = worker.communicate(timeout=5)
    assert ready == f"ready {handles_sigterm}\n", stderr
    if exits_through_atexit:
        assert worker.returncode == 128 + signal.SIGTERM, stderr
    else:
        assert worker.returncode == -signal.SIGTERM, stderr
    assert marker.exists() == exits_through_atexit
    assert stdout == ("span\n" if tracing else ""), stderr


@pytest.mark.parametrize(
    "level, survives_sigterm",
    [("off", False), ("kernel", True)],
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
