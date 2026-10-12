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
"""Races a broadcast's per-device launches against kernel loads on two GPUs.

A cold load holds its context's lock until its GPU stops spinning in the
broadcast, so a call into that context on the late GPU's launch path deadlocks.
"""

# One step's deadlock, time running down (ctx N is GPU N's driver context):
#
#   GPU 0's worker                    GPU 1's launch path
#   --------------------------------  ---------------------------------------
#   launches its broadcast share;
#   GPU 0 spins until GPU 1's arrives
#   cold-loads its next kernel: takes
#   ctx 0's lock, waits for GPU 0 to
#   stop spinning
#                                     host_free: GPU 1's worker frees GPU 0's
#                                     pinned staging; the free records an
#                                     event on ctx 0 and waits for its lock
#                                     peer_copy: the copy thread holds ctx 1
#                                     and waits for ctx 0; GPU 1's worker
#                                     waits for ctx 1
#   (never returns)                   GPU 1's share never launches, so GPU 0
#                                     never stops spinning
#
# The bypass cases make every launch a cold load; the shape-kernel case keeps
# the function cache on and cold-loads a new kernel variant per token bucket.

from __future__ import annotations

import ctypes
import faulthandler
import math
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Generator, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from max.driver import (
    CPU,
    Accelerator,
    Buffer,
    DeviceQueue,
    Usage,
    accelerator_count,
    batch_inplace_copy,
)
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    ops,
)
from max.nn import Signals

# Seeds the batch-shape generator. Random by default so repeated runs explore
# different step sequences; each run prints it, and `STRESS_SEED` replays one.
SEED = int(
    os.environ.get("STRESS_SEED", random.SystemRandom().randrange(2**32))
)

# Mirrors the multi-turn benchmark client: `NUM_SESSIONS` is its
# `max_concurrency`, sessions start on Poisson arrivals, and each turn sends a
# new prompt, decodes a reply, then thinks before the next turn.
NUM_SESSIONS = 32
MEAN_ARRIVAL_STEPS = 4.0
PROMPT_TOKENS = (64, 1024)
REPLY_TOKENS = (16, 128)
THINK_STEPS = (0, 20)
# Chunked-prefill cap on one step's tokens, and the granularity prefill sizes
# are rounded to so the per-shape buffer cache stays small.
MAX_STEP_TOKENS = 8192
PREFILL_ROUNDING = 128

DURATION_SEC = float(os.environ.get("STRESS_DURATION_SEC", "300"))
# A healthy run ends shortly after its steps, so anything past this is a wedged
# process rather than a slow one. The margin also covers compile and leaves a
# hung process alive long enough for an external debugger capture.
HANG_TIMEOUT_SEC = DURATION_SEC + float(
    os.environ.get("STRESS_HANG_MARGIN_SEC", "1200")
)
# How often the step loop reports its count, so a hang's onset shows in the log.
HEARTBEAT_SEC = 5.0

# Width of each step's activations.
HIDDEN = 1024
# Most trailing tokens per sequence a step picks, as spec decode verifies its
# drafts plus the bonus token.
MAX_PICKS = 4

# The copy thread's onload requests: one KV block per copy, and enough blocks
# per request that a copy is usually in flight when a load stalls.
COPY_BYTES = 2 << 20
COPY_BLOCKS = 32
CU_EVENT_DISABLE_TIMING = 0x2

# Shape-kernel case: each step's token bucket picks one of the shape ops'
# kernel variants, so a step that reaches a new bucket cold-loads it with the
# function cache on, as a shape-dispatched GEMM does in serving. Each GPU chains
# this many shape ops before the broadcasts and as many after.
SHAPE_POSITIONS = 8
# Seconds over which the step-token cap grows to `MAX_STEP_TOKENS`, so larger
# buckets, and their cold loads, keep arriving as a concurrency sweep brings new
# batch sizes.
TOKEN_RAMP_SEC = 480.0

# Seconds without a new step, once steps begin, that count as a hang: the
# watchdog then captures and exits rather than waiting out the deadline. 0
# turns it off for runs that attach their own debugger.
STALL_SEC = float(os.environ.get("STRESS_STALL_SEC", "60"))
# Bounds each capture tool, so a stuck tracer cannot hold the process stopped.
CAPTURE_TIMEOUT_SEC = 300
# prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY): lets `eu-stack`, a child of this
# process, attach to it under Yama ptrace_scope 1.
PR_SET_PTRACER = 0x59616D61
PR_SET_PTRACER_ANY = 2**64 - 1


@pytest.fixture(autouse=True)
def hang_watchdog() -> Generator[None, None, None]:
    # A thread wedged in a driver call cannot be interrupted from Python, so
    # dump every thread's stack and exit instead of waiting out the bazel
    # timeout.
    faulthandler.dump_traceback_later(HANG_TIMEOUT_SEC, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


def _allow_ptrace_by_children() -> None:
    try:
        ctypes.CDLL(None).prctl(
            PR_SET_PTRACER, ctypes.c_ulong(PR_SET_PTRACER_ANY), 0, 0, 0
        )
    except (OSError, AttributeError):
        pass


def _capture_hang() -> None:
    """Records the GPUs' state and every thread's native stack."""
    pid = str(os.getpid())
    out_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    for name, argv in (
        (
            "nvidia-smi",
            [
                "nvidia-smi",
                "--query-gpu=index,power.draw,utilization.gpu,memory.used",
                "--format=csv",
            ],
        ),
        ("eu-stack", ["eu-stack", "-p", pid]),
    ):
        if shutil.which(argv[0]) is None:
            text = f"{argv[0]} not on PATH\n"
        else:
            # SIGKILL only after SIGTERM has had 30 s, so a tracer can detach
            # rather than leave this process stopped.
            result = subprocess.run(
                ["timeout", "-k", "30", str(CAPTURE_TIMEOUT_SEC), *argv],
                capture_output=True,
                text=True,
                check=False,
            )
            text = result.stdout + result.stderr
        print(
            f"===== hang capture: {name} =====\n{text}",
            file=sys.stderr,
            flush=True,
        )
        if out_dir:
            with open(os.path.join(out_dir, f"hang-{name}.txt"), "w") as f:
                f.write(text)


class _StallWatchdog:
    """Captures the hang and exits once steps stop advancing for `STALL_SEC`.

    `execute` releases the GIL while it waits, so this thread keeps running
    through a hang.
    """

    def __init__(self) -> None:
        self.steps = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)

    def __enter__(self) -> _StallWatchdog:
        if STALL_SEC > 0:
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()

    def _watch(self) -> None:
        seen, since = self.steps, time.monotonic()
        while not self._stop.wait(1.0):
            if self.steps != seen:
                seen, since = self.steps, time.monotonic()
            elif time.monotonic() - since >= STALL_SEC:
                print(
                    f"hang: no step completed for {STALL_SEC:.0f}s"
                    f" after {seen} steps",
                    file=sys.stderr,
                    flush=True,
                )
                try:
                    _capture_hang()
                finally:
                    faulthandler.dump_traceback(all_threads=True)
                    os._exit(1)


@dataclass
class _Step:
    tokens: int
    sequences: int


@dataclass
class _Session:
    start: int
    decode_left: int = 0
    think_left: int = 0


def _steps(rng: random.Random) -> Iterator[_Step]:
    """Yields the batch shape a continuous-batching scheduler would run."""
    sessions: list[_Session] = []
    start = 0.0
    for _ in range(NUM_SESSIONS):
        sessions.append(_Session(start=int(start)))
        start += rng.expovariate(1.0 / MEAN_ARRIVAL_STEPS)

    step = 0
    while True:
        tokens = sequences = prefill = 0
        for s in sessions:
            if step < s.start:
                continue
            if s.think_left > 0:
                s.think_left -= 1
                continue
            if s.decode_left == 0:
                prefill += rng.randint(*PROMPT_TOKENS)
                s.decode_left = rng.randint(*REPLY_TOKENS)
            else:
                tokens += 1
                s.decode_left -= 1
                if s.decode_left == 0:
                    s.think_left = rng.randint(*THINK_STEPS)
            sequences += 1
        prefill = math.ceil(prefill / PREFILL_ROUNDING) * PREFILL_ROUNDING
        tokens = min(tokens + prefill, MAX_STEP_TOKENS)
        step += 1
        if sequences:
            yield _Step(tokens=tokens, sequences=sequences)


def _ramped(step: _Step, elapsed_sec: float) -> _Step:
    """Caps the step's tokens on a ramp that reaches `MAX_STEP_TOKENS`."""
    cap = int(MAX_STEP_TOKENS * min(1.0, elapsed_sec / TOKEN_RAMP_SEC))
    return _Step(
        tokens=max(step.sequences, min(step.tokens, cap)),
        sequences=step.sequences,
    )


def _rms_norm(x: TensorValue) -> TensorValue:
    return x * ops.rsqrt(ops.mean(x * x, axis=-1) + 1e-6)


def _step_graph(shape_positions: int = 0) -> Graph:
    """One step on two GPUs: per-GPU compute around GPU 0's broadcasts."""
    devices = [DeviceRef.GPU(i) for i in range(2)]
    input_types: list[TensorType | BufferType] = [
        TensorType(DType.float32, ["tokens", HIDDEN], device=devices[0]),
        *(
            TensorType(DType.float32, ["tokens", HIDDEN], device=d)
            for d in devices
        ),
        *(TensorType(DType.uint32, ["batch"], device=d) for d in devices),
        TensorType(DType.int64, [1], device=DeviceRef.CPU()),
    ]
    input_types.extend(Signals(devices=devices).input_types())
    extensions = (
        [Path(os.environ["STRESS_SHAPE_OPS_PATH"])] if shape_positions else []
    )
    with Graph(
        "stress_step", input_types=input_types, custom_extensions=extensions
    ) as graph:
        it = iter(graph.inputs)
        x = next(it).tensor
        hidden = [next(it).tensor for _ in devices]
        cache_lengths = [next(it).tensor for _ in devices]
        picks = next(it).tensor
        signal_buffers = [v.buffer for v in it]
        for k in range(shape_positions):
            hidden = [
                _shape_kernel(h, d, k)
                for h, d in zip(hidden, devices, strict=True)
            ]

        xs, tail = _hang_window(x, picks, signal_buffers)

        outputs: list[TensorValue] = []
        for i in range(len(devices)):
            y = ops.concat([xs[i], _rms_norm(hidden[i])], axis=-1)
            # Consumes the cache lengths so the step waits on their copy, as a
            # model's attention waits on its KV inputs.
            y = y + ops.cast(cache_lengths[i][0:1], DType.float32)
            y = ops.tanh(y[:, :HIDDEN] + y[:, HIDDEN:])
            for k in range(shape_positions):
                y = _shape_kernel(y, devices[i], shape_positions + k)
            outputs += [y, tail[i]]
        graph.output(*outputs)
    return graph


def _hang_window(
    x: TensorValue, picks: TensorValue, signal_buffers: Sequence[BufferValue]
) -> tuple[list[TensorValue], list[TensorValue]]:
    """Broadcasts GPU 0's activations, then a range fed only by a host scalar.

    GPU 0 loads the range kernel while its share of the first broadcast still
    spins for GPU 1's, so a cold load there holds GPU 0's context lock.
    """
    xs = ops.distributed_broadcast(_rms_norm(x), signal_buffers)
    tail = ops.distributed_broadcast(
        ops.range(
            picks[0], 0, -1, out_dim="picks", dtype=DType.int64, device=x.device
        ),
        signal_buffers,
    )
    return xs, tail


def _shape_kernel(x: TensorValue, device: DeviceRef, salt: int) -> TensorValue:
    return ops.custom(
        "shape_kernel",
        device=device,
        values=[x],
        out_types=[TensorType(DType.float32, x.shape, device=device)],
        parameters={"salt": salt},
    )[0].tensor


class _StepInputs:
    """Builds a step's inputs."""

    def __init__(self, devices: Sequence[Accelerator]) -> None:
        self._devices = devices
        self._cache_lengths = [
            Buffer(shape=(NUM_SESSIONS,), dtype=DType.uint32, device=d)
            for d in devices
        ]

    def args(self, step: _Step, stage: bool) -> list[Buffer]:
        """Returns the step's inputs.

        With `stage`, cache lengths go through fresh pinned staging on GPU 0, as
        the graph input stager does.
        """
        leader = self._devices[0]
        cache_lengths = [b[: step.sequences] for b in self._cache_lengths]
        if stage:
            host = Buffer(
                shape=(step.sequences,),
                dtype=DType.uint32,
                device=leader,
                usage=Usage.STAGING | Usage.UNTRACKED,
            )
            host.to_numpy()[:] = np.arange(step.sequences, dtype=np.uint32)
            batch_inplace_copy(cache_lengths, [host] * len(cache_lengths))
        picks = min(MAX_PICKS, step.tokens // step.sequences)
        return [
            Buffer.zeros((step.tokens, HIDDEN), DType.float32, device=leader),
            *(
                Buffer.zeros((step.tokens, HIDDEN), DType.float32, device=d)
                for d in self._devices
            ),
            *cache_lengths,
            Buffer.from_numpy(np.array([picks], dtype=np.int64)),
        ]


def _cuda_check(result: int, call: str) -> None:
    if result != 0:
        raise RuntimeError(f"{call} failed with CUresult {result}")


def _stream_context(
    cuda: ctypes.CDLL, stream: ctypes.c_void_p
) -> ctypes.c_void_p:
    context = ctypes.c_void_p()
    _cuda_check(
        cuda.cuStreamGetCtx(stream, ctypes.byref(context)), "cuStreamGetCtx"
    )
    return context


class _ComputeGate:
    """Snapshots every device's compute queue between steps for the copy thread.

    As in the KV tier connector's `ComputeGate`, the step thread records it and
    the copy thread's onload streams wait on it, so no copy waits on a step
    still launching.
    """

    def __init__(
        self, cuda: ctypes.CDLL, devices: Sequence[Accelerator]
    ) -> None:
        self._cuda = cuda
        self._compute = [
            ctypes.c_void_p(d.default_queue.native_stream_handle)
            for d in devices
        ]
        self._events: list[ctypes.c_void_p] = []
        for stream in self._compute:
            _cuda_check(
                cuda.cuCtxPushCurrent_v2(_stream_context(cuda, stream)),
                "cuCtxPushCurrent",
            )
            event = ctypes.c_void_p()
            _cuda_check(
                cuda.cuEventCreate(
                    ctypes.byref(event), CU_EVENT_DISABLE_TIMING
                ),
                "cuEventCreate",
            )
            _cuda_check(
                cuda.cuCtxPopCurrent_v2(ctypes.byref(ctypes.c_void_p())),
                "cuCtxPopCurrent",
            )
            self._events.append(event)
        self._requested = threading.Event()
        self._recorded = threading.Event()

    def record_if_requested(self) -> None:
        """Records the snapshot the copy thread asked for, between steps."""
        if not self._requested.is_set():
            return
        self._requested.clear()
        for event, stream in zip(self._events, self._compute, strict=True):
            _cuda_check(
                self._cuda.cuEventRecord(event, stream), "cuEventRecord"
            )
        self._recorded.set()

    def wait(
        self, streams: Sequence[ctypes.c_void_p], stop: threading.Event
    ) -> bool:
        """Gates `streams` on the next snapshot; returns False once stopped."""
        self._recorded.clear()
        self._requested.set()
        while not self._recorded.wait(0.1):
            if stop.is_set():
                return False
        for event, stream in zip(self._events, streams, strict=True):
            _cuda_check(
                self._cuda.cuStreamWaitEvent(stream, event, 0),
                "cuStreamWaitEvent",
            )
        return True

    def close(self) -> None:
        for event in self._events:
            _cuda_check(self._cuda.cuEventDestroy_v2(event), "cuEventDestroy")


class _CrossContext:
    """Makes a thread on the late GPU call into the stalled GPU's context."""

    stages_inputs = False

    def __init__(self, devices: Sequence[Accelerator]) -> None:
        self._devices = devices

    def start(self) -> None:
        """Runs once, after the first step.

        Every load on that step is cold whatever the function cache does, so
        crossing contexts then forms a hang serving never sees.
        """

    def between_steps(self) -> None:
        """Runs on the step thread after each step."""

    def stop(self) -> None:
        """Runs at the end, whether or not `start` ran."""


class _HostFree(_CrossContext):
    """Has `_StepInputs` stage cache lengths in GPU 0's pinned memory.

    GPU 1's worker frees the staging, recording an event on GPU 0's context.
    """

    def start(self) -> None:
        self.stages_inputs = True


class _PeerCopy(_CrossContext):
    """Copies GPU 0 to its peer as the KV tier connector's onload lane does.

    The copy holds the peer's context while it waits on GPU 0's.
    """

    def __init__(self, devices: Sequence[Accelerator]) -> None:
        super().__init__(devices)
        self._gate = _ComputeGate(ctypes.CDLL("libcuda.so.1"), devices)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._copy_loop, daemon=True)

    def start(self) -> None:
        for d in self._devices:
            d.synchronize()
        self._thread.start()

    def between_steps(self) -> None:
        self._gate.record_if_requested()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join()
        self._gate.close()

    def _copy_loop(self) -> None:
        """Runs `cuMemcpyPeerAsync` on the peer's onload stream.

        It copies block by block behind one compute gate, synced after each.
        """
        cuda = ctypes.CDLL("libcuda.so.1")
        queues = [DeviceQueue(d) for d in self._devices]
        streams = [ctypes.c_void_p(q.native_stream_handle) for q in queues]
        contexts = [_stream_context(cuda, stream) for stream in streams]
        bufs = [
            Buffer.zeros((COPY_BYTES,), DType.uint8, device=d)
            for d in self._devices
        ]
        peers = range(1, len(self._devices))
        while self._gate.wait(streams, self._stop):
            for _ in range(COPY_BLOCKS):
                for k in peers:
                    _cuda_check(
                        cuda.cuMemcpyPeerAsync(
                            ctypes.c_uint64(bufs[k]._data_ptr()),
                            contexts[k],
                            ctypes.c_uint64(bufs[0]._data_ptr()),
                            contexts[0],
                            ctypes.c_size_t(COPY_BYTES),
                            streams[k],
                        ),
                        "cuMemcpyPeerAsync",
                    )
                for k in peers:
                    _cuda_check(
                        cuda.cuStreamSynchronize(streams[k]),
                        "cuStreamSynchronize",
                    )


CROSS_CONTEXTS: dict[str, type[_CrossContext]] = {
    "host_free": _HostFree,
    "peer_copy": _PeerCopy,
}


def _run_steps(
    session: InferenceSession,
    devices: Sequence[Accelerator],
    cross_context_type: type[_CrossContext],
    shape_positions: int = 0,
) -> int:
    signal_buffers = Signals(
        devices=[DeviceRef.GPU(i) for i in range(2)]
    ).buffers()
    load_start = time.monotonic()
    model: Model = session.load(_step_graph(shape_positions))
    print(
        f"model loaded in {time.monotonic() - load_start:.0f}s",
        file=sys.stderr,
        flush=True,
    )
    inputs = _StepInputs(devices)
    cross_context = cross_context_type(devices)
    print(
        f"seed {SEED}; replay with STRESS_SEED={SEED}",
        file=sys.stderr,
        flush=True,
    )
    _allow_ptrace_by_children()
    # Log marker for external hang watchers: steps start here.
    print(
        f"[ns {time.monotonic_ns()}] steps begin", file=sys.stderr, flush=True
    )
    ran = 0
    start = time.monotonic()
    heartbeat = start + HEARTBEAT_SEC
    try:
        with _StallWatchdog() as watchdog:
            for step in _steps(random.Random(SEED)):
                now = time.monotonic()
                if now >= start + DURATION_SEC:
                    break
                if now >= heartbeat:
                    print(
                        f"[ns {time.monotonic_ns()}] steps {ran}",
                        file=sys.stderr,
                        flush=True,
                    )
                    heartbeat = now + HEARTBEAT_SEC
                if shape_positions:
                    step = _ramped(step, now - start)
                model.execute(
                    *inputs.args(step, cross_context.stages_inputs),
                    *signal_buffers,
                )
                ran += 1
                watchdog.steps = ran
                if ran == 1:
                    cross_context.start()
                cross_context.between_steps()
    finally:
        cross_context.stop()
    return ran


# The function-cache bypass is per process, so each cold-load source is its own
# test, and its own bazel target.
@pytest.mark.parametrize("case", CROSS_CONTEXTS)
def test_steps_survive_kernel_loads(case: str) -> None:
    """Runs steps that cold-load every kernel as a thread crosses contexts."""
    if accelerator_count() < 2:
        pytest.skip("requires 2 GPUs")
    devices = [Accelerator(i) for i in range(2)]
    session = InferenceSession(devices=[CPU(), *devices])
    steps = _run_steps(session, devices, CROSS_CONTEXTS[case])
    print(f"{case} steps in {DURATION_SEC}s: {steps}")
    assert steps > 0


def test_steps_survive_shape_kernel_loads() -> None:
    """Runs `peer_copy` steps whose shape kernels cold-load, cache on."""
    if accelerator_count() < 2:
        pytest.skip("requires 2 GPUs")
    devices = [Accelerator(i) for i in range(2)]
    # A child compiles the graph into the compile cache, so this process loads
    # its model from the cache, as a server started on a cached model does. A
    # process that has just compiled cold-loads kernels faster and rarely hangs.
    subprocess.run(
        [sys.executable, __file__, "--compile-into-cache"],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(filter(None, sys.path)),
        },
        check=True,
    )
    session = InferenceSession(devices=[CPU(), *devices])
    steps = _run_steps(session, devices, _PeerCopy, SHAPE_POSITIONS)
    print(f"shape_kernels steps in {DURATION_SEC}s: {steps}")
    assert steps > 0


if __name__ == "__main__" and sys.argv[1:] == ["--compile-into-cache"]:
    InferenceSession(devices=[CPU(), Accelerator(0), Accelerator(1)]).load(
        _step_graph(SHAPE_POSITIONS)
    )
