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
"""Collective deadlock shapes, driven by the real driver.

Same scenario as test_kernel_load_gate.mojo, with a different call in the
window. Device 0 runs a kernel that fills the GPU and spins until device 1
releases it, standing in for a collective's barrier. A lane pinned to device
1's launch worker then makes one call into device 0's context. If the driver
blocks that call until device 0 goes idle, the lane parks the worker, device
1 never launches, and device 0 spins until its bound expires.

Nothing here instruments the runtime. The device really is busy and the call
really is the driver's, so the control flow is production's. The spins are
bounded, so a regression fails on elapsed time rather than wedging the GPU.

`BLOCKING_CALL` picks the call:

- `control`: the lane sleeps but calls nothing. Must always pass.
- `device_free`: allocates and frees a device 0 buffer. The free records an
  event on device 0's context, which DRIV-462 suspected of waiting. It does
  not: the round passes, and stays as a tripwire in case that changes.
- `pageable_copy`: a device-to-host copy into pageable memory (DRIV-462). The
  copy waits out device 0's whole spin bound, so this round deadlocks today.

Each round prints a mark around every step, so a failure says which call
blocked rather than only that one did.
"""

from std.os import getenv
from std.runtime._asyncrt import TaskGroup
from std.testing import assert_true
from std.time import global_perf_counter_ns, monotonic, sleep

from max.runtime.asyncrt import task_id_for_device

from comm.device_collective import _launch_device_collective
from comm.sync import enable_p2p
from max.gpu.host import DeviceBuffer, DeviceContext, FuncAttribute

# How long device 0 waits for its peer before giving up. Long enough that a
# regression is unambiguous, short enough to leave the GPU usable afterwards.
comptime SPIN_TIMEOUT_NS = 20_000_000_000

# The lane's call starts here: inside the window, after device 0 is certainly
# resident.
comptime CALLER_DELAY_SEC = 0.1

# A healthy run finishes shortly after CALLER_DELAY_SEC. A run whose launch was
# starved cannot finish before SPIN_TIMEOUT_NS, when device 0 gives up.
comptime MAX_HEALTHY_SEC = 10.0

# Large enough that device 1's launch must opt in through cuFuncSetAttribute,
# as the peer launch did in the captured deadlock, but safely under the
# per-block cap on every target.
comptime OPT_IN_SMEM_BYTES = UInt32(64 * 1024)

comptime BUFFER_ELEMS = 1024

comptime _CONTROL = 0
comptime _DEVICE_FREE = 1
comptime _PAGEABLE_COPY = 2


def _spin_until_released(flag: MutPointer[Int32, MutAnyOrigin]):
    """Occupy the whole GPU until a peer sets `flag`, or the bound expires.

    Filling the device is the point: a one-block kernel leaves enough of the
    GPU idle that a concurrent call can still find its quiesce window.
    """
    var start = global_perf_counter_ns()
    while flag.unsafe_load[volatile=True]() == 0:
        if global_perf_counter_ns() - start > UInt64(SPIN_TIMEOUT_NS):
            return


def _release(flag: MutPointer[Int32, MutAnyOrigin]):
    """Peer arrival: releases device 0's spin over P2P."""
    flag.unsafe_store[volatile=True](Int32(1))


struct _Clock(ImplicitlyCopyable, Movable):
    """Elapsed-time source, and the trace that reports the interleaving."""

    var _start_ns: Int

    def __init__(out self):
        self._start_ns = monotonic()

    def mark(self, label: StaticString):
        """Prints `label` stamped with the time since the run began."""
        print(
            "[",
            Float64(monotonic() - self._start_ns) / 1.0e6,
            "ms ] ",
            label,
            sep="",
        )

    def elapsed_sec(self) -> Float64:
        """Seconds since the run began."""
        return Float64(monotonic() - self._start_ns) / 1.0e9


def _require_two_peer_gpus() raises:
    """Skips out unless two GPUs can reach each other's memory."""
    assert_true(
        DeviceContext.number_of_devices() > 1, "must have multiple GPUs"
    )
    assert_true(enable_p2p(), "failed to enable P2P access between GPUs")


def _create_release_flag(ctx: DeviceContext) raises -> DeviceBuffer[.int32]:
    """Allocates the flag device 0 spins on and device 1 sets over P2P."""
    var flag_buf = ctx.enqueue_create_buffer[.int32](1)
    ctx.enqueue_memset(flag_buf, Int32(0))
    ctx.synchronize()
    return flag_buf


def _enqueue_spinner(
    ctx: DeviceContext, flag: MutPointer[Int32, MutAnyOrigin], clock: _Clock
) raises:
    """Saturates device 0 until its peer arrives, modelling a barrier wait."""
    comptime hw_info = type_of(ctx).default_device_info
    clock.mark("dev0: enqueueing spinner")
    ctx.enqueue_function[_spin_until_released](
        flag,
        grid_dim=hw_info.sm_count,
        block_dim=hw_info.max_thread_block_size,
    )
    clock.mark("dev0: spinner enqueued")


def _enqueue_release(
    ctx: DeviceContext, flag: MutPointer[Int32, MutAnyOrigin], clock: _Clock
) raises:
    """Releases device 0's spin, modelling the last peer reaching the barrier.

    Takes the shared-memory opt-in so this launch also reaches
    cuFuncSetAttribute, as the peer launch did in the captured deadlock.
    """
    clock.mark("dev1: enqueueing release")
    ctx.enqueue_function[_release](
        flag,
        grid_dim=1,
        block_dim=1,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            OPT_IN_SMEM_BYTES
        ),
    )
    clock.mark("dev1: release enqueued")


def _make_call[
    call: Int
](ctx0: DeviceContext, src: DeviceBuffer[.int32], clock: _Clock):
    """Makes the round's call into device 0's context, which is saturated.

    Errors are printed rather than raised: the round fails on elapsed time,
    and a raise here would only hide which call blocked.
    """
    try:
        comptime if call == _DEVICE_FREE:
            clock.mark("lane: allocating on dev0")
            var scratch = ctx0.enqueue_create_buffer[.int32](BUFFER_ELEMS)
            clock.mark("lane: allocated, freeing")
            _ = scratch^
            clock.mark("lane: freed")
        elif call == _PAGEABLE_COPY:
            var host = List[Int32](length=BUFFER_ELEMS, fill=0)
            clock.mark("lane: starting pageable DtoH copy")
            ctx0.enqueue_copy(host.unsafe_ptr(), src)
            clock.mark("lane: copy returned, synchronizing")
            ctx0.synchronize()
            clock.mark("lane: synchronized")
            _ = host^
    except e:
        print("lane: call raised: ", e)


def _spin_and_release(
    ctx0: DeviceContext,
    ctx1: DeviceContext,
    flag: MutPointer[Int32, MutAnyOrigin],
    clock: _Clock,
) -> Optional[Error]:
    """Runs the two-device collective whose launch window the call lands in.

    Returns the error instead of raising it, because the caller has to join
    the lane first: unwinding with that task still pending aborts on its
    non-available AsyncValue and buries whatever the collective reported.
    """

    @inline(.always)
    def launch[index: Int]() raises {imm ctx0, imm ctx1, imm flag, imm clock}:
        comptime if index == 0:
            _enqueue_spinner(ctx0, flag, clock)
        else:
            _enqueue_release(ctx1, flag, clock)

    clock.mark("collective: opening")
    var error = Optional[Error]()
    try:
        _launch_device_collective[2](launch, [ctx0, ctx1])
    except e:
        error = e^
    clock.mark("collective: closed")
    return error^


def _assert_peer_arrived(
    ctx: DeviceContext, flag_buf: DeviceBuffer[.int32]
) raises:
    """Checks device 1 released the spin rather than device 0 timing out."""
    var host_flag = ctx.enqueue_create_host_buffer[.int32](1)
    flag_buf.enqueue_copy_to(host_flag)
    ctx.synchronize()
    assert_true(
        host_flag.unsafe_ptr()[0] == 1,
        (
            "peer never released the spin: its launch was stranded behind a"
            " blocking driver call parked on its worker"
        ),
    )


def _assert_no_stall(clock: _Clock) raises:
    """Checks the run finished near the lane's delay, not the spin bound."""
    var elapsed_sec = clock.elapsed_sec()
    assert_true(
        elapsed_sec < MAX_HEALTHY_SEC,
        String(
            "collective took ",
            elapsed_sec,
            "s; the lane's driver call blocked until the spin bound expired",
        ),
    )


def _run_round[call: Int](ctx0: DeviceContext, ctx1: DeviceContext) raises:
    """Lands the round's call in the collective's launch window.

    Parameters:
        call: Which call the lane makes into device 0's context.
    """
    var flag_buf = _create_release_flag(ctx0)
    var flag = flag_buf.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()

    var src = ctx0.enqueue_create_buffer[.int32](BUFFER_ELEMS)
    ctx0.enqueue_memset(src, Int32(7))
    ctx0.synchronize()

    var clock = _Clock()

    # The call has to be in flight before the collective opens its window, and
    # on the worker that device 1's launch will be pinned to. Affinity tasks
    # sit on a private queue only their own worker dequeues, so a worker parked
    # in a blocking driver call starves the launch queued behind it. The sleep
    # holds that worker until device 0 is saturated, landing the call
    # mid-fan-out, before the peer has launched. See DRIV-462.
    var tg = TaskGroup()
    # Resolved before the coroutine exists: a raise between the two would
    # abandon it, and Coroutine is not implicitly destroyable.
    var caller_worker = task_id_for_device(Int(ctx1.id()))

    @inline(.always)
    __async def lane() {var} -> None:
        sleep(CALLER_DELAY_SEC)
        comptime if call != _CONTROL:
            _make_call[call](ctx0, src, clock)
        else:
            clock.mark("lane: no call")

    tg._create_task(lane(), desired_worker_id=caller_worker)

    var collective_error = _spin_and_release(ctx0, ctx1, flag, clock)
    tg.wait()
    clock.mark("lane: joined")
    if collective_error:
        raise collective_error.take()

    ctx0.synchronize()
    ctx1.synchronize()
    clock.mark("devices synchronized")

    _assert_peer_arrived(ctx0, flag_buf)
    _assert_no_stall(clock)
    _ = src^


def main() raises:
    _require_two_peer_gpus()

    var round_name = getenv("BLOCKING_CALL", "control")
    print("round:", round_name)

    var ctx0 = DeviceContext(device_id=0)
    var ctx1 = DeviceContext(device_id=1)

    if round_name == "control":
        _run_round[_CONTROL](ctx0, ctx1)
    elif round_name == "device_free":
        _run_round[_DEVICE_FREE](ctx0, ctx1)
    elif round_name == "pageable_copy":
        _run_round[_PAGEABLE_COPY](ctx0, ctx1)
    else:
        raise Error(String("unknown BLOCKING_CALL: ", round_name))
