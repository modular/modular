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
"""Regression test for the kernel-load gate (MXSERV-441 / GEX-4227).

Reproduces the deadlock the gate exists to prevent. Device 0 runs a
GPU-filling kernel that waits for device 1 to release it, standing in for a
collective's barrier peer-wait, and an unrelated lane cold-loads a kernel
onto device 0 while it spins.

A first-time load blocks until its device quiesces, so that load cannot
return until device 1 arrives. The lane is pinned to the very worker device
1's launch task is pinned to, which is what closes the cycle: affinity tasks
sit on a private queue only their own worker dequeues, so a worker parked on
a blocking driver call starves the launch behind it and device 1 never
arrives. See GEX-3234 for the same shape reached through a one-thread
runtime, and DRIV-462 for the pinned-worker edge.

The gate breaks the cycle by making the load wait on an AsyncRT value rather
than parking the worker, so the donated worker still runs that launch.
Measured on 2xB200: each round finishes in 110-370ms with its gate, and in
20s without it, bounded only by the spin timeout below. Both spins are bounded
so a regression fails on elapsed time rather than wedging the GPU and timing
out.

Three calls block this way, each behind its own gate, and each gets its own
round with a fresh spinner: bringing up a vendor BLAS handle, a vendor
dispatch that first uses a GEMM kernel, which the library loads lazily, and a
first-time kernel load. Neither vendor call reaches `loadFunction`. Any one of
them starves the worker, so each has to be gated. They cannot share a window:
a gated wait donates the worker, the peer launches and the collective closes,
so whatever the lane does next runs on an idle device and proves nothing.

It also cannot see a gate whose state got duplicated by static linking: this
binary links one copy of everything, where a served process links five. Only
an end-to-end run across the real shared objects covers that, so keep the
gate's state in the driver object and off file scope. See KernelLoadGate.h.
"""

import linalg.matmul.vendor.blas as vendor_blas

from layout import TileTensor, row_major
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

# The unrelated load starts here: inside the window, after device 0 is
# certainly resident.
comptime LOADER_DELAY_SEC = 0.1

# A healthy run finishes shortly after LOADER_DELAY_SEC. A run whose launch
# was starved cannot finish before SPIN_TIMEOUT_NS, when device 0 gives up.
# Anything between the two is a regression.
comptime MAX_HEALTHY_SEC = 10.0

# Large enough that the load must opt in through cuFuncSetAttribute, which is
# the call that waits for the device to quiesce, but safely under the per-block
# cap on every target. The per-SM figure is not a legal per-block request.
comptime OPT_IN_SMEM_BYTES = UInt32(64 * 1024)

# Side of the square operands the dispatch round multiplies. The handle round
# dispatches nothing, so this is the first GEMM the library runs.
comptime MM = 2048

# The blocking call each round lands in the window.
comptime _HANDLE_BRING_UP = 0
comptime _VENDOR_DISPATCH = 1
comptime _COLD_LOAD = 2


def _spin_until_released(flag: MutPointer[Int32, MutAnyOrigin]):
    """Occupy the whole GPU until a peer sets `flag`, or the bound expires.

    Filling the device is the point: a one-block kernel leaves enough of the
    GPU idle that a concurrent load can still find its quiesce window, which
    is why the small barrier kernels never reproduced this.
    """
    var start = global_perf_counter_ns()
    while flag.load[volatile=True]() == 0:
        if global_perf_counter_ns() - start > UInt64(SPIN_TIMEOUT_NS):
            return


def _release(flag: MutPointer[Int32, MutAnyOrigin]):
    """Peer arrival: releases device 0's spin over P2P."""
    flag.store[volatile=True](Int32(1))


def _unrelated():
    """Stands in for any kernel a concurrent lane happens to load first."""
    pass


struct _Clock(ImplicitlyCopyable, Movable):
    """Elapsed-time source, and the trace that reports the interleaving.

    A failure is diagnosed from the order of these marks, so every step of
    the scenario emits one.
    """

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

    Takes the same shared-memory opt-in as the loader, so this launch also
    reaches cuFuncSetAttribute, as the peer launch did in the captured
    deadlock. It does not contend with the loader through the driver here: the
    driver's locks are per context, and the two run on different devices. What
    strands it is the loader parked on the worker this launch is pinned to.
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


def _bring_up_vendor_handle_on_busy_device(ctx: DeviceContext, clock: _Clock):
    """Brings up the vendor BLAS handle for `ctx` while it is saturated.

    Creating the handle blocks on a busy device like a kernel load does, but
    never reaches `loadFunction`. Nothing is dispatched, so the dispatch round
    still meets an unused library.
    """
    clock.mark("loader: starting vendor handle bring-up on dev0")
    try:
        with ctx.push_context():
            _ = vendor_blas._get_global_handle[DType.float32](ctx)
    except e:
        print("loader: vendor handle bring-up raised: ", e)
    clock.mark("loader: vendor handle bring-up returned")


def _vendor_matmul_on_busy_device(
    ctx: DeviceContext,
    c: TileTensor[mut=True, ...],
    a: TileTensor,
    b: TileTensor,
    clock: _Clock,
):
    """Runs a vendor BLAS matmul on `ctx` while it is saturated.

    The handle already exists, so only the dispatch can block: the library
    loads the GEMM kernel it picks on first use.
    """
    clock.mark("loader: starting vendor matmul on dev0")
    try:
        vendor_blas.matmul(ctx, c, a, b, c_row_major=True)
    except e:
        print("loader: vendor matmul raised: ", e)
    clock.mark("loader: vendor matmul returned")


def _cold_load_onto_busy_device(ctx: DeviceContext, clock: _Clock):
    """Cold-loads an unrelated kernel onto `ctx` while it is saturated.

    The shared-memory opt-in is what gives this teeth. `loadFunction` issues
    cuFuncSetAttribute for MAX_DYNAMIC_SHARED_SIZE_BYTES only when a size is
    requested, and that is the call that waits for the device to quiesce. A
    bare compile_function passes -1, skips it, and never blocks.
    """
    clock.mark("loader: starting cold load on dev0")
    try:
        _ = ctx.compile_function[_unrelated](
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                OPT_IN_SMEM_BYTES
            )
        )
    except e:
        print("loader: cold load raised: ", e)
    clock.mark("loader: cold load returned")


def _spin_and_release(
    ctx0: DeviceContext,
    ctx1: DeviceContext,
    flag: MutPointer[Int32, MutAnyOrigin],
    clock: _Clock,
) -> Optional[Error]:
    """Runs the two-device collective whose enqueue window the load lands in.

    Returns the error instead of raising it, because the caller has to join
    the loader task first: unwinding with that task still pending aborts on
    its non-available AsyncValue and buries whatever the collective reported.
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
            " kernel load parked on its worker"
        ),
    )


def _assert_no_stall(clock: _Clock) raises:
    """Checks the run finished near the loader's delay, not the spin bound."""
    var elapsed_sec = clock.elapsed_sec()
    assert_true(
        elapsed_sec < MAX_HEALTHY_SEC,
        String(
            "collective took ",
            elapsed_sec,
            (
                "s; a blocking driver call interleaved with the launch and"
                " blocked until the spin bound expired"
            ),
        ),
    )


def _run_round[call: Int](ctx0: DeviceContext, ctx1: DeviceContext) raises:
    """Lands one blocking call from the starved lane in a fresh window.

    Parameters:
        call: Which blocking call the lane makes: `_HANDLE_BRING_UP`,
            `_VENDOR_DISPATCH` or `_COLD_LOAD`.
    """
    var flag_buf = _create_release_flag(ctx0)
    var flag = flag_buf.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()

    var a_buf = ctx0.enqueue_create_buffer[.float32](MM * MM)
    var b_buf = ctx0.enqueue_create_buffer[.float32](MM * MM)
    var c_buf = ctx0.enqueue_create_buffer[.float32](MM * MM)
    ctx0.enqueue_memset(a_buf, Float32(1.0))
    ctx0.enqueue_memset(b_buf, Float32(1.0))
    ctx0.enqueue_memset(c_buf, Float32(0.0))
    ctx0.synchronize()
    var a_tt = TileTensor(a_buf, row_major(MM, MM)).as_imm()
    var b_tt = TileTensor(b_buf, row_major(MM, MM)).as_imm()
    var c_tt = TileTensor(c_buf, row_major(MM, MM))

    var clock = _Clock()
    comptime if call == _HANDLE_BRING_UP:
        clock.mark("round: vendor handle bring-up")
    elif call == _VENDOR_DISPATCH:
        clock.mark("round: vendor dispatch")
    else:
        clock.mark("round: cold load")

    # The load has to be in flight before the collective opens its window, and
    # on the worker that device 1's launch will be pinned to. Affinity tasks
    # sit on a private queue only their own worker dequeues, so a worker parked
    # in a blocking driver call starves the launch queued behind it and the
    # collective can never complete. The loader's sleep holds that worker until
    # device 0 is saturated, landing the load mid-fan-out, before the peer has
    # launched: the window the gate has to survive. See DRIV-462.
    var tg = TaskGroup()
    # Resolved before the coroutine exists: a raise between the two would
    # abandon it, and Coroutine is not implicitly destroyable.
    var loader_worker = task_id_for_device(Int(ctx1.id()))

    @__copy_capture(a_tt)
    @__copy_capture(b_tt)
    @__copy_capture(c_tt)
    @inline(.always)
    @__parameter
    __async def loader() -> None:
        sleep(LOADER_DELAY_SEC)
        comptime if call == _HANDLE_BRING_UP:
            _bring_up_vendor_handle_on_busy_device(ctx0, clock)
        elif call == _VENDOR_DISPATCH:
            _vendor_matmul_on_busy_device(ctx0, c_tt, a_tt, b_tt, clock)
        else:
            _cold_load_onto_busy_device(ctx0, clock)

    tg._create_task(loader(), desired_worker_id=loader_worker)

    var collective_error = _spin_and_release(ctx0, ctx1, flag, clock)
    tg.wait()
    clock.mark("loader: joined")
    if collective_error:
        raise collective_error.take()

    ctx0.synchronize()
    ctx1.synchronize()
    clock.mark("devices synchronized")

    _assert_peer_arrived(ctx0, flag_buf)
    _assert_no_stall(clock)
    _ = a_buf^
    _ = b_buf^
    _ = c_buf^


def main() raises:
    _require_two_peer_gpus()

    var ctx0 = DeviceContext(device_id=0)
    var ctx1 = DeviceContext(device_id=1)
    # Handle bring-up happens once per device, so it goes first, and the
    # dispatch round then finds the handle built and only the dispatch left.
    _run_round[_HANDLE_BRING_UP](ctx0, ctx1)
    _run_round[_VENDOR_DISPATCH](ctx0, ctx1)
    _run_round[_COLD_LOAD](ctx0, ctx1)
