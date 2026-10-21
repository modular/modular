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
"""Regression test for nesting under the kernel-load gate's collective side.

The thread that opens a collective waits for its fan-out with
`TaskGroup.wait()`, which on an AsyncRT worker runs other tasks nested on the
same stack. The opener must hold nothing while it waits, or a nested task that
needs the gate would wait for a release only the buried opener can perform.
Nothing spins on a GPU here.

Three scenarios run in order, each bounded so a hang fails as an abort. In the
control, the cold load is pinned to another worker, waits at the gate, and
proceeds once the collective releases it. In the nested load, the cold load is
pinned to the opener's own worker, so the opener's wait dequeues it. In the
nested collective, the opener's wait dequeues a task that opens a second
collective, which must queue behind the first rather than wait on it.
"""

from std.os import abort
from std.runtime import parallelism_level
from std.runtime._asyncrt import TaskGroup, _async_wait_timeout
from std.testing import assert_true
from std.time import monotonic, sleep

from max.runtime.asyncrt import task_id_for_device

from comm.device_collective import _launch_device_collective
from max.gpu.host import DeviceContext

# How long device 1's launch body holds the collective's enqueue window open.
comptime HOLD_WINDOW_SEC = 0.5

# When the intruding task is queued: after the opener is waiting inside the
# window.
comptime LOADER_DELAY_SEC = 0.1

# A healthy scenario finishes shortly after HOLD_WINDOW_SEC.
comptime TIMEOUT_NS = 10_000_000_000


def _unrelated[tag: Int]():
    """A distinct kernel per scenario, so each load is a first-time load."""
    pass


def _mark(start_ns: Int, label: String):
    print("[", Float64(monotonic() - start_ns) / 1.0e6, "ms ] ", label, sep="")


def _abort_hung(name: StaticString) -> Never:
    abort(
        String(
            "scenario '",
            name,
            "' did not finish within ",
            TIMEOUT_NS // 1_000_000_000,
            (
                "s: the intruding task is waiting on a gate that only the frame"
                " beneath it can release"
            ),
        )
    )


@inline(.never)
def _open_collective[
    *, hold_window: Bool
](
    ctx0: DeviceContext,
    ctx1: DeviceContext,
    start_ns: Int,
    label: StaticString,
) raises:
    """Runs a two-device collective whose launch bodies only mark the trace.

    With `hold_window`, device 1's body sleeps first, keeping the collective's
    enqueue window open. Kept out of line: inlining the collective into the
    async tasks that call this crashes the compiler in LowerAsyncFunctions.
    """

    @inline(.always)
    def launch[index: Int]() raises {imm start_ns, imm label}:
        comptime if hold_window and index == 1:
            sleep(HOLD_WINDOW_SEC)
        _mark(start_ns, String(label, ": device ", index, " enqueued"))

    _launch_device_collective[2](launch, [ctx0, ctx1])


def _run_scenario[
    tag: Int, *, inner_collective: Bool = False
](
    ctx0: DeviceContext,
    ctx1: DeviceContext,
    opener_worker: Int,
    loader_worker: Int,
    name: StaticString,
) raises:
    """Opens a collective on `opener_worker`, then intrudes on `loader_worker`.

    The intruding task cold-loads a kernel, or opens a second collective if
    `inner_collective` is set. Aborts if the two do not both finish within
    TIMEOUT_NS.
    """
    print("==== scenario:", name)
    var start_ns = monotonic()
    var collective_error = Optional[Error]()

    @inline(.always)
    __async def opener() {mut collective_error, var} -> None:
        _mark(start_ns, "opener: opening collective")
        try:
            _open_collective[hold_window=True](
                ctx0, ctx1, start_ns, "collective"
            )
        except e:
            collective_error = e^
        _mark(start_ns, "opener: collective closed")

    # TODO(MOCO-5158): an async closure whose captures are all
    # register-passable is invoked with a pointer to a by-value context.
    # Capturing a String makes the context memory-only; drop it with the fix.
    var memory_only = String()

    @inline(.always)
    __async def loader() {var} -> None:
        _ = memory_only
        comptime if inner_collective:
            _mark(start_ns, "loader: opening inner collective")
            try:
                _open_collective[hold_window=False](
                    ctx0, ctx1, start_ns, "inner collective"
                )
            except e:
                print("loader: inner collective raised: ", e)
            _mark(start_ns, "loader: inner collective closed")
        else:
            _mark(start_ns, "loader: starting cold load")
            try:
                _ = ctx0.compile_function[_unrelated[tag]]()
            except e:
                print("loader: cold load raised: ", e)
            _mark(start_ns, "loader: cold load returned")

    var tg = TaskGroup()
    tg._create_task(opener(), desired_worker_id=opener_worker)
    sleep(LOADER_DELAY_SEC)
    tg._create_task(loader(), desired_worker_id=loader_worker)

    # TaskGroup.wait() without the unbounded wait.
    tg._task_complete()
    var finished = _async_wait_timeout(Pointer(to=tg.chain), TIMEOUT_NS)
    if not finished:
        _abort_hung(name)
    # Keeps `tg` alive through the branch above. Mojo destroys a value after
    # its last use, and destroying a group whose tasks are still pending
    # aborts with a runtime error that buries the hang report.
    _ = tg.chain
    _mark(start_ns, "scenario finished")
    if collective_error:
        raise collective_error.take()


def main() raises:
    assert_true(
        DeviceContext.number_of_devices() > 1, "must have multiple GPUs"
    )
    var ctx0 = DeviceContext(device_id=0)
    var ctx1 = DeviceContext(device_id=1)

    # The opener and the control's loader must avoid the device workers, which
    # run the per-device launch bodies, and each other.
    var dev0_worker = task_id_for_device(Int(ctx0.id()))
    var dev1_worker = task_id_for_device(Int(ctx1.id()))
    var spare = List[Int]()
    for worker in range(1, parallelism_level()):
        if worker != dev0_worker and worker != dev1_worker:
            spare.append(worker)
    assert_true(len(spare) >= 2, "need two workers besides the device workers")
    var opener_worker = spare[0]
    var other_worker = spare[1]
    print(
        "workers: dev0=",
        dev0_worker,
        " dev1=",
        dev1_worker,
        " opener=",
        opener_worker,
        " other=",
        other_worker,
        sep="",
    )

    _run_scenario[0](ctx0, ctx1, opener_worker, other_worker, "control")
    _run_scenario[1](ctx0, ctx1, opener_worker, opener_worker, "nested load")
    _run_scenario[2, inner_collective=True](
        ctx0, ctx1, opener_worker, opener_worker, "nested collective"
    )
