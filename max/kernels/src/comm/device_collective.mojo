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
"""Helpers for dispatching collective operations across devices."""

from std.collections import Array, Optional
from std.runtime._asyncrt import TaskGroup

from max.gpu.host import DeviceContext, DeviceContextArray
from max.gpu.host._kernel_load_gate import (
    CollectiveEnqueueScope,
    CollectiveGate,
)
from max.runtime.asyncrt import task_id_for_device


@inline(.always)
def _launch_device_collective[
    num_devices: Int,
    F: def[Int]() raises -> None,
](func: F, var dev_ctxs: Array[DeviceContext, num_devices]) raises:
    """Dispatch async tasks to call func[i]() for each device in dev_ctxs.

    The tasks run concurrently, one per device affinity worker. They must:
    every collective kernel spins at a device-side barrier until all peers
    arrive, and several fill the GPU while they do, so a peer still compiling
    its own copy would wait for a quiesce that never comes.

    Unrelated kernel loads are held off until every device has been enqueued.
    The tasks are dispatched only once this collective owns the kernel-load
    gate, and the last of them releases it; this frame never holds the gate
    while it waits, because the wait may run a task that needs the gate.

    `func` must not wait or launch a collective of its own. A nested collective
    queues behind this one, which releases the gate only after `func` returns,
    so the two deadlock. A wait lets other work run on a thread registered as
    this collective's, and that work's kernel loads would pass the gate.

    Parameters:
        num_devices: Number of devices participating in the collective.
        F: Type of the per-device launch function.

    Args:
        func: Per-device launch function, called once with each device index.
        dev_ctxs: Contexts of the participating devices.
    """

    comptime assert num_devices > 0, "a collective needs at least one device"

    # One Optional[Error] slot per device; None means no error.
    # Each task writes only to its own index, so there is no data race.
    var errors = Array[Optional[Error], num_devices](fill=Optional[Error]())

    # Resolved here because the deferred dispatch below cannot raise.
    var worker_ids = Array[Int, num_devices](fill=-1)
    comptime for i in range(num_devices):
        worker_ids[i] = task_id_for_device(Int(dev_ctxs[i].id()))

    var gate = CollectiveGate(dev_ctxs[0], num_devices)
    var tg = TaskGroup()
    # The deferred dispatch adds tasks to the group it runs in. The group
    # outlives it, which the origin checker cannot see through the capture.
    var group = Pointer(to=tg).unsafe_origin_cast[MutUntrackedOrigin]()

    # Wrap the launch function in a Mojo async function which does not raise.
    @inline(.always)
    __async def wrapper[index: Int]() {mut errors, mut gate, imm} -> None:
        # Lets this worker's own loads through the gate its collective holds.
        with CollectiveEnqueueScope(dev_ctxs[index]):
            try:
                func[index]()
            except e:
                errors[index] = e^
        gate.enqueue_done()

    # Dispatch to the worker threads that have affinity for each device.
    @inline(.always)
    __async def dispatch_when_granted() {imm} -> None:
        comptime for i in range(num_devices):
            group[]._create_task(wrapper[i](), desired_worker_id=worker_ids[i])

    if gate.try_acquire():
        comptime for i in range(num_devices):
            tg._create_task(wrapper[i](), desired_worker_id=worker_ids[i])
    else:
        # A copy of the reference; `gate` keeps the grant itself alive.
        var grant = gate.grant
        tg._create_task_after(dispatch_when_granted(), grant)

    # Holds nothing while it waits, so a task this wait runs may take the gate.
    tg.wait()
    # The tasks reference `gate` until the last of them releases it.
    _ = gate.grant

    # Re-raise the first error encountered.
    comptime for i in range(num_devices):
        if errors[i]:
            raise errors[i].take()


@inline(.always)
def _launch_device_collective[
    num_devices: Int,
    F: def[Int]() raises -> None,
](func: F, var dev_ctxs: DeviceContextArray) raises:
    """Dispatch async tasks to call func[i]() for each device in dev_ctxs.

    `DeviceContextArray` overload. Forwards to the `Array` overload
    by unpacking the array's underlying storage.

    Parameters:
        num_devices: Number of devices participating in the collective.
        F: Type of the per-device launch function.

    Args:
        func: Per-device launch function, called once with each device index.
        dev_ctxs: Contexts of the participating devices.
    """

    comptime assert (
        dev_ctxs.length == num_devices
    ), "expected dev_ctxs to have the same number of elements as num_devices"

    _launch_device_collective[num_devices](
        func,
        rebind[Array[DeviceContext, num_devices]](
            dev_ctxs.device_contexts^
        ).copy(),
    )
