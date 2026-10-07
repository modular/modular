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
"""Mojo bindings for the driver's kernel-load gate.

The gate keeps first-time kernel loads from interleaving with a multi-device
collective's per-device enqueues. A load blocks until its device quiesces
while holding that context's driver lock. One that starts partway through the
fan-out can strand the peers that already reached the device-side barrier:
the driver's locks are per context, but a thread spanning two contexts, or
the worker a peer's enqueue is pinned to, ties that enqueue to the load.
`KernelLoadGate` in `MLRT/include/MLRT/Driver/DeviceContext/KernelLoadGate.h`
states what the gate guarantees and why; these types only reach it from Mojo.

Load waits suspend on an AsyncRT value rather than parking the thread, so a
worker waiting on the gate still runs the collective sub-task pinned to it.
The collective side never waits at all; see `CollectiveGate`.

Every type here is a no-op on a backend whose driver has no gate, and every
scope releases on the raising path as well as through `__exit__`: leaking the
gate wedges every later load.
"""

from std.atomic import Atomic
from std.ffi import external_call
from std.runtime._asyncrt import _Chain, _del_asyncrt_chain

from .device_context import DeviceContext


struct CollectiveGate:
    """Claims the gate for one collective's enqueues, without blocking.

    `try_acquire` either grants the gate now or leaves a pending grant in
    `grant` for the caller to dispatch from once it becomes available. Each
    per-device enqueue then calls `enqueue_done`, and the last of them releases
    the gate, from whichever worker that is.

    Nothing here waits. A donating wait runs other tasks nested on the waiting
    thread's stack, so a frame that held the gate while waiting for its fan-out
    could be buried beneath a task that needs the gate, with only that frame
    able to release it. See test_kernel_load_gate_nested.

    Keep it alive until the fan-out has finished: the enqueues reference it.
    """

    var _ctx: DeviceContext
    var _remaining: Atomic[Int]
    var grant: _Chain
    """The pending grant when `try_acquire` returned False; empty otherwise."""

    def __init__(out self, ctx: DeviceContext, num_enqueues: Int):
        # Any participating context resolves the same gate: it belongs to the
        # backend's driver, which every device in the collective shares.
        self._ctx = ctx
        self._remaining = Atomic[Int](num_enqueues)
        self.grant = _Chain()

    def __deinit__(deinit self):
        if self.grant:
            _del_asyncrt_chain(Pointer(to=self.grant))

    def try_acquire(mut self) -> Bool:
        """Claims the gate, or queues this collective for it.

        Returns:
            True if this collective owns the gate now. False if it must
            dispatch from `grant` once that becomes available.
        """
        return external_call[
            "AsyncRT_KernelLoadGate_tryAcquireCollective", Bool
        ](self._ctx._handle, Pointer(to=self.grant))

    def enqueue_done(mut self):
        """Records one per-device enqueue as finished; the last releases."""
        if self._remaining.fetch_sub(1) == 1:
            external_call["AsyncRT_KernelLoadGate_releaseCollective", NoneType](
                self._ctx._handle
            )


struct CollectiveEnqueueScope:
    """Registers this thread as enqueueing one device's share of a collective.

    Its loads then proceed instead of waiting on the gate their own collective
    holds. The driver keys this on thread identity, which is sound only because
    a per-device launch body never suspends.
    """

    var _ctx: DeviceContext
    var _held: Bool

    def __init__(out self, ctx: DeviceContext):
        self._ctx = ctx
        self._held = False

    def __deinit__(deinit self):
        if self._held:
            self._close()

    def __enter__(mut self):
        external_call["AsyncRT_KernelLoadGate_beginEnqueue", NoneType](
            self._ctx._handle
        )
        self._held = True

    def __exit__(mut self):
        self._close()
        self._held = False

    def _close(self):
        external_call["AsyncRT_KernelLoadGate_endEnqueue", NoneType](
            self._ctx._handle
        )


struct KernelLoadScope:
    """Keeps a driver call that can block on a busy device out of a
    collective's enqueue window.

    A first-time kernel load is gated inside `loadFunction`, but a vendor
    library call reaches the driver without passing through it while blocking
    the same way. Wrapping such a call here gives it the same treatment: it
    waits if a collective is mid-enqueue, and that wait suspends on an AsyncRT
    value so the worker stays free to run the launch it would otherwise
    starve. On a thread inside a collective's own per-device enqueue it is
    counted but does not wait.
    """

    var _ctx: DeviceContext
    var _held: Bool

    def __init__(out self, ctx: DeviceContext):
        self._ctx = ctx
        self._held = False

    def __deinit__(deinit self):
        if self._held:
            self._close()

    def __enter__(mut self):
        external_call["AsyncRT_KernelLoadGate_beginLoad", NoneType](
            self._ctx._handle
        )
        self._held = True

    def __exit__(mut self):
        self._close()
        self._held = False

    def _close(self):
        external_call["AsyncRT_KernelLoadGate_endLoad", NoneType](
            self._ctx._handle
        )
