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
from .plugin import (
    RawDriver,
    OutParam,
    EventHandle,
    QueueHandle,
    FunctionHandle,
    MemoryHandle,
)
from .context import Context
from .event import Event, EventFlags, EVENT_FLAG_NONE, Waitable, _EventInner
from .device import DeviceSpec
from .status import STATUS_SUCCESS, HALError
from std.collections import InlineArray
from std.memory import (
    ImmutPointer,
    OpaquePointer,
    UnsafePointer,
    UnsafeMaybeUninit,
)


@fieldwise_init
struct Queue[context_origin: ImmutOrigin, device_spec: DeviceSpec](Movable):
    """A command queue bound to a context.

    Parameters:
        context_origin: The origin of the parent Context pointer.
        device_spec: The compilation target this queue is set up for.
    """

    var _handle: QueueHandle
    var _raw: ImmutPointer[RawDriver, Self.context_origin]
    var _context: ImmutPointer[
        Context[Self.context_origin, Self.device_spec], Self.context_origin
    ]

    def __init__[
        o1: ImmutOrigin, o2: ImmutOrigin
    ](
        out self: Queue[origin_of(o1, o2), Self.device_spec],
        ref[o1] context: Context[o2, Self.device_spec],
    ) raises HALError:
        # This is a horrible hack that should be revisited as soon as
        # we can express subtyping relations between origins and/or
        # inferred/unbound inner origin params for arguments.
        # See MOCO-3661, MOCO-3326.
        self._context = rebind[type_of(self._context)](Pointer(to=context))
        self._raw = rebind[type_of(self._raw)](context._raw)

        ref raw = context._raw[]

        var queue_handle_uninit = UnsafeMaybeUninit[QueueHandle]()
        var status = raw._raw.queue_create.f(
            context._handle, OutParam[QueueHandle](to=queue_handle_uninit)
        )

        if status != STATUS_SUCCESS:
            var err = raw.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to create queue: {err.message}"),
            )

        self._handle = queue_handle_uninit.unsafe_assume_init_ref()

    def __del__(deinit self):
        try:
            self._raw[].destroy_queue(self._context[]._handle, self._handle)
        except e:
            print("warning: destroy_queue failed:", e)

    # TODO: revisit all of these when we get to queue dependency ordering
    def execute(
        self,
        func: FunctionHandle,
        grid: Tuple[UInt32, UInt32, UInt32],
        block: Tuple[UInt32, UInt32, UInt32],
        args: UnsafePointer[OpaquePointer[MutExternalOrigin], MutAnyOrigin],
        arg_sizes: UnsafePointer[UInt64, MutAnyOrigin],
        num_args: UInt32,
    ) raises HALError:
        """
        Enqueue an execution of the passed function as a kernel on this queue.

        Totally ordered with respect to other operations within this queue
        if backed by a stream.
        """
        self._raw[].execute_function(
            self._handle, func, grid, block, args, arg_sizes, num_args
        )

    def copy_to_device(
        self,
        dst: Buffer,
        src: UnsafePointer[UInt8, MutAnyOrigin],
        size: UInt64,
    ) raises HALError:
        """
        Explicit host-to-device buffer copy. Enqueues on this queue.
        Totally ordered with respect to other operations within this queue
        if backed by a stream.
        """
        self._raw[].copy_to_device(self._handle, dst._handle, src, size)

    def copy_from_device(
        self,
        dst: UnsafePointer[UInt8, MutAnyOrigin],
        src: Buffer,
        size: UInt64,
    ) raises HALError:
        """
        Explicit device-to-host buffer copy. Enqueues on this queue.
        Totally ordered with respect to other operations within this queue
        if backed by a stream.
        """
        self._raw[].copy_from_device(self._handle, dst, src._handle, size)

    def copy_intra_device(
        self,
        dst: Buffer,
        src: Buffer,
        size: UInt64,
    ) raises HALError:
        """
        Same-device buffer copy. Enqueues on this queue.
        Totally ordered with respect to other operations within this queue
        if backed by a stream.
        """
        self._raw[].copy_intra_device(
            self._handle, dst._handle, src._handle, size
        )

    def record_event[
        flags: EventFlags = EVENT_FLAG_NONE,
    ](self, out event: Event[Self.context_origin, flags],) raises HALError:
        """Creates a fresh event, records it on this queue's timeline, and
        returns it.

        The returned event is signaled when all operations enqueued on the
        queue before this call have completed.

        Parameters:
            flags: Capability bitmask. Default `EVENT_FLAG_NONE` is intra-GPU
                only — the cheapest path. Pass `EVENT_FLAG_CPU_VISIBLE` to
                enable host-side `synchronize()` / `is_ready()` calls.
        """
        var event_handle = self._raw[].create_event(
            self._context[]._handle, flags
        )
        event = Event[Self.context_origin, flags](
            _EventInner[Self.context_origin](
                _handle=event_handle,
                _context_handle=self._context[]._handle,
                _raw=self._raw,
            )
        )
        self._raw[].record_event(self._handle, event._inner[]._handle)

    def wait_for_events[
        *EventTypes: Waitable,
    ](self, *events: *EventTypes,) raises HALError:
        """Enqueues a wait for the given events on this queue.

        Accepts any combination of events with different flag combos.
        """
        comptime n = events.__len__()

        comptime if n == 0:
            return

        var handles = InlineArray[EventHandle, n](uninitialized=True)
        comptime for i in range(n):
            handles[i] = events[i]._handle()
        self._raw[].wait_for_events(
            self._handle, handles.unsafe_ptr(), UInt32(n)
        )

    def synchronize(self) raises HALError:
        """
        Totally ordered with respect to other operations within this queue
        if backed by a stream.
        """
        self._raw[].synchronize_queue(self._handle)
