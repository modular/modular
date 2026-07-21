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
)
from .buffer import Buffer, BufferView
from .context import Context
from .memory import _enqueue_copy
from .event import Event, EventFlags, EVENT_FLAG_NONE, Waitable, _EventInner
from .device import DeviceSpec
from .status import STATUS_SUCCESS, HALError
from std.collections import InlineArray, OptionalReg
from std.memory import (
    ArcPointer,
    OpaquePointer,
    UnsafePointer,
    UnsafeMaybeUninit,
)
from std.memory.arc_pointer import WeakPointer
from _hal.execution_config import (
    ExecutionConfig,
    BlockExecutionConfig,
    GridBlockExecutionConfig,
    GPUExecutionConfiguration,
    NearComputeGeneralPurposeScratchpadExecutionConfig,
)


@fieldwise_init
struct Queue[device_spec: DeviceSpec](ImplicitlyDeletable, Movable):
    """A command queue bound to a context.

    Parameters:
        device_spec: The compilation target this queue is set up for.
    """

    var _handle: QueueHandle
    var _raw: ArcPointer[RawDriver]
    var _context: ArcPointer[Context[Self.device_spec]]
    var _self_ref: WeakPointer[Self]

    @staticmethod
    def _create(
        out _self: ArcPointer[Self], context: Context[Self.device_spec]
    ) raises HALError:
        _self = ArcPointer(Self(context))
        _self[]._self_ref = WeakPointer(downgrade=_self)

    @doc_hidden
    def __init__(
        out self: Queue[Self.device_spec],
        ref context: Context[Self.device_spec],
    ) raises HALError:
        self._self_ref = WeakPointer[Self]()
        self._context = context._self_ref.try_upgrade().value()
        self._raw = context._raw

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
            self._raw[].destroy_queue(self._context[].handle(), self._handle)
        except e:
            print("warning: destroy_queue failed:", e)

    # TODO: revisit all of these when we get to queue dependency ordering
    def execute(
        self,
        func: FunctionHandle,
        grid: Tuple[UInt32, UInt32, UInt32],
        block: Tuple[UInt32, UInt32, UInt32],
        args: UnsafePointer[mut=True, OpaquePointer[MutUntrackedOrigin], _],
        arg_sizes: UnsafePointer[mut=True, UInt64, _],
        num_args: UInt32,
        shared_mem_bytes: UInt32 = 0,
        attributes: OptionalReg[OpaquePointer[MutUntrackedOrigin]] = None,
        num_attributes: UInt32 = 0,
    ) raises HALError:
        """
        Enqueue an execution of the passed function as a kernel on this queue.

        Totally ordered with respect to other operations within this queue
        if backed by a stream.
        """
        self._raw[].execute_function(
            self._handle,
            func,
            grid,
            block,
            args,
            arg_sizes,
            num_args,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes,
            num_attributes=num_attributes,
        )

    def execute[
        ExecutionConfigType: GridBlockExecutionConfig,
        //,
    ](
        self,
        func: FunctionHandle,
        execution_config: ExecutionConfigType,
        args: UnsafePointer[mut=True, OpaquePointer[MutUntrackedOrigin], _],
        arg_sizes: UnsafePointer[mut=True, UInt64, _],
        num_args: UInt32,
        attributes: OptionalReg[OpaquePointer[MutUntrackedOrigin]] = None,
        num_attributes: UInt32 = 0,
    ) raises HALError:
        """
        Enqueue an execution of the passed function as a kernel on this queue.

        Totally ordered with respect to other operations within this queue
        if backed by a stream.
        """
        var grid_dim = execution_config.get_grid_dim()
        var block_dim = execution_config.get_block_dim()

        debug_assert(
            grid_dim.x() > 0 and grid_dim.y() > 0 and grid_dim.z() > 0,
            "grid dimensions must be positive",
        )
        debug_assert(
            block_dim.x() > 0 and block_dim.y() > 0 and block_dim.z() > 0,
            "block dimensions must be positive",
        )
        debug_assert(
            grid_dim.x() <= 0xFFFFFFFF
            and grid_dim.y() <= 0xFFFFFFFF
            and grid_dim.z() <= 0xFFFFFFFF,
            "grid dimensions must fit in 32 bits",
        )
        debug_assert(
            block_dim.x() <= 0xFFFFFFFF
            and block_dim.y() <= 0xFFFFFFFF
            and block_dim.z() <= 0xFFFFFFFF,
            "block dimensions must fit in 32 bits",
        )

        var grid = Tuple(
            UInt32(grid_dim.x()), UInt32(grid_dim.y()), UInt32(grid_dim.z())
        )
        var block = Tuple(
            UInt32(block_dim.x()), UInt32(block_dim.y()), UInt32(block_dim.z())
        )

        var near_compute_scratchpad_usage: UInt64 = 0
        comptime if conforms_to(
            ExecutionConfigType,
            NearComputeGeneralPurposeScratchpadExecutionConfig,
        ):
            near_compute_scratchpad_usage = (
                execution_config.get_near_compute_scratchpad_usage()
            )

        debug_assert(
            near_compute_scratchpad_usage <= 0xFFFFFFFF,
            "near compute scratchpad usage must fit in 32 bits",
        )

        self._raw[].execute_function(
            self._handle,
            func,
            grid,
            block,
            args,
            arg_sizes,
            num_args,
            shared_mem_bytes=UInt32(near_compute_scratchpad_usage),
            attributes=attributes,
            num_attributes=num_attributes,
        )

    # Direction-specific transports. Callable directly for fine-grained
    # control, and the raw-host path (a bare pointer, not a `Buffer`) that
    # `copy` cannot express; `copy` (below) dispatches to these by residency.
    def copy_to_device(
        self,
        dst: BufferView,
        src: UnsafePointer[mut=False, UInt8, _],
    ) raises HALError:
        """
        Explicit host-to-device copy of `dst.byte_size` bytes into `dst`.
        Enqueues on this queue. Totally ordered with respect to other
        operations within this queue if backed by a stream.
        """
        self._raw[].copy_to_device(self._handle, dst._view, src)

    def copy_from_device(
        self,
        dst: UnsafePointer[mut=True, UInt8, _],
        src: BufferView,
    ) raises HALError:
        """
        Explicit device-to-host copy of `src.byte_size` bytes from `src`.
        Enqueues on this queue. Totally ordered with respect to other
        operations within this queue if backed by a stream.
        """
        self._raw[].copy_from_device(self._handle, dst, src._view)

    def copy_intra_device(
        self,
        dst: BufferView,
        src: BufferView,
    ) raises HALError:
        """
        Same-device copy of `dst.byte_size` bytes from `src` into `dst`.
        Enqueues on this queue. Totally ordered with respect to other
        operations within this queue if backed by a stream.
        """
        debug_assert(
            src.byte_size() >= dst.byte_size(),
            "copy_intra_device source view smaller than destination",
        )
        self._raw[].copy_intra_device(self._handle, dst._view, src._view)

    def launch_host_func[
        origin: MutOrigin
    ](
        self,
        func: def(OpaquePointer[origin]) thin -> None,
        user_data: OpaquePointer[origin],
    ) raises HALError:
        """Enqueues a host function callback (e.g. cuLaunchHostFunc)."""
        self._raw[].queue_launch_host_func(self._handle, func, user_data)

    def set_memory(
        self,
        dst: BufferView,
        value: UInt8,
    ) raises HALError:
        """
        Set every byte of `dst` to `value`. Enqueues on this queue. Sets
        `dst.byte_size` bytes.
        """
        self._raw[].set_memory(self._handle, dst._view, value)

    def fill(
        self,
        dst: BufferView,
        value: UInt64,
        value_size: UInt64,
    ) raises HALError:
        """
        Fill `dst` with a repeated `value_size`-byte `value`. Enqueues on this
        queue. Fills `dst.byte_size` bytes. `value_size` must be one of 1, 2, 4,
        or 8; a `value_size` of 1 is equivalent to `set_memory`.
        """
        self._raw[].fill(self._handle, dst._view, value, value_size)

    # ===-------------------------------------------------------------------===#
    # Unified copy
    # ===-------------------------------------------------------------------===#

    def copy(
        self,
        *,
        dst: Buffer[Self.device_spec],
        src: Buffer[Self.device_spec],
    ) raises HALError:
        """Enqueues a buffer-to-buffer copy of `src` into the front of `dst`.

        Transfers exactly `src.byte_size` bytes; `dst` must be at least that
        large, and any remaining tail of `dst` is left untouched. The transfer
        runs on this queue, so the device-resident operand it touches must
        reside on this queue's device — `dst` for a to-device or same-device
        copy, `src` for a device-to-pinned-host copy. A pinned host operand is
        only a host pointer and may come from any device's context. A
        device-to-device copy whose source is on another device is a peer copy;

        Args:
            dst: Destination buffer.
            src: Source buffer.
        """
        _enqueue_copy(
            self._raw,
            self._handle,
            self._context[]._device[].id,
            dst=dst,
            src=src,
        )

    def record_event[
        flags: EventFlags = EVENT_FLAG_NONE,
    ](self, out event: Event[flags]) raises HALError:
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
        event = Event[flags](
            _EventInner(
                _handle=event_handle,
                _context_handle=self._context[]._handle,
                _raw=self._raw,
            )
        )
        self._raw[].record_event(self._handle, event._inner[]._handle)

    def wait_for_events[
        *EventTypes: Waitable,
    ](self, *events: *EventTypes) raises HALError:
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

    def native_handle(self) raises HALError -> OptionalReg[UInt64]:
        """Returns the backend stream/queue handle, or None if the queue has
        none (a device with no OS-level stream object, e.g. CPU)."""
        return self._raw[].get_optional_queue_property["native_handle", UInt64](
            self._handle
        )
