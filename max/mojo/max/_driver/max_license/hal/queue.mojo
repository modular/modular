# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
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
    ContextHandle,
)
from max._plugin import MaxPluginForTarget
from ._machine import DeviceRef
from .buffer import Buffer, BufferView
from .context import Context
from .memory import _enqueue_copy
from .event import Event, EventFlags, EVENT_FLAG_NONE, Waitable, _EventInner
from .device import DeviceSpec
from .status import STATUS_SUCCESS, STATUS_UNSUPPORTED, HALError
from std.collections import Array, OptionalReg
from std.memory import (
    ArcPointer,
    OpaquePointer,
    Pointer,
    MaybeUninit,
)
from std.memory.arc_pointer import WeakPointer
from std.builtin.rebind import downcast
from std.reflection import SourceLocation, call_location, reflect
from max.gpu.host import (
    ClusterExecutionConfig,
    ExecutionConfig,
    GridBlockExecutionConfig,
    NearComputeGeneralPurposeScratchpadExecutionConfig,
)
from max.gpu.host import LaunchAttribute
from max.gpu.host.nvidia import LaunchAttributeHolderExecutionConfig
from max.gpu.host._launch_args import DeviceFunctionWrapper
from .execution_config import check_config_fields_launchable
from .kernel_args import _is_kernel_arg


@fieldwise_init
struct Queue[device_spec: DeviceSpec](Deinitable, Movable):
    """A command queue bound to a context.

    Parameters:
        device_spec: The compilation target this queue is set up for.
    """

    var _handle: QueueHandle
    var _raw: ArcPointer[RawDriver]
    var _context: ArcPointer[Context[Self.device_spec]]
    var _self_ref: WeakPointer[Self]

    comptime HALExtensions = MaxPluginForTarget[
        DeviceSpec.target[Self.device_spec]()
    ].HalHooks.extensions

    var _extensions: Pointer[Self.HALExtensions, ImmUntrackedOrigin]

    comptime _HALHooks = MaxPluginForTarget[
        DeviceSpec.target[Self.device_spec]()
    ].HalHooks

    comptime _FunctionWrapper = MaxPluginForTarget[
        DeviceSpec.target[Self.device_spec]()
    ].FunctionWrapper

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
        self._extensions = self._context[]._extensions
        self._raw = context._raw

        ref raw = context._raw[]

        var queue_handle_uninit = MaybeUninit[QueueHandle]()
        var status = raw._raw.base.queue_create.get()(
            context._handle, OutParam[QueueHandle](to=queue_handle_uninit)
        )

        if status != STATUS_SUCCESS:
            var err = raw.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to create queue: {err.message}"),
            )

        self._handle = queue_handle_uninit.unsafe_assume_init()

    def __deinit__(deinit self):
        try:
            self._raw[].destroy_queue(self._context[].handle(), self._handle)
        except e:
            print("warning: destroy_queue failed:", e)

    def extensions(
        imm self,
    ) -> ref[origin_of(self)] Self.HALExtensions:
        """Returns a ref to the HAL extension carrier object that
        lives on the `Device` this is peripherally derived from.

        Returns:
            An immutable ref to the extension carrier.
        """

        # SAFETY: this ref is valid up through destruction of `self`,
        # as it must live on the `Device` that `self` was peripherally
        # derived from and that `self` keeps alive
        # through `self._context`.
        return self._extensions.unsafe_origin_cast[origin_of(self)]()[]

    @inline(.always)
    def execute[
        declared_arg_types: TypeList[Trait=AnyType, ...],
        ExecutionConfigType: ExecutionConfig,
        //,
        kernel: def(*args: *declared_arg_types) thin -> None,
        *Ts: AnyType,
    ](
        self,
        *args: *Ts,
        function: FunctionHandle,
        execution_config: ExecutionConfigType,
        location: Optional[SourceLocation] = None,
    ) raises HALError:
        """Enqueues `function`, the loaded `kernel`, with `args` as
        `execution_config` describes.

        `function` must be loaded from `kernel` compiled behind the plugin's
        `FunctionWrapper`, and stay loaded until the launch completes. Every
        argument must already be on this queue's device: a
        `DeviceMemoryHandle`, such as a `Buffer` or `BufferView`, fills a
        pointer argument with its device address, and any other argument is
        a `DevicePassable` value, such as a fixed-width scalar, a device
        pointer or a `TileTensor`, encoded as `kernel` declares it. `Int` and
        `UInt` are a compile-time error, since their width differs between
        host and device.

        The plugin's `ArgPacker` packs the arguments. A grid/block config
        launches through the plugin's launcher when it takes grid/block
        launches, and otherwise with the packed slots. Any other config
        launches through the plugin's launcher, and is a compile-time error
        for a plugin whose launcher doesn't take one. A grid/block config
        that can hold constant memory mappings is a compile-time error too,
        since they don't reach the launch, and so is a cluster config on a
        device that can't launch clusters. A plugin launcher that only takes
        the grid and block raises on launch attributes, clusters or shared
        memory.

        Totally ordered with respect to other operations within this queue
        if backed by a stream.

        Parameters:
            declared_arg_types: The kernel's declared argument types.
            ExecutionConfigType: The type of `execution_config`.
            kernel: The kernel `function` was compiled from.
            Ts: The argument types.

        Args:
            args: The kernel's arguments.
            function: The loaded function.
            execution_config: How to launch it.
            location: Where the launch was requested, for errors; the
                caller's location by default.

        Raises:
            If an argument can't be packed or the launch fails.
        """
        comptime assert Ts.length == declared_arg_types.length, String(
            "the kernel takes ",
            declared_arg_types.length,
            " arguments but the launch passes ",
            Ts.length,
        )
        comptime for i in range(Ts.length):
            comptime assert _is_kernel_arg[Ts[i]](), String(
                "argument #",
                i,
                " of type '",
                reflect[Ts[i]].name(),
                "' is neither a `DeviceMemoryHandle` nor `DevicePassable`",
            )
            comptime assert not (Ts[i] == Int or Ts[i] == UInt), String(
                "argument #",
                i,
                (
                    " is an `Int` or `UInt`, whose width differs between host"
                    " and device; use a fixed-width type such as `Int32` or"
                    " `Int64`"
                ),
            )

        try:
            self._launch[
                Self._FunctionWrapper,
                declared_arg_types,
                "HAL `Queue.execute`",
            ](function, execution_config, *args)
        except e:
            raise HALError(
                e.status,
                message=location.or_else(call_location()).prefix(e.message),
            )

    @doc_hidden
    @inline(.always)
    def _launch[
        ExecutionConfigType: ExecutionConfig,
        //,
        Wrapper: DeviceFunctionWrapper,
        declared_arg_types: TypeList[Trait=AnyType, ...],
        api_name: StringLiteral,
        *Ts: AnyType,
    ](
        self,
        func: FunctionHandle,
        execution_config: ExecutionConfigType,
        *args: *Ts,
    ) raises HALError:
        """Packs `args` with the plugin's `ArgPacker` and launches `func` as
        `execution_config` describes.

        Every typed launch ends here: `execute`, and the DeviceContext and
        plugin launches that reach the plugin's launcher, so the launch-support
        checks and the packed bytes' lifetime live in one place.

        Parameters:
            ExecutionConfigType: The type of `execution_config`.
            Wrapper: The wrapper `func`'s kernel was compiled behind.
            declared_arg_types: The kernel's declared argument types.
            api_name: The launch API, for error messages.
            Ts: The argument types.

        Args:
            func: The loaded function.
            execution_config: How to launch it.
            args: The kernel's arguments.

        Raises:
            If an argument can't be packed, the config asks for shared memory,
            launch attributes or clusters the plugin's grid/block launcher
            can't grant, or the launch fails.
        """
        comptime Hooks = Self._HALHooks
        comptime is_grid_block = conforms_to(
            ExecutionConfigType, GridBlockExecutionConfig
        )
        comptime assert is_grid_block or Hooks.Launcher.launches_config, String(
            api_name,
            (
                ": the active HAL plugin's launcher doesn't take this"
                " execution config"
            ),
        )
        comptime if is_grid_block:
            check_config_fields_launchable[ExecutionConfigType, api_name]()
        comptime if conforms_to(ExecutionConfigType, ClusterExecutionConfig):
            comptime assert conforms_to(
                Hooks.ExecutionConfigType[
                    DeviceRef(Self.device_spec, 0, private=())
                ],
                ClusterExecutionConfig,
            ), "cluster launch is not supported on this device"

        var packed = Hooks.ArgPacker.pack[Wrapper, declared_arg_types](
            self._context[], *args
        )
        comptime if not is_grid_block:
            Hooks.Launcher.enqueue(
                self, func, packed, execution_config=execution_config
            )
        else:
            ref config = rebind[
                downcast[ExecutionConfigType, GridBlockExecutionConfig]
            ](execution_config)
            comptime if Hooks.Launcher.overrides_grid_block:
                # It only gets the grid and block, so refuse anything else
                # rather than drop it.
                comptime if conforms_to(
                    ExecutionConfigType,
                    NearComputeGeneralPurposeScratchpadExecutionConfig,
                ):
                    if execution_config.get_near_compute_scratchpad_usage():
                        raise HALError(
                            STATUS_UNSUPPORTED,
                            message=(
                                "the active HAL plugin's launcher doesn't take"
                                " shared memory"
                            ),
                        )
                comptime if conforms_to(
                    ExecutionConfigType, LaunchAttributeHolderExecutionConfig
                ):
                    if execution_config.get_launch_attributes():
                        raise HALError(
                            STATUS_UNSUPPORTED,
                            message=(
                                "the active HAL plugin's launcher doesn't take"
                                " launch attributes"
                            ),
                        )
                comptime if conforms_to(
                    ExecutionConfigType, ClusterExecutionConfig
                ):
                    var cluster_dim = execution_config.get_cluster_dim()
                    if (
                        cluster_dim.x() != 1
                        or cluster_dim.y() != 1
                        or cluster_dim.z() != 1
                    ):
                        raise HALError(
                            STATUS_UNSUPPORTED,
                            message=(
                                "the active HAL plugin's launcher doesn't take"
                                " cluster dims"
                            ),
                        )
                Hooks.Launcher.enqueue_grid_block(
                    self,
                    func,
                    packed,
                    grid_dim=config.get_grid_dim(),
                    block_dim=config.get_block_dim(),
                )
            else:
                var slots = packed.slots()
                self.execute(
                    func,
                    config,
                    args=slots.addrs.unsafe_ptr(),
                    arg_sizes=slots.sizes.unsafe_ptr(),
                    num_args=UInt32(packed.num_slots()),
                )
                _ = slots^
        # The launcher may only have queued the launch; keep the packed
        # bytes alive until it returns.
        _ = packed^

    @doc_hidden
    def execute[
        declared_arg_types: TypeList[Trait=AnyType, ...],
        ExecutionConfigType: ExecutionConfig,
        //,
        kernel: def(*args: *declared_arg_types) thin raises -> None,
        *Ts: AnyType,
    ](
        self,
        *args: *Ts,
        function: FunctionHandle,
        execution_config: ExecutionConfigType,
        location: Optional[SourceLocation] = None,
    ) raises HALError:
        # Only a raising kernel lands here; the overload above takes the rest.
        comptime assert False, "a kernel launched by `execute` can't raise"

    @inline(.always)
    def execute[
        declared_arg_types: TypeList[Trait=AnyType, ...],
        ExecutionConfigType: ExecutionConfig,
        //,
        kernel: def(*args: *declared_arg_types) thin -> None,
        *Ts: AnyType,
    ](
        self,
        *args: *Ts,
        execution_config: ExecutionConfigType,
        location: Optional[SourceLocation] = None,
    ) raises HALError:
        """Compiles `kernel` for this queue's device, loads it, and enqueues
        it with `args` as `execution_config` describes.

        The context compiles and loads `kernel` behind the plugin's
        `FunctionWrapper` on first use and keeps it loaded, so later launches
        reuse it. See the overload that takes a loaded function for what
        `args` and `execution_config` must be.

        Parameters:
            declared_arg_types: The kernel's declared argument types.
            ExecutionConfigType: The type of `execution_config`.
            kernel: The kernel to launch.
            Ts: The argument types.

        Args:
            args: The kernel's arguments.
            execution_config: How to launch it.
            location: Where the launch was requested, for errors; the
                caller's location by default.

        Raises:
            If `kernel` can't be loaded, an argument can't be packed or the
            launch fails.
        """
        var loc = location.or_else(call_location())
        var func: FunctionHandle
        try:
            func = self._context[].compile_function_inline[
                Self._FunctionWrapper.wrap[kernel]
            ]()
        except e:
            raise HALError(e.status, message=loc.prefix(e.message))
        self.execute[kernel](
            *args,
            function=func,
            execution_config=execution_config,
            location=loc,
        )

    @doc_hidden
    def execute[
        declared_arg_types: TypeList[Trait=AnyType, ...],
        ExecutionConfigType: ExecutionConfig,
        //,
        kernel: def(*args: *declared_arg_types) thin raises -> None,
        *Ts: AnyType,
    ](
        self,
        *args: *Ts,
        execution_config: ExecutionConfigType,
        location: Optional[SourceLocation] = None,
    ) raises HALError:
        # Only a raising kernel lands here; the overload above takes the rest.
        comptime assert False, "a kernel launched by `execute` can't raise"

    # TODO: revisit all of these when we get to queue dependency ordering
    def execute[
        args_origin: ImmOrigin, //
    ](
        self,
        func: FunctionHandle,
        grid: Tuple[UInt32, UInt32, UInt32],
        block: Tuple[UInt32, UInt32, UInt32],
        args: Pointer[mut=True, OpaquePointer[args_origin], _],
        arg_sizes: ImmPointer[UInt64, _],
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
        args_origin: ImmOrigin,
        ExecutionConfigType: GridBlockExecutionConfig,
        //,
    ](
        self,
        func: FunctionHandle,
        execution_config: ExecutionConfigType,
        args: Pointer[mut=True, ImmOpaquePointer[args_origin], _],
        arg_sizes: Pointer[mut=True, UInt64, _],
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

        var near_compute_scratchpad_usage: Int = 0
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

        # Attributes the config carries are appended to the caller's. A
        # 1x1x1 cluster, the vendor configurations' default, emits no
        # attribute so that targets without cluster launch accept it.
        var config_attributes = List[LaunchAttribute]()
        comptime if conforms_to(
            ExecutionConfigType, LaunchAttributeHolderExecutionConfig
        ):
            config_attributes.extend(execution_config.get_launch_attributes())
        comptime if conforms_to(ExecutionConfigType, ClusterExecutionConfig):
            var cluster_dim = execution_config.get_cluster_dim()
            if (
                cluster_dim.x() != 1
                or cluster_dim.y() != 1
                or cluster_dim.z() != 1
            ):
                config_attributes.append(
                    LaunchAttribute.from_cluster_dim(cluster_dim)
                )

        if config_attributes:
            var all_attributes = List[LaunchAttribute](
                capacity=Int(num_attributes) + len(config_attributes)
            )
            if case .Some(caller_attributes) = attributes:
                var src = caller_attributes.unsafe_bitcast[LaunchAttribute]()
                for i in range(Int(num_attributes)):
                    all_attributes.append(src[unsafe_offset=i])
            all_attributes.extend(config_attributes^)
            self._raw[].execute_function(
                self._handle,
                func,
                grid,
                block,
                args,
                arg_sizes,
                num_args,
                shared_mem_bytes=UInt32(near_compute_scratchpad_usage),
                attributes=OptionalReg(
                    all_attributes.unsafe_ptr()
                    .unsafe_bitcast[NoneType]()
                    .unsafe_origin_cast[MutUntrackedOrigin]()
                ),
                num_attributes=UInt32(len(all_attributes)),
            )
            # Keep attributes alive until the plugin has read them.
            _ = all_attributes^
            return

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
        src: Pointer[mut=False, UInt8, _],
    ) raises HALError:
        """
        Explicit host-to-device copy of `dst.byte_size` bytes into `dst`.
        Enqueues on this queue. Totally ordered with respect to other
        operations within this queue if backed by a stream.
        """
        self._raw[].copy_to_device(self._handle, dst._view, src)

    def copy_from_device(
        self,
        dst: Pointer[mut=True, UInt8, _],
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

    def copy_inter_device(
        self,
        dst: BufferView,
        src: BufferView,
        src_context: ContextHandle,
    ) raises HALError:
        """
        Peer copy of `dst.byte_size` bytes from `src` on `src_context`'s
        device into `dst`. Not ordered against work on the source device.
        """
        debug_assert(
            src.byte_size() >= dst.byte_size(),
            "copy_inter_device source view smaller than destination",
        )
        self._raw[].copy_inter_device(
            self._handle, dst._view, src._view, src_context
        )

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

        var handles = Array[EventHandle, n](
            fill_with_unrolled=lambda [i: Int]() -> EventHandle: events[
                i
            ]._handle()
        )
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
