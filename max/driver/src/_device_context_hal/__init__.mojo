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

# Implementation of DeviceContext backed by the HAL

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.builtin.rebind import downcast
from std.collections.optional import OptionalReg
from std.ffi import CStringSlice, _CPointer, _Global, c_size_t, external_call
from std.compile import CompiledFunctionInfo
from std.math import align_up
from std.memory import (
    alloc,
    dealloc,
    memcpy,
    ThinAllocation,
    ArcPointer,
    Layout,
    UnsafeMaybeUninit,
    UnsafePointer,
)
from std.memory.unsafe import bitcast
from std.pathlib import Path
from std.utils import Variant
from std.memory import stack_allocation
from std.memory.pointer import AddressSpace
from std.os import abort, getenv
from std.reflection import call_location, reflect, SourceLocation
from std.sys import bit_width_of, size_of
from std.sys.info import _TargetType, _current_target, is_gpu
from std.time import monotonic
from std.gpu.host._hal import (
    Buffer,
    Context,
    Device,
    Driver,
    Event,
    FunctionHandle,
    RuntimeBundle,
    Stream,
    get_device_spec,
)
from std.gpu.host._hal.event import EVENT_FLAG_CPU_VISIBLE
from std.gpu.host._hal.execution_config import (
    ExecutionConfig,
    BlockExecutionConfig,
    GridBlockExecutionConfig,
    NearComputeGeneralPurposeScratchpadExecutionConfig,
    GPUExecutionConfiguration,
    ClusterExecutionConfig,
    LaunchAttributeHolderExecutionConfig,
    ConstantMemoryMappingExecutionConfig,
)
from std.builtin.variadics import TypeList

from std.gpu.host.constant_memory_mapping import ConstantMemoryMapping
from std.gpu.host.device_attribute import DeviceAttribute
from std.gpu.host.dim import Dim
from std.gpu.host.info import GPUInfo
from std.gpu.host.launch_attribute import LaunchAttribute


def _check_device_context_hal_only_supported_exec_config[
    ExecutionConfigType: ExecutionConfig
](execution_config: ExecutionConfigType) raises:
    # HAL doesn't yet support cluster launch or arbitrary launch
    # attributes; the underlying Stream.execute primitive surfaces only
    # `shared_mem_bytes`. Refuse non-default values rather than silently
    # dropping them.
    comptime assert not conforms_to(
        ExecutionConfigType, ClusterExecutionConfig
    ), "HAL DeviceContext.enqueue_function does not support `cluster_dim`."

    comptime assert not conforms_to(
        ExecutionConfigType, LaunchAttributeHolderExecutionConfig
    ), (
        "HAL DeviceContext.enqueue_function does not support launch"
        " `attributes`."
    )
    comptime assert not conforms_to(
        ExecutionConfigType, ConstantMemoryMappingExecutionConfig
    ), (
        "HAL DeviceContext.enqueue_function does not support"
        " `constant_memory` mappings."
    )


def _check_dim[
    func_name_for_msg: StringLiteral, dim_name_for_msg: StringLiteral
](dim: Dim, *, location: SourceLocation) raises:
    if dim.x() <= 0:
        comptime msg = String(
            func_name_for_msg,
            ": Dim value ",
            dim_name_for_msg,
            ".x must be a positive number.",
        )
        raise Error(location.prefix(msg))
    if dim.y() <= 0:
        comptime msg = String(
            func_name_for_msg,
            ": Dim value ",
            dim_name_for_msg,
            ".y must be a positive number.",
        )
        raise Error(location.prefix(msg))
    if dim.z() <= 0:
        comptime msg = String(
            func_name_for_msg,
            ": Dim value ",
            dim_name_for_msg,
            ".z must be a positive number.",
        )
        raise Error(location.prefix(msg))


trait _HALFunctionEnqueuer:
    """HAL equivalent of DeviceContext's `_FunctionEnqueuer`.

    Both `DeviceContext` and `DeviceStream` conform; their `_hal_stream()`
    surfaces the underlying `Stream` that `DeviceFunction._call_with_pack_checked`
    enqueues kernels on.
    """

    def _hal_stream(
        self,
    ) -> ArcPointer[Stream[get_device_spec[0]()]]:
        ...


@fieldwise_init
struct _DeviceFunctionInner(Movable):
    """Wrapper around a HAL-loaded `FunctionHandle`.

    Owns the function handle, the `RuntimeBundle` it was loaded from, and
    the `Context` needed to unload the bundle. The bundle is kept alive for the
    lifetime of the function handle - destroying the bundle invalidates the
    function symbol it owns.
    """

    var _func_handle: FunctionHandle
    var _bundle: RuntimeBundle
    var _context: ArcPointer[Context[get_device_spec[0]()]]
    # Stable copy of the kernel's per-capture byte sizes, taken at
    # `DeviceFunction.__init__`. `CompiledFunctionInfo.capture_sizes` points at
    # elaborator stack storage that may be clobbered by the time the  HAL launch
    # path reads it.
    var _capture_sizes: UnsafePointer[UInt64, MutUntrackedOrigin]
    var _num_captures: Int

    def __del__(deinit self):
        try:
            self._context[].unload_function(self._func_handle)
        except e:
            print("warning: unload_function failed:", e)
        dealloc(
            ThinAllocation(
                unsafe_assume_ownership=self._capture_sizes
            ).unsafe_with_layout({count = max(self._num_captures, 1)})
        )
        # Unloading must precede the bundle release (the bundle owns the
        # loaded binary), and the context must outlive both.
        _ = self._bundle^
        _ = self._context^


def _driver_satisfies_api(driver_name: String, api: String) -> Bool:
    """Reports whether the single loaded HAL driver satisfies a requested API.

    The HAL backend loads exactly one accelerator driver. A request is
    satisfied by a case-insensitive match against the driver's name (for
    example `"cuda" == "cuda"`), or by the generic accelerator alias `"gpu"`,
    which any loaded GPU driver answers regardless of its concrete vendor name.
    """
    var d = driver_name.lower()
    var a = api.lower()
    if a == "gpu":
        return d != "cpu"
    return d == a


def _load_gpu_driver() -> Optional[ArcPointer[Driver]]:
    """Loads the accelerator plugin named by `MODULAR_DRIVER_PLUGINS`.

    Used as the one-time initializer for `_GPU_DRIVER`; see that global. It
    backs a `_Global` initializer, so it must not raise: a missing or
    unloadable plugin is reported as an empty `Optional`.
    """
    var plugin_spec = getenv("MODULAR_DRIVER_PLUGINS")
    if not plugin_spec:
        return None
    try:
        # The spec may list several plugins ("CUDA@/x.so;CPU@/y.so"); load the
        # first accelerator (non-CPU) entry.
        for entry in plugin_spec.split(";"):
            var name = String(entry.split("@")[0])
            if _driver_satisfies_api(name, "gpu"):
                return Optional(Driver.create(String(entry)))
        return None
    except:
        return None


comptime _GPU_DRIVER = _Global["MODULAR_HAL_GPU_DRIVER", _load_gpu_driver]


@fieldwise_init
struct StreamPriorityRange(TrivialRegisterPassable, Writable):
    """Represents the range of valid stream priorities for a GPU device.

    Stream priorities control the scheduling of GPU operations, with higher
    priority streams being executed preferentially over lower priority streams.
    """

    var least: Int
    """The lowest (numerically smallest) priority value."""

    var greatest: Int
    """The highest (numerically largest) priority value."""

    @always_inline
    def write_to(self, mut writer: Some[Writer]):
        """Writes the stream priority range to the given writer.

        Args:
            writer: The writer to output the stream priority range to.
        """
        writer.write(
            "StreamPriorityRange(least=",
            self.least,
            ", greatest=",
            self.greatest,
            ")",
        )


struct DeviceContext(
    ImplicitlyCopyable, RegisterPassable, _HALFunctionEnqueuer
):
    """Represents a single stream of execution on a particular accelerator
    (GPU).

    A `DeviceContext` serves as the low-level interface to the
    accelerator inside a MAX [custom operation](/max/develop/custom-ops/) and provides
    methods for allocating buffers on the device, copying data between host and
    device, and for compiling and running functions (also known as kernels) on
    the device.

    The device context can be used as a
    [context manager](/docs/manual/errors/#use-a-context-manager). For example:

    ```mojo
    from std.gpu.host import DeviceContext
    from std.gpu import thread_idx

    def kernel():
        print("hello from thread:", thread_idx.x, thread_idx.y, thread_idx.z)

    with DeviceContext() as ctx:
        ctx.enqueue_function[kernel, kernel](grid_dim=1, block_dim=(2, 2, 2))
        ctx.synchronize()
    ```

    A custom operation receives the `DeviceContext` for the target device
    directly as an argument to its `execute` method:

    ```text
    from std.gpu.host import DeviceContext
    from extensibility import register

    @register("custom_op")
    struct CustomOp:
        @staticmethod
        def execute(ctx: DeviceContext) raises:
            ctx.enqueue_function[kernel, kernel](grid_dim=1, block_dim=(2, 2, 2))
            ctx.synchronize()
    ```
    """

    comptime device_spec = get_device_spec[0]()

    comptime default_device_info = GPUInfo.from_target[
        Self.device_spec._mlir_target()
    ]()

    var _driver: ArcPointer[Driver]
    var _device: ArcPointer[Device[Self.device_spec]]
    var _context: ArcPointer[Context[Self.device_spec]]
    var _stream: ArcPointer[Stream[Self.device_spec]]
    # Null until the C++ runtime shim provides it; read by the CUDA/HIP
    # interop accessors (`_nvidia_cuda` / `_amdgpu_hip`).
    var _handle: _DeviceContextPtr[mut=True]

    @always_inline
    def __init__(
        out self,
        device_id: Int = 0,
        *,
        var api: String = String(Self.default_device_info.api),
    ) raises:
        """Constructs a `DeviceContext` for the specified device.

        This initializer creates a new device context for the specified accelerator device.
        The device context provides an interface for interacting with the GPU, including
        memory allocation, data transfer, and kernel execution.

        Args:
            device_id: ID of the accelerator device. If not specified, uses
                the default accelerator (device 0).
            api: Requested device API (for example, "cuda" or "hip"). Defaults
                to the device API specified by current target accelerator.

        Raises:
            If device initialization fails or the specified device is not available.

        Example:

        ```mojo
        from std.gpu.host import DeviceContext

        # Create a context for the default GPU
        var ctx = DeviceContext()

        # Create a context for a specific GPU (device 1)
        var ctx2 = DeviceContext(1)
        ```
        """

        var plugin_spec = getenv("MODULAR_DRIVER_PLUGINS")
        if not plugin_spec:
            raise Error("MODULAR_DRIVER_PLUGINS not set")

        # The spec may list several plugins ("CUDA@/x.so;CPU@/y.so"); load the
        # first whose name satisfies the requested api (its concrete vendor
        # name, "cpu", or the generic accelerator alias "gpu").
        var chosen = String("")
        for entry in plugin_spec.split(";"):
            var name = String(entry.split("@")[0])
            if _driver_satisfies_api(name, api):
                chosen = String(entry)
                break
        if not chosen:
            raise Error(
                String(
                    t"Requested API {api} not provided by"
                    t" MODULAR_DRIVER_PLUGINS"
                )
            )

        self._handle = None
        self._driver = Driver.create(chosen)

        # Validate that the loaded plugin satisfies the requested api (its
        # concrete vendor name, or the generic accelerator alias "gpu").
        var driver_name = self._driver[].get_name()
        if not _driver_satisfies_api(String(driver_name), api):
            raise Error(
                String(
                    t"Requested API {api} not supported by driver {driver_name}"
                )
            )

        # TODO: DRIV-163 - Use real device_id
        self._device = self._driver[].get_device[0]()
        self._context = self._device[].get_context()
        self._stream = self._context[].create_stream()

    def __enter__(var self) -> Self:
        """Enables the use of `DeviceContext` in a `with` statement context manager.

        Returns:
            The `DeviceContext` instance to be used within the context manager block.
        """
        return self^

    def synchronize(self) raises:
        """Blocks until all asynchronous calls on the stream associated with
        this device context have completed.

        Raises:
            If the operation fails.
        """
        self._stream[].synchronize()

    def id(self) -> Int64:
        """Returns the ID associated with this device.

        Returns:
            The unique device ID as an `Int64`.
        """
        return self._device[].id

    @always_inline
    def get_memory_info(self) raises -> Tuple[c_size_t, c_size_t]:
        """Returns the free and total memory size for this device.

        This method queries the current state of device memory, providing information
        about how much memory is available and the total memory capacity of the device.
        This is useful for memory management and determining if there's enough space
        for planned operations.

        Returns:
            A tuple of (free memory, total memory) in bytes.

        Raises:
            If there's an error retrieving the memory information.

        Example:

        ```mojo
        from std.gpu.host import DeviceContext

        var ctx = DeviceContext()
        try:
            (free, total) = ctx.get_memory_info()
            print("Free memory:", free / (1024*1024), "MB")
            print("Total memory:", total / (1024*1024), "MB")
        except:
            print("Failed to get memory information")
        ```
        """
        var info = self._context[].get_memory_info()
        return (c_size_t(info[0]), c_size_t(info[1]))

    @always_inline
    def get_attribute(self, attr: DeviceAttribute) raises -> Int:
        """Returns the specified attribute for this device.

        Use the aliases defined by
        [DeviceAttribute](/docs/std/gpu/host/device_attribute/DeviceAttribute/)
        to specify attributes. For example:

        ```mojo
        from std.gpu.host import DeviceAttribute, DeviceContext

        def main() raises:
            var ctx = DeviceContext()
            var attr = DeviceAttribute.MAX_BLOCKS_PER_MULTIPROCESSOR
            var max_blocks = ctx.get_attribute(attr)
            print(max_blocks)
        ```

        Args:
            attr: The device attribute to query.

        Returns:
            The value for `attr` on this device.

        Raises:
            If the operation fails.
        """
        return Int(self._device[].get_attribute(attr._value))

    @doc_hidden
    @always_inline
    def compute_capability(self) raises -> Int:
        """Returns the compute capability of this NVIDIA GPU device.

        This internal method retrieves the compute capability version of the current
        NVIDIA GPU device. The compute capability is a version number that identifies
        the features supported by the CUDA hardware.

        Returns:
            The compute capability as an integer (e.g., 70 for 7.0, 86 for 8.6).

        Raises:
            If there's an error retrieving the compute capability.

        Notes:

        This is a private method intended for internal use only.
        """
        var major = self.get_attribute(DeviceAttribute.COMPUTE_CAPABILITY_MAJOR)
        var minor = self.get_attribute(DeviceAttribute.COMPUTE_CAPABILITY_MINOR)
        return major * 10 + minor

    def set_as_current(self) raises:
        """For use with libraries that require a specific GPU context to be
        active. Sets the current device to the one associated with this
        DeviceContext.

        Example:

        ```mojo
        from std.gpu.host import DeviceContext
        var ctx = DeviceContext(device_id=1)
        ctx.set_as_current()
        ```

        Raises:
            If there's an error setting the current device.
        """
        self._context[].set_current()

    def push_context(self) raises -> _DeviceContextScopeHAL:
        """Returns a context manager that ensures this device's driver context is active.

        This method returns a context manager that pushes this device's driver
        context as the current context on entry and restores the previous context
        on exit. This is useful for operations that require a specific GPU context
        to be active, such as cuDNN operations on multi-GPU systems.

        Returns:
            A context manager that manages the driver context stack.

        Raises:
            If there's an error switching contexts.

        Example:

        ```mojo
        from std.gpu.host import DeviceContext

        var ctx = DeviceContext(device_id=1)
        # Ensure GPU 1's context is active for these operations.
        with ctx.push_context():
            # All GPU operations here will use GPU 1's context.
            ...  # call external stateful APIs, such as cudnn.
        # Previous context is automatically restored
        ```
        """
        return _DeviceContextScopeHAL(self)

    def stream(self) -> DeviceStream:
        return DeviceStream(self)

    @doc_hidden
    def _hal_stream(
        self,
    ) -> ArcPointer[Stream[get_device_spec[0]()]]:
        return self._stream

    def create_stream(self, *, priority: Int = 0) raises -> DeviceStream:
        """Creates a new stream associated with the given device context.

        Args:
            priority: Scheduling priority for the stream. The HAL queue API
                has no priority support, so this is accepted for parity with
                the legacy backend and ignored.

        Returns:
            The newly created device stream.

        Raises:
            If stream creation fails.
        """
        var hal_stream = self._context[].create_stream()
        return DeviceStream(self, hal_stream^)

    def create_event(self) -> DeviceEvent:
        """Creates a new event for synchronization between streams.

        Returns:
            A DeviceEvent that can be used for synchronization.

        Example:

        ```mojo
        from std.gpu.host import DeviceContext

        var ctx = DeviceContext()

        var default_stream = ctx.stream()
        var new_stream = ctx.create_stream()

        # Create an event
        var event = ctx.create_event()

        # Wait for the event in new_stream
        new_stream.enqueue_wait_for(event)

        # new_stream can continue
        default_stream.record_event(event)
        default_stream.synchronize()
        ```
        """

        # The HAL uses logical events where creation and recording happen in
        # one step. To work around this, initially create a DeviceEvent without
        # a backing HAL event.
        return DeviceEvent._create_unrecorded(self)

    @always_inline
    def compile_function[
        declared_arg_types: TypeList[Trait=AnyType, ...],
        //,
        func: def(* args: * declared_arg_types) thin -> None,
        # Passthrough parameters, accepted for API parity with the legacy
        # backend and ignored: the HAL compile path derives compile options
        # from the device spec and does not emit asm/LLVM/SASS dumps or honor
        # link options.
        compile_options: StaticString = "",
        link_options: StaticString = "",
        dump_asm: _DumpPath = False,
        dump_llvm: _DumpPath = False,
        _dump_sass: _DumpPath = False,
        _ptxas_info_verbose: Bool = False,
    ](
        self, *, func_attribute: OptionalReg[FuncAttribute] = None
    ) raises -> DeviceFunction[func, declared_arg_types]:
        """Compiles the provided function for execution on this device.

        Parameters:
            declared_arg_types: Types of the arguments to pass to the device function.
            func: The function to compile.

        Returns:
            The compiled function.

        Raises:
            If the operation fails.
        """
        return DeviceFunction[func, declared_arg_types](
            self, func_attribute=func_attribute
        )

    @always_inline
    def compile_function[
        declared_arg_types: TypeList[Trait=AnyType, ...],
        //,
        func: def(* args: * declared_arg_types) capturing -> None,
        # Passthrough parameters, accepted for API parity with the legacy
        # backend and ignored: the HAL compile path derives compile options
        # from the device spec and does not emit asm/LLVM/SASS dumps or honor
        # link options.
        compile_options: StaticString = "",
        link_options: StaticString = "",
        dump_asm: _DumpPath = False,
        dump_llvm: _DumpPath = False,
        _dump_sass: _DumpPath = False,
        _ptxas_info_verbose: Bool = False,
    ](
        self, *, func_attribute: OptionalReg[FuncAttribute] = None
    ) raises -> DeviceFunction[func, declared_arg_types]:
        """Compiles the provided `capturing` function for execution on this
        device.

        Parameters:
            declared_arg_types: Types of the arguments to pass to the device function.
            func: The function to compile.

        Returns:
            The compiled function.

        Raises:
            If the operation fails.
        """
        return DeviceFunction[func, declared_arg_types](
            self, func_attribute=func_attribute
        )

    @parameter
    @always_inline
    def enqueue_function[
        declared_arg_types: TypeList[Trait=AnyType, ...],
        ExecutionConfigType: GridBlockExecutionConfig,
        //,
        func: def(* args: * declared_arg_types) thin -> None,
        *actual_arg_types: DevicePassable,
    ](
        self,
        mut execution_config: ExecutionConfigType,
        *args: *actual_arg_types,
        location: OptionalReg[SourceLocation] = None,
    ) raises:
        """Compiles and enqueues a kernel for execution on this device.

        Parameters:
            declared_arg_types: Types of the arguments to pass to the device function.
            func: The function to compile and launch.
            actual_arg_types: The dtypes of the arguments being passed to the function.

        Args:
            execution_config: The execution configuration specifying device specific kernel launch arguments.
            args: Variadic arguments which are passed to the `func`.
            location: Source location for the function call.

        You can pass the function directly to `enqueue_function`
        without compiling it first:

        ```mojo
        from std.gpu.host import DeviceContext
        from max.driver._hal.execution_config import GPUExecutionConfiguration

        def kernel():
            print("hello from the GPU")

        with DeviceContext() as ctx:
            ctx.enqueue_function[kernel](GPUExecutionConfiguration(grid_dim=Dim(1, 1, 1), block_dim=Dim(1, 1, 1)))
            ctx.synchronize()
        ```

        If you are reusing the same function and parameters multiple times,
        this incurs 50-500 nanoseconds of overhead per enqueue, so you can
        compile it first to remove the overhead:

        ```mojo
        from std.gpu.host import DeviceContext

        def kernel():
            print("hello from the GPU")

        with DeviceContext() as ctx:
            var compiled_func = ctx.compile_function[kernel]()
            var config = GPUExecutionConfiguration(grid_dim=Dim(1, 1, 1), block_dim=Dim(1, 1, 1))
            ctx.enqueue_function(compiled_func, config)
            ctx.synchronize()
        ```

        Raises:
            If the operation fails.
        """
        var gpu_kernel = self.compile_function[func]()
        self.enqueue_function(
            execution_config, gpu_kernel, *args, location=location
        )

    @parameter
    @always_inline
    def enqueue_function[
        FuncType: def() -> None,
        ExecutionConfigType: GridBlockExecutionConfig,
        //,
    ](
        self,
        mut execution_config: ExecutionConfigType,
        func: FuncType,
        location: OptionalReg[SourceLocation] = None,
    ) raises:
        """Compiles and enqueues a capturing kernel for execution on this device.

        This overload is for kernels that capture variables from their enclosing scope.
        The `capturing` annotation on the signature function indicates that the kernel
        can access variables from the surrounding context.

        Parameters:
            FuncType: The type of the function to launch (usually inferred).

        Args:
            execution_config: The execution configuration specifying device specific kernel launch arguments.
            func: The capturing kernel function to compile and launch.
            location: Source location for the function call.

        Raises:
            If the operation fails.
        """
        var grid_dim = execution_config.get_grid_dim()
        var block_dim = execution_config.get_block_dim()

        _check_dim["DeviceContext.enqueue_function", "grid_dim"](
            grid_dim, location=call_location()
        )
        _check_dim["DeviceContext.enqueue_function", "block_dim"](
            block_dim, location=call_location()
        )

        var gpu_kernel = DeviceFunction[
            FuncType.__call__, TypeList.of[Trait=AnyType]()
        ](self)

        gpu_kernel._call_with_pack(
            self,
            execution_config,
            func,
            location=location.or_else(call_location()),
        )

    @parameter
    @always_inline
    def enqueue_function[
        ExecutionConfigType: GridBlockExecutionConfig,
        //,
        *Ts: DevicePassable,
    ](
        self,
        mut execution_config: ExecutionConfigType,
        f: DeviceFunction,
        *args: *Ts,
        location: OptionalReg[SourceLocation] = None,
    ) raises:
        """Enqueues a pre-compiled checked function for execution on this device.

        This overload requires a `DeviceFunction` that was compiled with
        type checking enabled (via `compile_function`). The function
        will verify that the argument types match the declared types at
        compile time.

        Parameters:
            ExecutionConfigType: The type of the execution configuration.
            Ts: Argument dtypes.

        Args:
            execution_config: The execution configuration specifying device specific kernel launch arguments.
            f: The compiled function to execute.
            args: Arguments to pass to the function.
            location: Source location for the function call.

        ```mojo
        from std.gpu.host import DeviceContext

        def kernel(x: Int):
            print("Value:", x)

        with DeviceContext() as ctx:
            var compiled_func = ctx.compile_function[kernel]()
            var cfg = GPUExecutionConfiguration(grid_dim=Dim(1, 1, 1), block_dim=Dim(1, 1, 1))
            ctx.enqueue_function(cfg, compiled_func, 42)
            ctx.synchronize()
        ```

        Raises:
            If the operation fails.
        """

        var grid_dim = execution_config.get_grid_dim()
        var block_dim = execution_config.get_block_dim()

        _check_dim["DeviceContext.enqueue_function", "grid_dim"](
            grid_dim, location=call_location()
        )
        _check_dim["DeviceContext.enqueue_function", "block_dim"](
            block_dim, location=call_location()
        )
        f._call_with_pack_checked(
            self,
            execution_config,
            *args,
            location=location.or_else(call_location()),
        )

    @parameter
    @always_inline
    def enqueue_function[
        declared_arg_types: TypeList[Trait=AnyType, ...],
        //,
        func: def(* args: * declared_arg_types) thin -> None,
        *actual_arg_types: DevicePassable,
        # Debug/link passthrough parameters, accepted for API parity with the
        # legacy backend and ignored (the HAL compile path does not emit
        # asm/LLVM/SASS dumps or honor link options).
        link_options: StaticString = "",
        dump_asm: _DumpPath = False,
        dump_llvm: _DumpPath = False,
        _dump_sass: _DumpPath = False,
        _ptxas_info_verbose: Bool = False,
    ](
        self,
        *args: *actual_arg_types,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        func_attribute: OptionalReg[FuncAttribute] = None,
        location: OptionalReg[SourceLocation] = None,
    ) raises:
        """Compiles and enqueues a kernel for execution on this device.

        Parameters:
            declared_arg_types: Types of the arguments to pass to the device function.
            func: The function to compile and launch.
            actual_arg_types: The dtypes of the arguments being passed to the function.

        Args:
            args: Variadic arguments which are passed to the `func`.
            grid_dim: The grid dimensions.
            block_dim: The block dimensions.
            cluster_dim: The cluster dimensions.
            shared_mem_bytes: Per-block memory shared between blocks.
            attributes: A `List` of launch attributes.
            constant_memory: A `List` of constant memory mappings.
            location: Source location for the function call.

        You can pass the function directly to `enqueue_function`
        without compiling it first:

        ```mojo
        from std.gpu.host import DeviceContext

        def kernel():
            print("hello from the GPU")

        with DeviceContext() as ctx:
            ctx.enqueue_function[kernel](grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```

        If you are reusing the same function and parameters multiple times,
        this incurs 50-500 nanoseconds of overhead per enqueue, so you can
        compile it first to remove the overhead:

        ```mojo
        from std.gpu.host import DeviceContext

        def kernel():
            print("hello from the GPU")

        with DeviceContext() as ctx:
            var compiled_func = ctx.compile_function[kernel]()
            ctx.enqueue_function(compiled_func, grid_dim=1, block_dim=1)
            ctx.enqueue_function(compiled_func, grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```

        Raises:
            If the operation fails.
        """
        _check_dim["DeviceContext.enqueue_function", "grid_dim"](
            grid_dim, location=call_location()
        )
        _check_dim["DeviceContext.enqueue_function", "block_dim"](
            block_dim, location=call_location()
        )
        var inferred_func_attribute = func_attribute
        if not func_attribute and shared_mem_bytes:
            var max_shared = self._get_max_dynamic_shared_memory_bytes(
                shared_mem_bytes.value()
            )
            if max_shared > 0:
                inferred_func_attribute = (
                    FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(max_shared)
                )
        var gpu_kernel = self.compile_function[func](
            func_attribute=inferred_func_attribute
        )
        gpu_kernel._call_with_pack_checked(
            self,
            *args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
            location=location.or_else(call_location()),
        )

    @parameter
    @always_inline
    def enqueue_function[
        declared_arg_types: TypeList[Trait=AnyType, ...],
        //,
        func: def(* args: * declared_arg_types) capturing -> None,
        *actual_arg_types: DevicePassable,
        # Debug/link passthrough params, accepted for API parity with the
        # legacy backend and ignored (the HAL compile path does not emit
        # asm/LLVM/SASS dumps or honor link options).
        link_options: StaticString = "",
        dump_asm: _DumpPath = False,
        dump_llvm: _DumpPath = False,
        _dump_sass: _DumpPath = False,
        _ptxas_info_verbose: Bool = False,
    ](
        self,
        *args: *actual_arg_types,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        func_attribute: OptionalReg[FuncAttribute] = None,
        location: OptionalReg[SourceLocation] = None,
    ) raises:
        """Compiles and enqueues a `capturing` kernel for execution on this
        device.

        This overload accepts a kernel whose signature is `capturing` (for
        example, one parameterized by captured compile-time functions) and is
        otherwise identical to the non-capturing overload.

        Parameters:
            declared_arg_types: Types of the arguments to pass to the device function.
            func: The function to compile and launch.
            actual_arg_types: The dtypes of the arguments being passed to the function.
            link_options: Ignored; accepted for parity with the AsyncRT backend.
            dump_asm: Ignored; accepted for parity with the AsyncRT backend.
            dump_llvm: Ignored; accepted for parity with the AsyncRT backend.
            _dump_sass: Ignored; accepted for parity with the AsyncRT backend.
            _ptxas_info_verbose: Ignored; accepted for parity with the AsyncRT backend.

        Args:
            args: Variadic arguments which are passed to the `func`.
            grid_dim: The grid dimensions.
            block_dim: The block dimensions.
            cluster_dim: The cluster dimensions.
            shared_mem_bytes: Per-block memory shared between blocks.
            attributes: A `List` of launch attributes.
            constant_memory: A `List` of constant memory mappings.
            location: Source location for the function call.

        Raises:
            If the operation fails.
        """
        _check_dim["DeviceContext.enqueue_function", "grid_dim"](
            grid_dim, location=call_location()
        )
        _check_dim["DeviceContext.enqueue_function", "block_dim"](
            block_dim, location=call_location()
        )
        var inferred_func_attribute = func_attribute
        if not func_attribute and shared_mem_bytes:
            var max_shared = self._get_max_dynamic_shared_memory_bytes(
                shared_mem_bytes.value()
            )
            if max_shared > 0:
                inferred_func_attribute = (
                    FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(max_shared)
                )
        var gpu_kernel = DeviceFunction[func, declared_arg_types](
            self, func_attribute=inferred_func_attribute
        )
        gpu_kernel._call_with_pack_checked(
            self,
            *args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
            location=location.or_else(call_location()),
        )

    @parameter
    @always_inline
    def enqueue_function[
        FuncType: def() -> None,
        //,
    ](
        self,
        func: FuncType,
        *,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        location: OptionalReg[SourceLocation] = None,
    ) raises:
        """Compiles and enqueues a capturing kernel for execution on this device.

        This overload is for kernels that capture variables from their enclosing scope.
        The `capturing` annotation on the signature function indicates that the kernel
        can access variables from the surrounding context.

        Parameters:
            FuncType: The type of the function to launch (usually inferred).

        Args:
            func: The capturing kernel function to compile and launch.
            grid_dim: The grid dimensions.
            block_dim: The block dimensions.
            cluster_dim: The cluster dimensions.
            shared_mem_bytes: Per-block memory shared between blocks.
            attributes: A `List` of launch attributes.
            constant_memory: A `List` of constant memory mappings.
            location: Source location for the function call.

        Raises:
            If the operation fails.
        """
        _check_dim["DeviceContext.enqueue_function", "grid_dim"](
            grid_dim, location=call_location()
        )
        _check_dim["DeviceContext.enqueue_function", "block_dim"](
            block_dim, location=call_location()
        )
        var gpu_kernel = DeviceFunction[
            FuncType.__call__, TypeList.of[Trait=AnyType]()
        ](self)
        gpu_kernel._call_with_pack(
            self,
            func,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
            location=location.or_else(call_location()),
        )

    @parameter
    @always_inline
    def enqueue_function[
        *Ts: DevicePassable,
    ](
        self,
        f: DeviceFunction,
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        location: OptionalReg[SourceLocation] = None,
    ) raises:
        """Enqueues a pre-compiled checked function for execution on this device.

        This overload requires a `DeviceFunction` that was compiled with
        type checking enabled (via `compile_function`). The function
        will verify that the argument types match the declared types at
        compile time.

        Parameters:
            Ts: Argument dtypes.

        Args:
            f: The compiled function to execute.
            args: Arguments to pass to the function.
            grid_dim: Dimensions of the compute grid, made up of thread
                blocks.
            block_dim: Dimensions of each thread block in the grid.
            cluster_dim: Dimensions of clusters (if the thread blocks are
                grouped into clusters).
            shared_mem_bytes: Amount of shared memory per thread block.
            attributes: Launch attributes.
            constant_memory: Constant memory mapping.
            location: Source location for the function call.

        ```mojo
        from std.gpu.host import DeviceContext

        def kernel(x: Int):
            print("Value:", x)

        with DeviceContext() as ctx:
            var compiled_func = ctx.compile_function[kernel]()
            ctx.enqueue_function(compiled_func, 42, grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```

        Raises:
            If the operation fails.
        """
        _check_dim["DeviceContext.enqueue_function", "grid_dim"](
            grid_dim, location=call_location()
        )
        _check_dim["DeviceContext.enqueue_function", "block_dim"](
            block_dim, location=call_location()
        )
        f._call_with_pack_checked(
            self,
            *args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
            location=location.or_else(call_location()),
        )

    @always_inline
    def load_function[
        func_type: TrivialRegisterPassable,
        //,
        func: func_type,
    ](
        self,
        *,
        var function_name: String,
        var asm: String,
        out result: DeviceExternalFunction,
    ) raises:
        """Loads a pre-compiled device function from assembly code.

        This method loads an external GPU function from provided assembly code (PTX/SASS)
        rather than compiling it from Mojo source. This is useful for integrating with
        existing CUDA/HIP code or for using specialized assembly optimizations.

        Parameters:
            func_type: The dtype of the function to load.
            func: The function reference.

        Args:
            function_name: The name of the function in the assembly code.
            asm: The assembly code (PTX/SASS) containing the function.

        Returns:
            The loaded function is stored in the `result` parameter.

        Raises:
            If loading the function fails or the assembly code is invalid.
        """
        return DeviceExternalFunction(self, function_name^, asm)

    @parameter
    @always_inline
    def enqueue_function[
        *Ts: AnyType,
    ](
        self,
        f: DeviceExternalFunction,
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        location: OptionalReg[SourceLocation] = None,
    ) raises:
        """Enqueues an external device function for execution on this device.

        Parameters:
            Ts: Argument types to pass to the external function.

        Args:
            f: The external device function to execute.
            args: Arguments to pass to the function.
            grid_dim: The grid dimensions.
            block_dim: The block dimensions.
            cluster_dim: Optional cluster dimensions.
            shared_mem_bytes: Optional shared memory per block.
            attributes: Launch attributes.
            constant_memory: Constant memory mapping.
            location: Source location for the function call.

        Raises:
            If the operation fails.
        """
        _check_dim["DeviceContext.enqueue_function", "grid_dim"](
            grid_dim, location=call_location()
        )
        _check_dim["DeviceContext.enqueue_function", "block_dim"](
            block_dim, location=call_location()
        )
        f._call_with_pack(
            self,
            *args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
            location=location.or_else(call_location()),
        )

    # ===-------------------------------------------------------------------===#
    # Buffer operations
    # ===-------------------------------------------------------------------===#

    def enqueue_create_buffer[
        dtype: DType
    ](self, size: Int) raises -> DeviceBuffer[dtype]:
        """Enqueues a buffer creation using the `DeviceBuffer` constructor.

        For GPU devices, the space is allocated in the device's global memory.

        Parameters:
            dtype: The data type to be stored in the allocated memory.

        Args:
            size: The number of elements of `type` to allocate memory for.

        Returns:
            The allocated buffer.

        Raises:
            If the operation fails.
        """

        # NOTE: The HAL supports doing this asynchronously, but to match
        # existing DeviceContext semantics we create a buffer immediately.
        return DeviceBuffer[dtype](self, size)

    def create_buffer_sync[
        dtype: DType
    ](self, size: Int) raises -> DeviceBuffer[dtype]:
        """Creates a buffer synchronously using the `DeviceBuffer` constructor.

        Parameters:
            dtype: The data type to be stored in the allocated memory.

        Args:
            size: The number of elements of `type` to allocate memory for.

        Returns:
            The allocated buffer.

        Raises:
            If the operation fails.
        """
        return DeviceBuffer[dtype](self, size)

    def enqueue_create_host_buffer[
        dtype: DType
    ](self, size: Int) raises -> HostBuffer[dtype]:
        """Enqueues the creation of a HostBuffer.

        This function allocates memory on the host that is accessible by the device.
        The memory is page-locked (pinned) for efficient data transfer between host and device.

        Pinned memory is guaranteed to remain resident in the host's RAM, not be
        paged/swapped out to disk. Memory allocated normally (for example, using
        [`alloc()`](/docs/std/memory/unsafe_pointer/alloc/))
        is pageable—individual pages of memory can be moved to secondary storage
        (disk/SSD) when main memory fills up.

        Using pinned memory allows devices to make fast transfers
        between host memory and device memory, because they can use direct
        memory access (DMA) to transfer data without relying on the CPU.

        Allocating too much pinned memory can cause performance issues, since it
        reduces the amount of memory available for other processes.

        Parameters:
            dtype: The data type to be stored in the allocated memory.

        Args:
            size: The number of elements of `type` to allocate memory for.

        Returns:
            A `HostBuffer` object that wraps the allocated host memory.

        Raises:
            If memory allocation fails or if the device context is invalid.

        Example:

        ```mojo
        from std.gpu.host import DeviceContext

        with DeviceContext() as ctx:
            # Allocate host memory accessible by the device
            var host_buffer = ctx.enqueue_create_host_buffer[DType.float32](1024)

            # Use the host buffer for device operations
            # ...
        ```
        """
        return HostBuffer[dtype](self, size)

    def enqueue_copy[
        dtype: DType
    ](
        self,
        dst_ptr: UnsafePointer[mut=True, Scalar[dtype], _],
        src_ptr: UnsafePointer[Scalar[dtype], _],
        size: Int,
    ) raises:
        """Enqueues an async copy of `size` elements from a device pointer to
        another device pointer.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_ptr: Host pointer to copy to.
            src_ptr: Device pointer to copy from.
            size: Number of elements (of the specified `DType`) to copy.

        Raises:
            If the operation fails.
        """
        self.enqueue_copy(
            DeviceBuffer[dtype](self, dst_ptr, size, owning=False),
            DeviceBuffer[dtype](self, src_ptr, size, owning=False),
        )

    def enqueue_copy[
        dtype: DType
    ](
        self,
        dst_buf: DeviceBuffer[dtype],
        src_ptr: UnsafePointer[Scalar[dtype], _],
    ) raises:
        """Enqueues an async copy from the host to the provided device
        buffer. The number of bytes copied is determined by the size of the
        device buffer.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_buf: Device buffer to copy to.
            src_ptr: Host pointer to copy from.

        Raises:
            If the operation fails.
        """
        self._stream[].copy_to_device(
            dst_buf._inner[]._buffer.view(),
            src_ptr.bitcast[UInt8](),
        )

    def enqueue_copy[
        dtype: DType
    ](
        self,
        dst_ptr: UnsafePointer[mut=True, Scalar[dtype], _],
        src_buf: DeviceBuffer[dtype],
    ) raises:
        """Enqueues an async copy from the device to the host. The
        number of bytes copied is determined by the size of the device buffer.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_ptr: Host pointer to copy to.
            src_buf: Device buffer to copy from.

        Raises:
            If the operation fails.
        """
        self._stream[].copy_from_device(
            dst_ptr.bitcast[UInt8](),
            src_buf._inner[]._buffer.view(),
        )

    def enqueue_copy[
        dtype: DType
    ](self, dst_buf: DeviceBuffer[dtype], src_buf: DeviceBuffer[dtype]) raises:
        """Enqueues an async copy from one device buffer to another. The amount
        of data transferred is determined by the size of the destination buffer.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_buf: Device buffer to copy to.
            src_buf: Device buffer to copy from. Must be at least as large as
                `dst`.

        Raises:
            If the operation fails.
        """
        self._stream[].copy_intra_device(
            dst_buf._inner[]._buffer.view(),
            src_buf._inner[]._buffer.view(),
        )

    def enqueue_copy[
        dtype: DType
    ](self, dst_buf: DeviceBuffer[dtype], src_buf: HostBuffer[dtype]) raises:
        """Enqueues an async copy from one device buffer to another. The amount
        of data transferred is determined by the size of the destination buffer.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_buf: Device buffer to copy to.
            src_buf: Device buffer to copy from. Must be at least as large as
                `dst`.

        Raises:
            If the operation fails.
        """
        self._stream[].copy_to_device(
            dst_buf._inner[]._buffer.view(),
            src_buf.unsafe_ptr().bitcast[UInt8](),
        )

    def enqueue_copy[
        dtype: DType
    ](self, dst_buf: HostBuffer[dtype], src_buf: DeviceBuffer[dtype]) raises:
        """Enqueues an async copy from one device buffer to another. The amount
        of data transferred is determined by the size of the destination buffer.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_buf: Device buffer to copy to.
            src_buf: Device buffer to copy from. Must be at least as large as
                `dst`.

        Raises:
            If the operation fails.
        """
        self._stream[].copy_from_device(
            dst_buf.unsafe_ptr().bitcast[UInt8](),
            src_buf._inner[]._buffer.view(),
        )

    def enqueue_copy[
        dtype: DType
    ](
        self,
        dst_buf: HostBuffer[dtype],
        src_ptr: UnsafePointer[Scalar[dtype], _],
    ) raises:
        """Enqueues an async copy from the host to the provided device
        buffer. The number of bytes copied is determined by the size of the
        device buffer.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_buf: Device buffer to copy to.
            src_ptr: Host pointer to copy from.

        Raises:
            If the operation fails.
        """
        self._stream[].copy_to_device(
            dst_buf._inner[]._buffer.view(),
            src_ptr.bitcast[UInt8](),
        )

    def enqueue_copy[
        dtype: DType
    ](
        self,
        dst_ptr: UnsafePointer[mut=True, Scalar[dtype], _],
        src_buf: HostBuffer[dtype],
    ) raises:
        """Enqueues an async copy from the device to the host. The
        number of bytes copied is determined by the size of the device buffer.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_ptr: Host pointer to copy to.
            src_buf: Device buffer to copy from.

        Raises:
            If the operation fails.
        """
        self._stream[].copy_from_device(
            dst_ptr.bitcast[UInt8](),
            src_buf._inner[]._buffer.view(),
        )

    def enqueue_copy[
        dtype: DType
    ](self, dst_buf: HostBuffer[dtype], src_buf: HostBuffer[dtype]) raises:
        """Enqueues an async copy from one device buffer to another. The amount
        of data transferred is determined by the size of the destination buffer.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_buf: Device buffer to copy to.
            src_buf: Device buffer to copy from. Must be at least as large as
                `dst`.

        Raises:
            If the operation fails.
        """
        self._stream[].copy_intra_device(
            dst_buf._inner[]._buffer.view(),
            src_buf._inner[]._buffer.view(),
        )

    def enqueue_copy[
        dtype: DType
    ](self, dst_buf: DeviceBuffer[dtype], src: Span[Scalar[dtype], _]) raises:
        """Enqueues an async copy from a host `Span` to a device buffer.

        The number of bytes copied is determined by the size of the device
        buffer. The span must contain at least as many elements as the
        destination buffer; this invariant is checked via `debug_assert`.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_buf: Device buffer to copy to.
            src: Host span to copy from.

        Raises:
            If the operation fails.
        """
        debug_assert(
            len(src) >= len(dst_buf),
            "source span length must be >= destination buffer length",
        )
        self.enqueue_copy(dst_buf, src.unsafe_ptr())

    def enqueue_copy[
        dtype: DType
    ](
        self,
        dst: Span[mut=True, Scalar[dtype], _],
        src_buf: DeviceBuffer[dtype],
    ) raises:
        """Enqueues an async copy from a device buffer to a host `Span`.

        The number of bytes copied is determined by the size of the device
        buffer. The span must contain at least as many elements as the source
        buffer; this invariant is checked via `debug_assert` (debug builds
        only).

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst: Host span to copy to.
            src_buf: Device buffer to copy from.

        Raises:
            If the operation fails.
        """
        debug_assert(
            len(dst) >= len(src_buf),
            "destination span length must be >= source buffer length",
        )
        self.enqueue_copy(dst.unsafe_ptr(), src_buf)

    def enqueue_copy[
        dtype: DType
    ](self, dst_buf: HostBuffer[dtype], src: Span[Scalar[dtype], _]) raises:
        """Enqueues an async copy from a host `Span` to a pinned host buffer.

        The number of bytes copied is determined by the size of the device
        buffer. The span must contain at least as many elements as the
        destination buffer; this invariant is checked via `debug_assert`.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_buf: Device buffer to copy to.
            src: Host span to copy from.

        Raises:
            If the operation fails.
        """
        debug_assert(
            len(src) >= len(dst_buf),
            "source span length must be >= destination buffer length",
        )
        self.enqueue_copy(dst_buf, src.unsafe_ptr())

    def enqueue_copy[
        dtype: DType
    ](
        self,
        dst: Span[mut=True, Scalar[dtype], _],
        src_buf: HostBuffer[dtype],
    ) raises:
        """Enqueues an async copy from a host buffer to a host `Span`.

        The number of bytes copied is determined by the size of the source
        buffer. The span must contain at least as many elements as the source
        buffer; this invariant is checked via `debug_assert` (debug builds
        only).

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst: Host span to copy to.
            src_buf: Host buffer to copy from.

        Raises:
            If the operation fails.
        """
        debug_assert(
            len(dst) >= len(src_buf),
            "destination span length must be >= source buffer length",
        )
        self.enqueue_copy(dst.unsafe_ptr(), src_buf)

    def api(self) -> String:
        """Returns the name of the API used to program the device.

        Possible values are:

        - "cpu": Generic host device (CPU).
        - "cuda": NVIDIA GPUs.
        - "hip": AMD GPUs.

        Returns:
            A string identifying the device API.
        """
        # The HAL plugin names are "CUDA"/"HIP"/"Metal"; AsyncRT exposes
        # them lowercased so dispatchers can match `ctx.api() == "cuda"`.
        return String(self._driver[].get_name()).lower()

    def _get_max_dynamic_shared_memory_bytes(
        self, requested_bytes: Int
    ) -> UInt32:
        """Gets the maximum dynamic shared memory bytes for this device.

        For NVIDIA GPUs, dynamic shared memory defaults to 48KB max. For larger
        allocations, we set MAX_DYNAMIC_SHARED_SIZE_BYTES to the minimum of:
        - The device's maximum opt-in shared memory per block
        - The requested size rounded up to nearest 1KB boundary

        For smaller allocations (<= 48KB), we return 0 to skip setting the
        attribute (avoiding unnecessary API calls and potential errors).

        For AMD GPUs, the MAX_SHARED_MEMORY_PER_BLOCK_OPTIN attribute doesn't
        exist, so we return 0 (no automatic inference) and rely on explicit
        func_attribute settings when needed.

        Args:
            requested_bytes: The amount of shared memory requested by the kernel.

        Returns:
            Maximum dynamic shared memory bytes to set, or 0 if not needed.
        """
        # NVIDIA GPUs have a 48KB default limit for dynamic shared memory
        comptime NVIDIA_DEFAULT_DYNAMIC_SHARED_LIMIT = 48 * 1024

        # Only set the attribute if we need more than the default limit
        if requested_bytes <= NVIDIA_DEFAULT_DYNAMIC_SHARED_LIMIT:
            return 0

        # Try to query the maximum opt-in shared memory limit from the device.
        # This attribute is NVIDIA-specific (via cudaFuncSetAttribute) and may
        # not be available on AMD GPUs or other vendors.
        try:
            var capacity = self.get_attribute(
                DeviceAttribute.MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
            )

            # Sanity check: capacity should be reasonable (at least 48KB)
            if capacity < NVIDIA_DEFAULT_DYNAMIC_SHARED_LIMIT:
                # If the opt-in capacity is less than the default, something is wrong.
                # Fall back to not setting the attribute.
                return 0

            # Round requested_bytes up to nearest 1KB and use the minimum of
            # that and the device capacity minus 1KB system reservation
            var rounded_request = ((requested_bytes + 1023) // 1024) * 1024
            return UInt32(min(rounded_request, capacity - 1024))
        except:
            # Attribute not available (e.g., on AMD GPUs). Return 0 to skip
            # automatic inference. Code that needs >48KB on AMD should explicitly
            # set func_attribute.
            return 0

    def enqueue_copy_no_cross_stream_sync[
        dtype: DType
    ](self, dst_buf: DeviceBuffer[dtype], src_buf: DeviceBuffer[dtype],) raises:
        """Enqueues a device-to-device copy without cross-stream synchronization.

        This behaves like `enqueue_copy` for two device buffers, except that
        when the source and destination are on different streams the driver does
        not insert its own cross-stream synchronization; the caller is
        responsible for ordering (for example via explicit device-wait ops).

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_buf: Destination device buffer.
            src_buf: Source device buffer.

        Raises:
            If the operation fails.
        """
        # The HAL copy path never inserts cross-stream synchronization, so the
        # plain device-to-device copy already has the required semantics.
        self.enqueue_copy(dst_buf, src_buf)

    def enqueue_cpu_range[
        func: def(count: Int) capturing -> None,
    ](self, count: Int) raises:
        """Runs a function over a 1D range on a CPU `DeviceContext`.

        The function is called as `func(i)` for each `i` in `range(count)`.

        The HAL backend has no host-function enqueue plugin entry, so the range
        runs serially and synchronously here; `synchronize()` is then a no-op
        for this work. Results match the AsyncRT backend; only the parallelism
        differs.

        Parameters:
            func: The function to execute.

        Args:
            count: The number of instances of the function to run.

        Raises:
            If self is not a CPU DeviceContext.
        """
        if self.api() != "cpu":
            raise Error(
                "enqueue_cpu_range is only supported on CPU DeviceContexts"
            )
        for i in range(count):
            func(i)

    @always_inline
    def enqueue_cpu_range[
        FuncType: def(Int) -> None,
    ](self, func: FuncType, count: Int) raises:
        """Runs a function closure over a 1D range on a CPU `DeviceContext`.

        See the parameterized overload; this runs serially and synchronously
        under the HAL backend.

        Parameters:
            FuncType: The type of function to execute.

        Args:
            func: The function closure to execute.
            count: The number of instances of the function to run.

        Raises:
            If self is not a CPU DeviceContext.
        """
        if self.api() != "cpu":
            raise Error(
                "enqueue_cpu_range is only supported on CPU DeviceContexts"
            )
        for i in range(count):
            func(i)

    @always_inline
    def execution_time[
        func: def(Self) raises capturing[_] -> None
    ](self, num_iters: Int) raises -> Int:
        """Measures the execution time, in nanoseconds, of a function that takes
        this `DeviceContext`, run `num_iters` times (host monotonic clock,
        bracketed by `synchronize()`).

        Parameters:
            func: A function that takes a `DeviceContext` to execute and time.

        Args:
            num_iters: The number of iterations to run the function.

        Returns:
            The total elapsed time in nanoseconds for all iterations.
        """
        self.synchronize()
        var start = monotonic()
        for _ in range(num_iters):
            func(self)
        self.synchronize()
        return Int(monotonic() - start)

    @always_inline
    def execution_time[
        FuncType: def(Self) raises -> None,
    ](self, func: FuncType, num_iters: Int) raises -> Int:
        """Measures the execution time, in nanoseconds, of `func(self)` run
        `num_iters` times (host monotonic clock, bracketed by `synchronize()`).

        Parameters:
            FuncType: The body function type.

        Args:
            func: The closure carrying the captured state of the body function.
            num_iters: The number of iterations to run the function.

        Returns:
            The total elapsed time in nanoseconds for all iterations.
        """
        self.synchronize()
        var start = monotonic()
        for _ in range(num_iters):
            func(self)
        self.synchronize()
        return Int(monotonic() - start)

    @always_inline
    def execution_time[
        func: def() raises capturing[_] -> None
    ](self, num_iters: Int) raises -> Int:
        """Measures the execution time, in nanoseconds, of a no-argument function
        run `num_iters` times.

        Parameters:
            func: A function to execute and time.

        Args:
            num_iters: The number of iterations to run the function.

        Returns:
            The total elapsed time in nanoseconds for all iterations.
        """

        self.synchronize()
        var start = monotonic()
        for _ in range(num_iters):
            func()
        self.synchronize()
        return Int(monotonic() - start)

    @always_inline
    def execution_time[
        FuncType: def() raises -> None,
    ](self, func: FuncType, num_iters: Int) raises -> Int:
        """Measures the execution time, in nanoseconds, of `func()` run
        `num_iters` times (host monotonic clock, bracketed by `synchronize()`).

        Parameters:
            FuncType: The body function type.

        Args:
            func: The closure carrying the captured state of the body function.
            num_iters: The number of iterations to run the function.

        Returns:
            The total elapsed time in nanoseconds for all iterations.
        """
        self.synchronize()
        var start = monotonic()
        for _ in range(num_iters):
            func()
        self.synchronize()
        return Int(monotonic() - start)

    @always_inline
    def execution_time_iter[
        func: def(Self, Int) raises capturing[_] -> None
    ](self, num_iters: Int) raises -> Int:
        """Measures the execution time, in nanoseconds, of a function that takes
        this `DeviceContext` and the iteration index, run for `num_iters`
        iterations.

        Parameters:
            func: A function that takes a `DeviceContext` and iteration index.

        Args:
            num_iters: The number of iterations to run the function.

        Returns:
            The total elapsed time in nanoseconds for all iterations.
        """

        self.synchronize()
        var start = monotonic()
        for i in range(num_iters):
            func(self, i)
        self.synchronize()
        return Int(monotonic() - start)

    @always_inline
    def execution_time_iter[
        FuncType: def(Self, Int) raises -> None,
    ](self, func: FuncType, num_iters: Int) raises -> Int:
        """Measures the execution time, in nanoseconds, of `func(self, i)` for
        `i` in `range(num_iters)` (host monotonic clock, bracketed by
        `synchronize()`).

        Parameters:
            FuncType: The body function type.

        Args:
            func: The closure carrying the captured state of the body function.
            num_iters: The number of iterations to run the function.

        Returns:
            The total elapsed time in nanoseconds for all iterations.
        """
        self.synchronize()
        var start = monotonic()
        for i in range(num_iters):
            func(self, i)
        self.synchronize()
        return Int(monotonic() - start)

    def enqueue_memset[
        dtype: DType
    ](self, dst: DeviceBuffer[dtype], val: Scalar[dtype]) raises:
        """Enqueues an async memset operation, setting all of the elements in
        the destination device buffer to the specified value.

        Parameters:
            dtype: Type of the data stored in the buffer.

        Args:
            dst: Destination buffer.
            val: Value to set all elements of `dst` to.

        Raises:
            If the operation fails.
        """
        self._stream[].fill(
            dst._inner[]._buffer.view(),
            _memset_value_as_u64[dtype](val),
            UInt64(size_of[dtype]()),
        )

    def enqueue_memset[
        dtype: DType
    ](self, dst: HostBuffer[dtype], val: Scalar[dtype]) raises:
        """Enqueues an async memset operation, setting all of the elements in
        the destination host buffer to the specified value.

        Parameters:
            dtype: Type of the data stored in the buffer.

        Args:
            dst: Destination buffer.
            val: Value to set all elements of `dst` to.

        Raises:
            If the operation fails.
        """
        self._stream[].fill(
            dst._inner[]._buffer.view(),
            _memset_value_as_u64[dtype](val),
            UInt64(size_of[dtype]()),
        )

    @staticmethod
    def enable_all_peer_access() raises:
        """Enables peer-to-peer access between all accelerators.

        The HAL backend manages a single device, so there are no peers to
        enable and this is a no-op."""
        pass

    @staticmethod
    def all_peer_access_enabled() raises -> Bool:
        """Returns whether peer-to-peer access is enabled between all GPU pairs.

        The HAL backend manages a single device (fewer than two GPUs), so this
        returns False, matching the AsyncRT semantics for that case."""
        return False

    def enqueue_wait_for(self, other: DeviceContext) raises:
        """Enqueues a wait operation for another device context to complete its work.

        This method creates a dependency between two device contexts, ensuring that operations
        in the current context will not begin execution until all previously enqueued operations
        in the other context have completed. This is useful for synchronizing work across
        multiple devices or streams.

        Args:
            other: The device context whose operations must complete before operations in this context can proceed.

        Raises:
            If there's an error enqueuing the wait operation or if the operation
            is not supported by the underlying device API.

        Example:

        ```mojo
        from std.gpu.host import DeviceContext

        # Create two device contexts
        var ctx1 = DeviceContext(0)  # First GPU
        var ctx2 = DeviceContext(1)  # Second GPU

        # Enqueue operations on ctx1
        # ...

        # Make ctx2 wait for ctx1 to complete before proceeding
        ctx2.enqueue_wait_for(ctx1)

        # Enqueue operations on ctx2 that depend on ctx1's completion
        # ...
        ```
        """
        # Compose existing primitives: record an event on `other`'s default
        # stream, then make this context's stream wait on it.
        var event = other.create_event()
        other.stream().record_event(event)
        self.stream().enqueue_wait_for(event)

    @staticmethod
    def number_of_devices(
        *, var api: String = String(Self.default_device_info.api)
    ) -> Int:
        """Returns the number of devices available that support the specified API.

        Args:
            api: Requested device API (for example, "cuda" or "hip"). Defaults
                to the device API specified by the current target accelerator.

        Returns:
            The number of devices, or 0 if no driver plugin is available or the
            loaded plugin does not support the requested API.
        """
        # The built-in host (CPU) backend always provides a single device and
        # needs no plugin spec.
        if api.lower() == "cpu":
            return 1
        try:
            if not getenv("MODULAR_DRIVER_PLUGINS"):
                return 0
            # Reuse the process-global accelerator driver (loaded once).
            var maybe_driver = _GPU_DRIVER.get_or_create_ptr()
            if not maybe_driver[]:
                return 0
            var driver = maybe_driver[].value()
            if not _driver_satisfies_api(driver[].get_name(), api):
                return 0
            return Int(driver[].get_device_count())
        except:
            return 0

    def name(self) raises -> String:
        """Returns the device name as reported by the native driver.

        Returns:
            The device name (e.g. "NVIDIA B200").
        """
        return self._device[].get_name()

    @doc_hidden
    def __init__(out self, ctx_ptr: _DeviceContextPtr[mut=True]):
        """Create a Mojo DeviceContext from the opaque device-context handle
        carried by the runtime."""
        self = Self._from_unsafe_cpp_handle(
            ctx_ptr.unsafe_value().bitcast[NoneType]()
        )

    @always_inline
    def __init__(out self, handle: OpaquePointer[MutUntrackedOrigin]):
        """Reconstructs a `DeviceContext` from an opaque runtime handle.

        The graph runtime hands op code the opaque device-context handle; this
        recovers the Mojo `DeviceContext` it wraps.

        Args:
            handle: The opaque device-context handle carried by the runtime.
        """
        self = Self._from_unsafe_cpp_handle(handle)

    @staticmethod
    def _from_unsafe_cpp_handle(
        handle: OpaquePointer[MutUntrackedOrigin],
    ) -> DeviceContext:
        return handle.bitcast[DeviceContext]()[]

    def is_compatible(self) -> Bool:
        """Returns True if this device is compatible with MAX.

        This method checks whether the current device is compatible with the
        Modular Accelerated Execution (MAX) runtime. It's useful for validating
        that the device can execute the compiled code before attempting operations.

        Returns:
            True if the device is compatible with MAX, False otherwise.

        Example:

        ```mojo
        from std.gpu.host import DeviceContext

        var ctx = DeviceContext()
        print("Device is compatible with MAX:", ctx.is_compatible())
        ```
        """
        # TODO: HAL stub - always returns True. Hook this up to a plugin-side
        # compute-capability + driver-version check when real semantics
        # are needed.
        return True

    def stream_priority_range(self) raises -> StreamPriorityRange:
        """Returns the range of stream priorities supported by this device context.

        Returns:
            A StreamPriorityRange object containing the minimum and maximum stream priorities.

        Raises:
            If the operation fails.
        """
        # TODO: HAL stub - returns the trivial range.
        return StreamPriorityRange(0, 0)

    def get_api_version(self) raises -> Int:
        """Returns the API version associated with this device.

        This method retrieves the version number of the GPU driver currently installed
        on the system for the device associated with this context. The version is
        returned as an integer that can be used to check compatibility with specific
        features or to troubleshoot driver-related issues.

        Returns:
            An integer representing the driver version.

        Raises:
            If the driver version cannot be retrieved or if the device context is invalid.

        Example:

        ```mojo
        from std.gpu.host import DeviceContext

        with DeviceContext() as ctx:
            # Get the API version
            var api_version = ctx.get_api_version()
            print("GPU API version:", api_version)
        ```
        """
        # TODO: HAL stub - returns 0. Plumb plugin-side `cuDriverGetVersion`
        # / `hipDriverGetVersion` when callers actually need to gate on
        # driver version.
        return 0

    def run_healthcheck(self) raises:
        """Runs lightweight GPU health validation.

        Checks for hardware throttling, uncorrectable ECC errors, and stuck
        VRAM. Raises an error if the GPU is unhealthy. The healthcheck runs
        automatically during device initialization; this method allows
        re-running it explicitly.

        Disable with `MODULAR_DEVICE_CONTEXT_DISABLE_HEALTHCHECK=true`.

        Raises:
            Error: If the GPU is in an unhealthy state.
        """
        # TODO: HAL stub - no-op. Port AsyncRT's `evaluateGPUHealth` when
        # the healthcheck signal is required.
        pass

    def arch_name(self) raises -> String:
        """Returns the architecture name of this device.

        This internal method retrieves the architecture name of AMD GPUs.

        Returns:
            The compute capability as a string (e.g., `gfx942` for `MI300`).

        Raises:
            If there's an error retrieving the compute capability.

        Notes:

        This is a private method intended for internal use only.
        """
        if self.api() == "hip":
            return self._device[].get_arch()
        var cc = self.compute_capability()
        var arch = String("sm_", cc)
        if cc >= 90:
            arch += "a"
        return arch


struct _DeviceContextScopeHAL(Movable):
    var _device_context: DeviceContext
    # Driver-context handle to restore on exit; only meaningful when `_active`
    # (any value, including zero, can be a live handle).
    var _prev_driver_ctx: Int
    var _active: Bool

    def __init__(out self, device_context: DeviceContext):
        self._device_context = device_context
        self._prev_driver_ctx = 0
        self._active = False

    def __del__(deinit self):
        # Ensure restoration in all cases.
        if self._active:
            try:
                self._device_context._context[].set_current_driver_context(
                    self._prev_driver_ctx
                )
            except e:
                print("warning: restoring driver context failed:", e)

    def __enter__(mut self) raises -> DeviceContext:
        self._prev_driver_ctx = (
            self._device_context._context[].get_current_driver_context()
        )
        self._device_context._context[].set_current()
        self._active = True
        return self._device_context

    def __exit__(mut self) raises:
        if self._active:
            self._device_context._context[].set_current_driver_context(
                self._prev_driver_ctx
            )
            self._active = False


# ===-----------------------------------------------------------------------===#
# DeviceFunction
# ===-----------------------------------------------------------------------===#


struct DeviceFunction[
    func_type: TrivialRegisterPassable,
    //,
    func: func_type,
    declared_arg_types: TypeList[Trait=AnyType, ...],
](ImplicitlyCopyable, Movable):
    """Represents a compiled device function ready for execution on a GPU.

    The `DeviceFunction` struct encapsulates a compiled GPU kernel that can be
    executed on a device. It provides methods for managing the function's lifecycle,
    copying data to constant memory, and dumping debug information such as
    assembly code, LLVM IR, or SASS code.

    `DeviceFunction` is typically created through the `compile_function()`
    method of a `DeviceContext` rather than directly instantiated.

    Example:

    ```mojo
    from std.gpu.host import DeviceContext

    def my_kernel():
        # Kernel implementation
        pass

    with DeviceContext() as ctx:
        # Compile the kernel
        var kernel = ctx.compile_function[my_kernel, my_kernel]()
        # Enqueue the kernel for execution
        ctx.enqueue_function(kernel, grid_dim=1, block_dim=1)
    ```

    Parameters:
        func_type: Type of the kernel function (inferred).
        func: The kernel function value to compile.
        declared_arg_types: The kernel argument types used for compile-time
            validation in `_call_with_pack_checked`.
    """

    var _ctx: DeviceContext
    var _inner: ArcPointer[_DeviceFunctionInner]
    # Null until the C++ runtime shim provides it; read by the CUDA/HIP
    # interop accessors (`_nvidia_cuda` / `_amdgpu_hip`).
    var _handle: _DeviceFunctionPtr[mut=True]
    # Hold the `CompiledFunctionInfo` by value so the closure environment of
    # `Self.func` travels with this `DeviceFunction` value.
    var _func_info: CompiledFunctionInfo[
        Self.func_type, Self.func, get_device_spec[0]()._mlir_target()
    ]

    @doc_hidden
    @always_inline
    def __init__(
        out self,
        ctx: DeviceContext,
        *,
        func_attribute: OptionalReg[FuncAttribute] = None,
    ) raises:
        """Compiles `Self.func` for `ctx`'s device and loads the function.

        Args:
            ctx: The device context to compile for.
            func_attribute: Optional attribute to apply to the loaded function
                (for example a raised dynamic shared-memory cap).

        Raises:
            If compilation or function loading fails, or if an unsupported
            function attribute is provided.
        """
        # Compile and load the bundle as two steps (exactly what
        # `Context.compile` does internally).
        var info = ctx._context[]._compile_inner[Self.func_type, Self.func]()
        var bundle = ctx._context[].load_bundle(info.asm)
        var func_handle = ctx._context[].load_function(
            bundle, info.function_name
        )
        # Snapshot the per-capture byte sizes into stable heap storage now,
        # while `info.capture_sizes` is still valid. It points at elaborator
        # stack storage that is not guaranteed to survive to launch time.
        var snap_num_captures = max(0, info.num_captures)
        var snap_capture_sizes = alloc(
            Layout[UInt64](count=max(snap_num_captures, 1))
        ).unsafe_leak()
        for i in range(snap_num_captures):
            snap_capture_sizes[i] = info.capture_sizes[i]
        if func_attribute:
            if (
                func_attribute.value().attribute
                == Attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES
            ):
                ctx._context[].set_function_attribute(
                    func_handle,
                    func_attribute.value().attribute.code,
                    Int32(func_attribute.value().value),
                )
            else:
                raise Error(
                    "the function attribute '",
                    func_attribute.value().attribute,
                    "' is not currently supported",
                )
        self._ctx = ctx
        self._handle = None
        self._func_info = info
        # The `RuntimeBundle` owns the loaded binary; the function handle is
        # only valid while the bundle is alive. Move it into the refcounted
        # inner struct.
        self._inner = ArcPointer(
            _DeviceFunctionInner(
                func_handle,
                bundle^,
                ctx._context,
                snap_capture_sizes,
                snap_num_captures,
            )
        )

    @always_inline
    @staticmethod
    def _validate_arguments[
        *Ts: DevicePassable,
        num_args: Int,
    ]() -> Tuple[Int, InlineArray[Int, num_args]]:
        comptime declared_num_args = Self.declared_arg_types.size

        comptime assert (
            declared_num_args == num_args
        ), "Wrong number of arguments to enqueue"

        # For each argument determine the size of the device dtype and
        # calculate the offset into a contiguous memory area which will
        # be used to remap the passed arguments into the device dtypes.
        var tmp_arg_offset = 0
        var translated_arg_offsets = InlineArray[Int, num_args](
            uninitialized=True
        )
        var num_translated_args = 0

        comptime for i in range(num_args):
            comptime declared_arg_type = Self.declared_arg_types[i]
            comptime actual_arg_type = Ts[i]

            def declared_arg_type_name() -> String:
                comptime if conforms_to(declared_arg_type, DevicePassable):
                    return declared_arg_type.get_type_name()
                else:
                    return reflect[declared_arg_type].name()

            comptime is_convertible: Bool = actual_arg_type._is_convertible_to_device_type[
                declared_arg_type
            ]()

            comptime if actual_arg_type == actual_arg_type.device_type:
                comptime assert is_convertible, String(
                    "argument #",
                    i,
                    " of type '",
                    actual_arg_type.get_type_name(),
                    "' does not match the declared function argument type '",
                    declared_arg_type_name(),
                    "'",
                )
            else:
                comptime assert is_convertible, String(
                    "argument #",
                    i,
                    " of type '",
                    actual_arg_type.get_type_name(),
                    "' (which became device of type '",
                    declared_arg_type_name(),
                    "') does not match the declared function argument type",
                )
            var aligned_type_size = align_up(
                size_of[actual_arg_type.device_type](), 8
            )
            if aligned_type_size != 0:
                num_translated_args += 1
                translated_arg_offsets[i] = tmp_arg_offset
                tmp_arg_offset += aligned_type_size
            else:
                translated_arg_offsets[i] = -1

        return (num_translated_args, translated_arg_offsets^)

    @always_inline
    @parameter
    def _call_with_pack_checked[
        *Ts: DevicePassable,
        ContextT: _HALFunctionEnqueuer,
    ](
        imm self,
        ctx: ContextT,
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        location: OptionalReg[SourceLocation] = None,
    ) raises:
        # Launch `attributes` (e.g. PDL / programmatic stream serialization,
        # cluster dimensions) are forwarded to the plugin via the attribute
        # array assembled below. `constant_memory` mappings DO change
        # semantics and are not yet plumbed through, so refuse them rather
        # than silently dropping them.
        if cluster_dim:
            attributes.append(
                LaunchAttribute.from_cluster_dim(cluster_dim.value())
            )
        if constant_memory:
            raise Error(
                "HAL DeviceContext.enqueue_function does not support"
                " `constant_memory` mappings."
            )

        debug_assert(
            shared_mem_bytes.or_else(0) >= 0,
            "shared_mem_bytes must be non-negative",
        )

        var config = GPUExecutionConfiguration(grid_dim, block_dim)
        config.set_near_compute_scratchpad_usage(
            UInt64(shared_mem_bytes.or_else(0))
        )

        self._call_with_pack_checked[*Ts, ContextT=ContextT](
            ctx, config, *args, attributes=attributes^, location=location
        )

    @always_inline
    @parameter
    def _call_with_pack_checked[
        ExecutionConfigType: GridBlockExecutionConfig,
        //,
        *Ts: DevicePassable,
        ContextT: _HALFunctionEnqueuer,
    ](
        imm self,
        ctx: ContextT,
        mut execution_config: ExecutionConfigType,
        *args: *Ts,
        var attributes: List[LaunchAttribute] = [],
        location: OptionalReg[SourceLocation] = None,
    ) raises:
        _check_device_context_hal_only_supported_exec_config(execution_config)

        var shared_mem_bytes: UInt64 = 0

        comptime if conforms_to(
            ExecutionConfigType,
            NearComputeGeneralPurposeScratchpadExecutionConfig,
        ):
            shared_mem_bytes = (
                execution_config.get_near_compute_scratchpad_usage()
            )

        debug_assert(
            shared_mem_bytes >= 0,
            "shared_mem_bytes must be non-negative",
        )

        comptime num_passed_args = Ts.size
        var validated_args = Self._validate_arguments[
            *Ts, num_args=num_passed_args
        ]()
        var num_translated_args = validated_args[0]
        var translated_arg_offsets = validated_args[1].copy()

        ref func_info = self._func_info
        var num_captures = max(0, func_info.num_captures)
        comptime populate = type_of(func_info).populate
        comptime num_captures_static = 16

        @parameter
        def calculate_args_size() -> Int:
            var tmp_args_size = 8  # reserve 8 extra bytes for alignment
            comptime for i in range(num_passed_args):
                comptime actual_arg_type = Ts[i]
                tmp_args_size += align_up(
                    size_of[actual_arg_type.device_type](), 8
                )
            return tmp_args_size

        comptime args_size = calculate_args_size()

        var translated_args = InlineArray[Byte, args_size](uninitialized=True)
        var start_addr = Int(translated_args.unsafe_ptr())
        var extra_align = align_up(start_addr, 8) - start_addr

        var dense_args_addrs: UnsafePointer[
            OpaquePointer[MutAnyOrigin], MutUntrackedOrigin
        ]
        var dense_args_sizes: UnsafePointer[UInt64, MutUntrackedOrigin]
        if num_captures > num_captures_static:
            dense_args_addrs = alloc(
                Layout[OpaquePointer[MutAnyOrigin]](
                    count=num_captures + num_passed_args
                )
            ).unsafe_leak()
            dense_args_sizes = alloc(
                Layout[UInt64](count=num_captures + num_passed_args)
            ).unsafe_leak()
            for i in range(num_captures + num_passed_args):
                dense_args_sizes[i] = 0
        else:
            dense_args_addrs = stack_allocation[
                num_captures_static + num_passed_args,
                OpaquePointer[MutAnyOrigin],
            ]()
            dense_args_sizes = stack_allocation[
                num_captures_static + num_passed_args, UInt64
            ]()
            for i in range(num_captures_static + num_passed_args):
                dense_args_sizes[i] = 0

        # Unlike the legacy path — whose `ctx.enqueue` is `@always_inline`, so
        # the launch reads that storage in the same frame — HAL dispatches the
        # launch through a chain of non-inlined calls (`Stream.execute` ->
        # `Queue.execute` -> plugin -> C ABI). The compiler treats `populate`'s
        # alloca as dead once `populate` returns and reuses the storage for
        # those call frames, corrupting the kernel-param values. Copy the
        # capture values into stable heap storage right after `populate` and
        # repoint `dense_args_addrs` at the copies so they survive dispatch.
        # Capture byte sizes come from the `_DeviceFunctionInner` snapshot taken
        # in `DeviceFunction.__init__`.
        var capture_blob = Optional[UnsafePointer[Byte, MutUntrackedOrigin]]()
        var capture_blob_size = 0
        if num_captures > 0:
            for i in range(num_captures):
                dense_args_sizes[
                    num_passed_args + i
                ] = self._inner[]._capture_sizes[i]
                capture_blob_size = align_up(capture_blob_size, 16) + Int(
                    self._inner[]._capture_sizes[i]
                )
            var capture_args_start = dense_args_addrs + num_translated_args
            populate(
                capture_args_start.bitcast[NoneType]().as_unsafe_any_origin()
            )

            var blob = alloc(
                Layout[Byte](count=max(capture_blob_size, 1))
            ).unsafe_leak()
            var blob_off = 0
            for i in range(num_captures):
                blob_off = align_up(blob_off, 16)
                var sz = Int(self._inner[]._capture_sizes[i])
                memcpy(
                    dest=blob + blob_off,
                    src=dense_args_addrs[num_translated_args + i].bitcast[
                        Byte
                    ](),
                    count=sz,
                )
                dense_args_addrs[num_translated_args + i] = (
                    (blob + blob_off).bitcast[NoneType]().as_unsafe_any_origin()
                )
                blob_off += sz
            capture_blob = Optional(blob)

        var translated_arg_idx = 0

        var device_type_encoder = DefaultDeviceTypeEncoder()

        comptime for i in range(num_passed_args):
            var translated_arg_offset = translated_arg_offsets[i]
            if translated_arg_offset >= 0:
                comptime actual_arg_type = Ts[i]
                var first_word_addr = UnsafePointer(
                    to=translated_args.unsafe_ptr()[
                        translated_arg_offset + extra_align
                    ]
                ).bitcast[NoneType]()
                args[i]._to_device_type(device_type_encoder, first_word_addr)
                dense_args_addrs[
                    translated_arg_idx
                ] = first_word_addr.as_unsafe_any_origin()
                dense_args_sizes[translated_arg_idx] = UInt64(
                    size_of[
                        actual_arg_type.device_type,
                        target=device_type_encoder.target(),
                    ]()
                )
                translated_arg_idx += 1

        # Kernels that use `with PDL()` emit `griddepcontrol` instructions and
        # require the launch to be configured with the matching attribute;
        # dropping it faults the launch.
        var attr_ptr = OptionalReg[OpaquePointer[MutUntrackedOrigin]](None)
        if len(attributes) > 0:
            attr_ptr = OptionalReg(
                attributes.unsafe_ptr()
                .bitcast[NoneType]()
                .unsafe_origin_cast[MutUntrackedOrigin]()
            )

        ctx._hal_stream()[].execute(
            self._inner[]._func_handle,
            execution_config,
            args=rebind[
                UnsafePointer[OpaquePointer[MutUntrackedOrigin], MutAnyOrigin]
            ](dense_args_addrs),
            arg_sizes=rebind[UnsafePointer[UInt64, MutAnyOrigin]](
                dense_args_sizes
            ),
            num_args=UInt32(num_translated_args + num_captures),
            attributes=attr_ptr,
            num_attributes=UInt32(len(attributes)),
        )

        # Keep `attributes` and the marshaled arg bytes alive past the launch:
        # `attr_ptr` and `dense_args_addrs` point into them, and ASAP-drop
        # would otherwise release the storage before the plugin reads it.
        _ = attributes^
        _ = translated_args^

        if capture_blob:
            dealloc(
                ThinAllocation(
                    unsafe_assume_ownership=capture_blob.value()
                ).unsafe_with_layout({count = max(capture_blob_size, 1)})
            )
        if num_captures > num_captures_static:
            dealloc(
                ThinAllocation(
                    unsafe_assume_ownership=dense_args_addrs
                ).unsafe_with_layout({count = num_captures + num_passed_args})
            )
            dealloc(
                ThinAllocation(
                    unsafe_assume_ownership=dense_args_sizes
                ).unsafe_with_layout({count = num_captures + num_passed_args})
            )

    @always_inline
    @parameter
    def _call_with_pack[
        *Ts: AnyType,
        ContextT: _HALFunctionEnqueuer,
    ](
        imm self,
        ctx: ContextT,
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        location: OptionalReg[SourceLocation] = None,
    ) raises:
        # Launch `attributes` (e.g. PDL / programmatic stream serialization,
        # cluster dimensions) are forwarded to the plugin via the attribute
        # array assembled below. `constant_memory` mappings DO change
        # semantics and are not yet plumbed through, so refuse them rather
        # than silently dropping them.
        if cluster_dim:
            attributes.append(
                LaunchAttribute.from_cluster_dim(cluster_dim.value())
            )
        if constant_memory:
            raise Error(
                "HAL DeviceContext.enqueue_function does not support"
                " `constant_memory` mappings."
            )

        comptime num_args = Ts.size
        ref func_info = self._func_info
        var num_captures = max(0, func_info.num_captures)
        comptime populate = type_of(func_info).populate
        comptime num_captures_static = 16

        var dense_args_addrs: UnsafePointer[
            OpaquePointer[MutAnyOrigin], MutUntrackedOrigin
        ]
        var dense_args_sizes: UnsafePointer[UInt64, MutUntrackedOrigin]
        if num_captures > num_captures_static:
            dense_args_addrs = alloc(
                Layout[OpaquePointer[MutAnyOrigin]](
                    count=num_captures + num_args
                )
            ).unsafe_leak()
            dense_args_sizes = alloc(
                Layout[UInt64](count=num_captures + num_args)
            ).unsafe_leak()
            for i in range(num_captures + num_args):
                dense_args_sizes[i] = 0
        else:
            dense_args_addrs = stack_allocation[
                num_captures_static + num_args, OpaquePointer[MutAnyOrigin]
            ]()
            dense_args_sizes = stack_allocation[
                num_captures_static + num_args, UInt64
            ]()
            for i in range(num_captures_static + num_args):
                dense_args_sizes[i] = 0

        comptime for i in range(num_args):
            dense_args_addrs[i] = (
                UnsafePointer(to=args[i])
                .bitcast[NoneType]()
                .unsafe_mut_cast[True]()
                .as_unsafe_any_origin()
            )

        @parameter
        def _populate_arg_sizes[i: Int]():
            dense_args_sizes[i] = UInt64(size_of[Ts[i]]())

        comptime for i in range(num_args):
            _populate_arg_sizes[i]()

        # See `_call_with_pack_checked`: `populate`'s stack capture storage
        # does not survive HAL's non-inlined launch chain, so copy the capture
        # values into a stable heap blob and repoint `dense_args_addrs` at the
        # copies.
        #
        # Unchecked path: this is reached via `enqueue_function(func_value)`,
        # so the `DeviceFunction` outlives the frame that filled
        # `func_info.capture_sizes` (elaborator stack storage) — by here it
        # has been reused/clobbered. Use the `_DeviceFunctionInner` snapshot
        # taken at `__init__` instead, which captured the sizes while they
        # were valid.
        var capture_blob = Optional[UnsafePointer[Byte, MutUntrackedOrigin]]()
        var capture_blob_size = 0
        if num_captures > 0:
            for i in range(num_captures):
                dense_args_sizes[num_args + i] = self._inner[]._capture_sizes[i]
                capture_blob_size = align_up(capture_blob_size, 16) + Int(
                    self._inner[]._capture_sizes[i]
                )
            var capture_args_start = dense_args_addrs + num_args
            populate(
                capture_args_start.bitcast[NoneType]().as_unsafe_any_origin()
            )

            var blob = alloc(
                Layout[Byte](count=max(capture_blob_size, 1))
            ).unsafe_leak()
            var blob_off = 0
            for i in range(num_captures):
                blob_off = align_up(blob_off, 16)
                var sz = Int(self._inner[]._capture_sizes[i])
                memcpy(
                    dest=blob + blob_off,
                    src=dense_args_addrs[num_args + i].bitcast[Byte](),
                    count=sz,
                )
                dense_args_addrs[num_args + i] = (
                    (blob + blob_off).bitcast[NoneType]().as_unsafe_any_origin()
                )
                blob_off += sz
            capture_blob = Optional(blob)

        # Kernels that use `with PDL()` emit `griddepcontrol` instructions and
        # require the launch to be configured with the matching attribute;
        # dropping it faults the launch.
        var attr_ptr = OptionalReg[OpaquePointer[MutUntrackedOrigin]](None)
        if len(attributes) > 0:
            attr_ptr = OptionalReg(
                attributes.unsafe_ptr()
                .bitcast[NoneType]()
                .unsafe_origin_cast[MutUntrackedOrigin]()
            )

        ctx._hal_stream()[].execute(
            self._inner[]._func_handle,
            grid=(
                UInt32(grid_dim.x()),
                UInt32(grid_dim.y()),
                UInt32(grid_dim.z()),
            ),
            block=(
                UInt32(block_dim.x()),
                UInt32(block_dim.y()),
                UInt32(block_dim.z()),
            ),
            args=rebind[
                UnsafePointer[OpaquePointer[MutUntrackedOrigin], MutAnyOrigin]
            ](dense_args_addrs),
            arg_sizes=rebind[UnsafePointer[UInt64, MutAnyOrigin]](
                dense_args_sizes
            ),
            num_args=UInt32(num_args + num_captures),
            shared_mem_bytes=UInt32(shared_mem_bytes.or_else(0)),
            attributes=attr_ptr,
            num_attributes=UInt32(len(attributes)),
        )

        # Keep `attributes` alive past the launch
        _ = attributes^

        if capture_blob:
            dealloc(
                ThinAllocation(
                    unsafe_assume_ownership=capture_blob.value()
                ).unsafe_with_layout({count = max(capture_blob_size, 1)})
            )
        if num_captures > num_captures_static:
            dealloc(
                ThinAllocation(
                    unsafe_assume_ownership=dense_args_addrs
                ).unsafe_with_layout({count = num_captures + num_args})
            )
            dealloc(
                ThinAllocation(
                    unsafe_assume_ownership=dense_args_sizes
                ).unsafe_with_layout({count = num_captures + num_args})
            )

    @always_inline
    @parameter
    def _call_with_pack[
        ExecutionConfigType: GridBlockExecutionConfig,
        //,
        *Ts: AnyType,
        ContextT: _HALFunctionEnqueuer,
    ](
        imm self,
        ctx: ContextT,
        mut execution_config: ExecutionConfigType,
        *args: *Ts,
        location: OptionalReg[SourceLocation] = None,
    ) raises:
        _check_device_context_hal_only_supported_exec_config(execution_config)

        comptime num_args = Ts.size
        ref func_info = self._func_info
        var num_captures = max(0, func_info.num_captures)
        comptime populate = type_of(func_info).populate
        comptime num_captures_static = 16

        var dense_args_addrs: UnsafePointer[
            OpaquePointer[MutAnyOrigin], MutUntrackedOrigin
        ]
        var dense_args_sizes: UnsafePointer[UInt64, MutUntrackedOrigin]
        if num_captures > num_captures_static:
            dense_args_addrs = alloc(
                Layout[OpaquePointer[MutAnyOrigin]](
                    count=num_captures + num_args
                )
            ).unsafe_leak()
            dense_args_sizes = alloc(
                Layout[UInt64](count=num_captures + num_args)
            ).unsafe_leak()
            for i in range(num_captures + num_args):
                dense_args_sizes[i] = 0
        else:
            dense_args_addrs = stack_allocation[
                num_captures_static + num_args, OpaquePointer[MutAnyOrigin]
            ]()
            dense_args_sizes = stack_allocation[
                num_captures_static + num_args, UInt64
            ]()
            for i in range(num_captures_static + num_args):
                dense_args_sizes[i] = 0

        comptime for i in range(num_args):
            dense_args_addrs[i] = (
                UnsafePointer(to=args[i])
                .bitcast[NoneType]()
                .unsafe_mut_cast[True]()
                .as_unsafe_any_origin()
            )

        @parameter
        def _populate_arg_sizes[i: Int]():
            dense_args_sizes[i] = UInt64(size_of[Ts[i]]())

        comptime for i in range(num_args):
            _populate_arg_sizes[i]()

        # See `_call_with_pack_checked`: keep capture data alive once
        # `populate`'s stack storage is dead.
        var capture_blob = Optional[UnsafePointer[Byte, MutUntrackedOrigin]]()
        var capture_blob_size = 0
        if num_captures > 0:
            for i in range(num_captures):
                dense_args_sizes[num_args + i] = self._inner[]._capture_sizes[i]
                capture_blob_size = align_up(capture_blob_size, 16) + Int(
                    self._inner[]._capture_sizes[i]
                )
            var capture_args_start = dense_args_addrs + num_args
            populate(
                capture_args_start.bitcast[NoneType]().as_unsafe_any_origin()
            )

            var blob = alloc(
                Layout[Byte](count=max(capture_blob_size, 1))
            ).unsafe_leak()
            var blob_off = 0
            for i in range(num_captures):
                blob_off = align_up(blob_off, 16)
                var sz = Int(self._inner[]._capture_sizes[i])
                memcpy(
                    dest=blob + blob_off,
                    src=dense_args_addrs[num_args + i].bitcast[Byte](),
                    count=sz,
                )
                dense_args_addrs[num_args + i] = (
                    (blob + blob_off).bitcast[NoneType]().as_unsafe_any_origin()
                )
                blob_off += sz
            capture_blob = Optional(blob)

        ctx._hal_stream()[].execute(
            self._inner[]._func_handle,
            execution_config,
            args=rebind[
                UnsafePointer[OpaquePointer[MutUntrackedOrigin], MutAnyOrigin]
            ](dense_args_addrs),
            arg_sizes=rebind[UnsafePointer[UInt64, MutAnyOrigin]](
                dense_args_sizes
            ),
            num_args=UInt32(num_args + num_captures),
        )

        if capture_blob:
            dealloc(
                ThinAllocation(
                    unsafe_assume_ownership=capture_blob.value()
                ).unsafe_with_layout({count = max(capture_blob_size, 1)})
            )
        if num_captures > num_captures_static:
            dealloc(
                ThinAllocation(
                    unsafe_assume_ownership=dense_args_addrs
                ).unsafe_with_layout({count = num_captures + num_args})
            )
            dealloc(
                ThinAllocation(
                    unsafe_assume_ownership=dense_args_sizes
                ).unsafe_with_layout({count = num_captures + num_args})
            )

    @always_inline
    def occupancy_max_active_blocks_per_multiprocessor(
        self, block_size: Int, dynamic_shared_mem_size: Int
    ) raises -> Int:
        """Returns the maximum number of active blocks per multiprocessor for the given function.

        Args:
            block_size: The number of threads per block.
            dynamic_shared_mem_size: The size of dynamically allocated shared memory in bytes.

        Returns:
            The maximum number of active blocks that can run concurrently per multiprocessor.

        Raises:
            If the occupancy calculation fails.
        """
        return Int(
            self._inner[]
            ._context[]
            .function_occupancy_max_active_blocks(
                self._inner[]._func_handle,
                Int32(block_size),
                UInt64(dynamic_shared_mem_size),
            )
        )


# ===-----------------------------------------------------------------------===#
# DeviceExternalFunction
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _ExternalFunctionInner(Movable):
    """Holder for an externally-loaded HAL `FunctionHandle`.

    Owns the function handle, the `RuntimeBundle` it was loaded from, and
    the `Context` needed to unload them. Distinct from `_DeviceFunctionInner`
    because external functions don't carry `CompiledFunctionInfo`.
    """

    var _func_handle: FunctionHandle
    var _bundle: RuntimeBundle
    var _context: ArcPointer[Context[get_device_spec[0]()]]

    def __del__(deinit self):
        try:
            self._context[].unload_function(self._func_handle)
        except e:
            print("warning: unload_function failed:", e)


struct DeviceExternalFunction(ImplicitlyCopyable, Movable):
    """Represents an external device function loaded from PTX/SASS assembly.

    This class provides functionality to load and execute pre-compiled GPU functions
    from assembly code rather than compiling them from Mojo source. This is useful
    for integrating with existing CUDA/HIP code or for using specialized assembly
    optimizations.
    """

    var _ctx: DeviceContext
    var _inner: ArcPointer[_ExternalFunctionInner]

    @doc_hidden
    def __init__(
        out self,
        ctx: DeviceContext,
        var function_name: String,
        asm: StringSlice,
    ) raises:
        """Loads a function from raw PTX/SASS bytes.

        Args:
            ctx: The device context to associate this function with.
            function_name: Mangled symbol name to look up inside the loaded
                bundle.
            asm: Pre-compiled assembly/object bytes.
        """
        var bundle = ctx._context[].load_bundle(asm)
        var func_handle = ctx._context[].load_function(bundle, function_name^)
        self._ctx = ctx
        self._inner = ArcPointer(
            _ExternalFunctionInner(func_handle, bundle^, ctx._context)
        )

    @always_inline
    @parameter
    def _call_with_pack[
        *Ts: AnyType,
        ContextT: _HALFunctionEnqueuer,
    ](
        imm self,
        ctx: ContextT,
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        location: OptionalReg[SourceLocation] = None,
    ) raises:
        # Launch `attributes` (e.g. PDL / programmatic stream serialization,
        # cluster dimensions) are forwarded to the plugin via the attribute
        # array assembled below. `constant_memory` mappings DO change
        # semantics and are not yet plumbed through, so refuse them rather
        # than silently dropping them.
        if cluster_dim:
            attributes.append(
                LaunchAttribute.from_cluster_dim(cluster_dim.value())
            )
        if constant_memory:
            raise Error(
                "HAL DeviceContext.enqueue_function does not support"
                " `constant_memory` mappings."
            )

        comptime num_args = Ts.size
        var dense_args_addrs = stack_allocation[
            num_args + 1, OpaquePointer[MutAnyOrigin]
        ]()
        var dense_args_sizes = stack_allocation[num_args + 1, UInt64]()

        comptime for i in range(num_args):
            dense_args_addrs[i] = (
                UnsafePointer(to=args[i])
                .bitcast[NoneType]()
                .unsafe_mut_cast[True]()
                .as_unsafe_any_origin()
            )

        @parameter
        def _populate_arg_sizes[i: Int]():
            dense_args_sizes[i] = UInt64(size_of[Ts[i]]())

        comptime for i in range(num_args):
            _populate_arg_sizes[i]()

        # Kernels that use `with PDL()` emit `griddepcontrol` instructions
        # and require the launch to be configured with the matching attribute;
        # dropping it faults the launch.
        var attr_ptr = OptionalReg[OpaquePointer[MutUntrackedOrigin]](None)
        if len(attributes) > 0:
            attr_ptr = OptionalReg(
                attributes.unsafe_ptr()
                .bitcast[NoneType]()
                .unsafe_origin_cast[MutUntrackedOrigin]()
            )

        ctx._hal_stream()[].execute(
            self._inner[]._func_handle,
            grid=(
                UInt32(grid_dim.x()),
                UInt32(grid_dim.y()),
                UInt32(grid_dim.z()),
            ),
            block=(
                UInt32(block_dim.x()),
                UInt32(block_dim.y()),
                UInt32(block_dim.z()),
            ),
            args=rebind[
                UnsafePointer[OpaquePointer[MutUntrackedOrigin], MutAnyOrigin]
            ](dense_args_addrs),
            arg_sizes=rebind[UnsafePointer[UInt64, MutAnyOrigin]](
                dense_args_sizes
            ),
            num_args=UInt32(num_args),
            shared_mem_bytes=UInt32(shared_mem_bytes.or_else(0)),
            attributes=attr_ptr,
            num_attributes=UInt32(len(attributes)),
        )

    @always_inline
    @parameter
    def _call_with_pack[
        ExecutionConfigType: GridBlockExecutionConfig,
        //,
        *Ts: AnyType,
        ContextT: _HALFunctionEnqueuer,
    ](
        imm self,
        ctx: ContextT,
        mut execution_config: ExecutionConfigType,
        *args: *Ts,
        location: OptionalReg[SourceLocation] = None,
    ) raises:
        _check_device_context_hal_only_supported_exec_config(execution_config)

        comptime num_args = Ts.size
        var dense_args_addrs = stack_allocation[
            num_args + 1, OpaquePointer[MutAnyOrigin]
        ]()
        var dense_args_sizes = stack_allocation[num_args + 1, UInt64]()

        comptime for i in range(num_args):
            dense_args_addrs[i] = (
                UnsafePointer(to=args[i])
                .bitcast[NoneType]()
                .unsafe_mut_cast[True]()
                .as_unsafe_any_origin()
            )

        @parameter
        def _populate_arg_sizes[i: Int]():
            dense_args_sizes[i] = UInt64(size_of[Ts[i]]())

        comptime for i in range(num_args):
            _populate_arg_sizes[i]()

        ctx._hal_stream()[].execute(
            self._inner[]._func_handle,
            execution_config,
            args=rebind[
                UnsafePointer[OpaquePointer[MutUntrackedOrigin], MutAnyOrigin]
            ](dense_args_addrs),
            arg_sizes=rebind[UnsafePointer[UInt64, MutAnyOrigin]](
                dense_args_sizes
            ),
            num_args=UInt32(num_args),
        )


# ===-----------------------------------------------------------------------===#
# DeviceStream
# ===-----------------------------------------------------------------------===#


struct DeviceStream(ImplicitlyCopyable, Movable, _HALFunctionEnqueuer):
    """Represents a CUDA/HIP stream for asynchronous GPU operations.
    A DeviceStream provides a queue for GPU operations that can execute concurrently
    with operations in other streams. Operations within a single stream execute in
    the order they are issued, but operations in different streams may execute in
    any relative order or concurrently.

    This abstraction allows for better utilization of GPU resources by enabling
    overlapping of computation and data transfers.

    Example:

    ```mojo
    from std.gpu.host import DeviceContext, DeviceStream
    var ctx = DeviceContext(0)  # Select first GPU
    var stream = DeviceStream(ctx)

    # Launch operations on the stream
    # ...

    # Wait for all operations in the stream to complete
    stream.synchronize()
    ```
    """

    var _ctx: DeviceContext
    var _stream: ArcPointer[Stream[get_device_spec[0]()]]
    # Null until the C++ runtime shim provides it; read by the CUDA/HIP
    # interop accessors (`_nvidia_cuda` / `_amdgpu_hip`).
    var _handle: _DeviceStreamPtr[mut=True]

    @doc_hidden
    def __init__(out self, ctx: DeviceContext):
        """Retrieves the stream associated with the given device context.

        Args:
            ctx: The device context to retrieve the stream from.
        """
        self._ctx = ctx
        self._stream = ctx._stream
        self._handle = None

    @doc_hidden
    def __init__(
        out self,
        ctx: DeviceContext,
        var hal_stream: ArcPointer[Stream[get_device_spec[0]()]],
    ):
        """Initializes a new DeviceStream with the given stream handle.

        Args:
            ctx: The device context that owns the stream.
            hal_stream: The stream handle to initialize the DeviceStream with.
        """
        self._ctx = ctx
        self._stream = hal_stream^
        self._handle = None

    def synchronize(self) raises:
        """Blocks the calling CPU thread until all operations in this stream complete.

        This function waits until all previously issued commands in this stream
        have completed execution. It provides a synchronization point between
        host and device code.

        Raises:
            If synchronization fails.

        Example:

        ```mojo
        from std.gpu.host import DeviceContext

        var ctx = DeviceContext()
        var stream = ctx.create_stream()

        # Launch kernel or memory operations on the stream
        # ...

        # Wait for completion
        stream.synchronize()

        # Now it's safe to use results on the host
        ```
        """
        self._stream[].synchronize()

    def enqueue_host_func[
        origin: MutOrigin
    ](
        self,
        func: def(OpaquePointer[origin]) thin -> None,
        user_data: OpaquePointer[origin],
    ) raises:
        """Enqueues a host callback to run on this stream.

        This corresponds to CUDA's `cuLaunchHostFunc`. The callback `func`
        runs on a driver thread once all preceding work on this stream has
        completed, and receives `user_data` as its only argument. Per the
        CUDA contract, the callback must not call any device APIs.

        Currently only implemented for CUDA streams; other backends raise.

        Parameters:
            origin: The origin of `user_data`, shared with the callback's
                argument so the two are coupled at the type level.

        Args:
            func: A `thin` C-compatible function pointer that accepts a
                single `void*` argument.
            user_data: An opaque pointer passed through to `func` when it
                runs.

        Raises:
            If the underlying device does not support host callbacks, or if
            the driver rejects the enqueue.
        """
        self._stream[].enqueue_host_func(func, user_data)

    def enqueue_wait_for(self, event: DeviceEvent) raises:
        """Makes this stream wait for the specified event.

        This function inserts a wait operation into this stream that will
        block all subsequent operations in the stream until the specified
        event has been recorded and completed.

        Args:
            event: The event to wait for.

        Raises:
            If the wait operation fails.
        """
        # If an event hasn't been recorded yet, there's nothing to wait on
        if not event._event[]:
            return
        self._stream[].wait_for_events(event._event[].value())

    def record_event(self, event: DeviceEvent) raises:
        """Records an event in this stream.

        This function records the given event at the current point in this stream.
        All operations in the stream that were enqueued before this call will
        complete before the event is triggered.

        Args:
            event: The event to record.

        Raises:
            If event recording fails.

        Example:

        ```mojo
        from std.gpu.host import DeviceContext

        var ctx = DeviceContext()

        var default_stream = ctx.stream()
        var new_stream = ctx.create_stream()

        # Create event on the context
        var event = ctx.create_event()

        # Wait for the event on the new stream
        new_stream.enqueue_wait_for(event)

        # Stream 2 can continue
        default_stream.record_event(event)
        ```
        """
        # Create and record the backing event for this DeviceEvent. If it was
        # previously recorded and no other DeviceEvents have been constructed
        # by copy from this event, the existing backing event will be released.
        var hal_event = self._stream[].record_event[EVENT_FLAG_CPU_VISIBLE]()
        event._event[] = Optional(hal_event^)

    @doc_hidden
    def wait_for_host_value(
        self,
        flag: CompletionFlag,
        value: UInt64,
    ) raises:
        """Stalls the stream until a host-visible flag reaches a given value.

        Corresponds to CUDA's `cuStreamWaitValue64` on the slot owned by
        `flag`.

        Args:
            flag: A non-owning handle to a ``M::Driver::CompletionFlag``.
            value: The 64-bit value to wait for.

        Raises:
            If the underlying device does not support stream memory ops,
            or if enqueueing the wait fails.
        """
        # The flag's value slot lives inside a C++-driver-owned object; the
        # exchange becomes functional once the HAL C++ runtime shim lands.
        raise Error("wait_for_host_value requires the HAL C++ runtime shim")

    def _native_stream[
        origin: Origin, //
    ](ref[origin] self) raises -> OpaquePointer[origin]:
        """Returns this stream's native driver-level handle (for example a
        `CUstream`), tied to this borrow of `self` so the owning queue stays
        alive while the handle is in use."""
        var handle = self._stream[].native_handle()
        if not handle:
            raise Error("this device's stream has no native driver handle")
        return OpaquePointer[origin](unsafe_from_address=Int(handle.value()))

    def _hal_stream(
        self,
    ) -> ArcPointer[Stream[get_device_spec[0]()]]:
        return self._stream

    @parameter
    @always_inline
    def enqueue_function[
        declared_arg_types: TypeList[Trait=AnyType, ...],
        //,
        func: def(* args: * declared_arg_types) thin -> None,
        *actual_arg_types: DevicePassable,
    ](
        self,
        *args: *actual_arg_types,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        location: OptionalReg[SourceLocation] = None,
    ) raises:
        """Compiles and enqueues a kernel for execution on this stream.

        Parameters:
            declared_arg_types: Types of the arguments to pass to the device function.
            func: The function to compile and launch.
            actual_arg_types: The dtypes of the arguments being passed to the function.

        Args:
            args: Variadic arguments which are passed to the `func`.
            grid_dim: The grid dimensions.
            block_dim: The block dimensions.
            cluster_dim: The cluster dimensions.
            shared_mem_bytes: Per-block memory shared between blocks.
            attributes: A `List` of launch attributes.
            constant_memory: A `List` of constant memory mappings.
            location: Source location for the function call.

        Raises:
            If the operation fails.
        """
        _check_dim["DeviceStream.enqueue_function", "grid_dim"](
            grid_dim, location=call_location()
        )
        _check_dim["DeviceStream.enqueue_function", "block_dim"](
            block_dim, location=call_location()
        )
        var gpu_kernel = self._ctx.compile_function[func]()
        gpu_kernel._call_with_pack_checked(
            self,
            *args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
            location=location.or_else(call_location()),
        )

    @parameter
    @always_inline
    def enqueue_function[
        declared_arg_types: TypeList[Trait=AnyType, ...],
        ExecutionConfigType: GridBlockExecutionConfig,
        //,
        func: def(* args: * declared_arg_types) thin -> None,
        *actual_arg_types: DevicePassable,
    ](
        self,
        mut execution_config: ExecutionConfigType,
        *args: *actual_arg_types,
        location: OptionalReg[SourceLocation] = None,
    ) raises:
        """Compiles and enqueues a kernel for execution on this stream.

        Parameters:
            declared_arg_types: Types of the arguments to pass to the device function.
            func: The function to compile and launch.
            actual_arg_types: The dtypes of the arguments being passed to the function.

        Args:
            execution_config: The execution configuration for the kernel launch.
            args: Variadic arguments which are passed to the `func`.
            location: Source location for the function call.

        Raises:
            If the operation fails.
        """
        var gpu_kernel = self._ctx.compile_function[func]()
        self.enqueue_function(
            execution_config,
            gpu_kernel,
            *args,
            location=location.or_else(call_location()),
        )

    @parameter
    @always_inline
    def enqueue_function[
        FuncType: def() -> None,
        //,
    ](
        self,
        func: FuncType,
        *,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        location: OptionalReg[SourceLocation] = None,
    ) raises:
        """Compiles and enqueues a capturing kernel for execution on this stream.

        Parameters:
            FuncType: The type of the function to launch (usually inferred).

        Args:
            func: The capturing kernel function to compile and launch.
            grid_dim: The grid dimensions.
            block_dim: The block dimensions.
            cluster_dim: The cluster dimensions.
            shared_mem_bytes: Per-block memory shared between blocks.
            attributes: A `List` of launch attributes.
            constant_memory: A `List` of constant memory mappings.
            location: Source location for the function call.

        Raises:
            If the operation fails.
        """
        _check_dim["DeviceStream.enqueue_function", "grid_dim"](
            grid_dim, location=call_location()
        )
        _check_dim["DeviceStream.enqueue_function", "block_dim"](
            block_dim, location=call_location()
        )
        var gpu_kernel = DeviceFunction[
            FuncType.__call__, TypeList.of[Trait=AnyType]()
        ](self._ctx)
        gpu_kernel._call_with_pack(
            self,
            func,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
            location=location.or_else(call_location()),
        )

    @parameter
    @always_inline
    def enqueue_function[
        ExecutionConfigType: GridBlockExecutionConfig,
        FuncType: def() -> None,
        //,
    ](
        self,
        mut execution_config: ExecutionConfigType,
        func: FuncType,
        *,
        location: OptionalReg[SourceLocation] = None,
    ) raises:
        """Compiles and enqueues a capturing kernel for execution on this stream.

        Parameters:
            ExecutionConfigType: The type of the execution configuration (usually inferred).
            FuncType: The type of the function to launch (usually inferred).

        Args:
            execution_config: The execution configuration for the kernel launch.
            func: The capturing kernel function to compile and launch.
            location: Source location for the function call.

        Raises:
            If the operation fails.
        """
        var gpu_kernel = DeviceFunction[
            FuncType.__call__, TypeList.of[Trait=AnyType]()
        ](self._ctx)
        self.enqueue_function(
            execution_config,
            gpu_kernel,
            location=location.or_else(call_location()),
        )

    @parameter
    @always_inline
    def enqueue_function[
        *Ts: DevicePassable,
    ](
        self,
        f: DeviceFunction,
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        location: OptionalReg[SourceLocation] = None,
    ) raises:
        """Enqueues a pre-compiled checked function for execution on this stream.

        Parameters:
            Ts: Argument dtypes.

        Args:
            f: The compiled function to execute.
            args: Arguments to pass to the function.
            grid_dim: Dimensions of the compute grid.
            block_dim: Dimensions of each thread block in the grid.
            cluster_dim: Dimensions of clusters.
            shared_mem_bytes: Amount of shared memory per thread block.
            attributes: Launch attributes.
            constant_memory: Constant memory mapping.
            location: Source location for the function call.

        Raises:
            If the operation fails.
        """
        _check_dim["DeviceStream.enqueue_function", "grid_dim"](
            grid_dim, location=call_location()
        )
        _check_dim["DeviceStream.enqueue_function", "block_dim"](
            block_dim, location=call_location()
        )
        f._call_with_pack_checked(
            self,
            *args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
            location=location.or_else(call_location()),
        )

    @parameter
    @always_inline
    def enqueue_function[
        ExecutionConfigType: GridBlockExecutionConfig,
        //,
        *Ts: DevicePassable,
    ](
        self,
        mut execution_config: ExecutionConfigType,
        f: DeviceFunction,
        *args: *Ts,
        location: OptionalReg[SourceLocation] = None,
    ) raises:
        """Enqueues a pre-compiled checked function for execution on this stream.

        Parameters:
            ExecutionConfigType: The type of the execution configuration (usually inferred).
            Ts: Argument dtypes.

        Args:
            execution_config: The execution configuration for the kernel launch.
            f: The compiled function to execute.
            args: Arguments to pass to the function.
            location: Source location for the function call.

        Raises:
            If the operation fails.
        """
        var grid_dim = execution_config.get_grid_dim()
        var block_dim = execution_config.get_block_dim()

        _check_dim["DeviceStream.enqueue_function", "grid_dim"](
            grid_dim, location=call_location()
        )
        _check_dim["DeviceStream.enqueue_function", "block_dim"](
            block_dim, location=call_location()
        )

        f._call_with_pack_checked(
            self,
            execution_config,
            *args,
            location=location.or_else(call_location()),
        )

    @parameter
    @always_inline
    def enqueue_function[
        *Ts: AnyType,
    ](
        self,
        f: DeviceExternalFunction,
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        location: OptionalReg[SourceLocation] = None,
    ) raises:
        """Enqueues an external device function for execution on this stream.

        Parameters:
            Ts: Argument types to pass to the external function.

        Args:
            f: The external device function to execute.
            args: Arguments to pass to the function.
            grid_dim: The grid dimensions.
            block_dim: The block dimensions.
            cluster_dim: Optional cluster dimensions.
            shared_mem_bytes: Optional shared memory per block.
            attributes: Launch attributes.
            constant_memory: Constant memory mapping.
            location: Source location for the function call.

        Raises:
            If the operation fails.
        """
        _check_dim["DeviceStream.enqueue_function", "grid_dim"](
            grid_dim, location=call_location()
        )
        _check_dim["DeviceStream.enqueue_function", "block_dim"](
            block_dim, location=call_location()
        )
        f._call_with_pack(
            self,
            *args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
            location=location.or_else(call_location()),
        )

    @parameter
    @always_inline
    def enqueue_function[
        ExecutionConfigType: GridBlockExecutionConfig,
        //,
        *Ts: AnyType,
    ](
        self,
        mut execution_config: ExecutionConfigType,
        f: DeviceExternalFunction,
        *args: *Ts,
        location: OptionalReg[SourceLocation] = None,
    ) raises:
        """Enqueues an external device function for execution on this stream.

        Parameters:
            ExecutionConfigType: The type of the execution configuration (usually inferred).
            Ts: Argument types to pass to the external function.

        Args:
            execution_config: The execution configuration for the kernel launch.
            f: The external device function to execute.
            args: Arguments to pass to the function.
            location: Source location for the function call.

        Raises:
            If the operation fails.
        """
        var grid_dim = execution_config.get_grid_dim()
        var block_dim = execution_config.get_block_dim()

        _check_dim["DeviceStream.enqueue_function", "grid_dim"](
            grid_dim, location=call_location()
        )
        _check_dim["DeviceStream.enqueue_function", "block_dim"](
            block_dim, location=call_location()
        )
        f._call_with_pack(
            self,
            execution_config,
            *args,
            location=location.or_else(call_location()),
        )


# ===-----------------------------------------------------------------------===#
# DeviceEvent
# ===-----------------------------------------------------------------------===#


struct DeviceEvent(ImplicitlyCopyable, Movable):
    """Represents a GPU event for synchronization between streams.

    A DeviceEvent allows for fine-grained synchronization between different
    GPU streams. Events can be recorded in one stream and waited for in another,
    enabling efficient coordination of asynchronous GPU operations.

    Example:

    ```mojo
    from std.gpu.host import DeviceContext

    var ctx = DeviceContext()

    var default_stream = ctx.stream()
    var new_stream = ctx.create_stream()

    # Create event in default_stream
    var event = ctx.create_event()

    # Wait for the event in new_stream
    new_stream.enqueue_wait_for(event)

    # Stream 2 can continue
    default_stream.record_event(event)
    ```
    """

    # `_event` is an `ArcPointer[Optional[...]]` so that all copies of a
    # DeviceEvent  share the same backing event, which matches the behavior
    # of the existing reference-counted DeviceEvent implementation.
    var _ctx: DeviceContext
    var _event: ArcPointer[Optional[Event[EVENT_FLAG_CPU_VISIBLE]]]

    @doc_hidden
    def __init__(out self, ctx: DeviceContext) raises:
        """Creates a new event recorded on the given context's default stream.

        Args:
            ctx: The device context to record the event on.

        Raises:
            If event creation or recording fails.
        """
        var hal_event = ctx._stream[].record_event[EVENT_FLAG_CPU_VISIBLE]()
        self._ctx = ctx
        self._event = ArcPointer(Optional(hal_event^))

    @doc_hidden
    def __init__(
        out self,
        ctx: DeviceContext,
        var event: Optional[Event[EVENT_FLAG_CPU_VISIBLE]],
    ):
        """Initializes a DeviceEvent wrapping a possibly-empty HAL event.

        Args:
            ctx: The device context that owns the event.
            event: The (possibly-unrecorded) HAL event to wrap.
        """
        self._ctx = ctx
        self._event = ArcPointer(event^)

    @doc_hidden
    @staticmethod
    def _create_unrecorded(ctx: DeviceContext) -> DeviceEvent:
        """Returns a `DeviceEvent` with no HAL event allocated yet."""
        return DeviceEvent(ctx, Optional[Event[EVENT_FLAG_CPU_VISIBLE]](None))

    def synchronize(self) raises:
        """Blocks the calling CPU thread until this event completes.

        This function waits until the event has been recorded and all
        operations before the event in the stream have completed.

        Raises:
            If synchronization fails.
        """
        # If an event hasn't been recorded yet, there's nothing to wait on
        if not self._event[]:
            return
        self._event[].value().synchronize()


# ===-----------------------------------------------------------------------===#
# DeviceBuffer
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _HALBufferInner(Movable):
    """Owning wrapper around a non-owning HAL Buffers.

    Owns the `Buffer` and holds a reference to the parent `Context` so
    destruction can call `Context.free_sync`.
    """

    var _buffer: Buffer[get_device_spec[0]()]
    var _context: ArcPointer[Context[get_device_spec[0]()]]
    var _device_addr: UInt64
    # For sub-buffer views: keeps the parent allocation alive for the view's
    # lifetime (a view's own `_buffer` is a non-owning wrapper). A `List` (heap
    # storage) rather than `Optional` so the struct does not embed itself.
    var _parent: List[ArcPointer[_HALBufferInner]]

    def __del__(deinit self):
        try:
            self._context[].free_sync(self._buffer^)
        except e:
            print("warning: free_sync failed:", e)
        # Release the parent only after the view's wrapper is freed.
        _ = self._parent^


struct DeviceBuffer[dtype: DType](
    DevicePassable, ImplicitlyCopyable, Movable, Sized
):
    """Represents a block of device-resident storage. For GPU devices, a device
    buffer is allocated in the device's global memory.

    To allocate a `DeviceBuffer`, use one of the methods provided by
    `DeviceContext`, such as
    [`enqueue_create_buffer()`](/docs/std/gpu/host/device_context/DeviceContext/#enqueue_create_buffer).

    Parameters:
        dtype: Data dtype to be stored in the buffer.
    """

    # Implementation of `DevicePassable`
    comptime device_type: AnyType = UnsafePointer[
        mut=True, Scalar[Self.dtype], AnyOrigin[mut=True]
    ]
    """`DeviceBuffer` dtypes are remapped to `UnsafePointer` when passed to
    accelerator devices."""

    def _to_device_type(
        self,
        mut encoder: Some[DeviceTypeEncoder],
        target: MutOpaquePointer[_],
    ):
        """Device dtype mapping from `DeviceBuffer` to the device's
        `UnsafePointer`.
        """
        try:
            encoder.encode_device_ptr(self.device_ptr(), target)
        except:
            pass

    @staticmethod
    def get_type_name() -> String:
        """Gets this dtype's name, for use in error messages when handing
        arguments to kernels.

        Returns:
            This dtype's name.
        """
        return String(t"DeviceBuffer[{Self.dtype}]")

    comptime _DevicePtr = UnsafePointer[Scalar[Self.dtype], MutUntrackedOrigin]

    # `_device_ptr` must be the first member in the struct. Kernel dispatch code
    # `rebind`s a `DeviceBuffer` directly to its `device_type` (an
    # `UnsafePointer`), which reinterprets the struct's initial data as the
    # device address.
    var _device_ptr: Self._DevicePtr
    # Wrap the inner buffer in an ArcPointer so copies of this DeviceBuffer hold
    # a reference to the same underlying buffer.
    var _ctx: DeviceContext
    var _inner: ArcPointer[_HALBufferInner]

    @doc_hidden
    def __init__(out self, ctx: DeviceContext, size: Int) raises:
        """This init takes in a constructed `DeviceContext` and schedules an
        owned buffer allocation using the stream in the device context.
        """
        var byte_size = UInt64(size * size_of[Self.dtype]())
        # Cache the GPU address up front so `unsafe_ptr` is non-raising.
        var buffer = ctx._context[].alloc_sync(byte_size)
        var addr = UInt64(0)
        if byte_size > 0:
            addr = ctx._context[].memory_get_address(buffer)
        self._device_ptr = Self._DevicePtr(unsafe_from_address=Int(addr))
        self._ctx = ctx
        self._inner = ArcPointer(
            _HALBufferInner(buffer^, ctx._context, addr, [])
        )

    @doc_hidden
    def __init__(
        out self,
        ctx: DeviceContext,
        ptr: UnsafePointer[Scalar[Self.dtype], ...],
        size: Int,
        *,
        owning: Bool,
    ):
        """Constructs a `DeviceBuffer` that wraps an externally-managed device
        pointer.

        When `owning` is True the wrapper takes ownership of the underlying
        allocation and will free it when the last reference drops. When
        `owning` is False the wrapper is non-owning; releasing it leaves the
        underlying pointer to whoever allocated it.

        This mirrors the non-raising legacy constructor; a wrap failure
        aborts rather than raising.

        Args:
            ctx: The device context the pointer belongs to.
            ptr: The device pointer to wrap.
            size: Number of elements addressable through the pointer.
            owning: Whether releasing this buffer should free the pointer.
        """
        var byte_size = UInt64(size * size_of[Self.dtype]())
        var addr = UInt64(Int(ptr))
        try:
            # The plugin rejects wrapping a null address; represent the empty
            # buffer the way a zero-byte allocation does (null memory handle).
            var buffer = ctx._context[].alloc_sync(
                0
            ) if addr == 0 else ctx._context[].wrap_memory(
                addr, byte_size, owning=owning
            )
            self._device_ptr = Self._DevicePtr(unsafe_from_address=Int(addr))
            self._ctx = ctx
            self._inner = ArcPointer(
                _HALBufferInner(buffer^, ctx._context, addr, [])
            )
        except e:
            abort("DeviceBuffer: failed to wrap external memory")

    @staticmethod
    @doc_hidden
    def empty(context: DeviceContext) -> Self:
        return Self(
            context,
            Self._DevicePtr.unsafe_dangling(),
            0,
            owning=False,
        )

    def device_ptr(
        ref self,
    ) raises -> DevicePointer[Self.dtype, origin_of(self)]:
        """Returns a `DevicePointer` referencing the start of this buffer.

        The returned `DevicePointer` is a non-owning borrow of this
        `DeviceBuffer` and must not outlive it. A function that returns a
        `DevicePointer` must also return (or otherwise keep alive) the backing
        `DeviceBuffer`; returning a pointer into a buffer created locally within
        the function is a borrow-check error.

        Returns:
            A `DevicePointer` referencing offset 0 of this buffer.

        Raises:
            If this buffer has size 0.
        """
        comptime assert not is_gpu(), "DeviceBuffer is not supported on GPUs"
        return DevicePointer[Self.dtype, origin_of(self)](self)

    def context(self) -> DeviceContext:
        """Returns the device context associated with this buffer.

        This method retrieves the device context that owns this buffer and is
        responsible for managing its lifecycle and operations.

        Returns:
            The device context associated with this buffer.
        """
        return self._ctx

    def __len__(self) -> Int:
        """Returns the number of elements in this buffer.

        This method calculates the number of elements by dividing the total byte size
        of the buffer by the size of each element.

        Returns:
            The number of elements in the buffer.
        """
        return Int(self._inner[]._buffer.byte_size) // size_of[Self.dtype]()

    @doc_hidden
    @always_inline
    def take_handle(var self) -> _DeviceBufferPtr[mut=True]:
        """Transfers this buffer to the runtime as a single owning handle.

        Under HAL the handle is this Mojo `DeviceBuffer` moved into a heap
        box; the receiving runtime entry wraps the box in the C++ shim's
        `DeviceBuffer`, which releases it when the last runtime reference
        drops. Moving into the box suppresses this value's destructor, so
        exactly one live reference transfers, net-zero.
        """
        var box = alloc[DeviceBuffer[Self.dtype]](1)
        box.init_pointee_move(self^)
        return _DeviceBufferPtr[mut=True](
            box.bitcast[_DeviceBufferCpp]().unsafe_origin_cast[
                UntrackedOrigin[mut=True]
            ]()
        )

    def take_ptr(
        var self,
    ) -> Self._DevicePtr:
        """Takes ownership of the device pointer from this buffer.

        This method releases the device pointer from the buffer's control and
        returns it to the caller. After this call, the buffer no longer owns
        the pointer, and the caller is responsible for managing its lifecycle.

        Returns:
            The raw device pointer that was owned by this buffer.
        """
        # Ownership of the underlying allocation transfers to the caller: the
        # wrapper handle (freed when the last reference drops) no longer frees
        # the memory.
        try:
            _ = self._ctx._context[].unwrap_memory(self._inner[]._buffer)
        except e:
            abort("DeviceBuffer.take_ptr: failed to release ownership")
        return Self._DevicePtr(
            unsafe_from_address=Int(self._inner[]._device_addr)
        )

    def enqueue_fill(self, val: Scalar[Self.dtype]) raises:
        """Enqueues an operation to fill this buffer with a specified value.

        This method schedules a memory set operation that fills the entire buffer
        with the specified value. The operation is asynchronous and will be executed
        in the stream associated with this buffer's context.

        Args:
            val: The value to fill the buffer with.

        Raises:
            If the operation fails.
        """
        self._ctx.enqueue_memset(self, val)

    @always_inline
    def create_sub_buffer[
        view_type: DType
    ](self, offset: Int, size: Int) raises -> DeviceBuffer[view_type]:
        """Creates a sub-buffer view of this buffer with a different element dtype.

        This method creates a new buffer that references a subset of the memory in this
        buffer, potentially with a different element dtype. The sub-buffer shares the
        underlying memory with the original buffer.

        Parameters:
            view_type: The data type for elements in the new sub-buffer.

        Args:
            offset: The starting offset, in view_type elements, from the beginning of this buffer.
            size: The number of elements in the new sub-buffer.

        Returns:
            A new DeviceBuffer referencing the specified region with the specified element dtype.

        Raises:
            If the operation fails.
        """
        comptime elem_size = size_of[view_type]()
        var byte_offset = UInt64(offset * elem_size)
        var byte_size = UInt64(size * elem_size)
        if byte_offset + byte_size > self._inner[]._buffer.byte_size:
            raise Error("create_sub_buffer: view is out of range")
        var view = DeviceBuffer[view_type](
            self._ctx,
            UnsafePointer[Scalar[view_type], MutUntrackedOrigin](
                unsafe_from_address=Int(
                    self._inner[]._device_addr + byte_offset
                )
            ),
            size,
            owning=False,
        )
        # The view's wrapper is non-owning; retaining the parent's inner keeps
        # the shared allocation alive for the view's lifetime.
        view._inner[]._parent = [self._inner]
        return view^

    def unsafe_ptr(
        self,
    ) -> UnsafePointer[Scalar[Self.dtype], MutUntrackedOrigin]:
        """Returns the raw device pointer without transferring ownership.

        This method provides direct access to the underlying device pointer
        for advanced use cases. The buffer retains ownership of the pointer.

        Returns:
            The raw device pointer owned by this buffer.
        """
        return self._device_ptr

    def _tensor_map_encode_tiled(
        self,
        tensor_map: MutOpaquePointer[_],
        data_type: Int32,
        rank: Int32,
        global_dim: UnsafePointer[mut=False, Int64, _],
        global_strides: UnsafePointer[mut=False, Int64, _],
        box_dim: UnsafePointer[mut=False, Int32, _],
        element_strides: UnsafePointer[mut=False, Int32, _],
        interleave: Int32,
        swizzle: Int32,
        l2_promotion: Int32,
        oob_fill: Int32,
    ) raises:
        """Encodes a tiled TMA descriptor for this buffer via the HAL plugin.
        Used by `std.gpu.host._tensormap.create_tensormap`."""
        # Same-width reinterpret of the callers' signed arrays to the C ABI's
        # unsigned element types.
        self._ctx._context[].tensor_map_encode_tiled(
            tensor_map,
            data_type,
            rank,
            self._inner[]._device_addr,
            global_dim.bitcast[UInt64](),
            global_strides.bitcast[UInt64](),
            box_dim.bitcast[UInt32](),
            element_strides.bitcast[UInt32](),
            interleave,
            swizzle,
            l2_promotion,
            oob_fill,
        )

    def _tensor_map_encode_im2col(
        self,
        tensor_map: MutOpaquePointer[_],
        data_type: Int32,
        rank: Int32,
        global_dim: UnsafePointer[mut=False, Int64, _],
        global_strides: UnsafePointer[mut=False, Int64, _],
        pixel_box_lower_corner: UnsafePointer[mut=False, Int32, _],
        pixel_box_upper_corner: UnsafePointer[mut=False, Int32, _],
        channels_per_pixel: Int32,
        pixels_per_column: Int32,
        element_strides: UnsafePointer[mut=False, Int32, _],
        interleave: Int32,
        swizzle: Int32,
        l2_promotion: Int32,
        oob_fill: Int32,
    ) raises:
        """Encodes an im2col TMA descriptor for this buffer via the HAL
        plugin. Used by `std.gpu.host._tensormap.create_tensormap_im2col`."""
        self._ctx._context[].tensor_map_encode_im2col(
            tensor_map,
            data_type,
            rank,
            self._inner[]._device_addr,
            global_dim.bitcast[UInt64](),
            global_strides.bitcast[UInt64](),
            pixel_box_lower_corner,
            pixel_box_upper_corner,
            channels_per_pixel,
            pixels_per_column,
            element_strides.bitcast[UInt32](),
            interleave,
            swizzle,
            l2_promotion,
            oob_fill,
        )

    def enqueue_copy_to(self, dst: DeviceBuffer[Self.dtype]) raises:
        """Enqueues an asynchronous copy from this buffer to another device buffer.

        This method schedules a memory copy operation from this buffer to the
        destination buffer. The operation is asynchronous and will be executed
        in the stream associated with this buffer's context.

        Args:
            dst: The destination device buffer to copy data to.

        Raises:
            If the operation fails.
        """
        self._ctx.enqueue_copy(dst, self)

    def enqueue_copy_from(self, src: DeviceBuffer[Self.dtype]) raises:
        """Enqueues an asynchronous copy to this buffer from another device buffer.

        This method schedules a memory copy operation to this buffer from the
        source buffer. The operation is asynchronous and will be executed in
        the stream associated with this buffer's context.

        Args:
            src: The source device buffer to copy data from.

        Raises:
            If the operation fails.
        """
        self._ctx.enqueue_copy(self, src)

    def enqueue_copy_from(
        self, src_ptr: UnsafePointer[Scalar[Self.dtype], _]
    ) raises:
        """Enqueues an asynchronous copy to this buffer from host memory.

        This method schedules a memory copy operation to this buffer from the
        given host pointer. The operation is asynchronous and will be executed
        in the stream associated with this buffer's context.

        Args:
            src_ptr: The source host pointer to copy data from.

        Raises:
            If the operation fails.
        """
        self._ctx.enqueue_copy(self, src_ptr)

    def enqueue_copy_to(self, dst: HostBuffer[Self.dtype]) raises:
        """Enqueues an asynchronous copy from this buffer to a host buffer.

        This method schedules a memory copy operation from this buffer to the destination
        host buffer. The operation is asynchronous and will be executed in the stream
        associated with this buffer's context.

        Args:
            dst: The destination host buffer to copy data to.

        Raises:
            If the operation fails.
        """
        self._ctx.enqueue_copy(dst, self)

    def enqueue_copy_from(self, src: HostBuffer[Self.dtype]) raises:
        """Enqueues an asynchronous copy from a host buffer to this buffer.

        This method schedules a memory copy operation from the source host buffer
        to this buffer. The operation is asynchronous and will be executed in the stream
        associated with this buffer's context.

        Args:
            src: The source host buffer to copy data from.

        Raises:
            If the operation fails.
        """
        self._ctx.enqueue_copy(self, src)

    def enqueue_copy_to(
        self, dst: Span[mut=True, Scalar[Self.dtype], _]
    ) raises:
        """Enqueues an asynchronous copy from this buffer to a host `Span`.

        Args:
            dst: The destination host span to copy data to.

        Raises:
            If the operation fails.
        """
        self._ctx.enqueue_copy(dst, self)

    def enqueue_copy_from(self, src: Span[Scalar[Self.dtype], _]) raises:
        """Enqueues an asynchronous copy from a host `Span` to this buffer.

        Args:
            src: The source host span to copy data from.

        Raises:
            If the operation fails.
        """
        self._ctx.enqueue_copy(self, src)

    def map_to_host(
        self,
        out mapped_buffer: _HostMappedBuffer[Self.dtype],
    ) raises:
        """Maps this device buffer to host memory for CPU access.

        This method creates a host-accessible view of the device buffer's contents.
        The mapping operation may involve copying data from device to host memory.

        Returns:
            A host-mapped buffer that provides CPU access to the device buffer's
            contents inside a with-statement.

        Raises:
            If there's an error during buffer creation or data transfer.

        Notes:

        Values modified inside the `with` statement are updated on the
        device when the `with` statement exits.

        Example:

        ```mojo
        from std.gpu.host import DeviceContext

        var ctx = DeviceContext()
        var length = 1024
        var in_dev = ctx.enqueue_create_buffer[DType.float32](length)
        var out_dev = ctx.enqueue_create_buffer[DType.float32](length)

        # Initialize the input and output with known values.
        with in_dev.map_to_host() as in_host, out_dev.map_to_host() as out_host:
            for i in range(length):
                in_host[i] = i
                out_host[i] = 255
        ```
        """
        mapped_buffer = _HostMappedBuffer[Self.dtype](self.context(), self)


# ===-----------------------------------------------------------------------===#
# HostBuffer
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _HostBufferInner(Movable):
    """Refcountable wrapper around a pinned-host HAL `Buffer`.

    Owns the pinned `Buffer` and the parent `Context` so destruction can
    call `Context.free_host_pinned`. Caches the host pointer up front so
    `unsafe_ptr` / `__getitem__` / `__setitem__` are non-raising.
    """

    var _buffer: Buffer[get_device_spec[0]()]
    var _context: ArcPointer[Context[get_device_spec[0]()]]
    var _host_ptr: UnsafePointer[UInt8, MutUntrackedOrigin]

    def __del__(deinit self):
        try:
            self._context[].free_host_pinned(self._buffer^)
        except e:
            print("warning: free_host_pinned failed:", e)


struct HostBuffer[dtype: DType](ImplicitlyCopyable, Movable, Sized):
    """Represents a block of host-resident storage. For GPU devices, a host
    buffer is allocated in the host's global memory.

    To allocate a `HostBuffer`, use one of the methods provided by
    `DeviceContext`, such as
    [`enqueue_create_host_buffer()`](/docs/std/gpu/host/device_context/DeviceContext/#enqueue_create_host_buffer).

    Parameters:
        dtype: Data type to be stored in the buffer.
    """

    var _ctx: DeviceContext
    var _inner: ArcPointer[_HostBufferInner]

    @doc_hidden
    def __init__(out self, ctx: DeviceContext, size: Int) raises:
        """This init takes in a constructed `DeviceContext` and schedules an
        owned buffer allocation using the stream in the device context.
        """
        var byte_size = UInt64(size * size_of[Self.dtype]())
        var buffer = ctx._context[].alloc_host_pinned(byte_size)
        var addr = UInt64(0)
        var host_ptr: UnsafePointer[UInt8, MutUntrackedOrigin]
        if byte_size > 0:
            host_ptr = ctx._context[].memory_get_host_address[
                MutUntrackedOrigin
            ](buffer)
        else:
            host_ptr = UnsafePointer[UInt8, MutUntrackedOrigin](
                unsafe_from_address=Int(addr)
            )
        self._ctx = ctx
        self._inner = ArcPointer(
            _HostBufferInner(buffer^, ctx._context, host_ptr)
        )

    def __len__(self) -> Int:
        """Returns the number of elements in this buffer.

        This method calculates the number of elements by dividing the total byte size
        of the buffer by the size of each element.

        Returns:
            The number of elements in the buffer.
        """
        return Int(self._inner[]._buffer.byte_size) // size_of[Self.dtype]()

    def context(self) -> DeviceContext:
        """Returns the device context associated with this buffer.

        This method retrieves the device context that owns this buffer and is
        responsible for managing its lifecycle and operations.

        Returns:
            The device context associated with this buffer.
        """
        return self._ctx

    def unsafe_ptr(
        self,
    ) -> UnsafePointer[Scalar[Self.dtype], MutUntrackedOrigin]:
        """Returns the raw device pointer without transferring ownership.

        This method provides direct access to the underlying device pointer
        for advanced use cases. The buffer retains ownership of the pointer.

        Returns:
            The raw device pointer owned by this buffer.
        """
        return (
            self._inner[]
            ._host_ptr.bitcast[Scalar[Self.dtype]]()
            .unsafe_origin_cast[MutUntrackedOrigin]()
        )

    def __getitem__(self, idx: Int) -> Scalar[Self.dtype]:
        """Retrieves the element at the specified index from the host buffer.

        This operator allows direct access to individual elements in the host buffer
        using array indexing syntax.

        Args:
            idx: The index of the element to retrieve.

        Returns:
            The scalar value at the specified index.
        """
        return self.unsafe_ptr()[idx]

    def __setitem__(self, idx: Int, val: Scalar[Self.dtype]):
        """Sets the element at the specified index in the host buffer.

        This operator allows direct modification of individual elements in the host buffer
        using array indexing syntax.

        Args:
            idx: The index of the element to modify.
            val: The new value to store at the specified index.
        """
        self.unsafe_ptr()[idx] = val

    def as_span[
        mut: Bool, origin: Origin[mut=mut], //
    ](ref[origin] self) -> Span[Scalar[Self.dtype], origin]:
        """Returns a `Span` pointing to the underlying memory of the `HostBuffer`.

        Parameters:
            mut: Whether the span should be mutable.
            origin: The origin of the buffer reference.

        Returns:
            A `Span` pointing to the underlying memory of the `HostBuffer`.
        """
        # Safety: We are casting the pointer to the mutability and origin of
        # self and the pointer is already mutable.
        return {
            unsafe_ptr = self.unsafe_ptr()
            .unsafe_mut_cast[mut]()
            .unsafe_origin_cast[origin](),
            length = len(self),
        }

    def enqueue_copy_to(self, dst: DeviceBuffer[Self.dtype]) raises:
        """Enqueues an asynchronous copy from this buffer to a device buffer.

        This method schedules a memory copy operation from this buffer to the destination
        device buffer. The operation is asynchronous and will be executed in the stream
        associated with this buffer's context.

        Args:
            dst: The destination device buffer to copy data to.

        Raises:
            If the operation fails.
        """
        self._ctx.enqueue_copy(dst, self)

    def enqueue_copy_from(self, src: DeviceBuffer[Self.dtype]) raises:
        """Enqueues an asynchronous copy from a device buffer to this buffer.

        This method schedules a memory copy operation from the source device buffer
        to this buffer. The operation is asynchronous and will be executed in the stream
        associated with this buffer's context.

        Args:
            src: The source device buffer to copy data from.

        Raises:
            If the operation fails.
        """
        self._ctx.enqueue_copy(self, src)

    def enqueue_copy_to(
        self, dst: Span[mut=True, Scalar[Self.dtype], _]
    ) raises:
        """Enqueues an asynchronous copy from this buffer to a host `Span`.

        Args:
            dst: The destination host span to copy data to.

        Raises:
            If the operation fails.
        """
        self._ctx.enqueue_copy(dst, self)

    def enqueue_copy_from(self, src: Span[Scalar[Self.dtype], _]) raises:
        """Enqueues an asynchronous copy from a host `Span` to this buffer.

        Args:
            src: The source host span to copy data from.

        Raises:
            If the operation fails.
        """
        self._ctx.enqueue_copy(self, src)


# ===-----------------------------------------------------------------------===#
# _HostMappedBuffer
# ===-----------------------------------------------------------------------===#


struct _HostMappedBuffer[dtype: DType]:
    var _ctx: DeviceContext
    var _dev_buf: DeviceBuffer[Self.dtype]
    var _cpu_buf: HostBuffer[Self.dtype]

    def __init__(
        out self, ctx: DeviceContext, buf: DeviceBuffer[Self.dtype]
    ) raises:
        var cpu_buf = ctx.enqueue_create_host_buffer[Self.dtype](len(buf))
        self._ctx = ctx
        self._dev_buf = buf
        self._cpu_buf = cpu_buf

    def __del__(deinit self):
        pass

    def __enter__(mut self) raises -> HostBuffer[Self.dtype]:
        self._dev_buf.enqueue_copy_to(self._cpu_buf)
        self._ctx.synchronize()
        return self._cpu_buf

    def __exit__(mut self) raises:
        self._ctx.synchronize()
        self._cpu_buf.enqueue_copy_to(self._dev_buf)
        self._ctx.synchronize()


# Create empty structs to ensure dtype checking when using the C++ handles.
struct _DeviceContextCpp:
    pass


struct _DeviceBufferCpp:
    pass


struct _DeviceFunctionCpp:
    pass


struct _DeviceMulticastBufferCpp:
    pass


struct _DeviceStreamCpp:
    pass


struct _DeviceEventCpp:
    pass


struct _DeviceTimerCpp:
    pass


struct _CompletionFlagCpp:
    pass


struct _DeviceContextScopeCpp:
    pass


struct _DeviceGraphBuilderCpp:
    pass


struct _DeviceGraphCpp:
    pass


comptime _DeviceContextPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = _CPointer[_DeviceContextCpp, origin]

comptime _DeviceBufferPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = _CPointer[_DeviceBufferCpp, origin]

comptime _DeviceFunctionPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = _CPointer[_DeviceFunctionCpp, origin]

comptime _DeviceMulticastBufferPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = _CPointer[_DeviceMulticastBufferCpp, origin]

comptime _DeviceStreamPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = _CPointer[_DeviceStreamCpp, origin]

comptime _DeviceEventPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = _CPointer[_DeviceEventCpp, origin]

comptime _DeviceTimerPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = _CPointer[_DeviceTimerCpp, origin]

comptime _CompletionFlagPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = _CPointer[_CompletionFlagCpp, origin]

comptime _DeviceContextScopePtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = _CPointer[_DeviceContextScopeCpp, origin]

comptime _DeviceGraphBuilderPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = _CPointer[_DeviceGraphBuilderCpp, origin]

comptime _DeviceGraphPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = _CPointer[_DeviceGraphCpp, origin]

comptime _CString[
    origin: Origin[mut=False] = UntrackedOrigin[mut=False]
] = Optional[CStringSlice[origin]]

comptime _DumpPath = Variant[Bool, Path, StaticString, def() capturing -> Path]


def _string_from_owned_charptr(c_str: _CString) -> String:
    var result = String()
    if c_str:
        result = String(unsafe_from_utf8_ptr=c_str.unsafe_value().unsafe_ptr())
    # void AsyncRT_DeviceContext_strfree(const char* ptr)
    external_call["AsyncRT_DeviceContext_strfree", NoneType](c_str)
    return result^


@no_inline
def _raise_checked_impl(
    err_msg: _CString, msg: String, location: SourceLocation
) raises:
    var err = _string_from_owned_charptr(err_msg)
    raise Error(location.prefix(err + ((" " + msg) if msg else "")))


def _memset_value_as_u64[dtype: DType](val: Scalar[dtype]) -> UInt64:
    """Packs a scalar value into the low bytes of a UInt64 for `set_memory`.

    This logic is copied directly from the existing DeviceContext, but is
    refactored into a separate function for conciceness.
    """
    comptime bitwidth = bit_width_of[dtype]()
    comptime assert (
        bitwidth == 8 or bitwidth == 16 or bitwidth == 32 or bitwidth == 64
    ), "bitwidth of memset dtype must be one of [8, 16, 32, 64]"
    comptime if bitwidth == 8:
        return UInt64(Int(bitcast[DType.uint8, 1](val)))
    elif bitwidth == 16:
        return UInt64(Int(bitcast[DType.uint16, 1](val)))
    elif bitwidth == 32:
        return UInt64(bitcast[DType.uint32, 1](val))
    else:
        return bitcast[DType.uint64, 1](val)


# ===-----------------------------------------------------------------------===#
# DeviceFunction
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct DefaultDeviceTypeEncoder(DeviceTypeEncoder):
    """Provides a default implementation of the `DeviceTypeEncoder` trait."""

    @staticmethod
    def target() -> _TargetType:
        """Returns the target architecture this encoder is encoding for.

        Returns:
            The target architecture this encoder is encoding for.
        """
        return _current_target()

    def encode_device_ptr(
        mut self, value: DevicePointer, dst: MutOpaquePointer[_]
    ):
        """Encodes a `DevicePointer` into `dst`.

        By default treat `DevicePointer` as `UnsafePointer`, works for Unified
        Memory targets such as CUDA and HIP.

        Args:
            value: The `DevicePointer` instance to encode into `dst`.
            dst: The opaque destination pointer to encode into.
        """
        value.unsafe_ptr()._to_device_type(self, dst)


# ===-----------------------------------------------------------------------===#
# DevicePointer
# ===-----------------------------------------------------------------------===#


struct DevicePointer[
    mut: Bool,
    //,
    dtype: DType,
    origin: Origin[mut=mut],
](
    DevicePassable,
    Equatable,
    ImplicitlyCopyable,
    TrivialRegisterPassable,
    Writable,
):
    """A host-side representation of a pointer to device memory that resides
    within a `DeviceBuffer`.

    A `DevicePointer` is a non-owning borrow of a `DeviceBuffer`; it must not
    outlive the buffer it points into.

    - Supports pointer arithmetic which may result in a new `DevicePointer`
      instance referring to the same `DeviceBuffer` with a new offset.
    - Supports equality comparison and ordering of `DevicePointer`s pointing
      into the same `DeviceBuffer`.
    - May support accessing device pointer address on supported hardware.
    - Does not support load/store operations.

    At the device function execution boundary a `DevicePointer` is transformed
    into an `UnsafePointer` on the device at the point of being handed over to
    the device driver.

    Parameters:
        mut: Whether the borrow of the underlying `DeviceBuffer` is mutable
            (inferred from `origin`).
        dtype: Data dtype to be stored in the pointer.
        origin: The origin of the borrowed `DeviceBuffer`.
    """

    var _buffer: Pointer[DeviceBuffer[Self.dtype], Self.origin]
    var _offset: Int
    var _size: Int

    # ===------------------------------------------------------------------=== #
    # Constructors
    # ===------------------------------------------------------------------=== #

    def __init__(
        out self, ref[Self.origin] buffer: DeviceBuffer[Self.dtype]
    ) raises:
        """Constructs a `DevicePointer` referencing the start of `buffer`.

        Args:
            buffer: The `DeviceBuffer` this pointer references. Must outlive
                the resulting `DevicePointer`.

        Raises:
            If `buffer` has size 0.
        """
        var size = len(buffer)
        if size == 0:
            raise Error("DevicePointer: size of DeviceBuffer must not be 0")
        self._buffer = Pointer(to=buffer)
        self._offset = 0
        self._size = size

    def __init__(
        out self, ref[Self.origin] buffer: DeviceBuffer[Self.dtype], offset: Int
    ) raises:
        """Constructs a `DevicePointer` into `buffer` at `offset` with `size`
        elements in range.

        Args:
            buffer: The `DeviceBuffer` this pointer references. Must outlive
                the resulting `DevicePointer`.
            offset: Element offset from the start of `buffer`.

        Raises:
            If `buffer` has size 0, or if `offset` is outside the half-open
            range `[0, len(buffer))`.
        """
        var size = len(buffer)
        if size == 0:
            raise Error("DevicePointer: invalid DeviceBuffer of size '0'")
        if offset < 0 or offset >= size:
            raise Error(
                t"DevicePointer: invalid offset '{offset}' for DeviceBuffer of"
                t" size '{size}'"
            )
        self._buffer = Pointer(to=buffer)
        self._offset = offset
        self._size = size

    # ===------------------------------------------------------------------=== #
    # Accessors
    # ===------------------------------------------------------------------=== #

    def buffer(self) -> ref[Self.origin] DeviceBuffer[Self.dtype]:
        """Returns a reference to the `DeviceBuffer` this pointer references.

        The reference is non-owning; the underlying `DeviceBuffer` must
        outlive `self`.

        Returns:
            A reference to the referenced `DeviceBuffer`.
        """
        return self._buffer[]

    def offset(self) -> Int:
        """Returns the element offset from the start of the owning buffer.

        Returns:
            The element offset.
        """
        return self._offset

    @doc_hidden
    def unsafe_ptr(
        ref self,
    ) -> UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]:
        """Returns the raw device pointer, if supported by the target.

        On targets that expose raw device pointers (for example CUDA and HIP),
        this returns the underlying address adjusted by the current offset.
        On targets that do not (for example Metal), this raises an error.

        Returns:
            The raw device pointer.
        """
        # TODO: GEX-3693: Assert/raise when target doesn't support raw device
        # pointer access
        # `DeviceBuffer.unsafe_ptr()` now ties its mutability to the borrow of
        # the buffer; force mutable to preserve this helper's `MutAnyOrigin`
        # contract.
        return (
            (self._buffer[].unsafe_ptr() + self._offset)
            .unsafe_mut_cast[True]()
            .as_unsafe_any_origin()
        )

    # ===------------------------------------------------------------------=== #
    # Pointer arithmetic
    # ===------------------------------------------------------------------=== #

    def __add__(self, n: Int) raises -> Self:
        """Returns a new `DevicePointer` offset forward by `n` elements.

        Args:
            n: Number of elements to offset by.

        Returns:
            A new `DevicePointer` referencing the same `DeviceBuffer` at the
            new offset.

        Raises:
            If the resulting offset is outside the bounds of the owning
            `DeviceBuffer`.
        """
        var offset = self._offset + n
        if offset < 0 or offset >= self._size:
            raise Error(
                t"DevicePointer: addition of '{n}' results in invalid offset of"
                t" '{offset}' for DeviceBuffer of size {self._size}"
            )
        return DevicePointer(self._buffer[], offset)

    def __sub__(self, n: Int) raises -> Self:
        """Returns a new `DevicePointer` offset backward by `n` elements.

        Args:
            n: Number of elements to offset by.

        Returns:
            A new `DevicePointer` referencing the same `DeviceBuffer` at the
            new offset.

        Raises:
            If the resulting offset is outside the bounds of the owning
            `DeviceBuffer`.
        """
        var offset = self._offset - n
        if offset < 0 or offset >= self._size:
            raise Error(
                t"DevicePointer: subtraction of '{n}' results in invalid offset"
                t" of '{offset}' for DeviceBuffer of size {self._size}"
            )
        return DevicePointer(self._buffer[], offset)

    def __iadd__(mut self, n: Int) raises:
        """Offsets this `DevicePointer` forward by `n` elements in place.

        Args:
            n: Number of elements to offset by.

        Raises:
            If the resulting offset is outside the bounds of the owning
            `DeviceBuffer`.
        """
        var offset = self._offset + n
        if offset < 0 or offset >= self._size:
            raise Error(
                t"DevicePointer: addition of '{n}' results in invalid offset of"
                t" '{offset}' for DeviceBuffer of size {self._size}"
            )
        self._offset = offset

    def __isub__(mut self, n: Int) raises:
        """Offsets this `DevicePointer` backward by `n` elements in place.

        Args:
            n: Number of elements to offset by.

        Raises:
            If the resulting offset is outside the bounds of the owning
            `DeviceBuffer`.
        """
        var offset = self._offset - n
        if offset < 0 or offset >= self._size:
            raise Error(
                t"DevicePointer: subtraction of '{n}' results in invalid offset"
                t" of '{offset}' for DeviceBuffer of size {self._size}"
            )
        self._offset = offset

    # ===------------------------------------------------------------------=== #
    # Comparison
    # ===------------------------------------------------------------------=== #

    @__unsafe_nested_origins_read_only
    def __eq__(self, other: Self) -> Bool:
        """Returns `True` if `self` and `other` reference the same buffer and
        offset.

        Args:
            other: The other `DevicePointer` to compare.

        Returns:
            `True` if equal.
        """
        # Buffer identity = base device address (HAL buffers have no C++ handle).
        return (
            self._buffer[]._device_ptr == other._buffer[]._device_ptr
            and self._offset == other._offset
        )

    @__unsafe_nested_origins_read_only
    def __eq__(self, other: DevicePointer[Self.dtype, _]) -> Bool:
        """Returns `True` if `self` and `other` reference the same buffer and
        offset.

        Args:
            other: The other `DevicePointer` to compare.

        Returns:
            `True` if equal.
        """
        return (
            self._buffer[]._device_ptr == other._buffer[]._device_ptr
            and self._offset == other._offset
        )

    @__unsafe_nested_origins_read_only
    def __ne__(self, other: Self) -> Bool:
        """Returns `True` if `self` and `other` differ in buffer or offset.

        Args:
            other: The other `DevicePointer` to compare.

        Returns:
            `True` if not equal.
        """
        return not (self == other)

    @__unsafe_nested_origins_read_only
    def __ne__(self, other: DevicePointer[Self.dtype, _]) -> Bool:
        """Returns `True` if `self` and `other` differ in buffer or offset.

        Args:
            other: The other `DevicePointer` to compare.

        Returns:
            `True` if not equal.
        """
        return not (self == other)

    @__unsafe_nested_origins_read_only
    def __lt__(self, other: DevicePointer[Self.dtype, _]) raises -> Bool:
        """Returns `True` if `self` precedes `other` within the same buffer.

        Args:
            other: The other `DevicePointer` to compare.

        Returns:
            `True` if `self` is ordered before `other`.

        Raises:
            If `self` and `other` reference different `DeviceBuffer`s.
        """
        if self._buffer[]._device_ptr != other._buffer[]._device_ptr:
            raise Error(
                "DevicePointer: less than comparison not supported when the"
                " underlying DeviceBuffer does not match"
            )
        return self._offset < other._offset

    @__unsafe_nested_origins_read_only
    def __le__(self, other: DevicePointer[Self.dtype, _]) raises -> Bool:
        """Returns `True` if `self` precedes or equals `other` within the
        same buffer.

        Args:
            other: The other `DevicePointer` to compare.

        Returns:
            `True` if `self` is ordered before or equal to `other`.

        Raises:
            If `self` and `other` reference different `DeviceBuffer`s.
        """
        if self._buffer[]._device_ptr != other._buffer[]._device_ptr:
            raise Error(
                "DevicePointer: less than or equal comparison not supported"
                " when the underlying DeviceBuffer does not match"
            )
        return self._offset <= other._offset

    @__unsafe_nested_origins_read_only
    def __gt__(self, other: DevicePointer[Self.dtype, _]) raises -> Bool:
        """Returns `True` if `self` follows `other` within the same buffer.

        Args:
            other: The other `DevicePointer` to compare.

        Returns:
            `True` if `self` is ordered after `other`.

        Raises:
            If `self` and `other` reference different `DeviceBuffer`s.
        """
        if self._buffer[]._device_ptr != other._buffer[]._device_ptr:
            raise Error(
                "DevicePointer: greater than comparison not supported when the"
                " underlying DeviceBuffer does not match"
            )
        return self._offset > other._offset

    @__unsafe_nested_origins_read_only
    def __ge__(self, other: DevicePointer[Self.dtype, _]) raises -> Bool:
        """Returns `True` if `self` follows or equals `other` within the same
        buffer.

        Args:
            other: The other `DevicePointer` to compare.

        Returns:
            `True` if `self` is ordered after or equal to `other`.

        Raises:
            If `self` and `other` reference different `DeviceBuffer`s.
        """
        if self._buffer[]._device_ptr != other._buffer[]._device_ptr:
            raise Error(
                "DevicePointer: greater than or equal comparison not supported"
                " when the underlying DeviceBuffer does not match"
            )
        return self._offset >= other._offset

    # ===------------------------------------------------------------------=== #
    # Writable
    # ===------------------------------------------------------------------=== #

    def write_to(self, mut writer: Some[Writer]):
        """Writes a string representation of this `DevicePointer` to `writer`.

        Args:
            writer: The writer to output the formatted string to.
        """
        writer.write(
            t"DevicePointer[{Self.dtype}]("
            t"buffer=DeviceBuffer(size={len(self._buffer[])}), "
            t"offset={self._offset})"
        )

    # ===------------------------------------------------------------------=== #
    # DevicePassable
    # ===------------------------------------------------------------------=== #

    comptime device_type: AnyType = UnsafePointer[
        mut=True, Scalar[Self.dtype], AnyOrigin[mut=True]
    ]
    """`DevicePointer` is remapped to `UnsafePointer` when passed to
    accelerator devices."""

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        """Device type mapping from `DevicePointer` to the device's
        `UnsafePointer`.
        """
        encoder.encode_device_ptr(self, target)

    @staticmethod
    def get_type_name() -> String:
        """Gets this type's name, for use in error messages when handing
        arguments to kernels.

        Returns:
            This type's name.
        """
        return String(t"DevicePointer[{Self.dtype}]")


# ===-----------------------------------------------------------------------===#
# Unsupported feature stubs
# ===-----------------------------------------------------------------------===#
#
# The HAL `DeviceContext` targets basic single-device execution. The following
# types exist so the `gpu.host` package surface matches the AsyncRT
# implementation, but their features (CUDA-graph capture, multi-GPU collectives
# and multicast, host-visible completion flags) are not implemented here.


struct CompletionFlag(ImplicitlyCopyable):
    """Host-visible completion flag.

    This struct is intentionally non-owning: it wraps the raw address of a
    ``M::Driver::CompletionFlag`` created by the C++ driver layer. The HAL
    backend has no C++ counterpart yet, so the handle is inert until the HAL
    C++ runtime shim lands; `DeviceStream.wait_for_host_value` raises before
    dereferencing it.
    """

    var _handle: OpaquePointer[MutUntrackedOrigin]

    @always_inline
    def __init__(out self, *, unsafe_from_address: Int):
        """Constructs a non-owning handle from an integer address.

        Intended for graph-op execute methods that extract a packed
        pointer from a payload buffer (mirroring how
        `mo.launch_host_func` rebuilds its trampoline/user-data
        pointers). The caller asserts that ``unsafe_from_address``
        points to a valid ``M::Driver::CompletionFlag`` and that the
        underlying object outlives any in-flight use.

        Args:
            unsafe_from_address: Raw address of an
                ``M::Driver::CompletionFlag`` (as packed into a graph
                payload buffer by the producer side).
        """
        self._handle = OpaquePointer[MutUntrackedOrigin](
            unsafe_from_address=unsafe_from_address
        )


struct DeviceContextList[size: Int](Copyable, ImplicitlyCopyable, Sized):
    """A fixed-size collection of `DeviceContext` values.

    Used by multi-device custom-op `execute` methods to receive one
    `DeviceContext` per participating device. The graph compiler recognizes
    this type and synthesizes it from the per-device contexts discovered on the
    operation.

    Parameters:
        size: The number of `DeviceContext` values in the collection.
    """

    var device_contexts: InlineArray[DeviceContext, Self.size]
    """The underlying storage for the per-device contexts."""

    @always_inline
    def __init__(
        out self, device_contexts: InlineArray[DeviceContext, Self.size]
    ):
        """Initialize from an `InlineArray` of `DeviceContext` values.

        Args:
            device_contexts: The per-device contexts to store.
        """
        self.device_contexts = device_contexts

    @always_inline
    def __init__(
        out self,
        var *device_contexts: DeviceContext,
        __list_literal__: NoneType = None,
    ):
        """Initialize from a variadic sequence of `DeviceContext` values.

        Args:
            device_contexts: One `DeviceContext` per device, exactly `size` of
                them.
            __list_literal__: Marker that lets this constructor accept
                list-literal syntax.
        """
        assert (
            len(device_contexts) == Self.size
        ), "mismatch in the number of elements"
        self.device_contexts = InlineArray[DeviceContext, Self.size](
            *device_contexts^, __list_literal__=None
        )

    def __getitem_param__[index: Int](self) -> DeviceContext:
        """Access a `DeviceContext` at a compile-time known index.

        Parameters:
            index: A compile-time integer index.

        Returns:
            The `DeviceContext` at the specified index.
        """
        return self.device_contexts[index]

    def __getitem__[I: Indexer, //](self, idx: I) -> DeviceContext:
        """Access a `DeviceContext` using a runtime index value.

        Parameters:
            I: A type that conforms to the `Indexer` trait.

        Args:
            idx: A runtime index value that conforms to the `Indexer` trait.

        Returns:
            The `DeviceContext` at the specified index.
        """
        return self.device_contexts[idx]

    def __len__(self) -> Int:
        """Get the number of `DeviceContext` values in the collection.

        Returns:
            The size of the collection as specified by the `size` parameter.
        """
        return Self.size

    def filter_gpu_contexts[
        num_gpu_devices: Int
    ](self) raises -> InlineArray[DeviceContext, num_gpu_devices]:
        """Filters CPU contexts out and returns the GPU contexts in order.

        Some kernels receive a `DeviceContextList` that mixes GPU contexts
        with CPU contexts carrying host-side pointers. Most kernels only
        want the GPU contexts in launch order, packed into a fixed-size
        `InlineArray`.

        Parameters:
            num_gpu_devices: The expected number of GPU contexts. Used as
                the size of the returned `InlineArray`.

        Returns:
            An `InlineArray` of size `num_gpu_devices` containing the GPU
            contexts in their original order.

        Raises:
            If the number of GPU contexts in the list is not equal to
            `num_gpu_devices`.
        """
        # Validate the count up front. Passing a partially-filled staging
        # array to `unsafe_assume_initialized=` would still be UB at the
        # eventual destruction of the returned `InlineArray`.
        var gpu_count = 0
        for i in range(Self.size):
            if self[i].api() != "cpu":
                gpu_count += 1
        if gpu_count != num_gpu_devices:
            raise Error("Invalid number of GPU device contexts")

        # Build the result in an `UnsafeMaybeUninit` staging array. Its
        # `__del__` is a no-op, so the staging array is safe to drop even
        # with uninitialized slots in scope (e.g. on an early raise). The
        # `unsafe_assume_initialized=` constructor then moves every slot
        # into a fully-initialized `InlineArray[DeviceContext]`.
        var staging = InlineArray[
            UnsafeMaybeUninit[DeviceContext], num_gpu_devices
        ](uninitialized=True)
        var dev_idx = 0
        for i in range(Self.size):
            if self[i].api() != "cpu":
                staging[dev_idx].init_from(DeviceContext(copy=self[i]))
                dev_idx += 1
        return InlineArray[DeviceContext, num_gpu_devices](
            unsafe_assume_initialized=staging^
        )


struct DeviceMulticastBuffer[dtype: DType]:
    """A multi-GPU multicast buffer (unsupported by the HAL `DeviceContext`).

    Parameters:
        dtype: Data dtype stored in the buffer.
    """

    pass


@always_inline
def _checked(
    err: _CString,
    *,
    msg: String = "",
    location: OptionalReg[SourceLocation] = None,
) raises:
    if err:
        _raise_checked_impl(err, msg, location.or_else(call_location()))
