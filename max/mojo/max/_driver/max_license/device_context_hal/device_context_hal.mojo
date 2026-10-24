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

# Implementation of DeviceContext backed by the HAL
"""This module provides functionality for interacting with accelerators. In
particular the
[`DeviceContext`](/docs/max/gpu/host/device_context/DeviceContext/) struct,
which represents a single stream of execution on a given accelerator. You can
use this struct to allocate accelerator memory, copy data to and from the
accelerator, and compile and execute functions on the accelerator.

This variant of the module provides the `max.gpu.host._legacy.device_context`
implementation when the HAL backend is enabled."""

from std.atomic import Atomic
from std.builtin.device_passable import (
    DevicePassable,
    DevicePointerLike,
    DeviceTypeEncoder,
)
from std.builtin.rebind import downcast
from std.collections.optional import OptionalReg
from std.ffi import CStringSpan, _Global, c_size_t, external_call
from std.compile import CompiledFunctionInfo
from std.math import align_up
from std.memory import (
    alloc,
    dealloc,
    unsafe_memcpy,
    ThinAllocation,
    ArcPointer,
    Layout,
    MaybeUninit,
    Pointer,
)
from std.memory.unsafe import bitcast
from std.pathlib import Path
from std.utils import Variant
from std.utils.lock import BlockingScopedLock, BlockingSpinLock
from std.memory import unsafe_stack_allocation
from std.os import abort, getenv
from std.reflection import call_location, reflect, SourceLocation
from std.sys import bit_width_of, size_of, CompilationTarget
from std.sys.info import is_gpu
from std.time import monotonic
from ..hal import (
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
from ..hal._machine import DeviceSpec
from ..hal.event import EVENT_FLAG_CPU_VISIBLE
from ..hal.extensions.traits import SupportsPeerAccessV1
from ..hal.execution_config import (
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

from max.gpu.host import (
    ConstantMemoryMapping,
    Dim,
    LaunchAttribute,
    default_accelerator,
)
from max.gpu.host import DeviceAttribute
from max.gpu.host.info import (
    GPUInfo,
    get_gpu_info,
    is_cpu,
    _device_type_encoder_target,
)
from max.gpu.host import Attribute, FuncAttribute
from max.gpu.host import _launch_args
from max.gpu.host._launch_args import DeviceFunctionWrapper, _IdentityWrapper


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

    The function handle (and the `RuntimeBundle` defining it) is owned by the
    `Context`'s function cache and stays loaded for the context's lifetime;
    the held `Context` `ArcPointer` keeps it valid.
    """

    var _func_handle: FunctionHandle
    var _context: ArcPointer[Context[get_device_spec[0]()]]


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


struct _PeerState(Movable):
    """Process-wide peer-access state, as in the legacy C++ `canAccessPeer_`.

    `lock` guards only this bookkeeping, never a driver call, because
    collectives read `all_enabled` from AsyncRT workers. The writer that
    completes the matrix publishes `all_enabled`, so readers never lock.
    """

    var lock: BlockingSpinLock
    var all_enabled: Atomic[Int32]
    var device_count: Int
    var enabled: List[Bool]
    # Releasing the last ref resets the CUDA primary context and its peer grant.
    var contexts: List[Optional[ArcPointer[Context[get_device_spec[0]()]]]]

    def __init__(out self, device_count: Int):
        self.lock = BlockingSpinLock()
        self.all_enabled = Atomic[Int32](0)
        self.device_count = device_count
        self.enabled = List[Bool](
            length=device_count * device_count, fill=False
        )
        self.contexts = List[
            Optional[ArcPointer[Context[get_device_spec[0]()]]]
        ](length=device_count, fill=None)

    def retain(
        mut self,
        device_id: Int,
        context: ArcPointer[Context[get_device_spec[0]()]],
    ):
        if not self.contexts[device_id]:
            self.contexts[device_id] = context

    def record_enabled(
        mut self,
        src: Int,
        src_context: ArcPointer[Context[get_device_spec[0]()]],
        dst: Int,
        dst_context: ArcPointer[Context[get_device_spec[0]()]],
    ):
        """Records a granted `src` -> `dst` pair. Call with `lock` held."""
        self.retain(src, src_context)
        self.retain(dst, dst_context)
        self.enabled[src * self.device_count + dst] = True
        self.enabled[src * self.device_count + src] = True
        if self.all_set():
            self.all_enabled.store(1)

    def all_set(self) -> Bool:
        if self.device_count < 2:
            return False
        for b in self.enabled:
            if not b:
                return False
        return True


def _init_peer_state() -> _PeerState:
    # `_Global` initializers must not raise; no plugin means zero devices.
    try:
        var maybe_driver = _GPU_DRIVER.get_or_create_ptr()
        if not maybe_driver[]:
            return _PeerState(0)
        return _PeerState(Int(maybe_driver[].value()[].get_device_count()))
    except:
        return _PeerState(0)


comptime _PEER_STATE = _Global["MODULAR_HAL_PEER_STATE", _init_peer_state]


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

    @inline(.always)
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


comptime DeviceContext = DeviceContextBase[]
"""Temporary type alias as DeviceContext transitions to being
parameterized by compilation target.

See `DeviceContextBase` for additional documentation.
"""


struct DeviceContextBase[target: CompilationTarget = default_accelerator()](
    ImplicitlyCopyable, RegisterPassable, _HALFunctionEnqueuer
):
    """Represents a single stream of execution on a particular accelerator
    (GPU).

    A `DeviceContext` serves as the low-level interface to the
    accelerator inside a MAX [custom operation](/max/develop/custom-ops/) and provides
    methods for allocating buffers on the device, copying data between host and
    device, and for compiling and running functions (also known as kernels) on
    the device.

    Parameters:
        target: The accelerator targeted by this device context.

    The device context can be used as a
    [context manager](/docs/manual/errors/#use-a-context-manager). For example:

    ```mojo
    from max.gpu.host import DeviceContext
    from max.gpu import thread_idx

    def kernel():
        print("hello from thread:", thread_idx.x, thread_idx.y, thread_idx.z)

    with DeviceContext() as ctx:
        ctx.enqueue_function[kernel, kernel](grid_dim=1, block_dim=(2, 2, 2))
        ctx.synchronize()
    ```

    A custom operation receives the `DeviceContext` for the target device
    directly as an argument to its `execute` method:

    ```text
    from max.gpu.host import DeviceContext
    from extensibility import register

    @register("custom_op")
    struct CustomOp:
        @staticmethod
        def execute(ctx: DeviceContext) raises:
            ctx.enqueue_function[kernel, kernel](grid_dim=1, block_dim=(2, 2, 2))
            ctx.synchronize()
    ```
    """

    # ===-------------------------------------------------------------------===#
    # Aliases
    # ===-------------------------------------------------------------------===#

    comptime Generic = DeviceContextBase[_]
    """DeviceContext type with unbound compilation target parameter.

    This is a tempoary type alias as DeviceContext transitions to being
    parameterized by compilation target.
    """

    # DeviceContext is only usable on hosts so `current()` will always be a CPU.
    comptime CPU = DeviceContextBase[CompilationTarget.current()]
    """DeviceContext type for the current CPU target."""

    comptime device_spec = get_device_spec[0]()
    """`DeviceSpec` describing the device this context targets."""

    comptime default_device_info = get_gpu_info[
        DeviceSpec.target[Self.device_spec]()._arch()
    ]()
    """`GPUInfo` object for the default accelerator."""

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var _driver: ArcPointer[Driver]
    var _device: ArcPointer[Device[Self.device_spec]]
    var _context: ArcPointer[Context[Self.device_spec]]
    var _stream: ArcPointer[Stream[Self.device_spec]]
    # Null until the C++ runtime shim provides it; read by the CUDA/HIP
    # interop accessors (`_nvidia_cuda` / `_amdgpu_hip`).
    var _handle: _DeviceContextPtr[mut=True]

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @inline(.always)
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
        from max.gpu.host import DeviceContext

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
        if not _driver_satisfies_api(String(chosen.split("@")[0]), "gpu"):
            self._driver = Driver.create(chosen)

            # Validate that the loaded plugin satisfies the requested api (its
            # concrete vendor name, or the generic accelerator alias "gpu").
            var driver_name = self._driver[].get_name()
            if not _driver_satisfies_api(String(driver_name), api):
                raise Error(
                    String(
                        t"Requested API {api} not supported by driver"
                        t" {driver_name}"
                    )
                )
            self._device = self._driver[].get_device[0]()
        else:
            # Shared so peer calls get both contexts from one driver.
            var maybe_driver = _GPU_DRIVER.get_or_create_ptr()
            if not maybe_driver[]:
                # `_Global` init drops the load error; reload to raise it.
                _ = Driver.create(chosen)
                raise Error(
                    String(t"failed to load accelerator plugin '{chosen}'")
                )
            var driver = maybe_driver[].value()
            var driver_name = driver[].get_name()
            if not _driver_satisfies_api(String(driver_name), api):
                raise Error(
                    String(
                        t"Requested API {api} not supported by driver"
                        t" {driver_name}"
                    )
                )
            var count = Int(driver[].get_device_count())
            if device_id < 0 or device_id >= count:
                raise Error(
                    String(t"device_id {device_id} out of range [0, {count})")
                )
            self._driver = driver
            # The spec stays id 0 for typing; the runtime id picks the device.
            self._device = rebind[ArcPointer[Device[Self.device_spec]]](
                self._driver[].get_device_dynamic[Optional[Int64](0)](
                    Int64(device_id)
                )
            )
        self._context = self._device[].get_context()
        self._stream = self._context[].create_stream()

    @doc_hidden
    def __init__(out self, ctx_ptr: _DeviceContextPtr[mut=True]):
        """Create a Mojo DeviceContext from the opaque device-context handle
        carried by the runtime."""
        # FIXME(MSTDL-3258): Remove default target assumption rebind/erasure.
        self = rebind[Self](
            Self._from_unsafe_cpp_handle(
                ctx_ptr.unsafe_value().unsafe_bitcast[NoneType]()
            )
        )

    @inline(.always)
    def __init__(out self, handle: OpaquePointer[MutUntrackedOrigin]):
        """Reconstructs a `DeviceContext` from an opaque runtime handle.

        The graph runtime hands op code the opaque device-context handle; this
        recovers the Mojo `DeviceContext` it wraps.

        Args:
            handle: The opaque device-context handle carried by the runtime.
        """
        # FIXME(MSTDL-3258): Remove default target assumption rebind/erasure.
        self = rebind[Self](Self._from_unsafe_cpp_handle(handle))

    @staticmethod
    def _from_unsafe_cpp_handle(
        handle: OpaquePointer[MutUntrackedOrigin],
    ) -> DeviceContext:
        return handle.unsafe_bitcast[DeviceContext]()[]

    # ===-------------------------------------------------------------------===#
    # Factory methods
    # ===-------------------------------------------------------------------===#

    @staticmethod
    def cpu() -> DeviceContext.CPU:
        """Creates a CPU `DeviceContext`.

        Returns:
            A new CPU `DeviceContext`.
        """
        try:
            return DeviceContext.CPU(api="cpu")
        except e:
            abort(t"unable to create CPU DeviceContext: {e}")

    # ===-------------------------------------------------------------------===#
    # TODO(MSTDL-3208): Temporary target parameterization helpers
    # ===-------------------------------------------------------------------===#

    # TODO(MSTDL-3265): Remove `target: StaticString` parameters from all
    #   callers of this method after verifying all callers instead indicate
    #   target device via `DeviceContext.target`.
    @inline(.always)
    def _comptime_assert_matches_target_is_cpu[
        target_name: StaticString
    ](self,):
        comptime assert Self.target.is_cpu() == is_cpu[target_name](), String(
            "`target` parameter '",
            target_name,
            "' does not match the DeviceContext target, which is ",
            "a CPU" if Self.target.is_cpu() else "not a CPU",
            " (triple: ",
            StaticString(Self.target.__triple()),
            ", arch: ",
            Self.target._arch(),
            ")",
        )

    @inline(.always)
    def cast_if_cpu[
        target_str: StaticString
    ](
        var self,
        out result: DeviceContextBase[
            CompilationTarget[
                _mlir_value=(
                    CompilationTarget.current()._mlir_value if target_str
                    == "cpu" else Self.target._mlir_value
                )
            ]()
        ],
    ):
        """Temporary function to cast this DeviceContext to the CPU target if
        the device label is "cpu".

        Parameters:
            target_str: The device label to cast to. Typically "cpu" or "gpu".

        Returns:
            A DeviceContext with a target parameter matching the provided device
            label.
        """
        # Assert that this is only called on `ctx: DeviceContext` concrete /
        # default accelerator instances for the time being.
        var _ = self.assert_default_target()
        comptime if target_str == "cpu":
            # Defensive sanity check, since we make this assumption in the
            # result type.
            comptime assert CompilationTarget.current().is_cpu()

            var self_cpu = self.assert_cpu_target()
            return rebind_var[type_of(result)](self_cpu^)
        else:
            return rebind_var[type_of(result)](self^)

    @inline(.always)
    def assert_cpu_target(var self) -> DeviceContext.CPU:
        """Temporary function to assert and cast this DeviceContext to the CPU
        target.

        This method will be removed once all uses have migrated to static CPU
        DeviceContext target types.

        The check is performed at runtime on the context's API, since the
        static `target` of a plain `DeviceContext` is always the default
        accelerator, even for contexts created with `api="cpu"`.

        Returns:
            This device context cast to the CPU target.
        """
        if self.api() != "cpu":
            abort(
                String(
                    t"using DeviceContext with non-CPU target is not supported"
                    t" from: {call_location()}"
                )
            )
        return rebind_var[DeviceContext.CPU](self)

    # FIXME: The more obvious spelling of this currently crashes the compiler:
    #    context.map(DeviceContext.assert_cpu_target)
    @staticmethod
    def _map_assert_cpu_target(
        ctx: Optional[DeviceContext],
    ) -> Optional[DeviceContext.CPU]:
        var cpu_ctx = Optional[DeviceContext.CPU]()
        if ctx:
            cpu_ctx = ctx.value().assert_cpu_target()
        return cpu_ctx^

    # TODO(MSTDL-3258): Remove once all APIs fully support generic DeviceContext
    @inline(.always)
    def assert_default_target(self) -> DeviceContext:
        """Temporary function to assert and cast this DeviceContext to the
        default target.

        This method will be removed once all APIs fully support generic target
        parameters.

        Constraints:
            This DeviceContext must be using the default target, else a compile
            time assertion will fail.

        Returns:
            This device context cast to the default target.
        """
        comptime assert Self.target._eq_triple_and_arch[
            default_accelerator()
        ](), String(
            t"using DeviceContext with non-default target is not supported fro"
            t"m: {call_location()}"
        )
        return rebind[DeviceContext](self)

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    def __enter__(var self) -> Self:
        """Enables the use of `DeviceContext` in a `with` statement context manager.

        Returns:
            The `DeviceContext` instance to be used within the context manager block.
        """
        return self^

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    def synchronize(self) raises:
        """Blocks until all asynchronous calls on the stream associated with
        this device context have completed.

        Raises:
            If the operation fails.
        """
        self._stream[].synchronize()

    def id(self) raises -> Int64:
        """Returns the ID associated with this device.

        Returns:
            The unique device ID as an `Int64`.

        Raises:
            If there's an error retrieving the device ID.
        """
        return self._device[].id

    @inline(.always)
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
        from max.gpu.host import DeviceContext

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

    @inline(.always)
    def get_attribute(self, attr: DeviceAttribute) raises -> Int:
        """Returns the specified attribute for this device.

        Use the aliases defined by
        [DeviceAttribute](/docs/max/gpu/host/device_attribute/DeviceAttribute/)
        to specify attributes. For example:

        ```mojo
        from max.gpu.host import DeviceAttribute, DeviceContext

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
    @inline(.always)
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
        # TODO: Apple GPUs report the chip generation (M1-M5) directly as the
        # compute capability, matching the native MetalDeviceContext
        # (`MetalDevice::computeCapability` returns the MAJOR only). The
        # major/minor split in device attributes is not portable and should be
        # changed.
        if self.api() == "metal":
            return major
        var minor = self.get_attribute(DeviceAttribute.COMPUTE_CAPABILITY_MINOR)
        return major * 10 + minor

    def set_as_current(self) raises:
        """For use with libraries that require a specific GPU context to be
        active. Sets the current device to the one associated with this
        DeviceContext.

        Example:

        ```mojo
        from max.gpu.host import DeviceContext
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
        from max.gpu.host import DeviceContext

        var ctx = DeviceContext(device_id=1)
        # Ensure GPU 1's context is active for these operations.
        with ctx.push_context():
            # All GPU operations here will use GPU 1's context.
            ...  # call external stateful APIs, such as cudnn.
        # Previous context is automatically restored
        ```
        """
        return _DeviceContextScopeHAL(self.assert_default_target())

    @doc_hidden
    def stream(self) -> DeviceStream:
        return DeviceStream(self.assert_default_target())

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
        return DeviceStream(self.assert_default_target(), hal_stream^)

    def create_event(self) -> DeviceEvent:
        """Creates a new event for synchronization between streams.

        Returns:
            A DeviceEvent that can be used for synchronization.

        Example:

        ```mojo
        from max.gpu.host import DeviceContext

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
        return DeviceEvent._create_unrecorded(self.assert_default_target())

    @inline(.always)
    def compile_function[
        declared_arg_types: TypeList[Trait=AnyType, ...],
        //,
        func: def(*args: *declared_arg_types) thin -> None,
        # Ignored; accepted for API parity with the legacy backend.
        compile_options: StaticString = "",
        link_options: StaticString = "",
        dump_asm: _DumpPath = False,
        dump_llvm: _DumpPath = False,
        _dump_sass: _DumpPath = False,
        _ptxas_info_verbose: Bool = False,
    ](
        self, *, func_attribute: OptionalReg[FuncAttribute] = None
    ) raises -> DeviceFunction[func]:
        """Compiles the provided `capturing` function for execution on this
        device.

        Parameters:
            declared_arg_types: Types of the arguments to pass to the device function.
            func: The function to compile.
            compile_options: Change the compile options to different options
                than the ones associated with this `DeviceContext`.
            link_options: Additional linker flags and options as a string.
            dump_asm: To dump the compiled assembly, pass `True`, or a file
                path to dump to, or a function returning a file path.
            dump_llvm: To dump the generated LLVM code, pass `True`, or a file
                path to dump to, or a function returning a file path.
            _dump_sass: Only runs on NVIDIA targets, and requires CUDA Toolkit
                to be installed. Pass `True`, or a file path to dump to, or a
                function returning a file path.
            _ptxas_info_verbose: Only runs on NVIDIA targets, and requires CUDA
                Toolkit to be installed. Changes `dump_asm` to output verbose
                PTX assembly (default `False`).

        Args:
            func_attribute: An attribute to use when compiling the code (such
                as maximum shared memory size).

        Returns:
            The compiled function.

        Raises:
            If the operation fails.
        """
        return DeviceFunction[func](
            self.assert_default_target(), func_attribute=func_attribute
        )

    @inline(.always)
    def enqueue_function[
        declared_arg_types: TypeList[Trait=AnyType, ...],
        ExecutionConfigType: GridBlockExecutionConfig,
        //,
        func: def(*args: *declared_arg_types) thin -> None,
        *actual_arg_types: DevicePassable,
    ](
        self,
        mut execution_config: ExecutionConfigType,
        *args: *actual_arg_types,
        location: Optional[SourceLocation] = None,
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
        from max.gpu.host import DeviceContext
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
        from max.gpu.host import DeviceContext

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

    @inline(.always)
    def enqueue_function[
        FuncType: ImplicitlyCopyable & RegisterPassable & def() -> None,
        ExecutionConfigType: GridBlockExecutionConfig,
        //,
    ](
        self,
        mut execution_config: ExecutionConfigType,
        func: FuncType,
        location: Optional[SourceLocation] = None,
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

        var gpu_kernel = DeviceFunction[FuncType.__call__](
            self.assert_default_target()
        )

        gpu_kernel._do_call_with_pack_checked(
            self,
            execution_config,
            host_args=(func,),
            location=location.or_else(call_location()),
        )

    @inline(.always)
    def enqueue_function[
        ExecutionConfigType: GridBlockExecutionConfig,
        //,
        *Ts: DevicePassable,
    ](
        self,
        mut execution_config: ExecutionConfigType,
        f: DeviceFunction,
        *args: *Ts,
        location: Optional[SourceLocation] = None,
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
        from max.gpu.host import DeviceContext

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
        f._do_call_with_pack_checked(
            self,
            execution_config,
            *args,
            location=location.or_else(call_location()),
        )

    @inline(.always)
    def enqueue_function[
        declared_arg_types: TypeList[Trait=AnyType, ...],
        HostArgs: TypeList[Trait=RegisterPassable, ...],
        //,
        func: def(*args: *declared_arg_types) thin -> None,
        *encoded_types: DevicePassable,
        # Ignored; accepted for API parity with the legacy backend.
        link_options: StaticString = "",
        dump_asm: _DumpPath = False,
        dump_llvm: _DumpPath = False,
        _dump_sass: _DumpPath = False,
        _ptxas_info_verbose: Bool = False,
    ](
        self,
        *encoded_args: *encoded_types,
        host_args: Tuple[*HostArgs] = (),
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        func_attribute: OptionalReg[FuncAttribute] = None,
        location: Optional[SourceLocation] = None,
    ) raises:
        """Compiles and enqueues a capturing kernel, encoding `DevicePassable` args and
        passing the trailing host arguments by host layout.

        Put capturing closures in `host_args`. The named kernel is compiled
        against those host callables; encoding them would rewrite captures to a
        synthesized `device_type` the kernel cannot read. Kernel parameters
        must be encoded args first, then the host arguments.

        Build `host_args` inline in the call with a tuple:

        ```mojo
        ctx.enqueue_function[kernel](
            buffer,
            host_args=(epilogue_fn,),
            grid_dim=1,
            block_dim=1,
        )
        ```

        Parameters:
            declared_arg_types: Types of the arguments to pass to the device function.
            HostArgs: The host-layout types of the trailing arguments.
            func: The function to compile and launch.
            encoded_types: The `DevicePassable` argument types encoded to
                device layout.
            link_options: Ignored; accepted for parity with the legacy backend.
            dump_asm: Ignored; accepted for parity with the legacy backend.
            dump_llvm: Ignored; accepted for parity with the legacy backend.
            _dump_sass: Ignored; accepted for parity with the legacy backend.
            _ptxas_info_verbose: Ignored; accepted for parity with the legacy backend.

        Args:
            encoded_args: The variadic `DevicePassable` arguments encoded to
                device layout.
            host_args: The trailing arguments passed by host layout.
            grid_dim: The grid dimensions.
            block_dim: The block dimensions.
            cluster_dim: The cluster dimensions.
            shared_mem_bytes: Per-block memory shared between blocks.
            attributes: A `List` of launch attributes.
            constant_memory: A `List` of constant memory mappings.
            func_attribute: An attribute to use when compiling the code (such
                as maximum shared memory size).
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
        var gpu_kernel = self.compile_function[func](
            func_attribute=self._func_attribute_for_launch(
                func_attribute, shared_mem_bytes
            ),
        )
        gpu_kernel._call_with_pack_checked(
            self,
            *encoded_args,
            host_args=host_args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
            location=location.or_else(call_location()),
        )

    @inline(.always)
    def enqueue_function[
        FuncType: ImplicitlyCopyable & RegisterPassable & def() -> None,
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
        func_attribute: OptionalReg[FuncAttribute] = None,
        location: Optional[SourceLocation] = None,
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
            func_attribute: An attribute to use when compiling the code (such
                as maximum shared memory size).
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
        var gpu_kernel = DeviceFunction[FuncType.__call__](
            self.assert_default_target(),
            func_attribute=self._func_attribute_for_launch(
                func_attribute, shared_mem_bytes
            ),
        )
        gpu_kernel._call_with_pack_checked(
            self,
            host_args=(func,),
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
            location=location.or_else(call_location()),
        )

    @inline(.always)
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
        location: Optional[SourceLocation] = None,
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
        from max.gpu.host import DeviceContext

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

    @inline(.always)
    def load_function[
        ret_type: AnyType,
        declared_arg_types: TypeList[Trait=AnyType, ...],
        //,
        func: def(*args: *declared_arg_types) thin -> ret_type,
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
            ret_type: The return type of `func` (inferred).
            declared_arg_types: The argument types of `func` (inferred).
            func: A function with the loaded kernel's signature. Only its type
                is used.

        Args:
            function_name: The name of the function in the assembly code.
            asm: The assembly code (PTX/SASS) containing the function.

        Returns:
            The loaded function is stored in the `result` parameter.

        Raises:
            If loading the function fails or the assembly code is invalid.
        """
        return DeviceExternalFunction(
            self.assert_default_target(), function_name^, asm
        )

    @inline(.always)
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
        location: Optional[SourceLocation] = None,
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
        return DeviceBuffer[dtype](self.assert_default_target(), size)

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
        return DeviceBuffer[dtype](self.assert_default_target(), size)

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
        from max.gpu.host import DeviceContext

        with DeviceContext() as ctx:
            # Allocate host memory accessible by the device
            var host_buffer = ctx.enqueue_create_host_buffer[.float32](1024)

            # Use the host buffer for device operations
            # ...
        ```
        """
        return HostBuffer[dtype](self.assert_default_target(), size)

    def enqueue_copy[
        dtype: DType
    ](
        self,
        dst_ptr: Pointer[mut=True, Scalar[dtype], _],
        src_ptr: Pointer[Scalar[dtype], _],
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
            DeviceBuffer[dtype](
                self.assert_default_target(), dst_ptr, size, owning=False
            ),
            DeviceBuffer[dtype](
                self.assert_default_target(), src_ptr, size, owning=False
            ),
        )

    def enqueue_copy[
        dtype: DType
    ](
        self,
        dst_buf: DeviceBuffer[dtype],
        src_ptr: Pointer[Scalar[dtype], _],
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
            src_ptr.unsafe_bitcast[UInt8](),
        )

    def enqueue_copy[
        dtype: DType
    ](
        self,
        dst_ptr: Pointer[mut=True, Scalar[dtype], _],
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
            dst_ptr.unsafe_bitcast[UInt8](),
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
            If the operation fails, or if `dst_buf` is not on this context's
            device.
        """
        # The plugin writes `dst` through this queue's context.
        var dst_device = dst_buf._inner[]._context[].get_device_id()
        if dst_device != self._device[].id:
            raise Error(
                String(
                    t"enqueue_copy destination is on device {dst_device}, not"
                    t" on this context's device {self._device[].id}"
                )
            )
        var src_context = src_buf._inner[]._context
        var dst_view = dst_buf._inner[]._buffer.view()
        var src_view = src_buf._inner[]._buffer.view()
        if src_context[].get_device_id() == self._device[].id:
            self._stream[].copy_intra_device(dst_view, src_view)
        else:
            # This plugin would misread another driver's context handle.
            if not (src_context[]._raw is self._context[]._raw):
                raise Error(
                    "enqueue_copy source is on another driver; cross-device"
                    " copies need both buffers on the same driver"
                )
            self._stream[].copy_inter_device(
                dst_view, src_view, src_context[]._handle
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
            If the operation fails, or if `src_buf` is smaller than `dst_buf`.
        """
        _check_copy_src_size(dst_buf._inner[]._buffer, src_buf._inner[]._buffer)
        self._stream[].copy_to_device(
            dst_buf._inner[]._buffer.view(),
            src_buf.unsafe_ptr().unsafe_bitcast[UInt8](),
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
            If the operation fails, or if `src_buf` is smaller than `dst_buf`.
        """
        ref dst_mem = dst_buf._inner[]._buffer
        ref src_mem = src_buf._inner[]._buffer
        _check_copy_src_size(dst_mem, src_mem)
        # `copy_from_device` copies the whole source view, so narrow it to the
        # destination's size.
        self._stream[].copy_from_device(
            dst_buf.unsafe_ptr().unsafe_bitcast[UInt8](),
            src_mem.view(byte_offset=0, byte_size=dst_mem.byte_size),
        )

    def enqueue_copy[
        dtype: DType
    ](
        self,
        dst_buf: HostBuffer[dtype],
        src_ptr: Pointer[Scalar[dtype], _],
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
            src_ptr.unsafe_bitcast[UInt8](),
        )

    def enqueue_copy[
        dtype: DType
    ](
        self,
        dst_ptr: Pointer[mut=True, Scalar[dtype], _],
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
            dst_ptr.unsafe_bitcast[UInt8](),
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
            If the operation fails, or if `src_buf` is smaller than `dst_buf`.
        """
        _check_copy_src_size(dst_buf._inner[]._buffer, src_buf._inner[]._buffer)
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

    @inline(.always)
    def _func_attribute_for_launch(
        self,
        func_attribute: OptionalReg[FuncAttribute],
        shared_mem_bytes: OptionalReg[Int],
    ) -> OptionalReg[FuncAttribute]:
        """Returns the function attribute to compile a launch with.

        CUDA's default cap is 48KB minus the kernel's static shared memory, so
        the cap is set for every request, not only past a threshold.
        """
        var requested_bytes = shared_mem_bytes.or_else(0)
        if func_attribute or requested_bytes <= 0:
            return func_attribute
        return FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(requested_bytes)
        )

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

    @inline(.always)
    def enqueue_cpu_range[
        FuncType: def(Int) -> None,
    ](self, func: FuncType, count: Int) raises:
        """Runs a function closure over a 1D range on a CPU `DeviceContext`.

        The function is called as `func(i)` for each `i` in `range(count)`.

        The HAL backend has no host-function enqueue plugin entry, so the range
        runs serially and synchronously here; `synchronize()` is then a no-op
        for this work. Results match the legacy backend; only the parallelism
        differs.

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

    @inline(.always)
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

        Raises:
            If synchronization fails or if the function raises an exception.
        """
        self.synchronize()
        var start = monotonic()
        for _ in range(num_iters):
            func(self)
        self.synchronize()
        return Int(monotonic() - start)

    @inline(.always)
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

        Raises:
            If synchronization fails or if the function raises an exception.
        """
        self.synchronize()
        var start = monotonic()
        for _ in range(num_iters):
            func()
        self.synchronize()
        return Int(monotonic() - start)

    @inline(.always)
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

        Raises:
            If synchronization fails or if the function raises an exception.
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

    def can_access(self, peer: DeviceContext) raises -> Bool:
        """Returns True if this device can access the identified peer device.

        Args:
            peer: The peer device context.

        Returns:
            False for self, a peer on another driver, or a build without
            peer access; otherwise the plugin's answer.

        Raises:
            If the plugin query fails.
        """
        # A foreign handle would be misread; CPU and GPU 0 also share id 0.
        if not (self._driver is peer._driver):
            return False
        if self._device[].id == peer._device[].id:
            return False
        comptime Ext = type_of(self._device[].extensions())
        comptime if conforms_to(Ext, SupportsPeerAccessV1):
            return self._device[].can_access_peer(peer._device[])
        else:
            return False

    def enable_peer_access(self, peer: DeviceContext) raises:
        """Enables direct memory access to the peer device.

        Args:
            peer: The peer device context whose memory becomes accessible.

        Raises:
            If the build or `peer`'s driver lacks peer access, or the
            plugin refuses the pair.
        """
        if not (self._driver is peer._driver):
            raise Error("Peer access not implemented")
        comptime Ext = type_of(self._device[].extensions())
        comptime if conforms_to(Ext, SupportsPeerAccessV1):
            if not self._device[].extensions().peer_access():
                raise Error("Peer access not implemented")
            self._context[].enable_peer_access(peer._context[])
            var src = Int(self._device[].id)
            var dst = Int(peer._device[].id)
            var state = _PEER_STATE.get_or_create_ptr()
            with BlockingScopedLock(state[].lock):
                state[].record_enabled(src, self._context, dst, peer._context)
        else:
            raise Error("Peer access not implemented")

    @staticmethod
    def enable_all_peer_access() raises:
        """Enables peer-to-peer access between all accelerators, both ways.

        A silent no-op without peer access or with one accelerator, as on the
        legacy backend; collective setup warns on a raise.

        Raises:
            On the first pair the plugin refuses.
        """
        comptime Ext = Device[Self.device_spec].Extensions
        comptime if conforms_to(Ext, SupportsPeerAccessV1):
            var maybe_driver = _GPU_DRIVER.get_or_create_ptr()
            if not maybe_driver[]:
                return
            var driver = maybe_driver[].value()
            var count = Int(driver[].get_device_count())
            if count <= 1:
                return
            var state = _PEER_STATE.get_or_create_ptr()
            if state[].all_enabled.load() != 0:
                return
            var contexts = List[ArcPointer[Context[Self.device_spec]]]()
            for i in range(count):
                # Matches `__init__`'s lookup, so the contexts are shared.
                var device = rebind[ArcPointer[Device[Self.device_spec]]](
                    driver[].get_device_dynamic[Optional[Int64](0)](Int64(i))
                )
                # A non-vendor `.so` may lack the group; stay a no-op.
                if not device[].extensions().peer_access():
                    return
                contexts.append(device[].get_context())
            # Concurrent callers may repeat a grant; the plugin treats an
            # already-enabled pair as success.
            for i in range(count):
                for j in range(count):
                    if i == j:
                        continue
                    contexts[i][].enable_peer_access(contexts[j][])
                    with BlockingScopedLock(state[].lock):
                        state[].record_enabled(i, contexts[i], j, contexts[j])

    @staticmethod
    def all_peer_access_enabled() raises -> Bool:
        """Returns whether peer access is enabled between all GPU pairs.

        Returns:
            False with fewer than two accelerators or without peer access.

        Raises:
            If the process-global peer state cannot be created.
        """
        return _PEER_STATE.get_or_create_ptr()[].all_enabled.load() != 0

    def supports_multicast(self) raises -> Bool:
        """Returns True if this device supports multicast memory mappings.

        Returns:
            False. Multicast is not implemented on the HAL backend.

        Raises:
            Never raises; declared `raises` to match the default
            `DeviceContext` API.
        """
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
        from max.gpu.host import DeviceContext

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

        Raises:
            If querying the device name from the driver fails.
        """
        return self._device[].get_name()

    def is_compatible(self) -> Bool:
        """Returns True if this device is compatible with MAX.

        This method checks whether the current device is compatible with the
        Modular Accelerated Execution (MAX) runtime. It's useful for validating
        that the device can execute the compiled code before attempting operations.

        Returns:
            True if the device is compatible with MAX, False otherwise.

        Example:

        ```mojo
        from max.gpu.host import DeviceContext

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
        from max.gpu.host import DeviceContext

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
        return self._device[].get_arch()

    def is_host_unified(self) raises -> Bool:
        """Returns True if this device and the host share one physical memory
        pool.

        Reports hardware topology, so it does not predict whether
        `DeviceBuffer.unsafe_host_ptr()` succeeds for any particular buffer.

        Returns:
            True if device and host memory are one physical pool.

        Raises:
            If there's an error querying the device.
        """
        return self._device[].is_host_unified()


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

    def __deinit__(deinit self):
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
    ret_type: AnyType,
    declared_arg_types: TypeList[Trait=AnyType, ...],
    //,
    func: def(*args: *declared_arg_types) thin -> ret_type,
    Wrapper: DeviceFunctionWrapper = _IdentityWrapper,
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
    from max.gpu.host import DeviceContext

    def my_kernel():
        # Kernel implementation
        pass

    with DeviceContext() as ctx:
        # Compile the kernel
        var kernel = ctx.compile_function[my_kernel]()
        # Enqueue the kernel for execution
        ctx.enqueue_function(kernel, grid_dim=1, block_dim=1)
    ```

    Parameters:
        ret_type: The return type of `func` (inferred).
        declared_arg_types: The argument types of `func` (inferred). Launch
            arguments are checked against these.
        func: The kernel to compile.
        Wrapper: Selects the function that is actually compiled for `func`,
            such as a generated entry point. Defaults to compiling `func` as
            written.
    """

    comptime _target = DeviceSpec.target[get_device_spec[0]()]()
    comptime _wrapped_func = Self.Wrapper.wrap[Self.func]

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var _ctx: DeviceContext
    var _inner: ArcPointer[_DeviceFunctionInner]
    # Null until the C++ runtime shim provides it; read by the CUDA/HIP
    # interop accessors (`_nvidia_cuda` / `_amdgpu_hip`).
    var _handle: _DeviceFunctionPtr[mut=True]
    # The compilation result for `Self._wrapped_func`: its assembly and
    # linkage name.
    var _func_info: CompiledFunctionInfo[
        type_of(Self._wrapped_func), Self._wrapped_func, Self._target
    ]

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @doc_hidden
    @inline(.always)
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
        var info = ctx._context[]._compile_inner[
            type_of(Self._wrapped_func), Self._wrapped_func
        ]()
        var attr_code = OptionalReg[Int32](None)
        var attr_value = Int32(0)
        if case .Some(func_attribute) = func_attribute:
            if (
                func_attribute.attribute
                == Attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES
            ):
                # Metal and HIP have no per-function cap; their plugins
                # reject it.
                if ctx.api() == "cuda":
                    attr_code = OptionalReg[Int32](
                        func_attribute.attribute.code
                    )
                    attr_value = Int32(func_attribute.value)
            else:
                raise Error(
                    "the function attribute '",
                    func_attribute.attribute,
                    "' is not currently supported",
                )
        # The context's function cache owns the loaded module: repeated
        # constructions of the same kernel reuse the cached module.
        var func_handle = ctx._context[].get_or_load_function(
            info.asm,
            info.function_name,
            attr_code=attr_code,
            attr_value=attr_value,
        )
        self._ctx = ctx
        self._handle = None
        self._func_info = info
        self._inner = ArcPointer(
            _DeviceFunctionInner(func_handle, ctx._context)
        )

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @inline(.always)
    @staticmethod
    def _check_trailing_host[Host: RegisterPassable, index: Int]():
        """Require `Host` to be the declared kernel argument at `index`."""
        comptime declared_host_type = Self.declared_arg_types[index]
        comptime assert Host == declared_host_type, String(
            "host argument of type '",
            reflect[Host].name(),
            "' does not match the declared function argument type '",
            reflect[declared_host_type].name(),
            "'",
        )

    @inline(.always)
    def _call_with_pack_checked[
        HostArgs: TypeList[Trait=RegisterPassable, ...],
        //,
        *Encoded: DevicePassable,
        ContextT: _HALFunctionEnqueuer,
    ](
        imm self,
        ctx: ContextT,
        *encoded_args: *Encoded,
        host_args: Tuple[*HostArgs] = (),
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        location: Optional[SourceLocation] = None,
    ) raises:
        # Launch `attributes` (e.g. PDL / programmatic stream serialization,
        # cluster dimensions) are forwarded to the plugin via the attribute
        # array assembled below. `constant_memory` mappings DO change
        # semantics and are not yet plumbed through, so refuse them rather
        # than silently dropping them.
        if case .Some(cluster_dim) = cluster_dim:
            attributes.append(LaunchAttribute.from_cluster_dim(cluster_dim))
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
        config.set_near_compute_scratchpad_usage(shared_mem_bytes.or_else(0))

        comptime for idx in range(HostArgs.length):
            Self._check_trailing_host[HostArgs[idx], Encoded.length + idx]()

        self._do_call_with_pack_checked[*Encoded, ContextT=ContextT](
            ctx,
            config,
            *encoded_args,
            host_args=host_args,
            attributes=attributes^,
            location=location,
        )

    @inline(.always)
    def _do_call_with_pack_checked[
        ExecutionConfigType: GridBlockExecutionConfig,
        HostArgs: TypeList[Trait=RegisterPassable, ...],
        //,
        *Ts: DevicePassable,
        ContextT: _HALFunctionEnqueuer,
    ](
        imm self,
        ctx: ContextT,
        mut execution_config: ExecutionConfigType,
        *args: *Ts,
        host_args: Tuple[*HostArgs] = (),
        var attributes: List[LaunchAttribute] = [],
        location: Optional[SourceLocation] = None,
    ) raises:
        comptime extra_host_count = HostArgs.length

        _check_device_context_hal_only_supported_exec_config(execution_config)

        var shared_mem_bytes: Int = 0

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

        comptime num_encoded = Ts.length
        comptime num_passed_args = num_encoded + extra_host_count
        var validated_args = _launch_args._validate_encodable_arguments[
            Self._target,
            DeclaredArgTypes=Self.declared_arg_types,
            ActualArgTypes=Ts,
            host_args_count=extra_host_count,
        ]()

        var num_translated_args = validated_args[0]
        comptime for i in range(extra_host_count):
            comptime size = size_of[HostArgs[i]]()
            # Skip zero-sized types (ZSTs).
            comptime if size != 0:
                num_translated_args += 1

        var translated_arg_offsets = validated_args[1].copy()

        def calculate_args_size() -> Int:
            var tmp_args_size = 8  # reserve 8 extra bytes for alignment
            comptime for i in range(num_encoded):
                comptime actual_arg_type = Ts[i]
                tmp_args_size += align_up(
                    size_of[actual_arg_type.device_type](), 8
                )
            return tmp_args_size

        comptime args_size = calculate_args_size()

        var translated_args = Array[Byte, args_size](uninitialized=True)
        var start_addr = Int(translated_args.unsafe_ptr())
        var extra_align = align_up(start_addr, 8) - start_addr

        # Allocation made at function scope to outlive the enqueue, otherwise
        # the stack bytes are free to be reused immediately. A kernel may take
        # no arguments; keep the arrays non-empty.
        var stack_addrs = Array[_, max(num_passed_args, 1)](
            fill=OpaquePointer[MutAnyOrigin].unsafe_dangling()
        )
        var stack_sizes = Array[UInt64, max(num_passed_args, 1)](fill=0)
        var dense_args_addrs = stack_addrs.unsafe_ptr().unsafe_origin_cast[
            MutUntrackedOrigin
        ]()
        var dense_args_sizes = stack_sizes.unsafe_ptr().unsafe_origin_cast[
            MutUntrackedOrigin
        ]()

        var translated_arg_idx = 0

        var device_type_encoder = DefaultDeviceTypeEncoder()

        comptime for i in range(num_encoded):
            var translated_arg_offset = translated_arg_offsets[i]
            if translated_arg_offset >= 0:
                comptime actual_arg_type = Ts[i]
                var first_word_addr = Pointer(
                    to=translated_args.unsafe_ptr()[
                        unsafe_offset=translated_arg_offset + extra_align
                    ]
                ).unsafe_bitcast[NoneType]()
                args[i]._to_device_type(device_type_encoder, first_word_addr)
                dense_args_addrs[
                    unsafe_offset=translated_arg_idx
                ] = first_word_addr.as_unsafe_any_origin()
                dense_args_sizes[unsafe_offset=translated_arg_idx] = UInt64(
                    size_of[
                        actual_arg_type.device_type,
                        target=device_type_encoder.target(),
                    ]()
                )
                translated_arg_idx += 1

        comptime for i in range(extra_host_count):
            comptime size = size_of[HostArgs[i]]()
            # Skip zero-sized types (ZSTs).
            if size == 0:
                continue

            var addr = (
                Pointer(to=host_args[i])
                .unsafe_bitcast[NoneType]()
                .unsafe_mut_cast[True]()
                .as_unsafe_any_origin()
            )

            dense_args_addrs[unsafe_offset=translated_arg_idx] = addr
            dense_args_sizes[unsafe_offset=translated_arg_idx] = UInt64(size)
            translated_arg_idx += 1

        # Kernels that use `with PDL()` emit `griddepcontrol` instructions and
        # require the launch to be configured with the matching attribute;
        # dropping it faults the launch.
        var attr_ptr = OptionalReg[OpaquePointer[MutUntrackedOrigin]](None)
        if len(attributes) > 0:
            attr_ptr = OptionalReg(
                attributes.unsafe_ptr()
                .unsafe_bitcast[NoneType]()
                .unsafe_origin_cast[MutUntrackedOrigin]()
            )

        ctx._hal_stream()[].execute(
            self._inner[]._func_handle,
            execution_config,
            args=rebind[
                Pointer[OpaquePointer[MutUntrackedOrigin], MutAnyOrigin]
            ](dense_args_addrs),
            arg_sizes=rebind[Pointer[UInt64, MutAnyOrigin]](dense_args_sizes),
            num_args=UInt32(num_translated_args),
            attributes=attr_ptr,
            num_attributes=UInt32(len(attributes)),
        )

        # Keep these alive until the plugin has read them.
        _ = attributes^
        _ = translated_args^
        _ = stack_addrs^
        _ = stack_sizes^

    @inline(.always)
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
        location: Optional[SourceLocation] = None,
    ) raises:
        # Launch `attributes` (e.g. PDL / programmatic stream serialization,
        # cluster dimensions) are forwarded to the plugin via the attribute
        # array assembled below. `constant_memory` mappings DO change
        # semantics and are not yet plumbed through, so refuse them rather
        # than silently dropping them.
        if case .Some(cluster_dim) = cluster_dim:
            attributes.append(LaunchAttribute.from_cluster_dim(cluster_dim))
        if constant_memory:
            raise Error(
                "HAL DeviceContext.enqueue_function does not support"
                " `constant_memory` mappings."
            )

        comptime num_args = Ts.length
        # Function scope so the storage outlives the enqueue; see
        # `_call_with_pack_checked`.
        var stack_addrs = Array[_, max(num_args, 1)](
            fill=OpaquePointer[MutAnyOrigin].unsafe_dangling()
        )
        var stack_sizes = Array[UInt64, max(num_args, 1)](fill=0)
        var dense_args_addrs = stack_addrs.unsafe_ptr().unsafe_origin_cast[
            MutUntrackedOrigin
        ]()
        var dense_args_sizes = stack_sizes.unsafe_ptr().unsafe_origin_cast[
            MutUntrackedOrigin
        ]()

        comptime for i in range(num_args):
            dense_args_addrs[unsafe_offset=i] = (
                Pointer(to=args[i])
                .unsafe_bitcast[NoneType]()
                .unsafe_mut_cast[True]()
                .as_unsafe_any_origin()
            )

        def _populate_arg_sizes[i: Int]() {imm}:
            dense_args_sizes[unsafe_offset=i] = UInt64(size_of[Ts[i]]())

        comptime for i in range(num_args):
            _populate_arg_sizes[i]()

        # Kernels that use `with PDL()` emit `griddepcontrol` instructions and
        # require the launch to be configured with the matching attribute;
        # dropping it faults the launch.
        var attr_ptr = OptionalReg[OpaquePointer[MutUntrackedOrigin]](None)
        if len(attributes) > 0:
            attr_ptr = OptionalReg(
                attributes.unsafe_ptr()
                .unsafe_bitcast[NoneType]()
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
                Pointer[OpaquePointer[MutUntrackedOrigin], MutAnyOrigin]
            ](dense_args_addrs),
            arg_sizes=rebind[Pointer[UInt64, MutAnyOrigin]](dense_args_sizes),
            num_args=UInt32(num_args),
            shared_mem_bytes=UInt32(shared_mem_bytes.or_else(0)),
            attributes=attr_ptr,
            num_attributes=UInt32(len(attributes)),
        )

        # Keep attributes alive until the plugin has read them.
        _ = attributes^
        _ = stack_addrs^
        _ = stack_sizes^

    @inline(.always)
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
        location: Optional[SourceLocation] = None,
    ) raises:
        _check_device_context_hal_only_supported_exec_config(execution_config)

        comptime num_args = Ts.length
        # Function scope so the storage outlives the enqueue; see
        # `_call_with_pack_checked`.
        var stack_addrs = Array[_, max(num_args, 1)](
            fill=OpaquePointer[MutAnyOrigin].unsafe_dangling()
        )
        var stack_sizes = Array[UInt64, max(num_args, 1)](fill=0)
        var dense_args_addrs = stack_addrs.unsafe_ptr().unsafe_origin_cast[
            MutUntrackedOrigin
        ]()
        var dense_args_sizes = stack_sizes.unsafe_ptr().unsafe_origin_cast[
            MutUntrackedOrigin
        ]()

        comptime for i in range(num_args):
            dense_args_addrs[unsafe_offset=i] = (
                Pointer(to=args[i])
                .unsafe_bitcast[NoneType]()
                .unsafe_mut_cast[True]()
                .as_unsafe_any_origin()
            )

        def _populate_arg_sizes[i: Int]() {imm}:
            dense_args_sizes[unsafe_offset=i] = UInt64(size_of[Ts[i]]())

        comptime for i in range(num_args):
            _populate_arg_sizes[i]()

        ctx._hal_stream()[].execute(
            self._inner[]._func_handle,
            execution_config,
            args=rebind[
                Pointer[OpaquePointer[MutUntrackedOrigin], MutAnyOrigin]
            ](dense_args_addrs),
            arg_sizes=rebind[Pointer[UInt64, MutAnyOrigin]](dense_args_sizes),
            num_args=UInt32(num_args),
        )
        _ = stack_addrs^
        _ = stack_sizes^

    @inline(.always)
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

    Borrows the handle from the context's loaded-function cache, which owns it
    and its `RuntimeBundle`, so there is nothing to unload here. Distinct from
    `_DeviceFunctionInner` because external functions don't carry
    `CompiledFunctionInfo`.
    """

    var _func_handle: FunctionHandle


struct DeviceExternalFunction(ImplicitlyCopyable, Movable):
    """Represents an external device function loaded from PTX/SASS assembly.

    This class provides functionality to load and execute pre-compiled GPU functions
    from assembly code rather than compiling them from Mojo source. This is useful
    for integrating with existing CUDA/HIP code or for using specialized assembly
    optimizations.
    """

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var _ctx: DeviceContext
    var _inner: ArcPointer[_ExternalFunctionInner]

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

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
        var func_handle = ctx._context[].get_or_load_function_dynamic(
            asm, function_name
        )
        self._ctx = ctx
        self._inner = ArcPointer(_ExternalFunctionInner(func_handle))

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @inline(.always)
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
        location: Optional[SourceLocation] = None,
    ) raises:
        # Launch `attributes` (e.g. PDL / programmatic stream serialization,
        # cluster dimensions) are forwarded to the plugin via the attribute
        # array assembled below. `constant_memory` mappings DO change
        # semantics and are not yet plumbed through, so refuse them rather
        # than silently dropping them.
        if case .Some(cluster_dim) = cluster_dim:
            attributes.append(LaunchAttribute.from_cluster_dim(cluster_dim))
        if constant_memory:
            raise Error(
                "HAL DeviceContext.enqueue_function does not support"
                " `constant_memory` mappings."
            )

        comptime num_args = Ts.length
        var dense_args_addrs = unsafe_stack_allocation[
            num_args + 1, OpaquePointer[MutAnyOrigin]
        ]()
        var dense_args_sizes = unsafe_stack_allocation[num_args + 1, UInt64]()

        comptime for i in range(num_args):
            dense_args_addrs[unsafe_offset=i] = (
                Pointer(to=args[i])
                .unsafe_bitcast[NoneType]()
                .unsafe_mut_cast[True]()
                .as_unsafe_any_origin()
            )

        def _populate_arg_sizes[i: Int]() {imm}:
            dense_args_sizes[unsafe_offset=i] = UInt64(size_of[Ts[i]]())

        comptime for i in range(num_args):
            _populate_arg_sizes[i]()

        # Kernels that use `with PDL()` emit `griddepcontrol` instructions
        # and require the launch to be configured with the matching attribute;
        # dropping it faults the launch.
        var attr_ptr = OptionalReg[OpaquePointer[MutUntrackedOrigin]](None)
        if len(attributes) > 0:
            attr_ptr = OptionalReg(
                attributes.unsafe_ptr()
                .unsafe_bitcast[NoneType]()
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
                Pointer[OpaquePointer[MutUntrackedOrigin], MutAnyOrigin]
            ](dense_args_addrs),
            arg_sizes=rebind[Pointer[UInt64, MutAnyOrigin]](dense_args_sizes),
            num_args=UInt32(num_args),
            shared_mem_bytes=UInt32(shared_mem_bytes.or_else(0)),
            attributes=attr_ptr,
            num_attributes=UInt32(len(attributes)),
        )

    @inline(.always)
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
        location: Optional[SourceLocation] = None,
    ) raises:
        _check_device_context_hal_only_supported_exec_config(execution_config)

        comptime num_args = Ts.length
        var dense_args_addrs = unsafe_stack_allocation[
            num_args + 1, OpaquePointer[MutAnyOrigin]
        ]()
        var dense_args_sizes = unsafe_stack_allocation[num_args + 1, UInt64]()

        comptime for i in range(num_args):
            dense_args_addrs[unsafe_offset=i] = (
                Pointer(to=args[i])
                .unsafe_bitcast[NoneType]()
                .unsafe_mut_cast[True]()
                .as_unsafe_any_origin()
            )

        def _populate_arg_sizes[i: Int]() {imm}:
            dense_args_sizes[unsafe_offset=i] = UInt64(size_of[Ts[i]]())

        comptime for i in range(num_args):
            _populate_arg_sizes[i]()

        ctx._hal_stream()[].execute(
            self._inner[]._func_handle,
            execution_config,
            args=rebind[
                Pointer[OpaquePointer[MutUntrackedOrigin], MutAnyOrigin]
            ](dense_args_addrs),
            arg_sizes=rebind[Pointer[UInt64, MutAnyOrigin]](dense_args_sizes),
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
    from max.gpu.host import DeviceContext, DeviceStream
    var ctx = DeviceContext(0)  # Select first GPU
    var stream = DeviceStream(ctx)

    # Launch operations on the stream
    # ...

    # Wait for all operations in the stream to complete
    stream.synchronize()
    ```
    """

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var _ctx: DeviceContext
    var _stream: ArcPointer[Stream[get_device_spec[0]()]]
    # Null until the C++ runtime shim provides it; read by the CUDA/HIP
    # interop accessors (`_nvidia_cuda` / `_amdgpu_hip`).
    var _handle: _DeviceStreamPtr[mut=True]

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

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

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    def synchronize(self) raises:
        """Blocks the calling CPU thread until all operations in this stream complete.

        This function waits until all previously issued commands in this stream
        have completed execution. It provides a synchronization point between
        host and device code.

        Raises:
            If synchronization fails.

        Example:

        ```mojo
        from max.gpu.host import DeviceContext

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
        from max.gpu.host import DeviceContext

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

    @inline(.always)
    def enqueue_function[
        declared_arg_types: TypeList[Trait=AnyType, ...],
        //,
        func: def(*args: *declared_arg_types) thin -> None,
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
        location: Optional[SourceLocation] = None,
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

    @inline(.always)
    def enqueue_function[
        declared_arg_types: TypeList[Trait=AnyType, ...],
        ExecutionConfigType: GridBlockExecutionConfig,
        //,
        func: def(*args: *declared_arg_types) thin -> None,
        *actual_arg_types: DevicePassable,
    ](
        self,
        mut execution_config: ExecutionConfigType,
        *args: *actual_arg_types,
        location: Optional[SourceLocation] = None,
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

    @inline(.always)
    def enqueue_function[
        FuncType: ImplicitlyCopyable & RegisterPassable & def() -> None,
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
        location: Optional[SourceLocation] = None,
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
        var gpu_kernel = DeviceFunction[FuncType.__call__](self._ctx)
        gpu_kernel._call_with_pack_checked(
            self,
            host_args=(func,),
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
            location=location.or_else(call_location()),
        )

    @inline(.always)
    def enqueue_function[
        ExecutionConfigType: GridBlockExecutionConfig,
        FuncType: ImplicitlyCopyable & RegisterPassable & def() -> None,
        //,
    ](
        self,
        mut execution_config: ExecutionConfigType,
        func: FuncType,
        *,
        location: Optional[SourceLocation] = None,
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
        var grid_dim = execution_config.get_grid_dim()
        var block_dim = execution_config.get_block_dim()

        _check_dim["DeviceStream.enqueue_function", "grid_dim"](
            grid_dim, location=call_location()
        )
        _check_dim["DeviceStream.enqueue_function", "block_dim"](
            block_dim, location=call_location()
        )
        var gpu_kernel = DeviceFunction[FuncType.__call__](self._ctx)
        gpu_kernel._do_call_with_pack_checked(
            self,
            execution_config,
            host_args=(func,),
            location=location.or_else(call_location()),
        )

    @inline(.always)
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
        location: Optional[SourceLocation] = None,
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

    @inline(.always)
    def enqueue_function[
        ExecutionConfigType: GridBlockExecutionConfig,
        //,
        *Ts: DevicePassable,
    ](
        self,
        mut execution_config: ExecutionConfigType,
        f: DeviceFunction,
        *args: *Ts,
        location: Optional[SourceLocation] = None,
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

        f._do_call_with_pack_checked(
            self,
            execution_config,
            *args,
            location=location.or_else(call_location()),
        )

    @inline(.always)
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
        location: Optional[SourceLocation] = None,
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

    @inline(.always)
    def enqueue_function[
        ExecutionConfigType: GridBlockExecutionConfig,
        //,
        *Ts: AnyType,
    ](
        self,
        mut execution_config: ExecutionConfigType,
        f: DeviceExternalFunction,
        *args: *Ts,
        location: Optional[SourceLocation] = None,
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
    from max.gpu.host import DeviceContext

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

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    # `_event` is an `ArcPointer[Optional[...]]` so that all copies of a
    # DeviceEvent  share the same backing event, which matches the behavior
    # of the existing reference-counted DeviceEvent implementation.
    var _ctx: DeviceContext
    var _event: ArcPointer[Optional[Event[EVENT_FLAG_CPU_VISIBLE]]]

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

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

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

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
    # Same, for `DeviceBuffer` views over a `HostBuffer`'s pinned allocation,
    # which the HAL C exports create for the C++ `DeviceContext`.
    var _parent_host: List[ArcPointer[_HostBufferInner]]

    def __deinit__(deinit self):
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
    [`enqueue_create_buffer()`](/docs/max/gpu/host/device_context/DeviceContext/#enqueue_create_buffer).

    Parameters:
        dtype: Data dtype to be stored in the buffer.
    """

    # ===-------------------------------------------------------------------===#
    # Aliases
    # ===-------------------------------------------------------------------===#

    comptime _DevicePtr = Pointer[Scalar[Self.dtype], MutUntrackedOrigin]

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    # `_device_ptr` must be the first member in the struct. Kernel dispatch code
    # `rebind`s a `DeviceBuffer` directly to its `device_type` (an
    # `Pointer`), which reinterprets the struct's initial data as the
    # device address.
    var _device_ptr: Self._DevicePtr
    # Wrap the inner buffer in an ArcPointer so copies of this DeviceBuffer hold
    # a reference to the same underlying buffer.
    var _ctx: DeviceContext
    var _inner: ArcPointer[_HALBufferInner]

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

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
            _HALBufferInner(buffer^, ctx._context, addr, [], [])
        )

    @doc_hidden
    def __init__(
        out self,
        ctx: DeviceContext,
        ptr: Pointer[Scalar[Self.dtype], ...],
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
                _HALBufferInner(buffer^, ctx._context, addr, [], [])
            )
        except e:
            abort(String(t"DeviceBuffer: failed to wrap external memory: {e}"))

    @staticmethod
    @doc_hidden
    def empty(context: DeviceContext) -> Self:
        return Self(
            context,
            Self._DevicePtr.unsafe_dangling(),
            0,
            owning=False,
        )

    # ===-------------------------------------------------------------------===#
    # Trait: DevicePassable
    # ===-------------------------------------------------------------------===#

    # Implementation of `DevicePassable`
    comptime device_type: AnyType = Pointer[
        mut=True, Scalar[Self.dtype], AnyOrigin[mut=True]
    ]
    """`DeviceBuffer` dtypes are remapped to `Pointer` when passed to
    accelerator devices."""

    def _to_device_type(
        self,
        mut encoder: Some[DeviceTypeEncoder],
        target: MutOpaquePointer[_],
    ):
        """Device dtype mapping from `DeviceBuffer` to the device's
        `Pointer`.
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

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    def __len__(self) -> Int:
        """Returns the number of elements in this buffer.

        This method calculates the number of elements by dividing the total byte size
        of the buffer by the size of each element.

        Returns:
            The number of elements in the buffer.
        """
        return Int(self._inner[]._buffer.byte_size) // size_of[Self.dtype]()

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

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

    @doc_hidden
    @inline(.always)
    def take_handle(var self) -> _DeviceBufferPtr[mut=True]:
        """Transfers this buffer to the runtime as a single owning handle.

        The buffer is boxed and wrapped in a C++ shim `DeviceBuffer` with
        refcount 1, the same contract as the legacy handle. The shim frees the
        box when its last runtime reference drops.
        """
        var ptr = self._device_ptr
        var byte_size = Int(self._inner[]._buffer.byte_size)
        var box = alloc[DeviceBuffer[Self.dtype]]({length = 1}).unsafe_leak()
        box.unsafe_write(self^)
        return external_call[
            "AsyncRT_DeviceBuffer_adoptMojoBox", _DeviceBufferPtr[mut=True]
        ](box, ptr, byte_size)

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

    @inline(.always)
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
            Pointer[Scalar[view_type], MutUntrackedOrigin](
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
    ) -> Pointer[Scalar[Self.dtype], MutUntrackedOrigin]:
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
        global_dim: Pointer[mut=False, Int64, _],
        global_strides: Pointer[mut=False, Int64, _],
        box_dim: Pointer[mut=False, Int32, _],
        element_strides: Pointer[mut=False, Int32, _],
        interleave: Int32,
        swizzle: Int32,
        l2_promotion: Int32,
        oob_fill: Int32,
    ) raises:
        """Encodes a tiled TMA descriptor for this buffer via the HAL plugin.
        Used by `max.gpu.host._legacy._tensormap.create_tensormap`."""
        # Same-width reinterpret of the callers' signed arrays to the C ABI's
        # unsigned element types.
        self._ctx._context[].tensor_map_encode_tiled(
            tensor_map,
            data_type,
            rank,
            self._inner[]._device_addr,
            global_dim.unsafe_bitcast[UInt64](),
            global_strides.unsafe_bitcast[UInt64](),
            box_dim.unsafe_bitcast[UInt32](),
            element_strides.unsafe_bitcast[UInt32](),
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
        global_dim: Pointer[mut=False, Int64, _],
        global_strides: Pointer[mut=False, Int64, _],
        pixel_box_lower_corner: Pointer[mut=False, Int32, _],
        pixel_box_upper_corner: Pointer[mut=False, Int32, _],
        channels_per_pixel: Int32,
        pixels_per_column: Int32,
        element_strides: Pointer[mut=False, Int32, _],
        interleave: Int32,
        swizzle: Int32,
        l2_promotion: Int32,
        oob_fill: Int32,
    ) raises:
        """Encodes an im2col TMA descriptor for this buffer via the HAL
        plugin. Used by `max.gpu.host._legacy._tensormap.create_tensormap_im2col`.
        """
        self._ctx._context[].tensor_map_encode_im2col(
            tensor_map,
            data_type,
            rank,
            self._inner[]._device_addr,
            global_dim.unsafe_bitcast[UInt64](),
            global_strides.unsafe_bitcast[UInt64](),
            pixel_box_lower_corner,
            pixel_box_upper_corner,
            channels_per_pixel,
            pixels_per_column,
            element_strides.unsafe_bitcast[UInt32](),
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

    def enqueue_copy_from(self, src_ptr: Pointer[Scalar[Self.dtype], _]) raises:
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
        from max.gpu.host import DeviceContext

        var ctx = DeviceContext()
        var length = 1024
        var in_dev = ctx.enqueue_create_buffer[.float32](length)
        var out_dev = ctx.enqueue_create_buffer[.float32](length)

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
    var _host_ptr: Pointer[UInt8, MutUntrackedOrigin]
    # For sub-buffer views: keeps the parent allocation alive for the view's
    # lifetime. A view's `_buffer` is a non-owning wrapper over the parent's
    # pinned memory, so it is released with `free_sync`, not freed.
    var _parent: List[ArcPointer[_HostBufferInner]]

    def __deinit__(deinit self):
        if self._parent:
            try:
                self._context[].free_sync(self._buffer^)
            except e:
                print("warning: free_sync failed:", e)
            # Release the parent only after the view's wrapper is freed.
            _ = self._parent^
            return
        try:
            self._context[].free_host_pinned(self._buffer^)
        except e:
            print("warning: free_host_pinned failed:", e)


struct HostBuffer[dtype: DType](ImplicitlyCopyable, Movable, Sized):
    """Represents a block of host-resident storage. For GPU devices, a host
    buffer is allocated in the host's global memory.

    To allocate a `HostBuffer`, use one of the methods provided by
    `DeviceContext`, such as
    [`enqueue_create_host_buffer()`](/docs/max/gpu/host/device_context/DeviceContext/#enqueue_create_host_buffer).

    Parameters:
        dtype: Data type to be stored in the buffer.
    """

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var _ctx: DeviceContext
    var _inner: ArcPointer[_HostBufferInner]

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @doc_hidden
    def __init__(out self, ctx: DeviceContext, size: Int) raises:
        """This init takes in a constructed `DeviceContext` and schedules an
        owned buffer allocation using the stream in the device context.
        """
        var byte_size = UInt64(size * size_of[Self.dtype]())
        var buffer = ctx._context[].alloc_host_pinned(byte_size)
        var addr = UInt64(0)
        var host_ptr: Pointer[UInt8, MutUntrackedOrigin]
        if byte_size > 0:
            host_ptr = ctx._context[].memory_get_host_address[
                MutUntrackedOrigin
            ](buffer)
        else:
            host_ptr = Pointer[UInt8, MutUntrackedOrigin](
                unsafe_from_address=Int(addr)
            )
        self._ctx = ctx
        self._inner = ArcPointer(
            _HostBufferInner(buffer^, ctx._context, host_ptr, [])
        )

    @doc_hidden
    def __init__(
        out self, ctx: DeviceContext, var inner: ArcPointer[_HostBufferInner]
    ):
        self._ctx = ctx
        self._inner = inner^

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    def __getitem__(self, idx: Int) -> Scalar[Self.dtype]:
        """Retrieves the element at the specified index from the host buffer.

        This operator allows direct access to individual elements in the host buffer
        using array indexing syntax.

        Args:
            idx: The index of the element to retrieve.

        Returns:
            The scalar value at the specified index.
        """
        return self.unsafe_ptr()[unsafe_offset=idx]

    def __setitem__(self, idx: Int, val: Scalar[Self.dtype]):
        """Sets the element at the specified index in the host buffer.

        This operator allows direct modification of individual elements in the host buffer
        using array indexing syntax.

        Args:
            idx: The index of the element to modify.
            val: The new value to store at the specified index.
        """
        self.unsafe_ptr()[unsafe_offset=idx] = val

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    def __len__(self) -> Int:
        """Returns the number of elements in this buffer.

        This method calculates the number of elements by dividing the total byte size
        of the buffer by the size of each element.

        Returns:
            The number of elements in the buffer.
        """
        return Int(self._inner[]._buffer.byte_size) // size_of[Self.dtype]()

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

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
    ) -> Pointer[Scalar[Self.dtype], MutUntrackedOrigin]:
        """Returns the raw device pointer without transferring ownership.

        This method provides direct access to the underlying device pointer
        for advanced use cases. The buffer retains ownership of the pointer.

        Returns:
            The raw device pointer owned by this buffer.
        """
        return self._inner[]._host_ptr.unsafe_bitcast[Scalar[Self.dtype]]()

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

    def create_sub_buffer[
        view_type: DType
    ](self, offset: Int, size: Int) raises -> HostBuffer[view_type]:
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
            A new HostBuffer referencing the specified region with the specified element dtype.

        Raises:
            If the operation fails.
        """
        comptime assert not is_gpu(), "HostBuffer is not supported on GPUs"
        comptime elem_size = size_of[view_type]()
        # Check in elements so a huge `offset` or `size` can't wrap the byte
        # arithmetic back into range.
        var max_elems = Int(self._inner[]._buffer.byte_size) // elem_size
        if (
            offset < 0
            or size < 0
            or size > max_elems
            or offset > max_elems - size
        ):
            raise Error("create_sub_buffer: view is out of range")
        var byte_offset = offset * elem_size
        var byte_size = size * elem_size
        var host_ptr = self._inner[]._host_ptr.unsafe_offset(byte_offset)
        var addr = UInt64(Int(host_ptr))
        # The view's buffer is a non-owning wrapper over the parent's pinned
        # memory. The plugin rejects wrapping a null address, so an empty
        # parent's view uses a zero-byte allocation instead, as `DeviceBuffer`
        # does; both are released with `free_sync`.
        var context = self._inner[]._context
        var buffer = context[].alloc_sync(
            0
        ) if addr == 0 else context[].wrap_memory(addr, UInt64(byte_size))
        return HostBuffer[view_type](
            self._ctx,
            ArcPointer(
                _HostBufferInner(buffer^, context, host_ptr, [self._inner])
            ),
        )

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

    def __deinit__(deinit self):
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
] = OptionalPointer[_DeviceContextCpp, origin]

comptime _DeviceBufferPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = OptionalPointer[_DeviceBufferCpp, origin]

comptime _DeviceFunctionPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = OptionalPointer[_DeviceFunctionCpp, origin]

comptime _DeviceMulticastBufferPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = OptionalPointer[_DeviceMulticastBufferCpp, origin]

comptime _DeviceStreamPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = OptionalPointer[_DeviceStreamCpp, origin]

comptime _DeviceEventPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = OptionalPointer[_DeviceEventCpp, origin]

comptime _DeviceTimerPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = OptionalPointer[_DeviceTimerCpp, origin]

comptime _CompletionFlagPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = OptionalPointer[_CompletionFlagCpp, origin]

comptime _DeviceContextScopePtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = OptionalPointer[_DeviceContextScopeCpp, origin]

comptime _DeviceGraphBuilderPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = OptionalPointer[_DeviceGraphBuilderCpp, origin]

comptime _DeviceGraphPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = OptionalPointer[_DeviceGraphCpp, origin]

comptime _CString[origin: ImmOrigin = ImmUntrackedOrigin] = Optional[
    CStringSpan[origin]
]

comptime _DumpPath = Variant[Bool, Path, StaticString, def() thin -> Path]


def _string_from_owned_charptr(c_str: _CString) -> String:
    var result = String()
    if case .Some(c_str) = c_str:
        result = String(unsafe_from_utf8_ptr=c_str.ptr())
    # void AsyncRT_DeviceContext_strfree(const char* ptr)
    external_call["AsyncRT_DeviceContext_strfree", NoneType](c_str)
    return result^


@inline(.never)
def _raise_checked_impl(
    err_msg: _CString, msg: String, location: SourceLocation
) raises:
    var err = _string_from_owned_charptr(err_msg)
    raise Error(location.prefix(err + ((" " + msg) if msg else "")))


def _check_copy_src_size(
    dst: Buffer[get_device_spec[0]()], src: Buffer[get_device_spec[0]()]
) raises:
    """Raises if `src` holds fewer bytes than a copy sized by `dst` reads."""
    if src.byte_size < dst.byte_size:
        raise Error(
            String(
                t"enqueue_copy: src is {src.byte_size} bytes but dst needs"
                t" {dst.byte_size} bytes"
            )
        )


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
        return UInt64(Int(bitcast[.uint8, 1](val)))
    elif bitwidth == 16:
        return UInt64(Int(bitcast[.uint16, 1](val)))
    elif bitwidth == 32:
        return UInt64(bitcast[.uint32, 1](val))
    else:
        return bitcast[.uint64, 1](val)


# ===-----------------------------------------------------------------------===#
# DeviceFunction
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct DefaultDeviceTypeEncoder(DeviceTypeEncoder):
    """Provides a default implementation of the `DeviceTypeEncoder` trait."""

    comptime _raw_mlir_target = _device_type_encoder_target()

    def encode_device_ptr[
        DevicePointerType: DevicePointerLike
    ](mut self, value: DevicePointerType, dst: MutOpaquePointer[_],):
        """Encodes a device pointer into `dst` as its raw pointer.

        Parameters:
            DevicePointerType: The type of the device pointer.

        Args:
            value: The device pointer to encode.
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
    DevicePointerLike,
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
    into an `Pointer` on the device at the point of being handed over to
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

    comptime PointeeType: AnyType = Scalar[Self.dtype]
    """DevicePointerLike encoded pointer pointee type."""

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
    ) -> Pointer[Scalar[Self.dtype], MutAnyOrigin]:
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
            self._buffer[]
            .unsafe_ptr()
            .unsafe_offset(self._offset)
            .as_unsafe_any_origin()
        )

    # ===------------------------------------------------------------------=== #
    # Origin and mutability casts
    # ===------------------------------------------------------------------=== #

    # `origin` appears only in the `_buffer` field's type parameters, never in
    # any field's layout, so the casts below reinterpret the whole handle, as
    # `Span.as_imm` does.
    comptime _OriginCastType[
        target_mut: Bool, //, target_origin: Origin[mut=target_mut]
    ] = DevicePointer[Self.dtype, target_origin]

    @always_inline("builtin")
    def unsafe_mut_cast[
        target_mut: Bool
    ](self) -> Self._OriginCastType[Self.origin.unsafe_mut_cast[target_mut]()]:
        """Changes the mutability of the borrow of the `DeviceBuffer`.

        To unconditionally drop mutability use `as_imm`.

        Parameters:
            target_mut: Mutability of the resulting `DevicePointer`.

        Returns:
            A `DevicePointer` referencing the same `DeviceBuffer` at the same
            offset, but with the newly specified mutability.

        Safety:
            Casting an immutable borrow to mutable claims mutation rights the
            caller does not hold, which defeats exclusivity checking: mutating
            the `DeviceBuffer` through the result while another borrow of it is
            live is undefined behavior. Prefer binding the mutability at the
            function signature level, for example taking a
            `DevicePointer[mut=True, dtype, _]` argument over an unbound
            `DevicePointer[dtype, _]`.
        """
        return rebind[
            Self._OriginCastType[Self.origin.unsafe_mut_cast[target_mut]()]
        ](self)

    @always_inline("builtin")
    def unsafe_origin_cast[
        target_origin: Origin[mut=Self.mut]
    ](self) -> Self._OriginCastType[target_origin]:
        """Changes the origin of the borrow of the `DeviceBuffer`.

        To unconditionally discard the origin use `as_unsafe_any_origin`.

        Parameters:
            target_origin: Origin of the resulting `DevicePointer`.

        Returns:
            A `DevicePointer` referencing the same `DeviceBuffer` at the same
            offset, but with the newly specified origin.

        Safety:
            The result names `target_origin` rather than the `DeviceBuffer` it
            refers into, so the lifetime checker no longer keeps that buffer
            alive; using the result once the buffer is destroyed is undefined
            behavior. Prefer parameterizing the origin at the function level
            over casting it.
        """
        return rebind[Self._OriginCastType[target_origin]](self)

    @always_inline("builtin")
    def as_imm(self) -> Self._OriginCastType[ImmOrigin(Self.origin)]:
        """Changes the borrow of the `DeviceBuffer` to immutable.

        Unlike `unsafe_mut_cast` this is always safe: dropping mutability
        cannot introduce aliasing.

        Returns:
            A `DevicePointer` referencing the same `DeviceBuffer` at the same
            offset, but with an immutable borrow.
        """
        return self.unsafe_mut_cast[False]()

    @always_inline("builtin")
    def as_unsafe_any_origin(
        self,
    ) -> Self._OriginCastType[UnsafeAnyOrigin[mut=Self.mut]]:
        """Discards the origin of the borrow of the `DeviceBuffer`.

        Returns:
            A `DevicePointer` with the origin set to `UnsafeAnyOrigin`.

        Safety:
            `UnsafeAnyOrigin` might alias any live value, which forces the
            lifetime checker into its most conservative behavior: it extends
            unrelated lifetimes and turns off exclusivity checking. The caller
            takes over keeping the `DeviceBuffer` alive. A concrete origin is
            always preferable.
        """
        return self.unsafe_origin_cast[UnsafeAnyOrigin[mut=Self.mut]]()

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

    comptime device_type: AnyType = Pointer[
        mut=True, Scalar[Self.dtype], AnyOrigin[mut=True]
    ]
    """`DevicePointer` is remapped to `Pointer` when passed to
    accelerator devices."""

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        """Device type mapping from `DevicePointer` to the device's
        `Pointer`.
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
# The HAL `DeviceContext` targets basic device execution. The following
# types exist so the `gpu.host` package surface matches the AsyncRT
# implementation, but their features (CUDA-graph capture, multicast,
# host-visible completion flags) are not implemented here.


struct CompletionFlag(ImplicitlyCopyable):
    """Host-visible completion flag.

    This struct is intentionally non-owning: it wraps the raw address of a
    ``M::Driver::CompletionFlag`` created by the C++ driver layer. The HAL
    backend has no C++ counterpart yet, so the handle is inert until the HAL
    C++ runtime shim lands; `DeviceStream.wait_for_host_value` raises before
    dereferencing it.
    """

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var _handle: OpaquePointer[MutUntrackedOrigin]

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @inline(.always)
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


struct DeviceContextArray[length: Int](Copyable, Sized):
    """A fixed-size collection of `DeviceContext` values.

    Used by multi-device custom-op `execute` methods to receive one
    `DeviceContext` per participating device. The graph compiler recognizes
    this type and synthesizes it from the per-device contexts discovered on the
    operation.

    Parameters:
        length: The number of `DeviceContext` values in the collection.
    """

    # ===-------------------------------------------------------------------===#
    # Aliases
    # ===-------------------------------------------------------------------===#

    @deprecated(
        "`DeviceContextArray.size` is deprecated, use"
        " `DeviceContextArray.length`."
    )
    comptime size = Self.length
    """The number of `DeviceContext` values in the collection. Deprecated
    alias for `length`."""

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var device_contexts: Array[DeviceContext, Self.length]
    """The underlying storage for the per-device contexts."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @inline(.always)
    def __init__(
        out self, var device_contexts: Array[DeviceContext, Self.length]
    ):
        """Initialize from an `Array` of `DeviceContext` values.

        Args:
            device_contexts: The per-device contexts to store.
        """
        self.device_contexts = device_contexts^

    @inline(.always)
    def __init__[
        *, __literal_size__: Int
    ](
        out self: DeviceContextArray[__literal_size__],
        var *device_contexts: DeviceContext,
        __list_literal__: NoneType = None,
    ):
        """Initialize from a variadic sequence of `DeviceContext` values.

        Parameters:
            __literal_size__: The number of contexts in the literal, inferred
                from the number of elements given.

        Args:
            device_contexts: One `DeviceContext` per device, exactly `length`
                of them.
            __list_literal__: Marker that lets this constructor accept
                list-literal syntax.
        """
        self.device_contexts = {uninitialized = True}
        var ptr = self.device_contexts.unsafe_ptr()
        comptime for i in range(__literal_size__):
            ptr.unsafe_offset(i).unsafe_write_move_from(
                Pointer(to=device_contexts[i])
            )

        # Do not destroy the elements when their backing storage goes away.
        device_contexts^._annihilate()

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

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

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    def __len__(self) -> Int:
        """Get the number of `DeviceContext` values in the collection.

        Returns:
            The size of the collection as specified by the `length` parameter.
        """
        return Self.length

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    def filter_gpu_contexts[
        num_gpu_devices: Int
    ](self) raises -> Array[DeviceContext, num_gpu_devices]:
        """Filters CPU contexts out and returns the GPU contexts in order.

        Some kernels receive a `DeviceContextArray` that mixes GPU contexts
        with CPU contexts carrying host-side pointers. Most kernels only
        want the GPU contexts in launch order, packed into a fixed-size
        `Array`.

        Parameters:
            num_gpu_devices: The expected number of GPU contexts. Used as
                the size of the returned `Array`.

        Returns:
            An `Array` of size `num_gpu_devices` containing the GPU
            contexts in their original order.

        Raises:
            If the number of GPU contexts in the list is not equal to
            `num_gpu_devices`.
        """
        # Validate the count up front. Passing a partially-filled staging
        # array to `unsafe_assume_initialized=` would still be UB at the
        # eventual destruction of the returned `Array`.
        var gpu_count = 0
        for i in range(Self.length):
            if self[i].api() != "cpu":
                gpu_count += 1
        if gpu_count != num_gpu_devices:
            raise Error("Invalid number of GPU device contexts")

        # Build the result in a `MaybeUninit` staging array. Its
        # `__deinit__` is a no-op, so the staging array is safe to drop even
        # with uninitialized slots in scope (e.g. on an early raise). The
        # `unsafe_assume_initialized=` constructor then moves every slot
        # into a fully-initialized `Array[DeviceContext]`.
        var staging = Array[MaybeUninit[DeviceContext], num_gpu_devices](
            uninitialized=True
        )
        var dev_idx = 0
        for i in range(Self.length):
            if self[i].api() != "cpu":
                staging[dev_idx].unsafe_write(DeviceContext(copy=self[i]))
                dev_idx += 1
        return Array[DeviceContext, num_gpu_devices](
            unsafe_assume_initialized=staging^
        )


@deprecated(use=DeviceContextArray)
comptime DeviceContextList = DeviceContextArray
"""Deprecated: A fixed-size collection of `DeviceContext` values.

This struct has been renamed to `DeviceContextArray`. This alias will be
removed in a future version of Mojo."""


struct DeviceMulticastBuffer[dtype: DType]:
    """A multi-GPU multicast buffer. Not supported by the HAL `DeviceContext`:
    every operation raises, so multicast-aware callers still compile.

    Parameters:
        dtype: Data dtype stored in the buffer.
    """

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @doc_hidden
    def __init__(out self, var contexts: List[DeviceContext], size: Int) raises:
        raise Error("Multicast not supported by the HAL DeviceContext")

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @doc_hidden
    def unicast_buffer_for(
        self, ctx: DeviceContext
    ) raises -> DeviceBuffer[Self.dtype]:
        raise Error("Multicast not supported by the HAL DeviceContext")

    @doc_hidden
    def multicast_buffer_for(
        self, ctx: DeviceContext
    ) raises -> DeviceBuffer[Self.dtype]:
        raise Error("Multicast not supported by the HAL DeviceContext")


@inline(.always)
def _checked(
    err: _CString,
    *,
    msg: String = "",
    location: Optional[SourceLocation] = None,
) raises:
    if err:
        _raise_checked_impl(err, msg, location.or_else(call_location()))
