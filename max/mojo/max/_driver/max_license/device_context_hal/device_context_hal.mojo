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


from std.builtin.device_passable import (
    DevicePointerLike,
    DeviceTypeEncoder,
)
from std.collections.optional import OptionalReg
from std.ffi import _Global
from std.compile import CompiledFunctionInfo
from std.memory import ArcPointer
from std.memory.unsafe import bitcast


from std.os import abort, getenv
from std.reflection import call_location, SourceLocation
from std.sys import (
    bit_width_of,
    CompilationTarget,
)


from ..hal import (
    Buffer,
    Context,
    Device as _HALDevice,
    Driver,
    FunctionHandle,
    RuntimeBundle,
    Stream,
    get_device_spec,
)
from max._plugin import MaxPluginForTarget
from ..hal._machine import DeviceRef, DeviceSpec
from ..hal.event import EVENT_FLAG_CPU_VISIBLE

from ..hal.extensions.traits import (
    SupportsCompletionFlagResetV1,
)

from max.gpu.host import (
    ExecutionConfig,
    GridBlockExecutionConfig,
    ClusterExecutionConfig,
    ClusterLaunchConfig,
    GridBlockLaunchConfig,
    LaunchAttribute,
    default_accelerator,
    Attribute,
    FuncAttribute,
)
from ..hal.execution_config import check_config_fields_launchable
from max.gpu.host.nvidia import (
    ClusterAttributeLaunchConfig,
    GridBlockAttributeLaunchConfig,
    LaunchAttributeHolderExecutionConfig,
)
from std.builtin.variadics import TypeList

from max.gpu.host.info import (
    GPUInfo,
    get_gpu_info,
    _device_type_encoder_target,
)
from max.gpu.host._launch_args import (
    DeviceFunctionWrapper,
    _IdentityWrapper,
)

# Re-exports any `__extension`s an overlay's hook file declares.
from ._device_context_hal_extras import *
from max.gpu.host._driver_handles import (
    _DeviceContextPtr,
    _DeviceFunctionPtr,
)
from max.gpu.host._vendor_handles import CUstream
from max.gpu.host._driver_common import (
    _DumpPath,
    _check_dim_in_range,
)


def _launch_attributes_ptr(
    mut attributes: List[LaunchAttribute],
) -> OptionalReg[OpaquePointer[MutUntrackedOrigin]]:
    # The caller keeps `attributes` alive until the plugin has read them.
    if len(attributes) == 0:
        return None
    return (
        attributes.unsafe_ptr()
        .unsafe_bitcast[NoneType]()
        .unsafe_origin_cast[MutUntrackedOrigin]()
    )


def _check_exec_config_launch_support[
    ExecutionConfigType: ExecutionConfig, target: CompilationTarget
]():
    comptime if conforms_to(ExecutionConfigType, ClusterExecutionConfig):
        comptime assert conforms_to(
            HALDeviceContext[target].ExecutionConfigType, ClusterExecutionConfig
        ), "cluster launch is not supported on this device"

    comptime if conforms_to(
        ExecutionConfigType, LaunchAttributeHolderExecutionConfig
    ):
        comptime assert conforms_to(
            HALDeviceContext[target].ExecutionConfigType,
            LaunchAttributeHolderExecutionConfig,
        ), "this target does not support launch attributes"

    check_config_fields_launchable[
        ExecutionConfigType, "HAL DeviceContext.enqueue_function"
    ]()


def _check_config_dims[
    func_name_for_msg: StringLiteral,
    ExecutionConfigType: ExecutionConfig,
](execution_config: ExecutionConfigType, *, location: SourceLocation) raises:
    """Checks a grid/block config's dims; any other config has none."""
    comptime if conforms_to(ExecutionConfigType, GridBlockExecutionConfig):
        _check_dim_in_range[func_name_for_msg, "grid_dim"](
            execution_config.get_grid_dim(), location=location
        )
        _check_dim_in_range[func_name_for_msg, "block_dim"](
            execution_config.get_block_dim(), location=location
        )


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


def _plugin_spec_entry(api: String) raises -> String:
    """Returns the first `MODULAR_DRIVER_PLUGINS` entry that satisfies `api`.

    Args:
        api: The requested device API, such as "cpu", "gpu" or "cuda".

    Returns:
        The matching `name@/path/to/plugin.so` entry.

    Raises:
        If `MODULAR_DRIVER_PLUGINS` is not set or no entry satisfies `api`.
    """
    var plugin_spec = getenv("MODULAR_DRIVER_PLUGINS")
    if not plugin_spec:
        raise Error("MODULAR_DRIVER_PLUGINS not set")
    # The spec may list several plugins ("CUDA@/x.so;CPU@/y.so").
    for entry in plugin_spec.split(";"):
        var name = String(entry.split("@")[0])
        if _driver_satisfies_api(name, api):
            return String(entry)
    raise Error(
        String(t"Requested API {api} not provided by MODULAR_DRIVER_PLUGINS")
    )


def _load_gpu_driver() -> Optional[ArcPointer[Driver]]:
    """Loads the accelerator plugin named by `MODULAR_DRIVER_PLUGINS`.

    Used as the one-time initializer for `_GPU_DRIVER`; see that global. It
    backs a `_Global` initializer, so it must not raise: a missing or
    unloadable plugin is reported as an empty `Optional`.
    """
    try:
        return Optional(Driver.create(_plugin_spec_entry("gpu")))
    except:
        return None


comptime _GPU_DRIVER = _Global["MODULAR_HAL_GPU_DRIVER", _load_gpu_driver]


@fieldwise_init
struct _CPUPlugin(Copyable, Movable):
    """The CPU plugin's driver and its device 0."""

    var driver: ArcPointer[Driver]
    var device: ArcPointer[_HALDevice[get_device_spec[0]()]]


def _open_cpu_plugin() raises -> _CPUPlugin:
    """Loads the CPU plugin named by `MODULAR_DRIVER_PLUGINS` and opens device 0.

    Returns:
        The loaded driver and device.

    Raises:
        If the plugin is not named, does not load, or has no device 0.
    """
    var driver = Driver.create(_plugin_spec_entry("cpu"))
    var device = driver[].get_device[0]()
    return _CPUPlugin(driver=driver^, device=device^)


def _load_cpu_plugin() -> Optional[_CPUPlugin]:
    """Opens the CPU plugin; the one-time initializer for `_CPU_PLUGIN`.

    It backs a `_Global` initializer, so it must not raise: a missing or
    unloadable plugin is reported as an empty `Optional`.
    """
    try:
        return Optional(_open_cpu_plugin())
    except:
        return None


# CPU kernels create a CPU context per call (`sync_parallelize` does when it is
# given none). Loading the plugin per context would put a `dlopen`, and a
# symbol lookup per entry point and extension, on every CPU kernel launch.
comptime _CPU_PLUGIN = _Global["MODULAR_HAL_CPU_PLUGIN", _load_cpu_plugin]


def _accelerator_api_and_arch(device_id: Int) raises -> Tuple[String, String]:
    """Returns the accelerator plugin's API and a device's arch.

    Reads both from the process-global driver without opening a context, which
    some plugins allow only one of per device.

    Args:
        device_id: The device to query.

    Returns:
        The device API, as `DeviceContext.api()` reports it, and the plugin's
        arch name.

    Raises:
        If no accelerator plugin is loaded, the plugin was loaded without an
        API name, or the plugin has no arch for the device.
    """
    var maybe_driver = _GPU_DRIVER.get_or_create_ptr()
    __guard case .Some(driver) = maybe_driver[] else:
        raise Error("no accelerator driver plugin is loaded")
    var api = String(driver[].name()).lower()
    # A plugin listed as a bare path, without a `name@`, is named after
    # its file, which is no API and so no `--target-accelerator` vendor
    # either.
    if "/" in api or api.endswith(".so"):
        raise Error("the accelerator plugin has no API name: ", api)
    var device = driver[].get_device_dynamic[Optional[Int64](0)](
        Int64(device_id)
    )
    return (api, device[].arch())


struct HALDeviceContext[target: CompilationTarget = default_accelerator()](
    ImplicitlyCopyable, RegisterPassable
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

    # DeviceContext is only usable on hosts so `current()` will always be a CPU.
    comptime CPU = HALDeviceContext[CompilationTarget.current()]
    """DeviceContext type for the current CPU target."""

    comptime Unknown = HALDeviceContext[CompilationTarget._unknown()]
    """DeviceContext type whose compilation target is not statically known."""

    comptime Optional[
        target: CompilationTarget = default_accelerator()
    ] = Optional[HALDeviceContext[target]]
    """Convenience type alias for an `Optional[DeviceContext[_]]` type parameter.

    This associated type worksaround Mojo's missing support for nested
    auto-parameterization. Mojo does not currently support argument types like:

    ```mojo
    ctx: Optional[DeviceContext[_]],
    ```

    because `_` unbound parameters can only be used in the top-level type of a
    argument type annotation.

    However, using this alias, auto-parameterization works:

    ```mojo
    ctx: DeviceContext.Optional,

    # equivalent, also works:
    ctx: DeviceContext.Optional,
    ```

    This alias may be removed once Mojo supports nested auto-parameterization.
    """

    comptime NoneValue = Optional[HALDeviceContext.CPU]()
    """Convenience helper for constructing a default value for
    `DeviceContext.Optional` parameters. This guides type inference to the CPU
    `target` when the default argument value is used, so omitting the context
    means "run on the CPU".

    Intended to be used like:

    ```mojo
    ctx: DeviceContext.Optional = DeviceContext.NoneValue,
    ```
    """

    comptime device_spec = get_device_spec[0]()
    """`DeviceSpec` describing the device this context targets."""

    comptime _PluginExecutionConfigType: ExecutionConfig = MaxPluginForTarget[
        DeviceSpec.target[Self.device_spec]()
    ].HalHooks.ExecutionConfigType[DeviceRef(Self.device_spec, 0, private=())]

    comptime _plugin_has_clusters = conforms_to(
        Self._PluginExecutionConfigType, ClusterExecutionConfig
    )
    comptime _plugin_has_attributes = conforms_to(
        Self._PluginExecutionConfigType, LaunchAttributeHolderExecutionConfig
    )

    comptime ExecutionConfigType: ExecutionConfig = (
        (
            ClusterAttributeLaunchConfig if Self._plugin_has_clusters else GridBlockAttributeLaunchConfig
        ) if Self._plugin_has_attributes else (
            ClusterLaunchConfig if Self._plugin_has_clusters else (
                Self._PluginExecutionConfigType if (
                    Self._HALHooks.Launcher.launches_config
                    and not conforms_to(
                        Self._PluginExecutionConfigType,
                        GridBlockExecutionConfig,
                    )
                ) else GridBlockLaunchConfig
            )
        )
    )
    """The launch configuration type with every launch capability this
    device supports.

    On NVIDIA devices it carries launch attributes, and from sm_90 on also
    thread block clusters, as on the legacy `DeviceContext`. When the plugin's
    own configuration isn't grid/block shaped and the plugin's launcher takes
    it, it's that configuration.

    Generic code checks for a capability with `conforms_to`, for example
    `conforms_to(DeviceContext[].ExecutionConfigType, ClusterExecutionConfig)`.
    """

    comptime default_device_info = get_gpu_info[Self.target._arch()]()
    """`GPUInfo` object for the default accelerator."""

    comptime _HALHooks = MaxPluginForTarget[Self.target].HalHooks

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var _driver: ArcPointer[Driver]
    var _device: ArcPointer[_HALDevice[Self.device_spec]]
    var _context: ArcPointer[Context[Self.device_spec]]
    var _stream: ArcPointer[Stream[Self.device_spec]]
    # Null until the C++ runtime shim provides it; read by the CUDA/HIP
    # interop accessors (`_nvidia_cuda` / `_amdgpu_hip`).
    var _handle: _DeviceContextPtr[mut=True]

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @staticmethod
    @inline(.always)
    def _default_api() -> String:
        """Returns the device API that matches this context's target.

        Resolving the API from `Self.target` keeps CPU targets from looking up
        GPU device info, which only exists for accelerator targets.

        Returns:
            `"cpu"` for a CPU target, otherwise the API of the target
            accelerator.
        """
        comptime if Self.target.is_cpu():
            return "cpu"
        else:
            return String(Self.default_device_info.api)

    @staticmethod
    def _api_matches_target(api: String) -> Bool:
        """Checks that a requested device API is one this target can run on.

        Args:
            api: Requested device API, such as "cpu", "gpu", "cuda" or "hip".

        Returns:
            `True` if a CPU target is paired with "cpu", or any other target
            is paired with a non-CPU API.
        """
        var is_cpu_api = api.lower() == "cpu"
        return is_cpu_api == Self.target.is_cpu()

    @inline(.always)
    def __init__(
        out self,
        device_id: Int = 0,
        *,
        var api: String = Self._default_api(),
    ) raises:
        """Constructs a `DeviceContext` for the specified device.

        This initializer creates a new device context for the specified accelerator device.
        The device context provides an interface for interacting with the GPU, including
        memory allocation, data transfer, and kernel execution.

        Args:
            device_id: ID of the accelerator device. If not specified, uses
                the default accelerator (device 0).
            api: Requested device API (for example, "cuda" or "hip"). Defaults
                to the device API of this context's target, which is "cpu" for
                a CPU target.

        Raises:
            If `api` does not match this context's target (a CPU target
            requires "cpu" and any other target requires a non-CPU API), if
            device initialization fails, or if the specified device is not
            available.

        Example:

        ```mojo
        from max.gpu.host import DeviceContext

        # Create a context for the default GPU
        var ctx = DeviceContext()

        # Create a context for a specific GPU (device 1)
        var ctx2 = DeviceContext(1)
        ```
        """

        if not Self._api_matches_target(api):
            raise Error(
                String(
                    t"DeviceContext with target"
                    t" {StaticString(Self.target.__triple())} cannot be created"
                    t" with api '{api}'"
                )
            )

        self._handle = None
        # `_api_matches_target` above pairs a CPU target with the "cpu" api, so
        # the branch is known at compile time. Choosing it here keeps the
        # accelerator path out of CPU-only code. The CPU branch also skips
        # parsing `MODULAR_DRIVER_PLUGINS`, which the cached load already did.
        comptime if Self.target.is_cpu():
            # `_open_cpu_plugin` only loads a plugin that satisfies "cpu", so
            # the driver needs no api check here.
            var maybe_plugin = _CPU_PLUGIN.get_or_create_ptr()
            __guard case .Some(plugin) = maybe_plugin[] else:
                # The cached load drops its error; load again to raise it.
                _ = _open_cpu_plugin()
                raise Error("failed to load the CPU plugin")
            self._driver = plugin.driver
            self._device = plugin.device
        else:
            # Load the first plugin whose name satisfies the requested api: its
            # concrete vendor name, or the generic accelerator alias "gpu".
            var chosen = _plugin_spec_entry(api)
            # Shared so peer calls get both contexts from one driver.
            var maybe_driver = _GPU_DRIVER.get_or_create_ptr()
            if case .Some(driver) = maybe_driver[]:
                self._driver = driver
            else:
                # `_Global` init drops the load error; reload to raise it.
                _ = Driver.create(chosen)
                raise Error(
                    String(t"failed to load accelerator plugin '{chosen}'")
                )
            var driver_name = self._driver[].name()
            if not _driver_satisfies_api(String(driver_name), api):
                raise Error(
                    String(
                        t"Requested API {api} not supported by driver"
                        t" {driver_name}"
                    )
                )
            var count = Int(self._driver[].device_count())
            if device_id < 0 or device_id >= count:
                raise Error(
                    String(t"device_id {device_id} out of range [0, {count})")
                )
            # The spec stays id 0 for typing; the runtime id picks the device.
            self._device = rebind[ArcPointer[_HALDevice[Self.device_spec]]](
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
        self = Self._from_unsafe_cpp_handle(
            ctx_ptr.unsafe_value().unsafe_bitcast[NoneType]()
        )

    @inline(.always)
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
    ) -> HALDeviceContext[Self.target]:
        return handle.unsafe_bitcast[HALDeviceContext[Self.target]]()[]

    # ===-------------------------------------------------------------------===#
    # Factory methods
    # ===-------------------------------------------------------------------===#

    @staticmethod
    def cpu() -> HALDeviceContext.CPU:
        """Creates a CPU `DeviceContext`.

        Returns:
            A new CPU `DeviceContext`.
        """
        try:
            return HALDeviceContext.CPU(api="cpu")
        except e:
            abort(t"unable to create CPU DeviceContext: {e}")

    # TODO(MSTDL-3258): Remove once all APIs fully support generic DeviceContext
    @inline(.always)
    def assert_default_target(self) -> HALDeviceContext[]:
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
        return rebind[HALDeviceContext[]](self)

    @inline(.always)
    def into_unknown_target(self) -> HALDeviceContext.Unknown:
        """Erases this DeviceContext's static target.

        For code that only uses the context's handle, so it neither needs nor
        should assume a target.

        Returns:
            This device context with an unknown target.
        """
        return rebind[HALDeviceContext.Unknown](self)

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

    def push_context(self) raises -> _HALDeviceContextScope:
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
        return _HALDeviceContextScope(self)

    def supports_flag_wait(self) -> Bool:
        """Returns whether this device supports completion flags.

        Returns:
            True when the device's plugin exports both the completion-flag
            and the flag-reset extensions.
        """
        # A context boxed under another target's type, such as a CPU context
        # in the accelerator's `DeviceContext[]`, would read the wrong
        # extension struct, so check the runtime API first.
        if self.api() != Self._default_api():
            return False
        comptime Ext = type_of(self._context[].extensions())
        comptime if conforms_to(Ext, SupportsCompletionFlagResetV1):
            ref ext = self._context[].extensions()
            return Bool(ext.completion_flag()) and Bool(
                ext.completion_flag_reset()
            )
        return False

    @doc_hidden
    def _hal_stream(
        self,
    ) -> ArcPointer[Stream[get_device_spec[0]()]]:
        return self._stream

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
    ) raises -> HALDeviceFunction[
        func,
        MaxPluginForTarget[Self.target].FunctionWrapper,
        target=Self.target,
    ]:
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
        return HALDeviceFunction[
            func,
            MaxPluginForTarget[Self.target].FunctionWrapper,
            target=Self.target,
        ](self, func_attribute=func_attribute)

    # ===-------------------------------------------------------------------===#
    # Buffer operations
    # ===-------------------------------------------------------------------===#

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
        return String(self._driver[].name()).lower()

    def enqueue_wait_for(self, other: HALDeviceContext) raises:
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
        var event = other._stream[].record_event[EVENT_FLAG_CPU_VISIBLE]()
        self._stream[].wait_for_events(event)


struct _HALDeviceContextScope(Movable):
    var _device_context: HALDeviceContext.Unknown
    # Driver-context handle to restore on exit; only meaningful when `_active`
    # (any value, including zero, can be a live handle).
    var _prev_driver_ctx: Int
    var _active: Bool

    def __init__(out self, device_context: HALDeviceContext):
        self._device_context = device_context.into_unknown_target()
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

    def __enter__(mut self) raises -> HALDeviceContext[]:
        self._prev_driver_ctx = (
            self._device_context._context[].get_current_driver_context()
        )
        self._device_context._context[].set_current()
        self._active = True
        return rebind[HALDeviceContext[]](self._device_context)

    def __exit__(mut self) raises:
        if self._active:
            self._device_context._context[].set_current_driver_context(
                self._prev_driver_ctx
            )
            self._active = False


# ===-----------------------------------------------------------------------===#
# DeviceFunction
# ===-----------------------------------------------------------------------===#


struct HALDeviceFunction[
    ret_type: AnyType,
    declared_arg_types: TypeList[Trait=AnyType, ...],
    //,
    func: def(*args: *declared_arg_types) thin -> ret_type,
    Wrapper: DeviceFunctionWrapper = _IdentityWrapper,
    *,
    target: CompilationTarget = default_accelerator(),
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
        target: The accelerator the function is compiled for.
    """

    comptime _target = DeviceSpec.target[get_device_spec[0]()]()
    comptime _wrapped_func = Self.Wrapper.wrap[Self.func]

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var _ctx: HALDeviceContext.Unknown
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
        ctx: HALDeviceContext,
        *,
        func_attribute: OptionalReg[FuncAttribute] = None,
    ) raises:
        """Compiles `Self.func` for `ctx`'s device and loads the function.

        Args:
            ctx: The device context to compile for.
            func_attribute: Optional attribute for the loaded function. Only
                `MAX_DYNAMIC_SHARED_SIZE_BYTES` is accepted, and it has no
                effect.

        Raises:
            If compilation or function loading fails, or if an unsupported
            function attribute is provided.
        """
        if case .Some(func_attribute) = func_attribute:
            # The CUDA plugin lifts the dynamic shared memory cap to the
            # device's opt-in limit when it loads a function, and Metal and
            # HIP have no cap, so a requested cap needs nothing more.
            if (
                func_attribute.attribute
                != Attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES
            ):
                raise Error(
                    "the function attribute '",
                    func_attribute.attribute,
                    "' is not currently supported",
                )
        var info = ctx._context[]._compile_inner[
            type_of(Self._wrapped_func), Self._wrapped_func
        ]()
        # The context's function cache owns the loaded module: repeated
        # constructions of the same kernel reuse the cached module.
        var func_handle = ctx._context[].get_or_load_function(
            info.asm, info.function_name
        )
        self._ctx = ctx.into_unknown_target()
        self._handle = None
        self._func_info = info
        self._inner = ArcPointer(
            _DeviceFunctionInner(func_handle, ctx._context)
        )


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
