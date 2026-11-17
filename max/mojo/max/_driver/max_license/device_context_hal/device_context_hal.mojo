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


from std.ffi import _Global
from std.memory import ArcPointer


from std.os import getenv
from std.reflection import call_location
from std.sys import CompilationTarget


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

from max.gpu.host import (
    ExecutionConfig,
    GridBlockExecutionConfig,
    ClusterExecutionConfig,
    ClusterLaunchConfig,
    GridBlockLaunchConfig,
    default_accelerator,
)
from max.gpu.host.nvidia import (
    ClusterAttributeLaunchConfig,
    GridBlockAttributeLaunchConfig,
    LaunchAttributeHolderExecutionConfig,
)


# Re-exports any `__extension`s an overlay's hook file declares.
from ._device_context_hal_extras import *
from max.gpu.host._driver_handles import _DeviceContextPtr
from max.gpu.host._vendor_handles import CUstream


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

    comptime Unknown = HALDeviceContext[CompilationTarget._unknown()]
    """DeviceContext type whose compilation target is not statically known."""

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
        var api: String,
    ) raises:
        """Constructs a `DeviceContext` for the specified device.

        This initializer creates a new device context for the specified accelerator device.
        The device context provides an interface for interacting with the GPU, including
        memory allocation, data transfer, and kernel execution.

        Args:
            device_id: ID of the accelerator device. If not specified, uses
                the default accelerator (device 0).
            api: Requested device API (for example, "cuda" or "hip"). A CPU
                target requires "cpu" and any other target requires a non-CPU
                API.

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
    # Methods
    # ===-------------------------------------------------------------------===#

    @doc_hidden
    def _hal_stream(
        self,
    ) -> ArcPointer[Stream[get_device_spec[0]()]]:
        return self._stream
