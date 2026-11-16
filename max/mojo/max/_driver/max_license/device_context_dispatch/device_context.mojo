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
"""Backend-dispatching `DeviceContext`, `DeviceBuffer`, and `HostBuffer`.

Each struct forwards to the legacy or the HAL implementation, whichever
`use_legacy_backend` names for the target. Where the two backends disagree on a
signature, the wrapper keeps the legacy one, because that is what call sites are
written against.
"""

from std.builtin.device_passable import DevicePassable
from std.collections.optional import OptionalReg
from std.atomic import Atomic
from std.ffi import _Global, c_size_t
from std.iter import downcast
from std.memory import ArcPointer, Pointer
from std.os import abort, getenv
from std.runtime import parallelism_level
from std.runtime._asyncrt import TaskGroup
from std.time import monotonic
from std.utils.lock import BlockingScopedLock, BlockingSpinLock
from std.reflection import (
    SourceLocation,
    call_location,
)
from std.sys import size_of
from std.sys.info import CompilationTarget

from max._driver.max_license.device_context_dispatch.device_function import (
    DeviceExternalFunction,
    DeviceFunction,
    _DispatchEnqueuer,
    _HALDeviceExternalFunction,
    _HALDeviceFunction,
)
from max._driver.max_license.device_context_hal.device_context_hal import (
    HALDeviceContext,
    _HALDeviceContextScope,
    _GPU_DRIVER,
    _check_config_dims,
    _check_copy_src_size,
    _memset_value_as_u64,
    _driver_satisfies_api,
)
from max._driver.max_license.hal import (
    Context,
    Device as _HALDevice,
    Event,
    MEMORY_UNKNOWN,
    Stream,
    get_device_spec,
)
from max._driver.max_license.hal.extensions.traits import (
    SupportsPeerAccessV1,
)
from max._driver.max_license.hal.event import EVENT_FLAG_CPU_VISIBLE
from max._driver.max_license.hal._plugin._trait import HalLaunchOptions
from max._plugin import MaxPluginForTarget
from max.gpu.host._legacy.constant_memory_mapping import ConstantMemoryMapping
from max.gpu.host._legacy.device_attribute import DeviceAttribute
from max.gpu.host._legacy.dim import Dim
from max.gpu.host._legacy.func_attribute import FuncAttribute
from max.gpu.host._legacy.launch_attribute import LaunchAttribute
from max.gpu.host._legacy.device_context import (
    LegacyDeviceClosureFunction,
    LegacyDeviceContext,
    _LegacyDeviceContextScope,
)
from max.gpu.host._driver_common import _DumpPath
from max.gpu.host._launch_args import DeviceFunctionWrapper
from max.gpu.host._overlay import HAL_DEVICE_CONTEXT
from max.gpu.host._driver_common import (
    StreamPriorityRange as _OverlayStreamPriorityRange,
    _check_dim_in_range,
)
from max.gpu.host._driver_handles import _DeviceContextPtr
from max.gpu.host.info import default_accelerator
from max.gpu.host.launch_config import (
    ExecutionConfig,
    GridBlockExecutionConfig,
)
from .host_types import (
    CompletionFlag,
    _CompletionFlagBox,
    Device,
    DeviceEvent,
    DeviceStream,
    _HALDeviceStream,
)
from max._driver.max_license.device_context_dispatch.device_buffer import (
    DeviceBuffer,
    _HALDeviceBuffer,
)
from max._driver.max_license.device_context_dispatch.host_buffer import (
    HostBuffer,
)
from ._device_context_extras import *


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
        if case .Some(driver) = maybe_driver[]:
            return _PeerState(Int(driver[].device_count()))
        return _PeerState(0)
    except:
        return _PeerState(0)


comptime _PEER_STATE = _Global["MODULAR_HAL_PEER_STATE", _init_peer_state]


def _check_device_to_device_copy[
    target: CompilationTarget, dtype: DType
](
    ctx: HALDeviceContext[target],
    dst_buf: _HALDeviceBuffer[dtype],
    src_buf: _HALDeviceBuffer[dtype],
) raises:
    # Runs before any cross-stream wait, which would hand this plugin
    # another driver's event.
    var dst_device = dst_buf._inner[]._context[].device_id()
    if dst_device != ctx._device[].id:
        raise Error(
            String(
                t"enqueue_copy destination is on device {dst_device}, not"
                t" on this context's device {ctx._device[].id}"
            )
        )
    var src_context = src_buf._inner[]._context
    if src_context[].device_id() != ctx._device[].id and not (
        src_context[]._raw is ctx._context[]._raw
    ):
        raise Error(
            "enqueue_copy source is on another driver; cross-device"
            " copies need both buffers on the same driver"
        )


def _enqueue_copy_device_to_device[
    target: CompilationTarget, dtype: DType
](
    ctx: HALDeviceContext[target],
    dst_buf: _HALDeviceBuffer[dtype],
    src_buf: _HALDeviceBuffer[dtype],
) raises:
    var src_context = src_buf._inner[]._context
    var dst_view = dst_buf._inner[]._buffer.view()
    var src_view = src_buf._inner[]._buffer.view()
    if src_context[].device_id() == ctx._device[].id:
        ctx._stream[].copy_intra_device(dst_view, src_view)
    else:
        ctx._stream[].copy_inter_device(
            dst_view, src_view, src_context[]._handle
        )


def use_legacy_backend() -> Bool:
    """Returns whether this build executes through the legacy backend.

    A build uses exactly one of the two device context implementations, so every
    wrapper's `A if use_legacy_backend() else B` shares this one condition. Keep
    it a plain call, not an alias or `@always_inline`: the compiler refines such
    a type to an arm inside a matching `comptime if` only while the condition
    stays unevaluated.

    Returns:
        True when the build uses the legacy device context.
    """
    return not HAL_DEVICE_CONTEXT


struct DeviceContext[target: CompilationTarget = default_accelerator()](
    ImplicitlyCopyable, RegisterPassable, _DispatchEnqueuer
):
    """Represents a single stream of execution on a particular accelerator
    (GPU).

    A `DeviceContext` serves as the low-level interface to the accelerator
    inside a MAX [custom operation](/max/develop/custom-ops/) and provides
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
        ctx.enqueue_function[kernel](grid_dim=1, block_dim=(2, 2, 2))
        ctx.synchronize()
    ```
    """

    # ===-------------------------------------------------------------------===#
    # Backend selection
    # ===-------------------------------------------------------------------===#

    comptime _Legacy = LegacyDeviceContext[Self.target]
    comptime _HAL = HALDeviceContext[Self.target]
    comptime _Impl = Self._Legacy if use_legacy_backend() else Self._HAL

    comptime _HALHooks = MaxPluginForTarget[Self.target].HalHooks

    # ===-------------------------------------------------------------------===#
    # Aliases
    # ===-------------------------------------------------------------------===#

    # DeviceContext is only usable on hosts so `current()` will always be a CPU.
    comptime CPU = DeviceContext[CompilationTarget.current()]
    """DeviceContext type for the current CPU target."""

    comptime Unknown = DeviceContext[CompilationTarget._unknown()]
    """DeviceContext type whose compilation target is not statically known."""

    comptime Optional[
        target: CompilationTarget = default_accelerator()
    ] = Optional[DeviceContext[target]]
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

    comptime NoneValue = Optional[DeviceContext.CPU]()
    """Convenience helper for constructing a default value for
    `DeviceContext.Optional` parameters. This guides type inference to the CPU
    `target` when the default argument value is used, so omitting the context
    means "run on the CPU".

    Intended to be used like:

    ```mojo
    ctx: DeviceContext.Optional = DeviceContext.NoneValue,
    ```
    """

    comptime default_device_info = (
        Self._Legacy.default_device_info if use_legacy_backend() else Self._HAL.default_device_info
    )
    """`GPUInfo` object for the default accelerator."""

    comptime FunctionWrapper: DeviceFunctionWrapper = MaxPluginForTarget[
        Self.target
    ].FunctionWrapper
    """Selects the function `compile_function` compiles for a user kernel."""

    comptime ExecutionConfigType: ExecutionConfig = (
        Self._Legacy.ExecutionConfigType if use_legacy_backend() else Self._HAL.ExecutionConfigType
    )
    """The launch configuration type with every launch capability this target
    supports.

    On NVIDIA targets it carries launch attributes, and from sm_90 on also
    thread block clusters.

    Generic code checks for a capability with `conforms_to`, for example
    `conforms_to(DeviceContext[].ExecutionConfigType, ClusterExecutionConfig)`.
    """

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var _impl: Self._Impl

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
        comptime if use_legacy_backend():
            return LegacyDeviceContext[Self.target]._default_api()
        else:
            return HALDeviceContext[Self.target]._default_api()

    @staticmethod
    def _api_matches_target(api: String) -> Bool:
        """Checks that a requested device API is one this target can run on.

        Args:
            api: Requested device API, such as "cpu", "gpu", "cuda" or "hip".

        Returns:
            `True` if a CPU target is paired with "cpu", or any other target
            is paired with a non-CPU API.
        """
        comptime if use_legacy_backend():
            return LegacyDeviceContext[Self.target]._api_matches_target(api)
        else:
            return HALDeviceContext[Self.target]._api_matches_target(api)

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
        comptime if use_legacy_backend():
            self._impl = Self._Legacy(device_id=device_id, api=api^)
        else:
            self._impl = Self._HAL(device_id=device_id, api=api^)

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
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            return DeviceContext.CPU(_impl=self._impl.assert_cpu_target())
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        return rebind_var[DeviceContext.CPU](self^)

    # FIXME: The more obvious spelling of this currently crashes the compiler:
    #    context.map(DeviceContext.assert_cpu_target)
    @staticmethod
    def _map_assert_cpu_target(
        ctx: DeviceContext.Optional,
    ) -> Optional[DeviceContext.CPU]:
        var cpu_ctx = Optional[DeviceContext.CPU]()
        if case .Some(gpu_ctx) = ctx:
            cpu_ctx = gpu_ctx.assert_cpu_target()
        return cpu_ctx^

    @inline(.always)
    def assert_default_target(self) -> DeviceContext[]:
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
        comptime if use_legacy_backend():
            return DeviceContext[](_impl=self._impl.assert_default_target())
        else:
            return DeviceContext[](_impl=self._impl.assert_default_target())

    @inline(.always)
    def into_unknown_target(self) -> DeviceContext.Unknown:
        """Erases this DeviceContext's static target.

        For code that only uses the context's handle, so it neither needs nor
        should assume a target.

        Returns:
            This device context with an unknown target.
        """
        comptime if use_legacy_backend():
            return DeviceContext.Unknown(_impl=self._impl.into_unknown_target())
        else:
            return DeviceContext.Unknown(_impl=self._impl.into_unknown_target())

    # ===-------------------------------------------------------------------===#
    # Escape hatches
    # ===-------------------------------------------------------------------===#

    @doc_hidden
    def __init__(out self, *, var _impl: Self._Impl):
        """Wraps an already-constructed backend context.

        Args:
            _impl: The backend context to take ownership of.
        """
        self._impl = _impl^

    @doc_hidden
    def __init__(out self, ctx_ptr: _DeviceContextPtr[mut=True]):
        """Wraps the runtime's native context handle.

        Args:
            ctx_ptr: The native handle of the context.
        """
        comptime if use_legacy_backend():
            self._impl = Self._Legacy(ctx_ptr)
        else:
            self._impl = Self._HAL(ctx_ptr)

    @doc_hidden
    def __init__(out self, handle: OpaquePointer[MutUntrackedOrigin]):
        """Wraps an opaque pointer to the runtime's native context.

        Args:
            handle: The opaque pointer to the native context.
        """
        comptime if use_legacy_backend():
            self._impl = Self._Legacy(
                rebind[OpaquePointer[UntrackedOrigin[mut=True]]](handle)
            )
        else:
            self._impl = Self._HAL(handle)

    def unwrap_legacy(self) -> Self._Legacy:
        """Returns the underlying legacy context.

        For code that needs a legacy-only member. Only valid in a build whose
        backend is the legacy one.

        Returns:
            The legacy device context this wrapper holds.
        """
        comptime assert (
            use_legacy_backend()
        ), "unwrap_legacy is only available in a legacy DeviceContext build"
        return self._impl

    def unwrap_hal(self) -> Self._HAL:
        """Returns the underlying HAL context.

        For code that needs a HAL-only member. Only valid in a build whose
        backend is the HAL one.

        Returns:
            The HAL device context this wrapper holds.
        """
        comptime assert (
            not use_legacy_backend()
        ), "unwrap_hal is only available in a HAL DeviceContext build"
        return self._impl

    @doc_hidden
    def _handle_ptr(self) -> _DeviceContextPtr[mut=True]:
        """Returns the backend's native context handle.

        For the AsyncRT calls that the shared kernel-load and capture helpers
        make on the context directly.

        Returns:
            The handle of the backend context this wrapper holds.
        """
        comptime if use_legacy_backend():
            return self._impl._handle
        else:
            return self._impl._handle

    # ===-------------------------------------------------------------------===#
    # Function enqueuer
    # ===-------------------------------------------------------------------===#

    comptime _LegacyEnqueuer = Self._Legacy

    def _legacy_enqueuer(self) -> Self._LegacyEnqueuer:
        comptime assert (
            use_legacy_backend()
        ), "this context is not on the legacy backend"
        return self._impl

    def _hal_stream(self) -> ArcPointer[Stream[get_device_spec[0]()]]:
        comptime assert (
            not use_legacy_backend()
        ), "this context is not on the HAL backend"
        return self._impl._hal_stream()

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    def __enter__(var self) -> Self:
        """Enables the use of DeviceContext in a 'with' statement context manager.

        This method allows DeviceContext to be used with Python-style context managers,
        which ensures proper resource management and cleanup when the context exits.

        Returns:
            The DeviceContext instance to be used within the context manager block.

        Example:

        ```mojo
        from max.gpu.host import DeviceContext

        # Using DeviceContext as a context manager
        with DeviceContext() as ctx:
            # Perform GPU operations
            # Resources are automatically released when exiting the block
            pass
        ```
        """
        return self^

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    def synchronize(self) raises:
        """Synchronizes the device context, waiting for all enqueued operations
        to complete.

        This method blocks until all previously enqueued work on the context's
        stream has finished.

        Raises:
            If the synchronization operation fails.
        """
        comptime if use_legacy_backend():
            self._impl.synchronize()
        else:
            self._impl.synchronize()

    @deprecated("Use `DeviceContext.device().id` instead.")
    def id(self) raises -> Int64:
        """Returns the ID of the device this context is associated with.

        Returns:
            The device ID.

        Raises:
            If querying the device ID fails.
        """
        return self.device().id

    def get_memory_info(self) raises -> Tuple[c_size_t, c_size_t]:
        """Returns the free and total memory size for this device.

        This method queries the current state of device memory, providing
        information about how much memory is available and the total memory
        capacity of the device. This is useful for memory management and
        determining if there's enough space for planned operations.

        Returns:
            A tuple of (free memory, total memory) in bytes.

        Raises:
            If there's an error retrieving the memory information, or the
            device can't measure its free memory.

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
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            return self._impl.get_memory_info()
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        var info = self._impl._context[].get_memory_info()
        if info[0] == MEMORY_UNKNOWN:
            raise Error("free memory could not be measured on this device")
        return (c_size_t(info[0]), c_size_t(info[1]))

    @deprecated("Use `DeviceContext.device().get_attribute()` instead.")
    def get_attribute(self, attr: DeviceAttribute) raises -> Int:
        """Returns the value of a device attribute.

        Args:
            attr: The device attribute to query.

        Returns:
            The attribute's value.

        Raises:
            If querying the attribute fails.
        """
        return self.device().get_attribute(attr)

    def compute_capability(self) raises -> Int:
        """Returns the compute capability of this device.

        Returns:
            The device's compute capability.

        Raises:
            If querying the compute capability fails.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            return self._impl.compute_capability()
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        var major = self.device().get_attribute(
            DeviceAttribute.COMPUTE_CAPABILITY_MAJOR
        )
        # TODO: Apple GPUs report the chip generation (M1-M5) directly as the
        # compute capability, matching the native MetalDeviceContext
        # (`MetalDevice::computeCapability` returns the MAJOR only). The
        # major/minor split in device attributes is not portable and should be
        # changed.
        if self.api() == "metal":
            return major
        var minor = self.device().get_attribute(
            DeviceAttribute.COMPUTE_CAPABILITY_MINOR
        )
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
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            self._impl.set_as_current()
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        self._impl._context[].set_current()

    def api(self) -> String:
        """Returns the name of the API used to program the device.

        This method queries the underlying device context to determine which GPU programming
        API is being used for the current device. This information is useful for writing
        code that can adapt to different GPU architectures and programming models.

        Possible values are:

        - `"cuda"`: NVIDIA GPUs using CUDA.
        - `"hip"`: AMD GPUs using HIP.
        - `"metal"`: Apple GPUs using Metal.
        - `"cpu"`: CPU.
        - `""`: No accelerator available.

        Returns:
            The name of the API.
        """
        comptime if use_legacy_backend():
            return self._impl.api()
        else:
            return self._impl.api()

    @deprecated("Use `DeviceContext.device().get_name()` instead.")
    def name(self) -> String:
        """Returns the device name, defined by the native device API.

        Returns:
            A string containing the device name.
        """
        try:
            return self.device().get_name()
        except e:
            abort(String(e))

    def get_api_version(self) raises -> Int:
        """Returns the API version associated with this device.

        This method retrieves the version number of the GPU driver currently
        installed on the system for the device associated with this context.
        The version is returned as an integer that can be used to check
        compatibility with specific features or to troubleshoot driver-related
        issues.

        Returns:
            An integer representing the driver version.

        Raises:
            If the driver version cannot be retrieved or if the device context
            is invalid.

        Example:

        ```mojo
        from max.gpu.host import DeviceContext

        with DeviceContext() as ctx:
            # Get the API version
            var api_version = ctx.get_api_version()
            print("GPU API version:", api_version)
        ```
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            return self._impl.get_api_version()
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        # TODO: HAL stub - returns 0. Plumb plugin-side `cuDriverGetVersion`
        # / `hipDriverGetVersion` when callers actually need to gate on
        # driver version.
        return 0

    @deprecated("Use `DeviceContext.device().get_arch()` instead.")
    def arch_name(self) raises -> String:
        """Returns the architecture name of this device.

        Returns:
            The device architecture name.

        Raises:
            If querying the architecture fails.
        """
        return self.device().get_arch()

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
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            return self._impl.is_compatible()
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        # TODO: HAL stub - always returns True. Hook this up to a plugin-side
        # compute-capability + driver-version check when real semantics
        # are needed.
        return True

    @deprecated("Use `DeviceContext.device().is_host_unified()` instead.")
    def is_host_unified(self) raises -> Bool:
        """Returns whether this device shares memory with the host.

        Returns:
            True when device and host memory are unified.

        Raises:
            If the query fails.
        """
        return self.device().is_host_unified()

    def is_graph_capturing(self) raises -> Bool:
        """Returns whether work enqueued on this context is only recorded.

        Returns:
            True while a graph capture is in progress. Always False on the HAL
            backend, which has no stream capture.

        Raises:
            If the query fails. The HAL backend never raises.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            return self._impl.is_graph_capturing()
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        # TODO(DRIV-687): answer from the queue once the HAL has stream capture.
        return False

    @staticmethod
    def number_of_devices(*, var api: String = Self._default_api()) -> Int:
        """Returns the number of devices available that support the specified
        API.

        Args:
            api: Requested device API (for example, "cuda" or "hip"). Defaults
                to the device API specified by the current target accelerator.

        Returns:
            The number of devices, or 0 if no driver plugin is available or the
            loaded plugin does not support the requested API.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            return LegacyDeviceContext[Self.target].number_of_devices(api=api^)
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        # The built-in host (CPU) backend always provides a single device and
        # needs no plugin spec.
        if api.lower() == "cpu":
            return 1
        try:
            if not getenv("MODULAR_DRIVER_PLUGINS"):
                return 0
            # Reuse the process-global accelerator driver (loaded once).
            var maybe_driver = _GPU_DRIVER.get_or_create_ptr()
            if case .Some(driver) = maybe_driver[]:
                if _driver_satisfies_api(driver[].name(), api):
                    return Int(driver[].device_count())
            return 0
        except:
            return 0

    # ===-------------------------------------------------------------------===#
    # Streams, events and completion flags
    # ===-------------------------------------------------------------------===#

    def stream(self) raises -> DeviceStream:
        """Returns the context's default stream.

        Returns:
            The default `DeviceStream`.

        Raises:
            If the stream cannot be created.
        """
        comptime if use_legacy_backend():
            return DeviceStream(_impl=self._impl.stream())
        else:
            return DeviceStream(
                _impl=rebind_var[DeviceStream._Impl](
                    _HALDeviceStream(self._impl, self._impl._stream)
                )
            )

    def create_stream(self, *, priority: Int = 0) raises -> DeviceStream:
        """Creates a new stream on this context's device.

        Args:
            priority: Requested stream priority. The HAL queue API has no
                priority support, so the HAL backend accepts it for parity with
                the legacy backend and ignores it.

        Returns:
            The new `DeviceStream`.

        Raises:
            If the stream cannot be created.
        """
        comptime if use_legacy_backend():
            return DeviceStream(
                _impl=self._impl.create_stream(priority=priority)
            )
        else:
            return DeviceStream(
                _impl=rebind_var[DeviceStream._Impl](
                    _HALDeviceStream(
                        self._impl, self._impl._context[].create_stream()
                    )
                )
            )

    def create_event(self) raises -> DeviceEvent:
        """Creates a new event for synchronization between streams.

        Returns:
            A `DeviceEvent` that can be used for synchronization.

        Raises:
            If the event cannot be created.

        Example:

        ```mojo
        from max.gpu.host import DeviceContext

        var ctx = DeviceContext()

        var default_stream = ctx.stream()
        var new_stream = ctx.create_stream()

        # Create an event
        var event = ctx.create_event()

        # Record the event in the default stream
        default_stream.record_event(event)

        # new_stream waits for the event, then continues
        new_stream.enqueue_wait_for(event)
        default_stream.synchronize()
        ```
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            return DeviceEvent(_impl=self._impl.create_event())
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        # The HAL uses logical events where creation and recording happen in
        # one step. To work around this, create a `DeviceEvent` without a
        # backing HAL event and let the first `record_event` fill it in.
        return DeviceEvent(
            _impl=DeviceEvent._HAL(
                self.into_unknown_target(),
                ArcPointer(Optional[Event[EVENT_FLAG_CPU_VISIBLE]](None)),
            )
        )

    def allocate_completion_flag(self) raises -> CompletionFlag:
        """Allocates a `CompletionFlag`, initialized to 0, on this device.

        Returns:
            The new flag.

        Raises:
            If the allocation fails, or if the device's plugin has no
            completion flags.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            return CompletionFlag(_impl=self._impl.allocate_completion_flag())
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        return CompletionFlag(
            _impl=CompletionFlag._HAL(
                ArcPointer(
                    _CompletionFlagBox(self._impl.assert_default_target())
                )
            )
        )

    def supports_flag_wait(self) -> Bool:
        """Returns whether this device supports waiting on a completion flag.

        Returns:
            True when flag waits are supported.
        """
        comptime if use_legacy_backend():
            return self._impl.supports_flag_wait()
        else:
            return self._impl.supports_flag_wait()

    def num_streams(self) -> Int:
        """Returns the number of streams available on this device context.

        Returns:
            The number of streams available on this device context.
        """
        comptime assert (
            use_legacy_backend()
        ), "num_streams is only available on the legacy DeviceContext"
        return self._impl.num_streams()

    def select_stream(
        self, stream_id: Int
    ) raises -> DeviceContext[Self.target]:
        """Selects the stream that this context's work runs on.

        Args:
            stream_id: The index of the stream to select.

        Returns:
            A context bound to the selected stream.

        Raises:
            If the stream cannot be selected.
        """
        comptime assert (
            use_legacy_backend()
        ), "select_stream is only available on the legacy DeviceContext"
        return DeviceContext[Self.target](
            _impl=self._impl.select_stream(stream_id)
        )

    def create_external_stream(
        self, external_stream: Optional[MutPointer[NoneType, _]]
    ) raises -> DeviceStream:
        """Wraps a stream created outside of MAX.

        Args:
            external_stream: The native stream to wrap.

        Returns:
            A device stream backed by `external_stream`.

        Raises:
            If the stream cannot be wrapped.
        """
        comptime assert use_legacy_backend(), (
            "create_external_stream is only available on the legacy"
            " DeviceContext"
        )
        return DeviceStream(
            _impl=self._impl.create_external_stream(external_stream)
        )

    def stream_priority_range(self) raises -> _OverlayStreamPriorityRange:
        """Returns the range of stream priorities supported by this device
        context.

        Returns:
            A `StreamPriorityRange` containing the minimum and maximum stream
            priorities.

        Raises:
            If the query fails.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            return self._impl.stream_priority_range()
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        # TODO: HAL stub - returns the trivial range.
        return _OverlayStreamPriorityRange(0, 0)

    def push_context(self) raises -> _DeviceContextScope[Self.target]:
        """Returns a context manager that makes this device's driver context
        current for the duration of a `with` block.

        Returns:
            A context manager that manages the driver context stack.

        Raises:
            If the driver contexts cannot be switched.
        """
        return _DeviceContextScope[Self.target](self)

    def __eq__(self, other: Self) -> Bool:
        """Compares two contexts for identity.

        Args:
            other: The context to compare with.

        Returns:
            Whether both contexts are the same.
        """
        comptime assert (
            use_legacy_backend()
        ), "== is only available on the legacy DeviceContext"
        return self._impl == other._impl

    def __ne__(self, other: Self) -> Bool:
        """Compares two contexts for identity.

        Args:
            other: The context to compare with.

        Returns:
            Whether the contexts differ.
        """
        comptime assert (
            use_legacy_backend()
        ), "!= is only available on the legacy DeviceContext"
        return self._impl != other._impl

    # ===-------------------------------------------------------------------===#
    # Device views
    # ===-------------------------------------------------------------------===#

    def device(ref self) -> Device[Self.target, ImmOrigin(origin_of(self))]:
        """Returns a view of the device this context runs on.

        Returns:
            A `Device` that borrows this context.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            return Device[Self.target, ImmOrigin(origin_of(self))](
                _impl=rebind_var[
                    Device[Self.target, ImmOrigin(origin_of(self))]._Impl
                ](self._impl.device())
            )
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        return Device[Self.target, ImmOrigin(origin_of(self))](
            _impl=Device[Self.target, ImmOrigin(origin_of(self))]._HAL(
                Pointer(to=self._impl)
                .as_imm()
                .unsafe_origin_cast[ImmOrigin(origin_of(self))](),
                self._impl._device[].id,
            )
        )

    # ===-------------------------------------------------------------------===#
    # Buffer allocation
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
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            return DeviceBuffer[dtype](
                _impl=self._impl.enqueue_create_buffer[dtype](size)
            )
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        # NOTE: The HAL supports doing this asynchronously, but to match
        # existing DeviceContext semantics we create a buffer immediately.
        return DeviceBuffer[dtype](self, size)

    def create_buffer_sync[
        dtype: DType
    ](self, size: Int) raises -> DeviceBuffer[dtype]:
        """Allocates a device buffer and waits for the allocation to complete.

        Parameters:
            dtype: The element dtype of the buffer.

        Args:
            size: Number of elements to allocate.

        Returns:
            The allocated device buffer.

        Raises:
            If the allocation fails.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            return DeviceBuffer[dtype](
                _impl=self._impl.create_buffer_sync[dtype](size)
            )
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        return DeviceBuffer[dtype](self, size)

    def enqueue_create_host_buffer[
        dtype: DType
    ](self, size: Int) raises -> HostBuffer[dtype]:
        """Enqueues the creation of a HostBuffer.

        This function allocates memory on the host that is accessible by the
        device. The memory is page-locked (pinned) for efficient data transfer
        between host and device.

        Pinned memory is guaranteed to remain resident in the host's RAM, not be
        paged/swapped out to disk. Memory allocated normally (for example, using
        [`alloc()`](/docs/std/memory/alloc/alloc/))
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
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            return HostBuffer[dtype](
                _impl=self._impl.enqueue_create_host_buffer[dtype](size)
            )
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        return HostBuffer[dtype](self, size)

    def wrap_host_memory[
        origin: MutOrigin, //, dtype: DType
    ](
        self,
        host_ptr: Pointer[Scalar[dtype], origin],
        size: Int,
    ) raises -> DeviceBuffer[dtype]:
        """Makes a range of the caller's host memory accessible to this device.

        The returned buffer grants access, not ownership. `host_ptr` stays the
        caller's to keep mapped and to free, and dropping the buffer undoes
        only the access.

        Address the range through the returned buffer, not through `host_ptr`:
        only on CUDA are the two the same address.

        Parameters:
            origin: The origin of `host_ptr`.
            dtype: The element dtype of the range.

        Args:
            host_ptr: The host memory to make accessible.
            size: Number of elements in the range.

        Returns:
            A device buffer addressing the same memory.

        Raises:
            If the range cannot be registered with the driver.
        """
        comptime assert (
            use_legacy_backend()
        ), "wrap_host_memory is only available on the legacy DeviceContext"
        return DeviceBuffer[dtype](
            _impl=self._impl.wrap_host_memory(host_ptr, size)
        )

    # ===-------------------------------------------------------------------===#
    # Copies and fills
    # ===-------------------------------------------------------------------===#

    def enqueue_copy[
        dtype: DType
    ](
        self,
        dst_ptr: Pointer[mut=True, Scalar[dtype], _],
        src_ptr: Pointer[mut=False, Scalar[dtype], _],
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
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            self._impl.enqueue_copy(dst_ptr, src_ptr, size)
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        self.enqueue_copy(
            DeviceBuffer[dtype](self, dst_ptr, size, owning=False),
            DeviceBuffer[dtype](self, src_ptr, size, owning=False),
        )

    def enqueue_copy[
        dtype: DType
    ](
        self,
        dst_buf: DeviceBuffer[dtype],
        src_ptr: Pointer[mut=False, Scalar[dtype], _],
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
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            self._impl.enqueue_copy(dst_buf._impl, src_ptr)
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        self._impl._stream[].copy_to_device(
            dst_buf._impl._inner[]._buffer.view(),
            src_ptr.unsafe_bitcast[UInt8](),
        )

    def enqueue_copy[
        dtype: DType
    ](
        self,
        dst_buf: HostBuffer[dtype],
        src_ptr: Pointer[mut=False, Scalar[dtype], _],
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
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            self._impl.enqueue_copy(dst_buf._impl, src_ptr)
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        self._impl._stream[].copy_to_device(
            dst_buf._impl._inner[]._buffer.view(),
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
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            self._impl.enqueue_copy(dst_ptr, src_buf._impl)
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        self._impl._stream[].copy_from_device(
            dst_ptr.unsafe_bitcast[UInt8](),
            src_buf._impl._inner[]._buffer.view(),
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
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            self._impl.enqueue_copy(dst_buf._impl, src_buf._impl)
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        # As in legacy: the copy waits for `src_buf`'s writes, and `src_buf`'s
        # stream can't overwrite or free it until the copy has read it.
        _check_device_to_device_copy(self._impl, dst_buf._impl, src_buf._impl)
        var src_ctx = src_buf._impl._ctx._impl
        var sync = not (src_ctx._hal_stream() is self._impl._hal_stream())
        if sync:
            self._impl.enqueue_wait_for(src_ctx)
        _enqueue_copy_device_to_device(self._impl, dst_buf._impl, src_buf._impl)
        if sync:
            src_ctx.enqueue_wait_for(self._impl)

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
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            self._impl.enqueue_copy(dst_buf._impl, src_buf._impl)
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        _check_copy_src_size(
            dst_buf._impl._inner[]._buffer, src_buf._impl._inner[]._buffer
        )
        self._impl._stream[].copy_to_device(
            dst_buf._impl._inner[]._buffer.view(),
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
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            self._impl.enqueue_copy(dst_buf._impl, src_buf._impl)
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        ref dst_mem = dst_buf._impl._inner[]._buffer
        ref src_mem = src_buf._impl._inner[]._buffer
        _check_copy_src_size(dst_mem, src_mem)
        # `copy_from_device` copies the whole source view, so narrow it to the
        # destination's size.
        self._impl._stream[].copy_from_device(
            dst_buf.unsafe_ptr().unsafe_bitcast[UInt8](),
            src_mem.view(byte_offset=0, byte_size=dst_mem.byte_size),
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
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            self._impl.enqueue_copy(dst_buf._impl, src_buf._impl)
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        _check_copy_src_size(
            dst_buf._impl._inner[]._buffer, src_buf._impl._inner[]._buffer
        )
        self._impl._stream[].copy_intra_device(
            dst_buf._impl._inner[]._buffer.view(),
            src_buf._impl._inner[]._buffer.view(),
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
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            self._impl.enqueue_copy(dst_ptr, src_buf._impl)
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        self._impl._stream[].copy_from_device(
            dst_ptr.unsafe_bitcast[UInt8](),
            src_buf._impl._inner[]._buffer.view(),
        )

    def enqueue_copy[
        dtype: DType
    ](
        self,
        dst_buf: DeviceBuffer[dtype],
        src: Span[mut=False, Scalar[dtype], _],
    ) raises:
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
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            self._impl.enqueue_copy(dst_buf._impl, src)
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        debug_assert(
            len(src) >= len(dst_buf),
            "source span length must be >= destination buffer length",
        )
        self.enqueue_copy(dst_buf, src.unsafe_ptr())

    def enqueue_copy[
        dtype: DType
    ](
        self, dst_buf: HostBuffer[dtype], src: Span[mut=False, Scalar[dtype], _]
    ) raises:
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
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            self._impl.enqueue_copy(dst_buf._impl, src)
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

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
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            self._impl.enqueue_copy(dst, src_buf._impl)
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        debug_assert(
            len(dst) >= len(src_buf),
            "destination span length must be >= source buffer length",
        )
        self.enqueue_copy(dst.unsafe_ptr(), src_buf)

    def enqueue_copy[
        dtype: DType
    ](
        self, dst: Span[mut=True, Scalar[dtype], _], src_buf: HostBuffer[dtype]
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
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            self._impl.enqueue_copy(dst, src_buf._impl)
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        debug_assert(
            len(dst) >= len(src_buf),
            "destination span length must be >= source buffer length",
        )
        self.enqueue_copy(dst.unsafe_ptr(), src_buf)

    def enqueue_copy_no_cross_stream_sync[
        dtype: DType
    ](self, dst_buf: DeviceBuffer[dtype], src_buf: DeviceBuffer[dtype]) raises:
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
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            self._impl.enqueue_copy_no_cross_stream_sync(
                dst_buf._impl, src_buf._impl
            )
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        _check_device_to_device_copy(self._impl, dst_buf._impl, src_buf._impl)
        _enqueue_copy_device_to_device(self._impl, dst_buf._impl, src_buf._impl)

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
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            self._impl.enqueue_memset(dst._impl, val)
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        self._impl._stream[].fill(
            dst._impl._inner[]._buffer.view(),
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
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            self._impl.enqueue_memset(dst._impl, val)
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        self._impl._stream[].fill(
            dst._impl._inner[]._buffer.view(),
            _memset_value_as_u64[dtype](val),
            UInt64(size_of[dtype]()),
        )

    def enqueue_cpu_range[
        FuncType: ImplicitlyCopyable & Deinitable & def(Int) -> None,
    ](self, func: FuncType, count: Int) raises:
        """Runs a function closure over a 1D range on a CPU `DeviceContext`.

        The function is called as `func(i)` for each `i` in `range(count)`.
        Instances run in parallel on the AsyncRT worker threads, but it is not
        guaranteed that all instances run at the same time.

        The HAL backend has no host-function enqueue plugin entry, so the range
        runs synchronously there: it returns when every instance is done, and
        `synchronize()` is then a no-op for this work.

        Parameters:
            FuncType: The type of function to execute.

        Args:
            func: The function closure to execute.
            count: The number of instances of the function to run.

        Raises:
            If self is not a CPU DeviceContext.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            self._impl.enqueue_cpu_range(func, count)
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        if self.api() != "cpu":
            raise Error(
                "enqueue_cpu_range is only supported on CPU DeviceContexts"
            )
        if count <= 0:
            return
        if count == 1:
            func(0)
            return

        @inline(.always)
        __async def run(i: Int) {imm} -> None:
            func(i)

        # Spread instances over the workers the way the legacy CPU context
        # does. Running one on this thread instead measured slower: a worker
        # that waits here leaves its own queued instance to be stolen late.
        var num_workers = max(parallelism_level(), 1)
        var group = TaskGroup()
        for i in range(count):
            group._create_task(run(i), desired_worker_id=i % num_workers)
        group.wait[origin_of(func)._mlir_origin]()

    def enqueue_cpu_function[
        FuncType: ImplicitlyCopyable & Deinitable & def() -> None,
    ](self, func: FuncType) raises:
        """Enqueues a function for execution on CPU.

        Parameters:
            FuncType: The function type.

        Args:
            func: The function to execute.

        Raises:
            If the operation fails.
            If self is not a CPU DeviceContext.
        """
        comptime assert (
            use_legacy_backend()
        ), "enqueue_cpu_function is only available on the legacy DeviceContext"
        self._impl.enqueue_cpu_function(func)

    # ===-------------------------------------------------------------------===#
    # Compiling and launching functions
    # ===-------------------------------------------------------------------===#

    def compile_function[
        declared_arg_types: TypeList[Trait=AnyType, ...],
        //,
        func: def(*args: *declared_arg_types) thin -> None,
        *,
        compile_options: StaticString = (Self.target.default_compile_options()),
        link_options: StaticString = "",
        dump_asm: _DumpPath = False,
        dump_llvm: _DumpPath = False,
        _dump_sass: _DumpPath = False,
        _ptxas_info_verbose: Bool = False,
    ](
        self,
        *,
        func_attribute: OptionalReg[FuncAttribute] = None,
        out result: DeviceFunction[
            func,
            Self.FunctionWrapper,
            target=Self.target,
            compile_options=compile_options,
            link_options=link_options,
            _ptxas_info_verbose=_ptxas_info_verbose,
        ],
    ) raises:
        """Compiles the provided function for execution on this device.

        Parameters:
            declared_arg_types: Types of the arguments of the kernel.
            func: The function to compile.
            compile_options: The compile options for the function.
            link_options: The link options for the function.
            dump_asm: To dump the compiled assembly, pass `True`, or a file
                path to dump to, or a function returning a file path.
            dump_llvm: To dump the generated LLVM code, pass `True`, or a file
                path to dump to, or a function returning a file path.
            _dump_sass: Only runs on NVIDIA targets, and requires CUDA toolkit
                to be installed. Pass `True`, or a file path to dump to, or a
                function returning a file path.
            _ptxas_info_verbose: Only runs on NVIDIA targets, and requires CUDA
                toolkit to be installed. Changes `dump_asm` to output verbose
                PTX assembly (default `False`).

        Args:
            func_attribute: An attribute to use when compiling the function.

        Returns:
            The compiled function.

        Raises:
            If the compilation fails.
        """
        comptime if use_legacy_backend():
            result = type_of(result)(
                _impl=self._impl.compile_function[
                    func,
                    compile_options=compile_options,
                    link_options=link_options,
                    dump_asm=dump_asm,
                    dump_llvm=dump_llvm,
                    _dump_sass=_dump_sass,
                    _ptxas_info_verbose=_ptxas_info_verbose,
                ](func_attribute=func_attribute)
            )
        else:
            # The HAL backend ignores the compile options, link options and
            # dump flags; they are accepted for parity with the legacy backend.
            result = type_of(result)(self, func_attribute=func_attribute)

    def compile_function[
        ClosureType: RegisterPassable & def() -> None,
    ](self) raises -> LegacyDeviceClosureFunction[ClosureType, Self.target]:
        """Compiles a closure as a kernel that takes no arguments.

        Launch the result with `enqueue_function(kernel, closure, ...)`. Only
        targets whose MAX plugin handles closure kernels support this, and the
        HAL backend has no closure kernels.

        Parameters:
            ClosureType: The closure's type, such as `type_of(closure)`.

        Returns:
            The compiled kernel.

        Raises:
            If compilation or loading fails.
        """
        comptime assert (
            use_legacy_backend()
        ), "closure kernels are only available on the legacy DeviceContext"
        return self._impl.compile_function[ClosureType]()

    def enqueue_function[
        ClosureType: RegisterPassable & def() -> None,
    ](
        self,
        f: LegacyDeviceClosureFunction[ClosureType, Self.target],
        closure: ClosureType,
        *,
        grid_dim: Dim,
        block_dim: Dim,
        location: Optional[SourceLocation] = None,
    ) raises:
        """Enqueues a closure kernel compiled by `compile_function[ClosureType]`.

        Parameters:
            ClosureType: The closure's type.

        Args:
            f: The compiled kernel.
            closure: The closure whose captures to launch with.
            grid_dim: The grid dimensions.
            block_dim: The block dimensions.
            location: Source location for the function call.

        Raises:
            If the launch fails.
        """
        comptime assert (
            use_legacy_backend()
        ), "closure kernels are only available on the legacy DeviceContext"
        self._impl.enqueue_function[ClosureType](
            f,
            closure,
            grid_dim=grid_dim,
            block_dim=block_dim,
            location=location,
        )

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
        func_attribute: OptionalReg[FuncAttribute] = None,
        out result: DeviceExternalFunction[target=Self.target],
    ) raises:
        """Loads a function from precompiled assembly.

        This loads an external GPU function from the provided assembly code
        (PTX/SASS) rather than compiling it from Mojo source. This is useful
        for integrating with existing CUDA/HIP code or for using specialized
        assembly optimizations.

        Parameters:
            ret_type: The return type of the function.
            declared_arg_types: Types of the arguments of the function.
            func: The function signature.

        Args:
            function_name: The name of the function in the assembly.
            asm: The assembly containing the function.
            func_attribute: An attribute to use when loading the function.
                The HAL backend ignores it.

        Returns:
            The loaded function.

        Raises:
            If the load fails.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            result = type_of(result)(
                _impl=self._impl.load_function[func](
                    function_name=function_name^,
                    asm=asm^,
                    func_attribute=func_attribute,
                )
            )
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        result = type_of(result)(self, function_name=function_name^, asm=asm^)

    @inline(.always)
    def enqueue_function[
        *Ts: AnyType
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
        """Enqueues a loaded external function for execution on this device.

        Parameters:
            Ts: The types of the arguments.

        Args:
            f: The loaded function to launch.
            args: The arguments to pass to the function.
            grid_dim: The grid dimensions.
            block_dim: The block dimensions.
            cluster_dim: The cluster dimensions.
            shared_mem_bytes: The dynamic shared memory in bytes.
            attributes: The launch attributes.
            constant_memory: The constant memory mappings.
            location: The source location of the call.

        Raises:
            If the launch fails.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            self._impl.enqueue_function(
                f._impl,
                *args,
                grid_dim=grid_dim,
                block_dim=block_dim,
                cluster_dim=cluster_dim,
                shared_mem_bytes=shared_mem_bytes,
                attributes=attributes^,
                constant_memory=constant_memory^,
                location=location,
            )
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        var hal_f: _HALDeviceExternalFunction = f._impl
        comptime if Self._HALHooks.Enqueuer.overrides:
            var launch = HalLaunchOptions(
                grid_dim,
                block_dim,
                cluster_dim,
                shared_mem_bytes,
                attributes^,
                constant_memory^,
                None,
            )
            Self._HALHooks.Enqueuer.enqueue_external(
                self._hal_stream()[], hal_f._inner[]._func_handle, launch, *args
            )
        else:
            _check_dim_in_range["DeviceContext.enqueue_function", "grid_dim"](
                grid_dim, location=call_location()
            )
            _check_dim_in_range["DeviceContext.enqueue_function", "block_dim"](
                block_dim, location=call_location()
            )
            f._call_with_pack(
                self._hal_stream(),
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
        *Ts: DevicePassable
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
        """Enqueues a compiled function for execution on this device.

        The function must have been compiled with type checking enabled (via
        `compile_function`), which verifies at compile time that the argument
        types match the declared types.

        ```mojo
        from max.gpu.host import DeviceContext

        def kernel(x: Int):
            print("Value:", x)

        with DeviceContext() as ctx:
            var compiled_func = ctx.compile_function[kernel]()
            ctx.enqueue_function(compiled_func, 42, grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```

        Parameters:
            Ts: The types of the arguments.

        Args:
            f: The compiled function to launch.
            args: The arguments to pass to the function.
            grid_dim: The grid dimensions.
            block_dim: The block dimensions.
            cluster_dim: The cluster dimensions.
            shared_mem_bytes: The dynamic shared memory in bytes.
            attributes: The launch attributes.
            constant_memory: The constant memory mappings.
            location: The source location of the call.

        Raises:
            If the launch fails.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            self._impl.enqueue_function(
                f._impl,
                *args,
                grid_dim=grid_dim,
                block_dim=block_dim,
                cluster_dim=cluster_dim,
                shared_mem_bytes=shared_mem_bytes,
                attributes=attributes^,
                constant_memory=constant_memory^,
                location=location,
            )
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        var hal_f: _HALDeviceFunction[...] = f._impl
        comptime if Self._HALHooks.Enqueuer.overrides:
            var launch = HalLaunchOptions(
                grid_dim,
                block_dim,
                cluster_dim,
                shared_mem_bytes,
                attributes^,
                constant_memory^,
                None,
            )
            Self._HALHooks.Enqueuer.enqueue_compiled[
                hal_f.Wrapper, hal_f.declared_arg_types
            ](self._hal_stream()[], hal_f._inner[]._func_handle, launch, *args)
        else:
            comptime if not Self._HALHooks.Launcher.overrides_grid_block:
                _check_dim_in_range[
                    "DeviceContext.enqueue_function", "grid_dim"
                ](grid_dim, location=call_location())
                _check_dim_in_range[
                    "DeviceContext.enqueue_function", "block_dim"
                ](block_dim, location=call_location())
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
        declared_arg_types: TypeList[Trait=AnyType, ...],
        HostArgs: TypeList[Trait=RegisterPassable, ...],
        //,
        func: def(*args: *declared_arg_types) thin -> None,
        *encoded_types: DevicePassable,
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
        """Compiles and enqueues a kernel for execution on this device.

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
            declared_arg_types: Types of the arguments of the kernel.
            HostArgs: Types of the host-only arguments.
            func: The function to compile and launch.
            encoded_types: The types of the device arguments.
            link_options: The link options for the function.
            dump_asm: To dump the compiled assembly, pass `True`, or a file
                path to dump to, or a function returning a file path.
            dump_llvm: To dump the generated LLVM code, pass `True`, or a file
                path to dump to, or a function returning a file path.
            _dump_sass: Only runs on NVIDIA targets, and requires CUDA toolkit
                to be installed.
            _ptxas_info_verbose: Only runs on NVIDIA targets, and requires CUDA
                toolkit to be installed.

        Args:
            encoded_args: The device arguments to pass to the kernel.
            host_args: The host-only arguments to pass to the kernel.
            grid_dim: The grid dimensions.
            block_dim: The block dimensions.
            cluster_dim: The cluster dimensions.
            shared_mem_bytes: The dynamic shared memory in bytes.
            attributes: The launch attributes.
            constant_memory: The constant memory mappings.
            func_attribute: An attribute to use when compiling the function.
            location: The source location of the call.

        Raises:
            If the compilation or the launch fails.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            # Leaving `host_args` out when there are none lets a plugin's own
            # `enqueue_function[func]` overload, which takes no host arguments,
            # win over the base one that raises for targets that launch through
            # their own overloads.
            comptime if HostArgs.length == 0:
                self._impl.enqueue_function[
                    func,
                    link_options=link_options,
                    dump_asm=dump_asm,
                    dump_llvm=dump_llvm,
                    _dump_sass=_dump_sass,
                    _ptxas_info_verbose=_ptxas_info_verbose,
                ](
                    *encoded_args,
                    grid_dim=grid_dim,
                    block_dim=block_dim,
                    cluster_dim=cluster_dim,
                    shared_mem_bytes=shared_mem_bytes,
                    attributes=attributes^,
                    constant_memory=constant_memory^,
                    func_attribute=func_attribute,
                    location=location,
                )
            else:
                self._impl.enqueue_function[
                    func,
                    link_options=link_options,
                    dump_asm=dump_asm,
                    dump_llvm=dump_llvm,
                    _dump_sass=_dump_sass,
                    _ptxas_info_verbose=_ptxas_info_verbose,
                ](
                    *encoded_args,
                    host_args=host_args,
                    grid_dim=grid_dim,
                    block_dim=block_dim,
                    cluster_dim=cluster_dim,
                    shared_mem_bytes=shared_mem_bytes,
                    attributes=attributes^,
                    constant_memory=constant_memory^,
                    func_attribute=func_attribute,
                    location=location,
                )
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        comptime if Self._HALHooks.Enqueuer.overrides:
            var launch = HalLaunchOptions(
                grid_dim,
                block_dim,
                cluster_dim,
                shared_mem_bytes,
                attributes^,
                constant_memory^,
                func_attribute,
            )
            Self._HALHooks.Enqueuer.enqueue_function[func](
                self._hal_stream()[], launch, *encoded_args, host_args=host_args
            )
        else:
            comptime if not Self._HALHooks.Launcher.overrides_grid_block:
                _check_dim_in_range[
                    "DeviceContext.enqueue_function", "grid_dim"
                ](grid_dim, location=call_location())
                _check_dim_in_range[
                    "DeviceContext.enqueue_function", "block_dim"
                ](block_dim, location=call_location())
            var gpu_kernel = self.compile_function[func](
                func_attribute=func_attribute
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

    def enqueue_function[
        FuncType: DevicePassable & def() -> None,
        //,
        dump_asm: _DumpPath = False,
        dump_llvm: _DumpPath = False,
        _dump_sass: _DumpPath = False,
        _ptxas_info_verbose: Bool = False,
    ](
        self,
        func: FuncType,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        func_attribute: OptionalReg[FuncAttribute] = None,
        location: Optional[SourceLocation] = None,
    ) raises:
        """Compiles and enqueues a capturing kernel whose captures are device
        passable.

        Parameters:
            FuncType: The type of the capturing kernel.
            dump_asm: To dump the compiled assembly, pass `True`, or a file
                path to dump to, or a function returning a file path.
            dump_llvm: To dump the generated LLVM code, pass `True`, or a file
                path to dump to, or a function returning a file path.
            _dump_sass: Only runs on NVIDIA targets, and requires CUDA toolkit
                to be installed.
            _ptxas_info_verbose: Only runs on NVIDIA targets, and requires CUDA
                toolkit to be installed.

        Args:
            func: The capturing kernel function to compile and launch.
            grid_dim: The grid dimensions.
            block_dim: The block dimensions.
            cluster_dim: The cluster dimensions.
            shared_mem_bytes: The dynamic shared memory in bytes.
            attributes: The launch attributes.
            constant_memory: The constant memory mappings.
            func_attribute: An attribute to use when compiling the function.
            location: The source location of the call.

        Raises:
            If the compilation or the launch fails.
        """
        comptime if use_legacy_backend():
            self._impl.enqueue_function[
                dump_asm=dump_asm,
                dump_llvm=dump_llvm,
                _dump_sass=_dump_sass,
                _ptxas_info_verbose=_ptxas_info_verbose,
            ](
                func,
                grid_dim,
                block_dim,
                cluster_dim=cluster_dim,
                shared_mem_bytes=shared_mem_bytes,
                attributes=attributes^,
                constant_memory=constant_memory^,
                func_attribute=func_attribute,
                location=location,
            )
        else:
            comptime assert conforms_to(
                FuncType, ImplicitlyCopyable & RegisterPassable
            ), (
                "the HAL backend launches only copyable, register-passable"
                " closures"
            )
            self._enqueue_closure_hal(
                rebind[
                    downcast[FuncType, ImplicitlyCopyable & RegisterPassable]
                ](func),
                grid_dim=grid_dim,
                block_dim=block_dim,
                cluster_dim=cluster_dim,
                shared_mem_bytes=shared_mem_bytes,
                attributes=attributes^,
                constant_memory=constant_memory^,
                func_attribute=func_attribute,
                location=location,
            )

    def enqueue_function[
        FuncType: ImplicitlyCopyable & RegisterPassable & def() -> None,
        //,
        dump_asm: _DumpPath = False,
        dump_llvm: _DumpPath = False,
        _dump_sass: _DumpPath = False,
        _ptxas_info_verbose: Bool = False,
    ](
        self,
        func: FuncType,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        func_attribute: OptionalReg[FuncAttribute] = None,
        location: Optional[SourceLocation] = None,
    ) raises where not conforms_to(FuncType, DevicePassable):
        """Compiles and enqueues a capturing kernel whose captures are plain
        register-passable values.

        Parameters:
            FuncType: The type of the capturing kernel.
            dump_asm: To dump the compiled assembly, pass `True`, or a file
                path to dump to, or a function returning a file path.
            dump_llvm: To dump the generated LLVM code, pass `True`, or a file
                path to dump to, or a function returning a file path.
            _dump_sass: Only runs on NVIDIA targets, and requires CUDA toolkit
                to be installed.
            _ptxas_info_verbose: Only runs on NVIDIA targets, and requires CUDA
                toolkit to be installed.

        Args:
            func: The capturing kernel function to compile and launch.
            grid_dim: The grid dimensions.
            block_dim: The block dimensions.
            cluster_dim: The cluster dimensions.
            shared_mem_bytes: The dynamic shared memory in bytes.
            attributes: The launch attributes.
            constant_memory: The constant memory mappings.
            func_attribute: An attribute to use when compiling the function.
            location: The source location of the call.

        Raises:
            If the compilation or the launch fails.
        """
        comptime if use_legacy_backend():
            self._impl.enqueue_function[
                dump_asm=dump_asm,
                dump_llvm=dump_llvm,
                _dump_sass=_dump_sass,
                _ptxas_info_verbose=_ptxas_info_verbose,
            ](
                func,
                grid_dim,
                block_dim,
                cluster_dim=cluster_dim,
                shared_mem_bytes=shared_mem_bytes,
                attributes=attributes^,
                constant_memory=constant_memory^,
                func_attribute=func_attribute,
                location=location,
            )
        else:
            self._enqueue_closure_hal(
                func,
                grid_dim=grid_dim,
                block_dim=block_dim,
                cluster_dim=cluster_dim,
                shared_mem_bytes=shared_mem_bytes,
                attributes=attributes^,
                constant_memory=constant_memory^,
                func_attribute=func_attribute,
                location=location,
            )

    @inline(.always)
    def _enqueue_closure_hal[
        FuncType: ImplicitlyCopyable & RegisterPassable & def() -> None,
    ](
        self,
        func: FuncType,
        *,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim],
        shared_mem_bytes: OptionalReg[Int],
        var attributes: List[LaunchAttribute],
        var constant_memory: List[ConstantMemoryMapping],
        func_attribute: OptionalReg[FuncAttribute],
        location: Optional[SourceLocation],
    ) raises where not use_legacy_backend():
        comptime if Self._HALHooks.Enqueuer.overrides:
            var launch = HalLaunchOptions(
                grid_dim,
                block_dim,
                cluster_dim,
                shared_mem_bytes,
                attributes^,
                constant_memory^,
                func_attribute,
            )
            Self._HALHooks.Enqueuer.enqueue_closure(
                self._hal_stream()[], launch, func
            )
        else:
            comptime if not Self._HALHooks.Launcher.overrides_grid_block:
                _check_dim_in_range[
                    "DeviceContext.enqueue_function", "grid_dim"
                ](grid_dim, location=call_location())
                _check_dim_in_range[
                    "DeviceContext.enqueue_function", "block_dim"
                ](block_dim, location=call_location())
            var gpu_kernel = DeviceFunction[
                FuncType.__call__, target=Self.target
            ](self, func_attribute=func_attribute)
            gpu_kernel._call_closure_checked(
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

    @inline(.always)
    def enqueue_function[
        declared_arg_types: TypeList[Trait=AnyType, ...],
        ConfigType: ExecutionConfig,
        HostArgs: TypeList[Trait=RegisterPassable, ...],
        //,
        func: def(*args: *declared_arg_types) thin -> None,
        *encoded_types: DevicePassable,
    ](
        self,
        config: ConfigType,
        *encoded_args: *encoded_types,
        host_args: Tuple[*HostArgs] = (),
        location: Optional[SourceLocation] = None,
    ) raises:
        """Compiles and enqueues a kernel with an execution configuration.

        Put capturing closures in `host_args`, after the encoded arguments,
        as for the keyword overload.

        You can pass the function directly to `enqueue_function` without
        compiling it first:

        ```mojo
        from max.gpu.host import DeviceContext
        from max.driver._hal.execution_config import GPUExecutionConfiguration

        def kernel():
            print("hello from the GPU")

        with DeviceContext() as ctx:
            ctx.enqueue_function[kernel](GPUExecutionConfiguration(grid_dim=Dim(1, 1, 1), block_dim=Dim(1, 1, 1)))
            ctx.synchronize()
        ```

        Parameters:
            declared_arg_types: Types of the arguments of the kernel.
            ConfigType: The type of the execution configuration.
            HostArgs: Types of the host-only arguments.
            func: The function to compile and launch.
            encoded_types: The types of the device arguments.

        Args:
            config: The execution configuration of the launch.
            encoded_args: The device arguments to pass to the kernel.
            host_args: The host-only arguments to pass to the kernel.
            location: The source location of the call.

        Raises:
            If the compilation or the launch fails.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            # The legacy backend doesn't support the full
            # `DeviceContext.ExecutionConfigType`, so the legacy
            # context's grid and block requirement is asserted here.
            comptime assert conforms_to(ConfigType, GridBlockExecutionConfig), (
                "the legacy DeviceContext launches only grid and block"
                " configurations"
            )
            self._impl.enqueue_function[func](
                config,
                *encoded_args,
                host_args=host_args,
                location=location,
            )
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        comptime if Self._HALHooks.Enqueuer.overrides:
            Self._HALHooks.Enqueuer.enqueue_function[func](
                self._hal_stream()[],
                config,
                *encoded_args,
                host_args=host_args,
            )
        else:
            var gpu_kernel = self.compile_function[func]()
            comptime if not Self._HALHooks.Launcher.overrides_grid_block:
                _check_config_dims["DeviceContext.enqueue_function"](
                    config, location=call_location()
                )
            comptime for idx in range(HostArgs.length):
                type_of(gpu_kernel)._check_trailing_host[
                    HostArgs[idx], encoded_types.length + idx
                ]()
            gpu_kernel._do_call_with_pack_checked(
                self._hal_stream(),
                config,
                *encoded_args,
                host_args=host_args,
                location=location.or_else(call_location()),
            )

    @inline(.always)
    def enqueue_function[
        ExecutionConfigType: ExecutionConfig,
        //,
        *Ts: DevicePassable,
    ](
        self,
        execution_config: ExecutionConfigType,
        f: DeviceFunction,
        *args: *Ts,
        location: Optional[SourceLocation] = None,
    ) raises:
        """Enqueues a pre-compiled checked function for execution on this device.

        Only available on the HAL `DeviceContext`.

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
        comptime assert not use_legacy_backend(), (
            "enqueue_function with an execution config is only available on a"
            " HAL DeviceContext"
        )
        var hal_f: _HALDeviceFunction[...] = f._impl
        comptime if Self._HALHooks.Enqueuer.overrides:
            Self._HALHooks.Enqueuer.enqueue_compiled[
                hal_f.Wrapper, hal_f.declared_arg_types
            ](
                self._hal_stream()[],
                hal_f._inner[]._func_handle,
                execution_config,
                *args,
            )
        else:
            comptime if not Self._HALHooks.Launcher.overrides_grid_block:
                _check_config_dims["DeviceContext.enqueue_function"](
                    execution_config, location=call_location()
                )
            f._do_call_with_pack_checked(
                self._hal_stream(),
                execution_config,
                *args,
                location=location.or_else(call_location()),
            )

    @inline(.always)
    def enqueue_function[
        FuncType: ImplicitlyCopyable & RegisterPassable & def() -> None,
        ExecutionConfigType: ExecutionConfig,
        //,
    ](
        self,
        execution_config: ExecutionConfigType,
        func: FuncType,
        location: Optional[SourceLocation] = None,
    ) raises:
        """Compiles and enqueues a capturing kernel for execution on this device.

        Only available on the HAL `DeviceContext`.

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
        comptime assert not use_legacy_backend(), (
            "enqueue_function with an execution config and a closure is only"
            " available on the HAL DeviceContext"
        )
        comptime if Self._HALHooks.Enqueuer.overrides:
            Self._HALHooks.Enqueuer.enqueue_closure(
                self._hal_stream()[], execution_config, func
            )
        else:
            comptime if not Self._HALHooks.Launcher.overrides_grid_block:
                _check_config_dims["DeviceContext.enqueue_function"](
                    execution_config, location=call_location()
                )

            var gpu_kernel = DeviceFunction[
                FuncType.__call__, target=Self.target
            ](self)

            gpu_kernel._do_call_closure_checked(
                self._hal_stream(),
                execution_config,
                func,
                location=location.or_else(call_location()),
            )

    def execution_time[
        FuncType: def(Self) raises -> None,
    ](self, func: FuncType, num_iters: Int) raises -> Int:
        """Measures the execution time, in nanoseconds, of `func(self)` run
        `num_iters` times (host monotonic clock, bracketed by `synchronize()`).

        It runs the function for the specified number of iterations and
        returns the total elapsed time in nanoseconds.

        Parameters:
            FuncType: The body function type.

        Args:
            func: The closure carrying the captured state of the body function.
            num_iters: The number of iterations to run the function.

        Returns:
            The total elapsed time in nanoseconds for all iterations.

        Raises:
            If the timing fails.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)

            def run_legacy(inner: Self._Legacy) raises {imm}:
                func(Self(_impl=inner.copy()))

            return self._impl.execution_time(run_legacy, num_iters)
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        self.synchronize()
        var start = monotonic()
        for _ in range(num_iters):
            func(self)
        self.synchronize()
        return Int(monotonic() - start)

    def execution_time[
        FuncType: def() raises -> None,
    ](self, func: FuncType, num_iters: Int) raises -> Int:
        """Measures the execution time of a function over multiple iterations.

        This overload is for a function that doesn't take the DeviceContext.

        Parameters:
            FuncType: The body function type.

        Args:
            func: The closure carrying the captured state of the body function.
            num_iters: The number of iterations to run the function.

        Returns:
            The total elapsed time in nanoseconds for all iterations.

        Raises:
            If the timing fails.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            return self._impl.execution_time(func, num_iters)
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        self.synchronize()
        var start = monotonic()
        for _ in range(num_iters):
            func()
        self.synchronize()
        return Int(monotonic() - start)

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
            If the timing fails.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)

            def run_legacy(inner: Self._Legacy, i: Int) raises {imm}:
                func(Self(_impl=inner.copy()), i)

            return self._impl.execution_time_iter(run_legacy, num_iters)
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        self.synchronize()
        var start = monotonic()
        for i in range(num_iters):
            func(self, i)
        self.synchronize()
        return Int(monotonic() - start)

    # ===-------------------------------------------------------------------===#
    # Peer access and waits
    # ===-------------------------------------------------------------------===#

    @deprecated(
        "Use `DeviceContext.device().can_access_peer(peer.device())` instead."
    )
    def can_access(self, peer: DeviceContext) raises -> Bool:
        """Returns whether this context can access `peer`'s memory.

        Args:
            peer: The peer context.

        Returns:
            True if peer access is available.

        Raises:
            If the query fails.
        """
        return self.device().can_access_peer(peer.device())

    def enable_peer_access(self, peer: DeviceContext) raises:
        """Enables direct memory access to the peer device.

        Args:
            peer: The peer device context whose memory becomes accessible.

        Raises:
            If enabling peer access fails: the build or `peer`'s driver lacks
            peer access, or the plugin refuses the pair.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            self._impl.enable_peer_access(peer._impl)
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        if not (self._impl._driver is peer._impl._driver):
            raise Error("Peer access not implemented")
        comptime Ext = type_of(self._impl._context[].extensions())
        comptime if conforms_to(Ext, SupportsPeerAccessV1):
            self._impl._context[].extensions().enable_peer_access(
                self._impl._context[].raw_driver()[],
                self._impl._context[].handle(),
                peer._impl._context[].handle(),
            )
            var src = Int(self._impl._device[].id)
            var dst = Int(peer._impl._device[].id)
            var state = _PEER_STATE.get_or_create_ptr()
            with BlockingScopedLock(state[].lock):
                state[].record_enabled(
                    src, self._impl._context, dst, peer._impl._context
                )
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
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            LegacyDeviceContext[Self.target].enable_all_peer_access()
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        comptime device_spec = HALDeviceContext[Self.target].device_spec
        comptime Ext = _HALDevice[device_spec].Extensions
        comptime if conforms_to(Ext, SupportsPeerAccessV1):
            var maybe_driver = _GPU_DRIVER.get_or_create_ptr()
            __guard case .Some(driver) = maybe_driver[] else:
                return
            var count = Int(driver[].device_count())
            if count <= 1:
                return
            var state = _PEER_STATE.get_or_create_ptr()
            if state[].all_enabled.load() != 0:
                return
            var contexts = List[ArcPointer[Context[device_spec]]]()
            for i in range(count):
                # Matches `__init__`'s lookup, so the contexts are shared.
                var device = rebind[ArcPointer[_HALDevice[device_spec]]](
                    driver[].get_device_dynamic[Optional[Int64](0)](Int64(i))
                )
                # A non-vendor `.so` may lack the group; stay a no-op.
                if not device[].extensions().supports_peer_access():
                    return
                contexts.append(device[].get_context())
            # Concurrent callers may repeat a grant; the plugin treats an
            # already-enabled pair as success.
            for i in range(count):
                for j in range(count):
                    if i == j:
                        continue
                    contexts[i][].extensions().enable_peer_access(
                        contexts[i][].raw_driver()[],
                        contexts[i][].handle(),
                        contexts[j][].handle(),
                    )
                    with BlockingScopedLock(state[].lock):
                        state[].record_enabled(i, contexts[i], j, contexts[j])

    @staticmethod
    def all_peer_access_enabled() raises -> Bool:
        """Returns whether peer access is enabled between every pair of devices.

        Returns:
            True when all peer-access pairs are enabled.

        Raises:
            If the query fails.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            return LegacyDeviceContext[Self.target].all_peer_access_enabled()
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        return _PEER_STATE.get_or_create_ptr()[].all_enabled.load() != 0

    def supports_multicast(self) raises -> Bool:
        """Returns True if this device supports multicast memory mappings.

        Returns:
            True when multicast is available. Always False on the HAL backend,
            where multicast is not implemented.

        Raises:
            If the query fails. The HAL backend never raises.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            return self._impl.supports_multicast()
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        return False

    def enqueue_wait_for(self, other: DeviceContext) raises:
        """Enqueues a wait for all work currently pending on `other`.

        Args:
            other: The context to wait for.

        Raises:
            If the operation fails.
        """
        comptime if use_legacy_backend():
            self._impl.enqueue_wait_for(other._impl)
        else:
            self._impl.enqueue_wait_for(other._impl)


struct _DeviceContextScope[target: CompilationTarget]:
    """Makes a device's driver context current for a `with` block."""

    comptime _Legacy = _LegacyDeviceContextScope[Self.target]
    comptime _HAL = _HALDeviceContextScope
    comptime _Impl = Self._Legacy if use_legacy_backend() else Self._HAL

    var _impl: Self._Impl

    def __init__(out self, ctx: DeviceContext[Self.target]):
        """Creates a scope over `ctx`.

        Args:
            ctx: The device context whose driver context to make current.
        """
        comptime if use_legacy_backend():
            self._impl = Self._Legacy(ctx._impl)
        else:
            self._impl = Self._HAL(ctx._impl)

    def __enter__(mut self) raises -> DeviceContext[Self.target]:
        """Makes the driver context current.

        Returns:
            The device context of this scope.

        Raises:
            If the driver context cannot be made current.
        """
        comptime if use_legacy_backend():
            return DeviceContext[Self.target](_impl=self._impl.__enter__())
        else:
            return rebind_var[DeviceContext[Self.target]](
                self._impl.__enter__()
            )

    def __exit__(mut self) raises:
        """Restores the previous driver context.

        Raises:
            If the previous driver context cannot be restored.
        """
        comptime if use_legacy_backend():
            self._impl.__exit__()
        else:
            self._impl.__exit__()
