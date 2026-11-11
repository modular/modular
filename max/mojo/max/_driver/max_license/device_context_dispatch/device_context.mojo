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

from std.ffi import c_size_t
from std.memory import Pointer
from std.os import abort
from std.reflection import call_location
from std.sys.info import CompilationTarget

from max._driver.max_license.device_context_dispatch.function import (
    _DispatchEnqueuer,
)
from max._driver.max_license.device_context_hal.device_context_hal import (
    HALDeviceContext,
    _HALDeviceContextScope,
)
from max.gpu.host._driver_handles import (
    _DeviceContextPtr as _HALDeviceContextPtr,
)
from max.gpu.host._legacy.device_attribute import DeviceAttribute
from max.gpu.host._legacy.device_context import (
    LegacyDeviceContext,
    _LegacyDeviceContextScope,
)
from max.gpu.host._driver_handles import (
    _DeviceContextPtr as _LegacyDeviceContextPtr,
)
from max.gpu.host._overlay import HAL_DEVICE_CONTEXT
from max.gpu.host._driver_common import (
    StreamPriorityRange as _OverlayStreamPriorityRange,
)
from max.gpu.host._driver_handles import (
    _DeviceContextPtr as _OverlayDeviceContextPtr,
)
from max.gpu.host.info import default_accelerator
from max.gpu.host.launch_config import ExecutionConfig
from .host_types import (
    CompletionFlag,
    Device,
    DeviceEvent,
    DeviceStream,
)
from max._driver.max_license.device_context_dispatch.device_buffer import (
    DeviceBuffer,
)
from max._driver.max_license.device_context_dispatch.host_buffer import (
    HostBuffer,
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
            self._impl = rebind_var[Self._Impl](
                Self._Legacy(device_id=device_id, api=api^)
            )
        else:
            self._impl = rebind_var[Self._Impl](
                Self._HAL(device_id=device_id, api=api^)
            )

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
        comptime if use_legacy_backend():
            return rebind_var[DeviceContext.CPU](
                rebind[Self._Legacy](self._impl).assert_cpu_target()
            )
        else:
            return rebind_var[DeviceContext.CPU](
                rebind[Self._HAL](self._impl).assert_cpu_target()
            )

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
            return rebind[DeviceContext[]](
                rebind[Self._Legacy](self._impl).assert_default_target()
            )
        else:
            return rebind[DeviceContext[]](
                rebind[Self._HAL](self._impl).assert_default_target()
            )

    @inline(.always)
    def into_unknown_target(self) -> DeviceContext.Unknown:
        """Erases this DeviceContext's static target.

        For code that only uses the context's handle, so it neither needs nor
        should assume a target.

        Returns:
            This device context with an unknown target.
        """
        comptime if use_legacy_backend():
            return rebind_var[DeviceContext.Unknown](
                rebind[Self._Legacy](self._impl).into_unknown_target()
            )
        else:
            return rebind_var[DeviceContext.Unknown](
                rebind[Self._HAL](self._impl).into_unknown_target()
            )

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
    def __init__(out self, ctx_ptr: _OverlayDeviceContextPtr[mut=True]):
        """Wraps the runtime's native context handle.

        Args:
            ctx_ptr: The native handle of the context.
        """
        comptime if use_legacy_backend():
            self._impl = rebind_var[Self._Impl](
                Self._Legacy(rebind[_LegacyDeviceContextPtr[mut=True]](ctx_ptr))
            )
        else:
            self._impl = rebind_var[Self._Impl](
                Self._HAL(rebind[_HALDeviceContextPtr[mut=True]](ctx_ptr))
            )

    @doc_hidden
    def __init__(out self, handle: OpaquePointer[MutUntrackedOrigin]):
        """Wraps an opaque pointer to the runtime's native context.

        Args:
            handle: The opaque pointer to the native context.
        """
        comptime if use_legacy_backend():
            self._impl = rebind_var[Self._Impl](
                Self._Legacy(
                    rebind[OpaquePointer[UntrackedOrigin[mut=True]]](handle)
                )
            )
        else:
            self._impl = rebind_var[Self._Impl](Self._HAL(handle))

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
        return rebind[Self._Legacy](self._impl)

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
        return rebind[Self._HAL](self._impl)

    @doc_hidden
    def _handle_ptr(self) -> _OverlayDeviceContextPtr[mut=True]:
        """Returns the backend's native context handle.

        For the AsyncRT calls that the shared kernel-load and capture helpers
        make on the context directly.

        Returns:
            The handle of the backend context this wrapper holds.
        """
        comptime if use_legacy_backend():
            return rebind[_OverlayDeviceContextPtr[mut=True]](
                rebind[Self._Legacy](self._impl)._handle
            )
        else:
            return rebind[_OverlayDeviceContextPtr[mut=True]](
                rebind[Self._HAL](self._impl)._handle
            )

    # ===-------------------------------------------------------------------===#
    # Function enqueuer
    # ===-------------------------------------------------------------------===#

    comptime _LegacyEnqueuer = Self._Legacy
    comptime _HALEnqueuer = Self._HAL

    def _legacy_enqueuer(self) -> Self._LegacyEnqueuer:
        comptime assert (
            use_legacy_backend()
        ), "this context is not on the legacy backend"
        return rebind[Self._Legacy](self._impl)

    def _hal_enqueuer(self) -> Self._HALEnqueuer:
        comptime assert (
            not use_legacy_backend()
        ), "this context is not on the HAL backend"
        return rebind[Self._HAL](self._impl)

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
            rebind[Self._Legacy](self._impl).synchronize()
        else:
            rebind[Self._HAL](self._impl).synchronize()

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
        """Returns the free and total memory of this device.

        Returns:
            A tuple of `(free, total)` bytes.

        Raises:
            If querying the device memory fails.
        """
        comptime if use_legacy_backend():
            return rebind[Self._Legacy](self._impl).get_memory_info()
        else:
            return rebind[Self._HAL](self._impl).get_memory_info()

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
        comptime if use_legacy_backend():
            return rebind[Self._Legacy](self._impl).compute_capability()
        else:
            return rebind[Self._HAL](self._impl).compute_capability()

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
        comptime if use_legacy_backend():
            rebind[Self._Legacy](self._impl).set_as_current()
        else:
            rebind[Self._HAL](self._impl).set_as_current()

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
            return rebind[Self._Legacy](self._impl).api()
        else:
            return rebind[Self._HAL](self._impl).api()

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
        """Returns the version of the device API in use.

        Returns:
            The API version.

        Raises:
            If querying the API version fails.
        """
        comptime if use_legacy_backend():
            return rebind[Self._Legacy](self._impl).get_api_version()
        else:
            return rebind[Self._HAL](self._impl).get_api_version()

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
        """
        comptime if use_legacy_backend():
            return rebind[Self._Legacy](self._impl).is_compatible()
        else:
            return rebind[Self._HAL](self._impl).is_compatible()

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
        """Returns whether this context is currently capturing a graph.

        Returns:
            True while a graph capture is in progress.

        Raises:
            If the query fails.
        """
        comptime if use_legacy_backend():
            return rebind[Self._Legacy](self._impl).is_graph_capturing()
        else:
            return rebind[Self._HAL](self._impl).is_graph_capturing()

    @staticmethod
    def number_of_devices(*, var api: String = Self._default_api()) -> Int:
        """Returns the number of devices available for the given API.

        Args:
            api: The device API to count devices for.

        Returns:
            The number of devices.
        """
        comptime if use_legacy_backend():
            return LegacyDeviceContext[Self.target].number_of_devices(api=api^)
        else:
            return HALDeviceContext[Self.target].number_of_devices(api=api^)

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
            return DeviceStream(
                _impl=rebind_var[DeviceStream._Impl](
                    rebind[Self._Legacy](self._impl).stream()
                )
            )
        else:
            return DeviceStream(
                _impl=rebind_var[DeviceStream._Impl](
                    rebind[Self._HAL](self._impl).stream()
                )
            )

    def create_stream(self, *, priority: Int = 0) raises -> DeviceStream:
        """Creates a new stream on this context's device.

        Args:
            priority: Requested stream priority.

        Returns:
            The new `DeviceStream`.

        Raises:
            If the stream cannot be created.
        """
        comptime if use_legacy_backend():
            return DeviceStream(
                _impl=rebind_var[DeviceStream._Impl](
                    rebind[Self._Legacy](self._impl).create_stream(
                        priority=priority
                    )
                )
            )
        else:
            return DeviceStream(
                _impl=rebind_var[DeviceStream._Impl](
                    rebind[Self._HAL](self._impl).create_stream(
                        priority=priority
                    )
                )
            )

    def create_event(self) raises -> DeviceEvent:
        """Creates a new event for cross-stream synchronization.

        Returns:
            The new `DeviceEvent`.

        Raises:
            If the event cannot be created.
        """
        comptime if use_legacy_backend():
            return DeviceEvent(
                _impl=rebind_var[DeviceEvent._Impl](
                    rebind[Self._Legacy](self._impl).create_event()
                )
            )
        else:
            return DeviceEvent(
                _impl=rebind_var[DeviceEvent._Impl](
                    rebind[Self._HAL](self._impl).create_event()
                )
            )

    def allocate_completion_flag(self) raises -> CompletionFlag:
        """Allocates a device-side completion flag.

        Returns:
            The new `CompletionFlag`.

        Raises:
            If the allocation fails.
        """
        comptime if use_legacy_backend():
            return CompletionFlag(
                _impl=rebind_var[CompletionFlag._Impl](
                    rebind[Self._Legacy](self._impl).allocate_completion_flag()
                )
            )
        else:
            return CompletionFlag(
                _impl=rebind_var[CompletionFlag._Impl](
                    rebind[Self._HAL](self._impl).allocate_completion_flag()
                )
            )

    def supports_flag_wait(self) -> Bool:
        """Returns whether this device supports waiting on a completion flag.

        Returns:
            True when flag waits are supported.
        """
        comptime if use_legacy_backend():
            return rebind[Self._Legacy](self._impl).supports_flag_wait()
        else:
            return rebind[Self._HAL](self._impl).supports_flag_wait()

    def num_streams(self) -> Int:
        """Returns the number of streams available on this device context.

        Returns:
            The number of streams available on this device context.
        """
        comptime assert (
            use_legacy_backend()
        ), "num_streams is only available on the legacy DeviceContext"
        return rebind[Self._Legacy](self._impl).num_streams()

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
        return rebind_var[DeviceContext[Self.target]](
            rebind[Self._Legacy](self._impl).select_stream(stream_id)
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
            _impl=rebind_var[DeviceStream._Impl](
                rebind[Self._Legacy](self._impl).create_external_stream(
                    external_stream
                )
            )
        )

    def stream_priority_range(self) raises -> _OverlayStreamPriorityRange:
        """Returns the range of stream priorities this device supports.

        Returns:
            The least and greatest stream priorities.

        Raises:
            If the query fails.
        """
        comptime if use_legacy_backend():
            return rebind_var[_OverlayStreamPriorityRange](
                rebind[Self._Legacy](self._impl).stream_priority_range()
            )
        else:
            return rebind_var[_OverlayStreamPriorityRange](
                rebind[Self._HAL](self._impl).stream_priority_range()
            )

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
        return rebind[Self._Legacy](self._impl) == rebind[Self._Legacy](
            other._impl
        )

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
        return rebind[Self._Legacy](self._impl) != rebind[Self._Legacy](
            other._impl
        )

    # ===-------------------------------------------------------------------===#
    # Device views
    # ===-------------------------------------------------------------------===#

    def device(ref self) -> Device[Self.target, ImmOrigin(origin_of(self))]:
        """Returns a view of the device this context runs on.

        Returns:
            A `Device` that borrows this context.
        """
        comptime if use_legacy_backend():
            return Device[Self.target, ImmOrigin(origin_of(self))](
                _impl=rebind_var[
                    Device[Self.target, ImmOrigin(origin_of(self))]._Impl
                ](rebind[Self._Legacy](self._impl).device())
            )
        else:
            return Device[Self.target, ImmOrigin(origin_of(self))](
                _impl=rebind_var[
                    Device[Self.target, ImmOrigin(origin_of(self))]._Impl
                ](rebind[Self._HAL](self._impl).device())
            )

    # ===-------------------------------------------------------------------===#
    # Buffer allocation
    # ===-------------------------------------------------------------------===#

    def enqueue_create_buffer[
        dtype: DType
    ](self, size: Int) raises -> DeviceBuffer[dtype]:
        """Enqueues an allocation of a device buffer of `size` elements.

        Parameters:
            dtype: The element dtype of the buffer.

        Args:
            size: Number of elements to allocate.

        Returns:
            The allocated device buffer.

        Raises:
            If the allocation fails.
        """
        comptime if use_legacy_backend():
            return rebind_var[DeviceBuffer[dtype]](
                DeviceBuffer[dtype](
                    _impl=rebind_var[DeviceBuffer[dtype]._Impl](
                        rebind[Self._Legacy](self._impl).enqueue_create_buffer[
                            dtype
                        ](size)
                    )
                )
            )
        else:
            return rebind_var[DeviceBuffer[dtype]](
                DeviceBuffer[dtype](
                    _impl=rebind_var[DeviceBuffer[dtype]._Impl](
                        rebind[Self._HAL](self._impl).enqueue_create_buffer[
                            dtype
                        ](size)
                    )
                )
            )

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
        comptime if use_legacy_backend():
            return rebind_var[DeviceBuffer[dtype]](
                DeviceBuffer[dtype](
                    _impl=rebind_var[DeviceBuffer[dtype]._Impl](
                        rebind[Self._Legacy](self._impl).create_buffer_sync[
                            dtype
                        ](size)
                    )
                )
            )
        else:
            return rebind_var[DeviceBuffer[dtype]](
                DeviceBuffer[dtype](
                    _impl=rebind_var[DeviceBuffer[dtype]._Impl](
                        rebind[Self._HAL](self._impl).create_buffer_sync[dtype](
                            size
                        )
                    )
                )
            )

    def enqueue_create_host_buffer[
        dtype: DType
    ](self, size: Int) raises -> HostBuffer[dtype]:
        """Enqueues an allocation of a host buffer of `size` elements.

        Parameters:
            dtype: The element dtype of the buffer.

        Args:
            size: Number of elements to allocate.

        Returns:
            The allocated host buffer.

        Raises:
            If the allocation fails.
        """
        comptime if use_legacy_backend():
            return rebind_var[HostBuffer[dtype]](
                HostBuffer[dtype](
                    _impl=rebind_var[HostBuffer[dtype]._Impl](
                        rebind[Self._Legacy](
                            self._impl
                        ).enqueue_create_host_buffer[dtype](size)
                    )
                )
            )
        else:
            return rebind_var[HostBuffer[dtype]](
                HostBuffer[dtype](
                    _impl=rebind_var[HostBuffer[dtype]._Impl](
                        rebind[Self._HAL](
                            self._impl
                        ).enqueue_create_host_buffer[dtype](size)
                    )
                )
            )

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
        return rebind_var[DeviceBuffer[dtype]](
            DeviceBuffer[dtype](
                _impl=rebind_var[DeviceBuffer[dtype]._Impl](
                    rebind[Self._Legacy](self._impl).wrap_host_memory(
                        host_ptr, size
                    )
                )
            )
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
            dst_ptr: Pointer to copy to.
            src_ptr: Pointer to copy from.
            size: Number of elements (of the specified `DType`) to copy.

        Raises:
            If the operation fails.
        """
        comptime if use_legacy_backend():
            rebind[Self._Legacy](self._impl).enqueue_copy(
                dst_ptr, src_ptr, size
            )
        else:
            rebind[Self._HAL](self._impl).enqueue_copy(dst_ptr, src_ptr, size)

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
        comptime if use_legacy_backend():
            rebind[Self._Legacy](self._impl).enqueue_copy(
                rebind[DeviceBuffer[dtype]._Legacy](dst_buf._impl), src_ptr
            )
        else:
            rebind[Self._HAL](self._impl).enqueue_copy(
                rebind[DeviceBuffer[dtype]._HAL](dst_buf._impl), src_ptr
            )

    def enqueue_copy[
        dtype: DType
    ](
        self,
        dst_buf: HostBuffer[dtype],
        src_ptr: Pointer[mut=False, Scalar[dtype], _],
    ) raises:
        """Enqueues an async copy from a device pointer to a host buffer.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_buf: Host buffer to copy to.
            src_ptr: Device pointer to copy from.

        Raises:
            If the operation fails.
        """
        comptime if use_legacy_backend():
            rebind[Self._Legacy](self._impl).enqueue_copy(
                rebind[HostBuffer[dtype]._Legacy](dst_buf._impl), src_ptr
            )
        else:
            rebind[Self._HAL](self._impl).enqueue_copy(
                rebind[HostBuffer[dtype]._HAL](dst_buf._impl), src_ptr
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
        comptime if use_legacy_backend():
            rebind[Self._Legacy](self._impl).enqueue_copy(
                dst_ptr, rebind[DeviceBuffer[dtype]._Legacy](src_buf._impl)
            )
        else:
            rebind[Self._HAL](self._impl).enqueue_copy(
                dst_ptr, rebind[DeviceBuffer[dtype]._HAL](src_buf._impl)
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
        comptime if use_legacy_backend():
            rebind[Self._Legacy](self._impl).enqueue_copy(
                rebind[DeviceBuffer[dtype]._Legacy](dst_buf._impl),
                rebind[DeviceBuffer[dtype]._Legacy](src_buf._impl),
            )
        else:
            rebind[Self._HAL](self._impl).enqueue_copy(
                rebind[DeviceBuffer[dtype]._HAL](dst_buf._impl),
                rebind[DeviceBuffer[dtype]._HAL](src_buf._impl),
            )

    def enqueue_copy[
        dtype: DType
    ](self, dst_buf: DeviceBuffer[dtype], src_buf: HostBuffer[dtype]) raises:
        """Enqueues an async copy from host memory to a device buffer.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_buf: Device buffer to copy to.
            src_buf: Host buffer to copy from.

        Raises:
            If the operation fails.
        """
        comptime if use_legacy_backend():
            rebind[Self._Legacy](self._impl).enqueue_copy(
                rebind[DeviceBuffer[dtype]._Legacy](dst_buf._impl),
                rebind[HostBuffer[dtype]._Legacy](src_buf._impl),
            )
        else:
            rebind[Self._HAL](self._impl).enqueue_copy(
                rebind[DeviceBuffer[dtype]._HAL](dst_buf._impl),
                rebind[HostBuffer[dtype]._HAL](src_buf._impl),
            )

    def enqueue_copy[
        dtype: DType
    ](self, dst_buf: HostBuffer[dtype], src_buf: DeviceBuffer[dtype]) raises:
        """Enqueues an async copy from a device buffer to host memory.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_buf: Host buffer to copy to.
            src_buf: Device buffer to copy from.

        Raises:
            If the operation fails.
        """
        comptime if use_legacy_backend():
            rebind[Self._Legacy](self._impl).enqueue_copy(
                rebind[HostBuffer[dtype]._Legacy](dst_buf._impl),
                rebind[DeviceBuffer[dtype]._Legacy](src_buf._impl),
            )
        else:
            rebind[Self._HAL](self._impl).enqueue_copy(
                rebind[HostBuffer[dtype]._HAL](dst_buf._impl),
                rebind[DeviceBuffer[dtype]._HAL](src_buf._impl),
            )

    def enqueue_copy[
        dtype: DType
    ](self, dst_buf: HostBuffer[dtype], src_buf: HostBuffer[dtype]) raises:
        """Enqueues an async copy from one host buffer to another.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_buf: Host buffer to copy to.
            src_buf: Host buffer to copy from.

        Raises:
            If the operation fails.
        """
        comptime if use_legacy_backend():
            rebind[Self._Legacy](self._impl).enqueue_copy(
                rebind[HostBuffer[dtype]._Legacy](dst_buf._impl),
                rebind[HostBuffer[dtype]._Legacy](src_buf._impl),
            )
        else:
            rebind[Self._HAL](self._impl).enqueue_copy(
                rebind[HostBuffer[dtype]._HAL](dst_buf._impl),
                rebind[HostBuffer[dtype]._HAL](src_buf._impl),
            )

    def enqueue_copy[
        dtype: DType
    ](
        self,
        dst_ptr: Pointer[mut=True, Scalar[dtype], _],
        src_buf: HostBuffer[dtype],
    ) raises:
        """Enqueues an async copy from host memory to a host pointer.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_ptr: Host pointer to copy to.
            src_buf: Host buffer to copy from.

        Raises:
            If the operation fails.
        """
        comptime if use_legacy_backend():
            rebind[Self._Legacy](self._impl).enqueue_copy(
                dst_ptr, rebind[HostBuffer[dtype]._Legacy](src_buf._impl)
            )
        else:
            rebind[Self._HAL](self._impl).enqueue_copy(
                dst_ptr, rebind[HostBuffer[dtype]._HAL](src_buf._impl)
            )

    def enqueue_copy[
        dtype: DType
    ](
        self,
        dst_buf: DeviceBuffer[dtype],
        src: Span[mut=False, Scalar[dtype], _],
    ) raises:
        """Enqueues an async copy from a host span to a device buffer.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_buf: Device buffer to copy to.
            src: Host span to copy from.

        Raises:
            If the operation fails.
        """
        comptime if use_legacy_backend():
            rebind[Self._Legacy](self._impl).enqueue_copy(
                rebind[DeviceBuffer[dtype]._Legacy](dst_buf._impl), src
            )
        else:
            rebind[Self._HAL](self._impl).enqueue_copy(
                rebind[DeviceBuffer[dtype]._HAL](dst_buf._impl), src
            )

    def enqueue_copy[
        dtype: DType
    ](
        self, dst_buf: HostBuffer[dtype], src: Span[mut=False, Scalar[dtype], _]
    ) raises:
        """Enqueues an async copy from a host span to a host buffer.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_buf: Host buffer to copy to.
            src: Host span to copy from.

        Raises:
            If the operation fails.
        """
        comptime if use_legacy_backend():
            rebind[Self._Legacy](self._impl).enqueue_copy(
                rebind[HostBuffer[dtype]._Legacy](dst_buf._impl), src
            )
        else:
            rebind[Self._HAL](self._impl).enqueue_copy(
                rebind[HostBuffer[dtype]._HAL](dst_buf._impl), src
            )

    def enqueue_copy[
        dtype: DType
    ](
        self,
        dst: Span[mut=True, Scalar[dtype], _],
        src_buf: DeviceBuffer[dtype],
    ) raises:
        """Enqueues an async copy from a device buffer to a host span.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst: Host span to copy to.
            src_buf: Device buffer to copy from.

        Raises:
            If the operation fails.
        """
        comptime if use_legacy_backend():
            rebind[Self._Legacy](self._impl).enqueue_copy(
                dst, rebind[DeviceBuffer[dtype]._Legacy](src_buf._impl)
            )
        else:
            rebind[Self._HAL](self._impl).enqueue_copy(
                dst, rebind[DeviceBuffer[dtype]._HAL](src_buf._impl)
            )

    def enqueue_copy[
        dtype: DType
    ](
        self, dst: Span[mut=True, Scalar[dtype], _], src_buf: HostBuffer[dtype]
    ) raises:
        """Enqueues an async copy from a host buffer to a host span.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst: Host span to copy to.
            src_buf: Host buffer to copy from.

        Raises:
            If the operation fails.
        """
        comptime if use_legacy_backend():
            rebind[Self._Legacy](self._impl).enqueue_copy(
                dst, rebind[HostBuffer[dtype]._Legacy](src_buf._impl)
            )
        else:
            rebind[Self._HAL](self._impl).enqueue_copy(
                dst, rebind[HostBuffer[dtype]._HAL](src_buf._impl)
            )

    def enqueue_copy_no_cross_stream_sync[
        dtype: DType
    ](self, dst_buf: DeviceBuffer[dtype], src_buf: DeviceBuffer[dtype]) raises:
        """Enqueues a device-to-device copy that skips the cross-stream waits.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_buf: Device buffer to copy to.
            src_buf: Device buffer to copy from.

        Raises:
            If the operation fails.
        """
        comptime if use_legacy_backend():
            rebind[Self._Legacy](self._impl).enqueue_copy_no_cross_stream_sync(
                rebind[DeviceBuffer[dtype]._Legacy](dst_buf._impl),
                rebind[DeviceBuffer[dtype]._Legacy](src_buf._impl),
            )
        else:
            rebind[Self._HAL](self._impl).enqueue_copy_no_cross_stream_sync(
                rebind[DeviceBuffer[dtype]._HAL](dst_buf._impl),
                rebind[DeviceBuffer[dtype]._HAL](src_buf._impl),
            )

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
        comptime if use_legacy_backend():
            rebind[Self._Legacy](self._impl).enqueue_memset(
                rebind[DeviceBuffer[dtype]._Legacy](dst._impl), val
            )
        else:
            rebind[Self._HAL](self._impl).enqueue_memset(
                rebind[DeviceBuffer[dtype]._HAL](dst._impl), val
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
        comptime if use_legacy_backend():
            rebind[Self._Legacy](self._impl).enqueue_memset(
                rebind[HostBuffer[dtype]._Legacy](dst._impl), val
            )
        else:
            rebind[Self._HAL](self._impl).enqueue_memset(
                rebind[HostBuffer[dtype]._HAL](dst._impl), val
            )

    def enqueue_cpu_range[
        FuncType: def(Int) -> None,
    ](self, func: FuncType, count: Int) raises:
        """Enqueues `count` calls to `func` on the CPU, each with its index.

        Parameters:
            FuncType: The function type.

        Args:
            func: The function to execute.
            count: Number of times to call it.

        Raises:
            If the operation fails.
        """
        comptime if use_legacy_backend():
            rebind[Self._Legacy](self._impl).enqueue_cpu_range(func, count)
        else:
            rebind[Self._HAL](self._impl).enqueue_cpu_range(func, count)

    def enqueue_cpu_function[
        FuncType: def() -> None,
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
        rebind[Self._Legacy](self._impl).enqueue_cpu_function(func)

    def execution_time[
        FuncType: def(Self) raises -> None,
    ](self, func: FuncType, num_iters: Int) raises -> Int:
        """Measures the execution time of a function that takes a DeviceContext
        parameter.

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
        comptime if use_legacy_backend():

            def run_legacy(inner: Self._Legacy) raises {imm}:
                func(Self(_impl=rebind_var[Self._Impl](inner.copy())))

            return rebind[Self._Legacy](self._impl).execution_time(
                run_legacy, num_iters
            )
        else:

            def run_hal(inner: Self._HAL) raises {imm}:
                func(Self(_impl=rebind_var[Self._Impl](inner.copy())))

            return rebind[Self._HAL](self._impl).execution_time(
                run_hal, num_iters
            )

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
        comptime if use_legacy_backend():
            return rebind[Self._Legacy](self._impl).execution_time(
                func, num_iters
            )
        else:
            return rebind[Self._HAL](self._impl).execution_time(func, num_iters)

    def execution_time_iter[
        FuncType: def(Self, Int) raises -> None,
    ](self, func: FuncType, num_iters: Int) raises -> Int:
        """Measures the execution time of a function that takes the iteration
        index as input.

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
        comptime if use_legacy_backend():

            def run_legacy(inner: Self._Legacy, i: Int) raises {imm}:
                func(Self(_impl=rebind_var[Self._Impl](inner.copy())), i)

            return rebind[Self._Legacy](self._impl).execution_time_iter(
                run_legacy, num_iters
            )
        else:

            def run_hal(inner: Self._HAL, i: Int) raises {imm}:
                func(Self(_impl=rebind_var[Self._Impl](inner.copy())), i)

            return rebind[Self._HAL](self._impl).execution_time_iter(
                run_hal, num_iters
            )

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
        """Enables access to `peer`'s memory from this context.

        Args:
            peer: The peer context.

        Raises:
            If enabling peer access fails.
        """
        comptime if use_legacy_backend():
            rebind[Self._Legacy](self._impl).enable_peer_access(
                rebind[Self._Legacy](peer._impl)
            )
        else:
            rebind[Self._HAL](self._impl).enable_peer_access(
                rebind[Self._HAL](peer._impl)
            )

    @staticmethod
    def enable_all_peer_access() raises:
        """Enables peer access between every device it can be enabled on.

        Raises:
            If enabling peer access fails.
        """
        comptime if use_legacy_backend():
            LegacyDeviceContext[Self.target].enable_all_peer_access()
        else:
            HALDeviceContext[Self.target].enable_all_peer_access()

    @staticmethod
    def all_peer_access_enabled() raises -> Bool:
        """Returns whether peer access is enabled between every pair of devices.

        Returns:
            True when all peer-access pairs are enabled.

        Raises:
            If the query fails.
        """
        comptime if use_legacy_backend():
            return LegacyDeviceContext[Self.target].all_peer_access_enabled()
        else:
            return HALDeviceContext[Self.target].all_peer_access_enabled()

    def supports_multicast(self) raises -> Bool:
        """Returns whether this device supports multicast memory.

        Returns:
            True when multicast is available.

        Raises:
            If the query fails.
        """
        comptime if use_legacy_backend():
            return rebind[Self._Legacy](self._impl).supports_multicast()
        else:
            return rebind[Self._HAL](self._impl).supports_multicast()

    def enqueue_wait_for(self, other: DeviceContext) raises:
        """Enqueues a wait for all work currently pending on `other`.

        Args:
            other: The context to wait for.

        Raises:
            If the operation fails.
        """
        comptime if use_legacy_backend():
            rebind[Self._Legacy](self._impl).enqueue_wait_for(
                rebind[Self._Legacy](other._impl)
            )
        else:
            rebind[Self._HAL](self._impl).enqueue_wait_for(
                rebind[Self._HAL](other._impl)
            )


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
            self._impl = rebind_var[Self._Impl](
                Self._Legacy(
                    rebind[DeviceContext[Self.target]._Legacy](ctx._impl)
                )
            )
        else:
            self._impl = rebind_var[Self._Impl](
                Self._HAL(rebind[DeviceContext[Self.target]._HAL](ctx._impl))
            )

    def __enter__(mut self) raises -> DeviceContext[Self.target]:
        """Makes the driver context current.

        Returns:
            The device context of this scope.

        Raises:
            If the driver context cannot be made current.
        """
        comptime if use_legacy_backend():
            return rebind_var[DeviceContext[Self.target]](
                rebind[Self._Legacy](self._impl).__enter__()
            )
        else:
            return rebind_var[DeviceContext[Self.target]](
                rebind[Self._HAL](self._impl).__enter__()
            )

    def __exit__(mut self) raises:
        """Restores the previous driver context.

        Raises:
            If the previous driver context cannot be restored.
        """
        comptime if use_legacy_backend():
            rebind[Self._Legacy](self._impl).__exit__()
        else:
            rebind[Self._HAL](self._impl).__exit__()
