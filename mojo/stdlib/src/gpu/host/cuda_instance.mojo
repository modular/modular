# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from sys.ffi import c_char, c_size_t

from utils.static_tuple import StaticTuple

from ._utils import (
    _check_error,
    _ContextHandle,
    _EventHandle,
    _get_dylib_function,
    _human_memory,
    _ModuleHandle,
    _StreamHandle,
)
from .device import Device
from .event import Flag
from .function import CacheConfig, _FunctionHandle


@register_passable("trivial")
struct _dylib_function[fn_name: StringLiteral, type: AnyTrivialRegType]:
    @staticmethod
    fn load() -> type:
        return _get_dylib_function[fn_name, type]()


# ===----------------------------------------------------------------------=== #
# Types
# ===----------------------------------------------------------------------=== #

alias _DeviceHandle = Int32

alias cuDeviceGetCount = _dylib_function[
    "cuDeviceGetCount", fn (UnsafePointer[Int32]) -> Result
]

alias cuDeviceGetAttribute = _dylib_function[
    "cuDeviceGetAttribute",
    fn (UnsafePointer[Int32], DeviceAttribute, _DeviceHandle) -> Result,
]

alias cuDriverGetVersion = _dylib_function[
    "cuDriverGetVersion", fn (UnsafePointer[Int32]) -> Result
]

alias cuDeviceGetName = _dylib_function[
    "cuDeviceGetName",
    fn (UnsafePointer[Int8], Int32, _DeviceHandle) -> Result,
]

alias cuDeviceTotalMem = _dylib_function[
    "cuDeviceTotalMem_v2", fn (UnsafePointer[Int], _DeviceHandle) -> Result
]

alias cuDevicePrimaryCtxRetain = _dylib_function[
    "cuDevicePrimaryCtxRetain",
    fn (UnsafePointer[_ContextHandle], _DeviceHandle) -> Result,
]

alias cuDevicePrimaryCtxSetFlags = _dylib_function[
    "cuDevicePrimaryCtxSetFlags",
    fn (_DeviceHandle, Int32) -> Result,
]

alias cuDevicePrimaryCtxRelease = _dylib_function[
    "cuDevicePrimaryCtxRelease", fn (_DeviceHandle) -> Result
]

alias cuDevicePrimaryCtxReset = _dylib_function[
    "cuDevicePrimaryCtxReset", fn (_DeviceHandle) -> Result
]

alias cuCtxCreate = _dylib_function[
    "cuCtxCreate_v2",
    fn (UnsafePointer[_ContextHandle], Int32, _DeviceHandle) -> Result,
]

alias cuCtxPushCurrent = _dylib_function[
    "cuCtxPushCurrent_v2",
    fn (_ContextHandle) -> Result,
]

alias cuCtxGetCurrent = _dylib_function[
    "cuCtxGetCurrent",
    fn (UnsafePointer[_ContextHandle]) -> Result,
]

alias cuCtxDestroy = _dylib_function[
    "cuCtxDestroy_v2", fn (_ContextHandle) -> Result
]

alias cuCtxSetCurrent = _dylib_function[
    "cuCtxSetCurrent", fn (_ContextHandle) -> Result
]

alias cuCtxSynchronize = _dylib_function["cuCtxSynchronize", fn () -> Result]

alias cuEventCreate = _dylib_function[
    "cuEventCreate", fn (UnsafePointer[_EventHandle], Flag) -> Result
]

alias cuEventDestroy = _dylib_function[
    "cuEventDestroy_v2", fn (_EventHandle) -> Result
]

alias cuEventSynchronize = _dylib_function[
    "cuEventSynchronize", fn (_EventHandle) -> Result
]

alias cuEventRecord = _dylib_function[
    "cuEventRecord", fn (_EventHandle, _StreamHandle) -> Result
]

alias cuEventElapsedTime = _dylib_function[
    "cuEventElapsedTime",
    fn (UnsafePointer[Float32], _EventHandle, _EventHandle) -> Result,
]

alias cuStreamCreate = _dylib_function[
    "cuStreamCreate", fn (UnsafePointer[_StreamHandle], Int32) -> Result
]

alias cuStreamDestroy = _dylib_function[
    "cuStreamDestroy", fn (_StreamHandle) -> Result
]

alias cuStreamSynchronize = _dylib_function[
    "cuStreamSynchronize", fn (_StreamHandle) -> Result
]

alias cuMemAllocHost = _dylib_function[
    "cuMemAllocHost_v2", fn (UnsafePointer[UnsafePointer[Int]], Int) -> Result
]

alias cuMemAlloc = _dylib_function[
    "cuMemAlloc_v2", fn (UnsafePointer[UnsafePointer[Int]], Int) -> Result
]

alias cuMemAllocAsync = _dylib_function[
    "cuMemAllocAsync",
    fn (UnsafePointer[UnsafePointer[Int]], Int, _StreamHandle) -> Result,
]

alias cuMemAllocManaged = _dylib_function[
    "cuMemAllocManaged",
    fn (UnsafePointer[UnsafePointer[Int]], Int, UInt32) -> Result,
]

alias cuMemFreeHost = _dylib_function[
    "cuMemFreeHost", fn (UnsafePointer[Int]) -> Result
]

alias cuMemFree = _dylib_function[
    "cuMemFree_v2", fn (UnsafePointer[Int]) -> Result
]

alias cuMemFreeAsync = _dylib_function[
    "cuMemFreeAsync", fn (UnsafePointer[Int], _StreamHandle) -> Result
]

alias cuMemcpyHtoD = _dylib_function[
    "cuMemcpyHtoD_v2",
    fn (UnsafePointer[Int], UnsafePointer[NoneType], Int) -> Result,
]

alias cuMemcpyHtoDAsync = _dylib_function[
    "cuMemcpyHtoDAsync_v2",
    fn (
        UnsafePointer[NoneType], UnsafePointer[Int], Int, _StreamHandle
    ) -> Result,
]

alias cuMemcpyDtoH = _dylib_function[
    "cuMemcpyDtoH_v2",
    fn (UnsafePointer[NoneType], UnsafePointer[Int], Int) -> Result,
]

alias cuMemcpyDtoHAsync = _dylib_function[
    "cuMemcpyDtoHAsync_v2",
    fn (
        UnsafePointer[NoneType], UnsafePointer[Int], Int, _StreamHandle
    ) -> Result,
]

alias cuMemcpyDtoDAsync = _dylib_function[
    "cuMemcpyDtoDAsync_v2",
    fn (
        UnsafePointer[NoneType], UnsafePointer[Int], Int, _StreamHandle
    ) -> Result,
]

alias cuMemcpyDtoD = _dylib_function[
    "cuMemcpyDtoD_v2",
    fn (UnsafePointer[Int], UnsafePointer[Int], Int) -> Result,
]

alias cuMemsetD8 = _dylib_function[
    "cuMemsetD8_v2", fn (UnsafePointer[Int], UInt8, Int) -> Result
]

alias cuMemsetD8Async = _dylib_function[
    "cuMemsetD8Async",
    fn (UnsafePointer[UInt8], UInt8, Int, _StreamHandle) -> Result,
]

alias cuMemsetD16Async = _dylib_function[
    "cuMemsetD16Async",
    fn (UnsafePointer[UInt16], UInt16, Int, _StreamHandle) -> Result,
]

alias cuMemsetD32Async = _dylib_function[
    "cuMemsetD32Async",
    fn (UnsafePointer[UInt32], UInt32, Int, _StreamHandle) -> Result,
]

alias cuLaunchKernelEx = _dylib_function[
    "cuLaunchKernelEx",
    fn (
        UnsafePointer[LaunchConfig],
        _FunctionHandle,
        UnsafePointer[UnsafePointer[NoneType]],  # Args
        UnsafePointer[NoneType],  # Extra
    ) -> Result,
]

alias cuFuncSetCacheConfig = _dylib_function[
    "cuFuncSetCacheConfig",
    fn (_FunctionHandle, Int32) -> Result,
]

alias cuFuncSetAttribute = _dylib_function[
    "cuFuncSetAttribute",
    fn (_FunctionHandle, Int32, Int32) -> Result,
]

alias cuFuncGetAttribute = _dylib_function[
    "cuFuncGetAttribute",
    fn (UnsafePointer[Int32], Int32, _FunctionHandle) -> Result,
]

alias cuModuleLoad = _dylib_function[
    "cuModuleLoad",
    fn (UnsafePointer[_ModuleHandle], UnsafePointer[c_char]) -> Result,
]

alias cuModuleLoadData = _dylib_function[
    "cuModuleLoadData",
    fn (UnsafePointer[_ModuleHandle], UnsafePointer[UInt8]) -> Result,
]

alias cuModuleLoadDataEx = _dylib_function[
    "cuModuleLoadDataEx",
    fn (
        UnsafePointer[_ModuleHandle],
        UnsafePointer[UInt8],
        UInt32,
        UnsafePointer[JitOptions],
        UnsafePointer[Int],
    ) -> Result,
]

alias cuModuleUnload = _dylib_function[
    "cuModuleUnload", fn (_ModuleHandle) -> Result
]

alias cuModuleGetFunction = _dylib_function[
    "cuModuleGetFunction",
    fn (
        UnsafePointer[_FunctionHandle],
        _ModuleHandle,
        UnsafePointer[c_char],
    ) -> Result,
]

alias cuCtxGetLimit = _dylib_function[
    "cuCtxGetLimit",
    fn (UnsafePointer[Int], LimitProperty) -> Result,
]

alias cuCtxSetLimit = _dylib_function[
    "cuCtxSetLimit",
    fn (LimitProperty, Int) -> Result,
]

alias cuCtxSetCacheConfig = _dylib_function[
    "cuCtxSetCacheConfig",
    fn (CacheConfig) -> Result,
]

alias cuCtxResetPersistingL2Cache = _dylib_function[
    "cuCtxResetPersistingL2Cache",
    fn () -> Result,
]

alias cuModuleGetGlobal = _dylib_function[
    "cuModuleGetGlobal_v2",
    fn (
        UnsafePointer[UnsafePointer[NoneType]],
        UnsafePointer[Int],
        _ModuleHandle,
        UnsafePointer[c_char],
    ) -> Result,
]

alias cuMemGetInfo = _dylib_function[
    "cuMemGetInfo_v2",
    fn (UnsafePointer[c_size_t], UnsafePointer[c_size_t]) -> Result,
]

# ===----------------------------------------------------------------------===#
# AccessProperty
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct AccessProperty:
    """Specifies performance hint with AccessPolicyWindow for hit_prop and
    miss_prop fields."""

    var _value: Int32

    alias NORMAL = Self(0)
    """Normal cache persistence."""
    alias STREAMING = Self(1)
    """Streaming access is less likely to persit from cache."""
    alias PERSISTING = Self(2)
    """Persisting access is more likely to persist in cache."""

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    fn __init__(inout self, *, other: Self):
        """Explicitly construct a deep copy of the provided value.

        Args:
            other: The value to copy.
        """
        self = other

    @no_inline
    fn __str__(self) -> String:
        return String.format_sequence(self)

    @no_inline
    fn format_to(self, inout writer: Formatter):
        if self is Self.NORMAL:
            return writer.write("NORMAL")
        if self is Self.STREAMING:
            return writer.write("STREAMING")
        return writer.write("PERSISTING")


# ===----------------------------------------------------------------------===#
# AccessPolicyWindow
# ===----------------------------------------------------------------------===#


@register_passable("trivial")
struct AccessPolicyWindow:
    """Specifies an access policy for a window, a contiguous extent of
    memory beginning at base_ptr and ending at base_ptr + num_bytes.
    num_bytes is limited by
    CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE. Partition into
    many segments and assign segments such that: sum of "hit segments"
    / window == approx. ratio. sum of "miss segments" / window ==
    approx 1-ratio. Segments and ratio specifications are fitted to the
    capabilities of the architecture. Accesses in a hit segment apply
    the hitProp access policy. Accesses in a miss segment apply the
    missProp access policy.
    """

    var base_ptr: UnsafePointer[NoneType]
    """Starting address of the access policy window. Driver may align it."""
    var num_bytes: Int
    """Size in bytes of the window policy. CUDA driver may restrict the
    maximum size and alignment."""
    var hit_ratio: Float32
    """Specifies percentage of lines assigned hitProp, rest are assigned
    missProp."""
    var hit_prop: AccessProperty
    """AccessProperty set for hit."""
    var miss_prop: AccessProperty
    """AccessProperty set for miss. Must be either NORMAL or STREAMING."""

    fn __init__(inout self):
        self.base_ptr = UnsafePointer[NoneType]()
        self.num_bytes = 0
        self.hit_ratio = 0
        self.hit_prop = AccessProperty.NORMAL
        self.miss_prop = AccessProperty.NORMAL

    fn __init__[
        T: AnyType
    ](
        inout self,
        *,
        base_ptr: UnsafePointer[T, *_, **_],
        count: Int,
        hit_ratio: Float32,
        hit_prop: AccessProperty = AccessProperty.NORMAL,
        miss_prop: AccessProperty = AccessProperty.NORMAL,
    ):
        self.base_ptr = base_ptr.bitcast[
            NoneType, address_space = AddressSpace.GENERIC
        ]().address
        self.num_bytes = count * sizeof[T]()
        self.hit_ratio = hit_ratio
        self.hit_prop = hit_prop
        self.miss_prop = miss_prop

    @no_inline
    fn __str__(self) -> String:
        return String.format_sequence(self)

    @no_inline
    fn format_to(self, inout writer: Formatter):
        return writer.write(
            "base_ptr: ",
            self.base_ptr,
            ", num_bytes: ",
            self.num_bytes,
            ", hit_ratio: ",
            self.hit_ratio,
            ", hit_prop: ",
            self.hit_prop,
            ", miss_prop: ",
            self.miss_prop,
        )


# ===----------------------------------------------------------------------===#
# LimitProperty
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct LimitProperty:
    var _value: Int32

    alias STACK_SIZE = 0x00
    """Controls the stack size in bytes of each GPU thread. The driver
    automatically increases the per-thread stack size for each kernel launch as
    needed. This size isn't reset back to the original value after each launch.
    Setting this value will take effect immediately, and if necessary, the device
  w ill block until all preceding requested tasks are complete."""

    alias PRINTF_FIFO_SIZE = 0x01
    """Controls the size in bytes of the FIFO used by the printf() device
    system call. Setting CU_LIMIT_PRINTF_FIFO_SIZE must be performed before
    launching any kernel that uses the printf() device system call, otherwise
    CUDA_ERROR_INVALID_VALUE will be returned."""

    alias MALLOC_HEAP_SIZE = 0x02
    """Controls the size in bytes of the heap used by the malloc() and free()
    device system calls. Setting CU_LIMIT_MALLOC_HEAP_SIZE must be performed
    before launching any kernel that uses the malloc() or free() device system
    calls, otherwise CUDA_ERROR_INVALID_VALUE will be returned."""

    alias DEV_RUNTIME_SYNC_DEPTH = 0x03
    """GPU device runtime launch synchronize depth."""

    alias DEV_RUNTIME_PENDING_LAUNCH_COUNT = 0x04
    """GPU device runtime pending launch count."""

    alias MAX_L2_FETCH_GRANULARITY = 0x05
    """A value between 0 and 128 that indicates the maximum fetch granularity
    of L2 (in Bytes). This is a hint."""

    alias PERSISTING_L2_CACHE_SIZE = 0x06
    """A size in bytes for L2 persisting lines cache size."""

    alias SHMEM_SIZE = 0x07
    """A maximum size in bytes of shared memory available to CUDA kernels on a
    CIG context. Can only be queried, cannot be set."""

    alias CIG_ENABLED = 0x08
    """A non-zero value indicates this CUDA context is a CIG-enabled context.
    Can only be queried, cannot be set."""

    alias CIG_SHMEM_FALLBACK_ENABLED = 0x09
    """When set to a non-zero value, CUDA will fail to launch a kernel on a
    CIG context, instead of using the fallback path, if the kernel uses more
    shared memory than available."""

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    fn __init__(inout self, *, other: Self):
        """Explicitly construct a deep copy of the provided value.

        Args:
            other: The value to copy.
        """
        self = other

    @no_inline
    fn __str__(self) -> String:
        return String.format_sequence(self)

    @no_inline
    fn format_to(self, inout writer: Formatter):
        return writer.write(self._value)


# ===----------------------------------------------------------------------===#
# ConstantMemoryMapping
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct ConstantMemoryMapping:
    var name: StringLiteral
    var ptr: UnsafePointer[NoneType]
    var byte_count: Int

    fn __init__(
        inout self,
        name: StringLiteral,
        ptr: UnsafePointer[NoneType],
        byte_count: Int,
    ):
        self.name = name
        self.ptr = ptr
        self.byte_count = byte_count


# ===----------------------------------------------------------------------===#
# LaunchAttribute
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct LaunchAttribute:
    var id: LaunchAttributeID
    var __pad: StaticTuple[UInt8, 8 - sizeof[LaunchAttributeID]()]
    var value: LaunchAttributeValue

    fn __init__(inout self):
        self.id = LaunchAttributeID.IGNORE
        self.__pad = __type_of(self.__pad)()
        self.value = LaunchAttributeValue()

    fn __init__(inout self, policy: AccessPolicyWindow):
        self = Self()
        self.id = LaunchAttributeID.ACCESS_POLICY_WINDOW
        self.value = LaunchAttributeValue(policy)

    @staticmethod
    fn from_cluster_dim(dim: Dim) -> Self:
        var res = Self()
        res.id = LaunchAttributeID.CLUSTER_DIMENSION
        res.value = LaunchAttributeValue(dim)
        return res


# ===----------------------------------------------------------------------===#
# LaunchAttributeValue
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct LaunchAttributeValue:
    # TODO: This should be a union as defined in
    # https://docs.nvidia.com/cuda/cuda-driver-api/unionCUlaunchAttributeValue.html#unionCUlaunchAttributeValue
    # but we can emulate that by just having different constructors with
    # different storages.
    alias _storage_type = StaticTuple[UInt8, 64]
    var _storage: Self._storage_type

    fn __init__(inout self):
        self._storage = StaticTuple[UInt8, 64](0)

    fn __init__(inout self, policy: AccessPolicyWindow):
        var tmp = policy
        var ptr = UnsafePointer.address_of(tmp)
        self._storage = ptr.bitcast[Self._storage_type]()[]

    fn __init__(inout self, dim: Dim):
        var tmp = StaticTuple[UInt32, 3](dim.x(), dim.y(), dim.z())
        var ptr = UnsafePointer.address_of(tmp)
        self._storage = ptr.bitcast[Self._storage_type]()[]


# ===----------------------------------------------------------------------===#
# LaunchAttributeID
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct LaunchAttributeID:
    var _value: Int32

    alias IGNORE = Self(0)
    """Ignored entry, for convenient composition."""

    alias ACCESS_POLICY_WINDOW = Self(1)
    """Valid for streams, graph nodes, launches."""

    alias COOPERATIVE = Self(2)
    """Valid for graph nodes, launches."""

    alias SYNCHRONIZATION_POLICY = Self(3)
    """Valid for streams."""

    alias CLUSTER_DIMENSION = Self(4)
    """Valid for graph nodes, launches."""

    alias CLUSTER_SCHEDULING_POLICY_PREFERENCE = Self(5)
    """Valid for graph nodes, launches."""

    alias PROGRAMMATIC_STREAM_SERIALIZATION = Self(6)
    """Valid for launches. Setting CUlaunchAttributeValue::
    programmaticStreamSerializationAllowed to non-0 signals that the kernel
    will use programmatic means to resolve its stream dependency, so that the
    CUDA runtime should opportunistically allow the grid's execution to overlap
    with the previous kernel in the stream, if that kernel requests the overlap.
    The dependent launches can choose to wait on the dependency using the
    programmatic sync."""

    alias PROGRAMMATIC_EVENT = Self(7)
    """Valid for launches. Set CUlaunchAttributeValue::programmaticEvent to
    record the event. Event recorded through this launch attribute is guaranteed
    to only trigger after all block in the associated kernel trigger the event.
    A block can trigger the event through PTX launchdep.release or CUDA builtin
    function cudaTriggerProgrammaticLaunchCompletion(). A trigger can also be
    inserted at the beginning of each block's execution if triggerAtBlockStart
    is set to non-0. The dependent launches can choose to wait on the dependency
    using the programmatic sync (cudaGridDependencySynchronize() or equivalent
    PTX instructions). Note that dependents (including the CPU thread calling
    cuEventSynchronize()) are not guaranteed to observe the release precisely
    when it is released. For example, cuEventSynchronize() may only observe
    the event trigger long after the associated kernel has completed. This
    recording type is primarily meant for establishing programmatic dependency
    between device tasks. Note also this type of dependency allows, but does not
    guarantee, concurrent execution of tasks. The event supplied must not be an
    interprocess or interop event. The event must disable timing (i.e. must be
    created with the CU_EVENT_DISABLE_TIMING flag set)."""

    alias PRIORITY = Self(8)
    """Valid for streams, graph nodes, launches."""

    alias MEM_SYNC_DOMAIN_MAP = Self(9)
    """Valid for streams, graph nodes, launches."""

    alias MEM_SYNC_DOMAIN = Self(10)
    """Valid for streams, graph nodes, launches."""

    alias LAUNCH_COMPLETION_EVENT = Self(12)
    """Valid for launches. Set CUlaunchAttributeValue::launchCompletionEvent to
    record the event. Nominally, the event is triggered once all blocks of the
    kernel have begun execution. Currently this is a best effort. If a kernel
    B has a launch completion dependency on a kernel A, B may wait until A is
    complete. Alternatively, blocks of B may begin before all blocks of A have
    begun, for example if B can claim execution resources unavailable to A
    (e.g. they run on different GPUs) or if B is a higher priority than A.
    Exercise caution if such an ordering inversion could lead to deadlock.
    A launch completion event is nominally similar to a programmatic event
    with triggerAtBlockStart set except that it is not visible to
    cudaGridDependencySynchronize() and can be used with compute capability
    less than 9.0. The event supplied must not be an interprocess or interop
    event. The event must disable timing (i.e. must be created with the
    CU_EVENT_DISABLE_TIMING flag set)."""

    alias DEVICE_UPDATABLE_KERNEL_NODE = Self(13)
    """Valid for graph nodes, launches. This attribute is graphs-only,
    and passing it to a launch in a non-capturing stream will result in an
    error. CUlaunchAttributeValue::deviceUpdatableKernelNode::deviceUpdatable
    can only be set to 0 or 1. Setting the field to 1 indicates that the
    corresponding kernel node should be device-updatable. On success, a handle
    will be returned via CUlaunchAttributeValue::deviceUpdatableKernelNode::devNode
    which can be passed to the various device-side update functions to update
    the node's kernel parameters from within another kernel. For more
    information on the types of device updates that can be made, as well as the
    relevant limitations thereof, see cudaGraphKernelNodeUpdatesApply. Nodes
    which are device-updatable have additional restrictions compared to regular
    kernel nodes. Firstly, device-updatable nodes cannot be removed from their
    graph via cuGraphDestroyNode. Additionally, once opted-in to this
    functionality, a node cannot opt out, and any attempt to set the
    deviceUpdatable attribute to 0 will result in an error. Device-updatable
    kernel nodes also cannot have their attributes copied to/from another kernel
    node via cuGraphKernelNodeCopyAttributes. Graphs containing one or more
    device-updatable nodes also do not allow multiple instantiation, and neither
    the graph nor its instantiated version can be passed to cuGraphExecUpdate.
    If a graph contains device-updatable nodes and updates those nodes from the
    device from within the graph, the graph must be uploaded with cuGraphUpload
    before it is launched. For such a graph, if host-side executable graph
    updates are made to the device-updatable nodes, the graph must be uploaded
    before it is launched again."""

    alias PREFERRED_SHARED_MEMORY_CARVEOUT = Self(14)
    """Valid for launches. On devices where the L1 cache and shared memory use
    the same hardware resources, setting CUlaunchAttributeValue::sharedMemCarveout
    to a percentage between 0-100 signals the CUDA driver to set the shared
    memory carveout preference, in percent of the total shared memory for that
    kernel launch. This attribute takes precedence over
    CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT. This is only a hint,
    and the CUDA driver can choose a different configuration if required for
    the launch."""

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    fn __init__(inout self, *, other: Self):
        """Explicitly construct a deep copy of the provided value.

        Args:
            other: The value to copy.
        """
        self = other

    @no_inline
    fn __str__(self) -> String:
        return String.format_sequence(self)

    @no_inline
    fn format_to(self, inout writer: Formatter):
        return writer.write(self._value)


# ===----------------------------------------------------------------------===#
# Launch Config
# ===----------------------------------------------------------------------===#


@register_passable("trivial")
struct LaunchConfig:
    var grid_dim_x: UInt32
    """Width of grid in blocks."""
    var grid_dim_y: UInt32
    """Height of grid in blocks."""
    var grid_dim_z: UInt32
    """Depth of grid in blocks."""
    var block_dim_x: UInt32
    """X dimension of each thread block."""
    var block_dim_y: UInt32
    """Y dimension of each thread block."""
    var block_dim_z: UInt32
    """Z dimension of each thread block."""
    var shared_mem_bytes: UInt32
    """Dynamic shared-memory size per thread block in bytes."""
    var stream: _StreamHandle
    """Stream identifier."""
    var attrs: UnsafePointer[LaunchAttribute]
    """List of attributes; nullable if num_attrs == 0."""
    var num_attrs: UInt32
    """Number of attributes populated in attrs."""

    @always_inline
    fn __init__(
        inout self,
        *,
        grid_dim_x: UInt32,
        block_dim_x: UInt32,
        block_dim_y: UInt32 = 1,
        block_dim_z: UInt32 = 1,
        grid_dim_y: UInt32 = 1,
        grid_dim_z: UInt32 = 1,
        shared_mem_bytes: UInt32 = 0,
        stream: _StreamHandle = _StreamHandle(),
        attrs: UnsafePointer[LaunchAttribute] = UnsafePointer[
            LaunchAttribute
        ](),
        num_attrs: UInt32 = 0,
    ):
        self.grid_dim_x = grid_dim_x
        self.grid_dim_y = grid_dim_y
        self.grid_dim_z = grid_dim_z
        self.block_dim_x = block_dim_x
        self.block_dim_y = block_dim_y
        self.block_dim_z = block_dim_z
        self.shared_mem_bytes = shared_mem_bytes
        self.stream = stream
        self.attrs = attrs
        self.num_attrs = num_attrs


# ===----------------------------------------------------------------------===#
# LaunchAttributeID
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct CacheMode:
    """Caching modes for dlcm."""

    var _value: Int

    alias NONE = Self(0)
    """Compile with no -dlcm flag specified."""

    alias L1_CACHE_DISABLED = Self(1)
    """Compile with L1 cache disabled."""

    alias L1_CACHE_ENABLED = Self(2)
    """Compile with L1 cache enabled."""

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    fn __init__(inout self, *, other: Self):
        """Explicitly construct a deep copy of the provided value.

        Args:
            other: The value to copy.
        """
        self = other

    @always_inline
    fn __int__(self) -> Int:
        return self._value

    @no_inline
    fn __str__(self) -> String:
        return String.format_sequence(self)

    @no_inline
    fn format_to(self, inout writer: Formatter):
        return writer.write(self._value)


# ===----------------------------------------------------------------------=== #
# CudaDLL
# ===----------------------------------------------------------------------=== #


@value
struct CudaDLL:
    # cuDevice
    var cuDeviceGetCount: cuDeviceGetCount.type
    var cuDeviceGetAttribute: cuDeviceGetAttribute.type
    var cuDriverGetVersion: cuDriverGetVersion.type
    var cuDeviceGetName: cuDeviceGetName.type
    var cuDeviceTotalMem: cuDeviceTotalMem.type

    # cuDevicePrimaryCtx
    var cuDevicePrimaryCtxRetain: cuDevicePrimaryCtxRetain.type
    var cuDevicePrimaryCtxSetFlags: cuDevicePrimaryCtxSetFlags.type
    var cuDevicePrimaryCtxRelease: cuDevicePrimaryCtxRelease.type
    var cuDevicePrimaryCtxReset: cuDevicePrimaryCtxReset.type

    # cuCtx
    var cuCtxCreate: cuCtxCreate.type
    var cuCtxPushCurrent: cuCtxPushCurrent.type
    var cuCtxGetCurrent: cuCtxGetCurrent.type
    var cuCtxDestroy: cuCtxDestroy.type
    var cuCtxSynchronize: cuCtxSynchronize.type
    var cuCtxSetCurrent: cuCtxSetCurrent.type
    var cuCtxSetCacheConfig: cuCtxSetCacheConfig.type
    var cuCtxResetPersistingL2Cache: cuCtxResetPersistingL2Cache.type

    # cuEvent
    var cuEventCreate: cuEventCreate.type
    var cuEventDestroy: cuEventDestroy.type
    var cuEventSynchronize: cuEventSynchronize.type
    var cuEventRecord: cuEventRecord.type
    var cuEventElapsedTime: cuEventElapsedTime.type

    # cuStream
    var cuStreamCreate: cuStreamCreate.type
    var cuStreamDestroy: cuStreamDestroy.type
    var cuStreamSynchronize: cuStreamSynchronize.type

    # cuMalloc
    var cuMemAllocHost: cuMemAllocHost.type
    var cuMemAlloc: cuMemAlloc.type
    var cuMemAllocAsync: cuMemAllocAsync.type
    var cuMemAllocManaged: cuMemAllocManaged.type
    var cuMemFreeHost: cuMemFreeHost.type
    var cuMemFree: cuMemFree.type
    var cuMemFreeAsync: cuMemFreeAsync.type

    # cuMemcpy
    var cuMemcpyHtoD: cuMemcpyHtoD.type
    var cuMemcpyHtoDAsync: cuMemcpyHtoDAsync.type
    var cuMemcpyDtoH: cuMemcpyDtoH.type
    var cuMemcpyDtoHAsync: cuMemcpyDtoHAsync.type
    var cuMemcpyDtoD: cuMemcpyDtoD.type
    var cuMemcpyDtoDAsync: cuMemcpyDtoDAsync.type

    # cuMemSet
    var cuMemsetD8: cuMemsetD8.type
    var cuMemsetD8Async: cuMemsetD8Async.type
    var cuMemsetD16Async: cuMemsetD16Async.type
    var cuMemsetD32Async: cuMemsetD32Async.type

    # cuFunc
    var cuLaunchKernelEx: cuLaunchKernelEx.type
    var cuFuncSetCacheConfig: cuFuncSetCacheConfig.type
    var cuFuncSetAttribute: cuFuncSetAttribute.type
    var cuFuncGetAttribute: cuFuncGetAttribute.type

    # cuModule
    var cuModuleLoad: cuModuleLoad.type
    var cuModuleLoadData: cuModuleLoadData.type
    var cuModuleLoadDataEx: cuModuleLoadDataEx.type
    var cuModuleUnload: cuModuleUnload.type
    var cuModuleGetFunction: cuModuleGetFunction.type
    var cuModuleGetGlobal: cuModuleGetGlobal.type

    # cuMem
    var cuMemGetInfo: cuMemGetInfo.type

    fn __init__(inout self):
        self.cuDeviceGetCount = cuDeviceGetCount.load()
        self.cuDeviceGetAttribute = cuDeviceGetAttribute.load()
        self.cuDriverGetVersion = cuDriverGetVersion.load()
        self.cuDeviceGetName = cuDeviceGetName.load()
        self.cuDeviceTotalMem = cuDeviceTotalMem.load()
        self.cuDevicePrimaryCtxRelease = cuDevicePrimaryCtxRelease.load()
        self.cuDevicePrimaryCtxReset = cuDevicePrimaryCtxReset.load()
        self.cuDevicePrimaryCtxRetain = cuDevicePrimaryCtxRetain.load()
        self.cuDevicePrimaryCtxSetFlags = cuDevicePrimaryCtxSetFlags.load()
        self.cuCtxCreate = cuCtxCreate.load()
        self.cuCtxResetPersistingL2Cache = cuCtxResetPersistingL2Cache.load()
        self.cuCtxPushCurrent = cuCtxPushCurrent.load()
        self.cuCtxGetCurrent = cuCtxGetCurrent.load()
        self.cuCtxSetCacheConfig = cuCtxSetCacheConfig.load()
        self.cuCtxDestroy = cuCtxDestroy.load()
        self.cuCtxSynchronize = cuCtxSynchronize.load()
        self.cuCtxSetCurrent = cuCtxSetCurrent.load()
        self.cuEventCreate = cuEventCreate.load()
        self.cuEventDestroy = cuEventDestroy.load()
        self.cuEventSynchronize = cuEventSynchronize.load()
        self.cuEventRecord = cuEventRecord.load()
        self.cuEventElapsedTime = cuEventElapsedTime.load()
        self.cuStreamCreate = cuStreamCreate.load()
        self.cuStreamDestroy = cuStreamDestroy.load()
        self.cuStreamSynchronize = cuStreamSynchronize.load()
        self.cuMemAllocHost = cuMemAllocHost.load()
        self.cuMemAlloc = cuMemAlloc.load()
        self.cuMemAllocAsync = cuMemAllocAsync.load()
        self.cuMemAllocManaged = cuMemAllocManaged.load()
        self.cuMemFreeHost = cuMemFreeHost.load()
        self.cuMemFree = cuMemFree.load()
        self.cuMemFreeAsync = cuMemFreeAsync.load()
        self.cuMemcpyHtoD = cuMemcpyHtoD.load()
        self.cuMemcpyHtoDAsync = cuMemcpyHtoDAsync.load()
        self.cuMemcpyDtoH = cuMemcpyDtoH.load()
        self.cuMemcpyDtoHAsync = cuMemcpyDtoHAsync.load()
        self.cuMemcpyDtoDAsync = cuMemcpyDtoDAsync.load()
        self.cuMemcpyDtoD = cuMemcpyDtoD.load()
        self.cuMemsetD8 = cuMemsetD8.load()
        self.cuMemsetD8Async = cuMemsetD8Async.load()
        self.cuMemsetD16Async = cuMemsetD16Async.load()
        self.cuMemsetD32Async = cuMemsetD32Async.load()
        self.cuLaunchKernelEx = cuLaunchKernelEx.load()
        self.cuFuncSetCacheConfig = cuFuncSetCacheConfig.load()
        self.cuFuncSetAttribute = cuFuncSetAttribute.load()
        self.cuFuncGetAttribute = cuFuncGetAttribute.load()
        self.cuModuleLoad = cuModuleLoad.load()
        self.cuModuleLoadData = cuModuleLoadData.load()
        self.cuModuleLoadDataEx = cuModuleLoadDataEx.load()
        self.cuModuleUnload = cuModuleUnload.load()
        self.cuModuleGetFunction = cuModuleGetFunction.load()
        self.cuModuleGetGlobal = cuModuleGetGlobal.load()
        self.cuMemGetInfo = cuMemGetInfo.load()

    fn __init__(inout self, *, other: Self):
        """Explicitly construct a deep copy of the provided value.

        Args:
            other: The value to copy.
        """
        self = other


struct CudaInstance:
    var cuda_dll: CudaDLL

    fn __init__(inout self) raises:
        self.cuda_dll = CudaDLL()

    fn __copyinit__(inout self, existing: Self):
        self.cuda_dll = existing.cuda_dll

    fn num_devices(self) raises -> Int:
        var res: Int32 = 0
        _check_error(
            self.cuda_dll.cuDeviceGetCount(UnsafePointer.address_of(res))
        )
        return int(res)
