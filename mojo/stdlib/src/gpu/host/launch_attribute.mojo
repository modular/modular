# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


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

    fn __init__(out self, *, other: Self):
        """Explicitly construct a deep copy of the provided value.

        Args:
            other: The value to copy.
        """
        self = other

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        return writer.write(self._value)


@value
@register_passable("trivial")
struct LaunchAttributeValue:
    # TODO: This should be a union as defined in
    # https://docs.nvidia.com/cuda/cuda-driver-api/unionCUlaunchAttributeValue.html#unionCUlaunchAttributeValue
    # but we can emulate that by just having different constructors with
    # different storages.
    alias _storage_type = StaticTuple[UInt8, 64]
    var _storage: Self._storage_type

    fn __init__(out self):
        self._storage = StaticTuple[UInt8, 64](0)

    @implicit
    fn __init__(out self, policy: AccessPolicyWindow):
        var tmp = policy
        var ptr = UnsafePointer.address_of(tmp)
        self._storage = ptr.bitcast[Self._storage_type]()[]

    @implicit
    fn __init__(out self, dim: Dim):
        var tmp = StaticTuple[UInt32, 3](dim.x(), dim.y(), dim.z())
        var ptr = UnsafePointer.address_of(tmp)
        self._storage = ptr.bitcast[Self._storage_type]()[]


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

    fn __init__(out self, *, other: Self):
        """Explicitly construct a deep copy of the provided value.

        Args:
            other: The value to copy.
        """
        self = other

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        if self is Self.NORMAL:
            return writer.write("NORMAL")
        if self is Self.STREAMING:
            return writer.write("STREAMING")
        return writer.write("PERSISTING")


@value
@register_passable("trivial")
struct LaunchAttribute:
    var id: LaunchAttributeID
    var __pad: StaticTuple[UInt8, 8 - sizeof[LaunchAttributeID]()]
    var value: LaunchAttributeValue

    fn __init__(out self):
        self.id = LaunchAttributeID.IGNORE
        self.__pad = __type_of(self.__pad)()
        self.value = LaunchAttributeValue()

    @implicit
    fn __init__(out self, policy: AccessPolicyWindow):
        self = Self()
        self.id = LaunchAttributeID.ACCESS_POLICY_WINDOW
        self.value = LaunchAttributeValue(policy)

    @staticmethod
    fn from_cluster_dim(dim: Dim) -> Self:
        var res = Self()
        res.id = LaunchAttributeID.CLUSTER_DIMENSION
        res.value = LaunchAttributeValue(dim)
        return res


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

    fn __init__(out self):
        self.base_ptr = UnsafePointer[NoneType]()
        self.num_bytes = 0
        self.hit_ratio = 0
        self.hit_prop = AccessProperty.NORMAL
        self.miss_prop = AccessProperty.NORMAL

    fn __init__[
        T: AnyType
    ](
        mut self,
        *,
        base_ptr: UnsafePointer[T, **_],
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
        return String.write(self)

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
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
