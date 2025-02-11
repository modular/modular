# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@value
@register_passable("trivial")
struct Attribute:
    var code: Int32

    alias MAX_THREADS_PER_BLOCK = Self(0)
    """The maximum number of threads per block, beyond which a launch of the
    function would fail. This number depends on both the function and the device
    on which the function is currently loaded."""

    alias SHARED_SIZE_BYTES = Self(1)
    """The size in bytes of statically-allocated shared memory required by this
    function. This does not include dynamically-allocated shared memory
    requested by the user at runtime."""

    alias CONST_SIZE_BYTES = Self(2)
    """The size in bytes of user-allocated constant memory required by this
    function."""

    alias LOCAL_SIZE_BYTES = Self(3)
    """The size in bytes of local memory used by each thread of this function."""

    alias NUM_REGS = Self(4)
    """The number of registers used by each thread of this function."""

    alias PTX_VERSION = Self(5)
    """The PTX virtual architecture version for which the function was compiled.
    This value is the major PTX version * 10 + the minor PTX version, so a PTX
    version 1.3 function would return the value 13. Note that this may return
    the undefined value of 0 for cubins compiled prior to CUDA 3.0.."""

    alias BINARY_VERSION = Self(6)
    """The binary architecture version for which the function was compiled.
    This value is the major binary version * 10 + the minor binary version,
    so a binary version 1.3 function would return the value 13. Note that this
    will return a value of 10 for legacy cubins that do not have a properly-
    encoded binary architecture version.."""

    alias CACHE_MODE_CA = Self(7)
    """The attribute to indicate whether the function has been compiled with
    user specified option "-Xptxas --dlcm=ca" set ."""

    alias MAX_DYNAMIC_SHARED_SIZE_BYTES = Self(8)
    """The maximum size in bytes of dynamically-allocated shared memory that
    can be used by this function. If the user-specified dynamic shared memory
    size is larger than this value."""

    alias PREFERRED_SHARED_MEMORY_CARVEOUT = Self(9)
    """On devices where the L1 cache and shared memory use the same hardware
    resources, this sets the shared memory carveout preference, in percent of
    the total shared memory."""

    alias CLUSTER_SIZE_MUST_BE_SET = Self(10)
    """If this attribute is set, the kernel must launch with a valid cluster
    size specified."""

    alias REQUIRED_CLUSTER_WIDTH = Self(11)
    """The required cluster width in blocks. The values must either all be 0 or
    all be positive. The validity of the cluster dimensions is otherwise checked
    at launch time."""

    alias REQUIRED_CLUSTER_HEIGHT = Self(12)
    """The required cluster height in blocks. The values must either all be 0 or
    all be positive. The validity of the cluster dimensions is otherwise checked
    at launch time."""

    alias REQUIRED_CLUSTER_DEPTH = Self(13)
    """The required cluster depth in blocks. The values must either all be 0 or
    all be positive. The validity of the cluster dimensions is otherwise checked
    at launch time."""

    alias NON_PORTABLE_CLUSTER_SIZE_ALLOWED = Self(14)
    """Whether the function can be launched with non-portable cluster size. 1 is
    allowed, 0 is disallowed. A non-portable cluster size may only function on
    the specific SKUs the program is tested on. The launch might fail if the
    program is run on a different hardware platform.CUDA API provides
    cudaOccupancyMaxActiveClusters to assist with checking whether the desired
    size can be launched on the current device.Portable Cluster SizeA portable
    cluster size is guaranteed to be functional on all compute capabilities
    higher than the target compute capability. The portable cluster size for
    sm_90 is 8 blocks per cluster."""

    alias CLUSTER_SCHEDULING_POLICY_PREFERENCE = Self(15)
    """The block scheduling policy of a function. The value type is
    CUclusterSchedulingPolicy / cudaClusterSchedulingPolicy."""

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        return self.code == other.code

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @always_inline("nodebug")
    fn __is__(self, other: Self) -> Bool:
        return self == other

    @always_inline("nodebug")
    fn __isnot__(self, other: Self) -> Bool:
        return not (self is other)

    fn write_to[W: Writer](self, mut writer: W):
        if self is Attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES:
            return writer.write("MAX_DYNAMIC_SHARED_SIZE_BYTES")
        if self is Attribute.PREFERRED_SHARED_MEMORY_CARVEOUT:
            return writer.write("PREFERRED_SHARED_MEMORY_CARVEOUT")
        if self is Attribute.CACHE_MODE_CA:
            return writer.write("CACHE_MODE_CA")
        if self is Attribute.PTX_VERSION:
            return writer.write("PTX_VERSION")
        if self is Attribute.BINARY_VERSION:
            return writer.write("BINARY_VERSION")
        if self is Attribute.NON_PORTABLE_CLUSTER_SIZE_ALLOWED:
            return writer.write("NON_PORTABLE_CLUSTER_SIZE_ALLOWED")
        if self is Attribute.CLUSTER_SCHEDULING_POLICY_PREFERENCE:
            return writer.write("CLUSTER_SCHEDULING_POLICY_PREFERENCE")
        if self is Attribute.CLUSTER_SIZE_MUST_BE_SET:
            return writer.write("CLUSTER_SIZE_MUST_BE_SET")
        if self is Attribute.REQUIRED_CLUSTER_WIDTH:
            return writer.write("REQUIRED_CLUSTER_WIDTH")
        if self is Attribute.REQUIRED_CLUSTER_HEIGHT:
            return writer.write("REQUIRED_CLUSTER_HEIGHT")
        if self is Attribute.REQUIRED_CLUSTER_DEPTH:
            return writer.write("REQUIRED_CLUSTER_DEPTH")


@value
@register_passable("trivial")
struct FuncAttribute(CollectionElement, EqualityComparable):
    """Implement Cuda's CUfunction_attribute enum.
    https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g9d955dde0904a9b43ca4d875ac1551bc.

    Only add 'max_dynamic_shared_size_bytes`.
    """

    var attribute: Attribute
    var value: Int32

    alias NULL = FuncAttribute(Attribute(-1), -1)

    fn __init__(out self, *, other: Self):
        """Explicitly construct a deep copy of the provided value.

        Args:
            other: The value to copy.
        """
        self = other

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        return self.attribute == other.attribute and self.value == other.value

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @always_inline
    @staticmethod
    fn CACHE_MODE_CA(val: Bool) -> FuncAttribute:
        """Indicates whether the function has been compiled with user specified
        option CacheMode.L1_CACHE_DISABLED set."""
        return FuncAttribute(Attribute.CACHE_MODE_CA, Int(val))

    @always_inline
    @staticmethod
    fn MAX_DYNAMIC_SHARED_SIZE_BYTES(val: UInt32) -> FuncAttribute:
        """The maximum size in bytes of dynamically-allocated shared memory that
        can be used by this function. If the user-specified dynamic shared memory
        size is larger than this value, the launch will fail."""
        return FuncAttribute(
            Attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES, val.cast[DType.int32]()
        )

    @always_inline
    @staticmethod
    fn PREFERRED_SHARED_MEMORY_CARVEOUT(val: Int32) -> FuncAttribute:
        """On devices where the L1 cache and shared memory use the same hardware
        resources, this sets the shared memory carveout preference, in percent
        of the total shared memory."""
        return FuncAttribute(Attribute.PREFERRED_SHARED_MEMORY_CARVEOUT, val)
