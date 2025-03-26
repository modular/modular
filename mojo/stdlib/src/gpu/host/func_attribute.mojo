# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""
GPU Kernel Function Attributes Module

This module provides structures for defining and managing GPU kernel function attributes.
It implements functionality similar to CUDA's CUfunction_attribute enum, allowing
for querying and setting various attributes that control kernel execution behavior
and resource allocation.

The module includes:
- `Attribute`: A value type representing different GPU kernel function attribute types
- `FuncAttribute`: A structure that pairs an attribute type with its value

These structures enable fine-grained control over GPU kernel execution parameters
such as shared memory allocation, cache behavior, and cluster configuration.
"""


@value
@register_passable("trivial")
struct Attribute(Writable):
    """Represents GPU kernel function attributes.

    This struct defines constants for various function attributes that can be queried
    or set for GPU kernels. These attributes provide information about resource
    requirements and execution constraints of kernel functions.
    """

    var code: Int32
    """The numeric code representing the attribute type."""

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
        """Checks if two Attribute instances are equal.

        Args:
            other: The Attribute to compare with.

        Returns:
            True if both attributes have the same code, False otherwise.
        """
        return self.code == other.code

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        """Checks if two Attribute instances are not equal.

        Args:
            other: The Attribute to compare with.

        Returns:
            True if the attributes have different codes, False otherwise.
        """
        return not (self == other)

    @always_inline("nodebug")
    fn __is__(self, other: Self) -> Bool:
        """Identity comparison operator for Attribute instances.

        Args:
            other: The Attribute to compare with.

        Returns:
            True if both attributes are identical (have the same code), False otherwise.
        """
        return self == other

    @always_inline("nodebug")
    fn __isnot__(self, other: Self) -> Bool:
        """Negative identity comparison operator for Attribute instances.

        Args:
            other: The Attribute to compare with.

        Returns:
            True if the attributes are not identical, False otherwise.
        """
        return not (self is other)

    fn write_to[W: Writer](self, mut writer: W):
        """Writes a string representation of the `Attribute` to the provided writer.

            This method converts the `Attribute` enum value to its corresponding string name
            and writes it to the provided writer object.

        Parameters:
            W: The type of writer to use for output. Must implement the Writer trait.

        Args:
            writer: A Writer object that will receive the string representation.
        """
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
    """Implements CUDA's CUfunction_attribute enum for GPU kernel function attributes.

    This struct represents function attributes that can be set or queried for GPU kernels,
    following NVIDIA's CUDA driver API conventions. Each attribute consists of a type
    (represented by the Attribute enum) and an associated value.

    The struct provides factory methods for creating common attribute configurations,
    such as cache mode settings and shared memory allocations.

    Reference: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g9d955dde0904a9b43ca4d875ac1551bc
    """

    var attribute: Attribute
    """The type of function attribute."""

    var value: Int32
    """The value associated with this attribute."""

    alias NULL = FuncAttribute(Attribute(-1), -1)
    """A null/invalid function attribute constant."""

    fn __init__(out self, *, other: Self):
        """Explicitly construct a deep copy of the provided value.

        Args:
            other: The value to copy.
        """
        self = other

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        """Checks if two `FuncAttribute` instances are equal.

        Args:
            other: The FuncAttribute to compare with.

        Returns:
            True if both the attribute type and value are equal, False otherwise.
        """
        return self.attribute == other.attribute and self.value == other.value

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        """Checks if two `FuncAttribute` instances are not equal.

        Args:
            other: The `FuncAttribute` to compare with.

        Returns:
            True if either the attribute type or value differs, False otherwise.
        """
        return not (self == other)

    @always_inline
    @staticmethod
    fn CACHE_MODE_CA(val: Bool) -> FuncAttribute:
        """Creates a CACHE_MODE_CA function attribute.

        Indicates whether the function has been compiled with user specified
        option `CacheMode.L1_CACHE_DISABLED` set.

        Args:
            val: Boolean value indicating if L1 cache is disabled.

        Returns:
            A `FuncAttribute` instance with CACHE_MODE_CA attribute type.
        """
        return FuncAttribute(Attribute.CACHE_MODE_CA, Int(val))

    @always_inline
    @staticmethod
    fn MAX_DYNAMIC_SHARED_SIZE_BYTES(val: UInt32) -> FuncAttribute:
        """Creates a MAX_DYNAMIC_SHARED_SIZE_BYTES function attribute.

        The maximum size in bytes of dynamically-allocated shared memory that
        can be used by this function. If the user-specified dynamic shared memory
        size is larger than this value, the launch will fail.

        Args:
            val: Maximum dynamic shared memory size in bytes.

        Returns:
            A `FuncAttribute` instance with `MAX_DYNAMIC_SHARED_SIZE_BYTES` attribute type.
        """
        return FuncAttribute(
            Attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES, val.cast[DType.int32]()
        )

    @always_inline
    @staticmethod
    fn PREFERRED_SHARED_MEMORY_CARVEOUT(val: Int32) -> FuncAttribute:
        """Creates a PREFERRED_SHARED_MEMORY_CARVEOUT function attribute.

        On devices where the L1 cache and shared memory use the same hardware
        resources, this sets the shared memory carveout preference, in percent
        of the total shared memory.

        Args:
            val: Shared memory carveout preference as a percentage (0-100).

        Returns:
            A FuncAttribute instance with `PREFERRED_SHARED_MEMORY_CARVEOUT` attribute type.
        """
        return FuncAttribute(Attribute.PREFERRED_SHARED_MEMORY_CARVEOUT, val)
