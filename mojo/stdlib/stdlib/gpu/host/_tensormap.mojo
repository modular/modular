# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from sys import external_call, size_of
from gpu.host import DeviceBuffer
from gpu.host.device_context import _checked, _ConstCharPtr, _DeviceBufferPtr
from memory import (
    LegacyOpaquePointer as OpaquePointer,
    LegacyUnsafePointer as UnsafePointer,
    stack_allocation,
)
from utils import IndexList, StaticTuple


@fieldwise_init("implicit")
@register_passable("trivial")
struct DataType:
    """
    Enum representing acceptable data types for the TensorMap descriptor.
    """

    var _value: Int32

    alias UINT8 = Self(0)
    alias UINT16 = Self(1)
    alias UINT32 = Self(2)
    alias INT32 = Self(3)
    alias UINT64 = Self(4)
    alias INT64 = Self(5)
    alias FLOAT16 = Self(6)
    alias FLOAT32 = Self(7)
    alias FLOAT64 = Self(8)
    alias BFLOAT16 = Self(9)
    alias FLOAT32_FTZ = Self(10)
    alias TFLOAT32 = Self(11)
    alias TFLOAT32_FTZ = Self(12)

    @staticmethod
    fn from_dtype[dtype: DType]() -> Self:
        """
        Convert a DType to a DataType enum value.

        Parameters:
            dtype: The data type to convert.

        Returns:
            The DataType enum value corresponding to the input data type.
        """
        constrained[
            dtype in (DType.float32, DType.bfloat16, DType.float8_e4m3fn),
            "Unsupported dtype",
        ]()

        @parameter
        if dtype is DType.float32:
            return Self.FLOAT32
        elif dtype is DType.float8_e4m3fn:
            return Self.UINT8
        else:
            return Self.BFLOAT16


@fieldwise_init("implicit")
@register_passable("trivial")
struct InterleaveMode:
    """Enum representing interleave modes for tensor memory access.

    Interleaving controls how data is distributed across memory channels
    to optimize memory bandwidth utilization.
    """

    var _value: Int32

    alias NONE = Self(0)
    alias _16B = Self(1)
    alias _32B = Self(2)


@fieldwise_init("implicit")
@register_passable("trivial")
struct SwizzleMode(
    Equatable,
    ImplicitlyCopyable,
    Intable,
    Movable,
    Stringable,
    Writable,
):
    """Enum representing memory swizzling patterns for tensor access optimization.

    Swizzling rearranges memory layout to improve cache performance and reduce
    memory bank conflicts. Different swizzle modes correspond to different
    memory access patterns optimized for specific tensor operations.
    """

    var _value: Int32

    alias NONE = Self(0)
    alias _32B = Self(1)
    alias _64B = Self(2)
    alias _128B = Self(3)

    @always_inline("nodebug")
    fn __int__(self) -> Int:
        """Convert SwizzleMode to integer representation.

        Returns:
            The integer value of the swizzle mode.
        """
        return Int(self._value)

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        """Check equality between two SwizzleMode instances.

        Args:
            other: The SwizzleMode to compare against.

        Returns:
            True if both instances have the same swizzle mode value.
        """
        return self._value == other._value

    @always_inline
    fn __ne__(self, other: Self) -> Bool:
        """Check inequality between two SwizzleMode instances.

        Args:
            other: The SwizzleMode to compare against.

        Returns:
            True if the instances have different swizzle mode values.
        """
        return self._value != other._value

    @always_inline
    fn bytes(self) -> Int:
        """Get the swizzle size in bytes.

        Returns:
            The swizzle pattern size in bytes. For example:
            - NONE (0): 16 bytes.
            - _32B (1): 32 bytes.
            - _64B (2): 64 bytes.
            - _128B (3): 128 bytes.
        """
        return Int((2**self._value) * 16)

    @no_inline
    fn __str__(self) -> String:
        """Convert SwizzleMode to string representation.

        Returns:
            A human-readable string describing the swizzle mode.
        """
        return String.write(self)

    @always_inline
    fn write_to(self, mut writer: Some[Writer]):
        """Write a human-readable representation of the SwizzleMode to a writer.

        Args:
            writer: The writer to output the string representation to.
        """
        if self._value == 1:
            writer.write("32B swizzle")
        elif self._value == 2:
            writer.write("64B swizzle")
        elif self._value == 3:
            writer.write("128B swizzle")
        elif self._value == 0:
            writer.write("no swizzle")
        else:
            writer.write("invalid swizzle")


@fieldwise_init("implicit")
@register_passable("trivial")
struct L2Promotion:
    """Enum representing L2 cache promotion policies for tensor data.

    L2 promotion controls how tensor data is cached in the L2 cache
    to improve memory access performance.
    """

    var _value: Int32

    alias NONE = Self(0)
    alias _64B = Self(1)
    alias _128B = Self(2)
    alias _256B = Self(3)


@fieldwise_init("implicit")
@register_passable("trivial")
struct OOBFill:
    """Enum representing out-of-bounds fill behavior for tensor access.

    Controls what values are returned when accessing tensor elements
    outside the defined tensor boundaries.
    """

    var _value: Int32

    alias NONE = Self(0)
    alias NAN_REQUEST_ZERO_FMA = Self(1)


# The TMA descriptor is a 128-byte opaque object filled by the driver API.
# It should be 64-byte aligned both on the host and the device (if passed to constant memory).
struct TensorMap(ImplicitlyCopyable):
    """A tensor memory access descriptor for optimized GPU tensor operations.

    TensorMap encapsulates a 128-byte opaque descriptor that is filled by the
    CUDA driver API. This descriptor contains all the information needed for
    efficient tensor memory access patterns, including tiling, swizzling, and
    caching policies.

    The descriptor must be 64-byte aligned both on the host and device memory
    to ensure proper hardware access patterns.
    """

    var data: StaticTuple[UInt8, 128]
    """The underlying 128-byte opaque descriptor data filled by the CUDA driver API."""

    @always_inline
    fn __init__(out self):
        """Initialize an empty TensorMap descriptor.

        Creates a zero-initialized 128-byte tensor map descriptor.
        The descriptor will need to be populated using create_tensormap()
        before it can be used for tensor operations.
        """
        self.data = StaticTuple[UInt8, 128]()

    @always_inline
    fn __copyinit__(out self, other: Self):
        """Copy constructor for TensorMap.

        Args:
            other: The TensorMap instance to copy from.
        """
        self.data = other.data


@always_inline
fn create_tensormap[
    dtype: DType,
    rank: Int, //,
](
    global_buf: DeviceBuffer[dtype],
    global_shape: IndexList[rank],
    global_strides: IndexList[rank],
    shared_mem_shape: IndexList[rank],
    swizzle_mode: SwizzleMode = SwizzleMode.NONE,
) raises -> TensorMap:
    """Create a tensor map descriptor object representing tiled memory region.

    Creates a TensorMap descriptor that enables efficient tensor memory access
    patterns on GPU devices. The descriptor encapsulates information about tensor
    layout, memory addressing, and access patterns to optimize data movement
    between global memory and shared memory.

    The function handles the conversion from row-major tensor layout (where the
    last dimension is contiguous) to the column-major layout expected by the
    underlying CUDA TensorMap API.

    Parameters:
        dtype: The data type of tensor elements (float32, bfloat16, or float8_e4m3fn).
        rank: The number of tensor dimensions.

    Args:
        global_buf: Device buffer containing the tensor data in global memory.
        global_shape: Shape of the tensor in global memory for each dimension.
        global_strides: Stride values for each dimension in the global tensor.
        shared_mem_shape: Shape of the tensor tile to be loaded into shared memory.
        swizzle_mode: Memory swizzling pattern to optimize cache performance. Defaults to SwizzleMode.NONE.

    Returns:
        A TensorMap descriptor configured for the specified tensor layout and
        access patterns.

    Raises:
        Error if the tensor configuration is invalid or if the underlying
        CUDA driver call fails.
    """
    # Tensormap must be 64 bytes aligned on host.
    # https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html
    var tensormap = stack_allocation[1, TensorMap, alignment=64]()[0]
    var tensormap_ptr = UnsafePointer(to=tensormap).bitcast[NoneType]()

    var global_dim_arg = InlineArray[Int64, rank](uninitialized=True)
    var global_strides_arg = InlineArray[Int64, rank](uninitialized=True)
    var box_dim_arg = InlineArray[Int32, rank](uninitialized=True)
    var element_stride_arg = InlineArray[Int32, rank](fill=1)

    # The input are row-major i.e. last dim is contiguous. The tensormap arguments
    # goes from the least rapidly varying dim to the highest. Here we inverse the
    # inputs for the tensormap constructor arguments.

    @parameter
    for i in range(rank):
        global_dim_arg[i] = global_shape[rank - i - 1]
        global_strides_arg[i] = global_strides[rank - i - 1] * size_of[dtype]()
        box_dim_arg[i] = shared_mem_shape[rank - i - 1]

    debug_assert(
        global_strides_arg[0] == size_of[dtype](),
        "TMA GMEM should be row-major, global stride",
        " at dim 0 should be size_of[dtype](): ",
        size_of[dtype](),
        " but is: ",
        global_strides_arg[0],
    )

    # Call cuDriver function `cuTensorMapEncodeTiled` via AsyncRT.
    # const char *AsyncRT_cuda_tensorMapEncodeTiled(
    #     void *tensorMap, int32_t tensorDataType, uint32_t tensorRank,
    #     const DeviceBuffer *globalAddress, const uint64_t *globalDim,
    #     const uint64_t *globalStrides, const uint32_t *boxDim,
    #     const uint32_t *elementStrides, int32_t interleave, int32_t swizzle,
    #     int32_t l2Promotion, int32_t oobFill) {

    _checked(
        external_call[
            "AsyncRT_cuda_tensorMapEncodeTiled",
            _ConstCharPtr,
            OpaquePointer,  # tensorMap
            Int32,  # tensorDataType
            Int32,  # tensorRank
            _DeviceBufferPtr,  #  globalAddress
            UnsafePointer[Int64],  # globalDim
            UnsafePointer[Int64],  # globalStrides
            UnsafePointer[Int32],  # boxDim
            UnsafePointer[Int32],  # elementStrides
            Int32,  # interleave
            Int32,  # swizzle
            Int32,  # l2Promotion
            Int32,  # oobFill
        ](
            tensormap_ptr,
            DataType.from_dtype[dtype]()._value,
            rank,
            global_buf._handle,
            global_dim_arg.unsafe_ptr(),
            # global_strides_arg[0] is implicitly size_of[dtype]()
            global_strides_arg.unsafe_ptr() + 1,
            box_dim_arg.unsafe_ptr(),
            element_stride_arg.unsafe_ptr(),
            InterleaveMode.NONE._value,
            swizzle_mode._value,
            L2Promotion.NONE._value,
            OOBFill.NONE._value,
        )
    )

    return tensormap
