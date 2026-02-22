# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
"""NVIDIA Tensor Memory Accelerator (TMA) module.

Provides types and functions for working with NVIDIA's Tensor Memory Accelerator,
which enables efficient asynchronous data movement between global and shared memory
on GPUs with Hopper architecture and newer.

The TMA hardware provides hardware-accelerated multi-dimensional memory copies with
features like swizzling for bank conflict avoidance, L2 cache promotion hints, and
support for various data types and memory layouts.
"""

from ffi import external_call
from sys import size_of

from gpu._utils import to_llvm_ptr
from gpu.host.device_context import (
    _ConstCharPtr,
    _checked,
    _DeviceBufferPtr,
)

from utils import IndexList, StaticTuple
from builtin.device_passable import DevicePassable


@fieldwise_init("implicit")
struct TensorMapDataType(TrivialRegisterPassable):
    """Data type enumeration for TMA tensor map descriptors.

    Specifies the element data type for TMA operations. The TMA hardware supports
    various numeric types including integers, floating-point, and specialized formats.
    """

    var _value: Int32

    comptime UINT8 = Self(0)
    """Unsigned 8-bit integer."""
    comptime UINT16 = Self(1)
    """Unsigned 16-bit integer."""
    comptime UINT32 = Self(2)
    """Unsigned 32-bit integer."""
    comptime INT32 = Self(3)
    """Signed 32-bit integer."""
    comptime UINT64 = Self(4)
    """Unsigned 64-bit integer."""
    comptime INT64 = Self(5)
    """Signed 64-bit integer."""
    comptime FLOAT16 = Self(6)
    """IEEE 754 16-bit floating-point."""
    comptime FLOAT32 = Self(7)
    """IEEE 754 32-bit floating-point."""
    comptime FLOAT64 = Self(8)
    """IEEE 754 64-bit floating-point."""
    comptime BFLOAT16 = Self(9)
    """Brain floating-point 16-bit format."""
    comptime FLOAT32_FTZ = Self(10)
    """32-bit float with flush-to-zero for denormals."""
    comptime TFLOAT32 = Self(11)
    """TensorFloat-32 format."""
    comptime TFLOAT32_FTZ = Self(12)
    """TensorFloat-32 with flush-to-zero for denormals."""

    @staticmethod
    fn from_dtype[dtype: DType]() -> Self:
        """Converts a Mojo `DType` to the corresponding TMA data type.

        Parameters:
            dtype: The Mojo data type to convert. Must be one of `DType.float32`,
                `DType.float16`, `DType.bfloat16`, `DType.uint8`,
                `DType.float8_e4m3fn`, or `DType.float8_e8m0fnu`.

        Constraints:
            The dtype must be one of the supported types listed above.

        Returns:
            The corresponding `TensorMapDataType` value.
        """
        comptime assert dtype in (
            DType.float32,
            DType.float16,
            DType.bfloat16,
            DType.uint8,
            DType.float8_e4m3fn,
            DType.float8_e8m0fnu,
        ), "Unsupported dtype"

        comptime if dtype == DType.float32:
            return Self.FLOAT32
        elif dtype == DType.float16:
            return Self.FLOAT16
        elif dtype in (DType.float8_e4m3fn, DType.float8_e8m0fnu, DType.uint8):
            return Self.UINT8
        else:
            return Self.BFLOAT16


@fieldwise_init("implicit")
struct TensorMapInterleave(TrivialRegisterPassable):
    """Interleave mode for TMA tensor map descriptors.

    Specifies how data elements are interleaved in memory for TMA operations.
    Interleaving can improve memory access patterns for certain workloads.
    """

    var _value: Int32

    comptime INTERLEAVE_NONE = Self(0)
    """No interleaving."""
    comptime INTERLEAVE_16B = Self(1)
    """16-byte interleaving."""
    comptime INTERLEAVE_32B = Self(2)
    """32-byte interleaving."""


@fieldwise_init("implicit")
struct TensorMapSwizzle(
    Equatable,
    Hashable,
    ImplicitlyCopyable,
    Intable,
    Stringable,
    TrivialRegisterPassable,
    Writable,
):
    """Swizzle mode for TMA tensor map descriptors.

    Swizzling permutes memory addresses to reduce shared memory bank conflicts
    on NVIDIA GPUs. Different swizzle modes apply XOR-based address transformations
    with different granularities (32B, 64B, or 128B).
    """

    var _value: Int32

    comptime SWIZZLE_NONE = Self(0)
    """No swizzling applied."""
    comptime SWIZZLE_32B = Self(1)
    """32-byte swizzle pattern."""
    comptime SWIZZLE_64B = Self(2)
    """64-byte swizzle pattern."""
    comptime SWIZZLE_128B = Self(3)
    """128-byte swizzle pattern."""

    @always_inline("nodebug")
    fn __int__(self) -> Int:
        """Converts the swizzle mode to an integer value.

        Returns:
            The integer representation of the swizzle mode.
        """
        return Int(self._value)

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        """Checks if two swizzle modes are equal.

        Args:
            other: The swizzle mode to compare with.

        Returns:
            True if the swizzle modes are equal, False otherwise.
        """
        return self._value == other._value

    @always_inline
    fn __ne__(self, other: Self) -> Bool:
        """Checks if two swizzle modes are not equal.

        Args:
            other: The swizzle mode to compare with.

        Returns:
            True if the swizzle modes are not equal, False otherwise.
        """
        return self._value != other._value

    @always_inline
    fn bytes(self) -> Int:
        """Gets the swizzle size in bytes.

        Returns:
            The swizzle size in bytes (0, 32, 64, or 128).
        """
        return Int((2**self._value) * 16)

    @no_inline
    fn __str__(self) -> String:
        """Converts the swizzle mode to a string representation.

        Returns:
            A string describing the swizzle mode.
        """
        return String.write(self)

    @always_inline
    fn write_to(self, mut writer: Some[Writer]):
        """Writes the swizzle mode to a writer.

        Args:
            writer: The writer to write to.
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
struct TensorMapL2Promotion(TrivialRegisterPassable):
    """L2 cache promotion hint for TMA tensor map descriptors.

    Specifies how much data to promote into the L2 cache during TMA operations.
    Promoting data to L2 can improve performance when the same data will be
    accessed multiple times.
    """

    var _value: Int32

    comptime NONE = Self(0)
    """No L2 promotion."""
    comptime L2_64B = Self(1)
    """Promote 64 bytes to L2 cache."""
    comptime L2_128B = Self(2)
    """Promote 128 bytes to L2 cache."""
    comptime L2_256B = Self(3)
    """Promote 256 bytes to L2 cache."""


@fieldwise_init("implicit")
struct TensorMapFloatOOBFill(TrivialRegisterPassable):
    """Out-of-bounds fill mode for floating-point TMA operations.

    Specifies how out-of-bounds memory accesses are handled for floating-point
    data types during TMA operations.
    """

    var _value: Int32

    comptime NONE = Self(0)
    """No special out-of-bounds handling."""
    comptime NAN_REQUEST_ZERO_FMA = Self(1)
    """Fill out-of-bounds values with NaN, request zero for FMA operations."""


# The TMA descriptor is a 128-byte opaque object filled by the driver API.
# It should be 64-byte aligned both on the host and the device (if passed to constant memory).
@align(64)
struct TMADescriptor(DevicePassable, ImplicitlyCopyable):
    """TMA tensor map descriptor.

    An opaque 128-byte descriptor that encodes all parameters for a TMA operation,
    including tensor dimensions, strides, data type, swizzle mode, and other
    configuration. This descriptor is created on the host using `create_tma_descriptor()`
    and can be passed to device code for use with TMA hardware instructions.

    The descriptor must be 64-byte aligned both on the host and device.
    """

    var data: StaticTuple[UInt8, 128]
    """The opaque 128-byte descriptor data."""

    comptime device_type: AnyType = TMADescriptor
    """The device-side type for this TMA descriptor."""

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        """Gets the type name for this descriptor.

        Returns:
            The string "TMADescriptor".
        """
        return "TMADescriptor"

    @always_inline
    fn __init__(out self):
        """Initializes an empty TMA descriptor.

        The descriptor data is uninitialized and must be filled using
        `create_tma_descriptor()` before use.
        """
        self.data = StaticTuple[UInt8, 128]()

    @always_inline
    fn __init__(out self, *, copy: Self):
        """Creates a copy of a TMA descriptor.

        Args:
            copy: The descriptor to copy.
        """
        self.data = copy.data


fn prefetch_tma_descriptor(desc_ptr: OpaquePointer[mut=False]):
    """Prefetches a TMA descriptor into the constant cache.

    Issues a hardware prefetch instruction to bring the TMA descriptor into
    the constant cache, which can improve performance when the descriptor
    will be used soon.

    Args:
        desc_ptr: Pointer to the TMA descriptor to prefetch.
    """
    __mlir_op.`nvvm.prefetch`[tensormap = __mlir_attr.unit](
        to_llvm_ptr(desc_ptr),
    )


@always_inline
fn create_tma_descriptor[
    dtype: DType,
    rank: Int,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
](
    global_buf: DeviceBuffer[dtype],
    global_shape: IndexList[rank],
    global_strides: IndexList[rank],
    shared_mem_shape: IndexList[rank],
) raises -> TMADescriptor:
    """Creates a TMA descriptor for tiled memory operations.

    Encodes tensor layout information into a 128-byte TMA descriptor that can
    be used with TMA hardware instructions to efficiently copy data between
    global and shared memory on NVIDIA GPUs.

    The descriptor specifies a mapping from a tile in shared memory to a region
    in global memory, including dimensions, strides, data type, and optional
    swizzling for bank conflict avoidance.

    Parameters:
        dtype: The element data type of the tensor.
        rank: The number of dimensions (1-5).
        swizzle_mode: The swizzle pattern to apply in shared memory.

    Args:
        global_buf: Device buffer containing the global memory tensor.
        global_shape: Dimensions of the tensor in global memory.
        global_strides: Strides (in elements) for each dimension in global memory.
            The tensor must be row-major (stride at innermost dimension equals 1).
        shared_mem_shape: Dimensions of the tile to be copied to shared memory.

    Returns:
        A TMA descriptor configured for the specified tensor layout.

    Raises:
        An error if the descriptor creation fails.
    """
    # TMADescriptor is @align(64) so stack allocation is automatically 64-byte aligned.
    var tma_descriptor = TMADescriptor()
    var tensor_map_ptr = UnsafePointer(to=tma_descriptor).bitcast[NoneType]()

    # NOTE: These are initialized in the comptime loop below.
    var global_dim_arg = InlineArray[Int64, rank](uninitialized=True)
    var global_strides_arg = InlineArray[Int64, rank](uninitialized=True)
    var box_dim_arg = InlineArray[Int32, rank](uninitialized=True)
    var element_stride_arg = InlineArray[Int32, rank](fill=1)

    comptime for i in range(rank):
        global_dim_arg[i] = Int64(global_shape[rank - i - 1])
        global_strides_arg[i] = Int64(
            global_strides[rank - i - 1] * size_of[dtype]()
        )
        box_dim_arg[i] = Int32(shared_mem_shape[rank - i - 1])

    debug_assert(
        global_strides_arg[0] == Int64(size_of[dtype]()),
        "TMA GMEM should be row-major, global stride",
        " at dim 0 should be size_of[dtype](): ",
        size_of[dtype](),
        " but is: ",
        global_strides_arg[0],
    )
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
            OpaquePointer[MutAnyOrigin],  # tensorMap
            Int32,  # tensorDataType
            Int32,  # tensorRank
            _DeviceBufferPtr,  #  globalAddress
            UnsafePointer[Int64, MutAnyOrigin],  # globalDim
            UnsafePointer[Int64, MutAnyOrigin],  # globalStrides
            UnsafePointer[Int32, MutAnyOrigin],  # boxDim
            UnsafePointer[Int32, MutAnyOrigin],  # elementStrides
            Int32,  # interleave
            Int32,  # swizzle
            Int32,  # l2Promotion
            Int32,  # oobFill
        ](
            tensor_map_ptr,
            TensorMapDataType.from_dtype[dtype]()._value,
            Int32(rank),
            global_buf._handle,
            global_dim_arg.unsafe_ptr(),
            # global_strides_arg[0] is implicitly size_of[dtype]()
            global_strides_arg.unsafe_ptr() + 1,
            box_dim_arg.unsafe_ptr(),
            element_stride_arg.unsafe_ptr(),
            TensorMapInterleave.INTERLEAVE_NONE._value,
            swizzle_mode._value,
            TensorMapL2Promotion.NONE._value,
            TensorMapFloatOOBFill.NONE._value,
        )
    )
    return tma_descriptor
