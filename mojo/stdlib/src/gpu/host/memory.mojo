# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements CUDA memory operations."""

from sys.info import sizeof

from memory import UnsafePointer
from memory.unsafe import bitcast

from ._utils import _check_error, _get_dylib_function
from .stream import Stream, _StreamHandle

# ===----------------------------------------------------------------------===#
# Memory
# ===----------------------------------------------------------------------===#


fn _malloc[type: AnyType](count: Int) raises -> UnsafePointer[type]:
    """Allocates GPU device memory."""

    var ptr = UnsafePointer[Int]()
    _check_error(
        _get_dylib_function[
            "cuMemAlloc_v2",
            fn (UnsafePointer[UnsafePointer[Int]], Int) -> Result,
        ]()(UnsafePointer.address_of(ptr), count * sizeof[type]())
    )
    return ptr.bitcast[type]()


fn _malloc[type: DType](count: Int) raises -> UnsafePointer[Scalar[type]]:
    return _malloc[Scalar[type]](count)


fn _malloc_managed[type: AnyType](count: Int) raises -> UnsafePointer[type]:
    """Allocates memory that will be automatically managed by the Unified Memory system.
    """
    alias CU_MEM_ATTACH_GLOBAL = UInt32(1)
    var ptr = UnsafePointer[Int]()
    _check_error(
        _get_dylib_function[
            "cuMemAllocManaged",
            fn (UnsafePointer[UnsafePointer[Int]], Int, UInt32) -> Result,
        ]()(
            UnsafePointer.address_of(ptr),
            count * sizeof[type](),
            CU_MEM_ATTACH_GLOBAL,
        )
    )
    return ptr.bitcast[type]()


fn _malloc_managed[
    type: DType
](count: Int) raises -> UnsafePointer[Scalar[type]]:
    return _malloc_managed[Scalar[type]](count)


fn _free[type: AnyType](ptr: UnsafePointer[type]) raises:
    """Frees allocated GPU device memory."""

    _check_error(
        _get_dylib_function[
            "cuMemFree_v2", fn (UnsafePointer[Int]) -> Result
        ]()(ptr.bitcast[Int]())
    )


fn _copy_host_to_device[
    type: AnyType
](
    device_dest: UnsafePointer[type], host_src: UnsafePointer[type], count: Int
) raises:
    """Copies memory from host to device."""

    _check_error(
        _get_dylib_function[
            "cuMemcpyHtoD_v2",
            fn (UnsafePointer[Int], UnsafePointer[NoneType], Int) -> Result,
        ]()(
            device_dest.bitcast[Int](),
            host_src.bitcast[NoneType](),
            count * sizeof[type](),
        )
    )


fn _copy_host_to_device_async[
    type: AnyType
](
    device_dst: UnsafePointer[type],
    host_src: UnsafePointer[type],
    count: Int,
    stream: Stream,
) raises:
    """Copies memory from host to device asynchronously."""

    _check_error(
        _get_dylib_function[
            "cuMemcpyHtoDAsync_v2",
            fn (
                UnsafePointer[NoneType], UnsafePointer[Int], Int, _StreamHandle
            ) -> Result,
        ]()(
            device_dst.bitcast[NoneType](),
            host_src.bitcast[Int](),
            count * sizeof[type](),
            stream.stream,
        )
    )


fn _copy_device_to_host[
    type: AnyType
](
    host_dest: UnsafePointer[type], device_src: UnsafePointer[type], count: Int
) raises:
    """Copies memory from device to host."""

    _check_error(
        _get_dylib_function[
            "cuMemcpyDtoH_v2",
            fn (UnsafePointer[NoneType], UnsafePointer[Int], Int) -> Result,
        ]()(
            host_dest.bitcast[NoneType](),
            device_src.bitcast[Int](),
            count * sizeof[type](),
        )
    )


fn _copy_device_to_host_async[
    type: AnyType
](
    host_dest: UnsafePointer[type],
    device_src: UnsafePointer[type],
    count: Int,
    stream: Stream,
) raises:
    """Copies memory from device to host asynchronously."""

    _check_error(
        _get_dylib_function[
            "cuMemcpyDtoHAsync_v2",
            fn (
                UnsafePointer[NoneType], UnsafePointer[Int], Int, _StreamHandle
            ) -> Result,
        ]()(
            host_dest.bitcast[NoneType](),
            device_src.bitcast[Int](),
            count * sizeof[type](),
            stream.stream,
        )
    )


fn _copy_device_to_device_async[
    type: AnyType
](
    dst: UnsafePointer[type],
    src: UnsafePointer[type],
    count: Int,
    stream: Stream,
) raises:
    """Copies memory from device to device asynchronously."""

    _check_error(
        _get_dylib_function[
            "cuMemcpyDtoDAsync_v2",
            fn (
                UnsafePointer[NoneType], UnsafePointer[Int], Int, _StreamHandle
            ) -> Result,
        ]()(
            dst.bitcast[NoneType](),
            src.bitcast[Int](),
            count * sizeof[type](),
            stream.stream,
        )
    )


fn _memset[
    type: AnyType
](device_dest: UnsafePointer[type], val: UInt8, count: Int) raises:
    """Sets the memory range of N 8-bit values to a specified value."""

    _check_error(
        _get_dylib_function[
            "cuMemsetD8_v2", fn (UnsafePointer[Int], UInt8, Int) -> Result
        ]()(
            device_dest.bitcast[Int](),
            val,
            count * sizeof[type](),
        )
    )


fn _memset_async[
    type: DType
](
    device_dest: UnsafePointer[Scalar[type]],
    val: Scalar[type],
    count: Int,
    stream: Stream,
) raises:
    """Sets the memory range of N 8-bit, 16-bit and 32-bit values to a specified value asynchronously.
    """

    alias bitwidth = bitwidthof[type]()
    constrained[
        bitwidth == 8 or bitwidth == 16 or bitwidth == 32,
        "bitwidth of memset type must be one of [8,16,32]",
    ]()

    @parameter
    if bitwidth == 8:
        _check_error(
            _get_dylib_function[
                "cuMemsetD8Async",
                fn (UnsafePointer[UInt8], UInt8, Int, _StreamHandle) -> Result,
            ]()(
                device_dest.bitcast[DType.uint8](),
                bitcast[DType.uint8, 1](val),
                count * sizeof[type](),
                stream.stream,
            )
        )
    elif bitwidth == 16:
        _check_error(
            _get_dylib_function[
                "cuMemsetD16Async",
                fn (
                    UnsafePointer[UInt16], UInt16, Int, _StreamHandle
                ) -> Result,
            ]()(
                device_dest.bitcast[DType.uint16](),
                bitcast[DType.uint16, 1](val),
                count,
                stream.stream,
            )
        )
    elif bitwidth == 32:
        _check_error(
            _get_dylib_function[
                "cuMemsetD32Async",
                fn (
                    UnsafePointer[UInt32], UInt32, Int, _StreamHandle
                ) -> Result,
            ]()(
                device_dest.bitcast[DType.uint32](),
                bitcast[DType.uint32, 1](val),
                count,
                stream.stream,
            )
        )


@always_inline
fn _copy_device_to_device[
    type: AnyType
](
    device_dest: UnsafePointer[type],
    device_src: UnsafePointer[type],
    count: Int,
) raises:
    """Copies memory from device to device."""

    _check_error(
        _get_dylib_function[
            "cuMemcpyDtoD_v2",
            fn (
                UnsafePointer[Int],
                UnsafePointer[Int],
                Int,
            ) -> Result,
        ]()(
            device_dest.bitcast[Int](),
            device_src.bitcast[Int](),
            count * sizeof[type](),
        )
    )


fn _malloc_async[
    type: AnyType
](count: Int, stream: Stream) raises -> UnsafePointer[type]:
    """Allocates memory with stream ordered semantics."""

    var ptr = UnsafePointer[Int]()
    _check_error(
        _get_dylib_function[
            "cuMemAllocAsync",
            fn (
                UnsafePointer[UnsafePointer[Int]], Int, _StreamHandle
            ) -> Result,
        ]()(
            UnsafePointer.address_of(ptr), count * sizeof[type](), stream.stream
        )
    )
    return ptr.bitcast[type]()


fn _free_async[type: AnyType](ptr: UnsafePointer[type], stream: Stream) raises:
    """Frees memory with stream ordered semantics."""

    _check_error(
        _get_dylib_function[
            "cuMemFreeAsync", fn (UnsafePointer[Int], _StreamHandle) -> Result
        ]()(ptr.bitcast[Int](), stream.stream)
    )


# ===----------------------------------------------------------------------===#
# TMA
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct TensorMapDataType:
    var _value: Int32

    alias UINT8 = Self(0)
    alias UINT16 = Self(1)
    alias UINT32 = Self(3)
    alias INT32 = Self(4)
    alias UINT64 = Self(5)
    alias INT64 = Self(6)
    alias FLOAT16 = Self(7)
    alias FLOAT32 = Self(8)
    alias FLOAT64 = Self(9)
    alias BFLOAT16 = Self(10)
    alias FLOAT32_FTZ = Self(11)
    alias TFLOAT32 = Self(12)
    alias TFLOAT32_FTZ = Self(13)

    fn __init__(inout self, value: Int32):
        self._value = value

    @staticmethod
    fn from_dtype[dtype: DType]() -> Self:
        constrained[
            dtype in (DType.float32, DType.bfloat16), "Unsupported dtype"
        ]()

        @parameter
        if dtype is DType.float32:
            return Self.FLOAT32
        else:
            return Self.BFLOAT16


@value
@register_passable("trivial")
struct TensornsorMapInterleave:
    var _value: Int32

    alias INTERLEAVE_NONE = Self(0)
    alias INTERLEAVE_16B = Self(1)
    alias INTERLEAVE_32B = Self(2)

    fn __init__(inout self, value: Int32):
        self._value = value


@value
@register_passable("trivial")
struct TensorMapSwizzle:
    var _value: Int32

    alias SWIZZLE_NONE = Self(0)
    alias SWIZZLE_32B = Self(1)
    alias SWIZZLE_64B = Self(2)
    alias SWIZZLE_128B = Self(3)

    fn __init__(inout self, value: Int32):
        self._value = value


@value
@register_passable("trivial")
struct TensorMapL2Promotion:
    var _value: Int32

    alias NONE = Self(0)
    alias L2_64B = Self(1)
    alias L2_128B = Self(2)
    alias L2_256B = Self(3)

    fn __init__(inout self, value: Int32):
        self._value = value


@value
@register_passable
struct TensorMapFloatOOBFill:
    var _value: Int32

    alias NONE = Self(0)
    alias NAN_REQUEST_ZERO_FMA = Self(1)

    fn __init__(inout self, value: Int32):
        self._value = value


fn create_tma_descriptor[
    dtype: DType, rank: Int
](
    global_ptr: UnsafePointer[Scalar[dtype]],
    global_shape: StaticIntTuple[rank],
    global_strides: StaticIntTuple[rank],
    shared_mem_shape: StaticIntTuple[rank],
    element_stride: StaticIntTuple[rank] = StaticIntTuple[rank](1),
) raises -> UnsafePointer[NoneType]:
    """Create a tensor map descriptor object representing tiled memory region.
    """

    var tensor_map_ptr = UnsafePointer[Int].alloc(32).bitcast[
        NoneType
    ]()  # size the sizeof(TensorMap{})

    var global_dim_arg = stack_allocation[rank, Int64]()
    var global_strides_arg = stack_allocation[rank - 1, Int64]()
    var box_dim_arg = stack_allocation[rank, Int32]()
    var element_stride_arg = stack_allocation[rank, Int32]()

    @parameter
    for i in range(rank):
        global_dim_arg[i] = global_shape[rank - i - 1]
        box_dim_arg[i] = shared_mem_shape[rank - i - 1]
        element_stride_arg[i] = element_stride[rank - i - 1]

    @parameter
    for i in range(rank - 1):
        global_strides_arg[i] = global_strides[rank - i - 2] * sizeof[dtype]()

    _check_error(
        _get_dylib_function[
            "cuTensorMapEncodeTiled",
            fn (
                UnsafePointer[NoneType],  # tensorMap
                Int32,  # tensorDataType
                Int32,  # tensorRank
                UnsafePointer[NoneType],  #  globalAddress
                UnsafePointer[Int64],  # globalDim
                UnsafePointer[Int64],  # globalStrides
                UnsafePointer[Int32],  # boxDim
                UnsafePointer[Int32],  # elementStrides
                Int32,  # interleave
                Int32,  # swizzle
                Int32,  # l2Promotion
                Int32,  # oobFill
            ) -> Result,
        ]()(
            tensor_map_ptr.bitcast[NoneType](),
            TensorMapDataType.from_dtype[dtype]()._value,
            rank,
            global_ptr.bitcast[NoneType](),
            global_dim_arg,
            global_strides_arg,
            box_dim_arg,
            element_stride_arg,
            TensornsorMapInterleave.INTERLEAVE_NONE._value,
            TensorMapSwizzle.SWIZZLE_NONE._value,
            TensorMapL2Promotion.NONE._value,
            TensorMapFloatOOBFill.NONE._value,
        )
    )
    return tensor_map_ptr.bitcast[NoneType]()
