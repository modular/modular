# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements CUDA memory operations."""

from os.path import isdir
from pathlib import Path
from sys import sizeof
from sys.ffi import DLHandle, _get_dylib_function

from builtin._location import __call_location, _SourceLocation
from gpu.host.result_v1 import Result
from memory import UnsafePointer, stack_allocation
from memory.unsafe import bitcast

from utils import StaticTuple

# ===-----------------------------------------------------------------------===#
# Utilities
# ===-----------------------------------------------------------------------===#


@always_inline
fn _check_error(
    err: Result,
    *,
    msg: StringLiteral = "",
    location: OptionalReg[_SourceLocation] = None,
) raises:
    _check_error_impl(err, msg, location.or_else(__call_location()))


@no_inline
fn _check_error_impl(
    err: Result,
    msg: StringLiteral,
    location: _SourceLocation,
) raises:
    """We do not want to inline this code since we want to make sure that the
    stringification of the error is not duplicated many times."""
    if err != Result.SUCCESS:
        raise Error(location.prefix(str(err) + " " + msg))


# ===-----------------------------------------------------------------------===#
# Library Load
# ===-----------------------------------------------------------------------===#


fn _get_cuda_driver_path() raises -> Path:
    alias _CUDA_DRIVER_LIB_NAME = "libcuda.so"
    var _DEFAULT_CUDA_DRIVER_BASE_PATHS = List[Path](
        Path("/usr/lib/x86_64-linux-gnu"),  # Ubuntu like
        Path("/usr/lib64/nvidia"),  # Redhat like
        Path("/usr/lib/wsl/lib"),  # WSL
    )

    # Quick lookup for libcuda.so assuming it's symlinked.
    for loc in _DEFAULT_CUDA_DRIVER_BASE_PATHS:
        var lib_path = loc[] / _CUDA_DRIVER_LIB_NAME
        if lib_path.exists():
            return lib_path

    # If we cannot find libcuda.so, then search harder.
    for loc in _DEFAULT_CUDA_DRIVER_BASE_PATHS:
        if isdir(loc[]):
            for file in loc[].listdir():
                var lib_path = loc[] / file[]
                if not lib_path.is_file():
                    continue
                if _CUDA_DRIVER_LIB_NAME in str(file[]):
                    return lib_path

    raise "the CUDA library was not found"


fn _init_dylib(ignored: UnsafePointer[NoneType]) -> UnsafePointer[NoneType]:
    try:
        var driver_path = _get_cuda_driver_path()
        var ptr = UnsafePointer[DLHandle].alloc(1)
        ptr.init_pointee_move(DLHandle(str(driver_path)))
        _ = ptr[].get_function[fn (UInt32) -> Result]("cuInit")(0)
        return ptr.bitcast[NoneType]()
    except e:
        return abort[UnsafePointer[NoneType]](e)


fn _destroy_dylib(ptr: UnsafePointer[NoneType]):
    ptr.bitcast[DLHandle]()[].close()
    ptr.free()


@always_inline
fn _get_cuda_dylib_function[
    func_name: StringLiteral, result_type: AnyTrivialRegType
]() -> result_type:
    return _get_dylib_function[
        "CUDA_DRIVER_LIBRARY",
        func_name,
        _init_dylib,
        _destroy_dylib,
        result_type,
    ]()


# ===-----------------------------------------------------------------------===#
# Memory
# ===-----------------------------------------------------------------------===#


fn _malloc[type: AnyType](count: Int) raises -> UnsafePointer[type]:
    """Allocates GPU device memory."""

    var ptr = UnsafePointer[Int]()
    _check_error(
        _get_cuda_dylib_function[
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
        _get_cuda_dylib_function[
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
        _get_cuda_dylib_function[
            "cuMemFree_v2", fn (UnsafePointer[Int]) -> Result
        ]()(ptr.bitcast[Int]())
    )


fn _memset[
    type: AnyType
](device_dest: UnsafePointer[type], val: UInt8, count: Int) raises:
    """Sets the memory range of N 8-bit values to a specified value."""

    _check_error(
        _get_cuda_dylib_function[
            "cuMemsetD8_v2", fn (UnsafePointer[Int], UInt8, Int) -> Result
        ]()(
            device_dest.bitcast[Int](),
            val,
            count * sizeof[type](),
        )
    )


# ===-----------------------------------------------------------------------===#
# TMA
# ===-----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct TensorMapDataType:
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

    @implicit
    fn __init__(out self, value: Int32):
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

    @implicit
    fn __init__(out self, value: Int32):
        self._value = value


@value
@register_passable("trivial")
struct TensorMapSwizzle:
    var _value: Int32

    alias SWIZZLE_NONE = Self(0)
    alias SWIZZLE_32B = Self(1)
    alias SWIZZLE_64B = Self(2)
    alias SWIZZLE_128B = Self(3)

    @implicit
    fn __init__(out self, value: Int32):
        self._value = value


@value
@register_passable("trivial")
struct TensorMapL2Promotion:
    var _value: Int32

    alias NONE = Self(0)
    alias L2_64B = Self(1)
    alias L2_128B = Self(2)
    alias L2_256B = Self(3)

    @implicit
    fn __init__(out self, value: Int32):
        self._value = value


@value
@register_passable
struct TensorMapFloatOOBFill:
    var _value: Int32

    alias NONE = Self(0)
    alias NAN_REQUEST_ZERO_FMA = Self(1)

    @implicit
    fn __init__(out self, value: Int32):
        self._value = value


# The TMA descriptor is a 128-byte opaque object filled by the driver API.
# It should be 64-byte aligned both on the host and the device (if passed to constant memory).
struct TMADescriptor:
    var data: StaticTuple[UInt8, 128]

    @always_inline
    fn __copyinit__(out self, other: Self):
        self.data = other.data


@always_inline
fn create_tma_descriptor[
    dtype: DType, rank: Int
](
    global_ptr: UnsafePointer[
        Scalar[dtype], address_space = AddressSpace.GENERIC
    ],
    global_shape: IndexList[rank],
    global_strides: IndexList[rank],
    shared_mem_shape: IndexList[rank],
    element_stride: IndexList[rank] = IndexList[rank](1),
) raises -> TMADescriptor:
    """Create a tensor map descriptor object representing tiled memory region.
    """
    # Enforces host-side aligment
    var tma_descriptor = stack_allocation[1, TMADescriptor, alignment=64]()[0]
    var tensor_map_ptr = UnsafePointer.address_of(tma_descriptor).bitcast[
        NoneType
    ]()

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
        _get_cuda_dylib_function[
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
            tensor_map_ptr,
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
    return tma_descriptor
