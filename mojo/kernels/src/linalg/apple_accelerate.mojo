# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from os import abort
from sys.info import os_is_macos
from sys.ffi import DLHandle
from sys.ffi import _get_dylib_function as _ffi_get_dylib_function
from collections import OptionalReg as Optional
from buffer.buffer import NDBuffer
from .BatchedMatmul import _reshape_nd_buffer_with_batch_to_3d
from utils.index import Index

# ===----------------------------------------------------------------------===#
# Constants
# ===----------------------------------------------------------------------===#

alias LIB_ACC_PATH = "/System/Library/Frameworks/Accelerate.framework/Accelerate"
alias LIB_ACC_PLIST = "/System/Library/Frameworks/Accelerate.framework/Versions/Current/Resources/Info.plist"


# ===----------------------------------------------------------------------===#
# Library Load
# ===----------------------------------------------------------------------===#


fn _init_dylib(ignored: Pointer[NoneType]) -> Pointer[NoneType]:
    var handle = DLHandle(LIB_ACC_PATH)
    if not handle:
        abort("the accelerate library was not found at " + LIB_ACC_PATH)
    var ptr = Pointer[DLHandle].alloc(1)
    ptr[] = handle
    return ptr.bitcast[NoneType]()


fn _destroy_dylib(ptr: Pointer[NoneType]):
    ptr.bitcast[DLHandle]()[].close()
    ptr.free()


@always_inline
fn _get_dylib_function[
    func_name: StringLiteral, result_type: AnyRegType
]() -> result_type:
    constrained[os_is_macos(), "operating system must be macOS"]()
    return _ffi_get_dylib_function[
        "ACCELERATE_PATH",
        func_name,
        _init_dylib,
        _destroy_dylib,
        result_type,
    ]()


# ===----------------------------------------------------------------------===#
# CBLAS
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct _CBLASOrder:
    var value: Int32
    alias ROW_MAJOR = _CBLASOrder(101)
    alias COL_MAJOR = _CBLASOrder(102)


@value
@register_passable("trivial")
struct _CBLASTranspose:
    var value: Int32
    alias NO_TRANSPOSE = _CBLASTranspose(111)
    alias TRANSPOSE = _CBLASTranspose(112)
    alias CONJ_TRANSPOSE = _CBLASTranspose(113)


@always_inline
fn _cblas_f32[
    *,
    transpose_b: Bool = False,
](c: NDBuffer, a: NDBuffer, b: NDBuffer):
    constrained[a.rank == b.rank == c.rank == 2, "rank must be 2"]()
    constrained[
        a.type == b.type == c.type == DType.float32,
        "input and output types must be float32",
    ]()

    var M = Int32(a.dim[0]())
    var N = Int32(b.dim[0]() if transpose_b else b.dim[1]())
    var K = Int32(a.dim[1]())

    var lda = K
    var ldb = N
    var ldc = N

    # void cblas_sgemm(const enum CBLAS_ORDER ORDER,
    #                  const enum CBLAS_TRANSPOSE TRANSA,
    #                  const enum CBLAS_TRANSPOSE TRANSB,
    #                  const int M,
    #                  const int N,
    #                  const int K,
    #                  const float ALPHA,
    #                  const float *A,
    #                  const int LDA,
    #                  const float *B,
    #                  const int LDB,
    #                  const float BETA,
    #                  float *C,
    #                  const int LDC);
    var cblas_gemm = _get_dylib_function[
        "cblas_sgemm",
        fn (
            _CBLASOrder,
            _CBLASTranspose,
            _CBLASTranspose,
            Int32,
            Int32,
            Int32,
            Float32,
            DTypePointer[DType.float32],
            Int32,
            DTypePointer[DType.float32],
            Int32,
            Float32,
            DTypePointer[DType.float32],
            Int32,
        ) -> NoneType,
    ]()
    cblas_gemm(
        _CBLASOrder.ROW_MAJOR,
        _CBLASTranspose.NO_TRANSPOSE,
        _CBLASTranspose.TRANSPOSE if transpose_b else _CBLASTranspose.NO_TRANSPOSE,
        M,
        N,
        K,
        Float32(1.0),
        rebind[DTypePointer[DType.float32]](a.data),
        lda,
        rebind[DTypePointer[DType.float32]](b.data),
        ldb,
        Float32(0.0),
        rebind[DTypePointer[DType.float32]](c.data),
        ldc,
    )


# ===----------------------------------------------------------------------===#
# BNNS
# ===----------------------------------------------------------------------===#

alias _BNNS_MAX_TENSOR_DIMENSION = 8


@value
@register_passable("trivial")
struct _BNNSNDArrayFlags:
    var value: UInt32
    alias BACK_PROP_SET = _BNNSNDArrayFlags(0)
    """During back propogation, the elements of this ndarray are overwritten by
    the jacobian. If the NDArray does not represent a backpropgation gradient
    calculated by the function, this variable is ignored (i.e. this will typically
    only be used for output variables with names ending _delta).
    """
    alias BACK_PROP_ACCUMULATE = _BNNSNDArrayFlags(1)
    """During back propogation, the value of the jacobian is added to the elements
    of this ndarray (i.e. if the value is input_delta on entry to the function
    and dL/dx is the jacobian, then on exit it will be input_delta + dL/dx).
    If the NDArray does not represent a backpropgation gradient calculated by the
    function, this variable is ignored (i.e. this will typically
    only be used for output variables with names ending _delta).
    """


@value
@register_passable("trivial")
struct _BNNSDataLayout:
    """The layout defines the nature of the data stored in a N-dimensional
    array.
    """

    var value: UInt32

    # 1-dimensional layouts (are identical and only included for ease of use
    # and consistent numbering).
    alias VECTOR = _BNNSDataLayout(0x10000)
    alias LAST_MAJOR_1D = _BNNSDataLayout(0x18000)
    alias FIRST_MAJOR_1D = _BNNSDataLayout(0x18001)

    # 2-dimensional layouts.
    alias ROW_MAJOR_MATRIX = _BNNSDataLayout(0x20000)
    alias COLUMN_MAJOR_MATRIX = _BNNSDataLayout(0x20001)
    alias LAST_MAJOR_2D = _BNNSDataLayout(0x28000)
    alias FIRST_MAJOR_2D = _BNNSDataLayout(0x28001)
    alias FULLY_CONNECTED_SPARSE = _BNNSDataLayout(0x21001)

    # 3-dimensional layouts.
    alias IMAGE_CHW = _BNNSDataLayout(0x30000)
    alias SNE = _BNNSDataLayout(0x30001)
    alias NSE = _BNNSDataLayout(0x30002)
    alias MHA_DHK = _BNNSDataLayout(0x30003)
    alias LAST_MAJOR_3D = _BNNSDataLayout(0x38000)
    alias FIRST_MAJOR_3D = _BNNSDataLayout(0x38001)

    # 4-dimensional layouts.
    alias CONVOLUTION_WEIGHTS_OIHW = _BNNSDataLayout(0x40000)
    alias CONVOLUTION_WEIGHTS_OIHrWr = _BNNSDataLayout(0x40001)
    alias CONVOLUTION_WEIGHTS_IOHrWr = _BNNSDataLayout(0x40002)
    alias CONVOLUTION_WEIGHTS_IOHrWr_PACK32 = _BNNSDataLayout(0x40010)
    alias LAST_MAJOR_4D = _BNNSDataLayout(0x48000)
    alias FIRST_MAJOR_4D = _BNNSDataLayout(0x48001)

    # 5-dimensional layouts.
    alias LAST_MAJOR_5D = _BNNSDataLayout(0x58000)
    alias FIRST_MAJOR_5D = _BNNSDataLayout(0x58001)

    # 6-dimensional layouts.
    alias LAST_MAJOR_6D = _BNNSDataLayout(0x68000)
    alias FIRST_MAJOR_6D = _BNNSDataLayout(0x68001)

    # 7-dimensional layouts.
    alias LAST_MAJOR_7D = _BNNSDataLayout(0x78000)
    alias FIRST_MAJOR_7D = _BNNSDataLayout(0x78001)

    # 8-dimensional layouts.
    alias LAST_MAJOR_8D = _BNNSDataLayout(0x88000)
    alias FIRST_MAJOR_8D = _BNNSDataLayout(0x88001)


@value
@register_passable("trivial")
struct _BNNSDataType:
    var value: UInt32

    alias FLOAT_BIT = _BNNSDataType(0x10000)
    alias FLOAT16 = Self.FLOAT_BIT | 16
    alias FLOAT32 = Self.FLOAT_BIT | 32
    alias BFLOAT16 = Self.FLOAT_BIT | 0x8000 | 16

    alias INT_BIT = _BNNSDataType(0x20000)
    alias INT1 = Self.INT_BIT | 1
    alias INT2 = Self.INT_BIT | 2
    alias INT4 = Self.INT_BIT | 4
    alias INT8 = Self.INT_BIT | 8
    alias INT16 = Self.INT_BIT | 16
    alias INT32 = Self.INT_BIT | 32
    alias INT64 = Self.INT_BIT | 64

    alias UINT_BIT = _BNNSDataType(0x40000)
    alias UINT1 = Self.UINT_BIT | 1
    alias UINT2 = Self.UINT_BIT | 2
    alias UINT4 = Self.UINT_BIT | 4
    alias UINT8 = Self.UINT_BIT | 8
    alias UINT16 = Self.UINT_BIT | 16
    alias UINT32 = Self.UINT_BIT | 32
    alias UINT64 = Self.UINT_BIT | 64

    alias INDEXED_BIT = _BNNSDataType(0x80000)
    alias INDEXED_BIT1 = Self.INDEXED_BIT | 1
    alias INDEXED_BIT2 = Self.INDEXED_BIT | 2
    alias INDEXED_BIT4 = Self.INDEXED_BIT | 4
    alias INDEXED_BIT8 = Self.INDEXED_BIT | 8

    alias MISC_BIT = _BNNSDataType(0x100000)
    alias BOOL = Self.MISC_BIT | 8

    @always_inline
    fn __or__(self, other: Self) -> Self:
        return _BNNSDataType(self.value | other.value)

    @always_inline
    fn __or__(self, other: Int) -> Self:
        return self | _BNNSDataType(other)


@value
@register_passable("trivial")
struct _BNNSNDArrayDescriptor:
    """This type  is used to represent an N-dimensional array of values (N=1,2,3,4).
    The nature and dimension of the data is determined by the layout field."""

    var flags: _BNNSNDArrayFlags
    """Used to control some behaviors of the NDArray."""
    var layout: _BNNSDataLayout
    """Defines the dimension (n) of the array, and how data is stored."""

    var size: StaticIntTuple[_BNNS_MAX_TENSOR_DIMENSION]
    """The number of values in each dimension; only the first n values are used."""
    var stride: StaticIntTuple[_BNNS_MAX_TENSOR_DIMENSION]
    """the increment (in values) between a value and the next in each dimension;
    only the first n values are used.
    """

    var data: Pointer[NoneType]
    """Points to the data; can be NULL."""
    var data_type: _BNNSDataType
    """Defines the size and type of the values stored in data."""

    var table_data: Pointer[NoneType]
    """Points to the lookup table; used only when data_type is Indexed<N>."""
    var table_data_type: _BNNSDataType
    """Defines the size and type of the values stored in table_data; used only
    when data_type is Indexed<K>.
    """

    var data_scale: Float32
    """Used in the conversion of integer values to floating point; used only
    when data_type is Int<K> or UInt<K>.
    """
    var data_bias: Float32
    """Used in the conversion of integer values to floating point; used only
    when data_type is Int<K> or UInt<K>.
    """


# ===----------------------------------------------------------------------===#
# Matmul
# ===----------------------------------------------------------------------===#


@always_inline
fn matmul[
    *,
    transpose_b: Bool = False,
](c: NDBuffer, a: NDBuffer, b: NDBuffer):
    @parameter
    if a.type == b.type == c.type == DType.float32:
        return _cblas_f32[transpose_b=transpose_b](c, a, b)

    constrained[False, "unsupported type in apple accelerate"]()


# ===----------------------------------------------------------------------===#
# Batched Matmul
# ===----------------------------------------------------------------------===#


@always_inline
fn batched_matmul[
    *,
    transpose_b: Bool = False,
](c: NDBuffer, a: NDBuffer, b: NDBuffer):
    var c3 = _reshape_nd_buffer_with_batch_to_3d(c)
    var a3 = _reshape_nd_buffer_with_batch_to_3d(a)
    var b3 = _reshape_nd_buffer_with_batch_to_3d(b)
    var batch_size = c.dim[0]()

    var m = c3.dim[1]()
    var n = c3.dim[2]()
    var k = a3.dim[2]()

    var c_shape = Index(c3.dim[1](), c3.dim[2]())
    var a_shape = Index(a3.dim[1](), a3.dim[2]())
    var b_shape = Index(b3.dim[1](), b3.dim[2]())

    for batch in range(batch_size):
        var c2 = NDBuffer[c.type, 2, address_space = c.address_space](
            c3.data + (c_shape[0] * c_shape[1]) * batch, c_shape
        )
        var a2 = NDBuffer[a.type, 2, address_space = a.address_space](
            a3.data + (a_shape[0] * a_shape[1]) * batch, a_shape
        )
        var b2 = NDBuffer[b.type, 2, address_space = b.address_space](
            b3.data + (b_shape[0] * b_shape[1]) * batch, b_shape
        )

        matmul[transpose_b=transpose_b](c2, a2, b2)
