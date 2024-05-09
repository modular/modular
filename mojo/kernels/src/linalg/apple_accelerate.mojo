# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from os import abort
from sys.info import os_is_macos, bitwidthof
from sys.ffi import DLHandle
from sys.ffi import _get_dylib_function as _ffi_get_dylib_function
from collections import OptionalReg
from buffer.buffer import NDBuffer
from .BatchedMatmul import _reshape_nd_buffer_with_batch_to_3d
from utils.index import Index
from .MatmulUtils import elementwise_epilogue_type
from algorithm import elementwise, vectorize
from algorithm.functional import parallelize_over_rows
from buffer.list import DimList
from math import fma
from .MatmulPack import pack_b_ndbuffer


# ===----------------------------------------------------------------------===#
# Constants
# ===----------------------------------------------------------------------===#

alias LIB_ACC_PATH = "/System/Library/Frameworks/Accelerate.framework/Accelerate"
alias LIB_ACC_PLIST = "/System/Library/Frameworks/Accelerate.framework/Versions/Current/Resources/Info.plist"


# ===----------------------------------------------------------------------===#
# Library Load
# ===----------------------------------------------------------------------===#


fn _init_dylib(ignored: UnsafePointer[NoneType]) -> UnsafePointer[NoneType]:
    var handle = DLHandle(LIB_ACC_PATH)
    if not handle:
        abort("the accelerate library was not found at " + LIB_ACC_PATH)
    var ptr = UnsafePointer[DLHandle].alloc(1)
    ptr[] = handle
    return ptr.bitcast[NoneType]()


fn _destroy_dylib(ptr: UnsafePointer[NoneType]):
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


@always_inline
fn use_apple_accelerate_lib[
    c_type: DType,
    a_type: DType,
    b_type: DType,
]() -> Bool:
    return os_is_macos() and a_type == b_type == c_type == DType.float32


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
    var ldb = N if not transpose_b else K
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

    fn __init__(inout self):
        self.value = 0


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

    fn __init__(inout self):
        self.value = 0


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

    fn __init__(inout self):
        self.value = 0

    fn __init__(inout self, type: DType):
        if type == DType.bool:
            self = Self.BOOL
        elif type == DType.float16:
            self = Self.FLOAT16
        elif type == DType.bfloat16:
            self = Self.BFLOAT16
        elif type == DType.float32:
            self = Self.FLOAT32
        elif type == DType.int8:
            self = Self.INT8
        elif type == DType.uint8:
            self = Self.UINT8
        elif type == DType.int16:
            self = Self.INT16
        elif type == DType.uint16:
            self = Self.UINT16
        elif type == DType.int32:
            self = Self.INT32
        elif type == DType.uint32:
            self = Self.UINT32
        elif type == DType.int64:
            self = Self.INT64
        elif type == DType.uint64:
            self = Self.UINT64
        elif type == DType.index:
            self = Self.INT64 if bitwidthof[DType.index]() == 64 else Self.INT32
        else:
            abort("invalid dtype " + str(type))
            self = Self.MISC_BIT

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

    fn __init__(inout self):
        self.flags = _BNNSNDArrayFlags()
        self.layout = _BNNSDataLayout()

        self.size = StaticIntTuple[_BNNS_MAX_TENSOR_DIMENSION](0)
        self.stride = StaticIntTuple[_BNNS_MAX_TENSOR_DIMENSION](0)

        self.data = Pointer[NoneType]()
        self.data_type = _BNNSDataType()

        self.table_data = Pointer[NoneType]()
        self.table_data_type = _BNNSDataType()

        self.data_scale = 0
        self.data_bias = 0

    fn __init__(
        inout self,
        *,
        flags: _BNNSNDArrayFlags,
        layout: _BNNSDataLayout,
        size: StaticIntTuple[_BNNS_MAX_TENSOR_DIMENSION],
        stride: StaticIntTuple[_BNNS_MAX_TENSOR_DIMENSION],
        data: Pointer[NoneType],
        data_type: _BNNSDataType,
        table_data: Pointer[NoneType] = Pointer[NoneType](),
        table_data_type: _BNNSDataType = _BNNSDataType(),
        data_scale: Float32 = 1,
        data_bias: Float32 = 0,
    ):
        self.flags = flags
        self.layout = layout
        self.size = size
        self.stride = stride
        self.data = data
        self.data_type = data_type
        self.table_data = table_data
        self.table_data_type = table_data_type
        self.data_scale = data_scale
        self.data_bias = data_bias


@value
@register_passable("trivial")
struct _BNNSLayerParametersBroadcastMatMul:
    var alpha: Float32
    var beta: Float32
    var trans_a: Bool
    var trans_b: Bool
    var quadratic: Bool
    var a_is_weights: Bool
    var b_is_weights: Bool
    var a_desc: _BNNSNDArrayDescriptor
    var b_desc: _BNNSNDArrayDescriptor
    var o_desc: _BNNSNDArrayDescriptor

    fn __init__(
        inout self,
        alpha: Float32 = 1,
        beta: Float32 = 0,
        trans_a: Bool = False,
        trans_b: Bool = False,
        a_is_weights: Bool = False,
        b_is_weights: Bool = True,
        a_desc: _BNNSNDArrayDescriptor = _BNNSNDArrayDescriptor(),
        b_desc: _BNNSNDArrayDescriptor = _BNNSNDArrayDescriptor(),
        o_desc: _BNNSNDArrayDescriptor = _BNNSNDArrayDescriptor(),
    ):
        self.alpha = alpha
        self.beta = beta
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.quadratic = False
        self.a_is_weights = a_is_weights
        self.b_is_weights = b_is_weights
        self.a_desc = a_desc
        self.b_desc = b_desc
        self.o_desc = o_desc


@always_inline
fn _bnns_matmul[
    *,
    transpose_b: Bool = False,
](c: NDBuffer, a: NDBuffer, b: NDBuffer):
    constrained[a.rank == b.rank == c.rank == 2, "rank must be 2"]()

    var M = a.dim[0]()
    var N = b.dim[0]() if transpose_b else b.dim[1]()
    var K = a.dim[1]()

    var lda = K
    var ldb = N
    var ldc = N

    var a_desc = _BNNSNDArrayDescriptor(
        flags=_BNNSNDArrayFlags.BACK_PROP_SET,
        layout=_BNNSDataLayout.ROW_MAJOR_MATRIX,
        size=StaticIntTuple[_BNNS_MAX_TENSOR_DIMENSION](M, K, 0, 0, 0, 0, 0, 0),
        stride=StaticIntTuple[_BNNS_MAX_TENSOR_DIMENSION](
            1, lda, 0, 0, 0, 0, 0, 0
        ),
        data=a.data.address.bitcast[
            NoneType, address_space = AddressSpace.GENERIC
        ](),
        data_type=a.type,
    )

    var b_desc = _BNNSNDArrayDescriptor(
        flags=_BNNSNDArrayFlags.BACK_PROP_SET,
        layout=_BNNSDataLayout.ROW_MAJOR_MATRIX,
        size=StaticIntTuple[_BNNS_MAX_TENSOR_DIMENSION](K, N, 0, 0, 0, 0, 0, 0),
        stride=StaticIntTuple[_BNNS_MAX_TENSOR_DIMENSION](
            1, ldb, 0, 0, 0, 0, 0, 0
        ),
        data=b.data.address.bitcast[
            NoneType, address_space = AddressSpace.GENERIC
        ](),
        data_type=b.type,
    )

    var c_desc = _BNNSNDArrayDescriptor(
        flags=_BNNSNDArrayFlags.BACK_PROP_SET,
        layout=_BNNSDataLayout.ROW_MAJOR_MATRIX,
        size=StaticIntTuple[_BNNS_MAX_TENSOR_DIMENSION](M, N, 0, 0, 0, 0, 0, 0),
        stride=StaticIntTuple[_BNNS_MAX_TENSOR_DIMENSION](
            1, ldc, 0, 0, 0, 0, 0, 0
        ),
        data=c.data.address.bitcast[
            NoneType, address_space = AddressSpace.GENERIC
        ](),
        data_type=c.type,
    )

    var params = _BNNSLayerParametersBroadcastMatMul(
        alpha=1,
        beta=0,
        trans_b=transpose_b,
        b_is_weights=True,
        a_desc=a_desc,
        b_desc=b_desc,
        o_desc=c_desc,
    )


# ===----------------------------------------------------------------------===#
# GEMV (for M=1)
# ===----------------------------------------------------------------------===#


# Parallelized/vectorized version of GEMV for M = 1.
# Currently, use is limited in Apple Float32 case.
# apple_matmul (which internally calls cblas_sgemm, which in turns calls a
# cblas_sgemv has been found to have suboptimal performance compared to this.
@always_inline
fn apple_gemv[
    c_shape: DimList,
    c_type: DType,
    a_shape: DimList,
    a_type: DType,
    b_shape: DimList,
    b_type: DType,
    b_packed: Bool,
    transpose_b: Bool = False,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    c: NDBuffer[c_type, 2, c_shape],
    a: NDBuffer[a_type, 2, a_shape],
    b: NDBuffer[b_type, 2, b_shape],
):
    # Recall:
    # if b_packed=True, this will be called AFTER pack shape and actual packing
    # function (in MatmulPack.mojo), which will TRANSPOSE the input.
    var M = a.dim[0]()
    var K = a.dim[1]() if b_packed else b.dim[0]()
    var N = b.dim[0]() if transpose_b or b_packed else b.dim[1]()

    var transposed_b = NDBuffer[b_type, 2]()
    var transposed_b_ptr = DTypePointer[b_type]()
    # If both b_packed and transpose_b are False, we need to transpose B at
    # runtime (which is suboptimal, but enables faster gemv below).
    if b_packed == False and not transpose_b:
        var transposed_b_shape = Index(b.dim[1](), b.dim[0]())
        transposed_b_ptr = DTypePointer[b_type].alloc(b.num_elements())
        transposed_b = NDBuffer[b_type, 2](transposed_b_ptr, transposed_b_shape)

        pack_b_ndbuffer[
            a_type,
            a_shape,
            b_type,
            b_shape,
            c_type,
            c_shape,
        ](b, transposed_b)

    # If b_packed == False and B comes transposed (transpose_b == True) we need
    # to adjust K accordingly.
    # We will also need to use the original B instead of transposed_b in the
    # calculations further below.
    if b_packed == False and transpose_b == True:
        K = b.dim(1)

    alias simd_width = simdwidthof[c.type]()

    @always_inline
    @__copy_capture(c, a, b, K)
    @parameter
    fn process_rows(start_row: Int, end_row: Int):
        for n in range(start_row, end_row):
            var acc_vector = SIMD[c.type, simd_width]()
            var acc_scalar = Scalar[c.type]()

            @always_inline
            @parameter
            fn compute_fn[width: Int](k: Int):
                @parameter
                if width == 1:
                    acc_scalar += (
                        a[0, k].cast[c.type]()
                        * b[n, k].cast[c.type]() if b_packed
                        or (not b_packed and transpose_b) else transposed_b[
                            n, k
                        ].cast[c.type]()
                    )
                else:
                    acc_vector = fma(
                        a.load[width=simd_width](0, k).cast[c.type](),
                        b.load[width=simd_width](n, k).cast[
                            c.type
                        ]() if b_packed
                        or (
                            not b_packed and transpose_b
                        ) else transposed_b.load[width=simd_width](n, k).cast[
                            c.type
                        ](),
                        acc_vector,
                    )

            vectorize[compute_fn, simd_width](K)

            var val = acc_vector.reduce_add() + acc_scalar

            @parameter
            if elementwise_lambda_fn:
                alias func = elementwise_lambda_fn.value()
                func[c.type, 1](Index(0, n), val)
            else:
                c[Index(0, n)] = val

    # TODO: Experiment with this.
    alias parallelism_grain_size = 16
    parallelize_over_rows[process_rows](
        StaticIntTuple[2](N, K), 1, parallelism_grain_size
    )

    transposed_b_ptr.free()


# ===----------------------------------------------------------------------===#
# Matmul
# ===----------------------------------------------------------------------===#


@always_inline
fn apple_matmul[
    *,
    transpose_b: Bool = False,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](c: NDBuffer, a: NDBuffer, b: NDBuffer):
    @parameter
    if a.type == b.type == c.type == DType.float32:
        _cblas_f32[transpose_b=transpose_b](c, a, b)

        @parameter
        if elementwise_lambda_fn:
            var m = c.dim[0]()
            var n = c.dim[1]()
            alias epilogue = elementwise_lambda_fn.value()
            alias simd_size = simdwidthof[c.type]()

            @always_inline
            @parameter
            fn epilogue_on_col_chunk[
                simd_width: Int, rank: Int
            ](idx: StaticIntTuple[rank]):
                var c_coord = (idx[0], idx[1])
                var c_val = c.load[width=simd_width](c_coord)
                epilogue[c.type, simd_width](c_coord, c_val)

            elementwise[epilogue_on_col_chunk, simd_size, 2](
                StaticIntTuple[2](m, n)
            )
        return

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

        apple_matmul[transpose_b=transpose_b](c2, a2, b2)
