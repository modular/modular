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

from collections import OptionalReg
from collections.string import StaticString
from math import fma
from os import abort
from sys import os_is_macos, simdwidthof
from sys.ffi import _get_dylib_function as _ffi_get_dylib_function
from sys.ffi import _Global, _OwnedDLHandle

from algorithm import elementwise, vectorize
from algorithm.functional import (
    _get_start_indices_of_nth_subvolume,
    parallelize_over_rows,
)
from buffer.buffer import NDBuffer
from buffer.dimlist import DimList
from memory import UnsafePointer

from utils import IndexList
from utils.index import Index

from .bmm import _reshape_nd_buffer_with_batch_to_3d
from .bmm import (
    elementwise_epilogue_type as batched_matmul_elementwise_epilogue_type,
)
from .packing import pack_b_ndbuffer
from .utils import elementwise_epilogue_type as matmul_elementwise_epilogue_type

alias cblas_gemm_type = fn (
    _CBLASOrder,
    _CBLASTranspose,
    _CBLASTranspose,
    Int32,
    Int32,
    Int32,
    Float32,
    UnsafePointer[Float32],
    Int32,
    UnsafePointer[Float32],
    Int32,
    Float32,
    UnsafePointer[Float32],
    Int32,
) -> None

# ===-----------------------------------------------------------------------===#
# Constants
# ===-----------------------------------------------------------------------===#

alias LIB_ACC_PATH = "/System/Library/Frameworks/Accelerate.framework/Accelerate"


# ===-----------------------------------------------------------------------===#
# Library Load
# ===-----------------------------------------------------------------------===#

alias APPLE_ACCELERATE = _Global[
    "APPLE_ACCELERATE", _OwnedDLHandle, _init_dylib
]


fn _init_dylib() -> _OwnedDLHandle:
    # Note: we can't use _find_dylib here because this is not a real path
    # (it's a framework path).
    try:
        return _OwnedDLHandle(LIB_ACC_PATH)
    except:
        return abort[_OwnedDLHandle](
            "the accelerate library was not found at " + LIB_ACC_PATH
        )


@always_inline
fn _get_dylib_function[
    func_name: StaticString, result_type: AnyTrivialRegType
]() -> result_type:
    constrained[os_is_macos(), "operating system must be macOS"]()
    return _ffi_get_dylib_function[
        APPLE_ACCELERATE(),
        func_name,
        result_type,
    ]()


@always_inline
fn get_cblas_f32_function() -> cblas_gemm_type:
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
    return _get_dylib_function["cblas_sgemm", cblas_gemm_type]()


# ===-----------------------------------------------------------------------===#
# CBLAS
# ===-----------------------------------------------------------------------===#


@always_inline
fn use_apple_accelerate_lib[
    c_type: DType,
    a_type: DType,
    b_type: DType,
]() -> Bool:
    return os_is_macos() and a_type == b_type == c_type is DType.float32


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


# _cblas_f32 used by apple_batched_matmul (via the corresponding apple_matmul)
@always_inline
fn _cblas_f32[
    *,
    transpose_b: Bool = False,
](
    cblas_gemm_fn: cblas_gemm_type,
    m: Int32,
    n: Int32,
    k: Int32,
    lda: Int32,
    ldb: Int32,
    ldc: Int32,
    alpha: Float32,
    beta: Float32,
    c_ptr: UnsafePointer[Float32, **_],
    a_ptr: UnsafePointer[Float32, **_],
    b_ptr: UnsafePointer[Float32, **_],
):
    cblas_gemm_fn(
        _CBLASOrder.ROW_MAJOR,
        _CBLASTranspose.NO_TRANSPOSE,
        _CBLASTranspose.TRANSPOSE if transpose_b else _CBLASTranspose.NO_TRANSPOSE,
        m,
        n,
        k,
        alpha,
        rebind[UnsafePointer[Float32]](a_ptr),
        lda,
        rebind[UnsafePointer[Float32]](b_ptr),
        ldb,
        beta,
        rebind[UnsafePointer[Float32]](c_ptr),
        ldc,
    )


# _cblas_f32 used by apple_matmul (except via the apple_matmul in
# apple_batched_matmul)
@always_inline
fn _cblas_f32[
    *,
    transpose_b: Bool = False,
](
    m: Int32,
    n: Int32,
    k: Int32,
    lda: Int32,
    ldb: Int32,
    ldc: Int32,
    alpha: Float32,
    beta: Float32,
    c_ptr: UnsafePointer[Float32, **_],
    a_ptr: UnsafePointer[Float32, **_],
    b_ptr: UnsafePointer[Float32, **_],
):
    var cblas_gemm = get_cblas_f32_function()

    _cblas_f32[transpose_b=transpose_b](
        cblas_gemm,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        alpha,
        beta,
        c_ptr,
        a_ptr,
        b_ptr,
    )


# ===-----------------------------------------------------------------------===#
# GEMV (for M=1)
# ===-----------------------------------------------------------------------===#


# Parallelized/vectorized version of GEMV for M = 1.
# Currently, use is limited in Apple Float32 case.
# apple_matmul (which internally calls cblas_sgemm, which in turns calls a
# cblas_sgemv has been found to have suboptimal performance compared to this.
@always_inline
fn apple_gemv[
    *,
    b_packed: Bool,
    transpose_b: Bool = False,
    elementwise_lambda_fn: OptionalReg[matmul_elementwise_epilogue_type] = None,
](
    c: NDBuffer[mut=True, _, 2, _, _],
    a: NDBuffer[_, 2, _, _],
    b: NDBuffer[_, 2, _, _],
) raises:
    # Recall:
    # if b_packed=True, this will be called AFTER pack shape and actual packing
    # function (in MatmulPack.mojo), which will TRANSPOSE the input.
    var K = a.dim[1]() if b_packed else b.dim[0]()
    var N = b.dim[0]() if transpose_b or b_packed else b.dim[1]()

    var transposed_b = NDBuffer[b.type, 2, MutableAnyOrigin]()
    var transposed_b_ptr = UnsafePointer[Scalar[b.type]]()

    # If both b_packed and transpose_b are False, we need to transpose B at
    # runtime (which is suboptimal, but enables faster gemv below).
    @parameter
    if b_packed == False and not transpose_b:
        var transposed_b_shape = Index(b.dim[1](), b.dim[0]())
        transposed_b_ptr = UnsafePointer[Scalar[b.type]].alloc(b.num_elements())
        transposed_b = NDBuffer[b.type, 2](transposed_b_ptr, transposed_b_shape)

        pack_b_ndbuffer[
            a.type,
            a.shape,
            b.type,
            b.shape,
            c.type,
            c.shape,
        ](b, transposed_b)

    # If b_packed == False and B comes transposed (transpose_b == True) we need
    # to adjust K accordingly.
    # We will also need to use the original B instead of transposed_b in the
    # calculations further below.
    @parameter
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
                var a_val = a.load[width=width](0, k).cast[c.type]()
                var b_val = b.load[width=width](n, k).cast[
                    c.type
                ]() if b_packed or (
                    not b_packed and transpose_b
                ) else transposed_b.load[
                    width=width
                ](
                    n, k
                ).cast[
                    c.type
                ]()

                @parameter
                if width == 1:
                    acc_scalar = fma(
                        rebind[Scalar[c.type]](a_val),
                        rebind[Scalar[c.type]](b_val),
                        acc_scalar,
                    )
                else:
                    acc_vector = fma(
                        rebind[SIMD[c.type, simd_width]](a_val),
                        rebind[SIMD[c.type, simd_width]](b_val),
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
        IndexList[2](N, K), 1, parallelism_grain_size
    )

    transposed_b_ptr.free()


# ===-----------------------------------------------------------------------===#
# Matmul
# ===-----------------------------------------------------------------------===#


# apple_matmul used by apple_batched_matmul
@always_inline
fn apple_matmul[
    *,
    transpose_b: Bool = False,
    elementwise_lambda_fn: OptionalReg[matmul_elementwise_epilogue_type] = None,
](cblas_gemm_fn: cblas_gemm_type, c: NDBuffer, a: NDBuffer, b: NDBuffer) raises:
    @parameter
    if a.type == b.type == c.type is DType.float32:
        var m = Int32(a.dim[0]())
        var n = Int32(b.dim[0]() if transpose_b else b.dim[1]())
        var k = Int32(a.dim[1]())

        var lda = k
        var ldb = n if not transpose_b else k
        var ldc = n

        alias alpha = 1.0
        alias beta = 0.0

        _cblas_f32[transpose_b=transpose_b](
            cblas_gemm_fn,
            m,
            n,
            k,
            lda,
            ldb,
            ldc,
            alpha,
            beta,
            rebind[UnsafePointer[Float32, address_space = c.address_space]](
                c.data
            ),
            rebind[UnsafePointer[Float32, address_space = a.address_space]](
                a.data
            ),
            rebind[UnsafePointer[Float32, address_space = b.address_space]](
                b.data
            ),
        )

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
            ](idx: IndexList[rank]):
                var c_coord = IndexList[2](idx[0], idx[1])
                var c_val = c.load[width=simd_width](c_coord)
                epilogue[c.type, simd_width](c_coord, c_val)

            elementwise[epilogue_on_col_chunk, simd_size](IndexList[2](m, n))
        return

    constrained[False, "unsupported type in apple accelerate"]()


# apple_matmul used by all matmuls except apple_batched_matmul
@always_inline
fn apple_matmul[
    *,
    transpose_b: Bool = False,
    elementwise_lambda_fn: OptionalReg[matmul_elementwise_epilogue_type] = None,
](c: NDBuffer, a: NDBuffer, b: NDBuffer) raises:
    @parameter
    if a.type == b.type == c.type is DType.float32:
        var cblas_gemm = get_cblas_f32_function()

        apple_matmul[
            transpose_b=transpose_b, elementwise_lambda_fn=elementwise_lambda_fn
        ](cblas_gemm, c, a, b)

        return

    constrained[False, "unsupported type in apple accelerate"]()


# ===-----------------------------------------------------------------------===#
# Batched Matmul
# ===-----------------------------------------------------------------------===#


@always_inline
fn apple_batched_matmul[
    *,
    transpose_b: Bool = False,
    elementwise_epilogue_fn: OptionalReg[
        batched_matmul_elementwise_epilogue_type
    ] = None,
](c: NDBuffer, a: NDBuffer, b: NDBuffer) raises:
    var c3 = _reshape_nd_buffer_with_batch_to_3d(c)
    var a3 = _reshape_nd_buffer_with_batch_to_3d(a)
    var b3 = _reshape_nd_buffer_with_batch_to_3d(b)
    var batch_size = c3.dim[0]()

    var c_shape = Index(c3.dim[1](), c3.dim[2]())
    var a_shape = Index(a3.dim[1](), a3.dim[2]())
    var b_shape = Index(b3.dim[1](), b3.dim[2]())

    var cblas_gemm = get_cblas_f32_function()

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

        alias rank = c.rank
        var batch_coords = _get_start_indices_of_nth_subvolume[2](
            batch, rebind[IndexList[rank]](c.get_shape())
        )

        @parameter
        @__copy_capture(batch_coords)
        fn elementwise_lambda_2d[
            c_type: DType, width: Int, *, alignment: Int = 1
        ](out_coords: IndexList[2], out_val: SIMD[c_type, width]):
            var local_batch_coords = batch_coords
            local_batch_coords[rank - 1] = out_coords[1]
            local_batch_coords[rank - 2] = out_coords[0]

            alias func = elementwise_epilogue_fn.value()
            func[c_type, width, rank](local_batch_coords, out_val)

        apple_matmul[
            transpose_b=transpose_b,
            elementwise_lambda_fn = OptionalReg[
                matmul_elementwise_epilogue_type
            ](elementwise_lambda_2d) if elementwise_epilogue_fn else None,
        ](cblas_gemm, c2, a2, b2)
