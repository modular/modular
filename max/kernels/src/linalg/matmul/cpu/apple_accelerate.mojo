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

from std.collections import Optional
from std.math import fma
from std.memory import alloc
from std.sys import CompilationTarget, simd_width_of
from std.ffi import _get_dylib_function as _ffi_get_dylib_function
from std.ffi import _Global, OwnedDLHandle

from std.algorithm import elementwise, vectorize
from std.algorithm.functional import (
    _get_start_indices_of_nth_subvolume,
    parallelize_over_rows,
)
from std.utils import IndexList
from std.utils.index import Index

from ...bmm import (
    elementwise_epilogue_type as batched_matmul_elementwise_epilogue_type,
)
from std.gpu.memory import AddressSpace
from layout import Coord, Idx, TileTensor, row_major
from ...packing import pack_b_ndbuffer
from ...utils import (
    elementwise_epilogue_type as matmul_elementwise_epilogue_type,
)

comptime cblas_gemm_type = def(
    _CBLASOrder,
    _CBLASTranspose,
    _CBLASTranspose,
    Int32,
    Int32,
    Int32,
    Float32,
    UnsafePointer[Float32, ImmutAnyOrigin],
    Int32,
    UnsafePointer[Float32, ImmutAnyOrigin],
    Int32,
    Float32,
    UnsafePointer[Float32, MutAnyOrigin],
    Int32,
) -> None

# ===-----------------------------------------------------------------------===#
# Constants
# ===-----------------------------------------------------------------------===#

comptime LIB_ACC_PATH = (
    "/System/Library/Frameworks/Accelerate.framework/Accelerate"
)


# ===-----------------------------------------------------------------------===#
# Library Load
# ===-----------------------------------------------------------------------===#


def _on_error_msg() -> Error:
    return Error(
        (
            "Cannot find the Apple Accelerate libraries. Please make sure that "
            "the XCode package is installed and that the library path is "
            "correctly set in one of the following paths ["
        ),
        ", ".join(Span([LIB_ACC_PATH])),
        "].",
    )


comptime APPLE_ACCELERATE = _Global[
    "APPLE_ACCELERATE", _init_dylib, on_error_msg=_on_error_msg
]


def _init_dylib() -> OwnedDLHandle:
    # Note: we can't use _find_dylib here because this is not a real path
    # (it's a framework path).
    try:
        return OwnedDLHandle(LIB_ACC_PATH)
    except:
        return OwnedDLHandle(unsafe_uninitialized=True)


@always_inline
def _get_dylib_function[
    func_name: StaticString, result_type: TrivialRegisterPassable
]() raises -> result_type:
    comptime assert (
        CompilationTarget.is_macos()
    ), "operating system must be macOS"
    return _ffi_get_dylib_function[
        APPLE_ACCELERATE(),
        func_name,
        result_type,
    ]()


@always_inline
def get_cblas_f32_function() raises -> cblas_gemm_type:
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
def use_apple_accelerate_lib[
    c_type: DType,
    a_type: DType,
    b_type: DType,
]() -> Bool:
    return (
        CompilationTarget.is_macos()
        and a_type == b_type == c_type == DType.float32
    )


@fieldwise_init
struct _CBLASOrder(TrivialRegisterPassable):
    var value: Int32
    comptime ROW_MAJOR = _CBLASOrder(101)
    comptime COL_MAJOR = _CBLASOrder(102)


@fieldwise_init
struct _CBLASTranspose(TrivialRegisterPassable):
    var value: Int32
    comptime NO_TRANSPOSE = _CBLASTranspose(111)
    comptime TRANSPOSE = _CBLASTranspose(112)
    comptime CONJ_TRANSPOSE = _CBLASTranspose(113)


# _cblas_f32 used by apple_batched_matmul (via the corresponding apple_matmul)
@always_inline
def _cblas_f32[
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
    c_ptr: UnsafePointer[mut=True, Float32, ...],
    a_ptr: UnsafePointer[Float32, ...],
    b_ptr: UnsafePointer[Float32, ...],
):
    cblas_gemm_fn(
        _CBLASOrder.ROW_MAJOR,
        _CBLASTranspose.NO_TRANSPOSE,
        _CBLASTranspose.TRANSPOSE if transpose_b else _CBLASTranspose.NO_TRANSPOSE,
        m,
        n,
        k,
        alpha,
        rebind[UnsafePointer[Float32, ImmutAnyOrigin]](a_ptr),
        lda,
        rebind[UnsafePointer[Float32, ImmutAnyOrigin]](b_ptr),
        ldb,
        beta,
        rebind[UnsafePointer[Float32, MutAnyOrigin]](c_ptr),
        ldc,
    )


# _cblas_f32 used by apple_matmul (except via the apple_matmul in
# apple_batched_matmul)
@always_inline
def _cblas_f32[
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
    c_ptr: UnsafePointer[mut=True, Float32, ...],
    a_ptr: UnsafePointer[Float32, ...],
    b_ptr: UnsafePointer[Float32, ...],
) raises:
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
def apple_gemv[
    *,
    b_packed: Bool,
    transpose_b: Bool = False,
    elementwise_lambda_fn: Optional[matmul_elementwise_epilogue_type] = None,
](
    c: TileTensor[mut=True, address_space=AddressSpace.GENERIC, ...],
    a: TileTensor[mut=False, address_space=AddressSpace.GENERIC, ...],
    b: TileTensor[mut=False, address_space=AddressSpace.GENERIC, ...],
) raises:
    comptime assert c.flat_rank >= 2
    comptime assert a.flat_rank >= 2
    comptime assert b.flat_rank >= 2

    # Recall:
    # if b_packed=True, this will be called AFTER pack shape and actual packing
    # function (in MatmulPack.mojo), which will TRANSPOSE the input.
    var K = Int(a.dim[1]()) if b_packed else Int(b.dim[0]())
    var N = Int(b.dim[0]()) if transpose_b or b_packed else Int(b.dim[1]())

    var transposed_b_ptr = UnsafePointer[Scalar[b.dtype], MutExternalOrigin]()
    var transposed_b = TileTensor(
        UnsafePointer[Scalar[b.dtype], MutExternalOrigin](),
        row_major(Coord(Idx(Int(0)), Idx(Int(0)))),
    )

    # If both b_packed and transpose_b are False, we need to transpose B at
    # runtime (which is suboptimal, but enables faster gemv below).
    comptime if b_packed == False and not transpose_b:
        var transposed_b_shape = Index(Int(b.dim[1]()), Int(b.dim[0]()))
        transposed_b_ptr = alloc[Scalar[b.dtype]](b.num_elements())
        transposed_b = TileTensor(
            transposed_b_ptr,
            row_major(
                Coord(Idx(transposed_b_shape[0]), Idx(transposed_b_shape[1]))
            ),
        )

        var _kernel_type_m = 0
        comptime if a.static_shape[0] > -1:
            _kernel_type_m = a.static_shape[0]
        pack_b_ndbuffer[a_type=a.dtype, c_type=c.dtype](
            b, transposed_b, _kernel_type_m
        )

    # If b_packed == False and B comes transposed (transpose_b == True) we need
    # to adjust K accordingly.
    # We will also need to use the original B instead of transposed_b in the
    # calculations further below.
    comptime if b_packed == False and transpose_b == True:
        K = Int(b.dim[1]())

    comptime simd_width = simd_width_of[c.dtype]()

    @always_inline
    @__copy_capture(c, a, b, K)
    @parameter
    def process_rows(start_row: Int, end_row: Int):
        for var n in range(start_row, end_row):
            var acc_vector = SIMD[c.dtype, simd_width]()
            var acc_scalar = Scalar[c.dtype]()

            @always_inline
            def compute_fn[width: Int](k: Int) unified {mut}:
                var a_val = a.load[width=width](Coord(Idx(0), Idx(k))).cast[
                    c.dtype
                ]()
                var b_val = (
                    b.load[width=width](Coord(Idx(n), Idx(k))).cast[
                        c.dtype
                    ]() if b_packed
                    or (not b_packed and transpose_b) else transposed_b.load[
                        width=width
                    ](Coord(Idx(n), Idx(k))).cast[c.dtype]()
                )

                comptime if width == 1:
                    acc_scalar = fma(
                        rebind[Scalar[c.dtype]](a_val),
                        rebind[Scalar[c.dtype]](b_val),
                        acc_scalar,
                    )
                else:
                    acc_vector = fma(
                        rebind[SIMD[c.dtype, simd_width]](a_val),
                        rebind[SIMD[c.dtype, simd_width]](b_val),
                        acc_vector,
                    )

            vectorize[simd_width](K, compute_fn)

            var val = acc_vector.reduce_add() + acc_scalar

            comptime if elementwise_lambda_fn:
                comptime func = elementwise_lambda_fn.value()
                func[c.dtype, 1](Index(0, n), val)
            else:
                c.store[width=1](Coord(Idx(0), Idx(n)), val)

    # TODO: Experiment with this.
    comptime parallelism_grain_size = 16
    parallelize_over_rows[process_rows](
        IndexList[2](N, K), 1, parallelism_grain_size
    )

    transposed_b_ptr.free()


# ===-----------------------------------------------------------------------===#
# Matmul
# ===-----------------------------------------------------------------------===#


# apple_matmul used by apple_batched_matmul
@always_inline
def apple_matmul[
    *,
    transpose_b: Bool = False,
    elementwise_lambda_fn: Optional[matmul_elementwise_epilogue_type] = None,
](
    cblas_gemm_fn: cblas_gemm_type,
    c: TileTensor[mut=True, ...],
    a: TileTensor[...],
    b: TileTensor[...],
) raises:
    comptime assert c.flat_rank >= 2
    comptime assert a.flat_rank >= 2
    comptime assert b.flat_rank >= 2
    comptime assert (
        a.dtype == b.dtype == c.dtype == DType.float32
    ), "unsupported type in apple accelerate"
    var m = Int32(Int(a.dim[0]()))
    var n = Int32(Int(b.dim[0]()) if transpose_b else Int(b.dim[1]()))
    var k = Int32(Int(a.dim[1]()))

    var lda = k
    var ldb = n if not transpose_b else k
    var ldc = n

    comptime alpha = 1.0
    comptime beta = 0.0

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
        rebind[UnsafePointer[Float32, MutAnyOrigin]](c.ptr),
        rebind[UnsafePointer[Float32, ImmutAnyOrigin]](a.ptr),
        rebind[UnsafePointer[Float32, ImmutAnyOrigin]](b.ptr),
    )

    comptime if elementwise_lambda_fn:
        var m = Int(c.dim[0]())
        var n = Int(c.dim[1]())
        comptime epilogue = elementwise_lambda_fn.value()
        comptime simd_size = simd_width_of[c.dtype]()

        @always_inline
        @parameter
        def epilogue_on_col_chunk[
            simd_width: Int, rank: Int, alignment: Int = 1
        ](idx: IndexList[rank]):
            var c_coord = IndexList[2](idx[0], idx[1])
            var c_val = c.load[width=simd_width](
                Coord(Idx(idx[0]), Idx(idx[1]))
            )
            epilogue[c.dtype, simd_width](c_coord, c_val)

        elementwise[epilogue_on_col_chunk, simd_size](IndexList[2](m, n))


# apple_matmul used by all matmuls except apple_batched_matmul
@always_inline
def apple_matmul[
    *,
    transpose_b: Bool = False,
    elementwise_lambda_fn: Optional[matmul_elementwise_epilogue_type] = None,
](c: TileTensor[mut=True, ...], a: TileTensor[...], b: TileTensor[...]) raises:
    comptime assert (
        a.dtype == b.dtype == c.dtype == DType.float32
    ), "unsupported type in apple accelerate"
    var cblas_gemm = get_cblas_f32_function()

    apple_matmul[
        transpose_b=transpose_b, elementwise_lambda_fn=elementwise_lambda_fn
    ](cblas_gemm, c, a, b)


# ===-----------------------------------------------------------------------===#
# Batched Matmul
# ===-----------------------------------------------------------------------===#


@always_inline
def apple_batched_matmul[
    rank: Int,
    *,
    transpose_b: Bool = False,
    elementwise_epilogue_fn: Optional[
        batched_matmul_elementwise_epilogue_type
    ] = None,
](
    c: TileTensor[mut=True, ...],
    a: TileTensor[...],
    b: TileTensor[...],
    c_shape_idx: IndexList[rank],
) raises:
    comptime assert rank >= 3, "expecting at least rank-3 TileTensor"

    # Compute batch dimensions by collapsing all but the last two dims.
    var batch_size = 1
    comptime for i in range(rank - 2):
        batch_size *= Int(c_shape_idx[i])

    var c_rows = Int(c_shape_idx[rank - 2])
    var c_cols = Int(c_shape_idx[rank - 1])

    # Extract the last two dims from a and b using their own rank.
    var a_rows = Int(a.dim[a.rank - 2]())
    var a_cols = Int(a.dim[a.rank - 1]())
    var b_rows = Int(b.dim[b.rank - 2]())
    var b_cols = Int(b.dim[b.rank - 1]())

    var c_stride = c_rows * c_cols
    var a_stride = a_rows * a_cols
    var b_stride = b_rows * b_cols

    var cblas_gemm = get_cblas_f32_function()

    for batch in range(batch_size):
        var c2 = TileTensor(
            c.ptr + batch * c_stride,
            row_major(Coord(Idx(c_rows), Idx(c_cols))),
        )
        var a2 = TileTensor(
            a.ptr + batch * a_stride,
            row_major(Coord(Idx(a_rows), Idx(a_cols))),
        )
        var b2 = TileTensor(
            b.ptr + batch * b_stride,
            row_major(Coord(Idx(b_rows), Idx(b_cols))),
        )

        var batch_coords = _get_start_indices_of_nth_subvolume[2](
            batch, c_shape_idx
        )

        @parameter
        @__copy_capture(batch_coords)
        def elementwise_lambda_2d[
            c_type: DType, width: Int, *, alignment: Int = 1
        ](out_coords: IndexList[2], out_val: SIMD[c_type, width]):
            var local_batch_coords = batch_coords
            local_batch_coords[rank - 1] = out_coords[1]
            local_batch_coords[rank - 2] = out_coords[0]

            comptime func = elementwise_epilogue_fn.value()
            func[c_type, width, rank](local_batch_coords, out_val)

        apple_matmul[
            transpose_b=transpose_b,
            elementwise_lambda_fn=Optional[matmul_elementwise_epilogue_type](
                elementwise_lambda_2d
            ) if elementwise_epilogue_fn else None,
        ](cblas_gemm, c2, a2, b2)
