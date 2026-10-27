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
"""Tests the `_matmul_gpu` epilogue and compute closures.

The shapes land on the GEMV, small-M, tile GEMM and vendor BLAS paths of the
SM100 dispatch, and on the 4-wave, split-K and multistage paths of the AMD
CDNA dispatch, with and without `transpose_b`. Each epilogue scales by a power
of two, so its result must match the plain `_matmul_gpu` exactly. The store
epilogue owns every output store, so its `c` is a shape-only tile over a null
pointer: no path may read or write it.

The tuned SM100 (N, K) = (2112, 14336) has split-K rows that the epilogues skip,
so its plain dispatch sums K in another order and cannot match them bit for
bit. Its reference runs through an identity compute epilogue instead, which
takes the same kernel as the epilogues.
"""

from std.random import rand

from max.gpu.host import DeviceContext
from layout import Coord, Idx, TileTensor, row_major
from linalg.matmul.gpu import _matmul_gpu
from std.testing import assert_equal


def test_epilogues[
    dtype: DType,
    N: Int,
    K: Int,
    *,
    transpose_b: Bool = True,
    split_k_tuned: Bool = False,
](ctx: DeviceContext, m: Int) raises:
    """Checks each epilogue against the plain matmul at every output index.
    With `split_k_tuned`, the reference is the identity compute epilogue."""
    print(dtype, m, "x", N, "x", K, "transpose_b:", transpose_b)
    comptime b_rows = N if transpose_b else K
    comptime b_cols = K if transpose_b else N
    var a_dev = ctx.enqueue_create_buffer[dtype](m * K)
    var b_dev = ctx.enqueue_create_buffer[dtype](N * K)
    var c_ref_dev = ctx.enqueue_create_buffer[dtype](m * N)
    var c_dev = ctx.enqueue_create_buffer[dtype](m * N)
    var d_dev = ctx.enqueue_create_buffer[dtype](m * N)

    with a_dev.map_to_host() as ha, b_dev.map_to_host() as hb:
        rand(ha.unsafe_ptr(), m * K, min=-1.0, max=1.0)
        rand(hb.unsafe_ptr(), N * K, min=-1.0, max=1.0)
    ctx.enqueue_memset(c_ref_dev, 0)
    ctx.enqueue_memset(d_dev, 0)

    var a = TileTensor(a_dev, row_major(m, Idx[K])).as_imm()
    var b = TileTensor(b_dev, row_major(Idx[b_rows], Idx[b_cols])).as_imm()
    var c_ref = TileTensor(c_ref_dev, row_major(m, Idx[N]))
    var c = TileTensor(c_dev, row_major(m, Idx[N]))
    var null_c_ptr: Optional[
        MutPointer[Scalar[dtype], MutUntrackedOrigin]
    ] = None
    var c_shape_only = TileTensor(
        null_c_ptr._unsafe_nullable(), row_major(m, Idx[N])
    )
    # A raw-pointer tile: the epilogue closure copies its capture.
    var d = TileTensor(d_dev.unsafe_ptr(), row_major(m, Idx[N]))

    def identity_fn[
        _dtype: DType, width: SIMDLength, *, alignment: Int
    ](idx: Coord, val: SIMD[_dtype, width]) -> SIMD[_dtype, width]:
        return val

    comptime if split_k_tuned:
        _matmul_gpu[use_tensor_core=True, transpose_b=transpose_b](
            c_ref, a, b, identity_fn, ctx
        )
    else:
        _matmul_gpu[use_tensor_core=True, transpose_b=transpose_b](
            c_ref, a, b, ctx
        )

    @inline(.always)
    def epilogue_fn[
        _dtype: DType, width: SIMDLength, *, alignment: Int
    ](idx: Coord, val: SIMD[_dtype, width]) {var d}:
        comptime assert idx.rank == 2, "the matmul epilogue contract is rank 2"
        var row_scale = -2 if Int(idx[0]) % 2 == 1 else 2
        d.store(idx, (val * SIMD[_dtype, width](row_scale)).cast[dtype]())

    def compute_fn[
        _dtype: DType, width: SIMDLength, *, alignment: Int
    ](idx: Coord, val: SIMD[_dtype, width]) -> SIMD[_dtype, width]:
        comptime assert idx.rank == 2, "the matmul epilogue contract is rank 2"
        var row_scale = -2 if Int(idx[0]) % 2 == 1 else 2
        return val * SIMD[_dtype, width](row_scale)

    _matmul_gpu[use_tensor_core=True, transpose_b=transpose_b](
        c_shape_only, a, b, epilogue_fn, ctx
    )
    with c_ref_dev.map_to_host() as h_ref, d_dev.map_to_host() as h_d:
        for i in range(m * N):
            var row_scale = -2 if (i // N) % 2 == 1 else 2
            assert_equal(h_d[i], h_ref[i] * Scalar[dtype](row_scale))

    ctx.enqueue_memset(c_dev, 0)
    _matmul_gpu[use_tensor_core=True, transpose_b=transpose_b](
        c, a, b, compute_fn, ctx
    )
    with c_ref_dev.map_to_host() as h_ref, c_dev.map_to_host() as h_c:
        for i in range(m * N):
            var row_scale = -2 if (i // N) % 2 == 1 else 2
            assert_equal(h_c[i], h_ref[i] * Scalar[dtype](row_scale))


def main() raises:
    with DeviceContext() as ctx:
        test_epilogues[.bfloat16, 4096, 4096](ctx, 1)
        test_epilogues[.bfloat16, 384, 7168](ctx, 28)
        test_epilogues[.bfloat16, 4096, 4096](ctx, 64)
        test_epilogues[.bfloat16, 4096, 4096](ctx, 256)
        test_epilogues[.bfloat16, 4096, 4096](ctx, 512)
        test_epilogues[.bfloat16, 4096, 4096](ctx, 2048)
        test_epilogues[.bfloat16, 2112, 14336, split_k_tuned=True](ctx, 64)
        test_epilogues[.bfloat16, 1024, 1024, transpose_b=False](ctx, 256)
        test_epilogues[.float32, 1024, 1024](ctx, 100)
