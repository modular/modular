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
"""Regression test for `matmul[transpose_b=True]` with N == 1 and M > 1.

`gemv_gpu_dispatch`'s scalar `GemvKernel` branch used the transposed launch
(A and B swapped, one warp per column of C) for every `transpose_b` product.
That launch is the M == 1 case; with N == 1 and M > 1 it ran a single warp and
wrote only c[0].
"""

from max.gpu.host import DeviceContext
from max.gpu.primitives import PDLLevel
from layout import Coord, Idx, TileTensor, row_major
from linalg.gemv import GEMVAlgorithm, gemv_gpu_dispatch
from linalg.matmul.gpu import _matmul_gpu
from std.testing import assert_equal


def test[
    dtype: DType,
    M: Int,
    N: Int,
    K: Int,
    transpose_b: Bool,
    *,
    direct: Bool = False,
    algorithm: GEMVAlgorithm = GEMVAlgorithm.GemvKernel,
    epilogue: Int = 0,
](ctx: DeviceContext) raises:
    """Compare matmul or a forced GEMV dispatch against a host dot product.

    epilogue=0 stores directly, 1 stores through an epilogue closure into a
    separate buffer, 2 stores a compute closure's result into c.
    Small signed integers keep float32 accumulation exact; the reference is
    cast to the output dtype before comparison.
    """
    comptime assert epilogue == 0 or direct
    print(
        M, "x", N, "x", K, "transpose_b", transpose_b, dtype, direct, epilogue
    )

    # A square epilogue destination keeps swapped coordinates in bounds, where
    # checking every sentinel can detect them safely. Plain stores are packed.
    comptime extent = max(M, N) + 1
    comptime c_size = extent * extent
    var a_host = ctx.enqueue_create_host_buffer[dtype](M * K)
    var b_host = ctx.enqueue_create_host_buffer[dtype](K * N)
    var c_host = ctx.enqueue_create_host_buffer[dtype](c_size)
    var expected = ctx.enqueue_create_host_buffer[dtype](c_size)
    for i in range(M):
        for k in range(K):
            a_host[i * K + k] = Scalar[dtype](i + 1 + k % 7 - 3)
    for k in range(K):
        for j in range(N):
            b_host[j * K + k if transpose_b else k * N + j] = Scalar[dtype](
                j + 1 + k % 5 - 2
            )
    for i in range(c_size):
        c_host[i] = -8192
        expected[i] = -8192
    for i in range(M):
        for j in range(N):
            var dot = Float32(0)
            for k in range(K):
                dot += (
                    a_host[i * K + k].cast[.float32]()
                    * b_host[j * K + k if transpose_b else k * N + j].cast[
                        .float32
                    ]()
                )
            comptime if epilogue == 1:
                expected[i * extent + j] = dot.cast[dtype]() + Scalar[dtype](
                    2 * i + 3 * j
                )
            elif epilogue == 2:
                expected[i * N + j] = dot.cast[dtype]() + Scalar[dtype](
                    2 * i + 3 * j
                )
            else:
                expected[i * N + j] = dot.cast[dtype]()

    var a_dev = ctx.enqueue_create_buffer[dtype](M * K)
    var b_dev = ctx.enqueue_create_buffer[dtype](K * N)
    var c_dev = ctx.enqueue_create_buffer[dtype](c_size)
    var epilogue_dev = ctx.enqueue_create_buffer[dtype](c_size)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.enqueue_copy(c_dev, c_host)
    ctx.enqueue_copy(epilogue_dev, c_host)

    comptime b_dim0 = N if transpose_b else K
    comptime b_dim1 = K if transpose_b else N
    var a_tensor = TileTensor(a_dev, row_major(Coord(M, Idx[K])))
    var b_tensor = TileTensor(b_dev, row_major(Coord(Idx[b_dim0], Idx[b_dim1])))
    var c_tensor = TileTensor(c_dev, row_major(Coord(M, Idx[N])))

    comptime if direct:
        # Force the changed branch on every GPU, independent of matmul's
        # architecture-specific choices. Also exercise runtime M, N and K.
        var a_dynamic = a_tensor.reshape(row_major(M, K)).as_imm()
        var b_dynamic = b_tensor.reshape(row_major(b_dim0, b_dim1)).as_imm()
        var c_dynamic = c_tensor.reshape(row_major(M, N))
        var epilogue_tensor = TileTensor(
            epilogue_dev, row_major(Idx[extent], Idx[extent])
        )
        var c_out = epilogue_tensor.reshape(epilogue_tensor.layout)

        def epilogue_store[
            out_type: DType, width: SIMDLength, *, alignment: Int
        ](idx: Coord, val: SIMD[out_type, width]) {var c_out}:
            var offset = 2 * Int(idx[0].value()) + 3 * Int(idx[1].value())
            c_out.store(
                idx,
                rebind[SIMD[dtype, width]](val + SIMD[out_type, width](offset)),
            )

        def compute_offset[
            out_type: DType, width: SIMDLength, *, alignment: Int
        ](idx: Coord, val: SIMD[out_type, width]) -> SIMD[out_type, width]:
            var offset = 2 * Int(idx[0].value()) + 3 * Int(idx[1].value())
            return val + SIMD[out_type, width](offset)

        comptime if epilogue == 1:
            gemv_gpu_dispatch[transpose_b=transpose_b, pdl_level=PDLLevel.OFF](
                algorithm, c_dynamic, a_dynamic, b_dynamic, epilogue_store, ctx
            )
        elif epilogue == 2:
            gemv_gpu_dispatch[transpose_b=transpose_b, pdl_level=PDLLevel.OFF](
                algorithm, c_dynamic, a_dynamic, b_dynamic, compute_offset, ctx
            )
        else:
            gemv_gpu_dispatch[transpose_b=transpose_b, pdl_level=PDLLevel.OFF](
                algorithm, c_dynamic, a_dynamic, b_dynamic, ctx
            )
    else:
        _matmul_gpu[use_tensor_core=True, transpose_b=transpose_b](
            c_tensor, a_tensor.as_imm(), b_tensor.as_imm(), ctx
        )
    ctx.enqueue_copy(c_host, c_dev)
    ctx.synchronize()

    comptime if epilogue == 1:
        # An epilogue owns the store; the nominal output must be untouched.
        for i in range(c_size):
            assert_equal(c_host[i], Scalar[dtype](-8192))
        ctx.enqueue_copy(c_host, epilogue_dev)
        ctx.synchronize()

    for i in range(c_size):
        assert_equal(
            c_host[i].cast[.float32](),
            expected[i].cast[.float32](),
            String("output/guard offset ", i),
        )


def main() raises:
    with DeviceContext() as ctx:
        comptime K = 1024
        # Integration coverage: matmul's selected implementation must work.
        comptime for m in range(1, 5):
            test[.float32, m, 1, K, False](ctx)
            test[.float32, m, 1, K, True](ctx)
        test[.float32, 17, 1, K, False](ctx)
        test[.float32, 17, 1, K, True](ctx)
        test[.float32, 1, 3, K, True](ctx)
        test[.bfloat16, 3, 1, K, True](ctx)
        test[.bfloat16, 1, 3, K, True](ctx)

        # Multiple blocks and a partial final block on both 32- and 64-lane
        # GPUs; K includes sub-warp, exact-warp and partial-warp reductions.
        comptime reduction_sizes = (1, 31, 32, 33, 65)
        comptime for index in range(len(reduction_sizes)):
            comptime k = reduction_sizes[index]
            test[.float32, 65, 1, k, True, direct=True](ctx)
        test[.float16, 17, 1, 33, True, direct=True](ctx)
        test[.bfloat16, 17, 1, 33, True, direct=True](ctx)
        test[
            .bfloat16,
            3,
            1,
            K,
            True,
            direct=True,
            algorithm=GEMVAlgorithm.GemvKernelVector,
        ](ctx)

        # Both store APIs must receive (row, 0) for N == 1 and (0, col) for
        # the unchanged M == 1 transposed launch. Plain stores cover it too.
        comptime for epilogue in range(3):
            test[.float32, 65, 1, 33, True, direct=True, epilogue=epilogue](ctx)
            test[.float32, 65, 1, 33, False, direct=True, epilogue=epilogue](
                ctx
            )
            test[.float32, 1, 65, 33, True, direct=True, epilogue=epilogue](ctx)
            test[.float32, 1, 1, 1, True, direct=True, epilogue=epilogue](ctx)
