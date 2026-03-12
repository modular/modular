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

from std.random import rand
from std.sys import get_defined_dtype, get_defined_int

from std.benchmark import Bench, BenchConfig, Bencher, BenchId
from buffer import Dim, DimList, NDBuffer
from std.gpu.host import DeviceContext
from linalg.dual_gemm import multistage_dual_gemm
from linalg.utils_gpu import MatmulConfig, _bk_base

from std.utils.index import Index, IndexList


fn bench_dual_gemm_gpu[
    dtype: DType, M: Int, N: Int, K: Int
](ctx: DeviceContext, mut b: Bench) raises:
    comptime transpose_b = True
    comptime warp_shape = Index(64, 64, _bk_base[dtype]())
    comptime config = MatmulConfig[dtype, dtype, dtype, transpose_b](
        block_tile_shape=Index(128, 128, 32),
        warp_tile_shape=warp_shape,
        num_pipeline_stages=4,
        num_k_partitions=1,
    )

    var a_h = alloc[Scalar[dtype]](M * K)
    var b0_h = alloc[Scalar[dtype]](N * K)
    var b1_h = alloc[Scalar[dtype]](N * K)

    rand[dtype](a_h, M * K)
    rand[dtype](b0_h, N * K)
    rand[dtype](b1_h, N * K)

    var a_d = ctx.enqueue_create_buffer[dtype](M * K)
    var b0_d = ctx.enqueue_create_buffer[dtype](N * K)
    var b1_d = ctx.enqueue_create_buffer[dtype](N * K)
    var c_d = ctx.enqueue_create_buffer[dtype](M * N)

    ctx.enqueue_copy(a_d, a_h)
    ctx.enqueue_copy(b0_d, b0_h)
    ctx.enqueue_copy(b1_d, b1_h)

    var a_ptr = a_d.unsafe_ptr()
    var b0_ptr = b0_d.unsafe_ptr()
    var b1_ptr = b1_d.unsafe_ptr()
    var c_ptr = c_d.unsafe_ptr()

    @always_inline
    @__copy_capture(a_ptr, b0_ptr, b1_ptr, c_ptr)
    @parameter
    fn bench_fn(mut b: Bencher) raises:
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            var a_buf = NDBuffer[
                rank=2, dtype, MutAnyOrigin, DimList[Dim(), K]()
            ](a_ptr, Index(M, K))
            var b0_buf = NDBuffer[rank=2, dtype, MutAnyOrigin, DimList[N, K]()](
                b0_ptr, Index(N, K)
            )
            var b1_buf = NDBuffer[rank=2, dtype, MutAnyOrigin, DimList[N, K]()](
                b1_ptr, Index(N, K)
            )
            var c_buf = NDBuffer[
                rank=2, dtype, MutAnyOrigin, DimList[Dim(), N]()
            ](c_ptr, Index(M, N))
            multistage_dual_gemm[transpose_b=transpose_b, config=config](
                c_buf, a_buf, b0_buf, b1_buf, ctx
            )

        b.iter_custom[kernel_launch](ctx)

    b.bench_function[bench_fn](
        BenchId(
            "dual_gemm",
            input_id=String(dtype, "/", M, "x", N, "x", K),
        )
    )

    ctx.synchronize()

    _ = a_d
    _ = b0_d
    _ = b1_d
    _ = c_d

    a_h.free()
    b0_h.free()
    b1_h.free()


def main() raises:
    comptime dtype = get_defined_dtype["dtype", DType.bfloat16]()
    comptime M = get_defined_int["M", 128]()
    comptime N = get_defined_int["N", 14336]()
    comptime K = get_defined_int["K", 4096]()

    var m = Bench(BenchConfig(num_repetitions=1))
    with DeviceContext() as ctx:
        bench_dual_gemm_gpu[dtype, M, N, K](ctx, m)

    m.dump_report()
