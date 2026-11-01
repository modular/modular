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

from std.random import random_float64

from max.benchmark import bencher_iter_custom
from std.benchmark import Bench, BenchConfig, Bencher, BenchId
from max.gpu.host import DeviceContext
from layout import Coord, TileTensor, row_major
from nn.normalization import group_norm

from std.utils.index import Index, IndexList


# The shape is a runtime value: `group_norm` takes a runtime `Coord` anyway, so
# one binary can sweep every shape instead of recompiling per shape.
def bench_group_norm_gpu[
    rank: Int, //, dtype: DType
](
    ctx: DeviceContext, mut b: Bench, shape: IndexList[rank], num_groups: Int
) raises:
    var C = shape[1]
    var total = shape.flattened_length()

    var data_h = List(length=total, fill=Scalar[dtype](0))
    var gamma_h = List(length=C, fill=Scalar[dtype](0))
    var beta_h = List(length=C, fill=Scalar[dtype](0))

    for i in range(total):
        data_h[i] = random_float64(0, 100).cast[dtype]()
    for i in range(C):
        gamma_h[i] = (Float64(i + C) / Float64(C)).cast[dtype]()
        beta_h[i] = (Float64(i) / Float64(C)).cast[dtype]()

    var data_d = ctx.enqueue_create_buffer[dtype](total)
    # Distinct output buffer: the input closure (reads) and `output` (writes)
    # must reference distinct buffer origins.
    var out_d = ctx.enqueue_create_buffer[dtype](total)
    var gamma_d = ctx.enqueue_create_buffer[dtype](C)
    var beta_d = ctx.enqueue_create_buffer[dtype](C)

    var data_buf = TileTensor(data_d, row_major(Coord(shape)))
    var out_buf = TileTensor(out_d, row_major(Coord(shape)))
    var gamma = TileTensor(gamma_d, row_major(Coord(Index(C))))
    var beta = TileTensor(beta_d, row_major(Coord(Index(C))))
    var epsilon = Float32(1e-5)

    ctx.enqueue_copy(data_d, data_h)
    ctx.enqueue_copy(gamma_d, gamma_h)
    ctx.enqueue_copy(beta_d, beta_h)

    @inline(.always)
    def input_fn[
        width: Int
    ](coords: Coord) {var data_buf} -> SIMD[dtype, width]:
        return data_buf.load[width=width](coords)

    @inline(.always)
    def gamma_fn[width: Int](coords: Coord) {var gamma} -> SIMD[dtype, width]:
        return gamma.load[width=width](coords)

    @inline(.always)
    def beta_fn[width: Int](coords: Coord) {var beta} -> SIMD[dtype, width]:
        return beta.load[width=width](coords)

    @inline(.always)
    def kernel_launch(ctx: DeviceContext) raises {imm}:
        group_norm[dtype, rank, target="gpu"](
            input_fn,
            gamma_fn,
            beta_fn,
            Coord(shape),
            epsilon,
            Int32(num_groups),
            out_buf,
            ctx=ctx,
        )

    @inline(.always)
    def bench_fn(mut b: Bencher) raises {imm}:
        bencher_iter_custom(b, kernel_launch, ctx)

    b.bench_function(
        bench_fn,
        BenchId(
            "group_norm",
            input_id=String(dtype, shape, num_groups, sep="/"),
        ),
    )

    ctx.synchronize()

    _ = data_d
    _ = out_d
    _ = gamma_d
    _ = beta_d
    _ = data_h^
    _ = gamma_h^
    _ = beta_h^


def bench_shapes[dtype: DType](ctx: DeviceContext, mut m: Bench) raises:
    # Warp-tiling dispatch (small groups).
    bench_group_norm_gpu[dtype](ctx, m, Index(4, 256, 16, 16), 32)
    bench_group_norm_gpu[dtype](ctx, m, Index(2, 128, 32, 32), 32)
    bench_group_norm_gpu[dtype](ctx, m, Index(16, 512, 16, 16), 32)
    bench_group_norm_gpu[dtype](ctx, m, Index(32, 128, 32, 32), 32)
    # H*W (or L) not a multiple of the SIMD width: SIMD windows straddle
    # channel boundaries.
    bench_group_norm_gpu[dtype](ctx, m, Index(2, 256, 15, 15), 32)
    bench_group_norm_gpu[dtype](ctx, m, Index(2, 256, 225), 32)

    # Multi-block dispatch, NCHW diffusion / VAE-like shapes, 32 groups.
    bench_group_norm_gpu[dtype](ctx, m, Index(1, 128, 512, 512), 32)
    bench_group_norm_gpu[dtype](ctx, m, Index(1, 256, 256, 256), 32)
    bench_group_norm_gpu[dtype](ctx, m, Index(1, 512, 64, 64), 32)
    bench_group_norm_gpu[dtype](ctx, m, Index(2, 320, 64, 64), 32)
    bench_group_norm_gpu[dtype](ctx, m, Index(2, 640, 32, 32), 32)
    bench_group_norm_gpu[dtype](ctx, m, Index(1, 512, 45, 45), 32)
    # NCL
    bench_group_norm_gpu[dtype](ctx, m, Index(1, 512, 4096), 32)
    bench_group_norm_gpu[dtype](ctx, m, Index(2, 320, 4096), 32)
    bench_group_norm_gpu[dtype](ctx, m, Index(2, 1280, 1024), 32)

    # TODO(adakkak): `group_norm_gpu_block` with SIMD loads (a group too wide
    # for warp tiling but too small to split, or 256+ rows) fails to launch
    # with CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES for both float32 and bfloat16.
    # Add those cells once it is fixed. (2, 1280, 16, 16) lands there for
    # bfloat16, so it only runs for float32.
    comptime if dtype == DType.float32:
        bench_group_norm_gpu[dtype](ctx, m, Index(2, 1280, 16, 16), 32)


def main() raises:
    var m = Bench(BenchConfig(num_repetitions=1))
    with DeviceContext() as ctx:
        bench_shapes[DType.bfloat16](ctx, m)
        bench_shapes[DType.float32](ctx, m)
    m.dump_report()
