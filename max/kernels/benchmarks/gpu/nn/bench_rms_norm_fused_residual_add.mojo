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
from std.sys import get_defined_dtype

from std.benchmark import Bench, BenchConfig, Bencher, BenchId
from std.gpu.host import DeviceContext
from internal_utils import get_defined_shape, int_list_to_tuple
from layout import Coord, Idx, TileTensor, row_major
from nn.normalization import rms_norm_fused_residual_add_gpu

from std.utils.index import Index, IndexList


def bench_rms_norm_fused_residual_add_gpu[
    rank: Int, //, dtype: DType, shape: IndexList[rank]
](ctx: DeviceContext, mut b: Bench, fn_name: String) raises:
    comptime cols = shape[rank - 1]
    comptime rows = shape.flattened_length() // cols

    var data_h = alloc[Scalar[dtype]](rows * cols)
    var residual_h = alloc[Scalar[dtype]](rows * cols)
    var gamma1_h = alloc[Scalar[dtype]](cols)
    var gamma2_h = alloc[Scalar[dtype]](cols)

    for i in range(rows * cols):
        data_h[i] = Scalar[dtype](random_float64(0, 100).cast[dtype]())
        residual_h[i] = Scalar[dtype](random_float64(0, 100).cast[dtype]())

    for i in range(cols):
        gamma1_h[i] = (Float64(i + cols) / Float64(cols)).cast[dtype]()
        gamma2_h[i] = (Float64(i + cols) / Float64(cols)).cast[dtype]()

    var data_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var residual_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var output_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var residual_output_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var gamma1_d = ctx.enqueue_create_buffer[dtype](cols)
    var gamma2_d = ctx.enqueue_create_buffer[dtype](cols)

    var param_shape = Index(cols)

    var data_buf = TileTensor(data_d, row_major(Coord(shape)))
    var residual_buf = TileTensor(residual_d, row_major(Coord(shape)))
    var output_buf = TileTensor(output_d, row_major(Coord(shape)))
    var residual_output_buf = TileTensor(
        residual_output_d, row_major(Coord(shape))
    )
    var gamma1 = TileTensor(gamma1_d, row_major(Coord(param_shape)))
    var gamma2 = TileTensor(gamma2_d, row_major(Coord(param_shape)))
    var epsilon1 = Scalar[dtype](0.001)
    var epsilon2 = Scalar[dtype](0.001)
    var weight_offset1 = Scalar[dtype](0.0)
    var weight_offset2 = Scalar[dtype](0.0)

    ctx.enqueue_copy(data_d, data_h)
    ctx.enqueue_copy(residual_d, residual_h)
    ctx.enqueue_copy(gamma1_d, gamma1_h)
    ctx.enqueue_copy(gamma2_d, gamma2_h)

    @__copy_capture(data_buf)
    @always_inline
    @parameter
    def input_fn[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
        var idx = data_buf.layout(Coord(coords))
        return data_buf.ptr.load[width=width](idx)

    @__copy_capture(residual_buf)
    @always_inline
    @parameter
    def residual_input_fn[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
        var idx = residual_buf.layout(Coord(coords))
        return residual_buf.ptr.load[width=width](idx)

    @always_inline
    @__copy_capture(output_buf)
    @parameter
    def output_fn[
        width: Int, alignment: Int
    ](coords: IndexList[rank], val: SIMD[dtype, width]) -> None:
        var idx = output_buf.layout(Coord(coords))
        output_buf.ptr.store[width=width, alignment=alignment](idx, val)

    @always_inline
    @__copy_capture(residual_output_buf)
    @parameter
    def residual_output_fn[
        width: Int, alignment: Int
    ](coords: IndexList[rank], val: SIMD[dtype, width]) -> None:
        var idx = residual_output_buf.layout(Coord(coords))
        residual_output_buf.ptr.store[width=width, alignment=alignment](
            idx, val
        )

    @always_inline
    @__copy_capture(
        shape,
        gamma1,
        epsilon1,
        weight_offset1,
        gamma2,
        epsilon2,
        weight_offset2,
    )
    @parameter
    def bench_fn(mut b: Bencher) raises:
        @parameter
        @always_inline
        def kernel_launch(ctx: DeviceContext) raises:
            rms_norm_fused_residual_add_gpu[
                input_fn,
                residual_input_fn,
                residual_output_fn,
                output_fn,
                multiply_before_cast=True,
            ](
                shape,
                gamma1,
                epsilon1,
                weight_offset1,
                gamma2,
                epsilon2,
                weight_offset2,
                ctx,
            )

        b.iter_custom[kernel_launch](ctx)

    b.bench_function[bench_fn](
        BenchId(
            "rms_norm_fused_residual_add",
            input_id=String(fn_name, "/", dtype, "/", shape),
        ),
    )

    ctx.synchronize()

    _ = data_d
    _ = residual_d
    _ = output_d
    _ = residual_output_d
    _ = gamma1_d
    _ = gamma2_d

    data_h.free()
    residual_h.free()
    gamma1_h.free()
    gamma2_h.free()


def main() raises:
    comptime dtype = DType.bfloat16
    comptime label = "rms_norm_fused_residual_add_gpu"

    var m = Bench(BenchConfig(num_repetitions=1))
    with DeviceContext() as ctx:
        # Warp tiling path (cols <= WARP_SIZE * simd_width * max_warps_per_block)
        bench_rms_norm_fused_residual_add_gpu[dtype, Index(1, 512, 4096)](
            ctx, m, label
        )
        bench_rms_norm_fused_residual_add_gpu[dtype, Index(1, 1024, 4096)](
            ctx, m, label
        )
        bench_rms_norm_fused_residual_add_gpu[dtype, Index(1, 512, 8192)](
            ctx, m, label
        )
        bench_rms_norm_fused_residual_add_gpu[dtype, Index(1, 1024, 8192)](
            ctx, m, label
        )
        # Block path (cols > WARP_SIZE * simd_width * max_warps_per_block)
        bench_rms_norm_fused_residual_add_gpu[dtype, Index(1, 512, 16384)](
            ctx, m, label
        )
        bench_rms_norm_fused_residual_add_gpu[dtype, Index(1, 1024, 16384)](
            ctx, m, label
        )
        # Float32 uses block path at lower dims (simd_width=4)
        bench_rms_norm_fused_residual_add_gpu[
            DType.float32, Index(1, 512, 8192)
        ](ctx, m, label)

    m.dump_report()
