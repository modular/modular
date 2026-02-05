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

from math import ceildiv, rsqrt
from sys import simd_width_of

from gpu import WARP_SIZE
from gpu.host import DeviceContext, get_gpu_target
from layout._coord import Coord, Idx, coord_to_index_list
from layout._layout import row_major
from layout._tile_tensor import TileTensor
from layout.math import mean, variance
from nn.normalization import *
from testing import assert_almost_equal

from utils.index import Index, IndexList


fn run_layer_norm_block[
    dtype: DType,
    *,
    simd_width: Int = simd_width_of[dtype, target = get_gpu_target()](),
](ctx: DeviceContext, rows: Int, cols: Int, rtol: Float64 = 0.01) raises:
    print("== run_layer_norm_gpu block kernel")

    var data_h = alloc[Scalar[dtype]](rows * cols)
    var res = alloc[Scalar[dtype]](rows * cols)
    var gamma_h = alloc[Scalar[dtype]](cols)
    var beta_h = alloc[Scalar[dtype]](cols)

    for i in range(rows * cols):
        var val = Scalar[dtype](i)
        data_h[i] = val

    for i in range(cols):
        gamma_h[i] = (Float64(i + cols) / Float64(cols)).cast[dtype]()
        beta_h[i] = (Float64(i) / Float64(cols)).cast[dtype]()

    var data_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var gamma_d = ctx.enqueue_create_buffer[dtype](cols)
    var beta_d = ctx.enqueue_create_buffer[dtype](cols)

    var data_shape = Index(rows, cols)
    var param_shape = Index(cols)

    var data_buf = TileTensor(data_d, row_major(Coord(data_shape)))
    var gamma = TileTensor(gamma_d, row_major(Coord(param_shape)))
    var beta = TileTensor(beta_d, row_major(Coord(param_shape)))
    var epsilon = Scalar[dtype]()

    ctx.enqueue_copy(data_d, data_h)
    ctx.enqueue_copy(gamma_d, gamma_h)
    ctx.enqueue_copy(beta_d, beta_h)

    @__copy_capture(data_buf)
    @always_inline
    @parameter
    fn input_fn[width: Int](row: Int, col: Int) -> SIMD[dtype, width]:
        var idx = data_buf.layout(Coord(Idx(row), Idx(col)))
        return data_buf.ptr.load[width=width](idx)

    @__copy_capture(gamma)
    @always_inline
    @parameter
    fn gamma_fn[
        width: Int, rank: Int
    ](coords: IndexList[rank]) -> SIMD[dtype, width]:
        var idx = gamma.layout(Idx(coords[0]))
        return gamma.ptr.load[width=width](idx)

    @__copy_capture(data_buf)
    @always_inline
    @parameter
    fn output_fn[
        width: Int, alignment: Int
    ](row: Int, col: Int, val: SIMD[dtype, width]):
        var idx = data_buf.layout(Coord(Idx(row), Idx(col)))
        data_buf.ptr.store[width=width, alignment=alignment](
            idx, rebind[SIMD[dtype, width]](val)
        )

    var max_warps_per_block = (
        ctx.default_device_info.max_thread_block_size // WARP_SIZE
    )

    @always_inline
    @parameter
    @__copy_capture(data_buf, gamma, beta, epsilon)
    fn run_func_ln() raises:
        comptime kernel = layer_norm_gpu_block[
            origin = beta.origin,
            LayoutType = beta.LayoutType,
            UInt(simd_width),
            input_fn,
            gamma_fn,
            output_fn,
        ]
        ctx.enqueue_function[kernel, kernel](
            IndexList[2](rows, cols),
            beta,
            epsilon,
            grid_dim=(rows, 1),
            block_dim=min(
                ceildiv(ceildiv(cols, simd_width), WARP_SIZE) * WARP_SIZE,
                WARP_SIZE * max_warps_per_block,
            ),
        )

    run_func_ln()
    ctx.enqueue_copy(res, data_d)
    ctx.synchronize()

    for r in range(rows):
        var vec = TileTensor(
            data_h + r * cols,
            row_major(Idx(cols)),
        )
        var mean_ref = mean(vec)
        var var_ref = variance(vec, correction=0)
        var norm_factor_ref = rsqrt(var_ref + epsilon)
        for c in range(cols):
            var idx = r * cols + c
            var val = ((data_h[idx] - mean_ref) * norm_factor_ref) * gamma_h[
                c
            ] + beta_h[c]
            assert_almost_equal(val, res[idx], rtol=rtol)

    _ = data_d
    _ = gamma_d
    _ = beta_d

    data_h.free()
    res.free()
    gamma_h.free()
    beta_h.free()


fn run_layer_norm_gpu[
    dtype: DType, rank: Int
](ctx: DeviceContext, shape: IndexList[rank], rtol: Float64 = 0.01) raises:
    print("== run_layer_norm_gpu")

    var cols = shape[rank - 1]
    var rows = shape.flattened_length() // cols

    var data_h = alloc[Scalar[dtype]](rows * cols)
    var res = alloc[Scalar[dtype]](rows * cols)
    var gamma_h = alloc[Scalar[dtype]](cols)
    var beta_h = alloc[Scalar[dtype]](cols)

    for i in range(rows * cols):
        var val = Scalar[dtype](i)
        data_h[i] = val

    for i in range(cols):
        gamma_h[i] = (Float64(i + cols) / Float64(cols)).cast[dtype]()
        beta_h[i] = (Float64(i) / Float64(cols)).cast[dtype]()

    var data_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var gamma_d = ctx.enqueue_create_buffer[dtype](cols)
    var beta_d = ctx.enqueue_create_buffer[dtype](cols)

    var param_shape = Index(cols)

    var data_buf = TileTensor(data_d, row_major(Coord(shape)))
    var gamma = TileTensor(gamma_d, row_major(Coord(param_shape)))
    var beta = TileTensor(beta_d, row_major(Coord(param_shape)))
    var epsilon = Scalar[dtype]()

    ctx.enqueue_copy(data_d, data_h)
    ctx.enqueue_copy(gamma_d, gamma_h)
    ctx.enqueue_copy(beta_d, beta_h)

    @__copy_capture(data_buf)
    @always_inline
    @parameter
    fn input_fn[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
        var idx = data_buf.layout(Coord(coords))

        return data_buf.ptr.load[width=width](idx)

    @__copy_capture(gamma)
    @always_inline
    @parameter
    fn gamma_fn[
        width: Int, rank: Int
    ](coords: IndexList[rank]) -> SIMD[dtype, width]:
        var idx = gamma.layout(Idx(coords[0]))
        return gamma.ptr.load[width=width](idx[0])

    @__copy_capture(data_buf)
    @always_inline
    @parameter
    fn output_fn[
        width: Int, rank_: Int, alignment: Int
    ](coords: IndexList[rank_], val: SIMD[dtype, width]):
        var idx = data_buf.layout(Coord(coords))
        data_buf.ptr.store[width=width, alignment=alignment](
            idx, rebind[SIMD[dtype, width]](val)
        )

    layer_norm_gpu[input_fn, gamma_fn, output_fn](shape, beta, epsilon, ctx=ctx)
    ctx.enqueue_copy(res, data_d)
    ctx.synchronize()

    for r in range(rows):
        var vec = TileTensor(
            data_h + r * cols,
            row_major(Idx(cols)),
        )
        var mean_ref = mean(vec)
        var var_ref = variance(vec, correction=0)
        var norm_factor_ref = rsqrt(var_ref + epsilon)
        for c in range(cols):
            var idx = r * cols + c
            var val = ((data_h[idx] - mean_ref) * norm_factor_ref) * gamma_h[
                c
            ] + beta_h[c]
            assert_almost_equal(val, res[idx], rtol=rtol)

    _ = data_d
    _ = gamma_d
    _ = beta_d

    data_h.free()
    res.free()
    gamma_h.free()
    beta_h.free()


fn run_layer_norm_warp_tiling[
    dtype: DType,
    *,
    simd_width: Int = simd_width_of[dtype, target = get_gpu_target()](),
](ctx: DeviceContext, rows: Int, cols: Int, rtol: Float64 = 0.01) raises:
    print("== run_layer_norm_gpu warp tiling kernel")

    var data_h = alloc[Scalar[dtype]](rows * cols)
    var res = alloc[Scalar[dtype]](rows * cols)
    var gamma_h = alloc[Scalar[dtype]](cols)
    var beta_h = alloc[Scalar[dtype]](cols)

    for i in range(rows * cols):
        var val = Scalar[dtype](i)
        data_h[i] = val

    for i in range(cols):
        gamma_h[i] = (Float64(i + cols) / Float64(cols)).cast[dtype]()
        beta_h[i] = (Float64(i) / Float64(cols)).cast[dtype]()

    var data_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var gamma_d = ctx.enqueue_create_buffer[dtype](cols)
    var beta_d = ctx.enqueue_create_buffer[dtype](cols)

    var data_shape = Index(rows, cols)
    var param_shape = Index(cols)

    var data_buf = TileTensor(data_d, row_major(Coord(data_shape)))
    var gamma = TileTensor(gamma_d, row_major(Coord(param_shape)))
    var beta = TileTensor(beta_d, row_major(Coord(param_shape)))
    var epsilon = Scalar[dtype]()

    ctx.enqueue_copy(data_d, data_h)
    ctx.enqueue_copy(gamma_d, gamma_h)
    ctx.enqueue_copy(beta_d, beta_h)

    @__copy_capture(data_buf)
    @always_inline
    @parameter
    fn input_fn[width: Int](row: Int, col: Int) -> SIMD[dtype, width]:
        var idx = data_buf.layout(Coord(Idx(row), Idx(col)))

        return data_buf.ptr.load[width=width](idx)

    @__copy_capture(gamma)
    @always_inline
    @parameter
    fn gamma_fn[
        width: Int, rank: Int
    ](coords: IndexList[rank]) -> SIMD[dtype, width]:
        var idx = gamma.layout(Idx(coords[0]))
        return gamma.ptr.load[width=width](idx)

    @__copy_capture(data_buf)
    @always_inline
    @parameter
    fn output_fn[
        width: Int, alignment: Int
    ](row: Int, col: Int, val: SIMD[dtype, width]):
        var idx = data_buf.layout(Coord(Idx(row), Idx(col)))
        data_buf.ptr.store[width=width, alignment=alignment](
            idx, rebind[SIMD[dtype, width]](val)
        )

    var max_warps_per_block = (
        ctx.default_device_info.max_thread_block_size // WARP_SIZE
    )

    @always_inline
    @parameter
    @__copy_capture(data_buf, gamma, beta, epsilon)
    fn run_func_ln() raises:
        comptime kernel = layer_norm_gpu_warp_tiling[
            origin = beta.origin,
            LayoutType = beta.LayoutType,
            UInt(simd_width),
            input_fn,
            gamma_fn,
            output_fn,
        ]
        ctx.enqueue_function[kernel, kernel](
            IndexList[2](rows, cols),
            beta,
            epsilon,
            grid_dim=(rows, 1),
            block_dim=min(
                ceildiv(ceildiv(cols, simd_width), WARP_SIZE) * WARP_SIZE,
                WARP_SIZE * max_warps_per_block,
            ),
        )

    run_func_ln()
    ctx.enqueue_copy(res, data_d)
    ctx.synchronize()

    for r in range(rows):
        var vec = TileTensor(
            data_h + r * cols,
            row_major(Idx(cols)),
        )
        var mean_ref = mean(vec)
        var var_ref = variance(vec, correction=0)
        var norm_factor_ref = rsqrt(var_ref + epsilon)
        for c in range(cols):
            var idx = r * cols + c
            var val = ((data_h[idx] - mean_ref) * norm_factor_ref) * gamma_h[
                c
            ] + beta_h[c]
            assert_almost_equal(val, res[idx], rtol=rtol)

    _ = data_d
    _ = gamma_d
    _ = beta_d

    data_h.free()
    res.free()
    gamma_h.free()
    beta_h.free()


def main():
    with DeviceContext() as ctx:
        run_layer_norm_block[DType.float32, simd_width=1](ctx, rows=3, cols=5)
        run_layer_norm_block[DType.float32](ctx, rows=3, cols=8)
        run_layer_norm_block[DType.float32, simd_width=1](ctx, rows=7, cols=33)
        run_layer_norm_block[DType.float32](ctx, rows=1, cols=1024)
        run_layer_norm_block[DType.float32](ctx, rows=1, cols=8192, rtol=0.1)

        run_layer_norm_warp_tiling[DType.float32, simd_width=1](
            ctx, rows=3, cols=5
        )
        run_layer_norm_warp_tiling[DType.float32](ctx, rows=3, cols=8)
        run_layer_norm_warp_tiling[DType.float32, simd_width=1](
            ctx, rows=7, cols=33
        )
        run_layer_norm_warp_tiling[DType.float32](ctx, rows=1, cols=1024)
        run_layer_norm_warp_tiling[DType.float32](ctx, rows=10, cols=4096)

        # variable rank
        run_layer_norm_gpu[DType.float32](ctx, Index(5))
        run_layer_norm_gpu[DType.float32](ctx, Index(3, 4, 10, 20, 8))
        run_layer_norm_gpu[DType.float32](ctx, Index(1, 5, 6, 10, 128))
