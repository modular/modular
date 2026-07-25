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

from std.math import ceildiv, rsqrt
from std.sys import simd_width_of

from std.gpu import WARP_SIZE
from std.gpu.host import DeviceContext, get_gpu_target
from layout import Coord, Idx, TileTensor, row_major
from layout.math import mean, variance
from nn.normalization import *
from std.testing import assert_almost_equal, assert_equal

from std.utils.index import Index, IndexList


def run_layer_norm_block[
    dtype: DType,
    *,
    simd_width: Int = simd_width_of[dtype, target=get_gpu_target()](),
](ctx: DeviceContext, rows: Int, cols: Int, rtol: Float64 = 0.01) raises:
    print("== run_layer_norm_gpu block kernel")

    var data_h = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var res = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var gamma_h = ctx.enqueue_create_host_buffer[dtype](cols)
    var beta_h = ctx.enqueue_create_host_buffer[dtype](cols)

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
    var epsilon = Float32(0)

    ctx.enqueue_copy(data_d, data_h)
    ctx.enqueue_copy(gamma_d, gamma_h)
    ctx.enqueue_copy(beta_d, beta_h)

    @__copy_capture(data_buf)
    @always_inline
    @parameter
    def input_fn[
        width: Int, alignment: Int
    ](row: Int, col: Int) -> SIMD[dtype, width]:
        var idx = data_buf.layout(Coord(row, col))
        return data_buf.raw_load[width=width, alignment=alignment](idx)

    @__copy_capture(gamma)
    @always_inline
    @parameter
    def gamma_fn[
        width: Int, rank: Int, alignment: Int
    ](coords: IndexList[rank]) -> SIMD[dtype, width]:
        var idx = gamma.layout(coords[0])
        return gamma.raw_load[width=width, alignment=alignment](idx)

    @__copy_capture(data_buf)
    @always_inline
    @parameter
    def output_fn[
        width: SIMDLength, alignment: Int
    ](row: Int, col: Int, val: SIMD[dtype, width]):
        var idx = data_buf.layout(Coord(row, col))
        data_buf.raw_store[width=width, alignment=alignment](
            idx, rebind[SIMD[dtype, width]](val)
        )

    var max_warps_per_block = (
        ctx.default_device_info.max_thread_block_size // WARP_SIZE
    )

    @always_inline
    @parameter
    @__copy_capture(data_buf, gamma, beta, epsilon)
    def run_func_ln() raises:
        comptime kernel = layer_norm_gpu_block[
            LayoutType=beta.LayoutType,
            origin=beta.origin,
            Storage=beta.Storage,
            simd_width,
            input_fn,
            gamma_fn,
            output_fn,
        ]
        ctx.enqueue_function[kernel](
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
            data_h.unsafe_ptr() + r * cols,
            row_major(cols),
        )
        var mean_ref = mean(vec)
        var var_ref = variance(vec, correction=0)
        var norm_factor_ref = rsqrt(var_ref + epsilon.cast[dtype]())
        for c in range(cols):
            var idx = r * cols + c
            var val = ((data_h[idx] - mean_ref) * norm_factor_ref) * gamma_h[
                c
            ] + beta_h[c]
            assert_almost_equal(val, res[idx], rtol=rtol)

    _ = data_d
    _ = gamma_d
    _ = beta_d


def run_layer_norm_gpu[
    dtype: DType, rank: Int
](ctx: DeviceContext, shape: IndexList[rank], rtol: Float64 = 0.01) raises:
    print("== run_layer_norm_gpu")

    var cols = shape[rank - 1]
    var rows = shape.flattened_length() // cols

    var data_h = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var res = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var gamma_h = ctx.enqueue_create_host_buffer[dtype](cols)
    var beta_h = ctx.enqueue_create_host_buffer[dtype](cols)

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
    var epsilon = Float32(0)

    ctx.enqueue_copy(data_d, data_h)
    ctx.enqueue_copy(gamma_d, gamma_h)
    ctx.enqueue_copy(beta_d, beta_h)

    @__copy_capture(data_buf)
    @always_inline
    @parameter
    def input_fn[
        width: Int, alignment: Int
    ](coords: Coord) -> SIMD[dtype, width]:
        var idx = data_buf.layout(coords)

        return data_buf.raw_load[width=width, alignment=alignment](idx)

    @__copy_capture(gamma)
    @always_inline
    @parameter
    def gamma_fn[
        width: Int, rank: Int, alignment: Int
    ](coords: IndexList[rank]) -> SIMD[dtype, width]:
        var idx = gamma.layout(coords[0])
        return gamma.raw_load[width=width, alignment=alignment](idx[0])

    @__copy_capture(data_buf)
    @always_inline
    @parameter
    def output_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[dtype, width]):
        var idx = data_buf.layout(coords)
        data_buf.raw_store[width=width, alignment=alignment](
            idx, rebind[SIMD[dtype, width]](val)
        )

    # `layer_norm_gpu` migrated to a `Coord` shape boundary (mirror of
    # `rms_norm_gpu` / softmax migration); `rank` is now an explicit parameter
    # (no longer inferred from `shape`).
    layer_norm_gpu[rank, input_fn, gamma_fn, output_fn](
        Coord(shape), beta, epsilon, ctx=ctx
    )
    ctx.enqueue_copy(res, data_d)
    ctx.synchronize()

    for r in range(rows):
        var vec = TileTensor(
            data_h.unsafe_ptr() + r * cols,
            row_major(cols),
        )
        var mean_ref = mean(vec)
        var var_ref = variance(vec, correction=0)
        var norm_factor_ref = rsqrt(var_ref + epsilon.cast[dtype]())
        for c in range(cols):
            var idx = r * cols + c
            var val = ((data_h[idx] - mean_ref) * norm_factor_ref) * gamma_h[
                c
            ] + beta_h[c]
            assert_almost_equal(val, res[idx], rtol=rtol)

    _ = data_d
    _ = gamma_d
    _ = beta_d


def run_layer_norm_static_vs_dynamic[
    dtype: DType, *dims: Int
](ctx: DeviceContext) raises:
    """Static-divisor fold numerical-equivalence check for `layer_norm_gpu`.

    Runs `layer_norm_gpu` twice on identical data: once with a fully static
    shape `Coord` (`row_major[*dims]().shape_coord()`, all `ComptimeInt`, so
    the flattened-row -> n-D `divmod` strength-reduces to magic-multiply +
    shift) and once with the dynamic `Coord(IndexList)` shape (runtime `IDIV`).
    The two device outputs must be bit-identical: the fold only changes which
    instructions the compiler emits, not the arithmetic result. Use rank>=3 so
    dims `1..rank-2` actually decompose (rank-2 hits the `{n, 0}` fast path and
    never divides, so the fold is a no-op there).
    """
    print("== run_layer_norm_gpu static vs dynamic")

    comptime static_layout = row_major[*dims]()
    comptime rank = type_of(static_layout).rank
    comptime assert rank >= 3, "fold only differs for rank>=3 shapes"

    var shape = IndexList[rank]()

    comptime for i in range(rank):
        shape[i] = dims[i]

    var cols = shape[rank - 1]
    var rows = shape.flattened_length() // cols

    var data_h = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var gamma_h = ctx.enqueue_create_host_buffer[dtype](cols)
    var beta_h = ctx.enqueue_create_host_buffer[dtype](cols)

    for i in range(rows * cols):
        data_h[i] = Scalar[dtype](i)
    for i in range(cols):
        gamma_h[i] = (Float64(i + cols) / Float64(cols)).cast[dtype]()
        beta_h[i] = (Float64(i) / Float64(cols)).cast[dtype]()

    var gamma_d = ctx.enqueue_create_buffer[dtype](cols)
    var beta_d = ctx.enqueue_create_buffer[dtype](cols)
    var param_shape = Index(cols)
    var gamma = TileTensor(gamma_d, row_major(Coord(param_shape)))
    var beta = TileTensor(beta_d, row_major(Coord(param_shape)))
    var epsilon = Float32(0)
    ctx.enqueue_copy(gamma_d, gamma_h)
    ctx.enqueue_copy(beta_d, beta_h)

    @__copy_capture(gamma)
    @always_inline
    @parameter
    def gamma_fn[
        width: Int, rank: Int, alignment: Int
    ](coords: IndexList[rank]) -> SIMD[dtype, width]:
        var idx = gamma.layout(coords[0])
        return gamma.raw_load[width=width, alignment=alignment](idx[0])

    # Static run: the `Coord` carries `ComptimeInt` dims, so the fold fires.
    var out_static_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var out_static_h = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    ctx.enqueue_copy(out_static_d, data_h)
    var data_static = TileTensor(out_static_d, static_layout)

    @__copy_capture(data_static)
    @always_inline
    @parameter
    def input_fn_static[
        width: Int, alignment: Int
    ](coords: Coord) -> SIMD[dtype, width]:
        var idx = data_static.layout(coords)
        return data_static.raw_load[width=width, alignment=alignment](idx)

    @__copy_capture(data_static)
    @always_inline
    @parameter
    def output_fn_static[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[dtype, width]):
        var idx = data_static.layout(coords)
        data_static.raw_store[width=width, alignment=alignment](
            idx, rebind[SIMD[dtype, width]](val)
        )

    layer_norm_gpu[rank, input_fn_static, gamma_fn, output_fn_static](
        static_layout.shape_coord(), beta, epsilon, ctx=ctx
    )
    ctx.enqueue_copy(out_static_h, out_static_d)

    # Dynamic run: the `Coord` carries runtime leaves, forcing the `IDIV`.
    var out_dyn_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var out_dyn_h = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    ctx.enqueue_copy(out_dyn_d, data_h)
    var data_dyn = TileTensor(out_dyn_d, row_major(Coord(shape)))

    @__copy_capture(data_dyn)
    @always_inline
    @parameter
    def input_fn_dyn[
        width: Int, alignment: Int
    ](coords: Coord) -> SIMD[dtype, width]:
        var idx = data_dyn.layout(coords)
        return data_dyn.raw_load[width=width, alignment=alignment](idx)

    @__copy_capture(data_dyn)
    @always_inline
    @parameter
    def output_fn_dyn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[dtype, width]):
        var idx = data_dyn.layout(coords)
        data_dyn.raw_store[width=width, alignment=alignment](
            idx, rebind[SIMD[dtype, width]](val)
        )

    layer_norm_gpu[rank, input_fn_dyn, gamma_fn, output_fn_dyn](
        Coord(shape), beta, epsilon, ctx=ctx
    )
    ctx.enqueue_copy(out_dyn_h, out_dyn_d)
    ctx.synchronize()

    # The fold is a codegen change only; results must be bit-identical.
    for i in range(rows * cols):
        assert_equal(out_static_h[i], out_dyn_h[i])

    _ = gamma_d
    _ = beta_d
    _ = out_static_d
    _ = out_dyn_d


def run_layer_norm_warp_tiling[
    dtype: DType,
    *,
    simd_width: Int = simd_width_of[dtype, target=get_gpu_target()](),
](ctx: DeviceContext, rows: Int, cols: Int, rtol: Float64 = 0.01) raises:
    print("== run_layer_norm_gpu warp tiling kernel")

    var data_h = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var res = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var gamma_h = ctx.enqueue_create_host_buffer[dtype](cols)
    var beta_h = ctx.enqueue_create_host_buffer[dtype](cols)

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
    var epsilon = Float32(0)

    ctx.enqueue_copy(data_d, data_h)
    ctx.enqueue_copy(gamma_d, gamma_h)
    ctx.enqueue_copy(beta_d, beta_h)

    @__copy_capture(data_buf)
    @always_inline
    @parameter
    def input_fn[
        width: Int, alignment: Int
    ](row: Int, col: Int) -> SIMD[dtype, width]:
        var idx = data_buf.layout(Coord(row, col))

        return data_buf.raw_load[width=width, alignment=alignment](idx)

    @__copy_capture(gamma)
    @always_inline
    @parameter
    def gamma_fn[
        width: Int, rank: Int, alignment: Int
    ](coords: IndexList[rank]) -> SIMD[dtype, width]:
        var idx = gamma.layout(coords[0])
        return gamma.raw_load[width=width, alignment=alignment](idx)

    @__copy_capture(data_buf)
    @always_inline
    @parameter
    def output_fn[
        width: SIMDLength, alignment: Int
    ](row: Int, col: Int, val: SIMD[dtype, width]):
        var idx = data_buf.layout(Coord(row, col))
        data_buf.raw_store[width=width, alignment=alignment](
            idx, rebind[SIMD[dtype, width]](val)
        )

    comptime max_warps_per_block = ctx.default_device_info.max_thread_block_size // WARP_SIZE

    @always_inline
    @parameter
    @__copy_capture(data_buf, gamma, beta, epsilon)
    def run_func_ln() raises:
        comptime kernel = layer_norm_gpu_warp_tiling[
            LayoutType=beta.LayoutType,
            origin=beta.origin,
            Storage=beta.Storage,
            simd_width,
            max_warps_per_block,
            input_fn,
            gamma_fn,
            output_fn,
        ]
        ctx.enqueue_function[kernel](
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
            data_h.unsafe_ptr() + r * cols,
            row_major(cols),
        )
        var mean_ref = mean(vec)
        var var_ref = variance(vec, correction=0)
        var norm_factor_ref = rsqrt(var_ref + epsilon.cast[dtype]())
        for c in range(cols):
            var idx = r * cols + c
            var val = ((data_h[idx] - mean_ref) * norm_factor_ref) * gamma_h[
                c
            ] + beta_h[c]
            assert_almost_equal(val, res[idx], rtol=rtol)

    _ = data_d
    _ = gamma_d
    _ = beta_d


def main() raises:
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

        # Rank-3 `[B, S, H]` cases exercise the rank>=3 outer-dim decomposition
        # that the `IndexList`->`Coord` static-divisor folding path targets
        # (mirror of the rank-3 cases added to `test_rms_norm`). Kept at fp32:
        # the reference fills `data[i] = i` and `layer_norm` subtracts the row
        # mean, so a bf16 variant (integers above 256 are no longer exactly
        # representable) drives the post-mean values into catastrophic
        # cancellation that swamps the kernel comparison rather than testing it.
        run_layer_norm_gpu[DType.float32](ctx, Index(8, 3072, 256))
        run_layer_norm_gpu[DType.float32](ctx, Index(4, 128, 2048))

        # The static-shape path is the whole point of this migration, but the
        # reference checks above run only the dynamic `Coord(IndexList)` form.
        # These assert the static (folded-divisor) and dynamic (runtime-divide)
        # paths produce bit-identical output at rank>=3, so the fold can't
        # silently diverge (mirror of concat's static-vs-dynamic check).
        run_layer_norm_static_vs_dynamic[DType.float32, 8, 3072, 256](ctx)
        run_layer_norm_static_vs_dynamic[DType.float32, 4, 128, 2048](ctx)
