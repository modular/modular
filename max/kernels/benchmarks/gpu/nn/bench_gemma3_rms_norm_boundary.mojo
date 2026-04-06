from std.random import random_float64
from std.sys import get_defined_dtype, simd_width_of

from std.algorithm.functional import elementwise
from std.benchmark import Bench, BenchConfig, Bencher, BenchId
from std.gpu.host import DeviceContext
from std.testing import assert_almost_equal
from internal_utils import get_defined_shape, int_list_to_tuple
from layout import Coord, Idx, TileTensor, row_major
from nn.normalization import rms_norm_fused_residual_add_gpu, rms_norm_gpu

from std.utils.index import Index, IndexList


def bench_gemma3_rms_norm_boundary[
    rank: Int, //, dtype: DType, shape: IndexList[rank]
](ctx: DeviceContext, mut b: Bench) raises:
    comptime cols = shape[rank - 1]
    comptime rows = shape.flattened_length() // cols
    comptime add_simd_width = simd_width_of[dtype]()

    var data_h = alloc[Scalar[dtype]](rows * cols)
    var residual_h = alloc[Scalar[dtype]](rows * cols)
    var gamma1_h = alloc[Scalar[dtype]](cols)
    var gamma2_h = alloc[Scalar[dtype]](cols)
    var baseline_sum_h = alloc[Scalar[dtype]](rows * cols)
    var baseline_output_h = alloc[Scalar[dtype]](rows * cols)
    var fused_sum_h = alloc[Scalar[dtype]](rows * cols)
    var fused_output_h = alloc[Scalar[dtype]](rows * cols)

    for i in range(rows * cols):
        data_h[i] = Scalar[dtype](random_float64(0, 100).cast[dtype]())
        residual_h[i] = Scalar[dtype](random_float64(0, 100).cast[dtype]())

    for i in range(cols):
        gamma1_h[i] = (Float64(i + cols) / Float64(cols)).cast[dtype]()
        gamma2_h[i] = (Float64(i + cols + 1) / Float64(cols)).cast[dtype]()

    var data_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var residual_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var baseline_norm_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var baseline_sum_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var baseline_output_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var fused_sum_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var fused_output_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var gamma1_d = ctx.enqueue_create_buffer[dtype](cols)
    var gamma2_d = ctx.enqueue_create_buffer[dtype](cols)

    var param_shape = Index(cols)

    var data_buf = TileTensor(data_d, row_major(Coord(shape)))
    var residual_buf = TileTensor(residual_d, row_major(Coord(shape)))
    var baseline_norm_buf = TileTensor(
        baseline_norm_d, row_major(Coord(shape))
    )
    var baseline_sum_buf = TileTensor(baseline_sum_d, row_major(Coord(shape)))
    var baseline_output_buf = TileTensor(
        baseline_output_d, row_major(Coord(shape))
    )
    var fused_sum_buf = TileTensor(fused_sum_d, row_major(Coord(shape)))
    var fused_output_buf = TileTensor(fused_output_d, row_major(Coord(shape)))
    var gamma1 = TileTensor(gamma1_d, row_major(Coord(param_shape)))
    var gamma2 = TileTensor(gamma2_d, row_major(Coord(param_shape)))
    var epsilon1 = Scalar[dtype](0.001)
    var epsilon2 = Scalar[dtype](0.001)
    # Gemma-family RMSNorm uses (1 + gamma).
    var weight_offset1 = Scalar[dtype](1.0)
    var weight_offset2 = Scalar[dtype](1.0)

    ctx.enqueue_copy(data_d, data_h)
    ctx.enqueue_copy(residual_d, residual_h)
    ctx.enqueue_copy(gamma1_d, gamma1_h)
    ctx.enqueue_copy(gamma2_d, gamma2_h)

    @always_inline
    @__copy_capture(data_buf)
    @parameter
    def input_fn[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
        return data_buf.load[width=width](Coord(coords))

    @always_inline
    @__copy_capture(residual_buf)
    @parameter
    def residual_input_fn[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
        return residual_buf.load[width=width](Coord(coords))

    @always_inline
    @__copy_capture(baseline_norm_buf)
    @parameter
    def baseline_norm_output_fn[
        width: Int, alignment: Int
    ](coords: IndexList[rank], val: SIMD[dtype, width]) -> None:
        baseline_norm_buf.store[alignment=alignment](Coord(coords), val)

    @always_inline
    @__copy_capture(baseline_sum_buf)
    @parameter
    def baseline_sum_output_fn[
        width: Int, alignment: Int
    ](coords: IndexList[rank], val: SIMD[dtype, width]) -> None:
        baseline_sum_buf.store[alignment=alignment](Coord(coords), val)

    @always_inline
    @__copy_capture(baseline_sum_buf)
    @parameter
    def baseline_sum_input_fn[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
        return baseline_sum_buf.load[width=width](Coord(coords))

    @always_inline
    @__copy_capture(baseline_output_buf)
    @parameter
    def baseline_output_fn[
        width: Int, alignment: Int
    ](coords: IndexList[rank], val: SIMD[dtype, width]) -> None:
        baseline_output_buf.store[alignment=alignment](Coord(coords), val)

    @always_inline
    @__copy_capture(fused_sum_buf)
    @parameter
    def fused_sum_output_fn[
        width: Int, alignment: Int
    ](coords: IndexList[rank], val: SIMD[dtype, width]) -> None:
        fused_sum_buf.store[alignment=alignment](Coord(coords), val)

    @always_inline
    @__copy_capture(fused_output_buf)
    @parameter
    def fused_output_fn[
        width: Int, alignment: Int
    ](coords: IndexList[rank], val: SIMD[dtype, width]) -> None:
        fused_output_buf.store[alignment=alignment](Coord(coords), val)

    @always_inline
    @__copy_capture(baseline_norm_buf, residual_buf, baseline_sum_buf)
    @parameter
    def add_kernel[
        width: Int, _rank: Int, alignment: Int = 1
    ](coords: IndexList[_rank]) -> None:
        var coord = Coord(coords)
        var val = baseline_norm_buf.load[width=width](coord) + residual_buf.load[
            width=width
        ](coord)
        baseline_sum_buf.store[width=width](coord, val)

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
    def run_baseline(ctx: DeviceContext) raises:
        rms_norm_gpu[
            input_fn, baseline_norm_output_fn, multiply_before_cast=True
        ](shape, gamma1, epsilon1, weight_offset1, ctx)
        elementwise[add_kernel, simd_width=add_simd_width, target="gpu"](
            shape, ctx
        )
        rms_norm_gpu[
            baseline_sum_input_fn,
            baseline_output_fn,
            multiply_before_cast=True,
        ](shape, gamma2, epsilon2, weight_offset2, ctx)

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
    def run_fused(ctx: DeviceContext) raises:
        rms_norm_fused_residual_add_gpu[
            input_fn,
            residual_input_fn,
            fused_sum_output_fn,
            fused_output_fn,
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

    @always_inline
    @parameter
    def baseline_bench(mut bencher: Bencher) raises:
        bencher.iter_custom[run_baseline](ctx)

    @always_inline
    @parameter
    def fused_bench(mut bencher: Bencher) raises:
        bencher.iter_custom[run_fused](ctx)

    b.bench_function[baseline_bench](
        BenchId(
            "gemma3_rms_norm_boundary_baseline",
            input_id=String(dtype, "/", shape),
        ),
    )
    b.bench_function[fused_bench](
        BenchId(
            "gemma3_rms_norm_boundary_fused",
            input_id=String(dtype, "/", shape),
        ),
    )

    run_baseline(ctx)
    run_fused(ctx)
    ctx.enqueue_copy(baseline_sum_h, baseline_sum_d)
    ctx.enqueue_copy(baseline_output_h, baseline_output_d)
    ctx.enqueue_copy(fused_sum_h, fused_sum_d)
    ctx.enqueue_copy(fused_output_h, fused_output_d)
    ctx.synchronize()

    for i in range(rows * cols):
        assert_almost_equal(
            baseline_sum_h[i], fused_sum_h[i], rtol=2e-2, atol=2e-2
        )
        assert_almost_equal(
            baseline_output_h[i], fused_output_h[i], rtol=2e-2, atol=2e-2
        )

    _ = data_d
    _ = residual_d
    _ = baseline_norm_d
    _ = baseline_sum_d
    _ = baseline_output_d
    _ = fused_sum_d
    _ = fused_output_d
    _ = gamma1_d
    _ = gamma2_d

    data_h.free()
    residual_h.free()
    gamma1_h.free()
    gamma2_h.free()
    baseline_sum_h.free()
    baseline_output_h.free()
    fused_sum_h.free()
    fused_output_h.free()


def main() raises:
    comptime dtype = get_defined_dtype["dtype", DType.bfloat16]()
    comptime shape = int_list_to_tuple[get_defined_shape["shape", "1x64x4608"]()]()

    var b = Bench(BenchConfig(num_repetitions=1))
    with DeviceContext() as ctx:
        bench_gemma3_rms_norm_boundary[dtype, shape](ctx, b)

    b.dump_report()
