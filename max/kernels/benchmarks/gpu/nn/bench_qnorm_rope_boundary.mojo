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

from std.random import random_float64, seed
from std.sys import get_defined_dtype, get_defined_int
from std.sys.info import align_of, simd_width_of

from std.benchmark import Bench, BenchConfig, Bencher, BenchId
from std.gpu import WARP_SIZE, block_idx, thread_idx
from std.gpu.host import DeviceContext
from std.gpu.primitives import warp
from std.testing import assert_almost_equal
from std.utils.numerics import get_accum_type

from internal_utils import arg_parse
from layout import Coord, Idx, TileTensor, row_major
from nn._ragged_utils import get_batch_from_row_offsets
from nn.normalization import rms_norm_gpu
from nn.rope import q_rms_norm_rope_ragged, rope_ragged

from std.utils import IndexList


@always_inline
def _rope_complex_mul_half_local[
    dtype: DType,
    freq_dtype: DType,
    width_2: Int,
    freq_width: Int,
](
    x_re: SIMD[dtype, width_2],
    x_im: SIMD[dtype, width_2],
    freq: SIMD[freq_dtype, freq_width],
) -> Tuple[SIMD[dtype, width_2], SIMD[dtype, width_2]]:
    var f_re = rebind[SIMD[freq_dtype, width_2]](freq.deinterleave()[0])
    var f_im = rebind[SIMD[freq_dtype, width_2]](freq.deinterleave()[1])
    var xr = x_re.cast[freq_dtype]()
    var xi = x_im.cast[freq_dtype]()
    var res_re = (xr * f_re - xi * f_im).cast[dtype]()
    var res_im = (xr * f_im + xi * f_re).cast[dtype]()
    return (res_re, res_im)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(warps_per_block * WARP_SIZE)
    )
)
def qnorm_rope_ragged_non_interleaved_128_kernel[
    dtype: DType,
    freq_dtype: DType,
    //,
    warps_per_block: Int,
](
    q_proj: TileTensor[dtype, ...],
    input_row_offsets: TileTensor[DType.uint32, ...],
    start_pos: TileTensor[DType.uint32, ...],
    freqs_cis: TileTensor[freq_dtype, ...],
    gamma: TileTensor[dtype, ...],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    num_rows: Int,
    output: TileTensor[mut=True, dtype, ...],
):
    comptime assert q_proj.flat_rank == 3
    comptime assert input_row_offsets.flat_rank == 1
    comptime assert start_pos.flat_rank == 1
    comptime assert freqs_cis.flat_rank == 2
    comptime assert gamma.flat_rank == 1
    comptime num_q_heads = Int(q_proj.static_shape[1])
    comptime head_dim = Int(q_proj.static_shape[2])
    comptime assert head_dim == 128
    comptime simd_width = simd_width_of[dtype]()
    comptime assert simd_width == 8
    comptime vec_width = simd_width // 2
    comptime half_warp_size = WARP_SIZE // 2
    comptime align = align_of[SIMD[dtype, vec_width]]()
    comptime accum_type = get_accum_type[dtype]()

    var eps_accum = epsilon.cast[accum_type]()
    var weight_offset_accum = weight_offset.cast[accum_type]()

    var tid = thread_idx.x
    var block_row = Int(block_idx.x) * (warps_per_block * 2)
    var warp_id = Int(tid // UInt(WARP_SIZE))
    var sub_warp_id = Int((tid % UInt(WARP_SIZE)) // UInt(half_warp_size))
    var row = block_row + (warp_id * 2) + sub_warp_id
    var local_tid = Int(tid % UInt(half_warp_size))

    if row < num_rows:
        var global_token_idx = row // num_q_heads
        var head_idx = row % num_q_heads
        var batch_idx = get_batch_from_row_offsets(
            input_row_offsets, global_token_idx
        )
        var token_idx = Int(
            UInt32(global_token_idx) - input_row_offsets[batch_idx]
        )
        var position_idx = Int(start_pos[batch_idx] + UInt32(token_idx))

        var re_col = local_tid * vec_width
        var im_col = re_col + (head_dim // 2)
        var freq_col = local_tid * simd_width

        var coord_re = IndexList[3](global_token_idx, head_idx, re_col)
        var coord_im = IndexList[3](global_token_idx, head_idx, im_col)
        var q_re = q_proj.load[width=vec_width, alignment=align](Coord(coord_re))
        var q_im = q_proj.load[width=vec_width, alignment=align](Coord(coord_im))

        var thread_m2 = (q_re.cast[accum_type]() ** 2).reduce_add() + (
            q_im.cast[accum_type]() ** 2
        ).reduce_add()
        var row_m2 = warp.lane_group_sum[num_lanes=half_warp_size](thread_m2)
        var norm_factor = rsqrt(
            (row_m2 / Scalar[accum_type](head_dim)) + eps_accum
        )

        var gamma_re = gamma.load[width=vec_width, alignment=align](
            Coord(Idx(re_col))
        )
        var gamma_im = gamma.load[width=vec_width, alignment=align](
            Coord(Idx(im_col))
        )
        var norm_re = (
            q_re.cast[accum_type]()
            * norm_factor
            * (gamma_re.cast[accum_type]() + weight_offset_accum)
        ).cast[dtype]()
        var norm_im = (
            q_im.cast[accum_type]()
            * norm_factor
            * (gamma_im.cast[accum_type]() + weight_offset_accum)
        ).cast[dtype]()

        var freq = freqs_cis.load[width=simd_width](
            Coord(Idx(position_idx), Idx(freq_col))
        )
        var rope_out = _rope_complex_mul_half_local(norm_re, norm_im, freq)

        output.store[alignment=align](Coord(coord_re), rope_out[0])
        output.store[alignment=align](Coord(coord_im), rope_out[1])


def qnorm_rope_ragged_non_interleaved_128[
    dtype: DType,
    freq_dtype: DType,
](
    q_proj: TileTensor[dtype, ...],
    input_row_offsets: TileTensor[DType.uint32, ...],
    start_pos: TileTensor[DType.uint32, ...],
    freqs_cis: TileTensor[freq_dtype, ...],
    gamma: TileTensor[dtype, ...],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    output: TileTensor[mut=True, dtype, ...],
    ctx: DeviceContext,
) raises:
    comptime num_q_heads = Int(q_proj.static_shape[1])
    var num_rows = q_proj.dim(0) * num_q_heads
    comptime sm_count = ctx.default_device_info.sm_count
    comptime default_warps_per_block = 2
    comptime large_row_warps_per_block = 8
    comptime min_large_row_blocks_per_sm = 6

    var large_row_grid_dim = ceildiv(num_rows, large_row_warps_per_block * 2)
    if large_row_grid_dim >= sm_count * min_large_row_blocks_per_sm:
        comptime kernel = qnorm_rope_ragged_non_interleaved_128_kernel[
            dtype, freq_dtype, large_row_warps_per_block
        ]
        ctx.enqueue_function[kernel, kernel](
            q_proj,
            input_row_offsets,
            start_pos,
            freqs_cis,
            gamma,
            epsilon,
            weight_offset,
            num_rows,
            output,
            grid_dim=large_row_grid_dim,
            block_dim=large_row_warps_per_block * WARP_SIZE,
        )
    else:
        comptime kernel = qnorm_rope_ragged_non_interleaved_128_kernel[
            dtype, freq_dtype, default_warps_per_block
        ]
        ctx.enqueue_function[kernel, kernel](
            q_proj,
            input_row_offsets,
            start_pos,
            freqs_cis,
            gamma,
            epsilon,
            weight_offset,
            num_rows,
            output,
            grid_dim=ceildiv(num_rows, default_warps_per_block * 2),
            block_dim=default_warps_per_block * WARP_SIZE,
        )


def _get_run_name[
    dtype: DType,
    num_q_heads: Int,
    head_dim: Int,
](batch_size: Int, seq_len: Int, cache_len: Int,) -> String:
    return String(
        "qnorm_rope_boundary(",
        dtype,
        ") : num_q_heads=",
        num_q_heads,
        ", head_dim=",
        head_dim,
        " : batch_size=",
        batch_size,
        ", seq_len=",
        seq_len,
        ", cache_len=",
        cache_len,
    )


def bench_qnorm_rope_boundary[
    dtype: DType,
    head_dim: Int,
    num_q_heads: Int,
](
    ctx: DeviceContext,
    mut b: Bench,
    batch_size: Int,
    seq_len: Int,
    cache_len: Int,
) raises:
    comptime max_seq_len = 2048
    var input_row_offsets_device = ctx.enqueue_create_buffer[DType.uint32](
        batch_size + 1
    )
    var start_pos_device = ctx.enqueue_create_buffer[DType.uint32](batch_size)
    var total_seq_len: UInt32 = 0

    with start_pos_device.map_to_host() as start_pos_host:
        with input_row_offsets_device.map_to_host() as input_row_offsets_host:
            var running_offset: UInt32 = 0
            for i in range(batch_size):
                input_row_offsets_host[i] = running_offset
                start_pos_host[i] = UInt32(cache_len)
                running_offset += UInt32(seq_len)

            total_seq_len = running_offset
            input_row_offsets_host[batch_size] = total_seq_len

    var q_layout = row_major(
        (Idx(total_seq_len), Idx[num_q_heads](), Idx[head_dim]())
    )
    var q_device = ctx.enqueue_create_buffer[dtype](
        Int(total_seq_len) * num_q_heads * head_dim
    )
    var baseline_norm_device = ctx.enqueue_create_buffer[dtype](len(q_device))
    var baseline_output_device = ctx.enqueue_create_buffer[dtype](len(q_device))
    var fused_output_device = ctx.enqueue_create_buffer[dtype](len(q_device))
    var gamma_device = ctx.enqueue_create_buffer[dtype](head_dim)
    comptime freqs_cis_layout = row_major[max_seq_len, head_dim]()
    var freqs_cis_device = ctx.enqueue_create_buffer[dtype](
        freqs_cis_layout.static_product
    )

    with q_device.map_to_host() as q_host:
        var q_tensor = TileTensor(q_host, q_layout)
        for i in range(Int(total_seq_len) * num_q_heads * head_dim):
            q_host[i] = Scalar[dtype](random_float64(-1, 1).cast[dtype]())

    with freqs_cis_device.map_to_host() as freqs_h:
        for i in range(freqs_cis_layout.static_product):
            freqs_h[i] = Scalar[dtype](random_float64(-1, 1).cast[dtype]())

    with gamma_device.map_to_host() as gamma_h:
        for i in range(head_dim):
            gamma_h[i] = Scalar[dtype]((Float64(i + head_dim) / Float64(head_dim)).cast[dtype]())

    var q_tensor = TileTensor(q_device, q_layout)
    var baseline_norm_tensor = TileTensor(baseline_norm_device, q_layout)
    var baseline_output_tensor = TileTensor(baseline_output_device, q_layout)
    var fused_output_tensor = TileTensor(fused_output_device, q_layout)
    var input_row_offsets_tensor = TileTensor(
        input_row_offsets_device, row_major(Idx(batch_size + 1))
    )
    var start_pos_tensor = TileTensor(start_pos_device, row_major(Idx(batch_size)))
    var freqs_cis_tensor = TileTensor(freqs_cis_device, freqs_cis_layout)
    var gamma_tensor = TileTensor(gamma_device, row_major(Idx[head_dim]()))
    var epsilon = Scalar[dtype](0.001)
    var weight_offset = Scalar[dtype](1.0)

    @always_inline
    @__copy_capture(q_tensor)
    @parameter
    def rms_input_fn[
        width: Int, rank: Int
    ](coords: IndexList[rank]) -> SIMD[dtype, width]:
        return q_tensor.load[width=width](Coord(coords))

    @always_inline
    @__copy_capture(baseline_norm_tensor)
    @parameter
    def rms_output_fn[
        width: Int, alignment: Int
    ](coords: IndexList[3], val: SIMD[dtype, width]) -> None:
        baseline_norm_tensor.store[alignment=alignment](Coord(coords), val)

    @always_inline
    @__copy_capture(baseline_output_tensor)
    @parameter
    def rope_output_fn[
        width: Int, alignment: Int
    ](idx: IndexList[3], val: SIMD[dtype, width]) -> None:
        baseline_output_tensor.store[alignment=alignment](Coord(idx), val)

    @always_inline
    @__copy_capture(q_layout, gamma_tensor, epsilon, weight_offset)
    @parameter
    def run_baseline(ctx: DeviceContext) raises:
        rms_norm_gpu[rms_input_fn, rms_output_fn, multiply_before_cast=True](
            IndexList[3](Int(total_seq_len), num_q_heads, head_dim),
            gamma_tensor,
            epsilon,
            weight_offset,
            ctx,
        )
        rope_ragged[
            dtype,
            dtype,
            interleaved=False,
            target="gpu",
            output_fn=rope_output_fn,
        ](
            x=baseline_norm_tensor,
            input_row_offsets=input_row_offsets_tensor,
            start_pos=start_pos_tensor,
            freqs_cis=freqs_cis_tensor,
            context=Optional[DeviceContext](ctx),
        )

    @always_inline
    @__copy_capture(q_tensor, gamma_tensor, epsilon, weight_offset)
    @parameter
    def run_fused(ctx: DeviceContext) raises:
        q_rms_norm_rope_ragged[dtype, dtype, target="gpu"](
            q_tensor,
            input_row_offsets_tensor,
            start_pos_tensor,
            freqs_cis_tensor,
            gamma_tensor,
            epsilon,
            weight_offset,
            fused_output_tensor,
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
            "qnorm_rope_boundary_baseline",
            input_id=_get_run_name[dtype, num_q_heads, head_dim](
                batch_size, seq_len, cache_len
            ),
        ),
    )
    b.bench_function[fused_bench](
        BenchId(
            "qnorm_rope_boundary_fused",
            input_id=_get_run_name[dtype, num_q_heads, head_dim](
                batch_size, seq_len, cache_len
            ),
        ),
    )

    run_baseline(ctx)
    run_fused(ctx)
    var baseline_host = alloc[Scalar[dtype]](len(q_device))
    var fused_host = alloc[Scalar[dtype]](len(q_device))
    ctx.enqueue_copy(baseline_host, baseline_output_device)
    ctx.enqueue_copy(fused_host, fused_output_device)
    ctx.synchronize()

    for i in range(len(q_device)):
        assert_almost_equal(
            baseline_host[i], fused_host[i], rtol=2e-2, atol=2e-2
        )

    baseline_host.free()
    fused_host.free()


def main() raises:
    comptime dtype = get_defined_dtype["dtype", DType.bfloat16]()
    comptime head_dim = get_defined_int["head_dim", 128]()
    comptime num_q_heads = get_defined_int["num_q_heads", 32]()

    var batch_size = arg_parse("batch_size", 1)
    var seq_len = arg_parse("seq_len", 1)
    var cache_len = arg_parse("cache_len", 0)

    seed(0)

    var b = Bench(BenchConfig(num_repetitions=1))
    with DeviceContext() as ctx:
        bench_qnorm_rope_boundary[dtype, head_dim, num_q_heads](
            ctx, b, batch_size, seq_len, cache_len
        )

    b.dump_report()
