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

from std.math import ceildiv
from std.random import random_ui64, seed
from std.sys import get_defined_dtype, get_defined_int
from std.sys.info import align_of, simd_width_of

from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx_uint as block_idx,
    thread_idx_uint as thread_idx,
)
from std.gpu.host import DeviceContext
from internal_utils import arg_parse
from layout import Coord, Idx, TensorLayout, TileTensor, row_major
from layout._fillers import random
from nn.fused_qk_rope import _rope_complex_mul_half
from nn.rope import rope_ragged

from std.utils.index import IndexList
from std.utils.static_tuple import StaticTuple


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(block_size))
)
def _rope_qonly_decode_ragged_trial_kernel[
    dtype: DType,
    freq_dtype: DType,
    QLayoutType: TensorLayout,
    OutputLayoutType: TensorLayout,
    StartPosLayoutType: TensorLayout,
    FreqLayoutType: TensorLayout,
    block_size: Int,
    warps_per_block: Int,
](
    x: TileTensor[dtype, QLayoutType, MutAnyOrigin],
    output: TileTensor[mut=True, dtype, OutputLayoutType, MutAnyOrigin],
    start_pos: TileTensor[DType.uint32, StartPosLayoutType, MutAnyOrigin],
    freqs_cis: TileTensor[freq_dtype, FreqLayoutType, MutAnyOrigin],
    num_rows: Int,
):
    comptime assert x.flat_rank == 3
    comptime assert output.flat_rank == 3
    comptime assert start_pos.flat_rank == 1
    comptime assert freqs_cis.flat_rank == 2

    comptime simd_width = simd_width_of[dtype]()
    comptime vec_width = simd_width // 2
    comptime align = align_of[SIMD[dtype, vec_width]]()
    comptime half_warp_size = WARP_SIZE // 2
    comptime num_q_heads = x.static_shape[1]
    comptime head_dim = x.static_shape[2]

    comptime assert (
        head_dim == 128
    ), "Only 128-column BF16 query rows are supported"
    comptime assert head_dim == half_warp_size * simd_width
    comptime assert freqs_cis.static_shape[1] == head_dim

    var tid = thread_idx.x
    var warp_idx = tid // UInt(WARP_SIZE)
    var sub_warp_idx = (tid % UInt(WARP_SIZE)) // UInt(half_warp_size)
    var local_tid = tid % UInt(half_warp_size)
    var row = (
        block_idx.x * UInt(warps_per_block * 2) + warp_idx * 2 + sub_warp_idx
    )
    if row < UInt(num_rows):
        var row_int = Int(row)
        var global_token_idx = row_int // num_q_heads
        var head_idx = row_int % num_q_heads
        var position = Int(start_pos[global_token_idx])

        var re_offset = Int(local_tid) * vec_width
        var im_offset = re_offset + head_dim // 2
        var freq_offset = Int(local_tid) * simd_width

        var q_re = x.load[width=vec_width, alignment=align](
            Coord(Idx(global_token_idx), Idx(head_idx), Idx(re_offset))
        )
        var q_im = x.load[width=vec_width, alignment=align](
            Coord(Idx(global_token_idx), Idx(head_idx), Idx(im_offset))
        )
        var freq = freqs_cis.load[width=simd_width, alignment=1](
            Coord(Idx(position), Idx(freq_offset))
        )
        var rope_val = _rope_complex_mul_half[
            dtype,
            freq_dtype,
            vec_width,
            simd_width,
        ](q_re, q_im, freq)

        output.store[alignment=align](
            Coord(Idx(global_token_idx), Idx(head_idx), Idx(re_offset)),
            rope_val[0],
        )
        output.store[alignment=align](
            Coord(Idx(global_token_idx), Idx(head_idx), Idx(im_offset)),
            rope_val[1],
        )


def _get_run_name[
    dtype: DType,
    num_q_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
](
    batch_size: Int,
    seq_len: Int,
    cache_len: Int,
    use_random_seq_lengths: Bool,
    use_decode_fastpath: Bool,
) -> String:
    # fmt: off
    return String(
        "q_ragged_rope(", dtype, ") : ",

        # head_info
        "num_q_heads=", num_q_heads, ", ",
        "ref_num_kv_heads=", num_kv_heads, ", ",
        "head_dim=", head_dim, " : ",

        "batch_size=", batch_size, ", ",
        "seq_len=", seq_len, ", ",
        "cache_len=", cache_len, ", ",
        "use_random_seq_lengths=", use_random_seq_lengths, ", ",
        "use_decode_fastpath=", use_decode_fastpath, ", ",
    )
    # fmt: on


def execute_kv_cache_ragged_rope[
    dtype: DType, head_dim: Int, num_q_heads: Int, num_kv_heads: Int
](
    ctx: DeviceContext,
    mut m: Bench,
    batch_size: Int,
    seq_len: Int,
    cache_len: Int,
    use_random_seq_lengths: Bool,
    use_decode_fastpath: Bool,
) raises:
    comptime max_seq_len = 2048

    var input_row_offsets_device = ctx.enqueue_create_buffer[dtype.uint32](
        batch_size + 1
    )
    var start_pos_device = ctx.enqueue_create_buffer[DType.uint32](batch_size)
    var total_seq_len: UInt32 = 0

    var flop_count = 0
    with start_pos_device.map_to_host() as start_pos_host:
        with input_row_offsets_device.map_to_host() as input_row_offsets_host:
            var running_offset: UInt32 = 0
            for i in range(batch_size):
                var curr_seq_length: UInt32
                if use_random_seq_lengths:
                    curr_seq_length = random_ui64(1, UInt64(seq_len)).cast[
                        DType.uint32
                    ]()
                else:
                    curr_seq_length = UInt32(seq_len)

                input_row_offsets_host[i] = running_offset
                start_pos_host[i] = UInt32(cache_len)
                running_offset += curr_seq_length

            total_seq_len = running_offset
            input_row_offsets_host[batch_size] = total_seq_len

    var q_device = ctx.enqueue_create_buffer[dtype](
        Int(total_seq_len) * num_q_heads * head_dim
    )
    var output_device = ctx.enqueue_create_buffer[dtype](len(q_device))
    var q_layout = row_major(
        (Idx(total_seq_len), Idx[num_q_heads](), Idx[head_dim]())
    )
    with q_device.map_to_host() as q_host:
        var q_tensor = TileTensor(q_host, q_layout)
        random(q_tensor)

    var q_device_tensor = TileTensor(q_device, q_layout)
    var output_device_tensor = TileTensor(
        output_device,
        row_major(Idx(total_seq_len), Idx[num_q_heads](), Idx[head_dim]()),
    )
    var input_row_offsets_tensor = TileTensor(
        input_row_offsets_device, row_major(Idx(batch_size + 1))
    )
    var start_pos_tensor = TileTensor(
        start_pos_device, row_major(Idx(batch_size))
    )

    comptime freqs_cis_table_layout = row_major[max_seq_len, head_dim]()
    var freqs_cis_table_device = ctx.enqueue_create_buffer[dtype](
        freqs_cis_table_layout.static_product
    )
    var freqs_cis_table_tensor = TileTensor(
        freqs_cis_table_device, freqs_cis_table_layout
    )

    num_flops_per_elem = 6
    num_elems = Int(total_seq_len) * num_q_heads * head_dim // 2
    flop_count = num_flops_per_elem * num_elems
    var is_decode_uniform = (
        not use_random_seq_lengths
        and seq_len == 1
        and Int(total_seq_len) == batch_size
    )

    @always_inline
    @__copy_capture(output_device_tensor)
    def output_fn[
        width: Int, alignment: Int
    ](idx: IndexList[3], val: SIMD[dtype, width]) capturing -> None:
        output_device_tensor.store[width=width, alignment=alignment](
            Coord(idx), val
        )

    @parameter
    @__copy_capture(
        q_device_tensor,
        input_row_offsets_tensor,
        start_pos_tensor,
        freqs_cis_table_tensor,
    )
    @always_inline
    def bench_func(mut b: Bencher):
        @parameter
        @always_inline
        def kernel_launch(ctx: DeviceContext) raises:
            comptime if dtype == DType.bfloat16 and head_dim == 128:
                if use_decode_fastpath and is_decode_uniform:
                    comptime warps_per_block = 2
                    comptime block_size = warps_per_block * WARP_SIZE
                    comptime kernel = _rope_qonly_decode_ragged_trial_kernel[
                        dtype,
                        dtype,
                        q_device_tensor.LayoutType,
                        output_device_tensor.LayoutType,
                        start_pos_tensor.LayoutType,
                        freqs_cis_table_tensor.LayoutType,
                        block_size,
                        warps_per_block,
                    ]
                    var num_rows = Int(total_seq_len) * num_q_heads
                    ctx.enqueue_function[kernel, kernel](
                        q_device_tensor,
                        output_device_tensor,
                        start_pos_tensor,
                        freqs_cis_table_tensor,
                        num_rows,
                        grid_dim=ceildiv(num_rows, warps_per_block * 2),
                        block_dim=block_size,
                    )
                else:
                    rope_ragged[
                        dtype,
                        dtype,
                        interleaved=False,
                        target="gpu",
                        output_fn=output_fn,
                    ](
                        x=q_device_tensor,
                        input_row_offsets=input_row_offsets_tensor,
                        start_pos=start_pos_tensor,
                        freqs_cis=freqs_cis_table_tensor,
                        context=Optional[DeviceContext](ctx),
                    )
            else:
                rope_ragged[
                    dtype,
                    dtype,
                    interleaved=False,
                    target="gpu",
                    output_fn=output_fn,
                ](
                    x=q_device_tensor,
                    input_row_offsets=input_row_offsets_tensor,
                    start_pos=start_pos_tensor,
                    freqs_cis=freqs_cis_table_tensor,
                    context=Optional[DeviceContext](ctx),
                )

        b.iter_custom[kernel_launch](ctx)

    m.bench_function[bench_func](
        BenchId(
            _get_run_name[dtype, num_q_heads, num_kv_heads, head_dim](
                batch_size,
                seq_len,
                cache_len,
                use_random_seq_lengths,
                use_decode_fastpath,
            )
        ),
        [ThroughputMeasure(BenchMetric.flops, flop_count)],
    )


def main() raises:
    comptime dtype = get_defined_dtype["dtype", DType.bfloat16]()

    comptime head_dim = get_defined_int["head_dim", 128]()
    comptime num_q_heads = get_defined_int["num_q_heads", 32]()
    comptime num_kv_heads = get_defined_int["num_kv_heads", 8]()

    var batch_size = arg_parse("batch_size", 1)
    var cache_len = arg_parse("cache_len", 10)
    var use_random_seq_lengths = arg_parse("use_random_lengths", False)
    var use_decode_fastpath = arg_parse("use_decode_fastpath", False)
    var seq_len = arg_parse("seq_len", 1)

    seed(0)

    var m = Bench()
    with DeviceContext() as ctx:
        execute_kv_cache_ragged_rope[
            dtype,
            head_dim,
            num_q_heads,
            num_kv_heads,
        ](
            ctx,
            m,
            batch_size,
            seq_len,
            cache_len,
            use_random_seq_lengths,
            use_decode_fastpath,
        )

    m.dump_report()
