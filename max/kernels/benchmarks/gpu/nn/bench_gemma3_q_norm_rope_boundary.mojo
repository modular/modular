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
from std.sys import get_defined_dtype, get_defined_int

from std.benchmark import Bench, BenchConfig, Bencher, BenchId
from std.gpu.host import DeviceContext
from std.testing import assert_almost_equal

from internal_utils import arg_parse
from layout import Coord, Idx, TileTensor, row_major
from nn.normalization import rms_norm_gpu
from nn.rope import q_rms_norm_rope_ragged, rope_ragged

from std.utils.index import Index, IndexList


def bench_gemma3_q_norm_rope_boundary[
    dtype: DType, head_dim: Int, num_q_heads: Int
](
    ctx: DeviceContext,
    mut bench: Bench,
    batch_size: Int,
    seq_len: Int,
    cache_len: Int,
) raises:
    comptime max_seq_len = 2048

    var total_seq_len = batch_size * seq_len
    var num_elems = total_seq_len * num_q_heads * head_dim
    var q_shape = IndexList[3](total_seq_len, num_q_heads, head_dim)
    var q_layout = row_major(
        (Idx(total_seq_len), Idx[num_q_heads](), Idx[head_dim]())
    )

    var q_h = alloc[Scalar[dtype]](num_elems)
    var gamma_h = alloc[Scalar[dtype]](head_dim)
    var input_row_offsets_h = alloc[Scalar[DType.uint32]](batch_size + 1)
    var start_pos_h = alloc[Scalar[DType.uint32]](batch_size)
    var freqs_h = alloc[Scalar[dtype]](max_seq_len * head_dim)
    var baseline_h = alloc[Scalar[dtype]](num_elems)
    var fused_h = alloc[Scalar[dtype]](num_elems)

    for i in range(num_elems):
        q_h[i] = Scalar[dtype](random_float64(0, 100).cast[dtype]())

    for i in range(head_dim):
        gamma_h[i] = (Float64(i + head_dim) / Float64(head_dim)).cast[dtype]()

    var running_offset: UInt32 = 0
    for i in range(batch_size):
        input_row_offsets_h[i] = running_offset
        start_pos_h[i] = UInt32(cache_len)
        running_offset += UInt32(seq_len)
    input_row_offsets_h[batch_size] = running_offset

    for i in range(max_seq_len * head_dim):
        freqs_h[i] = Scalar[dtype](random_float64(-1, 1).cast[dtype]())

    var q_d = ctx.enqueue_create_buffer[dtype](num_elems)
    var gamma_d = ctx.enqueue_create_buffer[dtype](head_dim)
    var input_row_offsets_d = ctx.enqueue_create_buffer[DType.uint32](
        batch_size + 1
    )
    var start_pos_d = ctx.enqueue_create_buffer[DType.uint32](batch_size)
    var freqs_d = ctx.enqueue_create_buffer[dtype](max_seq_len * head_dim)
    var baseline_norm_d = ctx.enqueue_create_buffer[dtype](num_elems)
    var baseline_output_d = ctx.enqueue_create_buffer[dtype](num_elems)
    var fused_output_d = ctx.enqueue_create_buffer[dtype](num_elems)

    ctx.enqueue_copy(q_d, q_h)
    ctx.enqueue_copy(gamma_d, gamma_h)
    ctx.enqueue_copy(input_row_offsets_d, input_row_offsets_h)
    ctx.enqueue_copy(start_pos_d, start_pos_h)
    ctx.enqueue_copy(freqs_d, freqs_h)

    var q_tensor = TileTensor(q_d, q_layout)
    var gamma_tensor = TileTensor(gamma_d, row_major(Idx[head_dim]()))
    var input_row_offsets_tensor = TileTensor(
        input_row_offsets_d, row_major(Idx(batch_size + 1))
    )
    var start_pos_tensor = TileTensor(start_pos_d, row_major(Idx(batch_size)))
    comptime freqs_layout = row_major[max_seq_len, head_dim]()
    var freqs_tensor = TileTensor(freqs_d, freqs_layout)
    var baseline_norm_tensor = TileTensor(baseline_norm_d, q_layout)
    var baseline_output_tensor = TileTensor(baseline_output_d, q_layout)
    var fused_output_tensor = TileTensor(fused_output_d, q_layout)

    var epsilon = Scalar[dtype](1e-6)
    var weight_offset = Scalar[dtype](1.0)

    @always_inline
    @__copy_capture(q_tensor)
    @parameter
    def input_fn[
        width: Int, rank: Int
    ](coords: IndexList[rank]) -> SIMD[dtype, width]:
        return q_tensor.load[width=width](Coord(coords))

    @always_inline
    @__copy_capture(baseline_norm_tensor)
    @parameter
    def baseline_norm_output_fn[
        width: Int, alignment: Int
    ](coords: IndexList[3], val: SIMD[dtype, width]) -> None:
        baseline_norm_tensor.store[alignment=alignment](Coord(coords), val)

    @always_inline
    @__copy_capture(baseline_output_tensor)
    def rope_output_fn[
        width: Int, alignment: Int
    ](idx: IndexList[3], val: SIMD[dtype, width]) capturing -> None:
        baseline_output_tensor.store[alignment=alignment](Coord(idx), val)

    @always_inline
    @__copy_capture(
        q_layout,
        gamma_tensor,
        epsilon,
        weight_offset,
        baseline_norm_tensor,
        input_row_offsets_tensor,
        start_pos_tensor,
        freqs_tensor,
    )
    @parameter
    def run_baseline(ctx: DeviceContext) raises:
        rms_norm_gpu[
            input_fn,
            baseline_norm_output_fn,
            multiply_before_cast=True,
        ](q_shape, gamma_tensor, epsilon, weight_offset, ctx)
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
            freqs_cis=freqs_tensor,
            context=Optional[DeviceContext](ctx),
        )

    @always_inline
    @__copy_capture(
        q_tensor,
        fused_output_tensor,
        input_row_offsets_tensor,
        start_pos_tensor,
        freqs_tensor,
        gamma_tensor,
        epsilon,
        weight_offset,
    )
    @parameter
    def run_fused(ctx: DeviceContext) raises:
        q_rms_norm_rope_ragged[
            dtype,
            dtype,
            target="gpu",
        ](
            x=q_tensor,
            input_row_offsets=input_row_offsets_tensor,
            start_pos=start_pos_tensor,
            freqs_cis=freqs_tensor,
            gamma=gamma_tensor,
            epsilon=epsilon,
            weight_offset=weight_offset,
            output=fused_output_tensor,
            context=ctx,
        )

    @always_inline
    @parameter
    def baseline_bench(mut bencher: Bencher) raises:
        bencher.iter_custom[run_baseline](ctx)

    @always_inline
    @parameter
    def fused_bench(mut bencher: Bencher) raises:
        bencher.iter_custom[run_fused](ctx)

    bench.bench_function[baseline_bench](
        BenchId(
            "gemma3_q_norm_rope_boundary_baseline",
            input_id=String(
                dtype,
                "/bs=",
                batch_size,
                "/seq=",
                seq_len,
                "/cache=",
                cache_len,
            ),
        ),
    )
    bench.bench_function[fused_bench](
        BenchId(
            "gemma3_q_norm_rope_boundary_fused",
            input_id=String(
                dtype,
                "/bs=",
                batch_size,
                "/seq=",
                seq_len,
                "/cache=",
                cache_len,
            ),
        ),
    )

    run_baseline(ctx)
    run_fused(ctx)
    ctx.enqueue_copy(baseline_h, baseline_output_d)
    ctx.enqueue_copy(fused_h, fused_output_d)
    ctx.synchronize()

    for i in range(num_elems):
        assert_almost_equal(baseline_h[i], fused_h[i], rtol=2e-2, atol=2e-2)

    _ = q_d
    _ = gamma_d
    _ = input_row_offsets_d
    _ = start_pos_d
    _ = freqs_d
    _ = baseline_norm_d
    _ = baseline_output_d
    _ = fused_output_d

    q_h.free()
    gamma_h.free()
    input_row_offsets_h.free()
    start_pos_h.free()
    freqs_h.free()
    baseline_h.free()
    fused_h.free()


def main() raises:
    comptime dtype = get_defined_dtype["dtype", DType.bfloat16]()
    comptime head_dim = get_defined_int["head_dim", 128]()
    comptime num_q_heads = get_defined_int["num_q_heads", 32]()

    var batch_size = arg_parse("batch_size", 64)
    var seq_len = arg_parse("seq_len", 1)
    var cache_len = arg_parse("cache_len", 1024)

    var bench = Bench(BenchConfig(num_repetitions=1))
    with DeviceContext() as ctx:
        bench_gemma3_q_norm_rope_boundary[
            dtype,
            head_dim,
            num_q_heads,
        ](ctx, bench, batch_size, seq_len, cache_len)

    bench.dump_report()
