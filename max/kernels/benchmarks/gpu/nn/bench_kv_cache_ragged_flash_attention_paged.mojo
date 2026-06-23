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

from std.collections import Set
from std.math import ceildiv, rsqrt
from std.random import random_ui64, seed
from std.sys import get_defined_bool, get_defined_dtype, get_defined_int

from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from layout import (
    Coord,
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from std.gpu.host import DeviceContext
from internal_utils import arg_parse
from layout._fillers import random
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from nn.attention.gpu.mha import flash_attention
from nn.attention.mha_mask import CausalMask, SlidingWindowCausalMask

from std.utils import IndexList


def flops(
    batch: Int, nheads: Int, seqlen_q: Int, seqlen_k: Int, headdim: Int
) raises -> Int:
    var avg_seqlen = Float64(max(seqlen_k - seqlen_q, 0) + seqlen_k) / 2
    return Int(
        Float64(batch * nheads * 2 * seqlen_q)
        * avg_seqlen
        * Float64((headdim + headdim))
    )


def _get_run_name[
    dtype: DType,
    num_q_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    cross_attention: Bool,
    sink: Bool,
](
    batch_size: Int,
    seq_len: Int,
    use_random_seq_lengths: Bool,
    cache_len: Int,
    use_random_cache_lengths: Bool,
) -> String:
    return String(
        "fused_qkv_ragged_flash_attention",
        "(",
        dtype,
        ") : ",
        # head_info
        "num_q_heads=",
        num_q_heads,
        ", num_kv_heads=",
        num_kv_heads,
        ", head_dim=",
        head_dim,
        " : ",
        "batch_size=",
        batch_size,
        ", seq_len=",
        seq_len,
        ", use_random_seq_lengths=",
        use_random_seq_lengths,
        ", cache_len=",
        cache_len,
        ", use_random_cache_lengths=",
        use_random_cache_lengths,
        ", cross_attention=",
        cross_attention,
        ", sink=",
        sink,
    )


def execute_kv_cache_ragged_flash_attention[
    dtype: DType,
    *,
    head_dim: Int,
    num_q_heads: Int,
    num_kv_heads: Int,
    page_size: Int,
    local_window_size: Int = -1,
    cross_attention: Bool = False,
    sink: Bool = False,
](
    ctx: DeviceContext,
    mut m: Bench,
    batch_size: Int,
    seq_len: Int,
    use_random_seq_lengths: Bool,
    cache_len: Int,
    use_random_cache_lengths: Bool,
    run_benchmark: Bool,
) raises:
    comptime num_layers = 1
    comptime layer_idx = 0
    var num_pages = batch_size * ceildiv(seq_len + cache_len, page_size) * 2
    comptime CollectionType = PagedKVCacheCollection[
        dtype,
        KVCacheStaticParams(num_heads=num_kv_heads, head_size=head_dim),
        page_size,
        ...,
    ]

    debug_assert(
        batch_size < num_pages,
        "batch_size passed to unit test (",
        batch_size,
        ") is larger than configured num_pages (",
        num_pages,
        ")",
    )

    # Host allocations for row offsets and cache lengths
    var input_row_offsets_host_ptr = List(
        length=batch_size + 1, fill=Scalar[DType.uint32](0)
    )
    var cache_lengths_host_ptr = List(
        length=batch_size, fill=Scalar[DType.uint32](0)
    )
    var max_context_length = 0
    var max_seq_length: UInt32 = 0
    var total_seq_len: UInt32 = 0
    var valid_lengths = List[Int]()

    for i in range(batch_size):
        var curr_seq_length: UInt32
        if use_random_seq_lengths:
            curr_seq_length = random_ui64(1, UInt64(seq_len)).cast[
                DType.uint32
            ]()
        else:
            curr_seq_length = UInt32(seq_len)
        valid_lengths.append(Int(curr_seq_length))

        var curr_cache_length: UInt32
        if use_random_cache_lengths:
            curr_cache_length = random_ui64(0, UInt64(cache_len)).cast[
                DType.uint32
            ]()
        else:
            curr_cache_length = UInt32(cache_len)

        curr_context_length = Int(curr_cache_length) + Int(curr_seq_length)

        max_context_length = max(max_context_length, curr_context_length)
        max_seq_length = max(max_seq_length, curr_seq_length)

        input_row_offsets_host_ptr[i] = total_seq_len
        cache_lengths_host_ptr[i] = curr_cache_length
        total_seq_len += curr_seq_length

    input_row_offsets_host_ptr[batch_size] = total_seq_len

    # Device allocations and copies for row offsets
    var input_row_offsets_dev_buffer = ctx.enqueue_create_buffer[DType.uint32](
        batch_size + 1
    )
    ctx.enqueue_copy(input_row_offsets_dev_buffer, input_row_offsets_host_ptr)

    # Device allocation and copy for cache lengths
    var cache_lengths_dev_buffer = ctx.enqueue_create_buffer[DType.uint32](
        batch_size
    )
    ctx.enqueue_copy(cache_lengths_dev_buffer, cache_lengths_host_ptr)

    # Q tensor allocation
    var q_size = Int(total_seq_len) * num_q_heads * head_dim
    var q_host_ptr = List(length=q_size, fill=Scalar[dtype](0))
    random(
        TileTensor(
            q_host_ptr,
            row_major(
                (
                    total_seq_len,
                    Idx[num_q_heads],
                    Idx[head_dim],
                )
            ),
        )
    )
    var q_dev_buffer = ctx.enqueue_create_buffer[dtype](q_size)
    ctx.enqueue_copy(q_dev_buffer, q_host_ptr)

    # Output tensor allocation
    var output_size = Int(total_seq_len) * num_q_heads * head_dim
    var output_host_ptr = List(length=output_size, fill=Scalar[dtype](0))
    var output_dev_buffer = ctx.enqueue_create_buffer[dtype](output_size)
    var output_device_tensor = TileTensor(
        output_dev_buffer,
        row_major((total_seq_len, Idx[num_q_heads], Idx[head_dim])),
    )
    # Paged LUT allocation. The LUT row stride (columns per sequence)
    # must satisfy `PagedKVCache.populate`'s SIMD-path contract: round
    # the page count up to a multiple of 8 so the `ld.global.v{chunk}.u32`
    # read (chunk capped at 8) is naturally aligned for every
    # `batch_idx * row_stride` offset, and add a 16-element tail pad so a
    # max-width SIMD load at any valid `first_lut_idx` stays in-bounds.
    # Mirrors `_padded_lut_cols` (cache_manager.py) / `padded_lut_cols`
    # (kv_cache_test_utils.mojo). Without it, an odd page count yields an
    # odd row stride and odd `batch_idx` produces a 4-byte-misaligned
    # v2.u32 load -> CUDA_ERROR_MISALIGNED_ADDRESS.
    var paged_lut_cols = (
        (ceildiv(max_context_length, page_size) + 7) // 8
    ) * 8 + 16
    var paged_lut_size = batch_size * paged_lut_cols

    def _ri(v: Int) -> Int64:
        return Int64(v)

    var paged_lut_host_ptr = List(
        length=paged_lut_size, fill=Scalar[DType.uint32](0)
    )
    var paged_lut_host = TileTensor(
        paged_lut_host_ptr,
        row_major(Coord(_ri(batch_size), _ri(paged_lut_cols))),
    )
    paged_lut_set = Set[Int]()
    for bs in range(batch_size):
        curr_seq_len = Int(cache_lengths_host_ptr[bs]) + valid_lengths[bs]
        for block_idx in range(0, ceildiv(curr_seq_len, page_size)):
            var randval = Int(random_ui64(0, UInt64(num_pages - 1)))
            while randval in paged_lut_set:
                randval = Int(random_ui64(0, UInt64(num_pages - 1)))

            paged_lut_set.add(randval)
            paged_lut_host[bs, block_idx] = UInt32(randval)

    var paged_lut_dev_buffer = ctx.enqueue_create_buffer[DType.uint32](
        paged_lut_size
    )
    ctx.enqueue_copy(paged_lut_dev_buffer, paged_lut_host_ptr)

    # KV block paged allocation
    var kv_block_size = (
        num_pages * 2 * num_layers * page_size * num_kv_heads * head_dim
    )
    var kv_block_paged_host_ptr = List(
        length=kv_block_size, fill=Scalar[dtype](0)
    )
    random(
        LayoutTensor[dtype, Layout.row_major[6](), MutAnyOrigin](
            kv_block_paged_host_ptr,
            RuntimeLayout[Layout.row_major[6]()].row_major(
                IndexList[6](
                    num_pages, 2, num_layers, page_size, num_kv_heads, head_dim
                )
            ),
        )
    )
    var kv_block_paged_dev_buffer = ctx.enqueue_create_buffer[dtype](
        kv_block_size
    )
    ctx.enqueue_copy(kv_block_paged_dev_buffer, kv_block_paged_host_ptr)

    # Create LayoutTensors for KV collection
    comptime kv_block_layout = Layout.row_major[6]()
    var kv_block_layout_tensor = LayoutTensor[dtype, kv_block_layout](
        kv_block_paged_dev_buffer.unsafe_ptr(),
        RuntimeLayout[kv_block_layout].row_major(
            IndexList[6](
                num_pages, 2, num_layers, page_size, num_kv_heads, head_dim
            )
        ),
    )

    comptime cache_lengths_layout = Layout(UNKNOWN_VALUE)
    var cache_lengths_layout_tensor = LayoutTensor[
        mut=False, DType.uint32, cache_lengths_layout
    ](
        cache_lengths_dev_buffer.unsafe_ptr(),
        RuntimeLayout[cache_lengths_layout].row_major(IndexList[1](batch_size)),
    )

    comptime paged_lut_layout = Layout.row_major[2]()
    var paged_lut_layout_tensor = LayoutTensor[
        mut=False, DType.uint32, paged_lut_layout
    ](
        paged_lut_dev_buffer.unsafe_ptr(),
        RuntimeLayout[paged_lut_layout].row_major(
            IndexList[2](batch_size, paged_lut_cols)
        ),
    )

    kv_collection_device = CollectionType(
        kv_block_layout_tensor,
        cache_lengths_layout_tensor,
        paged_lut_layout_tensor,
        max_seq_length,
        UInt32(max_context_length),
    )

    k_cache_device = kv_collection_device.get_key_cache(layer_idx)
    v_cache_device = kv_collection_device.get_value_cache(layer_idx)

    # Create tensors for flash_attention inputs
    var q_device_tensor = TileTensor(
        q_dev_buffer,
        row_major((total_seq_len, Idx[num_q_heads], Idx[head_dim])),
    )

    var input_row_offsets_tensor = TileTensor(
        input_row_offsets_dev_buffer,
        row_major(batch_size + 1),
    )

    # Phase-10 cross-attention path: an independent kv-side
    # input_row_offsets. For the bench harness we set it equal to
    # the Q-side so the dispatcher routes through the cross-attention
    # launcher (`mha_prefill_v2_ragged[cross_attention=True]`) but
    # `num_keys` derives identically; this measures the comptime-
    # monomorphized path, not a true encoder-decoder shape.
    var kv_input_row_offsets_dev_buffer = ctx.enqueue_create_buffer[
        DType.uint32
    ](batch_size + 1)
    comptime if cross_attention:
        ctx.enqueue_copy(
            kv_input_row_offsets_dev_buffer, input_row_offsets_host_ptr
        )

    var kv_input_row_offsets_view = LayoutTensor[
        mut=False, DType.uint32, Layout.row_major(UNKNOWN_VALUE)
    ](
        kv_input_row_offsets_dev_buffer.unsafe_ptr(),
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
            IndexList[1](batch_size + 1)
        ),
    )

    # Phase-5b sink path: per-q-head sink weight buffer. Filled with
    # a small fixed value so the seeded `(max_vec, norm_vec)` init
    # state has a non-trivial sink contribution.
    var sink_weights_dev_buffer = ctx.enqueue_create_buffer[dtype](num_q_heads)
    comptime if sink:
        var sw_host = List(length=num_q_heads, fill=Scalar[dtype](0.05))
        ctx.enqueue_copy(sink_weights_dev_buffer, sw_host)
        _ = sw_host^

    var sink_weights_view = LayoutTensor[
        dtype,
        Layout.row_major(UNKNOWN_VALUE),
    ](
        sink_weights_dev_buffer.unsafe_ptr()
        .as_immutable()
        .as_unsafe_any_origin(),
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
            IndexList[1](num_q_heads)
        ),
    )

    if run_benchmark:

        @parameter
        @__copy_capture(
            q_device_tensor,
            k_cache_device,
            v_cache_device,
            output_device_tensor,
            input_row_offsets_tensor,
            kv_input_row_offsets_view,
            sink_weights_view,
        )
        @always_inline
        def bench_func(mut b: Bencher):
            @parameter
            @always_inline
            def kernel_launch(ctx: DeviceContext) raises:
                comptime if local_window_size > 0:
                    comptime assert (
                        not sink
                    ), "sliding window mask does not support sink"
                    comptime assert (
                        not cross_attention
                    ), "sliding window mask does not support cross_attention"
                    flash_attention[ragged=True](
                        output_device_tensor.to_layout_tensor().as_unsafe_any_origin(),
                        q_device_tensor.to_layout_tensor(),
                        k_cache_device,
                        v_cache_device,
                        SlidingWindowCausalMask[local_window_size](),
                        input_row_offsets_tensor.to_layout_tensor(),
                        rsqrt(Float32(head_dim)),
                        ctx,
                    )
                else:
                    # Sink/cross_attention dispatch: passing
                    # `sink_weights=…` to `flash_attention[sink=True]`
                    # selects the dispatcher's `comptime if sink:`
                    # branch. Passing
                    # `kv_input_row_offsets=…` selects the runtime
                    # `if kv_input_row_offsets:` branch.
                    comptime if sink and cross_attention:
                        flash_attention[ragged=True, sink=True](
                            output_device_tensor.to_layout_tensor().as_unsafe_any_origin(),
                            q_device_tensor.to_layout_tensor(),
                            k_cache_device,
                            v_cache_device,
                            CausalMask(),
                            input_row_offsets_tensor.to_layout_tensor(),
                            rsqrt(Float32(head_dim)),
                            ctx,
                            kv_input_row_offsets=kv_input_row_offsets_view.as_unsafe_any_origin(),
                            sink_weights=sink_weights_view,
                        )
                    elif sink:
                        flash_attention[ragged=True, sink=True](
                            output_device_tensor.to_layout_tensor().as_unsafe_any_origin(),
                            q_device_tensor.to_layout_tensor(),
                            k_cache_device,
                            v_cache_device,
                            CausalMask(),
                            input_row_offsets_tensor.to_layout_tensor(),
                            rsqrt(Float32(head_dim)),
                            ctx,
                            sink_weights=sink_weights_view,
                        )
                    elif cross_attention:
                        flash_attention[ragged=True](
                            output_device_tensor.to_layout_tensor().as_unsafe_any_origin(),
                            q_device_tensor.to_layout_tensor(),
                            k_cache_device,
                            v_cache_device,
                            CausalMask(),
                            input_row_offsets_tensor.to_layout_tensor(),
                            rsqrt(Float32(head_dim)),
                            ctx,
                            kv_input_row_offsets=kv_input_row_offsets_view.as_unsafe_any_origin(),
                        )
                    else:
                        flash_attention[ragged=True](
                            output_device_tensor.to_layout_tensor().as_unsafe_any_origin(),
                            q_device_tensor.to_layout_tensor(),
                            k_cache_device,
                            v_cache_device,
                            CausalMask(),
                            input_row_offsets_tensor.to_layout_tensor(),
                            rsqrt(Float32(head_dim)),
                            ctx,
                        )

            b.iter_custom[kernel_launch](ctx)

        flop_count = flops(
            batch_size,
            num_q_heads,
            seq_len,
            cache_len + seq_len,
            head_dim,
        )
        m.bench_function[bench_func](
            BenchId(
                _get_run_name[
                    dtype,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    cross_attention,
                    sink,
                ](
                    batch_size,
                    seq_len,
                    use_random_seq_lengths,
                    cache_len,
                    use_random_cache_lengths,
                )
            ),
            [ThroughputMeasure(BenchMetric.flops, flop_count)],
        )
    else:
        # `False` is useful for profiling with NCU.
        # We don't want to run the benchmark, as this makes the profiling
        # take a very long time and bloats the prof full of extra runs that
        # we don't look at.
        comptime if local_window_size > 0:
            flash_attention[ragged=True](
                output_device_tensor.to_layout_tensor().as_unsafe_any_origin(),
                q_device_tensor.to_layout_tensor(),
                k_cache_device,
                v_cache_device,
                SlidingWindowCausalMask[local_window_size](),
                input_row_offsets_tensor.to_layout_tensor(),
                rsqrt(Float32(head_dim)),
                ctx,
            )
        else:
            flash_attention[ragged=True](
                output_device_tensor.to_layout_tensor().as_unsafe_any_origin(),
                q_device_tensor.to_layout_tensor(),
                k_cache_device,
                v_cache_device,
                CausalMask(),
                input_row_offsets_tensor.to_layout_tensor(),
                rsqrt(Float32(head_dim)),
                ctx,
            )

    # Consume device buffers
    _ = input_row_offsets_dev_buffer^
    _ = cache_lengths_dev_buffer^
    _ = q_dev_buffer^
    _ = output_dev_buffer^
    _ = paged_lut_dev_buffer^
    _ = kv_block_paged_dev_buffer^
    _ = kv_input_row_offsets_dev_buffer^
    _ = sink_weights_dev_buffer^
    _ = kv_block_paged_host_ptr^
    _ = paged_lut_host_ptr^
    _ = output_host_ptr^
    _ = q_host_ptr^
    _ = cache_lengths_host_ptr^
    _ = input_row_offsets_host_ptr^


def main() raises:
    comptime dtype = get_defined_dtype["dtype", DType.bfloat16]()

    comptime head_dim = get_defined_int["head_dim", 128]()
    comptime num_q_heads = get_defined_int["num_q_heads", 32]()
    comptime num_kv_heads = get_defined_int["num_kv_heads", 8]()
    comptime page_size = get_defined_int["page_size", 256]()
    comptime local_window_size = get_defined_int["local_window_size", -1]()
    comptime cross_attention = get_defined_bool["cross_attention", False]()
    comptime sink = get_defined_bool["sink", False]()

    var batch_size = arg_parse("batch_size", 1)
    var use_random_seq_lengths = arg_parse("use_random_seq_lengths", False)
    var seq_len = arg_parse("seq_len", 1)
    var cache_len = arg_parse("cache_len", 1)
    var use_random_cache_lengths = arg_parse("use_random_cache_lengths", False)
    var run_benchmark = arg_parse("run_benchmark", True)

    seed(0)

    var m = Bench()
    try:
        with DeviceContext() as ctx:
            # benchmarking flash attention
            execute_kv_cache_ragged_flash_attention[
                dtype,
                head_dim=head_dim,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                page_size=page_size,
                local_window_size=local_window_size,
                cross_attention=cross_attention,
                sink=sink,
            ](
                ctx,
                m,
                batch_size,
                seq_len,
                use_random_seq_lengths,
                cache_len,
                use_random_cache_lengths,
                run_benchmark,
            )

    except e:
        print("CUDA_ERROR:", e)

    m.dump_report()
