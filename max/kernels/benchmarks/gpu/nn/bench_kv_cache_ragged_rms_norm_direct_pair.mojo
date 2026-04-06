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
from std.random import seed
from std.sys import get_defined_dtype, get_defined_int
from std.sys.info import size_of

from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from std.gpu import WARP_SIZE
from std.gpu.host import DeviceContext
from std.gpu.primitives.grid_controls import PDLLevel, pdl_launch_attributes
from std.testing import assert_almost_equal
from internal_utils import arg_parse
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import (
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from layout._fillers import random
from nn.kv_cache import rms_norm_kv_cache_ragged_paged
from nn.normalization import rms_norm_gpu_warp_tiling_128

from std.utils import IndexList


def _rms_norm_kv_cache_decode_direct_trial[
    dtype: DType,
    params: KVCacheStaticParams,
    page_size: Int,
    cache_dtype: DType,
    //,
    multiply_before_cast: Bool,
    per_head_norm: Bool,
](
    kv_collection: PagedKVCacheCollection[cache_dtype, params, page_size],
    gamma: TileTensor[dtype, ...],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    layer_idx: UInt32,
    total_seq_len: UInt32,
    input_row_offsets: TileTensor[DType.uint32, ...],
    ctx: DeviceContext,
) raises:
    comptime assert per_head_norm, (
        "The direct KV RMSNorm trial expects per-head norm"
    )
    comptime rms_norm_cols = gamma.static_shape[0]
    comptime head_size = Int(params.head_size)
    comptime assert rms_norm_cols != -1, "Need static gamma shape"

    comptime if dtype != DType.bfloat16 or rms_norm_cols != 128 or head_size != 128:
        rms_norm_kv_cache_ragged_paged[
            dtype=dtype,
            params=params,
            page_size=page_size,
            cache_dtype=cache_dtype,
            target="gpu",
            multiply_before_cast=multiply_before_cast,
            per_head_norm=per_head_norm,
        ](
            kv_collection,
            gamma,
            epsilon,
            weight_offset,
            layer_idx,
            total_seq_len,
            input_row_offsets,
            ctx,
        )
    else:
        var batch_size = Int(input_row_offsets.dim(0)) - 1
        var is_decode_uniform = Int(total_seq_len) == batch_size

        if not is_decode_uniform:
            rms_norm_kv_cache_ragged_paged[
                dtype=dtype,
                params=params,
                page_size=page_size,
                cache_dtype=cache_dtype,
                target="gpu",
                multiply_before_cast=multiply_before_cast,
                per_head_norm=per_head_norm,
            ](
                kv_collection,
                gamma,
                epsilon,
                weight_offset,
                layer_idx,
                total_seq_len,
                input_row_offsets,
                ctx,
            )
            return

        var k_cache = kv_collection.get_key_cache(Int(layer_idx))
        var num_rows = Int(total_seq_len) * Int(params.num_heads)
        comptime simd_width = 8
        comptime default_warps_per_block = 2
        comptime large_row_warps_per_block = 8
        comptime default_block_size = default_warps_per_block * WARP_SIZE
        comptime large_row_block_size = large_row_warps_per_block * WARP_SIZE
        comptime min_large_row_blocks_per_sm = 5
        var min_large_row_blocks = (
            ctx.default_device_info.sm_count * min_large_row_blocks_per_sm
        )
        var large_row_grid_dim = ceildiv(
            num_rows, large_row_warps_per_block * 2
        )

        @always_inline
        @parameter
        @__copy_capture(k_cache)
        def direct_input_fn_2d[width: Int](
            row: Int, col: Int
        ) -> SIMD[dtype, width]:
            var batch_idx = row // Int(params.num_heads)
            var head_idx = row % Int(params.num_heads)
            var cache_token_idx = Int(k_cache.cache_length(batch_idx))
            return k_cache.load[width=width](
                bs=batch_idx,
                tok_idx=cache_token_idx,
                head_idx=head_idx,
                head_dim_idx=col,
            ).cast[dtype]()

        @always_inline
        @parameter
        @__copy_capture(k_cache)
        def direct_output_fn_2d[width: Int, alignment: Int](
            row: Int, col: Int, val: SIMD[dtype, width]
        ) -> None:
            var batch_idx = row // Int(params.num_heads)
            var head_idx = row % Int(params.num_heads)
            var cache_token_idx = Int(k_cache.cache_length(batch_idx))
            k_cache.store(
                bs=batch_idx,
                tok_idx=cache_token_idx,
                head_idx=head_idx,
                head_dim_idx=col,
                val=val.cast[cache_dtype](),
            )

        if large_row_grid_dim >= min_large_row_blocks:
            comptime kernel = rms_norm_gpu_warp_tiling_128[
                mut=gamma.mut,
                LayoutType=gamma.LayoutType,
                origin=gamma.origin,
                simd_width,
                large_row_warps_per_block,
                direct_input_fn_2d,
                direct_output_fn_2d,
                multiply_before_cast=multiply_before_cast,
            ]
            ctx.enqueue_function[kernel, kernel](
                gamma,
                epsilon,
                weight_offset,
                num_rows,
                rms_norm_cols,
                grid_dim=large_row_grid_dim,
                block_dim=large_row_block_size,
                attributes=pdl_launch_attributes(PDLLevel(1)),
            )
        else:
            comptime kernel = rms_norm_gpu_warp_tiling_128[
                mut=gamma.mut,
                LayoutType=gamma.LayoutType,
                origin=gamma.origin,
                simd_width,
                default_warps_per_block,
                direct_input_fn_2d,
                direct_output_fn_2d,
                multiply_before_cast=multiply_before_cast,
            ]
            ctx.enqueue_function[kernel, kernel](
                gamma,
                epsilon,
                weight_offset,
                num_rows,
                rms_norm_cols,
                grid_dim=ceildiv(num_rows, default_warps_per_block * 2),
                block_dim=default_block_size,
                attributes=pdl_launch_attributes(PDLLevel(1)),
            )


def execute_kv_cache_ragged_rms_norm_direct_pair[
    dtype: DType, head_dim: Int, num_kv_heads: Int, page_size: Int
](
    ctx: DeviceContext,
    mut m: Bench,
    batch_size: Int,
    seq_len: Int,
    cache_len: Int,
    cache_len_step: Int,
) raises:
    comptime num_layers = 1
    comptime layer_idx = 0
    comptime kv_params = KVCacheStaticParams(
        num_heads=UInt(num_kv_heads), head_size=UInt(head_dim)
    )
    comptime CollectionType = PagedKVCacheCollection[dtype, kv_params, page_size]

    var total_seq_len = UInt32(batch_size * seq_len)
    var max_cache_len = cache_len + (batch_size - 1) * cache_len_step
    var max_context_length = seq_len + max_cache_len
    var paged_lut_cols = ceildiv(max_context_length, page_size)
    var num_pages = batch_size * paged_lut_cols

    var input_row_offsets_host_ptr = alloc[Scalar[DType.uint32]](batch_size + 1)
    var cache_lengths_host_ptr = alloc[Scalar[DType.uint32]](batch_size)
    var gamma_host_ptr = alloc[Scalar[dtype]](head_dim)

    for i in range(batch_size):
        input_row_offsets_host_ptr[i] = UInt32(i * seq_len)
        cache_lengths_host_ptr[i] = UInt32(cache_len + i * cache_len_step)
    input_row_offsets_host_ptr[batch_size] = total_seq_len

    for i in range(head_dim):
        gamma_host_ptr[i] = (
            Float64(i + head_dim) / Float64(head_dim)
        ).cast[dtype]()

    var input_row_offsets_dev_buffer = ctx.enqueue_create_buffer[DType.uint32](
        batch_size + 1
    )
    ctx.enqueue_copy(input_row_offsets_dev_buffer, input_row_offsets_host_ptr)

    var cache_lengths_dev_buffer = ctx.enqueue_create_buffer[DType.uint32](
        batch_size
    )
    ctx.enqueue_copy(cache_lengths_dev_buffer, cache_lengths_host_ptr)

    var gamma_dev_buffer = ctx.enqueue_create_buffer[dtype](head_dim)
    ctx.enqueue_copy(gamma_dev_buffer, gamma_host_ptr)

    var paged_lut_size = batch_size * paged_lut_cols
    var paged_lut_host_ptr = alloc[Scalar[DType.uint32]](paged_lut_size)
    comptime paged_lut_layout = Layout.row_major[2]()
    var paged_lut_host = LayoutTensor[
        DType.uint32, paged_lut_layout, MutAnyOrigin
    ](
        paged_lut_host_ptr,
        RuntimeLayout[paged_lut_layout].row_major(
            IndexList[2](batch_size, paged_lut_cols)
        ),
    )

    for bs in range(batch_size):
        for page_idx in range(paged_lut_cols):
            paged_lut_host[bs, page_idx] = UInt32(bs * paged_lut_cols + page_idx)

    var paged_lut_dev_buffer = ctx.enqueue_create_buffer[DType.uint32](
        paged_lut_size
    )
    ctx.enqueue_copy(paged_lut_dev_buffer, paged_lut_host_ptr)

    var kv_block_shape = IndexList[6](
        num_pages,
        2,
        num_layers,
        page_size,
        num_kv_heads,
        head_dim,
    )
    var kv_block_size = kv_block_shape.flattened_length()
    var initial_kv_block_host_ptr = alloc[Scalar[dtype]](kv_block_size)
    var baseline_kv_block_host_ptr = alloc[Scalar[dtype]](kv_block_size)
    var direct_kv_block_host_ptr = alloc[Scalar[dtype]](kv_block_size)
    random(
        LayoutTensor[dtype, Layout.row_major[6](), MutAnyOrigin](
            initial_kv_block_host_ptr,
            RuntimeLayout[Layout.row_major[6]()].row_major(kv_block_shape),
        )
    )
    var baseline_kv_block_dev_buffer = ctx.enqueue_create_buffer[dtype](
        kv_block_size
    )
    var direct_kv_block_dev_buffer = ctx.enqueue_create_buffer[dtype](
        kv_block_size
    )
    ctx.enqueue_copy(baseline_kv_block_dev_buffer, initial_kv_block_host_ptr)
    ctx.enqueue_copy(direct_kv_block_dev_buffer, initial_kv_block_host_ptr)

    comptime kv_block_layout = Layout.row_major[6]()
    comptime cache_lengths_layout = Layout(UNKNOWN_VALUE)
    var cache_lengths_layout_tensor = LayoutTensor[
        DType.uint32, cache_lengths_layout, ImmutAnyOrigin
    ](
        cache_lengths_dev_buffer.unsafe_ptr(),
        RuntimeLayout[cache_lengths_layout].row_major(IndexList[1](batch_size)),
    )

    var paged_lut_layout_tensor = LayoutTensor[
        DType.uint32, paged_lut_layout, ImmutAnyOrigin
    ](
        paged_lut_dev_buffer.unsafe_ptr(),
        RuntimeLayout[paged_lut_layout].row_major(
            IndexList[2](batch_size, paged_lut_cols)
        ),
    )

    var baseline_kv_collection = CollectionType(
        LayoutTensor[dtype, kv_block_layout, MutAnyOrigin](
            baseline_kv_block_dev_buffer.unsafe_ptr(),
            RuntimeLayout[kv_block_layout].row_major(kv_block_shape),
        ),
        cache_lengths_layout_tensor,
        paged_lut_layout_tensor,
        UInt32(seq_len),
        UInt32(max_context_length),
    )
    var direct_kv_collection = CollectionType(
        LayoutTensor[dtype, kv_block_layout, MutAnyOrigin](
            direct_kv_block_dev_buffer.unsafe_ptr(),
            RuntimeLayout[kv_block_layout].row_major(kv_block_shape),
        ),
        cache_lengths_layout_tensor,
        paged_lut_layout_tensor,
        UInt32(seq_len),
        UInt32(max_context_length),
    )

    var gamma_tensor = TileTensor(
        gamma_dev_buffer.unsafe_ptr(), row_major(Idx[head_dim]())
    )
    var input_row_offsets_tensor = TileTensor(
        input_row_offsets_dev_buffer.unsafe_ptr(),
        row_major(Idx(batch_size + 1)),
    )
    var epsilon = Scalar[dtype](1e-6)
    var weight_offset = Scalar[dtype](1.0)
    var num_bytes = (
        3 * Int(total_seq_len) * num_kv_heads * head_dim * size_of[dtype]()
    )

    @always_inline
    @__copy_capture(
        baseline_kv_collection,
        gamma_tensor,
        epsilon,
        weight_offset,
        input_row_offsets_tensor,
        total_seq_len,
    )
    @parameter
    def run_baseline(ctx: DeviceContext) raises:
        rms_norm_kv_cache_ragged_paged[
            target="gpu",
            multiply_before_cast=True,
            per_head_norm=True,
        ](
            baseline_kv_collection,
            gamma_tensor,
            epsilon,
            weight_offset,
            UInt32(layer_idx),
            total_seq_len,
            input_row_offsets_tensor,
            ctx,
        )

    @always_inline
    @__copy_capture(
        direct_kv_collection,
        gamma_tensor,
        epsilon,
        weight_offset,
        input_row_offsets_tensor,
        total_seq_len,
    )
    @parameter
    def run_direct_trial(ctx: DeviceContext) raises:
        _rms_norm_kv_cache_decode_direct_trial[
            dtype=dtype,
            params=kv_params,
            page_size=page_size,
            cache_dtype=dtype,
            multiply_before_cast=True,
            per_head_norm=True,
        ](
            direct_kv_collection,
            gamma_tensor,
            epsilon,
            weight_offset,
            UInt32(layer_idx),
            total_seq_len,
            input_row_offsets_tensor,
            ctx,
        )

    @always_inline
    @parameter
    def baseline_bench(mut bencher: Bencher) raises:
        bencher.iter_custom[run_baseline](ctx)

    @always_inline
    @parameter
    def direct_trial_bench(mut bencher: Bencher) raises:
        bencher.iter_custom[run_direct_trial](ctx)

    var input_id = String(
        dtype,
        "/bs=",
        batch_size,
        "/seq=",
        seq_len,
        "/cache=",
        cache_len,
        "/step=",
        cache_len_step,
    )

    m.bench_function[baseline_bench](
        BenchId(
            "kv_cache_ragged_rms_norm_baseline",
            input_id=input_id,
        ),
        [ThroughputMeasure(BenchMetric.bytes, num_bytes)],
    )
    m.bench_function[direct_trial_bench](
        BenchId(
            "kv_cache_ragged_rms_norm_direct_trial",
            input_id=input_id,
        ),
        [ThroughputMeasure(BenchMetric.bytes, num_bytes)],
    )

    run_baseline(ctx)
    run_direct_trial(ctx)
    ctx.enqueue_copy(baseline_kv_block_host_ptr, baseline_kv_block_dev_buffer)
    ctx.enqueue_copy(direct_kv_block_host_ptr, direct_kv_block_dev_buffer)
    ctx.synchronize()

    var baseline_kv_collection_host = CollectionType(
        LayoutTensor[dtype, kv_block_layout, MutAnyOrigin](
            baseline_kv_block_host_ptr,
            RuntimeLayout[kv_block_layout].row_major(kv_block_shape),
        ),
        LayoutTensor[DType.uint32, cache_lengths_layout, ImmutAnyOrigin](
            cache_lengths_host_ptr,
            RuntimeLayout[cache_lengths_layout].row_major(
                IndexList[1](batch_size)
            ),
        ),
        LayoutTensor[DType.uint32, paged_lut_layout, ImmutAnyOrigin](
            paged_lut_host_ptr,
            RuntimeLayout[paged_lut_layout].row_major(
                IndexList[2](batch_size, paged_lut_cols)
            ),
        ),
        UInt32(seq_len),
        UInt32(max_context_length),
    )
    var direct_kv_collection_host = CollectionType(
        LayoutTensor[dtype, kv_block_layout, MutAnyOrigin](
            direct_kv_block_host_ptr,
            RuntimeLayout[kv_block_layout].row_major(kv_block_shape),
        ),
        LayoutTensor[DType.uint32, cache_lengths_layout, ImmutAnyOrigin](
            cache_lengths_host_ptr,
            RuntimeLayout[cache_lengths_layout].row_major(
                IndexList[1](batch_size)
            ),
        ),
        LayoutTensor[DType.uint32, paged_lut_layout, ImmutAnyOrigin](
            paged_lut_host_ptr,
            RuntimeLayout[paged_lut_layout].row_major(
                IndexList[2](batch_size, paged_lut_cols)
            ),
        ),
        UInt32(seq_len),
        UInt32(max_context_length),
    )

    var baseline_k_cache_host = baseline_kv_collection_host.get_key_cache(layer_idx)
    var direct_k_cache_host = direct_kv_collection_host.get_key_cache(layer_idx)

    for bs_idx in range(batch_size):
        for tok_idx in range(seq_len):
            var cache_tok_idx = Int(cache_lengths_host_ptr[bs_idx]) + tok_idx
            for head_idx in range(num_kv_heads):
                for dim_idx in range(head_dim):
                    assert_almost_equal(
                        baseline_k_cache_host.load[width=1](
                            bs=bs_idx,
                            tok_idx=cache_tok_idx,
                            head_idx=head_idx,
                            head_dim_idx=dim_idx,
                        ),
                        direct_k_cache_host.load[width=1](
                            bs=bs_idx,
                            tok_idx=cache_tok_idx,
                            head_idx=head_idx,
                            head_dim_idx=dim_idx,
                        ),
                        rtol=2e-2,
                        atol=2e-2,
                    )

    input_row_offsets_host_ptr.free()
    cache_lengths_host_ptr.free()
    gamma_host_ptr.free()
    paged_lut_host_ptr.free()
    initial_kv_block_host_ptr.free()
    baseline_kv_block_host_ptr.free()
    direct_kv_block_host_ptr.free()

    _ = input_row_offsets_dev_buffer^
    _ = cache_lengths_dev_buffer^
    _ = gamma_dev_buffer^
    _ = paged_lut_dev_buffer^
    _ = baseline_kv_block_dev_buffer^
    _ = direct_kv_block_dev_buffer^


def main() raises:
    comptime dtype = get_defined_dtype["dtype", DType.bfloat16]()
    comptime head_dim = get_defined_int["head_dim", 128]()
    comptime num_kv_heads = get_defined_int["num_kv_heads", 16]()
    comptime page_size = get_defined_int["page_size", 512]()

    var batch_size = arg_parse("batch_size", 1)
    var seq_len = arg_parse("seq_len", 1)
    var cache_len = arg_parse("cache_len", 0)
    var cache_len_step = arg_parse("cache_len_step", 0)

    seed(0)

    var m = Bench()
    with DeviceContext() as ctx:
        execute_kv_cache_ragged_rms_norm_direct_pair[
            dtype,
            head_dim,
            num_kv_heads,
            page_size,
        ](ctx, m, batch_size, seq_len, cache_len, cache_len_step)

    m.dump_report()
