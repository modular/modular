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

from std.math import ceildiv, gcd
from std.random import seed
from std.sys import get_defined_dtype, get_defined_int

from std.benchmark import Bench, BenchConfig, Bencher, BenchId
from std.gpu import WARP_SIZE
from std.gpu.host import DeviceContext
from std.gpu.primitives.grid_controls import PDLLevel, pdl_launch_attributes
from std.testing import assert_almost_equal

from internal_utils import arg_parse
from kv_cache.types import KVCacheStaticParams, KVCollectionT, PagedKVCacheCollection
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
from nn.rope import (
    _rope_k_cache_decode_ragged_kernel,
    _rope_k_cache_ragged,
    _rope_k_cache_ragged_kernel,
)

from std.utils import IndexList


def _rope_k_cache_direct_row_trial[
    freq_dtype: DType,
    collection_t: KVCollectionT,
](
    total_seq_len: Int,
    input_row_offsets: TileTensor[DType.uint32, ...],
    kv_collection: collection_t,
    freqs_cis: TileTensor[freq_dtype, ...],
    layer_idx: UInt32,
    ctx: DeviceContext,
) raises:
    comptime assert (
        input_row_offsets.flat_rank == 1
    ), "input_row_offsets must be rank 1"
    comptime assert freqs_cis.flat_rank == 2, "freqs_cis must be rank 2"
    comptime assert collection_t.CacheType.dtype == DType.bfloat16, (
        "The direct-row K-rope trial only supports BF16 cache storage"
    )
    comptime head_size = Int(collection_t.CacheType.kv_params.head_size)
    comptime assert head_size == 128, (
        "The direct-row K-rope trial only supports 128-column keys"
    )

    var k_cache = kv_collection.get_key_cache(Int(layer_idx))
    var total_rows = total_seq_len * Int(collection_t.CacheType.kv_params.num_heads)
    var batch_size = Int(input_row_offsets.dim(0)) - 1
    var is_decode_uniform = total_seq_len == batch_size
    comptime default_warps_per_block = 2
    comptime default_block_size = default_warps_per_block * WARP_SIZE
    comptime large_row_warps_per_block = 8
    comptime large_row_block_size = large_row_warps_per_block * WARP_SIZE
    comptime min_large_row_blocks_per_sm = 6
    var large_row_grid_dim = ceildiv(total_rows, large_row_warps_per_block * 2)
    var min_large_row_blocks = (
        ctx.default_device_info.sm_count * min_large_row_blocks_per_sm
    )

    if is_decode_uniform:
        if large_row_grid_dim >= min_large_row_blocks:
            comptime kernel = _rope_k_cache_decode_ragged_kernel[
                freq_dtype,
                collection_t.CacheType,
                freqs_cis.LayoutType,
                large_row_block_size,
                large_row_warps_per_block,
            ]
            ctx.enqueue_function[kernel, kernel](
                k_cache,
                freqs_cis,
                total_rows,
                grid_dim=large_row_grid_dim,
                block_dim=large_row_block_size,
                attributes=pdl_launch_attributes(PDLLevel(1)),
            )
        else:
            comptime kernel = _rope_k_cache_decode_ragged_kernel[
                freq_dtype,
                collection_t.CacheType,
                freqs_cis.LayoutType,
                default_block_size,
                default_warps_per_block,
            ]
            ctx.enqueue_function[kernel, kernel](
                k_cache,
                freqs_cis,
                total_rows,
                grid_dim=ceildiv(total_rows, default_warps_per_block * 2),
                block_dim=default_block_size,
                attributes=pdl_launch_attributes(PDLLevel(1)),
            )
    else:
        if large_row_grid_dim >= min_large_row_blocks:
            comptime kernel = _rope_k_cache_ragged_kernel[
                freq_dtype,
                collection_t.CacheType,
                input_row_offsets.LayoutType,
                freqs_cis.LayoutType,
                large_row_block_size,
                large_row_warps_per_block,
            ]
            ctx.enqueue_function[kernel, kernel](
                input_row_offsets,
                k_cache,
                freqs_cis,
                total_rows,
                grid_dim=large_row_grid_dim,
                block_dim=large_row_block_size,
                attributes=pdl_launch_attributes(PDLLevel(1)),
            )
        else:
            comptime kernel = _rope_k_cache_ragged_kernel[
                freq_dtype,
                collection_t.CacheType,
                input_row_offsets.LayoutType,
                freqs_cis.LayoutType,
                default_block_size,
                default_warps_per_block,
            ]
            ctx.enqueue_function[kernel, kernel](
                input_row_offsets,
                k_cache,
                freqs_cis,
                total_rows,
                grid_dim=ceildiv(total_rows, default_warps_per_block * 2),
                block_dim=default_block_size,
                attributes=pdl_launch_attributes(PDLLevel(1)),
                )


def _fill_paged_lut[origin: MutOrigin, //](
    paged_lut_h: UnsafePointer[Scalar[DType.uint32], origin],
    row_page_counts_h: UnsafePointer[Scalar[DType.uint32], _],
    batch_size: Int,
    paged_lut_cols: Int,
    num_pages: Int,
    packed_pages: Bool,
    fragment_pages: Bool,
):
    var paged_lut_host = LayoutTensor[
        DType.uint32, Layout.row_major[2](), origin
    ](
        paged_lut_h,
        RuntimeLayout[Layout.row_major[2]()].row_major(
            IndexList[2](batch_size, paged_lut_cols)
        ),
    )
    assert not (
        packed_pages and fragment_pages
    ), "pack_pages and fragment_pages are mutually exclusive"

    var next_page_id = 0
    var fragment_cursor = 0
    var fragment_stride = 1
    var fragment_offset = 0
    if fragment_pages:
        fragment_stride = num_pages // 2 + 1
        while gcd(fragment_stride, num_pages) != 1:
            fragment_stride += 1
        fragment_offset = num_pages // 3

    for bs in range(batch_size):
        var row_page_count = Int(row_page_counts_h[bs])
        for page_idx in range(paged_lut_cols):
            if packed_pages:
                if page_idx < row_page_count:
                    paged_lut_host[bs, page_idx] = UInt32(next_page_id)
                    next_page_id += 1
                else:
                    paged_lut_host[bs, page_idx] = UInt32(num_pages)
            elif fragment_pages:
                if page_idx < row_page_count:
                    # A deterministic co-prime walk scatters used pages across
                    # the dense pool while keeping the exact layout reproducible.
                    var fragment_page_id = (
                        fragment_offset + fragment_cursor * fragment_stride
                    ) % num_pages
                    fragment_cursor += 1
                    paged_lut_host[bs, page_idx] = UInt32(fragment_page_id)
                else:
                    paged_lut_host[bs, page_idx] = UInt32(num_pages)
            else:
                paged_lut_host[bs, page_idx] = UInt32(
                    bs * paged_lut_cols + page_idx
                )


def _assert_k_cache_match[collection_t: KVCollectionT](
    baseline_kv_collection_host: collection_t,
    trial_kv_collection_host: collection_t,
    cache_lengths_h: UnsafePointer[Scalar[DType.uint32], _],
    layer_idx: Int,
    batch_size: Int,
    seq_len: Int,
    num_kv_heads: Int,
    head_dim: Int,
) raises:
    var baseline_k_cache_host = baseline_kv_collection_host.get_key_cache(layer_idx)
    var trial_k_cache_host = trial_kv_collection_host.get_key_cache(layer_idx)

    for bs_idx in range(batch_size):
        for tok_idx in range(seq_len):
            var post_seq_idx = Int(cache_lengths_h[bs_idx]) + tok_idx
            for head_idx in range(num_kv_heads):
                for dim_idx in range(head_dim):
                    assert_almost_equal(
                        baseline_k_cache_host.load[width=1](
                            bs_idx,
                            head_idx,
                            post_seq_idx,
                            dim_idx,
                        ),
                        trial_k_cache_host.load[width=1](
                            bs_idx,
                            head_idx,
                            post_seq_idx,
                            dim_idx,
                        ),
                        rtol=2e-2,
                        atol=2e-2,
                    )


def bench_gemma3_k_rope_boundary[
    dtype: DType,
    head_dim: Int,
    num_kv_heads: Int,
    page_size: Int,
](
    ctx: DeviceContext,
    mut bench: Bench,
    batch_size: Int,
    seq_len: Int,
    cache_len: Int,
    cache_len_step: Int,
    packed_pages: Bool,
    fragment_pages: Bool,
    paired_fragment_pages: Bool,
) raises:
    comptime layer_idx = 0
    comptime num_layers = 1
    comptime kv_params = KVCacheStaticParams(
        num_heads=UInt(num_kv_heads), head_size=UInt(head_dim)
    )
    comptime CollectionType = PagedKVCacheCollection[dtype, kv_params, page_size]
    comptime kv_block_layout = Layout.row_major[6]()
    comptime cache_lengths_layout = Layout(UNKNOWN_VALUE)

    var total_seq_len = UInt32(batch_size * seq_len)
    var max_cache_len = cache_len + (batch_size - 1) * cache_len_step
    var max_context_length = seq_len + max_cache_len

    var input_row_offsets_h = alloc[Scalar[DType.uint32]](batch_size + 1)
    var cache_lengths_h = alloc[Scalar[DType.uint32]](batch_size)
    var row_page_counts_h = alloc[Scalar[DType.uint32]](batch_size)
    var freqs_h = alloc[Scalar[dtype]](max_context_length * head_dim)
    var paged_lut_cols = 0
    var num_pages = 0

    for i in range(batch_size):
        input_row_offsets_h[i] = UInt32(i * seq_len)
        var row_cache_len = cache_len + i * cache_len_step
        cache_lengths_h[i] = UInt32(row_cache_len)
        var row_page_count = ceildiv(row_cache_len + seq_len, page_size)
        row_page_counts_h[i] = UInt32(row_page_count)
        if row_page_count > paged_lut_cols:
            paged_lut_cols = row_page_count
        if packed_pages:
            num_pages += row_page_count
    input_row_offsets_h[batch_size] = total_seq_len
    if not packed_pages:
        num_pages = batch_size * paged_lut_cols

    assert not (
        paired_fragment_pages and (packed_pages or fragment_pages)
    ), (
        "paired_fragment_pages cannot be combined with pack_pages or "
        "fragment_pages"
    )

    var paged_lut_h = alloc[Scalar[DType.uint32]](batch_size * paged_lut_cols)
    _fill_paged_lut(
        paged_lut_h,
        row_page_counts_h,
        batch_size,
        paged_lut_cols,
        num_pages,
        packed_pages,
        fragment_pages,
    )

    random(
        LayoutTensor[dtype, Layout.row_major[2](), MutAnyOrigin](
            freqs_h,
            RuntimeLayout[Layout.row_major[2]()].row_major(
                IndexList[2](max_context_length, head_dim)
            ),
        )
    )

    var input_row_offsets_d = ctx.enqueue_create_buffer[DType.uint32](
        batch_size + 1
    )
    var cache_lengths_d = ctx.enqueue_create_buffer[DType.uint32](batch_size)
    var freqs_d = ctx.enqueue_create_buffer[dtype](max_context_length * head_dim)
    var paged_lut_d = ctx.enqueue_create_buffer[DType.uint32](
        batch_size * paged_lut_cols
    )

    ctx.enqueue_copy(input_row_offsets_d, input_row_offsets_h)
    ctx.enqueue_copy(cache_lengths_d, cache_lengths_h)
    ctx.enqueue_copy(freqs_d, freqs_h)
    ctx.enqueue_copy(paged_lut_d, paged_lut_h)

    var kv_block_shape = IndexList[6](
        num_pages,
        2,
        num_layers,
        page_size,
        num_kv_heads,
        head_dim,
    )
    var kv_block_size = kv_block_shape.flattened_length()
    var initial_kv_block_h = alloc[Scalar[dtype]](kv_block_size)
    var baseline_kv_block_h = alloc[Scalar[dtype]](kv_block_size)
    var trial_kv_block_h = alloc[Scalar[dtype]](kv_block_size)
    random(
        LayoutTensor[dtype, Layout.row_major[6](), MutAnyOrigin](
            initial_kv_block_h,
            RuntimeLayout[Layout.row_major[6]()].row_major(kv_block_shape),
        )
    )

    var baseline_kv_block_d = ctx.enqueue_create_buffer[dtype](kv_block_size)
    var trial_kv_block_d = ctx.enqueue_create_buffer[dtype](kv_block_size)
    ctx.enqueue_copy(baseline_kv_block_d, initial_kv_block_h)
    ctx.enqueue_copy(trial_kv_block_d, initial_kv_block_h)

    var input_row_offsets_tensor = TileTensor(
        input_row_offsets_d.unsafe_ptr(), row_major(Idx(batch_size + 1))
    )
    var freqs_tensor = TileTensor(
        freqs_d.unsafe_ptr(),
        row_major(Idx(max_context_length), Idx[head_dim]()),
    )

    if paired_fragment_pages:
        var fragmented_paged_lut_h = alloc[Scalar[DType.uint32]](
            batch_size * paged_lut_cols
        )
        _fill_paged_lut(
            fragmented_paged_lut_h,
            row_page_counts_h,
            batch_size,
            paged_lut_cols,
            num_pages,
            False,
            True,
        )
        var fragmented_paged_lut_d = ctx.enqueue_create_buffer[DType.uint32](
            batch_size * paged_lut_cols
        )
        ctx.enqueue_copy(fragmented_paged_lut_d, fragmented_paged_lut_h)

        var fragment_baseline_kv_block_h = alloc[Scalar[dtype]](kv_block_size)
        var fragment_trial_kv_block_h = alloc[Scalar[dtype]](kv_block_size)
        var fragment_baseline_kv_block_d = ctx.enqueue_create_buffer[dtype](
            kv_block_size
        )
        var fragment_trial_kv_block_d = ctx.enqueue_create_buffer[dtype](
            kv_block_size
        )
        ctx.enqueue_copy(fragment_baseline_kv_block_d, initial_kv_block_h)
        ctx.enqueue_copy(fragment_trial_kv_block_d, initial_kv_block_h)

        var dense_baseline_kv_collection = CollectionType(
            LayoutTensor[dtype, kv_block_layout, MutAnyOrigin](
                baseline_kv_block_d.unsafe_ptr(),
                RuntimeLayout[kv_block_layout].row_major(kv_block_shape),
            ),
            LayoutTensor[DType.uint32, cache_lengths_layout, ImmutAnyOrigin](
                cache_lengths_d.unsafe_ptr(),
                RuntimeLayout[cache_lengths_layout].row_major(
                    IndexList[1](batch_size)
                ),
            ),
            LayoutTensor[DType.uint32, Layout.row_major[2](), ImmutAnyOrigin](
                paged_lut_d.unsafe_ptr(),
                RuntimeLayout[Layout.row_major[2]()].row_major(
                    IndexList[2](batch_size, paged_lut_cols)
                ),
            ),
            UInt32(seq_len),
            UInt32(max_context_length),
        )
        var dense_trial_kv_collection = CollectionType(
            LayoutTensor[dtype, kv_block_layout, MutAnyOrigin](
                trial_kv_block_d.unsafe_ptr(),
                RuntimeLayout[kv_block_layout].row_major(kv_block_shape),
            ),
            LayoutTensor[DType.uint32, cache_lengths_layout, ImmutAnyOrigin](
                cache_lengths_d.unsafe_ptr(),
                RuntimeLayout[cache_lengths_layout].row_major(
                    IndexList[1](batch_size)
                ),
            ),
            LayoutTensor[DType.uint32, Layout.row_major[2](), ImmutAnyOrigin](
                paged_lut_d.unsafe_ptr(),
                RuntimeLayout[Layout.row_major[2]()].row_major(
                    IndexList[2](batch_size, paged_lut_cols)
                ),
            ),
            UInt32(seq_len),
            UInt32(max_context_length),
        )
        var fragment_baseline_kv_collection = CollectionType(
            LayoutTensor[dtype, kv_block_layout, MutAnyOrigin](
                fragment_baseline_kv_block_d.unsafe_ptr(),
                RuntimeLayout[kv_block_layout].row_major(kv_block_shape),
            ),
            LayoutTensor[DType.uint32, cache_lengths_layout, ImmutAnyOrigin](
                cache_lengths_d.unsafe_ptr(),
                RuntimeLayout[cache_lengths_layout].row_major(
                    IndexList[1](batch_size)
                ),
            ),
            LayoutTensor[DType.uint32, Layout.row_major[2](), ImmutAnyOrigin](
                fragmented_paged_lut_d.unsafe_ptr(),
                RuntimeLayout[Layout.row_major[2]()].row_major(
                    IndexList[2](batch_size, paged_lut_cols)
                ),
            ),
            UInt32(seq_len),
            UInt32(max_context_length),
        )
        var fragment_trial_kv_collection = CollectionType(
            LayoutTensor[dtype, kv_block_layout, MutAnyOrigin](
                fragment_trial_kv_block_d.unsafe_ptr(),
                RuntimeLayout[kv_block_layout].row_major(kv_block_shape),
            ),
            LayoutTensor[DType.uint32, cache_lengths_layout, ImmutAnyOrigin](
                cache_lengths_d.unsafe_ptr(),
                RuntimeLayout[cache_lengths_layout].row_major(
                    IndexList[1](batch_size)
                ),
            ),
            LayoutTensor[DType.uint32, Layout.row_major[2](), ImmutAnyOrigin](
                fragmented_paged_lut_d.unsafe_ptr(),
                RuntimeLayout[Layout.row_major[2]()].row_major(
                    IndexList[2](batch_size, paged_lut_cols)
                ),
            ),
            UInt32(seq_len),
            UInt32(max_context_length),
        )

        @always_inline
        @__copy_capture(
            dense_baseline_kv_collection,
            input_row_offsets_tensor,
            total_seq_len,
            freqs_tensor,
        )
        @parameter
        def run_dense_baseline(ctx: DeviceContext) raises:
            _rope_k_cache_ragged[target="gpu", interleaved=False](
                Int(total_seq_len),
                input_row_offsets_tensor,
                dense_baseline_kv_collection,
                freqs_tensor,
                UInt32(layer_idx),
                ctx,
            )

        @always_inline
        @__copy_capture(
            fragment_baseline_kv_collection,
            input_row_offsets_tensor,
            total_seq_len,
            freqs_tensor,
        )
        @parameter
        def run_fragment_baseline(ctx: DeviceContext) raises:
            _rope_k_cache_ragged[target="gpu", interleaved=False](
                Int(total_seq_len),
                input_row_offsets_tensor,
                fragment_baseline_kv_collection,
                freqs_tensor,
                UInt32(layer_idx),
                ctx,
            )

        @always_inline
        @__copy_capture(
            dense_trial_kv_collection,
            input_row_offsets_tensor,
            total_seq_len,
            freqs_tensor,
        )
        @parameter
        def run_dense_trial(ctx: DeviceContext) raises:
            _rope_k_cache_direct_row_trial[dtype, CollectionType](
                Int(total_seq_len),
                input_row_offsets_tensor,
                dense_trial_kv_collection,
                freqs_tensor,
                UInt32(layer_idx),
                ctx,
            )

        @always_inline
        @__copy_capture(
            fragment_trial_kv_collection,
            input_row_offsets_tensor,
            total_seq_len,
            freqs_tensor,
        )
        @parameter
        def run_fragment_trial(ctx: DeviceContext) raises:
            _rope_k_cache_direct_row_trial[dtype, CollectionType](
                Int(total_seq_len),
                input_row_offsets_tensor,
                fragment_trial_kv_collection,
                freqs_tensor,
                UInt32(layer_idx),
                ctx,
            )

        # Warm all four variants in one process, then reset them back to the
        # same initial KV state before collecting timings.
        run_dense_baseline(ctx)
        run_fragment_baseline(ctx)
        run_dense_trial(ctx)
        run_fragment_trial(ctx)
        ctx.synchronize()
        ctx.enqueue_copy(baseline_kv_block_d, initial_kv_block_h)
        ctx.enqueue_copy(trial_kv_block_d, initial_kv_block_h)
        ctx.enqueue_copy(fragment_baseline_kv_block_d, initial_kv_block_h)
        ctx.enqueue_copy(fragment_trial_kv_block_d, initial_kv_block_h)
        ctx.synchronize()

        @always_inline
        @parameter
        def dense_baseline_bench(mut bencher: Bencher) raises:
            bencher.iter_custom[run_dense_baseline](ctx)

        @always_inline
        @parameter
        def fragment_baseline_bench(mut bencher: Bencher) raises:
            bencher.iter_custom[run_fragment_baseline](ctx)

        @always_inline
        @parameter
        def dense_trial_bench(mut bencher: Bencher) raises:
            bencher.iter_custom[run_dense_trial](ctx)

        @always_inline
        @parameter
        def fragment_trial_bench(mut bencher: Bencher) raises:
            bencher.iter_custom[run_fragment_trial](ctx)

        bench.bench_function[dense_baseline_bench](
            BenchId(
                "gemma3_k_rope_boundary_dense_baseline",
                input_id=String(
                    dtype,
                    "/bs=",
                    batch_size,
                    "/seq=",
                    seq_len,
                    "/cache=",
                    cache_len,
                    "/step=",
                    cache_len_step,
                ),
            ),
        )
        bench.bench_function[fragment_baseline_bench](
            BenchId(
                "gemma3_k_rope_boundary_fragmented_baseline",
                input_id=String(
                    dtype,
                    "/bs=",
                    batch_size,
                    "/seq=",
                    seq_len,
                    "/cache=",
                    cache_len,
                    "/step=",
                    cache_len_step,
                ),
            ),
        )
        bench.bench_function[dense_trial_bench](
            BenchId(
                "gemma3_k_rope_boundary_dense_trial",
                input_id=String(
                    dtype,
                    "/bs=",
                    batch_size,
                    "/seq=",
                    seq_len,
                    "/cache=",
                    cache_len,
                    "/step=",
                    cache_len_step,
                ),
            ),
        )
        bench.bench_function[fragment_trial_bench](
            BenchId(
                "gemma3_k_rope_boundary_fragmented_trial",
                input_id=String(
                    dtype,
                    "/bs=",
                    batch_size,
                    "/seq=",
                    seq_len,
                    "/cache=",
                    cache_len,
                    "/step=",
                    cache_len_step,
                ),
            ),
        )

        run_dense_baseline(ctx)
        run_fragment_baseline(ctx)
        run_dense_trial(ctx)
        run_fragment_trial(ctx)
        ctx.enqueue_copy(baseline_kv_block_h, baseline_kv_block_d)
        ctx.enqueue_copy(trial_kv_block_h, trial_kv_block_d)
        ctx.enqueue_copy(fragment_baseline_kv_block_h, fragment_baseline_kv_block_d)
        ctx.enqueue_copy(fragment_trial_kv_block_h, fragment_trial_kv_block_d)
        ctx.synchronize()

        var dense_baseline_kv_collection_host = CollectionType(
            LayoutTensor[dtype, kv_block_layout, MutAnyOrigin](
                baseline_kv_block_h,
                RuntimeLayout[kv_block_layout].row_major(kv_block_shape),
            ),
            LayoutTensor[DType.uint32, cache_lengths_layout, ImmutAnyOrigin](
                cache_lengths_h,
                RuntimeLayout[cache_lengths_layout].row_major(
                    IndexList[1](batch_size)
                ),
            ),
            LayoutTensor[DType.uint32, Layout.row_major[2](), ImmutAnyOrigin](
                paged_lut_h,
                RuntimeLayout[Layout.row_major[2]()].row_major(
                    IndexList[2](batch_size, paged_lut_cols)
                ),
            ),
            UInt32(seq_len),
            UInt32(max_context_length),
        )
        var dense_trial_kv_collection_host = CollectionType(
            LayoutTensor[dtype, kv_block_layout, MutAnyOrigin](
                trial_kv_block_h,
                RuntimeLayout[kv_block_layout].row_major(kv_block_shape),
            ),
            LayoutTensor[DType.uint32, cache_lengths_layout, ImmutAnyOrigin](
                cache_lengths_h,
                RuntimeLayout[cache_lengths_layout].row_major(
                    IndexList[1](batch_size)
                ),
            ),
            LayoutTensor[DType.uint32, Layout.row_major[2](), ImmutAnyOrigin](
                paged_lut_h,
                RuntimeLayout[Layout.row_major[2]()].row_major(
                    IndexList[2](batch_size, paged_lut_cols)
                ),
            ),
            UInt32(seq_len),
            UInt32(max_context_length),
        )
        var fragment_baseline_kv_collection_host = CollectionType(
            LayoutTensor[dtype, kv_block_layout, MutAnyOrigin](
                fragment_baseline_kv_block_h,
                RuntimeLayout[kv_block_layout].row_major(kv_block_shape),
            ),
            LayoutTensor[DType.uint32, cache_lengths_layout, ImmutAnyOrigin](
                cache_lengths_h,
                RuntimeLayout[cache_lengths_layout].row_major(
                    IndexList[1](batch_size)
                ),
            ),
            LayoutTensor[DType.uint32, Layout.row_major[2](), ImmutAnyOrigin](
                fragmented_paged_lut_h,
                RuntimeLayout[Layout.row_major[2]()].row_major(
                    IndexList[2](batch_size, paged_lut_cols)
                ),
            ),
            UInt32(seq_len),
            UInt32(max_context_length),
        )
        var fragment_trial_kv_collection_host = CollectionType(
            LayoutTensor[dtype, kv_block_layout, MutAnyOrigin](
                fragment_trial_kv_block_h,
                RuntimeLayout[kv_block_layout].row_major(kv_block_shape),
            ),
            LayoutTensor[DType.uint32, cache_lengths_layout, ImmutAnyOrigin](
                cache_lengths_h,
                RuntimeLayout[cache_lengths_layout].row_major(
                    IndexList[1](batch_size)
                ),
            ),
            LayoutTensor[DType.uint32, Layout.row_major[2](), ImmutAnyOrigin](
                fragmented_paged_lut_h,
                RuntimeLayout[Layout.row_major[2]()].row_major(
                    IndexList[2](batch_size, paged_lut_cols)
                ),
            ),
            UInt32(seq_len),
            UInt32(max_context_length),
        )

        _assert_k_cache_match(
            dense_baseline_kv_collection_host,
            dense_trial_kv_collection_host,
            cache_lengths_h,
            layer_idx,
            batch_size,
            seq_len,
            num_kv_heads,
            head_dim,
        )
        _assert_k_cache_match(
            fragment_baseline_kv_collection_host,
            fragment_trial_kv_collection_host,
            cache_lengths_h,
            layer_idx,
            batch_size,
            seq_len,
            num_kv_heads,
            head_dim,
        )

        _ = input_row_offsets_d
        _ = cache_lengths_d
        _ = freqs_d
        _ = paged_lut_d
        _ = fragmented_paged_lut_d
        _ = baseline_kv_block_d
        _ = trial_kv_block_d
        _ = fragment_baseline_kv_block_d
        _ = fragment_trial_kv_block_d

        input_row_offsets_h.free()
        cache_lengths_h.free()
        row_page_counts_h.free()
        freqs_h.free()
        paged_lut_h.free()
        fragmented_paged_lut_h.free()
        initial_kv_block_h.free()
        baseline_kv_block_h.free()
        trial_kv_block_h.free()
        fragment_baseline_kv_block_h.free()
        fragment_trial_kv_block_h.free()
        return

    var baseline_kv_collection = CollectionType(
        LayoutTensor[dtype, kv_block_layout, MutAnyOrigin](
            baseline_kv_block_d.unsafe_ptr(),
            RuntimeLayout[kv_block_layout].row_major(kv_block_shape),
        ),
        LayoutTensor[DType.uint32, cache_lengths_layout, ImmutAnyOrigin](
            cache_lengths_d.unsafe_ptr(),
            RuntimeLayout[cache_lengths_layout].row_major(
                IndexList[1](batch_size)
            ),
        ),
        LayoutTensor[DType.uint32, Layout.row_major[2](), ImmutAnyOrigin](
            paged_lut_d.unsafe_ptr(),
            RuntimeLayout[Layout.row_major[2]()].row_major(
                IndexList[2](batch_size, paged_lut_cols)
            ),
        ),
        UInt32(seq_len),
        UInt32(max_context_length),
    )
    var trial_kv_collection = CollectionType(
        LayoutTensor[dtype, kv_block_layout, MutAnyOrigin](
            trial_kv_block_d.unsafe_ptr(),
            RuntimeLayout[kv_block_layout].row_major(kv_block_shape),
        ),
        LayoutTensor[DType.uint32, cache_lengths_layout, ImmutAnyOrigin](
            cache_lengths_d.unsafe_ptr(),
            RuntimeLayout[cache_lengths_layout].row_major(
                IndexList[1](batch_size)
            ),
        ),
        LayoutTensor[DType.uint32, Layout.row_major[2](), ImmutAnyOrigin](
            paged_lut_d.unsafe_ptr(),
            RuntimeLayout[Layout.row_major[2]()].row_major(
                IndexList[2](batch_size, paged_lut_cols)
            ),
        ),
        UInt32(seq_len),
        UInt32(max_context_length),
    )

    @always_inline
    @__copy_capture(
        baseline_kv_collection,
        input_row_offsets_tensor,
        total_seq_len,
        freqs_tensor,
    )
    @parameter
    def run_baseline(ctx: DeviceContext) raises:
        _rope_k_cache_ragged[target="gpu", interleaved=False](
            Int(total_seq_len),
            input_row_offsets_tensor,
            baseline_kv_collection,
            freqs_tensor,
            UInt32(layer_idx),
            ctx,
        )

    @always_inline
    @__copy_capture(
        trial_kv_collection,
        input_row_offsets_tensor,
        total_seq_len,
        freqs_tensor,
    )
    @parameter
    def run_trial(ctx: DeviceContext) raises:
        _rope_k_cache_direct_row_trial[dtype, CollectionType](
            Int(total_seq_len),
            input_row_offsets_tensor,
            trial_kv_collection,
            freqs_tensor,
            UInt32(layer_idx),
            ctx,
        )

    @always_inline
    @parameter
    def baseline_bench(mut bencher: Bencher) raises:
        bencher.iter_custom[run_baseline](ctx)

    @always_inline
    @parameter
    def trial_bench(mut bencher: Bencher) raises:
        bencher.iter_custom[run_trial](ctx)

    bench.bench_function[baseline_bench](
        BenchId(
            "gemma3_k_rope_boundary_baseline",
            input_id=String(
                dtype,
                "/bs=",
                batch_size,
                "/seq=",
                seq_len,
                "/cache=",
                cache_len,
                "/step=",
                cache_len_step,
            ),
        ),
    )
    bench.bench_function[trial_bench](
        BenchId(
            "gemma3_k_rope_boundary_trial",
            input_id=String(
                dtype,
                "/bs=",
                batch_size,
                "/seq=",
                seq_len,
                "/cache=",
                cache_len,
                "/step=",
                cache_len_step,
            ),
        ),
    )

    run_baseline(ctx)
    run_trial(ctx)
    ctx.enqueue_copy(baseline_kv_block_h, baseline_kv_block_d)
    ctx.enqueue_copy(trial_kv_block_h, trial_kv_block_d)
    ctx.synchronize()

    var baseline_kv_collection_host = CollectionType(
        LayoutTensor[dtype, kv_block_layout, MutAnyOrigin](
            baseline_kv_block_h,
            RuntimeLayout[kv_block_layout].row_major(kv_block_shape),
        ),
        LayoutTensor[DType.uint32, cache_lengths_layout, ImmutAnyOrigin](
            cache_lengths_h,
            RuntimeLayout[cache_lengths_layout].row_major(
                IndexList[1](batch_size)
            ),
        ),
        LayoutTensor[DType.uint32, Layout.row_major[2](), ImmutAnyOrigin](
            paged_lut_h,
            RuntimeLayout[Layout.row_major[2]()].row_major(
                IndexList[2](batch_size, paged_lut_cols)
            ),
        ),
        UInt32(seq_len),
        UInt32(max_context_length),
    )
    var trial_kv_collection_host = CollectionType(
        LayoutTensor[dtype, kv_block_layout, MutAnyOrigin](
            trial_kv_block_h,
            RuntimeLayout[kv_block_layout].row_major(kv_block_shape),
        ),
        LayoutTensor[DType.uint32, cache_lengths_layout, ImmutAnyOrigin](
            cache_lengths_h,
            RuntimeLayout[cache_lengths_layout].row_major(
                IndexList[1](batch_size)
            ),
        ),
        LayoutTensor[DType.uint32, Layout.row_major[2](), ImmutAnyOrigin](
            paged_lut_h,
            RuntimeLayout[Layout.row_major[2]()].row_major(
                IndexList[2](batch_size, paged_lut_cols)
            ),
        ),
        UInt32(seq_len),
        UInt32(max_context_length),
    )
    _assert_k_cache_match(
        baseline_kv_collection_host,
        trial_kv_collection_host,
        cache_lengths_h,
        layer_idx,
        batch_size,
        seq_len,
        num_kv_heads,
        head_dim,
    )

    _ = input_row_offsets_d
    _ = cache_lengths_d
    _ = freqs_d
    _ = paged_lut_d
    _ = baseline_kv_block_d
    _ = trial_kv_block_d

    input_row_offsets_h.free()
    cache_lengths_h.free()
    row_page_counts_h.free()
    freqs_h.free()
    paged_lut_h.free()
    initial_kv_block_h.free()
    baseline_kv_block_h.free()
    trial_kv_block_h.free()


def main() raises:
    comptime dtype = get_defined_dtype["dtype", DType.bfloat16]()
    comptime head_dim = get_defined_int["head_dim", 128]()
    comptime num_kv_heads = get_defined_int["num_kv_heads", 16]()
    comptime page_size = 128

    var batch_size = arg_parse("batch_size", 64)
    var seq_len = arg_parse("seq_len", 1)
    var cache_len = arg_parse("cache_len", 1024)
    var cache_len_step = arg_parse("cache_len_step", 0)
    var packed_pages = arg_parse("pack_pages", 0) != 0
    var fragment_pages = arg_parse("fragment_pages", 0) != 0
    var paired_fragment_pages = arg_parse("paired_fragment_pages", 0) != 0

    seed(0)

    var bench = Bench(BenchConfig(num_repetitions=1))
    with DeviceContext() as ctx:
        bench_gemma3_k_rope_boundary[
            dtype,
            head_dim,
            num_kv_heads,
            page_size,
        ](
            ctx,
            bench,
            batch_size,
            seq_len,
            cache_len,
            cache_len_step,
            packed_pages,
            fragment_pages,
            paired_fragment_pages,
        )

    bench.dump_report()
