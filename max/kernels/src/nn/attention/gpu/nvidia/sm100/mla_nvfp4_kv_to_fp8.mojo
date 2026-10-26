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

"""Sparse MLA prefill over an NVFP4 cache through a once-converted FP8 copy.

Target family: Blackwell sm_100 (the conversion kernels are plain SIMT).

`mla_prefill_sparse_qkv_fp8` on an NVFP4 cache makes every CTA (one query
token) expand its own `top_k` K rows into an FP8 tile in shared memory, so a
row selected by `n` queries is expanded `n` times. At 8 heads per GPU that
costs ~36% over the FP8-cache kernel. When the batch's context is small next
to its query count, it is cheaper to expand every context row once, into an FP8
scratch, and run the unchanged FP8 kernel on that.

A row is `352 B` = 256 B packed e2m1 latent + 64 B FP8 rope + (in the separate
scale pool) 32 B E4M3 scales, one per 16 latent elements. The FP8 row is
`576 B`. The expansion is the helper the NVFP4 kernel uses
(`nvfp4_dequant_group`), so the result is bit-identical to that kernel's.

The scratch holds whole pages: batch entry `b`'s page `p` goes to scratch page
`b * pages_per_seq + p`. Converting whole pages keeps a block that two entries
share (a cached prefix) complete under either entry's copy. A table from
physical block to scratch page then rewrites the indexer's value rows
(`block * page_size + offset`) into scratch rows; the scale half of the index
row is not read, since the conversion takes each row's scales through the
cache's own scale page table.
"""

from std.math import align_up, ceildiv
from std.memory import bitcast

from max.gpu import WARP_SIZE, block_idx, global_idx, lane_id, thread_idx
from max.gpu.host import DeviceContext

from kv_cache.types import (
    KVCacheStaticParams,
    KVCacheT,
    PagedKVCacheCollection,
)
from layout import Idx, TileTensor, row_major
from nn.attention.gpu.nvidia.sm100.mla_prefill_sparse_qkv_fp8 import (
    mla_prefill_sparse_qkv_fp8,
)
from nn.attention.gpu.nvidia.sm100.mla_prefill_sparse_utils import (
    MLASparseConfig,
)
from nn.attention.gpu.nvidia.sm100.nvfp4_kv_dequant import (
    NVFP4_MLA_ROPE_DIM,
    nvfp4_dequant_group,
    nvfp4_mla_row_bytes,
    nvfp4_mla_scale_row_bytes,
)

comptime _LATENT_DIM = 512
comptime _FP8_ROW_BYTES = _LATENT_DIM + NVFP4_MLA_ROPE_DIM
comptime _PACKED_ROW_BYTES = nvfp4_mla_row_bytes[_FP8_ROW_BYTES]()
comptime _SF_ROW_BYTES = nvfp4_mla_scale_row_bytes[_FP8_ROW_BYTES]()
comptime _WARPS_PER_BLOCK = 8
comptime _NO_PAGE = UInt32(0xFFFFFFFF)
# Slots per remap thread; `top_k` and both strides are multiples of it.
comptime _REMAP_WIDTH = 4

comptime _BytePtr = MutPointer[Scalar[DType.uint8], MutAnyOrigin]
comptime _U32Ptr = MutPointer[Scalar[DType.uint32], MutAnyOrigin]
comptime _I32Ptr = MutPointer[Scalar[DType.int32], MutAnyOrigin]


def nvfp4_prefill_scratch_rows[
    page_size: Int
](batch_size: Int, max_context_length: Int) -> Int:
    """Rows of the FP8 scratch a batch converts into.

    Every entry gets the pages of the longest context, so the scratch is
    addressed without a prefix sum; `max_context_length` (the cache's
    `max_cache_length`) already counts the step's own tokens.
    """
    return batch_size * align_up(max_context_length, page_size)


def nvfp4_prefill_converts_once(
    scratch_rows: Int, query_tokens: Int, ratio: Int, max_rows: Int
) -> Bool:
    """Whether converting the batch's context once beats the NVFP4 kernel.

    Converting costs per context row and saves per query token, so it pays
    off while rows per query token stay under `ratio` (0 turns it off).
    `max_rows` caps the scratch, which the memory planner reserves through
    `nvfp4_mla_prefill_convert_bytes` in `max.nn.kernels`.
    """
    return (
        ratio > 0
        and 0 < scratch_rows
        and scratch_rows <= max_rows
        and scratch_rows <= ratio * query_tokens
    )


@always_inline
def _convert_row(
    packed_row: _BytePtr,
    scale_row: _BytePtr,
    fp8_row: _BytePtr,
    lane: Int,
):
    """Expands one row; lane `g` owns scale group `g` (16 latent elements).

    The 32 lanes read 256 contiguous packed bytes and write 512 contiguous
    FP8 bytes; lanes 0-3 then copy the 64 rope bytes unchanged.
    """
    var packed = packed_row.unsafe_offset(lane * 8).unsafe_load[
        width=8, alignment=8
    ]()
    var scale = bitcast[DType.float8_e4m3fn, 1](
        SIMD[DType.uint8, 1](scale_row[lane])
    ).cast[DType.float32]()[0]
    fp8_row.unsafe_offset(lane * 16).unsafe_bitcast[
        Scalar[DType.uint32]
    ]().unsafe_store[width=4, alignment=16](nvfp4_dequant_group(packed, scale))
    if lane < NVFP4_MLA_ROPE_DIM // 16:
        fp8_row.unsafe_offset(_LATENT_DIM + lane * 16).unsafe_bitcast[
            Scalar[DType.uint32]
        ]().unsafe_store[width=4, alignment=16](
            packed_row.unsafe_offset(_LATENT_DIM // 2 + lane * 16)
            .unsafe_bitcast[Scalar[DType.uint32]]()
            .unsafe_load[width=4, alignment=16]()
        )


def _convert_pages_kernel[
    cache_t: KVCacheT
](
    cache: cache_t,
    input_row_offsets: _U32Ptr,
    fp8: _BytePtr,
    page_slots: _U32Ptr,
    pages_per_seq: Int32,
):
    """Converts scratch page `block_idx.x` and records its physical block."""
    comptime page_size = cache_t.page_size_
    var slot = Int(block_idx.x)
    var b, page = divmod(slot, Int(pages_per_seq))
    var context = cache.cache_length(b) + Int(
        input_row_offsets[b + 1] - input_row_offsets[b]
    )
    if page * page_size >= context:
        return
    var first_tok = UInt32(page * page_size)
    var value_base = Int(cache.row_idx(UInt32(b), first_tok))
    var scale_base = Int(cache.scale_row_idx(UInt32(b), first_tok))
    if thread_idx.x == 0:
        # `get_tma_row(page_size)` is the distance between blocks in rows.
        var block_rows = Int(cache.get_tma_row(Int32(page_size)))
        page_slots[value_base // block_rows] = UInt32(slot)
    var values = cache.blocks_raw_ptr().unsafe_bitcast[Scalar[DType.uint8]]()
    var scales = cache.scales_raw_ptr().unsafe_bitcast[Scalar[DType.uint8]]()
    var lane = Int(lane_id())
    for r in range(Int(thread_idx.x // WARP_SIZE), page_size, _WARPS_PER_BLOCK):
        _convert_row(
            values.unsafe_offset((value_base + r) * _PACKED_ROW_BYTES),
            scales.unsafe_offset((scale_base + r) * _SF_ROW_BYTES),
            fp8.unsafe_offset((slot * page_size + r) * _FP8_ROW_BYTES),
            lane,
        )


def _remap_kernel[
    page_size: Int
](
    indices: _I32Ptr,
    dense: _I32Ptr,
    page_slots: _U32Ptr,
    src_stride: Int32,
    top_k: Int32,
    dst_stride: Int32,
    num_tokens: Int32,
    num_blocks: Int32,
):
    """Rewrites `_REMAP_WIDTH` value rows of one token into scratch rows.

    Unused slots, slots past `top_k`, and rows whose block was not converted
    become `-1`, which the FP8 kernel zero-fills.
    """
    var i = Int(global_idx.x) * _REMAP_WIDTH
    var tok, slot = divmod(i, Int(dst_stride))
    if tok >= Int(num_tokens):
        return
    var out = SIMD[DType.int32, _REMAP_WIDTH](-1)
    if slot < Int(top_k):
        var rows = indices.unsafe_offset(
            tok * Int(src_stride) + slot
        ).unsafe_load[width=_REMAP_WIDTH, alignment=16]()
        comptime for j in range(_REMAP_WIDTH):
            var row = Int(rows[j])
            if row >= 0:
                var block, offset = divmod(row, page_size)
                if block < Int(num_blocks):
                    var page = page_slots[block]
                    if page != _NO_PAGE:
                        out[j] = Int32(Int(page) * page_size + offset)
    dense.unsafe_offset(i).unsafe_store[alignment=16](out)


def mla_prefill_sparse_nvfp4_via_fp8[
    cache_t: KVCacheT,
    //,
    config: MLASparseConfig,
    group: Int,
    q_depth: Int,
](
    output: TileTensor[mut=True, address_space=.GENERIC, ...],
    q: TileTensor[address_space=.GENERIC, ...],
    kv_cache: cache_t,
    indices: _I32Ptr,
    indices_stride: Int,
    topk_lengths: TileTensor[.uint32, address_space=.GENERIC, ...],
    input_row_offsets: TileTensor[.uint32, address_space=.GENERIC, ...],
    attn_sink_ptr: Optional[ImmPointer[Float32, ImmutAnyOrigin]],
    scale: Float32,
    ctx: DeviceContext,
) raises:
    """Runs `mla_prefill_sparse_qkv_fp8` on an FP8 copy of the batch's context.

    Takes the arguments of the NVFP4 call: `indices` rows are `indices_stride`
    wide, the value rows followed by the scale rows. The caller decides with
    `nvfp4_prefill_converts_once` whether this beats that call.

    The scratch buffers are allocated on `ctx`'s stream for this call only, as
    the other prefill temporaries are. Sparse prefill never runs under device
    graph capture, whose replays are decode or verify steps.
    """
    comptime page_size = cache_t.page_size_
    comptime top_k = config.indices_stride // 2
    comptime block_threads = (
        _WARPS_PER_BLOCK * ctx.default_device_info.warp_size
    )
    comptime assert (
        cache_t.dtype == DType.uint8 and cache_t.quantization_enabled
    ), "the convert-once prefill reads a packed NVFP4 cache"
    comptime assert (
        q_depth == _FP8_ROW_BYTES
    ), "a GLM/DSv3.2 MLA row is 576 wide"
    comptime assert (
        top_k % _REMAP_WIDTH == 0
    ), "the index remap reads 4 slots at a time"
    debug_assert(
        indices_stride == config.indices_stride,
        "indices_stride must match the config",
    )

    var num_tokens = Int(q.dim(0))
    var batch_size = Int(input_row_offsets.dim(0)) - 1
    var pages_per_seq = ceildiv(Int(kv_cache.max_context_length()), page_size)
    var num_pages = batch_size * pages_per_seq
    var block_rows = Int(kv_cache.get_tma_row(Int32(page_size)))
    var num_blocks = (kv_cache.num_kv_rows() - page_size) // block_rows + 1
    # The FP8 kernel reads whole `B_TOPK` tiles of a row, so pad the row to a
    # tile with `-1` rather than let the last tile read the next row.
    var dense_stride = align_up(top_k, config.B_TOPK)

    var fp8_buf = ctx.enqueue_create_buffer[DType.float8_e4m3fn](
        num_pages * page_size * _FP8_ROW_BYTES
    )
    var page_slots_buf = ctx.enqueue_create_buffer[DType.uint32](num_blocks)
    var dense_buf = ctx.enqueue_create_buffer[DType.int32](
        num_tokens * dense_stride
    )
    page_slots_buf.enqueue_fill(_NO_PAGE)

    var fp8_bytes = fp8_buf.unsafe_ptr().unsafe_bitcast[Scalar[DType.uint8]]()
    var page_slots = page_slots_buf.unsafe_ptr()
    var row_offsets = input_row_offsets.ptr.unsafe_mut_cast[
        True
    ]().as_unsafe_any_origin()
    ctx.enqueue_function[_convert_pages_kernel[cache_t]](
        kv_cache,
        row_offsets,
        fp8_bytes,
        page_slots,
        Int32(pages_per_seq),
        grid_dim=num_pages,
        block_dim=block_threads,
    )
    var remap_threads = num_tokens * dense_stride // _REMAP_WIDTH
    ctx.enqueue_function[_remap_kernel[page_size]](
        indices,
        dense_buf.unsafe_ptr(),
        page_slots,
        Int32(indices_stride),
        Int32(top_k),
        Int32(dense_stride),
        Int32(num_tokens),
        Int32(num_blocks),
        grid_dim=ceildiv(remap_threads, block_threads),
        block_dim=block_threads,
    )

    # The FP8 kernel addresses rows through the indices alone, so the scratch
    # cache borrows the NVFP4 cache's lengths and page table as filler.
    comptime ScratchCollection = PagedKVCacheCollection[
        DType.float8_e4m3fn,
        KVCacheStaticParams(num_heads=1, head_size=_FP8_ROW_BYTES, is_mla=True),
        page_size,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    var scratch_blocks = TileTensor(
        fp8_buf.unsafe_ptr(),
        row_major(
            Int64(num_pages),
            Idx[1],
            Int64(1),
            Idx[page_size],
            Idx[1],
            Idx[_FP8_ROW_BYTES],
        ),
    )
    var lengths_ptr = kv_cache.cache_lengths_nd().ptr.as_unsafe_any_origin()
    var scratch = ScratchCollection(
        rebind[ScratchCollection.blocks_tt_type](scratch_blocks),
        TileTensor(lengths_ptr, row_major(Int64(batch_size))),
        TileTensor(lengths_ptr, row_major(Int64(1), Int64(1))),
        kv_cache.max_prompt_length(),
        kv_cache.max_context_length(),
    ).get_key_cache(0)

    mla_prefill_sparse_qkv_fp8[
        config=config,
        group=group,
        q_depth=q_depth,
        scale_block_size=0,
    ](
        output,
        q,
        scratch,
        TileTensor(
            dense_buf.unsafe_ptr().unsafe_bitcast[UInt32](),
            row_major(num_tokens * dense_stride),
        ),
        topk_lengths,
        attn_sink_ptr,
        scale,
        Int32(dense_stride),
        ctx,
    )
    _ = fp8_buf^
    _ = page_slots_buf^
    _ = dense_buf^
