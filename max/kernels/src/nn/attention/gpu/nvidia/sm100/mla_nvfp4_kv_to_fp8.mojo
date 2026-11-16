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
from std.sys import get_defined_int

from max.gpu import WARP_SIZE, block_idx, global_idx, lane_id, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.host.nvidia import GridBlockAttributeLaunchConfig
from max.gpu.primitives.grid_controls import (
    PDL,
    PDLLevel,
    pdl_launch_attributes,
)

from kv_cache.types import (
    KVCacheStaticParams,
    KVCacheT,
    PagedKVCacheCollection,
)
from layout import Idx, TileTensor, row_major
from nn.attention.mha_utils import OptionalPointer
from nn.attention.gpu.nvidia.sm100.mla_sparse_attention import (
    mla_sparse_attention,
)
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
# Slots per remap thread; only the padded destination stride must align.
comptime _REMAP_WIDTH = 4

comptime _BytePtr = MutPointer[Scalar[DType.uint8], MutAnyOrigin]
comptime _U32Ptr = MutPointer[Scalar[DType.uint32], MutAnyOrigin]
comptime _I32Ptr = MutPointer[Scalar[DType.int32], MutAnyOrigin]


@fieldwise_init
struct NVFP4PrefillWorkspaceView(TrivialRegisterPassable):
    """Borrows conversion storage; its owner must outlive every graph replay."""

    var fp8: MutPointer[Scalar[DType.float8_e4m3fn], MutUntrackedOrigin]
    var page_slots: MutPointer[UInt32, MutUntrackedOrigin]
    var indices: MutPointer[Int32, MutUntrackedOrigin]
    var fp8_capacity: Int
    var page_slots_capacity: Int
    var indices_capacity: Int


struct NVFP4PrefillWorkspace(Movable):
    """Owns reusable conversion storage allocated outside graph recording."""

    var fp8: DeviceBuffer[DType.float8_e4m3fn]
    var page_slots: DeviceBuffer[DType.uint32]
    var indices: DeviceBuffer[DType.int32]

    def __init__(
        out self,
        fp8_elems: Int,
        num_blocks: Int,
        indices_elems: Int,
        ctx: DeviceContext,
    ) raises:
        self.fp8 = ctx.enqueue_create_buffer[DType.float8_e4m3fn](fp8_elems)
        self.page_slots = ctx.enqueue_create_buffer[DType.uint32](num_blocks)
        self.indices = ctx.enqueue_create_buffer[DType.int32](indices_elems)

    def view(mut self) -> NVFP4PrefillWorkspaceView:
        return NVFP4PrefillWorkspaceView(
            self.fp8.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
            self.page_slots.unsafe_ptr().unsafe_origin_cast[
                MutUntrackedOrigin
            ](),
            self.indices.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
            len(self.fp8),
            len(self.page_slots),
            len(self.indices),
        )


def nvfp4_prefill_workspace[
    cache_t: KVCacheT,
    //,
    config: MLASparseConfig,
](
    q: TileTensor[address_space=.GENERIC, ...],
    kv_cache: cache_t,
    input_row_offsets: TileTensor[.uint32, address_space=.GENERIC, ...],
    ctx: DeviceContext,
) raises -> NVFP4PrefillWorkspace:
    """Allocates conversion storage before capturing a helper invocation."""
    var batch_size = Int(input_row_offsets.dim(0)) - 1
    var pages = batch_size * ceildiv(
        Int(kv_cache.max_context_length()), cache_t.page_size_
    )
    var block_rows = Int(kv_cache.get_tma_row(Int32(cache_t.page_size_)))
    var num_blocks = (
        kv_cache.num_kv_rows() - cache_t.page_size_
    ) // block_rows + 1
    return NVFP4PrefillWorkspace(
        pages * cache_t.page_size_ * _FP8_ROW_BYTES,
        num_blocks,
        Int(q.dim(0)) * align_up(config.indices_stride // 2, config.B_TOPK),
        ctx,
    )


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
    `nvfp4_mla_prefill_convert_bytes` in `max.nn._graph_ops`.
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
    var slot = block_idx.x
    var b, page = divmod(slot, Int(pages_per_seq))

    with PDL():
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
        var values = cache.blocks_raw_ptr().unsafe_bitcast[
            Scalar[DType.uint8]
        ]()
        var scales = cache.scales_raw_ptr().unsafe_bitcast[
            Scalar[DType.uint8]
        ]()
        var lane = Int(lane_id())
        for r in range(
            Int(thread_idx.x // WARP_SIZE), page_size, _WARPS_PER_BLOCK
        ):
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
    var i = global_idx.x * _REMAP_WIDTH
    var tok, slot = divmod(i, Int(dst_stride))
    var out = SIMD[DType.int32, _REMAP_WIDTH](-1)

    with PDL():
        if tok >= Int(num_tokens):
            return
        if slot < Int(top_k):
            var rows = SIMD[DType.int32, _REMAP_WIDTH](-1)
            if Int(
                src_stride
            ) % _REMAP_WIDTH == 0 and slot + _REMAP_WIDTH <= Int(top_k):
                rows = indices.unsafe_offset(
                    tok * Int(src_stride) + slot
                ).unsafe_load[width=_REMAP_WIDTH, alignment=16]()
            else:
                comptime for j in range(_REMAP_WIDTH):
                    if slot + j < Int(top_k):
                        rows[j] = indices[tok * Int(src_stride) + slot + j]
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
    SinkType: OptionalPointer,
    //,
    config: MLASparseConfig,
    group: Int,
    q_depth: Int,
    causal: Bool = False,
](
    output: TileTensor[mut=True, address_space=.GENERIC, ...],
    q: TileTensor[address_space=.GENERIC, ...],
    kv_cache: cache_t,
    indices: _I32Ptr,
    indices_stride: Int,
    topk_lengths: TileTensor[.uint32, address_space=.GENERIC, ...],
    input_row_offsets: TileTensor[.uint32, address_space=.GENERIC, ...],
    attn_sink_ptr: SinkType,
    scale: Float32,
    ctx: DeviceContext,
    num_partitions_in: Optional[Int] = None,
    workspace: Optional[NVFP4PrefillWorkspaceView] = None,
) raises:
    """Runs sparse attention on an FP8 copy of the batch's context.

    Takes the arguments of the NVFP4 call: `indices` rows are `indices_stride`
    wide, the value rows followed by the scale rows. The caller decides with
    `nvfp4_prefill_converts_once` whether this beats that call. `causal`
    requires rows whose valid entries are packed from the query's causal
    prefix.

    Eager calls allocate scratch on `ctx`'s stream. Capture requires a
    workspace allocated beforehand and retained through graph destruction;
    replay retains the captured batch, query, and allocation geometry.
    Capture requires `num_partitions_in=1`, until split accumulator storage is
    caller-owned. Eager split calls are supported.
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

    var buffers: NVFP4PrefillWorkspaceView
    var owned_workspace = Optional[NVFP4PrefillWorkspace](None)
    if case .Some(supplied) = workspace:
        buffers = supplied
    else:
        var owned = NVFP4PrefillWorkspace(
            num_pages * page_size * _FP8_ROW_BYTES,
            num_blocks,
            num_tokens * dense_stride,
            ctx,
        )
        buffers = owned.view()
        owned_workspace = Optional[NVFP4PrefillWorkspace](owned^)
    if (
        buffers.fp8_capacity < num_pages * page_size * _FP8_ROW_BYTES
        or buffers.page_slots_capacity < num_blocks
        or buffers.indices_capacity < num_tokens * dense_stride
    ):
        raise Error("NVFP4 prefill workspace is too small for this invocation")
    var page_slots_buf = DeviceBuffer[DType.uint32](
        ctx, buffers.page_slots, num_blocks, owning=False
    )
    ctx.enqueue_memset(page_slots_buf, _NO_PAGE)
    var fp8_ptr = buffers.fp8.as_unsafe_any_origin()
    var dense_indices = buffers.indices.as_unsafe_any_origin()
    var fp8_bytes = fp8_ptr.unsafe_bitcast[Scalar[DType.uint8]]()
    var page_slots = buffers.page_slots.as_unsafe_any_origin()
    var row_offsets = input_row_offsets.ptr.unsafe_mut_cast[
        True
    ]().as_unsafe_any_origin()
    ctx.enqueue_function[_convert_pages_kernel[cache_t]](
        GridBlockAttributeLaunchConfig(
            grid_dim=num_pages,
            block_dim=block_threads,
            attributes=pdl_launch_attributes(PDLLevel.ON),
        ),
        kv_cache,
        row_offsets,
        fp8_bytes,
        page_slots,
        Int32(pages_per_seq),
    )
    var remap_threads = num_tokens * dense_stride // _REMAP_WIDTH
    ctx.enqueue_function[_remap_kernel[page_size]](
        GridBlockAttributeLaunchConfig(
            grid_dim=ceildiv(remap_threads, block_threads),
            block_dim=block_threads,
            attributes=pdl_launch_attributes(PDLLevel.ON),
        ),
        indices,
        dense_indices,
        page_slots,
        Int32(indices_stride),
        Int32(top_k),
        Int32(dense_stride),
        Int32(num_tokens),
        Int32(num_blocks),
    )

    # The FP8 kernel addresses rows through the indices alone, so the scratch
    # cache's page table is filler. Its lengths must be the NVFP4 cache's
    # real ones: the causal key count and the bound-only softmax read them.
    var scratch_blocks = TileTensor(
        fp8_ptr,
        row_major(
            Int64(num_pages),
            Idx[1],
            Int64(1),
            Idx[page_size],
            Idx[1],
            Idx[_FP8_ROW_BYTES],
        ),
    )
    var lengths_ptr = kv_cache.cache_lengths_nd().ptr
    comptime ScratchCollection = PagedKVCacheCollection[
        DType.float8_e4m3fn,
        KVCacheStaticParams(num_heads=1, head_size=_FP8_ROW_BYTES, is_mla=True),
        page_size,
        type_of(scratch_blocks).origin,
        type_of(lengths_ptr).origin,
        type_of(lengths_ptr).origin,
        MutUntrackedOrigin,
    ]
    var scratch = ScratchCollection(
        rebind[ScratchCollection.blocks_tt_type](scratch_blocks),
        TileTensor(lengths_ptr, row_major(Int64(batch_size))),
        TileTensor(lengths_ptr, row_major(Int64(1), Int64(1))),
        kv_cache.max_prompt_length(),
        kv_cache.max_context_length(),
    ).get_key_cache(0)

    mla_sparse_attention[
        num_heads=config.num_q_heads, depth=q_depth, causal=causal
    ](
        output,
        q,
        scratch,
        dense_indices,
        dense_stride,
        topk_lengths,
        input_row_offsets,
        attn_sink_ptr,
        scale,
        Int(kv_cache.max_prompt_length()),
        Int(kv_cache.max_context_length()),
        ctx,
        num_partitions_in=num_partitions_in,
    )
    _ = page_slots_buf^
    _ = owned_workspace^
