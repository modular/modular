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
"""SM100 (B200) ragged-paged MHA with FP8 KV cache + per-block fp32 scales.

Target hardware family: NVIDIA SM100 (Blackwell, B200).

Implements the correctness MVP of the SM100 MHA fp8-KV decode path.
Architecture:

    bf16 Q
       |
       v
    +-------------------------------+
    | mha_fp8_kv_decode_ragged(...) |
    +-------------------------------+
       |
       | (1) dequant_paged_fp8_to_bf16:
       |     per-page, per-token, per-head_dim_block convert
       |     fp8 K, fp8 V -> staging bf16 K, V, applying
       |     `bf16(float(fp8) * fp32_scale[block(head_dim)])`
       v
    +-------------------------------+
    | flash_attention_ragged (bf16) |
    | -> existing FA4 SM100 path    |
    +-------------------------------+
       |
       v
    bf16 output

A future fused convert+scale-apply variant (mirroring
`mla_decode_kv_fp8.mojo`'s `kv_load2cvt_pipe → kv_cvt2mma_pipe`
staging) would replace the external dequant pass with an in-pipeline
convert WG.  The surface exposed here is stable across that change.

Constraints:
- The dequant kernel only queries `load_scale` (and the underlying
  `PagedKVCache._get_scale_idx`) at block-start head_dim indices
  `(d // g) * g`, matching the floordiv semantics in `_get_scale_idx`.
- The dequant operates on FP8 paged blocks indexed by the same
  `lookup_table` as the bf16 staging buffer.
"""

from std.math import ceildiv
from std.sys import size_of

from std.atomic import Atomic
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_dim,
    block_idx,
    grid_dim,
    thread_idx,
)
from std.gpu.host import DeviceBuffer, DeviceContext
from std.utils.numerics import get_accum_type
from std.utils.static_tuple import StaticTuple

from kv_cache.types import (
    KVCacheStaticParams,
    PagedKVCache,
    PagedKVCacheCollection,
)


# ------------------------------------------------------------------------------
# Dequant kernel: paged FP8 KV + per-block fp32 scales -> contiguous BF16 KV
# ------------------------------------------------------------------------------
#
# The kernel is a thin per-element loop. The goal is correctness against a
# host reference; the dispatch surface is designed to be stable across a
# future fused convert variant.
#
# Grid:  (ceildiv(page_size, block_dim.x), num_kv_heads, num_paged_blocks*2)
# Block: 128 threads
#
# Each thread handles `head_dim` elements across one (block, kv_idx, page_row,
# head). The outermost grid dim packs both K (kv_idx=0) and V (kv_idx=1) into
# a single launch so the host call sequence is one kernel, not two.
#
# This is a *device-resident* layout transform: it reads FP8 K/V + fp32 scales
# from the paged cache, dequantizes to bf16, and writes to a flat staging
# buffer whose layout matches what `flash_attention_ragged` consumes as a
# RaggedMHAOperand (rank-3 ragged: [total_seq_len, num_kv_heads, head_dim]).
# We dequant the paged cache view per layer up to max_cache_length, which is
# wasteful at long context relative to a fused convert but acceptable here.


def _compact_live_blocks_kernel[
    page_size: Int,
](
    # `kv_lookup_table`: [batch_size, max_blocks_per_seq] of UInt32 physical
    # block indices.  Per `cache_manager.py:570`, slots beyond the per-seq
    # allocated range are sentinel-filled with `num_paged_blocks`.
    kv_lookup_table_ptr: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    # `compact_buf`: output buffer holding live physical_block IDs.  Sized at
    # `num_paged_blocks` (worst case all blocks live).
    compact_buf_ptr: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    # `compact_count`: atomic counter; output value is the count of live
    # blocks written to `compact_buf`.  Must be zero-initialised before
    # launch.
    compact_count_ptr: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    lut_extent: Int32,
    num_paged_blocks: Int32,
):
    """Compact preprocessor: scan the LUT and write live physical_block IDs
    to `compact_buf`.

    Each thread inspects one LUT slot and skips the sentinel-filled (= `num_paged_blocks`)
    entries, atomically appending live IDs to `compact_buf`.  The LUT is
    fully sentinel-initialised by `PagedKVCacheManager.runtime_inputs`
    (`lut_table_np.fill(self._total_num_pages)`) and only the first
    `len(blocks)` entries per row are overwritten with real physical
    block IDs.  Thus the sentinel check alone gives the correct live set;
    we do NOT need to inspect `cache_lengths` separately.
    (cache_lengths is the *existing* cache size BEFORE this batch; for a
    fresh prompt cache_lengths[s]=0 but the LUT correctly holds the
    just-allocated block IDs for the new tokens.)

    Launch shape: grid_x = ceildiv(lut_extent, 128), block_dim_x = 128.
    Output `compact_count` is the runtime live-block count (= the size of
    `compact_buf` that the dequant kernel should iterate).

    Cost is tiny: one read per LUT slot + one atomic per live slot.  No
    HBM-write amplification — the staging buffer is untouched here.
    """
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= Int(lut_extent):
        return

    # Skip sentinel-filled (unallocated) LUT entries.
    var phys = Int(kv_lookup_table_ptr[tid])
    if phys >= Int(num_paged_blocks):
        return

    # Live block.  Atomically append.
    var write_idx = Atomic[DType.uint32].fetch_add(compact_count_ptr, UInt32(1))
    compact_buf_ptr[Int(write_idx)] = UInt32(phys)


def _dequant_paged_fp8_kv_to_bf16_kernel[
    kv_params: KVCacheStaticParams,
    page_size: Int,
    quantization_granularity: Int,
    # When True, grid.z indexes *logical* (LUT) slots rather than physical
    # blocks; each thread block reads `kv_lookup_table[lut_slot]` to find
    # the physical block to dequant.  This "lookup-table-aware dequant"
    # path lets us launch a tiny grid (proportional to live LUT slots)
    # and still write to the correct staging slot.
    lut_aware: Bool,
    # When True, grid.z indexes into `compact_buf_ptr` (a list of live
    # physical_block IDs).  Each block reads `compact_buf[block_idx.z >>
    # 1]` directly — no LUT scan, no cache_lengths check — and exits if
    # `block_idx.z >= compact_count * 2`.  Only `compact_count` of the
    # launched blocks do work; the rest early-return on a single atomic
    # load.  Production-typical compact_count ≈ 200-500 out of
    # ~3876 launched blocks.
    compact_aware: Bool = False,
](
    # FP8 paged blocks: [total_num_blocks, 2, num_layers, page_size, num_heads,
    # head_size]. We treat all 6 dims as a contiguous row-major view here.
    fp8_blocks_ptr: UnsafePointer[Scalar[DType.float8_e4m3fn], MutAnyOrigin],
    # FP32 scales: [total_num_blocks, 2, num_layers, page_size, num_heads,
    # head_dim_granularity].
    fp32_scales_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    # BF16 staging output: [total_num_blocks, 2, dst_num_layers, page_size,
    # num_heads, head_size]. num_layers==1 for single-layer staging buffers.
    bf16_blocks_ptr: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    num_layers: Int32,
    layer_idx: Int32,
    num_paged_blocks: Int32,
    # Destination layout: staging buffer may have num_layers=1 and layer_idx=0
    # so that a per-layer scratch buffer holds only one layer's worth of KV data.
    dst_num_layers: Int32,
    dst_layer_idx: Int32,
    # LUT-aware path inputs (ignored when lut_aware==False).
    # kv_lookup_table: [batch_size, padded_num_pages] of UInt32 physical block
    # indices.  Sentinel value `num_paged_blocks` marks unused slots.
    kv_lookup_table_ptr: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    # Flat extent of the LUT (= shape[0] * shape[1]).  grid.z = lut_extent * 2
    # under lut_aware mode.
    lut_extent: Int32,
    # compact_aware-only inputs (ignored otherwise).
    # `compact_buf_ptr`: list of live physical_block IDs (output of the
    # preprocessor `_compact_live_blocks_kernel`).
    # `compact_count_ptr`: scalar with the runtime count of live blocks
    # written to `compact_buf_ptr`.
    compact_buf_ptr: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    compact_count_ptr: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
):
    """Per-element dequant: bf16(float(fp8) * fp32_scale[block(head_dim)]).

    The grid is structured so that:
      - block_idx.z is `(physical_block * 2 + kv_idx)` in the non-LUT path,
        or `(lut_slot * 2 + kv_idx)` in the LUT-aware path
      - block_idx.y is head_idx
      - block_idx.x * block_dim.x + thread_idx.x is the (page_row, head_dim)
        flat offset within (page_size * head_dim) for this (block, kv_idx,
        head).

    Each thread handles one element. Scale lookup uses block-start head_dim
    indexing `(d // g) * g`, never a mid-block index, matching the floordiv
    semantics in `_get_scale_idx`.

    LUT-aware path: when `lut_aware=True`, `block_idx.z >> 1` is a LUT
    slot index in `[0, lut_extent)`.  Each block reads
    `kv_lookup_table[lut_slot]` to find the physical block to dequant,
    skipping the work if the entry is the sentinel `num_paged_blocks`
    (padded/unallocated slot).  This lets us launch a grid sized at
    `lut_extent * 2` instead of `num_paged_blocks * 2` while still writing
    to the correct staging slot (indexed by `physical_block`), since the
    downstream attention's LUT indirection reads from the same physical
    slot we wrote.

    Grid-stride loop: CUDA's grid.z is capped at 65535, so for production-
    scale LUTs (e.g. `max_batch_size=128 × max_blocks_per_seq=1960 =
    250880` slots × 2 = 501760) the launcher clamps
    `grid.z = min(lut_extent * 2, 65535)` and we iterate the
    `(lut_slot, kv_idx)` space inside the kernel with stride `grid_dim.z`.
    The sentinel-skip handles padded LUT entries inside the stride loop.
    Threshold: at page_size=128, the stride loop activates above
    batch_size≈17 at the 1938-block production pool size
    (`(17 * 1960 * 2) > 65535`).  Below that, the loop runs exactly once
    and the path is identical to the pre-clamp single-pass behaviour.
    """
    var head_dim = kv_params.head_size
    var page_elems = page_size * head_dim  # per (block, kv_idx, head)
    var t = Int(block_idx.x * block_dim.x + thread_idx.x)
    if t >= page_elems:
        return

    var page_row = t // head_dim
    var d = t % head_dim
    var head = Int(block_idx.y)

    # Total logical (slot, kv_idx) pairs to process.  Mode-dependent:
    #   compact_aware: total = compact_count_runtime * 2
    #                  (read once from the preprocessor's atomic counter).
    #   lut_aware:     total = lut_extent * 2.
    #   legacy:        total = num_paged_blocks * 2.
    var total_pairs: Int
    comptime if compact_aware:
        total_pairs = Int(compact_count_ptr[0]) * 2
    elif lut_aware:
        total_pairs = Int(lut_extent) * 2
    else:
        total_pairs = Int(num_paged_blocks) * 2

    # Grid-stride loop: each block processes pairs
    # `block_idx.z, block_idx.z + grid_dim.z, block_idx.z + 2*grid_dim.z, ...`
    # until total_pairs is exhausted.  This is a no-op single iteration when
    # the launcher's `grid_z` equals `total_pairs` (sub-65535 case).
    var stride = Int(grid_dim.z)
    var kv_idx_and_block_start = Int(block_idx.z)

    var kv_idx_and_block = kv_idx_and_block_start
    while kv_idx_and_block < total_pairs:
        var kv_idx = kv_idx_and_block & 1

        # Resolve the physical block index (compact buf / LUT / direct).
        var physical_block: Int
        comptime if compact_aware:
            # Each (lut_slot >> 1) indexes into compact_buf.  The buf only
            # holds live physical_blocks (preprocessor pre-filtered both
            # sentinels and cache_lengths overflow).  Both K (kv_idx=0)
            # and V (kv_idx=1) for the same physical_block share one entry.
            physical_block = Int(compact_buf_ptr[kv_idx_and_block >> 1])
        elif lut_aware:
            var lut_slot = kv_idx_and_block >> 1
            var entry = Int(kv_lookup_table_ptr[lut_slot])
            # Sentinel = num_paged_blocks marks an unallocated/padded LUT
            # slot; such slots correspond to no live block, so skip.
            if entry >= Int(num_paged_blocks):
                kv_idx_and_block += stride
                continue
            physical_block = entry
        else:
            physical_block = kv_idx_and_block >> 1

        # Source: flat 6D index into [num_blocks, 2, num_layers, page_size,
        # num_heads, head_size] using the full num_layers and real layer_idx.
        var head_dim_gran = ceildiv(head_dim, quantization_granularity)
        var src_base = (
            ((physical_block * 2 + kv_idx) * Int(num_layers) + Int(layer_idx))
            * page_size
            + page_row
        ) * kv_params.num_heads + head
        var src_offset = src_base * head_dim + d
        var scale_block_idx = d // quantization_granularity
        var scales_offset = src_base * head_dim_gran + scale_block_idx

        # Destination: flat 6D index into [num_blocks, 2, dst_num_layers,
        # page_size, num_heads, head_size].  For a per-layer staging
        # buffer, dst_num_layers==1 and dst_layer_idx==0 so the write
        # lands at layer slot 0 regardless of the real layer_idx used
        # for the source read.
        var dst_base = (
            (
                (physical_block * 2 + kv_idx) * Int(dst_num_layers)
                + Int(dst_layer_idx)
            )
            * page_size
            + page_row
        ) * kv_params.num_heads + head
        var dst_offset = dst_base * head_dim + d

        var fp8_v = fp8_blocks_ptr[src_offset]
        var scale = fp32_scales_ptr[scales_offset]
        # Per design §(1): bf16(float(fp8) * fp32_scale).
        var dequantized = fp8_v.cast[DType.float32]() * scale
        bf16_blocks_ptr[dst_offset] = dequantized.cast[DType.bfloat16]()

        kv_idx_and_block += stride


# ------------------------------------------------------------------------------
# Host-side launcher: enqueue the dequant kernel + return a bf16 view.
# ------------------------------------------------------------------------------


@always_inline
def dequant_paged_fp8_kv_to_bf16[
    kv_params: KVCacheStaticParams,
    page_size: Int,
    quantization_granularity: Int,
](
    fp8_blocks_ptr: UnsafePointer[Scalar[DType.float8_e4m3fn], MutAnyOrigin],
    fp32_scales_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    bf16_blocks_ptr: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    num_paged_blocks: Int,
    num_layers: Int,
    layer_idx: Int,
    ctx: DeviceContext,
    dst_num_layers: Int = 1,
    dst_layer_idx: Int = 0,
    kv_lookup_table_ptr: UnsafePointer[
        Scalar[DType.uint32], MutAnyOrigin
    ] = UnsafePointer[Scalar[DType.uint32], MutAnyOrigin].unsafe_dangling(),
    lut_extent: Int = 0,
    # Compact-then-dequant inputs.  Pass all of these (and batch_size /
    # max_blocks_per_seq) to enable the 2-kernel compact-then-dequant
    # path.  When `compact_buf_ptr` is the dangling default, the launcher
    # falls back to the lut_aware path.
    compact_buf_ptr: UnsafePointer[
        Scalar[DType.uint32], MutAnyOrigin
    ] = UnsafePointer[Scalar[DType.uint32], MutAnyOrigin].unsafe_dangling(),
    compact_count_ptr: UnsafePointer[
        Scalar[DType.uint32], MutAnyOrigin
    ] = UnsafePointer[Scalar[DType.uint32], MutAnyOrigin].unsafe_dangling(),
    batch_size: Int = 0,
    max_blocks_per_seq: Int = 0,
    # The dequant grid_z in compact_aware mode.  The launcher uses
    # `min(num_paged_blocks, dequant_grid_z_cap) * 2` so the captured
    # grid is statically bounded.  Default to num_paged_blocks (which
    # matches the pre-compact upper bound).
    dequant_grid_z_cap: Int = -1,
) raises:
    """Enqueue the dequant kernel for one layer of a paged FP8 KV cache.

    Reads `fp8_blocks_ptr` + `fp32_scales_ptr`, writes `bf16_blocks_ptr`.
    Covers both K (kv_idx=0) and V (kv_idx=1) in one launch.

    `dst_num_layers` and `dst_layer_idx` control the destination (staging)
    buffer layout.  Pass `dst_num_layers=1, dst_layer_idx=0` when the
    staging buffer has shape `[num_blocks, 2, 1, page_size, num_heads,
    head_dim]` (single-layer scratch, 50x smaller for Gemma4-31B).
    When not provided they default to `1`/`0` matching that compact layout.

    LUT-aware mode — pass `kv_lookup_table_ptr != null` AND
    `lut_extent > 0` to enable: the launcher sizes `grid_z = 2 *
    lut_extent` (vs. `2 * num_paged_blocks` for the legacy path) and each
    thread block reads `kv_lookup_table[lut_slot]` to discover its
    physical-block target.  Padded/unallocated LUT entries (sentinel ==
    `num_paged_blocks`) are skipped inside the kernel.  This is correct
    even when `lut_extent < num_paged_blocks` because the downstream
    attention reads through the same LUT — physical blocks not in the LUT
    are NEVER read, so they don't need to be written.

    `lut_extent` is the **static** flat extent of the LUT tensor
    (`shape[0] * shape[1]`).  Under CUDA graph capture this is fixed at
    capture time, so the captured grid stays stable.  For typical decode
    workloads (`max_batch_size=128`, low cache occupancy) `lut_extent` is
    1-2 orders of magnitude smaller than `num_paged_blocks`, giving real
    per-launch speedup on the launch-overhead-bound kernel.

    Default args (no LUT) preserve the direct-test behavior:
    grid_z = 2 * num_paged_blocks, write every block in the pool.

    Grid-stride loop: CUDA caps `grid.z <= 65535`, but at production scale
    (`max_batch_size=128 * padded_lut=1960 = 250880` slots × 2 = 501760)
    this is exceeded.  The launcher clamps
    `grid_z = min(total_pairs, 65535)` and the kernel iterates the
    `(lut_slot, kv_idx)` space with stride `grid_dim.z` — see kernel
    docstring.  Activation threshold at the production pool/lut shape is
    `batch_size >= 17` (= 65535 // (1960*2) + 1); below that the loop
    runs once and is equivalent to the pre-clamp single-pass behaviour.
    """
    comptime block_dim_x: Int = 128
    # CUDA grid.z hard cap (per H100/B200 device specs).  Above this the
    # launcher must use a grid-stride loop inside the kernel.
    comptime CUDA_MAX_GRID_Z: Int = 65535

    var page_elems = page_size * kv_params.head_size
    var grid_x = ceildiv(page_elems, block_dim_x)
    var grid_y = kv_params.num_heads

    var use_compact = batch_size > 0 and max_blocks_per_seq > 0
    var use_lut = lut_extent > 0

    var effective_lut_extent = lut_extent if use_lut else num_paged_blocks

    if use_compact:
        # ---- Compact-then-dequant path ----
        # Step 1: scan the LUT to compact live physical block IDs into
        # `compact_buf_ptr`, write count to `compact_count_ptr`.  Tiny
        # kernel — bounded by lut_extent slots.
        var lut_extent_actual = batch_size * max_blocks_per_seq
        var compact_grid_x = ceildiv(lut_extent_actual, block_dim_x)
        comptime compact_kernel = _compact_live_blocks_kernel[page_size]
        # Reset compact_count to 0 before launch (each kernel call must
        # start fresh — graph capture replays the count tensor's residual
        # contents otherwise).
        ctx.enqueue_memset(
            DeviceBuffer[DType.uint32](ctx, compact_count_ptr, 1, owning=False),
            UInt32(0),
        )
        ctx.enqueue_function[compact_kernel](
            kv_lookup_table_ptr,
            compact_buf_ptr,
            compact_count_ptr,
            Int32(lut_extent_actual),
            Int32(num_paged_blocks),
            grid_dim=(compact_grid_x, 1, 1),
            block_dim=(block_dim_x, 1, 1),
        )
        # Step 2: dequant grid sized by `dequant_grid_z_cap` (static).
        # The kernel reads runtime `compact_count_ptr[0]` and only does
        # work for blocks within `compact_count * 2`.
        var compact_cap = num_paged_blocks if dequant_grid_z_cap < 0 else min(
            dequant_grid_z_cap, num_paged_blocks
        )
        var total_pairs_compact = compact_cap * 2
        var grid_z_compact = min(total_pairs_compact, CUDA_MAX_GRID_Z)
        comptime kernel_compact = _dequant_paged_fp8_kv_to_bf16_kernel[
            kv_params,
            page_size,
            quantization_granularity,
            lut_aware=False,
            compact_aware=True,
        ]
        ctx.enqueue_function[kernel_compact](
            fp8_blocks_ptr,
            fp32_scales_ptr,
            bf16_blocks_ptr,
            Int32(num_layers),
            Int32(layer_idx),
            Int32(num_paged_blocks),
            Int32(dst_num_layers),
            Int32(dst_layer_idx),
            kv_lookup_table_ptr,
            Int32(effective_lut_extent),
            compact_buf_ptr,
            compact_count_ptr,
            grid_dim=(grid_x, grid_y, grid_z_compact),
            block_dim=(block_dim_x, 1, 1),
        )
        return

    var grid_blocks = lut_extent if use_lut else num_paged_blocks
    # Total (lut_slot, kv_idx) pairs the kernel must cover.  The grid-z
    # dimension is clamped to CUDA's hardware cap, and the kernel
    # iterates internally if `total_pairs > grid_z`.
    var total_pairs = grid_blocks * 2  # K and V
    var grid_z = min(total_pairs, CUDA_MAX_GRID_Z)

    if use_lut:
        comptime kernel_lut = _dequant_paged_fp8_kv_to_bf16_kernel[
            kv_params,
            page_size,
            quantization_granularity,
            lut_aware=True,
            compact_aware=False,
        ]
        ctx.enqueue_function[kernel_lut](
            fp8_blocks_ptr,
            fp32_scales_ptr,
            bf16_blocks_ptr,
            Int32(num_layers),
            Int32(layer_idx),
            Int32(num_paged_blocks),
            Int32(dst_num_layers),
            Int32(dst_layer_idx),
            kv_lookup_table_ptr,
            Int32(effective_lut_extent),
            compact_buf_ptr,
            compact_count_ptr,
            grid_dim=(grid_x, grid_y, grid_z),
            block_dim=(block_dim_x, 1, 1),
        )
    else:
        comptime kernel_full = _dequant_paged_fp8_kv_to_bf16_kernel[
            kv_params,
            page_size,
            quantization_granularity,
            lut_aware=False,
            compact_aware=False,
        ]
        ctx.enqueue_function[kernel_full](
            fp8_blocks_ptr,
            fp32_scales_ptr,
            bf16_blocks_ptr,
            Int32(num_layers),
            Int32(layer_idx),
            Int32(num_paged_blocks),
            Int32(dst_num_layers),
            Int32(dst_layer_idx),
            kv_lookup_table_ptr,
            Int32(effective_lut_extent),
            compact_buf_ptr,
            compact_count_ptr,
            grid_dim=(grid_x, grid_y, grid_z),
            block_dim=(block_dim_x, 1, 1),
        )
