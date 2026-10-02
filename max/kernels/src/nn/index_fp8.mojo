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
"""Implements tensor indexing and gather kernels with FP8 quantization support on Blackwell GPUs."""
from std.math.uutils import ufloordiv, udivmod
from std.sys import size_of, simd_width_of
from std.sys.info import _has_blackwell_tcgen05
from std.math import ceildiv
from layout import (
    Coord,
    Idx,
    DefaultEngine,
    TensorLayout,
    TensorEngine,
    TileTensor,
    stack_allocation,
)
from layout.tile_tensor import ImmTileTensor, MutTileTensor
from layout.tile_layout import row_major
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.sync import barrier
from max.gpu.memory import external_memory
from kv_cache.types import create_flat_kv_tma_tile
from nn.attention.mha_operand import RaggedMHAOperand, MHAOperand
from nn.attention.gpu.nvidia.common import q_tma
from nn.attention.gpu.sparse_index_fp8_sm100 import (
    _BM_KEY,
    _INDEX_SWIZZLE,
    SPEC_DECODE_N_TOKENS_ALT,
    fp8_index_score_sm100,
)
from std.utils.index import Index


struct IndexSmemStorage[
    dtype: DType,
    num_heads: Int,
    depth: Int,
    BN: Int,
]:
    """Holds shared-memory buffers for the query, key, and scratch tiles used by the FP8 index kernel.

    Parameters:
        dtype: Data type of the query and key tiles.
        num_heads: Number of attention heads stored in the query tile.
        depth: Per-head feature depth of the query and key tiles.
        BN: Number of key rows staged in shared memory per block tile.
    """

    var q_smem: Array[Scalar[Self.dtype], Self.num_heads * Self.depth]
    var k_smem: Array[Scalar[Self.dtype], Self.BN * Self.depth]
    var scratch: Array[Float32, Self.BN * 8]


@__name(t"fp8_index_{dtype}")
def fp8_index_kernel[
    dtype: DType,
    OutputLT: TensorLayout,
    QLT: TensorLayout,
    QSLT: TensorLayout,
    k_operand_type: MHAOperand,
    ks_operand_type: MHAOperand,
    block_tile_shape: Array[Int, 2],
    VLLT: TensorLayout,
    num_heads: Int,
    depth: Int,
    # When False: num_keys = cache_length + seq_len (cache_length excludes new tokens).
    # When True: num_keys = cache_length (cache_length already includes new tokens).
    _is_cache_length_accurate: Bool = False,
    *,
    OutputEngine: TensorEngine = DefaultEngine[element_width=1],
    QEngine: TensorEngine = DefaultEngine[element_width=1],
    QSEngine: TensorEngine = DefaultEngine[element_width=1],
    VLEngine: TensorEngine = DefaultEngine[element_width=1],
](
    output: TileTensor[.float32, OutputLT, MutAnyOrigin, Engine=OutputEngine],
    # [total_seq_len, num_heads, depth]
    q: TileTensor[dtype, QLT, ImmutAnyOrigin, Engine=QEngine],
    # [total_seq_len, num_heads]
    q_s: TileTensor[.float32, QSLT, ImmutAnyOrigin, Engine=QSEngine],
    # MHAOperand for K values
    k_operand: k_operand_type,
    # MHAOperand for K scales
    ks_operand: ks_operand_type,
    valid_length: TileTensor[.uint32, VLLT, ImmutAnyOrigin, Engine=VLEngine],
):
    """Computes the scalar FP8 index/gather score kernel as a Blackwell tensor-core fallback.

    Each block computes a slice of the query sequence against a tile of the
    paged key cache, accumulating per-head logits in shared memory and writing
    the scale-weighted row sum to the output tensor.

    Parameters:
        dtype: Data type of the query and key tiles.
        OutputLT: Layout type of the output score tensor.
        QLT: Layout type of the query tensor.
        QSLT: Layout type of the query scale tensor.
        k_operand_type: MHAOperand type used to address the paged key cache.
        ks_operand_type: MHAOperand type used to address the key scales.
        block_tile_shape: Block tile shape as `[BM, BN]` rows of sequence and keys.
        VLLT: Layout type of the valid-length (sequence offset) tensor.
        num_heads: Number of attention heads.
        depth: Per-head feature depth.
        _is_cache_length_accurate: When True, `cache_length` already includes new tokens.
        OutputEngine: Engine of the `output_tt` tile.
        QEngine: Engine of the `q_tt` tile.
        QSEngine: Engine of the `q_s_tt` tile.
        VLEngine: Engine of the `valid_length_tt` tile.

    Args:
        output: Output score tensor of shape `[total_seq_len, num_keys]`.
        q: Query tensor of shape `[total_seq_len, num_heads, depth]`.
        q_s: Per-query scale tensor of shape `[total_seq_len, num_heads]`.
        k_operand: Ragged paged operand providing key rows.
        ks_operand: Ragged paged operand providing per-key scales.
        valid_length: Cumulative sequence offsets of shape `[batch_size + 1]`.
    """

    comptime assert q.flat_rank == 3
    comptime assert q_s.flat_rank == 2
    comptime assert output.flat_rank == 2
    comptime assert valid_length.flat_rank == 1, "valid_length must be 1D"
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]

    comptime thread_dim_x = 16
    comptime thread_dim_y = 8

    comptime simd_width = simd_width_of[dtype]()

    var batch_idx = block_idx.x
    var seq_offset = block_idx.y
    var key_offset = block_idx.z * BM
    var tid = thread_idx.x * 8 + thread_idx.y

    var start_of_seq = valid_length[batch_idx][0]
    var end_of_seq = valid_length[batch_idx + 1][0]
    var seq_len = end_of_seq - start_of_seq

    var num_keys = k_operand.cache_length(batch_idx)

    comptime if not _is_cache_length_accurate:
        num_keys += Int(seq_len)

    if seq_offset >= Int(seq_len) or key_offset >= num_keys:
        return

    ref smem_ptr = external_memory[
        UInt8,
        address_space=.SHARED,
        alignment=128,
    ]().unsafe_bitcast[IndexSmemStorage[dtype, num_heads, depth, BN]]()[]

    ref q_smem = smem_ptr.q_smem
    ref k_smem = smem_ptr.k_smem
    ref scratch_smem = smem_ptr.scratch

    var q_smem_tile = TileTensor(
        q_smem.unsafe_ptr(), row_major[num_heads, depth]()
    )

    var k_smem_tile = TileTensor(k_smem.unsafe_ptr(), row_major[BN, depth]())

    var k_smem_ptr: Pointer[
        Scalar[dtype], origin_of(k_smem), address_space=.SHARED
    ] = k_smem.unsafe_ptr()

    var q_ptr = q.ptr_at_offset(Coord(start_of_seq + UInt32(seq_offset), 0, 0))
    var q_s_ptr = q_s.ptr_at_offset(Coord(start_of_seq + UInt32(seq_offset), 0))
    var o_ptr = output.ptr_at_offset(
        Coord(start_of_seq + UInt32(seq_offset), UInt32(key_offset))
    )

    var q_tile = TileTensor(q_ptr, row_major[num_heads, depth]())
    var q_s_tile = TileTensor(q_s_ptr, row_major[1, num_heads]())
    var logits = stack_allocation[.float32, .LOCAL](
        row_major[BN // thread_dim_x, num_heads // thread_dim_y]()
    )
    var q_s_reg_tile = stack_allocation[.float32, .LOCAL](
        row_major[1, num_heads // thread_dim_y]()
    )
    var logits_sum = stack_allocation[.float32, .LOCAL](
        row_major[BN // thread_dim_x, 1]()
    )
    var scratch = TileTensor(
        scratch_smem.unsafe_ptr().as_unsafe_any_origin(),
        row_major[BN, thread_dim_y](),
    )

    var q_s_frag = q_s_tile.tile[1, num_heads // thread_dim_y](
        ufloordiv(thread_idx.x, thread_dim_x), thread_idx.y
    )

    comptime for q_frag_idx in range(num_heads // thread_dim_y):
        q_s_reg_tile[0, q_frag_idx] = q_s_frag[0, q_frag_idx][0]

    comptime num_threads = thread_dim_x * thread_dim_y
    comptime assert (
        depth % simd_width == 0
    ), "depth must be a multiple of the SIMD width"

    # Flat thread-strided copy of the contiguous [num_heads, depth] Q tile.
    # A layout-distributed copy over the [16, 8] thread shape floor-divides
    # the tile shape per axis and silently stages NOTHING whenever
    # num_heads < 16 or depth // simd_width < 8 (e.g. depth == 64).
    comptime q_vecs = num_heads * depth // simd_width
    var q_smem_dst: Pointer[
        Scalar[dtype], origin_of(q_smem), address_space=.SHARED
    ] = q_smem.unsafe_ptr()
    for v in range(Int(tid), q_vecs, num_threads):
        q_smem_dst.unsafe_store(
            v * simd_width, q_ptr.unsafe_load[width=simd_width](v * simd_width)
        )

    for i in range(BM // BN):
        var current_key_offset = key_offset + i * BN
        if current_key_offset >= num_keys:
            break
        for k_row in range(tid, BN, num_threads):
            var row_base = k_row * depth
            if current_key_offset + k_row >= num_keys:
                comptime for d in range(0, depth, simd_width):
                    k_smem_ptr.unsafe_store(
                        row_base + d, SIMD[dtype, simd_width](0)
                    )
            else:
                var k_ptr = k_operand.block_paged_ptr[1](
                    UInt32(batch_idx),
                    UInt32(current_key_offset + k_row),
                    UInt32(0),  # head_idx = 0 for MLA (single head for K)
                    UInt32(0),
                )
                comptime for d in range(0, depth, simd_width):
                    k_smem_ptr.unsafe_store(
                        row_base + d,
                        k_ptr.unsafe_load[width=simd_width](d).cast[dtype](),
                    )

        barrier()

        # Load K scales for current tile
        var k_s_reg: Float32 = 0.0
        if current_key_offset + tid < num_keys:
            var ks_ptr = ks_operand.block_paged_ptr[1](
                UInt32(batch_idx),
                UInt32(current_key_offset + tid),
                UInt32(0),
                UInt32(0),
            )
            k_s_reg = ks_ptr[].cast[.float32]()

        var q_smem_frag = q_smem_tile.tile[num_heads // thread_dim_y, depth](
            thread_idx.y, 0
        )
        var k_smem_frag = k_smem_tile.tile[BN // thread_dim_x, depth](
            thread_idx.x, 0
        )

        _ = logits.fill(0)
        _ = logits_sum.fill(0)

        for k in range(depth):
            comptime for mma_m in range(BN // thread_dim_x):
                comptime for mma_n in range(num_heads // thread_dim_y):
                    logits[mma_m, mma_n] += (
                        k_smem_frag[mma_m, k][0].cast[.float32]()
                        * q_smem_frag[mma_n, k][0].cast[.float32]()
                    )

        comptime for l_i in range(BN // thread_dim_x):
            comptime for l_j in range(num_heads // thread_dim_y):
                logits[l_i, l_j] = (
                    max(logits[l_i, l_j], 0) * q_s_reg_tile[0, l_j][0]
                )

                logits_sum[l_i, 0] += logits[l_i, l_j]

            scratch[
                thread_idx.x * (BN // thread_dim_x) + l_i,
                thread_idx.y,
            ] = logits_sum[l_i, 0]

        barrier()

        if current_key_offset + tid < num_keys:
            # Sum logits across heads
            var row_sum: Float32 = 0.0

            for col_idx in range(thread_dim_y):
                row_sum += scratch[tid, col_idx][0]

            o_ptr[unsafe_offset=i * BN + tid] = k_s_reg * row_sum


@inline(.always)
def fp8_index[
    dtype: DType,
    output_layout: TensorLayout,
    q_layout: TensorLayout,
    qs_layout: TensorLayout,
    k_layout: TensorLayout,
    ks_layout: TensorLayout,
    vl_layout: TensorLayout,
    cro_layout: TensorLayout,
    //,
    num_heads: Int,
    depth: Int,
](
    output: MutTileTensor[.float32, output_layout, _],
    q: ImmTileTensor[dtype, q_layout, _],
    q_s: ImmTileTensor[.float32, qs_layout, _],
    k: ImmTileTensor[dtype, k_layout, _],
    k_s: ImmTileTensor[.float32, ks_layout, _],
    valid_length: ImmTileTensor[.uint32, vl_layout, _],
    cache_row_offsets: ImmTileTensor[.uint32, cro_layout, _],
    batch_size: Int,
    max_seq_len: Int,
    max_num_keys: Int,
    ctx: DeviceContext,
) raises:
    """Dispatches the FP8 index/gather scorer on the given device context.

    Selects the Blackwell tcgen05/TMA tensor-core scorer when the device and
    operand layout support it, otherwise falls back to the scalar
    `fp8_index_kernel` path.

    Parameters:
        dtype: Data type of the query and key tensors.
        output_layout: Layout of the output score tensor.
        q_layout: Layout of the query tensor.
        qs_layout: Layout of the per-query scale tensor.
        k_layout: Layout of the key tensor.
        ks_layout: Layout of the per-key scale tensor.
        vl_layout: Layout of the cumulative sequence offsets.
        cro_layout: Layout of the cache row offsets.
        num_heads: Number of attention heads.
        depth: Per-head feature depth.

    Args:
        output: Output score tensor of shape `[total_seq_len, max_num_keys]`.
        q: Query tensor of shape `[total_seq_len, num_heads, depth]`.
        q_s: Per-query scale tensor of shape `[total_seq_len, num_heads]`.
        k: Key tensor of shape `[total_keys, 1, depth]`.
        k_s: Per-key scale tensor of shape `[total_keys]`.
        valid_length: Cumulative sequence offsets of shape `[batch_size + 1]`.
        cache_row_offsets: Per-batch row offsets into the paged key cache.
        batch_size: Number of sequences in the batch.
        max_seq_len: Maximum sequence length across the batch.
        max_num_keys: Maximum key count across the batch.
        ctx: Device context used to enqueue the selected kernel.

    Raises:
        When the underlying kernel enqueue reports a device-side error.
    """
    var total_keys = Int(k.dim[0]())

    var k_operand = RaggedMHAOperand(k, cache_row_offsets)
    # The operand wants a rank-3 buffer; the per-key scales are one scalar per
    # key row.
    comptime assert k_s.is_row_major
    var ks_operand = RaggedMHAOperand(
        k_s.reshape(Coord(total_keys, Idx[1], Idx[1])), cache_row_offsets
    )

    comptime assert num_heads % 4 == 0, "num_heads must be a multiple of 4"

    # RaggedMHAOperand.cache_length() returns full key length directly, so the
    # SM100 tensor-core scorer and the scalar fallback both run with
    # _is_cache_length_accurate=True (skip adding seq_len in the kernel).
    # The scorer uses tcgen05/TMA (Blackwell-only), so gate on
    # _has_blackwell_tcgen05(): H100/A100/other NVIDIA and AMD take the scalar
    # fallback. The SM100 scorer stages a BM_key-row K tile with one TMA copy,
    # so a paged K cache must have page_size == 0 (contiguous, as this ragged
    # path is) or a multiple of BM_key; any other page_size falls back too.
    comptime if (
        _has_blackwell_tcgen05()
        and (
            num_heads == 64
            or num_heads == 32
            or num_heads == 8
            or num_heads == 4
        )
        and depth == 128
        and (
            type_of(k_operand).page_size == 0
            or type_of(k_operand).page_size % _BM_KEY == 0
        )
    ):
        # One-block form of the same descriptor: a ragged buffer has no
        # blocks, so it declares one and is addressed at block 0.
        var k_tma_tile = create_flat_kv_tma_tile[
            BN=_BM_KEY, BK=depth, swizzle_mode=_INDEX_SWIZZLE
        ](ctx, k.unsafe_ptr().as_unsafe_any_origin(), total_keys, 1, depth)
        fp8_index_score_sm100[
            dtype,
            type_of(k_operand),
            type_of(ks_operand),
            num_heads,
            depth,
            _is_cache_length_accurate=True,
            # GLM 5.x MTP decodes 6 tokens (num_draft_tokens + 1), which the
            # default 4-token N-tile at nh=32 covers with two blocks spending 256
            # MMA columns on 192 live ones. 3 divides 6, so it tiles the step
            # exactly at 96 columns -- and unlike 6, its TMEM footprint still
            # leaves room for two co-resident CTAs. Inert wherever 3 tokens are
            # not a legal UMMA N or the default already divides the step (nh=64).
            #
            # This is a speculative-decode tile and nothing else. The bound is
            # arithmetic, not a threshold: a 3-token tile needs `msl // 3` blocks
            # where the default needs `ceildiv(msl, 4)`, and
            # `msl // 3 <= ceildiv(msl, 4)` iff `msl <= 9`, so the gap grows
            # without bound and only max_seq_len in {3, 6, 9} survives at nh=32
            # (12 is excluded because the default tile already divides it). A
            # 2-token hint was tried first, back when only 64 columns could be
            # hoisted without spilling -- but it needs THREE blocks to cover the
            # step, so it pays 1.5x the CTA prologues to reach the same hoist. Once
            # the index arithmetic was narrowed to 32-bit, every width hoists at
            # zero spill and 3 tokens is simply the exact divisor.
            N_TOKENS_ALT=SPEC_DECODE_N_TOKENS_ALT,
        ](
            output,
            q,
            q_s,
            k_operand,
            ks_operand,
            k_tma_tile,
            valid_length,
            batch_size,
            max_seq_len,
            max_num_keys,
            False,
            ctx,
        )
    else:
        comptime assert num_heads % 16 == 0, (
            "the scalar fp8_index_kernel tiles heads by thread_dim_y == 8 and"
            " is unvalidated below 16 heads; num_heads in {4, 8} requires the"
            " SM100 tensor-core path"
        )
        comptime block_tile_shape: Array[Int, 2] = [512, 128]
        comptime BM = block_tile_shape[0]
        comptime BN = block_tile_shape[1]
        comptime smem_use = size_of[
            IndexSmemStorage[dtype, num_heads, depth, BN]
        ]()
        comptime smem_available = ctx.default_device_info.shared_memory_per_multiprocessor - 1024

        comptime kernel = fp8_index_kernel[
            dtype,
            output_layout,
            q_layout,
            qs_layout,
            type_of(k_operand),
            type_of(ks_operand),
            block_tile_shape,
            vl_layout,
            num_heads,
            depth,
            _is_cache_length_accurate=True,
        ]

        ctx.enqueue_function[kernel](
            output,
            q,
            q_s,
            k_operand,
            ks_operand,
            valid_length,
            grid_dim=(
                batch_size,
                max_seq_len,
                ceildiv(max_num_keys, BM),
            ),
            block_dim=(16, 8, 1),
            shared_mem_bytes=smem_use,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(smem_available)
            ),
        )


@__name(t"fp8_index_matmul_max_{dtype}")
def _index_matmul_max[
    dtype: DType,
    output_layout: TensorLayout,
    q_layout: TensorLayout,
    qs_layout: TensorLayout,
    k_layout: TensorLayout,
    vl_layout: TensorLayout,
    cro_layout: TensorLayout,
](
    output: TileTensor[.float32, output_layout, MutAnyOrigin],
    q: TileTensor[dtype, q_layout, ImmutAnyOrigin],
    q_s: TileTensor[.float32, qs_layout, ImmutAnyOrigin],
    k: TileTensor[dtype, k_layout, ImmutAnyOrigin],
    valid_length: TileTensor[.uint32, vl_layout, ImmutAnyOrigin],
    cache_row_offsets: TileTensor[.uint32, cro_layout, ImmutAnyOrigin],
):
    comptime assert q.flat_rank == 3
    comptime assert k.flat_rank == 3
    comptime assert output.flat_rank == 3

    comptime num_heads = q_layout.static_shape[1]
    comptime depth = q_layout.static_shape[2]

    var batch_idx, head_idx = udivmod(block_idx.z, num_heads)
    var seq_idx = block_idx.x * 16 + thread_idx.x
    var key_idx = block_idx.y * 16 + thread_idx.y

    var start_of_seq = valid_length[batch_idx]
    var end_of_seq = valid_length[batch_idx + 1]
    var seq_len = end_of_seq - start_of_seq

    var k_row_start = cache_row_offsets[batch_idx]
    var k_row_end = cache_row_offsets[batch_idx + 1]
    var num_keys = k_row_end - k_row_start

    if key_idx >= Int(num_keys) or seq_idx >= Int(seq_len):
        return

    var q_batch = q[Int(start_of_seq) : Int(end_of_seq), :, :]
    var k_batch = k[Int(k_row_start) : Int(k_row_end), :, :]
    # Slice only the leading axis: the view inherits the dense parent's
    # `max_num_keys` row stride, which differs from this entry's own key count
    # on a ragged batch.
    var o_batch = output[Int(start_of_seq) : Int(end_of_seq), :, :]

    # Cast each FP8 code to F32 before multiply so we match TileLang-style GEMM
    # (FP8×FP8 in low precision can saturate/widen differently than F32 products).
    var accum = Float32(0.0)
    for d in range(Int(depth)):
        var kd = k_batch[key_idx, 0, d][0].cast[.float32]()
        var qd = q_batch[seq_idx, head_idx, d][0].cast[.float32]()
        accum += kd * qd

    accum = max(accum, 0) * q_s[start_of_seq + UInt32(seq_idx), head_idx][0]
    o_batch[seq_idx, key_idx, head_idx] = accum


@__name(t"fp8_index_reduce_logits")
def _reduce_logits[
    logits_layout: TensorLayout,
    output_layout: TensorLayout,
    ks_layout: TensorLayout,
    vl_layout: TensorLayout,
    cro_layout: TensorLayout,
](
    logits: TileTensor[.float32, logits_layout, MutAnyOrigin],
    output: TileTensor[.float32, output_layout, MutAnyOrigin],
    k_s: TileTensor[.float32, ks_layout, ImmutAnyOrigin],
    valid_length: TileTensor[.uint32, vl_layout, ImmutAnyOrigin],
    cache_row_offsets: TileTensor[.uint32, cro_layout, ImmutAnyOrigin],
):
    comptime num_heads = logits_layout.static_shape[2]
    var batch_idx = block_idx.z
    var seq_idx = block_idx.x * 16 + thread_idx.x
    var key_idx = block_idx.y * 16 + thread_idx.y

    var start_of_seq = valid_length[batch_idx][0]
    var end_of_seq = valid_length[batch_idx + 1][0]
    var seq_len = end_of_seq - start_of_seq

    var k_row_start = cache_row_offsets[batch_idx][0]
    var k_row_end = cache_row_offsets[batch_idx + 1][0]
    var num_keys = k_row_end - k_row_start

    if seq_idx >= Int(seq_len) or key_idx >= Int(num_keys):
        return

    # Slice only the leading axis so both views keep the dense parents'
    # `max_num_keys` row stride rather than this entry's own key count.
    var o_batch = output[Int(start_of_seq) : Int(end_of_seq), :]
    var logits_batch = logits[Int(start_of_seq) : Int(end_of_seq), :, :]
    var k_s_batch = k_s[Int(k_row_start) : Int(k_row_end)]

    var sum = Float32(0.0)
    for head in range(num_heads):
        sum += logits_batch[seq_idx, key_idx, head][0]

    o_batch[seq_idx, key_idx] = sum * k_s_batch[key_idx][0]


@inline(.always)
def fp8_index_naive[
    dtype: DType,
    output_layout: TensorLayout,
    q_layout: TensorLayout,
    qs_layout: TensorLayout,
    k_layout: TensorLayout,
    ks_layout: TensorLayout,
    vl_layout: TensorLayout,
    cro_layout: TensorLayout,
    //,
    num_heads: Int,
    depth: Int,
](
    output: MutTileTensor[.float32, output_layout, _],
    q: ImmTileTensor[dtype, q_layout, _],
    q_s: ImmTileTensor[.float32, qs_layout, _],
    k: ImmTileTensor[dtype, k_layout, _],
    k_s: ImmTileTensor[.float32, ks_layout, _],
    valid_length: ImmTileTensor[.uint32, vl_layout, _],
    cache_row_offsets: ImmTileTensor[.uint32, cro_layout, _],
    batch_size: Int,
    max_seq_len: Int,
    max_num_keys: Int,
    ctx: DeviceContext,
) raises:
    """Computes the FP8 index/gather score via a two-pass matmul-then-reduce reference path.

    Enqueues `_index_matmul_max` to produce per-head logits followed by
    `_reduce_logits` to sum across heads and apply the per-key scale, serving
    as a correctness reference for the optimized tensor-core kernels.

    Parameters:
        dtype: Data type of the query and key tensors.
        output_layout: Layout of the output score tensor.
        q_layout: Layout of the query tensor; its head and depth extents must
            be static and equal `num_heads` and `depth`.
        qs_layout: Layout of the per-query scale tensor.
        k_layout: Layout of the key tensor.
        ks_layout: Layout of the per-key scale tensor.
        vl_layout: Layout of the cumulative sequence offsets.
        cro_layout: Layout of the cache row offsets.
        num_heads: Number of attention heads.
        depth: Per-head feature depth.

    Args:
        output: Output score tensor of shape `[total_seq_len, max_num_keys]`.
        q: Query tensor of shape `[total_seq_len, num_heads, depth]`.
        q_s: Per-query scale tensor of shape `[total_seq_len, num_heads]`.
        k: Key tensor of shape `[total_keys, 1, depth]`.
        k_s: Per-key scale tensor of shape `[total_keys]`.
        valid_length: Cumulative sequence offsets of shape `[batch_size + 1]`.
        cache_row_offsets: Per-batch row offsets into the paged key cache.
        batch_size: Number of sequences in the batch.
        max_seq_len: Maximum sequence length across the batch.
        max_num_keys: Maximum key count across the batch.
        ctx: Device context used to enqueue the kernels.

    Raises:
        When the underlying kernel enqueue reports a device-side error.
    """
    comptime assert (
        q_layout.static_shape[1] == num_heads
        and q_layout.static_shape[2] == depth
    ), "fp8_index_naive: q must have static [num_heads, depth] trailing dims"

    var logits_size = batch_size * max_seq_len * max_num_keys * num_heads
    var logits_dev = ctx.enqueue_create_buffer[.float32](logits_size)
    logits_dev.enqueue_fill(Float32(0.0))
    var logits_buf = TileTensor(
        logits_dev.unsafe_ptr(),
        row_major(
            Coord(batch_size * max_seq_len, max_num_keys, Idx[num_heads])
        ),
    )

    comptime mm = _index_matmul_max[
        dtype,
        type_of(logits_buf).LayoutType,
        q_layout,
        qs_layout,
        k_layout,
        vl_layout,
        cro_layout,
    ]

    ctx.enqueue_function[mm](
        logits_buf,
        q,
        q_s,
        k,
        valid_length,
        cache_row_offsets,
        grid_dim=(
            ceildiv(max_seq_len, 16),
            ceildiv(max_num_keys, 16),
            batch_size * num_heads,
        ),
        block_dim=(16, 16, 1),
    )

    comptime reduce_logits = _reduce_logits[
        type_of(logits_buf).LayoutType,
        output_layout,
        ks_layout,
        vl_layout,
        cro_layout,
    ]

    ctx.enqueue_function[reduce_logits](
        logits_buf,
        output,
        k_s,
        valid_length,
        cache_row_offsets,
        grid_dim=(
            ceildiv(max_seq_len, 16),
            ceildiv(max_num_keys, 16),
            batch_size,
        ),
        block_dim=(16, 16, 1),
    )

    _ = logits_dev
