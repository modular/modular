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
"""Graph-op bindings for the MiniMax-M3 block-sparse attention (MSA) kernels.

BF16, head_dim 128, paged KV with page_size 128 (== block size BN), single
index-K head.  Two ops:

  * `mo.msa.indexer.ragged.paged`   -> `sparse_indexer_{prefill,decode}`
  * `mo.msa.attention.ragged.paged` -> `msa_sm100_decode` (decode) /
    `msa_sm100_prefill_{plan,run}` (prefill) on NVIDIA SM100;
    `msa_amd_decode_dispatch` (decode) / `msa_amd_prefill_run` (prefill) on
    AMD gfx950 (MI355).

The attention op is dual-arch: its body comptime-branches on
`has_amd_gpu_accelerator()` so the dead arch's kernels never codegen (the SM100
tcgen05/TMA prefill cannot lower on gfx950, and the AMD MFMA path cannot lower
on SM100).  The AMD prefill `plan` is the arch-neutral `msa_sm100_prefill_plan`
(host sizing + buffer alloc only -- no SM100 device kernel), shared by both
arches; only the pure-device run differs.  The indexer op is still SM100-only
(the MSA indexer was not ported to AMD); an AMD M3 deployment must source
`d_indices` from an AMD indexer or a host build -- the attention op only
consumes `d_indices`.

Each op takes the same arguments for prefill and decode and picks the kernel at
runtime from `kv_collection.max_seq_length` (the max number of *new* query tokens
in the batch): `== 1` is a single-token decode step, anything larger is a
prefill / context-encoding step.  (Unlike the DeepSeek MLA indexer, the MSA
prefill and decode paths take the same operands, so they need only one op each
rather than separate prefill/decode entry points.)

The indexer op emits top-k *block* ids per (index head, token); the attention
op consumes those block ids (`d_indices`) to gather a sparse band of KV blocks
from the main paged cache.  Both K caches (index-K and main-KV) are BF16 with no
scales, so they build with `generic_get_paged_cache` (NOT the `_with_scales`
variant the MLA FP8 indexer uses).

Modeled on the MLA FP8 indexer registration in `attention.mojo`
(`mo.mla.indexer.ragged.float8.paged`) for the comptime cache-param extraction
+ paged-collection build, and on the in-tree MSA tests
(`Kernels/test/msa/test_msa_sm100_d128_decode_paged.mojo`,
`test_msa_sm100_d128_prefill_device_csr.mojo`) for the exact call shapes.

Three attention routes, picked at runtime from `kv_collection.max_seq_length`
(`max_q_len`, the max number of *new* query tokens; MAX draft length = 4):

  * `== 1`            -> single-token DECODE (`msa_sm100_decode`, NullMask, the
                        SM-fill split-K heuristic).  Causal is a no-op (the
                        single query sits at the sequence END, so every selected
                        past KV position is causal-valid and nothing is masked).
  * `2 / 3 / 4`        -> sparse SPECULATIVE decode (`msa_sm100_decode` with
                        `spec_max_seq_len` bound to the matched length, which
                        derives the spec mode in-entry): one CTA per (draft
                        token, split-K partition), in-kernel per-token causal,
                        capture-stable over-launched grid (`batch * spec_max_seq_len`
                        on the token axis, `max_num_partitions` on the partition
                        axis).  Split-K is REAL (the SM-fill heuristic picks
                        `num_partitions` from `batch * spec_max_seq_len`, NOT
                        NoPartition); at np>1 the partials key on the RAGGED
                        global query row and the shared `mha_splitk_reduce`
                        combine writes the ragged output directly (Frame R).
                        Causal is REAL (a draft token can precede some selected
                        KV); the kernel derives each slot's logical KV start
                        in-kernel as `d_idx_base[blk] * BN` and each token's
                        logical query position as `cache_lengths[batch] +
                        tok_in_seq`, so the op never builds a `kv_logical_pos` or
                        `q_positions` array (mirrors the prefill `use_causal`
                        path, which derives the diagonal from cu_seqlens +
                        cache_lengths).  A short prefill of 2-4 is correctly
                        handled by this path, so no prefill/spec disambiguation
                        is needed.
  * `> 4`              -> PREFILL (`msa_sm100_prefill_{plan,run}`, device CSR).
"""

import extensibility as compiler

from std.collections import OptionalReg
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import ceildiv, min
from std.memory import UnsafePointer
from std.sys.info import has_amd_gpu_accelerator

from layout import row_major, TileTensor, Coord
from layout.tile_tensor import row_major as tt_row_major

from extensibility import InputTensor, OutputTensor
from extensibility import _MutableInputTensor as MutableInputTensor

from nn.kv_cache import generic_get_paged_cache
from nn.attention.mha_operand import KVCacheMHAOperand
from nn.attention.mha_mask import NullMask
from nn.attention.mha_utils import MHAConfig, StaticInt

from msa.sparse_indexer_prefill import sparse_indexer_prefill
from msa.sparse_indexer_decode import sparse_indexer_decode
from msa.msa_1q import msa_sm100_decode
from msa.msa_2q import msa_sm100_prefill_plan, msa_sm100_prefill_run
from msa.amd.decode import msa_amd_decode_dispatch
from msa.amd.prefill import msa_amd_prefill_run


# ===-----------------------------------------------------------------------===#
# Indexer (top-k block selection)
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.msa.indexer.ragged.paged")
struct Struct_msa_indexer_ragged_paged:
    @always_inline
    @staticmethod
    def execute[
        *,
        num_index_heads: Int,
        idx_head_dim: Int,
        block_size: Int,
        topk: Int,
        init_blocks: Int,
        local_blocks: Int,
    ](
        out_idxs: OutputTensor[dtype=DType.int32, rank=3, ...],
        q: InputTensor[dtype=DType.bfloat16, rank=3, ...],
        input_row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        prefix_lens: InputTensor[dtype=DType.uint32, rank=1, ...],
        k_blocks: MutableInputTensor[dtype=DType.bfloat16, rank=6, ...],
        k_cache_lengths: InputTensor[dtype=DType.uint32, rank=1, ...],
        k_lookup_table: InputTensor[dtype=DType.uint32, rank=2, ...],
        k_max_prompt_length: InputTensor[dtype=DType.uint32, rank=1, ...],
        k_max_cache_length: InputTensor[dtype=DType.uint32, rank=1, ...],
        msa_scalar_args: InputTensor[dtype=DType.int64, rank=1, ...],
        layer_idx: UInt32,
        score_scratch: MutableInputTensor[dtype=DType.float32, rank=3, ...],
        scale: Float32,
        ctx: DeviceContext,
    ) raises:
        """Select top-k key *blocks* per (index head, query token).

        Dispatches to the decode kernel when `kv_collection.max_seq_length == 1`
        (one new index-K token per sequence) and to the prefill kernel
        otherwise.

        Parameters:
            num_index_heads: Number of index (query) heads.
            idx_head_dim: Index head dimension.
            block_size: KV block size in tokens (== page_size).
            topk: Number of blocks to select per token.
            init_blocks: Always-keep leading blocks (forced high score).
            local_blocks: Always-keep trailing/local blocks (forced score).

        Args:
            out_idxs: Output block indices `[num_index_heads, num_rows, topk]`,
                int32, `-1`-padded (`num_rows` == total_q on prefill, batch on
                decode).
            q: Query tensor `[num_rows, num_index_heads, idx_head_dim]` BF16.
            input_row_offsets: Ragged query offsets `[batch + 1]` uint32 (used on
                the prefill path; on decode it is `[0, 1, ..., batch]`).
            prefix_lens: Per-batch cached-key count `[batch]` uint32 (pass the
                index-K `cache_lengths`); used as the decode `seq_lens`.
            k_blocks: Index-K paged blocks `[num_blocks, 1, num_layers,
                page_size, 1, idx_head_dim]` BF16.
            k_cache_lengths: Index-K cache lengths `[batch]` uint32.
            k_lookup_table: Index-K page table `[batch, max_pages]` uint32.
            k_max_prompt_length: Index-K max prompt (query) length `[1]` uint32.
            k_max_cache_length: Index-K max cache length `[1]` uint32.
            msa_scalar_args: On-device scalar arguments for the decode indexer
                msa_scalar_args[0] = batch_size
                msa_scalar_args[1] = max_cache_valid_length.
            layer_idx: Layer index for the index-K cache.
            score_scratch: Persistent decode score scratch
                `[num_index_heads, max_batch, MAX_NUM_BLOCKS]`.
            scale: QK scale.
            ctx: Device context.
        """
        var k_collection = generic_get_paged_cache(
            k_blocks,
            k_cache_lengths,
            k_lookup_table,
            k_max_prompt_length,
            k_max_cache_length,
        )
        var k_cache = k_collection.get_key_cache(Int(layer_idx))
        var k_operand = KVCacheMHAOperand(k_cache)

        var total_q = Int(q.dim_size[0]())
        if total_q == 0:
            return
        var max_num_blocks = ceildiv(
            Int(k_cache.max_context_length())
            + Int(k_cache.max_prompt_length()),
            block_size,
        )

        # Decode == one new index-K token per sequence (`num_rows == batch`);
        # anything larger is a prefill / context-encoding step.
        if Int(k_collection.max_seq_length) == 1:
            var batch = total_q  # 1 token/seq on decode
            # Use persistent score scratch. This is required for graph capture.
            var score = score_scratch.to_tile_tensor[DType.int64]()

            # `prefix_lens` is the index-K `cache_lengths` BEFORE this step's
            # IndexK was scattered (MAX `_is_cache_length_accurate=False`
            # convention). The decode scorer needs the POST-write key count, so
            # it adds this step's per-batch query-token count (from
            # `input_row_offsets`) internally. On the decode route there is
            # exactly one new token per sequence, so that count is 1; passing
            # `input_row_offsets` keeps it correct for any ragged in-step count.
            sparse_indexer_decode[
                DType.bfloat16,
                type_of(k_operand),
                num_index_heads,
                idx_head_dim,
                block_size,
            ](
                q.to_tile_tensor[DType.int64](),
                k_operand,
                prefix_lens.to_tile_tensor[DType.int64](),
                input_row_offsets.to_tile_tensor[DType.int64](),
                score,
                out_idxs.to_tile_tensor[DType.int64](),
                batch,
                max_num_blocks,
                topk,
                init_blocks,
                local_blocks,
                scale,
                ctx,
            )
        else:
            var batch = Int(input_row_offsets.dim_size[0]()) - 1

            # Caller-owned score scratch [num_index_heads, total_q, max_num_blocks].
            var score_size = num_index_heads * total_q * max_num_blocks
            var score_buf = ctx.enqueue_create_buffer[DType.float32](score_size)
            score_buf.enqueue_fill(Float32(0))
            var score = TileTensor(
                score_buf,
                tt_row_major(num_index_heads, total_q, max_num_blocks),
            )

            sparse_indexer_prefill[
                DType.bfloat16,
                type_of(k_operand),
                num_index_heads,
                idx_head_dim,
                block_size,
            ](
                q.to_tile_tensor[DType.int64](),
                k_operand,
                input_row_offsets.to_tile_tensor[DType.int64](),
                prefix_lens.to_tile_tensor[DType.int64](),
                score,
                out_idxs.to_tile_tensor[DType.int64](),
                batch,
                total_q,
                Int(k_cache.max_prompt_length()),  # max_seqlen_q
                max_num_blocks,
                topk,
                init_blocks,
                local_blocks,
                scale,
                ctx,
            )
            _ = score_buf^


# ===-----------------------------------------------------------------------===#
# Sparse attention (block-gathered MHA)
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.msa.attention.ragged.paged")
struct Struct_msa_attention_ragged_paged:
    @always_inline
    @staticmethod
    def execute[
        *,
        group: Int,
        topk: Int,
    ](
        output: OutputTensor[dtype=DType.bfloat16, rank=3, ...],
        q: InputTensor[dtype=DType.bfloat16, rank=3, ...],
        input_row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        cache_row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        total_context_length: InputTensor[dtype=DType.uint32, rank=1, ...],
        kv_blocks: MutableInputTensor[dtype=DType.bfloat16, rank=6, ...],
        cache_lengths: InputTensor[dtype=DType.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=DType.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=DType.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=DType.uint32, rank=1, ...],
        msa_scalar_args: InputTensor[dtype=DType.int64, rank=1, ...],
        layer_idx: UInt32,
        d_indices: InputTensor[dtype=DType.int32, rank=3, ...],
        scale: Float32,
        ctx: DeviceContext,
    ) raises:
        """Block-sparse MHA for SM100 (BF16, head_dim 128).

        Gathers `topk` KV blocks per (kv head, query token) using the block ids
        in `d_indices`.  Dispatches to the decode kernel when
        `kv_collection.max_seq_length == 1` (one query token per sequence) and to
        the prefill kernel otherwise.

        Decode uses `NullMask` + an SM-fill split-K heuristic
        (`get_mha_decoding_max_num_partitions` clamped by `topk`): `num_partitions
        > 1` runs the block-major fwd over partitioned KV bands then combines via
        the shared `mha_splitk_reduce`; `num_partitions == 1` takes the no-combine
        `NoPartition` path.  Prefill uses the
        device-CSR plan/run path (`msa_sm100_prefill_plan` +
        `msa_sm100_prefill_run`): the run is pure-device, but the plan sizes its
        buffers from the per-batch cu-seqlens on host, so one D2H readback +
        sync per call is unavoidable while this stays a single stateless op.

        Routing is purely by the runtime query length
        `max_q_len = kv_collection.max_seq_length` (the max new query tokens):
        `== 1` decode, `2 / 3 / 4` sparse speculative decode (one CTA per draft
        token, real per-token causal, capture-stable over-launch -- see the
        module docstring; `spec_max_seq_len` is bound to the matched length per
        branch), and `> 4` prefill.  A short 2-4 prefill is correctly handled by
        the spec path, so no prefill/spec disambiguation is needed.  Spec decode
        derives each draft token's logical query position in-kernel from
        `cache_lengths + tok_in_seq` (mirrors the prefill `use_causal` path), so
        no `q_positions` array is built or passed.

        Parameters:
            group: Query heads per kv-head (`n_heads // n_kv_heads`); asserts
                `group <= MMA_M` in the kernel.
            topk: Number of gathered KV blocks per token (`d_indices` stride).

        Args:
            output: Output `[num_rows, n_heads, head_dim]` BF16.
            q: Query `[num_rows, n_heads, head_dim]` BF16 (`num_rows` == total_q
                on prefill, batch on decode).
            input_row_offsets: Ragged query offsets `[batch + 1]` uint32 (1
                token/seq on decode).
            cache_row_offsets: Ragged valid cache offsets `[batch + 1]` uint32.
            total_context_length: Total context length of the current batch.
            kv_blocks: Main-KV paged blocks `[num_blocks, 2, num_layers,
                page_size, n_kv_heads, head_dim]` BF16.
            cache_lengths: Main-KV cache lengths `[batch]` uint32.
            kv_lookup_table: Main-KV page table `[batch, max_pages]` uint32.
            max_prompt_length: Main-KV max prompt (query) length `[1]` uint32.
            max_cache_length: Main-KV max cache length `[1]` uint32.
            msa_scalar_args: On-device scalar arguments for the MSA decode
                msa_scalar_args[0] = batch_size
                msa_scalar_args[1] = max_cache_valid_length.
            layer_idx: Layer index for the main-KV cache.
            d_indices: Selected block ids `[n_kv_heads, num_rows, topk]` int32.
            scale: QK scale.
            ctx: Device context.
        """
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )
        var k_cache = kv_collection.get_key_cache(Int(layer_idx))
        var v_cache = kv_collection.get_value_cache(Int(layer_idx))
        var k_op = KVCacheMHAOperand(k_cache)
        var v_op = KVCacheMHAOperand(v_cache)

        comptime k_num_heads = Int(kv_blocks.static_spec.shape_tuple[4])
        comptime head_dim = Int(kv_blocks.static_spec.shape_tuple[5])
        comptime page_size = Int(kv_blocks.static_spec.shape_tuple[3])
        comptime num_heads = group * k_num_heads
        comptime config = MHAConfig[DType.bfloat16](num_heads, head_dim)

        # `num_rows` == total query tokens (== batch on decode, 1 token/seq).
        var num_rows = Int(q.dim_size[0]())

        # Non-owning DeviceBuffer views over the graph tensors.
        var out_lt = output.to_layout_tensor()
        var q_lt = q.to_layout_tensor()
        var output_buf = DeviceBuffer[DType.bfloat16](
            ctx, out_lt.ptr, num_rows * num_heads * head_dim, owning=False
        )
        var q_buf = DeviceBuffer[DType.bfloat16](
            ctx, q_lt.ptr, num_rows * num_heads * head_dim, owning=False
        )

        # Route purely on the runtime query length.  MAX speculative draft
        # length is 4; `2/3/4` route to spec decode, `> 4` to prefill (a short
        # 2-4 prefill is correctly served by the spec/prefill path).  Spec
        # decode is SM100-only; on AMD the `2-4` case falls through to the
        # paged prefill run (the AMD decode dispatch has no spec mode).
        comptime MAX_SPEC_DRAFT = 4
        var max_q_len = Int(kv_collection.max_seq_length)

        # Decode == one query token per sequence (`max_q_len == 1`).
        if max_q_len == 1:
            var topk_tokens = topk * page_size

            var iro_lt = input_row_offsets.to_layout_tensor()
            var valid_length = DeviceBuffer[DType.uint32](
                ctx,
                iro_lt.ptr,
                Int(input_row_offsets.dim_size[0]()),
                owning=False,
            )
            var d_indices_tt = TileTensor(
                d_indices.to_layout_tensor().ptr,
                row_major(Coord(d_indices.to_layout_tensor().size())),
            ).as_immut()

            # `np` is owned by the decode entry (computed from batch_size,
            # topk_tokens, topk via the dense-MHA heuristic).
            #
            # `mask_unselected=True`: the indexer `-1`-pads `d_indices` when
            # the sequence has fewer than `topk` selectable blocks (e.g. a
            # short first decode step). Without this the `-1` blocks attend
            # phantom rows, and -- with np == topk -- each `-1` lands in its
            # own fully-masked partition whose NaN exp-sum poisons the combine.
            #
            # AMD and SM100 take the same call signature; the two arms differ
            # only in kernel name.  No `valid_key` (the indexer's trailing
            # block is whole; a sub-BN partial would need per-batch clamp).
            comptime if has_amd_gpu_accelerator():
                msa_amd_decode_dispatch[
                    config=config,
                    group=group,
                    ragged=True,
                    _is_cache_length_accurate=False,
                    mask_unselected=True,
                ](
                    output_buf,
                    q_buf,
                    k_op,
                    v_op,
                    d_indices_tt,
                    topk,  # indices_stride (topk in BLOCKS)
                    num_rows,  # num_rows_q (1 token/seq)
                    NullMask(),
                    valid_length,
                    StaticInt[1](),  # max_prompt_len (decode)
                    topk_tokens,  # max_cache_valid_length
                    scale,
                    None,  # kv_input_row_offsets
                    num_rows,  # batch_size
                    ctx,
                )
            else:
                msa_sm100_decode[
                    config=config,
                    group=group,
                    ragged=True,
                    _is_cache_length_accurate=False,
                    mask_unselected=True,
                ](
                    output_buf,
                    q_buf,
                    k_op,
                    v_op,
                    d_indices_tt,
                    topk,  # indices_stride (topk in BLOCKS)
                    num_rows,  # num_rows_q (1 token/seq)
                    NullMask(),
                    valid_length,
                    StaticInt[1](),  # max_prompt_len (decode)
                    topk_tokens,  # max_cache_valid_length
                    scale,
                    None,  # kv_input_row_offsets
                    num_rows,  # batch_size
                    ctx,
                )
        elif (not has_amd_gpu_accelerator()) and max_q_len <= MAX_SPEC_DRAFT:
            # ---- Sparse SPECULATIVE decode (`2 <= max_q_len <= 4`, SM100) ----
            # Each draft token runs on its OWN CTA via the per-token decode
            # kernel (`spec_max_seq_len > 1` derives the spec mode in-entry =>
            # per_token_index + causal + the over-launched
            # `batch * spec_max_seq_len` grid).  Selection reuses the PREFILL
            # indexer (per-token `[head_kv, total_q, topk]`), so the block ids
            # here are per draft token.  `input_row_offsets` is the ragged Q
            # offset array the kernel reads for the over-launch token tail and
            # the global-query-row remap (`iro[b] + tok_in_seq`).  Causal is
            # REAL here (a draft token can precede some selected KV): the kernel
            # poisons slots whose logical position exceeds the token's logical
            # query position, deriving the slot's logical start in-kernel from
            # `d_idx_base[blk]*BN` (no `kv_logical_pos` array) and the token's
            # logical query position in-kernel from
            # `cache_lengths[batch_of_token] + tok_in_seq` (no `q_positions`
            # array -- mirrors the prefill `use_causal` path, which derives the
            # diagonal from cu_seqlens + cache_lengths).  REAL split-K:
            # `msa_sm100_decode` feeds `batch * spec_max_seq_len` to the decode
            # partition heuristic so the partition axis fills the SM array at
            # low batch, and launches the shared `mha_splitk_reduce` combine
            # when np > 1 (the causal dead-partition salvage in the partial
            # writeback keeps the combine NaN-free).  The partials key on the
            # ragged global query row, so the combine writes the ragged output
            # directly (no dense intermediate / gather).  SM100 only: AMD has
            # no spec mode (the `2-4` case falls through to paged prefill).
            # The inner `comptime if` keeps the tcgen05 `msa_1q` body from
            # codegen'ing on gfx950 even though the outer `elif` is runtime.
            comptime if not has_amd_gpu_accelerator():
                var iro_lt = input_row_offsets.to_layout_tensor()
                var valid_length = DeviceBuffer[DType.uint32](
                    ctx,
                    iro_lt.ptr,
                    Int(input_row_offsets.dim_size[0]()),
                    owning=False,
                )
                var d_indices_tt = TileTensor(
                    d_indices.to_layout_tensor().ptr,
                    row_major(Coord(d_indices.to_layout_tensor().size())),
                ).as_immut()
                var topk_tokens = topk * page_size
                var batch = Int(input_row_offsets.dim_size[0]()) - 1

                # The over-launch span `spec_max_seq_len` is a graph constant,
                # so bind it to the matched runtime length per branch (one CTA
                # per (draft token, partition) over `batch * spec_max_seq_len`).
                # The entry derives the spec mode from `spec_max_seq_len > 1`.
                comptime for n in range(2, MAX_SPEC_DRAFT + 1):
                    if max_q_len == n:
                        msa_sm100_decode[
                            config=config,
                            group=group,
                            ragged=True,
                            _is_cache_length_accurate=False,
                            mask_unselected=True,
                            spec_max_seq_len=n,  # over-launch span (graph const)
                        ](
                            output_buf,
                            q_buf,
                            k_op,
                            v_op,
                            d_indices_tt,
                            topk,  # indices_stride (topk in BLOCKS)
                            num_rows,  # num_rows_q (total draft tokens)
                            NullMask(),
                            valid_length,  # ragged Q offsets (tail + row remap)
                            StaticInt[
                                1
                            ](),  # max_prompt_len: tile is decode-shaped
                            topk_tokens,  # max_cache_valid_length
                            scale,
                            None,  # kv_input_row_offsets
                            batch,  # batch_size (grid.x = batch*spec_max_seq_len)
                            ctx,
                            # Spec decode derives BOTH the per-block logical
                            # start and the per-token logical query position
                            # in-kernel (the latter from `cache_lengths +
                            # tok_in_seq`), so it carries neither a
                            # `kv_logical_pos` nor a `q_positions` array.  The
                            # kernel keys causal off the derived spec mode (=>
                            # `causal`), not off the presence of a `q_positions`
                            # pointer.
                        )
                        return
        else:
            var batch = Int(input_row_offsets.dim_size[0]()) - 1

            var lse_buf = ctx.enqueue_create_buffer[DType.float32](
                num_rows * num_heads
            )

            var d_lt = d_indices.to_layout_tensor()
            var d_indices_buf = DeviceBuffer[DType.int32](
                ctx, d_lt.ptr, k_num_heads * num_rows * topk, owning=False
            )

            var plan = msa_sm100_prefill_plan[
                output_type=DType.bfloat16,
                config=config,
                group=group,
                topk=topk,
            ](
                num_rows,
                Int(total_context_length[0]),
                batch,
                Int(kv_collection.max_seq_length),
                Int(kv_collection.max_cache_length),
                ctx,
            )

            # bitcast input_row_offsets and cache_row_offsets to int32, then
            # wrap then in DeviceBuffer.
            var cuq_d = DeviceBuffer[DType.int32](
                ctx,
                input_row_offsets._ptr.bitcast[Int32](),
                batch + 1,
                owning=False,
            )
            var cuk_d = DeviceBuffer[DType.int32](
                ctx,
                cache_row_offsets._ptr.bitcast[Int32](),
                batch + 1,
                owning=False,
            )

            # The plan (host sizing + buffer alloc) is arch-neutral (no SM100
            # device kernel), so it codegens on gfx950.  Only the device run
            # differs: AMD chains CSR-build -> block-major fwd -> combine;
            # SM100 runs its tcgen05 path.  Both take the identical signature;
            # the comptime branch keeps the dead arch's kernels from codegen'ing.
            comptime if has_amd_gpu_accelerator():
                msa_amd_prefill_run[
                    config=config,
                    group=group,
                    topk=topk,
                    use_causal=True,
                ](
                    plan,
                    output_buf,
                    lse_buf,
                    q_buf,
                    k_op,
                    v_op,
                    d_indices_buf,
                    cuq_d,
                    cuk_d,
                    scale,
                    ctx,
                )
            else:
                msa_sm100_prefill_run[
                    config=config,
                    group=group,
                    topk=topk,
                    use_causal=True,
                ](
                    plan,
                    output_buf,
                    lse_buf,
                    q_buf,
                    k_op,
                    v_op,
                    d_indices_buf,
                    cuq_d,
                    cuk_d,
                    scale,
                    ctx,
                )

            _ = lse_buf^
