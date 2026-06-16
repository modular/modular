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

Target: NVIDIA SM100 (B200), BF16, head_dim 128, paged KV with page_size 128
(== block size BN), single index-K head.  Two ops:

  * `mo.msa.indexer.ragged.paged`   -> `sparse_indexer_{prefill,decode}`
  * `mo.msa.attention.ragged.paged` -> `msa_sm100_{prefill_,}dispatch`

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
`test_msa_sm100_d128_prefill_b.mojo`) for the exact dispatch call shapes.

TODO(causal): the forward op passes `q_positions` (per-token logical query
position) and leaves `kv_logical_pos=None` for the first cut -- the indexer
already restricts selection to causal-valid blocks, so the only residual
causal work is the diagonal (partial) block, which we validate with a logit
check later before enabling in-kernel `kv_logical_pos` masking.
"""

import extensibility as compiler

from std.collections import OptionalReg
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import ceildiv
from std.memory import UnsafePointer

from layout import row_major, TileTensor
from layout.tile_tensor import row_major as tt_row_major

from extensibility import InputTensor, OutputTensor
from extensibility import _MutableInputTensor as MutableInputTensor

from nn.kv_cache import generic_get_paged_cache
from nn.attention.mha_operand import KVCacheMHAOperand
from nn.attention.mha_mask import NullMask
from nn.attention.mha_utils import MHAConfig, NoPartition, StaticInt
from std.utils.numerics import get_accum_type

from msa.sparse_indexer_prefill import sparse_indexer_prefill
from msa.sparse_indexer_decode import sparse_indexer_decode
from msa.msa_1q import msa_sm100_dispatch
from msa.msa_2q import msa_sm100_prefill_b_device_csr_dispatch


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
        k_max_lengths: InputTensor[dtype=DType.uint32, rank=2, ...],
        layer_idx: UInt32,
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
            k_max_lengths: Index-K max lengths `[1, 2]` uint32.
            layer_idx: Layer index for the index-K cache.
            scale: QK scale.
            ctx: Device context.
        """
        var k_collection = generic_get_paged_cache(
            k_blocks,
            k_cache_lengths,
            k_lookup_table,
            k_max_lengths,
        )
        var k_cache = k_collection.get_key_cache(Int(layer_idx))
        var k_operand = KVCacheMHAOperand(k_cache)

        var total_q = Int(q.dim_size[0]())
        var max_num_blocks = ceildiv(
            Int(k_cache.max_context_length())
            + Int(k_cache.max_prompt_length()),
            block_size,
        )

        # Decode == one new index-K token per sequence (`num_rows == batch`);
        # anything larger is a prefill / context-encoding step.
        if Int(k_collection.max_seq_length) == 1:
            var batch = total_q  # 1 token/seq on decode

            # Caller-owned score scratch [num_index_heads, batch, max_num_blocks].
            var score_size = num_index_heads * batch * max_num_blocks
            var score_buf = ctx.enqueue_create_buffer[DType.float32](score_size)
            score_buf.enqueue_fill(Float32(0))
            var score = TileTensor(
                score_buf,
                tt_row_major(num_index_heads, batch, max_num_blocks),
            )

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
            _ = score_buf^
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
        kv_blocks: MutableInputTensor[dtype=DType.bfloat16, rank=6, ...],
        cache_lengths: InputTensor[dtype=DType.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=DType.uint32, rank=2, ...],
        max_lengths: InputTensor[dtype=DType.uint32, rank=2, ...],
        layer_idx: UInt32,
        d_indices: InputTensor[dtype=DType.int32, rank=3, ...],
        q_positions: InputTensor[dtype=DType.int32, rank=1, ...],
        scale: Float32,
        ctx: DeviceContext,
    ) raises:
        """Block-sparse MHA for SM100 (BF16, head_dim 128).

        Gathers `topk` KV blocks per (kv head, query token) using the block ids
        in `d_indices`.  Dispatches to the decode kernel when
        `kv_collection.max_seq_length == 1` (one query token per sequence) and to
        the prefill kernel otherwise.

        Decode uses `NullMask` + `NoPartition` (no split-K).  Prefill uses the
        device-CSR path (`msa_sm100_prefill_b_device_csr_dispatch`) which
        requires cumulative sequence lengths copied to host.  Both paths pass
        `q_positions` for causal with `kv_logical_pos=None` (see module TODO).

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
            kv_blocks: Main-KV paged blocks `[num_blocks, 2, num_layers,
                page_size, n_kv_heads, head_dim]` BF16.
            cache_lengths: Main-KV cache lengths `[batch]` uint32.
            kv_lookup_table: Main-KV page table `[batch, max_pages]` uint32.
            max_lengths: Main-KV max lengths `[1, 2]` uint32.
            layer_idx: Layer index for the main-KV cache.
            d_indices: Selected block ids `[n_kv_heads, num_rows, topk]` int32.
            q_positions: Per-token logical query position `[num_rows]` int32
                (used for causal; see module TODO).
            scale: QK scale.
            ctx: Device context.
        """
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_lengths,
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
        comptime accum_type = get_accum_type[DType.bfloat16]()
        comptime KVPtrT = UnsafePointer[Int32, MutAnyOrigin]

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

        var q_positions_ptr = rebind[KVPtrT](q_positions.to_layout_tensor().ptr)

        # Decode == one query token per sequence (`max_seq_length == 1`).
        if Int(kv_collection.max_seq_length) == 1:
            var iro_lt = input_row_offsets.to_layout_tensor()
            var valid_length = DeviceBuffer[DType.uint32](
                ctx,
                iro_lt.ptr,
                Int(input_row_offsets.dim_size[0]()),
                owning=False,
            )
            var d_indices_ptr = rebind[KVPtrT](d_indices.to_layout_tensor().ptr)
            var topk_tokens = topk * page_size
            msa_sm100_dispatch[
                config=config,
                group=group,
                ragged=True,
                _is_cache_length_accurate=False,
            ](
                output_buf,
                q_buf,
                k_op,
                v_op,
                d_indices_ptr,
                topk,  # indices_stride (topk in BLOCKS)
                num_rows,  # num_rows_q (1 token/seq)
                NullMask(),
                valid_length,
                StaticInt[1](),  # max_prompt_len (decode)
                topk_tokens,  # max_cache_valid_length
                scale,
                None,  # kv_input_row_offsets
                num_rows,  # batch_size
                NoPartition[accum_type](),
                ctx,
                kv_logical_pos=None,  # TODO(causal): see module docstring
                q_positions=OptionalReg[KVPtrT](q_positions_ptr),
            )
        else:
            var batch = Int(input_row_offsets.dim_size[0]()) - 1

            var lse_buf = ctx.enqueue_create_buffer[DType.float32](
                num_rows * num_heads
            )

            var d_lt = d_indices.to_layout_tensor()
            var d_indices_buf = DeviceBuffer[DType.int32](
                ctx, d_lt.ptr, k_num_heads * num_rows * topk, owning=False
            )

            var iro_lt = input_row_offsets.to_layout_tensor()
            var iro_dev = DeviceBuffer[DType.uint32](
                ctx, iro_lt.ptr, batch + 1, owning=False
            )
            var iro_host = ctx.enqueue_create_host_buffer[DType.uint32](
                batch + 1
            )
            ctx.enqueue_copy(iro_host, iro_dev)

            var cl_lt = cache_lengths.to_layout_tensor()
            var cl_dev = DeviceBuffer[DType.uint32](
                ctx, cl_lt.ptr, batch, owning=False
            )
            var cl_host = ctx.enqueue_create_host_buffer[DType.uint32](batch)
            ctx.enqueue_copy(cl_host, cl_dev)
            ctx.synchronize()

            var cu_seqlens_q = List[Int32](length=batch + 1, fill=Int32(0))
            for i in range(batch + 1):
                cu_seqlens_q[i] = iro_host[i].cast[DType.int32]()

            var cu_seqlens_k = List[Int32](length=batch + 1, fill=Int32(0))
            for i in range(batch):
                var seq_len_q = (
                    iro_host[i + 1].cast[DType.int32]()
                    - iro_host[i].cast[DType.int32]()
                )
                cu_seqlens_k[i + 1] = (
                    cu_seqlens_k[i] + cl_host[i].cast[DType.int32]() + seq_len_q
                )

            msa_sm100_prefill_b_device_csr_dispatch[
                config=config,
                group=group,
                topk=topk,
            ](
                output_buf,
                lse_buf,
                q_buf,
                k_op,
                v_op,
                d_indices_buf,
                cu_seqlens_q,
                cu_seqlens_k,
                scale,
                ctx,
                q_positions=OptionalReg[KVPtrT](q_positions_ptr),
            )

            _ = lse_buf^
