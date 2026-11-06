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


# ===-----------------------------------------------------------------------===#
# General imports
# ===-----------------------------------------------------------------------===#

"""Registers attention graph ops (MHA, MLA, fused QKV) and dispatches them to the `nn.attention` kernels."""

from std.collections import OptionalReg
import extensibility

# ===-----------------------------------------------------------------------===#
# Kernel imports
# ===-----------------------------------------------------------------------===#

from max.gpu.host import DeviceAttribute, DeviceBuffer, DeviceContext
from layout import Coord, Idx
from layout.tile_tensor import row_major as tt_row_major
from max.gpu.host.info import is_cpu, is_gpu
from max.gpu.primitives.grid_controls import PDLLevel, pdl_launch_attributes
from kv_cache.paged_sparse_kv_index_remap import (
    NoRemap,
    PagedKVRemap,
    PagedKVScaleRemap,
    SparseIndexRemap,
    paged_sparse_kv_index_remap,
)
from kv_cache.types import KVCacheStaticParams
from linalg.mx_format import MXFormat
from layout import (
    TileTensor,
    RowMajorLayout,
    UNKNOWN_VALUE,
)
from nn.attention.cpu.mha import flash_attention as nn_flash_attention
from nn.kv_cache import (
    generic_flash_attention_kv_cache_padded,
    generic_fused_qkv_matmul_kv_cache_bshd_paged,
    generic_get_paged_cache,
    generic_get_paged_cache_with_scales,
    paged_cache_rows,
)
from nn.kv_cache_ragged import (
    generic_cross_attention_kv_cache,
    generic_flare_mla_decode_kv_cache_ragged,
    generic_flare_mla_decompress_k_cache_ragged_paged,
    generic_flare_mla_prefill_kv_cache_ragged,
    generic_flare_mla_prefill_ragged_paged_plan,
    generic_flash_attention_kv_cache_ragged,
    generic_fused_qk_rope_bshd_paged_ragged,
    generic_fused_qkv_index_matmul_kv_cache_paged_ragged,
    generic_fused_qkv_index_matmul_kv_cache_paged_ragged_scale_float4,
    generic_fused_qkv_matmul_kv_cache_paged_ragged,
    generic_fused_qkv_matmul_kv_cache_paged_ragged_bias,
    generic_fused_qkv_matmul_kv_cache_paged_ragged_scale,
    generic_fused_qkv_matmul_kv_cache_paged_ragged_scale_float4,
)
from nn.attention.gpu.mha import flash_attention, flash_attention_ragged
from nn.attention.gpu.nvidia.common import (
    ImmutTileTensor1D,
    immut_tile_tensor_1d,
)
from nn.attention.gpu.mha_decode_partition_heuristic import (
    amd_sparse_mla_inplace_decode_num_partitions,
)
from nn.attention.mha_mask import MHAMask
from kv_cache.mixed_step_split import (
    MIXED_STEP_DECODE_WIDTH,
    is_packed,
    read_mixed_split,
    rebase_row_offsets,
)
from nn.attention.mha_utils import (
    NonNullPointer,
    dispatch_mask,
    maybe_null_pointer,
)
from nn.attention.gpu.mla_graph import (
    AMDMLAGraph,
    LegacyNVMLAGraph,
    MLAGraphKernels,
    SM100MLAGraph,
    mla_prefill_branch_fp8,
    mla_prefill_branch_bf16,
    mla_decode_branch_fp8,
    mla_decode_branch_bf16,
    mla_graph_takes_decode_branch,
    mla_sparse_check_topk_lengths,
    mla_sparse_prefill_on_decode_kernel,
    mla_prefill_decode_graph_fp8,
    mla_prefill_decode_graph_bf16,
    mla_store_ragged_paged,
)
from nn.attention.gpu.mla_index_fp8 import (
    BlockSplitTopk,
    IndexerTopk,
    mla_indexer_ragged_float8_paged,
    mla_indexer_ragged_float8_paged_cp,
)
from nn.attention.gpu.cp_merge_topk import cp_merge_topk
from nn.attention.compress_windows import compress_windows_ragged_paged
from nn.attention.indexer_score import indexer_score_ragged_paged
from nn.attention.gpu.mla_common import SparseRowLayout
from max.runtime.tracing import Trace, TraceLevel, get_safe_task_id
from extensibility import DynamicTensor, InputTensor, OutputTensor
from extensibility import Fused, Input, LoadAccess, TensorTuple
from extensibility import (
    _FusedInputTensor as FusedInputTensor,
)
from extensibility import (
    _MutableInputTensor as MutableInputTensor,
)
from std.logger import Logger

comptime logger = Logger()

from std.utils import IndexList

# ===-----------------------------------------------------------------------===#
from .kernels import *
from nn.attention.gpu.mla_index_kpool import (
    kpool_compress_kernel,
    KPOOL_EXPAND_BLOCK_SIZE,
    kpool_expand_topk_kernel,
    kpool_ring_close_kernel,
    kpool_seed_tail_kernel,
    kpool_tail_update_kernel,
)

from .kernels import (
    _execute_mha_ragged_paged_scalar_args,
    _unmarshal_mha_decode_dispatch_metadata,
    _unsafe_str_to_coord,
)


@extensibility.register("attention.mla.kpool.expand_topk")
struct MLAKPoolExpandTopK:
    """Registers the `attention.mla.kpool.expand_topk` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        *,
        kpool: Int,
        pool_topk: Int,
        always_select_tail: Bool,
    ](
        out_indices: Some[Output & Tensor],
        pool_ids: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        """Turns selected pool ids back into the token positions they cover.

        The indexer selects pools; attention reads tokens. Pool `p` covers
        positions `[p * kpool, (p + 1) * kpool)`, so the selection widens from
        `pool_topk` to `pool_topk * kpool`. Each row is packed: the selected
        pools in selection order, then the tail, then `-1`, never a clamped
        valid position, which would point attention at a token the indexer did
        not choose.

        Parameters:
            kpool: Tokens per pool.
            pool_topk: Pools selected per query; the width of `pool_ids`.
            always_select_tail: Append the `kpool - 1` positions after the last
                complete pool -- the query's most recent tokens.

        Args:
            out_indices: Output `[total_seq_len, pool_topk * kpool + tail]`
                token positions, `-1` where unused.
            pool_ids: Selected pool ids `[total_seq_len, pool_topk]`.
            input_row_offsets: Token row offsets per request, `[batch + 1]`.
            cache_lengths: Cached tokens per request, `[batch]`.
            ctx: Device context for GPU execution.
        """
        comptime assert conforms_to(type_of(out_indices), TileTensorable)
        comptime assert conforms_to(type_of(pool_ids), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert type_of(out_indices).rank == 2
        comptime assert type_of(pool_ids).rank == 2
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        var out_tt = out_indices.to_tile_tensor[.int32]()
        var pool_tt = pool_ids.to_tile_tensor[.int32]()
        var iro_tt = input_row_offsets.to_tile_tensor[.uint32]()
        var clen_tt = cache_lengths.to_tile_tensor[.uint32]()

        var total_seq_len = Int(pool_ids.shape()[0].value())
        if total_seq_len == 0:
            # The architecture sets `supports_empty_batches`, so a step with
            # no tokens reaches here, and a zero-width grid is a launch error.
            # The sibling kpool registrations guard the same way.
            return

        ctx.enqueue_function[
            kpool_expand_topk_kernel[
                out_tt.LayoutType,
                out_tt.origin,
                type_of(pool_tt.as_imm()).LayoutType,
                ImmOrigin(pool_tt.origin),
                type_of(iro_tt.as_imm()).LayoutType,
                ImmOrigin(iro_tt.origin),
                type_of(clen_tt.as_imm()).LayoutType,
                out_tt.Engine,
                type_of(pool_tt.as_imm()).Engine,
                type_of(iro_tt.as_imm()).Engine,
                type_of(clen_tt.as_imm()).Engine,
                kpool,
                pool_topk,
                always_select_tail,
            ]
        ](
            out_tt,
            pool_tt.as_imm(),
            iro_tt.as_imm(),
            clen_tt.as_imm(),
            Int32(total_seq_len),
            grid_dim=(total_seq_len, 1, 1),
            block_dim=(KPOOL_EXPAND_BLOCK_SIZE, 1, 1),
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )


@extensibility.register("attention.mla.kpool.compress")
struct MLAKPoolCompress:
    """Registers the `attention.mla.kpool.compress` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        *,
        head_dim: Int,
        kpool: Int,
    ](
        pooled: Some[Output & Tensor],
        k: Some[Input & Tensor],
        gate: Some[Input & Tensor],
        ape: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        pool_row_offsets: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        """Compresses each complete k-pool into one candidate key.

        Parameters:
            head_dim: Channels per key; also the kernel's block width.
            kpool: Tokens per pool.

        Args:
            pooled: Output `[total_pools, head_dim]` pooled keys.
            k: Layer-normed indexer keys `[total_tokens, head_dim]`.
            gate: Per-token gate scores `[total_tokens, head_dim]`.
            ape: Within-pool position embedding `[kpool, head_dim]`, float32.
            input_row_offsets: Token row offsets `[batch + 1]`.
            pool_row_offsets: Pool row offsets `[batch + 1]`.
            cache_lengths: Cached-prefix length per request `[batch]`. A pool
                covers absolute positions, so this is what places the call's
                tokens on the pool grid.
            ctx: Device context for GPU execution.
        """
        comptime assert conforms_to(type_of(pooled), TileTensorable)
        comptime assert conforms_to(type_of(k), TileTensorable)
        comptime assert conforms_to(type_of(gate), TileTensorable)
        comptime assert conforms_to(type_of(ape), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(pool_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert type_of(pooled).rank == 2
        comptime assert type_of(k).rank == 2
        comptime assert type_of(gate).rank == 2
        comptime assert type_of(ape).rank == 2
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(pool_row_offsets).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        var pooled_tt = pooled.to_tile_tensor()
        var k_tt = k.to_tile_tensor()
        var gate_tt = gate.to_tile_tensor()
        var ape_tt = ape.to_tile_tensor[.float32]()
        var iro_tt = input_row_offsets.to_tile_tensor[.uint32]()
        var pro_tt = pool_row_offsets.to_tile_tensor[.uint32]()
        var clen_tt = cache_lengths.to_tile_tensor[.uint32]()

        var total_pools = Int(pooled.shape()[0].value())
        if total_pools == 0:
            # A step can legitimately close no pool: a decode step carries one
            # token per request, and a pool needs `kpool` of them. There is
            # nothing to compress, and a zero-width grid is a launch error.
            return

        ctx.enqueue_function[
            kpool_compress_kernel[
                type_of(k).dtype,
                type_of(k_tt.as_imm()).LayoutType,
                ImmOrigin(k_tt.origin),
                type_of(gate_tt.as_imm()).LayoutType,
                ImmOrigin(gate_tt.origin),
                type_of(ape_tt.as_imm()).LayoutType,
                ImmOrigin(ape_tt.origin),
                type_of(iro_tt.as_imm()).LayoutType,
                ImmOrigin(iro_tt.origin),
                type_of(pro_tt.as_imm()).LayoutType,
                ImmOrigin(pro_tt.origin),
                type_of(clen_tt.as_imm()).LayoutType,
                pooled_tt.LayoutType,
                pooled_tt.origin,
                type_of(k_tt.as_imm()).Engine,
                type_of(gate_tt.as_imm()).Engine,
                type_of(ape_tt.as_imm()).Engine,
                type_of(iro_tt.as_imm()).Engine,
                type_of(pro_tt.as_imm()).Engine,
                type_of(clen_tt.as_imm()).Engine,
                pooled_tt.Engine,
                head_dim,
                kpool,
            ]
        ](
            pooled_tt,
            k_tt.as_imm(),
            gate_tt.as_imm(),
            ape_tt.as_imm(),
            iro_tt.as_imm(),
            pro_tt.as_imm(),
            clen_tt.as_imm(),
            grid_dim=(total_pools, 1, 1),
            block_dim=(head_dim, 1, 1),
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )


@extensibility.register("attention.mla.kpool.seed_tail")
struct MLAKPoolSeedTail:
    """Registers the `attention.mla.kpool.seed_tail` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        *,
        head_dim: Int,
        kpool: Int,
    ](
        # `tail` is a slot-indexed ring, read+written in place at
        # `slot_idx[r]`. It must be a `MutableInput` (not an `Output`) so
        # the graph binds the caller's persistent ring rather than treating
        # it as a freshly-produced output -- mirroring the `state_pool`
        # precedent in `kda.mojo`.
        tail: Some[MutableInput & Tensor],
        k: Some[Input & Tensor],
        gate: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        slot_idx: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        """Stashes a prefill chunk's trailing tokens into the tail ring.

        Whole pools compress directly from `k`/`gate`; the tokens after the
        last complete pool are that request's in-progress pool and have
        nowhere else to go until later tokens complete it, so they seed this
        per-request ring. `attention.mla.kpool.ring_close` reads them back and
        closes the pool once it fills.

        Parameters:
            head_dim: Channels per key; also the kernel's block width.
            kpool: Tokens per pool.

        Args:
            tail: Persistent slot-indexed ring `[max_slots, 2, kpool,
                head_dim]`, mutated in place.
            k: Layer-normed indexer keys `[total_tokens, head_dim]`.
            gate: Per-token gate scores `[total_tokens, head_dim]`.
            input_row_offsets: Token row offsets per request `[batch + 1]`.
            cache_lengths: Cached tokens per request `[batch]`.
            slot_idx: Ring slot per request `[batch]`.
            ctx: Device context for GPU execution.
        """
        comptime assert conforms_to(type_of(tail), TileTensorable)
        comptime assert conforms_to(type_of(k), TileTensorable)
        comptime assert conforms_to(type_of(gate), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(slot_idx), TileTensorable)
        comptime assert type_of(tail).rank == 4
        comptime assert type_of(k).rank == 2
        comptime assert type_of(gate).rank == 2
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(slot_idx).rank == 1
        var tail_tt = tail.to_tile_tensor()
        var k_tt = k.to_tile_tensor()
        var gate_tt = gate.to_tile_tensor()
        var iro_tt = input_row_offsets.to_tile_tensor[.uint32]()
        var clen_tt = cache_lengths.to_tile_tensor[.uint32]()
        var slot_tt = slot_idx.to_tile_tensor[.uint32]()

        var batch_size = Int(input_row_offsets.shape()[0].value()) - 1
        if batch_size == 0:
            # A step can legitimately carry no requests, and a zero-width
            # grid is a launch error.
            return

        ctx.enqueue_function[
            kpool_seed_tail_kernel[
                type_of(k).dtype,
                tail_tt.LayoutType,
                tail_tt.origin,
                type_of(k_tt.as_imm()).LayoutType,
                ImmOrigin(k_tt.origin),
                type_of(gate_tt.as_imm()).LayoutType,
                ImmOrigin(gate_tt.origin),
                type_of(iro_tt.as_imm()).LayoutType,
                ImmOrigin(iro_tt.origin),
                type_of(clen_tt.as_imm()).LayoutType,
                type_of(slot_tt.as_imm()).LayoutType,
                ImmOrigin(slot_tt.origin),
                tail_tt.Engine,
                type_of(k_tt.as_imm()).Engine,
                type_of(gate_tt.as_imm()).Engine,
                type_of(iro_tt.as_imm()).Engine,
                type_of(clen_tt.as_imm()).Engine,
                type_of(slot_tt.as_imm()).Engine,
                head_dim,
                kpool,
            ]
        ](
            tail_tt,
            k_tt.as_imm(),
            gate_tt.as_imm(),
            iro_tt.as_imm(),
            clen_tt.as_imm(),
            slot_tt.as_imm(),
            Int32(batch_size),
            grid_dim=(batch_size, 1, 1),
            block_dim=(head_dim, 1, 1),
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )


@extensibility.register("attention.mla.kpool.ring_close")
struct MLAKPoolRingClose:
    """Registers the `attention.mla.kpool.ring_close` graph op with the graph
    compiler.
    """

    @staticmethod
    def execute[
        *,
        head_dim: Int,
        kpool: Int,
    ](
        pooled: Some[Output & Tensor],
        closed_pool: Some[Output & Tensor],
        # `tail` is only read here, but `attention.mla.kpool.seed_tail` mutates the
        # same buffer in this same step (to extend whatever pool this op
        # leaves pending), so it must stay a `MutableInput` argument to get
        # correct ordering against that writer -- mirroring how
        # `MLAIndexerRaggedFloat8Paged` above reads `k_blocks`/`k_scales`
        # through `MutableInputTensor` even though it never writes them,
        # because `kv_cache.store.paged.ragged` does.
        tail: Some[MutableInput & Tensor],
        k: Some[Input & Tensor],
        gate: Some[Input & Tensor],
        ape: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        slot_idx: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        """Closes each request's pending tail-ring pool, ragged and
        unconditional -- see `kpool_ring_close_kernel`'s docstring for the
        pool-splitting semantics this implements.

        Parameters:
            head_dim: Channels per key; also the kernel's block width.
            kpool: Tokens per pool.

        Args:
            pooled: Output `[batch, head_dim]` pooled keys, meaningful only
                where `closed_pool` is non-negative.
            closed_pool: Output `[batch]` pool id closed this call, or -1.
            tail: Persistent slot-indexed ring `[max_slots, 2, kpool,
                head_dim]`. Read only.
            k: This call's layer-normed keys `[total_tokens, head_dim]`.
            gate: This call's gate scores `[total_tokens, head_dim]`.
            ape: Within-pool position embedding `[kpool, head_dim]`, float32.
            input_row_offsets: Token row offsets per request `[batch + 1]`.
            cache_lengths: Cached-prefix length per request `[batch]`.
            slot_idx: Ring slot per request `[batch]`.
            ctx: Device context for GPU execution.
        """
        comptime assert conforms_to(type_of(pooled), TileTensorable)
        comptime assert conforms_to(type_of(closed_pool), TileTensorable)
        comptime assert conforms_to(type_of(tail), TileTensorable)
        comptime assert conforms_to(type_of(k), TileTensorable)
        comptime assert conforms_to(type_of(gate), TileTensorable)
        comptime assert conforms_to(type_of(ape), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(slot_idx), TileTensorable)
        comptime assert type_of(pooled).rank == 2
        comptime assert type_of(closed_pool).rank == 1
        comptime assert type_of(tail).rank == 4
        comptime assert type_of(k).rank == 2
        comptime assert type_of(gate).rank == 2
        comptime assert type_of(ape).rank == 2
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(slot_idx).rank == 1
        var pooled_tt = pooled.to_tile_tensor()
        var closed_tt = closed_pool.to_tile_tensor[.int32]()
        var tail_tt = tail.to_tile_tensor()
        var k_tt = k.to_tile_tensor()
        var gate_tt = gate.to_tile_tensor()
        var ape_tt = ape.to_tile_tensor[.float32]()
        var iro_tt = input_row_offsets.to_tile_tensor[.uint32]()
        var clen_tt = cache_lengths.to_tile_tensor[.uint32]()
        var slot_tt = slot_idx.to_tile_tensor[.uint32]()

        var batch_size = Int(pooled.shape()[0].value())
        if batch_size == 0:
            # A step can legitimately carry no requests, and a zero-width
            # grid is a launch error.
            return

        ctx.enqueue_function[
            kpool_ring_close_kernel[
                type_of(k).dtype,
                pooled_tt.LayoutType,
                pooled_tt.origin,
                closed_tt.LayoutType,
                closed_tt.origin,
                type_of(tail_tt.as_imm()).LayoutType,
                ImmOrigin(tail_tt.origin),
                type_of(k_tt.as_imm()).LayoutType,
                ImmOrigin(k_tt.origin),
                type_of(gate_tt.as_imm()).LayoutType,
                ImmOrigin(gate_tt.origin),
                type_of(ape_tt.as_imm()).LayoutType,
                ImmOrigin(ape_tt.origin),
                type_of(iro_tt.as_imm()).LayoutType,
                ImmOrigin(iro_tt.origin),
                type_of(clen_tt.as_imm()).LayoutType,
                type_of(slot_tt.as_imm()).LayoutType,
                ImmOrigin(slot_tt.origin),
                pooled_tt.Engine,
                closed_tt.Engine,
                type_of(tail_tt.as_imm()).Engine,
                type_of(k_tt.as_imm()).Engine,
                type_of(gate_tt.as_imm()).Engine,
                type_of(ape_tt.as_imm()).Engine,
                type_of(iro_tt.as_imm()).Engine,
                type_of(clen_tt.as_imm()).Engine,
                type_of(slot_tt.as_imm()).Engine,
                head_dim,
                kpool,
            ]
        ](
            pooled_tt,
            closed_tt,
            tail_tt.as_imm(),
            k_tt.as_imm(),
            gate_tt.as_imm(),
            ape_tt.as_imm(),
            iro_tt.as_imm(),
            clen_tt.as_imm(),
            slot_tt.as_imm(),
            Int32(batch_size),
            grid_dim=(batch_size, 1, 1),
            block_dim=(head_dim, 1, 1),
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )


@inline(.always)
def _mla_indexer_ragged_float8_paged_execute[
    R: SparseIndexRemap,
    Topk: IndexerTopk = BlockSplitTopk,
    //,
    *,
    num_heads: Int,
    depth: Int,
    k: Int,
    quantization_granularity: Int,
    mask_str: StaticString,
    kpool: Int,
    write_logical: Bool,
](
    output_indices: TileTensor[
        mut=True, DType.int32, address_space=.GENERIC, ...
    ],
    q: TileTensor[mut=False, DType.float8_e4m3fn, address_space=.GENERIC, ...],
    qs: TileTensor[mut=False, DType.float32, address_space=.GENERIC, ...],
    input_row_offsets: TileTensor[
        mut=False, DType.uint32, address_space=.GENERIC, ...
    ],
    k_blocks: TileTensor[
        mut=True, DType.float8_e4m3fn, address_space=.GENERIC, ...
    ],
    page_stride: TileTensor[
        mut=False, DType.int64, address_space=.GENERIC, ...
    ],
    k_cache_lengths: TileTensor[
        mut=False, DType.uint32, address_space=.GENERIC, ...
    ],
    k_lookup_table: TileTensor[
        mut=False, DType.uint32, address_space=.GENERIC, ...
    ],
    k_max_prompt_length: TileTensor[
        mut=False, DType.uint32, address_space=.GENERIC, ...
    ],
    k_max_cache_length: TileTensor[
        mut=False, DType.uint32, address_space=.GENERIC, ...
    ],
    k_scales: TileTensor[mut=True, DType.float32, address_space=.GENERIC, ...],
    scales_page_stride: TileTensor[
        mut=False, DType.int64, address_space=.GENERIC, ...
    ],
    # The scales collection stores both lookup tables under one origin.
    k_scales_lookup_table: TileTensor[
        mut=False,
        DType.uint32,
        _,
        k_lookup_table.origin,
        address_space=.GENERIC,
        ...,
    ],
    layer_idx: UInt32,
    remap: R,
    ctx: DeviceContext,
) raises:
    """Shared body of the indexer ops.

    `remap` is the optional second output; `Topk` is the top-k dispatch,
    which the op's registration picks.
    """
    comptime assert output_indices.rank == 2
    comptime assert q.rank == 3
    comptime assert qs.rank == 2
    comptime assert input_row_offsets.rank == 1
    comptime assert k_blocks.rank == 6
    comptime assert page_stride.rank == 1
    comptime assert k_cache_lengths.rank == 1
    comptime assert k_lookup_table.rank == 2
    comptime assert k_max_prompt_length.rank == 1
    comptime assert k_max_cache_length.rank == 1
    comptime assert k_scales.rank == 6
    comptime assert scales_page_stride.rank == 1
    comptime assert k_scales_lookup_table.rank == 2
    # Extract cache parameters from block shapes
    comptime page_size = k_blocks.static_shape[3]
    comptime head_dim = k_blocks.static_shape[5]
    comptime k_num_heads = k_blocks.static_shape[4]
    comptime is_mla = k_blocks.static_shape[1] == 1
    comptime kv_params = KVCacheStaticParams(k_num_heads, head_dim, is_mla)
    comptime assert quantization_granularity >= depth, (
        "quantization_granularity must be >= depth for MLA (one scale per"
        " token per head)"
    )

    # K cache with scales (k_s values are stored in k_collection.scales). The
    # builder takes pointer-backed tiles; a non-pointer storage policy aborts
    # here at run time.
    var k_collection = generic_get_paged_cache_with_scales[
        DType.float8_e4m3fn,
        DType.float32,
        kv_params,
        page_size,
        quantization_granularity,
    ](
        _mut_pointer_tile(k_blocks),
        _imm_pointer_tile(page_stride),
        _imm_pointer_tile(k_cache_lengths),
        _imm_pointer_tile(k_lookup_table),
        _imm_pointer_tile(k_max_prompt_length),
        _imm_pointer_tile(k_max_cache_length),
        _mut_pointer_tile(k_scales),
        _imm_pointer_tile(scales_page_stride),
        _imm_pointer_tile(k_scales_lookup_table),
    )

    mla_indexer_ragged_float8_paged[
        DType.float8_e4m3fn,
        type_of(k_collection),
        num_heads,
        depth,
        k,
        mask_str,
        kpool=kpool,
        write_logical=write_logical,
        Topk=Topk,
    ](
        _mut_pointer_tile(output_indices),
        _imm_pointer_tile(q),
        _imm_pointer_tile(qs),
        _imm_pointer_tile(input_row_offsets),
        k_collection,
        layer_idx,
        ctx,
        remap,
    )


@extensibility.register("attention.mla.indexer.ragged.float8.paged")
struct MLAIndexerRaggedFloat8Paged:
    """Registers the `attention.mla.indexer.ragged.float8.paged` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        *,
        num_heads: Int,
        depth: Int,
        k: Int,
        quantization_granularity: Int,
        mask_str: StaticString,
        kpool: Int = 1,
    ](
        output_indices: Some[Output & Tensor],
        q: Some[Input & Tensor],
        qs: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        k_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        k_cache_lengths: Some[Input & Tensor],
        k_lookup_table: Some[Input & Tensor],
        k_max_prompt_length: Some[Input & Tensor],
        k_max_cache_length: Some[Input & Tensor],
        k_scales: Some[MutableInput & Tensor],
        scales_page_stride: Some[Input & Tensor],
        # Resolves a request's scale pages. Pass `k_lookup_table` itself when
        # the scales share the values' block-id space; pass a distinct table
        # when they are paged independently.
        k_scales_lookup_table: Some[Input & Tensor],
        layer_idx: UInt32,
        ctx: DeviceContext,
    ) raises:
        """Compute FP8 attention scores and return top-k key indices per token.

        This kernel is designed for Multi-head Latent Attention (MLA) architectures.
        It computes FP8 matmul between queries and cached keys (with scales), applies
        masking, and returns the indices of the top-k highest-scoring keys per token.
        Scores are aggregated (summed) across all attention heads.

        Parameters:
            num_heads: Number of query attention heads (must be 128).
            depth: Head dimension (must be 128).
            k: Number of top indices to return per token.
            quantization_granularity: Quantization granularity for the K cache.
            mask_str: Mask type - either MaskName.NULL (no mask) or MaskName.CAUSAL.
            kpool: Tokens per pooled cache row. `1` scores one row per token;
                `k > 1` scores one pooled key per `kpool` consecutive tokens,
                so the K cache holds pooled keys and `k` counts pools.

        Args:
            output_indices: Output tensor [total_seq_len, top_k] containing
                top-k key indices per token. Invalid positions (where there are
                fewer than top_k valid keys) are filled with -1.
            q: Query tensor [total_seq_len, num_heads, depth] in FP8.
            qs: Query scales [total_seq_len, num_heads] in float32.
            input_row_offsets: Ragged row offsets [batch_size + 1] for queries.
            k_blocks: Paged K cache blocks [num_blocks, 1, num_layers, page_size,
                num_heads, head_size] in FP8.
            page_stride: Page-to-page distance in elements; -1 when packed.
            k_cache_lengths: Cache lengths [batch_size] - number of cached tokens
                per sequence.
            k_lookup_table: Page lookup table [batch_size, pages_per_seq] mapping
                sequence pages to block indices.
            k_max_prompt_length: Max prompt (query) length scalar tensor [1].
            k_max_cache_length: Max cache length scalar tensor [1].
            k_scales: K scale blocks matching k_blocks shape with scale values.
            scales_page_stride: The same, for the scales leaf.
            k_scales_lookup_table: Page lookup table for the scale blocks
                [batch_size, pages_per_seq]. Equal to `k_lookup_table` when the
                scales share the values' block-id space.
            layer_idx: Layer index for retrieving the correct cache layer.
            ctx: Device context for GPU execution.
        """
        comptime assert conforms_to(type_of(output_indices), TileTensorable)
        comptime assert conforms_to(type_of(q), TileTensorable)
        comptime assert conforms_to(type_of(qs), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(k_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(k_cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(k_lookup_table), TileTensorable)
        comptime assert conforms_to(
            type_of(k_max_prompt_length), TileTensorable
        )
        comptime assert conforms_to(type_of(k_max_cache_length), TileTensorable)
        comptime assert conforms_to(type_of(k_scales), TileTensorable)
        comptime assert conforms_to(type_of(scales_page_stride), TileTensorable)
        comptime assert conforms_to(
            type_of(k_scales_lookup_table), TileTensorable
        )
        comptime assert type_of(output_indices).rank == 2
        comptime assert type_of(q).rank == 3
        comptime assert type_of(qs).rank == 2
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(k_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(k_cache_lengths).rank == 1
        comptime assert type_of(k_lookup_table).rank == 2
        comptime assert type_of(k_max_prompt_length).rank == 1
        comptime assert type_of(k_max_cache_length).rank == 1
        comptime assert type_of(k_scales).rank == 6
        comptime assert type_of(scales_page_stride).rank == 1
        comptime assert type_of(k_scales_lookup_table).rank == 2
        _mla_indexer_ragged_float8_paged_execute[
            num_heads=num_heads,
            depth=depth,
            k=k,
            quantization_granularity=quantization_granularity,
            mask_str=mask_str,
            kpool=kpool,
            write_logical=True,
        ](
            output_indices.to_tile_tensor[.int32](),
            q.to_tile_tensor[.float8_e4m3fn](),
            qs.to_tile_tensor[.float32](),
            input_row_offsets.to_tile_tensor[.uint32](),
            k_blocks.to_tile_tensor[.float8_e4m3fn](),
            page_stride.to_tile_tensor[.int64](),
            k_cache_lengths.to_tile_tensor[.uint32](),
            k_lookup_table.to_tile_tensor[.uint32](),
            k_max_prompt_length.to_tile_tensor[.uint32](),
            k_max_cache_length.to_tile_tensor[.uint32](),
            k_scales.to_tile_tensor[.float32](),
            scales_page_stride.to_tile_tensor[.int64](),
            k_scales_lookup_table.to_tile_tensor[.uint32](),
            layer_idx,
            NoRemap(),
            ctx,
        )


@extensibility.register("attention.mla.indexer.ragged.float8.paged.remap")
struct MLAIndexerRaggedFloat8PagedRemap:
    """Registers the `attention.mla.indexer.ragged.float8.paged.remap` graph op.

    `attention.mla.indexer.ragged.float8.paged` plus a second output: each selected
    position's physical row in the MLA cache, exactly what
    `paged_sparse_kv_index_remap` would compute from the logical output. A
    sparse MLA op given it skips that remap.
    """

    @staticmethod
    def execute[
        *,
        num_heads: Int,
        depth: Int,
        k: Int,
        quantization_granularity: Int,
        mask_str: StaticString,
        write_logical: Bool = False,
    ](
        output_indices: Some[Output & Tensor],
        physical_indices: Some[Output & Tensor],
        q: Some[Input & Tensor],
        qs: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        k_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        k_cache_lengths: Some[Input & Tensor],
        k_lookup_table: Some[Input & Tensor],
        k_max_prompt_length: Some[Input & Tensor],
        k_max_cache_length: Some[Input & Tensor],
        k_scales: Some[MutableInput & Tensor],
        scales_page_stride: Some[Input & Tensor],
        k_scales_lookup_table: Some[Input & Tensor],
        layer_idx: UInt32,
        mla_kv_blocks: Some[MutableInput & Tensor],
        mla_lookup_table: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        """Computes top-k key positions and their physical MLA cache rows.

        Parameters:
            num_heads: Number of query attention heads.
            depth: Head dimension.
            k: Number of top indices to return per token.
            quantization_granularity: Quantization granularity for the K cache.
            mask_str: Mask type - either MaskName.NULL or MaskName.CAUSAL.
            write_logical: Whether to write `output_indices` (defaults to
                `False`). With `False` only the physical rows are written, and
                `output_indices` may have no rows.

        Args:
            output_indices: Logical top-k key positions `[total_seq_len, k]`,
                `-1` past a token's candidates.
            physical_indices: Physical MLA cache row of each `output_indices`
                slot, `block_id * page_size + offset`, `-1` where the
                position is `-1` or its page is unmapped.
            q: Query tensor [total_seq_len, num_heads, depth] in FP8.
            qs: Query scales [total_seq_len, num_heads] in float32.
            input_row_offsets: Ragged row offsets [batch_size + 1] for queries.
            k_blocks: Paged indexer K cache blocks in FP8.
            page_stride: Page-to-page distance in elements; -1 when packed.
            k_cache_lengths: Cache lengths [batch_size].
            k_lookup_table: Indexer cache page lookup table.
            k_max_prompt_length: Max prompt (query) length scalar tensor [1].
            k_max_cache_length: Max cache length scalar tensor [1].
            k_scales: K scale blocks matching k_blocks shape with scale values.
            scales_page_stride: The same, for the scales leaf.
            k_scales_lookup_table: Page lookup table for the scale blocks.
            layer_idx: Layer index for retrieving the correct cache layer.
            mla_kv_blocks: The MLA cache's blocks; their page count is the
                invalid block id and their page size the remap's.
            mla_lookup_table: The MLA cache's page lookup table
                `[batch_size, pages_per_seq]`.
            ctx: Device context for GPU execution.
        """
        comptime assert conforms_to(type_of(output_indices), TileTensorable)
        comptime assert conforms_to(type_of(physical_indices), TileTensorable)
        comptime assert conforms_to(type_of(q), TileTensorable)
        comptime assert conforms_to(type_of(qs), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(k_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(k_cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(k_lookup_table), TileTensorable)
        comptime assert conforms_to(
            type_of(k_max_prompt_length), TileTensorable
        )
        comptime assert conforms_to(type_of(k_max_cache_length), TileTensorable)
        comptime assert conforms_to(type_of(k_scales), TileTensorable)
        comptime assert conforms_to(type_of(scales_page_stride), TileTensorable)
        comptime assert conforms_to(
            type_of(k_scales_lookup_table), TileTensorable
        )
        comptime assert conforms_to(type_of(mla_kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(mla_lookup_table), TileTensorable)
        comptime assert type_of(output_indices).rank == 2
        comptime assert type_of(physical_indices).rank == 2
        comptime assert type_of(q).rank == 3
        comptime assert type_of(qs).rank == 2
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(k_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(k_cache_lengths).rank == 1
        comptime assert type_of(k_lookup_table).rank == 2
        comptime assert type_of(k_max_prompt_length).rank == 1
        comptime assert type_of(k_max_cache_length).rank == 1
        comptime assert type_of(k_scales).rank == 6
        comptime assert type_of(scales_page_stride).rank == 1
        comptime assert type_of(k_scales_lookup_table).rank == 2
        comptime assert type_of(mla_kv_blocks).rank == 6
        comptime assert type_of(mla_lookup_table).rank == 2
        comptime mla_page_size = type_of(mla_kv_blocks)._shape_types[
            3
        ].static_value
        # The remap reads the index tables through their flat pointers; a
        # non-pointer storage policy aborts here at run time.
        var remap = PagedKVRemap[mla_page_size](
            physical_indices.to_tile_tensor[.int32]().unsafe_ptr(),
            mla_lookup_table.to_tile_tensor[.uint32]().unsafe_ptr(),
            Int(mla_lookup_table.shape()[1].value()),
            Int(mla_lookup_table.shape()[0].value()),
            UInt32(Int(mla_kv_blocks.shape()[0].value())),
        )
        _mla_indexer_ragged_float8_paged_execute[
            num_heads=num_heads,
            depth=depth,
            k=k,
            quantization_granularity=quantization_granularity,
            mask_str=mask_str,
            kpool=1,
            write_logical=write_logical,
        ](
            output_indices.to_tile_tensor[.int32](),
            q.to_tile_tensor[.float8_e4m3fn](),
            qs.to_tile_tensor[.float32](),
            input_row_offsets.to_tile_tensor[.uint32](),
            k_blocks.to_tile_tensor[.float8_e4m3fn](),
            page_stride.to_tile_tensor[.int64](),
            k_cache_lengths.to_tile_tensor[.uint32](),
            k_lookup_table.to_tile_tensor[.uint32](),
            k_max_prompt_length.to_tile_tensor[.uint32](),
            k_max_cache_length.to_tile_tensor[.uint32](),
            k_scales.to_tile_tensor[.float32](),
            scales_page_stride.to_tile_tensor[.int64](),
            k_scales_lookup_table.to_tile_tensor[.uint32](),
            layer_idx,
            remap,
            ctx,
        )


@extensibility.register("attention.mla.indexer.ragged.float8.paged.remap.cp")
struct MLAIndexerRaggedFloat8PagedRemapCP:
    """Registers the `attention.mla.indexer.ragged.float8.paged.remap.cp` graph op.

    `attention.mla.indexer.ragged.float8.paged.remap` for one rank of a decode
    context-parallel (DCP) group. The rank's caches hold only the positions it
    owns, at their local slots (`kv_cache.context_parallel`), behind the
    group's shared lookup tables and global cache lengths. The op selects the
    top-k of those local slots, returns their physical rows, and also returns
    their scores, which the group all-gathers for `attention.mla.cp_merge_topk`.
    """

    @staticmethod
    def execute[
        *,
        num_heads: Int,
        depth: Int,
        k: Int,
        quantization_granularity: Int,
        mask_str: StaticString,
        dcp_size: Int,
        dcp_interleave: Int,
        write_logical: Bool = False,
    ](
        output_indices: Some[Output & Tensor],
        physical_indices: Some[Output & Tensor],
        scores: Some[Output & Tensor],
        q: Some[Input & Tensor],
        qs: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        k_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        k_cache_lengths: Some[Input & Tensor],
        k_lookup_table: Some[Input & Tensor],
        k_max_prompt_length: Some[Input & Tensor],
        k_max_cache_length: Some[Input & Tensor],
        k_scales: Some[MutableInput & Tensor],
        scales_page_stride: Some[Input & Tensor],
        k_scales_lookup_table: Some[Input & Tensor],
        layer_idx: UInt32,
        mla_kv_blocks: Some[MutableInput & Tensor],
        mla_lookup_table: Some[Input & Tensor],
        dcp_rank: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        """Selects this rank's local top-k physical rows and their scores.

        Parameters:
            num_heads: Number of query attention heads.
            depth: Head dimension.
            k: Number of local slots to select per token.
            quantization_granularity: Quantization granularity for the K cache.
            mask_str: Mask type - either MaskName.NULL or MaskName.CAUSAL.
            dcp_size: Ranks in the decode context-parallel group.
            dcp_interleave: Consecutive positions a rank owns in turn.
            write_logical: Whether to write `output_indices`, the selected
                local slots (defaults to `False`, which allows no rows).

        Args:
            output_indices: Selected local slots `[total_seq_len, k]`.
            physical_indices: Physical MLA cache row of each selection
                `[total_seq_len, k]`, `-1` past a token's local candidates.
            scores: Each selection's bf16 score `[total_seq_len, k]`, `-inf`
                past a token's local candidates.
            q: Query tensor [total_seq_len, num_heads, depth] in FP8.
            qs: Query scales [total_seq_len, num_heads] in float32.
            input_row_offsets: Ragged row offsets [batch_size + 1] for queries.
            k_blocks: This rank's paged indexer K cache blocks in FP8.
            page_stride: Page-to-page distance in elements; -1 when packed.
            k_cache_lengths: Global cache lengths [batch_size].
            k_lookup_table: Indexer cache page lookup table.
            k_max_prompt_length: Max prompt (query) length scalar tensor [1].
            k_max_cache_length: Global max cache length scalar tensor [1].
            k_scales: K scale blocks matching k_blocks shape with scale values.
            scales_page_stride: The same, for the scales leaf.
            k_scales_lookup_table: Page lookup table for the scale blocks.
            layer_idx: Layer index for retrieving the correct cache layer.
            mla_kv_blocks: This rank's MLA cache blocks.
            mla_lookup_table: The MLA cache's page lookup table
                `[batch_size, pages_per_seq]`.
            dcp_rank: This rank in `[0, dcp_size)`, int32 `[1]` on the device.
            ctx: Device context for GPU execution.
        """
        comptime assert conforms_to(type_of(output_indices), TileTensorable)
        comptime assert conforms_to(type_of(physical_indices), TileTensorable)
        comptime assert conforms_to(type_of(scores), TileTensorable)
        comptime assert conforms_to(type_of(q), TileTensorable)
        comptime assert conforms_to(type_of(qs), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(k_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(k_cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(k_lookup_table), TileTensorable)
        comptime assert conforms_to(
            type_of(k_max_prompt_length), TileTensorable
        )
        comptime assert conforms_to(type_of(k_max_cache_length), TileTensorable)
        comptime assert conforms_to(type_of(k_scales), TileTensorable)
        comptime assert conforms_to(type_of(scales_page_stride), TileTensorable)
        comptime assert conforms_to(
            type_of(k_scales_lookup_table), TileTensorable
        )
        comptime assert conforms_to(type_of(mla_kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(mla_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(dcp_rank), TileTensorable)
        comptime assert type_of(output_indices).rank == 2
        comptime assert type_of(physical_indices).rank == 2
        comptime assert type_of(scores).rank == 2
        comptime assert type_of(q).rank == 3
        comptime assert type_of(qs).rank == 2
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(k_blocks).rank == 6
        comptime assert type_of(mla_kv_blocks).rank == 6
        comptime assert type_of(mla_lookup_table).rank == 2
        comptime assert type_of(dcp_rank).rank == 1
        comptime page_size = type_of(k_blocks)._shape_types[3].static_value
        comptime head_dim = type_of(k_blocks)._shape_types[5].static_value
        comptime k_num_heads = type_of(k_blocks)._shape_types[4].static_value
        comptime is_mla = type_of(k_blocks)._shape_types[1].static_value == 1
        comptime kv_params = KVCacheStaticParams(k_num_heads, head_dim, is_mla)
        comptime mla_page_size = type_of(mla_kv_blocks)._shape_types[
            3
        ].static_value
        var k_collection = generic_get_paged_cache_with_scales[
            DType.float8_e4m3fn,
            DType.float32,
            kv_params,
            page_size,
            quantization_granularity,
        ](
            _mut_pointer_tile(k_blocks.to_tile_tensor[.float8_e4m3fn]()),
            _imm_pointer_tile(page_stride.to_tile_tensor[.int64]()),
            _imm_pointer_tile(k_cache_lengths.to_tile_tensor[.uint32]()),
            _imm_pointer_tile(k_lookup_table.to_tile_tensor[.uint32]()),
            _imm_pointer_tile(k_max_prompt_length.to_tile_tensor[.uint32]()),
            _imm_pointer_tile(k_max_cache_length.to_tile_tensor[.uint32]()),
            _mut_pointer_tile(k_scales.to_tile_tensor[.float32]()),
            _imm_pointer_tile(scales_page_stride.to_tile_tensor[.int64]()),
            _imm_pointer_tile(k_scales_lookup_table.to_tile_tensor[.uint32]()),
        )
        # The remap reads the index tables through their flat pointers; a
        # non-pointer storage policy aborts here at run time.
        var remap = PagedKVRemap[mla_page_size](
            physical_indices.to_tile_tensor[.int32]().unsafe_ptr(),
            mla_lookup_table.to_tile_tensor[.uint32]().unsafe_ptr(),
            Int(mla_lookup_table.shape()[1].value()),
            Int(mla_lookup_table.shape()[0].value()),
            UInt32(Int(mla_kv_blocks.shape()[0].value())),
        )
        mla_indexer_ragged_float8_paged_cp[
            DType.float8_e4m3fn,
            type_of(k_collection),
            num_heads,
            depth,
            k,
            mask_str,
            dcp_size,
            dcp_interleave,
            write_logical=write_logical,
        ](
            _mut_pointer_tile(output_indices.to_tile_tensor[.int32]()),
            _mut_pointer_tile(scores.to_tile_tensor[.bfloat16]()),
            _imm_pointer_tile(q.to_tile_tensor[.float8_e4m3fn]()),
            _imm_pointer_tile(qs.to_tile_tensor[.float32]()),
            _imm_pointer_tile(input_row_offsets.to_tile_tensor[.uint32]()),
            k_collection,
            layer_idx,
            ctx,
            remap,
            dcp_rank.to_tile_tensor[.int32]().unsafe_ptr().as_imm(),
        )


@extensibility.register("attention.mla.cp_merge_topk")
struct MLACPMergeTopk:
    """Registers the `attention.mla.cp_merge_topk` graph op.

    Merges a decode context-parallel group's local indexer selections (see
    `attention.mla.indexer.ragged.float8.paged.remap.cp`) into the group's top-k and
    keeps this rank's share, compacted to the front of each row.
    """

    @staticmethod
    def execute[
        *,
        dcp_size: Int,
        k: Int,
    ](
        rows: Some[Output & Tensor],
        lengths: Some[Output & Tensor],
        local_rows: Some[Input & Tensor],
        scores: Some[Input & Tensor],
        dcp_rank: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        """Keeps this rank's selections that are in the group's top-k.

        Parameters:
            dcp_size: Ranks in the decode context-parallel group.
            k: Selections per token, on each rank and after the merge.

        Args:
            rows: `[T, k]` int32 rows of this rank's selections in the
                group's top-k, compacted to the front, then `-1`.
            lengths: `[T]` int32 count of `rows`' selected slots.
            local_rows: `[T, k]` int32 this rank's local selections.
            scores: `[T, dcp_size * k]` bf16 every rank's selection scores,
                rank-major, `-inf` for padding.
            dcp_rank: This rank in `[0, dcp_size)`, int32 `[1]` on the device.
            ctx: Device context for GPU execution.
        """
        comptime assert conforms_to(type_of(rows), TileTensorable)
        comptime assert conforms_to(type_of(lengths), TileTensorable)
        comptime assert conforms_to(type_of(local_rows), TileTensorable)
        comptime assert conforms_to(type_of(scores), TileTensorable)
        comptime assert conforms_to(type_of(dcp_rank), TileTensorable)
        comptime assert type_of(rows).rank == 2
        comptime assert type_of(lengths).rank == 1
        comptime assert type_of(local_rows).rank == 2
        comptime assert type_of(scores).rank == 2
        comptime assert type_of(dcp_rank).rank == 1
        cp_merge_topk[dcp_size, k](
            _mut_pointer_tile(rows.to_tile_tensor[.int32]()),
            _mut_pointer_tile(lengths.to_tile_tensor[.int32]()),
            _imm_pointer_tile(local_rows.to_tile_tensor[.int32]()),
            _imm_pointer_tile(scores.to_tile_tensor[.bfloat16]()),
            dcp_rank.to_tile_tensor[.int32]().unsafe_ptr().as_imm(),
            ctx,
        )


@extensibility.register("attention.mla.indexer.ragged.float8.paged.remap.nvfp4")
struct MLAIndexerRaggedFloat8PagedRemapNVFP4:
    """Registers the `attention.mla.indexer.ragged.float8.paged.remap.nvfp4` graph op.

    `attention.mla.indexer.ragged.float8.paged.remap` for an NVFP4 MLA cache, whose
    scales are paged apart from its values. Each row of `physical_indices` is
    `2 * k` wide: the selected positions' physical rows in the values, then
    their rows in the scale pool, resolved through the cache's scales lookup
    table. The sparse NVFP4 MLA kernels read the second half instead of
    deriving a scale row from a value row.
    """

    @staticmethod
    def execute[
        *,
        num_heads: Int,
        depth: Int,
        k: Int,
        quantization_granularity: Int,
        mask_str: StaticString,
        write_logical: Bool = False,
    ](
        output_indices: Some[Output & Tensor],
        physical_indices: Some[Output & Tensor],
        q: Some[Input & Tensor],
        qs: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        k_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        k_cache_lengths: Some[Input & Tensor],
        k_lookup_table: Some[Input & Tensor],
        k_max_prompt_length: Some[Input & Tensor],
        k_max_cache_length: Some[Input & Tensor],
        k_scales: Some[MutableInput & Tensor],
        scales_page_stride: Some[Input & Tensor],
        k_scales_lookup_table: Some[Input & Tensor],
        layer_idx: UInt32,
        mla_kv_blocks: Some[MutableInput & Tensor],
        mla_lookup_table: Some[Input & Tensor],
        mla_kv_scales: Some[MutableInput & Tensor],
        mla_scales_page_stride: Some[Input & Tensor],
        mla_scales_lookup_table: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        """Computes top-k key positions and their value and scale rows.

        Parameters:
            num_heads: Number of query attention heads.
            depth: Head dimension.
            k: Number of top indices to return per token.
            quantization_granularity: Quantization granularity for the K cache.
            mask_str: Mask type - either MaskName.NULL or MaskName.CAUSAL.
            write_logical: Whether to write `output_indices` (defaults to
                `False`). With `False` only `physical_indices` is written, and
                `output_indices` may have no rows.

        Args:
            output_indices: Logical top-k key positions `[total_seq_len, k]`,
                `-1` past a token's candidates.
            physical_indices: `[total_seq_len, 2 * k]`. Columns `[0, k)` are
                each `output_indices` slot's physical MLA cache row,
                `block_id * page_size + offset`; columns `[k, 2 * k)` are its
                row in the MLA scale pool, `scale_block_id * rows_per_block +
                offset`. `-1` where the position is `-1` or its page is
                unmapped.
            q: Query tensor [total_seq_len, num_heads, depth] in FP8.
            qs: Query scales [total_seq_len, num_heads] in float32.
            input_row_offsets: Ragged row offsets [batch_size + 1] for queries.
            k_blocks: Paged indexer K cache blocks in FP8.
            page_stride: Page-to-page distance in elements; -1 when packed.
            k_cache_lengths: Cache lengths [batch_size].
            k_lookup_table: Indexer cache page lookup table.
            k_max_prompt_length: Max prompt (query) length scalar tensor [1].
            k_max_cache_length: Max cache length scalar tensor [1].
            k_scales: K scale blocks matching k_blocks shape with scale values.
            scales_page_stride: The same, for the scales leaf.
            k_scales_lookup_table: Page lookup table for the scale blocks.
            layer_idx: Layer index for retrieving the correct cache layer.
            mla_kv_blocks: The MLA cache's blocks; their page count is the
                invalid block id and their page size the remap's.
            mla_lookup_table: The MLA cache's page lookup table
                `[batch_size, pages_per_seq]`.
            mla_kv_scales: The MLA cache's scale blocks; their page count is
                the invalid scale block id.
            mla_scales_page_stride: Page-to-page distance of `mla_kv_scales`
                in elements; -1 when packed.
            mla_scales_lookup_table: The MLA cache's scales page lookup table
                `[batch_size, pages_per_seq]`.
            ctx: Device context for GPU execution.
        """
        comptime assert conforms_to(type_of(output_indices), TileTensorable)
        comptime assert conforms_to(type_of(physical_indices), TileTensorable)
        comptime assert conforms_to(type_of(q), TileTensorable)
        comptime assert conforms_to(type_of(qs), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(k_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(k_cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(k_lookup_table), TileTensorable)
        comptime assert conforms_to(
            type_of(k_max_prompt_length), TileTensorable
        )
        comptime assert conforms_to(type_of(k_max_cache_length), TileTensorable)
        comptime assert conforms_to(type_of(k_scales), TileTensorable)
        comptime assert conforms_to(type_of(scales_page_stride), TileTensorable)
        comptime assert conforms_to(
            type_of(k_scales_lookup_table), TileTensorable
        )
        comptime assert conforms_to(type_of(mla_kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(mla_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(mla_kv_scales), TileTensorable)
        comptime assert conforms_to(
            type_of(mla_scales_page_stride), TileTensorable
        )
        comptime assert conforms_to(
            type_of(mla_scales_lookup_table), TileTensorable
        )
        comptime assert type_of(output_indices).rank == 2
        comptime assert type_of(physical_indices).rank == 2
        comptime assert type_of(q).rank == 3
        comptime assert type_of(qs).rank == 2
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(k_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(k_cache_lengths).rank == 1
        comptime assert type_of(k_lookup_table).rank == 2
        comptime assert type_of(k_max_prompt_length).rank == 1
        comptime assert type_of(k_max_cache_length).rank == 1
        comptime assert type_of(k_scales).rank == 6
        comptime assert type_of(scales_page_stride).rank == 1
        comptime assert type_of(k_scales_lookup_table).rank == 2
        comptime assert type_of(mla_kv_blocks).rank == 6
        comptime assert type_of(mla_lookup_table).rank == 2
        comptime assert type_of(mla_kv_scales).rank == 6
        comptime assert type_of(mla_kv_scales).dtype == DType.float8_e4m3fn
        comptime assert type_of(mla_scales_page_stride).rank == 1
        comptime assert type_of(mla_scales_lookup_table).rank == 2
        comptime mla_page_size = type_of(mla_kv_blocks)._shape_types[
            3
        ].static_value
        # The scale pool's blocks sit a page stride apart, which is
        # `rows_per_block` rows of `heads * scales_per_row` elements.
        var mla_kv_scales_tile = mla_kv_scales.to_tile_tensor()
        var scales_row_elems = Int(mla_kv_scales_tile.dim[4]()) * Int(
            mla_kv_scales_tile.dim[5]()
        )
        var scales_page_elems = Int(
            mla_scales_page_stride.to_tile_tensor[.int64]()
            .as_imm()
            .load[width=1](Coord(0))
        )
        if scales_page_elems < 0:
            scales_page_elems = scales_row_elems
            for i in range(1, 4):
                scales_page_elems *= Int(mla_kv_scales_tile.dim(i))
        # The remap reads the index tables through their flat pointers; a
        # non-pointer storage policy aborts here at run time.
        var remap = PagedKVScaleRemap[mla_page_size, k](
            physical_indices.to_tile_tensor[.int32]().unsafe_ptr(),
            mla_lookup_table.to_tile_tensor[.uint32]().unsafe_ptr(),
            mla_scales_lookup_table.to_tile_tensor[.uint32]().unsafe_ptr(),
            Int(mla_lookup_table.shape()[1].value()),
            Int(mla_scales_lookup_table.shape()[1].value()),
            Int(mla_lookup_table.shape()[0].value()),
            UInt32(Int(mla_kv_blocks.shape()[0].value())),
            UInt32(Int(mla_kv_scales.shape()[0].value())),
            scales_page_elems // scales_row_elems,
        )
        _mla_indexer_ragged_float8_paged_execute[
            num_heads=num_heads,
            depth=depth,
            k=k,
            quantization_granularity=quantization_granularity,
            mask_str=mask_str,
            kpool=1,
            write_logical=write_logical,
        ](
            output_indices.to_tile_tensor[.int32](),
            q.to_tile_tensor[.float8_e4m3fn](),
            qs.to_tile_tensor[.float32](),
            input_row_offsets.to_tile_tensor[.uint32](),
            k_blocks.to_tile_tensor[.float8_e4m3fn](),
            page_stride.to_tile_tensor[.int64](),
            k_cache_lengths.to_tile_tensor[.uint32](),
            k_lookup_table.to_tile_tensor[.uint32](),
            k_max_prompt_length.to_tile_tensor[.uint32](),
            k_max_cache_length.to_tile_tensor[.uint32](),
            k_scales.to_tile_tensor[.float32](),
            scales_page_stride.to_tile_tensor[.int64](),
            k_scales_lookup_table.to_tile_tensor[.uint32](),
            layer_idx,
            remap,
            ctx,
        )


@extensibility.register("attention.mha.no_cache")
struct FlashAttentionGPU:
    """Registers the `attention.mha.no_cache` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        MaskIns: TypeList[Trait=Input & Fused & LoadAccess, ...],
        //,
        mask_str: StaticString,
        local_window_size: Int = -1,
    ](
        output: Some[Output & Tensor],
        q: Some[Input & Tensor],
        k: Some[Input & Tensor],
        v: Some[Input & Tensor],
        scale: Float32,
        mask_operands: TensorTuple[*MaskIns],
        ctx: DeviceContext,
    ) raises:
        """`attention.mha.no_cache` is a hand-fused operator which does
        something analogous to the following list of operations.

        **Step 0:
        Transpose:
        query_processed = transpose(query) # BSHD --> BHSD
        key_processed = transpose(key)     # BSHD --> BHDS
        value_processed = transpose(value) # BSHD --> BHSD

        **Step 1:
        attentionMatrix = query_processed @ key_processed

        **Step 2:
        norm = broadcast_to(normScalar, shape_of(attentionMatrix))

        **Step 3:
        # Normalize and apply masking
        attentionMatrixNormMasked = mask_functor(attentionMatrix * scale)

        **Step 4:
        # Apply softmax and reproject result
        attentionMatrixSoftMax = softmax(attentionMatrixNormMasked)
        answer = attentionMatrixSoftMax @ value_processed
        answer = transpose(answer) # BHSD --> BSHD

        Finally, this pattern supports grouped attention patterns. That is if we
        have G groups, then let h = H / G. Key and value are allowed to be BShD
        in these scenarios. Both key and value must be BShD if one is. If this is
        true the following is equivalently run before Step 0:

        ** Step -1:
        key = concat(key, ...) # concat BShD --> BSHD
        value = concat(value, ...) # concat BShD --> BSHD

        The underlying fusion follows ideas taken from the 2022 FlashAttention paper
        by Tri Dao et al.

        Parameters:
            MaskIns: The mask operand types.
            mask_str: Attention mask type string passed to
                `dispatch_mask` to select the mask functor.
            local_window_size: Sliding window size for windowed attention
                (defaults to -1, meaning no window).

        Args:
            output: Output attention tensor with the same shape as `q`.
            q: Query tensor in BSHD layout.
            k: Key tensor in BSHD layout, or BShD for grouped attention.
            v: Value tensor in BSHD layout, or BShD for grouped attention.
            scale: Scaling factor applied to attention scores before softmax.
            mask_operands: The mask's operands; `dispatch_mask` lists what
                each mask reads.
            ctx: Device context for GPU execution.
        """
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(q), TileTensorable)
        comptime assert conforms_to(type_of(k), TileTensorable)
        comptime assert conforms_to(type_of(v), TileTensorable)
        comptime rank = type_of(output).rank
        comptime assert type_of(q).rank == rank
        comptime assert type_of(k).rank == rank
        comptime assert type_of(v).rank == rank
        comptime assert ctx.target.is_gpu(), "only valid on GPUs"

        var output_buffer = output.to_tile_tensor()
        var q_buffer = q.to_tile_tensor()
        var k_buffer = k.to_tile_tensor()
        var v_buffer = v.to_tile_tensor()

        def _dispatch_flash_attention[
            mask_t: MHAMask
        ](mask: mask_t) raises {
            var output_buffer,
            var q_buffer,
            var k_buffer,
            var v_buffer,
            imm,
        }:
            flash_attention[](
                output_buffer,
                q_buffer,
                k_buffer,
                v_buffer,
                mask,
                scale,
                ctx,
            )

        dispatch_mask[mask_str, local_window_size](
            mask_operands, _dispatch_flash_attention
        )


@extensibility.register("attention.mha.no_cache", type="cpu")
struct FlashAttentionCPU:
    """Registers the CPU kernel of the `attention.mha.no_cache` graph op with the
    graph compiler."""

    @staticmethod
    def execute[
        MaskIns: TypeList[Trait=Input & Fused & LoadAccess, ...],
        //,
        mask_str: StaticString,
        local_window_size: Int = -1,
    ](
        output: Some[Output & Tensor],
        q: Some[Input & Tensor],
        k: Some[Input & Fused & LoadAccess],
        v: Some[Input & Fused & LoadAccess],
        scale: Float32,
        mask_operands: TensorTuple[*MaskIns],
        ctx: DeviceContext,
    ) raises:
        """`attention.mha.no_cache` on CPU: the GPU kernel's computation, with `k`
        and `v` read through loads so that their producers fuse into them.

        Parameters:
            MaskIns: The mask operand types.
            mask_str: Attention mask type string passed to `dispatch_mask`.
            local_window_size: Sliding window size for windowed attention
                (defaults to -1, meaning no window).

        Args:
            output: Output attention tensor with the same shape as `q`.
            q: Query tensor in BSHD layout.
            k: Key tensor in BSHD layout, or BShD for grouped attention.
            v: Value tensor in BSHD layout, or BShD for grouped attention.
            scale: Scaling factor applied to attention scores before softmax.
            mask_operands: The mask's operands; `dispatch_mask` lists what
                each mask reads.
            ctx: Device context for CPU execution.
        """
        comptime assert ctx.target.is_cpu(), "only valid on CPUs"
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(q), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime rank = type_of(output).rank
        comptime assert type_of(q).dtype == dtype
        comptime assert type_of(k).dtype == dtype
        comptime assert type_of(v).dtype == dtype
        comptime assert type_of(q).rank == rank
        comptime assert type_of(k).rank == rank
        comptime assert type_of(v).rank == rank

        comptime assert conforms_to(type_of(k), UnsafeFusedView)
        var k_view = unsafe_fused_view(k)
        comptime assert conforms_to(type_of(k_view), Loadable)
        comptime assert conforms_to(type_of(v), UnsafeFusedView)
        var v_view = unsafe_fused_view(v)
        comptime assert conforms_to(type_of(v_view), Loadable)

        @inline(.always)
        def k_input_fn[
            width: Int
        ](coords: Coord) {var k_view} -> SIMD[dtype, width]:
            return rebind[SIMD[dtype, width]](k_view.load[width=width](coords))

        @inline(.always)
        def v_input_fn[
            width: Int
        ](coords: Coord) {var v_view} -> SIMD[dtype, width]:
            return rebind[SIMD[dtype, width]](v_view.load[width=width](coords))

        var k_shape = rebind[IndexList[rank]](coord_to_index_list(k.shape()))
        var v_shape = rebind[IndexList[rank]](coord_to_index_list(v.shape()))

        def _run[mask_t: MHAMask](mask: mask_t) raises {imm}:
            nn_flash_attention(
                k_input_fn,
                v_input_fn,
                q.to_tile_tensor[dtype]().as_imm(),
                k_shape,
                v_shape,
                output.to_tile_tensor[dtype](),
                mask,
                scale,
                ctx=Optional(ctx.assert_cpu_target()),
            )

        dispatch_mask[mask_str, local_window_size](mask_operands, _run)


@extensibility.register("attention.mha.padded.no_cache")
struct PaddedFlashAttentionGPU:
    """Registers the `attention.mha.padded.no_cache` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        mask_str: StaticString,
        local_window_size: Int = -1,
    ](
        output: Some[Output & Tensor],
        q: Some[Input & Tensor],
        k: Some[Input & Tensor],
        v: Some[Input & Tensor],
        valid_length: Some[Input & Tensor],
        scale: Float32,
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(q), TileTensorable)
        comptime assert conforms_to(type_of(k), TileTensorable)
        comptime assert conforms_to(type_of(v), TileTensorable)
        comptime assert conforms_to(type_of(valid_length), TileTensorable)
        comptime rank = type_of(output).rank
        comptime assert type_of(q).rank == rank
        comptime assert type_of(k).rank == rank
        comptime assert type_of(v).rank == rank
        comptime assert type_of(valid_length).rank == 1
        comptime assert ctx.target.is_gpu(), "only valid on GPUs"

        var output_buffer = output.to_tile_tensor()
        var q_buffer = q.to_tile_tensor()
        var k_buffer = k.to_tile_tensor()
        var v_buffer = v.to_tile_tensor()

        comptime valid_length_t = ImmutTileTensor1D[.uint32]
        # `valid_length` is read through its flat pointer; a non-pointer storage policy aborts here at run time.
        var _valid_length = immut_tile_tensor_1d(
            valid_length.to_tile_tensor[.uint32]().unsafe_ptr(),
            Int(valid_length.shape()[0].value()),
        )

        def _dispatch_flash_attention[
            mask_t: MHAMask
        ](mask: mask_t) raises {
            var output_buffer,
            var q_buffer,
            var k_buffer,
            var v_buffer,
            imm,
        }:
            flash_attention[
                _use_valid_length=True,
                _padded_ndbuffer=True,
            ](
                output_buffer,
                q_buffer,
                k_buffer,
                v_buffer,
                mask,
                scale,
                ctx,
                valid_length=OptionalReg[valid_length_t](_valid_length),
            )

        dispatch_mask[mask_str, local_window_size](_dispatch_flash_attention)


@extensibility.register("attention.mha.ragged.no_cache")
struct RaggedFlashAttentionGPU:
    """Registers the `attention.mha.ragged.no_cache` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        MaskIns: TypeList[Trait=Input & Fused & LoadAccess, ...],
        //,
        mask_str: StaticString,
        local_window_size: Int = -1,
    ](
        output: Some[Output & Tensor],
        q: Some[Input & Tensor],
        k: Some[Input & Tensor],
        v: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        q_max_seq_len: Some[Input & Tensor],
        scale: Float32,
        mask_operands: TensorTuple[*MaskIns],
        ctx: DeviceContext,
    ) raises:
        """`attention.mha.ragged.no_cache` computes flash attention for ragged inputs without KV cache.

        The inputs q, k, v are in ragged format with shape [total_seq_len, num_heads, head_dim].
        input_row_offsets indicates where each sequence starts and ends in the ragged tensors.

        Parameters:
            MaskIns: The mask operand types.
            mask_str: Attention mask type string passed to
                `dispatch_mask` to select the mask functor.
            local_window_size: Sliding window size for windowed attention
                (defaults to -1, meaning no window).

        Args:
            output: Output attention tensor with the same shape as `q`.
            q: Query tensor in ragged format with shape
                [total_seq_len, num_heads, head_dim].
            k: Key tensor in ragged format with shape
                [total_seq_len, num_heads, head_dim].
            v: Value tensor in ragged format with shape
                [total_seq_len, num_heads, head_dim].
            input_row_offsets: Row offsets [batch_size + 1] marking the
                start and end of each sequence in the ragged tensors.
            q_max_seq_len: Maximum query sequence length across the batch.
            scale: Scaling factor applied to attention scores before softmax.
            mask_operands: The mask's operands; `dispatch_mask` lists what
                each mask reads.
            ctx: Device context for GPU execution.
        """
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(q), TileTensorable)
        comptime assert conforms_to(type_of(k), TileTensorable)
        comptime assert conforms_to(type_of(v), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(q_max_seq_len), TileTensorable)
        comptime rank = type_of(output).rank
        comptime assert type_of(q).rank == rank
        comptime assert type_of(k).rank == rank
        comptime assert type_of(v).rank == rank
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(q_max_seq_len).rank == 1
        comptime assert ctx.target.is_gpu(), "only valid on GPUs"

        var output_buffer = output.to_tile_tensor()
        var q_buffer = q.to_tile_tensor()
        var k_buffer = k.to_tile_tensor()
        var v_buffer = v.to_tile_tensor()

        var _input_row_offsets = input_row_offsets.to_tile_tensor[.uint32]()

        def _dispatch_flash_attention[
            mask_t: MHAMask
        ](mask: mask_t) raises {
            var output_buffer,
            var q_buffer,
            var k_buffer,
            var v_buffer,
            imm,
        }:
            flash_attention_ragged[](
                output_buffer,
                q_buffer,
                k_buffer,
                v_buffer,
                _input_row_offsets,
                q_max_seq_len.to_tile_tensor[.uint32](),
                mask,
                scale,
                ctx,
            )

        dispatch_mask[mask_str, local_window_size](
            mask_operands, _dispatch_flash_attention
        )


@extensibility.register("kv_cache.fused_qkv_matmul.padded.paged")
struct Struct_fused_qkv_matmul_padded_paged:
    """Registers the `kv_cache.fused_qkv_matmul.padded.paged` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute(
        output: Some[Output & Tensor],
        hidden_state: Some[Input & Tensor],
        weight: Some[Input & Tensor],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        layer_idx: UInt32,
        valid_lengths: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(hidden_state), TileTensorable)
        comptime assert conforms_to(type_of(weight), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert conforms_to(type_of(valid_lengths), TileTensorable)
        comptime dtype = type_of(kv_blocks).dtype
        comptime assert type_of(output).rank == 3
        comptime assert type_of(hidden_state).rank == 3
        comptime assert type_of(weight).rank == 2
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(valid_lengths).rank == 1
        var kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )

        generic_fused_qkv_matmul_kv_cache_bshd_paged(
            hidden_state.to_tile_tensor[dtype](),
            weight.to_tile_tensor[dtype](),
            kv_collection,
            layer_idx,
            valid_lengths.to_tile_tensor[.uint32](),
            output.to_tile_tensor[dtype](),
            ctx,
        )


@extensibility.register("kv_cache.fused_qkv_matmul.ragged.paged")
struct Struct_fused_qkv_matmul_padded_ragged:
    """Registers the `kv_cache.fused_qkv_matmul.ragged.paged` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute(
        output: Some[Output & Tensor],
        hidden_state: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        weight: Some[Input & Tensor],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        layer_idx: UInt32,
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(hidden_state), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(weight), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert type_of(output).rank == 2
        comptime assert type_of(hidden_state).rank == 2
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(weight).rank == 2
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        var kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )
        return generic_fused_qkv_matmul_kv_cache_paged_ragged(
            hidden_state.to_tile_tensor[type_of(output).dtype](),
            input_row_offsets.to_tile_tensor[.uint32](),
            weight.to_tile_tensor(),
            kv_collection,
            layer_idx,
            output.to_tile_tensor(),
            ctx,
        )


@extensibility.register("kv_cache.fused_qkv_matmul.ragged.paged.quantized")
struct Struct_fused_qkv_matmul_padded_ragged_quantized:
    """Registers the `kv_cache.fused_qkv_matmul.ragged.paged.quantized` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute[
        group_size: Int,
        has_zp_int: Int,
        //,
    ](
        output: Some[Output & Tensor],
        hidden_state: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        weight: Some[Input & Tensor],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        layer_idx: UInt32,
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(hidden_state), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(weight), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert type_of(output).rank == 2
        comptime assert type_of(hidden_state).rank == 2
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(weight).rank == 2
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(kv_blocks).dtype == type_of(output).dtype
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        # In the group-wise quantization scheme, every `group_size` quantized weights
        # share the same scale. If `has_zp_int` is non-zero, there is also a group-wise
        # zero point that need to be subtracted from the quantized weights.
        comptime has_zp = True if has_zp_int == 1 else False
        var kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )
        return generic_fused_qkv_matmul_kv_cache_paged_ragged[
            group_size=group_size,
            has_zp=has_zp,
        ](
            hidden_state.to_tile_tensor[type_of(output).dtype](),
            input_row_offsets.to_tile_tensor[.uint32](),
            weight.to_tile_tensor(),
            kv_collection,
            layer_idx,
            output.to_tile_tensor(),
            ctx,
        )


@extensibility.register("kv_cache.fused_qkv_matmul.ragged.paged.bias")
struct Struct_fused_qkv_matmul_padded_ragged_bias:
    """Registers the `kv_cache.fused_qkv_matmul.ragged.paged.bias` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute(
        output: Some[Output & Tensor],
        hidden_state: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        weight: Some[Input & Tensor],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        layer_idx: UInt32,
        bias: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(hidden_state), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(weight), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert conforms_to(type_of(bias), TileTensorable)
        comptime assert type_of(output).rank == 2
        comptime assert type_of(hidden_state).rank == 2
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(weight).rank == 2
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(bias).rank == 1
        var kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )
        return generic_fused_qkv_matmul_kv_cache_paged_ragged_bias(
            hidden_state.to_tile_tensor[type_of(output).dtype](),
            input_row_offsets.to_tile_tensor[.uint32](),
            weight.to_tile_tensor(),
            kv_collection,
            layer_idx,
            output.to_tile_tensor(),
            bias.to_tile_tensor[type_of(output).dtype](),
            ctx,
        )


@extensibility.register("kv_cache.fused_qkv_matmul.ragged.paged.scale")
struct Struct_fused_qkv_matmul_padded_ragged_scale:
    """Registers the `kv_cache.fused_qkv_matmul.ragged.paged.scale` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute[
        kv_type: DType,
        //,
        m_scale_granularity: Int,
        n_scale_granularity: Int,
        k_scale_granularity: Int,
    ](
        output: Some[Output & Tensor],
        hidden_state: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        weight: Some[Input & Tensor],
        input_scale: Some[Input & Tensor],
        weight_scale: Some[Input & Tensor],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        layer_idx: UInt32,
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(hidden_state), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(weight), TileTensorable)
        comptime assert conforms_to(type_of(input_scale), TileTensorable)
        comptime assert conforms_to(type_of(weight_scale), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime dtype = type_of(hidden_state).dtype
        comptime scale_type = type_of(input_scale).dtype
        comptime assert type_of(output).rank == 2
        comptime assert type_of(hidden_state).rank == 2
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(weight).rank == 2
        comptime assert type_of(input_scale).rank == 2
        comptime assert type_of(weight_scale).rank == 2
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(kv_blocks).dtype == kv_type
        var kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )
        return generic_fused_qkv_matmul_kv_cache_paged_ragged_scale[
            scales_granularity_mnk=(
                m_scale_granularity,
                n_scale_granularity,
                k_scale_granularity,
            ),
        ](
            hidden_state.to_tile_tensor(),
            input_row_offsets.to_tile_tensor[.uint32](),
            weight.to_tile_tensor[dtype](),
            input_scale.to_tile_tensor(),
            weight_scale.to_tile_tensor[scale_type](),
            kv_collection,
            layer_idx,
            output.to_tile_tensor(),
            ctx,
            None,
        )


@inline(.always)
def _mut_pointer_tile(
    tile: TileTensor[mut=True, address_space=.GENERIC, ...],
) -> TileTensor[tile.dtype, tile.LayoutType, tile.origin]:
    """Rebuilds `tile` as a mutable view over its pointer.

    The paged KV cache collection holds pointer-backed tiles, so a projected
    argument with another storage policy aborts here at run time.
    """
    return {ptr = tile.unsafe_ptr(), layout = tile.layout}


@inline(.always)
def _imm_pointer_tile(
    tile: TileTensor[address_space=.GENERIC, ...],
) -> TileTensor[tile.dtype, tile.LayoutType, ImmOrigin(tile.origin)]:
    """Rebuilds `tile` as an immutable view over its pointer.

    The scaled fused QKV matmul helpers take pointer-backed tiles, so a
    projected argument with another storage policy aborts here at run time.
    """
    return TileTensor[tile.dtype, tile.LayoutType, ImmOrigin(tile.origin)](
        ptr=tile.unsafe_ptr().as_imm(), layout=tile.layout
    )


@extensibility.register("kv_cache.fused_qkv_matmul.ragged.paged.scale.float4")
struct Struct_fused_qkv_matmul_padded_ragged_scale_float4:
    """Registers the `kv_cache.fused_qkv_matmul.ragged.paged.scale.float4` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute[
        dtype: DType,
        scale_type: DType,
        kv_type: DType,
        //,
        SF_VECTOR_SIZE: Int,
    ](
        output: Some[Output & Tensor],
        hidden_state: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        weight: Some[Input & Tensor],
        input_scale: Some[Input & Tensor],
        weight_scale: Some[Input & Tensor],
        tensor_sf: Float32,
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        layer_idx: UInt32,
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(hidden_state), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(weight), TileTensorable)
        comptime assert conforms_to(type_of(input_scale), TileTensorable)
        comptime assert conforms_to(type_of(weight_scale), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert type_of(output).rank == 2
        comptime assert type_of(hidden_state).rank == 2
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(weight).rank == 2
        comptime assert type_of(input_scale).rank == 5
        comptime assert type_of(weight_scale).rank == 5
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(kv_blocks).dtype == kv_type
        var kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )
        return generic_fused_qkv_matmul_kv_cache_paged_ragged_scale_float4[
            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
        ](
            _imm_pointer_tile(hidden_state.to_tile_tensor[dtype]().as_imm()),
            input_row_offsets.to_tile_tensor[.uint32](),
            _imm_pointer_tile(weight.to_tile_tensor[dtype]().as_imm()),
            _imm_pointer_tile(
                input_scale.to_tile_tensor[scale_type]().as_imm()
            ),
            _imm_pointer_tile(
                weight_scale.to_tile_tensor[scale_type]().as_imm()
            ),
            tensor_sf,
            kv_collection,
            layer_idx,
            output.to_tile_tensor(),
            ctx,
        )


@extensibility.register("kv_cache.fused_qkv_matmul.ragged.paged.scale.mxfp8")
struct Struct_fused_qkv_matmul_padded_ragged_scale_mxfp8:
    """Registers the `kv_cache.fused_qkv_matmul.ragged.paged.scale.mxfp8` graph op with the graph compiler.
    """

    # Delegates to the NVFP4 entry point, which is dual-mode and also handles
    # MXFP8 from its data dtype, scale dtype, and SF_VECTOR_SIZE parameters. The
    # "float4" in the callee name is intentional. Do not split off a separate
    # MXFP8 path.
    @inline(.always)
    @staticmethod
    def execute[
        dtype: DType,
        scale_type: DType,
        kv_type: DType,
        //,
        SF_VECTOR_SIZE: Int,
    ](
        output: Some[Output & Tensor],
        hidden_state: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        weight: Some[Input & Tensor],
        input_scale: Some[Input & Tensor],
        weight_scale: Some[Input & Tensor],
        tensor_sf: Float32,
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        layer_idx: UInt32,
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(hidden_state), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(weight), TileTensorable)
        comptime assert conforms_to(type_of(input_scale), TileTensorable)
        comptime assert conforms_to(type_of(weight_scale), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert type_of(output).rank == 2
        comptime assert type_of(hidden_state).rank == 2
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(weight).rank == 2
        comptime assert type_of(input_scale).rank == 5
        comptime assert type_of(weight_scale).rank == 5
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(kv_blocks).dtype == kv_type
        var kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )
        return generic_fused_qkv_matmul_kv_cache_paged_ragged_scale_float4[
            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
        ](
            _imm_pointer_tile(hidden_state.to_tile_tensor[dtype]().as_imm()),
            input_row_offsets.to_tile_tensor[.uint32](),
            _imm_pointer_tile(weight.to_tile_tensor[dtype]().as_imm()),
            _imm_pointer_tile(
                input_scale.to_tile_tensor[scale_type]().as_imm()
            ),
            _imm_pointer_tile(
                weight_scale.to_tile_tensor[scale_type]().as_imm()
            ),
            tensor_sf,
            kv_collection,
            layer_idx,
            output.to_tile_tensor(),
            ctx,
        )


@extensibility.register(
    "kv_cache.fused_qkv_matmul.ragged.paged.scale.mxfp8.amd"
)
struct Struct_fused_qkv_matmul_padded_ragged_scale_mxfp8_amd:
    """Registers the `kv_cache.fused_qkv_matmul.ragged.paged.scale.mxfp8.amd` graph op with the graph compiler.
    """

    # CDNA4 sibling of the struct above, delegating to the same dual-mode entry
    # point and the same band routing. It exists only because the scale layout
    # differs by vendor: SM100 wants the rank-5 SF-atom interleave, CDNA4
    # consumes the checkpoint's plain rank-2 [N, K // 32] E8M0 scales.
    # Everything below the entry point is shared.
    @inline(.always)
    @staticmethod
    def execute[
        dtype: DType,
        scale_type: DType,
        kv_type: DType,
        //,
        SF_VECTOR_SIZE: Int,
    ](
        output: Some[Output & Tensor],
        hidden_state: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        weight: Some[Input & Tensor],
        input_scale: Some[Input & Tensor],
        weight_scale: Some[Input & Tensor],
        tensor_sf: Float32,
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        layer_idx: UInt32,
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(hidden_state), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(weight), TileTensorable)
        comptime assert conforms_to(type_of(input_scale), TileTensorable)
        comptime assert conforms_to(type_of(weight_scale), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert type_of(output).rank == 2
        comptime assert type_of(hidden_state).rank == 2
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(weight).rank == 2
        comptime assert type_of(input_scale).rank == 2
        comptime assert type_of(weight_scale).rank == 2
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(kv_blocks).dtype == kv_type
        var kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )
        return generic_fused_qkv_matmul_kv_cache_paged_ragged_scale_float4[
            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
        ](
            _imm_pointer_tile(hidden_state.to_tile_tensor[dtype]().as_imm()),
            input_row_offsets.to_tile_tensor[.uint32](),
            _imm_pointer_tile(weight.to_tile_tensor[dtype]().as_imm()),
            _imm_pointer_tile(
                input_scale.to_tile_tensor[scale_type]().as_imm()
            ),
            _imm_pointer_tile(
                weight_scale.to_tile_tensor[scale_type]().as_imm()
            ),
            tensor_sf,
            kv_collection,
            layer_idx,
            output.to_tile_tensor(),
            ctx,
        )


@extensibility.register(
    "kv_cache.fused_qkv_matmul.ragged.paged.scale.mxfp6.amd"
)
struct Struct_fused_qkv_matmul_padded_ragged_scale_mxfp6_amd:
    """Registers the `kv_cache.fused_qkv_matmul.ragged.paged.scale.mxfp6.amd` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute[
        dtype: DType,
        scale_type: DType,
        kv_type: DType,
        //,
        SF_VECTOR_SIZE: Int,
        FP6_FORMAT: Int,
    ](
        output: Some[Output & Tensor],
        hidden_state: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        weight: Some[Input & Tensor],
        input_scale: Some[Input & Tensor],
        weight_scale: Some[Input & Tensor],
        tensor_sf: Float32,
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        layer_idx: UInt32,
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(hidden_state), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(weight), TileTensorable)
        comptime assert conforms_to(type_of(input_scale), TileTensorable)
        comptime assert conforms_to(type_of(weight_scale), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert type_of(output).rank == 2
        comptime assert type_of(hidden_state).rank == 2
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(weight).rank == 2
        comptime assert type_of(input_scale).rank == 2
        comptime assert type_of(weight_scale).rank == 2
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(kv_blocks).dtype == kv_type
        comptime assert FP6_FORMAT in (
            0,
            1,
        ), "FP6_FORMAT must be 0 (E2M3) or 1 (E3M2)"
        comptime fmt = (
            MXFormat.FP6_E2M3 if FP6_FORMAT == 0 else MXFormat.FP6_E3M2
        )
        var kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )
        return generic_fused_qkv_matmul_kv_cache_paged_ragged_scale_float4[
            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            mx_format=fmt,
        ](
            _imm_pointer_tile(hidden_state.to_tile_tensor[dtype]().as_imm()),
            input_row_offsets.to_tile_tensor[.uint32](),
            _imm_pointer_tile(weight.to_tile_tensor[dtype]().as_imm()),
            _imm_pointer_tile(
                input_scale.to_tile_tensor[scale_type]().as_imm()
            ),
            _imm_pointer_tile(
                weight_scale.to_tile_tensor[scale_type]().as_imm()
            ),
            tensor_sf,
            kv_collection,
            layer_idx,
            output.to_tile_tensor(),
            ctx,
        )


@extensibility.register(
    "kv_cache.fused_qkv_index_matmul.ragged.paged.scale.mxfp8"
)
struct Struct_fused_qkv_index_matmul_padded_ragged_scale_mxfp8:
    """Registers the `kv_cache.fused_qkv_index_matmul.ragged.paged.scale.mxfp8` graph op with the graph compiler.
    """

    # Dual-cache fused QKV + index-QK matmul for MiniMax-M3. Like the
    # single-cache mxfp8 struct above, this delegates to the dual-mode NVFP4
    # entry point, which also handles MXFP8 (E8M0 scales, SF_VECTOR_SIZE=32)
    # from its dtype/scale-dtype/SF_VECTOR_SIZE parameters. The "float4" in the
    # callee name is intentional; do not split off a separate MXFP8 path.
    #
    # The MAIN cache operands (kv_blocks .. max_cache_length) drive the K/V
    # scatter; the INDEX cache operands (index_kv_blocks .. index_max_cache_length)
    # drive the IndexK scatter. Q is returned in `q_output` [M, q_dim] and IndexQ
    # in `iq_output` [M, iq_dim].
    #
    # `IQ_DIM` is the IndexQ output-band width (num_index_heads * idx_head_dim).
    # It is a parameter because, for the MLA index cache, it cannot be recovered
    # from the index cache's `num_heads` (== 1 for the single latent head).
    @inline(.always)
    @staticmethod
    def execute[
        dtype: DType,
        scale_type: DType,
        kv_type: DType,
        index_kv_type: DType,
        //,
        SF_VECTOR_SIZE: Int,
        IQ_DIM: Int,
    ](
        q_output: Some[Output & Tensor],
        iq_output: Some[Output & Tensor],
        hidden_state: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        weight: Some[Input & Tensor],
        input_scale: Some[Input & Tensor],
        weight_scale: Some[Input & Tensor],
        tensor_sf: Float32,
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        index_kv_blocks: Some[MutableInput & Tensor],
        index_page_stride: Some[Input & Tensor],
        index_cache_lengths: Some[Input & Tensor],
        index_kv_lookup_table: Some[Input & Tensor],
        index_max_prompt_length: Some[Input & Tensor],
        index_max_cache_length: Some[Input & Tensor],
        layer_idx: UInt32,
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(q_output), TileTensorable)
        comptime assert conforms_to(type_of(iq_output), TileTensorable)
        comptime assert conforms_to(type_of(hidden_state), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(weight), TileTensorable)
        comptime assert conforms_to(type_of(input_scale), TileTensorable)
        comptime assert conforms_to(type_of(weight_scale), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert conforms_to(type_of(index_kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(index_page_stride), TileTensorable)
        comptime assert conforms_to(
            type_of(index_cache_lengths), TileTensorable
        )
        comptime assert conforms_to(
            type_of(index_kv_lookup_table), TileTensorable
        )
        comptime assert conforms_to(
            type_of(index_max_prompt_length), TileTensorable
        )
        comptime assert conforms_to(
            type_of(index_max_cache_length), TileTensorable
        )
        comptime output_type = type_of(q_output).dtype
        comptime assert type_of(q_output).rank == 2
        comptime assert type_of(iq_output).rank == 2
        comptime assert type_of(hidden_state).rank == 2
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(weight).rank == 2
        comptime assert type_of(input_scale).rank == 5
        comptime assert type_of(weight_scale).rank == 5
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(index_kv_blocks).rank == 6
        comptime assert type_of(index_page_stride).rank == 1
        comptime assert type_of(index_cache_lengths).rank == 1
        comptime assert type_of(index_kv_lookup_table).rank == 2
        comptime assert type_of(index_max_prompt_length).rank == 1
        comptime assert type_of(index_max_cache_length).rank == 1
        comptime assert type_of(kv_blocks).dtype == kv_type
        comptime assert type_of(index_kv_blocks).dtype == index_kv_type
        var kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )
        var index_kv_collection = generic_get_paged_cache(
            index_kv_blocks.to_tile_tensor(),
            index_page_stride.to_tile_tensor[.int64](),
            index_cache_lengths.to_tile_tensor[.uint32](),
            index_kv_lookup_table.to_tile_tensor[.uint32](),
            index_max_prompt_length.to_tile_tensor[.uint32](),
            index_max_cache_length.to_tile_tensor[.uint32](),
        )
        return (
            generic_fused_qkv_index_matmul_kv_cache_paged_ragged_scale_float4[
                SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            ](
                _imm_pointer_tile(
                    hidden_state.to_tile_tensor[dtype]().as_imm()
                ),
                input_row_offsets.to_tile_tensor[.uint32](),
                _imm_pointer_tile(weight.to_tile_tensor[dtype]().as_imm()),
                _imm_pointer_tile(
                    input_scale.to_tile_tensor[scale_type]().as_imm()
                ),
                _imm_pointer_tile(
                    weight_scale.to_tile_tensor[scale_type]().as_imm()
                ),
                tensor_sf,
                kv_collection,
                index_kv_collection,
                layer_idx,
                IQ_DIM,
                q_output.to_tile_tensor(),
                iq_output.to_tile_tensor[output_type](),
                ctx,
            )
        )


@extensibility.register(
    "kv_cache.fused_qkv_index_matmul.ragged.paged.scale.mxfp8.amd"
)
struct Struct_fused_qkv_index_matmul_padded_ragged_scale_mxfp8_amd:
    """Registers the `kv_cache.fused_qkv_index_matmul.ragged.paged.scale.mxfp8.amd` graph op with the graph compiler.
    """

    # CDNA4 sibling of the struct above, delegating to the same dual-mode
    # entry point and the same column routing. It exists only because the
    # scale layout differs by vendor: SM100 wants the rank-5 SF-atom
    # interleave, CDNA4 consumes the checkpoint's plain rank-2 [N, K // 32]
    # E8M0 scales. Everything below the entry point is shared.
    #
    # The MAIN cache operands (kv_blocks .. max_cache_length) drive the K/V
    # scatter; the INDEX cache operands (index_kv_blocks .. index_max_cache_length)
    # drive the IndexK scatter. Q is returned in `q_output` [M, q_dim] and IndexQ
    # in `iq_output` [M, iq_dim].
    #
    # `IQ_DIM` is the IndexQ output-band width (num_index_heads * idx_head_dim).
    # It is a parameter because, for the MLA index cache, it cannot be recovered
    # from the index cache's `num_heads` (== 1 for the single latent head).
    @inline(.always)
    @staticmethod
    def execute[
        dtype: DType,
        scale_type: DType,
        kv_type: DType,
        index_kv_type: DType,
        //,
        SF_VECTOR_SIZE: Int,
        IQ_DIM: Int,
    ](
        q_output: Some[Output & Tensor],
        iq_output: Some[Output & Tensor],
        hidden_state: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        weight: Some[Input & Tensor],
        input_scale: Some[Input & Tensor],
        weight_scale: Some[Input & Tensor],
        tensor_sf: Float32,
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        index_kv_blocks: Some[MutableInput & Tensor],
        index_page_stride: Some[Input & Tensor],
        index_cache_lengths: Some[Input & Tensor],
        index_kv_lookup_table: Some[Input & Tensor],
        index_max_prompt_length: Some[Input & Tensor],
        index_max_cache_length: Some[Input & Tensor],
        layer_idx: UInt32,
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(q_output), TileTensorable)
        comptime assert conforms_to(type_of(iq_output), TileTensorable)
        comptime assert conforms_to(type_of(hidden_state), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(weight), TileTensorable)
        comptime assert conforms_to(type_of(input_scale), TileTensorable)
        comptime assert conforms_to(type_of(weight_scale), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert conforms_to(type_of(index_kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(index_page_stride), TileTensorable)
        comptime assert conforms_to(
            type_of(index_cache_lengths), TileTensorable
        )
        comptime assert conforms_to(
            type_of(index_kv_lookup_table), TileTensorable
        )
        comptime assert conforms_to(
            type_of(index_max_prompt_length), TileTensorable
        )
        comptime assert conforms_to(
            type_of(index_max_cache_length), TileTensorable
        )
        comptime output_type = type_of(q_output).dtype
        comptime assert type_of(q_output).rank == 2
        comptime assert type_of(iq_output).rank == 2
        comptime assert type_of(hidden_state).rank == 2
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(weight).rank == 2
        comptime assert type_of(input_scale).rank == 2
        comptime assert type_of(weight_scale).rank == 2
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(index_kv_blocks).rank == 6
        comptime assert type_of(index_page_stride).rank == 1
        comptime assert type_of(index_cache_lengths).rank == 1
        comptime assert type_of(index_kv_lookup_table).rank == 2
        comptime assert type_of(index_max_prompt_length).rank == 1
        comptime assert type_of(index_max_cache_length).rank == 1
        comptime assert type_of(kv_blocks).dtype == kv_type
        comptime assert type_of(index_kv_blocks).dtype == index_kv_type
        var kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )
        var index_kv_collection = generic_get_paged_cache(
            index_kv_blocks.to_tile_tensor(),
            index_page_stride.to_tile_tensor[.int64](),
            index_cache_lengths.to_tile_tensor[.uint32](),
            index_kv_lookup_table.to_tile_tensor[.uint32](),
            index_max_prompt_length.to_tile_tensor[.uint32](),
            index_max_cache_length.to_tile_tensor[.uint32](),
        )
        return (
            generic_fused_qkv_index_matmul_kv_cache_paged_ragged_scale_float4[
                SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            ](
                _imm_pointer_tile(
                    hidden_state.to_tile_tensor[dtype]().as_imm()
                ),
                input_row_offsets.to_tile_tensor[.uint32](),
                _imm_pointer_tile(weight.to_tile_tensor[dtype]().as_imm()),
                _imm_pointer_tile(
                    input_scale.to_tile_tensor[scale_type]().as_imm()
                ),
                _imm_pointer_tile(
                    weight_scale.to_tile_tensor[scale_type]().as_imm()
                ),
                tensor_sf,
                kv_collection,
                index_kv_collection,
                layer_idx,
                IQ_DIM,
                q_output.to_tile_tensor(),
                iq_output.to_tile_tensor[output_type](),
                ctx,
            )
        )


@extensibility.register(
    "kv_cache.fused_qkv_index_matmul.ragged.paged.scale.mxfp6.amd"
)
struct Struct_fused_qkv_index_matmul_padded_ragged_scale_mxfp6_amd:
    """Registers the `kv_cache.fused_qkv_index_matmul.ragged.paged.scale.mxfp6.amd` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute[
        dtype: DType,
        scale_type: DType,
        kv_type: DType,
        index_kv_type: DType,
        //,
        SF_VECTOR_SIZE: Int,
        IQ_DIM: Int,
        FP6_FORMAT: Int,
    ](
        q_output: Some[Output & Tensor],
        iq_output: Some[Output & Tensor],
        hidden_state: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        weight: Some[Input & Tensor],
        input_scale: Some[Input & Tensor],
        weight_scale: Some[Input & Tensor],
        tensor_sf: Float32,
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        index_kv_blocks: Some[MutableInput & Tensor],
        index_page_stride: Some[Input & Tensor],
        index_cache_lengths: Some[Input & Tensor],
        index_kv_lookup_table: Some[Input & Tensor],
        index_max_prompt_length: Some[Input & Tensor],
        index_max_cache_length: Some[Input & Tensor],
        layer_idx: UInt32,
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(q_output), TileTensorable)
        comptime assert conforms_to(type_of(iq_output), TileTensorable)
        comptime assert conforms_to(type_of(hidden_state), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(weight), TileTensorable)
        comptime assert conforms_to(type_of(input_scale), TileTensorable)
        comptime assert conforms_to(type_of(weight_scale), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert conforms_to(type_of(index_kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(index_page_stride), TileTensorable)
        comptime assert conforms_to(
            type_of(index_cache_lengths), TileTensorable
        )
        comptime assert conforms_to(
            type_of(index_kv_lookup_table), TileTensorable
        )
        comptime assert conforms_to(
            type_of(index_max_prompt_length), TileTensorable
        )
        comptime assert conforms_to(
            type_of(index_max_cache_length), TileTensorable
        )
        comptime output_type = type_of(q_output).dtype
        comptime assert type_of(q_output).rank == 2
        comptime assert type_of(iq_output).rank == 2
        comptime assert type_of(hidden_state).rank == 2
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(weight).rank == 2
        comptime assert type_of(input_scale).rank == 2
        comptime assert type_of(weight_scale).rank == 2
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(index_kv_blocks).rank == 6
        comptime assert type_of(index_page_stride).rank == 1
        comptime assert type_of(index_cache_lengths).rank == 1
        comptime assert type_of(index_kv_lookup_table).rank == 2
        comptime assert type_of(index_max_prompt_length).rank == 1
        comptime assert type_of(index_max_cache_length).rank == 1
        comptime assert type_of(kv_blocks).dtype == kv_type
        comptime assert type_of(index_kv_blocks).dtype == index_kv_type
        var kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )
        var index_kv_collection = generic_get_paged_cache(
            index_kv_blocks.to_tile_tensor(),
            index_page_stride.to_tile_tensor[.int64](),
            index_cache_lengths.to_tile_tensor[.uint32](),
            index_kv_lookup_table.to_tile_tensor[.uint32](),
            index_max_prompt_length.to_tile_tensor[.uint32](),
            index_max_cache_length.to_tile_tensor[.uint32](),
        )
        comptime assert FP6_FORMAT in (
            0,
            1,
        ), "FP6_FORMAT must be 0 (E2M3) or 1 (E3M2)"
        comptime fmt = (
            MXFormat.FP6_E2M3 if FP6_FORMAT == 0 else MXFormat.FP6_E3M2
        )
        return (
            generic_fused_qkv_index_matmul_kv_cache_paged_ragged_scale_float4[
                SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                mx_format=fmt,
            ](
                _imm_pointer_tile(
                    hidden_state.to_tile_tensor[dtype]().as_imm()
                ),
                input_row_offsets.to_tile_tensor[.uint32](),
                _imm_pointer_tile(weight.to_tile_tensor[dtype]().as_imm()),
                _imm_pointer_tile(
                    input_scale.to_tile_tensor[scale_type]().as_imm()
                ),
                _imm_pointer_tile(
                    weight_scale.to_tile_tensor[scale_type]().as_imm()
                ),
                tensor_sf,
                kv_collection,
                index_kv_collection,
                layer_idx,
                IQ_DIM,
                q_output.to_tile_tensor(),
                iq_output.to_tile_tensor[output_type](),
                ctx,
            )
        )


@extensibility.register("kv_cache.fused_qkv_index_matmul.ragged.paged")
struct Struct_fused_qkv_index_matmul_padded_ragged:
    # BF16 (non-scaled) dual-cache fused QKV + index-QK matmul for MiniMax-M3.
    # Non-scaled analog of the mxfp8 struct above: same dual-cache column routing
    # (MAIN K/V scatter + INDEX IndexK scatter) and `IQ_DIM` parameter, but plain
    # BF16 with no block-scaling operands (no input_scale / weight_scale /
    # tensor_sf / SF_VECTOR_SIZE). Hardware-agnostic (AMD CDNA4 + NVIDIA).
    #
    # The MAIN cache operands (kv_blocks .. max_cache_length) drive the K/V
    # scatter; the INDEX cache operands (index_kv_blocks .. index_max_cache_length)
    # drive the IndexK scatter. Q and IndexQ are returned in the combined
    # `output` tensor [M, q_dim + iq_dim].
    #
    # `IQ_DIM` is the IndexQ output-band width (num_index_heads * idx_head_dim).
    # It is a parameter because, for the MLA index cache, it cannot be recovered
    # from the index cache's `num_heads` (== 1 for the single latent head).
    @inline(.always)
    @staticmethod
    def execute[
        kv_type: DType,
        index_kv_type: DType,
        //,
        IQ_DIM: Int,
    ](
        output: Some[Output & Tensor],
        hidden_state: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        weight: Some[Input & Tensor],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        index_kv_blocks: Some[MutableInput & Tensor],
        index_page_stride: Some[Input & Tensor],
        index_cache_lengths: Some[Input & Tensor],
        index_kv_lookup_table: Some[Input & Tensor],
        index_max_prompt_length: Some[Input & Tensor],
        index_max_cache_length: Some[Input & Tensor],
        layer_idx: UInt32,
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(hidden_state), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(weight), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert conforms_to(type_of(index_kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(index_page_stride), TileTensorable)
        comptime assert conforms_to(
            type_of(index_cache_lengths), TileTensorable
        )
        comptime assert conforms_to(
            type_of(index_kv_lookup_table), TileTensorable
        )
        comptime assert conforms_to(
            type_of(index_max_prompt_length), TileTensorable
        )
        comptime assert conforms_to(
            type_of(index_max_cache_length), TileTensorable
        )
        comptime dtype = type_of(output).dtype
        comptime assert type_of(output).rank == 2
        comptime assert type_of(hidden_state).rank == 2
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(weight).rank == 2
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(index_kv_blocks).rank == 6
        comptime assert type_of(index_page_stride).rank == 1
        comptime assert type_of(index_cache_lengths).rank == 1
        comptime assert type_of(index_kv_lookup_table).rank == 2
        comptime assert type_of(index_max_prompt_length).rank == 1
        comptime assert type_of(index_max_cache_length).rank == 1
        comptime assert type_of(kv_blocks).dtype == kv_type
        comptime assert type_of(index_kv_blocks).dtype == index_kv_type
        var kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )
        var index_kv_collection = generic_get_paged_cache(
            index_kv_blocks.to_tile_tensor(),
            index_page_stride.to_tile_tensor[.int64](),
            index_cache_lengths.to_tile_tensor[.uint32](),
            index_kv_lookup_table.to_tile_tensor[.uint32](),
            index_max_prompt_length.to_tile_tensor[.uint32](),
            index_max_cache_length.to_tile_tensor[.uint32](),
        )
        return generic_fused_qkv_index_matmul_kv_cache_paged_ragged(
            hidden_state.to_tile_tensor[dtype](),
            input_row_offsets.to_tile_tensor[.uint32](),
            weight.to_tile_tensor[dtype](),
            kv_collection,
            index_kv_collection,
            layer_idx,
            IQ_DIM,
            output.to_tile_tensor(),
            ctx,
        )


@extensibility.register("kv_cache.fused_qkv_matmul.ragged.paged.scale.bias")
struct Struct_fused_qkv_matmul_padded_ragged_scale_bias:
    """Registers the `kv_cache.fused_qkv_matmul.ragged.paged.scale.bias` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute[
        kv_type: DType,
        //,
        m_scale_granularity: Int,
        n_scale_granularity: Int,
        k_scale_granularity: Int,
    ](
        output: Some[Output & Tensor],
        hidden_state: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        weight: Some[Input & Tensor],
        input_scale: Some[Input & Tensor],
        weight_scale: Some[Input & Tensor],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        layer_idx: UInt32,
        bias: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(hidden_state), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(weight), TileTensorable)
        comptime assert conforms_to(type_of(input_scale), TileTensorable)
        comptime assert conforms_to(type_of(weight_scale), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert conforms_to(type_of(bias), TileTensorable)
        comptime dtype = type_of(hidden_state).dtype
        comptime scale_type = type_of(input_scale).dtype
        comptime output_type = type_of(output).dtype
        comptime assert type_of(output).rank == 2
        comptime assert type_of(hidden_state).rank == 2
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(weight).rank == 2
        comptime assert type_of(input_scale).rank == 2
        comptime assert type_of(weight_scale).rank == 2
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(bias).rank == 1
        comptime assert type_of(kv_blocks).dtype == kv_type
        var kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )
        comptime ExpectedBiasType = TileTensor[
            output_type, RowMajorLayout[Int64], ImmutAnyOrigin
        ]
        var bias_tensor = bias.to_tile_tensor[output_type]()
        comptime assert bias_tensor.is_row_major
        var rebound_bias = rebind[ExpectedBiasType](
            bias_tensor.reshape(
                Coord(Int64(bias_tensor.num_elements()))
            ).as_unsafe_any_origin()
        )
        return generic_fused_qkv_matmul_kv_cache_paged_ragged_scale[
            scales_granularity_mnk=(
                m_scale_granularity,
                n_scale_granularity,
                k_scale_granularity,
            ),
        ](
            hidden_state.to_tile_tensor(),
            input_row_offsets.to_tile_tensor[.uint32](),
            weight.to_tile_tensor[dtype](),
            input_scale.to_tile_tensor(),
            weight_scale.to_tile_tensor[scale_type](),
            kv_collection,
            layer_idx,
            output.to_tile_tensor(),
            ctx,
            OptionalReg[ExpectedBiasType](rebound_bias),
        )


@extensibility.register("kv_cache.fused_qkv_matmul.ragged.paged.bias.quantized")
struct Struct_fused_qkv_matmul_padded_ragged_bias_quantized:
    """Registers the `kv_cache.fused_qkv_matmul.ragged.paged.bias.quantized` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute[
        group_size: Int,
        has_zp_int: Int,
        //,
    ](
        output: Some[Output & Tensor],
        hidden_state: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        weight: Some[Input & Tensor],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        layer_idx: UInt32,
        bias: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(hidden_state), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(weight), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert conforms_to(type_of(bias), TileTensorable)
        comptime assert type_of(output).rank == 2
        comptime assert type_of(hidden_state).rank == 2
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(weight).rank == 2
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(kv_blocks).dtype == type_of(output).dtype
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(bias).rank == 1
        # In the group-wise quantization scheme, every `group_size` quantized weights
        # share the same scale. If `has_zp_int` is non-zero, there is also a group-wise
        # zero point that need to be subtracted from the quantized weights.
        comptime has_zp = True if has_zp_int == 1 else False
        var kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )
        return generic_fused_qkv_matmul_kv_cache_paged_ragged_bias[
            group_size=group_size,
            has_zp=has_zp,
        ](
            hidden_state.to_tile_tensor[type_of(output).dtype](),
            input_row_offsets.to_tile_tensor[.uint32](),
            weight.to_tile_tensor(),
            kv_collection,
            layer_idx,
            output.to_tile_tensor(),
            bias.to_tile_tensor[type_of(output).dtype](),
            ctx,
        )


@extensibility.register("kv_cache.fused_qk_rope.ragged.paged.with_position_id")
struct Struct_fused_qk_rope_ragged_paged_with_position_id[interleaved: Bool]:
    """Registers the `kv_cache.fused_qk_rope.ragged.paged.with_position_id` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute[
        mrope_section: StaticString,
        cache_dtype: DType,
    ](
        output: Some[Output & Tensor],
        q_proj: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        freqs_cis: Some[Input & Tensor],
        position_ids: Some[Input & Tensor],
        layer_idx: UInt32,
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(q_proj), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert conforms_to(type_of(freqs_cis), TileTensorable)
        comptime assert conforms_to(type_of(position_ids), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime assert type_of(output).rank == 3
        comptime assert type_of(q_proj).rank == 3
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(freqs_cis).rank == 2
        comptime assert type_of(position_ids).rank == 2
        comptime assert type_of(kv_blocks).dtype == cache_dtype
        var kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )
        comptime mrope = _unsafe_str_to_coord[mrope_section]()
        generic_fused_qk_rope_bshd_paged_ragged[
            interleaved=Self.interleaved,
            has_position_ids=True,
            mrope_types=mrope.element_types,
            mrope_section=mrope,
        ](
            q_proj.to_tile_tensor[type_of(output).dtype](),
            input_row_offsets.to_tile_tensor[.uint32](),
            kv_collection,
            freqs_cis.to_tile_tensor(),
            position_ids.to_tile_tensor[.uint32](),
            layer_idx,
            output.to_tile_tensor(),
            context,
        )


@extensibility.register("kv_cache.fused_qk_rope.ragged.paged")
struct Struct_fused_qk_rope_ragged_paged[interleaved: Bool]:
    """Registers the `kv_cache.fused_qk_rope.ragged.paged` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute[
        cache_dtype: DType,
        v_head_dim: Int = -1,
    ](
        output: Some[Output & Tensor],
        q_proj: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        freqs_cis: Some[Input & Tensor],
        layer_idx: UInt32,
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(q_proj), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert conforms_to(type_of(freqs_cis), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime assert type_of(output).rank == 3
        comptime assert type_of(q_proj).rank == 3
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(freqs_cis).rank == 2
        comptime assert type_of(kv_blocks).dtype == cache_dtype
        # Dummy position_ids - won't be used since has_position_ids=False
        var dummy_position_ids = TileTensor(
            MutPointer[UInt32, MutUntrackedOrigin].unsafe_dangling(),
            row_major(0, 0),
        )
        var kv_collection = generic_get_paged_cache[v_head_size=v_head_dim](
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )
        generic_fused_qk_rope_bshd_paged_ragged[
            interleaved=Self.interleaved,
            has_position_ids=False,
        ](
            q_proj.to_tile_tensor[type_of(output).dtype](),
            input_row_offsets.to_tile_tensor[.uint32](),
            kv_collection,
            freqs_cis.to_tile_tensor(),
            dummy_position_ids,
            layer_idx,
            output.to_tile_tensor(),
            context,
        )


@extensibility.register("attention.mha.padded.paged")
struct Struct_mha_padded_paged:
    """Registers the `attention.mha.padded.paged` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute[
        mask_str: StaticString,
        local_window_size: Int = -1,
    ](
        output: Some[Output & Tensor],
        q: Some[Input & Tensor],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        layer_idx: UInt32,
        valid_lengths: Some[Input & Tensor],
        scale: Float32,
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(q), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert conforms_to(type_of(valid_lengths), TileTensorable)
        comptime dtype = type_of(kv_blocks).dtype
        comptime assert type_of(output).rank == 4
        comptime assert type_of(q).rank == 4
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(valid_lengths).rank == 1
        var kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )

        generic_flash_attention_kv_cache_padded[
            mask_str=mask_str,
            local_window_size=local_window_size,
        ](
            q.to_tile_tensor[dtype](),
            kv_collection,
            layer_idx,
            valid_lengths.to_tile_tensor[.uint32](),
            scale,
            output.to_tile_tensor[dtype](),
            context,
        )


@extensibility.register("attention.mha.ragged.paged")
struct Struct_mha_ragged_paged_scalar_args:
    """Registers the `attention.mha.ragged.paged` graph op with the graph compiler.

    `mixed_step_split` is the KV cache's split record
    (`kv_cache.mixed_step_split`). On NVIDIA, a valid record runs the decode
    rows and the prefill rows of one replica batch as two views, each on its
    own route, into the one output. Zeros, an invalid record, a strided
    operand, split head dims, mask operands or another target keep the single
    launch.
    """

    @inline(.always)
    @staticmethod
    def execute[
        MaskIns: TypeList[Trait=Input & Fused & LoadAccess, ...],
        //,
        mask_str: StaticString,
        local_window_size: Int = -1,
        v_head_dim: Int = -1,
    ](
        output: Some[Output & Tensor],
        q: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        layer_idx: UInt32,
        scale: Float32,
        mha_decode_dispatch_metadata: Some[Input & Tensor],
        mixed_step_split: Some[Input & Tensor],
        mask_operands: TensorTuple[*MaskIns],
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(q), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert conforms_to(
            type_of(mha_decode_dispatch_metadata), TileTensorable
        )
        comptime assert conforms_to(type_of(mixed_step_split), TileTensorable)
        comptime out_dtype = type_of(output).dtype
        comptime assert type_of(output).rank == 3
        comptime assert type_of(q).rank == 3
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(mha_decode_dispatch_metadata).rank == 1
        comptime assert type_of(mixed_step_split).rank == 1
        _execute_mha_ragged_paged[
            mask_str=mask_str,
            local_window_size=local_window_size,
            v_head_dim=v_head_dim,
        ](
            output.to_tile_tensor(),
            q.to_tile_tensor(),
            input_row_offsets.to_tile_tensor[.uint32](),
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
            layer_idx,
            scale,
            mha_decode_dispatch_metadata.to_tile_tensor[.int64](),
            mixed_step_split.to_tile_tensor[.uint32](),
            mask_operands,
            context,
        )


def _execute_mha_ragged_paged[
    q_dtype: DType,
    output_dtype: DType,
    MaskIns: TypeList[Trait=Input & Fused & LoadAccess, ...],
    //,
    *,
    mask_str: StaticString,
    local_window_size: Int,
    v_head_dim: Int,
](
    output: TileTensor[mut=True, output_dtype, address_space=.GENERIC, ...],
    q: TileTensor[q_dtype, address_space=.GENERIC, ...],
    input_row_offsets: TileTensor[DType.uint32, address_space=.GENERIC, ...],
    kv_blocks: TileTensor[mut=True, address_space=.GENERIC, ...],
    page_stride: TileTensor[
        mut=False, DType.int64, address_space=.GENERIC, ...
    ],
    cache_lengths: TileTensor[
        mut=False, DType.uint32, address_space=.GENERIC, ...
    ],
    kv_lookup_table: TileTensor[
        mut=False, DType.uint32, address_space=.GENERIC, ...
    ],
    max_prompt_length: TileTensor[
        mut=False, DType.uint32, address_space=.GENERIC, ...
    ],
    max_cache_length: TileTensor[
        mut=False, DType.uint32, address_space=.GENERIC, ...
    ],
    layer_idx: UInt32,
    scale: Float32,
    mha_decode_dispatch_metadata: TileTensor[
        DType.int64, address_space=.GENERIC, ...
    ],
    mixed_step_split: TileTensor[
        mut=False, DType.uint32, address_space=.GENERIC, ...
    ],
    mask_operands: TensorTuple[*MaskIns],
    context: DeviceContext,
) raises:
    """Runs `attention.mha.ragged.paged`: split by `mixed_step_split` where the
    record and the operands allow it, else one launch."""
    # Mask operands are indexed by the whole batch's sequence ids, which a
    # split view renumbers.
    comptime if (
        v_head_dim == -1
        and len(MaskIns) == 0
        and context.target.is_gpu()
        and context.target.is_nvidia_gpu()
    ):
        if _mha_ragged_mixed_step_split[
            mask_str=mask_str,
            local_window_size=local_window_size,
        ](
            output,
            q,
            input_row_offsets,
            kv_blocks,
            page_stride,
            cache_lengths,
            kv_lookup_table,
            Int(max_prompt_length[0]),
            Int(max_cache_length[0]),
            mixed_step_split,
            layer_idx,
            scale,
            context,
        ):
            return
    _execute_mha_ragged_paged_scalar_args[
        mask_str=mask_str,
        local_window_size=local_window_size,
        v_head_dim=v_head_dim,
    ](
        output,
        q,
        input_row_offsets,
        kv_blocks,
        page_stride,
        cache_lengths,
        kv_lookup_table,
        max_prompt_length,
        max_cache_length,
        layer_idx,
        scale,
        mha_decode_dispatch_metadata,
        mask_operands,
        context,
    )


def _mha_ragged_mixed_step_split[
    q_dtype: DType,
    out_dtype: DType,
    //,
    *,
    mask_str: StaticString,
    local_window_size: Int,
](
    output: TileTensor[mut=True, out_dtype, address_space=.GENERIC, ...],
    q: TileTensor[q_dtype, address_space=.GENERIC, ...],
    input_row_offsets: TileTensor[DType.uint32, address_space=.GENERIC, ...],
    kv_blocks: TileTensor[mut=True, address_space=.GENERIC, ...],
    page_stride: TileTensor[
        mut=False, DType.int64, address_space=.GENERIC, ...
    ],
    cache_lengths: TileTensor[
        mut=False, DType.uint32, address_space=.GENERIC, ...
    ],
    kv_lookup_table: TileTensor[
        mut=False, DType.uint32, address_space=.GENERIC, ...
    ],
    max_q_len: Int,
    max_cache_len: Int,
    mixed_step_split: TileTensor[
        mut=False, DType.uint32, address_space=.GENERIC, ...
    ],
    layer_idx: UInt32,
    scale: Float32,
    context: DeviceContext,
) raises -> Bool:
    """Runs ragged MHA on the head view, then on the tail view, into the
    one `output`, when `mixed_step_split` holds a split the operands allow.

    Returns False, with nothing launched, to keep the single launch. `q` and
    `output` are `[tokens, heads, depth]`, so a view is a pointer offset at
    any head count. The split-K combine covers every `q` row it gets, so
    each view gets its own rows and batch-local offsets.
    """
    if max_q_len <= MIXED_STEP_DECODE_WIDTH:
        return False
    var batch = Int(input_row_offsets.dim[0]()) - 1
    if case .Some(split) = read_mixed_split(
        mixed_step_split,
        batch,
        Int(q.dim[0]()),
        max_q_len,
        max_cache_len,
    ):
        if not (
            is_packed(q)
            and is_packed(output)
            and is_packed(input_row_offsets)
            and is_packed(cache_lengths)
            and is_packed(kv_lookup_table)
            and is_packed(mixed_step_split)
        ):
            return False
        comptime num_heads = q.static_shape[1]
        comptime depth = q.static_shape[2]
        comptime kv_heads = kv_blocks.static_shape[4]
        comptime row_width = num_heads * depth
        comptime for view_idx in range(2):
            comptime in_tail = view_idx == 1
            var view = split.tail if in_tail else split.head
            var row_offsets = input_row_offsets.ptr.unsafe_mut_cast[
                True
            ]().unsafe_origin_cast[MutAnyOrigin]()
            var rebased: Optional[DeviceBuffer[.uint32]] = None
            comptime if in_tail:
                rebased = rebase_row_offsets[with_cache_offsets=False](
                    row_offsets, row_offsets, batch, view, context
                )
                row_offsets = (
                    rebased.value()
                    .unsafe_ptr()
                    .unsafe_origin_cast[MutAnyOrigin]()
                )
            var cache = paged_cache_rows(
                kv_blocks,
                page_stride,
                cache_lengths,
                kv_lookup_table,
                mixed_step_split,
                view.first_row,
                view.rows,
                view.lengths_slot,
            )
            generic_flash_attention_kv_cache_ragged[
                mask_str=mask_str,
                local_window_size=local_window_size,
                output_dtype=out_dtype,
            ](
                TileTensor(
                    q.ptr.unsafe_offset(view.first_token * row_width),
                    tt_row_major(view.tokens, Idx[num_heads], Idx[depth]),
                ).as_imm(),
                TileTensor(row_offsets, tt_row_major(view.rows + 1)).as_imm(),
                cache,
                layer_idx,
                scale,
                TileTensor(
                    output.ptr.unsafe_offset(view.first_token * row_width),
                    tt_row_major(view.tokens, Idx[num_heads], Idx[depth]),
                ),
                context,
                MHADecodeDispatchMetadata.from_runtime_values[
                    num_heads, num_heads // kv_heads
                ](view.rows, view.max_q_len, view.max_cache_len, context),
                TensorTuple(),
            )
            _ = rebased^
        return True
    return False


@extensibility.register("attention.mha.ragged.paged.sink_weights")
struct Struct_mha_ragged_paged_sink_weights_scalar_args:
    """Registers the `attention.mha.ragged.paged.sink_weights` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute[
        mask_str: StaticString,
        local_window_size: Int = -1,
        v_head_dim: Int = -1,
    ](
        output: Some[Output & Tensor],
        q: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        layer_idx: UInt32,
        scale: Float32,
        sink_weights: Some[Input & Tensor],
        mha_decode_dispatch_metadata: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(q), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert conforms_to(type_of(sink_weights), TileTensorable)
        comptime assert conforms_to(
            type_of(mha_decode_dispatch_metadata), TileTensorable
        )
        comptime dtype = type_of(kv_blocks).dtype
        comptime assert type_of(output).rank == 3
        comptime assert type_of(output).dtype == dtype
        comptime assert type_of(q).rank == 3
        comptime assert type_of(q).dtype == dtype
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(sink_weights).rank == 1
        comptime assert type_of(mha_decode_dispatch_metadata).rank == 1
        var sink_weights_tt = sink_weights.to_tile_tensor[
            type_of(q).dtype
        ]().as_imm()
        _execute_mha_ragged_paged_scalar_args[
            mask_str=mask_str,
            sink=True,
            local_window_size=local_window_size,
            v_head_dim=v_head_dim,
        ](
            output.to_tile_tensor(),
            q.to_tile_tensor(),
            input_row_offsets.to_tile_tensor[.uint32](),
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
            layer_idx,
            scale,
            mha_decode_dispatch_metadata.to_tile_tensor[.int64](),
            TensorTuple(),
            context,
            OptionalReg[type_of(sink_weights_tt)](sink_weights_tt),
        )


trait _MLADecodeRaggedPagedOp:
    """Implements `attention.mla.decode.ragged.paged`.

    Each registration picks its kernels.
    """

    comptime kernels: MLAGraphKernels

    @inline(.always)
    @staticmethod
    def execute[
        mask_str: StaticString,
    ](
        output: Some[Output & Tensor],
        q: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        layer_idx: UInt32,
        scale: Float32,
        scalar_args: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(q), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert conforms_to(type_of(scalar_args), TileTensorable)
        comptime assert type_of(output).rank == 3
        comptime assert type_of(q).rank == 3
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(scalar_args).rank == 1
        comptime assert (
            type_of(kv_blocks)._shape_types[1].static_value == 1
        ), "Only support only_k=True for MLA decompress"
        var kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )
        generic_flare_mla_decode_kv_cache_ragged[
            mask_str=mask_str,
            kernels=Self.kernels.Kernels,
        ](
            q.to_tile_tensor(),
            input_row_offsets.to_tile_tensor[.uint32](),
            kv_collection,
            layer_idx,
            scale,
            output.to_tile_tensor(),
            scalar_args.to_tile_tensor[.int64](),
            context,
        )


@extensibility.register("attention.mla.decode.ragged.paged")
struct Struct_mla_decode_ragged_paged(_MLADecodeRaggedPagedOp):
    """Registers the op for pre-SM100 NVIDIA GPUs and any other device."""

    comptime kernels: MLAGraphKernels = LegacyNVMLAGraph


@extensibility.register(
    "attention.mla.decode.ragged.paged", type="gpu", api="hip"
)
struct Struct_mla_decode_ragged_paged_amd(_MLADecodeRaggedPagedOp):
    """Registers the op for AMD GPUs."""

    comptime kernels: MLAGraphKernels = AMDMLAGraph


@extensibility.register(
    "attention.mla.decode.ragged.paged", type="gpu", api="cuda", arch="sm_100a"
)
struct Struct_mla_decode_ragged_paged_sm100(_MLADecodeRaggedPagedOp):
    """Registers the op for B100 and B200."""

    comptime kernels: MLAGraphKernels = SM100MLAGraph


@extensibility.register(
    "attention.mla.decode.ragged.paged", type="gpu", api="cuda", arch="sm_103a"
)
struct Struct_mla_decode_ragged_paged_sm103(_MLADecodeRaggedPagedOp):
    """Registers the op for B300."""

    comptime kernels: MLAGraphKernels = SM100MLAGraph


trait _MLADecodeRaggedPagedScaledOp:
    """Implements `attention.mla.decode.ragged.paged.scaled`.

    Each registration picks its kernels.
    """

    comptime kernels: MLAGraphKernels

    @inline(.always)
    @staticmethod
    def execute[
        mask_str: StaticString,
        per_token_scale_rope_aware: Int = 0,
        quantization_granularity: Int = 640,
    ](
        output: Some[Output & Tensor],
        q: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        kv_scales: Some[MutableInput & Tensor],
        scales_page_stride: Some[Input & Tensor],
        # Resolves a request's scale pages. Pass `kv_lookup_table` itself when
        # the scales share the values' block-id space; pass a distinct table
        # when they are paged independently.
        kv_scales_lookup_table: Some[Input & Tensor],
        q_scales: Some[Input & Tensor],
        layer_idx: UInt32,
        scale: Float32,
        scalar_args: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(q), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert conforms_to(type_of(kv_scales), TileTensorable)
        comptime assert conforms_to(type_of(scales_page_stride), TileTensorable)
        comptime assert conforms_to(
            type_of(kv_scales_lookup_table), TileTensorable
        )
        comptime assert conforms_to(type_of(q_scales), TileTensorable)
        comptime assert conforms_to(type_of(scalar_args), TileTensorable)
        comptime kv_dtype = type_of(kv_blocks).dtype
        comptime assert type_of(output).rank == 3
        comptime assert type_of(q).rank == 3
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(kv_scales).rank == 6
        comptime assert type_of(scales_page_stride).rank == 1
        comptime assert type_of(kv_scales_lookup_table).rank == 2
        comptime assert type_of(q_scales).rank == 1
        comptime assert type_of(scalar_args).rank == 1
        comptime assert (
            type_of(kv_blocks)._shape_types[1].static_value == 1
        ), "Only support only_k=True for MLA decompress"

        comptime page_size = type_of(kv_blocks)._shape_types[3].static_value
        comptime head_dim = type_of(kv_blocks)._shape_types[5].static_value
        comptime kv_num_heads = type_of(kv_blocks)._shape_types[4].static_value
        comptime kv_params = KVCacheStaticParams(kv_num_heads, head_dim, True)

        var kv_collection = generic_get_paged_cache_with_scales[
            kv_dtype,
            DType.float32,
            kv_params,
            page_size,
            quantization_granularity,
        ](
            _mut_pointer_tile(kv_blocks.to_tile_tensor()),
            _imm_pointer_tile(page_stride.to_tile_tensor[.int64]().as_imm()),
            _imm_pointer_tile(cache_lengths.to_tile_tensor[.uint32]().as_imm()),
            _imm_pointer_tile(
                kv_lookup_table.to_tile_tensor[.uint32]().as_imm()
            ),
            _imm_pointer_tile(
                max_prompt_length.to_tile_tensor[.uint32]().as_imm()
            ),
            _imm_pointer_tile(
                max_cache_length.to_tile_tensor[.uint32]().as_imm()
            ),
            _mut_pointer_tile(kv_scales.to_tile_tensor[.float32]()),
            _imm_pointer_tile(
                scales_page_stride.to_tile_tensor[.int64]().as_imm()
            ),
            _imm_pointer_tile(
                kv_scales_lookup_table.to_tile_tensor[.uint32]().as_imm()
            ),
        )

        # Get the q_scales raw pointer for per-token Q scaling. The helper
        # takes a mutable pointer but only reads it; a non-pointer storage
        # policy aborts here at run time.
        var q_scale_ptr = (
            q_scales.to_tile_tensor[.float32]()
            .as_imm()
            .ptr.unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutAnyOrigin]()
        )

        generic_flare_mla_decode_kv_cache_ragged[
            mask_str=mask_str,
            per_token_scale_rope_aware=per_token_scale_rope_aware != 0,
            kernels=Self.kernels.Kernels,
        ](
            q.to_tile_tensor(),
            input_row_offsets.to_tile_tensor[.uint32](),
            kv_collection,
            layer_idx,
            scale,
            output.to_tile_tensor(),
            scalar_args.to_tile_tensor[.int64](),
            context,
            q_scale_ptr,
        )


@extensibility.register("attention.mla.decode.ragged.paged.scaled")
struct Struct_mla_decode_ragged_paged_scaled(_MLADecodeRaggedPagedScaledOp):
    """Registers the op for pre-SM100 NVIDIA GPUs and any other device."""

    comptime kernels: MLAGraphKernels = LegacyNVMLAGraph


@extensibility.register(
    "attention.mla.decode.ragged.paged.scaled", type="gpu", api="hip"
)
struct Struct_mla_decode_ragged_paged_scaled_amd(_MLADecodeRaggedPagedScaledOp):
    """Registers the op for AMD GPUs."""

    comptime kernels: MLAGraphKernels = AMDMLAGraph


@extensibility.register(
    "attention.mla.decode.ragged.paged.scaled",
    type="gpu",
    api="cuda",
    arch="sm_100a",
)
struct Struct_mla_decode_ragged_paged_scaled_sm100(
    _MLADecodeRaggedPagedScaledOp
):
    """Registers the op for B100 and B200."""

    comptime kernels: MLAGraphKernels = SM100MLAGraph


@extensibility.register(
    "attention.mla.decode.ragged.paged.scaled",
    type="gpu",
    api="cuda",
    arch="sm_103a",
)
struct Struct_mla_decode_ragged_paged_scaled_sm103(
    _MLADecodeRaggedPagedScaledOp
):
    """Registers the op for B300."""

    comptime kernels: MLAGraphKernels = SM100MLAGraph


trait _MLAPrefillRaggedPagedOp:
    """Implements `attention.mla.prefill.ragged.paged`.

    Each registration picks its kernels.
    """

    comptime kernels: MLAGraphKernels

    @inline(.always)
    @staticmethod
    def execute[
        mask_str: StaticString,
    ](
        output: Some[Output & Tensor],
        q: Some[Input & Tensor],
        k: Some[Input & Tensor],
        v: Some[Input & Tensor],
        buffer_row_offsets: Some[Input & Tensor],
        cache_offsets: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        layer_idx: UInt32,
        scale: Float32,
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(q), TileTensorable)
        comptime assert conforms_to(type_of(k), TileTensorable)
        comptime assert conforms_to(type_of(v), TileTensorable)
        comptime assert conforms_to(type_of(buffer_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(cache_offsets), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime qkv_dtype = type_of(kv_blocks).dtype
        comptime assert type_of(output).rank == 3
        comptime assert type_of(q).rank == 3
        comptime assert type_of(k).rank == 3
        comptime assert type_of(v).rank == 3
        comptime assert type_of(buffer_row_offsets).rank == 1
        comptime assert type_of(cache_offsets).rank == 1
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        var kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )
        generic_flare_mla_prefill_kv_cache_ragged[
            mask_str=mask_str,
            kernels=Self.kernels.Kernels,
        ](
            q.to_tile_tensor[qkv_dtype](),
            k.to_tile_tensor[qkv_dtype](),
            v.to_tile_tensor[qkv_dtype](),
            buffer_row_offsets.to_tile_tensor[.uint32](),
            cache_offsets.to_tile_tensor[.uint32](),
            input_row_offsets.to_tile_tensor[.uint32](),
            kv_collection,
            layer_idx,
            scale,
            output.to_tile_tensor(),
            context,
        )


@extensibility.register("attention.mla.prefill.ragged.paged")
struct Struct_mla_prefill_ragged_paged(_MLAPrefillRaggedPagedOp):
    """Registers the op for pre-SM100 NVIDIA GPUs and any other device."""

    comptime kernels: MLAGraphKernels = LegacyNVMLAGraph


@extensibility.register(
    "attention.mla.prefill.ragged.paged", type="gpu", api="hip"
)
struct Struct_mla_prefill_ragged_paged_amd(_MLAPrefillRaggedPagedOp):
    """Registers the op for AMD GPUs."""

    comptime kernels: MLAGraphKernels = AMDMLAGraph


@extensibility.register(
    "attention.mla.prefill.ragged.paged", type="gpu", api="cuda", arch="sm_100a"
)
struct Struct_mla_prefill_ragged_paged_sm100(_MLAPrefillRaggedPagedOp):
    """Registers the op for B100 and B200."""

    comptime kernels: MLAGraphKernels = SM100MLAGraph


@extensibility.register(
    "attention.mla.prefill.ragged.paged", type="gpu", api="cuda", arch="sm_103a"
)
struct Struct_mla_prefill_ragged_paged_sm103(_MLAPrefillRaggedPagedOp):
    """Registers the op for B300."""

    comptime kernels: MLAGraphKernels = SM100MLAGraph


@extensibility.register("attention.mla.prefill.ragged.plan")
struct Struct_mla_prefill_ragged_plan:
    """Registers the `attention.mla.prefill.ragged.plan` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute(
        buffer_row_offsets: Some[Output & Tensor],
        cache_offsets: Some[Output & Tensor],
        buffer_lengths: Some[Output & Tensor],
        input_row_offsets: Some[Input & Tensor],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        layer_idx: UInt32,
        buffer_tok_size: UInt32,
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(buffer_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(cache_offsets), TileTensorable)
        comptime assert conforms_to(type_of(buffer_lengths), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime dtype = type_of(kv_blocks).dtype
        comptime assert type_of(buffer_row_offsets).rank == 2
        comptime assert type_of(cache_offsets).rank == 2
        comptime assert type_of(buffer_lengths).rank == 1
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(kv_blocks)._shape_types[1].static_value == 1, (
            "Expected is_mla=True for MLA decompress, but found both k and"
            " v dimensions."
        )
        # The plan reads only cache lengths, so it builds the collection
        # without scales whatever the cache format, NVFP4 included. That
        # takes the tile overload: the trait one rejects a uint8 cache.
        var kv_collection = generic_get_paged_cache[
            dtype,
            KVCacheStaticParams(
                type_of(kv_blocks)._shape_types[4].static_value,
                type_of(kv_blocks)._shape_types[5].static_value,
                True,
            ),
            type_of(kv_blocks)._shape_types[3].static_value,
        ](
            _mut_pointer_tile(kv_blocks.to_tile_tensor[dtype]()),
            _imm_pointer_tile(page_stride.to_tile_tensor[.int64]().as_imm()),
            _imm_pointer_tile(cache_lengths.to_tile_tensor[.uint32]().as_imm()),
            _imm_pointer_tile(
                kv_lookup_table.to_tile_tensor[.uint32]().as_imm()
            ),
            _imm_pointer_tile(
                max_prompt_length.to_tile_tensor[.uint32]().as_imm()
            ),
            _imm_pointer_tile(
                max_cache_length.to_tile_tensor[.uint32]().as_imm()
            ),
        )
        generic_flare_mla_prefill_ragged_paged_plan(
            input_row_offsets.to_tile_tensor[.uint32](),
            kv_collection,
            layer_idx,
            buffer_tok_size,
            buffer_row_offsets.to_tile_tensor[.uint32](),
            cache_offsets.to_tile_tensor[.uint32](),
            buffer_lengths.to_tile_tensor[.int32](),
            context,
        )


@extensibility.register("attention.mla.decompress.k.cache.ragged.paged")
struct Struct_mla_decompress_k_cache_ragged_paged:
    """Registers the `attention.mla.decompress.k.cache.ragged.paged` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute(
        k_latent_buffer: Some[Output & Tensor],
        k_buffer: Some[Output & Tensor],
        buffer_row_offsets_1d: Some[Input & Tensor],
        cache_offsets_1d: Some[Input & Tensor],
        buffer_length: Int32,
        weight: Some[Input & Tensor],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        layer_idx: UInt32,
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(k_latent_buffer), TileTensorable)
        comptime assert conforms_to(type_of(k_buffer), TileTensorable)
        comptime assert conforms_to(
            type_of(buffer_row_offsets_1d), TileTensorable
        )
        comptime assert conforms_to(type_of(cache_offsets_1d), TileTensorable)
        comptime assert conforms_to(type_of(weight), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime dtype = type_of(kv_blocks).dtype
        comptime assert type_of(k_latent_buffer).rank == 2
        comptime assert type_of(k_buffer).rank == 2
        comptime assert type_of(buffer_row_offsets_1d).rank == 1
        comptime assert type_of(cache_offsets_1d).rank == 1
        comptime assert type_of(weight).rank == 2
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        var kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )
        generic_flare_mla_decompress_k_cache_ragged_paged(
            buffer_row_offsets_1d.to_tile_tensor[.uint32](),
            cache_offsets_1d.to_tile_tensor[.uint32](),
            buffer_length,
            weight.to_tile_tensor[dtype](),
            kv_collection,
            layer_idx,
            k_latent_buffer.to_tile_tensor[dtype](),
            k_buffer.to_tile_tensor[dtype](),
            context,
        )


trait _MLAPrefillGraphPagedOp:
    """Implements `attention.mla.graph.prefill.paged.fp8`.

    Each registration picks its kernels.
    """

    comptime kernels: MLAGraphKernels

    @inline(.always)
    @staticmethod
    def execute[
        m_scale_granularity: Int,
        n_scale_granularity: Int,
        k_scale_granularity: Int,
        mask_str: StaticString,
    ](
        output: Some[Output & Tensor],
        q: Some[Input & Tensor],
        kv: Some[Input & Fused & LoadAccess],
        input_row_offsets: Some[Input & Tensor],
        freqs_cis: Some[Input & Tensor],
        kv_norm_gamma: Some[Input & Tensor],
        buffer_row_offsets_1d: Some[Input & Tensor],
        cache_offsets_1d: Some[Input & Tensor],
        buffer_length: Int32,
        w_k: Some[Input & Tensor],
        w_uv: Some[Input & Tensor],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        layer_idx: UInt32,
        scale: Float32,
        epsilon: Float32,
        w_k_scale: Some[Input & Tensor],
        w_uv_scale: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(q), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(freqs_cis), TileTensorable)
        comptime assert conforms_to(type_of(kv_norm_gamma), TileTensorable)
        comptime assert conforms_to(
            type_of(buffer_row_offsets_1d), TileTensorable
        )
        comptime assert conforms_to(type_of(cache_offsets_1d), TileTensorable)
        comptime assert conforms_to(type_of(w_k), TileTensorable)
        comptime assert conforms_to(type_of(w_uv), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert conforms_to(type_of(w_k_scale), TileTensorable)
        comptime assert conforms_to(type_of(w_uv_scale), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime fp8_dtype = type_of(w_k).dtype
        comptime fp8_scale_dtype = type_of(w_k_scale).dtype
        comptime assert type_of(output).rank == 3
        comptime assert type_of(q).rank == 3
        comptime assert type_of(kv).rank == 2
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(freqs_cis).rank == 2
        comptime assert type_of(kv_norm_gamma).rank == 1
        comptime assert type_of(buffer_row_offsets_1d).rank == 1
        comptime assert type_of(cache_offsets_1d).rank == 1
        comptime assert type_of(w_k).rank == 2
        comptime assert type_of(w_uv).rank == 3
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(w_k_scale).rank == 2
        comptime assert type_of(w_uv_scale).rank == 3
        comptime assert type_of(kv).dtype == DType.bfloat16
        var kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor[dtype](),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )

        comptime assert (
            context.target.is_gpu()
        ), "attention.mla.graph.prefill.paged.fp8 is only supported on GPU"

        comptime assert conforms_to(type_of(kv), UnsafeFusedView)
        var kv_view = unsafe_fused_view(kv)
        comptime assert conforms_to(type_of(kv_view), Loadable)

        @inline(.always)
        def kv_input_fn[
            width: Int
        ](coords: IndexList[2]) {var kv_view} -> SIMD[.bfloat16, width]:
            return rebind[SIMD[.bfloat16, width]](
                kv_view.load[width=width, element_alignment=width](
                    Coord(coords)
                )
            )

        mla_prefill_branch_fp8[
            m_scale_granularity=m_scale_granularity,
            n_scale_granularity=n_scale_granularity,
            k_scale_granularity=k_scale_granularity,
            mask_str=mask_str,
            kernels=Self.kernels,
        ](
            output.to_tile_tensor(),
            q.to_tile_tensor[dtype](),
            input_row_offsets.to_tile_tensor[.uint32](),
            freqs_cis.to_tile_tensor(),
            kv_norm_gamma.to_tile_tensor(),
            kv_collection,
            layer_idx,
            scale,
            epsilon,
            buffer_row_offsets_1d.to_tile_tensor[.uint32](),
            cache_offsets_1d.to_tile_tensor[.uint32](),
            Int(buffer_length),
            w_k.to_tile_tensor(),
            w_k_scale.to_tile_tensor(),
            w_uv.to_tile_tensor[fp8_dtype](),
            w_uv_scale.to_tile_tensor[fp8_scale_dtype](),
            kv_input_fn,
            context,
        )


@extensibility.register("attention.mla.graph.prefill.paged.fp8")
struct Struct_mla_prefill_graph_paged(_MLAPrefillGraphPagedOp):
    """Registers the op for pre-SM100 NVIDIA GPUs and any other device."""

    comptime kernels: MLAGraphKernels = LegacyNVMLAGraph


@extensibility.register(
    "attention.mla.graph.prefill.paged.fp8", type="gpu", api="hip"
)
struct Struct_mla_prefill_graph_paged_amd(_MLAPrefillGraphPagedOp):
    """Registers the op for AMD GPUs."""

    comptime kernels: MLAGraphKernels = AMDMLAGraph


@extensibility.register(
    "attention.mla.graph.prefill.paged.fp8",
    type="gpu",
    api="cuda",
    arch="sm_100a",
)
struct Struct_mla_prefill_graph_paged_sm100(_MLAPrefillGraphPagedOp):
    """Registers the op for B100 and B200."""

    comptime kernels: MLAGraphKernels = SM100MLAGraph


@extensibility.register(
    "attention.mla.graph.prefill.paged.fp8",
    type="gpu",
    api="cuda",
    arch="sm_103a",
)
struct Struct_mla_prefill_graph_paged_sm103(_MLAPrefillGraphPagedOp):
    """Registers the op for B300."""

    comptime kernels: MLAGraphKernels = SM100MLAGraph


trait _MLADecodeGraphPagedFP8Op:
    """Implements `attention.mla.graph.decode.paged.fp8`.

    Each registration picks its kernels.
    """

    comptime kernels: MLAGraphKernels

    @inline(.always)
    @staticmethod
    def execute[
        m_scale_granularity: Int,
        n_scale_granularity: Int,
        k_scale_granularity: Int,
        mask_str: StaticString,
    ](
        output: Some[Output & Tensor],
        q: Some[Input & Tensor],
        kv: Some[Input & Fused & LoadAccess],
        input_row_offsets: Some[Input & Tensor],
        freqs_cis: Some[Input & Tensor],
        kv_norm_gamma: Some[Input & Tensor],
        w_uk: Some[Input & Tensor],
        w_uv: Some[Input & Tensor],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        layer_idx: UInt32,
        scale: Float32,
        epsilon: Float32,
        w_uk_scale: Some[Input & Tensor],
        w_uv_scale: Some[Input & Tensor],
        scalar_args: Some[Input & Tensor],
        num_partitions_scalar: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(q), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(freqs_cis), TileTensorable)
        comptime assert conforms_to(type_of(kv_norm_gamma), TileTensorable)
        comptime assert conforms_to(type_of(w_uk), TileTensorable)
        comptime assert conforms_to(type_of(w_uv), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert conforms_to(type_of(w_uk_scale), TileTensorable)
        comptime assert conforms_to(type_of(w_uv_scale), TileTensorable)
        comptime assert conforms_to(type_of(scalar_args), TileTensorable)
        comptime assert conforms_to(
            type_of(num_partitions_scalar), TileTensorable
        )
        comptime dtype = type_of(output).dtype
        comptime fp8_dtype = type_of(w_uk).dtype
        comptime fp8_scale_dtype = type_of(w_uk_scale).dtype
        comptime assert type_of(output).rank == 3
        comptime assert type_of(q).rank == 3
        comptime assert type_of(kv).rank == 2
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(freqs_cis).rank == 2
        comptime assert type_of(kv_norm_gamma).rank == 1
        comptime assert type_of(w_uk).rank == 3
        comptime assert type_of(w_uv).rank == 3
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(w_uk_scale).rank == 3
        comptime assert type_of(w_uv_scale).rank == 3
        comptime assert type_of(scalar_args).rank == 1
        comptime assert type_of(num_partitions_scalar).rank == 1
        comptime assert type_of(kv).dtype == DType.bfloat16
        var kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor[dtype](),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )

        comptime assert (
            context.target.is_gpu()
        ), "attention.mla.graph.decode.paged.fp8 is only supported on GPU"

        comptime assert conforms_to(type_of(kv), UnsafeFusedView)
        var kv_view = unsafe_fused_view(kv)
        comptime assert conforms_to(type_of(kv_view), Loadable)

        @inline(.always)
        def kv_input_fn[
            width: Int
        ](coords: IndexList[2]) {var kv_view} -> SIMD[.bfloat16, width]:
            return rebind[SIMD[.bfloat16, width]](
                kv_view.load[width=width, element_alignment=width](
                    Coord(coords)
                )
            )

        var num_partitions_proj = Int(
            num_partitions_scalar.to_tile_tensor[.int64]().load[width=1](
                Coord(0)
            )
        )

        with Trace[TraceLevel.OP, target=context.target.device_kind_label()](
            "attention.mla.graph.decode.paged.fp8",
            task_id=get_safe_task_id(context),
        ):
            mla_decode_branch_fp8[
                m_scale_granularity=m_scale_granularity,
                n_scale_granularity=n_scale_granularity,
                k_scale_granularity=k_scale_granularity,
                mask_str=mask_str,
                kernels=Self.kernels,
            ](
                output.to_tile_tensor(),
                q.to_tile_tensor[dtype](),
                input_row_offsets.to_tile_tensor[.uint32](),
                freqs_cis.to_tile_tensor(),
                kv_norm_gamma.to_tile_tensor(),
                kv_collection,
                layer_idx,
                scale,
                epsilon,
                w_uk.to_tile_tensor(),
                w_uk_scale.to_tile_tensor(),
                w_uv.to_tile_tensor[fp8_dtype](),
                w_uv_scale.to_tile_tensor[fp8_scale_dtype](),
                scalar_args.to_tile_tensor[.int64](),
                kv_input_fn,
                context,
                num_partitions_in=num_partitions_proj,
            )


@extensibility.register("attention.mla.graph.decode.paged.fp8")
struct Struct_mla_decode_graph_paged_fp8(_MLADecodeGraphPagedFP8Op):
    """Registers the op for pre-SM100 NVIDIA GPUs and any other device."""

    comptime kernels: MLAGraphKernels = LegacyNVMLAGraph


@extensibility.register(
    "attention.mla.graph.decode.paged.fp8", type="gpu", api="hip"
)
struct Struct_mla_decode_graph_paged_fp8_amd(_MLADecodeGraphPagedFP8Op):
    """Registers the op for AMD GPUs."""

    comptime kernels: MLAGraphKernels = AMDMLAGraph


@extensibility.register(
    "attention.mla.graph.decode.paged.fp8",
    type="gpu",
    api="cuda",
    arch="sm_100a",
)
struct Struct_mla_decode_graph_paged_fp8_sm100(_MLADecodeGraphPagedFP8Op):
    """Registers the op for B100 and B200."""

    comptime kernels: MLAGraphKernels = SM100MLAGraph


@extensibility.register(
    "attention.mla.graph.decode.paged.fp8",
    type="gpu",
    api="cuda",
    arch="sm_103a",
)
struct Struct_mla_decode_graph_paged_fp8_sm103(_MLADecodeGraphPagedFP8Op):
    """Registers the op for B300."""

    comptime kernels: MLAGraphKernels = SM100MLAGraph


@inline(.always)
def _sparse_indices_ptr[
    indices_stride: Int
](
    sparse_indices: TileTensor[DType.int32, address_space=.GENERIC, ...],
    num_tokens: Int,
) raises -> MutPointer[Int32, MutAnyOrigin]:
    """Returns the sparse indices after checking their shape.

    The kernels index the buffer flat, `row * indices_stride + slot`, one row
    per query token.
    """
    comptime assert sparse_indices.rank == 2
    if (
        Int(sparse_indices.dim[0]()) != num_tokens
        or Int(sparse_indices.dim[1]()) != indices_stride
    ):
        raise Error("sparse_indices must be [query tokens, indices_stride]")
    return sparse_indices.ptr.unsafe_mut_cast[True]().unsafe_origin_cast[
        MutAnyOrigin
    ]()


@inline(.always)
def _sparse_gather_indices[
    page_size: Int,
    indices_stride: Int,
](
    mut scratch: Optional[DeviceBuffer[DType.int32]],
    sparse_indices: TileTensor[DType.int32, address_space=.GENERIC, ...],
    input_row_offsets: TileTensor[DType.uint32, address_space=.GENERIC, ...],
    kv_lookup_table: TileTensor[DType.uint32, address_space=.GENERIC, ...],
    kv_blocks: TileTensor,
    ctx: DeviceContext,
) raises -> MutPointer[Int32, MutAnyOrigin]:
    """Remaps the logical `sparse_indices` to physical rows in `scratch`.

    The caller keeps `scratch` alive until its kernels are enqueued.
    """
    var buf = ctx.enqueue_create_buffer[.int32](sparse_indices.num_elements())
    var ptr = buf.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    paged_sparse_kv_index_remap[page_size, indices_stride](
        ptr,
        sparse_indices,
        input_row_offsets,
        kv_lookup_table,
        kv_blocks,
        ctx,
    )
    scratch = buf^
    return ptr


@inline(.always)
def _sparse_decode_indices[
    page_size: Int,
    indices_stride: Int,
    physical_indices: Bool,
](
    mut scratch: Optional[DeviceBuffer[DType.int32]],
    sparse_indices: TileTensor[DType.int32, address_space=.GENERIC, ...],
    input_row_offsets: TileTensor[DType.uint32, address_space=.GENERIC, ...],
    kv_lookup_table: TileTensor[DType.uint32, address_space=.GENERIC, ...],
    kv_blocks: TileTensor,
    num_tokens: Int,
    ctx: DeviceContext,
) raises -> MutPointer[Int32, MutAnyOrigin]:
    """Returns the sparse indices the decode branch reads.

    The AMD sparse MLA arm gathers paged physical rows, so logical indices are
    remapped for it here; the SM100 kernels look logical indices up
    themselves. Pair with `_sparse_decode_physical`.
    """
    var indices = _sparse_indices_ptr[indices_stride](
        sparse_indices, num_tokens
    )
    comptime if ctx.target.is_amd_gpu() and not physical_indices:
        return _sparse_gather_indices[page_size, indices_stride](
            scratch,
            sparse_indices,
            input_row_offsets,
            kv_lookup_table,
            kv_blocks,
            ctx,
        )
    return indices


@inline(.always)
def _sparse_decode_physical[physical_indices: Bool]() -> Bool:
    """Returns whether `_sparse_decode_indices` returns physical rows."""
    return physical_indices or DeviceContext[].target.is_amd_gpu()


@inline(.always)
def _sparse_prefill_indices[
    page_size: Int,
    indices_stride: Int,
    cache_dtype: DType,
    num_heads: Int,
    physical_indices: Bool,
    q_dtype: DType = DType.bfloat16,
](
    mut scratch: Optional[DeviceBuffer[DType.int32]],
    sparse_indices: TileTensor[DType.int32, address_space=.GENERIC, ...],
    input_row_offsets: TileTensor[DType.uint32, address_space=.GENERIC, ...],
    kv_lookup_table: TileTensor[DType.uint32, address_space=.GENERIC, ...],
    kv_blocks: TileTensor[cache_dtype, ...],
    max_seq_len: Int,
    ctx: DeviceContext,
) raises -> OptionalReg[MutPointer[Int32, MutAnyOrigin]]:
    """Returns the remapped rows for the prefill branch of a prefill+decode op.

    Without the indexer's physical rows only the prefill branch needs a remap;
    the decode kernels look logical indices up. The result is empty when the
    indexer supplied physical rows or the batch runs a decode kernel.
    """
    comptime if not physical_indices and not mla_sparse_prefill_on_decode_kernel[
        q_dtype, cache_dtype, num_heads
    ]():
        if not mla_graph_takes_decode_branch[
            cache_dtype, num_heads, sparse_mla=True
        ](max_seq_len):
            return _sparse_gather_indices[page_size, indices_stride](
                scratch,
                sparse_indices,
                input_row_offsets,
                kv_lookup_table,
                kv_blocks,
                ctx,
            )
    return None


def _amd_sparse_decode_partitions(
    batch_size: Int, indices_stride: Int, num_heads: Int, ctx: DeviceContext
) raises -> Optional[Int]:
    """Returns the split-K count an FP8 sparse MLA decode op forwards.

    AMD reads the selected rows in place at a count the op sizes itself
    rather than the graph's scalar, so a host whose heuristic follows the
    context (Mach's SM100 port gives 2 below 1.5k tokens, 4x slower) still
    gets the tuned split. MAX's capture key resolves the same function, so
    it equals the launched split. SM100 caps the split by the top-k mask
    itself, which a count would override, so it gets none.
    """
    comptime if ctx.target.is_amd_gpu():
        return amd_sparse_mla_inplace_decode_num_partitions(
            batch_size,
            indices_stride,
            num_heads,
            ctx.device().get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT),
        )
    return None


trait _MLADecodeGraphPagedFP8SparseOp:
    """Implements `attention.mla.graph.decode.paged.fp8.sparse`.

    Each registration picks its kernels.

    Under decode context parallelism (`dcp_size > 1`) every rank of the
    group receives the same lookup table and global `cache_lengths`, and the
    fused latent store writes a new token only on the rank that owns its
    position (`ContextParallelLayout[dcp_size, dcp_interleave]`), at that
    rank's local slot. `dcp_rank` is a host scalar after `selection_lag`;
    outside a group (`dcp_size == 1`) the caller passes 0 and the layout never
    reads it.
    """

    comptime kernels: MLAGraphKernels

    @inline(.always)
    @staticmethod
    def execute[
        m_scale_granularity: Int,
        n_scale_granularity: Int,
        k_scale_granularity: Int,
        mask_str: StaticString,
        indices_stride: Int,
        # Whether `sparse_indices` holds the indexer's paged physical rows
        # (`attention.mla.indexer.ragged.float8.paged.remap`) rather than logical
        # token positions, which the kernels look up in the page table.
        physical_indices: Bool,
        # Read-once shared-index MTP fold (KERN-3141). Set from the Python
        # `index_share` op attribute (default False -> per-position path,
        # byte-identical). Forwarded to the decode dispatch as
        # `fold_shared_index`.
        index_share: Bool = False,
        # The sparse index producer's row layout (`SparseRowLayout`), which
        # bounds each row by its closed-form valid length. The defaults are
        # plain top-k, which adds no bound.
        index_kpool: Int = 1,
        index_pool_topk: Int = 0,
        index_kpool_select_tail: Bool = True,
        # Set by the Python wrapper from whether each operand was supplied.
        # An absent operand arrives as a placeholder tensor that is never
        # read; the decode kernels are specialized on its absence.
        has_topk_lengths: Bool = True,
        has_attn_sink: Bool = True,
        # Set from the Python `attn_sink_inert` attribute when the caller knows
        # the sink is inert (-1e38); the AMD sparse arms reject a live one.
        attn_sink_inert: Bool = False,
        dcp_size: Int = 1,
        dcp_interleave: Int = 1,
    ](
        output: Some[Output & Tensor],
        q: Some[Input & Tensor],
        kv: Some[Input & Fused & LoadAccess],
        input_row_offsets: Some[Input & Tensor],
        freqs_cis: Some[Input & Tensor],
        kv_norm_gamma: Some[Input & Tensor],
        w_uk: Some[Input & Tensor],
        w_uv: Some[Input & Tensor],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        layer_idx: UInt32,
        scale: Float32,
        epsilon: Float32,
        w_uk_scale: Some[Input & Tensor],
        w_uv_scale: Some[Input & Tensor],
        scalar_args: Some[Input & Tensor],
        sparse_indices: Some[Input & Tensor],
        topk_lengths: Some[Input & Tensor],
        attn_sink: Some[Input & Tensor],
        num_partitions_scalar: Some[Input & Tensor],
        selection_lag: Int32,
        dcp_rank: Int32,
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(q), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(freqs_cis), TileTensorable)
        comptime assert conforms_to(type_of(kv_norm_gamma), TileTensorable)
        comptime assert conforms_to(type_of(w_uk), TileTensorable)
        comptime assert conforms_to(type_of(w_uv), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert conforms_to(type_of(w_uk_scale), TileTensorable)
        comptime assert conforms_to(type_of(w_uv_scale), TileTensorable)
        comptime assert conforms_to(type_of(scalar_args), TileTensorable)
        comptime assert conforms_to(type_of(sparse_indices), TileTensorable)
        comptime assert conforms_to(type_of(topk_lengths), TileTensorable)
        comptime assert conforms_to(type_of(attn_sink), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime fp8_dtype = type_of(w_uk).dtype
        comptime fp8_scale_dtype = type_of(w_uk_scale).dtype
        comptime assert type_of(output).rank == 3
        comptime assert type_of(q).rank == 3
        comptime assert type_of(kv).rank == 2
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(freqs_cis).rank == 2
        comptime assert type_of(kv_norm_gamma).rank == 1
        comptime assert type_of(w_uk).rank == 3
        comptime assert type_of(w_uv).rank == 3
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(w_uk_scale).rank == 3
        comptime assert type_of(w_uv_scale).rank == 3
        comptime assert type_of(scalar_args).rank == 1
        comptime assert type_of(sparse_indices).rank == 2
        comptime assert type_of(topk_lengths).rank == 1
        comptime assert type_of(attn_sink).rank == 1
        comptime assert type_of(num_partitions_scalar).rank == 1
        comptime assert type_of(q).dtype == dtype
        comptime assert type_of(w_uv).dtype == fp8_dtype
        comptime assert type_of(w_uv_scale).dtype == fp8_scale_dtype
        comptime assert type_of(kv).dtype == DType.bfloat16
        comptime if dcp_size > 1:
            if dcp_rank < 0 or Int(dcp_rank) >= dcp_size:
                raise Error(
                    t"dcp_rank {dcp_rank} is outside a group of {dcp_size}"
                )
        var kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )

        comptime assert context.target.is_gpu(), (
            "attention.mla.graph.decode.paged.fp8.sparse is only supported"
            " on GPU"
        )
        comptime assert not (index_share and physical_indices), (
            "index_share masks a shared row by logical key position, so it"
            " needs logical sparse_indices"
        )

        comptime assert conforms_to(type_of(kv), UnsafeFusedView)
        var kv_view = unsafe_fused_view(kv)
        comptime assert conforms_to(type_of(kv_view), Loadable)

        @inline(.always)
        def kv_input_fn[
            width: Int
        ](coords: IndexList[2]) {var kv_view} -> SIMD[.bfloat16, width]:
            return rebind[SIMD[.bfloat16, width]](
                kv_view.load[width=width, element_alignment=width](
                    Coord(coords)
                )
            )

        var num_tokens = Int(q.shape()[0].value())
        comptime if has_topk_lengths:
            var required_lengths = Int(input_row_offsets.shape()[0].value()) - 1
            if Int(topk_lengths.shape()[0].value()) < required_lengths:
                raise Error(
                    "topk_lengths does not cover its batch/token domain"
                )

        var topk_lengths_ptr = maybe_null_pointer[has_topk_lengths](
            topk_lengths.to_tile_tensor[.int32]().ptr
        )
        var attn_sink_ptr = maybe_null_pointer[has_attn_sink](
            attn_sink.to_tile_tensor[.float32]().ptr
        )
        # Rows the indexer computed for their own token are packed to a
        # closed-form length (`SparseRowLayout`), which the mask applies
        # without reading them. A shared list serving several query positions
        # needs each slot's logical key position for causality. With one query
        # position per sequence, the shared list is just a reused row, which
        # `selection_lag` bounds, and the fold does not launch.
        var logical_indices_ptr = OptionalReg[MutPointer[Int32, MutAnyOrigin]]()
        comptime if index_share:
            if max_prompt_length.to_tile_tensor[.uint32]().ptr[0] > 1:
                logical_indices_ptr = (
                    sparse_indices.to_tile_tensor[.int32]()
                    .ptr.unsafe_mut_cast[True]()
                    .unsafe_origin_cast[MutAnyOrigin]()
                )

        with Trace[TraceLevel.OP, target=context.target.device_kind_label()](
            "attention.mla.graph.decode.paged.fp8.sparse",
            task_id=get_safe_task_id(context),
        ):
            var gather_indices = _sparse_indices_ptr[indices_stride](
                sparse_indices.to_tile_tensor[.int32](), num_tokens
            )
            mla_decode_branch_fp8[
                m_scale_granularity=m_scale_granularity,
                n_scale_granularity=n_scale_granularity,
                k_scale_granularity=k_scale_granularity,
                mask_str=mask_str,
                sparse_mla=True,
                fold_shared_index=index_share,
                physical_indices=physical_indices,
                sparse_row=SparseRowLayout(
                    kpool=index_kpool,
                    pool_topk=index_pool_topk,
                    select_tail=index_kpool_select_tail,
                ),
                attn_sink_inert=attn_sink_inert,
                kernels=Self.kernels,
                dcp_size=dcp_size,
                dcp_interleave=dcp_interleave,
            ](
                output.to_tile_tensor[dtype](),
                q.to_tile_tensor[dtype](),
                input_row_offsets.to_tile_tensor[.uint32](),
                freqs_cis.to_tile_tensor(),
                kv_norm_gamma.to_tile_tensor(),
                kv_collection,
                layer_idx,
                scale,
                epsilon,
                w_uk.to_tile_tensor[fp8_dtype](),
                w_uk_scale.to_tile_tensor[fp8_scale_dtype](),
                w_uv.to_tile_tensor[fp8_dtype](),
                w_uv_scale.to_tile_tensor[fp8_scale_dtype](),
                scalar_args.to_tile_tensor[.int64](),
                kv_input_fn,
                context,
                gather_indices,
                indices_stride,
                topk_lengths_ptr,
                attn_sink_ptr,
                num_partitions_in=_amd_sparse_decode_partitions(
                    num_tokens,
                    indices_stride,
                    type_of(q)._shape_types[1].static_value,
                    context,
                ),
                logical_indices=logical_indices_ptr,
                selection_lag=Int(selection_lag),
                dcp_rank=Int(dcp_rank),
            )


@extensibility.register("attention.mla.graph.decode.paged.fp8.sparse")
struct Struct_mla_decode_graph_paged_fp8_sparse(
    _MLADecodeGraphPagedFP8SparseOp
):
    """Registers the op for pre-SM100 NVIDIA GPUs and any other device."""

    comptime kernels: MLAGraphKernels = LegacyNVMLAGraph


@extensibility.register(
    "attention.mla.graph.decode.paged.fp8.sparse", type="gpu", api="hip"
)
struct Struct_mla_decode_graph_paged_fp8_sparse_amd(
    _MLADecodeGraphPagedFP8SparseOp
):
    """Registers the op for AMD GPUs."""

    comptime kernels: MLAGraphKernels = AMDMLAGraph


@extensibility.register(
    "attention.mla.graph.decode.paged.fp8.sparse",
    type="gpu",
    api="cuda",
    arch="sm_100a",
)
struct Struct_mla_decode_graph_paged_fp8_sparse_sm100(
    _MLADecodeGraphPagedFP8SparseOp
):
    """Registers the op for B100 and B200."""

    comptime kernels: MLAGraphKernels = SM100MLAGraph


@extensibility.register(
    "attention.mla.graph.decode.paged.fp8.sparse",
    type="gpu",
    api="cuda",
    arch="sm_103a",
)
struct Struct_mla_decode_graph_paged_fp8_sparse_sm103(
    _MLADecodeGraphPagedFP8SparseOp
):
    """Registers the op for B300."""

    comptime kernels: MLAGraphKernels = SM100MLAGraph


trait _MLAPrefillGraphBF16PagedOp:
    """Implements `attention.mla.graph.prefill.paged`.

    Each registration picks its kernels.
    """

    comptime kernels: MLAGraphKernels

    @inline(.always)
    @staticmethod
    def execute[
        mask_str: StaticString,
    ](
        output: Some[Output & Tensor],
        q: Some[Input & Tensor],
        kv: Some[Input & Fused & LoadAccess],
        input_row_offsets: Some[Input & Tensor],
        freqs_cis: Some[Input & Tensor],
        kv_norm_gamma: Some[Input & Tensor],
        buffer_row_offsets_1d: Some[Input & Tensor],
        cache_offsets_1d: Some[Input & Tensor],
        buffer_length: Int32,
        w_k: Some[Input & Tensor],
        w_uv: Some[Input & Tensor],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        layer_idx: UInt32,
        scale: Float32,
        epsilon: Float32,
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(q), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(freqs_cis), TileTensorable)
        comptime assert conforms_to(type_of(kv_norm_gamma), TileTensorable)
        comptime assert conforms_to(
            type_of(buffer_row_offsets_1d), TileTensorable
        )
        comptime assert conforms_to(type_of(cache_offsets_1d), TileTensorable)
        comptime assert conforms_to(type_of(w_k), TileTensorable)
        comptime assert conforms_to(type_of(w_uv), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert type_of(output).rank == 3
        comptime assert type_of(q).rank == 3
        comptime assert type_of(kv).rank == 2
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(freqs_cis).rank == 2
        comptime assert type_of(kv_norm_gamma).rank == 1
        comptime assert type_of(buffer_row_offsets_1d).rank == 1
        comptime assert type_of(cache_offsets_1d).rank == 1
        comptime assert type_of(w_k).rank == 2
        comptime assert type_of(w_uv).rank == 3
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(kv).dtype == DType.bfloat16
        var kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )

        comptime assert (
            context.target.is_gpu()
        ), "attention.mla.graph.prefill.paged is only supported on GPU"

        comptime assert conforms_to(type_of(kv), UnsafeFusedView)
        var kv_view = unsafe_fused_view(kv)
        comptime assert conforms_to(type_of(kv_view), Loadable)

        @inline(.always)
        def kv_input_fn[
            width: Int
        ](coords: IndexList[2]) {var kv_view} -> SIMD[.bfloat16, width]:
            return rebind[SIMD[.bfloat16, width]](
                kv_view.load[width=width, element_alignment=width](
                    Coord(coords)
                )
            )

        mla_prefill_branch_bf16[
            mask_str=mask_str,
            kernels=Self.kernels,
        ](
            output.to_tile_tensor[.bfloat16](),
            q.to_tile_tensor[.bfloat16](),
            input_row_offsets.to_tile_tensor[.uint32](),
            freqs_cis.to_tile_tensor(),
            kv_norm_gamma.to_tile_tensor(),
            kv_collection,
            layer_idx,
            scale,
            epsilon,
            buffer_row_offsets_1d.to_tile_tensor[.uint32](),
            cache_offsets_1d.to_tile_tensor[.uint32](),
            Int(buffer_length),
            w_k.to_tile_tensor[.bfloat16](),
            w_uv.to_tile_tensor[.bfloat16](),
            kv_input_fn,
            context,
        )


@extensibility.register("attention.mla.graph.prefill.paged")
struct Struct_mla_prefill_graph_bf16_paged(_MLAPrefillGraphBF16PagedOp):
    """Registers the op for pre-SM100 NVIDIA GPUs and any other device."""

    comptime kernels: MLAGraphKernels = LegacyNVMLAGraph


@extensibility.register(
    "attention.mla.graph.prefill.paged", type="gpu", api="hip"
)
struct Struct_mla_prefill_graph_bf16_paged_amd(_MLAPrefillGraphBF16PagedOp):
    """Registers the op for AMD GPUs."""

    comptime kernels: MLAGraphKernels = AMDMLAGraph


@extensibility.register(
    "attention.mla.graph.prefill.paged", type="gpu", api="cuda", arch="sm_100a"
)
struct Struct_mla_prefill_graph_bf16_paged_sm100(_MLAPrefillGraphBF16PagedOp):
    """Registers the op for B100 and B200."""

    comptime kernels: MLAGraphKernels = SM100MLAGraph


@extensibility.register(
    "attention.mla.graph.prefill.paged", type="gpu", api="cuda", arch="sm_103a"
)
struct Struct_mla_prefill_graph_bf16_paged_sm103(_MLAPrefillGraphBF16PagedOp):
    """Registers the op for B300."""

    comptime kernels: MLAGraphKernels = SM100MLAGraph


trait _MLADecodeGraphBF16PagedOp:
    """Implements `attention.mla.graph.decode.paged`.

    Each registration picks its kernels.
    """

    comptime kernels: MLAGraphKernels

    @inline(.always)
    @staticmethod
    def execute[
        mask_str: StaticString,
    ](
        output: Some[Output & Tensor],
        q: Some[Input & Tensor],
        kv: Some[Input & Fused & LoadAccess],
        input_row_offsets: Some[Input & Tensor],
        freqs_cis: Some[Input & Tensor],
        kv_norm_gamma: Some[Input & Tensor],
        w_uk: Some[Input & Tensor],
        w_uv: Some[Input & Tensor],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        layer_idx: UInt32,
        scale: Float32,
        epsilon: Float32,
        scalar_args: Some[Input & Tensor],
        num_partitions_scalar: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(q), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(freqs_cis), TileTensorable)
        comptime assert conforms_to(type_of(kv_norm_gamma), TileTensorable)
        comptime assert conforms_to(type_of(w_uk), TileTensorable)
        comptime assert conforms_to(type_of(w_uv), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert conforms_to(type_of(scalar_args), TileTensorable)
        comptime assert conforms_to(
            type_of(num_partitions_scalar), TileTensorable
        )
        comptime assert type_of(output).rank == 3
        comptime assert type_of(q).rank == 3
        comptime assert type_of(kv).rank == 2
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(freqs_cis).rank == 2
        comptime assert type_of(kv_norm_gamma).rank == 1
        comptime assert type_of(w_uk).rank == 3
        comptime assert type_of(w_uv).rank == 3
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(scalar_args).rank == 1
        comptime assert type_of(num_partitions_scalar).rank == 1
        comptime assert type_of(kv).dtype == DType.bfloat16
        var kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )

        comptime assert (
            context.target.is_gpu()
        ), "attention.mla.graph.decode.paged is only supported on GPU"

        comptime assert conforms_to(type_of(kv), UnsafeFusedView)
        var kv_view = unsafe_fused_view(kv)
        comptime assert conforms_to(type_of(kv_view), Loadable)

        @inline(.always)
        def kv_input_fn[
            width: Int
        ](coords: IndexList[2]) {var kv_view} -> SIMD[.bfloat16, width]:
            return rebind[SIMD[.bfloat16, width]](
                kv_view.load[width=width, element_alignment=width](
                    Coord(coords)
                )
            )

        var num_partitions_proj = Int(
            num_partitions_scalar.to_tile_tensor[.int64]().load[width=1](
                Coord(0)
            )
        )

        with Trace[TraceLevel.OP, target=context.target.device_kind_label()](
            "attention.mla.graph.decode.paged",
            task_id=get_safe_task_id(context),
        ):
            mla_decode_branch_bf16[
                mask_str=mask_str,
                kernels=Self.kernels,
            ](
                output.to_tile_tensor[.bfloat16](),
                q.to_tile_tensor[.bfloat16](),
                input_row_offsets.to_tile_tensor[.uint32](),
                freqs_cis.to_tile_tensor(),
                kv_norm_gamma.to_tile_tensor(),
                kv_collection,
                layer_idx,
                scale,
                epsilon,
                w_uk.to_tile_tensor[.bfloat16](),
                w_uv.to_tile_tensor[.bfloat16](),
                scalar_args.to_tile_tensor[.int64](),
                kv_input_fn,
                context,
                num_partitions_in=num_partitions_proj,
            )


@extensibility.register("attention.mla.graph.decode.paged")
struct Struct_mla_decode_graph_bf16_paged(_MLADecodeGraphBF16PagedOp):
    """Registers the op for pre-SM100 NVIDIA GPUs and any other device."""

    comptime kernels: MLAGraphKernels = LegacyNVMLAGraph


@extensibility.register(
    "attention.mla.graph.decode.paged", type="gpu", api="hip"
)
struct Struct_mla_decode_graph_bf16_paged_amd(_MLADecodeGraphBF16PagedOp):
    """Registers the op for AMD GPUs."""

    comptime kernels: MLAGraphKernels = AMDMLAGraph


@extensibility.register(
    "attention.mla.graph.decode.paged", type="gpu", api="cuda", arch="sm_100a"
)
struct Struct_mla_decode_graph_bf16_paged_sm100(_MLADecodeGraphBF16PagedOp):
    """Registers the op for B100 and B200."""

    comptime kernels: MLAGraphKernels = SM100MLAGraph


@extensibility.register(
    "attention.mla.graph.decode.paged", type="gpu", api="cuda", arch="sm_103a"
)
struct Struct_mla_decode_graph_bf16_paged_sm103(_MLADecodeGraphBF16PagedOp):
    """Registers the op for B300."""

    comptime kernels: MLAGraphKernels = SM100MLAGraph


trait _MLADecodeGraphBF16PagedSparseOp:
    """Implements `attention.mla.graph.decode.paged.sparse`.

    Each registration picks its kernels.

    Under decode context parallelism (`dcp_size > 1`) every rank of the
    group receives the same lookup table and global `cache_lengths`, and the
    fused latent store writes a new token only on the rank that owns its
    position (`ContextParallelLayout[dcp_size, dcp_interleave]`), at that
    rank's local slot. `dcp_rank` is a host scalar after `selection_lag`;
    outside a group (`dcp_size == 1`) the caller passes 0 and the layout never
    reads it.
    """

    comptime kernels: MLAGraphKernels

    @inline(.always)
    @staticmethod
    def execute[
        mask_str: StaticString,
        indices_stride: Int,
        # Whether `sparse_indices` holds the indexer's paged physical rows
        # (`attention.mla.indexer.ragged.float8.paged.remap`) rather than logical
        # token positions, which the kernels look up in the page table.
        physical_indices: Bool,
        # The sparse index producer's row layout (`SparseRowLayout`), which
        # bounds each row by its closed-form valid length. The defaults are
        # plain top-k, which adds no bound.
        index_kpool: Int = 1,
        index_pool_topk: Int = 0,
        index_kpool_select_tail: Bool = True,
        # Set by the Python wrapper from whether each operand was supplied.
        # An absent operand arrives as a placeholder tensor that is never
        # read; the decode kernels are specialized on its absence.
        has_topk_lengths: Bool = True,
        has_attn_sink: Bool = True,
        dcp_size: Int = 1,
        dcp_interleave: Int = 1,
    ](
        output: Some[Output & Tensor],
        q: Some[Input & Tensor],
        kv: Some[Input & Fused & LoadAccess],
        input_row_offsets: Some[Input & Tensor],
        freqs_cis: Some[Input & Tensor],
        kv_norm_gamma: Some[Input & Tensor],
        w_uk: Some[Input & Tensor],
        w_uv: Some[Input & Tensor],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        layer_idx: UInt32,
        scale: Float32,
        epsilon: Float32,
        scalar_args: Some[Input & Tensor],
        sparse_indices: Some[Input & Tensor],
        topk_lengths: Some[Input & Tensor],
        attn_sink: Some[Input & Tensor],
        num_partitions_scalar: Some[Input & Tensor],
        selection_lag: Int32,
        dcp_rank: Int32,
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(q), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(freqs_cis), TileTensorable)
        comptime assert conforms_to(type_of(kv_norm_gamma), TileTensorable)
        comptime assert conforms_to(type_of(w_uk), TileTensorable)
        comptime assert conforms_to(type_of(w_uv), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert conforms_to(type_of(scalar_args), TileTensorable)
        comptime assert conforms_to(type_of(sparse_indices), TileTensorable)
        comptime assert conforms_to(type_of(topk_lengths), TileTensorable)
        comptime assert conforms_to(type_of(attn_sink), TileTensorable)
        comptime assert type_of(output).rank == 3
        comptime assert type_of(q).rank == 3
        comptime assert type_of(kv).rank == 2
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(freqs_cis).rank == 2
        comptime assert type_of(kv_norm_gamma).rank == 1
        comptime assert type_of(w_uk).rank == 3
        comptime assert type_of(w_uv).rank == 3
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(scalar_args).rank == 1
        comptime assert type_of(sparse_indices).rank == 2
        comptime assert type_of(topk_lengths).rank == 1
        comptime assert type_of(attn_sink).rank == 1
        comptime assert type_of(num_partitions_scalar).rank == 1
        comptime assert type_of(kv).dtype == DType.bfloat16
        comptime if dcp_size > 1:
            if dcp_rank < 0 or Int(dcp_rank) >= dcp_size:
                raise Error(
                    t"dcp_rank {dcp_rank} is outside a group of {dcp_size}"
                )
        var kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )

        comptime assert (
            context.target.is_gpu()
        ), "attention.mla.graph.decode.paged.sparse is only supported on GPU"

        comptime assert conforms_to(type_of(kv), UnsafeFusedView)
        var kv_view = unsafe_fused_view(kv)
        comptime assert conforms_to(type_of(kv_view), Loadable)

        @inline(.always)
        def kv_input_fn[
            width: Int
        ](coords: IndexList[2]) {var kv_view} -> SIMD[.bfloat16, width]:
            return rebind[SIMD[.bfloat16, width]](
                kv_view.load[width=width, element_alignment=width](
                    Coord(coords)
                )
            )

        var num_tokens = Int(q.shape()[0].value())

        comptime if has_topk_lengths:
            var required_lengths = Int(input_row_offsets.shape()[0].value()) - 1
            if Int(topk_lengths.shape()[0].value()) < required_lengths:
                raise Error(
                    "topk_lengths does not cover its batch/token domain"
                )

        var topk_lengths_ptr = maybe_null_pointer[has_topk_lengths](
            topk_lengths.to_tile_tensor[.int32]().ptr
        )
        var attn_sink_ptr = maybe_null_pointer[has_attn_sink](
            attn_sink.to_tile_tensor[.float32]().ptr
        )

        comptime mla_page_size = type_of(kv_blocks)._shape_types[3].static_value

        with Trace[TraceLevel.OP, target=context.target.device_kind_label()](
            "attention.mla.graph.decode.paged.sparse",
            task_id=get_safe_task_id(context),
        ):
            var scratch_sparse_indices = Optional[DeviceBuffer[DType.int32]]()
            var gather_indices = _sparse_decode_indices[
                mla_page_size,
                indices_stride,
                physical_indices,
            ](
                scratch_sparse_indices,
                sparse_indices.to_tile_tensor[.int32](),
                input_row_offsets.to_tile_tensor[.uint32](),
                kv_lookup_table.to_tile_tensor[.uint32](),
                kv_blocks.to_tile_tensor(),
                num_tokens,
                context,
            )
            mla_decode_branch_bf16[
                mask_str=mask_str,
                sparse_mla=True,
                physical_indices=_sparse_decode_physical[physical_indices](),
                sparse_row=SparseRowLayout(
                    kpool=index_kpool,
                    pool_topk=index_pool_topk,
                    select_tail=index_kpool_select_tail,
                ),
                kernels=Self.kernels,
                dcp_size=dcp_size,
                dcp_interleave=dcp_interleave,
            ](
                output.to_tile_tensor[.bfloat16](),
                q.to_tile_tensor[.bfloat16](),
                input_row_offsets.to_tile_tensor[.uint32](),
                freqs_cis.to_tile_tensor(),
                kv_norm_gamma.to_tile_tensor(),
                kv_collection,
                layer_idx,
                scale,
                epsilon,
                w_uk.to_tile_tensor[.bfloat16](),
                w_uv.to_tile_tensor[.bfloat16](),
                scalar_args.to_tile_tensor[.int64](),
                kv_input_fn,
                context,
                gather_indices,
                indices_stride,
                topk_lengths_ptr,
                attn_sink_ptr,
                num_partitions_in=None,
                selection_lag=Int(selection_lag),
                dcp_rank=Int(dcp_rank),
            )
            _ = scratch_sparse_indices^


@extensibility.register("attention.mla.graph.decode.paged.sparse")
struct Struct_mla_decode_graph_bf16_paged_sparse(
    _MLADecodeGraphBF16PagedSparseOp
):
    """Registers the op for pre-SM100 NVIDIA GPUs and any other device."""

    comptime kernels: MLAGraphKernels = LegacyNVMLAGraph


@extensibility.register(
    "attention.mla.graph.decode.paged.sparse", type="gpu", api="hip"
)
struct Struct_mla_decode_graph_bf16_paged_sparse_amd(
    _MLADecodeGraphBF16PagedSparseOp
):
    """Registers the op for AMD GPUs."""

    comptime kernels: MLAGraphKernels = AMDMLAGraph


@extensibility.register(
    "attention.mla.graph.decode.paged.sparse",
    type="gpu",
    api="cuda",
    arch="sm_100a",
)
struct Struct_mla_decode_graph_bf16_paged_sparse_sm100(
    _MLADecodeGraphBF16PagedSparseOp
):
    """Registers the op for B100 and B200."""

    comptime kernels: MLAGraphKernels = SM100MLAGraph


@extensibility.register(
    "attention.mla.graph.decode.paged.sparse",
    type="gpu",
    api="cuda",
    arch="sm_103a",
)
struct Struct_mla_decode_graph_bf16_paged_sparse_sm103(
    _MLADecodeGraphBF16PagedSparseOp
):
    """Registers the op for B300."""

    comptime kernels: MLAGraphKernels = SM100MLAGraph


trait _MLADecodeGraphBF16PagedSparseNVFP4Op:
    """Implements `attention.mla.graph.decode.paged.sparse.nvfp4`.

    Each registration picks its kernels.

    Under decode context parallelism (`dcp_size > 1`) every rank of the
    group receives the same lookup table and global `cache_lengths`, and the
    fused latent store writes a new token only on the rank that owns its
    position (`ContextParallelLayout[dcp_size, dcp_interleave]`), at that
    rank's local slot. `dcp_rank` is a host scalar after `selection_lag`;
    outside a group (`dcp_size == 1`) the caller passes 0 and the layout never
    reads it.
    """

    comptime kernels: MLAGraphKernels

    @inline(.always)
    @staticmethod
    def execute[
        mask_str: StaticString,
        indices_stride: Int,
        physical_indices: Bool,
        index_kpool: Int = 1,
        index_pool_topk: Int = 0,
        index_kpool_select_tail: Bool = True,
        has_topk_lengths: Bool = True,
        has_attn_sink: Bool = True,
        dcp_size: Int = 1,
        dcp_interleave: Int = 1,
    ](
        output: Some[Output & Tensor],
        q: Some[Input & Tensor],
        kv: Some[Input & Fused & LoadAccess],
        input_row_offsets: Some[Input & Tensor],
        freqs_cis: Some[Input & Tensor],
        kv_norm_gamma: Some[Input & Tensor],
        w_uk: Some[Input & Tensor],
        w_uv: Some[Input & Tensor],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        kv_scales: Some[MutableInput & Tensor],
        scales_page_stride: Some[Input & Tensor],
        scales_lookup_table: Some[Input & Tensor],
        layer_idx: UInt32,
        scale: Float32,
        epsilon: Float32,
        scalar_args: Some[Input & Tensor],
        sparse_indices: Some[Input & Tensor],
        topk_lengths: Some[Input & Tensor],
        attn_sink: Some[Input & Tensor],
        num_partitions_scalar: Some[Input & Tensor],
        selection_lag: Int32,
        dcp_rank: Int32,
        context: DeviceContext,
    ) raises:
        comptime if dcp_size > 1:
            if dcp_rank < 0 or Int(dcp_rank) >= dcp_size:
                raise Error(
                    t"dcp_rank {dcp_rank} is outside a group of {dcp_size}"
                )
        comptime assert context.target.is_gpu(), (
            "attention.mla.graph.decode.paged.sparse.nvfp4 is only supported"
            " on GPU"
        )
        comptime assert physical_indices, (
            "an NVFP4 MLA cache needs the indexer's physical and scale rows,"
            " not token positions"
        )
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(q), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(freqs_cis), TileTensorable)
        comptime assert conforms_to(type_of(kv_norm_gamma), TileTensorable)
        comptime assert conforms_to(type_of(w_uk), TileTensorable)
        comptime assert conforms_to(type_of(w_uv), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert conforms_to(type_of(kv_scales), TileTensorable)
        comptime assert conforms_to(type_of(scales_page_stride), TileTensorable)
        comptime assert conforms_to(
            type_of(scales_lookup_table), TileTensorable
        )
        comptime assert conforms_to(type_of(scalar_args), TileTensorable)
        comptime assert conforms_to(type_of(sparse_indices), TileTensorable)
        comptime assert conforms_to(type_of(topk_lengths), TileTensorable)
        comptime assert conforms_to(type_of(attn_sink), TileTensorable)
        comptime assert type_of(output).rank == 3
        comptime assert type_of(q).rank == 3
        comptime assert type_of(kv).rank == 2
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(freqs_cis).rank == 2
        comptime assert type_of(kv_norm_gamma).rank == 1
        comptime assert type_of(w_uk).rank == 3
        comptime assert type_of(w_uv).rank == 3
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(kv_scales).rank == 6
        comptime assert type_of(scales_page_stride).rank == 1
        comptime assert type_of(scales_lookup_table).rank == 2
        comptime assert type_of(scalar_args).rank == 1
        comptime assert type_of(sparse_indices).rank == 2
        comptime assert type_of(topk_lengths).rank == 1
        comptime assert type_of(attn_sink).rank == 1
        comptime assert type_of(num_partitions_scalar).rank == 1
        comptime assert type_of(kv).dtype == DType.bfloat16
        var kv_collection = generic_get_paged_cache_with_scales(
            kv_blocks.to_tile_tensor[.uint8](),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
            kv_scales.to_tile_tensor[.float8_e4m3fn](),
            scales_page_stride.to_tile_tensor[.int64](),
            scales_lookup_table.to_tile_tensor[.uint32](),
        )

        comptime assert conforms_to(type_of(kv), UnsafeFusedView)
        var kv_view = unsafe_fused_view(kv)
        comptime assert conforms_to(type_of(kv_view), Loadable)

        @inline(.always)
        def kv_input_fn[
            width: Int
        ](coords: IndexList[2]) {var kv_view} -> SIMD[.bfloat16, width]:
            return rebind[SIMD[.bfloat16, width]](
                kv_view.load[width=width, element_alignment=width](
                    Coord(coords)
                )
            )

        var num_tokens = Int(q.shape()[0].value())

        var topk_lengths_ptr = maybe_null_pointer[has_topk_lengths](
            topk_lengths.to_tile_tensor[.int32]().ptr
        )
        var attn_sink_ptr = maybe_null_pointer[has_attn_sink](
            attn_sink.to_tile_tensor[.float32]().ptr
        )

        with Trace[TraceLevel.OP, target=context.target.device_kind_label()](
            "attention.mla.graph.decode.paged.sparse.nvfp4",
            task_id=get_safe_task_id(context),
        ):
            var gather_indices = _sparse_indices_ptr[indices_stride](
                sparse_indices.to_tile_tensor[.int32](), num_tokens
            )
            mla_decode_branch_bf16[
                mask_str=mask_str,
                sparse_mla=True,
                physical_indices=physical_indices,
                sparse_row=SparseRowLayout(
                    kpool=index_kpool,
                    pool_topk=index_pool_topk,
                    select_tail=index_kpool_select_tail,
                ),
                kernels=Self.kernels,
                dcp_size=dcp_size,
                dcp_interleave=dcp_interleave,
            ](
                output.to_tile_tensor[.bfloat16](),
                q.to_tile_tensor[.bfloat16](),
                input_row_offsets.to_tile_tensor[.uint32](),
                freqs_cis.to_tile_tensor(),
                kv_norm_gamma.to_tile_tensor(),
                kv_collection,
                layer_idx,
                scale,
                epsilon,
                w_uk.to_tile_tensor[.bfloat16](),
                w_uv.to_tile_tensor[.bfloat16](),
                scalar_args.to_tile_tensor[.int64](),
                kv_input_fn,
                context,
                gather_indices,
                indices_stride,
                topk_lengths_ptr,
                attn_sink_ptr,
                num_partitions_in=None,
                selection_lag=Int(selection_lag),
                dcp_rank=Int(dcp_rank),
            )


# ===-----------------------------------------------------------------------===#
# Sparse MLA decode for decode context parallelism (`.cp`)
#
# Each rank holds a slice of every request's latent cache and attends over the
# selected rows it owns, compacted to the front of each token's index row with
# a per-token valid length, and stores a new token only if it owns the token's
# position. The ops return the attention LSE next to O so the ranks' partial
# results can be merged. They take no attention sink, k-pool row
# layout, shared-index fold or selection lag. They stay registrations of their
# own because of the LSE: an op's output list is fixed (the graph compiler binds
# no empty variadic), so the base ops could only take it as a placeholder
# output every non-CP caller would then allocate. The CP latent store alone is
# a parameter of the base ops.
# ===-----------------------------------------------------------------------===#


@inline(.always)
def _cp_sparse_lengths_ptr(
    sparse_lengths: InputTensor[dtype=.int32, rank=1, ...],
    num_tokens: Int,
) raises -> NonNullPointer[DType.int32]:
    """Returns the per-token valid lengths of a `.cp` op after checking them.

    The ops run the sparse decode with `skip_masked_tiles`, under which
    `topk_lengths` holds one count per query token.
    """
    if sparse_lengths.dim_size(0) != num_tokens:
        raise Error("sparse_lengths must be [query tokens]")
    return NonNullPointer[DType.int32](sparse_lengths.to_tile_tensor[.int64]())


@inline(.always)
def _cp_check_rank[dcp_size: Int](dcp_rank: Int32) raises:
    """Raises if `dcp_rank` is not a rank of a group of `dcp_size`."""
    if dcp_rank < 0 or Int(dcp_rank) >= dcp_size:
        raise Error(t"dcp_rank {dcp_rank} is outside a group of {dcp_size}")


@inline(.always)
def _cp_lse_ptr(
    lse: OutputTensor[dtype=.float32, rank=2, ...],
    num_tokens: Int,
    num_heads: Int,
) raises -> NonNullPointer[DType.float32]:
    """Returns the LSE output of a `.cp` op after checking its shape."""
    if lse.dim_size(0) != num_tokens or lse.dim_size(1) != num_heads:
        raise Error("lse must be [query tokens, num_heads]")
    return NonNullPointer[DType.float32](lse.to_tile_tensor[.int64]())


@extensibility.register("attention.mla.graph.decode.paged.fp8.sparse.cp")
struct Struct_mla_decode_graph_paged_fp8_sparse_cp:
    """Registers the `attention.mla.graph.decode.paged.fp8.sparse.cp` graph op.

    `attention.mla.graph.decode.paged.fp8.sparse` for decode context parallelism.
    Each query token `t` attends over the first `sparse_lengths[t]` entries of
    its `sparse_indices` row (0 is allowed), and the op also writes `lse`, the
    attention log-sum-exp per (token, head) in base 2:
    `log2(sum_k exp2(q.k * scale * log2(e)))`, before `W_UV`. A token with no
    keys gets `lse = -inf` and `output = 0`. Ranks merge as
    `w_r = exp2(lse_r - max_r lse_r)`, `O = sum_r w_r O_r / sum_r w_r`.
    `split_len_bound` (>0) bounds every length and sizes split-K in place of
    `indices_stride`; it only affects performance. As in the base op with
    `dcp_size > 1`, the fused latent store writes a new token only on the rank
    that owns its position (`ContextParallelLayout[dcp_size,
    dcp_interleave]`), at that rank's local slot; `dcp_rank` is a host scalar
    after `split_len_bound`.
    SM100 only.
    """

    @inline(.always)
    @staticmethod
    def execute[
        dtype: DType,
        freq_dtype: DType,
        gamma_dtype: DType,
        fp8_dtype: DType,
        fp8_scale_dtype: DType,
        cache_dtype: DType,
        //,
        m_scale_granularity: Int,
        n_scale_granularity: Int,
        k_scale_granularity: Int,
        mask_str: StaticString,
        target: StaticString,
        dcp_size: Int,
        dcp_interleave: Int,
        indices_stride: Int,
        # Whether `sparse_indices` holds the indexer's paged physical rows
        # rather than logical token positions.
        physical_indices: Bool,
    ](
        output: OutputTensor[dtype=dtype, rank=3, ...],
        lse: OutputTensor[dtype=.float32, rank=2, ...],
        q: InputTensor[dtype=dtype, rank=3, ...],
        kv: FusedInputTensor[dtype=.bfloat16, rank=2, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        freqs_cis: InputTensor[dtype=freq_dtype, rank=2, ...],
        kv_norm_gamma: InputTensor[dtype=gamma_dtype, rank=1, ...],
        w_uk: InputTensor[dtype=fp8_dtype, rank=3, ...],
        w_uv: InputTensor[dtype=fp8_dtype, rank=3, ...],
        kv_blocks: MutableInputTensor[dtype=cache_dtype, rank=6, ...],
        page_stride: InputTensor[dtype=.int64, rank=1, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        layer_idx: UInt32,
        scale: Float32,
        epsilon: Float32,
        w_uk_scale: InputTensor[dtype=fp8_scale_dtype, rank=3, ...],
        w_uv_scale: InputTensor[dtype=fp8_scale_dtype, rank=3, ...],
        scalar_args: InputTensor[dtype=.int64, rank=1, ...],
        sparse_indices: InputTensor[dtype=.int32, rank=2, ...],
        sparse_lengths: InputTensor[dtype=.int32, rank=1, ...],
        split_len_bound: Int32,
        dcp_rank: Int32,
        context: DeviceContext,
    ) raises:
        _cp_check_rank[dcp_size](dcp_rank)
        comptime assert is_gpu[target](), (
            "attention.mla.graph.decode.paged.fp8.sparse.cp is only supported"
            " on GPU"
        )
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            page_stride,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )

        @inline(.always)
        def kv_input_fn[
            width: Int
        ](coords: IndexList[2]) {var kv} -> SIMD[.bfloat16, width]:
            return kv._lambda_load[width=width, element_alignment=width](coords)

        var num_tokens = q.dim_size(0)
        var lengths_ptr = _cp_sparse_lengths_ptr(sparse_lengths, num_tokens)
        var lse_ptr = _cp_lse_ptr(
            lse, num_tokens, Int(q.static_spec.shape_tuple[1])
        )

        with Trace[TraceLevel.OP, target=target](
            "attention.mla.graph.decode.paged.fp8.sparse.cp",
            task_id=get_safe_task_id(context),
        ):
            var gather_indices = _sparse_indices_ptr[indices_stride](
                sparse_indices.to_tile_tensor[.int64](), num_tokens
            )
            mla_decode_branch_fp8[
                kernels=SM100MLAGraph,
                m_scale_granularity=m_scale_granularity,
                n_scale_granularity=n_scale_granularity,
                k_scale_granularity=k_scale_granularity,
                mask_str=mask_str,
                sparse_mla=True,
                physical_indices=physical_indices,
                skip_masked_tiles=True,
                dcp_size=dcp_size,
                dcp_interleave=dcp_interleave,
            ](
                output.to_tile_tensor[.int64](),
                q.to_tile_tensor[.int64](),
                input_row_offsets.to_tile_tensor[.int64](),
                freqs_cis.to_tile_tensor[.int64](),
                kv_norm_gamma.to_tile_tensor[.int64](),
                kv_collection,
                layer_idx,
                scale,
                epsilon,
                w_uk.to_tile_tensor[.int64](),
                w_uk_scale.to_tile_tensor[.int64](),
                w_uv.to_tile_tensor[.int64](),
                w_uv_scale.to_tile_tensor[.int64](),
                scalar_args.to_tile_tensor[.int64](),
                kv_input_fn,
                context,
                gather_indices,
                indices_stride,
                lengths_ptr,
                lse_out=lse_ptr,
                split_len_bound=Int(split_len_bound),
                dcp_rank=Int(dcp_rank),
            )


@extensibility.register("attention.mla.graph.decode.paged.sparse.cp")
struct Struct_mla_decode_graph_bf16_paged_sparse_cp:
    """Registers the `attention.mla.graph.decode.paged.sparse.cp` graph op.

    `attention.mla.graph.decode.paged.sparse` (BF16 weights, BF16 or FP8 latent
    cache) for decode context parallelism; see
    `attention.mla.graph.decode.paged.fp8.sparse.cp` for the per-token lengths, the
    `lse` output, `split_len_bound` and the CP store. SM100 only.
    """

    @inline(.always)
    @staticmethod
    def execute[
        kv_dtype: DType,
        freq_dtype: DType,
        gamma_dtype: DType,
        //,
        mask_str: StaticString,
        target: StaticString,
        dcp_size: Int,
        dcp_interleave: Int,
        indices_stride: Int,
        physical_indices: Bool,
        # Whether `q` is already absorbed, `[q_nope @ w_uk, q_rope]`, so
        # `w_uk` is not read. Decode context parallelism with a gathered
        # query absorbs each rank's heads before the all-gather.
        q_absorbed: Bool = False,
        # Whether `output` is the latent attention output `[T, H,
        # kv_latent_dim]`, so `w_uv` is not read: the ranks merge latents and
        # each applies `w_uv` to its own heads after the merge.
        latent_output: Bool = False,
    ](
        output: OutputTensor[dtype=.bfloat16, rank=3, ...],
        lse: OutputTensor[dtype=.float32, rank=2, ...],
        q: InputTensor[dtype=.bfloat16, rank=3, ...],
        kv: FusedInputTensor[dtype=.bfloat16, rank=2, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        freqs_cis: InputTensor[dtype=freq_dtype, rank=2, ...],
        kv_norm_gamma: InputTensor[dtype=gamma_dtype, rank=1, ...],
        w_uk: InputTensor[dtype=.bfloat16, rank=3, ...],
        w_uv: InputTensor[dtype=.bfloat16, rank=3, ...],
        kv_blocks: MutableInputTensor[dtype=kv_dtype, rank=6, ...],
        page_stride: InputTensor[dtype=.int64, rank=1, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        layer_idx: UInt32,
        scale: Float32,
        epsilon: Float32,
        scalar_args: InputTensor[dtype=.int64, rank=1, ...],
        sparse_indices: InputTensor[dtype=.int32, rank=2, ...],
        sparse_lengths: InputTensor[dtype=.int32, rank=1, ...],
        split_len_bound: Int32,
        dcp_rank: Int32,
        context: DeviceContext,
    ) raises:
        _cp_check_rank[dcp_size](dcp_rank)
        comptime assert is_gpu[
            target
        ](), "attention.mla.graph.decode.paged.sparse.cp is only supported on GPU"
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            page_stride,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )

        @inline(.always)
        def kv_input_fn[
            width: Int
        ](coords: IndexList[2]) {var kv} -> SIMD[.bfloat16, width]:
            return kv._lambda_load[width=width, element_alignment=width](coords)

        var num_tokens = q.dim_size(0)
        var lengths_ptr = _cp_sparse_lengths_ptr(sparse_lengths, num_tokens)
        var lse_ptr = _cp_lse_ptr(
            lse, num_tokens, Int(q.static_spec.shape_tuple[1])
        )

        with Trace[TraceLevel.OP, target=target](
            "attention.mla.graph.decode.paged.sparse.cp",
            task_id=get_safe_task_id(context),
        ):
            var gather_indices = _sparse_indices_ptr[indices_stride](
                sparse_indices.to_tile_tensor[.int64](), num_tokens
            )
            mla_decode_branch_bf16[
                kernels=SM100MLAGraph,
                mask_str=mask_str,
                sparse_mla=True,
                physical_indices=physical_indices,
                skip_masked_tiles=True,
                dcp_size=dcp_size,
                dcp_interleave=dcp_interleave,
                q_absorbed=q_absorbed,
                latent_output=latent_output,
            ](
                output.to_tile_tensor[.int64](),
                q.to_tile_tensor[.int64](),
                input_row_offsets.to_tile_tensor[.int64](),
                freqs_cis.to_tile_tensor[.int64](),
                kv_norm_gamma.to_tile_tensor[.int64](),
                kv_collection,
                layer_idx,
                scale,
                epsilon,
                w_uk.to_tile_tensor[.int64](),
                w_uv.to_tile_tensor[.int64](),
                scalar_args.to_tile_tensor[.int64](),
                kv_input_fn,
                context,
                gather_indices,
                indices_stride,
                lengths_ptr,
                lse_out=lse_ptr,
                split_len_bound=Int(split_len_bound),
                dcp_rank=Int(dcp_rank),
            )


@extensibility.register("attention.mla.graph.decode.paged.sparse.nvfp4.cp")
struct Struct_mla_decode_graph_bf16_paged_sparse_nvfp4_cp:
    """Registers the `attention.mla.graph.decode.paged.sparse.nvfp4.cp` graph op.

    `attention.mla.graph.decode.paged.sparse.nvfp4` for decode context parallelism;
    see `attention.mla.graph.decode.paged.fp8.sparse.cp` for the per-token lengths,
    the `lse` output, `split_len_bound` and the CP store. A row of
    `sparse_indices` is `indices_stride / 2` value rows then as many
    scale-pool rows, and a token's length counts entries of each half: the
    compacted value rows sit at `[0, len)` and their scale rows at
    `[indices_stride / 2, indices_stride / 2 + len)`. SM100 only.
    """

    @inline(.always)
    @staticmethod
    def execute[
        freq_dtype: DType,
        gamma_dtype: DType,
        //,
        mask_str: StaticString,
        target: StaticString,
        dcp_size: Int,
        dcp_interleave: Int,
        indices_stride: Int,
        physical_indices: Bool,
        # Whether `q` is already absorbed, `[q_nope @ w_uk, q_rope]`, so
        # `w_uk` is not read. Decode context parallelism with a gathered
        # query absorbs each rank's heads before the all-gather.
        q_absorbed: Bool = False,
        # Whether `output` is the latent attention output `[T, H,
        # kv_latent_dim]`, so `w_uv` is not read: the ranks merge latents and
        # each applies `w_uv` to its own heads after the merge.
        latent_output: Bool = False,
    ](
        output: OutputTensor[dtype=.bfloat16, rank=3, ...],
        lse: OutputTensor[dtype=.float32, rank=2, ...],
        q: InputTensor[dtype=.bfloat16, rank=3, ...],
        kv: FusedInputTensor[dtype=.bfloat16, rank=2, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        freqs_cis: InputTensor[dtype=freq_dtype, rank=2, ...],
        kv_norm_gamma: InputTensor[dtype=gamma_dtype, rank=1, ...],
        w_uk: InputTensor[dtype=.bfloat16, rank=3, ...],
        w_uv: InputTensor[dtype=.bfloat16, rank=3, ...],
        kv_blocks: MutableInputTensor[dtype=.uint8, rank=6, ...],
        page_stride: InputTensor[dtype=.int64, rank=1, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        kv_scales: MutableInputTensor[dtype=.float8_e4m3fn, rank=6, ...],
        scales_page_stride: InputTensor[dtype=.int64, rank=1, ...],
        scales_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        layer_idx: UInt32,
        scale: Float32,
        epsilon: Float32,
        scalar_args: InputTensor[dtype=.int64, rank=1, ...],
        sparse_indices: InputTensor[dtype=.int32, rank=2, ...],
        sparse_lengths: InputTensor[dtype=.int32, rank=1, ...],
        split_len_bound: Int32,
        dcp_rank: Int32,
        context: DeviceContext,
    ) raises:
        _cp_check_rank[dcp_size](dcp_rank)
        comptime assert is_gpu[target](), (
            "attention.mla.graph.decode.paged.sparse.nvfp4.cp is only supported"
            " on GPU"
        )
        comptime assert physical_indices, (
            "an NVFP4 MLA cache needs the indexer's physical and scale rows,"
            " not token positions"
        )
        var kv_collection = generic_get_paged_cache_with_scales(
            kv_blocks,
            page_stride,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
            kv_scales,
            scales_page_stride,
            scales_lookup_table,
        )

        @inline(.always)
        def kv_input_fn[
            width: Int
        ](coords: IndexList[2]) {var kv} -> SIMD[.bfloat16, width]:
            return kv._lambda_load[width=width, element_alignment=width](coords)

        var num_tokens = q.dim_size(0)
        var lengths_ptr = _cp_sparse_lengths_ptr(sparse_lengths, num_tokens)
        var lse_ptr = _cp_lse_ptr(
            lse, num_tokens, Int(q.static_spec.shape_tuple[1])
        )

        with Trace[TraceLevel.OP, target=target](
            "attention.mla.graph.decode.paged.sparse.nvfp4.cp",
            task_id=get_safe_task_id(context),
        ):
            var gather_indices = _sparse_indices_ptr[indices_stride](
                sparse_indices.to_tile_tensor[.int64](), num_tokens
            )
            mla_decode_branch_bf16[
                kernels=SM100MLAGraph,
                mask_str=mask_str,
                sparse_mla=True,
                physical_indices=physical_indices,
                skip_masked_tiles=True,
                dcp_size=dcp_size,
                dcp_interleave=dcp_interleave,
                q_absorbed=q_absorbed,
                latent_output=latent_output,
            ](
                output.to_tile_tensor[.int64](),
                q.to_tile_tensor[.int64](),
                input_row_offsets.to_tile_tensor[.int64](),
                freqs_cis.to_tile_tensor[.int64](),
                kv_norm_gamma.to_tile_tensor[.int64](),
                kv_collection,
                layer_idx,
                scale,
                epsilon,
                w_uk.to_tile_tensor[.int64](),
                w_uv.to_tile_tensor[.int64](),
                scalar_args.to_tile_tensor[.int64](),
                kv_input_fn,
                context,
                gather_indices,
                indices_stride,
                lengths_ptr,
                lse_out=lse_ptr,
                split_len_bound=Int(split_len_bound),
                dcp_rank=Int(dcp_rank),
            )


@extensibility.register("attention.mla.graph.decode.paged.sparse.nvfp4")
struct Struct_mla_decode_graph_bf16_paged_sparse_nvfp4(
    _MLADecodeGraphBF16PagedSparseNVFP4Op
):
    """Registers the op for pre-SM100 NVIDIA GPUs and any other device."""

    comptime kernels: MLAGraphKernels = LegacyNVMLAGraph


@extensibility.register(
    "attention.mla.graph.decode.paged.sparse.nvfp4", type="gpu", api="hip"
)
struct Struct_mla_decode_graph_bf16_paged_sparse_nvfp4_amd(
    _MLADecodeGraphBF16PagedSparseNVFP4Op
):
    """Registers the op for AMD GPUs."""

    comptime kernels: MLAGraphKernels = AMDMLAGraph


@extensibility.register(
    "attention.mla.graph.decode.paged.sparse.nvfp4",
    type="gpu",
    api="cuda",
    arch="sm_100a",
)
struct Struct_mla_decode_graph_bf16_paged_sparse_nvfp4_sm100(
    _MLADecodeGraphBF16PagedSparseNVFP4Op
):
    """Registers the op for B100 and B200."""

    comptime kernels: MLAGraphKernels = SM100MLAGraph


@extensibility.register(
    "attention.mla.graph.decode.paged.sparse.nvfp4",
    type="gpu",
    api="cuda",
    arch="sm_103a",
)
struct Struct_mla_decode_graph_bf16_paged_sparse_nvfp4_sm103(
    _MLADecodeGraphBF16PagedSparseNVFP4Op
):
    """Registers the op for B300."""

    comptime kernels: MLAGraphKernels = SM100MLAGraph


trait _MLAPrefillGraphDecodePagedFP8Op:
    """Implements `attention.mla.graph.prefill.decode.paged.fp8`.

    Each registration picks its kernels.
    """

    comptime kernels: MLAGraphKernels

    @inline(.always)
    @staticmethod
    def execute[
        m_scale_granularity: Int,
        n_scale_granularity: Int,
        k_scale_granularity: Int,
        mask_str: StaticString,
    ](
        output: Some[Output & Tensor],
        q: Some[Input & Tensor],
        kv: Some[Input & Fused & LoadAccess],
        input_row_offsets: Some[Input & Tensor],
        freqs_cis: Some[Input & Tensor],
        kv_norm_gamma: Some[Input & Tensor],
        buffer_row_offsets_1d: Some[Input & Tensor],
        cache_offsets_1d: Some[Input & Tensor],
        buffer_length: Int32,
        w_k: Some[Input & Tensor],
        w_uk: Some[Input & Tensor],
        w_uv: Some[Input & Tensor],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        layer_idx: UInt32,
        scale: Float32,
        epsilon: Float32,
        w_k_scale: Some[Input & Tensor],
        w_uk_scale: Some[Input & Tensor],
        w_uv_scale: Some[Input & Tensor],
        scalar_args: Some[Input & Tensor],
        num_partitions_scalar: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(q), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(freqs_cis), TileTensorable)
        comptime assert conforms_to(type_of(kv_norm_gamma), TileTensorable)
        comptime assert conforms_to(
            type_of(buffer_row_offsets_1d), TileTensorable
        )
        comptime assert conforms_to(type_of(cache_offsets_1d), TileTensorable)
        comptime assert conforms_to(type_of(w_k), TileTensorable)
        comptime assert conforms_to(type_of(w_uk), TileTensorable)
        comptime assert conforms_to(type_of(w_uv), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert conforms_to(type_of(w_k_scale), TileTensorable)
        comptime assert conforms_to(type_of(w_uk_scale), TileTensorable)
        comptime assert conforms_to(type_of(w_uv_scale), TileTensorable)
        comptime assert conforms_to(type_of(scalar_args), TileTensorable)
        comptime assert conforms_to(
            type_of(num_partitions_scalar), TileTensorable
        )
        comptime dtype = type_of(output).dtype
        comptime fp8_dtype = type_of(w_k).dtype
        comptime fp8_scale_dtype = type_of(w_k_scale).dtype
        comptime assert type_of(output).rank == 3
        comptime assert type_of(q).rank == 3
        comptime assert type_of(kv).rank == 2
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(freqs_cis).rank == 2
        comptime assert type_of(kv_norm_gamma).rank == 1
        comptime assert type_of(buffer_row_offsets_1d).rank == 1
        comptime assert type_of(cache_offsets_1d).rank == 1
        comptime assert type_of(w_k).rank == 2
        comptime assert type_of(w_uk).rank == 3
        comptime assert type_of(w_uv).rank == 3
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(w_k_scale).rank == 2
        comptime assert type_of(w_uk_scale).rank == 3
        comptime assert type_of(w_uv_scale).rank == 3
        comptime assert type_of(scalar_args).rank == 1
        comptime assert type_of(num_partitions_scalar).rank == 1
        comptime assert type_of(kv).dtype == DType.bfloat16
        var kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )

        comptime assert context.target.is_gpu(), (
            "attention.mla.graph.prefill.decode.paged.fp8 is only supported"
            " on GPU"
        )

        comptime assert conforms_to(type_of(kv), UnsafeFusedView)
        var kv_view = unsafe_fused_view(kv)
        comptime assert conforms_to(type_of(kv_view), Loadable)

        @inline(.always)
        def kv_input_fn[
            width: Int
        ](coords: IndexList[2]) {var kv_view} -> SIMD[.bfloat16, width]:
            return rebind[SIMD[.bfloat16, width]](
                kv_view.load[width=width, element_alignment=width](
                    Coord(coords)
                )
            )

        var num_partitions_proj = Int(
            num_partitions_scalar.to_tile_tensor[.int64]().load[width=1](
                Coord(0)
            )
        )

        with Trace[TraceLevel.OP, target=context.target.device_kind_label()](
            "attention.mla.graph.prefill.decode.paged.fp8",
            task_id=get_safe_task_id(context),
        ):
            mla_prefill_decode_graph_fp8[
                m_scale_granularity=m_scale_granularity,
                n_scale_granularity=n_scale_granularity,
                k_scale_granularity=k_scale_granularity,
                mask_str=mask_str,
                kernels=Self.kernels,
            ](
                output.to_tile_tensor(),
                q.to_tile_tensor[dtype](),
                input_row_offsets.to_tile_tensor[.uint32](),
                freqs_cis.to_tile_tensor(),
                kv_norm_gamma.to_tile_tensor(),
                kv_collection,
                layer_idx,
                scale,
                epsilon,
                buffer_row_offsets_1d.to_tile_tensor[.uint32](),
                cache_offsets_1d.to_tile_tensor[.uint32](),
                Int(buffer_length),
                Int(kv_collection.max_seq_length),
                w_k.to_tile_tensor(),
                w_k_scale.to_tile_tensor(),
                w_uk.to_tile_tensor[fp8_dtype](),
                w_uk_scale.to_tile_tensor[fp8_scale_dtype](),
                w_uv.to_tile_tensor[fp8_dtype](),
                w_uv_scale.to_tile_tensor[fp8_scale_dtype](),
                scalar_args.to_tile_tensor[.int64](),
                kv_input_fn,
                context,
                num_partitions_in=num_partitions_proj,
            )


@extensibility.register("attention.mla.graph.prefill.decode.paged.fp8")
struct Struct_mla_prefill_graph_decode_paged_fp8(
    _MLAPrefillGraphDecodePagedFP8Op
):
    """Registers the op for pre-SM100 NVIDIA GPUs and any other device."""

    comptime kernels: MLAGraphKernels = LegacyNVMLAGraph


@extensibility.register(
    "attention.mla.graph.prefill.decode.paged.fp8", type="gpu", api="hip"
)
struct Struct_mla_prefill_graph_decode_paged_fp8_amd(
    _MLAPrefillGraphDecodePagedFP8Op
):
    """Registers the op for AMD GPUs."""

    comptime kernels: MLAGraphKernels = AMDMLAGraph


@extensibility.register(
    "attention.mla.graph.prefill.decode.paged.fp8",
    type="gpu",
    api="cuda",
    arch="sm_100a",
)
struct Struct_mla_prefill_graph_decode_paged_fp8_sm100(
    _MLAPrefillGraphDecodePagedFP8Op
):
    """Registers the op for B100 and B200."""

    comptime kernels: MLAGraphKernels = SM100MLAGraph


@extensibility.register(
    "attention.mla.graph.prefill.decode.paged.fp8",
    type="gpu",
    api="cuda",
    arch="sm_103a",
)
struct Struct_mla_prefill_graph_decode_paged_fp8_sm103(
    _MLAPrefillGraphDecodePagedFP8Op
):
    """Registers the op for B300."""

    comptime kernels: MLAGraphKernels = SM100MLAGraph


trait _MLAPrefillGraphDecodePagedFP8SparseOp:
    """Implements `attention.mla.graph.prefill.decode.paged.fp8.sparse`.

    Each registration picks its kernels.
    """

    comptime kernels: MLAGraphKernels

    @inline(.always)
    @staticmethod
    def execute[
        m_scale_granularity: Int,
        n_scale_granularity: Int,
        k_scale_granularity: Int,
        mask_str: StaticString,
        indices_stride: Int,
        # Whether `sparse_indices` holds the indexer's paged physical rows
        # (`attention.mla.indexer.ragged.float8.paged.remap`) rather than logical
        # token positions, which decode looks up in the page table and prefill
        # remaps first.
        physical_indices: Bool,
        # Set by the Python wrapper from whether each operand was supplied.
        # An absent operand arrives as a placeholder tensor that is never
        # read; the decode kernels are specialized on its absence.
        has_topk_lengths: Bool = True,
        has_attn_sink: Bool = True,
        # The sparse index producer's row layout (`SparseRowLayout`), which
        # bounds each row by its closed-form valid length. The defaults are
        # plain top-k, which adds no bound.
        index_kpool: Int = 1,
        index_pool_topk: Int = 0,
        index_kpool_select_tail: Bool = True,
        # Set from the Python `attn_sink_inert` attribute when the caller knows
        # the sink is inert (-1e38); the AMD sparse arms reject a live one.
        attn_sink_inert: Bool = False,
    ](
        output: Some[Output & Tensor],
        q: Some[Input & Tensor],
        kv: Some[Input & Fused & LoadAccess],
        input_row_offsets: Some[Input & Tensor],
        freqs_cis: Some[Input & Tensor],
        kv_norm_gamma: Some[Input & Tensor],
        buffer_row_offsets_1d: Some[Input & Tensor],
        cache_offsets_1d: Some[Input & Tensor],
        buffer_length: Int32,
        w_k: Some[Input & Tensor],
        w_uk: Some[Input & Tensor],
        w_uv: Some[Input & Tensor],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        layer_idx: UInt32,
        scale: Float32,
        epsilon: Float32,
        w_k_scale: Some[Input & Tensor],
        w_uk_scale: Some[Input & Tensor],
        w_uv_scale: Some[Input & Tensor],
        scalar_args: Some[Input & Tensor],
        sparse_indices: Some[Input & Tensor],
        topk_lengths: Some[Input & Tensor],
        attn_sink: Some[Input & Tensor],
        num_partitions_scalar: Some[Input & Tensor],
        selection_lag: Int32,
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(q), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(freqs_cis), TileTensorable)
        comptime assert conforms_to(type_of(kv_norm_gamma), TileTensorable)
        comptime assert conforms_to(
            type_of(buffer_row_offsets_1d), TileTensorable
        )
        comptime assert conforms_to(type_of(cache_offsets_1d), TileTensorable)
        comptime assert conforms_to(type_of(w_k), TileTensorable)
        comptime assert conforms_to(type_of(w_uk), TileTensorable)
        comptime assert conforms_to(type_of(w_uv), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert conforms_to(type_of(w_k_scale), TileTensorable)
        comptime assert conforms_to(type_of(w_uk_scale), TileTensorable)
        comptime assert conforms_to(type_of(w_uv_scale), TileTensorable)
        comptime assert conforms_to(type_of(scalar_args), TileTensorable)
        comptime assert conforms_to(type_of(sparse_indices), TileTensorable)
        comptime assert conforms_to(type_of(topk_lengths), TileTensorable)
        comptime assert conforms_to(type_of(attn_sink), TileTensorable)
        comptime assert type_of(output).rank == 3
        comptime assert type_of(q).rank == 3
        comptime assert type_of(kv).rank == 2
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(freqs_cis).rank == 2
        comptime assert type_of(kv_norm_gamma).rank == 1
        comptime assert type_of(buffer_row_offsets_1d).rank == 1
        comptime assert type_of(cache_offsets_1d).rank == 1
        comptime assert type_of(w_k).rank == 2
        comptime assert type_of(w_uk).rank == 3
        comptime assert type_of(w_uv).rank == 3
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(w_k_scale).rank == 2
        comptime assert type_of(w_uk_scale).rank == 3
        comptime assert type_of(w_uv_scale).rank == 3
        comptime assert type_of(scalar_args).rank == 1
        comptime assert type_of(sparse_indices).rank == 2
        comptime assert type_of(topk_lengths).rank == 1
        comptime assert type_of(attn_sink).rank == 1
        comptime assert type_of(num_partitions_scalar).rank == 1
        comptime assert type_of(kv).dtype == DType.bfloat16
        comptime dtype = type_of(output).dtype
        comptime fp8_dtype = type_of(w_k).dtype
        comptime fp8_scale_dtype = type_of(w_k_scale).dtype
        comptime cache_dtype = type_of(kv_blocks).dtype
        comptime assert type_of(q).dtype == dtype
        comptime assert type_of(w_uk).dtype == fp8_dtype
        comptime assert type_of(w_uv).dtype == fp8_dtype
        comptime assert type_of(w_uk_scale).dtype == fp8_scale_dtype
        comptime assert type_of(w_uv_scale).dtype == fp8_scale_dtype
        var kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )

        comptime assert context.target.is_gpu(), (
            "attention.mla.graph.prefill.decode.paged.fp8.sparse is only"
            " supported on GPU"
        )

        comptime assert conforms_to(type_of(kv), UnsafeFusedView)
        var kv_view = unsafe_fused_view(kv)
        comptime assert conforms_to(type_of(kv_view), Loadable)

        @inline(.always)
        def kv_input_fn[
            width: Int
        ](coords: IndexList[2]) {var kv_view} -> SIMD[.bfloat16, width]:
            return rebind[SIMD[.bfloat16, width]](
                kv_view.load[width=width, element_alignment=width](
                    Coord(coords)
                )
            )

        comptime mla_page_size = type_of(kv_blocks)._shape_types[3].static_value
        comptime num_heads = type_of(q)._shape_types[1].static_value
        var num_tokens = Int(q.shape()[0].value())

        comptime if has_topk_lengths:
            mla_sparse_check_topk_lengths[cache_dtype, num_heads](
                Int(topk_lengths.shape()[0].value()),
                Int(input_row_offsets.shape()[0].value()) - 1,
                num_tokens,
                Int(max_prompt_length.to_tile_tensor[.uint32]().ptr[0]),
            )

        var topk_lengths_ptr = maybe_null_pointer[has_topk_lengths](
            topk_lengths.to_tile_tensor[.int32]().ptr
        )
        var attn_sink_ptr = maybe_null_pointer[has_attn_sink](
            attn_sink.to_tile_tensor[.float32]().ptr
        )

        with Trace[TraceLevel.OP, target=context.target.device_kind_label()](
            "attention.mla.graph.prefill.decode.paged.fp8.sparse",
            task_id=get_safe_task_id(context),
        ):
            var decode_indices = _sparse_indices_ptr[indices_stride](
                sparse_indices.to_tile_tensor[.int32](), num_tokens
            )
            var scratch_sparse_indices = Optional[DeviceBuffer[DType.int32]]()
            var prefill_indices = _sparse_prefill_indices[
                mla_page_size,
                indices_stride,
                cache_dtype,
                num_heads,
                physical_indices,
                q_dtype=dtype,
            ](
                scratch_sparse_indices,
                sparse_indices.to_tile_tensor[.int32](),
                input_row_offsets.to_tile_tensor[.uint32](),
                kv_lookup_table.to_tile_tensor[.uint32](),
                kv_blocks.to_tile_tensor(),
                Int(kv_collection.max_seq_length),
                context,
            )
            mla_prefill_decode_graph_fp8[
                m_scale_granularity=m_scale_granularity,
                n_scale_granularity=n_scale_granularity,
                k_scale_granularity=k_scale_granularity,
                mask_str=mask_str,
                sparse_mla=True,
                sparse_indices_stride=indices_stride,
                physical_indices=physical_indices,
                # TODO(KERN-3141): fold a shared row here once the decode
                # branch takes the logical positions to mask it by.
                sparse_row=SparseRowLayout(
                    kpool=index_kpool,
                    pool_topk=index_pool_topk,
                    select_tail=index_kpool_select_tail,
                ),
                attn_sink_inert=attn_sink_inert,
                kernels=Self.kernels,
            ](
                output.to_tile_tensor[dtype](),
                q.to_tile_tensor[dtype](),
                input_row_offsets.to_tile_tensor[.uint32](),
                freqs_cis.to_tile_tensor(),
                kv_norm_gamma.to_tile_tensor(),
                kv_collection,
                layer_idx,
                scale,
                epsilon,
                buffer_row_offsets_1d.to_tile_tensor[.uint32](),
                cache_offsets_1d.to_tile_tensor[.uint32](),
                Int(buffer_length),
                Int(kv_collection.max_seq_length),
                w_k.to_tile_tensor[fp8_dtype](),
                w_k_scale.to_tile_tensor[fp8_scale_dtype](),
                w_uk.to_tile_tensor[fp8_dtype](),
                w_uk_scale.to_tile_tensor[fp8_scale_dtype](),
                w_uv.to_tile_tensor[fp8_dtype](),
                w_uv_scale.to_tile_tensor[fp8_scale_dtype](),
                scalar_args.to_tile_tensor[.int64](),
                kv_input_fn,
                context,
                decode_indices,
                indices_stride,
                topk_lengths_ptr,
                attn_sink_ptr,
                num_partitions_in=_amd_sparse_decode_partitions(
                    num_tokens,
                    indices_stride,
                    num_heads,
                    context,
                ),
                prefill_d_indices=prefill_indices,
                selection_lag=Int(selection_lag),
            )
            _ = scratch_sparse_indices^


@extensibility.register("attention.mla.graph.prefill.decode.paged.fp8.sparse")
struct Struct_mla_prefill_graph_decode_paged_fp8_sparse(
    _MLAPrefillGraphDecodePagedFP8SparseOp
):
    """Registers the op for pre-SM100 NVIDIA GPUs and any other device."""

    comptime kernels: MLAGraphKernels = LegacyNVMLAGraph


@extensibility.register(
    "attention.mla.graph.prefill.decode.paged.fp8.sparse", type="gpu", api="hip"
)
struct Struct_mla_prefill_graph_decode_paged_fp8_sparse_amd(
    _MLAPrefillGraphDecodePagedFP8SparseOp
):
    """Registers the op for AMD GPUs."""

    comptime kernels: MLAGraphKernels = AMDMLAGraph


@extensibility.register(
    "attention.mla.graph.prefill.decode.paged.fp8.sparse",
    type="gpu",
    api="cuda",
    arch="sm_100a",
)
struct Struct_mla_prefill_graph_decode_paged_fp8_sparse_sm100(
    _MLAPrefillGraphDecodePagedFP8SparseOp
):
    """Registers the op for B100 and B200."""

    comptime kernels: MLAGraphKernels = SM100MLAGraph


@extensibility.register(
    "attention.mla.graph.prefill.decode.paged.fp8.sparse",
    type="gpu",
    api="cuda",
    arch="sm_103a",
)
struct Struct_mla_prefill_graph_decode_paged_fp8_sparse_sm103(
    _MLAPrefillGraphDecodePagedFP8SparseOp
):
    """Registers the op for B300."""

    comptime kernels: MLAGraphKernels = SM100MLAGraph


trait _MLAPrefillGraphDecodeBF16PagedOp:
    """Implements `attention.mla.graph.prefill.decode.paged`.

    Each registration picks its kernels.
    """

    comptime kernels: MLAGraphKernels

    @inline(.always)
    @staticmethod
    def execute[
        mask_str: StaticString,
    ](
        output: Some[Output & Tensor],
        q: Some[Input & Tensor],
        kv: Some[Input & Fused & LoadAccess],
        input_row_offsets: Some[Input & Tensor],
        freqs_cis: Some[Input & Tensor],
        kv_norm_gamma: Some[Input & Tensor],
        buffer_row_offsets_1d: Some[Input & Tensor],
        cache_offsets_1d: Some[Input & Tensor],
        buffer_length: Int32,
        w_k: Some[Input & Tensor],
        w_uk: Some[Input & Tensor],
        w_uv: Some[Input & Tensor],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        layer_idx: UInt32,
        scale: Float32,
        epsilon: Float32,
        scalar_args: Some[Input & Tensor],
        num_partitions_scalar: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(q), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(freqs_cis), TileTensorable)
        comptime assert conforms_to(type_of(kv_norm_gamma), TileTensorable)
        comptime assert conforms_to(
            type_of(buffer_row_offsets_1d), TileTensorable
        )
        comptime assert conforms_to(type_of(cache_offsets_1d), TileTensorable)
        comptime assert conforms_to(type_of(w_k), TileTensorable)
        comptime assert conforms_to(type_of(w_uk), TileTensorable)
        comptime assert conforms_to(type_of(w_uv), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert conforms_to(type_of(scalar_args), TileTensorable)
        comptime assert conforms_to(
            type_of(num_partitions_scalar), TileTensorable
        )
        comptime assert type_of(output).rank == 3
        comptime assert type_of(q).rank == 3
        comptime assert type_of(kv).rank == 2
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(freqs_cis).rank == 2
        comptime assert type_of(kv_norm_gamma).rank == 1
        comptime assert type_of(buffer_row_offsets_1d).rank == 1
        comptime assert type_of(cache_offsets_1d).rank == 1
        comptime assert type_of(w_k).rank == 2
        comptime assert type_of(w_uk).rank == 3
        comptime assert type_of(w_uv).rank == 3
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(scalar_args).rank == 1
        comptime assert type_of(num_partitions_scalar).rank == 1
        comptime assert type_of(kv).dtype == DType.bfloat16
        var kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )

        comptime assert (
            context.target.is_gpu()
        ), "attention.mla.graph.prefill.decode.paged is only supported on GPU"

        comptime assert conforms_to(type_of(kv), UnsafeFusedView)
        var kv_view = unsafe_fused_view(kv)
        comptime assert conforms_to(type_of(kv_view), Loadable)

        @inline(.always)
        def kv_input_fn[
            width: Int
        ](coords: IndexList[2]) {var kv_view} -> SIMD[.bfloat16, width]:
            return rebind[SIMD[.bfloat16, width]](
                kv_view.load[width=width, element_alignment=width](
                    Coord(coords)
                )
            )

        var num_partitions_proj = Int(
            num_partitions_scalar.to_tile_tensor[.int64]().load[width=1](
                Coord(0)
            )
        )

        with Trace[TraceLevel.OP, target=context.target.device_kind_label()](
            "attention.mla.graph.prefill.decode.paged",
            task_id=get_safe_task_id(context),
        ):
            mla_prefill_decode_graph_bf16[
                mask_str=mask_str,
                kernels=Self.kernels,
            ](
                output.to_tile_tensor[.bfloat16](),
                q.to_tile_tensor[.bfloat16](),
                input_row_offsets.to_tile_tensor[.uint32](),
                freqs_cis.to_tile_tensor(),
                kv_norm_gamma.to_tile_tensor(),
                kv_collection,
                layer_idx,
                scale,
                epsilon,
                buffer_row_offsets_1d.to_tile_tensor[.uint32](),
                cache_offsets_1d.to_tile_tensor[.uint32](),
                Int(buffer_length),
                Int(kv_collection.max_seq_length),
                w_k.to_tile_tensor[.bfloat16](),
                w_uk.to_tile_tensor[.bfloat16](),
                w_uv.to_tile_tensor[.bfloat16](),
                scalar_args.to_tile_tensor[.int64](),
                kv_input_fn,
                context,
                num_partitions_in=num_partitions_proj,
            )


@extensibility.register("attention.mla.graph.prefill.decode.paged")
struct Struct_mla_prefill_graph_decode_bf16_paged(
    _MLAPrefillGraphDecodeBF16PagedOp
):
    """Registers the op for pre-SM100 NVIDIA GPUs and any other device."""

    comptime kernels: MLAGraphKernels = LegacyNVMLAGraph


@extensibility.register(
    "attention.mla.graph.prefill.decode.paged", type="gpu", api="hip"
)
struct Struct_mla_prefill_graph_decode_bf16_paged_amd(
    _MLAPrefillGraphDecodeBF16PagedOp
):
    """Registers the op for AMD GPUs."""

    comptime kernels: MLAGraphKernels = AMDMLAGraph


@extensibility.register(
    "attention.mla.graph.prefill.decode.paged",
    type="gpu",
    api="cuda",
    arch="sm_100a",
)
struct Struct_mla_prefill_graph_decode_bf16_paged_sm100(
    _MLAPrefillGraphDecodeBF16PagedOp
):
    """Registers the op for B100 and B200."""

    comptime kernels: MLAGraphKernels = SM100MLAGraph


@extensibility.register(
    "attention.mla.graph.prefill.decode.paged",
    type="gpu",
    api="cuda",
    arch="sm_103a",
)
struct Struct_mla_prefill_graph_decode_bf16_paged_sm103(
    _MLAPrefillGraphDecodeBF16PagedOp
):
    """Registers the op for B300."""

    comptime kernels: MLAGraphKernels = SM100MLAGraph


@extensibility.register("attention.mla.store.ragged.paged")
struct Struct_mla_store_ragged_paged:
    """Registers the `attention.mla.store.ragged.paged` graph op with the graph
    compiler.
    """

    @inline(.always)
    @staticmethod
    def execute[](
        kv: Some[Input & Fused & LoadAccess],
        input_row_offsets: Some[Input & Tensor],
        freqs_cis: Some[Input & Tensor],
        kv_norm_gamma: Some[Input & Tensor],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        layer_idx: UInt32,
        epsilon: Float32,
        context: DeviceContext,
    ) raises:
        """Stores raw MLA latent rows into the paged cache, normed and roped.

        The store half of `attention.mla.graph.prefill.decode.paged` without the
        attention: each row of `kv` -- a request's `token_idx`-th row, found
        via `input_row_offsets` -- is RMSNormed on its first `kv_lora_rank`
        elements against `kv_norm_gamma`, roped on its trailing rope slice at
        its absolute position `cache_length + token_idx` through `freqs_cis`,
        cast to the cache dtype, and written to `kv_blocks`. Rows a
        speculative drafter stores but never queries need this; the attention
        op always attends every query row, so it cannot stand in for the
        store.

        Args:
            kv: Raw (un-normalized) latent rows, `[total_seq_len,
                kv_lora_rank + qk_rope_head_dim]`, bfloat16.
            input_row_offsets: Row offsets per request, `[batch + 1]`.
            freqs_cis: Interleaved RoPE table, `[max_position,
                qk_rope_head_dim]`; row `cache_length + token_idx` ropes a row.
            kv_norm_gamma: RMSNorm gamma, `[kv_lora_rank]`.
            kv_blocks: The paged cache's blocks, `[pages, 1, layers,
                page_size, 1, kv_lora_rank + qk_rope_head_dim]`.
            page_stride: Page-to-page distance in elements; -1 when packed.
            cache_lengths: Cached tokens per request, `[batch]`.
            kv_lookup_table: Page lookup table, `[batch, pages_per_seq]`.
            max_prompt_length: The batch's max prompt length, `[1]`.
            max_cache_length: The batch's max cache length, `[1]`.
            layer_idx: Layer of the cache to write.
            epsilon: RMSNorm epsilon.
            context: Device context for GPU execution.
        """
        comptime assert (
            context.target.is_gpu()
        ), "attention.mla.store.ragged.paged is only supported on GPU"
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(freqs_cis), TileTensorable)
        comptime assert conforms_to(type_of(kv_norm_gamma), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert type_of(kv).rank == 2
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(freqs_cis).rank == 2
        comptime assert type_of(kv_norm_gamma).rank == 1
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(kv).dtype == DType.bfloat16
        comptime cache_head_size = type_of(kv_blocks)._shape_types[
            5
        ].static_value
        comptime kv_norm_dim = type_of(kv_norm_gamma)._shape_types[
            0
        ].static_value
        comptime assert (
            type_of(kv)._shape_types[1].static_value == cache_head_size
        ), "kv's width must equal the MLA cache's head size"
        comptime assert (
            cache_head_size > kv_norm_dim
        ), "the store op needs a rope slice: a cache row wider than kv_norm_dim"

        var kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )

        var num_rows = Int(kv.shape()[0].value())
        comptime assert conforms_to(type_of(kv), UnsafeFusedView)
        var kv_view = unsafe_fused_view(kv)
        comptime assert conforms_to(type_of(kv_view), Loadable)

        @inline(.always)
        def kv_input_fn[
            width: Int
        ](coords: IndexList[2]) {var kv_view} -> SIMD[.bfloat16, width]:
            return rebind[SIMD[.bfloat16, width]](
                kv_view.load[width=width, element_alignment=width](
                    Coord(coords)
                )
            )

        with Trace[TraceLevel.OP, target=context.target.device_kind_label()](
            "attention.mla.store.ragged.paged",
            task_id=get_safe_task_id(context),
        ):
            mla_store_ragged_paged(
                num_rows,
                input_row_offsets.to_tile_tensor[.uint32](),
                freqs_cis.to_tile_tensor(),
                kv_norm_gamma.to_tile_tensor(),
                kv_collection,
                layer_idx,
                epsilon,
                kv_input_fn,
                context,
            )


trait _MLAPrefillGraphDecodePagedSparseOp:
    """Implements `attention.mla.graph.prefill.decode.paged.sparse`.

    Each registration picks its kernels.
    """

    comptime kernels: MLAGraphKernels

    @inline(.always)
    @staticmethod
    def execute[
        mask_str: StaticString,
        indices_stride: Int,
        # Whether `sparse_indices` holds the indexer's paged physical rows
        # (`attention.mla.indexer.ragged.float8.paged.remap`) rather than logical
        # token positions, which decode looks up in the page table and prefill
        # remaps first.
        physical_indices: Bool,
        # Set by the Python wrapper from whether each operand was supplied.
        # An absent operand arrives as a placeholder tensor that is never
        # read; the decode kernels are specialized on its absence.
        has_topk_lengths: Bool = True,
        has_attn_sink: Bool = True,
        # The sparse index producer's row layout (`SparseRowLayout`), which
        # bounds each row by its closed-form valid length. The defaults are
        # plain top-k, which adds no bound.
        index_kpool: Int = 1,
        index_pool_topk: Int = 0,
        index_kpool_select_tail: Bool = True,
    ](
        output: Some[Output & Tensor],
        q: Some[Input & Tensor],
        kv: Some[Input & Fused & LoadAccess],
        input_row_offsets: Some[Input & Tensor],
        freqs_cis: Some[Input & Tensor],
        kv_norm_gamma: Some[Input & Tensor],
        buffer_row_offsets_1d: Some[Input & Tensor],
        cache_offsets_1d: Some[Input & Tensor],
        buffer_length: Int32,
        w_k: Some[Input & Tensor],
        w_uk: Some[Input & Tensor],
        w_uv: Some[Input & Tensor],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        layer_idx: UInt32,
        scale: Float32,
        epsilon: Float32,
        scalar_args: Some[Input & Tensor],
        sparse_indices: Some[Input & Tensor],
        topk_lengths: Some[Input & Tensor],
        attn_sink: Some[Input & Tensor],
        num_partitions_scalar: Some[Input & Tensor],
        selection_lag: Int32,
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(q), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(freqs_cis), TileTensorable)
        comptime assert conforms_to(type_of(kv_norm_gamma), TileTensorable)
        comptime assert conforms_to(
            type_of(buffer_row_offsets_1d), TileTensorable
        )
        comptime assert conforms_to(type_of(cache_offsets_1d), TileTensorable)
        comptime assert conforms_to(type_of(w_k), TileTensorable)
        comptime assert conforms_to(type_of(w_uk), TileTensorable)
        comptime assert conforms_to(type_of(w_uv), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert conforms_to(type_of(scalar_args), TileTensorable)
        comptime assert conforms_to(type_of(sparse_indices), TileTensorable)
        comptime assert conforms_to(type_of(topk_lengths), TileTensorable)
        comptime assert conforms_to(type_of(attn_sink), TileTensorable)
        comptime assert type_of(output).rank == 3
        comptime assert type_of(q).rank == 3
        comptime assert type_of(kv).rank == 2
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(freqs_cis).rank == 2
        comptime assert type_of(kv_norm_gamma).rank == 1
        comptime assert type_of(buffer_row_offsets_1d).rank == 1
        comptime assert type_of(cache_offsets_1d).rank == 1
        comptime assert type_of(w_k).rank == 2
        comptime assert type_of(w_uk).rank == 3
        comptime assert type_of(w_uv).rank == 3
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(scalar_args).rank == 1
        comptime assert type_of(sparse_indices).rank == 2
        comptime assert type_of(topk_lengths).rank == 1
        comptime assert type_of(attn_sink).rank == 1
        comptime assert type_of(num_partitions_scalar).rank == 1
        comptime assert type_of(kv).dtype == DType.bfloat16
        comptime cache_dtype = type_of(kv_blocks).dtype
        var kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )

        comptime assert context.target.is_gpu(), (
            "attention.mla.graph.prefill.decode.paged.sparse is only supported"
            " on GPU"
        )

        comptime assert conforms_to(type_of(kv), UnsafeFusedView)
        var kv_view = unsafe_fused_view(kv)
        comptime assert conforms_to(type_of(kv_view), Loadable)

        @inline(.always)
        def kv_input_fn[
            width: Int
        ](coords: IndexList[2]) {var kv_view} -> SIMD[.bfloat16, width]:
            return rebind[SIMD[.bfloat16, width]](
                kv_view.load[width=width, element_alignment=width](
                    Coord(coords)
                )
            )

        comptime mla_page_size = type_of(kv_blocks)._shape_types[3].static_value
        comptime num_heads = type_of(q)._shape_types[1].static_value
        var num_tokens = Int(q.shape()[0].value())

        comptime if has_topk_lengths:
            mla_sparse_check_topk_lengths[cache_dtype, num_heads](
                Int(topk_lengths.shape()[0].value()),
                Int(input_row_offsets.shape()[0].value()) - 1,
                num_tokens,
                Int(max_prompt_length.to_tile_tensor[.uint32]().ptr[0]),
            )

        var topk_lengths_ptr = maybe_null_pointer[has_topk_lengths](
            topk_lengths.to_tile_tensor[.int32]().ptr
        )
        var attn_sink_ptr = maybe_null_pointer[has_attn_sink](
            attn_sink.to_tile_tensor[.float32]().ptr
        )

        with Trace[TraceLevel.OP, target=context.target.device_kind_label()](
            "attention.mla.graph.prefill.decode.paged.sparse",
            task_id=get_safe_task_id(context),
        ):
            comptime decode_physical = _sparse_decode_physical[
                physical_indices
            ]()
            var scratch_decode_indices = Optional[DeviceBuffer[DType.int32]]()
            var decode_indices = _sparse_decode_indices[
                mla_page_size,
                indices_stride,
                physical_indices,
            ](
                scratch_decode_indices,
                sparse_indices.to_tile_tensor[.int32](),
                input_row_offsets.to_tile_tensor[.uint32](),
                kv_lookup_table.to_tile_tensor[.uint32](),
                kv_blocks.to_tile_tensor(),
                num_tokens,
                context,
            )
            var scratch_sparse_indices = Optional[DeviceBuffer[DType.int32]]()
            var prefill_indices = _sparse_prefill_indices[
                mla_page_size,
                indices_stride,
                cache_dtype,
                num_heads,
                decode_physical,
            ](
                scratch_sparse_indices,
                sparse_indices.to_tile_tensor[.int32](),
                input_row_offsets.to_tile_tensor[.uint32](),
                kv_lookup_table.to_tile_tensor[.uint32](),
                kv_blocks.to_tile_tensor(),
                Int(kv_collection.max_seq_length),
                context,
            )
            mla_prefill_decode_graph_bf16[
                mask_str=mask_str,
                sparse_mla=True,
                sparse_indices_stride=indices_stride,
                physical_indices=decode_physical,
                sparse_row=SparseRowLayout(
                    kpool=index_kpool,
                    pool_topk=index_pool_topk,
                    select_tail=index_kpool_select_tail,
                ),
                kernels=Self.kernels,
            ](
                output.to_tile_tensor[.bfloat16](),
                q.to_tile_tensor[.bfloat16](),
                input_row_offsets.to_tile_tensor[.uint32](),
                freqs_cis.to_tile_tensor(),
                kv_norm_gamma.to_tile_tensor(),
                kv_collection,
                layer_idx,
                scale,
                epsilon,
                buffer_row_offsets_1d.to_tile_tensor[.uint32](),
                cache_offsets_1d.to_tile_tensor[.uint32](),
                Int(buffer_length),
                Int(kv_collection.max_seq_length),
                w_k.to_tile_tensor[.bfloat16](),
                w_uk.to_tile_tensor[.bfloat16](),
                w_uv.to_tile_tensor[.bfloat16](),
                scalar_args.to_tile_tensor[.int64](),
                kv_input_fn,
                context,
                decode_indices,
                indices_stride,
                topk_lengths_ptr,
                attn_sink_ptr,
                # Sparse path: let kernel use its own mask-aware computation.
                num_partitions_in=None,
                prefill_d_indices=prefill_indices,
                selection_lag=Int(selection_lag),
            )
            _ = scratch_decode_indices^
            _ = scratch_sparse_indices^


@extensibility.register("attention.mla.graph.prefill.decode.paged.sparse")
struct Struct_mla_prefill_graph_decode_paged_sparse(
    _MLAPrefillGraphDecodePagedSparseOp
):
    """Registers the op for pre-SM100 NVIDIA GPUs and any other device."""

    comptime kernels: MLAGraphKernels = LegacyNVMLAGraph


@extensibility.register(
    "attention.mla.graph.prefill.decode.paged.sparse", type="gpu", api="hip"
)
struct Struct_mla_prefill_graph_decode_paged_sparse_amd(
    _MLAPrefillGraphDecodePagedSparseOp
):
    """Registers the op for AMD GPUs."""

    comptime kernels: MLAGraphKernels = AMDMLAGraph


@extensibility.register(
    "attention.mla.graph.prefill.decode.paged.sparse",
    type="gpu",
    api="cuda",
    arch="sm_100a",
)
struct Struct_mla_prefill_graph_decode_paged_sparse_sm100(
    _MLAPrefillGraphDecodePagedSparseOp
):
    """Registers the op for B100 and B200."""

    comptime kernels: MLAGraphKernels = SM100MLAGraph


@extensibility.register(
    "attention.mla.graph.prefill.decode.paged.sparse",
    type="gpu",
    api="cuda",
    arch="sm_103a",
)
struct Struct_mla_prefill_graph_decode_paged_sparse_sm103(
    _MLAPrefillGraphDecodePagedSparseOp
):
    """Registers the op for B300."""

    comptime kernels: MLAGraphKernels = SM100MLAGraph


trait _MLAPrefillGraphDecodePagedSparseNVFP4Op:
    """Implements `attention.mla.graph.prefill.decode.paged.sparse.nvfp4`.

    Each registration picks its kernels.
    """

    comptime kernels: MLAGraphKernels

    @inline(.always)
    @staticmethod
    def execute[
        mask_str: StaticString,
        indices_stride: Int,
        physical_indices: Bool,
        has_topk_lengths: Bool = True,
        has_attn_sink: Bool = True,
        index_kpool: Int = 1,
        index_pool_topk: Int = 0,
        index_kpool_select_tail: Bool = True,
        # Sparse prefill converts the context to FP8 once while it has at most
        # this many rows per query token; see `nvfp4_prefill_converts_once`.
        nvfp4_convert_ratio: Int = 0,
        nvfp4_convert_max_rows: Int = 0,
    ](
        output: Some[Output & Tensor],
        q: Some[Input & Tensor],
        kv: Some[Input & Fused & LoadAccess],
        input_row_offsets: Some[Input & Tensor],
        freqs_cis: Some[Input & Tensor],
        kv_norm_gamma: Some[Input & Tensor],
        buffer_row_offsets_1d: Some[Input & Tensor],
        cache_offsets_1d: Some[Input & Tensor],
        buffer_length: Int32,
        w_k: Some[Input & Tensor],
        w_uk: Some[Input & Tensor],
        w_uv: Some[Input & Tensor],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        kv_scales: Some[MutableInput & Tensor],
        scales_page_stride: Some[Input & Tensor],
        scales_lookup_table: Some[Input & Tensor],
        layer_idx: UInt32,
        scale: Float32,
        epsilon: Float32,
        scalar_args: Some[Input & Tensor],
        sparse_indices: Some[Input & Tensor],
        topk_lengths: Some[Input & Tensor],
        attn_sink: Some[Input & Tensor],
        num_partitions_scalar: Some[Input & Tensor],
        selection_lag: Int32,
        context: DeviceContext,
    ) raises:
        comptime assert context.target.is_gpu(), (
            "attention.mla.graph.prefill.decode.paged.sparse.nvfp4 is only"
            " supported on GPU"
        )
        comptime assert physical_indices, (
            "an NVFP4 MLA cache needs the indexer's physical and scale rows,"
            " not token positions"
        )
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(q), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(freqs_cis), TileTensorable)
        comptime assert conforms_to(type_of(kv_norm_gamma), TileTensorable)
        comptime assert conforms_to(
            type_of(buffer_row_offsets_1d), TileTensorable
        )
        comptime assert conforms_to(type_of(cache_offsets_1d), TileTensorable)
        comptime assert conforms_to(type_of(w_k), TileTensorable)
        comptime assert conforms_to(type_of(w_uk), TileTensorable)
        comptime assert conforms_to(type_of(w_uv), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert conforms_to(type_of(kv_scales), TileTensorable)
        comptime assert conforms_to(type_of(scales_page_stride), TileTensorable)
        comptime assert conforms_to(
            type_of(scales_lookup_table), TileTensorable
        )
        comptime assert conforms_to(type_of(scalar_args), TileTensorable)
        comptime assert conforms_to(type_of(sparse_indices), TileTensorable)
        comptime assert conforms_to(type_of(topk_lengths), TileTensorable)
        comptime assert conforms_to(type_of(attn_sink), TileTensorable)
        comptime assert type_of(output).rank == 3
        comptime assert type_of(q).rank == 3
        comptime assert type_of(kv).rank == 2
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(freqs_cis).rank == 2
        comptime assert type_of(kv_norm_gamma).rank == 1
        comptime assert type_of(buffer_row_offsets_1d).rank == 1
        comptime assert type_of(cache_offsets_1d).rank == 1
        comptime assert type_of(w_k).rank == 2
        comptime assert type_of(w_uk).rank == 3
        comptime assert type_of(w_uv).rank == 3
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(kv_scales).rank == 6
        comptime assert type_of(scales_page_stride).rank == 1
        comptime assert type_of(scales_lookup_table).rank == 2
        comptime assert type_of(scalar_args).rank == 1
        comptime assert type_of(sparse_indices).rank == 2
        comptime assert type_of(topk_lengths).rank == 1
        comptime assert type_of(attn_sink).rank == 1
        comptime assert type_of(num_partitions_scalar).rank == 1
        comptime assert type_of(kv).dtype == DType.bfloat16
        var kv_collection = generic_get_paged_cache_with_scales(
            kv_blocks.to_tile_tensor[.uint8](),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
            kv_scales.to_tile_tensor[.float8_e4m3fn](),
            scales_page_stride.to_tile_tensor[.int64](),
            scales_lookup_table.to_tile_tensor[.uint32](),
        )

        comptime assert conforms_to(type_of(kv), UnsafeFusedView)
        var kv_view = unsafe_fused_view(kv)
        comptime assert conforms_to(type_of(kv_view), Loadable)

        @inline(.always)
        def kv_input_fn[
            width: Int
        ](coords: IndexList[2]) {var kv_view} -> SIMD[.bfloat16, width]:
            return rebind[SIMD[.bfloat16, width]](
                kv_view.load[width=width, element_alignment=width](
                    Coord(coords)
                )
            )

        comptime mla_page_size = type_of(kv_blocks)._shape_types[3].static_value
        comptime num_heads = type_of(q)._shape_types[1].static_value
        var num_tokens = Int(q.shape()[0].value())

        # The prefill-shaped branch reads one length per query token, while
        # the plain decode branch reads one per sequence, so the buffer must
        # cover whichever the batch takes. Sparse prefill over an NVFP4
        # cache that routes to the decode kernel with tile skipping reads
        # per token too.
        comptime if has_topk_lengths:
            mla_sparse_check_topk_lengths[DType.uint8, num_heads](
                Int(topk_lengths.shape()[0].value()),
                Int(input_row_offsets.shape()[0].value()) - 1,
                num_tokens,
                Int(max_prompt_length.to_tile_tensor[.uint32]().ptr[0]),
            )

        var topk_lengths_ptr = maybe_null_pointer[has_topk_lengths](
            topk_lengths.to_tile_tensor[.int32]().ptr
        )
        var attn_sink_ptr = maybe_null_pointer[has_attn_sink](
            attn_sink.to_tile_tensor[.float32]().ptr
        )

        with Trace[TraceLevel.OP, target=context.target.device_kind_label()](
            "attention.mla.graph.prefill.decode.paged.sparse.nvfp4",
            task_id=get_safe_task_id(context),
        ):
            var decode_indices = _sparse_indices_ptr[indices_stride](
                sparse_indices.to_tile_tensor[.int32](), num_tokens
            )
            var scratch_sparse_indices = Optional[DeviceBuffer[DType.int32]]()
            var prefill_indices = _sparse_prefill_indices[
                mla_page_size,
                indices_stride,
                DType.uint8,
                num_heads,
                physical_indices,
            ](
                scratch_sparse_indices,
                sparse_indices.to_tile_tensor[.int32](),
                input_row_offsets.to_tile_tensor[.uint32](),
                kv_lookup_table.to_tile_tensor[.uint32](),
                kv_blocks.to_tile_tensor[.uint8](),
                Int(kv_collection.max_seq_length),
                context,
            )
            mla_prefill_decode_graph_bf16[
                mask_str=mask_str,
                sparse_mla=True,
                sparse_indices_stride=indices_stride,
                physical_indices=physical_indices,
                sparse_row=SparseRowLayout(
                    kpool=index_kpool,
                    pool_topk=index_pool_topk,
                    select_tail=index_kpool_select_tail,
                ),
                nvfp4_convert_ratio=nvfp4_convert_ratio,
                nvfp4_convert_max_rows=nvfp4_convert_max_rows,
                kernels=Self.kernels,
            ](
                output.to_tile_tensor[.bfloat16](),
                q.to_tile_tensor[.bfloat16](),
                input_row_offsets.to_tile_tensor[.uint32](),
                freqs_cis.to_tile_tensor(),
                kv_norm_gamma.to_tile_tensor(),
                kv_collection,
                layer_idx,
                scale,
                epsilon,
                buffer_row_offsets_1d.to_tile_tensor[.uint32](),
                cache_offsets_1d.to_tile_tensor[.uint32](),
                Int(buffer_length),
                Int(kv_collection.max_seq_length),
                w_k.to_tile_tensor[.bfloat16](),
                w_uk.to_tile_tensor[.bfloat16](),
                w_uv.to_tile_tensor[.bfloat16](),
                scalar_args.to_tile_tensor[.int64](),
                kv_input_fn,
                context,
                decode_indices,
                indices_stride,
                topk_lengths_ptr,
                attn_sink_ptr,
                num_partitions_in=None,
                prefill_d_indices=prefill_indices,
                selection_lag=Int(selection_lag),
            )
            _ = scratch_sparse_indices^


@extensibility.register("attention.mla.graph.prefill.decode.paged.sparse.nvfp4")
struct Struct_mla_prefill_graph_decode_paged_sparse_nvfp4(
    _MLAPrefillGraphDecodePagedSparseNVFP4Op
):
    """Registers the op for pre-SM100 NVIDIA GPUs and any other device."""

    comptime kernels: MLAGraphKernels = LegacyNVMLAGraph


@extensibility.register(
    "attention.mla.graph.prefill.decode.paged.sparse.nvfp4",
    type="gpu",
    api="hip",
)
struct Struct_mla_prefill_graph_decode_paged_sparse_nvfp4_amd(
    _MLAPrefillGraphDecodePagedSparseNVFP4Op
):
    """Registers the op for AMD GPUs."""

    comptime kernels: MLAGraphKernels = AMDMLAGraph


@extensibility.register(
    "attention.mla.graph.prefill.decode.paged.sparse.nvfp4",
    type="gpu",
    api="cuda",
    arch="sm_100a",
)
struct Struct_mla_prefill_graph_decode_paged_sparse_nvfp4_sm100(
    _MLAPrefillGraphDecodePagedSparseNVFP4Op
):
    """Registers the op for B100 and B200."""

    comptime kernels: MLAGraphKernels = SM100MLAGraph


@extensibility.register(
    "attention.mla.graph.prefill.decode.paged.sparse.nvfp4",
    type="gpu",
    api="cuda",
    arch="sm_103a",
)
struct Struct_mla_prefill_graph_decode_paged_sparse_nvfp4_sm103(
    _MLAPrefillGraphDecodePagedSparseNVFP4Op
):
    """Registers the op for B300."""

    comptime kernels: MLAGraphKernels = SM100MLAGraph


trait _MLAPrefillGraphDecodeBF16PagedQuantizedOp:
    """Implements `attention.mla.graph.prefill.decode.paged.quantized`.

    Each registration picks its kernels.
    """

    comptime kernels: MLAGraphKernels

    @inline(.always)
    @staticmethod
    def execute[
        mask_str: StaticString,
    ](
        output: Some[Output & Tensor],
        q: Some[Input & Tensor],
        kv: Some[Input & Fused & LoadAccess],
        input_row_offsets: Some[Input & Tensor],
        freqs_cis: Some[Input & Tensor],
        kv_norm_gamma: Some[Input & Tensor],
        buffer_row_offsets_1d: Some[Input & Tensor],
        cache_offsets_1d: Some[Input & Tensor],
        buffer_length: Int32,
        w_k: Some[Input & Tensor],
        w_uk: Some[Input & Tensor],
        w_uv: Some[Input & Tensor],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        kv_scales: Some[MutableInput & Tensor],
        layer_idx: UInt32,
        scale: Float32,
        epsilon: Float32,
        scalar_args: Some[Input & Tensor],
        num_partitions_scalar: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(q), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(freqs_cis), TileTensorable)
        comptime assert conforms_to(type_of(kv_norm_gamma), TileTensorable)
        comptime assert conforms_to(
            type_of(buffer_row_offsets_1d), TileTensorable
        )
        comptime assert conforms_to(type_of(cache_offsets_1d), TileTensorable)
        comptime assert conforms_to(type_of(w_k), TileTensorable)
        comptime assert conforms_to(type_of(w_uk), TileTensorable)
        comptime assert conforms_to(type_of(w_uv), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert conforms_to(type_of(scalar_args), TileTensorable)
        comptime assert conforms_to(
            type_of(num_partitions_scalar), TileTensorable
        )
        comptime assert type_of(output).rank == 3
        comptime assert type_of(q).rank == 3
        comptime assert type_of(kv).rank == 2
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(freqs_cis).rank == 2
        comptime assert type_of(kv_norm_gamma).rank == 1
        comptime assert type_of(buffer_row_offsets_1d).rank == 1
        comptime assert type_of(cache_offsets_1d).rank == 1
        comptime assert type_of(w_k).rank == 2
        comptime assert type_of(w_uk).rank == 3
        comptime assert type_of(w_uv).rank == 3
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(kv_scales).rank == 6
        comptime assert type_of(scalar_args).rank == 1
        comptime assert type_of(num_partitions_scalar).rank == 1
        comptime assert type_of(kv).dtype == DType.bfloat16
        var kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )

        comptime assert context.target.is_gpu(), (
            "attention.mla.graph.prefill.decode.paged.quantized is only"
            " supported on GPU"
        )

        comptime assert conforms_to(type_of(kv), UnsafeFusedView)
        var kv_view = unsafe_fused_view(kv)
        comptime assert conforms_to(type_of(kv_view), Loadable)

        @inline(.always)
        def kv_input_fn[
            width: Int
        ](coords: IndexList[2]) {var kv_view} -> SIMD[.bfloat16, width]:
            return rebind[SIMD[.bfloat16, width]](
                kv_view.load[width=width, element_alignment=width](
                    Coord(coords)
                )
            )

        var num_partitions_proj = Int(
            num_partitions_scalar.to_tile_tensor[.int64]().load[width=1](
                Coord(0)
            )
        )

        with Trace[TraceLevel.OP, target=context.target.device_kind_label()](
            "attention.mla.graph.prefill.decode.paged.quantized",
            task_id=get_safe_task_id(context),
        ):
            mla_prefill_decode_graph_bf16[
                mask_str=mask_str,
                kernels=Self.kernels,
            ](
                output.to_tile_tensor[.bfloat16](),
                q.to_tile_tensor[.bfloat16](),
                input_row_offsets.to_tile_tensor[.uint32](),
                freqs_cis.to_tile_tensor(),
                kv_norm_gamma.to_tile_tensor(),
                kv_collection,
                layer_idx,
                scale,
                epsilon,
                buffer_row_offsets_1d.to_tile_tensor[.uint32](),
                cache_offsets_1d.to_tile_tensor[.uint32](),
                Int(buffer_length),
                Int(kv_collection.max_seq_length),
                w_k.to_tile_tensor[.bfloat16](),
                w_uk.to_tile_tensor[.bfloat16](),
                w_uv.to_tile_tensor[.bfloat16](),
                scalar_args.to_tile_tensor[.int64](),
                kv_input_fn,
                context,
                num_partitions_in=num_partitions_proj,
            )


@extensibility.register("attention.mla.graph.prefill.decode.paged.quantized")
struct Struct_mla_prefill_graph_decode_bf16_paged_quantized(
    _MLAPrefillGraphDecodeBF16PagedQuantizedOp
):
    """Registers the op for pre-SM100 NVIDIA GPUs and any other device."""

    comptime kernels: MLAGraphKernels = LegacyNVMLAGraph


@extensibility.register(
    "attention.mla.graph.prefill.decode.paged.quantized", type="gpu", api="hip"
)
struct Struct_mla_prefill_graph_decode_bf16_paged_quantized_amd(
    _MLAPrefillGraphDecodeBF16PagedQuantizedOp
):
    """Registers the op for AMD GPUs."""

    comptime kernels: MLAGraphKernels = AMDMLAGraph


@extensibility.register(
    "attention.mla.graph.prefill.decode.paged.quantized",
    type="gpu",
    api="cuda",
    arch="sm_100a",
)
struct Struct_mla_prefill_graph_decode_bf16_paged_quantized_sm100(
    _MLAPrefillGraphDecodeBF16PagedQuantizedOp
):
    """Registers the op for B100 and B200."""

    comptime kernels: MLAGraphKernels = SM100MLAGraph


@extensibility.register(
    "attention.mla.graph.prefill.decode.paged.quantized",
    type="gpu",
    api="cuda",
    arch="sm_103a",
)
struct Struct_mla_prefill_graph_decode_bf16_paged_quantized_sm103(
    _MLAPrefillGraphDecodeBF16PagedQuantizedOp
):
    """Registers the op for B300."""

    comptime kernels: MLAGraphKernels = SM100MLAGraph


@extensibility.register("attention.cross.ragged.paged")
struct Struct_cross_attention_ragged_paged:
    """Registers the `attention.cross.ragged.paged` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute[
        mask_str: StaticString,
        local_window_size: Int = -1,
    ](
        output: Some[Output & Tensor],
        q: Some[Input & Tensor],
        q_input_row_offsets: Some[Input & Tensor],
        q_max_seq_len: Some[Input & Tensor],
        kv_input_row_offsets: Some[Input & Tensor],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        layer_idx: UInt32,
        scale: Float32,
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(q), TileTensorable)
        comptime assert conforms_to(
            type_of(q_input_row_offsets), TileTensorable
        )
        comptime assert conforms_to(type_of(q_max_seq_len), TileTensorable)
        comptime assert conforms_to(
            type_of(kv_input_row_offsets), TileTensorable
        )
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime out_dtype = type_of(output).dtype
        comptime assert type_of(output).rank == 3
        comptime assert type_of(q).rank == 3
        comptime assert type_of(q_input_row_offsets).rank == 1
        comptime assert type_of(q_max_seq_len).rank == 1
        comptime assert type_of(kv_input_row_offsets).rank == 1
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        var kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )
        generic_cross_attention_kv_cache[
            mask_str=mask_str,
            local_window_size=local_window_size,
            output_dtype=out_dtype,
        ](
            q.to_tile_tensor(),
            q_input_row_offsets.to_tile_tensor[.uint32](),
            q_max_seq_len.to_tile_tensor[.uint32](),
            kv_input_row_offsets.to_tile_tensor[.uint32](),
            kv_collection,
            layer_idx,
            scale,
            output.to_tile_tensor(),
            context,
        )


# ===-----------------------------------------------------------------------===#
# Latent sparse attention (DeepSeek-V4 CSA): window leaf + compressed leaf
# ===-----------------------------------------------------------------------===#


trait _LatentSparseAttentionRaggedPagedOp:
    """Implements `attention.latent_sparse.ragged.paged`.

    Each registration picks its kernels.
    """

    comptime kernels: MLAGraphKernels

    @inline(.always)
    @staticmethod
    def execute[
        window: Int,
        split_k: Bool = True,
        portable_p_bf16: Bool = False,
    ](
        output: Some[Output & Tensor],
        q: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        comp_indices: Some[Input & Tensor],
        attn_sink: Some[Input & Tensor],
        swa_kv_blocks: Some[MutableInput & Tensor],
        swa_page_stride: Some[Input & Tensor],
        swa_cache_lengths: Some[Input & Tensor],
        swa_kv_lookup_table: Some[Input & Tensor],
        swa_max_prompt_length: Some[Input & Tensor],
        swa_max_cache_length: Some[Input & Tensor],
        comp_kv_blocks: Some[MutableInput & Tensor],
        comp_page_stride: Some[Input & Tensor],
        comp_cache_lengths: Some[Input & Tensor],
        comp_kv_lookup_table: Some[Input & Tensor],
        comp_max_prompt_length: Some[Input & Tensor],
        comp_max_cache_length: Some[Input & Tensor],
        layer_swa: UInt32,
        layer_comp: UInt32,
        scale: Float32,
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(q), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(comp_indices), TileTensorable)
        comptime assert conforms_to(type_of(attn_sink), TileTensorable)
        comptime assert conforms_to(type_of(swa_kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(swa_page_stride), TileTensorable)
        comptime assert conforms_to(type_of(swa_cache_lengths), TileTensorable)
        comptime assert conforms_to(
            type_of(swa_kv_lookup_table), TileTensorable
        )
        comptime assert conforms_to(
            type_of(swa_max_prompt_length), TileTensorable
        )
        comptime assert conforms_to(
            type_of(swa_max_cache_length), TileTensorable
        )
        comptime assert conforms_to(type_of(comp_kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(comp_page_stride), TileTensorable)
        comptime assert conforms_to(type_of(comp_cache_lengths), TileTensorable)
        comptime assert conforms_to(
            type_of(comp_kv_lookup_table), TileTensorable
        )
        comptime assert conforms_to(
            type_of(comp_max_prompt_length), TileTensorable
        )
        comptime assert conforms_to(
            type_of(comp_max_cache_length), TileTensorable
        )
        comptime q_type = type_of(output).dtype
        comptime assert type_of(output).rank == 3
        comptime assert type_of(q).rank == 3
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(comp_indices).rank == 2
        comptime assert type_of(attn_sink).rank == 1
        comptime assert type_of(swa_kv_blocks).rank == 6
        comptime assert type_of(swa_page_stride).rank == 1
        comptime assert type_of(swa_cache_lengths).rank == 1
        comptime assert type_of(swa_kv_lookup_table).rank == 2
        comptime assert type_of(swa_max_prompt_length).rank == 1
        comptime assert type_of(swa_max_cache_length).rank == 1
        comptime assert type_of(comp_kv_blocks).rank == 6
        comptime assert type_of(comp_page_stride).rank == 1
        comptime assert type_of(comp_cache_lengths).rank == 1
        comptime assert type_of(comp_kv_lookup_table).rank == 2
        comptime assert type_of(comp_max_prompt_length).rank == 1
        comptime assert type_of(comp_max_cache_length).rank == 1
        var swa_collection = generic_get_paged_cache(
            swa_kv_blocks.to_tile_tensor(),
            swa_page_stride.to_tile_tensor[.int64](),
            swa_cache_lengths.to_tile_tensor[.uint32](),
            swa_kv_lookup_table.to_tile_tensor[.uint32](),
            swa_max_prompt_length.to_tile_tensor[.uint32](),
            swa_max_cache_length.to_tile_tensor[.uint32](),
        )
        var comp_collection = generic_get_paged_cache(
            comp_kv_blocks.to_tile_tensor(),
            comp_page_stride.to_tile_tensor[.int64](),
            comp_cache_lengths.to_tile_tensor[.uint32](),
            comp_kv_lookup_table.to_tile_tensor[.uint32](),
            comp_max_prompt_length.to_tile_tensor[.uint32](),
            comp_max_cache_length.to_tile_tensor[.uint32](),
        )
        Self.kernels.latent_sparse_attention[
            window=window,
            split_k=split_k,
            portable_p_bf16=portable_p_bf16,
        ](
            output.to_tile_tensor(),
            q.to_tile_tensor[q_type](),
            input_row_offsets.to_tile_tensor[.uint32](),
            comp_indices.to_tile_tensor[.int32](),
            attn_sink.to_tile_tensor[.float32](),
            swa_collection.get_key_cache(Int(layer_swa)),
            comp_collection.get_key_cache(Int(layer_comp)),
            scale,
            context,
        )


@extensibility.register("attention.latent_sparse.ragged.paged")
struct Struct_latent_sparse_attention_ragged_paged(
    _LatentSparseAttentionRaggedPagedOp
):
    """Registers the op for pre-SM100 NVIDIA GPUs and any other device."""

    comptime kernels: MLAGraphKernels = LegacyNVMLAGraph


@extensibility.register(
    "attention.latent_sparse.ragged.paged", type="gpu", api="hip"
)
struct Struct_latent_sparse_attention_ragged_paged_amd(
    _LatentSparseAttentionRaggedPagedOp
):
    """Registers the op for AMD GPUs."""

    comptime kernels: MLAGraphKernels = AMDMLAGraph


@extensibility.register(
    "attention.latent_sparse.ragged.paged",
    type="gpu",
    api="cuda",
    arch="sm_100a",
)
struct Struct_latent_sparse_attention_ragged_paged_sm100(
    _LatentSparseAttentionRaggedPagedOp
):
    """Registers the op for B100 and B200."""

    comptime kernels: MLAGraphKernels = SM100MLAGraph


@extensibility.register(
    "attention.latent_sparse.ragged.paged",
    type="gpu",
    api="cuda",
    arch="sm_103a",
)
struct Struct_latent_sparse_attention_ragged_paged_sm103(
    _LatentSparseAttentionRaggedPagedOp
):
    """Registers the op for B300."""

    comptime kernels: MLAGraphKernels = SM100MLAGraph


@extensibility.register("attention.indexer_score.ragged.paged")
struct Struct_indexer_score_ragged_paged:
    """Registers the `attention.indexer_score.ragged.paged` graph op with the graph compiler.

    DeepSeek-V4 lightning-indexer scores: every query row against the live
    entries of one layer of a paged compressed leaf, relu'd, weighted per
    head and summed over the given heads. See `nn.attention.indexer_score`.
    """

    @inline(.always)
    @staticmethod
    def execute[
        num_heads: Int,
    ](
        output: Some[Output & Tensor],
        q: Some[Input & Tensor],
        weights: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        base: Some[Input & Tensor],
        cutoff: Some[Input & Tensor],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        layer: UInt32,
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(q), TileTensorable)
        comptime assert conforms_to(type_of(weights), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(base), TileTensorable)
        comptime assert conforms_to(type_of(cutoff), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert type_of(output).rank == 2
        comptime assert type_of(q).rank == 3
        comptime assert type_of(weights).rank == 2
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(base).rank == 1
        comptime assert type_of(cutoff).rank == 1
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        var collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )
        indexer_score_ragged_paged[num_heads=num_heads](
            output.to_tile_tensor[.float32](),
            q.to_tile_tensor(),
            weights.to_tile_tensor[.float32](),
            input_row_offsets.to_tile_tensor[.uint32](),
            base.to_tile_tensor[.int32](),
            cutoff.to_tile_tensor[.int32](),
            collection.get_key_cache(Int(layer)),
            context,
        )


@extensibility.register("attention.compress_windows.ragged.paged")
struct Struct_compress_windows_ragged_paged:
    """Registers the `attention.compress_windows.ragged.paged` graph op with the graph compiler.

    DeepSeek-V4's compressor over the windows of a ragged chunk, reading the
    positions before the chunk from a paged state leaf. See
    `nn.attention.compress_windows`.
    """

    @inline(.always)
    @staticmethod
    def execute[
        rotate: Bool,
    ](
        output: Some[Output & Tensor],
        kv: Some[Input & Tensor],
        score: Some[Input & Tensor],
        ape: Some[Input & Tensor],
        norm_weight: Some[Input & Tensor],
        freqs: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        window_offsets: Some[Input & Tensor],
        starts: Some[Input & Tensor],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        layer: UInt32,
        epsilon: Float32,
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(kv), TileTensorable)
        comptime assert conforms_to(type_of(score), TileTensorable)
        comptime assert conforms_to(type_of(ape), TileTensorable)
        comptime assert conforms_to(type_of(norm_weight), TileTensorable)
        comptime assert conforms_to(type_of(freqs), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(window_offsets), TileTensorable)
        comptime assert conforms_to(type_of(starts), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime assert type_of(output).rank == 2
        comptime assert type_of(kv).rank == 2
        comptime assert type_of(score).rank == 2
        comptime assert type_of(ape).rank == 2
        comptime assert type_of(norm_weight).rank == 1
        comptime assert type_of(freqs).rank == 3
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(window_offsets).rank == 1
        comptime assert type_of(starts).rank == 1
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(kv_blocks).dtype == DType.float32
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        var collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )
        compress_windows_ragged_paged[rotate=rotate](
            output.to_tile_tensor(),
            kv.to_tile_tensor[.float32](),
            score.to_tile_tensor[.float32](),
            ape.to_tile_tensor[.float32](),
            norm_weight.to_tile_tensor[dtype](),
            freqs.to_tile_tensor[.float32](),
            input_row_offsets.to_tile_tensor[.uint32](),
            window_offsets.to_tile_tensor[.uint32](),
            starts.to_tile_tensor[.int32](),
            collection.get_key_cache(Int(layer)),
            collection.get_value_cache(Int(layer)),
            epsilon,
            context,
        )
