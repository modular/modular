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

"""Registers KV-cache graph ops backed by the `kv_cache` and `nn.kv_cache` kernels."""

import extensibility

# ===-----------------------------------------------------------------------===#
# Kernel imports
# ===-----------------------------------------------------------------------===#

from max.gpu.host import DeviceContext
from layout.tile_tensor import row_major
from kv_cache.types import KVCacheStaticParams, KVCollectionT
from layout import (
    Coord,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from internal_utils.fp8_utils import cast_saturating
from nn.indexer_qk_quantize import (
    INDEX_HEAD_DIM,
    indexer_k_quantize_store,
    indexer_q_ssmax_rope_quantize,
)
from nn.kv_cache import (
    fused_dual_qk_rms_norm_rope_nvfp4_ragged_paged,
    fused_dual_qk_rms_norm_rope_ragged_paged,
    fused_qk_rms_norm_ragged_paged,
    fused_qk_rms_norm_rope_ragged_paged,
    generic_get_paged_cache,
    generic_get_paged_cache_with_scales,
    rms_norm_kv_cache_ragged_paged,
    rms_norm_value_cache_ragged_paged,
)
from nn.kv_cache_gather import kv_cache_gather_rows_ragged
from nn.kv_cache_ragged import (
    generic_kv_cache_radd_dispatch,
    k_matmul_ragged_paged,
    k_matmul_ragged_paged_scale,
    kv_cache_row_offsets_ragged_paged,
    kv_cache_2m_iadd_dispatch,
    kv_cache_store_ragged,
    kv_cache_store_padded,
    kv_cache_store_scales_ragged,
    kv_matmul_ragged_paged,
)
from extensibility import (
    Input,
    InputTensor,
    Output,
    OutputTensor,
)
from extensibility import (
    _FusedInputTensor as FusedInputTensor,
)
from extensibility import (
    _MutableInputTensor as MutableInputTensor,
)
from std.logger import Logger

comptime logger = Logger()

# ===-----------------------------------------------------------------------===#
from .kernels import *


@inline(.always)
def _mut_pointer_tile(
    tile: TileTensor[mut=True, address_space=.GENERIC, ...],
) -> TileTensor[tile.dtype, tile.LayoutType, tile.origin]:
    """Rebuilds `tile` as a mutable view over its pointer.

    The scaled paged KV cache collection holds pointer-backed tiles, so a
    projected argument with another storage policy aborts here at run time.
    """
    return {ptr = tile.unsafe_ptr(), layout = tile.layout}


@inline(.always)
def _imm_pointer_tile(
    tile: TileTensor[address_space=.GENERIC, ...],
) -> TileTensor[tile.dtype, tile.LayoutType, ImmOrigin(tile.origin)]:
    """Rebuilds `tile` as an immutable view over its pointer.

    The scaled paged KV cache collection holds pointer-backed tiles, so a
    projected argument with another storage policy aborts here at run time.
    """
    return TileTensor[tile.dtype, tile.LayoutType, ImmOrigin(tile.origin)](
        ptr=tile.unsafe_ptr().as_imm(), layout=tile.layout
    )


@extensibility.register("kv_cache.store.paged.ragged")
struct Struct_kv_cache_store_paged:
    """Registers the `kv_cache.store.paged.ragged` graph op with the graph compiler.

    Under decode context parallelism (`dcp_size > 1`) every rank of the
    group receives the same lookup table and global `cache_lengths`; a row is
    stored only on the rank that owns its position
    (`ContextParallelLayout[dcp_size, dcp_interleave]`), at that rank's local
    slot. `dcp_rank` is a host scalar after `layer_idx`; outside a group
    (`dcp_size == 1`) the caller passes 0 and the layout never reads it.
    """

    @inline(.always)
    @staticmethod
    def execute[
        key_or_value: Int,
        v_head_dim: Int = -1,
        dcp_size: Int = 1,
        dcp_interleave: Int = 1,
    ](
        inputs: Some[Input & Fused & LoadAccess],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        layer_idx: UInt32,
        dcp_rank: Int32,
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime dtype = type_of(inputs).dtype
        comptime kv_type = type_of(kv_blocks).dtype
        comptime assert type_of(inputs).rank == 3
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1

        comptime if dcp_size > 1:
            if dcp_rank < 0 or Int(dcp_rank) >= dcp_size:
                raise Error(
                    t"dcp_rank {dcp_rank} is outside a group of {dcp_size}"
                )

        # Project the host proxy to its access-enabled view so the closure can
        # load on the GPU lane.
        comptime assert conforms_to(type_of(inputs), UnsafeFusedView)
        var inputs_view = unsafe_fused_view(inputs)
        comptime assert conforms_to(type_of(inputs_view), Loadable)

        var paged_kv_collection = generic_get_paged_cache[
            v_head_size=v_head_dim
        ](
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )
        comptime KVCacheT = paged_kv_collection.CacheType
        var cache: KVCacheT

        # A split cache's value lands after the key in the same row.
        comptime kv_params = KVCacheT.kv_params
        comptime head_dim_offset = (
            0 if key_or_value == 0 else kv_params.v_col_base()
        )
        comptime if kv_params.is_split():
            var width = (
                kv_params.k_head_size() if key_or_value
                == 0 else kv_params.v_head_size
            )
            var row_width = Int(inputs.shape()[2].value())
            if row_width != width:
                raise Error(
                    t"a split KV cache stores {width}-wide rows here, got"
                    t" {row_width}"
                )

        comptime if key_or_value == 0:
            cache = paged_kv_collection.get_key_cache(Int(layer_idx))
        else:
            cache = paged_kv_collection.get_value_cache(Int(layer_idx))

        @inline(.always)
        def input_fn[
            width: Int, alignment: Int
        ](idx: Coord) {var inputs_view} -> SIMD[kv_type, width]:
            # The value dtype is the producer compute dtype (bf16), which need
            # not match the cache: an FP8 cache saturates on the store rather
            # than emitting NaN for out-of-range values.
            return cast_saturating[kv_type](
                rebind[SIMD[dtype, width]](
                    inputs_view.load[width=width, element_alignment=alignment](
                        idx
                    )
                )
            )

        kv_cache_store_ragged[
            head_dim_offset=head_dim_offset,
            dcp_size=dcp_size,
            dcp_interleave=dcp_interleave,
        ](
            cache,
            inputs.shape(),
            input_row_offsets.to_tile_tensor[.uint32](),
            input_fn,
            context,
            Int(dcp_rank),
        )


@extensibility.register("kv_cache.gather_rows.ragged.paged")
struct Struct_kv_cache_gather_rows_ragged_paged:
    """Registers the `kv_cache.gather_rows.ragged.paged` graph op with the graph compiler.

    Copies the key or value rows a per-row slot list names out of one layer
    of a paged cache, one head per slot. See `nn.kv_cache_gather`.
    """

    @inline(.always)
    @staticmethod
    def execute[
        key_or_value: Int,
    ](
        output: Some[Output & Tensor],
        slots: Some[Input & Tensor],
        row_offsets: Some[Input & Tensor],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        layer_idx: UInt32,
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(slots), TileTensorable)
        comptime assert conforms_to(type_of(row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime kv_type = type_of(kv_blocks).dtype
        comptime assert type_of(output).rank == 3
        comptime assert type_of(slots).rank == 2
        comptime assert type_of(row_offsets).rank == 1
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        var paged_kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )
        comptime KVCacheT = paged_kv_collection.CacheType
        var cache: KVCacheT

        comptime if key_or_value == 0:
            cache = paged_kv_collection.get_key_cache(Int(layer_idx))
        else:
            cache = paged_kv_collection.get_value_cache(Int(layer_idx))

        kv_cache_gather_rows_ragged(
            output.to_tile_tensor[kv_type](),
            slots.to_tile_tensor[.int32](),
            row_offsets.to_tile_tensor[.uint32](),
            cache,
            context,
        )


@extensibility.register("kv_cache.store_k_scales.paged.ragged")
struct Struct_kv_cache_store_k_scales_paged:
    """Registers the `kv_cache.store_k_scales.paged.ragged` graph op with the graph compiler.

    Under decode context parallelism (`dcp_size > 1`) every rank of the
    group receives the same lookup table and global `cache_lengths`; a row is
    stored only on the rank that owns its position
    (`ContextParallelLayout[dcp_size, dcp_interleave]`), at that rank's local
    slot. `dcp_rank` is a host scalar after `layer_idx`; outside a group
    (`dcp_size == 1`) the caller passes 0 and the layout never reads it.
    """

    @inline(.always)
    @staticmethod
    def execute[
        quantization_granularity: Int,
        dcp_size: Int = 1,
        dcp_interleave: Int = 1,
    ](
        input_k_scales: Some[Input & Fused & LoadAccess],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        k_scales_blocks: Some[MutableInput & Tensor],
        scales_page_stride: Some[Input & Tensor],
        # Resolves a request's scale pages. Pass `kv_lookup_table` itself when
        # the scales share the values' block-id space; pass a distinct table
        # when they are paged independently.
        k_scales_lookup_table: Some[Input & Tensor],
        layer_idx: UInt32,
        dcp_rank: Int32,
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert conforms_to(type_of(k_scales_blocks), TileTensorable)
        comptime assert conforms_to(type_of(scales_page_stride), TileTensorable)
        comptime assert conforms_to(
            type_of(k_scales_lookup_table), TileTensorable
        )
        comptime cache_dtype = type_of(kv_blocks).dtype
        comptime scale_dtype = type_of(k_scales_blocks).dtype
        comptime assert type_of(input_k_scales).rank == 3
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(input_k_scales).dtype == scale_dtype
        comptime assert type_of(k_scales_blocks).rank == 6
        comptime assert type_of(scales_page_stride).rank == 1
        comptime assert type_of(k_scales_lookup_table).rank == 2

        comptime if dcp_size > 1:
            if dcp_rank < 0 or Int(dcp_rank) >= dcp_size:
                raise Error(
                    t"dcp_rank {dcp_rank} is outside a group of {dcp_size}"
                )

        # Project the host proxy to its access-enabled view so the closure can
        # load on the GPU lane.
        comptime assert conforms_to(type_of(input_k_scales), UnsafeFusedView)
        var input_k_scales_view = unsafe_fused_view(input_k_scales)
        comptime assert conforms_to(type_of(input_k_scales_view), Loadable)

        comptime page_size = type_of(kv_blocks)._shape_types[3].static_value
        comptime head_dim = type_of(kv_blocks)._shape_types[5].static_value
        comptime num_heads = type_of(kv_blocks)._shape_types[4].static_value
        comptime is_mla = type_of(kv_blocks)._shape_types[1].static_value == 1
        comptime kv_params = KVCacheStaticParams(num_heads, head_dim, is_mla)

        # The builder takes pointer-backed tiles; a non-pointer storage policy
        # aborts here at run time.
        var k_collection = generic_get_paged_cache_with_scales[
            cache_dtype,
            scale_dtype,
            kv_params,
            page_size,
            quantization_granularity,
        ](
            _mut_pointer_tile(kv_blocks.to_tile_tensor()),
            _imm_pointer_tile(page_stride.to_tile_tensor[.int64]()),
            _imm_pointer_tile(cache_lengths.to_tile_tensor[.uint32]()),
            _imm_pointer_tile(kv_lookup_table.to_tile_tensor[.uint32]()),
            _imm_pointer_tile(max_prompt_length.to_tile_tensor[.uint32]()),
            _imm_pointer_tile(max_cache_length.to_tile_tensor[.uint32]()),
            _mut_pointer_tile(k_scales_blocks.to_tile_tensor()),
            _imm_pointer_tile(scales_page_stride.to_tile_tensor[.int64]()),
            _imm_pointer_tile(k_scales_lookup_table.to_tile_tensor[.uint32]()),
        )

        @inline(.always)
        def input_fn[
            width: Int, alignment: Int
        ](idx: Coord) {var input_k_scales_view} -> SIMD[scale_dtype, width]:
            return rebind[SIMD[scale_dtype, width]](
                input_k_scales_view.load[
                    width=width, element_alignment=alignment
                ](idx)
            )

        kv_cache_store_scales_ragged[
            dcp_size=dcp_size, dcp_interleave=dcp_interleave
        ](
            k_collection.get_key_cache(Int(layer_idx)),
            input_k_scales.shape(),
            input_row_offsets.to_tile_tensor[.uint32](),
            input_fn,
            context,
            Int(dcp_rank),
        )


@extensibility.register("indexer.k.quantize_store.fp8.paged.ragged")
struct Struct_indexer_k_quantize_store_fp8_paged:
    """Registers `indexer.k.quantize_store.fp8.paged.ragged`.

    Normalizes the 128 channels of each raw index key out of the merged
    projection's row (they start at `key_col_offset`) with a biased LayerNorm,
    rotates their leading `rope_dim` columns with interleaved RoPE,
    FP8-quantizes them with one E8M0 scale each, and stores the keys and their
    float32 scales in the paged indexer cache -- the chain
    `reduce.layer_norm`, `rope_ragged`,
    `quantize_dynamic_scaled_float8`, `kv_cache.store.paged.ragged` and
    `kv_cache.store_k_scales.paged.ragged` run, in one launch.
    """

    @inline(.always)
    @staticmethod
    def execute[
        quantization_granularity: Int,
        rope_dim: Int,
        key_col_offset: Int,
    ](
        k: InputTensor[dtype=.bfloat16, rank=2, ...],
        kv_blocks: MutableInputTensor[dtype=.float8_e4m3fn, rank=6, ...],
        page_stride: InputTensor[dtype=.int64, rank=1, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        k_scales_blocks: MutableInputTensor[dtype=.float32, rank=6, ...],
        scales_page_stride: InputTensor[dtype=.int64, rank=1, ...],
        k_scales_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        layer_idx: UInt32,
        gamma: InputTensor[rank=1, ...],
        beta: InputTensor[rank=1, ...],
        epsilon: Float32,
        freqs_cis: InputTensor[dtype=.bfloat16, rank=2, ...],
        start_pos: InputTensor[dtype=.uint32, rank=1, ...],
        context: DeviceContext,
    ) raises:
        comptime assert context.target.is_gpu(), "only valid on GPUs"
        # The kernel quantizes a whole 128-channel key with one scale and
        # stores it at block 0, so a finer granularity would drop scales.
        comptime assert (
            quantization_granularity == INDEX_HEAD_DIM
        ), "an indexer key takes one scale per 128 channels"
        comptime page_size = Int(kv_blocks.static_spec.shape_tuple[3])
        comptime head_dim = Int(kv_blocks.static_spec.shape_tuple[5])
        comptime num_heads = Int(kv_blocks.static_spec.shape_tuple[4])
        comptime is_mla = Int(kv_blocks.static_spec.shape_tuple[1]) == 1
        comptime kv_params = KVCacheStaticParams(num_heads, head_dim, is_mla)

        var k_collection = generic_get_paged_cache_with_scales[
            DType.float8_e4m3fn,
            DType.float32,
            kv_params,
            page_size,
            quantization_granularity,
        ](
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor(),
            cache_lengths.to_tile_tensor(),
            kv_lookup_table.to_tile_tensor(),
            max_prompt_length.to_tile_tensor(),
            max_cache_length.to_tile_tensor(),
            k_scales_blocks.to_tile_tensor(),
            scales_page_stride.to_tile_tensor(),
            k_scales_lookup_table.to_tile_tensor(),
        )
        indexer_k_quantize_store[
            rope_dim=rope_dim,
            key_col_offset=key_col_offset,
            gamma_dtype=type_of(gamma).dtype,
            beta_dtype=type_of(beta).dtype,
        ](
            k_collection.get_key_cache(Int(layer_idx)),
            k.to_tile_tensor(),
            gamma.to_tile_tensor(),
            beta.to_tile_tensor(),
            epsilon,
            freqs_cis.to_tile_tensor(),
            input_row_offsets.to_tile_tensor(),
            start_pos.to_tile_tensor(),
            context,
        )


@extensibility.register("indexer.q.ssmax_rope_quantize.fp8")
struct Struct_indexer_q_ssmax_rope_quantize_fp8:
    """Registers `indexer.q.ssmax_rope_quantize.fp8`.

    Applies SSMax to a `[rows, heads, 128]` index query, rotates the leading
    `rope_dim` columns of each head, and FP8-quantizes each head with one
    E8M0 scale, in one launch. See `nn.indexer_qk_quantize`.
    """

    @inline(.always)
    @staticmethod
    def execute[
        rope_dim: Int,
    ](
        q_out: OutputTensor[dtype=.float8_e4m3fn, rank=3, ...],
        scale_out: OutputTensor[dtype=.float8_e8m0fnu, rank=2, ...],
        q: InputTensor[dtype=.bfloat16, rank=3, ...],
        positions: InputTensor[dtype=.uint32, rank=1, ...],
        ssmax_s: InputTensor[dtype=.bfloat16, rank=1, ...],
        ssmax_b: InputTensor[dtype=.bfloat16, rank=1, ...],
        freqs_cis: InputTensor[dtype=.bfloat16, rank=2, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        start_pos: InputTensor[dtype=.uint32, rank=1, ...],
        context: DeviceContext,
    ) raises:
        comptime assert context.target.is_gpu(), "only valid on GPUs"
        indexer_q_ssmax_rope_quantize[rope_dim](
            q_out.to_tile_tensor(),
            scale_out.to_tile_tensor(),
            q.to_tile_tensor(),
            positions.to_tile_tensor(),
            ssmax_s.to_tile_tensor(),
            ssmax_b.to_tile_tensor(),
            freqs_cis.to_tile_tensor(),
            input_row_offsets.to_tile_tensor(),
            start_pos.to_tile_tensor(),
            context,
        )


@extensibility.register("kv_cache.store.paged.padded")
struct Struct_kv_cache_store_padded:
    """Registers the `kv_cache.store.paged.padded` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute[
        key_or_value: Int
    ](
        inputs: Some[Input & Fused & LoadAccess],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        valid_lengths: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        layer_idx: UInt32,
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(valid_lengths), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime dtype = type_of(kv_blocks).dtype
        comptime assert type_of(inputs).rank == 4
        comptime assert type_of(inputs).dtype == dtype
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(valid_lengths).rank == 1
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1

        # Project the host proxy to its access-enabled view so the closure can
        # load on the GPU lane.
        comptime assert conforms_to(type_of(inputs), UnsafeFusedView)
        var inputs_view = unsafe_fused_view(inputs)
        comptime assert conforms_to(type_of(inputs_view), Loadable)

        var paged_kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )
        comptime KVCacheT = paged_kv_collection.CacheType
        var cache: KVCacheT

        comptime if key_or_value == 0:
            cache = paged_kv_collection.get_key_cache(Int(layer_idx))
        else:
            cache = paged_kv_collection.get_value_cache(Int(layer_idx))

        @inline(.always)
        def input_fn[
            width: Int, alignment: Int
        ](idx: Coord) {var inputs_view} -> SIMD[dtype, width]:
            comptime assert idx.rank == 4
            return rebind[SIMD[dtype, width]](
                inputs_view.load[width=width, element_alignment=alignment](idx)
            )

        kv_cache_store_padded(
            cache,
            inputs.shape(),
            valid_lengths.to_tile_tensor[.uint32](),
            input_fn,
            context,
        )


@extensibility.register("kv_cache.rms_norm.ragged.paged")
struct Struct_rms_norm_kv_cache_ragged_paged:
    """Registers the `kv_cache.rms_norm.ragged.paged` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute[
        multiply_before_cast: Bool,
        per_head_norm: Bool,
        //,
        epsilon: Float32,
        weight_offset: Float32,
    ](
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        gamma: Some[Input & Tensor],
        layer_idx: UInt32,
        total_seq_len: UInt32,
        input_row_offsets: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert conforms_to(type_of(gamma), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(gamma).rank == 1
        comptime assert type_of(input_row_offsets).rank == 1
        var kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )
        rms_norm_kv_cache_ragged_paged[
            multiply_before_cast=multiply_before_cast,
            per_head_norm=per_head_norm,
            epsilon=epsilon,
            weight_offset=weight_offset,
        ](
            kv_collection,
            gamma.to_tile_tensor(),
            layer_idx,
            total_seq_len,
            input_row_offsets.to_tile_tensor[.uint32](),
            context,
        )


@extensibility.register("kv_cache.fused_qk_rms_norm.ragged.paged")
struct Struct_fused_qk_rms_norm_ragged_paged:
    """Registers the `kv_cache.fused_qk_rms_norm.ragged.paged` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute[
        dtype: DType,
        multiply_before_cast: Bool,
        //,
    ](
        q_output: Some[Output & Tensor],
        q_proj: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        q_gamma: Some[Input & Tensor],
        k_gamma: Some[Input & Tensor],
        epsilon: Float32,
        layer_idx: UInt32,
        weight_offset: Scalar[dtype=dtype],
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(q_output), TileTensorable)
        comptime assert conforms_to(type_of(q_proj), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert conforms_to(type_of(q_gamma), TileTensorable)
        comptime assert conforms_to(type_of(k_gamma), TileTensorable)
        comptime assert type_of(q_output).rank == 3
        comptime assert type_of(q_proj).rank == 3
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(q_gamma).rank == 1
        comptime assert type_of(k_gamma).rank == 1
        var kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )
        fused_qk_rms_norm_ragged_paged[
            multiply_before_cast=multiply_before_cast,
        ](
            q_proj.to_tile_tensor[dtype](),
            kv_collection,
            q_gamma.to_tile_tensor[dtype](),
            k_gamma.to_tile_tensor[dtype](),
            epsilon,
            weight_offset,
            layer_idx,
            input_row_offsets.to_tile_tensor[.uint32](),
            q_output.to_tile_tensor[dtype](),
            context,
        )


@extensibility.register("kv_cache.fused_qk_rms_norm_rope.ragged.paged")
struct Struct_fused_qk_rms_norm_rope_ragged_paged[interleaved: Bool]:
    """Registers the `kv_cache.fused_qk_rms_norm_rope.ragged.paged` graph op with the graph compiler.

    Parameters:
        interleaved: When true, RoPE rotates adjacent element pairs; when
            false, rotates pairs separated by half the head dimension.
    """

    @inline(.always)
    @staticmethod
    def execute[
        dtype: DType,
        multiply_before_cast: Bool,
        //,
    ](
        q_output: Some[Output & Tensor],
        q_proj: Some[Input & Fused & LoadAccess],
        input_row_offsets: Some[Input & Tensor],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        q_gamma: Some[Input & Tensor],
        k_gamma: Some[Input & Tensor],
        freqs_cis: Some[Input & Tensor],
        epsilon: Float32,
        layer_idx: UInt32,
        weight_offset: Scalar[dtype=dtype],
        context: DeviceContext,
    ) raises:
        # `q_proj` is a fused input, so any elementwise/view producer
        # feeding it (e.g. a `slice` + `reshape` that carves Q out of a combined
        # `[Q | IndexQ]` matmul output) is folded into the Q read lambda by the
        # graph compiler's input-prologue fusion. When there is no producer to
        # fuse, the fused load degrades to a plain strided load, so the
        # default rank-3 Q-projection path is unchanged.
        comptime assert conforms_to(type_of(q_output), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert conforms_to(type_of(q_gamma), TileTensorable)
        comptime assert conforms_to(type_of(k_gamma), TileTensorable)
        comptime assert conforms_to(type_of(freqs_cis), TileTensorable)
        comptime assert type_of(q_output).rank == 3
        comptime assert type_of(q_proj).rank == 3
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(q_gamma).rank == 1
        comptime assert type_of(k_gamma).rank == 1
        comptime assert type_of(freqs_cis).rank == 2
        comptime q_out_dtype = type_of(q_output).dtype
        comptime freq_dtype = type_of(freqs_cis).dtype
        comptime assert type_of(q_proj).dtype == dtype
        var kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )

        comptime assert conforms_to(type_of(q_proj), UnsafeFusedView)
        var q_view = unsafe_fused_view(q_proj)
        comptime assert conforms_to(type_of(q_view), Loadable)

        @inline(.always)
        def q_input_fn[
            width: Int, alignment: Int
        ](token: Int, head: Int, col: Int) {var q_view} -> SIMD[dtype, width]:
            return rebind[SIMD[dtype, width]](
                q_view.load[width=width, element_alignment=alignment](
                    Coord(token, head, col)
                )
            )

        fused_qk_rms_norm_rope_ragged_paged[
            multiply_before_cast=multiply_before_cast,
            interleaved=Self.interleaved,
        ](
            kv_collection,
            q_input_fn,
            q_gamma.to_tile_tensor[dtype](),
            k_gamma.to_tile_tensor[dtype](),
            freqs_cis.to_tile_tensor[freq_dtype](),
            epsilon,
            weight_offset,
            layer_idx,
            input_row_offsets.to_tile_tensor[.uint32](),
            q_output.to_tile_tensor[q_out_dtype](),
            context,
        )


@extensibility.register("kv_cache.fused_qk_rms_norm_rope.ragged.paged.dual")
struct Struct_fused_qk_rms_norm_rope_ragged_paged_dual[interleaved: Bool]:
    """Registers `kv_cache.fused_qk_rms_norm_rope.ragged.paged.dual` with the graph compiler.

    Fuses the two back-to-back `kv_cache.fused_qk_rms_norm_rope.ragged.paged` launches
    that a MiniMax-M3 sparse layer fires (main GQA attention + lightning
    indexer) into one kernel. Both bands read (disjoint) slices of the same
    combined QKV+IndexQ matmul output via their own fused Q read lambda; each band writes its Q to a separate DPS output and its K back into
    its own paged cache.

    Parameters:
        interleaved: When true, RoPE rotates adjacent element pairs; when
            false, rotates pairs separated by half the head dimension. Shared by
            both bands (they use the same rope table).
    """

    @inline(.always)
    @staticmethod
    def execute[
        dtype: DType,
        multiply_before_cast: Bool,
        //,
    ](
        q_main_output: Some[Output & Tensor],
        q_index_output: Some[Output & Tensor],
        q_main_proj: Some[Input & Fused & LoadAccess],
        q_index_proj: Some[Input & Fused & LoadAccess],
        input_row_offsets: Some[Input & Tensor],
        main_kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        main_cache_lengths: Some[Input & Tensor],
        main_kv_lookup_table: Some[Input & Tensor],
        main_max_prompt_length: Some[Input & Tensor],
        main_max_cache_length: Some[Input & Tensor],
        index_kv_blocks: Some[MutableInput & Tensor],
        index_page_stride: Some[Input & Tensor],
        index_cache_lengths: Some[Input & Tensor],
        index_kv_lookup_table: Some[Input & Tensor],
        index_max_prompt_length: Some[Input & Tensor],
        index_max_cache_length: Some[Input & Tensor],
        q_main_gamma: Some[Input & Tensor],
        k_main_gamma: Some[Input & Tensor],
        q_index_gamma: Some[Input & Tensor],
        k_index_gamma: Some[Input & Tensor],
        freqs_cis: Some[Input & Tensor],
        main_epsilon: Float32,
        index_epsilon: Float32,
        layer_idx: UInt32,
        weight_offset: Scalar[dtype=dtype],
        context: DeviceContext,
    ) raises:
        # `q_main_proj` / `q_index_proj` are fused inputs, so the
        # slice+reshape that carves each band's Q out of the combined
        # `[Q | IndexQ]` matmul output folds into that band's read lambda.
        comptime assert conforms_to(type_of(q_main_output), TileTensorable)
        comptime assert conforms_to(type_of(q_index_output), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(main_kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(main_cache_lengths), TileTensorable)
        comptime assert conforms_to(
            type_of(main_kv_lookup_table), TileTensorable
        )
        comptime assert conforms_to(
            type_of(main_max_prompt_length), TileTensorable
        )
        comptime assert conforms_to(
            type_of(main_max_cache_length), TileTensorable
        )
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
        comptime assert conforms_to(type_of(q_main_gamma), TileTensorable)
        comptime assert conforms_to(type_of(k_main_gamma), TileTensorable)
        comptime assert conforms_to(type_of(q_index_gamma), TileTensorable)
        comptime assert conforms_to(type_of(k_index_gamma), TileTensorable)
        comptime assert conforms_to(type_of(freqs_cis), TileTensorable)
        comptime assert type_of(q_main_output).rank == 3
        comptime assert type_of(q_index_output).rank == 3
        comptime assert type_of(q_main_proj).rank == 3
        comptime assert type_of(q_index_proj).rank == 3
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(main_kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(main_cache_lengths).rank == 1
        comptime assert type_of(main_kv_lookup_table).rank == 2
        comptime assert type_of(main_max_prompt_length).rank == 1
        comptime assert type_of(main_max_cache_length).rank == 1
        comptime assert type_of(index_kv_blocks).rank == 6
        comptime assert type_of(index_page_stride).rank == 1
        comptime assert type_of(index_cache_lengths).rank == 1
        comptime assert type_of(index_kv_lookup_table).rank == 2
        comptime assert type_of(index_max_prompt_length).rank == 1
        comptime assert type_of(index_max_cache_length).rank == 1
        comptime assert type_of(q_main_gamma).rank == 1
        comptime assert type_of(k_main_gamma).rank == 1
        comptime assert type_of(q_index_gamma).rank == 1
        comptime assert type_of(k_index_gamma).rank == 1
        comptime assert type_of(freqs_cis).rank == 2
        comptime q_main_out_dtype = type_of(q_main_output).dtype
        comptime q_index_out_dtype = type_of(q_index_output).dtype
        comptime freq_dtype = type_of(freqs_cis).dtype
        comptime assert type_of(q_main_proj).dtype == dtype
        comptime assert type_of(q_index_proj).dtype == dtype
        var main_kv_collection = generic_get_paged_cache(
            main_kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            main_cache_lengths.to_tile_tensor[.uint32](),
            main_kv_lookup_table.to_tile_tensor[.uint32](),
            main_max_prompt_length.to_tile_tensor[.uint32](),
            main_max_cache_length.to_tile_tensor[.uint32](),
        )
        var index_kv_collection = generic_get_paged_cache(
            index_kv_blocks.to_tile_tensor(),
            index_page_stride.to_tile_tensor[.int64](),
            index_cache_lengths.to_tile_tensor[.uint32](),
            index_kv_lookup_table.to_tile_tensor[.uint32](),
            index_max_prompt_length.to_tile_tensor[.uint32](),
            index_max_cache_length.to_tile_tensor[.uint32](),
        )

        comptime assert conforms_to(type_of(q_main_proj), UnsafeFusedView)
        var q_main_view = unsafe_fused_view(q_main_proj)
        comptime assert conforms_to(type_of(q_main_view), Loadable)
        comptime assert conforms_to(type_of(q_index_proj), UnsafeFusedView)
        var q_index_view = unsafe_fused_view(q_index_proj)
        comptime assert conforms_to(type_of(q_index_view), Loadable)

        @inline(.always)
        def main_q_input_fn[
            width: Int, alignment: Int
        ](token: Int, head: Int, col: Int) {var q_main_view} -> SIMD[
            dtype, width
        ]:
            return rebind[SIMD[dtype, width]](
                q_main_view.load[width=width, element_alignment=alignment](
                    Coord(token, head, col)
                )
            )

        @inline(.always)
        def index_q_input_fn[
            width: Int, alignment: Int
        ](token: Int, head: Int, col: Int) {var q_index_view} -> SIMD[
            dtype, width
        ]:
            return rebind[SIMD[dtype, width]](
                q_index_view.load[width=width, element_alignment=alignment](
                    Coord(token, head, col)
                )
            )

        fused_dual_qk_rms_norm_rope_ragged_paged[
            multiply_before_cast=multiply_before_cast,
            interleaved=Self.interleaved,
        ](
            main_kv_collection,
            index_kv_collection,
            q_main_gamma.to_tile_tensor[dtype](),
            k_main_gamma.to_tile_tensor[dtype](),
            q_index_gamma.to_tile_tensor[dtype](),
            k_index_gamma.to_tile_tensor[dtype](),
            freqs_cis.to_tile_tensor[freq_dtype](),
            main_epsilon,
            index_epsilon,
            weight_offset,
            layer_idx,
            input_row_offsets.to_tile_tensor[.uint32](),
            main_q_input_fn,
            index_q_input_fn,
            q_main_output.to_tile_tensor[q_main_out_dtype](),
            q_index_output.to_tile_tensor[q_index_out_dtype](),
            context,
        )


@inline(.always)
def load_staged_row[
    width: Int, alignment: Int
](proj: Some[Loadable], token: Int, head: Int, col: Int) -> SIMD[
    type_of(proj).dtype, width
]:
    """Reads `width` elements of one staged `[token, head, col]` row.

    Parameters:
        width: Number of elements to read.
        alignment: Alignment of the read, in elements.

    Args:
        proj: The staged projection's access-enabled fused view.
        token: Ragged token index.
        head: Head index.
        col: First column to read.

    Returns:
        The `width` elements starting at `[token, head, col]`.
    """
    return proj.load[width=width, element_alignment=alignment](
        Coord(token, head, col)
    )


@inline(.always)
def fused_dual_qk_rms_norm_rope_nvfp4_from_graph[
    dtype: DType,
    main_collection_t: KVCollectionT,
    index_collection_t: KVCollectionT,
    //,
    multiply_before_cast: Bool,
    interleaved: Bool,
](
    q_main_output: Some[Output & Tensor],
    q_index_output: Some[Output & Tensor],
    q_main_proj: Some[Input & Fused & LoadAccess],
    k_main_proj: Some[Input & Fused & LoadAccess],
    v_main_proj: Some[Input & Fused & LoadAccess],
    q_index_proj: Some[Input & Fused & LoadAccess],
    k_index_proj: Some[Input & Fused & LoadAccess],
    input_row_offsets: Some[Input & Tensor],
    main_kv_collection: main_collection_t,
    index_kv_collection: index_collection_t,
    q_main_gamma: Some[Input & Tensor],
    k_main_gamma: Some[Input & Tensor],
    q_index_gamma: Some[Input & Tensor],
    k_index_gamma: Some[Input & Tensor],
    freqs_cis: Some[Input & Tensor],
    main_epsilon: Float32,
    index_epsilon: Float32,
    layer_idx: UInt32,
    weight_offset: Scalar[dtype=dtype],
    context: DeviceContext,
) raises:
    """Runs `fused_dual_qk_rms_norm_rope_nvfp4_ragged_paged` from graph
    operands, once the ops have built their cache collections.

    Parameters:
        dtype: Staging and gamma dtype.
        main_collection_t: Main cache collection type.
        index_collection_t: Index-K cache collection type.
        multiply_before_cast: RMSNorm rounding order.
        interleaved: Whether RoPE rotates adjacent pairs.

    Args:
        q_main_output: Normed and roped main Q.
        q_index_output: Normed and roped IndexQ.
        q_main_proj: Staged main Q.
        k_main_proj: Staged main K.
        v_main_proj: Staged main V.
        q_index_proj: Staged IndexQ.
        k_index_proj: Staged IndexK.
        input_row_offsets: Ragged row offsets.
        main_kv_collection: Main cache.
        index_kv_collection: Index-K cache.
        q_main_gamma: Main Q RMSNorm weight.
        k_main_gamma: Main K RMSNorm weight.
        q_index_gamma: IndexQ RMSNorm weight.
        k_index_gamma: IndexK RMSNorm weight.
        freqs_cis: RoPE table.
        main_epsilon: Main RMSNorm epsilon.
        index_epsilon: Indexer RMSNorm epsilon.
        layer_idx: Layer index.
        weight_offset: RMSNorm weight offset.
        context: Device context.
    """
    comptime assert conforms_to(type_of(q_main_output), TileTensorable)
    comptime assert conforms_to(type_of(q_index_output), TileTensorable)
    comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
    comptime assert conforms_to(type_of(q_main_gamma), TileTensorable)
    comptime assert conforms_to(type_of(k_main_gamma), TileTensorable)
    comptime assert conforms_to(type_of(q_index_gamma), TileTensorable)
    comptime assert conforms_to(type_of(k_index_gamma), TileTensorable)
    comptime assert conforms_to(type_of(freqs_cis), TileTensorable)
    comptime assert type_of(q_main_output).rank == 3
    comptime assert type_of(q_index_output).rank == 3
    comptime assert type_of(q_main_proj).rank == 3
    comptime assert type_of(k_main_proj).rank == 3
    comptime assert type_of(v_main_proj).rank == 3
    comptime assert type_of(q_index_proj).rank == 3
    comptime assert type_of(k_index_proj).rank == 3
    comptime assert type_of(input_row_offsets).rank == 1
    comptime assert type_of(q_main_gamma).rank == 1
    comptime assert type_of(k_main_gamma).rank == 1
    comptime assert type_of(q_index_gamma).rank == 1
    comptime assert type_of(k_index_gamma).rank == 1
    comptime assert type_of(freqs_cis).rank == 2
    comptime q_main_out_dtype = type_of(q_main_output).dtype
    comptime q_index_out_dtype = type_of(q_index_output).dtype
    comptime freq_dtype = type_of(freqs_cis).dtype
    comptime assert type_of(q_main_proj).dtype == dtype
    comptime assert conforms_to(type_of(q_main_proj), UnsafeFusedView)
    var q_main_view = unsafe_fused_view(q_main_proj)
    comptime assert conforms_to(type_of(q_main_view), Loadable)
    comptime assert type_of(k_main_proj).dtype == dtype
    comptime assert conforms_to(type_of(k_main_proj), UnsafeFusedView)
    var k_main_view = unsafe_fused_view(k_main_proj)
    comptime assert conforms_to(type_of(k_main_view), Loadable)
    comptime assert type_of(v_main_proj).dtype == dtype
    comptime assert conforms_to(type_of(v_main_proj), UnsafeFusedView)
    var v_main_view = unsafe_fused_view(v_main_proj)
    comptime assert conforms_to(type_of(v_main_view), Loadable)
    comptime assert type_of(q_index_proj).dtype == dtype
    comptime assert conforms_to(type_of(q_index_proj), UnsafeFusedView)
    var q_index_view = unsafe_fused_view(q_index_proj)
    comptime assert conforms_to(type_of(q_index_view), Loadable)
    comptime assert type_of(k_index_proj).dtype == dtype
    comptime assert conforms_to(type_of(k_index_proj), UnsafeFusedView)
    var k_index_view = unsafe_fused_view(k_index_proj)
    comptime assert conforms_to(type_of(k_index_view), Loadable)

    @inline(.always)
    def q_main_fn[
        width: Int, alignment: Int
    ](token: Int, head: Int, col: Int) {var q_main_view} -> SIMD[dtype, width]:
        return rebind[SIMD[dtype, width]](
            load_staged_row[width, alignment](q_main_view, token, head, col)
        )

    @inline(.always)
    def k_main_fn[
        width: Int, alignment: Int
    ](token: Int, head: Int, col: Int) {var k_main_view} -> SIMD[dtype, width]:
        return rebind[SIMD[dtype, width]](
            load_staged_row[width, alignment](k_main_view, token, head, col)
        )

    @inline(.always)
    def v_main_fn[
        width: Int, alignment: Int
    ](token: Int, head: Int, col: Int) {var v_main_view} -> SIMD[dtype, width]:
        return rebind[SIMD[dtype, width]](
            load_staged_row[width, alignment](v_main_view, token, head, col)
        )

    @inline(.always)
    def q_index_fn[
        width: Int, alignment: Int
    ](token: Int, head: Int, col: Int) {var q_index_view} -> SIMD[dtype, width]:
        return rebind[SIMD[dtype, width]](
            load_staged_row[width, alignment](q_index_view, token, head, col)
        )

    @inline(.always)
    def k_index_fn[
        width: Int, alignment: Int
    ](token: Int, head: Int, col: Int) {var k_index_view} -> SIMD[dtype, width]:
        return rebind[SIMD[dtype, width]](
            load_staged_row[width, alignment](k_index_view, token, head, col)
        )

    fused_dual_qk_rms_norm_rope_nvfp4_ragged_paged[
        multiply_before_cast=multiply_before_cast,
        interleaved=interleaved,
    ](
        main_kv_collection,
        index_kv_collection,
        q_main_gamma.to_tile_tensor[dtype](),
        k_main_gamma.to_tile_tensor[dtype](),
        q_index_gamma.to_tile_tensor[dtype](),
        k_index_gamma.to_tile_tensor[dtype](),
        freqs_cis.to_tile_tensor[freq_dtype](),
        main_epsilon,
        index_epsilon,
        weight_offset,
        layer_idx,
        input_row_offsets.to_tile_tensor[.uint32](),
        q_main_fn,
        k_main_fn,
        v_main_fn,
        q_index_fn,
        k_index_fn,
        q_main_output.to_tile_tensor[q_main_out_dtype](),
        q_index_output.to_tile_tensor[q_index_out_dtype](),
        context,
    )


@extensibility.register(
    "kv_cache.fused_qk_rms_norm_rope.ragged.paged.dual.nvfp4"
)
struct Struct_fused_qk_rms_norm_rope_ragged_paged_dual_nvfp4[interleaved: Bool]:
    """Registers `kv_cache.fused_qk_rms_norm_rope.ragged.paged.dual.nvfp4` with the graph compiler.

    The dual MiniMax-M3 RMSNorm+RoPE op for a main cache with scales, which
    the projection leaves unwritten. Q, K, V, IndexQ and IndexK all come from
    staging through fused read lambdas. K and V are written to the
    main cache (NVFP4 when its dtype is `uint8`), IndexK to the index cache.
    The main cache's operands are in the order
    `flatten_without_attention_dispatch_metadata` emits for quantized params.

    Parameters:
        interleaved: When true, RoPE rotates adjacent element pairs; when
            false, rotates pairs separated by half the head dimension.
    """

    @inline(.always)
    @staticmethod
    def execute[
        dtype: DType,
        multiply_before_cast: Bool,
        //,
    ](
        q_main_output: Some[Output & Tensor],
        q_index_output: Some[Output & Tensor],
        q_main_proj: Some[Input & Fused & LoadAccess],
        k_main_proj: Some[Input & Fused & LoadAccess],
        v_main_proj: Some[Input & Fused & LoadAccess],
        q_index_proj: Some[Input & Fused & LoadAccess],
        k_index_proj: Some[Input & Fused & LoadAccess],
        input_row_offsets: Some[Input & Tensor],
        main_kv_blocks: Some[MutableInput & Tensor],
        main_page_stride: Some[Input & Tensor],
        main_cache_lengths: Some[Input & Tensor],
        main_kv_lookup_table: Some[Input & Tensor],
        main_max_prompt_length: Some[Input & Tensor],
        main_max_cache_length: Some[Input & Tensor],
        main_kv_scales: Some[MutableInput & Tensor],
        main_scales_page_stride: Some[Input & Tensor],
        main_scales_lookup_table: Some[Input & Tensor],
        index_kv_blocks: Some[MutableInput & Tensor],
        index_page_stride: Some[Input & Tensor],
        index_cache_lengths: Some[Input & Tensor],
        index_kv_lookup_table: Some[Input & Tensor],
        index_max_prompt_length: Some[Input & Tensor],
        index_max_cache_length: Some[Input & Tensor],
        q_main_gamma: Some[Input & Tensor],
        k_main_gamma: Some[Input & Tensor],
        q_index_gamma: Some[Input & Tensor],
        k_index_gamma: Some[Input & Tensor],
        freqs_cis: Some[Input & Tensor],
        main_epsilon: Float32,
        index_epsilon: Float32,
        layer_idx: UInt32,
        weight_offset: Scalar[dtype=dtype],
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(main_kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(main_page_stride), TileTensorable)
        comptime assert conforms_to(type_of(main_cache_lengths), TileTensorable)
        comptime assert conforms_to(
            type_of(main_kv_lookup_table), TileTensorable
        )
        comptime assert conforms_to(
            type_of(main_max_prompt_length), TileTensorable
        )
        comptime assert conforms_to(
            type_of(main_max_cache_length), TileTensorable
        )
        comptime assert conforms_to(type_of(main_kv_scales), TileTensorable)
        comptime assert conforms_to(
            type_of(main_scales_page_stride), TileTensorable
        )
        comptime assert conforms_to(
            type_of(main_scales_lookup_table), TileTensorable
        )
        comptime assert type_of(main_kv_blocks).rank == 6
        comptime assert type_of(main_page_stride).rank == 1
        comptime assert type_of(main_cache_lengths).rank == 1
        comptime assert type_of(main_kv_lookup_table).rank == 2
        comptime assert type_of(main_max_prompt_length).rank == 1
        comptime assert type_of(main_max_cache_length).rank == 1
        comptime assert type_of(main_kv_scales).rank == 6
        comptime assert type_of(main_scales_page_stride).rank == 1
        comptime assert type_of(main_scales_lookup_table).rank == 2
        var main_kv_collection = generic_get_paged_cache_with_scales(
            main_kv_blocks.to_tile_tensor(),
            main_page_stride.to_tile_tensor[.int64](),
            main_cache_lengths.to_tile_tensor[.uint32](),
            main_kv_lookup_table.to_tile_tensor[.uint32](),
            main_max_prompt_length.to_tile_tensor[.uint32](),
            main_max_cache_length.to_tile_tensor[.uint32](),
            main_kv_scales.to_tile_tensor(),
            main_scales_page_stride.to_tile_tensor[.int64](),
            main_scales_lookup_table.to_tile_tensor[.uint32](),
        )
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
        comptime assert type_of(index_kv_blocks).rank == 6
        comptime assert type_of(index_page_stride).rank == 1
        comptime assert type_of(index_cache_lengths).rank == 1
        comptime assert type_of(index_kv_lookup_table).rank == 2
        comptime assert type_of(index_max_prompt_length).rank == 1
        comptime assert type_of(index_max_cache_length).rank == 1
        var index_kv_collection = generic_get_paged_cache(
            index_kv_blocks.to_tile_tensor(),
            index_page_stride.to_tile_tensor[.int64](),
            index_cache_lengths.to_tile_tensor[.uint32](),
            index_kv_lookup_table.to_tile_tensor[.uint32](),
            index_max_prompt_length.to_tile_tensor[.uint32](),
            index_max_cache_length.to_tile_tensor[.uint32](),
        )

        fused_dual_qk_rms_norm_rope_nvfp4_from_graph[
            multiply_before_cast=multiply_before_cast,
            interleaved=Self.interleaved,
        ](
            q_main_output,
            q_index_output,
            q_main_proj,
            k_main_proj,
            v_main_proj,
            q_index_proj,
            k_index_proj,
            input_row_offsets,
            main_kv_collection,
            index_kv_collection,
            q_main_gamma,
            k_main_gamma,
            q_index_gamma,
            k_index_gamma,
            freqs_cis,
            main_epsilon,
            index_epsilon,
            layer_idx,
            weight_offset,
            context,
        )


@extensibility.register(
    "kv_cache.fused_qk_rms_norm_rope.ragged.paged.dual.nvfp4.nvfp4_index"
)
struct Struct_fused_qk_rms_norm_rope_ragged_paged_dual_nvfp4_nvfp4_index[
    interleaved: Bool
]:
    """Registers `kv_cache.fused_qk_rms_norm_rope.ragged.paged.dual.nvfp4.nvfp4_index` with the graph compiler.

    `kv_cache.fused_qk_rms_norm_rope.ragged.paged.dual.nvfp4` for an index-K cache
    that has scales too: IndexK is written as NVFP4 like the main K.

    The dual MiniMax-M3 RMSNorm+RoPE op for a main cache with scales, which
    the projection leaves unwritten. Q, K, V, IndexQ and IndexK all come from
    staging through fused read lambdas. K and V are written to the
    main cache (NVFP4 when its dtype is `uint8`), IndexK to the index cache.
    The main cache's operands are in the order
    `flatten_without_attention_dispatch_metadata` emits for quantized params.

    Parameters:
        interleaved: When true, RoPE rotates adjacent element pairs; when
            false, rotates pairs separated by half the head dimension.
    """

    @inline(.always)
    @staticmethod
    def execute[
        dtype: DType,
        multiply_before_cast: Bool,
        //,
    ](
        q_main_output: Some[Output & Tensor],
        q_index_output: Some[Output & Tensor],
        q_main_proj: Some[Input & Fused & LoadAccess],
        k_main_proj: Some[Input & Fused & LoadAccess],
        v_main_proj: Some[Input & Fused & LoadAccess],
        q_index_proj: Some[Input & Fused & LoadAccess],
        k_index_proj: Some[Input & Fused & LoadAccess],
        input_row_offsets: Some[Input & Tensor],
        main_kv_blocks: Some[MutableInput & Tensor],
        main_page_stride: Some[Input & Tensor],
        main_cache_lengths: Some[Input & Tensor],
        main_kv_lookup_table: Some[Input & Tensor],
        main_max_prompt_length: Some[Input & Tensor],
        main_max_cache_length: Some[Input & Tensor],
        main_kv_scales: Some[MutableInput & Tensor],
        main_scales_page_stride: Some[Input & Tensor],
        main_scales_lookup_table: Some[Input & Tensor],
        index_kv_blocks: Some[MutableInput & Tensor],
        index_page_stride: Some[Input & Tensor],
        index_cache_lengths: Some[Input & Tensor],
        index_kv_lookup_table: Some[Input & Tensor],
        index_max_prompt_length: Some[Input & Tensor],
        index_max_cache_length: Some[Input & Tensor],
        index_kv_scales: Some[MutableInput & Tensor],
        index_scales_page_stride: Some[Input & Tensor],
        index_scales_lookup_table: Some[Input & Tensor],
        q_main_gamma: Some[Input & Tensor],
        k_main_gamma: Some[Input & Tensor],
        q_index_gamma: Some[Input & Tensor],
        k_index_gamma: Some[Input & Tensor],
        freqs_cis: Some[Input & Tensor],
        main_epsilon: Float32,
        index_epsilon: Float32,
        layer_idx: UInt32,
        weight_offset: Scalar[dtype=dtype],
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(main_kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(main_page_stride), TileTensorable)
        comptime assert conforms_to(type_of(main_cache_lengths), TileTensorable)
        comptime assert conforms_to(
            type_of(main_kv_lookup_table), TileTensorable
        )
        comptime assert conforms_to(
            type_of(main_max_prompt_length), TileTensorable
        )
        comptime assert conforms_to(
            type_of(main_max_cache_length), TileTensorable
        )
        comptime assert conforms_to(type_of(main_kv_scales), TileTensorable)
        comptime assert conforms_to(
            type_of(main_scales_page_stride), TileTensorable
        )
        comptime assert conforms_to(
            type_of(main_scales_lookup_table), TileTensorable
        )
        comptime assert type_of(main_kv_blocks).rank == 6
        comptime assert type_of(main_page_stride).rank == 1
        comptime assert type_of(main_cache_lengths).rank == 1
        comptime assert type_of(main_kv_lookup_table).rank == 2
        comptime assert type_of(main_max_prompt_length).rank == 1
        comptime assert type_of(main_max_cache_length).rank == 1
        comptime assert type_of(main_kv_scales).rank == 6
        comptime assert type_of(main_scales_page_stride).rank == 1
        comptime assert type_of(main_scales_lookup_table).rank == 2
        var main_kv_collection = generic_get_paged_cache_with_scales(
            main_kv_blocks.to_tile_tensor(),
            main_page_stride.to_tile_tensor[.int64](),
            main_cache_lengths.to_tile_tensor[.uint32](),
            main_kv_lookup_table.to_tile_tensor[.uint32](),
            main_max_prompt_length.to_tile_tensor[.uint32](),
            main_max_cache_length.to_tile_tensor[.uint32](),
            main_kv_scales.to_tile_tensor(),
            main_scales_page_stride.to_tile_tensor[.int64](),
            main_scales_lookup_table.to_tile_tensor[.uint32](),
        )
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
        comptime assert conforms_to(type_of(index_kv_scales), TileTensorable)
        comptime assert conforms_to(
            type_of(index_scales_page_stride), TileTensorable
        )
        comptime assert conforms_to(
            type_of(index_scales_lookup_table), TileTensorable
        )
        comptime assert type_of(index_kv_blocks).rank == 6
        comptime assert type_of(index_page_stride).rank == 1
        comptime assert type_of(index_cache_lengths).rank == 1
        comptime assert type_of(index_kv_lookup_table).rank == 2
        comptime assert type_of(index_max_prompt_length).rank == 1
        comptime assert type_of(index_max_cache_length).rank == 1
        comptime assert type_of(index_kv_scales).rank == 6
        comptime assert type_of(index_scales_page_stride).rank == 1
        comptime assert type_of(index_scales_lookup_table).rank == 2
        var index_kv_collection = generic_get_paged_cache_with_scales(
            index_kv_blocks.to_tile_tensor(),
            index_page_stride.to_tile_tensor[.int64](),
            index_cache_lengths.to_tile_tensor[.uint32](),
            index_kv_lookup_table.to_tile_tensor[.uint32](),
            index_max_prompt_length.to_tile_tensor[.uint32](),
            index_max_cache_length.to_tile_tensor[.uint32](),
            index_kv_scales.to_tile_tensor(),
            index_scales_page_stride.to_tile_tensor[.int64](),
            index_scales_lookup_table.to_tile_tensor[.uint32](),
        )

        fused_dual_qk_rms_norm_rope_nvfp4_from_graph[
            multiply_before_cast=multiply_before_cast,
            interleaved=Self.interleaved,
        ](
            q_main_output,
            q_index_output,
            q_main_proj,
            k_main_proj,
            v_main_proj,
            q_index_proj,
            k_index_proj,
            input_row_offsets,
            main_kv_collection,
            index_kv_collection,
            q_main_gamma,
            k_main_gamma,
            q_index_gamma,
            k_index_gamma,
            freqs_cis,
            main_epsilon,
            index_epsilon,
            layer_idx,
            weight_offset,
            context,
        )


@extensibility.register("kv_cache.rms_norm_value_cache.ragged.paged")
struct Struct_rms_norm_value_cache_ragged_paged:
    """Registers the `kv_cache.rms_norm_value_cache.ragged.paged` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute[
        multiply_before_cast: Bool,
        per_head_norm: Bool,
        //,
        epsilon: Float32,
        weight_offset: Float32,
    ](
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        gamma: Some[Input & Tensor],
        layer_idx: UInt32,
        total_seq_len: UInt32,
        input_row_offsets: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert conforms_to(type_of(gamma), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(gamma).rank == 1
        comptime assert type_of(input_row_offsets).rank == 1
        var kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )
        rms_norm_value_cache_ragged_paged[
            multiply_before_cast=multiply_before_cast,
            per_head_norm=per_head_norm,
            epsilon=epsilon,
            weight_offset=weight_offset,
        ](
            kv_collection,
            gamma.to_tile_tensor(),
            layer_idx,
            total_seq_len,
            input_row_offsets.to_tile_tensor[.uint32](),
            context,
        )


@extensibility.register("kv_cache.kv_matmul.ragged.paged")
struct Struct_kv_matmul_ragged_paged:
    """Registers the `kv_cache.kv_matmul.ragged.paged` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute(
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
        comptime assert conforms_to(type_of(hidden_state), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(weight), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime dtype = type_of(kv_blocks).dtype
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
        kv_matmul_ragged_paged(
            hidden_state.to_tile_tensor[dtype](),
            input_row_offsets.to_tile_tensor[.uint32](),
            weight.to_tile_tensor[dtype](),
            kv_collection,
            layer_idx,
            ctx,
        )


@extensibility.register("kv_cache.k_matmul.ragged.paged")
struct Struct_k_matmul_ragged_paged:
    """Registers the `kv_cache.k_matmul.ragged.paged` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute(
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
        comptime assert conforms_to(type_of(hidden_state), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(weight), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime dtype = type_of(kv_blocks).dtype
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
        k_matmul_ragged_paged(
            hidden_state.to_tile_tensor[dtype](),
            input_row_offsets.to_tile_tensor[.uint32](),
            weight.to_tile_tensor[dtype](),
            kv_collection,
            layer_idx,
            ctx,
        )


@extensibility.register("kv_cache.k_matmul.ragged.paged.scale")
struct Struct_k_matmul_ragged_paged_scale:
    """Registers the `kv_cache.k_matmul.ragged.paged.scale` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute[
        m_scale_granularity: Int,
        n_scale_granularity: Int,
        k_scale_granularity: Int,
    ](
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
        comptime scale_dtype = type_of(input_scale).dtype
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
        var kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )
        k_matmul_ragged_paged_scale[
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
            weight_scale.to_tile_tensor[scale_dtype](),
            kv_collection,
            layer_idx,
            ctx,
        )


@extensibility.register("kv_cache.row_offsets.ragged.paged")
struct Struct_kv_cache_row_offsets_ragged_paged:
    """Registers the `kv_cache.row_offsets.ragged.paged` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute(
        cache_row_offsets: Some[Output & Tensor],
        input_row_offsets: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(cache_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        kv_cache_row_offsets_ragged_paged(
            cache_row_offsets.to_tile_tensor[.uint32](),
            input_row_offsets.to_tile_tensor[.uint32](),
            cache_lengths.to_tile_tensor[.uint32](),
            context,
        )


@extensibility.register("kv_cache.ragged.paged.radd")
struct Struct_kv_cache_ragged_paged_radd:
    """Registers the `kv_cache.ragged.paged.radd` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute(
        a: Some[Input & Tensor],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        batch_offset: UInt32,
        layer_idx: UInt32,
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(a), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime dtype = type_of(kv_blocks).dtype
        comptime assert type_of(a).rank == 2
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(input_row_offsets).rank == 1
        var kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )

        generic_kv_cache_radd_dispatch(
            a.to_tile_tensor[dtype](),
            kv_collection,
            input_row_offsets.to_tile_tensor[.uint32](),
            batch_offset,
            layer_idx,
            context,
        )


@extensibility.register("kv_cache.ragged.paged.2m_iadd")
struct Struct_kv_cache_ragged_paged_2m_iadd:
    """Registers the `kv_cache.ragged.paged.2m_iadd` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute(
        kv: Some[Input & Tensor],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        lora_end_idx: Some[Input & Tensor],
        batch_seq_len: Some[Input & Tensor],
        layer_idx: UInt32,
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(kv), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(lora_end_idx), TileTensorable)
        comptime assert conforms_to(type_of(batch_seq_len), TileTensorable)
        comptime dtype = type_of(kv_blocks).dtype
        comptime assert type_of(kv).rank == 2
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(lora_end_idx).rank == 1
        comptime assert type_of(batch_seq_len).rank == 1
        var kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )

        var kv_tensor = kv.to_tile_tensor[dtype]()

        if kv_tensor.dim[0]() == 0:
            return

        kv_cache_2m_iadd_dispatch(
            kv_tensor,
            kv_collection,
            input_row_offsets.to_tile_tensor[.uint32](),
            lora_end_idx.to_tile_tensor[.int64](),
            batch_seq_len.to_tile_tensor[.int64](),
            layer_idx,
            context,
        )
