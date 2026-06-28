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

from std.sys.info import simd_width_of, _current_target
import extensibility as compiler

# ===-----------------------------------------------------------------------===#
# Kernel imports
# ===-----------------------------------------------------------------------===#
from std.algorithm import elementwise

from std.gpu.host import DeviceContext, get_gpu_target
from layout.tile_tensor import row_major
from std.gpu.host.info import is_gpu
from kv_cache.types import KVCacheStaticParams
from layout import (
    Coord,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    UNKNOWN_VALUE,
    row_major,
)
from nn._ragged_utils import get_batch_from_row_offsets
from nn.kv_cache import (
    copy_kv_pages_d2h,
    fused_qk_rms_norm_ragged_paged,
    fused_qk_rms_norm_rope_ragged_paged,
    generic_get_paged_cache,
    generic_get_paged_cache_with_scales,
    rms_norm_kv_cache_ragged_paged,
    rms_norm_value_cache_ragged_paged,
)
from nn.kv_cache_ragged import (
    generic_kv_cache_radd_dispatch,
    k_matmul_ragged_paged,
    k_matmul_ragged_paged_scale,
    kv_cache_row_offsets_ragged_paged,
    kv_cache_2m_iadd_dispatch,
    kv_cache_store_ragged,
    kv_cache_store_padded,
    kv_matmul_ragged_paged,
)
from extensibility import InputTensor, OutputTensor
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


@compiler.register("mo.kv_cache.store.paged.ragged")
struct Struct_kv_cache_store_paged:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType, target: StaticString, key_or_value: Int
    ](
        inputs: FusedInputTensor[dtype=dtype, rank=3, ...],
        kv_blocks: MutableInputTensor[dtype=dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=DType.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=DType.uint32, rank=2, ...],
        input_row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        max_prompt_length: InputTensor[dtype=DType.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=DType.uint32, rank=1, ...],
        layer_idx: UInt32,
        context: DeviceContext,
    ) capturing raises:
        var paged_kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )
        comptime KVCacheT = paged_kv_collection.CacheType
        var cache: KVCacheT

        comptime if key_or_value == 0:
            cache = paged_kv_collection.get_key_cache(Int(layer_idx))
        else:
            cache = paged_kv_collection.get_value_cache(Int(layer_idx))

        @parameter
        @always_inline
        def input_fn[
            width: Int, alignment: Int
        ](idx: IndexList[3]) capturing -> SIMD[dtype, width]:
            return inputs._lambda_load[
                width=width, element_alignment=alignment
            ](
                idx,
            )

        kv_cache_store_ragged[input_fn=input_fn, target=target](
            cache,
            inputs.shape(),
            input_row_offsets.to_layout_tensor(),
            context,
        )


@compiler.register("mo.kv_cache.store_k_scales.paged.ragged")
struct Struct_kv_cache_store_k_scales_paged:
    @always_inline
    @staticmethod
    def execute[
        cache_dtype: DType,
        scale_dtype: DType,
        target: StaticString,
        //,
        quantization_granularity: Int,
    ](
        input_k_scales: FusedInputTensor[dtype=scale_dtype, rank=3, ...],
        kv_blocks: MutableInputTensor[dtype=cache_dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=DType.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=DType.uint32, rank=2, ...],
        input_row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        max_prompt_length: InputTensor[dtype=DType.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=DType.uint32, rank=1, ...],
        k_scales_blocks: MutableInputTensor[dtype=scale_dtype, rank=6, ...],
        layer_idx: UInt32,
        context: DeviceContext,
    ) capturing raises:
        comptime page_size = Int(kv_blocks.static_spec.shape_tuple[3])
        comptime head_dim = Int(kv_blocks.static_spec.shape_tuple[5])
        comptime num_heads = Int(kv_blocks.static_spec.shape_tuple[4])
        comptime is_mla = Int(kv_blocks.static_spec.shape_tuple[1]) == 1
        comptime kv_params = KVCacheStaticParams(num_heads, head_dim, is_mla)

        var k_collection = generic_get_paged_cache_with_scales[
            cache_dtype,
            scale_dtype,
            kv_params,
            page_size,
            quantization_granularity,
        ](
            LayoutTensor[cache_dtype, Layout.row_major[6](), MutAnyOrigin](
                kv_blocks.to_layout_tensor().ptr,
                RuntimeLayout[Layout.row_major[6]()].row_major(
                    kv_blocks.to_layout_tensor().runtime_layout.shape.value
                ),
            ),
            LayoutTensor[DType.uint32, Layout(UNKNOWN_VALUE), ImmutAnyOrigin](
                cache_lengths.to_layout_tensor().ptr,
                RuntimeLayout[Layout(UNKNOWN_VALUE)](
                    cache_lengths.to_layout_tensor().runtime_layout.shape.value,
                    cache_lengths.to_layout_tensor().runtime_layout.stride.value,
                ),
            ),
            LayoutTensor[DType.uint32, Layout.row_major[2](), ImmutAnyOrigin](
                kv_lookup_table.to_layout_tensor().ptr,
                RuntimeLayout[Layout.row_major[2]()].row_major(
                    kv_lookup_table.to_layout_tensor().runtime_layout.shape.value
                ),
            ),
            LayoutTensor[DType.uint32, Layout.row_major[1](), ImmutAnyOrigin](
                max_prompt_length.to_layout_tensor().ptr,
                RuntimeLayout[Layout.row_major[1]()].row_major(
                    max_prompt_length.to_layout_tensor().runtime_layout.shape.value
                ),
            ),
            LayoutTensor[DType.uint32, Layout.row_major[1](), ImmutAnyOrigin](
                max_cache_length.to_layout_tensor().ptr,
                RuntimeLayout[Layout.row_major[1]()].row_major(
                    max_cache_length.to_layout_tensor().runtime_layout.shape.value
                ),
            ),
            LayoutTensor[scale_dtype, Layout.row_major[6](), MutAnyOrigin](
                k_scales_blocks.to_layout_tensor().ptr,
                RuntimeLayout[Layout.row_major[6]()].row_major(
                    k_scales_blocks.to_layout_tensor().runtime_layout.shape.value
                ),
            ),
        )

        var k_cache = k_collection.get_key_cache(Int(layer_idx))

        var input_row_offsets_tt = input_row_offsets.to_tile_tensor[
            DType.int64
        ]()

        def write_scale_to_cache[
            width: Int,
            alignment: Int = 1,
        ](idx: Coord) {
            var k_cache,
            var input_row_offsets_tt,
            var input_row_offsets,
            var input_k_scales,
        }:
            var loaded_val = input_k_scales._lambda_load[
                width=width, element_alignment=alignment
            ](
                IndexList[3](
                    Int(idx[0].value()),
                    Int(idx[1].value()),
                    Int(idx[2].value()),
                ),
            )
            var batch_idx = get_batch_from_row_offsets(
                input_row_offsets_tt, Int(idx[0].value())
            )
            var token_idx = Int(
                UInt32(idx[0].value()) - input_row_offsets[batch_idx]
            )
            var h_idx = Int(idx[1].value())
            var hd_idx = Int(idx[2].value())
            var cache_length = k_cache.cache_length(batch_idx)
            var cache_token_idx = token_idx + cache_length
            k_cache.store_scale(
                batch_idx,
                h_idx,
                cache_token_idx,
                hd_idx,
                loaded_val,
            )

        comptime compile_target = get_gpu_target() if is_gpu[
            target
        ]() else _current_target()
        comptime simd_width = simd_width_of[
            scale_dtype, target=compile_target
        ]()

        elementwise[simd_width=simd_width, target=target](
            write_scale_to_cache, input_k_scales.shape_coord(), context
        )


@compiler.register("mo.kv_cache.store.paged.padded")
struct Struct_kv_cache_store_padded:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType, target: StaticString, key_or_value: Int
    ](
        inputs: FusedInputTensor[dtype=dtype, rank=4, ...],
        kv_blocks: MutableInputTensor[dtype=dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=DType.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=DType.uint32, rank=2, ...],
        valid_lengths: InputTensor[dtype=DType.uint32, rank=1, ...],
        max_prompt_length: InputTensor[dtype=DType.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=DType.uint32, rank=1, ...],
        layer_idx: UInt32,
        context: DeviceContext,
    ) capturing raises:
        var paged_kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )
        comptime KVCacheT = paged_kv_collection.CacheType
        var cache: KVCacheT

        comptime if key_or_value == 0:
            cache = paged_kv_collection.get_key_cache(Int(layer_idx))
        else:
            cache = paged_kv_collection.get_value_cache(Int(layer_idx))

        @parameter
        @always_inline
        def input_fn[
            width: Int, alignment: Int
        ](idx: IndexList[4]) capturing -> SIMD[dtype, width]:
            return inputs._lambda_load[
                width=width, element_alignment=alignment
            ](
                idx,
            )

        kv_cache_store_padded[input_fn=input_fn, target=target](
            cache,
            inputs.shape(),
            valid_lengths.to_layout_tensor(),
            context,
        )


@compiler.register("mo.rms_norm_kv_cache.ragged.paged")
struct Struct_rms_norm_kv_cache_ragged_paged:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        multiply_before_cast: Bool,
        per_head_norm: Bool,
        cache_dtype: DType,
        //,
        target: StaticString,
    ](
        kv_blocks: MutableInputTensor[dtype=cache_dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=DType.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=DType.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=DType.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=DType.uint32, rank=1, ...],
        gamma: InputTensor[dtype=dtype, rank=1, ...],
        epsilon: Scalar[dtype],
        layer_idx: UInt32,
        total_seq_len: UInt32,
        input_row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        weight_offset: Scalar[dtype=dtype],
        context: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )
        rms_norm_kv_cache_ragged_paged[
            target=target,
            multiply_before_cast=multiply_before_cast,
            per_head_norm=per_head_norm,
        ](
            kv_collection,
            gamma.to_tile_tensor[DType.int64](),
            epsilon,
            weight_offset,
            layer_idx,
            total_seq_len,
            input_row_offsets.to_tile_tensor[DType.int64](),
            context,
        )


@compiler.register("mo.fused_qk_rms_norm.ragged.paged")
struct Struct_fused_qk_rms_norm_ragged_paged:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        multiply_before_cast: Bool,
        cache_dtype: DType,
        //,
        target: StaticString,
    ](
        q_output: OutputTensor[dtype=dtype, rank=3, ...],
        q_proj: InputTensor[dtype=dtype, rank=3, ...],
        input_row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        kv_blocks: MutableInputTensor[dtype=cache_dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=DType.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=DType.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=DType.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=DType.uint32, rank=1, ...],
        q_gamma: InputTensor[dtype=dtype, rank=1, ...],
        k_gamma: InputTensor[dtype=dtype, rank=1, ...],
        epsilon: Scalar[dtype],
        layer_idx: UInt32,
        weight_offset: Scalar[dtype=dtype],
        context: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )
        fused_qk_rms_norm_ragged_paged[
            target=target,
            multiply_before_cast=multiply_before_cast,
        ](
            q_proj.to_tile_tensor[DType.int64](),
            kv_collection,
            q_gamma.to_tile_tensor[DType.int64](),
            k_gamma.to_tile_tensor[DType.int64](),
            epsilon,
            weight_offset,
            layer_idx,
            input_row_offsets.to_tile_tensor[DType.int64](),
            q_output.to_tile_tensor[DType.int64](),
            context,
        )


@compiler.register("mo.fused_qk_rms_norm_rope.ragged.paged")
struct Struct_fused_qk_rms_norm_rope_ragged_paged[interleaved: Bool]:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        freq_dtype: DType,
        multiply_before_cast: Bool,
        cache_dtype: DType,
        //,
        target: StaticString,
    ](
        q_output: OutputTensor[dtype=dtype, rank=3, ...],
        q_proj: InputTensor[dtype=dtype, rank=3, ...],
        input_row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        kv_blocks: MutableInputTensor[dtype=cache_dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=DType.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=DType.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=DType.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=DType.uint32, rank=1, ...],
        q_gamma: InputTensor[dtype=dtype, rank=1, ...],
        k_gamma: InputTensor[dtype=dtype, rank=1, ...],
        freqs_cis: InputTensor[dtype=freq_dtype, rank=2, ...],
        epsilon: Scalar[dtype],
        layer_idx: UInt32,
        weight_offset: Scalar[dtype=dtype],
        context: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )
        fused_qk_rms_norm_rope_ragged_paged[
            target=target,
            multiply_before_cast=multiply_before_cast,
            interleaved=Self.interleaved,
        ](
            q_proj.to_tile_tensor[DType.int64](),
            kv_collection,
            q_gamma.to_tile_tensor[DType.int64](),
            k_gamma.to_tile_tensor[DType.int64](),
            freqs_cis.to_tile_tensor[DType.int64](),
            epsilon,
            weight_offset,
            layer_idx,
            input_row_offsets.to_tile_tensor[DType.int64](),
            q_output.to_tile_tensor[DType.int64](),
            context,
        )


@compiler.register("mo.rms_norm_value_cache.ragged.paged")
struct Struct_rms_norm_value_cache_ragged_paged:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        multiply_before_cast: Bool,
        per_head_norm: Bool,
        cache_dtype: DType,
        //,
        target: StaticString,
    ](
        kv_blocks: MutableInputTensor[dtype=cache_dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=DType.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=DType.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=DType.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=DType.uint32, rank=1, ...],
        gamma: InputTensor[dtype=dtype, rank=1, ...],
        epsilon: Scalar[dtype],
        layer_idx: UInt32,
        total_seq_len: UInt32,
        input_row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        weight_offset: Scalar[dtype=dtype],
        context: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )
        rms_norm_value_cache_ragged_paged[
            target=target,
            multiply_before_cast=multiply_before_cast,
            per_head_norm=per_head_norm,
        ](
            kv_collection,
            gamma.to_tile_tensor[DType.int64](),
            epsilon,
            weight_offset,
            layer_idx,
            total_seq_len,
            input_row_offsets.to_tile_tensor[DType.int64](),
            context,
        )


@compiler.register("mo.print_kv_cache.paged")
struct Struct_print_kv_cache_paged:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        //,
        target: StaticString,
    ](
        valid_lengths: InputTensor[dtype=DType.uint32, rank=1, ...],
        kv_blocks: MutableInputTensor[dtype=dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=DType.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=DType.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=DType.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=DType.uint32, rank=1, ...],
        layer_idx: UInt32,
        is_print_compact: InputTensor[dtype=DType.bool, rank=1, ...],
        context: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )
        print_kv_cache_paged_generic_kernel_api[target](
            valid_lengths,
            kv_collection,
            layer_idx,
            is_print_compact,
            context,
        )


@compiler.register("mo.kv_matmul.ragged.paged")
struct Struct_kv_matmul_ragged_paged:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        //,
        target: StaticString,
    ](
        hidden_state: InputTensor[dtype=dtype, rank=2, ...],
        input_row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        weight: InputTensor[dtype=dtype, rank=2, ...],
        kv_blocks: MutableInputTensor[dtype=dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=DType.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=DType.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=DType.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=DType.uint32, rank=1, ...],
        layer_idx: UInt32,
        ctx: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )
        kv_matmul_ragged_paged[target=target](
            hidden_state.to_layout_tensor(),
            input_row_offsets.to_layout_tensor(),
            weight.to_layout_tensor(),
            kv_collection,
            layer_idx,
            ctx,
        )


@compiler.register("mo.k_matmul.ragged.paged")
struct Struct_k_matmul_ragged_paged:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        //,
        target: StaticString,
    ](
        hidden_state: InputTensor[dtype=dtype, rank=2, ...],
        input_row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        weight: InputTensor[dtype=dtype, rank=2, ...],
        kv_blocks: MutableInputTensor[dtype=dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=DType.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=DType.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=DType.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=DType.uint32, rank=1, ...],
        layer_idx: UInt32,
        ctx: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )
        k_matmul_ragged_paged[target=target](
            hidden_state.to_layout_tensor(),
            input_row_offsets.to_layout_tensor(),
            weight.to_layout_tensor(),
            kv_collection,
            layer_idx,
            ctx,
        )


@compiler.register("mo.k_matmul.ragged.paged.scale")
struct Struct_k_matmul_ragged_paged_scale:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        scale_dtype: DType,
        kv_cache_t: DType,
        //,
        m_scale_granularity: Int,
        n_scale_granularity: Int,
        k_scale_granularity: Int,
        target: StaticString,
    ](
        hidden_state: InputTensor[dtype=dtype, rank=2, ...],
        input_row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        weight: InputTensor[dtype=dtype, rank=2, ...],
        input_scale: InputTensor[dtype=scale_dtype, rank=2, ...],
        weight_scale: InputTensor[dtype=scale_dtype, rank=2, ...],
        kv_blocks: MutableInputTensor[dtype=kv_cache_t, rank=6, ...],
        cache_lengths: InputTensor[dtype=DType.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=DType.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=DType.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=DType.uint32, rank=1, ...],
        layer_idx: UInt32,
        ctx: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )
        k_matmul_ragged_paged_scale[
            target=target,
            scales_granularity_mnk=IndexList[3](
                m_scale_granularity, n_scale_granularity, k_scale_granularity
            ),
        ](
            hidden_state.to_layout_tensor(),
            input_row_offsets.to_layout_tensor(),
            weight.to_layout_tensor(),
            input_scale.to_layout_tensor(),
            weight_scale.to_layout_tensor(),
            kv_collection,
            layer_idx,
            ctx,
        )


@compiler.register("mo.kv_cache.row_offsets.ragged.paged")
struct Struct_kv_cache_row_offsets_ragged_paged:
    @always_inline
    @staticmethod
    def execute[
        target: StaticString,
    ](
        cache_row_offsets: OutputTensor[dtype=DType.uint32, rank=1, ...],
        input_row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        cache_lengths: InputTensor[dtype=DType.uint32, rank=1, ...],
        context: DeviceContext,
    ) raises:
        kv_cache_row_offsets_ragged_paged[target=target](
            cache_row_offsets.to_tile_tensor[DType.int64](),
            input_row_offsets.to_tile_tensor[DType.int64](),
            cache_lengths.to_tile_tensor[DType.int64](),
            context,
        )


@compiler.register("mo.kv_cache.ragged.paged.radd")
struct Struct_kv_cache_ragged_paged_radd:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        //,
        target: StaticString,
    ](
        a: InputTensor[dtype=dtype, rank=2, ...],
        kv_blocks: MutableInputTensor[dtype=dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=DType.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=DType.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=DType.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=DType.uint32, rank=1, ...],
        input_row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        batch_offset: UInt32,
        layer_idx: UInt32,
        context: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )

        generic_kv_cache_radd_dispatch[target=target,](
            a.to_layout_tensor(),
            kv_collection,
            input_row_offsets.to_layout_tensor(),
            batch_offset,
            layer_idx,
            context,
        )


@compiler.register("mo.kv_cache.ragged.paged.2m_iadd")
struct Struct_kv_cache_ragged_paged_2m_iadd:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        //,
        target: StaticString,
    ](
        kv: InputTensor[dtype=dtype, rank=2, ...],
        kv_blocks: MutableInputTensor[dtype=dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=DType.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=DType.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=DType.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=DType.uint32, rank=1, ...],
        input_row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        lora_end_idx: InputTensor[dtype=DType.int64, rank=1, ...],
        batch_seq_len: InputTensor[dtype=DType.int64, rank=1, ...],
        layer_idx: UInt32,
        context: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )

        var kv_layout_tensor = kv.to_layout_tensor()

        if kv_layout_tensor.shape[0]() == 0:
            return

        kv_cache_2m_iadd_dispatch[target=target,](
            kv_layout_tensor,
            kv_collection,
            input_row_offsets.to_layout_tensor(),
            lora_end_idx.to_layout_tensor(),
            batch_seq_len.to_layout_tensor(),
            layer_idx,
            context,
        )


@compiler.register("mo.kv_cache.copy_pages_d2h")
struct KVCacheCopyPagesD2H:
    @staticmethod
    def execute[
        dtype: DType,
        //,
        target: StaticString,
    ](
        device_kv_blocks: MutableInputTensor[dtype=dtype, rank=6, ...],
        host_kv_blocks: MutableInputTensor[dtype=dtype, rank=6, ...],
        src_page_ids: InputTensor[dtype=DType.int64, rank=1, ...],
        dst_page_ids: InputTensor[dtype=DType.int64, rank=1, ...],
        layer_idx: UInt32,
        ctx: DeviceContext,
    ) raises:
        var gpu_ctx = ctx

        copy_kv_pages_d2h(
            LayoutTensor[dtype, Layout.row_major[6](), MutAnyOrigin](
                device_kv_blocks.to_layout_tensor().ptr,
                RuntimeLayout[Layout.row_major[6]()].row_major(
                    device_kv_blocks.to_layout_tensor().runtime_layout.shape.value
                ),
            ),
            LayoutTensor[dtype, Layout.row_major[6](), MutAnyOrigin](
                host_kv_blocks.to_layout_tensor().ptr,
                RuntimeLayout[Layout.row_major[6]()].row_major(
                    host_kv_blocks.to_layout_tensor().runtime_layout.shape.value
                ),
            ),
            LayoutTensor[DType.int64, Layout.row_major[1](), MutAnyOrigin](
                src_page_ids.to_layout_tensor().ptr,
                RuntimeLayout[Layout.row_major[1]()].row_major(
                    src_page_ids.to_layout_tensor().runtime_layout.shape.value
                ),
            ),
            LayoutTensor[DType.int64, Layout.row_major[1](), MutAnyOrigin](
                dst_page_ids.to_layout_tensor().ptr,
                RuntimeLayout[Layout.row_major[1]()].row_major(
                    dst_page_ids.to_layout_tensor().runtime_layout.shape.value
                ),
            ),
            Int(layer_idx),
            gpu_ctx,
        )
