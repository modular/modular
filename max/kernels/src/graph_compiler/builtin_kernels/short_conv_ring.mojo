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
"""Registers the ring-state short convolution graph ops; see
`nn.short_conv_ring`."""

import extensibility
from extensibility import Input, MutableInput, Output, Tensor, TileTensorable
from max.gpu.host import DeviceContext

from nn.kv_cache import generic_get_paged_cache
from nn.short_conv_ring import (
    fused_qk_rms_norm_short_conv_ragged_paged,
    short_conv_ring,
)


@extensibility.register("mo.short_conv_ring")
struct Struct_short_conv_ring:
    """`x + conv(x)` over a ragged batch, pre-chunk taps from the ring; with
    `commit`, also writes each sequence's last inputs into its ring slot."""

    @inline(.always)
    @staticmethod
    def execute[
        commit: Bool,
        //,
        target: StaticString,
    ](
        output: Some[Output & Tensor],
        x: Some[Input & Tensor],
        weight: Some[Input & Tensor],
        ring: Some[MutableInput & Tensor],
        input_row_offsets: Some[Input & Tensor],
        positions: Some[Input & Tensor],
        conv_rows: Some[Input & Tensor],
        layer_row: UInt32,
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(x), TileTensorable)
        comptime assert conforms_to(type_of(weight), TileTensorable)
        comptime assert conforms_to(type_of(ring), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(positions), TileTensorable)
        comptime assert conforms_to(type_of(conv_rows), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime assert type_of(output).rank == 2
        comptime assert type_of(x).rank == 2
        comptime assert type_of(weight).rank == 2
        comptime assert type_of(ring).rank == 3
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(positions).rank == 1
        comptime assert type_of(conv_rows).rank == 2
        short_conv_ring[target=target, commit=commit](
            x.to_tile_tensor[dtype](),
            weight.to_tile_tensor[dtype](),
            ring.to_tile_tensor(),
            input_row_offsets.to_tile_tensor[.uint32](),
            positions.to_tile_tensor[.uint32](),
            conv_rows.to_tile_tensor[.uint32](),
            layer_row,
            output.to_tile_tensor(),
            context,
        )


@extensibility.register("mo.fused_qk_rms_norm_short_conv.ragged.paged")
struct Struct_fused_qk_rms_norm_short_conv_ragged_paged:
    """The fused prologue of a short-conv attention block; see
    `fused_qk_rms_norm_short_conv_ragged_paged` in `nn/short_conv_ring.mojo`."""

    @inline(.always)
    @staticmethod
    def execute[
        multiply_before_cast: Bool,
        apply_log_scaling: Bool,
        commit: Bool,
        //,
        target: StaticString,
    ](
        q_output: Some[Output & Tensor],
        qkvr: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        positions: Some[Input & Tensor],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        q_gamma: Some[Input & Tensor],
        k_gamma: Some[Input & Tensor],
        k_weight: Some[Input & Tensor],
        v_weight: Some[Input & Tensor],
        k_conv_ring: Some[MutableInput & Tensor],
        v_conv_ring: Some[MutableInput & Tensor],
        k_conv_rows: Some[Input & Tensor],
        v_conv_rows: Some[Input & Tensor],
        k_layer_row: UInt32,
        v_layer_row: UInt32,
        log_scaling: Some[Input & Tensor],
        epsilon: Float32,
        layer_idx: UInt32,
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(q_output), TileTensorable)
        comptime assert conforms_to(type_of(qkvr), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(positions), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert conforms_to(type_of(q_gamma), TileTensorable)
        comptime assert conforms_to(type_of(k_gamma), TileTensorable)
        comptime assert conforms_to(type_of(k_weight), TileTensorable)
        comptime assert conforms_to(type_of(v_weight), TileTensorable)
        comptime assert conforms_to(type_of(k_conv_ring), TileTensorable)
        comptime assert conforms_to(type_of(v_conv_ring), TileTensorable)
        comptime assert conforms_to(type_of(k_conv_rows), TileTensorable)
        comptime assert conforms_to(type_of(v_conv_rows), TileTensorable)
        comptime assert conforms_to(type_of(log_scaling), TileTensorable)
        comptime dtype = type_of(q_output).dtype
        comptime ring_dtype = type_of(k_conv_ring).dtype
        comptime assert type_of(q_output).rank == 3
        comptime assert type_of(qkvr).rank == 2
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(positions).rank == 1
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(q_gamma).rank == 1
        comptime assert type_of(k_gamma).rank == 1
        comptime assert type_of(k_weight).rank == 2
        comptime assert type_of(v_weight).rank == 2
        comptime assert type_of(k_conv_ring).rank == 3
        comptime assert type_of(v_conv_ring).rank == 3
        comptime assert type_of(k_conv_rows).rank == 2
        comptime assert type_of(v_conv_rows).rank == 2
        comptime assert type_of(log_scaling).rank == 1
        var kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )
        fused_qk_rms_norm_short_conv_ragged_paged[
            target=target,
            multiply_before_cast=multiply_before_cast,
            apply_log_scaling=apply_log_scaling,
            commit=commit,
        ](
            qkvr.to_tile_tensor[dtype](),
            kv_collection,
            q_gamma.to_tile_tensor[dtype](),
            k_gamma.to_tile_tensor[dtype](),
            k_weight.to_tile_tensor[dtype](),
            v_weight.to_tile_tensor[dtype](),
            k_conv_ring.to_tile_tensor(),
            v_conv_ring.to_tile_tensor[ring_dtype](),
            epsilon,
            layer_idx,
            input_row_offsets.to_tile_tensor[.uint32](),
            positions.to_tile_tensor[.uint32](),
            k_conv_rows.to_tile_tensor[.uint32](),
            v_conv_rows.to_tile_tensor[.uint32](),
            k_layer_row,
            v_layer_row,
            log_scaling.to_tile_tensor[.float32](),
            q_output.to_tile_tensor(),
            context,
        )
