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
from extensibility import InputTensor, OutputTensor
from extensibility import (
    _MutableInputTensor as MutableInputTensor,
)
from max.gpu.host import DeviceContext

from nn.kv_cache import generic_get_paged_cache
from nn.short_conv_ring import (
    fused_qk_rms_norm_short_conv_ragged_paged,
    short_conv_ring_commit,
    short_conv_ring_commit_kv,
    short_conv_ring_fwd,
)


@extensibility.register("mo.short_conv_ring_fwd")
struct Struct_short_conv_ring_fwd:
    """`x + conv(x)` over a ragged batch, pre-chunk taps from the ring."""

    @inline(.always)
    @staticmethod
    def execute[
        dtype: DType,
        ring_dtype: DType,
        //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        x: InputTensor[dtype=dtype, rank=2, ...],
        weight: InputTensor[dtype=dtype, rank=2, ...],
        # Only read here; mutable so the graph orders the commit after it.
        ring: MutableInputTensor[dtype=ring_dtype, rank=3, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        positions: InputTensor[dtype=.uint32, rank=1, ...],
        conv_row: InputTensor[dtype=.uint32, rank=1, ...],
        context: DeviceContext,
    ) raises:
        short_conv_ring_fwd[target=target](
            x.to_tile_tensor[.int64](),
            weight.to_tile_tensor[.int64](),
            ring.to_tile_tensor[.int64](),
            input_row_offsets.to_tile_tensor[.int64](),
            positions.to_tile_tensor[.int64](),
            conv_row.to_tile_tensor[.int64](),
            output.to_tile_tensor[.int64](),
            context,
        )


@extensibility.register("mo.short_conv_ring_commit")
struct Struct_short_conv_ring_commit:
    """Writes each sequence's last inputs into its conv ring slot."""

    @inline(.always)
    @staticmethod
    def execute[
        dtype: DType,
        ring_dtype: DType,
        //,
        target: StaticString,
    ](
        ring: MutableInputTensor[dtype=ring_dtype, rank=3, ...],
        x: InputTensor[dtype=dtype, rank=2, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        positions: InputTensor[dtype=.uint32, rank=1, ...],
        conv_row: InputTensor[dtype=.uint32, rank=1, ...],
        context: DeviceContext,
    ) raises:
        short_conv_ring_commit[target=target](
            x.to_tile_tensor[.int64](),
            ring.to_tile_tensor[.int64](),
            input_row_offsets.to_tile_tensor[.int64](),
            positions.to_tile_tensor[.int64](),
            conv_row.to_tile_tensor[.int64](),
            context,
        )


@extensibility.register("mo.short_conv_ring_commit_kv")
struct Struct_short_conv_ring_commit_kv[k_col: Int]:
    """Commits the K and V conv inputs of a fused `qkvr` projection.

    Parameters:
        k_col: First K column of `qkvr`; V follows K.
    """

    @inline(.always)
    @staticmethod
    def execute[
        dtype: DType,
        ring_dtype: DType,
        //,
        target: StaticString,
    ](
        k_ring: MutableInputTensor[dtype=ring_dtype, rank=3, ...],
        v_ring: MutableInputTensor[dtype=ring_dtype, rank=3, ...],
        qkvr: InputTensor[dtype=dtype, rank=2, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        positions: InputTensor[dtype=.uint32, rank=1, ...],
        k_conv_row: InputTensor[dtype=.uint32, rank=1, ...],
        v_conv_row: InputTensor[dtype=.uint32, rank=1, ...],
        context: DeviceContext,
    ) raises:
        short_conv_ring_commit_kv[target=target, k_col=Self.k_col](
            qkvr.to_tile_tensor[.int64](),
            k_ring.to_tile_tensor[.int64](),
            v_ring.to_tile_tensor[.int64](),
            input_row_offsets.to_tile_tensor[.int64](),
            positions.to_tile_tensor[.int64](),
            k_conv_row.to_tile_tensor[.int64](),
            v_conv_row.to_tile_tensor[.int64](),
            context,
        )


@extensibility.register("mo.fused_qk_rms_norm_short_conv.ragged.paged")
struct Struct_fused_qk_rms_norm_short_conv_ragged_paged:
    """The fused prologue of a short-conv attention block; see
    `fused_qk_rms_norm_short_conv_ragged_paged` in `nn/short_conv_ring.mojo`."""

    @inline(.always)
    @staticmethod
    def execute[
        dtype: DType,
        cache_dtype: DType,
        ring_dtype: DType,
        multiply_before_cast: Bool,
        apply_log_scaling: Bool,
        //,
        target: StaticString,
    ](
        q_output: OutputTensor[dtype=dtype, rank=3, ...],
        qkvr: InputTensor[dtype=dtype, rank=2, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        positions: InputTensor[dtype=.uint32, rank=1, ...],
        kv_blocks: MutableInputTensor[dtype=cache_dtype, rank=6, ...],
        page_stride: InputTensor[dtype=.int64, rank=1, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        q_gamma: InputTensor[dtype=dtype, rank=1, ...],
        k_gamma: InputTensor[dtype=dtype, rank=1, ...],
        k_weight: InputTensor[dtype=dtype, rank=2, ...],
        v_weight: InputTensor[dtype=dtype, rank=2, ...],
        # Only read here; mutable so the graph orders the commit after it.
        k_conv_ring: MutableInputTensor[dtype=ring_dtype, rank=3, ...],
        v_conv_ring: MutableInputTensor[dtype=ring_dtype, rank=3, ...],
        k_conv_row: InputTensor[dtype=.uint32, rank=1, ...],
        v_conv_row: InputTensor[dtype=.uint32, rank=1, ...],
        log_scaling: InputTensor[dtype=.float32, rank=1, ...],
        epsilon: Float32,
        layer_idx: UInt32,
        context: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            page_stride,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )
        fused_qk_rms_norm_short_conv_ragged_paged[
            target=target,
            multiply_before_cast=multiply_before_cast,
            apply_log_scaling=apply_log_scaling,
        ](
            qkvr.to_tile_tensor[.int64](),
            kv_collection,
            q_gamma.to_tile_tensor[.int64](),
            k_gamma.to_tile_tensor[.int64](),
            k_weight.to_tile_tensor[.int64](),
            v_weight.to_tile_tensor[.int64](),
            k_conv_ring.to_tile_tensor[.int64](),
            v_conv_ring.to_tile_tensor[.int64](),
            epsilon,
            layer_idx,
            input_row_offsets.to_tile_tensor[.int64](),
            positions.to_tile_tensor[.int64](),
            k_conv_row.to_tile_tensor[.int64](),
            v_conv_row.to_tile_tensor[.int64](),
            log_scaling.to_tile_tensor[.int64](),
            q_output.to_tile_tensor[.int64](),
            context,
        )
