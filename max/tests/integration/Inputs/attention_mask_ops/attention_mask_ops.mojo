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
"""Attention entry points for testing the masks in `nn.attention.operand_masks`
against `MaterializedMask`, which reads the same tensor through its pointer.

The `mask` string parameter picks the mask; the mask operands are the trailing
variadic.
"""

import extensibility
from extensibility import (
    Fused,
    Input,
    LoadAccess,
    Output,
    TensorTuple,
    TileTensorable,
)
from layout import ImmTileTensor
from max.gpu.host import DeviceContext
from max.gpu.host.info import is_gpu
from nn.attention.gpu.mha import flash_attention, flash_attention_ragged
from nn.attention.mha_mask import CausalMask, MaterializedMask, MHAMask
from nn.attention.operand_masks import AdditiveMask, RelativeLogitsMask


@inline(.always)
def _ragged_attention[
    M: MHAMask, *Ts: Input & Fused & LoadAccess
](
    output: Some[Output & TileTensorable],
    q: Some[Input & TileTensorable],
    k: Some[Input & TileTensorable],
    v: Some[Input & TileTensorable],
    input_row_offsets: Some[Input & TileTensorable],
    q_max_seq_len: Some[Input & TileTensorable],
    scale: Float32,
    mask_operands: TensorTuple[*Ts],
    ctx: DeviceContext,
) raises:
    flash_attention_ragged(
        output.to_tile_tensor(),
        q.to_tile_tensor(),
        k.to_tile_tensor(),
        v.to_tile_tensor(),
        input_row_offsets.to_tile_tensor[.uint32](),
        q_max_seq_len.to_tile_tensor[.uint32](),
        M.make(mask_operands),
        scale,
        ctx,
    )


@extensibility.register("test_ragged_attention_load_mask")
struct RaggedAttentionLoadMask:
    """Ragged attention with a mask from `nn.attention.operand_masks`."""

    @staticmethod
    def execute[
        MaskIns: TypeList[Trait=Input & Fused & LoadAccess, ...],
        //,
        mask: StaticString,
        window: Int,
    ](
        output: Some[Output & TileTensorable],
        q: Some[Input & TileTensorable],
        k: Some[Input & TileTensorable],
        v: Some[Input & TileTensorable],
        input_row_offsets: Some[Input & TileTensorable],
        q_max_seq_len: Some[Input & TileTensorable],
        scale: Float32,
        mask_operands: TensorTuple[*MaskIns],
        ctx: DeviceContext,
    ) raises:
        comptime assert ctx.target.is_gpu(), "only valid on GPUs"
        comptime match mask:
            case "causal":
                _ragged_attention[CausalMask](
                    output,
                    q,
                    k,
                    v,
                    input_row_offsets,
                    q_max_seq_len,
                    scale,
                    mask_operands,
                    ctx,
                )
            case "additive":
                _ragged_attention[AdditiveMask[]](
                    output,
                    q,
                    k,
                    v,
                    input_row_offsets,
                    q_max_seq_len,
                    scale,
                    mask_operands,
                    ctx,
                )
            case "relative_logits":
                _ragged_attention[RelativeLogitsMask[window]](
                    output,
                    q,
                    k,
                    v,
                    input_row_offsets,
                    q_max_seq_len,
                    scale,
                    mask_operands,
                    ctx,
                )
            case _:
                comptime assert False, "unknown mask: " + mask


@inline(.always)
def _imm_tile[
    dtype: DType, T: TileTensorable
](operand: T) -> ImmTileTensor[dtype, T.LayoutType, ImmUntrackedOrigin]:
    var tile = operand.to_tile_tensor[dtype, mut=False]()
    return {
        ptr = rebind[Pointer[Scalar[dtype], ImmUntrackedOrigin]](tile.ptr),
        layout = tile.layout,
    }


@extensibility.register("test_ragged_attention_pointer_mask")
struct RaggedAttentionPointerMask:
    """Ragged attention with `MaterializedMask`, which `AdditiveMask`
    replaces."""

    @staticmethod
    def execute[
        MaskIns: TypeList[Trait=Input & TileTensorable, ...],
        //,
        mask: StaticString,
        window: Int,
    ](
        output: Some[Output & TileTensorable],
        q: Some[Input & TileTensorable],
        k: Some[Input & TileTensorable],
        v: Some[Input & TileTensorable],
        input_row_offsets: Some[Input & TileTensorable],
        q_max_seq_len: Some[Input & TileTensorable],
        scale: Float32,
        mask_operands: TensorTuple[*MaskIns],
        ctx: DeviceContext,
    ) raises:
        comptime assert ctx.target.is_gpu(), "only valid on GPUs"
        var output_buffer = output.to_tile_tensor()
        var q_buffer = q.to_tile_tensor()
        var k_buffer = k.to_tile_tensor()
        var v_buffer = v.to_tile_tensor()
        var row_offsets = input_row_offsets.to_tile_tensor[.uint32]()
        var max_seq_len = q_max_seq_len.to_tile_tensor[.uint32]()

        def _run[
            mask_t: MHAMask
        ](mask_value: mask_t) raises {
            var output_buffer,
            var q_buffer,
            var k_buffer,
            var v_buffer,
            var row_offsets,
            var max_seq_len,
            imm,
        }:
            flash_attention_ragged(
                output_buffer,
                q_buffer,
                k_buffer,
                v_buffer,
                row_offsets,
                max_seq_len,
                mask_value,
                scale,
                ctx,
            )

        comptime match mask:
            case "additive":
                _run(
                    MaterializedMask(
                        _imm_tile[type_of(mask_operands[0]).dtype](
                            mask_operands[0]
                        )
                    )
                )
            case _:
                comptime assert False, "unknown mask: " + mask


@extensibility.register("test_dense_attention_pointer_mask")
struct DenseAttentionPointerMask:
    """Dense `[batch, seq, heads, depth]` attention with the mask read through
    its pointer, as `attention.masked_flash_attention_gpu` did."""

    @staticmethod
    def execute(
        output: Some[Output & TileTensorable],
        q: Some[Input & TileTensorable],
        k: Some[Input & TileTensorable],
        v: Some[Input & TileTensorable],
        scale: Float32,
        mask: Some[Input & TileTensorable],
        ctx: DeviceContext,
    ) raises:
        comptime assert ctx.target.is_gpu(), "only valid on GPUs"
        flash_attention(
            output.to_tile_tensor(),
            q.to_tile_tensor(),
            k.to_tile_tensor(),
            v.to_tile_tensor(),
            mask.to_tile_tensor(),
            scale,
            context=ctx,
        )
