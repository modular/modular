# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
from __future__ import annotations

from typing import Callable, Union, cast

from max.dtype import DType
from max.graph import (
    DeviceRef,
    TensorValue,
    ops,
)

from ..attention.attention_with_rope import AttentionWithRope
from ..attention.mask_config import MHAMaskVariant
from ..kernels import (
    flash_attention_ragged,
    fused_qk_ragged_rope,
    fused_qkv_ragged_matmul,
    sgmv_qkv_lora_kernel,
)
from ..kv_cache import (
    ContinuousBatchingKVCacheCollection,
    KVCacheParams,
    PagedKVCacheCollection,
)
from ..linear import Float8Config, Linear
from ..rotary_embedding import RotaryEmbedding
from .linear_lora import LinearLoRA


class AttentionWithRopeAndLoRA(AttentionWithRope):
    # This class will not use the RotaryEmbedding to
    # calculate rope, but it already includes a freqs_cis
    # calculation, which we will borrow
    rope: RotaryEmbedding

    def __init__(
        self,
        *,
        rope: RotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        devices: list[DeviceRef] | None = None,
        dtype: DType = DType.float32,
        linear_cls: Callable[..., Linear] = LinearLoRA,
        stacked_qkv: bool = False,
        scale: float | None = None,
        has_bias: bool = False,
        float8_config: Float8Config | None = None,
        clip_qkv: float | None = None,
    ):
        """Initializes the LoRA-enabled attention layer.

        Args:
            rope: The rope layer to borrow the freqs_cis value from.
            num_attention_heads: The number of attention heads.
            num_key_value_heads: Number of key/value heads.
            hidden_size: The dimension of the hidden states.
            kv_params: KV Cache Params, including the number of kv heads, the head dim, and data type.
            dtype: DType of the QKV and output projection weights.
            devices: Device to place the weights and run the computation. If
                multiple are provided, the first device is used. Use
                `DistributedAttentionWithRope` to use all devices during
                attention computation.
            linear_cls: Linear class to use for the outputs dense layer.
            stacked_qkv: Whether the weights are stacked together.
            scale: Value used to scale the results of the attention output.
            has_bias: Whether to use an attention bias.
            clip_qkv: If provided, the QKV weights are clamped between
                `[-clip_qkv, clip_qkv]`
        """
        if float8_config:
            raise NotImplementedError("Float8 is not implemented for LoRA.")

        if stacked_qkv:
            raise NotImplementedError("LoRA doesn't support stacked QKV.")

        super().__init__(
            rope=rope,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_size=hidden_size,
            kv_params=kv_params,
            devices=devices,
            dtype=dtype,
            linear_cls=linear_cls,
            stacked_qkv=stacked_qkv,
            scale=scale,
            has_bias=has_bias,
            float8_config=float8_config,
            clip_qkv=clip_qkv,
        )

    @property
    def qkv_loras(self) -> list[LinearLoRA]:
        # MyPy isn't intelligent enough to know we have checks elsewhere.
        #  Even if we put the check in the __init__ function...
        if (
            not isinstance(self.q_proj, LinearLoRA)
            or not isinstance(self.k_proj, LinearLoRA)
            or not isinstance(self.v_proj, LinearLoRA)
        ):
            raise ValueError("Attention projections must be LinearLoRAs.")

        return [self.q_proj, self.k_proj, self.v_proj]

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: Union[
            ContinuousBatchingKVCacheCollection, PagedKVCacheCollection
        ],
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        # Get attributes from input.
        total_seq_len = x.shape[0]

        xq = fused_qkv_ragged_matmul(
            self.kv_params,
            input=x,
            wqkv=self.wqkv,
            bias=self.wqkv_bias,
            input_row_offsets=input_row_offsets,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
        )

        xq += self.fused_qkv_lora(
            xq,  # should be X but graph comp crashes due to stubbing kernel
            input_row_offsets,
            kv_collection,
            layer_idx,
        )

        # Apply rope.
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        if xq.device is not None:
            freqs_cis = ops.cast(self.rope.freqs_cis, xq.dtype).to(xq.device)
        else:
            freqs_cis = ops.cast(self.rope.freqs_cis, xq.dtype)

        xq = fused_qk_ragged_rope(
            self.kv_params,
            xq,
            input_row_offsets,
            kv_collection,
            freqs_cis,
            layer_idx,
            interleaved=self.rope.interleaved,
        )
        # Calculate Flash Attention.
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=input_row_offsets,
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=self.scale,
        )

        attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])

        out = cast(LinearLoRA, self.o_proj).apply_lora(
            attn_out, input_row_offsets
        )

        return out

    def fused_qkv_lora(
        self,
        x: TensorValue,
        input_row_offsets: TensorValue,
        kv_collection: ContinuousBatchingKVCacheCollection
        | PagedKVCacheCollection,
        layer_idx: TensorValue,
    ):
        """
        Computes fused query, key, and value LoRAs with ragged input.

        Args:
            x (TensorValue): The input tensor of shape [total_tokens, hidden_dim].
            qkv_loras (list[LinearLoRA]): List of 3 LinearLoRA modules for Q, K, and V projections.
            input_row_offsets (TensorValue): 1D tensor indicating the start index of each sequence in `x`.
            kv_collection (ContinuousBatchingKVCacheCollection | PagedKVCacheCollection):
                The key/value cache collection structure.
            layer_idx (TensorValue): Index of the current transformer layer (used for caching).

        Returns:
            TensorValue: The query projections.

        Raises:
            ValueError: If 'set_lora_batch_info' has not been called on the LoRAs.
        """
        qkv_loras = self.qkv_loras

        lora_ids = qkv_loras[0].lora_ids
        lora_ranks = qkv_loras[0].lora_ranks

        if lora_ids is None or lora_ranks is None:
            raise ValueError(
                "'set_lora_batch_info' not called before executing forward pass."
            )

        if len(qkv_loras) == 3:
            lora_a = ops.concat([lora.lora_A for lora in qkv_loras], axis=1)
            lora_b = ops.concat([lora.lora_B for lora in qkv_loras], axis=1)
            lora_bias = None
        else:
            lora_a = qkv_loras[0].lora_A
            lora_b = qkv_loras[0].lora_B
            lora_bias = None

        y = sgmv_qkv_lora_kernel(
            input=x,
            lora_a=lora_a,
            lora_b=lora_b,
            lora_ids=lora_ids,
            lora_ranks=lora_ranks,
            input_row_offsets=input_row_offsets,
            kv_params=self.kv_params,
            kv_collection=kv_collection,
            n_heads=self.n_heads,
            bias=lora_bias,
        )
        return y
