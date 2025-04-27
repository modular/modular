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
"""An opaque KV Cache optimized vanilla attention mechanism, with Mask Variants provided inside the Kernel."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Union

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops

from ..clamp import clamp
from ..kernels import (
    MHAMaskVariant,
    flash_attention_ragged,
    fused_qkv_ragged_matmul,
)
from ..kv_cache import (
    ContinuousBatchingKVCacheCollection,
    KVCacheParams,
    PagedKVCacheCollection,
)
from ..layer import Module
from ..linear import LinearV2


@dataclass
class RaggedAttention(Module):
    """Layer that computes the self attention score for ragged inputs."""

    def __init__(
        self,
        *,
        mask_variant: MHAMaskVariant,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        layer_idx: int,
        devices: list[DeviceRef] | None = None,
        dtype: DType = DType.float32,
        linear_cls: Callable[..., LinearV2] = LinearV2,
        stacked_qkv: bool = False,
        scale: float | None = None,
        has_bias: bool = False,
        clip_qkv: float | None = None,
    ):
        """Initializes the attention layer.

        Args:
            rope: The rope layer to borrow the freq_cis value from.
            num_attention_heads: The number of attention heads.
            num_key_value_heads: Number of key/value heads.
            hidden_size: The dimension of the hidden states.
            kv_params: KV Cache Params, including the number of kv heads, the head dim, and data type.
            layer_idx: The layer number associated with this Attention block.
            dtype: DType of the
            devices: Device to place the weights and run the computation. If
                multiple are provided, the first device is used.
            linear_cls: Linear class to use for the outputs dense layer.
            stacked_qkv: Whether the weights are stacked together.
            scale: Value used to scale the results of the attention output.
            has_bias: Whether to use an attention bias.
            clip_qkv: If provided, the QKV weights are clamped between
                `[-clip_qkv, clip_qkv]`
        """
        if has_bias:
            raise ValueError(
                "RaggedAttention does not yet support `has_bias=True`."
            )
        if stacked_qkv and clip_qkv:
            raise ValueError(
                "`clip_qkv` not yet supported when `stack_qkv=True`."
            )
        if not kv_params.cache_strategy.uses_opaque():
            raise ValueError(
                f"{kv_params.cache_strategy} cache strategy, not supported"
                " in Attention layer."
            )

        super().__init__()
        self.mask_variant = mask_variant
        self.n_heads = num_attention_heads
        self.layer_idx = layer_idx
        self.kv_params = kv_params
        self.has_bias = has_bias
        self.scale = (
            scale if scale else math.sqrt(1.0 / self.kv_params.head_dim)
        )
        self.clip_qkv = clip_qkv
        self.devices = devices or [DeviceRef.CPU()]

        kv_weight_dim = (
            hidden_size // num_attention_heads
        ) * num_key_value_heads

        self.stacked_qkv = stacked_qkv

        if stacked_qkv:
            # To keep the weight names consistent with the transformers attention,
            # the names are suffixed ".weight".
            self.qkv_proj = Weight(
                name="qkv_proj.weight",
                dtype=dtype,
                shape=[hidden_size + 2 * kv_weight_dim, hidden_size],
                device=self.devices[0],
            )
        else:
            self.q_proj = Weight(
                name="q_proj.weight",
                dtype=dtype,
                shape=[hidden_size, hidden_size],
                device=self.devices[0],
            )
            self.k_proj = Weight(
                name="k_proj.weight",
                dtype=dtype,
                shape=[kv_weight_dim, hidden_size],
                device=self.devices[0],
            )
            self.v_proj = Weight(
                name="v_proj.weight",
                dtype=dtype,
                shape=[kv_weight_dim, hidden_size],
                device=self.devices[0],
            )

        if has_bias:
            assert not stacked_qkv, "Bias is not supported with stacked qkv."

            self.bias_q = Weight(
                name="q_proj.bias",
                dtype=dtype,
                shape=[hidden_size],
                device=self.devices[0],
            )
            self.bias_k = Weight(
                name="k_proj.bias",
                dtype=dtype,
                shape=[kv_weight_dim],
                device=self.devices[0],
            )
            self.bias_v = Weight(
                name="v_proj.bias",
                dtype=dtype,
                shape=[kv_weight_dim],
                device=self.devices[0],
            )

        self.o_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=hidden_size,
            dtype=dtype,
            device=self.devices[0],
        )

    @property
    def wqkv(self) -> TensorValue:
        """The concatenation of q, k, and v weight vectors."""
        if self.stacked_qkv:
            return self.qkv_proj
        else:
            wq: TensorValue = self.q_proj
            wk: TensorValue = self.k_proj
            wv: TensorValue = self.v_proj
            if self.clip_qkv:
                wq = clamp(wq, min=-self.clip_qkv, max=self.clip_qkv)
                wk = clamp(wk, min=-self.clip_qkv, max=self.clip_qkv)
                wv = clamp(wv, min=-self.clip_qkv, max=self.clip_qkv)
            return ops.concat((wq, wk, wv))

    def __call__(
        self,
        x: TensorValue,
        kv_collection: Union[
            ContinuousBatchingKVCacheCollection, PagedKVCacheCollection
        ],
        **kwargs,
    ) -> TensorValue:
        # Get attributes from input.
        total_seq_len = x.shape[0]
        layer_idx = ops.constant(self.layer_idx, DType.uint32)

        # Call into fused qkv ragged matmul.
        xq = fused_qkv_ragged_matmul(
            self.kv_params,
            input=x,
            wqkv=self.wqkv,
            input_row_offsets=kwargs["input_row_offsets"],
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
        )

        # Reshape for flash attention.
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        # Calculate Flash Attention.
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=kwargs["input_row_offsets"],
            mask_variant=self.mask_variant,
            scale=self.scale,
        )

        attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])

        return self.o_proj(attn_out)
