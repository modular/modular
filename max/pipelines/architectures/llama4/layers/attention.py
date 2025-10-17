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

"""Llama4 Attention layer."""

from __future__ import annotations

import math
from collections.abc import Callable

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    Weight,
    ops,
)
from max.nn.attention import MHAMaskVariant
from max.nn.attention.attention_with_rope import distribute_value
from max.nn.comm import Allreduce
from max.nn.kernels import (
    flash_attention_ragged,
    fused_qk_ragged_rope,
    fused_qkv_ragged_matmul,
    rms_norm_key_cache,
)
from max.nn.kv_cache import (
    KVCacheParams,
    PagedCacheValues,
)
from max.nn.layer import Module
from max.nn.linear import Linear
from max.nn.rotary_embedding import RotaryEmbedding

from .norm import l2_norm


def Llama4TextAttention(**kwargs):  # noqa: ANN201
    """Implementation of the attention layer for the Llama4 text model."""
    devices = kwargs["devices"]
    if len(devices) == 1:
        return _Llama4TextAttention(**kwargs)
    else:
        return _DistributedLlama4TextAttention(**kwargs)


class _Llama4TextAttention(Module):
    """Implementation of the attention layer for the Llama4 text model."""

    def __init__(
        self,
        *,
        rope: RotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        layer_idx: int,
        dtype: DType = DType.float32,
        attn_temperature_tuning: bool,
        floor_scale: float,
        attn_scale: float,
        devices: list[DeviceRef],
        linear_cls: Callable[..., Linear] = Linear,
        scale: float | None = None,
        has_bias: bool = False,
        use_rope: bool = True,
        use_qk_norm: bool = False,
        qk_norm_eps: float = 1e-6,
        local_window_size: int = 8192,
    ) -> None:
        """Initializes the attention layer.

        Args:
            rope: The rope layer to borrow the freqs_cis value from.
            num_attention_heads: The number of attention heads.
            num_key_value_heads: Number of key/value heads.
            hidden_size: The dimension of the hidden states.
            kv_params: KV Cache Params, including the number of kv heads, the head dim, and data type.
            layer_idx: The layer number associated with this Attention block.
            dtype: DType of the attention inputs and weights.
            attn_temperature_tuning: int, used to improve accuracy for long
                contexts.
            floor_scale: Float, used with `attn_temperature_tuning`.
            attn_scale: Float, used with `attn_temperature_tuning`.
            devices: Device to place the weights and run the computation. If
                multiple are provided, the first device is used. Use
                `TensorParallelAttentionWithRope` to use all devices during
                attention computation.
            linear_cls: Linear class to use for the outputs dense layer.
            scale: Value used to scale the results of the attention output.
            has_bias: Whether to use an attention bias. Defaults to False.
            use_rope: Whether to use rope in this layer. Defaults to True.
            use_qk_norm: Whether to normalize the qk values. Defaults to False.
            qk_norm_eps: Value to use for numerical stability. Defaults to 1e-6.
            local_window_size: The size of the local window for the attention.
                Defaults to 8192.
        """

        super().__init__()
        self.rope = rope
        self.n_heads = num_attention_heads
        self.layer_idx = layer_idx
        self.kv_params = kv_params
        self.has_bias = has_bias
        self.devices = devices
        self.scale = (
            scale if scale else math.sqrt(1.0 / self.kv_params.head_dim)
        )
        self.local_window_size = local_window_size
        # rope unused for dense layers
        self.use_rope = use_rope
        self.use_qk_norm = self.use_rope and use_qk_norm
        self.attn_scale = attn_scale
        self.floor_scale = floor_scale
        self.attn_temperature_tuning = attn_temperature_tuning
        self.qk_norm_eps = qk_norm_eps

        if not self.kv_params.cache_strategy.uses_opaque():
            raise ValueError(
                f"{self.kv_params.cache_strategy} cache strategy, not supported"
                " in Attention layer."
            )

        self.q_weight_dim = self.kv_params.head_dim * num_attention_heads
        self.kv_weight_dim = self.kv_params.head_dim * num_key_value_heads

        self.q_proj = Weight(
            name="q_proj.weight",
            dtype=dtype,
            shape=[self.q_weight_dim, hidden_size],
            device=devices[0],
        )
        self.k_proj = Weight(
            name="k_proj.weight",
            dtype=dtype,
            shape=[self.kv_weight_dim, hidden_size],
            device=devices[0],
        )
        self.v_proj = Weight(
            name="v_proj.weight",
            dtype=dtype,
            shape=[self.kv_weight_dim, hidden_size],
            device=devices[0],
        )

        if has_bias:
            self.bias_q = Weight(
                name="q_proj.bias",
                dtype=dtype,
                shape=[self.q_weight_dim],
                device=devices[0],
            )
            self.bias_k = Weight(
                name="k_proj.bias",
                dtype=dtype,
                shape=[self.kv_weight_dim],
                device=devices[0],
            )
            self.bias_v = Weight(
                name="v_proj.bias",
                dtype=dtype,
                shape=[self.kv_weight_dim],
                device=devices[0],
            )

        self.o_proj = linear_cls(
            in_dim=self.q_weight_dim,
            out_dim=hidden_size,
            dtype=dtype,
            device=devices[0],
        )

    @property
    def wqkv(self) -> TensorValue:
        """The concatenation of q, k, and v weight vectors."""
        wq: TensorValue = self.q_proj
        wk: TensorValue = self.k_proj
        wv: TensorValue = self.v_proj
        return ops.concat((wq, wk, wv)).to(self.devices[0])

    @property
    def wqkv_bias(self) -> TensorValue | None:
        """The concatenation of q, k, and v bias weight vectors."""
        if not self.has_bias:
            return None

        return ops.concat((self.bias_q, self.bias_k, self.bias_v))

    def __call__(
        self,
        xs: list[TensorValue],
        cache_positions_list: list[TensorValue],
        kv_collections: list[PagedCacheValues],
        **kwargs,
    ) -> list[TensorValue]:
        assert len(xs) == 1 and len(kv_collections) == 1
        x = xs[0]
        kv_collection = kv_collections[0]
        cache_positions = cache_positions_list[0]

        # Get attributes from input.
        total_seq_len = x.shape[0]

        layer_idx = ops.constant(
            self.layer_idx, DType.uint32, device=DeviceRef.CPU()
        )
        # Call into fused qkv ragged matmul.
        wqkv = self.wqkv
        xq = fused_qkv_ragged_matmul(
            self.kv_params,
            input=x,
            wqkv=wqkv,
            bias=self.wqkv_bias,
            input_row_offsets=kwargs["input_row_offsets"],
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
        )
        # Apply rope.
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        if self.use_rope:
            if xq.device is not None:
                freqs_cis = self.rope.freqs_cis.to(xq.device)
            else:
                freqs_cis = self.rope.freqs_cis
            xq = fused_qk_ragged_rope(
                self.kv_params,
                xq,
                kwargs["input_row_offsets"],
                kv_collection,
                freqs_cis,
                layer_idx,
                interleaved=self.rope.interleaved,
            )

        if self.use_qk_norm:
            # Apply QK norm to query and key states.
            xq = l2_norm(xq, self.qk_norm_eps, multiply_before_cast=False)
            rms_norm_key_cache(
                self.kv_params,
                kv_collection=kv_collection,
                gamma=ops.constant(
                    1, self.kv_params.dtype, device=DeviceRef.CPU()
                )
                .broadcast_to([self.kv_params.head_dim])
                .to(self.devices[0]),
                epsilon=self.qk_norm_eps,
                layer_idx=layer_idx,
                total_seq_len=total_seq_len,
                input_row_offsets=kwargs["input_row_offsets"],
                weight_offset=0.0,
            )

        if self.attn_temperature_tuning and not self.use_rope:
            attn_scales = (
                ops.log(
                    ops.floor(
                        (ops.cast(cache_positions, DType.float32) + 1.0)
                        / self.floor_scale
                    )
                    + 1.0
                )
                * self.attn_scale
                + 1.0
            ).to(self.devices[0])
            attn_scales = ops.reshape(attn_scales, [-1, 1, 1])
            xq = (xq.cast(DType.float32) * attn_scales).cast(xq.dtype)

        # Calculate Flash Attention.
        mask_variant = (
            MHAMaskVariant.CHUNKED_CAUSAL_MASK
            if int((self.layer_idx + 1) % 4) != 0
            else MHAMaskVariant.CAUSAL_MASK
        )
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=kwargs["input_row_offsets"],
            mask_variant=mask_variant,
            local_window_size=self.local_window_size,
            scale=self.scale,
        )
        attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])
        ret = self.o_proj(attn_out)
        return [ret]


class _DistributedLlama4TextAttention(_Llama4TextAttention):
    """Distributed implementation of the Llama4 text attention layer."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if not self.devices or len(self.devices) < 2:
            raise ValueError(
                f"Must provide at least 2 devices to `_DistributedLlama4TextAttention`, got {self.devices}"
            )
        # Shard weights into separate AttentionWithRope layers.
        num_devices = len(self.devices)
        self.allreduce = Allreduce(num_devices)

        self.q_proj.sharding_strategy = ShardingStrategy.rowwise(num_devices)
        self.k_proj.sharding_strategy = ShardingStrategy.rowwise(num_devices)
        self.v_proj.sharding_strategy = ShardingStrategy.rowwise(num_devices)
        self.o_proj.sharding_strategy = ShardingStrategy.columnwise(num_devices)

        self.list_of_attentions = []
        kwargs = kwargs.copy()
        kwargs["num_attention_heads"] //= len(self.devices)

        # Shard weights once for all devices
        q_proj_shards = self.q_proj.shard(self.devices)
        k_proj_shards = self.k_proj.shard(self.devices)
        v_proj_shards = self.v_proj.shard(self.devices)
        o_proj_weight_shards = self.o_proj.weight.shard(self.devices)

        for n, device in enumerate(self.devices):
            kwargs["devices"] = [device]
            layer = _Llama4TextAttention(**kwargs)
            layer.q_proj = q_proj_shards[n]
            layer.k_proj = k_proj_shards[n]
            layer.v_proj = v_proj_shards[n]
            layer.o_proj.weight = o_proj_weight_shards[n]
            self.list_of_attentions.append(layer)

    def __call__(
        self,
        xs: list[TensorValue],
        cache_positions_list: list[TensorValue],
        kv_collections: list[PagedCacheValues],
        **kwargs,
    ) -> list[TensorValue]:
        input_row_offsets = kwargs["input_row_offsets"]

        signal_buffers: list[BufferValue] = kwargs["signal_buffers"]
        assert isinstance(input_row_offsets, TensorValue)
        assert self.devices
        input_row_offsets_ = distribute_value(input_row_offsets, self.devices)
        return self.allreduce(
            inputs=[
                self.list_of_attentions[i](
                    [xs[i]],
                    [cache_positions_list[i]],
                    [kv_collections[i]],
                    input_row_offsets=input_row_offsets_[i],
                )[0]
                for i in range(len(self.devices))
            ],
            signal_buffers=signal_buffers,
        )
