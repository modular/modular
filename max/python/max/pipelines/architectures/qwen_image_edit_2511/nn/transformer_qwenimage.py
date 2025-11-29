# Copyright 2025 Qwen-Image Team, The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput
from diffusers.models.attention import AttentionMixin, FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.attention_processor import Attention
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous, RMSNorm

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    Dim,
    ShardingStrategy,
    TensorValue,
    TensorValueLike,
    dtype_promotion,
    ops,
)
from max.nn import (
    MLP,
    Allreduce,
    LayerList,
    Linear,
    RMSNorm,
    Shardable,
)
from max.nn.float8_config import Float8Config
from max.nn.layer import Module
from collections.abc import Callable, Iterable, Sequence

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    TensorValueLike,
    Weight,
    ops,
)
from max.nn import (
    MLP,
    ColumnParallelLinear,
    LayerList,
    Linear,
    Llama3RotaryEmbedding,
    Module,
    ReturnLogits,
    RMSNorm,
    VocabParallelEmbedding,
)
from max.nn.attention.attention_with_rope import _compute_shard_range
from max.nn.comm.allreduce import Allreduce
from max.nn.float8_config import Float8Config
from max.nn.kernels import (
    MHAMaskVariant,
    flash_attention_ragged,
    fused_qk_ragged_rope,
    fused_qkv_ragged_matmul,
)
from max.nn.kv_cache import KVCacheParams, PagedCacheValues
from max.nn.layer import Shardable
from max.nn.transformer.distributed_transformer import (
    ShardableCallable,
    forward_sharded_layers,
)
from max.pipelines.architectures.internvl.embedding_utils import (
    merge_multimodal_embeddings_with_gather,
)
from max.pipelines.architectures.internvl.internvl import distribute_value


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def get_timestep_embedding(
    timesteps: TensorValue,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> TensorValue:
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (TensorValue):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        TensorValue: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * ops.range(
        start=0, end=half_dim, dtype=DType.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = ops.exp(exponent).cast(timesteps.dtype)
    emb = timesteps[:, None].cast(DType.float32) * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = ops.concat([ops.sin(emb), ops.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = ops.concat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class QwenTimestepProjEmbeddings(Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep: TensorValue, hidden_states: TensorValue) -> TensorValue:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.cast(hidden_states.dtype))  # (N, D)

        conditioning = timesteps_emb

        return conditioning


class QwenImageAttentionWithRope(Module, Shardable):
    """QwenImage attention layer with multi-axis rotary position embedding (mrope).

    This implementation is based on the Qwen2.5VL language model architecture, which
    is similar to Llama3 but includes attention bias and multi-axis rotary position embedding (mrope).

    This is a distributed attention layer that supports tensor parallel and replicate sharding strategies.

    This attention implementation supports 2D position IDs for vision-language tasks.
    """

    # This class will not use the RotaryEmbedding to
    # apply rope to the query, but it already includes a freqs_cis
    # calculation, which we will borrow
    rope: Llama3RotaryEmbedding

    def __init__(
        self,
        *,
        rope: Llama3RotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        devices: Sequence[DeviceRef] | None = None,
        dtype: DType = DType.float32,
        linear_cls: Callable[..., Linear] = Linear,
        scale: float | None = None,
        has_bias: bool = True,
        float8_config: Float8Config | None = None,
    ) -> None:
        """Initializes the QwenImage attention layer with mrope support.

        Args:
            rope: The rope layer to borrow the freqs_cis value from.
            num_attention_heads: The number of attention heads.
            num_key_value_heads: Number of key/value heads.
            hidden_size: The dimension of the hidden states.
            kv_params: KV Cache Params, including the number of kv heads, the head dim, and data type.
            devices: Device to place the weights and run the computation. This is a distributed
                attention layer, so we use all devices during attention computation.
            dtype: DType of the QKV and output projection weights.
            linear_cls: Linear class to use for the outputs dense layer.
            scale: Value used to scale the results of the attention output.
            has_bias: Whether to use an attention bias.
        """

        super().__init__()
        self.rope = rope
        self.n_heads = num_attention_heads
        self.kv_params = kv_params
        self.has_bias = has_bias
        self.hidden_size = hidden_size
        self.scale = (
            scale if scale else math.sqrt(1.0 / self.kv_params.head_dim)
        )
        self.float8_config = float8_config

        self.devices = devices or [DeviceRef.CPU()]

        self._sharding_strategy: ShardingStrategy | None = None

        if not self.kv_params.cache_strategy.uses_opaque():
            raise ValueError(
                f"{self.kv_params.cache_strategy} cache strategy, not supported"
                " in Attention layer."
            )

        q_weight_dim = self.kv_params.head_dim * num_attention_heads
        kv_weight_dim = self.kv_params.head_dim * num_key_value_heads

        self.q_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=q_weight_dim,
            dtype=dtype,
            device=self.devices[0],
            has_bias=has_bias,
            float8_config=float8_config,
        )
        self.k_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=kv_weight_dim,
            dtype=dtype,
            device=self.devices[0],
            has_bias=has_bias,
            float8_config=float8_config,
        )
        self.v_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=kv_weight_dim,
            dtype=dtype,
            device=self.devices[0],
            has_bias=has_bias,
            float8_config=float8_config,
        )

        self.o_proj = linear_cls(
            in_dim=q_weight_dim,
            out_dim=hidden_size,
            dtype=dtype,
            device=self.devices[0],
            float8_config=float8_config,
        )

    @property
    def wqkv(self) -> TensorValue:
        """The concatenation of q, k, and v weight vectors."""

        wq: TensorValue = self.q_proj.weight
        wk: TensorValue = self.k_proj.weight
        wv: TensorValue = self.v_proj.weight

        wqkv = ops.concat((wq, wk, wv))
        return wqkv

    @property
    def wqkv_bias(self) -> TensorValue | None:
        """The concatenation of q, k, and v bias weight vectors."""
        if not self.has_bias:
            return None

        # Access bias, which should all exist since has_bias=True.
        assert self.q_proj.bias is not None
        assert self.k_proj.bias is not None
        assert self.v_proj.bias is not None
        return ops.concat(
            (self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)
        )

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
        position_ids: TensorValue,
        mrope_section: list[int],
    ) -> TensorValue:
        # Keep attention stack in BF16.
        total_seq_len = x.shape[0]
        self.kv_params.dtype = DType.bfloat16

        # Make sure activations are BF16 for the fused kernel.
        x_in = x if x.dtype == DType.bfloat16 else ops.cast(x, DType.bfloat16)

        # Helper: dequantize an FP8 weight to BF16 using its scale, if present.
        def _dequant_w_to_bf16(
            w: Weight, w_scale: Weight | None
        ) -> TensorValue:
            w_bf16 = ops.cast(w.to(x_in.device), DType.bfloat16)
            if w_scale is None:
                return w_bf16
            s = ops.cast(w_scale.to(x_in.device), DType.bfloat16)
            # Supports scalar or rowwise scales via broadcasting.
            return w_bf16 * s

        # Build BF16 WQKV for the fused kernel:
        # - FP8 models: dequantize Q/K/V with their scales.
        # - Non-FP8 models: just cast the concatenated weight to BF16.
        if self.q_proj.weight.dtype.is_float8():
            wq_bf16 = _dequant_w_to_bf16(
                self.q_proj.weight, self.q_proj.weight_scale
            )
            wk_bf16 = _dequant_w_to_bf16(
                self.k_proj.weight, self.k_proj.weight_scale
            )
            wv_bf16 = _dequant_w_to_bf16(
                self.v_proj.weight, self.v_proj.weight_scale
            )
            wqkv_bf16 = ops.concat((wq_bf16, wk_bf16, wv_bf16), axis=0)
        else:
            wqkv_bf16 = self.wqkv
            if wqkv_bf16.dtype != DType.bfloat16:
                wqkv_bf16 = ops.cast(wqkv_bf16, DType.bfloat16)

        # Fused QKV matmul: input and wqkv are both BF16 now.
        xq = fused_qkv_ragged_matmul(
            self.kv_params,
            input=x_in,
            wqkv=wqkv_bf16,
            bias=self.wqkv_bias,
            input_row_offsets=input_row_offsets,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
        )

        # Apply RoPE and flash attention (also in BF16).
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))
        freqs_cis = freqs_cis.to(xq.device)
        xq = fused_qk_ragged_rope(
            self.kv_params,
            xq,
            input_row_offsets,
            kv_collection,
            freqs_cis,
            layer_idx,
            interleaved=self.rope.interleaved,
            position_ids=position_ids,
            mrope_section=mrope_section,
        )

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
        return self.o_proj(attn_out)

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, sharding_strategy: ShardingStrategy) -> None:
        num_devices = sharding_strategy.num_devices

        if sharding_strategy.is_replicate:
            self.q_proj.sharding_strategy = sharding_strategy
            self.k_proj.sharding_strategy = sharding_strategy
            self.v_proj.sharding_strategy = sharding_strategy
            self.o_proj.sharding_strategy = sharding_strategy

        elif sharding_strategy.is_tensor_parallel:
            self.q_proj.sharding_strategy = ShardingStrategy.rowwise(
                num_devices
            )
            self.k_proj.sharding_strategy = ShardingStrategy.rowwise(
                num_devices
            )
            self.v_proj.sharding_strategy = ShardingStrategy.rowwise(
                num_devices
            )
            self.o_proj.sharding_strategy = (
                ShardingStrategy.head_aware_columnwise(
                    num_devices, self.n_heads, self.kv_params.head_dim
                )
            )

        else:
            raise ValueError(
                "Qwen25VLDecoderAttentionWithRope only supports tensor parallel and replicate sharding strategy"
            )

        self._sharding_strategy = sharding_strategy

    def shard(
        self, devices: Iterable[DeviceRef]
    ) -> list[Qwen25VLDecoderAttentionWithRope]:
        """Creates sharded views of this attention layer across multiple devices.

        Args:
            devices: Iterable of devices to place the shards on.

        Returns:
            List of sharded Gemma3Attention instances, one for each device.
        """
        if not self.sharding_strategy:
            raise ValueError(
                "Qwen25VLDecoderAttentionWithRope layer cannot be sharded because no sharding strategy was provided."
            )

        # Get sharded weights
        q_proj_shards = self.q_proj.shard(devices)
        k_proj_shards = self.k_proj.shard(devices)
        v_proj_shards = self.v_proj.shard(devices)
        o_proj_shards = self.o_proj.shard(devices)

        shards = []
        for shard_idx, device in enumerate(devices):
            # Calculate sharded dimensions - handle uneven head distribution
            # Calculate the number of heads for this device
            head_start, head_end = _compute_shard_range(
                self.n_heads, shard_idx, len(self.devices)
            )
            sharded_num_heads = head_end - head_start

            sharded_head_start, sharded_head_end = _compute_shard_range(
                self.kv_params.n_kv_heads,
                shard_idx,
                len(self.devices),
            )
            sharded_num_kv_heads = sharded_head_end - sharded_head_start

            # Create new attention instance with sharded configuration
            sharded = Qwen25VLDecoderAttentionWithRope(
                rope=self.rope,
                num_attention_heads=sharded_num_heads,
                num_key_value_heads=sharded_num_kv_heads,
                hidden_size=self.hidden_size,
                kv_params=self.kv_params,
                dtype=self.q_proj.weight.dtype,
                devices=[device],
                linear_cls=self.o_proj.__class__,
                scale=self.scale,
                has_bias=self.has_bias,
                float8_config=self.float8_config,
            )

            # Assign sharded weights
            sharded.q_proj = q_proj_shards[shard_idx]
            sharded.k_proj = k_proj_shards[shard_idx]
            sharded.v_proj = v_proj_shards[shard_idx]
            sharded.o_proj = o_proj_shards[shard_idx]

            shards.append(sharded)

        return shards



class QwenImageTransformerBlock(Module):
    """QwenImageTransformerBlock customized for supporting 2D position ids."""

    def __init__(
        self,
        attention: QwenImageAttentionWithRope,
        mlp: ShardableCallable,
        attention_norm: ShardableCallable,
        mlp_norm: ShardableCallable,
        devices: list[DeviceRef],
    ) -> None:
        super().__init__()
        self.self_attn = attention
        self.self_attn.sharding_strategy = ShardingStrategy.tensor_parallel(
            len(devices)
        )
        self.self_attn_shards = attention.shard(devices)

        self.mlp = mlp
        self.mlp.sharding_strategy = ShardingStrategy.tensor_parallel(
            len(devices)
        )
        self.mlp_shards = mlp.shard(devices)

        self.input_layernorm = attention_norm
        self.input_layernorm.sharding_strategy = ShardingStrategy.replicate(
            len(devices)
        )
        self.input_layernorm_shards = attention_norm.shard(devices)

        self.post_attention_layernorm = mlp_norm

        self.post_attention_layernorm.sharding_strategy = (
            ShardingStrategy.replicate(len(devices))
        )
        self.post_attention_layernorm_shards = mlp_norm.shard(devices)

        self.devices = devices
        self.allreduce = Allreduce(num_accelerators=len(devices))

    def __call__(
        self,
        layer_idx: TensorValue,
        xs: list[TensorValue],
        kv_collections: list[PagedCacheValues],
        freqs_cis: list[TensorValue],
        input_row_offsets: list[TensorValue],
        position_ids: TensorValue,
        mrope_section: list[int],
        signal_buffers: list[BufferValue],
    ) -> list[TensorValue]:
        norm_xs = forward_sharded_layers(self.input_layernorm_shards, xs)

        attn_out = [
            shard(
                layer_idx,
                norm_xs[i],
                kv_collections[i],
                freqs_cis=freqs_cis[i],
                input_row_offsets=input_row_offsets[i],
                # TODO: how to pass position_ids and mrope_section to each shard?
                position_ids=position_ids,
                mrope_section=mrope_section,
            )
            for i, shard in enumerate(self.self_attn_shards)
        ]
        attn_outs = self.allreduce(attn_out, signal_buffers)

        hs = [x + attn_out for x, attn_out in zip(xs, attn_outs, strict=True)]

        # Apply post attention layer norm to each shard
        norm_outs = forward_sharded_layers(
            self.post_attention_layernorm_shards, hs
        )
        mlp_outs = forward_sharded_layers(self.mlp_shards, norm_outs)

        mlp_outs = self.allreduce(mlp_outs, signal_buffers)

        hs = [h + mlp_out for h, mlp_out in zip(hs, mlp_outs, strict=True)]

        return hs


class QwenImageTransformer2DModel(
    Module,
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin
):
    """
    The Transformer model introduced in Qwen.

    Args:
        patch_size (`int`, defaults to `2`):
            Patch size to turn the input data into small patches.
        in_channels (`int`, defaults to `64`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `None`):
            The number of channels in the output. If not specified, it defaults to `in_channels`.
        num_layers (`int`, defaults to `60`):
            The number of layers of dual stream DiT blocks to use.
        attention_head_dim (`int`, defaults to `128`):
            The number of dimensions to use for each attention head.
        num_attention_heads (`int`, defaults to `24`):
            The number of attention heads to use.
        joint_attention_dim (`int`, defaults to `3584`):
            The number of dimensions to use for the joint attention (embedding/channel dimension of
            `encoder_hidden_states`).
        guidance_embeds (`bool`, defaults to `False`):
            Whether to use guidance embeddings for guidance-distilled variant of the model.
        axes_dims_rope (`Tuple[int]`, defaults to `(16, 56, 56)`):
            The dimensions to use for the rotary positional embeddings.
    """

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: Optional[int] = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        guidance_embeds: bool = False,  # TODO: this should probably be removed
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = QwenEmbedRope(theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True)

        self.time_text_embed = QwenTimestepProjEmbeddings(embedding_dim=self.inner_dim)

        self.txt_norm = RMSNorm(joint_attention_dim, eps=1e-6)

        self.img_in = nn.Linear(in_channels, self.inner_dim)
        self.txt_in = nn.Linear(joint_attention_dim, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                QwenImageTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        txt_seq_lens: Optional[List[int]] = None,
        guidance: torch.Tensor = None,  # TODO: this should probably be removed
        attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        The [`QwenTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            encoder_hidden_states_mask (`torch.Tensor` of shape `(batch_size, text_sequence_length)`):
                Mask of the input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        hidden_states = self.img_in(hidden_states)

        timestep = timestep.cast(hidden_states.dtype)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        if guidance is not None:
            guidance = guidance.cast(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, hidden_states)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states)
        )

        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)

        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    encoder_hidden_states_mask,
                    temb,
                    image_rotary_emb,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=attention_kwargs,
                )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        # Use only the image part (hidden_states) from the dual-stream blocks
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
