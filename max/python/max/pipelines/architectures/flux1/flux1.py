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

import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from os import PathLike
from typing import Any

import max.nn as nn
from max.driver import DLPackArray
from max.dtype import DType
from max.graph import DeviceRef, TensorType, TensorValue, ops
from max.graph.weights import SafetensorWeights
from max.nn import LayerNorm, Module

from .layers.embeddings import (
    CombinedTimestepGuidanceTextProjEmbeddings,
    CombinedTimestepTextProjEmbeddings,
)
from .layers.flux_attention import FeedForward, FluxAttention, FluxPosEmbed
from .layers.normalizations import (
    AdaLayerNormContinuous,
    AdaLayerNormZero,
    AdaLayerNormZeroSingle,
)
from .model_config import FluxConfig

logger = logging.getLogger(__name__)


def get_weight_registry_from_diffusers(
    safe_tensor_folder: PathLike,
) -> dict[str, DLPackArray]:
    weight_files = [
        os.path.join(safe_tensor_folder, f)
        for f in os.listdir(safe_tensor_folder)
        if f.endswith(".safetensors")
    ]
    weights = SafetensorWeights(weight_files)
    return {name: weight.data().data for name, weight in weights.items()}


class FluxSingleTransformerBlock(Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        device: DeviceRef = DeviceRef.CPU(),
        dtype: DType = DType.bfloat16,
    ):
        """Initialize Flux single transformer block.

        Args:
            dim: Dimension of the input/output.
            num_attention_heads: Number of attention heads.
            attention_head_dim: Dimension of each attention head.
            mlp_ratio: Ratio for MLP hidden dimension.
            device: Device to place the module on.
            dtype: Data type for the module.
        """
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim, device=device, dtype=dtype)
        self.proj_mlp = nn.Linear(
            dim, self.mlp_hidden_dim, has_bias=True, device=device, dtype=dtype
        )
        self.act_mlp = ops.gelu
        self.proj_out = nn.Linear(
            dim + self.mlp_hidden_dim,
            dim,
            has_bias=True,
            device=device,
            dtype=dtype,
        )
        self.attn = FluxAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            eps=1e-6,
            pre_only=True,
            device=device,
            dtype=dtype,
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        encoder_hidden_states: TensorValue,
        temb: TensorValue,
        image_rotary_emb: tuple[TensorValue, TensorValue] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[TensorValue, TensorValue]:
        """Apply single transformer block with attention and MLP.

        Args:
            hidden_states: Input hidden states.
            encoder_hidden_states: Encoder hidden states for cross-attention.
            temb: Time embedding.
            image_rotary_emb: Optional rotary position embeddings.
            joint_attention_kwargs: Optional attention kwargs.

        Returns:
            Tuple of (encoder_hidden_states, hidden_states).
        """
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = ops.concat(
            [encoder_hidden_states, hidden_states], axis=1
        )

        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(
            self.proj_mlp(norm_hidden_states), approximate="tanh"
        )
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        hidden_states = ops.concat([attn_output, mlp_hidden_states], axis=2)
        gate = ops.unsqueeze(gate, 1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == DType.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        encoder_hidden_states, hidden_states = (
            hidden_states[:, :text_seq_len],
            hidden_states[:, text_seq_len:],
        )
        return encoder_hidden_states, hidden_states


class FluxTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
        device: DeviceRef = DeviceRef.CPU(),
        dtype: DType = DType.bfloat16,
    ):
        """Initialize Flux transformer block.

        Args:
            dim: Dimension of the input/output.
            num_attention_heads: Number of attention heads.
            attention_head_dim: Dimension of each attention head.
            qk_norm: Type of normalization for query and key ("rms_norm").
            eps: Epsilon for normalization layers.
            device: Device to place the module on.
            dtype: Data type for the module.
        """
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim, device=device, dtype=dtype)
        self.norm1_context = AdaLayerNormZero(dim, device=device, dtype=dtype)

        self.attn = FluxAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            eps=eps,
            device=device,
            dtype=dtype,
        )

        self.norm2 = LayerNorm(
            dim,
            eps=1e-6,
            devices=[device],
            dtype=dtype,
            keep_dtype=True,
            elementwise_affine=False,
        )
        self.ff = FeedForward(
            dim=dim,
            dim_out=dim,
            activation_fn="gelu-approximate",
            device=device,
            dtype=dtype,
        )

        self.norm2_context = LayerNorm(
            dim,
            eps=1e-6,
            devices=[device],
            dtype=dtype,
            keep_dtype=True,
            elementwise_affine=False,
        )
        self.ff_context = FeedForward(
            dim=dim,
            dim_out=dim,
            activation_fn="gelu-approximate",
            device=device,
            dtype=dtype,
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        encoder_hidden_states: TensorValue,
        temb: TensorValue,
        image_rotary_emb: tuple[TensorValue, TensorValue] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[TensorValue, TensorValue]:
        """Apply transformer block with dual-stream attention and feedforward.

        Args:
            hidden_states: Input hidden states.
            encoder_hidden_states: Encoder hidden states for cross-attention.
            temb: Time embedding.
            image_rotary_emb: Optional rotary position embeddings.
            joint_attention_kwargs: Optional attention kwargs.

        Returns:
            Tuple of (encoder_hidden_states, hidden_states).
        """
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.norm1(hidden_states, emb=temb)
        )

        (
            norm_encoder_hidden_states,
            c_gate_msa,
            c_shift_mlp,
            c_scale_mlp,
            c_gate_mlp,
        ) = self.norm1_context(encoder_hidden_states, emb=temb)
        joint_attention_kwargs = joint_attention_kwargs or {}

        # Attention.
        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        attn_output, context_attn_output = attention_outputs

        # Process attention outputs for the `hidden_states`.
        attn_output = ops.unsqueeze(gate_msa, 1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = (
            norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        )

        ff_output = self.ff(norm_hidden_states)
        ff_output = ops.unsqueeze(gate_mlp, 1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.
        context_attn_output = ops.unsqueeze(c_gate_msa, 1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = (
            norm_encoder_hidden_states * (1 + c_scale_mlp[:, None])
            + c_shift_mlp[:, None]
        )

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = (
            encoder_hidden_states
            + ops.unsqueeze(c_gate_mlp, 1) * context_ff_output
        )
        if encoder_hidden_states.dtype == DType.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class FluxTransformer2DModel(nn.Module):
    def __init__(
        self,
        config: FluxConfig,
    ):
        """Initialize Flux Transformer 2D model.

        Args:
            config: Flux configuration containing model dimensions, attention
                settings, and device/dtype information.
        """
        super().__init__()
        patch_size = config.patch_size
        in_channels = config.in_channels
        out_channels = config.out_channels
        num_layers = config.num_layers
        num_single_layers = config.num_single_layers
        attention_head_dim = config.attention_head_dim
        num_attention_heads = config.num_attention_heads
        joint_attention_dim = config.joint_attention_dim
        pooled_projection_dim = config.pooled_projection_dim
        guidance_embeds = config.guidance_embeds
        axes_dims_rope = config.axes_dims_rope
        device = config.device
        dtype = config.dtype
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)
        self.guidance_embeds = guidance_embeds

        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings
            if guidance_embeds
            else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim,
            pooled_projection_dim=pooled_projection_dim,
            device=device,
            dtype=dtype,
        )
        self.context_embedder = nn.Linear(
            joint_attention_dim,
            self.inner_dim,
            has_bias=True,
            device=device,
            dtype=dtype,
        )
        self.x_embedder = nn.Linear(
            in_channels,
            self.inner_dim,
            has_bias=True,
            device=device,
            dtype=dtype,
        )

        self.transformer_blocks = nn.Sequential(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        self.single_transformer_blocks = nn.Sequential(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim, self.inner_dim, eps=1e-6, device=device, dtype=dtype
        )
        self.proj_out = nn.Linear(
            self.inner_dim,
            patch_size * patch_size * self.out_channels,
            has_bias=True,
            device=device,
            dtype=dtype,
        )

        self.gradient_checkpointing = False

        self.max_device = device
        self.max_dtype = dtype
        self.in_channels = in_channels
        self.joint_attention_dim = joint_attention_dim
        self.pooled_projection_dim = pooled_projection_dim

        self._cache_context_warning_shown = False

    def input_types(self) -> tuple[TensorType, ...]:
        """Define input tensor types for the model.

        Returns:
            Tuple of TensorType specifications for all model inputs.
        """
        hidden_states_type = TensorType(
            self.max_dtype,
            shape=["batch_size", "image_seq_len", self.in_channels],
            device=self.max_device,
        )
        encoder_hidden_states_type = TensorType(
            self.max_dtype,
            shape=["batch_size", "text_seq_len", self.joint_attention_dim],
            device=self.max_device,
        )
        pooled_projections_type = TensorType(
            self.max_dtype,
            shape=["batch_size", self.pooled_projection_dim],
            device=self.max_device,
        )
        timestep_type = TensorType(
            DType.float32, shape=["batch_size"], device=self.max_device
        )
        img_ids_type = TensorType(
            self.max_dtype, shape=["image_seq_len", 3], device=self.max_device
        )
        txt_ids_type = TensorType(
            self.max_dtype, shape=["text_seq_len", 3], device=self.max_device
        )
        guidance_type = TensorType(
            self.max_dtype, shape=["batch_size"], device=self.max_device
        )

        return (
            hidden_states_type,
            encoder_hidden_states_type,
            pooled_projections_type,
            timestep_type,
            img_ids_type,
            txt_ids_type,
            guidance_type,
        )

    @contextmanager
    def cache_context(self, name: str) -> Generator[None, None, None]:
        """Context manager for cache control (not implemented in MAX).

        Args:
            name: Name of the cache context.

        Yields:
            None.
        """
        if not self._cache_context_warning_shown:
            logger.warning(
                "cache_context is not implemented in MAX FluxTransformer2DModel. "
                "Caching optimizations are disabled."
            )
            self._cache_context_warning_shown = True
        yield

    def __call__(
        self,
        hidden_states: TensorValue,
        encoder_hidden_states: TensorValue = None,
        pooled_projections: TensorValue = None,
        timestep: TensorValue = None,
        img_ids: TensorValue = None,
        txt_ids: TensorValue = None,
        guidance: TensorValue = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        controlnet_block_samples: Any | None = None,
        controlnet_single_block_samples: Any | None = None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> tuple[TensorValue]:
        """Apply Flux Transformer 2D model forward pass.

        Args:
            hidden_states: Input latent hidden states.
            encoder_hidden_states: Text encoder hidden states.
            pooled_projections: Pooled text embeddings.
            timestep: Diffusion timestep.
            img_ids: Image position IDs.
            txt_ids: Text position IDs.
            guidance: Guidance scale values.
            joint_attention_kwargs: Additional attention arguments.
            controlnet_block_samples: Optional controlnet block samples.
            controlnet_single_block_samples: Optional controlnet single block samples.
            return_dict: Whether to return as dictionary.
            controlnet_blocks_repeat: Whether to repeat controlnet blocks.

        Returns:
            Tuple containing output tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()

        hidden_states = self.x_embedder(hidden_states)

        timestep = ops.cast(timestep, hidden_states.dtype)
        timestep = timestep * 1000.0
        if guidance is not None:
            guidance = guidance.cast(hidden_states.dtype) * 1000.0

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if not self.guidance_embeds
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        ids = ops.concat((txt_ids, img_ids), axis=0)
        image_rotary_emb = self.pos_embed(ids)

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        for block in self.single_transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        return (output,)
