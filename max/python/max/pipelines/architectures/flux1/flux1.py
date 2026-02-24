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

import logging
from collections.abc import Sequence

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import TensorType
from max.nn.module_v3 import Linear, Module
from max.nn.module_v3.norm import LayerNorm
from max.nn.module_v3.sequential import ModuleList

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
from .model_config import FluxConfigBase

logger = logging.getLogger(__name__)


class FluxSingleTransformerBlock(Module[..., tuple[Tensor, Tensor]]):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        dtype: DType = DType.bfloat16,
    ):
        """Initialize Flux single transformer block.

        Args:
            dim: Dimension of the input/output.
            num_attention_heads: Number of attention heads.
            attention_head_dim: Dimension of each attention head.
            mlp_ratio: Ratio for MLP hidden dimension.
            dtype: Data type for the module.
        """
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = Linear(dim, self.mlp_hidden_dim, bias=True)
        self.act_mlp = F.gelu
        self.proj_out = Linear(
            dim + self.mlp_hidden_dim,
            dim,
            bias=True,
        )
        self.attn = FluxAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            eps=1e-6,
            pre_only=True,
        )

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        temb: Tensor,
        image_rotary_emb: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply single transformer block with attention and MLP.

        Args:
            hidden_states: Input hidden states.
            encoder_hidden_states: Encoder hidden states for cross-attention.
            temb: Time embedding.
            image_rotary_emb: Optional rotary position embeddings.

        Returns:
            Tuple of (encoder_hidden_states, hidden_states).
        """
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = F.concat([encoder_hidden_states, hidden_states], axis=1)

        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(
            self.proj_mlp(norm_hidden_states), approximate="tanh"
        )

        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = F.concat([attn_output, mlp_hidden_states], axis=2)
        gate = F.unsqueeze(gate, 1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == DType.float16:
            hidden_states = hidden_states.clip(min=-65504, max=65504)

        encoder_hidden_states, hidden_states = (
            hidden_states[:, :text_seq_len],
            hidden_states[:, text_seq_len:],
        )
        return encoder_hidden_states, hidden_states


class FluxTransformerBlock(Module[..., tuple[Tensor, Tensor]]):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
        dtype: DType = DType.bfloat16,
    ):
        """Initialize Flux transformer block.

        Args:
            dim: Dimension of the input/output.
            num_attention_heads: Number of attention heads.
            attention_head_dim: Dimension of each attention head.
            qk_norm: Type of normalization for query and key ("rms_norm").
            eps: Epsilon for normalization layers.
            dtype: Data type for the module.
        """
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim)
        self.norm1_context = AdaLayerNormZero(dim)

        self.attn = FluxAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            eps=eps,
        )

        self.norm2 = LayerNorm(
            dim,
            eps=1e-6,
            keep_dtype=True,
            elementwise_affine=False,
            use_bias=False,
        )
        self.ff = FeedForward(
            dim=dim,
            dim_out=dim,
            activation_fn="gelu-approximate",
        )

        self.norm2_context = LayerNorm(
            dim,
            eps=1e-6,
            keep_dtype=True,
            elementwise_affine=False,
            use_bias=False,
        )
        self.ff_context = FeedForward(
            dim=dim,
            dim_out=dim,
            activation_fn="gelu-approximate",
        )

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        temb: Tensor,
        image_rotary_emb: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply transformer block with dual-stream attention and feedforward.

        Args:
            hidden_states: Input hidden states.
            encoder_hidden_states: Encoder hidden states for cross-attention.
            temb: Time embedding.
            image_rotary_emb: Optional rotary position embeddings.

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

        # Attention.
        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        assert isinstance(attention_outputs, tuple)
        attn_output, context_attn_output = attention_outputs

        # Process attention outputs for the `hidden_states`.
        attn_output = F.unsqueeze(gate_msa, 1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = (
            norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        )

        ff_output = self.ff(norm_hidden_states)
        ff_output = F.unsqueeze(gate_mlp, 1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.
        context_attn_output = F.unsqueeze(c_gate_msa, 1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = (
            norm_encoder_hidden_states * (1 + c_scale_mlp[:, None])
            + c_shift_mlp[:, None]
        )

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = (
            encoder_hidden_states
            + F.unsqueeze(c_gate_mlp, 1) * context_ff_output
        )
        if encoder_hidden_states.dtype == DType.float16:
            encoder_hidden_states = encoder_hidden_states.clip(
                min=-65504, max=65504
            )

        return encoder_hidden_states, hidden_states


class FluxTransformer2DModel(Module[..., Sequence[Tensor]]):
    def __init__(
        self,
        config: FluxConfigBase,
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
        )
        self.context_embedder = Linear(
            joint_attention_dim,
            self.inner_dim,
            bias=True,
        )
        self.x_embedder = Linear(
            in_channels,
            self.inner_dim,
            bias=True,
        )

        self.transformer_blocks: ModuleList[FluxTransformerBlock] = ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        self.single_transformer_blocks: ModuleList[
            FluxSingleTransformerBlock
        ] = ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dtype=dtype,
                )
                for _ in range(num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim, self.inner_dim, eps=1e-6
        )
        self.proj_out = Linear(
            self.inner_dim,
            patch_size * patch_size * self.out_channels,
            bias=True,
        )

        self.gradient_checkpointing = False

        self.max_device = device
        self.max_dtype = dtype
        self.in_channels = in_channels
        self.joint_attention_dim = joint_attention_dim
        self.pooled_projection_dim = pooled_projection_dim

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

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        pooled_projections: Tensor,
        timestep: Tensor,
        img_ids: Tensor,
        txt_ids: Tensor,
        guidance: Tensor | None = None,
    ) -> tuple[Tensor]:
        """Apply Flux Transformer 2D model forward pass.

        Args:
            hidden_states: Input latent hidden states.
            encoder_hidden_states: Text encoder hidden states.
            pooled_projections: Pooled text embeddings.
            timestep: Diffusion timestep.
            img_ids: Image position IDs.
            txt_ids: Text position IDs.
            guidance: Guidance scale values.

        Returns:
            Tuple containing output tensor.
        """

        hidden_states = self.x_embedder(hidden_states)

        timestep = F.cast(timestep, hidden_states.dtype)
        timestep = timestep * 1000.0
        if guidance is not None:
            guidance = F.cast(guidance, hidden_states.dtype) * 1000.0

        if self.guidance_embeds:
            assert isinstance(
                self.time_text_embed, CombinedTimestepGuidanceTextProjEmbeddings
            )
            assert isinstance(guidance, Tensor)
            temb = self.time_text_embed(timestep, guidance, pooled_projections)
        else:
            assert isinstance(
                self.time_text_embed, CombinedTimestepTextProjEmbeddings
            )
            temb = self.time_text_embed(timestep, pooled_projections)

        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        ids = F.concat((txt_ids, img_ids), axis=0)
        image_rotary_emb = self.pos_embed(ids)

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )

        for single_block in self.single_transformer_blocks:
            encoder_hidden_states, hidden_states = single_block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        return (output,)
