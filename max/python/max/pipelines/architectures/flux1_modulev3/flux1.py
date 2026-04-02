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
from collections.abc import Callable, Sequence

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Linear, Module
from max.experimental.nn.norm import LayerNorm
from max.experimental.nn.sequential import ModuleList
from max.experimental.tensor import Tensor
from max.graph import TensorType

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


from max.pipelines.lib.interfaces.cache_mixin import (
    fbcache_conditional_execution,
    teacache_conditional_execution,
    teacache_rescaled_delta,
)


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
        dtype = config.dtype
        self.patch_size = patch_size
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

        self.max_dtype = dtype
        self.in_channels = in_channels
        self.joint_attention_dim = joint_attention_dim
        self.pooled_projection_dim = pooled_projection_dim

        # Cache routing: pick the forward/input_types path once before
        # compilation.  model.py sets these based on cache_config.
        self._forward_impl: Callable[..., tuple[Tensor, ...]] = (
            self._forward_standard
        )
        self._input_types_impl: Callable[..., tuple[TensorType, ...]] = (
            self._input_types_standard
        )
        self._teacache_rel_l1_thresh: float = 0.2
        self._teacache_coefficients: tuple[float, ...] = ()

    def _fbcache_conditional_execution_output_types(self) -> list[TensorType]:
        """Return [residual_type, output_type] for fbcache_conditional_execution / input_types."""
        residual_type = TensorType(
            self.max_dtype,
            shape=["batch_size", "image_seq_len", self.inner_dim],
            device=self.device,
        )
        output_type = TensorType(
            self.max_dtype,
            shape=[
                "batch_size",
                "image_seq_len",
                self.patch_size * self.patch_size * self.out_channels,
            ],
            device=self.device,
        )
        return [residual_type, output_type]

    def _teacache_output_types(self) -> list[TensorType]:
        """Return TeaCache output types for updated cache state + output."""
        image_hidden_type = TensorType(
            self.max_dtype,
            shape=["batch_size", "image_seq_len", self.inner_dim],
            device=self.device,
        )
        accum_type = TensorType(DType.float32, shape=[1], device=self.device)
        output_type = TensorType(
            self.max_dtype,
            shape=[
                "batch_size",
                "image_seq_len",
                self.patch_size * self.patch_size * self.out_channels,
            ],
            device=self.device,
        )
        return [image_hidden_type, image_hidden_type, accum_type, output_type]

    def _teacache_modulated_input(
        self,
        hidden_states: Tensor,
        temb: Tensor,
    ) -> Tensor:
        """Compute TeaCache's modulated input from the first block's AdaLN."""
        norm_hidden_states, _gate, _shift, _scale, _gate2 = (
            self.transformer_blocks[0].norm1(hidden_states, emb=temb)
        )
        return norm_hidden_states

    def _base_input_types(self) -> tuple[TensorType, ...]:
        """Return the base input types shared by all forward paths."""
        return (
            TensorType(
                self.max_dtype,
                shape=["batch_size", "image_seq_len", self.in_channels],
                device=self.device,
            ),
            TensorType(
                self.max_dtype,
                shape=[
                    "batch_size",
                    "text_seq_len",
                    self.joint_attention_dim,
                ],
                device=self.device,
            ),
            TensorType(
                self.max_dtype,
                shape=["batch_size", self.pooled_projection_dim],
                device=self.device,
            ),
            TensorType(DType.float32, shape=["batch_size"], device=self.device),
            TensorType(
                self.max_dtype,
                shape=["image_seq_len", 3],
                device=self.device,
            ),
            TensorType(
                self.max_dtype,
                shape=["text_seq_len", 3],
                device=self.device,
            ),
            TensorType(
                self.max_dtype, shape=["batch_size"], device=self.device
            ),
        )

    def _input_types_standard(self) -> tuple[TensorType, ...]:
        return self._base_input_types()

    def _input_types_fbcache(self) -> tuple[TensorType, ...]:
        rdt_type = TensorType(DType.float32, shape=[], device=self.device)
        return (
            self._base_input_types()
            + tuple(self._fbcache_conditional_execution_output_types())
            + (rdt_type,)
        )

    def _input_types_teacache(self) -> tuple[TensorType, ...]:
        image_hidden_type = TensorType(
            self.max_dtype,
            shape=["batch_size", "image_seq_len", self.inner_dim],
            device=self.device,
        )
        accum_type = TensorType(DType.float32, shape=[1], device=self.device)
        force_compute_type = TensorType(
            DType.bool, shape=[1], device=self.device
        )
        return self._base_input_types() + (
            image_hidden_type,  # prev_modulated_input
            image_hidden_type,  # prev_residual
            accum_type,  # accumulated_rel_l1
            force_compute_type,  # force_compute
        )

    def input_types(self) -> tuple[TensorType, ...]:
        """Define input tensor types for the model."""
        return self._input_types_impl()

    def _run_first_block(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        temb: Tensor,
        image_rotary_emb: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Run the first dual-stream transformer block.

        Returns:
            (first_encoder_hidden_states, first_hidden_states).
        """
        return self.transformer_blocks[0](
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
        )

    def _run_remaining_blocks(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        temb: Tensor,
        image_rotary_emb: tuple[Tensor, Tensor],
    ) -> Tensor:
        """Run remaining dual-stream blocks 1..N and single-stream blocks.

        Returns:
            Pre-postamble image hidden states [B, image_seq_len, inner_dim].
        """
        for rem_block in self.transformer_blocks[1:]:
            encoder_hidden_states, hidden_states = rem_block(
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
        return hidden_states

    def _forward_postamble(self, hidden_states: Tensor, temb: Tensor) -> Tensor:
        """Final norm and projection after the transformer backbone."""
        return self.proj_out(self.norm_out(hidden_states, temb))

    def _forward_preamble(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        pooled_projections: Tensor,
        timestep: Tensor,
        img_ids: Tensor,
        txt_ids: Tensor,
        guidance: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, tuple[Tensor, Tensor]]:
        """Embeddings, projection, RoPE.

        Returns:
            (projected_hidden_states, encoder_hidden_states, temb,
             image_rotary_emb).
        """
        hidden_states = self.x_embedder(hidden_states)

        timestep = F.cast(timestep, hidden_states.dtype)
        timestep = timestep * 1000.0
        if guidance is not None:
            guidance = F.cast(guidance, hidden_states.dtype) * 1000.0

        if self.guidance_embeds:
            assert isinstance(
                self.time_text_embed,
                CombinedTimestepGuidanceTextProjEmbeddings,
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

        return (hidden_states, encoder_hidden_states, temb, image_rotary_emb)

    def _forward_standard(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        pooled_projections: Tensor,
        timestep: Tensor,
        img_ids: Tensor,
        txt_ids: Tensor,
        guidance: Tensor | None = None,
    ) -> tuple[Tensor]:
        """Standard forward pass (no step-cache)."""
        projected, encoder_hidden_states, temb, image_rotary_emb = (
            self._forward_preamble(
                hidden_states,
                encoder_hidden_states,
                pooled_projections,
                timestep,
                img_ids,
                txt_ids,
                guidance,
            )
        )
        first_encoder, first_hidden = self._run_first_block(
            projected, encoder_hidden_states, temb, image_rotary_emb
        )
        image_hidden = self._run_remaining_blocks(
            first_hidden, first_encoder, temb, image_rotary_emb
        )
        return (self._forward_postamble(image_hidden, temb),)

    def _forward_fbcache(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        pooled_projections: Tensor,
        timestep: Tensor,
        img_ids: Tensor,
        txt_ids: Tensor,
        guidance: Tensor | None,
        prev_residual: Tensor,
        prev_output: Tensor,
        residual_threshold: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Step-cache forward pass with F.cond branching for cache reuse."""
        projected, encoder_hidden_states, temb, image_rotary_emb = (
            self._forward_preamble(
                hidden_states,
                encoder_hidden_states,
                pooled_projections,
                timestep,
                img_ids,
                txt_ids,
                guidance,
            )
        )
        first_encoder, first_hidden = self._run_first_block(
            projected, encoder_hidden_states, temb, image_rotary_emb
        )
        first_block_residual = first_hidden - projected

        return fbcache_conditional_execution(
            first_block_residual,
            prev_residual,
            prev_output,
            residual_threshold,
            self._run_remaining_blocks,
            dict(
                hidden_states=first_hidden,
                encoder_hidden_states=first_encoder,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            ),
            self._forward_postamble,
            temb,
            self._fbcache_conditional_execution_output_types(),
        )

    def _forward_teacache(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        pooled_projections: Tensor,
        timestep: Tensor,
        img_ids: Tensor,
        txt_ids: Tensor,
        guidance: Tensor | None,
        prev_modulated_input: Tensor,
        prev_residual: Tensor,
        accumulated_rel_l1: Tensor,
        force_compute: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """TeaCache forward pass: skip decision gates the entire DiT backbone."""
        projected, encoder_hidden_states, temb, image_rotary_emb = (
            self._forward_preamble(
                hidden_states,
                encoder_hidden_states,
                pooled_projections,
                timestep,
                img_ids,
                txt_ids,
                guidance,
            )
        )
        modulated_input = self._teacache_modulated_input(projected, temb)

        delta = teacache_rescaled_delta(
            modulated_input,
            prev_modulated_input,
            self._teacache_coefficients,
        )
        next_accumulated = accumulated_rel_l1 + delta

        return teacache_conditional_execution(
            modulated_input=modulated_input,
            next_accumulated=next_accumulated,
            accumulated_rel_l1=accumulated_rel_l1,
            force_compute=force_compute,
            rel_l1_thresh=self._teacache_rel_l1_thresh,
            projected_hidden_states=projected,
            prev_residual=prev_residual,
            temb=temb,
            run_first_block=self._run_first_block,
            first_block_kwargs=dict(
                hidden_states=projected,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            ),
            run_remaining_blocks=self._run_remaining_blocks,
            remaining_blocks_kwargs=dict(
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            ),
            run_postamble=self._forward_postamble,
            output_types=self._teacache_output_types(),
        )

    def forward(self, *args: Tensor) -> tuple[Tensor, ...]:
        """Forward pass, dispatched to standard or step-cache impl."""
        return self._forward_impl(*args)
