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

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import TensorType
from max.nn.module_v3 import Linear, Module
from max.nn.module_v3.norm import LayerNorm
from max.nn.module_v3.sequential import ModuleList

from .layers.embeddings import TimestepEmbedding, Timesteps
from .layers.flux2_attention import (
    Flux2Attention,
    Flux2FeedForward,
    Flux2ParallelSelfAttention,
    Flux2PosEmbed,
)
from .layers.normalizations import AdaLayerNormContinuous
from .model_config import Flux2ConfigBase


class Flux2TimestepGuidanceEmbeddings(Module[[Tensor, Tensor], Tensor]):
    def __init__(
        self,
        in_channels: int = 256,
        embedding_dim: int = 6144,
        bias: bool = False,
    ):
        """Initialize Flux2TimestepGuidanceEmbeddings.

        Args:
            in_channels: Number of sinusoidal channels.
            embedding_dim: Output embedding dimension.
            bias: Whether to use bias in MLP layers.
        """
        self.time_proj = Timesteps(
            num_channels=in_channels,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=embedding_dim,
            sample_proj_bias=bias,
        )
        self.guidance_embedder = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=embedding_dim,
            sample_proj_bias=bias,
        )

    def forward(self, timestep: Tensor, guidance: Tensor) -> Tensor:
        """Compute combined timestep and guidance embeddings.

        Args:
            timestep: Timestep values of shape [B].
            guidance: Guidance scale values of shape [B].

        Returns:
            Combined embedding of shape [B, embedding_dim].
        """
        # Project timesteps to sinusoidal embeddings
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(
            timesteps_proj.cast(timestep.dtype)
        )

        # Project guidance to sinusoidal embeddings
        guidance_proj = self.time_proj(guidance)
        guidance_emb = self.guidance_embedder(
            guidance_proj.cast(guidance.dtype)
        )

        # Combine embeddings
        time_guidance_emb = timesteps_emb + guidance_emb

        return time_guidance_emb


class Flux2Modulation(
    Module[[Tensor], tuple[tuple[Tensor, Tensor, Tensor], ...]]
):
    def __init__(
        self,
        dim: int,
        mod_param_sets: int = 2,
        bias: bool = False,
    ):
        """Initialize Flux2Modulation.

        Args:
            dim: Input/output dimension.
            mod_param_sets: Number of parameter sets (2 for dual-stream, 1 for single-stream).
            bias: Whether to use bias in linear layer.
        """
        self.mod_param_sets = mod_param_sets
        self.linear = Linear(dim, dim * 3 * mod_param_sets, bias=bias)

    def forward(
        self, temb: Tensor
    ) -> tuple[tuple[Tensor, Tensor, Tensor], ...]:
        """Generate modulation parameters from timestep embedding.

        Args:
            temb: Timestep embedding of shape [B, dim] or [B, 1, dim].

        Returns:
            Tuple of mod_param_sets tuples, each containing (shift, scale, gate).
        """
        # Apply SiLU activation
        mod = F.silu(temb)

        # Project to modulation parameters
        mod = self.linear(mod)

        # Ensure 3D shape: [B, 1, dim * 3 * mod_param_sets]
        if len(mod.shape) == 2:
            mod = F.unsqueeze(mod, 1)

        # Split into 3 * mod_param_sets chunks
        # Each set gets: shift, scale, gate
        mod_params = F.chunk(mod, 3 * self.mod_param_sets, axis=-1)

        # Group into tuples of (shift, scale, gate)
        result = tuple(
            (mod_params[3 * i], mod_params[3 * i + 1], mod_params[3 * i + 2])
            for i in range(self.mod_param_sets)
        )

        return result


class Flux2TransformerBlock(Module[..., tuple[Tensor, Tensor]]):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        bias: bool = False,
    ):
        """Initialize Flux2TransformerBlock.

        Args:
            dim: Hidden dimension size.
            num_attention_heads: Number of attention heads.
            attention_head_dim: Dimension of each attention head.
            mlp_ratio: Multiplier for feedforward hidden dimension.
            eps: Epsilon for layer normalization.
            bias: Whether to use bias in linear layers.
        """
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        # Normalizations (elementwise_affine=False for all)
        self.norm1 = LayerNorm(
            dim, eps=eps, elementwise_affine=False, use_bias=False
        )
        self.norm1_context = LayerNorm(
            dim, eps=eps, elementwise_affine=False, use_bias=False
        )

        # Dual-stream attention
        self.attn = Flux2Attention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=bias,
            added_proj_bias=bias,
            out_bias=bias,
            eps=eps,
        )

        # Feedforward layers
        self.norm2 = LayerNorm(
            dim, eps=eps, elementwise_affine=False, use_bias=False
        )
        self.ff = Flux2FeedForward(
            dim=dim, dim_out=dim, mult=mlp_ratio, bias=bias
        )

        self.norm2_context = LayerNorm(
            dim, eps=eps, elementwise_affine=False, use_bias=False
        )
        self.ff_context = Flux2FeedForward(
            dim=dim, dim_out=dim, mult=mlp_ratio, bias=bias
        )

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        temb_mod_params_img: tuple[
            tuple[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]
        ],
        temb_mod_params_txt: tuple[
            tuple[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]
        ],
        image_rotary_emb: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass for dual-stream transformer block.

        Args:
            hidden_states: Image tokens of shape [B, S_img, D].
            encoder_hidden_states: Text tokens of shape [B, S_txt, D].
            temb_mod_params_img: ((shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp)) for image stream.
            temb_mod_params_txt: Same structure as temb_mod_params_img for text stream.
            image_rotary_emb: Optional (cos, sin) tuple for rotary embeddings.

        Returns:
            Tuple of (encoder_hidden_states, hidden_states) - (text, image) outputs.
        """
        # Unpack modulation parameters
        (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = (
            temb_mod_params_img
        )
        (
            (c_shift_msa, c_scale_msa, c_gate_msa),
            (c_shift_mlp, c_scale_mlp, c_gate_mlp),
        ) = temb_mod_params_txt

        # === Image stream - Attention ===
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = (1 + scale_msa) * norm_hidden_states + shift_msa

        # === Text stream - Attention ===
        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)
        norm_encoder_hidden_states = (
            1 + c_scale_msa
        ) * norm_encoder_hidden_states + c_shift_msa

        # === Dual-stream attention ===
        attn_result = self.attn.forward(
            norm_hidden_states,
            norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )
        if isinstance(attn_result, tuple):
            attn_output, context_attn_output = attn_result
        else:
            raise ValueError("Expected tuple from dual-stream attention")

        # === Image stream - Apply gate and residual ===
        attn_output = gate_msa * attn_output
        hidden_states = hidden_states + attn_output

        # === Image stream - Feedforward ===
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + gate_mlp * ff_output

        # === Text stream - Apply gate and residual ===
        context_attn_output = c_gate_msa * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        # === Text stream - Feedforward ===
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = (
            norm_encoder_hidden_states * (1 + c_scale_mlp) + c_shift_mlp
        )

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = (
            encoder_hidden_states + c_gate_mlp * context_ff_output
        )

        # === FP16 clipping for text stream ===
        if encoder_hidden_states.dtype == DType.float16:
            encoder_hidden_states = encoder_hidden_states.clip(
                min=-65504, max=65504
            )

        return encoder_hidden_states, hidden_states


class Flux2SingleTransformerBlock(Module[..., Tensor | tuple[Tensor, Tensor]]):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        bias: bool = False,
    ):
        """Initialize Flux2SingleTransformerBlock.

        Args:
            dim: Hidden dimension size.
            num_attention_heads: Number of attention heads.
            attention_head_dim: Dimension of each attention head.
            mlp_ratio: Multiplier for feedforward hidden dimension.
            eps: Epsilon for layer normalization.
            bias: Whether to use bias in linear layers.
        """
        # Single normalization (elementwise_affine=False)
        self.norm = LayerNorm(
            dim, eps=eps, elementwise_affine=False, use_bias=False
        )

        # Parallel attention+MLP
        self.attn = Flux2ParallelSelfAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=bias,
            out_bias=bias,
            eps=eps,
            mlp_ratio=mlp_ratio,
            mlp_mult_factor=2,
        )

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor | None = None,
        temb_mod_params: tuple[Tensor, Tensor, Tensor] | None = None,
        image_rotary_emb: tuple[Tensor, Tensor] | None = None,
        split_hidden_states: bool = False,
        text_seq_len: int | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Forward pass for single-stream transformer block.

        Args:
            hidden_states: Image tokens [B, S_img, D] or concatenated [B, S_txt+S_img, D].
            encoder_hidden_states: Optional text tokens [B, S_txt, D]. If provided, concatenates with hidden_states.
            temb_mod_params: (shift, scale, gate) tuple for modulation.
            image_rotary_emb: Optional (cos, sin) tuple for rotary embeddings.
            split_hidden_states: If True, split output back into text and image.
            text_seq_len: Length of text sequence (required if split_hidden_states=True).

        Returns:
            If split_hidden_states: (encoder_hidden_states, hidden_states) tuple.
            Otherwise: hidden_states (concatenated).
        """
        # Concatenate text and image if separate encoder_hidden_states provided
        if encoder_hidden_states is not None:
            from max.graph import StaticDim

            text_seq_len_dim = encoder_hidden_states.shape[1]
            if isinstance(text_seq_len_dim, StaticDim):
                text_seq_len = text_seq_len_dim.dim
            else:
                # For symbolic dims, we'll need to handle at runtime
                text_seq_len = None
            hidden_states = F.concat(
                [encoder_hidden_states, hidden_states], axis=1
            )

        # Unpack modulation parameters
        if temb_mod_params is None:
            raise ValueError("temb_mod_params cannot be None")
        mod_shift, mod_scale, mod_gate = temb_mod_params

        # Normalize and modulate
        norm_hidden_states = self.norm(hidden_states)
        norm_hidden_states = (1 + mod_scale) * norm_hidden_states + mod_shift

        # Parallel attention+MLP
        attn_output = self.attn.forward(
            norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        # Gate and residual
        hidden_states = hidden_states + mod_gate * attn_output

        # FP16 clipping
        if hidden_states.dtype == DType.float16:
            hidden_states = hidden_states.clip(min=-65504, max=65504)

        # Optionally split back to text and image
        if split_hidden_states:
            encoder_hidden_states = hidden_states[:, :text_seq_len, :]
            hidden_states = hidden_states[:, text_seq_len:, :]
            return encoder_hidden_states, hidden_states
        else:
            return hidden_states


class Flux2Transformer2DModel(Module[..., tuple[Tensor]]):
    def __init__(
        self,
        config: Flux2ConfigBase,
    ):
        """Initialize Flux2Transformer2DModel.

        Args:
            config: Flux2 configuration containing model dimensions, attention
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
        timestep_guidance_channels = config.timestep_guidance_channels
        mlp_ratio = config.mlp_ratio
        axes_dims_rope = config.axes_dims_rope
        rope_theta = config.rope_theta
        device = config.device
        dtype = config.dtype
        eps = config.eps

        self.patch_size = patch_size
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        # 1. Positional embeddings (RoPE)
        self.pos_embed = Flux2PosEmbed(
            theta=rope_theta, axes_dim=axes_dims_rope
        )

        # 2. Timestep and guidance embeddings
        self.time_guidance_embed = Flux2TimestepGuidanceEmbeddings(
            in_channels=timestep_guidance_channels,
            embedding_dim=self.inner_dim,
            bias=False,
        )

        # 3. Modulation layers
        # Two sets of modulation parameters for dual-stream blocks (attn + mlp per stream)
        self.double_stream_modulation_img = Flux2Modulation(
            self.inner_dim, mod_param_sets=2, bias=False
        )
        self.double_stream_modulation_txt = Flux2Modulation(
            self.inner_dim, mod_param_sets=2, bias=False
        )
        # One set for single-stream blocks (parallel attn+mlp)
        self.single_stream_modulation = Flux2Modulation(
            self.inner_dim, mod_param_sets=1, bias=False
        )

        # 4. Input embeddings
        self.x_embedder = Linear(in_channels, self.inner_dim, bias=False)
        self.context_embedder = Linear(
            joint_attention_dim, self.inner_dim, bias=False
        )

        # 5. Dual-stream transformer blocks
        self.transformer_blocks: ModuleList[Flux2TransformerBlock] = ModuleList(
            [
                Flux2TransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    bias=False,
                )
                for _ in range(num_layers)
            ]
        )

        # 6. Single-stream transformer blocks
        self.single_transformer_blocks: ModuleList[
            Flux2SingleTransformerBlock
        ] = ModuleList(
            [
                Flux2SingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    bias=False,
                )
                for _ in range(num_single_layers)
            ]
        )

        # 7. Output layers
        self.norm_out = AdaLayerNormContinuous(
            embedding_dim=self.inner_dim,
            conditioning_embedding_dim=self.inner_dim,
            elementwise_affine=False,
            eps=eps,
            bias=False,
        )
        self.proj_out = Linear(
            self.inner_dim,
            patch_size * patch_size * self.out_channels,
            bias=False,
        )

        # Store config for input_types
        self.max_device = device
        self.max_dtype = dtype
        self.in_channels = in_channels
        self.joint_attention_dim = joint_attention_dim

    def input_types(self) -> tuple[TensorType, ...]:
        """Define input tensor types for the model with symbolic shapes.

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
        timestep_type = TensorType(
            self.max_dtype, shape=["batch_size"], device=self.max_device
        )
        img_ids_type = TensorType(
            DType.int64,
            shape=["batch_size", "image_seq_len", 4],
            device=self.max_device,
        )
        txt_ids_type = TensorType(
            DType.int64,
            shape=["batch_size", "text_seq_len", 4],
            device=self.max_device,
        )
        guidance_type = TensorType(
            self.max_dtype, shape=["batch_size"], device=self.max_device
        )

        return (
            hidden_states_type,
            encoder_hidden_states_type,
            timestep_type,
            img_ids_type,
            txt_ids_type,
            guidance_type,
        )

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep: Tensor,
        img_ids: Tensor,
        txt_ids: Tensor,
        guidance: Tensor,
    ) -> tuple[Tensor]:
        """Forward pass through Flux2 Transformer.

        Args:
            hidden_states: Image latents of shape [B, H*W, in_channels].
            encoder_hidden_states: Text embeddings of shape [B, txt_len, joint_attention_dim].
            timestep: Denoising timestep of shape [B] (scaled to [0, 1] range).
            img_ids: Image position IDs of shape [image_seq_len, 4].
            txt_ids: Text position IDs of shape [text_seq_len, 4].
            guidance: Guidance scale of shape [B] (scaled to [0, 1] range).

        Returns:
            Denoised output of shape [B, H*W, patch_size^2 * out_channels].
        """
        # Handle batch dimension in ids (squeeze if needed)
        if img_ids.rank == 3:
            img_ids = img_ids[0]  # [H*W, 4]
        if txt_ids.rank == 3:
            txt_ids = txt_ids[0]  # [txt_len, 4]

        num_txt_tokens = encoder_hidden_states.shape[1]

        # 1. Calculate timestep embedding and modulation parameters
        # Scale timestep and guidance to [0, 1000] range
        timestep = (timestep * 1000.0).cast(hidden_states.dtype)
        guidance = (guidance * 1000.0).cast(hidden_states.dtype)

        temb = self.time_guidance_embed(timestep, guidance)

        # Generate modulation parameters
        double_stream_mod_img_tuple = self.double_stream_modulation_img(temb)
        double_stream_mod_txt_tuple = self.double_stream_modulation_txt(temb)
        single_stream_mod_tuple = self.single_stream_modulation(temb)
        # Cast to expected types (modulation returns variadic tuple, but we need fixed-length)
        double_stream_mod_img: tuple[
            tuple[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]
        ] = (double_stream_mod_img_tuple[0], double_stream_mod_img_tuple[1])
        double_stream_mod_txt: tuple[
            tuple[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]
        ] = (double_stream_mod_txt_tuple[0], double_stream_mod_txt_tuple[1])
        single_stream_mod = single_stream_mod_tuple[0]

        # 2. Input projection
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # 3. Calculate RoPE embeddings
        # Concatenate text and image position IDs
        ids = F.concat([txt_ids, img_ids], axis=0)
        image_rotary_emb = self.pos_embed(ids)

        # 4. Dual-stream transformer blocks
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod_params_img=double_stream_mod_img,
                temb_mod_params_txt=double_stream_mod_txt,
                image_rotary_emb=image_rotary_emb,
            )

        # 5. Concatenate text and image for single-stream blocks
        hidden_states = F.concat([encoder_hidden_states, hidden_states], axis=1)

        # 6. Single-stream transformer blocks
        for i in range(len(self.single_transformer_blocks)):
            single_block: Flux2SingleTransformerBlock = (
                self.single_transformer_blocks[i]
            )
            hidden_states = single_block(  # type: ignore[assignment]
                hidden_states=hidden_states,
                encoder_hidden_states=None,
                temb_mod_params=single_stream_mod,
                image_rotary_emb=image_rotary_emb,
                split_hidden_states=False,
            )

        # 7. Remove text tokens from concatenated stream
        hidden_states = hidden_states[:, num_txt_tokens:, :]

        # 8. Output projection
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        # Return as tuple for consistency with Flux1 (pipeline expects [0] indexing)
        return (output,)
