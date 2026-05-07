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
from max.graph import DeviceRef, Dim, TensorType, TensorValue, ops
from max.nn.layer import LayerList, Module
from max.nn.linear import Linear
from max.nn.quant_config import QuantConfig

from .layers.embeddings import TimestepEmbedding, Timesteps
from .layers.flux2_attention import (
    Flux2Attention,
    Flux2FeedForward,
    Flux2ParallelSelfAttention,
    Flux2PosEmbed,
)
from .layers.normalizations import AdaLayerNormContinuous, LayerNorm
from .model_config import Flux2BlockQuant, Flux2Config


class Flux2TimestepGuidanceEmbeddings(Module):
    def __init__(
        self,
        *,
        in_channels: int = 256,
        embedding_dim: int = 6144,
        bias: bool = False,
        guidance_embeds: bool = True,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        """Initialize Flux2TimestepGuidanceEmbeddings.

        Args:
            in_channels: Number of sinusoidal channels.
            embedding_dim: Output embedding dimension.
            bias: Whether to use bias in MLP layers.
            guidance_embeds: If True, include guidance embedder.
            dtype: Weight dtype.
            device: Weight device.
        """
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=in_channels,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=embedding_dim,
            sample_proj_bias=bias,
            dtype=dtype,
            device=device,
        )
        if guidance_embeds:
            self.guidance_embedder: TimestepEmbedding | None = (
                TimestepEmbedding(
                    in_channels=in_channels,
                    time_embed_dim=embedding_dim,
                    sample_proj_bias=bias,
                    dtype=dtype,
                    device=device,
                )
            )
        else:
            self.guidance_embedder = None

    def __call__(
        self, timestep: TensorValue, guidance: TensorValue
    ) -> TensorValue:
        """Compute combined timestep and guidance embeddings.

        Args:
            timestep: Timestep values of shape [B].
            guidance: Guidance scale values of shape [B].

        Returns:
            Combined embedding of shape [B, embedding_dim].
        """
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(
            ops.cast(timesteps_proj, timestep.dtype)
        )
        if guidance is not None and self.guidance_embedder is not None:
            guidance_proj = self.time_proj(guidance)
            guidance_emb = self.guidance_embedder(
                ops.cast(guidance_proj, guidance.dtype)
            )
            return timesteps_emb + guidance_emb
        return timesteps_emb


class Flux2Modulation(Module):
    def __init__(
        self,
        dim: int,
        *,
        dtype: DType,
        device: DeviceRef,
        mod_param_sets: int = 2,
        bias: bool = False,
    ) -> None:
        """Initialize Flux2Modulation.

        Args:
            dim: Input/output dimension.
            dtype: Weight dtype.
            device: Weight device.
            mod_param_sets: Number of modulation parameter sets.
            bias: Whether to use bias in the linear layer.
        """
        super().__init__()
        self.mod_param_sets = mod_param_sets
        self.linear = Linear(
            in_dim=dim,
            out_dim=dim * 3 * mod_param_sets,
            dtype=dtype,
            device=device,
            has_bias=bias,
        )

    def __call__(
        self, temb: TensorValue
    ) -> tuple[tuple[TensorValue, TensorValue, TensorValue], ...]:
        """Generate modulation parameters from timestep embedding.

        Args:
            temb: Timestep embedding of shape [B, dim] or [B, 1, dim].

        Returns:
            Tuple of modulation tuples, each containing (shift, scale, gate).
        """
        mod = self.linear(ops.silu(temb))
        if len(mod.shape) == 2:
            mod = ops.unsqueeze(mod, 1)
        mod_params = ops.split(
            mod,
            [temb.shape[-1]] * (3 * self.mod_param_sets),
            axis=-1,
        )
        return tuple(
            (mod_params[3 * i], mod_params[3 * i + 1], mod_params[3 * i + 2])
            for i in range(self.mod_param_sets)
        )


class Flux2TransformerBlock(Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        *,
        dtype: DType,
        device: DeviceRef,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        bias: bool = False,
        quant: Flux2BlockQuant = Flux2BlockQuant(),
    ) -> None:
        """Initialize Flux2TransformerBlock.

        Args:
            dim: Hidden dimension size.
            num_attention_heads: Number of attention heads.
            attention_head_dim: Dimension of each attention head.
            dtype: Weight dtype.
            device: Weight device.
            mlp_ratio: Multiplier for feedforward hidden dimension.
            eps: Epsilon for layer normalization.
            bias: Whether to use bias in linear layers.
            quant: Per-Linear quant plan; default leaves every Linear BF16.
        """
        super().__init__()
        self.norm1 = LayerNorm(
            dim,
            dtype=dtype,
            device=device,
            eps=eps,
            elementwise_affine=False,
            use_bias=False,
        )
        self.norm1_context = LayerNorm(
            dim,
            dtype=dtype,
            device=device,
            eps=eps,
            elementwise_affine=False,
            use_bias=False,
        )
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
            dtype=dtype,
            device=device,
            quant=quant,
        )
        self.norm2 = LayerNorm(
            dim,
            dtype=dtype,
            device=device,
            eps=eps,
            elementwise_affine=False,
            use_bias=False,
        )
        self.ff = Flux2FeedForward(
            dim=dim,
            dim_out=dim,
            mult=mlp_ratio,
            bias=bias,
            dtype=dtype,
            device=device,
            quant_config=quant.ff,
        )
        self.norm2_context = LayerNorm(
            dim,
            dtype=dtype,
            device=device,
            eps=eps,
            elementwise_affine=False,
            use_bias=False,
        )
        self.ff_context = Flux2FeedForward(
            dim=dim,
            dim_out=dim,
            mult=mlp_ratio,
            bias=bias,
            dtype=dtype,
            device=device,
            quant_config=quant.ff_context,
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        encoder_hidden_states: TensorValue,
        temb_mod_params_img: tuple[
            tuple[TensorValue, TensorValue, TensorValue],
            tuple[TensorValue, TensorValue, TensorValue],
        ],
        temb_mod_params_txt: tuple[
            tuple[TensorValue, TensorValue, TensorValue],
            tuple[TensorValue, TensorValue, TensorValue],
        ],
        image_rotary_emb: tuple[TensorValue, TensorValue] | None = None,
    ) -> tuple[TensorValue, TensorValue]:
        """Forward pass for dual-stream transformer block.

        Args:
            hidden_states: Image tokens of shape [B, S_img, D].
            encoder_hidden_states: Text tokens of shape [B, S_txt, D].
            temb_mod_params_img: Image-stream modulation parameters.
            temb_mod_params_txt: Text-stream modulation parameters.
            image_rotary_emb: Optional (cos, sin) tuple for rotary embeddings.

        Returns:
            Tuple of (encoder_hidden_states, hidden_states).
        """
        (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = (
            temb_mod_params_img
        )
        (
            (c_shift_msa, c_scale_msa, c_gate_msa),
            (c_shift_mlp, c_scale_mlp, c_gate_mlp),
        ) = temb_mod_params_txt

        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = (1 + scale_msa) * norm_hidden_states + shift_msa
        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)
        norm_encoder_hidden_states = (
            1 + c_scale_msa
        ) * norm_encoder_hidden_states + c_shift_msa

        attn_result = self.attn(
            norm_hidden_states,
            norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )
        if not isinstance(attn_result, tuple):
            raise ValueError("Expected tuple from dual-stream attention")
        attn_output, context_attn_output = attn_result

        attn_output = gate_msa * attn_output
        hidden_states = hidden_states + attn_output
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + gate_mlp * ff_output

        context_attn_output = c_gate_msa * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = (
            norm_encoder_hidden_states * (1 + c_scale_mlp) + c_shift_mlp
        )
        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = (
            encoder_hidden_states + c_gate_mlp * context_ff_output
        )

        if encoder_hidden_states.dtype == DType.float16:
            encoder_hidden_states = ops.min(
                ops.max(encoder_hidden_states, -65504),
                65504,
            )

        return encoder_hidden_states, hidden_states


class Flux2SingleTransformerBlock(Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        *,
        dtype: DType,
        device: DeviceRef,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        bias: bool = False,
        quant_config: QuantConfig | None = None,
    ) -> None:
        """Initialize Flux2SingleTransformerBlock.

        Args:
            dim: Hidden dimension size.
            num_attention_heads: Number of attention heads.
            attention_head_dim: Dimension of each attention head.
            dtype: Weight dtype.
            device: Weight device.
            mlp_ratio: Multiplier for feedforward hidden dimension.
            eps: Epsilon for layer normalization.
            bias: Whether to use bias in linear layers.
        """
        super().__init__()
        self.norm = LayerNorm(
            dim,
            dtype=dtype,
            device=device,
            eps=eps,
            elementwise_affine=False,
            use_bias=False,
        )
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
            dtype=dtype,
            device=device,
            quant_config=quant_config,
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        encoder_hidden_states: TensorValue | None = None,
        temb_mod_params: tuple[
            TensorValue,
            TensorValue,
            TensorValue,
        ]
        | None = None,
        image_rotary_emb: tuple[TensorValue, TensorValue] | None = None,
        split_hidden_states: bool = False,
        text_seq_len: int | Dim | None = None,
    ) -> TensorValue | tuple[TensorValue, TensorValue]:
        """Forward pass for single-stream transformer block.

        Args:
            hidden_states: Image tokens or concatenated text+image tokens.
            encoder_hidden_states: Optional text tokens to concatenate.
            temb_mod_params: (shift, scale, gate) tuple for modulation.
            image_rotary_emb: Optional (cos, sin) tuple for rotary embeddings.
            split_hidden_states: If True, split output back into text and image.
            text_seq_len: Length of text sequence when splitting.

        Returns:
            Either concatenated hidden states or (encoder_hidden_states, hidden_states).
        """
        if encoder_hidden_states is not None:
            text_seq_len = encoder_hidden_states.shape[1]
            hidden_states = ops.concat(
                [encoder_hidden_states, hidden_states],
                axis=1,
            )

        if temb_mod_params is None:
            raise ValueError("temb_mod_params cannot be None")
        mod_shift, mod_scale, mod_gate = temb_mod_params

        norm_hidden_states = self.norm(hidden_states)
        norm_hidden_states = (1 + mod_scale) * norm_hidden_states + mod_shift
        attn_output = self.attn(
            norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )
        hidden_states = hidden_states + mod_gate * attn_output

        if hidden_states.dtype == DType.float16:
            hidden_states = ops.min(ops.max(hidden_states, -65504), 65504)

        if split_hidden_states:
            if text_seq_len is None:
                raise ValueError("text_seq_len is required when splitting")
            encoder_hidden_states = hidden_states[:, :text_seq_len, :]
            hidden_states = hidden_states[:, text_seq_len:, :]
            return encoder_hidden_states, hidden_states
        return hidden_states


class Flux2Transformer2DModel(Module):
    def __init__(self, config: Flux2Config) -> None:
        """Initialize Flux2Transformer2DModel.

        Args:
            config: Flux2 configuration containing model dimensions,
                attention settings, and device/dtype information.
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
        devices = config.devices
        device = devices[0]
        dtype = config.dtype
        eps = config.eps
        quant_config = config.quant_config
        nvfp4_layers_bfl: frozenset[str] = getattr(
            config, "nvfp4_layers_bfl", frozenset()
        )

        self.devices = devices
        self.patch_size = patch_size
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.max_dtype = dtype
        self.in_channels = in_channels
        self.joint_attention_dim = joint_attention_dim

        self.pos_embed = Flux2PosEmbed(
            theta=rope_theta, axes_dim=axes_dims_rope
        )
        self.time_guidance_embed = Flux2TimestepGuidanceEmbeddings(
            in_channels=timestep_guidance_channels,
            embedding_dim=self.inner_dim,
            bias=False,
            guidance_embeds=getattr(config, "guidance_embeds", True),
            dtype=dtype,
            device=device,
        )
        self.double_stream_modulation_img = Flux2Modulation(
            self.inner_dim,
            dtype=dtype,
            device=device,
            mod_param_sets=2,
            bias=False,
        )
        self.double_stream_modulation_txt = Flux2Modulation(
            self.inner_dim,
            dtype=dtype,
            device=device,
            mod_param_sets=2,
            bias=False,
        )
        self.single_stream_modulation = Flux2Modulation(
            self.inner_dim,
            dtype=dtype,
            device=device,
            mod_param_sets=1,
            bias=False,
        )
        self.x_embedder = Linear(
            in_dim=in_channels,
            out_dim=self.inner_dim,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.context_embedder = Linear(
            in_dim=joint_attention_dim,
            out_dim=self.inner_dim,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.transformer_blocks = LayerList(
            [
                Flux2TransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dtype=dtype,
                    device=device,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    bias=False,
                    quant=Flux2BlockQuant.resolve(
                        i, quant_config, nvfp4_layers_bfl
                    ),
                )
                for i in range(num_layers)
            ]
        )
        self.single_transformer_blocks = LayerList(
            [
                Flux2SingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dtype=dtype,
                    device=device,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    bias=False,
                    quant_config=quant_config,
                )
                for _ in range(num_single_layers)
            ]
        )
        self.norm_out = AdaLayerNormContinuous(
            embedding_dim=self.inner_dim,
            conditioning_embedding_dim=self.inner_dim,
            elementwise_affine=False,
            dtype=dtype,
            device=device,
            eps=eps,
            bias=False,
        )
        self.proj_out = Linear(
            in_dim=self.inner_dim,
            out_dim=patch_size * patch_size * self.out_channels,
            dtype=dtype,
            device=device,
            has_bias=False,
        )

    def input_types(self) -> tuple[TensorType, ...]:
        """Define input tensor types for the model with symbolic shapes.

        Returns:
            Tuple of TensorType specifications for all model inputs.
        """
        primary = self.devices[0]
        return (
            TensorType(
                self.max_dtype,
                shape=["batch_size", "image_seq_len", self.in_channels],
                device=primary,
            ),
            TensorType(
                self.max_dtype,
                shape=["batch_size", "text_seq_len", self.joint_attention_dim],
                device=primary,
            ),
            TensorType(self.max_dtype, shape=["batch_size"], device=primary),
            TensorType(
                DType.int64,
                shape=["batch_size", "image_seq_len", 4],
                device=primary,
            ),
            TensorType(
                DType.int64,
                shape=["batch_size", "text_seq_len", 4],
                device=primary,
            ),
            TensorType(self.max_dtype, shape=["batch_size"], device=primary),
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        encoder_hidden_states: TensorValue,
        timestep: TensorValue,
        img_ids: TensorValue,
        txt_ids: TensorValue,
        guidance: TensorValue,
    ) -> tuple[TensorValue]:
        """Forward pass through Flux2 Transformer.

        Args:
            hidden_states: Image latents of shape [B, H*W, in_channels].
            encoder_hidden_states: Text embeddings of shape [B, txt_len, joint_attention_dim].
            timestep: Denoising timestep of shape [B].
            img_ids: Image position IDs of shape [batch_size, image_seq_len, 4]
                or [image_seq_len, 4].
            txt_ids: Text position IDs of shape [batch_size, text_seq_len, 4]
                or [text_seq_len, 4].
            guidance: Guidance scale of shape [B].

        Returns:
            Denoised output of shape [B, H*W, patch_size^2 * out_channels].
        """
        if img_ids.rank == 3:
            img_ids = img_ids[0]
        if txt_ids.rank == 3:
            txt_ids = txt_ids[0]

        num_txt_tokens = encoder_hidden_states.shape[1]
        timestep = ops.cast(timestep * 1000.0, hidden_states.dtype)
        guidance = ops.cast(guidance * 1000.0, hidden_states.dtype)
        temb = self.time_guidance_embed(timestep, guidance)

        double_stream_mod_img_tuple = self.double_stream_modulation_img(temb)
        double_stream_mod_txt_tuple = self.double_stream_modulation_txt(temb)
        single_stream_mod_tuple = self.single_stream_modulation(temb)
        double_stream_mod_img = (
            double_stream_mod_img_tuple[0],
            double_stream_mod_img_tuple[1],
        )
        double_stream_mod_txt = (
            double_stream_mod_txt_tuple[0],
            double_stream_mod_txt_tuple[1],
        )
        single_stream_mod = single_stream_mod_tuple[0]

        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        ids = ops.concat([txt_ids, img_ids], axis=0)
        image_rotary_emb = self.pos_embed(ids)

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod_params_img=double_stream_mod_img,
                temb_mod_params_txt=double_stream_mod_txt,
                image_rotary_emb=image_rotary_emb,
            )

        hidden_states = ops.concat(
            [encoder_hidden_states, hidden_states], axis=1
        )

        for single_block in self.single_transformer_blocks:
            hidden_states = single_block(
                hidden_states=hidden_states,
                encoder_hidden_states=None,
                temb_mod_params=single_stream_mod,
                image_rotary_emb=image_rotary_emb,
                split_hidden_states=False,
            )
            if isinstance(hidden_states, tuple):
                raise ValueError("Expected concatenated hidden states")

        hidden_states = hidden_states[:, num_txt_tokens:, :]
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)
        return (output,)
