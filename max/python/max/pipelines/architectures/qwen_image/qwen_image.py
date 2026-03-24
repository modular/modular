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

"""QwenImage Transformer 2D Model.

A 20B parameter MMDiT model for text-to-image generation with 60 dual-stream
blocks, 3D RoPE, and timestep-only embeddings (no guidance embedding).

Weight key naming matches HuggingFace diffusers:
- img_in.{weight,bias}          (input projection for image latents)
- txt_in.{weight,bias}          (input projection for text embeddings)
- time_text_embed.timestep_embedder.{linear_1,linear_2}.{weight,bias}
- txt_norm.weight               (RMSNorm for text output)
- transformer_blocks.{i}.*      (per-block: img_mod, txt_mod, attn, img_mlp, txt_mlp)
- norm_out.linear.{weight,bias} (AdaLayerNormContinuous)
- proj_out.{weight,bias}        (output projection)
"""

from max.dtype import DType
from max.graph import TensorType, TensorValue, ops
from max.nn.layer import LayerList, Module
from max.nn.linear import Linear
from max.nn.norm import RMSNorm

from .layers.embeddings import (
    QwenImagePosEmbed,
    QwenImageTimestepProjEmbeddings,
)
from .layers.normalizations import AdaLayerNormContinuous
from .layers.qwen_image_attention import QwenImageTransformerBlock
from .model_config import QwenImageConfigBase


class QwenImageTransformer2DModel(Module):
    """QwenImage Transformer with 60 dual-stream blocks.

    Key differences from Flux2:
    - No guidance embedding (timestep only)
    - No single-stream blocks (all 60 are dual-stream)
    - 3D RoPE with axes [16, 56, 56] (T, H, W)
    - Per-block modulation (img_mod, txt_mod per block)
    - inner_dim = 24 * 128 = 3072
    """

    def __init__(
        self,
        config: QwenImageConfigBase,
    ):
        super().__init__()
        patch_size = config.patch_size
        in_channels = config.in_channels
        out_channels = config.out_channels
        num_layers = config.num_layers
        attention_head_dim = config.attention_head_dim
        num_attention_heads = config.num_attention_heads
        joint_attention_dim = config.joint_attention_dim
        axes_dims_rope = config.axes_dims_rope
        rope_theta = config.rope_theta
        device = config.device
        dtype = config.dtype
        eps = config.eps

        self.patch_size = patch_size
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        # 1. Positional embeddings (3D RoPE: T, H, W)
        self.pos_embed = QwenImagePosEmbed(
            theta=rope_theta, axes_dim=axes_dims_rope
        )

        # 2. Timestep embeddings (no guidance) — key: time_text_embed.*
        self.time_text_embed = QwenImageTimestepProjEmbeddings(
            in_channels=256,
            embedding_dim=self.inner_dim,
            bias=True,
            dtype=dtype,
            device=device,
        )

        # 3. Input embeddings — keys: img_in.*, txt_in.*
        self.img_in = Linear(
            in_dim=in_channels,
            out_dim=self.inner_dim,
            dtype=dtype,
            device=device,
            has_bias=True,
        )
        self.txt_in = Linear(
            in_dim=joint_attention_dim,
            out_dim=self.inner_dim,
            dtype=dtype,
            device=device,
            has_bias=True,
        )

        # 4. Text input norm — key: txt_norm.weight
        self.txt_norm = RMSNorm(joint_attention_dim, dtype=dtype, eps=eps)

        # 5. Dual-stream transformer blocks (all 60 are dual-stream)
        self.transformer_blocks: LayerList = LayerList(
            [
                QwenImageTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=4.0,
                    eps=eps,
                    bias=True,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(num_layers)
            ]
        )

        # 6. Output layers — keys: norm_out.linear.*, proj_out.*
        self.norm_out = AdaLayerNormContinuous(
            embedding_dim=self.inner_dim,
            conditioning_embedding_dim=self.inner_dim,
            dtype=dtype,
            device=device,
            eps=eps,
            bias=True,
        )
        self.proj_out = Linear(
            in_dim=self.inner_dim,
            out_dim=patch_size * patch_size * self.out_channels,
            dtype=dtype,
            device=device,
            has_bias=True,
        )

        # Store config for input_types
        self.max_device = device
        self.max_dtype = dtype
        self.in_channels = in_channels
        self.joint_attention_dim = joint_attention_dim
        self.zero_cond_t = config.zero_cond_t
        # Set before graph build for zero_cond_t split modulation
        self.num_noise_tokens: int | None = None

    def input_types(self) -> tuple[TensorType, ...]:
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
        # 3D position IDs: (T, H, W)
        img_ids_type = TensorType(
            DType.int64,
            shape=["batch_size", "image_seq_len", 3],
            device=self.max_device,
        )
        txt_ids_type = TensorType(
            DType.int64,
            shape=["batch_size", "text_seq_len", 3],
            device=self.max_device,
        )

        result = (
            hidden_states_type,
            encoder_hidden_states_type,
            timestep_type,
            img_ids_type,
            txt_ids_type,
        )

        return result

    def __call__(
        self,
        hidden_states: TensorValue,
        encoder_hidden_states: TensorValue,
        timestep: TensorValue,
        img_ids: TensorValue,
        txt_ids: TensorValue,
    ) -> tuple[TensorValue]:
        """Forward pass through QwenImage Transformer.

        Args:
            hidden_states: Image latents [B, img_seq, in_channels].
            encoder_hidden_states: Text embeddings [B, txt_len, joint_attention_dim].
            timestep: Denoising timestep [B] (scaled to [0, 1] range).
            img_ids: Image position IDs [B, image_seq_len, 3] (T, H, W).
            txt_ids: Text position IDs [B, text_seq_len, 3].
            num_noise_tokens: [1] scalar — number of noise tokens in the
                image sequence. Condition tokens (positions >= this value)
                receive timestep=0 modulation. Only for zero_cond_t=True.

        Returns:
            Denoised output of shape [B, img_seq, patch_size^2 * out_channels].
        """
        # Handle batch dimension in ids
        img_ids = img_ids[0]  # [img_seq, 3]
        txt_ids = txt_ids[0]  # [txt_len, 3]

        # 1. Calculate timestep embedding
        timestep_scaled = ops.cast(timestep * 1000.0, hidden_states.dtype)
        temb = self.time_text_embed(timestep_scaled)

        # For zero_cond_t: compute temb for timestep=0 (condition tokens)
        temb_zero: TensorValue | None = None
        num_noise: int | None = None
        if self.zero_cond_t and self.num_noise_tokens is not None:
            zero_t = timestep_scaled * 0.0
            temb_zero = self.time_text_embed(zero_t)
            num_noise = self.num_noise_tokens

        # 2. Input projection (txt_norm applied before txt_in projection)
        hidden_states = self.img_in(hidden_states)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        # 3. Calculate RoPE embeddings
        ids = ops.concat([txt_ids, img_ids], axis=0)
        image_rotary_emb = self.pos_embed(ids)

        # 4. Dual-stream transformer blocks (all 60)
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                temb_zero=temb_zero,
                num_noise_tokens=num_noise,
            )

        # 5. Output projection (image tokens only, discard text)
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        return (output,)
