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
"""The Gemma4 vision tower DiffusionGemma registers.

The tower is built and weighted for every DiffusionGemma checkpoint, but
neither checkpoint accepts images, so it never runs on the serving path.

Attribute names match the graph arch's vision tree so the architecture's
vision weight adapter keys load unchanged.
"""

from __future__ import annotations

import math

from max import functional as F
from max.dtype import DType
from max.nn import Linear, Module
from max.nn.common_layers.mlp import MLP
from max.nn.kernels import (
    flash_attention_ragged_gpu,
)
from max.nn.legacy.attention.mask_config import MHAMaskVariant
from max.nn.sequential import ModuleList
from max.tensor import Tensor

from .layers.rms_norm import Gemma4RMSNorm
from .layers.rotary_embedding import (
    apply_multidimensional_rope,
    compute_vision_freqs_cis,
)
from .model_config import Gemma4ForConditionalGenerationConfig


class Gemma4VisionPatchEmbedder(Module[..., Tensor]):
    """Turns raw pixels into embedded patches.

    Each pixel is rescaled from ``[0, 1]`` to ``[-1, 1]``, projected
    linearly, and given its 2D position embedding.
    """

    def __init__(self, config: Gemma4ForConditionalGenerationConfig) -> None:
        super().__init__()
        vision_cfg = config.vision_config
        assert vision_cfg is not None
        self.dtype = config.unquantized_dtype
        self.hidden_size = vision_cfg.hidden_size
        self.position_embedding_size = vision_cfg.position_embedding_size

        self.input_proj = Linear(
            3 * vision_cfg.patch_size**2, self.hidden_size, bias=False
        )
        self.position_embedding_table = Tensor.zeros(
            [2, self.position_embedding_size, self.hidden_size]
        )

    def forward(
        self, patches_flat: Tensor, pixel_position_ids: Tensor
    ) -> Tensor:
        """Scales the pixels, projects them and adds their positions."""
        patches = patches_flat.cast(self.dtype) * 2.0 - 1.0
        hidden = self.input_proj(patches)

        table_flat = self.position_embedding_table.reshape(
            (2 * self.position_embedding_size, self.hidden_size)
        )
        x_ids = pixel_position_ids[:, 0].cast(DType.int64)
        y_ids = (
            pixel_position_ids[:, 1].cast(DType.int64)
            + self.position_embedding_size
        )
        position_emb = F.gather(table_flat, x_ids, axis=0) + F.gather(
            table_flat, y_ids, axis=0
        )
        return hidden + position_emb


class Gemma4VisionAttention(Module[..., Tensor]):
    """Ragged bidirectional attention over each image's patches.

    It applies per-head query and key norms, a weightless value norm and 2D
    RoPE, at scale 1.0.
    """

    def __init__(
        self, config: Gemma4ForConditionalGenerationConfig, layer_idx: int
    ) -> None:
        super().__init__()
        vision_cfg = config.vision_config
        assert vision_cfg is not None
        self.layer_idx = layer_idx
        self.head_dim = vision_cfg.head_dim
        self.num_attention_heads = vision_cfg.num_attention_heads
        self.num_key_value_heads = vision_cfg.num_key_value_heads

        bias = vision_cfg.attention_bias
        self.q_proj = Linear(
            vision_cfg.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=bias,
        )
        self.k_proj = Linear(
            vision_cfg.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=bias,
        )
        self.v_proj = Linear(
            vision_cfg.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=bias,
        )
        self.o_proj = Linear(
            self.num_attention_heads * self.head_dim,
            vision_cfg.hidden_size,
            bias=bias,
        )
        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=vision_cfg.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(self.head_dim, eps=vision_cfg.rms_norm_eps)
        self.v_norm = Gemma4RMSNorm(
            self.head_dim, eps=vision_cfg.rms_norm_eps, with_weight=False
        )

        self._qkv_split = [
            self.num_attention_heads * self.head_dim,
            self.num_key_value_heads * self.head_dim,
            self.num_key_value_heads * self.head_dim,
        ]

    @property
    def wqkv(self) -> Tensor:
        """The query, key and value projections, stacked for one matmul."""
        return F.concat(
            (self.q_proj.weight, self.k_proj.weight, self.v_proj.weight)
        )

    def forward(
        self,
        hidden_states: Tensor,
        freqs_cis: Tensor,
        cu_seqlens: Tensor,
        max_seq_len: Tensor,
    ) -> Tensor:
        """Runs the attention over the patches in each image."""
        total_patches = hidden_states.shape[0]

        # One fused QKV matmul, valid because this arch drops the checkpoint's
        # ClippableLinear clamps.
        x = hidden_states @ self.wqkv.T
        xq, xk, xv = (
            part.reshape(shape)
            for part, shape in zip(
                x.split(self._qkv_split, axis=-1),
                (
                    (total_patches, self.num_attention_heads, self.head_dim),
                    (
                        total_patches,
                        self.num_key_value_heads,
                        self.head_dim,
                    ),
                    (
                        total_patches,
                        self.num_key_value_heads,
                        self.head_dim,
                    ),
                ),
                strict=True,
            )
        )

        # Apply per-head RMS norms.
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)
        xv = self.v_norm(xv)

        # Add num_heads broadcast dim: [p, head_dim//2, 2] → [p, 1, head_dim//2, 2].
        freqs_bcast = freqs_cis.reshape(
            (total_patches, 1, self.head_dim // 2, 2)
        )
        # Apply 2-D multidimensional RoPE to Q and K.
        xq = apply_multidimensional_rope(xq, freqs_bcast, ndim=2)
        xk = apply_multidimensional_rope(xk, freqs_bcast, ndim=2)

        # Bidirectional packed flash attention (no causal mask, scale=1.0).
        output = flash_attention_ragged_gpu(
            xq,
            xk,
            xv,
            input_row_offsets=cu_seqlens,
            max_seq_len=max_seq_len,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=1.0,
        ).rebind_mapping(xq.mapping)
        # output: [total_patches, num_heads, head_dim]
        output = output.reshape((total_patches, -1))
        return self.o_proj(output)


class Gemma4VisionEncoderLayer(Module[..., Tensor]):
    """Full-attention transformer block, 4 norms, no layer scalar / MoE."""

    def __init__(
        self, config: Gemma4ForConditionalGenerationConfig, layer_idx: int
    ) -> None:
        super().__init__()
        vision_config = config.vision_config
        assert vision_config is not None
        hidden = vision_config.hidden_size
        eps = vision_config.rms_norm_eps

        self.input_layernorm = Gemma4RMSNorm(hidden, eps=eps)
        self.self_attn = Gemma4VisionAttention(config, layer_idx)
        self.post_attention_layernorm = Gemma4RMSNorm(hidden, eps=eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(hidden, eps=eps)
        self.mlp = MLP(
            hidden_dim=hidden,
            feed_forward_length=vision_config.intermediate_size,
            activation_function=vision_config.hidden_activation,
        )
        self.post_feedforward_layernorm = Gemma4RMSNorm(hidden, eps=eps)

    def forward(
        self,
        hidden_states: Tensor,
        freqs_cis: Tensor,
        cu_seqlens: Tensor,
        max_seq_len: Tensor,
    ) -> Tensor:
        """Runs the layer's attention, then its MLP."""
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        h = self.self_attn(h, freqs_cis, cu_seqlens, max_seq_len)
        h = residual + self.post_attention_layernorm(h)

        residual = h
        h = self.pre_feedforward_layernorm(h)
        h = self.mlp(h)
        return residual + self.post_feedforward_layernorm(h)


class Gemma4VisionEncoder(Module[..., Tensor]):
    """Stack of Gemma4VisionEncoderLayer blocks."""

    def __init__(self, config: Gemma4ForConditionalGenerationConfig) -> None:
        super().__init__()
        assert config.vision_config is not None
        self.layers = ModuleList(
            [
                Gemma4VisionEncoderLayer(config, layer_idx)
                for layer_idx in range(config.vision_config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        hidden_state: Tensor,
        freqs_cis: Tensor,
        cu_seqlens: Tensor,
        max_seq_len: Tensor,
    ) -> Tensor:
        """Runs the encoder stack over the patch embeddings."""
        for layer in self.layers:
            hidden_state = layer(
                hidden_state, freqs_cis, cu_seqlens, max_seq_len
            )
        return hidden_state


class Gemma4VisionPooler(Module[..., Tensor]):
    """Gather-based sparse average pooling; no learnable parameters.

    The float16 early-return branch of the graph arch is dropped: this arch
    is bf16-only.
    """

    def __init__(self, hidden_size: int, pooling_kernel_size: int) -> None:
        super().__init__()
        k2 = pooling_kernel_size * pooling_kernel_size
        self._post_scale: float = math.sqrt(hidden_size) / k2

    def forward(
        self, hidden_states: Tensor, pool_gather_index: Tensor
    ) -> Tensor:
        """Pools each image's patch features into one vector."""
        original_dtype = hidden_states.dtype
        h = hidden_states.cast(DType.float32)

        # Sentinel index (== total_patches) selects an appended zero row so
        # under-filled bins contribute nothing.
        zero_row = Tensor.zeros(
            [1, h.shape[1]], dtype=DType.float32, device=h.device
        )
        padded = F.concat([h, zero_row], axis=0)

        gathered = F.gather(padded, pool_gather_index, axis=0)
        pooled = gathered.sum(axis=1).squeeze(1)
        return (pooled * self._post_scale).cast(original_dtype)


class Gemma4MultimodalEmbedder(Module[[Tensor], Tensor]):
    """Weightless RMSNorm + projection into LM hidden space."""

    def __init__(
        self,
        multimodal_hidden_size: int,
        text_hidden_size: int,
        dtype: DType,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.embedding_projection = Linear(
            multimodal_hidden_size, text_hidden_size, bias=False
        )
        self.embedding_pre_projection_norm = Gemma4RMSNorm(
            multimodal_hidden_size, eps=eps, with_weight=False
        )

    def forward(self, inputs_embeds: Tensor) -> Tensor:
        """Normalizes the token embeddings and projects them."""
        normed = self.embedding_pre_projection_norm(inputs_embeds)
        return self.embedding_projection(normed.cast(self.dtype))


class Gemma4VisionModel(Module[..., Tensor]):
    """patch_embedder -> 2-D-RoPE encoder -> pooler -> (standardize) -> embed_vision."""

    def __init__(self, config: Gemma4ForConditionalGenerationConfig) -> None:
        super().__init__()
        vision_config = config.vision_config
        assert vision_config is not None
        self.rope_theta = vision_config.rope_theta
        self.head_dim = vision_config.head_dim
        self.dtype = config.unquantized_dtype

        self.patch_embedder = Gemma4VisionPatchEmbedder(config)
        self.encoder = Gemma4VisionEncoder(config)
        self.pooler = Gemma4VisionPooler(
            vision_config.hidden_size, vision_config.pooling_kernel_size
        )
        self.embed_vision = Gemma4MultimodalEmbedder(
            vision_config.hidden_size,
            config.text_config.hidden_size,
            self.dtype,
            eps=vision_config.rms_norm_eps,
        )
        self.standardize = vision_config.standardize
        if self.standardize:
            self.std_bias = Tensor.zeros([vision_config.hidden_size])
            self.std_scale = Tensor.ones([vision_config.hidden_size])

    def forward(
        self,
        patches_flat: Tensor,
        pixel_position_ids: Tensor,
        cu_seqlens: Tensor,
        pool_gather_index: Tensor,
        max_seq_len: Tensor,
    ) -> Tensor:
        """Runs the vision tower over the patches and pools each image."""
        hidden = self.patch_embedder(patches_flat, pixel_position_ids)

        freqs_cis = compute_vision_freqs_cis(
            pixel_position_ids,
            head_dim=self.head_dim,
            ndim=2,
            theta=self.rope_theta,
            dtype=self.dtype,
            device=pixel_position_ids.device,
        )

        encoded = self.encoder(hidden, freqs_cis, cu_seqlens, max_seq_len)
        pooled = self.pooler(encoded, pool_gather_index)
        if self.standardize:
            pooled = (
                pooled - self.std_bias.cast(pooled.dtype)
            ) * self.std_scale.cast(pooled.dtype)
        return self.embed_vision(pooled)


__all__ = ["Gemma4VisionModel"]
