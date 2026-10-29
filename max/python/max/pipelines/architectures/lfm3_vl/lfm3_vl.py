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
"""The LFM3-VL vision tower, projector and language tower.

The tower embeds each frame's patches, in merge-block order, through layers
that alternate a gated depthwise convolution over the frame's patch grid
with bidirectional attention within the frame. The projector merges each
block into one decoder embedding.
"""

from __future__ import annotations

from collections.abc import Callable

from max import functional as F
from max.dtype import DType
from max.nn import Linear, Module, ModuleList, as_subgraph
from max.nn.common_layers.functional_kernels import (
    flash_attention_ragged_gpu,
)
from max.nn.legacy.attention.mask_config import MHAMaskVariant
from max.nn.norm import LayerNorm, RMSNorm
from max.pipelines.lib import ModuleV3Outputs
from max.pipelines.lib.vlm_utils import F_merge_multimodal_embeddings
from max.tensor import Tensor

from ..lfm3.lfm3 import Lfm3, Lfm3MLP
from ..lfm3.model_config import FULL_ATTENTION
from .model_config import Lfm3VlConfig, Lfm3VlVisionConfig


def _rotate(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    xf = F.cast(x, DType.float32)
    rotated = F.concat([-xf[:, :, half:], xf[:, :, :half]], axis=-1)
    return F.cast(xf * cos + rotated * sin, x.dtype)


class Lfm3VlVisionAttention(Module[..., Tensor]):
    """Grouped-query attention within each frame.

    Uses per-head q/k norms and axial RoPE.
    """

    def __init__(self, config: Lfm3VlVisionConfig) -> None:
        hidden = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = hidden // self.num_heads
        self.q_proj = Linear(hidden, self.num_heads * self.head_dim, bias=False)
        self.k_proj = Linear(
            hidden, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = Linear(
            hidden, self.num_kv_heads * self.head_dim, bias=False
        )
        self.out_proj = Linear(
            self.num_heads * self.head_dim, hidden, bias=False
        )
        self.q_layernorm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_layernorm = RMSNorm(self.head_dim, eps=config.norm_eps)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: Tensor,
    ) -> Tensor:
        """Attends within each frame, with axial RoPE."""
        q = self.q_layernorm(
            self.q_proj(x).reshape((-1, self.num_heads, self.head_dim))
        )
        k = self.k_layernorm(
            self.k_proj(x).reshape((-1, self.num_kv_heads, self.head_dim))
        )
        v = self.v_proj(x).reshape((-1, self.num_kv_heads, self.head_dim))
        q, k = _rotate(q, cos, sin), _rotate(k, cos, sin)
        # The ragged kernel takes as many KV heads as query heads.
        rep = self.num_heads // self.num_kv_heads
        k, v = (
            F.concat(
                [t.reshape((-1, self.num_kv_heads, 1, self.head_dim))] * rep,
                axis=2,
            ).reshape((-1, self.num_heads, self.head_dim))
            for t in (k, v)
        )
        out = flash_attention_ragged_gpu(
            q,
            k,
            v,
            input_row_offsets=cu_seqlens,
            max_seq_len=max_seqlen,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=self.head_dim**-0.5,
        ).rebind_mapping(q.mapping)
        return self.out_proj(out.reshape((-1, self.num_heads * self.head_dim)))


class Lfm3VlVisionConv(Module[..., Tensor]):
    """The vision tower's gated depthwise convolution.

    Computes ``out_proj(C * conv(B * x))`` with a depthwise convolution over the
    frame's patch grid.

    ``conv_taps[i, t]`` is the row of patch ``i``'s ``t``-th neighbor in the
    kernel window, or the row count for a neighbor outside the frame, which
    reads a zero row. Gathering the taps keeps the merge-block row order, so
    no layout conversion to and from the raster grid is needed.
    """

    def __init__(self, config: Lfm3VlVisionConfig) -> None:
        hidden, k = config.hidden_size, config.conv_kernel_size
        self.in_proj = Linear(hidden, 3 * hidden, bias=False)
        self.spatial_conv = Tensor.zeros([hidden, 1, k, k])
        self.out_proj = Linear(hidden, hidden, bias=False)

    def forward(self, x: Tensor, conv_taps: Tensor) -> Tensor:
        """Applies the gated depthwise convolution over each frame's patches."""
        hidden = int(self.spatial_conv.shape[0])
        b, c, v = F.split(self.in_proj(x), [hidden] * 3, axis=-1)
        gated = b * v
        padded = F.concat(
            [
                gated,
                F.full([1, hidden], 0, dtype=gated.dtype, device=gated.device),
            ]
        )
        weight = F.cast(self.spatial_conv.reshape([hidden, -1]), DType.float32)
        out = F.cast(F.gather(padded, conv_taps[:, 0], axis=0), DType.float32)
        out = out * weight[:, 0]
        for t in range(1, int(weight.shape[1])):
            tap = F.gather(padded, conv_taps[:, t], axis=0)
            out = out + F.cast(tap, DType.float32) * weight[:, t]
        return self.out_proj(c * F.cast(out, x.dtype))


class Lfm3VlVisionLayer(Module[..., Tensor]):
    """A pre-norm residual block around the operator and the MLP."""

    def __init__(self, config: Lfm3VlVisionConfig, layer_type: str) -> None:
        self.is_attention = layer_type == FULL_ATTENTION
        if self.is_attention:
            self.self_attn = Lfm3VlVisionAttention(config)
        else:
            self.conv = Lfm3VlVisionConv(config)
        hidden, eps = config.hidden_size, config.norm_eps
        self.feed_forward = Lfm3MLP(hidden, config.intermediate_size)
        self.operator_norm = RMSNorm(hidden, eps=eps)
        self.ffn_norm = RMSNorm(hidden, eps=eps)

    def forward(
        self,
        h: Tensor,
        cos: Tensor,
        sin: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: Tensor,
        conv_taps: Tensor,
    ) -> Tensor:
        """Runs the operator and the MLP as residual branches."""
        x = self.operator_norm(h)
        if self.is_attention:
            h = h + self.self_attn(x, cos, sin, cu_seqlens, max_seqlen)
        else:
            h = h + self.conv(x, conv_taps)
        return h + self.feed_forward(self.ffn_norm(h))


class Lfm3VlVisionTower(Module[..., Tensor]):
    """The patchifier, its norm and the layer stack."""

    def __init__(self, config: Lfm3VlVisionConfig) -> None:
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.rope_theta = config.rope_theta
        self.patchifier = Linear(
            config.patch_dim, config.hidden_size, bias=False
        )
        self.embedding_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.layers = ModuleList(
            [Lfm3VlVisionLayer(config, t) for t in config.layer_types]
        )

    def forward(
        self,
        patches: Tensor,
        position_ids: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: Tensor,
        conv_taps: Tensor,
    ) -> Tensor:
        """Encodes frame patches into vision rows."""
        h = self.patchifier(F.cast(patches, self.patchifier.weight.dtype))
        h = self.embedding_norm(h)
        # Each axis rotates a quarter of the head with the same frequencies:
        # the head reads [h, w, h, w] in quarters.
        quarter = self.head_dim // 4
        iota = F.arange(0, quarter, dtype=DType.float32, device=h.device)
        inv_freqs = 1.0 / (self.rope_theta ** (iota / quarter))
        freqs = F.cast(position_ids, DType.float32).reshape((-1, 2, 1))
        freqs = (freqs * inv_freqs).reshape((-1, 1, 2 * quarter))
        angles = F.concat([freqs, freqs], axis=-1)
        cos, sin = F.cos(angles), F.sin(angles)
        for layer in self.layers:
            call: Callable[..., Tensor] = as_subgraph(
                layer,
                name="vision_attention"
                if layer.is_attention
                else "vision_conv",
            )
            h = call(h, cos, sin, cu_seqlens, max_seqlen, conv_taps)
        return h


class Lfm3VlProjector(Module[[Tensor], Tensor]):
    """Merges each block of patch embeddings into one decoder embedding."""

    def __init__(self, config: Lfm3VlConfig) -> None:
        vision_hidden = config.vision.hidden_size
        self.merged = vision_hidden * config.downsample_factor**2
        self.layer_norm = LayerNorm(
            vision_hidden, eps=config.projector_norm_eps, keep_dtype=False
        )
        self.linear_1 = Linear(
            self.merged,
            config.projector_hidden_size,
            bias=config.projector_bias,
        )
        self.linear_2 = Linear(
            config.projector_hidden_size,
            config.hidden_size,
            bias=config.projector_bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Merges patch blocks and projects them into the decoder's space."""
        n, hidden = x.shape
        factor = self.merged // int(hidden)
        h = F.rebind(self.layer_norm(x), ((n // factor) * factor, hidden))
        h = h.reshape((-1, self.merged))
        return self.linear_2(F.gelu(self.linear_1(h)))


class Lfm3VlVision(Module[..., Tensor]):
    """The vision tower and the projector, compiled as one graph."""

    def __init__(self, config: Lfm3VlConfig) -> None:
        self.vision_tower = Lfm3VlVisionTower(config.vision)
        self.multi_modal_projector = Lfm3VlProjector(config)

    def forward(
        self,
        patches: Tensor,
        position_ids: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: Tensor,
        conv_taps: Tensor,
    ) -> Tensor:
        """Encodes and projects frame patches for the decoder."""
        return self.multi_modal_projector(
            self.vision_tower(
                patches, position_ids, cu_seqlens, max_seqlen, conv_taps
            )
        )


class Lfm3VlLanguage(Lfm3):
    """The LFM3 decoder, reading image embeddings at their placeholders."""

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        *inputs: Tensor,
    ) -> ModuleV3Outputs:
        """Runs the decoder with image embeddings.

        Takes the image embeddings and their placeholder indices ahead of the
        cache inputs.
        """
        del return_n_logits
        image_embeddings, image_token_indices, *kv_inputs = inputs
        embeds = F_merge_multimodal_embeddings(
            self.model.embed_tokens(tokens),
            F.cast(image_embeddings, self.model.embed_tokens.weight.dtype),
            image_token_indices,
        )
        return self.decode(embeds, input_row_offsets, tuple(kv_inputs))
