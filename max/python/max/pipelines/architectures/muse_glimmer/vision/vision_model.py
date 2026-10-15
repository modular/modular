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

"""Muse Glimmer vision tower, adapter and projection (ModuleV3 API)."""

from __future__ import annotations

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.functional_kernels import (
    flash_attention_ragged_gpu,
)
from max.experimental.nn.linear import Linear
from max.experimental.nn.norm import LayerNorm
from max.experimental.nn.sequential import ModuleList
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorType
from max.nn.attention.mask_config import MHAMaskVariant
from max.pipelines.architectures.gemma4_modulev3.layers.rms_norm import (
    Gemma4RMSNorm,
)

from ..model_config import MuseGlimmerVisionConfig


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate-half RoPE in float32 on ``[P, heads, head_dim]``."""
    x32 = x.cast(DType.float32)
    x1, x2 = x32.split(int(x.shape[-1]) // 2, axis=-1)
    rotated = F.concat([-x2, x1], axis=-1)
    return (x32 * cos.unsqueeze(1) + rotated * sin.unsqueeze(1)).cast(x.dtype)


class MuseGlimmerVisionAttention(Module[..., Tensor]):
    """Bidirectional ragged attention with axial 2D RoPE."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = Linear(hidden_size, hidden_size)
        self.k_proj = Linear(hidden_size, hidden_size)
        self.v_proj = Linear(hidden_size, hidden_size)
        self.proj = Linear(hidden_size, hidden_size)

    def forward(
        self,
        x: Tensor,
        rot_cos: Tensor,
        rot_sin: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: Tensor,
    ) -> Tensor:
        shape = (-1, self.num_heads, self.head_dim)
        q = _apply_rope(self.q_proj(x).reshape(shape), rot_cos, rot_sin)
        k = _apply_rope(self.k_proj(x).reshape(shape), rot_cos, rot_sin)
        out = flash_attention_ragged_gpu(
            q,
            k,
            self.v_proj(x).reshape(shape),
            input_row_offsets=cu_seqlens,
            max_seq_len=max_seqlen,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=self.head_dim**-0.5,
        )
        return self.proj(out.reshape((-1, self.num_heads * self.head_dim)))


class MuseGlimmerVisionMLP(Module[[Tensor], Tensor]):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.fc1 = Linear(hidden_size, intermediate_size)
        self.fc2 = Linear(intermediate_size, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class MuseGlimmerVisionEncoderLayer(Module[..., Tensor]):
    """Pre-LayerNorm block; attends within a window or a whole image."""

    def __init__(self, config: MuseGlimmerVisionConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        eps = config.layer_norm_eps
        self.norm1 = LayerNorm(hidden, eps=eps, keep_dtype=False)
        self.attn = MuseGlimmerVisionAttention(
            hidden, config.num_attention_heads
        )
        self.norm2 = LayerNorm(hidden, eps=eps, keep_dtype=False)
        self.mlp = MuseGlimmerVisionMLP(hidden, config.intermediate_size)

    def forward(
        self,
        x: Tensor,
        rot_cos: Tensor,
        rot_sin: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: Tensor,
    ) -> Tensor:
        x = x + self.attn(
            self.norm1(x), rot_cos, rot_sin, cu_seqlens, max_seqlen
        )
        return x + self.mlp(self.norm2(x))


class MuseGlimmerVisionPatchEmbedder(Module[..., Tensor]):
    """Patch projection plus the bilinearly resampled position table."""

    def __init__(self, config: MuseGlimmerVisionConfig) -> None:
        super().__init__()
        self.patch_embedding = Linear(
            config.patch_dim, config.hidden_size, bias=False
        )
        self.position_embedding_table = Tensor.zeros(
            [config.pos_emb_height * config.pos_emb_width, config.hidden_size]
        )

    def forward(
        self, pixel_values: Tensor, interp_idx: Tensor, interp_w: Tensor
    ) -> Tensor:
        h = self.patch_embedding(pixel_values)
        taps = F.gather(self.position_embedding_table, interp_idx, axis=0)
        pos = (taps.cast(DType.float32) * interp_w.unsqueeze(-1)).sum(axis=1)
        return h + pos.squeeze(1).cast(h.dtype)


class MuseGlimmerVisionTower(Module[..., Tensor]):
    """Patch embed -> window-major blocks -> ``ln_post`` -> pixel shuffle."""

    def __init__(self, config: MuseGlimmerVisionConfig) -> None:
        super().__init__()
        self.merge_size = config.merge_size
        self.patch_embedder = MuseGlimmerVisionPatchEmbedder(config)
        self.ln_pre = LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, keep_dtype=False
        )
        self.layers = ModuleList(
            [
                MuseGlimmerVisionEncoderLayer(config)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.full_attention = [
            t == "full_attention" for t in config.layer_types
        ]
        self.ln_post = LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, keep_dtype=False
        )

    def forward(
        self,
        pixel_values: Tensor,
        interp_idx: Tensor,
        interp_w: Tensor,
        window_index: Tensor,
        reverse_index: Tensor,
        pixel_shuffle_index: Tensor,
        rot_cos: Tensor,
        rot_sin: Tensor,
        cu_seqlens: Tensor,
        cu_window_seqlens: Tensor,
        max_seqlen: Tensor,
        max_window_seqlen: Tensor,
    ) -> Tensor:
        h = self.patch_embedder(pixel_values, interp_idx, interp_w)
        h = F.gather(self.ln_pre(h), window_index, axis=0)
        for layer, full in zip(self.layers, self.full_attention, strict=True):
            if full:
                h = layer(h, rot_cos, rot_sin, cu_seqlens, max_seqlen)
            else:
                h = layer(
                    h, rot_cos, rot_sin, cu_window_seqlens, max_window_seqlen
                )
        h = self.ln_post(F.gather(h, reverse_index, axis=0))
        h = F.gather(h, pixel_shuffle_index, axis=0)
        blocks = self.merge_size**2
        n, hidden = h.shape
        # Every image grid is a multiple of the merge size on both axes.
        h = F.rebind(h, [(n // blocks) * blocks, hidden])
        return (
            h.reshape((n // blocks, blocks, hidden))
            .permute([0, 2, 1])
            .reshape((-1, blocks * hidden))
        )


class MuseGlimmerVisionAdapter(Module[[Tensor], Tensor]):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = Linear(in_dim, hidden_dim, bias=False)
        self.fc2 = Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(self.fc2(F.gelu(self.fc1(x))))


class MuseGlimmerVisionModel(Module[..., Tensor]):
    """Maps packed image patches to normed text-width image embeddings.

    Attribute names equal the checkpoint's ``model.*`` names, so the weight
    adapter only strips the prefix.
    """

    def __init__(
        self,
        config: MuseGlimmerVisionConfig,
        text_hidden_size: int,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        self.vision_tower = MuseGlimmerVisionTower(config)
        self.vision_adapter = MuseGlimmerVisionAdapter(
            config.out_hidden_size, config.projector_hidden_size
        )
        self.vision_projection = Linear(
            config.projector_hidden_size, text_hidden_size, bias=False
        )
        self.perception_emb_norm = Gemma4RMSNorm(
            text_hidden_size, eps=rms_norm_eps, with_weight=False
        )

    def forward(self, *inputs: Tensor) -> Tensor:
        h = self.vision_adapter(self.vision_tower(*inputs))
        return self.perception_emb_norm(self.vision_projection(h))


def vision_input_types(
    config: MuseGlimmerVisionConfig, device: DeviceRef, dtype: DType
) -> tuple[TensorType, ...]:
    """Returns the compile inputs of :class:`MuseGlimmerVisionModel`."""
    p = "total_patches"
    return (
        TensorType(dtype, [p, config.patch_dim], device=device),
        TensorType(DType.int32, [p, 4], device=device),
        TensorType(DType.float32, [p, 4], device=device),
        *(TensorType(DType.int64, [p], device=device) for _ in range(3)),
        *(
            TensorType(DType.float32, [p, config.head_dim], device=device)
            for _ in range(2)
        ),
        TensorType(DType.uint32, ["num_images_plus_1"], device=device),
        TensorType(DType.uint32, ["num_windows_plus_1"], device=device),
        *(
            TensorType(DType.uint32, [], device=DeviceRef.CPU())
            for _ in range(2)
        ),
    )
