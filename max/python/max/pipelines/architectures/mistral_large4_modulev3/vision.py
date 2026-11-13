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
"""Mistral Large 4's Pixtral vision tower and projector (ModuleV3).

The checkpoint's vision subtree is unquantized BF16 and carries Mistral's
consolidated names, so every module below binds :mod:`.weight_contract`'s
vision table directly: the tower under ``vision_encoder.*`` (conv patch
embed, ``ln_pre``, 30 blocks of ``attention.{wq,wk,wv,wo}`` /
``feed_forward.{w1,w2,w3}`` with axial 2D RoPE) and the projector's
``pre_mm_projector_norm``, ``patch_merger.merging_layer`` and
``vision_language_adapter.{w_in,w_out}`` (RMSNorm, 2x2 patch merge, then a
two-linear GELU adapter onto the text width).

Images attend block-diagonally: each image's patches form one ragged
sequence, and the tower's attention runs on the ragged flash kernel over
``cu_seqlens`` rather than a dense ``[1, 1, P, P]`` mask, which for a
maximal 12,100-patch image would be a 586 MB host-to-device copy per
forward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from max import functional as F
from max.dtype import DType
from max.graph import DeviceRef, Dim, TensorType, TensorValue, ops
from max.nn import Module
from max.nn.kernels import flash_attention_ragged_gpu
from max.nn.legacy.attention.mask_config import MHAMaskVariant
from max.nn.linear import Linear
from max.nn.norm import RMSNorm
from max.nn.sequential import ModuleList
from max.tensor import Tensor
from transformers import PretrainedConfig
from typing_extensions import Self

from ..pixtral_modulev3.vision_encoder.attention_utils import rotate_half
from ..pixtral_modulev3.vision_encoder.rotary_embedding_2d import (
    RotaryEmbedding2D,
)

_VISION_DTYPE = DType.bfloat16
"""The vision subtree's storage and compute dtype; it is never quantized."""


@dataclass(frozen=True)
class MistralLarge4VisionConfig:
    """The fields of the generated config that size the vision graph."""

    dtype: DType
    num_channels: int
    patch_size: int
    image_size: int
    vision_hidden_size: int
    vision_num_attention_heads: int
    vision_head_dim: int
    vision_rope_theta: float
    vision_num_hidden_layers: int
    vision_intermediate_size: int
    spatial_merge_size: int
    """Side of the patch square the merger folds into one embedding row."""

    hidden_size: int
    """Text width: the adapter's output, one row per merged token."""

    rms_norm_eps: float
    """The projector norm's epsilon, which is the text model's, as the HF
    reference does (modeling_mistral_large4.py:863)."""

    tower_rms_norm_eps: float = 1e-5
    """The tower norms' epsilon: HF's Pixtral default when the vision
    config omits it, as this checkpoint's does."""

    @property
    def patch_dim(self) -> int:
        """The flattened pixels of one patch."""
        return self.num_channels * self.patch_size**2

    @property
    def max_patches_per_side(self) -> int:
        """Patches per side of the largest image the RoPE table covers."""
        return self.image_size // self.patch_size

    @classmethod
    def from_hf_config(cls, huggingface_config: PretrainedConfig) -> Self:
        """Reads the vision geometry off the generated ``config.json``.

        Args:
            huggingface_config: The pipeline's ``MistralLarge4HFConfig``,
                whose ``vision_config`` block the config generator emits.

        Returns:
            The vision graph's configuration.
        """
        vision = huggingface_config.vision_config
        text = huggingface_config.text_config
        hidden = _int_field(vision, "hidden_size")
        heads = _int_field(vision, "num_attention_heads")
        return cls(
            dtype=_VISION_DTYPE,
            num_channels=_int_field(vision, "num_channels"),
            patch_size=_int_field(vision, "patch_size"),
            image_size=_int_field(vision, "image_size"),
            vision_hidden_size=hidden,
            vision_num_attention_heads=heads,
            vision_head_dim=int(getattr(vision, "head_dim", 0) or 0)
            or hidden // heads,
            vision_rope_theta=_rope_theta(vision),
            vision_num_hidden_layers=_int_field(vision, "num_hidden_layers"),
            vision_intermediate_size=_int_field(vision, "intermediate_size"),
            spatial_merge_size=_int_field(
                huggingface_config, "spatial_merge_size"
            ),
            hidden_size=_int_field(text, "hidden_size"),
            rms_norm_eps=_float_field(text, "rms_norm_eps"),
            tower_rms_norm_eps=getattr(vision, "rms_norm_eps", None) or 1e-5,
        )


def _int_field(config: PretrainedConfig, name: str) -> int:
    value = getattr(config, name, None)
    if value is None:
        raise ValueError(f"{type(config).__name__} has no {name}")
    return int(value)


def _float_field(config: PretrainedConfig, name: str) -> float:
    value = getattr(config, name, None)
    if value is None:
        raise ValueError(f"{type(config).__name__} has no {name}")
    return float(value)


def _rope_theta(vision: PretrainedConfig) -> float:
    """The tower's rope theta, from the axial ``rope_parameters`` block."""
    rope = getattr(vision, "rope_parameters", None)
    if rope is None:
        raise ValueError("vision_config has no rope_parameters")
    if isinstance(rope, PretrainedConfig):
        return _float_field(rope, "rope_theta")
    return float(rope["rope_theta"])


class MistralLarge4VisionAttention(Module[..., Tensor]):
    """Block-diagonal bidirectional attention over each image's patches.

    Every image is one ragged sequence in ``cu_seqlens``, so the ragged
    flash kernel with a null mask computes each image's full self-attention
    and images never see each other -- the same block-diagonal property a
    dense ``[1, 1, P, P]`` mask would encode.
    """

    n_heads: int
    head_dim: int
    wq: Linear
    wk: Linear
    wv: Linear
    wo: Linear

    def __init__(self, config: MistralLarge4VisionConfig) -> None:
        super().__init__()
        dim = config.vision_hidden_size
        self.n_heads = config.vision_num_attention_heads
        self.head_dim = config.vision_head_dim
        self.wq = Linear(in_dim=dim, out_dim=dim, bias=False)
        self.wk = Linear(in_dim=dim, out_dim=dim, bias=False)
        self.wv = Linear(in_dim=dim, out_dim=dim, bias=False)
        self.wo = Linear(in_dim=dim, out_dim=dim, bias=False)

    def forward(
        self,
        x: Tensor,
        position_embeddings: tuple[TensorValue, TensorValue],
        cu_seqlens: Tensor,
        max_seq_len: Tensor,
    ) -> Tensor:
        """Runs attention over the patches of every image.

        Args:
            x: Patch features, ``[total_patches, hidden]``.
            position_embeddings: Per-patch ``(cos, sin)``, each
                ``[total_patches, head_dim]``.
            cu_seqlens: Where each image's patches start, ``[images + 1]``.
            max_seq_len: The longest image's patch count, a scalar.
        """
        total_patches = x.shape[0]
        shape: list[int | Dim] = [
            total_patches,
            self.n_heads,
            self.head_dim,
        ]
        xq = TensorValue(self.wq(x).reshape(shape))
        xk = TensorValue(self.wk(x).reshape(shape))
        xv = TensorValue(self.wv(x).reshape(shape))

        # Axial 2D RoPE, broadcast over the heads.
        cos, sin = position_embeddings
        cos = ops.unsqueeze(cos, 1)
        sin = ops.unsqueeze(sin, 1)
        xq = xq * cos + rotate_half(xq) * sin
        xk = xk * cos + rotate_half(xk) * sin

        # The functional wrapper hands back a Tensor, not a graph value.
        output = flash_attention_ragged_gpu(
            xq,
            xk,
            xv,
            input_row_offsets=cu_seqlens,
            max_seq_len=max_seq_len,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=1.0 / math.sqrt(self.head_dim),
        ).reshape([total_patches, self.n_heads * self.head_dim])
        return self.wo(output)


class MistralLarge4VisionMLP(Module[[Tensor], Tensor]):
    """The tower block's SwiGLU feed-forward."""

    w1: Linear
    w2: Linear
    w3: Linear

    def __init__(self, config: MistralLarge4VisionConfig) -> None:
        super().__init__()
        hidden = config.vision_hidden_size
        intermediate = config.vision_intermediate_size
        self.w1 = Linear(in_dim=hidden, out_dim=intermediate, bias=False)
        self.w2 = Linear(in_dim=intermediate, out_dim=hidden, bias=False)
        self.w3 = Linear(in_dim=hidden, out_dim=intermediate, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Runs ``w2(silu(w1(x)) * w3(x))``."""
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MistralLarge4VisionTransformerBlock(Module[..., Tensor]):
    """One tower block: attention and feed-forward with pre-norms."""

    attention_norm: RMSNorm
    attention: MistralLarge4VisionAttention
    ffn_norm: RMSNorm
    feed_forward: MistralLarge4VisionMLP

    def __init__(self, config: MistralLarge4VisionConfig) -> None:
        super().__init__()
        hidden = config.vision_hidden_size
        eps = config.tower_rms_norm_eps
        self.attention_norm = RMSNorm(dim=hidden, eps=eps)
        self.attention = MistralLarge4VisionAttention(config)
        self.ffn_norm = RMSNorm(dim=hidden, eps=eps)
        self.feed_forward = MistralLarge4VisionMLP(config)

    def forward(
        self,
        x: Tensor,
        position_embeddings: tuple[TensorValue, TensorValue],
        cu_seqlens: Tensor,
        max_seq_len: Tensor,
    ) -> Tensor:
        """Runs the block's attention, then its feed-forward."""
        h = x + self.attention(
            self.attention_norm(x),
            position_embeddings,
            cu_seqlens,
            max_seq_len,
        )
        return h + self.feed_forward(self.ffn_norm(h))


class MistralLarge4VisionTransformer(Module[..., Tensor]):
    """The tower's block stack; there is no final norm."""

    layers: ModuleList

    def __init__(self, config: MistralLarge4VisionConfig) -> None:
        super().__init__()
        self.layers = ModuleList(
            MistralLarge4VisionTransformerBlock(config)
            for _ in range(config.vision_num_hidden_layers)
        )

    def forward(
        self,
        x: Tensor,
        position_embeddings: tuple[TensorValue, TensorValue],
        cu_seqlens: Tensor,
        max_seq_len: Tensor,
    ) -> Tensor:
        """Runs every block over the patch features."""
        for layer in self.layers:
            x = layer(x, position_embeddings, cu_seqlens, max_seq_len)
        return x


class MistralLarge4VisionEncoder(Module[..., Tensor]):
    """The Pixtral tower: conv patch embed, ``ln_pre`` and the blocks."""

    patch_conv: Tensor
    """The conv patch embedding, stored ``[hidden, channels, p, p]``."""

    ln_pre: RMSNorm
    patch_positional_embedding: RotaryEmbedding2D
    transformer: MistralLarge4VisionTransformer

    def __init__(self, config: MistralLarge4VisionConfig) -> None:
        super().__init__()
        hidden = config.vision_hidden_size
        # Keep the stored [out_ch, in_ch, kh, kw] layout so the weight binds
        # by shape; the reshape to [hidden, patch_dim] happens in the graph.
        self.patch_conv = Tensor.zeros(
            [
                hidden,
                config.num_channels,
                config.patch_size,
                config.patch_size,
            ]
        )
        self.ln_pre = RMSNorm(dim=hidden, eps=config.tower_rms_norm_eps)
        self.patch_positional_embedding = RotaryEmbedding2D(
            dim=hidden,
            n_heads=config.vision_num_attention_heads,
            theta=config.vision_rope_theta,
            max_patches_per_side=config.max_patches_per_side,
        )
        self.transformer = MistralLarge4VisionTransformer(config)
        self._hidden = hidden
        self._patch_dim = config.patch_dim

    def forward(
        self,
        pixel_patches: Tensor,
        position_ids: Tensor,
        cu_seqlens: Tensor,
        max_seq_len: Tensor,
    ) -> Tensor:
        """Encodes every image's patches.

        Args:
            pixel_patches: Flattened patches of all images, row-major,
                ``[total_patches, channels * patch_size**2]``.
            position_ids: Each patch's ``row * max_patches_per_side + col``
                index into the 2D RoPE table, ``[total_patches]``.
            cu_seqlens: Where each image's patches start, ``[images + 1]``.
            max_seq_len: The longest image's patch count, a scalar.

        Returns:
            The last block's features, ``[total_patches, hidden]``.
        """
        weight_2d = ops.reshape(
            TensorValue(self.patch_conv), [self._hidden, self._patch_dim]
        )
        patches = Tensor.from_graph_value(
            ops.cast(TensorValue(pixel_patches), weight_2d.dtype)
            @ ops.transpose(weight_2d, 0, 1)
        )
        h = self.ln_pre(patches)
        position_embeddings = self.patch_positional_embedding(
            h, TensorValue(position_ids)
        )
        return self.transformer(h, position_embeddings, cu_seqlens, max_seq_len)


class MistralLarge4PatchMerger(Module[[Tensor, Tensor], Tensor]):
    """Folds each square of patches into one row with a linear."""

    merging_layer: Linear

    def __init__(self, config: MistralLarge4VisionConfig) -> None:
        super().__init__()
        merge = config.spatial_merge_size
        self.merging_layer = Linear(
            in_dim=config.vision_hidden_size * merge**2,
            out_dim=config.vision_hidden_size,
            bias=False,
        )

    def forward(self, features: Tensor, merge_indices: Tensor) -> Tensor:
        """Merges the patches of every image.

        Args:
            features: Patch features of all images,
                ``[total_patches, hidden]``.
            merge_indices: For each merged row, the patch rows of its
                square, row-major, ``[merged_tokens, merge_size**2]``.

        Returns:
            The merged rows, ``[merged_tokens, hidden]``.
        """
        squares = F.gather(features, merge_indices, axis=0)
        # The reference unfolds the grid, which lays a square out with the
        # channel as the slowest axis.
        channel_major = squares.transpose(1, 2)
        merged_tokens, hidden, square = channel_major.shape
        return self.merging_layer(
            channel_major.reshape([merged_tokens, hidden * square])
        )


class MistralLarge4VisionLanguageAdapter(Module[[Tensor], Tensor]):
    """The projector's MLP: one linear up, GELU, one linear across."""

    w_in: Linear
    w_out: Linear

    def __init__(self, config: MistralLarge4VisionConfig) -> None:
        super().__init__()
        self.w_in = Linear(
            in_dim=config.vision_hidden_size,
            out_dim=config.hidden_size,
            bias=False,
        )
        self.w_out = Linear(
            in_dim=config.hidden_size, out_dim=config.hidden_size, bias=False
        )

    def forward(self, x: Tensor) -> Tensor:
        """Runs ``w_out(gelu(w_in(x)))``."""
        return self.w_out(F.gelu(self.w_in(x)))


class MistralLarge4Vision(Module[..., Tensor]):
    """The tower and projector, as the engine's vision submodel.

    One forward over a ragged batch of images emits one BF16
    ``[merged_tokens, hidden_size]`` row per merged patch square, in image
    order -- ready to scatter into the language model's prompt embeddings.
    """

    vision_encoder: MistralLarge4VisionEncoder
    pre_mm_projector_norm: RMSNorm
    patch_merger: MistralLarge4PatchMerger
    vision_language_adapter: MistralLarge4VisionLanguageAdapter

    def __init__(self, config: MistralLarge4VisionConfig) -> None:
        super().__init__()
        self.vision_encoder = MistralLarge4VisionEncoder(config)
        self.pre_mm_projector_norm = RMSNorm(
            dim=config.vision_hidden_size, eps=config.rms_norm_eps
        )
        self.patch_merger = MistralLarge4PatchMerger(config)
        self.vision_language_adapter = MistralLarge4VisionLanguageAdapter(
            config
        )

    def forward(
        self,
        pixel_patches: Tensor,
        position_ids: Tensor,
        cu_seqlens: Tensor,
        merge_indices: Tensor,
        max_seq_len: Tensor,
    ) -> Tensor:
        """Encodes the patches and projects them onto the text width.

        Args:
            pixel_patches: Flattened patches of all images, row-major,
                ``[total_patches, channels * patch_size**2]``.
            position_ids: Each patch's 2D RoPE table index,
                ``[total_patches]``.
            cu_seqlens: Where each image's patches start, ``[images + 1]``.
            merge_indices: The patch rows of each merged square,
                ``[merged_tokens, merge_size**2]``.
            max_seq_len: The longest image's patch count, a scalar.

        Returns:
            Post-adapter embeddings, ``[merged_tokens, hidden_size]``.
        """
        features = self.vision_encoder(
            pixel_patches, position_ids, cu_seqlens, max_seq_len
        )
        merged = self.patch_merger(
            self.pre_mm_projector_norm(features), merge_indices
        )
        return self.vision_language_adapter(merged)


def vision_input_types(
    config: MistralLarge4VisionConfig, device: DeviceRef
) -> list[TensorType]:
    """Returns the compile inputs of :class:`MistralLarge4Vision`.

    The ragged layout replaces the dense ``[1, 1, P, P]`` block-diagonal
    mask Small 4's ABI passed: the flash kernel derives image boundaries
    from ``cu_seqlens``, so no per-patch-pair bytes cross the host-device
    boundary.
    """
    return [
        TensorType(
            DType.float32,
            ["total_patches", config.patch_dim],
            device=device,
        ),
        TensorType(DType.int64, ["total_patches"], device=device),
        TensorType(DType.uint32, ["num_images_plus_1"], device=device),
        TensorType(
            DType.int32,
            ["merged_tokens", config.spatial_merge_size**2],
            device=device,
        ),
        TensorType(DType.uint32, [], device=DeviceRef.CPU()),
    ]
