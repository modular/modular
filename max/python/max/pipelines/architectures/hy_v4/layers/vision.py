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
"""Hy4 VL vision tower and its ``rmsnorm_mlp_adapter`` projector.

The tower is a 40-layer pre-LayerNorm ViT with a learned 128x128 position
table, bilinearly resampled to each image's patch grid, and full attention
inside each image. The projector applies RMSNorm, a 2x2 strided conv, GELU
and a 1x1 conv, and appends the learned ``image_newline`` after each row of
merged tokens. Module names follow the checkpoint's ``vit.*`` keys.

The host builds the gather indices with ``vision_index_inputs``.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from typing import ClassVar

import numpy as np
import numpy.typing as npt
from max import functional as F
from max.dtype import DType
from max.graph import DeviceRef, DimLike, TensorType
from max.nn import (
    Embedding,
    LayerNorm,
    Linear,
    Module,
    ModuleList,
    RMSNorm,
    Sequential,
    as_subgraph,
)
from max.nn.attention_mask import MHAMaskVariant
from max.nn.kernels import flash_attention_ragged_gpu
from max.nn.legacy.graph_groups import (
    VisionHostMax,
    VisionRowIndex,
    VisionRows,
    VisionSegments,
)
from max.tensor import Tensor, default_dtype

from ..model_config import HYV4VLConfig, HYV4VLVisionConfig


def _bilinear_axis(
    size: int, edge: int, copy: bool
) -> tuple[
    npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float32]
]:
    """Torch's bilinear source taps for ``scale_factor=(size + 0.1) / edge``.

    The reference passes a scale factor, not a size, so the sampling ratio is
    ``edge / (size + 0.1)`` rather than ``edge / size``. When the grid is the
    table's own size, torch's CUDA kernel copies the table instead (its CPU
    kernel does this per axis; the reference serves on CUDA).
    """
    ratio = np.float32(1.0 if copy else 1.0 / ((size + 0.1) / edge))
    src = np.maximum(
        ratio * (np.arange(size, dtype=np.float32) + np.float32(0.5))
        - np.float32(0.5),
        np.float32(0),
    )
    i0 = src.astype(np.int64)
    i1 = np.minimum(i0 + 1, edge - 1)
    return i0, i1, src - i0.astype(np.float32)


def vision_index_inputs(
    grids: Sequence[tuple[int, int]], config: HYV4VLVisionConfig
) -> dict[str, npt.NDArray[np.generic]]:
    """Builds the tower's index inputs for images packed in ``grids`` order.

    Args:
        grids: Each image's ``(h, w)`` patch grid.
        config: The vision config.

    Returns:
        ``pos_idx``/``pos_w`` ([patches, 4]), ``input_row_offsets``,
        ``max_seqlen``, ``fold_idx`` ([tokens, 4]) and ``newline`` ([tokens]).
    """
    edge = config.max_image_size // config.patch_size
    merge = config.spatial_merge_size
    pos_idx, pos_w, fold_idx, newline = [], [], [], []
    offsets = [0]
    for h, w in grids:
        copy = (h, w) == (edge, edge)
        y0, y1, ly = _bilinear_axis(h, edge, copy)
        x0, x1, lx = _bilinear_axis(w, edge, copy)
        # Row 0 of the table is an unused class-token slot.
        rows = [
            1 + ya[:, None] * edge + xa[None, :]
            for ya in (y0, y1)
            for xa in (x0, x1)
        ]
        wy, wx = (1 - ly, ly), (1 - lx, lx)
        weights = [
            wy[a][:, None] * wx[b][None, :] for a in (0, 1) for b in (0, 1)
        ]
        pos_idx.append(np.stack(rows, -1).reshape(-1, 4))
        pos_w.append(np.stack(weights, -1).reshape(-1, 4))

        # Each output row holds w // merge windows and then one newline, which
        # reads the row's last window and is replaced in the graph.
        cols = np.minimum(np.arange(w // merge + 1), w // merge - 1)
        top = np.arange(h // merge)[:, None] * merge * w + cols[None, :] * merge
        window = [dy * w + dx for dy in range(merge) for dx in range(merge)]
        fold_idx.append(
            offsets[-1]
            + (top[..., None] + np.array(window)).reshape(-1, merge * merge)
        )
        line = np.zeros((h // merge, w // merge + 1), dtype=np.bool_)
        line[:, -1] = True
        newline.append(line.reshape(-1))
        offsets.append(offsets[-1] + h * w)
    return {
        "pos_idx": np.concatenate(pos_idx).astype(np.uint32),
        "pos_w": np.concatenate(pos_w).astype(np.float32),
        "input_row_offsets": np.array(offsets, dtype=np.uint32),
        "max_seqlen": np.array([max(h * w for h, w in grids)], dtype=np.uint32),
        "fold_idx": np.concatenate(fold_idx).astype(np.uint32),
        "newline": np.concatenate(newline),
    }


class _Conv(Module[[Tensor], Tensor]):
    """A conv whose stride equals its kernel, as a matmul over its windows.

    Input rows are flattened windows in the weight's own ``(channel, ky, kx)``
    order.
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel: int
    ) -> None:
        self.weight = Tensor.zeros([out_channels, in_channels, kernel, kernel])
        self.bias = Tensor.zeros([out_channels])

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight.reshape([self.weight.shape[0], -1])
        return x @ weight.T + self.bias


class _GELU(Module[[Tensor], Tensor]):
    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(x)


class HYV4VLVisionEmbeddings(Module[..., Tensor]):
    """Patch embedding plus the bilinearly resampled position table."""

    def __init__(self, config: HYV4VLVisionConfig) -> None:
        edge = config.max_image_size // config.patch_size
        self.patch_embedding = _Conv(
            config.num_channels, config.hidden_size, config.patch_size
        )
        self.position_embedding = Embedding(
            edge * edge + 1, dim=config.hidden_size
        )

    def forward(
        self, pixel_values: Tensor, pos_idx: Tensor, pos_w: Tensor
    ) -> Tensor:
        """Embeds patch rows and adds their resampled positions."""
        table = self.position_embedding.weight
        patches = pixel_values.shape[0]
        # The reference resamples the table in fp32 and casts the result.
        taps = F.gather(table, pos_idx.reshape([patches * 4]), axis=0)
        taps = taps.reshape([patches, 4, table.shape[1]]).cast(DType.float32)
        positions = (taps * pos_w.reshape([patches, 4, 1])).sum(axis=1)
        positions = positions.reshape([patches, table.shape[1]]).cast(
            table.dtype
        )
        embeds = self.patch_embedding(pixel_values.cast(table.dtype))
        return embeds + positions


class HYV4VLVisionAttention(Module[..., Tensor]):
    """Bidirectional attention inside each image, all projections biased."""

    def __init__(self, config: HYV4VLVisionConfig) -> None:
        hidden = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = hidden // self.num_heads
        self.q_proj = Linear(hidden, hidden, bias=True)
        self.k_proj = Linear(hidden, hidden, bias=True)
        self.v_proj = Linear(hidden, hidden, bias=True)
        self.o_proj = Linear(hidden, hidden, bias=True)

    def forward(
        self, x: Tensor, input_row_offsets: Tensor, max_seqlen: Tensor
    ) -> Tensor:
        """Attends within each segment of ``input_row_offsets``."""
        shape: list[DimLike] = [x.shape[0], self.num_heads, self.head_dim]
        out = flash_attention_ragged_gpu(
            self.q_proj(x).reshape(shape),
            self.k_proj(x).reshape(shape),
            self.v_proj(x).reshape(shape),
            input_row_offsets=input_row_offsets,
            max_seq_len=max_seqlen,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=self.head_dim**-0.5,
        )
        return self.o_proj(out.reshape(x.shape))


class HYV4VLVisionMLP(Module[[Tensor], Tensor]):
    """``dense_4h_to_h(gelu(dense_h_to_4h(x)))``, biased."""

    def __init__(self, config: HYV4VLVisionConfig) -> None:
        self.dense_h_to_4h = Linear(
            config.hidden_size, config.intermediate_size, bias=True
        )
        self.dense_4h_to_h = Linear(
            config.intermediate_size, config.hidden_size, bias=True
        )

    def forward(self, x: Tensor) -> Tensor:
        """Applies the MLP."""
        return self.dense_4h_to_h(F.gelu(self.dense_h_to_4h(x)))


class HYV4VLVisionBlock(Module[..., Tensor]):
    """Pre-LayerNorm attention and MLP, each with a residual."""

    def __init__(self, config: HYV4VLVisionConfig) -> None:
        hidden, eps = config.hidden_size, config.rms_norm_eps
        self.input_layernorm = LayerNorm(hidden, eps=eps, keep_dtype=False)
        self.self_attn = HYV4VLVisionAttention(config)
        self.post_attention_layernorm = LayerNorm(
            hidden, eps=eps, keep_dtype=False
        )
        self.mlp = HYV4VLVisionMLP(config)

    def forward(
        self, x: Tensor, input_row_offsets: Tensor, max_seqlen: Tensor
    ) -> Tensor:
        """Applies one pre-LayerNorm block."""
        x = x + self.self_attn(
            self.input_layernorm(x), input_row_offsets, max_seqlen
        )
        return x + self.mlp(self.post_attention_layernorm(x))


class HYV4VLPerceive(Module[..., Tensor]):
    """The ``rmsnorm_mlp_adapter`` projector to the text width."""

    def __init__(self, config: HYV4VLVisionConfig) -> None:
        hidden, merge = config.hidden_size, config.spatial_merge_size
        merged = hidden * merge * merge
        self.norm = RMSNorm(
            hidden, eps=config.rms_norm_eps, multiply_before_cast=True
        )
        self.proj = Sequential(
            _Conv(hidden, merged, merge),
            _GELU(),
            _Conv(merged, config.out_hidden_size, 1),
        )
        self.image_newline = Tensor.zeros([config.out_hidden_size])

    def forward(self, x: Tensor, fold_idx: Tensor, newline: Tensor) -> Tensor:
        """Projects each 2x2 window; newline tokens take ``image_newline``."""
        tokens, taps = fold_idx.shape
        windows = F.gather(
            self.norm(x), fold_idx.reshape([tokens * taps]), axis=0
        )
        # [tokens, (ky, kx), channel] -> the conv weight's (channel, ky, kx).
        windows = windows.reshape([tokens, taps, x.shape[1]]).transpose(1, 2)
        out = self.proj(windows.reshape([tokens, -1]))
        return F.where(newline.reshape([tokens, 1]), self.image_newline, out)


class HYV4VLVisionModel(Module[..., Tensor]):
    """The vision tower and projector: one row per image token."""

    def __init__(self, config: HYV4VLVisionConfig, device: DeviceRef) -> None:
        self.config = config
        self.device = device
        self.embeddings = HYV4VLVisionEmbeddings(config)
        self.layers = ModuleList(
            [HYV4VLVisionBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.perceive = HYV4VLPerceive(config)

    def forward(
        self,
        pixel_values: Tensor,
        pos_idx: Tensor,
        pos_w: Tensor,
        input_row_offsets: Tensor,
        max_seqlen: Tensor,
        fold_idx: Tensor,
        newline: Tensor,
    ) -> Tensor:
        """Returns one row per image token, images in packing order."""
        h = self.embeddings(pixel_values, pos_idx, pos_w)
        for layer in self.layers:
            h = as_subgraph(layer, name="hy_v4_vision_block")(
                h, input_row_offsets, max_seqlen
            )
        return self.perceive(h, fold_idx, newline)

    def input_types(self) -> OrderedDict[str, object]:
        """The inputs in ``forward`` order, grouped as the engine packs them."""
        patch_dim = self.config.num_channels * self.config.patch_size**2
        device = self.device
        return OrderedDict(
            patches=VisionRows(
                pixel_values=TensorType(
                    DType.float32, ["vision_patches", patch_dim], device
                ),
                pos_idx=TensorType(DType.uint32, ["vision_patches", 4], device),
                pos_w=TensorType(DType.float32, ["vision_patches", 4], device),
            ),
            segments=VisionSegments(
                input_row_offsets=TensorType(
                    DType.uint32, ["vision_segments"], device
                )
            ),
            segment_bound=VisionHostMax(
                max_seqlen=TensorType(DType.uint32, [1], DeviceRef.CPU())
            ),
            fold=VisionRowIndex(
                fold_idx=TensorType(DType.uint32, ["image_tokens", 4], device)
            ),
            tokens=VisionRows(
                newline=TensorType(DType.bool, ["image_tokens"], device)
            ),
        )


class HYV4VLVisionEncoder(HYV4VLVisionModel):
    """The tower and projector, registered for the ``VISION`` role.

    It runs on the first device and gives BF16 rows of the text width.
    """

    graph_name: ClassVar[str] = "hy_v4_vl_vision"

    def __init__(self, config: HYV4VLConfig) -> None:
        with default_dtype(DType.bfloat16):
            super().__init__(config.vision, config.devices[0])
        self.embedding_dtype = DType.bfloat16
        self.embedding_dim = config.vision.out_hidden_size
