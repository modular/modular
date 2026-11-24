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
"""The Qwen3-VL vision tower, registered for the ``VISION`` role.

Every block attends over each image or clip segment in full, the MLPs
activate with the tanh approximation of GELU, every norm is a LayerNorm, and
positions come from a bilinearly interpolated learned grid plus a rotary
table. The tower runs on one device. Each output row is one merged patch:
its embedding, then the deepstack embeddings of the blocks
``deepstack_visual_indexes`` names, each as wide as the decoder's hidden
state.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import ClassVar

from max import functional as F
from max._artifact.vision import VisionMetadata
from max.dtype import DType
from max.graph import DeviceRef, Dim, DimLike, TensorType
from max.nn import Embedding, LayerNorm, Linear, Module, as_subgraph
from max.nn.kernels import spatial_merge
from max.nn.legacy.graph_groups import (
    VisionHostMax,
    VisionRows,
    VisionSegments,
)
from max.nn.sequential import ModuleList
from max.pipelines.lib import PipelineConfig
from max.pipelines.lib.graph_specs import GraphSpec
from max.tensor import Tensor, default_dtype

from ..model_config import Qwen3VLConfig, VisionConfig
from .attention import VisionAttention
from .preprocess import (
    image_preprocess_spec,
    qwen3vl_vision_metadata,
    video_preprocess_spec,
)

__all__ = ["Qwen3VLVisionEncoder", "Qwen3VLVisionTransformer"]

_ROTARY_THETA = 10000.0
_NORM_EPS = 1e-6


class _PatchProjection(Module[[Tensor], Tensor]):
    """The checkpoint's Conv3D patch projection, applied as the linear map.

    The weight keeps the stored ``[embed_dim, in_channels,
    temporal_patch_size, patch_size, patch_size]`` shape. Patches don't
    overlap, so the convolution is the matmul with that weight flattened.
    """

    def __init__(self, config: VisionConfig) -> None:
        self.embed_dim = config.hidden_size
        self.patch_dim = (
            config.in_channels
            * config.temporal_patch_size
            * config.patch_size
            * config.patch_size
        )
        self.weight = Tensor.zeros(
            [
                config.hidden_size,
                config.in_channels,
                config.temporal_patch_size,
                config.patch_size,
                config.patch_size,
            ]
        )
        self.bias = Tensor.zeros([config.hidden_size])

    def forward(self, x: Tensor) -> Tensor:
        """Projects ``[n_patches, patch_dim]`` patches to the embedding."""
        weight = self.weight.reshape([self.embed_dim, self.patch_dim])
        return x.cast(weight.dtype) @ weight.T + self.bias


class VisionPatchEmbed(Module[[Tensor], Tensor]):
    """Embeds each flattened patch."""

    def __init__(self, config: VisionConfig) -> None:
        self.proj = _PatchProjection(config)

    @property
    def patch_dim(self) -> int:
        """The width of a flattened patch."""
        return self.proj.patch_dim

    def forward(self, x: Tensor) -> Tensor:
        """Projects ``[n_patches, patch_dim]`` patches to the embedding."""
        return self.proj(x)


class BilinearPositionEmbedding(Module[..., Tensor]):
    """Interpolates the learned square position grid to each patch.

    Each position takes its four neighbours in the grid, weighted
    bilinearly. The grid is interpolated once per frame; each grid item's
    rows are then repeated over its temporal patches and reordered into the
    merge blocks the patches arrive in.
    """

    def __init__(self, config: VisionConfig) -> None:
        self.dim = config.hidden_size
        self.spatial_merge_size = config.spatial_merge_size
        self.embedding = Embedding(
            config.num_position_embeddings, dim=config.hidden_size
        )

    def forward(
        self,
        idxs: Tensor,
        weights: Tensor,
        grid_thw: Tensor,
        out_rows: DimLike,
    ) -> Tensor:
        """Returns each patch's position embedding.

        Args:
            idxs: ``[4, sum(h * w)]`` grid neighbours of each position.
            weights: ``[4, sum(h * w), 1]`` their interpolation weights.
            grid_thw: ``[n_images, 3]`` each item's ``(t, h, w)``.
            out_rows: The patch count, ``sum(t * h * w)``.

        Returns:
            ``[out_rows, hidden_size]``.
        """
        weights = weights.cast(self.embedding.weight.dtype)
        weighted = [
            self.embedding(idxs[i, :]) * weights[i, :] for i in range(4)
        ]
        embeds = weighted[0] + weighted[1] + weighted[2] + weighted[3]
        return spatial_merge(
            embeds,
            grid_thw,
            self.dim,
            self.spatial_merge_size,
            out_rows=out_rows,
        )


def rotary_table(
    rot_pos_ids: Tensor,
    max_grid_size: Tensor,
    seq_len: Dim,
    head_dim: int,
) -> tuple[Tensor, Tensor]:
    """Returns each patch's rotary ``(cos, sin)``, ``[seq_len, head_dim]``.

    Half of each head rotates by the patch's row and half by its column.

    Args:
        rot_pos_ids: ``[seq_len, 2]`` each patch's ``(row, column)``.
        max_grid_size: The largest row or column, an int32 host scalar.
        seq_len: The patch count.
        head_dim: The attention head width.
    """
    device = rot_pos_ids.device
    n = head_dim // 2
    # In float64, so the exponential can't overflow.
    iota = F.range(
        0, n - 1, 2, out_dim=n // 2, device=device, dtype=DType.float64
    )
    inv_freqs = (1.0 / (_ROTARY_THETA ** (iota / n))).cast(DType.float32)
    positions = F.range(
        0,
        max_grid_size,
        1,
        out_dim="max_grid_size",
        device=device,
        dtype=DType.float32,
    )
    table = F.gather(F.outer(positions, inv_freqs), rot_pos_ids, axis=0)
    freqs = F.flatten(table, 1).reshape([seq_len, -1])
    freqs = F.concat([freqs, freqs], axis=-1)
    return F.cos(freqs), F.sin(freqs)


class VisionMLP(Module[[Tensor], Tensor]):
    """Two projections around a tanh-approximated GELU."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        self.linear_fc1 = Linear(hidden_size, intermediate_size, bias=True)
        self.linear_fc2 = Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """Runs the feed-forward over ``x``."""
        return self.linear_fc2(F.gelu(self.linear_fc1(x), approximate="tanh"))


class VisionBlock(Module[..., Tensor]):
    """A pre-norm block: attention, then the MLP."""

    def __init__(self, config: VisionConfig) -> None:
        self.norm1 = LayerNorm(
            config.hidden_size, eps=_NORM_EPS, keep_dtype=False
        )
        self.norm2 = LayerNorm(
            config.hidden_size, eps=_NORM_EPS, keep_dtype=False
        )
        self.attn = VisionAttention(
            config.hidden_size, config.num_attention_heads
        )
        self.mlp = VisionMLP(config.hidden_size, config.intermediate_size)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: Tensor,
    ) -> Tensor:
        """Runs the block over the patches."""
        h = x + self.attn(self.norm1(x), cos, sin, cu_seqlens, max_seqlen)
        return h + self.mlp(self.norm2(h))


class VisionPatchMerger(Module[[Tensor], Tensor]):
    """Projects each merge block of patches into the decoder's width.

    A block's patch features are normalized, concatenated and run through a
    two-layer MLP. A deepstack merger normalizes the concatenated block
    instead of each patch.
    """

    def __init__(
        self, config: VisionConfig, *, use_postshuffle_norm: bool
    ) -> None:
        self.hidden_size = config.hidden_size
        self.input_dim = config.hidden_size * config.spatial_merge_size**2
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = LayerNorm(
            self.input_dim if use_postshuffle_norm else config.hidden_size,
            eps=_NORM_EPS,
            keep_dtype=False,
        )
        self.linear_fc1 = Linear(self.input_dim, self.input_dim, bias=True)
        self.linear_fc2 = Linear(
            self.input_dim, config.out_hidden_size, bias=True
        )

    def forward(self, x: Tensor) -> Tensor:
        """Maps ``[n_patches, hidden]`` to ``[n_patches / merge**2, out]``."""
        n_patches, hidden_size = x.shape
        factor = self.input_dim // self.hidden_size
        # The patch count is a multiple of the merge unit by construction,
        # which a dynamic dim doesn't carry into the reshape.
        x = F.rebind(x, [(n_patches // factor) * factor, hidden_size])
        if self.use_postshuffle_norm:
            x = x.reshape([n_patches // factor, self.input_dim])
        x = self.norm(x).reshape([n_patches // factor, self.input_dim])
        return self.linear_fc2(F.gelu(self.linear_fc1(x)))


class Qwen3VLVisionTransformer(Module[..., Tensor]):
    """The vision tower: patches in, one row per merge block out."""

    def __init__(self, config: VisionConfig, embedding_dtype: DType) -> None:
        self.vision_config = config
        self.embedding_dtype = embedding_dtype
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.deepstack_visual_indexes = list(config.deepstack_visual_indexes)
        self.patch_embed = VisionPatchEmbed(config)
        self.pos_embed = BilinearPositionEmbedding(config)
        self.blocks = ModuleList(
            [VisionBlock(config) for _ in range(config.depth)]
        )
        self.merger = VisionPatchMerger(config, use_postshuffle_norm=False)
        self.deepstack_merger_list = ModuleList(
            [
                VisionPatchMerger(config, use_postshuffle_norm=True)
                for _ in self.deepstack_visual_indexes
            ]
        )

    def forward(
        self,
        pixel_values: Tensor,
        weights: Tensor,
        idxs: Tensor,
        rot_pos_ids: Tensor,
        max_grid_size: Tensor,
        grid_thw: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: Tensor,
    ) -> Tensor:
        """Encodes the batch's patches.

        Args:
            pixel_values: ``[seq_len, patch_dim]`` float32 patches, a
                ``(t, h, w)`` grid per item, its merge blocks contiguous.
            weights: ``[4, sum(h * w), 1]`` bilinear weights of each
                position's grid neighbours. The grid is interpolated once
                per frame, so it has its own row count.
            idxs: ``[4, sum(h * w)]`` those neighbours.
            rot_pos_ids: ``[seq_len, 2]`` each patch's ``(row, column)``.
            max_grid_size: The largest row or column, an int32 host scalar.
            grid_thw: ``[n_images, 3]`` each item's ``(t, h, w)``.
            cu_seqlens: ``[n_segments + 1]`` uint32 offsets of the attention
                segments, one per image or temporal patch of a clip.
            max_seqlen: The longest segment, ``[1]`` uint32 on the host.

        Returns:
            ``[seq_len / merge**2, out_hidden_size * (1 + deepstack)]``
            rows in ``embedding_dtype``: each merged patch's embedding, then
            its deepstack embeddings in block order.
        """
        h = self.patch_embed(pixel_values)
        seq_len = h.shape[0]
        h = h + self.pos_embed(idxs, weights, grid_thw, seq_len)
        cos, sin = rotary_table(
            rot_pos_ids, max_grid_size, seq_len, self.head_dim
        )
        deepstack: list[Tensor] = []
        # Every block shares one body; each call binds its own weights.
        for index, block in enumerate(self.blocks):
            h = as_subgraph(block, name="vision_block")(
                h, cos, sin, cu_seqlens, max_seqlen
            )
            if index in self.deepstack_visual_indexes:
                merger = self.deepstack_merger_list[
                    self.deepstack_visual_indexes.index(index)
                ]
                deepstack.append(merger(h))
        rows = F.concat([self.merger(h), *deepstack], axis=-1)
        return rows.cast(self.embedding_dtype)


class Qwen3VLVisionEncoder(Qwen3VLVisionTransformer):
    """The vision tower, registered for the ``VISION`` role.

    Its graph takes the inputs named as a serving engine packs them, one
    group per packing rule, and runs on the config's first device.
    """

    graph_name: ClassVar[str] = "qwen3vl_vision"

    def __init__(self, config: Qwen3VLConfig) -> None:
        with default_dtype(config.declared_dtype):
            super().__init__(config.vision_config, config.declared_dtype)
        self.device_ref = config.devices[0]
        self.embedding_dim = config.vision_config.embedding_width

    def input_types(self) -> OrderedDict[str, object]:
        """Returns the tower's inputs, grouped as a serving engine packs them.

        The fields are :meth:`forward`'s arguments, in its graph's order.
        ``vision_seq_len`` counts the batch's patches and ``vision_pos_len``
        its interpolated positions, one per patch of each frame.
        """
        device = self.device_ref
        cpu = DeviceRef.CPU()
        return OrderedDict(
            patches=VisionRows(
                pixel_values=TensorType(
                    DType.float32,
                    ["vision_seq_len", self.patch_embed.patch_dim],
                    device,
                ),
                weights=TensorType(
                    DType.float32, [4, "vision_pos_len", 1], device
                ),
                idxs=TensorType(DType.int64, [4, "vision_pos_len"], device),
                rot_pos_ids=TensorType(
                    DType.int32, ["vision_seq_len", 2], device
                ),
            ),
            grid_bound=VisionHostMax(
                max_grid_size=TensorType(DType.int32, [], cpu)
            ),
            grids=VisionRows(
                grid_thw=TensorType(DType.int64, ["n_images", 3], device)
            ),
            segments=VisionSegments(
                cu_seqlens=TensorType(DType.uint32, ["n_seqlens"], device)
            ),
            segment_bound=VisionHostMax(
                max_seqlen=TensorType(DType.uint32, [1], cpu)
            ),
        )

    @classmethod
    def vision_metadata(
        cls, pipeline_config: PipelineConfig, config: Qwen3VLConfig
    ) -> VisionMetadata | None:
        """Returns the media layout the checkpoint is served with."""
        del config
        huggingface_config = pipeline_config.model.huggingface_config
        assert huggingface_config is not None
        return qwen3vl_vision_metadata(huggingface_config)

    @classmethod
    def vision_preprocess_spec(
        cls, pipeline_config: PipelineConfig, config: Qwen3VLConfig
    ) -> GraphSpec | None:
        """Returns the graph that preprocesses one image for the tower."""
        del pipeline_config
        return image_preprocess_spec(config.vision_config)

    @classmethod
    def vision_preprocess_video_spec(
        cls, pipeline_config: PipelineConfig, config: Qwen3VLConfig
    ) -> GraphSpec | None:
        """Returns the graph that preprocesses one temporal patch of a clip.

        The tower attends within each temporal patch, so a patch is a
        clip's attention segment.
        """
        del pipeline_config
        return video_preprocess_spec(config.vision_config)
