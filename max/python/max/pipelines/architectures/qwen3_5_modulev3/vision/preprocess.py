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
# Ported to ModuleV3 from
# max/python/max/pipelines/architectures/qwen3_5/preprocess_graph.py and the
# vision hooks of qwen3_5/model.py at main 4df8111afba.
"""Qwen3.5 image and video preprocessing as CPU graphs a serving engine runs.

The image graph does for one image what the tokenizer and
``pack_uncached_images`` do: it drops the alpha channel, resizes the image
with Pillow's antialiased bicubic filter, bit for bit, normalizes and
patchifies it, and computes the tower's bilinear position tables, rotary
positions, grid and segment bounds. The video graph does the same for one
temporal patch of a clip, the frames the tower attends over together, as
``video_processing.patchify_clip`` does. Each also returns the decoder's
M-RoPE positions for the item's placeholder tokens, as ``get_rope_index``
lays out a ``(1, h, w)`` grid.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
from max import functional as F
from max._artifact.vision import (
    ModalityLayout,
    PerClip,
    PromptText,
    SmartResize,
    Span,
    TemplateMarkers,
    VideoSmartResize,
    VisionMetadata,
)
from max.driver import CPU
from max.dtype import DType
from max.graph import Dim
from max.nn import Module
from max.nn.legacy.graph_groups import (
    ItemPositions,
    VisionHostMax,
    VisionRows,
    VisionSegments,
)
from max.nn.legacy.image_preprocess import (
    image_preprocess_input_types,
    resize_bicubic_to,
    video_preprocess_input_types,
)
from max.pipelines.lib.graph_specs import GraphSpec
from max.tensor import Tensor
from transformers import AutoConfig

from ..data_processing import QWEN3_5_MAX_PIXELS
from ..model_config import VisionConfig
from ..tokenizer import VISION_END_TOKEN, VISION_START_TOKEN
from ..video_processing import (
    QWEN3_5_VIDEO_MAX_PIXELS,
    QWEN3_5_VIDEO_MIN_PIXELS,
)

__all__ = [
    "Qwen3_5ImagePreprocess",
    "image_preprocess_spec",
    "qwen3_5_vision_metadata",
    "video_preprocess_spec",
]

_resize_bicubic_to = F.functional(resize_bicubic_to)


def _linspace(side: Dim, stop: int) -> Tensor:
    """Returns ``np.linspace(0, stop, side, dtype=np.float32)``.

    Like NumPy, it steps in float64, pins the last value to ``stop`` and
    casts.
    """
    cpu = CPU()
    index = F.range(0, side, 1, dtype=DType.int64, device=cpu).cast(
        DType.float64
    )
    last = F.shape_to_tensor([side]).cast(DType.float64) - 1.0
    step = float(stop) / F.max(last, 1.0)
    values = F.where(
        F.equal(index, last), float(stop) * F.min(last, 1.0), index * step
    )
    return values.cast(DType.float32)


class Qwen3_5ImagePreprocess(Module[..., OrderedDict[str, object]]):
    """Turns one RGBA8 image, or one temporal patch of a clip, into tower inputs.

    Takes ``image_preprocess_input_types()``, the size being the
    ``smart_resize`` target, or ``video_preprocess_input_types()`` with
    ``temporal_patch_size`` frames and one size, the clip's target. Returns
    the tower's input tree for the item, on the CPU, and the item's
    :class:`~max.nn.legacy.graph_groups.ItemPositions`.
    """

    def __init__(
        self,
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        num_position_embeddings: int,
    ) -> None:
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.num_grid_per_side = int(num_position_embeddings**0.5)

    def forward(self, image: Tensor, size: Tensor) -> OrderedDict[str, object]:
        """Returns the tower's input tree and positions for one item."""
        patch = self.patch_size
        merge = self.merge_size
        temporal = self.temporal_patch_size
        cpu = CPU()

        # The tokenizer converts to RGB, which drops alpha rather than
        # compositing it.
        frames = _resize_bicubic_to(
            image[:, :, :, :3], size.reshape([2]), patch * merge
        ).permute([1, 0, 2, 3])
        # Dividing by 127.5 in the graph lands a float32 ulp from numpy's
        # quotient for about half the levels. A table of the 256 levels,
        # divided in numpy as the tokenizer does, keeps it exact.
        levels = np.arange(256, dtype=np.float32)
        table = F.constant(
            (levels - 127.5) / np.float32(127.5), DType.float32, cpu
        )
        normalized = F.gather(table, frames.cast(DType.int32), axis=0)

        channels, _, height, width = normalized.shape
        # A still image fills every frame of its temporal patch.
        normalized = F.broadcast_to(
            normalized, [channels, temporal, height, width]
        )
        blocks_h = height // (patch * merge)
        blocks_w = width // (patch * merge)
        grid_h, grid_w = blocks_h * merge, blocks_w * merge
        seq = grid_h * grid_w
        # Rows run over merge blocks, then the patches within a block, so a
        # block's patches stay contiguous for the patch merger.
        pixel_values = (
            normalized.reshape(
                [
                    channels,
                    temporal,
                    blocks_h,
                    merge,
                    patch,
                    blocks_w,
                    merge,
                    patch,
                ]
            )
            .permute([2, 5, 3, 6, 0, 1, 4, 7])
            .reshape([seq, channels * temporal * patch * patch])
        )

        merged: list[Dim | int] = [blocks_h, blocks_w, merge, merge]

        def ids(side: Dim, shape: list[Dim | int]) -> Tensor:
            # Each patch's row or column, laid out like the rows above.
            positions = F.range(0, side, 1, dtype=DType.int32, device=cpu)
            return F.broadcast_to(positions.reshape(shape), merged)

        rot_pos_ids = F.stack(
            [
                ids(grid_h, [blocks_h, 1, merge, 1]),
                ids(grid_w, [1, blocks_w, 1, merge]),
            ],
            axis=-1,
        ).reshape([seq, 2])

        # The four neighbours of each patch in the learned position grid,
        # row-major over the patch grid: offsets in float64 from float32
        # positions, cast to float32 at the end.
        side = self.num_grid_per_side
        plane: list[Dim | int] = [grid_h, grid_w]
        h_pos, w_pos = _linspace(grid_h, side - 1), _linspace(grid_w, side - 1)
        h_floor, w_floor = h_pos.cast(DType.int64), w_pos.cast(DType.int64)
        h_ceil, w_ceil = (F.min(f + 1, side - 1) for f in (h_floor, w_floor))

        def rows(values: Tensor) -> Tensor:
            return F.broadcast_to(values.reshape([grid_h, 1]), plane)

        def cols(values: Tensor) -> Tensor:
            return F.broadcast_to(values.reshape([1, grid_w]), plane)

        dh = rows(h_pos.cast(DType.float64) - h_floor.cast(DType.float64))
        dw = cols(w_pos.cast(DType.float64) - w_floor.cast(DType.float64))
        w11 = dh * dw
        weights = (
            F.stack([1.0 - dh - dw + w11, dw - w11, dh - w11, w11], axis=0)
            .reshape([4, seq, 1])
            .cast(DType.float32)
        )
        top, bottom = rows(h_floor * side), rows(h_ceil * side)
        left, right = cols(w_floor), cols(w_ceil)
        idxs = F.stack(
            [top + left, top + right, bottom + left, bottom + right], axis=0
        ).reshape([4, seq])

        max_grid_size = F.max(
            F.shape_to_tensor([grid_h]), F.shape_to_tensor([grid_w])
        )

        # A placeholder token per merge block, row-major. Timestamp
        # semantics give a temporal patch of a clip a one-frame grid too, so
        # the temporal axis is constant. The next text continues one past
        # the largest position, as in get_rope_index.
        tokens = blocks_h * blocks_w
        blocks: list[Dim | int] = [blocks_h, blocks_w]

        def block_ids(side: Dim, shape: list[Dim | int]) -> Tensor:
            positions = F.range(0, side, 1, dtype=DType.int64, device=cpu)
            return F.broadcast_to(positions.reshape(shape), blocks).reshape(
                [tokens]
            )

        position_ids = F.stack(
            [
                F.broadcast_to(
                    F.constant(0, DType.int64, cpu).reshape([1]), [tokens]
                ),
                block_ids(blocks_h, [blocks_h, 1]),
                block_ids(blocks_w, [1, blocks_w]),
            ],
            axis=0,
        )
        position_advance = F.max(position_ids.reshape([3 * tokens])) + 1
        return OrderedDict(
            patches=VisionRows(
                pixel_values=(pixel_values,),
                weights=(weights,),
                idxs=(idxs,),
                rot_pos_ids=(rot_pos_ids,),
            ),
            grid_bound=VisionHostMax(
                max_grid_size=(max_grid_size.reshape([]).cast(DType.int32),)
            ),
            grids=VisionRows(
                grid_thw=(
                    F.shape_to_tensor([1, grid_h, grid_w])
                    .cast(DType.int64)
                    .reshape([1, 3]),
                )
            ),
            segments=VisionSegments(
                cu_seqlens=(F.shape_to_tensor([0, seq]).cast(DType.uint32),)
            ),
            segment_bound=VisionHostMax(
                max_seqlen=(F.shape_to_tensor([seq]).cast(DType.uint32),)
            ),
            positions=ItemPositions(
                position_ids=position_ids,
                position_advance=position_advance.reshape([]),
            ),
        )


def _preprocess(vision: VisionConfig) -> Qwen3_5ImagePreprocess:
    """Returns the preprocessing module for a checkpoint's tower."""
    return Qwen3_5ImagePreprocess(
        patch_size=vision.patch_size,
        temporal_patch_size=vision.temporal_patch_size,
        merge_size=vision.spatial_merge_size,
        num_position_embeddings=vision.num_position_embeddings,
    )


def image_preprocess_spec(vision: VisionConfig) -> GraphSpec:
    """Returns the graph that preprocesses one image for the tower.

    It does what the tokenizer and the vision input packing do.
    """
    return GraphSpec(
        _preprocess(vision),
        "qwen3_5_vision_preprocess",
        image_preprocess_input_types(),
        weights={},
    )


def video_preprocess_spec(vision: VisionConfig) -> GraphSpec:
    """Returns the graph that preprocesses one temporal patch of a clip.

    It does for the tower what the video processor does.
    """
    return GraphSpec(
        _preprocess(vision),
        "qwen3_5_vision_preprocess_video",
        video_preprocess_input_types(vision.temporal_patch_size, 1),
        weights={},
    )


def qwen3_5_vision_metadata(
    huggingface_config: AutoConfig,
) -> VisionMetadata | None:
    """Returns the media layout a Qwen3.5 checkpoint is served with.

    Returns ``None`` for a text-only checkpoint. The chat template writes the
    vision start and end markers around each placeholder as text, so the
    spliced runs are bare placeholders.
    """
    vision = getattr(huggingface_config, "vision_config", None)
    if vision is None:
        return None
    markers = TemplateMarkers(
        huggingface_config.vision_start_token_id,
        huggingface_config.vision_end_token_id,
    )
    return VisionMetadata.from_layouts(
        image=ModalityLayout(
            patch_size=vision.patch_size,
            merge_size=vision.spatial_merge_size,
            # The min_pixels Qwen3_5Tokenizer gives its image processor.
            strategy=SmartResize(
                min_pixels=65536, max_pixels=QWEN3_5_MAX_PIXELS
            ),
            span=Span(huggingface_config.image_token_id),
            template_markers=markers,
        ),
        video=ModalityLayout(
            patch_size=vision.patch_size,
            merge_size=vision.spatial_merge_size,
            strategy=VideoSmartResize(
                temporal_patch_size=vision.temporal_patch_size,
                min_pixels=QWEN3_5_VIDEO_MIN_PIXELS,
                max_pixels=QWEN3_5_VIDEO_MAX_PIXELS,
            ),
            span=Span(huggingface_config.video_token_id),
            template_markers=markers,
            # The clip budget bounds every frame at once, so a clip's frame
            # count changes how each frame is resized.
            video_layout=PerClip(),
        ),
        # MAX declares no byte caps for this family; 512 MiB is the serving
        # engine's own video cap, which real lecture clips need.
        max_image_bytes=0,
        max_video_bytes=512 << 20,
        prompt=PromptText(
            video_clip_open=VISION_START_TOKEN,
            video_clip_close=VISION_END_TOKEN,
        ),
    )
