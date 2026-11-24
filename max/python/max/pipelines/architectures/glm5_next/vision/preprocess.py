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
"""GLM-5.3-Flash image preprocessing as a CPU graph a serving engine runs.

The graph does for one image what :class:`.image_processor.Glm5NextImageProcessor`
and the batch processor's packing do: it drops the alpha channel, resizes the
content to fit inside the canvas the engine laid out with Pillow's antialiased
bicubic filter, bit for bit, zero-pads the rest of the canvas in pixel space,
rescales and normalizes with the processor's own arithmetic, patchifies in the
tower's row order and computes the tower's rotary positions and segment
bounds. A still image fills its temporal patch with copies of itself.

The engine sizes the canvas by the same token-budget ``smart_resize`` the
processor uses (``glm5_next_smart_resize`` in ``vision_config.json``), so the
graph only derives the content size from the canvas and the source size.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
from max import functional as F
from max._artifact.vision import (
    Glm5NextSmartResize,
    ModalityLayout,
    PromptText,
    Span,
    TemplateMarkers,
    VisionMetadata,
)
from max.driver import CPU
from max.dtype import DType
from max.graph import Dim, ops
from max.nn import Module
from max.nn.legacy.graph_groups import VisionHostMax, VisionRows, VisionSegments
from max.nn.legacy.image_preprocess import (
    image_preprocess_input_types,
    resize_bicubic_to,
)
from max.pipelines.lib.graph_specs import GraphSpec
from max.tensor import Tensor
from transformers import AutoConfig

from ..tokenizer import _load_processor_config
from .image_processor import Glm5NextImageProcessor

__all__ = [
    "Glm5NextImagePreprocess",
    "glm5_next_vision_metadata",
    "image_preprocess_spec",
]

_resize_bicubic_to = F.functional(resize_bicubic_to)
# The functional wrapper names no output dims; the graph op does.
_broadcast_to = F.functional(ops.broadcast_to)


def _normalization_table(processor: Glm5NextImageProcessor) -> np.ndarray:
    """Returns the normalized value of each uint8 level per channel, ``[3, 256]``.

    Computed as :meth:`Glm5NextPatchifier.normalize` does: the rescale in
    float64 stored to float32, then the mean and the standard deviation in
    float32, so a gathered level is the processor's value bit for bit.
    """
    table = np.empty((3, 256), dtype=np.float32)
    for channel in range(3):
        row = np.arange(256, dtype=np.float32)
        np.multiply(row, processor.rescale_factor, out=row, dtype=np.float64)
        np.subtract(row, processor.image_mean[channel], out=row)
        np.divide(row, processor.image_std[channel], out=row)
        table[channel] = row
    return table


class Glm5NextImagePreprocess(Module[..., OrderedDict[str, object]]):
    """Turns one RGBA8 image into the vision tower's inputs.

    Takes ``image_preprocess_input_types()``: the decoded image and the
    canvas ``(height, width)`` the engine laid out, both multiples of
    ``patch_size * merge_size``. Returns the tower's input tree for the
    image, on the CPU.
    """

    def __init__(self, processor: Glm5NextImageProcessor) -> None:
        self.patch_size = processor.patch_size
        self.temporal_patch_size = processor.temporal_patch_size
        self.merge_size = processor.merge_size
        self.min_image_tokens = processor.min_image_tokens
        self._table = _normalization_table(processor).reshape(-1)

    def _content_size(self, source: Tensor, canvas: Tensor) -> Tensor:
        """Returns the ``int64 [2]`` size the content is resized to.

        As :meth:`Glm5NextPatchifier.content_size`: the content keeps its
        aspect ratio inside the canvas and is never upscaled unless the
        source is under the token floor, which is the only way it reaches
        the floor. Every step is the processor's own float64 arithmetic.
        """
        factor = self.patch_size * self.merge_size
        floor_pixels = float(
            self.temporal_patch_size * factor**2 * self.min_image_tokens
        )
        scale = F.min(canvas / source)
        pixels = (source[0] * source[1]) * float(self.temporal_patch_size)
        scale = F.where(
            F.greater_equal(pixels, floor_pixels), F.min(scale, 1.0), scale
        )
        return F.max(F.min(canvas, F.floor(source * scale)), 1.0).cast(
            DType.int64
        )

    def forward(self, image: Tensor, size: Tensor) -> OrderedDict[str, object]:
        """Returns the tower's input tree for one image."""
        patch = self.patch_size
        merge = self.merge_size
        temporal = self.temporal_patch_size
        cpu = CPU()

        # The processor converts to RGB, which drops alpha rather than
        # compositing it.
        rgb = image[:, :, :, :3]
        _, height, width, _ = rgb.shape
        source = F.shape_to_tensor([height, width]).cast(DType.float64)
        canvas = size.reshape([2]).cast(DType.float64)
        content = self._content_size(source, canvas)
        # Pillow's uint8 bicubic resize, bit for bit; `[1, 3, h, w]`.
        resized = _resize_bicubic_to(rgb, content, 1)[0]
        _, content_height, content_width = resized.shape

        # The rest of the canvas is zero in pixel space, so a padded patch
        # normalizes to `-mean / std`, as the processor's does.
        zero = F.constant(0, DType.uint8, cpu)
        canvas_width = size[1:2]
        pad_right = _broadcast_to(
            zero,
            F.concat(
                [
                    F.constant([3], DType.int64, cpu),
                    F.shape_to_tensor([content_height]),
                    canvas_width - F.shape_to_tensor([content_width]),
                ],
                axis=0,
            ),
            out_dims=[3, content_height, "pad_width"],
        )
        row = F.rebind(
            F.concat([resized, pad_right], axis=2),
            [3, content_height, "canvas_width"],
            message="the padded row spans the canvas width",
        )
        _, _, canvas_width_dim = row.shape
        pad_bottom = _broadcast_to(
            zero,
            F.concat(
                [
                    F.constant([3], DType.int64, cpu),
                    size[0:1] - F.shape_to_tensor([content_height]),
                    canvas_width,
                ],
                axis=0,
            ),
            out_dims=[3, "pad_height", canvas_width_dim],
        )
        padded = F.rebind(
            F.concat([row, pad_bottom], axis=1),
            [3, "canvas_height", canvas_width_dim],
            message="the padded canvas is the laid-out size",
        )
        # The canvas sides are multiples of the merged patch, which the
        # patchify below needs to know and the engine's layout guarantees.
        factor = patch * merge
        _, canvas_height_dim, _ = padded.shape
        padded = F.rebind(
            padded,
            [
                3,
                canvas_height_dim // factor * factor,
                canvas_width_dim // factor * factor,
            ],
            message=f"the canvas sides must be multiples of {factor}",
        )
        _, canvas_height_dim, canvas_width_dim = padded.shape

        # Rescale and normalize by a table of the 256 levels per channel,
        # computed as the processor computes them.
        table = F.constant(self._table, DType.float32, cpu)
        offsets = F.constant(
            np.array([0, 256, 512], dtype=np.int32).reshape(3, 1, 1),
            DType.int32,
            cpu,
        )
        normalized = F.gather(table, padded.cast(DType.int32) + offsets, axis=0)

        # A still image fills every frame of its temporal patch.
        channels = 3
        normalized = F.broadcast_to(
            normalized.reshape(
                [channels, 1, canvas_height_dim, canvas_width_dim]
            ),
            [channels, temporal, canvas_height_dim, canvas_width_dim],
        )
        blocks_h = canvas_height_dim // (patch * merge)
        blocks_w = canvas_width_dim // (patch * merge)
        grid_h, grid_w = blocks_h * merge, blocks_w * merge
        seq = grid_h * grid_w
        # The processor's row order: merge blocks, then the patches within a
        # block, each row `(channel, temporal, y, x)`.
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
            # Each patch's row or column, laid out like the rows above
            # (`vision_position_ids`).
            positions = F.range(0, side, 1, dtype=DType.int32, device=cpu)
            return F.broadcast_to(positions.reshape(shape), merged)

        rot_pos_ids = F.stack(
            [
                ids(grid_h, [blocks_h, 1, merge, 1]),
                ids(grid_w, [1, blocks_w, 1, merge]),
            ],
            axis=-1,
        ).reshape([seq, 2])

        # One attention segment: the image is one frame (`vision_cu_seqlens`).
        return OrderedDict(
            patches=VisionRows(
                pixel_values=(pixel_values,),
                rot_pos_ids=(rot_pos_ids,),
            ),
            segments=VisionSegments(
                input_row_offsets=(
                    F.shape_to_tensor([0, seq]).cast(DType.uint32),
                )
            ),
            segment_bound=VisionHostMax(
                max_seqlen=(F.shape_to_tensor([seq]).cast(DType.uint32),)
            ),
        )


def image_preprocess_spec(processor: Glm5NextImageProcessor) -> GraphSpec:
    """Returns the graph that preprocesses one image for the tower."""
    return GraphSpec(
        Glm5NextImagePreprocess(processor),
        "glm5_next_vision_preprocess",
        image_preprocess_input_types(),
        weights={},
    )


def glm5_next_vision_metadata(
    huggingface_config: AutoConfig, processor: Glm5NextImageProcessor
) -> VisionMetadata | None:
    """Returns the media layout a GLM-5.3-Flash checkpoint is served with.

    Returns ``None`` for a checkpoint without a vision tower. The chat
    template renders ``<|begin_of_image|><|image|><|end_of_image|>`` and the
    tokenizer widens the one placeholder in place, so the spliced span is
    the bare run. Video is not served from an artifact yet: the clip graph
    is not exported, so the layout declares images only.
    """
    vision = getattr(huggingface_config, "vision_config", None)
    if not hasattr(vision, "patch_size"):
        return None
    return VisionMetadata.from_layouts(
        image=ModalityLayout(
            patch_size=processor.patch_size,
            merge_size=processor.merge_size,
            strategy=Glm5NextSmartResize(
                temporal_patch_size=processor.temporal_patch_size,
                min_tokens=processor.min_image_tokens,
                max_tokens=processor.max_image_tokens,
            ),
            span=Span(int(huggingface_config.image_token_id)),
            template_markers=TemplateMarkers(
                int(huggingface_config.image_start_token_id),
                int(huggingface_config.image_end_token_id),
            ),
        ),
        video=None,
        # The tokenizer sets no per-image byte cap; the pixel guard applies.
        max_image_bytes=0,
        max_video_bytes=0,
        prompt=PromptText(),
    )


def processor_for(pipeline_config: Any) -> Glm5NextImageProcessor:
    """Returns the image processor the pipeline's checkpoint ships."""
    path = pipeline_config.model
    return Glm5NextImageProcessor.from_processor_config(
        _load_processor_config(path.model_path, path.huggingface_model_revision)
    )
