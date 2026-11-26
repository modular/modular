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
"""Hy4 VL image preprocessing as a CPU graph that a serving engine runs.

For one image, the graph does what ``HYV4VLImageProcessor`` and
``vision_index_inputs`` do. It also returns the decoder's M-RoPE positions
for the image's tokens, as ``mrope_positions`` lays them out. The resize is
Pillow's uint8 bicubic. The reference quantizes its taps more coarsely, so
some pixels of a downscaled image can differ by one level.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
from max import functional as F
from max._artifact.vision import (
    ModalityLayout,
    PromptText,
    SmartResize,
    Span,
    TemplateMarkers,
    VisionMetadata,
)
from max.driver import CPU
from max.dtype import DType
from max.graph import Dim
from max.nn import Module
from max.nn.legacy.graph_groups import (
    ItemPositions,
    VisionHostMax,
    VisionRowIndex,
    VisionRows,
    VisionSegments,
)
from max.nn.legacy.image_preprocess import (
    composite_on_white,
    image_preprocess_input_types,
    resize_bicubic_to,
)
from max.pipelines.lib import PipelineConfig
from max.pipelines.lib.graph_specs import GraphSpec
from max.tensor import Tensor

from .image_processor import HYV4VLImageProcessor
from .model_config import HYV4VLVisionConfig

_composite_on_white = F.functional(composite_on_white)
_resize_bicubic_to = F.functional(resize_bicubic_to)


def _arange(size: Dim) -> Tensor:
    return F.range(0, size, 1, dtype=DType.int64, device=CPU())


def _scalar(dim: Dim) -> Tensor:
    return F.shape_to_tensor([dim]).reshape([])


def _bilinear_axis(
    size: Dim, edge: int, copy: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """``vision_index_inputs``' source taps along one axis, in the graph."""
    side = _scalar(size).cast(DType.float64)
    ratio = F.where(copy, 1.0, 1.0 / ((side + 0.1) / float(edge))).cast(
        DType.float32
    )
    # numpy rounds the product and the difference to float32 one at a time.
    # Each runs in float64, which holds the float32 product exactly, and is
    # rounded by a cast, so the compiler cannot fuse them into one FMA.
    centers = (_arange(size).cast(DType.float64) + 0.5) * ratio.cast(
        DType.float64
    )
    shifted = centers.cast(DType.float32).cast(DType.float64) - 0.5
    src = F.max(shifted, 0.0).cast(DType.float32)
    i0 = src.cast(DType.int64)
    return i0, F.min(i0 + 1, edge - 1), src - i0.cast(DType.float32)


class HYV4VLImagePreprocess(Module[..., OrderedDict[str, object]]):
    """Turns one RGBA8 image into the Hy4 VL tower's inputs and positions.

    Takes ``image_preprocess_input_types()``, the size being the
    ``smart_resize`` target.
    """

    def __init__(
        self, processor: HYV4VLImageProcessor, vision: HYV4VLVisionConfig
    ) -> None:
        self.patch = processor.patch_size
        self.merge = processor.merge_size
        self.edge = vision.max_image_size // vision.patch_size
        # Each channel's 256 levels, normalized in numpy as the processor
        # does, so the graph only gathers and matches it bit for bit.
        scale = np.float32(1.0 / processor.rescale_factor)
        mean = np.asarray(processor.image_mean, np.float32) * scale
        std = np.asarray(processor.image_std, np.float32) * scale
        levels = np.arange(256, dtype=np.float32)
        self.levels = (levels[None, :] - mean[:, None]) / std[:, None]

    def forward(self, image: Tensor, size: Tensor) -> OrderedDict[str, object]:
        """Returns the tower's input tree and positions for one image."""
        patch, merge, edge = self.patch, self.merge, self.edge
        cpu = CPU()

        rgb = _composite_on_white(image)
        pixels = _resize_bicubic_to(rgb, size.reshape([2]), patch * merge)
        _, channels, height, width = pixels.shape
        levels = F.constant(self.levels, DType.float32, cpu).reshape([-1])
        level = pixels.reshape([channels, height, width]).cast(DType.int64)
        offset = F.constant(np.arange(3) * 256, DType.int64, cpu)
        normalized = F.gather(levels, level + offset.reshape([3, 1, 1]), axis=0)
        blocks_h, blocks_w = height // (patch * merge), width // (patch * merge)
        rows, cols = blocks_h * merge, blocks_w * merge
        patches = rows * cols
        pixel_values = (
            normalized.reshape([3, rows, patch, cols, patch])
            .permute([1, 3, 0, 2, 4])
            .reshape([patches, 3 * patch * patch])
        )

        # The position table's four bilinear taps for each patch.
        copy = F.logical_and(
            F.equal(_scalar(rows), edge), F.equal(_scalar(cols), edge)
        )
        y0, y1, ly = _bilinear_axis(rows, edge, copy)
        x0, x1, lx = _bilinear_axis(cols, edge, copy)
        plane: list[Dim | int] = [rows, cols]

        def outer(y: Tensor, x: Tensor) -> tuple[Tensor, Tensor]:
            return (
                F.broadcast_to(y.reshape([rows, 1]), plane),
                F.broadcast_to(x.reshape([1, cols]), plane),
            )

        taps = [outer(y, x) for y in (y0, y1) for x in (x0, x1)]
        pos_idx = F.stack([1 + y * edge + x for y, x in taps], axis=-1)
        weights = [
            outer(wy, wx) for wy in (1.0 - ly, ly) for wx in (1.0 - lx, lx)
        ]
        pos_w = F.stack([y * x for y, x in weights], axis=-1)

        # Each token row holds blocks_w windows, then a newline that reads
        # the row's last window and is replaced in the tower.
        line = blocks_w + 1
        tokens = blocks_h * line
        token_plane: list[Dim | int] = [blocks_h, line]
        row, col = _arange(blocks_h), _arange(line)
        last = _scalar(blocks_w)
        top = F.broadcast_to(
            (row * merge * _scalar(cols)).reshape([blocks_h, 1]), token_plane
        ) + F.broadcast_to(
            (F.min(col, last - 1) * merge).reshape([1, line]), token_plane
        )
        dy, dx = np.divmod(np.arange(merge * merge), merge)
        window = F.constant(dy, DType.int64, cpu) * _scalar(cols) + F.constant(
            dx, DType.int64, cpu
        )
        fold_idx = top.reshape([tokens, 1]) + window.reshape([1, merge * merge])
        newline = F.broadcast_to(
            F.equal(col, last).reshape([1, line]), token_plane
        )

        # The rows x (cols + 1) token grid starts at the next text position,
        # and the text after it starts max(rows, cols + 1) later.
        def spread(values: Tensor, shape: list[Dim | int]) -> Tensor:
            return F.broadcast_to(values.reshape(shape), token_plane).reshape(
                [tokens]
            )

        position_ids = F.stack(
            [
                spread(F.constant(0, DType.int64, cpu), [1, 1]),
                spread(row, [blocks_h, 1]),
                spread(col, [1, line]),
            ],
            axis=0,
        )
        return OrderedDict(
            patches=VisionRows(
                pixel_values=pixel_values,
                pos_idx=pos_idx.reshape([patches, 4]).cast(DType.uint32),
                pos_w=pos_w.reshape([patches, 4]),
            ),
            segments=VisionSegments(
                input_row_offsets=F.shape_to_tensor([0, patches]).cast(
                    DType.uint32
                )
            ),
            segment_bound=VisionHostMax(
                max_seqlen=F.shape_to_tensor([patches]).cast(DType.uint32)
            ),
            fold=VisionRowIndex(fold_idx=fold_idx.cast(DType.uint32)),
            tokens=VisionRows(newline=newline.reshape([tokens])),
            positions=ItemPositions(
                position_ids=position_ids,
                position_advance=F.max(_scalar(blocks_h), _scalar(line)),
            ),
        )


def image_preprocess_spec(
    processor: HYV4VLImageProcessor, vision: HYV4VLVisionConfig
) -> GraphSpec:
    """Returns the graph that preprocesses one image for the tower."""
    return GraphSpec(
        HYV4VLImagePreprocess(processor, vision),
        "hy_v4_vl_vision_preprocess",
        image_preprocess_input_types(),
        weights={},
    )


def vision_metadata(
    pipeline_config: PipelineConfig, processor: HYV4VLImageProcessor
) -> VisionMetadata:
    """Returns the image layout a Hy4 VL checkpoint is served with.

    The chat template writes the image start and end markers around the one
    pad as text, so the spliced run is bare pads, one more per merged row.
    """
    hf_config = pipeline_config.model.huggingface_config
    assert hf_config is not None
    return VisionMetadata.from_layouts(
        image=ModalityLayout(
            patch_size=processor.patch_size,
            merge_size=processor.merge_size,
            strategy=SmartResize(
                min_pixels=processor.min_pixels,
                max_pixels=processor.max_pixels,
                newline_per_row=True,
            ),
            span=Span(hf_config.image_token_id),
            template_markers=TemplateMarkers(
                hf_config.image_start_token_id, hf_config.image_end_token_id
            ),
        ),
        video=None,
        max_image_bytes=0,
        max_video_bytes=0,
        # vLLM's template writes a message's parts with nothing between them.
        prompt=PromptText(),
        # vLLM's recipe serves one image per prompt.
        max_images_per_request=1,
    )
