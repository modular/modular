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

"""Exports Kimi K3 image layout metadata and its CPU preprocessing graph."""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from max import functional as F
from max._artifact.vision import (
    KimiK3NavitResize,
    ModalityLayout,
    PromptText,
    Span,
    VisionMetadata,
)
from max.driver import CPU
from max.dtype import DType
from max.graph import ops
from max.nn import Module
from max.nn.legacy.graph_groups import VisionHostMax, VisionRows, VisionSegments
from max.nn.legacy.image_preprocess import image_preprocess_input_types
from max.pipelines.lib.graph_specs import GraphSpec
from max.tensor import Tensor
from PIL import Image

from .vision_processor import (
    MediaProcConfig,
    load_media_config,
    normalization_table,
)

if TYPE_CHECKING:
    from max.pipelines.lib.config import PipelineConfig

    from .model_config import KimiK3Config

_resize_bicubic = F.functional(ops.resize_bicubic)


def _alpha_tables(
    config: MediaProcConfig,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Pins Pillow's uint8 alpha conversions and NumPy compositing rounding."""
    alpha, value = np.indices((256, 256), dtype=np.uint16)
    rgba = np.stack([value, value, value, alpha], axis=-1).astype(np.uint8)
    premultiply = np.asarray(Image.fromarray(rgba).convert("RGBa"))[
        :, :, 0
    ].copy()
    premultiplied = Image.frombytes("RGBa", (256, 256), rgba.tobytes())
    unpremultiply = np.asarray(premultiplied.convert("RGBA"))[:, :, 0].copy()
    backgrounds = (
        [config.chessboard_white_value, config.chessboard_gray_value]
        if config.transparent_pattern == "chessboard"
        else [
            {"white": 255, "gray": 128, "black": 0, None: 0}[
                config.transparent_pattern
            ]
        ]
    )
    fraction = alpha.astype(np.float32) / 255.0
    composite = np.stack(
        [
            (
                fraction * value.astype(np.uint8) + (1 - fraction) * background
            ).astype(np.uint8)
            for background in backgrounds
        ]
    )
    return (
        premultiply.reshape(-1),
        unpremultiply.reshape(-1),
        composite.reshape(-1),
    )


class KimiK3ImagePreprocess(Module[..., OrderedDict[str, object]]):
    """Produces one image's named tower inputs on the CPU."""

    def __init__(self, config: MediaProcConfig):
        self.config = config
        self.premultiply, self.unpremultiply, self.composite = _alpha_tables(
            config
        )
        self.normalized_levels = normalization_table(config)

    def _composite(self, rgba: Tensor) -> Tensor:
        cfg = self.config
        cpu = CPU()
        height, width = rgba.shape[:2]
        rgb = rgba[:, :, :3]
        if cfg.transparent_pattern is not None:
            indices = (
                rgb.cast(DType.int64) + rgba[:, :, 3:].cast(DType.int64) * 256
            )
            if cfg.transparent_pattern == "chessboard":
                y = F.range(
                    0, height, 1, dtype=DType.int64, device=cpu
                ).reshape([height, 1])
                x = F.range(0, width, 1, dtype=DType.int64, device=cpu).reshape(
                    [1, width]
                )
                gray_slot = (
                    y // cfg.chessboard_square_size
                    + x // cfg.chessboard_square_size
                ) % 2
                if not cfg.chessboard_square_on_top_left:
                    gray_slot = 1 - gray_slot
                indices = (
                    indices + gray_slot.reshape([height, width, 1]) * 65536
                )
            rgb = F.gather(
                F.constant(self.composite, DType.uint8, cpu), indices, axis=0
            )
        return rgb

    def forward(self, image: Tensor, size: Tensor) -> OrderedDict[str, object]:
        """Resizes RGBA, fills transparency, pads, and packs row-major patches."""
        cfg = self.config
        cpu = CPU()
        source = F.shape_to_tensor(image.shape[1:3])
        source_f64 = source.cast(DType.float64)
        side = cfg.patch_size * cfg.patch_limit_on_one_side
        # Matches navit_resize_image; the caller's `size` is the padded
        # canvas of the same rule.
        floor_patches = F.max(source // cfg.patch_size, 1).cast(DType.float64)
        budget_scale = F.sqrt(
            float(cfg.in_patch_limit) / (floor_patches[0] * floor_patches[1])
        )
        scale = F.min(
            F.min(F.min(budget_scale, 1.0), side / source_f64[0]),
            side / source_f64[1],
        )
        content_size = F.min(
            F.max((source_f64 * scale).cast(DType.int64), 1), side
        )
        if cfg.transparent_bg_fill_stage == "before_resize":
            rgb = self._composite(image[0])
            rgb = _resize_bicubic(
                rgb.unsqueeze(0).permute([0, 3, 1, 2]),
                F.concat([F.constant([1, 3], DType.int64, cpu), content_size]),
                antialias=True,
                out_dims=[1, 3, "k3_content_height", "k3_content_width"],
            ).permute([0, 2, 3, 1])[0]
        else:
            pixels = image
            alpha = pixels[:, :, :, 3:].cast(DType.int32)
            premultiplied = F.gather(
                F.constant(self.premultiply, DType.uint8, cpu),
                pixels[:, :, :, :3].cast(DType.int32) + alpha * 256,
                axis=0,
            )
            premultiplied = F.concat(
                [premultiplied, pixels[:, :, :, 3:]], axis=-1
            )
            resized = _resize_bicubic(
                premultiplied.permute([0, 3, 1, 2]),
                F.concat([F.constant([1, 4], DType.int64, cpu), content_size]),
                antialias=True,
                out_dims=[1, 4, "k3_content_height", "k3_content_width"],
            ).permute([0, 2, 3, 1])[0]
            height, width = resized.shape[:2]
            alpha = resized[:, :, 3:].cast(DType.int32)
            rgb = F.gather(
                F.constant(self.unpremultiply, DType.uint8, cpu),
                resized[:, :, :3].cast(DType.int32) + alpha * 256,
                axis=0,
            )
            rgba = F.concat([rgb, resized[:, :, 3:]], axis=-1)
            y = F.range(0, height, 1, dtype=DType.int64, device=cpu).reshape(
                [height, 1]
            )
            x = F.range(0, width, 1, dtype=DType.int64, device=cpu).reshape(
                [1, width]
            )
            # Pillow skips its RGBa conversion when no resize is needed.
            original = F.gather(
                image.reshape([-1, 4]),
                F.min(y, source[0] - 1) * source[1] + F.min(x, source[1] - 1),
                axis=0,
            )
            same_size = F.equal(source[0], content_size[0]) & F.equal(
                source[1], content_size[1]
            )
            rgba = F.where(same_size, original, rgba)
            rgb = self._composite(rgba)
        patch = cfg.patch_size
        rows = F.range(
            0,
            size[0] // patch,
            1,
            out_dim="k3_grid_height",
            dtype=DType.int64,
            device=cpu,
        )
        columns = F.range(
            0,
            size[1] // patch,
            1,
            out_dim="k3_grid_width",
            dtype=DType.int64,
            device=cpu,
        )
        inner = F.range(0, patch, 1, dtype=DType.int64, device=cpu)
        gh, gw = rows.shape[0], columns.shape[0]
        yy = (rows.reshape([gh, 1]) * patch + inner).reshape([gh, 1, patch, 1])
        xx = (columns.reshape([gw, 1]) * patch + inner).reshape(
            [1, gw, 1, patch]
        )
        inside = (yy < content_size[0]) & (xx < content_size[1])
        indices = F.min(yy, content_size[0] - 1) * content_size[1] + F.min(
            xx, content_size[1] - 1
        )
        rgb = F.gather(rgb.reshape([-1, 3]), indices, axis=0)
        rgb = F.where(inside.unsqueeze(-1), rgb, 0)
        channel = F.range(0, 3, 1, dtype=DType.int32, device=cpu)
        levels = F.gather(
            F.constant(self.normalized_levels.reshape(-1), DType.float32, cpu),
            rgb.cast(DType.int32) * 3 + channel,
            axis=0,
        )
        count = gh * gw
        patches = (
            levels.permute([0, 1, 4, 2, 3])
            .reshape([count, 3, patch, patch])
            .cast(DType.bfloat16)
        )
        positions = (
            rows.reshape([gh, 1]) * 512 + columns.reshape([1, gw])
        ).reshape([count])
        return OrderedDict(
            patches=VisionRows(
                pixel_values=(patches,), position_ids=(positions,)
            ),
            grids=VisionRows(
                grid_thws=(
                    F.shape_to_tensor([1, gh, gw])
                    .cast(DType.int64)
                    .reshape([1, 3]),
                )
            ),
            segments=VisionSegments(
                input_row_offsets=(
                    F.shape_to_tensor([0, count]).cast(DType.uint32),
                )
            ),
            segment_bound=VisionHostMax(
                max_seq_len=(F.shape_to_tensor([count]).cast(DType.uint32),)
            ),
        )


def vision_metadata(
    pipeline_config: PipelineConfig, config: KimiK3Config
) -> VisionMetadata:
    """Returns the artifact's still-image layout and bare embedding spans."""
    del config
    model = pipeline_config.model
    processor = load_media_config(
        model.model_path, model.huggingface_model_revision
    )
    return VisionMetadata.from_layouts(
        image=ModalityLayout(
            patch_size=processor.patch_size,
            merge_size=processor.merge_kernel_size,
            strategy=KimiK3NavitResize(
                processor.in_patch_limit, processor.patch_limit_on_one_side
            ),
            span=Span(int(model.huggingface_config.media_placeholder_token_id)),
        ),
        video=None,
        max_image_bytes=0,
        max_video_bytes=0,
        prompt=PromptText(),
    )


def vision_preprocess_spec(
    pipeline_config: PipelineConfig, config: KimiK3Config
) -> GraphSpec:
    """Returns the CPU graph that prepares the registered vision input tree."""
    del config
    model = pipeline_config.model
    processor = load_media_config(
        model.model_path, model.huggingface_model_revision
    )
    return GraphSpec(
        KimiK3ImagePreprocess(processor),
        "kimi_k3_vision_preprocess",
        image_preprocess_input_types(),
        weights={},
    )
