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

"""Processes still images with Kimi K3's resize, transparency, and patch order."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import numpy.typing as npt
from max.pipelines.lib.tokenizer import open_image
from PIL import Image
from transformers.utils.hub import cached_file


class ImageInput(TypedDict):
    """Describes one still-image upload."""

    type: Literal["image"]
    image: Image.Image | bytes


class ResizePlan(TypedDict):
    """Records resized content, canvas padding, and image token count."""

    new_width: int
    new_height: int
    pad_width: int
    pad_height: int
    num_tokens: int


class VisionOutput(TypedDict):
    """Contains normalized patches and their temporal and spatial grids."""

    pixel_values: npt.NDArray[np.float32]
    grid_thws: npt.NDArray[np.int64]


def _integer(config: Mapping[str, object], name: str, default: int) -> int:
    value = config.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _triple(
    config: Mapping[str, object], name: str
) -> tuple[float, float, float]:
    values = config.get(name, [0.5, 0.5, 0.5])
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"{name} must contain three numbers")
    if not all(isinstance(value, (int, float)) for value in values):
        raise ValueError(f"{name} must contain three numbers")
    return (float(values[0]), float(values[1]), float(values[2]))


@dataclass(frozen=True)
class MediaProcConfig:
    """Defines the checkpoint's still-image preprocessing.

    Each default is the reference processor's value for an absent key, so
    ``MediaProcConfig()`` equals ``MediaProcConfig.from_dict({})``.
    """

    in_patch_limit: int = 65536
    patch_size: int = 14
    merge_kernel_size: int = 2
    patch_limit_on_one_side: int = 512
    image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    image_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    transparent_pattern: str | None = None
    chessboard_square_size: int = 16
    chessboard_square_on_top_left: bool = True
    chessboard_white_value: int = 255
    chessboard_gray_value: int = 200
    transparent_bg_fill_stage: str = "before_resize"

    @classmethod
    def from_dict(cls, config: Mapping[str, object]) -> MediaProcConfig:
        """Reads the still-image fields of a checkpoint processor config."""
        if config.get("fixed_output_tokens") is not None:
            raise ValueError("Kimi K3 requires variable image token counts")
        bg = config.get("transparent_bg_config")
        if bg is None:
            bg_values: Mapping[str, object] = {}
            pattern = None
        else:
            if not isinstance(bg, dict):
                raise ValueError("transparent_bg_config must be a mapping")
            bg_values = bg
            pattern = bg.get("pattern", "black")
            if pattern not in ("white", "black", "gray", "chessboard"):
                raise ValueError(
                    f"Unsupported transparent background: {pattern}"
                )
        stage = config.get("transparent_bg_fill_stage", "before_resize")
        if stage not in ("before_resize", "after_resize"):
            raise ValueError(f"Unsupported transparency stage: {stage}")
        top_left = bg_values.get("chessboard_square_on_top_left", True)
        if not isinstance(top_left, bool):
            raise ValueError("chessboard_square_on_top_left must be a boolean")
        white = bg_values.get("chessboard_white_value", 255)
        gray = bg_values.get("chessboard_gray_value", 200)
        if not isinstance(white, int) or not 0 <= white <= 255:
            raise ValueError("chessboard_white_value must be uint8")
        if not isinstance(gray, int) or not 0 <= gray <= 255:
            raise ValueError("chessboard_gray_value must be uint8")
        std = _triple(config, "image_std")
        if min(std) <= 0:
            raise ValueError("image_std must be positive")
        return cls(
            in_patch_limit=_integer(config, "in_patch_limit", 65536),
            patch_size=_integer(config, "patch_size", 14),
            merge_kernel_size=_integer(config, "merge_kernel_size", 2),
            patch_limit_on_one_side=_integer(
                config, "patch_limit_on_one_side", 512
            ),
            image_mean=_triple(config, "image_mean"),
            image_std=std,
            transparent_pattern=pattern,
            chessboard_square_size=_integer(
                bg_values, "chessboard_square_size", 16
            ),
            chessboard_square_on_top_left=top_left,
            chessboard_white_value=white,
            chessboard_gray_value=gray,
            transparent_bg_fill_stage=stage,
        )


def load_media_config(
    model_path: str, revision: str | None = None
) -> MediaProcConfig:
    """Loads the checkpoint's processor settings without importing remote code."""
    config_path = cached_file(
        model_path, "preprocessor_config.json", revision=revision
    )
    if config_path is None:
        raise ValueError("Kimi K3 checkpoint has no preprocessor_config.json")
    payload = json.loads(Path(config_path).read_text())
    config = payload.get("media_proc_cfg")
    if not isinstance(config, dict):
        raise ValueError("media_proc_cfg must be a mapping")
    return MediaProcConfig.from_dict(config)


def navit_resize_image(
    width: int, height: int, config: MediaProcConfig
) -> ResizePlan:
    """Computes content size and bottom/right padding in reference order."""
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    patch = config.patch_size
    side = config.patch_limit_on_one_side * patch
    scale = min(
        1.0,
        math.sqrt(
            config.in_patch_limit
            / (max(1.0, width // patch) * max(1.0, height // patch))
        ),
        side / width,
        side / height,
    )
    new_width = min(max(1, int(width * scale)), side)
    new_height = min(max(1, int(height * scale)), side)
    factor = patch * config.merge_kernel_size
    pad_width = (-new_width) % factor
    pad_height = (-new_height) % factor
    grid_h = (new_height + pad_height) // patch
    grid_w = (new_width + pad_width) // patch
    if max(grid_h, grid_w) > config.patch_limit_on_one_side:
        raise ValueError(
            "Padded patch grid exceeds the checkpoint's side limit"
        )
    return ResizePlan(
        new_width=new_width,
        new_height=new_height,
        pad_width=pad_width,
        pad_height=pad_height,
        num_tokens=grid_h * grid_w // config.merge_kernel_size**2,
    )


def fill_transparent_background(
    image: Image.Image, config: MediaProcConfig
) -> Image.Image:
    """Composites alpha with the checkpoint's float32 truncation semantics."""
    pattern = config.transparent_pattern
    if pattern is None or (
        "A" not in image.getbands() and "transparency" not in image.info
    ):
        return image.convert("RGB")
    pixels = np.asarray(image.convert("RGBA"))
    height, width = pixels.shape[:2]
    if pattern == "chessboard":
        square = config.chessboard_square_size
        alternating = (
            np.arange(height)[:, None] // square
            + np.arange(width)[None, :] // square
        ) % 2
        gray = alternating == (1 if config.chessboard_square_on_top_left else 0)
        background = np.where(
            gray, config.chessboard_gray_value, config.chessboard_white_value
        ).astype(np.uint8)
        background = np.broadcast_to(background[:, :, None], (height, width, 3))
    else:
        value = {"white": 255, "black": 0, "gray": 128}[pattern]
        background = np.full((height, width, 3), value, dtype=np.uint8)
    alpha = pixels[:, :, 3].astype(np.float32) / 255.0
    alpha = np.stack([alpha] * 3, axis=2)
    result = alpha * pixels[:, :, :3] + (1 - alpha) * background
    return Image.fromarray(result.astype(np.uint8))


def normalization_table(config: MediaProcConfig) -> npt.NDArray[np.float32]:
    """Returns each uint8 level normalized in the reference's cast order."""
    values = np.broadcast_to(np.arange(256)[:, None], (256, 3))
    normalized = (values / 255.0).astype(np.float32)
    normalized -= np.asarray(config.image_mean)
    normalized *= 1.0 / np.asarray(config.image_std)
    return normalized


def compute_position_ids(
    grid_thws: npt.NDArray[np.int64], max_width: int = 512
) -> npt.NDArray[np.int64]:
    """Flattens each image's row-major rotary positions in media order."""
    parts = [
        np.tile(
            (
                np.arange(int(h))[:, None] * max_width
                + np.arange(int(w))[None, :]
            ).reshape(-1),
            int(t),
        )
        for t, h, w in grid_thws
    ]
    return (
        np.concatenate(parts).astype(np.int64)
        if parts
        else np.empty(0, dtype=np.int64)
    )


class KimiK3VisionProcessor:
    """Produces reference float32 patches and per-image patch grids."""

    def __init__(
        self,
        media_proc_cfg: Mapping[str, object] | MediaProcConfig | None = None,
    ):
        self.cfg = (
            media_proc_cfg
            if isinstance(media_proc_cfg, MediaProcConfig)
            else (
                MediaProcConfig.from_dict(media_proc_cfg)
                if media_proc_cfg is not None
                else MediaProcConfig()
            )
        )

    @staticmethod
    def make_image_prompt(width: int, height: int) -> str:
        """Frames one image with its original upload dimensions."""
        return f"<|media_begin|>image {width}x{height}<|media_content|><|media_pad|><|media_end|>"

    def preprocess(
        self, medias: Sequence[ImageInput], return_tensors: str | None = None
    ) -> VisionOutput:
        """Resizes, composites, pads, normalizes, and packs images in order."""
        if return_tensors not in (None, "np"):
            raise ValueError("Kimi K3 MAX preprocessing returns NumPy tensors")
        outputs: list[npt.NDArray[np.float32]] = []
        grids: list[list[int]] = []
        for media in medias:
            if media["type"] != "image":
                raise ValueError("Kimi K3 supports still images only")
            image = open_image(media["image"])
            if self.cfg.transparent_bg_fill_stage == "before_resize":
                image = fill_transparent_background(image, self.cfg)
            plan = navit_resize_image(*image.size, self.cfg)
            image = image.resize(
                (plan["new_width"], plan["new_height"]),
                Image.Resampling.BICUBIC,
            )
            if self.cfg.transparent_bg_fill_stage == "after_resize":
                image = fill_transparent_background(image, self.cfg)
            pixels = np.pad(
                np.asarray(image),
                ((0, plan["pad_height"]), (0, plan["pad_width"]), (0, 0)),
                constant_values=0,
            )
            pixels = (pixels / 255.0).astype(np.float32)
            pixels -= np.asarray(self.cfg.image_mean)
            pixels *= 1.0 / np.asarray(self.cfg.image_std)
            height, width = pixels.shape[:2]
            patch = self.cfg.patch_size
            outputs.append(
                pixels.reshape(height // patch, patch, width // patch, patch, 3)
                .transpose(0, 2, 4, 1, 3)
                .reshape(-1, 3, patch, patch)
            )
            grids.append([1, height // patch, width // patch])
        patch = self.cfg.patch_size
        return VisionOutput(
            pixel_values=np.concatenate(outputs)
            if outputs
            else np.empty((0, 3, patch, patch), dtype=np.float32),
            grid_thws=np.asarray(grids, dtype=np.int64).reshape(-1, 3),
        )
