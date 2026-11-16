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
"""The image processor: whole merge cells, no resize."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import huggingface_hub
import numpy as np
import numpy.typing as npt
from PIL import Image
from transformers import PreTrainedTokenizerBase

from .model_config import Inkling2VisionConfig

PROCESSOR_CONFIG_FILE = "processor_config.json"

# Raw pixel value the reference pads with, before rescaling
# (``image_processing_inkling2.py:90``: ``fill_value=-1.0``).
_PAD_RAW_VALUE = -1.0

# The reference's class defaults, used when the checkpoint's
# ``processor_config.json`` omits them (``image_processing_inkling2.py:35-38``).
_IMAGE_MEAN = (0.5, 0.5, 0.5)
_IMAGE_STD = (0.5, 0.5, 0.5)
_RESCALE_FACTOR = 1.0 / 255.0

# The transparency checkerboard the reference composites against before it
# drops the alpha channel (``image_processing_inkling2.py:64-70``): 8-pixel
# cells alternating 204 and 153 on the 0-255 scale.
_CHECKER_CELL = 8
_CHECKER_LIGHT = 204
_CHECKER_DARK = 153


class InklingImageProcessor:
    """Turns one image into the merged patch block the vision tower consumes."""

    def __init__(
        self,
        processor_config: Mapping[str, Any],
        vision_config: Inkling2VisionConfig,
    ) -> None:
        block = processor_config.get("image_processor")
        if not isinstance(block, Mapping):
            raise ValueError(
                f"{PROCESSOR_CONFIG_FILE} has no image_processor block, so the "
                "image normalization constants are unknown"
            )
        self.patch_size = vision_config.patch_size
        self.merge_size = vision_config.spatial_merge_size
        # One merge token's worth of pixels along each side.
        self.side = self.patch_size * self.merge_size
        self.num_channels = vision_config.num_channels
        self.image_mean = np.asarray(
            block.get("image_mean", _IMAGE_MEAN), dtype=np.float32
        )
        self.image_std = np.asarray(
            block.get("image_std", _IMAGE_STD), dtype=np.float32
        )
        self.rescale_factor = float(
            block.get("rescale_factor", _RESCALE_FACTOR)
        )
        declared = block.get("size")
        if isinstance(declared, Mapping):
            declared_size = (
                int(declared.get("height", self.side)),
                int(declared.get("width", self.side)),
            )
            if declared_size != (self.side, self.side):
                raise ValueError(
                    f"{PROCESSOR_CONFIG_FILE} cuts images into "
                    f"{declared_size[0]}x{declared_size[1]} cells, but the "
                    f"vision config's patch_size {self.patch_size} x "
                    f"spatial_merge_size {self.merge_size} is {self.side} "
                    "pixels per image token; the batch processor takes the "
                    "block side from the vision config, so the two must agree"
                )
        # The padding sentinel after rescale-and-normalize. The reference's
        # raw -1 lands just below the scale's floor (-1.0078 at
        # mean = std = 0.5) while the tower's contract is pixels in [-1, 1]
        # (``modeling_inkling2.py:1426-1427``), so the sentinel enters the
        # graph clamped to that floor. Only pad cells can be affected: a real
        # pixel normalizes into [-1, 1] exactly.
        pad_raw = np.float32(_PAD_RAW_VALUE) * self.rescale_factor
        pad = (pad_raw - self.image_mean) / self.image_std
        self._pad_value = np.clip(pad, -1.0, 1.0).astype(np.float32)

    def __call__(self, image: Image.Image) -> npt.NDArray[np.float32]:
        """Pixels of one image on the reference's ``[-1, 1]`` scale."""
        pixels = _rgb_pixels(image)
        height, width, channels = pixels.shape
        if channels != self.num_channels:
            raise ValueError(
                f"image has {channels} channels but the vision config says "
                f"{self.num_channels}"
            )
        rows = -(-height // self.side)
        columns = -(-width // self.side)
        padded = np.empty(
            (rows * self.side, columns * self.side, channels),
            dtype=np.float32,
        )
        padded[...] = self._pad_value
        padded[:height, :width] = (
            pixels.astype(np.float32) * self.rescale_factor - self.image_mean
        ) / self.image_std
        return padded.reshape(
            rows, self.side, columns, self.side, channels
        ).transpose(0, 2, 1, 3, 4)


def _rgb_pixels(image: Image.Image) -> npt.NDArray[np.uint8]:
    if image.mode != "RGB" and getattr(image, "has_transparency_data", False):
        rgba = np.asarray(image.convert("RGBA"), dtype=np.int32)
        alpha = rgba[..., 3:4]
        row_cell = np.arange(rgba.shape[0], dtype=np.int32)[:, None]
        column_cell = np.arange(rgba.shape[1], dtype=np.int32)[None, :]
        background = np.where(
            (row_cell // _CHECKER_CELL + column_cell // _CHECKER_CELL) % 2 == 0,
            _CHECKER_LIGHT,
            _CHECKER_DARK,
        )[..., None]
        composited = (
            rgba[..., :3] * alpha + background * (255 - alpha) + 127
        ) // 255
        return composited.astype(np.uint8)
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


class InklingProcessor:
    """Stands in for the ``AutoProcessor`` the checkpoint names but does not ship."""

    def __init__(
        self,
        delegate: PreTrainedTokenizerBase,
        processor_config: Mapping[str, Any],
        vision_config: Inkling2VisionConfig,
    ) -> None:
        self.delegate = delegate
        self.image_processor = InklingImageProcessor(
            processor_config, vision_config
        )
        image_token = processor_config["image_token"]
        vocab = delegate.get_vocab()
        if image_token not in vocab:
            raise ValueError(
                f"{PROCESSOR_CONFIG_FILE} names {image_token!r} as the image "
                "token, but the tokenizer vocabulary has no such entry"
            )
        self.image_token_id = int(vocab[image_token])

    def apply_chat_template(self, messages: Any, **options: Any) -> str:
        """Delegate the chat template to the checkpoint's processor."""
        templated = self.delegate.apply_chat_template(messages, **options)
        assert isinstance(templated, str)
        return templated

    def __call__(
        self,
        *,
        text: str | Sequence[int],
        images: Sequence[Image.Image] | None = None,
        add_special_tokens: bool = True,
        **unused_kwargs: Any,
    ) -> dict[str, Any]:
        """Build the prompt's inputs, keeping the image placement."""
        del unused_kwargs
        token_ids = (
            self.delegate.encode(text, add_special_tokens=add_special_tokens)
            if isinstance(text, str)
            else list(text)
        )
        blocks = [self.image_processor(image) for image in images or []]
        # One image token per merge cell, so the placeholder run is the
        # block's two leading axes multiplied out.
        grids = [
            [1, int(block.shape[0]), int(block.shape[1])] for block in blocks
        ]
        return {
            # Both keys are batched, as a HuggingFace processor returns them.
            "input_ids": [
                expand_image_placeholders(
                    token_ids,
                    self.image_token_id,
                    [grid[1] * grid[2] for grid in grids],
                )
            ],
            "pixel_values": [blocks],
            # The reference's own ``image_grid_thw`` (``[temporal, rows,
            # columns]`` counted in image tokens,
            # ``image_processing_inkling2.py:94``). MAX's ``ImageMetadata``
            # has no field for it, so the grid reaches the tower through the
            # block's leading axes; it is emitted here for parity with the
            # checkpoint's processor.
            "image_grid_thw": np.asarray(grids, dtype=np.int32).reshape(-1, 3),
        }


def load_processor_config(
    model_path: str, revision: str | None = None
) -> dict[str, Any]:
    """Reads ``processor_config.json`` out of the checkpoint."""
    local = Path(model_path) / PROCESSOR_CONFIG_FILE
    if not local.is_file():
        local = Path(
            huggingface_hub.hf_hub_download(
                repo_id=model_path,
                filename=PROCESSOR_CONFIG_FILE,
                revision=revision,
            )
        )
    return json.loads(local.read_text())


def expand_image_placeholders(
    token_ids: Sequence[int], placeholder_id: int, patch_counts: Sequence[int]
) -> list[int]:
    """Grows each single placeholder token into that image's run of tokens."""
    markers = [
        index
        for index, token_id in enumerate(token_ids)
        if token_id == placeholder_id
    ]
    if len(markers) != len(patch_counts):
        raise ValueError(
            f"prompt carries {len(markers)} image placeholder(s) but "
            f"{len(patch_counts)} image input(s) were provided"
        )
    # Downstream recovers each image's run as a maximal contiguous stretch of
    # placeholder tokens, so back-to-back runs would fuse into one image.
    if any(second - first == 1 for first, second in pairwise(markers)):
        raise ValueError(
            "prompt places two image placeholders back to back; separate "
            "consecutive images with at least one text token"
        )
    expanded = list(token_ids)
    # Back to front, so the untouched markers keep their indices.
    for index, count in reversed(list(zip(markers, patch_counts, strict=True))):
        expanded[index : index + 1] = [placeholder_id] * count
    return expanded
