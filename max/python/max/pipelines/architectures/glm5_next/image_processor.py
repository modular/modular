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
"""GLM-5.3-Flash image processor using pure numpy/PIL (no torch).

Reimplements HuggingFace's ``Glm5NextImageProcessorPil`` operation for
operation, including its float dtypes and their order, so the patch tensor is
bit-identical to the reference. The torchvision-backed
``Glm5NextImageProcessor`` resamples with antialiased bicubic rather than
Pillow's, so it agrees with this to about ``2/255`` on the pixels a resize
actually moves and exactly everywhere else.

Two things here are new relative to GLM-4.6V's processor, and both change the
pixels:

* The budget is counted in **decoder tokens**, not pixels, and ``smart_resize``
  binary-searches the largest aspect-preserving content height whose aligned
  canvas fits it, rather than rescaling by a closed-form ratio.
* The image is resized to that content size and then **zero-padded** up to the
  aligned canvas, rather than resized onto the canvas. The padding is applied
  in pixel space, before rescale and normalize, so padded patches carry
  ``-mean/std`` and not zero. They are real patches: they run through the tower
  and consume decoder tokens.

The parameters come from the checkpoint's ``processor_config.json`` -- note the
name, and note that it nests them under an ``image_processor`` key. GLM-4.6V
ships a flat ``preprocessor_config.json``; GLM-5.3-Flash ships no such file.

The resize / pad / normalize / patchify pipeline below is shared with
:mod:`.video_processor`, which differs only in its token budget, its frame
count and the sampler that picks the frames.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image
from typing_extensions import Self

__all__ = [
    "GLM5_IMAGE_MEAN",
    "GLM5_IMAGE_STD",
    "Glm5NextImageProcessor",
    "Glm5NextPatchifier",
    "smart_resize",
]

GLM5_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
"""OpenAI CLIP channel means, as shipped in ``processor_config.json``."""

GLM5_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)
"""OpenAI CLIP channel standard deviations."""


def _align(value: int, factor: int) -> int:
    """Rounds ``value`` up to the next multiple of ``factor``."""
    return math.ceil(value / factor) * factor


def smart_resize(
    num_frames: int,
    height: int,
    width: int,
    *,
    temporal_factor: int = 2,
    factor: int = 28,
    min_image_tokens: int = 16,
    max_image_tokens: int = 8000,
) -> tuple[int, int]:
    """Picks the aligned canvas ``(height, width)`` an entry is padded up to.

    Both returned edges are multiples of ``factor``. The budget is spatiotemporal
    and expressed in decoder tokens: one token covers
    ``temporal_factor * factor**2`` pixels, so a still image with
    ``max_image_tokens = 8000`` gets 12,544,000 pixels of canvas.

    The canvas is the *aligned* size, not the content size. The caller resizes
    content to fit inside it and pads the remainder.

    Args:
        num_frames: Frames the patchifier will see. ``temporal_patch_size`` for
            a still image, which is replicated to fill the temporal patch.
        height: Source height in pixels.
        width: Source width in pixels.
        temporal_factor: Temporal patch size the frame count is rounded to.
        factor: Spatial quantum, ``patch_size * merge_size``.
        min_image_tokens: Lower bound on the entry's decoder-token count.
        max_image_tokens: Upper bound on the entry's decoder-token count.

    Returns:
        The aligned ``(height, width)`` canvas.

    Raises:
        ValueError: If the budget cannot hold even one aligned patch.
    """
    pixels_per_token = temporal_factor * factor**2
    min_pixels = min_image_tokens * pixels_per_token
    max_pixels = max_image_tokens * pixels_per_token

    def fit_within_budget(aligned_frames: int) -> tuple[int, int]:
        minimum_pixels = aligned_frames * factor**2
        if max_pixels < minimum_pixels:
            raise ValueError(
                f"max_pixels={max_pixels} is too small. At least "
                f"{minimum_pixels} pixels are required for one aligned patch."
            )
        low, high = 1, height
        best_height, best_width = factor, factor
        while low <= high:
            content_height = (low + high) // 2
            content_width = max(1, math.floor(width * content_height / height))
            candidate_height = _align(content_height, factor)
            candidate_width = _align(content_width, factor)
            if (
                aligned_frames * candidate_height * candidate_width
                <= max_pixels
            ):
                best_height, best_width = candidate_height, candidate_width
                low = content_height + 1
            else:
                high = content_height - 1
        return best_height, best_width

    aligned_frames = max(
        temporal_factor, round(num_frames / temporal_factor) * temporal_factor
    )
    aligned_height = _align(height, factor)
    aligned_width = _align(width, factor)
    budget = aligned_frames * aligned_height * aligned_width

    if budget < min_pixels:
        scale = math.sqrt(min_pixels / (num_frames * height * width))
        aligned_height = _align(max(1, math.ceil(height * scale)), factor)
        aligned_width = _align(max(1, math.ceil(width * scale)), factor)
        budget = aligned_frames * aligned_height * aligned_width

    if budget > max_pixels:
        aligned_height, aligned_width = fit_within_budget(aligned_frames)

    return aligned_height, aligned_width


class Glm5NextPatchifier:
    """The pixel pipeline images and video share.

    Canvas selection, the resize-and-pad step, normalization and the patch
    reordering are identical for a still image and for a clip; the two differ
    only in their token budget, their frame count and -- for video -- the
    sampler that chooses the frames. Subclasses add the entry point.
    """

    def __init__(
        self,
        *,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        patch_expand_factor: int = 1,
        min_image_tokens: int = 16,
        max_image_tokens: int = 8000,
        image_mean: Sequence[float] = GLM5_IMAGE_MEAN,
        image_std: Sequence[float] = GLM5_IMAGE_STD,
        rescale_factor: float = 1.0 / 255.0,
        **unused_kwargs: Any,
    ) -> None:
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.patch_expand_factor = patch_expand_factor
        self.min_image_tokens = min_image_tokens
        self.max_image_tokens = max_image_tokens
        self.rescale_factor = rescale_factor
        self.image_mean = np.array(image_mean, dtype=np.float32, ndmin=1)
        self.image_std = np.array(image_std, dtype=np.float32, ndmin=1)

    #: Key of the section :meth:`from_processor_config` reads out of
    #: ``processor_config.json``. The video processor overrides it.
    CONFIG_SECTION = "image_processor"

    @classmethod
    def from_processor_config(
        cls, config: Mapping[str, Any], **overrides: Any
    ) -> Self:
        """Builds a processor from a parsed ``processor_config.json``.

        Accepts either the whole file or the relevant sub-object.
        """
        section = config.get(cls.CONFIG_SECTION, config)
        defaults = cls()
        kwargs: dict[str, Any] = {
            "patch_size": section.get("patch_size", defaults.patch_size),
            "temporal_patch_size": section.get(
                "temporal_patch_size", defaults.temporal_patch_size
            ),
            "merge_size": section.get("merge_size", defaults.merge_size),
            "patch_expand_factor": section.get(
                "patch_expand_factor", defaults.patch_expand_factor
            ),
            "min_image_tokens": section.get(
                "min_image_tokens", defaults.min_image_tokens
            ),
            "max_image_tokens": section.get(
                "max_image_tokens", defaults.max_image_tokens
            ),
            "image_mean": section.get("image_mean", GLM5_IMAGE_MEAN),
            "image_std": section.get("image_std", GLM5_IMAGE_STD),
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    @property
    def spatial_factor(self) -> int:
        """Quantum both canvas edges are a multiple of.

        ``patch_expand_factor`` is 1 in the shipped config and the reference's
        PIL backend leaves it out of this product entirely; it is kept as a
        field so a checkpoint that sets it is not silently mis-sized.
        """
        return self.patch_size * self.merge_size

    def canvas_size(
        self, height: int, width: int, num_frames: int | None = None
    ) -> tuple[int, int]:
        """Returns the aligned ``(height, width)`` an entry is padded up to."""
        if num_frames is None:
            num_frames = self.temporal_patch_size
        return smart_resize(
            num_frames,
            height,
            width,
            temporal_factor=self.temporal_patch_size,
            factor=self.spatial_factor,
            min_image_tokens=self.min_image_tokens,
            max_image_tokens=self.max_image_tokens,
        )

    def num_image_tokens(self, grid_thw: tuple[int, int, int]) -> int:
        """Placeholder tokens one vision entry with this grid expands to."""
        grid_t, grid_h, grid_w = grid_thw
        return grid_t * grid_h * grid_w // self.merge_size**2

    def content_size(
        self,
        height: int,
        width: int,
        canvas_height: int,
        canvas_width: int,
        num_frames: int | None = None,
    ) -> tuple[int, int]:
        """Returns the size the content is resized to inside the canvas.

        An image already inside the token budget is never upscaled -- the
        ``min(1.0, scale)`` clamp. An image *below* the minimum-token floor
        skips that clamp and is upscaled, which is the only way it reaches the
        floor.
        """
        if num_frames is None:
            num_frames = self.temporal_patch_size
        scale = min(canvas_height / height, canvas_width / width)
        pixels_per_token = self.temporal_patch_size * self.spatial_factor**2
        floor_pixels = pixels_per_token * self.min_image_tokens
        if num_frames * height * width >= floor_pixels:
            scale = min(1.0, scale)
        return (
            max(1, min(canvas_height, math.floor(height * scale))),
            max(1, min(canvas_width, math.floor(width * scale))),
        )

    def resize_and_pad(
        self,
        image: Image.Image,
        canvas_height: int,
        canvas_width: int,
        content_height: int,
        content_width: int,
        out: npt.NDArray[np.uint8] | None = None,
    ) -> npt.NDArray[np.uint8]:
        """Resizes one frame's content and zero-pads it up to the canvas.

        Padding runs on raw pixels, so a padded patch normalizes to
        ``-mean/std`` rather than zero. Padding after normalization would be a
        different tensor and a silently worse tower input.

        ``out``, a canvas whose padding is already zero, is written in place:
        a clip's frames share one content size, so a worker reuses one canvas
        rather than zeroing a fresh one per frame.
        """
        # HF converts with PIL's own RGB conversion, which drops alpha rather
        # than compositing it onto white the way qwen_vl_utils does.
        if image.mode != "RGB":
            image = image.convert("RGB")
        if (content_height, content_width) != image.size[::-1]:
            image = image.resize(
                (content_width, content_height), Image.Resampling.BICUBIC
            )
        if out is None:
            out = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        out[:content_height, :content_width] = np.asarray(image, dtype=np.uint8)
        return out

    def normalize(self, patches: npt.NDArray[np.float32]) -> None:
        """Rescales in float64 and normalizes in float32, in that order.

        Matching the reference's dtypes exactly: folding the two into one
        float32 multiply-add (as the qwen2_5vl processor does) perturbs the last
        mantissa bits. Works in place, since a clip's patches run to
        gigabytes.

        Args:
            patches: ``(num_patches, 3, k)`` pixel values, channel second.
        """
        np.multiply(patches, self.rescale_factor, out=patches, dtype=np.float64)
        np.subtract(patches, self.image_mean[:, np.newaxis], out=patches)
        np.divide(patches, self.image_std[:, np.newaxis], out=patches)

    def patchify(
        self,
        frames: npt.NDArray[np.uint8],
        out: npt.NDArray[np.float32] | None = None,
    ) -> tuple[npt.NDArray[np.float32], tuple[int, int, int]]:
        """Reorders padded frames into normalized vision-tower patch rows.

        The bytes are reordered first and normalized in place after, so the
        only allocation the size of the result is the result itself.

        Args:
            frames: ``(num_frames, height, width, 3)`` padded pixels. A frame
                count that is not a multiple of ``temporal_patch_size`` is
                padded by repeating the last frame, as the reference does.
            out: Rows to write the patches into, for a caller assembling one
                tensor from several calls. Allocated when ``None``.

        Returns:
            The ``(num_patches, 3 * temporal * patch**2)`` float32 patch tensor
            and its ``(t, h, w)`` grid.
        """
        patch_size = self.patch_size
        merge_size = self.merge_size
        temporal_patch_size = self.temporal_patch_size

        remainder = frames.shape[0] % temporal_patch_size
        if remainder != 0:
            repeats = np.repeat(
                frames[-1:], temporal_patch_size - remainder, axis=0
            )
            frames = np.concatenate([frames, repeats], axis=0)

        num_frames, height, width, channels = frames.shape
        grid_t = num_frames // temporal_patch_size
        grid_h = height // patch_size
        grid_w = width // patch_size
        num_patches = grid_t * grid_h * grid_w

        patches = frames.reshape(
            grid_t,
            temporal_patch_size,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
            channels,
        )
        # The reference's row order, (t, h block, w block, h merge, w merge)
        # by (channel, temporal, y, x), read from channel-last pixels.
        patches = np.transpose(patches, (0, 2, 5, 3, 6, 8, 1, 4, 7))
        if out is None:
            out = np.empty(
                (num_patches, channels * temporal_patch_size * patch_size**2),
                dtype=np.float32,
            )
        # A reshape of a non-contiguous `out` would copy, and the writes below
        # would land in the copy.
        assert out.flags.c_contiguous
        np.copyto(out.reshape(patches.shape), patches)
        self.normalize(out.reshape(num_patches, channels, -1))
        return out, (grid_t, grid_h, grid_w)


class Glm5NextImageProcessor(Glm5NextPatchifier):
    """Resizes, pads, normalizes and patchifies still images.

    One image is one entry with ``grid_t = 1``: the temporal patch is filled
    with two copies of the same frame.
    """

    def grid_thw(self, image: Image.Image) -> tuple[int, int, int]:
        """Returns the ``(t, h, w)`` patch grid without touching the pixels."""
        width, height = image.size
        canvas_height, canvas_width = self.canvas_size(height, width)
        return (
            1,
            canvas_height // self.patch_size,
            canvas_width // self.patch_size,
        )

    def preprocess(
        self, image: Image.Image
    ) -> tuple[npt.NDArray[np.float32], tuple[int, int, int]]:
        """Patchifies one image.

        Args:
            image: The decoded image, in any PIL mode.

        Returns:
            The ``(num_patches, channels * temporal_patch_size * patch_size**2)``
            float32 patch tensor and its ``(t, h, w)`` grid.
        """
        width, height = image.size
        canvas_height, canvas_width = self.canvas_size(height, width)
        content_height, content_width = self.content_size(
            height, width, canvas_height, canvas_width
        )
        padded = self.resize_and_pad(
            image, canvas_height, canvas_width, content_height, content_width
        )
        # A still image fills the temporal patch with copies of itself, which
        # `patchify`'s repeat-the-last-frame padding does.
        return self.patchify(padded[np.newaxis])

    def __call__(
        self, images: Sequence[Image.Image]
    ) -> tuple[list[npt.NDArray[np.float32]], npt.NDArray[np.int32]]:
        """Patchifies each image, returning per-image patches and the grids."""
        processed = [self.preprocess(image) for image in images]
        return (
            [patches for patches, _ in processed],
            np.array([grid for _, grid in processed], dtype=np.int32).reshape(
                len(processed), 3
            ),
        )
