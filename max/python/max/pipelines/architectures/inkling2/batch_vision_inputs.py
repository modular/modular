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
"""Packs a batch's uncached images for the Inkling 2 vision tower."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from max.driver import CPU, Buffer, Device
from max.pipelines.context import ImageMetadata, TextAndVisionContext

from .vision import Inkling2VisionConfig
from .vision_inputs import vision_inputs

HOST_INPUTS = frozenset({"positions", "max_seqlen"})
"""The tower inputs that stay on the host: the rotary angles and the longest
sequence."""

IMAGE_GRID_THW = "image_grid_thw"
"""``extra_model_args`` key of the ``[len(images), 3]`` image-token grids, one
row per entry of ``ctx.images``."""


def select_images(
    selection: Sequence[tuple[TextAndVisionContext, Sequence[ImageMetadata]]],
) -> tuple[list[npt.NDArray[np.int16]], npt.NDArray[np.int64]]:
    """Returns the pixel blocks and ``[n, 3]`` grids of the selected images.

    Images are matched to their grid by identity with ``ctx.images``, in
    context-major, image-minor order, the order the encoder must emit rows.

    Raises:
        ValueError: If a selected image is not one of its context's own, or
            its grid does not match its placeholder span.
    """
    pixels: list[npt.NDArray[np.int16]] = []
    grids: list[npt.NDArray[np.int64]] = []
    for ctx, miss_images in selection:
        wanted = {id(img) for img in miss_images}
        all_grids = ctx.extra_model_args[IMAGE_GRID_THW]
        for img, grid in zip(ctx.images, all_grids, strict=True):
            if id(img) not in wanted:
                continue
            wanted.discard(id(img))
            if int(np.prod(grid)) != img.end_idx - img.start_idx:
                raise ValueError(
                    f"image grid {grid.tolist()} does not fill its "
                    f"{img.end_idx - img.start_idx} placeholder tokens"
                )
            pixels.append(img.pixel_values)
            grids.append(grid)
        if wanted:
            raise ValueError(
                f"{len(wanted)} selected image(s) of request {ctx.request_id}"
                " are not in ctx.images"
            )
    return pixels, np.stack(grids) if grids else np.zeros((0, 3), np.int64)


def pack_uncached_images(
    selection: Sequence[tuple[TextAndVisionContext, Sequence[ImageMetadata]]],
    device: Device,
    config: Inkling2VisionConfig,
) -> list[Buffer] | None:
    """Packs the selected images into the vision tower's compile order.

    Returns:
        The inputs of :meth:`Inkling2Vision.input_types`, in its tree's
        order, on ``device`` except the scalar max length; ``None`` if nothing
        is selected.
    """
    pixels, grid_thw = select_images(selection)
    if not pixels:
        return None
    inputs = vision_inputs(
        grid_thw,
        np.concatenate(pixels),
        merge=config.spatial_merge_size,
        posemb_side=config.posemb_grid_size,
    )
    return [
        Buffer.from_numpy(a).to(CPU() if name in HOST_INPUTS else device)
        for name, a in inputs._asdict().items()
    ]
