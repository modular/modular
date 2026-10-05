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

"""Packs a batch's uncached images into Muse Glimmer vision tower inputs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.graph.buffer_utils import cast_tensor_to
from max.pipelines.context import ImageMetadata, TextAndVisionContext

from .model_config import MuseGlimmerVisionConfig
from .vision import VisionInputs, vision_inputs

IMAGE_GRID_THW = "image_grid_thw"
"""``extra_model_args`` key of the ``[len(images), 3]`` patch grids, one row
per entry of ``ctx.images``."""


def select_images(
    selection: Sequence[tuple[TextAndVisionContext, Sequence[ImageMetadata]]],
    merge_size: int,
) -> tuple[list[npt.NDArray[np.float32]], npt.NDArray[np.int64]]:
    """Returns the pixels and ``[n, 3]`` grids of the selected images.

    Images are matched to their grid by identity with ``ctx.images``, in
    context-major, image-minor order, the order the encoder must emit rows.

    Raises:
        ValueError: If a selected image is not one of its context's own, or
            its grid does not match its placeholder span.
    """
    pixels: list[npt.NDArray[np.float32]] = []
    grids: list[npt.NDArray[np.int64]] = []
    for ctx, miss_images in selection:
        wanted = {id(img) for img in miss_images}
        all_grids = ctx.extra_model_args[IMAGE_GRID_THW]
        for img, grid in zip(ctx.images, all_grids, strict=True):
            if id(img) not in wanted:
                continue
            wanted.discard(id(img))
            t, h, w = (int(x) for x in grid)
            if t * h * w // merge_size**2 != img.end_idx - img.start_idx:
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
    config: MuseGlimmerVisionConfig,
    dtype: DType,
) -> list[Buffer] | None:
    """Packs the selected images into the vision tower's compile order.

    Returns:
        ``pixel_values`` then every :class:`VisionInputs` field, on
        ``device`` except the two max lengths, or ``None`` if nothing is
        selected.
    """
    pixels, grid_thw = select_images(selection, config.merge_size)
    if not pixels:
        return None
    patches = Buffer.from_numpy(
        np.concatenate(pixels).astype(np.float32, copy=False)
    ).to(device)
    inputs: VisionInputs = vision_inputs(grid_thw, config)
    return [
        cast_tensor_to(patches, dtype),
        *(
            Buffer.from_numpy(a).to(CPU() if a.ndim == 0 else device)
            for a in inputs
        ),
    ]
