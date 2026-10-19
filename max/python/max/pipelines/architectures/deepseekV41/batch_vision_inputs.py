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

"""Packs a batch's uncached images into DeepSeek-V4.1 vision model inputs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.pipelines.context import ImageMetadata, TextAndVisionContext

from .vision import DeepseekV41VisionConfig, vision_inputs

IMAGE_GRIDS = "image_grids"
"""``extra_model_args`` key of the ``[len(images), 2]`` ViT patch grids
``(n_vit_h, n_vit_w)``, one row per entry of ``ctx.images``."""


def span_tokens(n_vit_h: int, n_vit_w: int, ratio: int) -> int:
    """Positions of the image span of an ``n_vit_h x n_vit_w`` patch grid."""
    n_llm_h, n_llm_w = -(-n_vit_h // ratio), -(-n_vit_w // ratio)
    return n_llm_h * (n_llm_w + 1) + 2


def select_images(
    selection: Sequence[tuple[TextAndVisionContext, Sequence[ImageMetadata]]],
    ratio: int,
) -> tuple[list[npt.NDArray[np.uint16]], list[tuple[int, int]]]:
    """Returns the patches and patch grids of the selected images.

    Images are matched to their grid by identity with ``ctx.images``, in
    context-major, image-minor order, the order the encoder emits rows.

    Raises:
        ValueError: If a selected image is not one of its context's own, or
            its grid does not fill its span.
    """
    patches: list[npt.NDArray[np.uint16]] = []
    grids: list[tuple[int, int]] = []
    for ctx, miss_images in selection:
        wanted = {id(img) for img in miss_images}
        all_grids = ctx.extra_model_args[IMAGE_GRIDS]
        for img, grid in zip(ctx.images, all_grids, strict=True):
            if id(img) not in wanted:
                continue
            wanted.discard(id(img))
            h, w = (int(x) for x in grid)
            if span_tokens(h, w, ratio) != img.end_idx - img.start_idx:
                raise ValueError(
                    f"image grid {(h, w)} does not fill its "
                    f"{img.end_idx - img.start_idx} span tokens"
                )
            patches.append(img.pixel_values)
            grids.append((h, w))
        if wanted:
            raise ValueError(
                f"{len(wanted)} selected image(s) of request {ctx.request_id}"
                " are not in ctx.images"
            )
    return patches, grids


def pack_uncached_images(
    selection: Sequence[tuple[TextAndVisionContext, Sequence[ImageMetadata]]],
    device: Device,
    config: DeepseekV41VisionConfig,
) -> list[Buffer] | None:
    """Packs the selected images in the vision model's compile order.

    Returns:
        The bf16 ``[P, 3 * patch_size**2]`` patches then every
        :class:`~.vision.VisionInputs` field, on ``device`` except
        ``max_seqlen``, or ``None`` if nothing is selected.
    """
    patches, grids = select_images(selection, config.downsample_ratio)
    if not patches:
        return None
    # The tokenizer keeps bf16 bit patterns as uint16.
    flat = np.concatenate([p.reshape(len(p), -1) for p in patches])
    host = vision_inputs(grids, config)
    return [
        Buffer.from_numpy(flat).view(DType.bfloat16).to(device),
        *(Buffer.from_numpy(a).to(device) for a in host[:3]),
        Buffer.from_numpy(host.max_seqlen).to(CPU()),
        *(Buffer.from_numpy(a).to(device) for a in host[4:]),
    ]
