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

"""Packs a batch's uncached images into Muse Spark vision tower inputs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.graph.buffer_utils import cast_tensor_to
from max.pipelines.architectures.muse_glimmer.batch_vision_inputs import (
    select_images,
)
from max.pipelines.architectures.muse_glimmer.model_config import (
    MuseGlimmerVisionConfig,
)
from max.pipelines.context import ImageMetadata, TextAndVisionContext

from .vision import vision_inputs


def pack_uncached_images(
    selection: Sequence[tuple[TextAndVisionContext, Sequence[ImageMetadata]]],
    device: Device,
    config: MuseGlimmerVisionConfig,
    dtype: DType,
) -> list[Buffer] | None:
    """Packs the selected images into the vision tower's compile order.

    Returns:
        ``pixel_values`` then every vision input, on ``device`` except the
        two max lengths, or ``None`` if nothing is selected.
    """
    pixels, grid_thw = select_images(selection, config.merge_size)
    if not pixels:
        return None
    patches = Buffer.from_numpy(
        np.concatenate(pixels).astype(np.float32, copy=False)
    ).to(device)
    return [
        cast_tensor_to(patches, dtype),
        *(
            Buffer.from_numpy(a).to(CPU() if a.ndim == 0 else device)
            for a in vision_inputs(grid_thw, config)
        ),
    ]
