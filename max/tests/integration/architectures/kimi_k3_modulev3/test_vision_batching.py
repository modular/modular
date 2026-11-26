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
"""Checks image selection against the original context's patch offsets."""

import numpy as np
import pytest
from max.pipelines.architectures.kimi_k3_modulev3.batch_processor import (
    KimiK3BatchProcessor,
)
from max.pipelines.architectures.kimi_k3_modulev3.context import (
    KimiK3TextAndVisionContext,
)
from max.pipelines.context import ImageMetadata


@pytest.mark.parametrize("selected", [[1], [1, 0]])
def test_selected_images_keep_their_original_patch_slices(
    selected: list[int],
) -> None:
    grids = np.array([[1, 2, 2], [1, 2, 4]], dtype=np.int64)
    images = [
        ImageMetadata(
            start_idx=10,
            end_idx=11,
            pixel_values=np.full((4, 3, 14, 14), 1, dtype=np.float32),
        ),
        ImageMetadata(
            start_idx=20,
            end_idx=22,
            pixel_values=np.full((8, 3, 14, 14), 2, dtype=np.float32),
        ),
    ]
    context = object.__new__(KimiK3TextAndVisionContext)
    context.images = images
    context.grid_thws = grids
    context.position_ids = np.arange(12, dtype=np.int64)
    batcher = object.__new__(KimiK3BatchProcessor)

    packed = batcher.pack_vision_inputs(
        [(context, [images[index] for index in selected])], []
    )

    assert len(packed) == len(selected)
    offsets = [0, 4, 12]
    for entry, index in zip(packed, selected, strict=True):
        np.testing.assert_array_equal(entry.pixels, images[index].pixel_values)
        np.testing.assert_array_equal(entry.grid, grids[index : index + 1])
        np.testing.assert_array_equal(
            entry.positions,
            context.position_ids[offsets[index] : offsets[index + 1]],
        )
        assert entry.token_count == images[index].embedding_rows
