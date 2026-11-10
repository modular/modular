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
"""GLM-5.3-Flash encodes a batch's images in chunks the planner sized.

The memory planner reserves the tower's activations for one encode of at most
``max_batch_input_tokens`` output rows, or one largest image when that is
larger. ``vision_execute`` must honor that bound: no chunk holds more rows
unless it is a single image, no image is split, and the rows come back in
the order the images were selected.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import MLAKVCacheParams, MultiKVCacheParams
from max.pipelines.architectures.glm5_next.batch_processor import (
    Glm5NextBatchProcessor,
)
from max.pipelines.architectures.glm5_next.model_config import Glm5NextConfig
from max.pipelines.architectures.glm5_next.vision.config import (
    Glm5NextVisionConfig,
)

MERGE = 2
HIDDEN = 4
PATCH_DIM = 6


def _config() -> Glm5NextConfig:
    leaf = MLAKVCacheParams(
        dtype=DType.bfloat16,
        head_dim=576,
        num_layers=1,
        page_size=128,
        devices=[DeviceRef.CPU()],
        num_q_heads=8,
    )
    return Glm5NextConfig(
        dtype=DType.bfloat16,
        kv_params=MultiKVCacheParams.from_params(
            {"mla": leaf, "indexer": leaf}
        ),
        devices=[DeviceRef.CPU()],
        max_seq_len=1024,
        hidden_size=HIDDEN,
        intermediate_size=8,
        num_attention_heads=8,
        num_key_value_heads=8,
        num_hidden_layers=1,
        qk_rope_head_dim=0,
        vision_config=Glm5NextVisionConfig(
            depth=1,
            hidden_size=8,
            num_heads=2,
            intermediate_size=8,
            projection_intermediate_size=8,
            out_hidden_size=HIDDEN,
            patch_size=1,
            temporal_patch_size=1,
            spatial_merge_size=MERGE,
        ),
    )


def _image(index: int, merged_rows: int) -> tuple[Any, np.ndarray]:
    """An image of ``merged_rows`` output rows, its pixels tagged by index."""
    grid = np.array([1, MERGE, MERGE * merged_rows], dtype=np.int64)
    patches = int(np.prod(grid))
    pixels = np.full((patches, PATCH_DIM), index, dtype=np.float32)
    return SimpleNamespace(start_idx=10 * index, pixel_values=pixels), grid


def _batcher(
    budget: int, monkeypatch: pytest.MonkeyPatch
) -> tuple[Glm5NextBatchProcessor, list[list[int]]]:
    """A batcher whose tower records each chunk and tags its rows."""
    batcher = object.__new__(Glm5NextBatchProcessor)
    batcher.config = _config()
    batcher.runtime = cast(
        Any, SimpleNamespace(max_batch_input_tokens=budget, devices=[CPU()])
    )
    batcher._vision_model = cast(Any, object())
    chunks: list[list[int]] = []

    def encode(
        entries: Sequence[tuple[np.ndarray, np.ndarray]],
    ) -> list[Buffer]:
        tags = [int(pixels[0, 0]) for pixels, _ in entries]
        chunks.append(tags)
        rows = [
            np.full((int(np.prod(grid)) // MERGE**2, HIDDEN), tag, np.float32)
            for tag, (_, grid) in zip(tags, entries, strict=True)
        ]
        return [Buffer.from_numpy(np.concatenate(rows))]

    monkeypatch.setattr(batcher, "_encode_images", encode)
    return batcher, chunks


def _select(
    sizes: Sequence[Sequence[int]],
) -> list[tuple[Any, list[Any]]]:
    """One context per inner list, every image a cache miss."""
    selection: list[tuple[Any, list[Any]]] = []
    index = 0
    for context_sizes in sizes:
        images, grids = [], []
        for size in context_sizes:
            image, grid = _image(index, size)
            images.append(image)
            grids.append(grid)
            index += 1
        context = SimpleNamespace(images=images, grid_thws=grids)
        selection.append((context, list(images)))
    return selection


@pytest.mark.parametrize(
    ("sizes", "budget", "expected"),
    [
        # Packs up to the budget, starting a chunk where the next image
        # would pass it.
        ([[60, 30], [20]], 100, [[0, 1], [2]]),
        # An image past the budget runs alone and whole.
        ([[20, 120, 10]], 100, [[0], [1], [2]]),
        # Exactly the budget still fits.
        ([[50], [50], [1]], 100, [[0, 1], [2]]),
        ([[5, 5, 5]], 100, [[0, 1, 2]]),
    ],
)
def test_chunks_stay_within_the_input_budget(
    sizes: list[list[int]],
    budget: int,
    expected: list[list[int]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batcher, chunks = _batcher(budget, monkeypatch)
    result = batcher.vision_execute(
        cast(Any, _select(sizes)), [CPU()], packed=None
    )
    assert chunks == expected
    flat = [size for context in sizes for size in context]
    assert result.per_image_token_counts == flat
    for chunk in chunks:
        rows = sum(flat[i] for i in chunk)
        assert rows <= budget or len(chunk) == 1
    (embeddings,) = result.embeddings
    tags = np.from_dlpack(embeddings)[:, 0].astype(int).tolist()
    assert tags == [i for i, size in enumerate(flat) for _ in range(size)]
