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
"""Checks that the vision cache's rows fit what the tower returns."""

from __future__ import annotations

from types import SimpleNamespace

from max.dtype import DType
from max.pipelines.architectures.qwen3vl_moe.model_config import (
    Qwen3VLConfig,
    VisionConfig,
)

# Qwen/Qwen3-VL-30B-A3B-Instruct's vision config.
_VISION = SimpleNamespace(
    patch_size=16,
    temporal_patch_size=2,
    in_channels=3,
    hidden_size=1152,
    num_heads=16,
    depth=27,
    intermediate_size=4304,
    out_hidden_size=2048,
    deepstack_visual_indexes=[8, 16, 24],
    spatial_merge_size=2,
    num_position_embeddings=2304,
)


def test_a_row_holds_the_embedding_then_each_deepstack_layer() -> None:
    vision = VisionConfig.initialize_from_config(_VISION)
    assert vision.embedding_width == 2048 * 4


def test_the_cache_rows_take_the_checkpoints_dtype() -> None:
    for declared, dtype in (
        ("bfloat16", DType.bfloat16),
        ("float32", DType.float32),
    ):
        hf_config = SimpleNamespace(
            vision_config=_VISION, text_config=SimpleNamespace(dtype=declared)
        )
        assert Qwen3VLConfig.get_vision_cache_row_spec(hf_config) == (
            2048 * 4,
            dtype,
        )
