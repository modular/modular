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
"""Tests Kimi K3 host image preprocessing and image token spans."""

import numpy as np
import pytest
from max.pipelines.architectures.kimi_k3_modulev3.tokenizer import (
    expand_image_tokens,
)
from max.pipelines.architectures.kimi_k3_modulev3.vision_processor import (
    KimiK3VisionProcessor,
    MediaProcConfig,
    compute_position_ids,
    navit_resize_image,
)
from max.pipelines.context.exceptions import InputError
from PIL import Image

PATCH = 14


def _normalized(level: int) -> float:
    return (level / 255.0 - 0.5) / 0.5


def test_patches_are_row_major_and_channel_first() -> None:
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, size=(28, 28, 3), dtype=np.uint8)
    output = KimiK3VisionProcessor().preprocess(
        [{"type": "image", "image": Image.fromarray(pixels)}]
    )

    np.testing.assert_array_equal(output["grid_thws"], [[1, 2, 2]])
    patches = output["pixel_values"]
    assert patches.shape == (4, 3, PATCH, PATCH)
    assert patches.dtype == np.float32
    expected = (pixels[:PATCH, PATCH:, :] / 255.0 - 0.5) / 0.5
    np.testing.assert_allclose(
        patches[1], expected.transpose(2, 0, 1), rtol=0, atol=1e-6
    )


def test_padding_fills_bottom_right_with_zero_level() -> None:
    plan = navit_resize_image(29, 15, MediaProcConfig())
    assert (plan["pad_width"], plan["pad_height"]) == (27, 13)
    assert plan["num_tokens"] == 2

    image = Image.new("RGB", (29, 15), (255, 255, 255))
    output = KimiK3VisionProcessor().preprocess(
        [{"type": "image", "image": image}]
    )

    np.testing.assert_array_equal(output["grid_thws"], [[1, 2, 4]])
    last = output["pixel_values"][-1]
    np.testing.assert_allclose(last, _normalized(0), atol=1e-6)


def test_transparency_uses_checkpoint_chessboard() -> None:
    config = MediaProcConfig.from_dict(
        {
            "transparent_bg_config": {
                "pattern": "chessboard",
                "chessboard_square_size": 8,
                "chessboard_gray_value": 180,
            },
            "transparent_bg_fill_stage": "after_resize",
        }
    )
    image = Image.new("RGBA", (28, 28), (0, 0, 0, 0))
    patches = KimiK3VisionProcessor(config).preprocess(
        [{"type": "image", "image": image}]
    )["pixel_values"]

    assert patches[0, 0, 0, 0] == pytest.approx(_normalized(255))
    assert patches[0, 0, 0, 8] == pytest.approx(_normalized(180))


def test_default_config_matches_an_empty_checkpoint_config() -> None:
    assert MediaProcConfig() == MediaProcConfig.from_dict({})


def test_rejects_image_bytes_outside_the_allowed_formats() -> None:
    eps = b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 1 1\n"
    with pytest.raises(InputError):
        KimiK3VisionProcessor().preprocess([{"type": "image", "image": eps}])


def test_adjacent_images_keep_separate_spans() -> None:
    pad = 9
    tokens, spans = expand_image_tokens(
        np.array([1, pad, pad, 2], dtype=np.int64), [2, 3], pad
    )

    np.testing.assert_array_equal(tokens, [1, pad, pad, pad, pad, pad, 2])
    assert spans == [(1, 3), (3, 6)]


def test_expand_rejects_a_media_count_mismatch() -> None:
    with pytest.raises(ValueError, match="placeholder count"):
        expand_image_tokens(np.array([1, 9], dtype=np.int64), [2, 3], 9)


def test_position_ids_use_rope_row_stride() -> None:
    grids = np.array([[1, 2, 3], [1, 1, 2]], dtype=np.int64)
    np.testing.assert_array_equal(
        compute_position_ids(grids), [0, 1, 2, 512, 513, 514, 0, 1]
    )
