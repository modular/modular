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

"""Compare Gemma4ImageProcessor outputs against HuggingFace reference."""

from __future__ import annotations

import hf_gemma4_reference
import numpy as np
import pytest
from max.pipelines.architectures.gemma4.image_processor import (
    Gemma4ImageProcessor,
)
from PIL import Image

PATCH_SIZE = 16
MAX_SOFT_TOKENS = 280
POOLING_K = 3


def _patchify_chw(img: np.ndarray, patch_size: int) -> np.ndarray:
    """Reshape a CHW image into ``(num_patches, patch_dim)``."""
    c, h, w = img.shape
    ph, pw = h // patch_size, w // patch_size
    return (
        img.reshape(c, ph, patch_size, pw, patch_size)
        .transpose(1, 3, 2, 4, 0)
        .reshape(ph * pw, patch_size * patch_size * c)
    )


def _run_both(
    images: list[Image.Image],
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], list[int]]:
    """Run both HF reference and MAX processors, return patchified outputs.

    Returns ``(max_patches, hf_patches, max_soft_tokens, hf_soft_tokens)``
    where each ``*_patches`` list has one array per image.
    """
    hf_proc = hf_gemma4_reference.Gemma4ImageProcessor(
        patch_size=PATCH_SIZE,
        max_soft_tokens=MAX_SOFT_TOKENS,
        pooling_kernel_size=POOLING_K,
        do_rescale=True,
        rescale_factor=1.0 / 255.0,
        do_normalize=False,
        image_mean=[0.0, 0.0, 0.0],
        image_std=[1.0, 1.0, 1.0],
    )
    hf_out = hf_proc(images=images, return_tensors="np")

    max_proc = Gemma4ImageProcessor(
        patch_size=PATCH_SIZE,
        max_soft_tokens=MAX_SOFT_TOKENS,
        pooling_kernel_size=POOLING_K,
    )
    max_pv, _, max_soft = max_proc(images)

    hf_patches = [
        _patchify_chw(pv, PATCH_SIZE) for pv in hf_out["pixel_values"]
    ]
    return max_pv, hf_patches, max_soft, hf_out["num_soft_tokens_per_image"]


@pytest.mark.parametrize(
    "images",
    [
        [Image.new("RGB", (640, 480), color="blue")],
        [Image.new("RGB", (32, 32), color="white")],
        [
            Image.new("RGB", (800, 200), color="red"),
            Image.new("RGB", (200, 800), color="green"),
            Image.new("RGB", (500, 500), color="blue"),
        ],
    ],
    ids=["single", "small", "multi-aspect"],
)
def test_matches_hf_reference(images: list[Image.Image]) -> None:
    """Patchified pixel values and soft token counts match HF reference."""
    max_pv, hf_pv, max_soft, hf_soft = _run_both(images)

    assert len(max_pv) == len(hf_pv)
    assert max_soft == hf_soft, f"Soft tokens: {max_soft} vs {hf_soft}"

    for i, (m, h) in enumerate(zip(max_pv, hf_pv, strict=True)):
        assert m.shape == h.shape, f"Image {i} shape: {m.shape} vs {h.shape}"
        np.testing.assert_allclose(
            m, h, rtol=1e-2, atol=1e-2, err_msg=f"Image {i} pixel mismatch"
        )


# -- Preprocessing-parameter tests (no HF dependency) -------------------------


def test_do_rescale_false() -> None:
    """With do_rescale=False pixel values stay in [0, 255]."""
    img = Image.new("RGB", (480, 480), color=(128, 64, 32))
    pv, _, _ = Gemma4ImageProcessor(do_rescale=False)([img])
    assert pv[0].max() > 1.0


def test_custom_rescale_factor() -> None:
    """A custom rescale_factor scales pixel values accordingly."""
    img = Image.new("RGB", (480, 480), color=(255, 255, 255))
    factor = 1.0 / 128.0
    pv, _, _ = Gemma4ImageProcessor(rescale_factor=factor)([img])
    np.testing.assert_allclose(pv[0].max(), 255.0 * factor, atol=0.01)


def test_do_normalize() -> None:
    """With do_normalize=True, values are shifted by mean/std."""
    img = Image.new("RGB", (480, 480), color=(128, 128, 128))
    pv, _, _ = Gemma4ImageProcessor(
        do_normalize=True,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
    )([img])
    assert np.all(pv[0] >= -1.1) and np.all(pv[0] <= 1.1)


def test_do_resize_false() -> None:
    """With do_resize=False, patch count reflects original dimensions."""
    side = 48
    pv, _, _ = Gemma4ImageProcessor(do_resize=False)(
        [Image.new("RGB", (side, side), color="red")]
    )
    assert pv[0].shape[0] == (side // 16) ** 2
