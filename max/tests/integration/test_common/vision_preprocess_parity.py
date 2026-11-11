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
"""Images, engine inputs and checks the vision_preprocess parity tests share.

Each model family's test runs its exported preprocessing graph on what a
serving engine passes and compares every output with what MAX serve feeds
the vision tower, bit for bit. The graph's uint8 resize is Pillow's, so the
pixels match on smooth content such as :func:`scene` and :func:`gradient`
images and on the hard edges of :func:`chart` images, which overshoot
[0, 255] between Pillow's two passes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from max.driver import Buffer
from max.dtype import DType
from max.tensor.compilation import CompiledCallable
from PIL import Image


def noise(
    height: int, width: int, mode: str, seed: int | None = None
) -> Image.Image:
    """Uniform noise in an ``RGB``, ``RGBA``, ``L`` or ``LA`` image."""
    channels = {"RGB": 3, "RGBA": 4, "L": 1, "LA": 2}[mode]
    pixels = np.random.default_rng(
        height * width if seed is None else seed
    ).integers(0, 256, (height, width, channels), dtype=np.uint8)
    return Image.fromarray(pixels.squeeze(-1) if channels == 1 else pixels)


def gradient(height: int, width: int) -> Image.Image:
    """Smooth ramps, one per channel."""
    y, x = np.mgrid[0:height, 0:width]
    rgb = np.stack(
        [
            x * 255 // max(width - 1, 1),
            y * 255 // max(height - 1, 1),
            (x + y) * 255 // max(width + height - 2, 1),
        ],
        axis=-1,
    )
    return Image.fromarray(rgb.astype(np.uint8))


def scene(
    height: int, width: int, mode: str = "RGB", seed: int | None = None
) -> Image.Image:
    """Smooth waves with mild noise, in an ``RGB``, ``RGBA``, ``L`` or ``LA``
    image: content like a photo's, whose resize stays within [0, 255]."""
    rng = np.random.default_rng(height * width if seed is None else seed)
    y, x = np.mgrid[0:height, 0:width]
    channels = []
    for _ in range(4):
        fx, fy = rng.uniform(0.01, 0.08, 2)
        phase = rng.uniform(0, 2 * np.pi)
        channels.append(
            128 + 60 * np.sin(x * fx + phase) + 50 * np.cos(y * fy - phase)
        )
    pixels = np.stack(channels, -1) + rng.normal(0, 6, (height, width, 4))
    rgba = np.clip(np.rint(pixels), 0, 255).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA").convert(mode)


# The strokes of a seven-segment glyph in a 7x5 cell: top, upper right, lower
# right, bottom, lower left, upper left and middle.
_SEGMENTS = (
    (slice(0, 1), slice(0, 5)),
    (slice(0, 4), slice(4, 5)),
    (slice(3, 7), slice(4, 5)),
    (slice(6, 7), slice(0, 5)),
    (slice(3, 7), slice(0, 1)),
    (slice(0, 4), slice(0, 1)),
    (slice(3, 4), slice(0, 5)),
)


def _text(
    rgb: npt.NDArray[np.uint8],
    rows: range,
    cols: range,
    ink: int,
    rng: np.random.Generator,
) -> None:
    """Writes lines of 1-pixel glyphs, with gaps between words, in ``ink``."""
    for top in rows:
        for left in cols:
            if rng.random() < 0.2:
                continue
            cell = rgb[top : top + 7, left : left + 5]
            for (ys, xs), on in zip(
                _SEGMENTS, rng.random(len(_SEGMENTS)) < 0.6, strict=True
            ):
                if on:
                    cell[ys, xs] = ink


def chart(height: int, width: int, seed: int | None = None) -> Image.Image:
    """A chart's hard edges, which take a resize past [0, 255]: black text,
    1- and 2-pixel axis and grid lines, and solid saturated bars on white,
    and a black panel with white text and lines."""
    rng = np.random.default_rng(height * width if seed is None else seed)
    rgb = np.full((height, width, 3), 255, np.uint8)
    split = width * 3 // 4
    # The black panel, with white text and a 1- and a 2-pixel line.
    rgb[: height // 2, split:] = 0
    _text(
        rgb,
        range(2, height // 2 - 7, 10),
        range(split + 2, width - 5, 7),
        255,
        rng,
    )
    rgb[height // 4, split:] = 255
    rgb[height // 3 : height // 3 + 2, split:] = 255
    # The plot's title and labels.
    _text(rgb, range(1, height // 5 - 7, 10), range(1, split - 5, 7), 0, rng)
    top, bottom, left = height // 5, height - 3, 3
    colors = np.array(
        [[255, 0, 0], [0, 0, 255], [0, 255, 0], [255, 255, 0], [255, 0, 255]],
        np.uint8,
    )
    bars = 2 * len(colors) + 1
    for i, color in enumerate(colors):
        x0 = left + (2 * i + 1) * (split - left) // bars
        x1 = left + (2 * i + 2) * (split - left) // bars
        y0 = top + int(rng.uniform(0, 0.8) * (bottom - top))
        rgb[y0:bottom, x0:x1] = color
    # 1-pixel grid lines over the bars, and 2-pixel axes.
    rgb[top : bottom : max((bottom - top) // 6, 2), left:split] = 0
    rgb[top:bottom, left - 2 : left] = 0
    rgb[bottom : bottom + 2, left - 2 : split] = 0
    return Image.fromarray(rgb)


def palette_with_transparency() -> Image.Image:
    """A palette image with a transparent index, which Pillow does not open
    as ``RGBA``."""
    image = scene(112, 140).quantize(16)
    image.info["transparency"] = 3
    return image


def graph_inputs(
    image: Image.Image, size: tuple[int, int]
) -> tuple[Buffer, Buffer]:
    """The pixels and size a serving engine passes: RGBA, with alpha 255
    for an image Pillow does not open as RGBA."""
    if image.mode == "RGBA":
        rgba = np.asarray(image)
    else:
        rgb = np.asarray(image.convert("RGB"))
        rgba = np.concatenate(
            [rgb, np.full((*rgb.shape[:2], 1), 255, np.uint8)], axis=-1
        )
    return (
        Buffer.from_numpy(np.ascontiguousarray(rgba[None])),
        Buffer.from_numpy(np.array(size, np.int64)),
    )


def bits(buffer: Buffer) -> npt.NDArray[Any]:
    """The buffer's values, bfloat16 as its raw bits."""
    if buffer.dtype == DType.bfloat16:
        buffer = buffer.view(DType.uint16)
    return buffer.to_numpy()


def check(
    graph: CompiledCallable[..., Any],
    case: str,
    outputs: list[Buffer],
    served: list[Buffer],
) -> None:
    """Asserts that the graph's outputs are what MAX serve feeds the tower,
    bit for bit, naming the case and output path of any difference.

    Args:
        graph: The compiled preprocessing graph.
        case: The case's name, for the failure message.
        outputs: The graph's outputs.
        served: What MAX serve feeds the tower, in the same order.
    """
    paths = graph.out_tree.leaf_paths
    assert len(outputs) == len(served) == len(paths), case
    for path, got, want in zip(paths, outputs, served, strict=True):
        assert got.dtype == want.dtype, f"{case}: {path}"
        np.testing.assert_array_equal(
            bits(got), bits(want), err_msg=f"{case}: {path}"
        )
