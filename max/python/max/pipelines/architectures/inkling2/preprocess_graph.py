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
"""Inkling 2 image preprocessing as a CPU graph.

For one image the graph does what :class:`Inkling2ImageProcessor` and
:func:`~.vision_inputs.vision_inputs` do: it composites transparency on the
checkerboard, pads the image to whole blocks and cuts it into them, and builds
the tower's position-table taps, class-row gather, rotary positions and
segment bounds. Its outputs are the tower's input tree for that one image.
"""

from __future__ import annotations

from collections import OrderedDict

from max import functional as F
from max.dtype import DType
from max.graph import Dim
from max.nn import Module
from max.nn.legacy.graph_groups import (
    VisionHostMax,
    VisionRowIndex,
    VisionRows,
    VisionSegments,
)
from max.tensor import Tensor

from .image_processor import (
    CHECKER_DARK,
    CHECKER_LIGHT,
    CHECKER_SIDE,
    PAD_RAW_VALUE,
)


def _composite_on_checkerboard(image: Tensor) -> Tensor:
    """Pastes ``uint8 [1, h, w, 4]`` RGBA on the checkerboard, as int32 RGB."""
    pixels = image[0].cast(DType.int32)
    height, width = pixels.shape[0], pixels.shape[1]
    device = image.device
    rows = F.arange(0, height, out_dim=height, dtype=DType.int32, device=device)
    cols = F.arange(0, width, out_dim=width, dtype=DType.int32, device=device)
    squares = (
        rows.reshape([height, 1]) // CHECKER_SIDE
        + cols.reshape([1, width]) // CHECKER_SIDE
    )
    light = (squares % 2 == 0).cast(DType.int32)
    background = CHECKER_DARK + light * (CHECKER_LIGHT - CHECKER_DARK)
    alpha = pixels[:, :, 3:]
    return (
        pixels[:, :, :3] * alpha
        + background.reshape([height, width, 1]) * (255 - alpha)
        + 127
    ) // 255


def _axis_taps(index: Tensor, size: Tensor, side: int) -> tuple[Tensor, Tensor]:
    """Returns two int64 taps and float32 weights per position of an axis."""
    source = ((index.cast(DType.float32) + 0.5) * float(side)) / size.cast(
        DType.float32
    ) - 0.5
    floor = F.floor(source)
    offsets = F.arange(0, 2, dtype=DType.float32, device=index.device)
    weights = F.elementwise_max(
        1.0 - F.abs(source.reshape([-1, 1]) - floor.reshape([-1, 1]) - offsets),
        0.0,
    )
    taps = floor.cast(DType.int64).reshape([-1, 1]) + F.arange(
        0, 2, dtype=DType.int64, device=index.device
    )
    return F.elementwise_min(F.elementwise_max(taps, 0), side - 1), weights


class Inkling2ImagePreprocess(
    Module[[Tensor, Tensor], OrderedDict[str, object]]
):
    """Turns one RGBA8 image into the vision tower's inputs.

    The graph takes
    :func:`~max.nn.legacy.image_preprocess.image_preprocess_input_types` and
    returns the tower's input tree for the image, on the CPU. The tree is the
    one :meth:`Inkling2Vision.input_types` declares, as one image's share of a
    packed launch: its segment bounds start at 0, and its row index counts the
    image's own patches.
    """

    def __init__(
        self, *, patch_size: int, merge_size: int, posemb_side: int
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.posemb_side = posemb_side

    def forward(self, image: Tensor, size: Tensor) -> OrderedDict[str, object]:
        """Returns the tower's input tree for one image.

        ``size`` is the padded image the host lays out, which the image's own
        shape already fixes, so the graph does not read it.
        """
        del size
        device = image.device
        merge = self.merge_size
        block = self.patch_size * merge
        rgb = _composite_on_checkerboard(image)
        height, width = rgb.shape[0], rgb.shape[1]
        block_rows = (height + (block - 1)) // block
        block_cols = (width + (block - 1)) // block
        extent = F.shape_to_tensor([height, width])

        block_row = F.arange(
            0, block_rows, out_dim=block_rows, dtype=DType.int64, device=device
        )
        block_col = F.arange(
            0, block_cols, out_dim=block_cols, dtype=DType.int64, device=device
        )
        inside = F.arange(0, block, dtype=DType.int64, device=device)
        y = (block_row.reshape([block_rows, 1]) * block + inside).reshape(
            [block_rows, 1, block, 1]
        )
        x = (block_col.reshape([block_cols, 1]) * block + inside).reshape(
            [1, block_cols, 1, block]
        )
        valid = F.logical_and(y < extent[0], x < extent[1])
        flat = F.elementwise_min(y, extent[0] - 1) * extent[1]
        flat = flat + F.elementwise_min(x, extent[1] - 1)
        values = F.gather(rgb.reshape([-1, 3]), flat, axis=0)
        tokens = block_rows * block_cols
        pixel_values = F.where(
            valid.unsqueeze(-1), values, PAD_RAW_VALUE
        ).reshape([tokens, block, block, 3])

        # A block's native patches are consecutive, row-major inside it.
        patches = tokens * merge * merge
        within = F.arange(0, merge, dtype=DType.int64, device=device)
        grid: list[Dim | int] = [block_rows, block_cols, merge, merge]
        native_row = F.broadcast_to(
            block_row.reshape([block_rows, 1, 1, 1]) * merge
            + within.reshape([1, 1, merge, 1]),
            grid,
        ).reshape([patches])
        native_col = F.broadcast_to(
            block_col.reshape([1, block_cols, 1, 1]) * merge
            + within.reshape([1, 1, 1, merge]),
            grid,
        ).reshape([patches])

        side = self.posemb_side
        native = F.shape_to_tensor([block_rows * merge, block_cols * merge])
        row_taps, row_weights = _axis_taps(native_row, native[0], side)
        col_taps, col_weights = _axis_taps(native_col, native[1], side)
        interp_idx = (
            row_taps.reshape([-1, 2, 1]) * side + col_taps.reshape([-1, 1, 2])
        ).reshape([patches, 4])
        interp_w = (
            row_weights.reshape([-1, 2, 1]) * col_weights.reshape([-1, 1, 2])
        ).reshape([patches, 4])

        # The sequence is the class row, at (0, 0), then the patches at their
        # 1-indexed positions. The class row gathers index `patches`.
        positions = F.concat(
            [
                F.zeros([1, 2], dtype=DType.int64, device=device),
                F.stack([native_row + 1, native_col + 1], axis=-1),
            ],
            axis=0,
        )
        seq_index = F.concat(
            [
                F.shape_to_tensor([patches]),
                F.arange(
                    0,
                    patches,
                    out_dim=patches,
                    dtype=DType.int64,
                    device=device,
                ),
            ]
        )
        return OrderedDict(
            patches=VisionRows(
                interp_idx=interp_idx.cast(DType.int32),
                interp_w=interp_w,
                positions=positions.cast(DType.int32),
                pixel_values=pixel_values.cast(DType.int16),
            ),
            segments=VisionSegments(
                cu_seqlens=F.shape_to_tensor([0, patches]).cast(DType.uint32)
            ),
            sequence=VisionRowIndex(seq_index=seq_index),
            bound=VisionHostMax(
                max_seqlen=F.shape_to_tensor([patches + 1])[0].cast(
                    DType.uint32
                )
            ),
        )
