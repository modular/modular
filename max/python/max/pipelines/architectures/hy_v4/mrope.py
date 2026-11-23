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
"""Hy4 VL multimodal RoPE: ``[3, T]`` positions and their per-token table."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from max import functional as F
from max.dtype import DType
from max.tensor import Tensor

__all__ = ["mla_rope_tables", "mrope_freqs_cis", "mrope_positions"]


def mrope_positions(
    tokens: npt.NDArray[np.integer],
    image_grids: Sequence[tuple[int, int, int]],
    *,
    image_token_id: int,
    spatial_merge_size: int,
) -> tuple[npt.NDArray[np.int64], int]:
    """Returns the temporal, height and width positions of one prompt.

    Text tokens count up by one on all three axes. Each run of image tokens
    is ``rows x (cols + 1)``, one newline per row: all three axes start at
    the next position, height counts rows and width counts columns. The
    next token starts ``max(rows, cols + 1)`` after the image start.

    Args:
        tokens: ``[T]`` prompt token ids.
        image_grids: The ``(t, h, w)`` patch grid of each image, in order.
        image_token_id: The id of every image placeholder token.
        spatial_merge_size: Patches merged per side into one token.

    Returns:
        The ``[3, T]`` positions, and the offset that decode adds to its
        cache position on all three axes.
    """
    is_image = np.diff(
        (tokens == image_token_id).astype(np.int8), prepend=0, append=0
    )
    starts, ends = np.flatnonzero(is_image == 1), np.flatnonzero(is_image == -1)
    positions = np.empty((3, len(tokens)), np.int64)
    next_pos = cursor = 0
    for start, end, (t, h, w) in zip(starts, ends, image_grids, strict=True):
        rows, cols = h // spatial_merge_size, w // spatial_merge_size + 1
        if t != 1 or end - start != rows * cols:
            raise ValueError(
                f"image tokens [{start}, {end}) do not match grid {(t, h, w)}"
            )
        positions[:, cursor:start] = np.arange(start - cursor) + next_pos
        next_pos += start - cursor
        positions[0, start:end] = next_pos
        positions[1, start:end] = np.repeat(np.arange(rows), cols) + next_pos
        positions[2, start:end] = np.tile(np.arange(cols), rows) + next_pos
        next_pos += max(rows, cols)
        cursor = end
    positions[:, cursor:] = np.arange(len(tokens) - cursor) + next_pos
    return positions, int(positions.max()) + 1 - len(tokens)


def mrope_freqs_cis(
    freqs_cis: Tensor, positions: Tensor, mrope_section: Sequence[int]
) -> Tensor:
    """Returns the ``[T, rope_dim]`` interleaved cos/sin table of each token.

    Interleaved M-RoPE rotates pair ``j`` by the height position at
    ``j = 1, 4, ...``, the width position at ``j = 2, 5, ...``, each below
    ``3 * section``, and the temporal position elsewhere. Positions are
    integers, so each pair is an entry of the 1D table: text tokens get its
    rows bit for bit.

    Args:
        freqs_cis: The 1D ``[positions, rope_dim]`` table.
        positions: The batch's ``[3, T]`` int64 positions.
        mrope_section: The pairs each axis rotates.
    """
    half = sum(mrope_section)
    axis = np.zeros(half, np.int64)
    axis[1 : 3 * mrope_section[1] : 3] = 1
    axis[2 : 3 * mrope_section[2] : 3] = 2
    mesh = positions.mesh
    # Row ``p * half + j`` of the flattened table is pair ``j`` at ``p``.
    rows = F.gather(
        positions, F.constant(axis, DType.int64, device=mesh), axis=0
    ).transpose(0, 1) * half + F.constant(
        np.arange(half), DType.int64, device=mesh
    )
    pairs = F.gather(freqs_cis.reshape([-1, 2]), rows, axis=0)
    return pairs.reshape([positions.shape[1], freqs_cis.shape[1]])


def mla_rope_tables(
    freqs_cis: Tensor, positions: Tensor, mrope_section: Sequence[int]
) -> tuple[Tensor, Tensor]:
    """Returns the table the MLA kernel reads and each token's rope rows.

    The MLA kernel rotates by ``freqs_cis`` at cache positions, so it gets
    the identity rotation; the attention blocks rotate by the rows.
    """
    rope_dim = int(freqs_cis.shape[1])
    identity = F.broadcast_to(
        F.constant(
            np.tile(np.array([1, 0], np.float32), rope_dim // 2),
            DType.float32,
            device=positions.mesh,
        ),
        freqs_cis.shape,
    )
    return identity, mrope_freqs_cis(freqs_cis, positions, mrope_section)
