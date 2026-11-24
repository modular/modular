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
"""Host-side patch-grid geometry for the GLM-5.3-Flash vision tower.

Pure NumPy, so it can run in the tokenizer / input-processing path.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["vision_cu_seqlens", "vision_position_ids"]


def vision_position_ids(
    grid_thw: npt.NDArray[np.integer[Any]], spatial_merge_size: int
) -> npt.NDArray[np.int32]:
    """Computes per-patch ``(h, w)`` grid coordinates in vision-tower token order.

    Mirrors ``transformers.vision_utils.get_vision_position_ids``: patches are
    emitted in ``spatial_merge_size x spatial_merge_size`` blocks so the tower's
    strided-conv downsample can merge four consecutive rows into one token, and
    each frame of a ``t``-frame entry repeats the same ``(h, w)`` coordinates
    (GLM's vision RoPE has no temporal axis).

    Args:
        grid_thw: ``(num_entries, 3)`` array of ``(t, h, w)`` patch-grid shapes.
        spatial_merge_size: Merge block size from the vision config.

    Returns:
        ``(total_patches, 2)`` int32 array of ``(h_coord, w_coord)`` per patch.
    """
    merge = spatial_merge_size
    per_entry: list[npt.NDArray[np.int32]] = []
    for t, h, w in np.asarray(grid_thw).reshape(-1, 3).tolist():
        t, h, w = int(t), int(h), int(w)
        hpos = np.broadcast_to(np.arange(h, dtype=np.int32)[:, None], (h, w))
        hpos = hpos.reshape(h // merge, merge, w // merge, merge)
        hpos = hpos.transpose(0, 2, 1, 3).reshape(-1)

        wpos = np.broadcast_to(np.arange(w, dtype=np.int32)[None, :], (h, w))
        wpos = wpos.reshape(h // merge, merge, w // merge, merge)
        wpos = wpos.transpose(0, 2, 1, 3).reshape(-1)

        per_entry.append(np.tile(np.stack([hpos, wpos], axis=-1), (t, 1)))

    return np.concatenate(per_entry, axis=0).astype(np.int32)


def vision_cu_seqlens(
    grid_thw: npt.NDArray[np.integer[Any]],
) -> npt.NDArray[np.uint32]:
    """Computes per-frame attention segment boundaries for the vision tower.

    Mirrors ``transformers.vision_utils.get_vision_cu_seqlens``: one segment of
    ``h * w`` patches per frame, ``t`` segments per entry. A frame never attends
    to another frame, which is what makes per-frame vision-cache granularity
    bit-equivalent to encoding a whole clip at once.

    Args:
        grid_thw: ``(num_entries, 3)`` array of ``(t, h, w)`` patch-grid shapes.

    Returns:
        ``(num_frames + 1,)`` uint32 prefix sum of frame lengths, starting at 0.
    """
    grid = np.asarray(grid_thw).reshape(-1, 3)
    lengths = np.repeat(grid[:, 1] * grid[:, 2], grid[:, 0])
    return np.concatenate(
        [np.zeros(1, dtype=np.uint32), np.cumsum(lengths).astype(np.uint32)]
    )
