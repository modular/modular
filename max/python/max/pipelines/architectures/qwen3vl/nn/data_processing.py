# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from max.profiler import traced


@traced
def get_bilinear_interpolation_weights_and_indices(
    grid_thw: npt.NDArray[np.integer[Any]],
    num_grid_per_side: int,
) -> tuple[npt.NDArray[np.integer[Any]], npt.NDArray[np.floating[Any]]]:
    """Calculate the bilinear interpolation weights and indices from the offsets
    of patches in the original pixel values.
    Converted the original implementation from torch to numpy. Original implementation from:
    https://github.com/vllm-project/vllm/blob/9fce7bee745230d61c60ad467966790553b0ba48/vllm/model_executor/models/qwen3_vl.py#L444

    Returns a tuple (indices, weights).

    indices : np.ndarray of shape (4, N), dtype=int64
        Indices of the four bilinear neighbors for each patch position, where N is the total number of patch positions.
        The first axis (size 4) corresponds to the four neighbors: (top-left, top-right, bottom-left, bottom-right).
    weights : np.ndarray of shape (4, N, 1), dtype=float32
        Bilinear interpolation weights for each neighbor and patch position.
        The first axis (size 4) matches the order of indices.
    """
    indices_list = []
    weights_list = []
    for _, h, w in grid_thw:
        h_idxs = np.linspace(0, num_grid_per_side - 1, h, dtype=np.float32)
        w_idxs = np.linspace(0, num_grid_per_side - 1, w, dtype=np.float32)

        h_floor = h_idxs.astype(np.int64)
        w_floor = w_idxs.astype(np.int64)
        h_ceil = np.clip(h_floor + 1, 0, num_grid_per_side - 1)
        w_ceil = np.clip(w_floor + 1, 0, num_grid_per_side - 1)

        dh = h_idxs - h_floor
        dw = w_idxs - w_floor

        # Create meshgrid view for all h, w vars
        dh_grid, dw_grid = np.meshgrid(dh, dw, indexing="ij")
        h_floor_grid, w_floor_grid = np.meshgrid(
            h_floor, w_floor, indexing="ij"
        )
        h_ceil_grid, w_ceil_grid = np.meshgrid(h_ceil, w_ceil, indexing="ij")
        h_floor_grid_idx = h_floor_grid * num_grid_per_side
        h_ceil_grid_idx = h_ceil_grid * num_grid_per_side

        # original computation of weights
        # w00 = (1 - dh_grid) * (1 - dw_grid)
        # w01 = (1 - dh_grid) * dw_grid
        # w10 = dh_grid * (1 - dw_grid)
        # w11 = dh_grid * dw_grid
        # we reuse w11 here to avoid duplicate
        # dh_grid * dw_grid computation
        w11 = dh_grid * dw_grid
        w10 = dh_grid - w11
        w01 = dw_grid - w11
        w00 = 1 - dh_grid - dw_grid + w11

        idx00 = h_floor_grid_idx + w_floor_grid
        idx01 = h_floor_grid_idx + w_ceil_grid
        idx10 = h_ceil_grid_idx + w_floor_grid
        idx11 = h_ceil_grid_idx + w_ceil_grid

        indices = np.stack([idx00, idx01, idx10, idx11], axis=0).reshape(4, -1)
        weights = np.stack([w00, w01, w10, w11], axis=0).reshape(4, -1, 1)
        indices_list.append(indices)
        weights_list.append(weights)

    return np.concatenate(indices_list, axis=1), np.concatenate(
        weights_list, axis=1
    )


@traced
def mrope_pos_ids_3d(
    grid_thw: npt.NDArray[np.integer[Any]],
    spatial_merge_size: int,
) -> npt.NDArray[np.integer[Any]]:
    """Calculate the rope index based on image and video's temporal, height, and width in LLM using NumPy.
    This is used in the vision transformer to calculate the rotary position embeddings.
    Converted the original implementation from torch to numpy. Original implementation from:
    https://github.com/vllm-project/vllm/blob/9fce7bee745230d61c60ad467966790553b0ba48/vllm/model_executor/models/qwen3_vl.py#L409

    Args:
        grid_thw: Array of shape [num_images, 3] with (t, h, w) for each image/video
        spatial_merge_size: Factor for downscaling spatial dimensions

    Returns:
        pos_ids: Array of shape [seq_len, 2] with (h, w) for each position
    """
    pos_ids = []
    for t, h, w in grid_thw:
        # Create height position IDs
        hpos_ids = (
            np.broadcast_to(np.arange(h).reshape(-1, 1), (h, w))
            .reshape(
                h // spatial_merge_size,
                spatial_merge_size,
                w // spatial_merge_size,
                spatial_merge_size,
            )
            .transpose(0, 2, 1, 3)
            .flatten()
        )

        # Create width position IDs
        wpos_ids = (
            np.broadcast_to(np.arange(w).reshape(1, -1), (h, w))
            .reshape(
                h // spatial_merge_size,
                spatial_merge_size,
                w // spatial_merge_size,
                spatial_merge_size,
            )
            .transpose(0, 2, 1, 3)
            .flatten()
        )

        # Stack height and width position IDs
        coords = np.stack([hpos_ids, wpos_ids], axis=-1)
        # Repeat for temporal dimension
        pos_ids.append(np.tile(coords, (t, 1)))

    return np.concatenate(pos_ids, axis=0)


@traced
def get_seqlens(
    grid_thw: npt.NDArray[np.integer[Any]],
) -> tuple[
    npt.NDArray[np.integer[Any]],
    int,
]:
    """Generate attention masks for visual tokens using seq_length and cu_seqlens.
    cu_seqlens is used for all blocks in Qwen3VL to implement full attention.

    Args:
        grid_thw: number of patches in spatial and temporal dims in images. Shape = [n_images, 3]

    Returns:
        Tuple of (cu_seqlens, max_seqlen)
    """
    repeated_sizes = np.repeat(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
    cu_seqlens = np.cumsum(repeated_sizes, dtype=np.int32)

    cu_seqlens = np.pad(cu_seqlens, (1, 0), constant_values=0)
    max_seqlen = int(np.max(np.diff(cu_seqlens)))

    return cu_seqlens, max_seqlen
