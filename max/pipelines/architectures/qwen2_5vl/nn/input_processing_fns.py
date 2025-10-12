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

"""NumPy implementations of PyTorch functions for Qwen2.5VL vision transformer.

This module provides NumPy equivalents of PyTorch functions used in the Qwen2.5VL
vision transformer, specifically for rotary positional embeddings and window indexing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def rot_pos_emb_numpy(
    grid_thw: npt.NDArray[np.integer[Any]], spatial_merge_size: int
) -> tuple[npt.NDArray[np.floating[Any]], int]:
    """NumPy implementation of rot_pos_emb function.

    Args:
        grid_thw: Array of shape [num_images, 3] with (t, h, w) for each image/video
        spatial_merge_size: Factor for downscaling spatial dimensions
        rotary_pos_emb_fn: Function that takes max_grid_size and returns rotary embeddings

    Returns:
        rotary_pos_emb: Rotary position embeddings of shape [seq_len, head_dim]
    """
    pos_ids = []

    for t, h, w in grid_thw:
        # Generate height position IDs: arange(h).unsqueeze(1).expand(-1, w)
        hpos_ids = np.arange(h).reshape(h, 1)  # unsqueeze(1)
        hpos_ids = np.tile(hpos_ids, (1, w))  # type: ignore  # expand(-1, w)
        hpos_ids = hpos_ids.reshape(  # type: ignore
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
        )
        hpos_ids = np.transpose(hpos_ids, (0, 2, 1, 3))  # type: ignore  # permute(0, 2, 1, 3)
        hpos_ids = hpos_ids.flatten()  # type: ignore

        # Generate width position IDs: arange(w).unsqueeze(0).expand(h, -1)
        wpos_ids = np.arange(w).reshape(1, w)  # unsqueeze(0)
        wpos_ids = np.tile(wpos_ids, (h, 1))  # type: ignore  # expand(h, -1)
        wpos_ids = wpos_ids.reshape(  # type: ignore
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
        )
        wpos_ids = np.transpose(wpos_ids, (0, 2, 1, 3))  # type: ignore  # permute(0, 2, 1, 3)
        wpos_ids = wpos_ids.flatten()  # type: ignore

        # Stack and repeat for temporal dimension
        pos_ids_hw = np.stack([hpos_ids, wpos_ids], axis=-1)
        pos_ids_t = np.repeat(
            pos_ids_hw, t, axis=0
        )  # repeat(t, 1) -> repeat along axis 0
        pos_ids.append(pos_ids_t)

    # Concatenate all position IDs
    pos_ids = np.concatenate(pos_ids, axis=0)
    max_grid_size = grid_thw[:, 1:].max()

    return pos_ids, max_grid_size


def get_window_index_numpy(
    grid_thw: npt.NDArray[np.integer[Any]],
    window_size: int,
    spatial_merge_size: int,
    patch_size: int,
) -> tuple[npt.NDArray[np.integer[Any]], npt.NDArray[np.integer[Any]]]:
    """NumPy implementation of get_window_index function.

    Args:
        grid_thw: Array of shape [num_images, 3] with (t, h, w) for each image/video
        window_size: Size of attention window
        spatial_merge_size: Factor for downscaling spatial dimensions
        patch_size: Size of each patch in the image
        spatial_merge_unit: Number of patches merged together

    Returns:
        Tuple of (window_index, cu_window_seqlens)
    """
    window_index = []
    cu_window_seqlens = [0]
    window_index_id = 0
    vit_merger_window_size = window_size // spatial_merge_size // patch_size

    spatial_merge_unit = spatial_merge_size * spatial_merge_size

    for grid_t, grid_h, grid_w in grid_thw:
        llm_grid_h = grid_h // spatial_merge_size
        llm_grid_w = grid_w // spatial_merge_size

        # Create index tensor: arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
        index = np.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
            grid_t, llm_grid_h, llm_grid_w
        )

        # Calculate padding
        pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
        pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
        num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
        num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size

        # Pad the index tensor: F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
        index_padded = np.pad(
            index,
            ((0, 0), (0, pad_h), (0, pad_w)),
            mode="constant",
            constant_values=-100,
        )

        # Reshape into windows
        index_padded = index_padded.reshape(
            grid_t,
            num_windows_h,
            vit_merger_window_size,
            num_windows_w,
            vit_merger_window_size,
        )

        # Permute: permute(0, 1, 3, 2, 4)
        index_padded = np.transpose(index_padded, (0, 1, 3, 2, 4))

        # Reshape to group windows
        index_padded = index_padded.reshape(
            grid_t,
            num_windows_h * num_windows_w,
            vit_merger_window_size,
            vit_merger_window_size,
        )

        # Calculate sequence lengths: (index_padded != -100).sum([2, 3]).reshape(-1)
        seqlens = (index_padded != -100).sum(axis=(2, 3)).reshape(-1)

        # Flatten and filter
        index_padded = index_padded.reshape(-1)
        index_new = index_padded[index_padded != -100]

        # Add to window index with offset
        window_index.append(index_new + window_index_id)

        # Update cumulative sequence lengths
        cu_seqlens_tmp = (
            np.cumsum(seqlens) * spatial_merge_unit + cu_window_seqlens[-1]
        )
        cu_window_seqlens.extend(cu_seqlens_tmp.tolist())

        # Update window index ID
        window_index_id += int(grid_t * llm_grid_h * llm_grid_w)

    # Concatenate all window indices
    window_index = np.concatenate(window_index, axis=0)

    return window_index, np.array(cu_window_seqlens)


def get_cu_window_seqlens_numpy(
    cu_window_seqlens: npt.NDArray[np.integer[Any]],
) -> npt.NDArray[np.integer[Any]]:
    """NumPy implementation to get cumulative window sequence lengths.

    Removes duplicate consecutive values from the cumulative window sequence lengths.

    Args:
        cu_window_seqlens: List of cumulative window sequence lengths.

    Returns:
        np.ndarray: Unique consecutive cumulative window sequence lengths.
    """

    # NumPy equivalent of torch.unique_consecutive
    # Find positions where consecutive elements are different
    if len(cu_window_seqlens) <= 1:
        return cu_window_seqlens

    # Create mask for elements that are different from the next element, plus the last element
    mask = np.ones(len(cu_window_seqlens), dtype=bool)
    mask[:-1] = cu_window_seqlens[:-1] != cu_window_seqlens[1:]

    return cu_window_seqlens[mask]


def get_cu_seqlens_numpy(
    grid_thw: npt.NDArray[np.integer[Any]],
) -> npt.NDArray[np.integer[Any]]:
    """NumPy implementation to compute cumulative sequence lengths.

    Computes cumulative sequence lengths for each image/video based on their
    temporal, height, and width dimensions. For each image with dimensions (t, h, w),
    the sequence length is h * w, repeated t times, then accumulated.

    Args:
        grid_thw: Array of shape [num_images, 3] with (t, h, w) for each image/video.

    Returns:
        np.ndarray: Cumulative sequence lengths with a leading zero, of shape [total_sequences + 1].
    """

    cu_seqlens = np.repeat(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(
        axis=0,
    )
    cu_seqlens = np.pad(cu_seqlens, (1, 0), mode="constant", constant_values=0)

    return cu_seqlens
