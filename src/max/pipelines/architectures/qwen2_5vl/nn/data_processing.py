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


import numpy as np


def mrope_pos_ids_3d(
    grid_thw: np.ndarray, spatial_merge_size: int
) -> np.ndarray:
    """Calculate the 3D rope index based on image and video's temporal, height, and width in LLM using NumPy."""
    pos_ids = []

    for t, h, w in grid_thw:
        hpos_ids = np.arange(h).reshape(h, 1)
        hpos_ids = np.tile(hpos_ids, (1, w))
        hpos_ids = hpos_ids.reshape(
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
        )
        hpos_ids = np.transpose(hpos_ids, (0, 2, 1, 3)).flatten()

        wpos_ids = np.arange(w).reshape(1, w)
        wpos_ids = np.tile(wpos_ids, (h, 1))
        wpos_ids = wpos_ids.reshape(
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
        )
        wpos_ids = np.transpose(wpos_ids, (0, 2, 1, 3)).flatten()

        pos_ids.append(
            np.stack([hpos_ids, wpos_ids], axis=-1).repeat(t, axis=0)
        )
    return np.concatenate(pos_ids, axis=0)


def get_window_index(
    grid_thw: np.ndarray,
    window_size: int,
    spatial_merge_size: int,
    patch_size: int,
    spatial_merge_unit: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes the indices of patches within windows, handles padding for uneven divisions,
    and computes cumulative window sequence lengths for the attention mechanism.

    Args:

        grid_thw: spatial and temporal coordinated of patches in images.
        window_size: Size of attention window.
        spatial_merge_size: number of patches to merge together.
        patch_size: Size of each patch in the image.
        spatial_merge_unit:
    """
    window_index = []
    cu_window_seqlens = [0]
    window_index_id = 0
    padding_value = -100
    # The effective window size after considering the spatial merge size and patch size.
    vit_merger_window_size = window_size // spatial_merge_size // patch_size

    for grid_t, grid_h, grid_w in grid_thw:
        # Computes the number of patches along height and width after spatial merging.
        # The merged height and width after applying spatial merging.
        llm_grid_h, llm_grid_w = (
            grid_h // spatial_merge_size,
            grid_w // spatial_merge_size,
        )
        # Generates a [grid_t, llm_grid_h, llm_grid_w] tensor of indices for all the patches.
        index = np.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
            grid_t, llm_grid_h, llm_grid_w
        )
        # No. of patches needed to make the grid divisible by vit_merger_window_size.
        pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
        pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
        num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
        num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size

        index_padded = np.pad(
            index,
            ((0, 0), (0, pad_h), (0, pad_w)),
            constant_values=padding_value,
        )
        # Reshape into a 5D tensor of windows of shape [grid_t, num_windows_h, vit_merger_window_size, num_windows_w, vit_merger_window_size].
        index_padded = index_padded.reshape(
            grid_t,
            num_windows_h,
            vit_merger_window_size,
            num_windows_w,
            vit_merger_window_size,
        )
        # Flatten the window dimensions into a 3D tensor of shape [grid_t, num_windows_h * num_windows_w, vit_merger_window_size, vit_merger_window_size].
        index_padded = np.transpose(index_padded, (0, 1, 3, 2, 4)).reshape(
            grid_t,
            num_windows_h * num_windows_w,
            vit_merger_window_size,
            vit_merger_window_size,
        )
        # Counts how many valid patches (i.e., not equal to -100) are present in each window.
        seqlens = np.sum(index_padded != padding_value, axis=(2, 3)).reshape(-1)
        index_padded = index_padded.reshape(-1)
        # Removes the padding (-100 values) and keeps only the valid indices
        index_new = index_padded[index_padded != padding_value]
        # Add valid indices with an offset (window_index_id) to ensure indices are unique across batches.
        window_index.append(np.add(index_new, window_index_id))
        cu_seqlens_tmp = (
            np.cumsum(seqlens) * spatial_merge_unit + cu_window_seqlens[-1]
        )
        cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
        window_index_id += grid_t * llm_grid_h * llm_grid_w

    return np.concatenate(window_index, axis=0), np.array(cu_window_seqlens)
