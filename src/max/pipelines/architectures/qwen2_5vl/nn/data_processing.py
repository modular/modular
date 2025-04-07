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
        window_size: Size of attention window. E.g.  window size of 112×112 corresponds to 8×8 patches of size 14.
        spatial_merge_size: number of patches to merge together.
        patch_size: Size of each patch in the image.
        spatial_merge_unit:
    """
    window_index = []
    cu_window_seqlens = [0]  # Total number of patches in prev. windows.
    window_index_id = 0
    padding_value = -100
    # The effective window size = Number of patches in each window after considering spatial merge size and patch size.
    vit_merger_window_size = window_size // spatial_merge_size // patch_size

    for grid_t, grid_h, grid_w in grid_thw:
        # Computes the number of patches along height and width after applying spatial merging.
        llm_grid_h, llm_grid_w = (
            grid_h // spatial_merge_size,
            grid_w // spatial_merge_size,
        )
        # Generates a [grid_t, llm_grid_h, llm_grid_w] tensor of indices for all the patches after merging.
        index = np.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
            grid_t, llm_grid_h, llm_grid_w
        )
        # Make no. of patches along h and w dims divisible by no. of patches per window.
        pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
        pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
        num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
        num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size

        index_padded = np.pad(
            index,
            ((0, 0), (0, pad_h), (0, pad_w)),
            constant_values=padding_value,
        )
        # Assign a window index to each patch by reshaping indieces tensor into
        # 5D tensor of windows of shape [grid_t, num_windows_h, vit_merger_window_size, num_windows_w, vit_merger_window_size].
        # from [num_patches_h, num_patches_w] to [num_windows_h, n_patches_per_window, num_windows_w, n_patches_per_window].
        index_padded = index_padded.reshape(
            grid_t,
            num_windows_h,
            vit_merger_window_size,
            num_windows_w,
            vit_merger_window_size,
        )
        # Flatten the window dimensions into a 3D tensor of shape [grid_t, total num_windows in h and w, n_patches_per_window, n_patches_per_window].
        index_padded = np.transpose(index_padded, (0, 1, 3, 2, 4)).reshape(
            grid_t,
            num_windows_h * num_windows_w,
            vit_merger_window_size,
            vit_merger_window_size,
        )
        # Counts how many valid patches (i.e., != -100) are present in each window.
        seqlens = np.sum(index_padded != padding_value, axis=(2, 3)).reshape(-1)
        index_padded = index_padded.reshape(-1)
        # Removes the padding (-100 values) and keeps only the valid indices
        index_new = index_padded[index_padded != padding_value]
        # Add valid indices with an offset of (Total number of patches in prev. batch/image) to ensure indices are unique across batches.
        window_index.append(np.add(index_new, window_index_id))
        # [0, n_patches_in_window1, n_patches_in_windows 1 & 2, ...] * n_patches_merged_together Offset by number of patches in prev images = cu_window_seqlens[-1]
        cu_seqlens_tmp = (
            np.cumsum(seqlens) * spatial_merge_unit + cu_window_seqlens[-1]
        )
        cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
        window_index_id += grid_t * llm_grid_h * llm_grid_w

    # Return a 1D tensor of length [seq_len // (window_size // spatial_merge_size // patch_size) = seq_len // 4] for Qwen2.5VL
    return np.concatenate(window_index, axis=0), np.array(cu_window_seqlens)


def generate_attention_mask(
    grid_thw: np.ndarray, seq_length: int, cu_win_seqlens: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Generate attention masks for visual tokens using seq_length and cu_seqlens.
    cu_seqlens is used when the block is in fullatt_block_indexes.

    Args:
        grid_thw: number of patches in spatial and temporal dims in images. Shape = [n_images, 3]
        seq_length: int represents total number of patches in all images and videos.
        cu_window_seqlens: cumulative window sequence lengths for the attention mechanism. Shape = [n_windows]
    """
    cu_window_seqlens = np.array(cu_win_seqlens, dtype=np.int32)
    # 1. Remove consecutive duplicates (equivalent to torch.unique_consecutive)
    _, unique_indices = np.unique(cu_window_seqlens, return_index=True)
    cu_window_seqlens = cu_window_seqlens[np.sort(unique_indices)]

    # 2. Compute cu_seqlens of cumulative window_seq_lens for full attention (used with layers not using window attention)
    window_sizes = (
        grid_thw[:, 1] * grid_thw[:, 2]
    )  # Window sizes = n_patches in each image
    repeated_sizes = np.repeat(window_sizes, grid_thw[:, 0])
    cu_seqlens = np.cumsum(repeated_sizes, dtype=np.int32)

    # 3. Pad cu_seqlens with 0 at the beginning
    cu_seqlens = np.pad(cu_seqlens, (1, 0), constant_values=0)

    # TODO(KERN-782): This fill_val should be -inf but softmax saturates with NaNs.
    fill_val = -10000.0
    attention_mask_full = np.full(
        (1, seq_length, seq_length), fill_val, dtype=np.float32
    )
    attention_mask_window = np.full(
        (1, seq_length, seq_length), fill_val, dtype=np.float32
    )

    for i in range(1, len(cu_seqlens)):
        attention_mask_full[
            ...,
            cu_seqlens[i - 1] : cu_seqlens[i],
            cu_seqlens[i - 1] : cu_seqlens[i],
        ] = 0

    for i in range(1, len(cu_window_seqlens)):
        attention_mask_window[
            ...,
            cu_window_seqlens[i - 1] : cu_window_seqlens[i],
            cu_window_seqlens[i - 1] : cu_window_seqlens[i],
        ] = 0
    return attention_mask_full, attention_mask_window
