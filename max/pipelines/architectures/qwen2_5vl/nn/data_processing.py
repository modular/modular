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


def mrope_pos_ids_3d(
    grid_thw: npt.NDArray[np.integer[Any]], spatial_merge_size: int
) -> npt.NDArray[np.integer[Any]]:
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
    grid_thw: npt.NDArray[np.integer[Any]],
    window_size: int,
    spatial_merge_size: int,
    patch_size: int,
    spatial_merge_unit: int,
) -> tuple[npt.NDArray[np.integer[Any]], npt.NDArray[np.integer[Any]]]:
    """Computes the indices of patches within windows, handles padding for uneven divisions,
    and computes cumulative window sequence lengths for the attention mechanism.

    Args:

        grid_thw: spatial and temporal coordinated of patches in images.
        window_size: Size of attention window. E.g.  window size of 112x112 corresponds to 8x8 patches of size 14.
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
    grid_thw: npt.NDArray[np.integer[Any]],
    seq_length: int,
    cu_win_seqlens: npt.NDArray[np.integer[Any]],
) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
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


def get_rope_index(
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    tokens_per_second: int,
    input_ids: npt.NDArray[np.integer[Any]],
    image_grid_thw: npt.NDArray[np.integer[Any]] | None = None,
    video_grid_thw: npt.NDArray[np.integer[Any]] | None = None,
    second_per_grid_ts: npt.NDArray[np.floating[Any]] | None = None,
    attention_mask: npt.NDArray[np.floating[Any]] | None = None,
) -> tuple[npt.NDArray[np.integer[Any]], npt.NDArray[np.integer[Any]]]:
    """Calculates position ids for 3D rotary position embeddings (RoPE) for vLLM.

    It determines the temporal, height, and width position indices for vision tokens
    and assigns linear position indices for text tokens.

    Image tokens (image_token_id) => Assigned 3D RoPE.
    Video tokens (video_token_id) => Assigned 3D RoPE with an additional time-step scaling factor.
    Text tokens => Assigned simple sequential position IDs.

    Args:
        spatial_merge_size: Factor for downscaling spatial dimensions (height & width).
        image_token_id: Token ID indicating an image token.
        video_token_id: Token ID indicating a video token.
        vision_start_token_id: Token ID marking the start of a vision sequence.
        tokens_per_second: Defines temporal granularity of video embeddings.
        input_ids: Tensor of token indices for the input sequence.
        image_grid_thw: Shape (num_images, 3), specifying (temporal, height, width) grid for each image.
        video_grid_thw: Shape (num_videos, 3), specifying (temporal, height, width) grid for each video.
        second_per_grid_ts: Time interval (in seconds) for each video grid along the temporal axis.
        attention_mask: Mask indicating valid tokens (1 = valid, 0 = ignored).

    Returns:
        position_ids: A (3, batch_size, sequence_length) tensor with (temporal, height, width) indices.
        mrope_position_deltas: A (batch_size, 1) tensor storing the positioning shift for text tokens.
    """
    mrope_position_deltas = []  # Tracks offsets for adjusting text token positions.
    if input_ids is not None and (
        image_grid_thw is not None or video_grid_thw is not None
    ):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = np.ones_like(total_input_ids)

        # Initialize position_ids
        position_ids = np.ones(
            (3, input_ids.shape[0], input_ids.shape[1]), dtype=np.int64
        )

        image_index, video_index = 0, 0

        for i, input_ids_row in enumerate(total_input_ids):
            # Extract valid input_ids using the attention_mask.
            input_ids_row = input_ids_row[attention_mask[i] == 1]
            vision_start_indices = np.where(
                input_ids_row == vision_start_token_id
            )[0]
            vision_tokens = input_ids_row[vision_start_indices + 1]

            image_nums = np.sum(vision_tokens == image_token_id)
            video_nums = np.sum(vision_tokens == video_token_id)

            input_tokens = input_ids_row.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums

            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1

                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1

                # For each image token: Set temporal position to 0 and extract H,W from grid.
                if ed_image < ed_video:
                    assert image_grid_thw is not None
                    t, h, w = image_grid_thw[image_index]
                    second_per_grid_t = 0.0
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    assert video_grid_thw is not None
                    t, h, w = video_grid_thw[video_index]
                    # Compute temporal intervals using tokens_per_second * second_per_grid_ts
                    second_per_grid_t = (
                        second_per_grid_ts[video_index]
                        if second_per_grid_ts is not None
                        else 1.0
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video

                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t,
                    h // spatial_merge_size,
                    w // spatial_merge_size,
                )
                text_len = ed - st
                st_idx = (
                    (llm_pos_ids_list[-1].max() + 1) if llm_pos_ids_list else 0
                )

                llm_pos_ids_list.append(
                    np.arange(text_len).reshape(1, -1).repeat(3, axis=0)
                    + st_idx
                )

                range_tensor = np.arange(llm_grid_t).reshape(-1, 1)
                expanded_range = np.tile(
                    range_tensor, (1, llm_grid_h * llm_grid_w)
                )
                time_tensor = (
                    expanded_range * second_per_grid_t * tokens_per_second
                )
                t_index = time_tensor.astype(np.int64).flatten()

                h_index = np.tile(
                    np.arange(llm_grid_h).reshape(1, -1, 1),
                    (llm_grid_t, 1, llm_grid_w),
                ).flatten()
                w_index = np.tile(
                    np.arange(llm_grid_w).reshape(1, 1, -1),
                    (llm_grid_t, llm_grid_h, 1),
                ).flatten()

                llm_pos_ids_list.append(
                    np.vstack([t_index, h_index, w_index]) + text_len + st_idx
                )
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            # Start text positions after the last vision position + 1
            if st < len(input_tokens):
                st_idx = (
                    (llm_pos_ids_list[-1].max() + 1) if llm_pos_ids_list else 0
                )
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    np.arange(text_len).reshape(1, -1).repeat(3, axis=0)
                    + st_idx
                )

            llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(
                3, -1
            )
            position_ids[:, i, attention_mask[i] == 1] = llm_positions
            mrope_position_deltas.append(
                llm_positions.max() + 1 - len(total_input_ids[i])
            )

        mrope_position_deltas_array = np.array(mrope_position_deltas).reshape(
            -1, 1
        )
        return position_ids, mrope_position_deltas_array
    else:
        if attention_mask is not None:
            position_ids = np.cumsum(attention_mask, axis=-1) - 1
            position_ids[attention_mask == 0] = 1
            position_ids = np.tile(position_ids[np.newaxis, ...], (3, 1, 1))
            max_position_ids = position_ids.max(axis=1, keepdims=True).max(
                axis=-1, keepdims=True
            )
            mrope_position_deltas_array = (
                max_position_ids + 1 - attention_mask.shape[-1]
            )
        else:
            position_ids = np.tile(
                np.arange(input_ids.shape[1])[np.newaxis, np.newaxis, :],
                (3, input_ids.shape[0], 1),
            )
            mrope_position_deltas_array = np.zeros(
                (input_ids.shape[0], 1), dtype=np.int64
            )

        return position_ids, mrope_position_deltas_array
