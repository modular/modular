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


def mrope_pos_ids_3d(grid_thw, spatial_merge_size) -> np.ndarray:
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
