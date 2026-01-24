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

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .context import Qwen2_5VLTextAndVisionContext

if TYPE_CHECKING:
    from ..qwen3vl_moe.context import Qwen3VLTextAndVisionContext


def compute_multimodal_merge_indices(
    batch: Sequence[
        Qwen2_5VLTextAndVisionContext | Qwen3VLTextAndVisionContext
    ],
) -> npt.NDArray[np.int32]:
    """Compute indices for a batch of VLM contexts to use in merge_multimodal_embeddings.

    Args:
        batch: Sequence of VLM contexts.

    Returns:
        npt.NDArray[np.int32]: Multimodal merge indices, some of which may be negative.
    """
    # Calculate sentinel OOB index value by finding the largest negative int32 value.
    oob_idx = np.iinfo(np.int32).min

    # Collect indices and offsets.
    indices_list = []
    total_active_tokens = 0

    for ctx in batch:
        if ctx.needs_vision_encoding:
            # This logic is quite tricky but is required for VLM prefix caching.
            # In the current approach, we run image decoding on all images.
            # We then select the rows of the image embeddings we want to use.
            # This may not be all of the rows in the event of a prefix cache
            # hit. This is done via a multimodal merge operation which filters
            # out negative indices.

            # First, get the pre-computed indices of where the image placeholder
            # tokens are in the prompt. This is populated by tokenizer.
            # eg: prompt = [0, 1, 2, 3, IMG, IMG, IMG, IMG, 8, 9, IMG, IMG]
            #    indices = [4, 5, 6, 7, 10, 11]
            indices = ctx.image_token_indices

            # Subtract all of the indices by the start_idx to get offsets
            # relative to the ragged next_tokens input sequence.
            # eg: start_idx = 6
            #     indices = [-2, -1, 0, 1, 4, 5]
            indices = indices - ctx.tokens.processed_length

            # Set any negative indices to -1, which means that they are ignored.
            # Bump remaining by accumulated value for the batch.
            indices_filtered = [
                idx + total_active_tokens if idx >= 0 else oob_idx
                for idx in indices.tolist()
            ]

            # Final scatter indices assuming the batch has 10 image tokens so far.
            # eg: indices_filtered = [-999, -999, 10, 11, 14, 15]
            #     This means that we will copy 4 image embeddings to the rows
            #     10-11 and 14-15 of the text embeddings.
            indices_list.append(indices_filtered)

        total_active_tokens += ctx.tokens.active_length

    # scatter_nd_skip_oob_indices uses int32 indices.
    if indices_list:
        return np.concatenate(indices_list, dtype=np.int32)
    else:
        return np.array([], dtype=np.int32)
