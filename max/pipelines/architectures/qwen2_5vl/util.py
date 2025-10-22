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

import numpy as np
import numpy.typing as npt
from max.pipelines.core import TextAndVisionContext


def compute_scatter_gather_indices(
    batch: Sequence[TextAndVisionContext],
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int64]]:
    """Compute scatter and gather indices for a batch of VLM contexts.

    These scatter and gather indices are used to perform a masked_scatter operation
    to merge image embeddings into the text embeddings.

    Args:
        batch: Sequence of VLM contexts.

    Returns:
        tuple[npt.NDArray[np.int32], npt.NDArray[np.int64]]: Scatter and gather indices.
    """
    # Collect indices and offsets.
    scatter_indices_list = []
    gather_indices_list = []
    image_tokens_in_active_tokens = 0
    image_tokens_in_all_tokens = 0

    for ctx in batch:
        if ctx.needs_vision_encoding:
            # This logic is quite tricky but is required for VLM prefix caching.
            # In the current approach, we run image decoding on all images.
            # We then select the rows of the image embeddings we want to use.
            # This may not be all of the rows in the event of a prefix cache
            # hit. This selection is done via a gather.
            #
            # Then we scatter those selected rows to the rows of the text
            # embeddings containing image placeholder tokens.
            #
            # This is essentially a masked_scatter operation.

            # First, get the pre-computed indices of where the image placeholder
            # tokens are in the prompt. This is populated by tokenizer.
            # eg: prompt = [0, 1, 2, 3, IMG, IMG, IMG, IMG, 8, 9]
            #    indices = [4, 5, 6, 7]
            indices = ctx.extra_model_args["image_token_indices"]

            # Subtract all of the indices by the start_idx to get offsets
            # relative to the ragged next_tokens input sequence.
            # eg: start_idx = 5
            #     indices = [-1, 0, 1, 2]
            indices = indices - ctx.start_idx

            # Filter out any indices that are negative, which means that they
            # are not included in next_tokens. Bump remaining by accumulated
            # value for the batch.
            indices_filtered = [
                idx + image_tokens_in_active_tokens
                for idx in indices.tolist()
                if idx >= 0
            ]

            # Final scatter indices assuming this is sole request in batch.
            # eg: indices_filtered = [0, 1, 2]
            #     This means that we will copy the 3 image embeddings to the
            #     rows 0-2 of the text embeddings.
            scatter_indices_list.append(indices_filtered)

            num_gathered = len(indices_filtered)
            num_skipped = indices.shape[0] - len(indices_filtered)

            image_tokens_in_all_tokens += num_skipped
            # This computes which rows of the image embeddings to gather.
            # This calculation drops the image embedding for the first IMG
            # but selects them for the next 3.
            # eg: gathered_indices = [1, 2, 3]
            gathered_indices = (
                np.arange(num_gathered, dtype=np.int64)
                + image_tokens_in_all_tokens
            )
            image_tokens_in_all_tokens += num_gathered
            gather_indices_list.append(gathered_indices)

        image_tokens_in_active_tokens += ctx.active_length

    # ops.scatter_nd uses int32 indices.
    # ops.gather_nd uses int64 indices.
    if not scatter_indices_list:
        scatter_indices = np.array([], dtype=np.int32)
        gather_indices = np.array([], dtype=np.int64)
    else:
        scatter_indices = np.concatenate(scatter_indices_list, dtype=np.int32)
        gather_indices = np.concatenate(gather_indices_list, dtype=np.int64)

    return scatter_indices, gather_indices
