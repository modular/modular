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

import math

import numpy as np


def causal_attention_mask(
    original_start_pos: list[int],
    original_seq_len: list[int],
    pad_to_multiple_of: int = 1,
) -> np.ndarray:
    # Each example in the batch has a "start position", which is the length
    # of the previously encoded tokens ("context"), and a "sequence length",
    # which is the number of additional tokens to be encoded in this pass.
    #
    # "Causal attention" means that each token can "see" tokens before it,
    # as well as itself.
    # The attention layer adds the mask to the attention scores and then
    # performs a softmax, so for tokens that a given token can "see" the mask
    # wants to produce a 0, meaning to pass the attention through as normal,
    # and for tokens that can't be "seen" the mask should produce -inf, which
    # will result in them being functionally ignored after the softmax operation.
    #
    # We call the total length "post_seq_len", referring to the total context
    # length after this pass concludes.
    start_pos: np.ndarray = np.array(original_start_pos, dtype=np.int64)
    seq_len: np.ndarray = np.array(original_seq_len, dtype=np.int64)

    # Provided `pad_to_multiple_of` ensure the padded_length is cleanly divisible
    # by this multiple.
    # If max_len is 1, we are presumably in a token generation phase batch.
    # W scenario, padding from 1 -> 2, does not result in a performance gain.
    if seq_len.max() == 1:
        padded_length = 1
    else:
        padded_length = (
            math.ceil(seq_len.max() / pad_to_multiple_of) * pad_to_multiple_of
        )

    # Mask shape: for each token being generated, attend to tokens _before_ it
    # in the entire sequence including context. Pad all values to the longest
    # sequence length and total length.
    post_seq_len = (start_pos + padded_length).max()
    mask_shape = (padded_length, post_seq_len)

    # TODO(KERN-782): This should be -inf but softmax saturates with NaNs.
    fill_val = -10000.0
    fill_matrix = np.full(mask_shape, fill_val, dtype=np.float32)

    return np.stack(
        # Set diagonal to k + 1 so that tokens attend to themselves.
        [np.triu(fill_matrix, k=k + 1) for k in start_pos]
    )
