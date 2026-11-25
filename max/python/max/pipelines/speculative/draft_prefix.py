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
"""Merges prompt tokens with each row's chosen prefix of draft tokens."""

__all__ = ["merge_draft_prefixes"]

from max.dtype import DType
from max.graph import Dim, TensorValue, ops
from max.nn._graph_ops import merge_ragged_tensors

from .gated_delta_rollback import _row_plan
from .ragged_token_merger import _shape_to_scalar


def merge_draft_prefixes(
    prompt_tokens: TensorValue,
    prompt_offsets: TensorValue,
    draft_tokens: TensorValue,
    verify_lengths: TensorValue,
    total: Dim,
) -> tuple[TensorValue, TensorValue]:
    """Merges prompt tokens with each row's chosen prefix of draft tokens.

    The caller's graph must name its batch dim ``batch_size``; the reused
    row plan reads that dim.

    Args:
        prompt_tokens: The prompt tokens of shape [S].
        prompt_offsets: The prompt offsets of shape [B+1], uint32.
        draft_tokens: The draft tokens of shape [B, K]; the output keeps
            this dtype.
        verify_lengths: Per-row draft counts of shape [B], int32; each in
            ``0..=K`` and their sum must equal ``total``.
        total: The packed draft count, the sum of ``verify_lengths``.

    Returns:
        A tuple of two tensors:
            - The merged tokens of shape [S + total].
            - The merged offsets of shape [B+1].
    """
    device = prompt_tokens.device
    K = _shape_to_scalar(draft_tokens.shape[1], device, dtype=DType.int64)
    flat_drafts = ops.reshape(draft_tokens, shape=(-1,))
    # The stride is the row pitch minus one: rows sit K slots apart.
    rows, draft_offsets = _row_plan(
        None,
        K - 1,
        verify_lengths.cast(DType.int64),
        flat_drafts.shape[0],
        total,
        device,
    )
    packed = ops.gather(flat_drafts, rows, axis=0)

    return merge_ragged_tensors(
        prompt_tokens,
        prompt_offsets,
        packed,
        draft_offsets.cast(DType.uint32).rebind([prompt_offsets.shape[0]]),
    )
