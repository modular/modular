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
"""The indexer's selection width and its tail-seed replay inputs."""

from __future__ import annotations

from typing import NamedTuple

from max.tensor import Tensor

__all__ = [
    "KpoolTailReplayInputs",
    "indexer_selection_width",
]


def indexer_selection_width(
    index_topk: int, index_kpool: int, always_select_tail: bool
) -> int:
    """Returns the width of one token's selected rows.

    ``index_topk + index_kpool - 1`` when the k-pool tail is always selected,
    ``index_topk`` otherwise.
    """
    if not always_select_tail or index_kpool == 1:
        return index_topk
    return index_topk + index_kpool - 1


class KpoolTailReplayInputs(NamedTuple):
    """One pooled indexer call's inputs to the tail-ring seed.

    A speculative verify keeps the ring at its pre-verify members and re-runs
    :func:`~max.nn._graph_ops.mla_kpool_seed_tail` over the accepted rows once
    acceptance is known. The ring is slot-indexed modulo ``kpool``, so a seed
    of a rejected draft can overwrite a pending member the next step closes
    its pool with, and no length pointer undoes that.
    """

    k: Tensor
    """``[total_tokens, index_head_dim]`` layer-normed keys."""
    gate: Tensor
    """``[total_tokens, index_head_dim]`` compression-gate scores."""
    tail: Tensor
    """The tail ring, every sparse layer's rows."""
    slot_idx: Tensor
    """``[batch]`` ring rows of this layer."""
    cache_lengths: Tensor
    """``[batch]`` cached-prefix length the call ran at."""
    kpool: int
