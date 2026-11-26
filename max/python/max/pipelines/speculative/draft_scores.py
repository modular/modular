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
"""The head-free draft score: the top token's log-odds per position."""

from __future__ import annotations

import functools
import operator

from max import functional as F
from max.dtype import DType
from max.graph import Dim, DimLike, TensorValue, ops
from max.nn._graph_ops import (
    top_token_logit as _fused_top_token_logit,
)
from max.nn._graph_ops import (
    top_token_pieces as _fused_top_token_pieces,
)
from max.tensor import Tensor
from max.tensor.sharding.rules import reduce_rule

# Wrapped so eager and mesh tensors dispatch per device; the last-axis
# reduction rule keeps a row-sharded input row-sharded.
_fused_rows = F.functional(_fused_top_token_logit, rule=reduce_rule)
_fused_pieces = F.functional(_fused_top_token_pieces, rule=reduce_rule)


__all__ = [
    "combine_shard_scores",
    "top_token_logit",
    "top_token_pieces",
]


def _flat_rows(
    logits: TensorValue | Tensor, lead: list[Dim]
) -> TensorValue | Tensor:
    """Flattens the ``lead`` axes of ``logits`` into one rows axis."""
    shape: list[DimLike] = [
        functools.reduce(operator.mul, lead, 1),
        logits.shape[-1],
    ]
    # Graph values take the graph op; F.reshape keeps a sharded Tensor's layout.
    if isinstance(logits, Tensor):
        return F.reshape(logits, shape)
    return ops.reshape(logits, shape)


def top_token_logit(logits: TensorValue | Tensor) -> Tensor:
    """Scores one position's sampling logits by the top token's log-odds.

    The top token's own mass is excluded from the denominator so
    hyper-peaked positions keep their deficit at full float32
    precision. One fused kernel launch reads the logits once.

    Args:
        logits: One position's sampling logits, graph values and
            eager tensors both serve. The last axis is reduced,
            every other axis is kept.

    Returns:
        The top token's log-odds per position, float32, the input
        with the last axis reduced and squeezed.
    """
    # The fused op takes rank-2 rows; flatten the leading axes and
    # restore them after, so any rank scores through the one kernel.
    lead = list(logits.shape[:-1])
    flat = _flat_rows(logits, lead)
    scored = _fused_rows(flat)
    return F.squeeze(F.reshape(scored, [*lead, 1]), axis=-1)


def top_token_pieces(logits: TensorValue | Tensor, start: int) -> Tensor:
    """Reduces one position's logits to its top token's allgather pieces.

    Args:
        logits: One position's sampling logits, graph values and
            eager tensors both serve. The last axis is reduced,
            every other axis is kept.
        start: The shard's start in the joined vocabulary, added to
            the local max index.

    Returns:
        The per-position pieces, float32, the input with the last
        axis replaced by the three columns ``(max, index, excluded
        mass)``.
    """
    # The fused op takes rank-2 rows; flatten the leading axes and
    # restore them after, so any rank reduces through the one kernel.
    lead = list(logits.shape[:-1])
    flat = _flat_rows(logits, lead)
    pieces = _fused_pieces(flat, start)
    return F.reshape(pieces, [*lead, 3])


def combine_shard_scores(
    pieces: Tensor,
    top_indices: Tensor,
    top_max: Tensor,
) -> Tensor:
    """Combines every position's per-shard pieces into its top log-odds.

    Args:
        pieces: ``[shards, b, K, 3]`` float32, every position's gathered
            columns: the shard maxima, their global indices, and each
            shard's excluded mass.
        top_indices: ``[b, K]`` int32, the argmax step's global top
            index per position.
        top_max: ``[b, K]`` float32, the global maximum the argmax
            step already reduced over the shards.

    Returns:
        The top token's log-odds per position, ``[b, K]`` float32.
    """
    # The shard axis stays where the allgather stacked it, so the
    # per-position math runs batched in the same op order.
    maxima = pieces[:, :, :, 0]
    indices = pieces[:, :, :, 1]
    excluded = pieces[:, :, :, 2]
    # Empty shards need a finite scale even when the whole row is masked.
    safe_top = F.where(F.is_inf(maxima), 0.0, top_max)
    rescaled = F.exp(maxima - safe_top)
    mass = F.sum(excluded * rescaled, axis=0)
    is_top_shard = F.equal(indices, F.cast(top_indices, indices.dtype))
    restored = F.sum(
        rescaled * F.cast(F.logical_not(is_top_shard), DType.float32),
        axis=0,
    )
    # The shard-axis sums keep rank; squeeze once for the [b, K] contract.
    return F.squeeze(-F.log(mass + restored), axis=0)
