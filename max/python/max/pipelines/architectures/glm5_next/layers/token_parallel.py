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

"""Running the feed-forward block on one rank's share of the tokens.

Under tensor-parallel attention every rank holds the whole sequence, so the
expert-parallel MoE dispatched every token from every rank while its dispatch
buffers were sized by dividing the batch by the tensor-parallel degree. A rank
allocated for a token count nobody reserved.

The MoE is per-token, so the fix is to give each rank a distinct slice of the
rows, run the experts on that, and gather the outputs back. The residual is
left at its full width: it crosses a subgraph boundary every layer, and a
sharded token axis there is an *expression* over ``total_seq_len``, which the
graph verifier rejects with "Expressions are not (currently) allowed in
function signatures". Keeping the split inside the sublayer keeps the
expression inside the subgraph body where it is legal.

The partition is ``[T*r // G, T*(r+1) // G)``: contiguous, covering, and every
rank lands on floor or ceil of ``T / G``. So ``ceildiv(T, G)`` remains the
per-rank upper bound, which is exactly what
:func:`~max.nn.comm.ep.ep_config.calculate_ep_max_tokens_per_rank` already
returns -- the shared helper needs no architecture-specific override.
:func:`~max.graph.ops.allgather` concatenates in device order, which inverts
the split exactly.
"""

from __future__ import annotations

from collections.abc import Sequence

from max.dtype import DType
from max.graph import BufferValue, TensorValue, ops

__all__ = ["gather_token_shards", "token_bound", "token_shard"]


def token_bound(
    total_tokens: TensorValue, index: int, degree: int
) -> TensorValue:
    """Returns ``total_tokens * index // degree`` as an int64 CPU scalar.

    The partition boundary for rank ``index``. Computed as a runtime value
    rather than from ``x.shape[0]``: :func:`~max.graph.ops.slice_tensor`
    converts a bound through a path that takes a tensor or an ``int``, not a
    ``Dim``, so an algebraic expression over the shape cannot be a bound.
    """
    scaled = total_tokens * index
    # `//` promotes to float64, so the floor comes back through a cast.
    return (scaled // degree).cast(DType.int64)


def token_shard(
    x: TensorValue, rank: int, degree: int, total_tokens: TensorValue
) -> TensorValue:
    """Returns this rank\'s contiguous slice of ``x``\'s token axis.

    A local slice rather than a collective: every rank already holds the whole
    tensor, so there is nothing to communicate. The result\'s row count is a
    freshly *named* dimension rather than an expression over the input\'s, the
    same way :func:`~max.nn.data_parallelism.split_batch` names its splits --
    a named dimension is legal in the subgraph signature this crosses, and an
    expression is not.

    Args:
        x: ``[total_tokens, ...]``, identical on every rank.
        rank: This device\'s index within the tensor-parallel group.
        degree: Size of that group.
        total_tokens: The row count as an int64 scalar, on CPU.

    Returns:
        ``[shard_tokens, ...]``.
    """
    start = token_bound(total_tokens, rank, degree)
    stop = token_bound(total_tokens, rank + 1, degree)
    return ops.slice_tensor(
        x, [(slice(start, stop), f"moe_token_shard_{rank}"), ...]
    )


def gather_token_shards(
    shards: Sequence[TensorValue],
    like: Sequence[TensorValue],
    signal_buffers: Sequence[BufferValue],
) -> list[TensorValue]:
    """Reassembles the per-rank shards into the full sequence on every rank.

    Args:
        shards: ``[shard_tokens, ...]`` per device, in device order.
        like: The tensors :func:`token_shard` was given, used to restore the
            token dimension\'s original symbol.
        signal_buffers: Collective synchronization buffers, one per device.

    Returns:
        ``[total_tokens, ...]`` per device.
    """
    gathered = ops.allgather(shards, signal_buffers, axis=0)
    # The gather\'s own row count is the sum of the shards\' named dimensions.
    # Rebinding to the input\'s dimension is what keeps that sum from reaching
    # the residual, and what lets the write-back see matching shapes -- the
    # same reason DeepSeek-V3.2 rebinds after its own scatter/gather trip.
    return [
        ops.rebind(g, ref.shape) for g, ref in zip(gathered, like, strict=True)
    ]
