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

"""Distributed reduction ops (sum, mean, softmax).

Reductions along an axis that is NOT the sharded axis are safe to run
per-shard — the placement is preserved. Reductions along the sharded
axis require cross-device communication (error by default).

Rules per mesh axis:
    Replicated, any reduce axis → Replicated
    Sharded(k), reduce axis != k → Sharded(k) (but output rank may change)
    Sharded(k), reduce axis == k → error (needs allreduce first)
    Partial, any reduce axis → Partial (reduction is linear)
"""

from __future__ import annotations

from collections.abc import Callable

from max.experimental import tensor
from max.experimental.sharding import PlacementMapping, Sharded
from max.graph import TensorValue, ops
from max.graph.value import TensorValueLike

from ._context_provider import functional
from .collectives import _has_distributed, make_distributed, to_shard_tvs


def _reduce_dispatch(
    graph_op: Callable[..., TensorValueLike],
    x: TensorValueLike,
    axis: int,
    op_name: str,
    **kwargs,
) -> TensorValueLike:
    """Generic dispatch for reduction ops with an axis argument.

    The reduction removes (or keeps with keepdim) the reduced axis.
    Sharded placements are updated accordingly.
    """
    if not _has_distributed(x):
        return graph_op(x, axis=axis, **kwargs)

    assert isinstance(x, tensor.Tensor)
    mesh = x.mesh
    ndim = len(x.shape)
    norm_axis = axis % ndim

    # Check: reducing along a sharded axis is not supported without
    # an explicit allreduce first.
    for p in x.placements:
        if isinstance(p, Sharded) and p.axis == norm_axis:
            raise ValueError(
                f"{op_name}: cannot reduce along sharded axis {norm_axis}. "
                "Call F.all_reduce_sum or F.all_gather first."
            )

    # Partial and Replicated pass through unchanged:
    #  - Partial: reduction is linear, so reduce(a₁+a₂) = reduce(a₁)+reduce(a₂).
    #    For non-linear reductions (softmax), Partial is auto-resolved by the
    #    @functional wrapper BEFORE reaching this function.
    #  - Replicated: per-shard reduction is correct as-is.
    # MAX graph ops keep the reduced dim (like keepdim=True), so no axis
    # shifting is needed.
    out_p = list(x.placements)

    shards = to_shard_tvs(x)
    results = [TensorValue(graph_op(s, axis=axis, **kwargs)) for s in shards]
    return make_distributed(results, PlacementMapping(mesh, tuple(out_p)))


def _check_axis_none_sharded(x: TensorValueLike, op_name: str) -> None:
    """Error if axis=None on a tensor with any Sharded placement.

    axis=None flattens and reduces all elements.  This is only valid
    when every shard holds the same or additively-complete data:
      - Replicated: all shards identical → per-shard flatten+reduce is correct.
      - Partial: sum/mean are linear → per-shard result stays Partial.
      - Sharded: each shard holds only a slice → per-shard flatten+reduce
        silently gives the wrong answer.  Error out.
    """
    if isinstance(x, tensor.Tensor) and x.is_distributed:
        for p in x.placements:
            if isinstance(p, Sharded):
                raise ValueError(
                    f"{op_name}: axis=None is not supported for tensors "
                    f"with Sharded placement (found Sharded({p.axis})). "
                    f"Call F.all_gather first, or reduce along a "
                    f"specific axis."
                )


def _flatten_for_reduce(x: TensorValueLike) -> TensorValue:
    """Get a flat TensorValue for axis=None reduction.

    For distributed tensors (Replicated/Partial only — Sharded is rejected
    by ``_check_axis_none_sharded`` before this is called), uses shard 0.
    """
    if isinstance(x, tensor.Tensor) and x.is_distributed:
        tv = x.local_shards[0].__tensorvalue__()
    else:
        tv = TensorValue(x)
    return tv.reshape([-1])


@functional(linear=True)
def sum(x: TensorValueLike, axis: int | None = -1) -> TensorValueLike:
    """Distributed sum. If ``axis`` is None, flattens and reduces all."""
    if axis is None:
        _check_axis_none_sharded(x, "sum")
        return ops.sum(_flatten_for_reduce(x), axis=0)
    return _reduce_dispatch(ops.sum, x, axis, "sum")


@functional(linear=True)
def mean(x: TensorValueLike, axis: int | None = -1) -> TensorValueLike:
    """Distributed mean. If ``axis`` is None, flattens and reduces all."""
    if axis is None:
        _check_axis_none_sharded(x, "mean")
        return ops.mean(_flatten_for_reduce(x), axis=0)
    return _reduce_dispatch(ops.mean, x, axis, "mean")


@functional
def softmax(x: TensorValueLike, axis: int = -1) -> TensorValueLike:
    """Distributed softmax along an axis, preserving placement."""
    return _reduce_dispatch(ops.softmax, x, axis, "softmax")
