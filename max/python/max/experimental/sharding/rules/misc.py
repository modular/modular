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

"""Placement rules for miscellaneous ops (resize, irfft, qmatmul, scatter_nd, ...)."""

from __future__ import annotations

from typing import Any

from max.experimental.sharding.placements import Sharded
from max.experimental.sharding.types import TensorLayout
from max.graph.dim import DimLike

from ..action import AxisAssignment, replicated_rows
from ..cost import P, R
from .elementwise import linear_unary_rule


def _block_axes_rows(
    x: TensorLayout, blocked: set[int]
) -> list[AxisAssignment]:
    """Linear-block helper: shard any non-blocked axis; Partial pass-through."""
    rows: list[AxisAssignment] = [AxisAssignment((R,), (R,))]
    for ax in range(x.rank):
        if ax in blocked:
            continue
        rows.append(AxisAssignment((Sharded(ax),), (Sharded(ax),)))
    rows.append(AxisAssignment((P,), (P,)))
    return rows


def band_part_rule(
    x: TensorLayout, *extra: Any, **kwargs: Any
) -> list[AxisAssignment]:
    """Strategies for ``band_part``: sharding only outside the last-two matrix axes."""
    return _block_axes_rows(x, {max(0, x.rank - 2), x.rank - 1})


def fold_rule(
    input: TensorLayout, *extra: Any, **kwargs: Any
) -> list[AxisAssignment]:
    """Strategies for ``fold``: sharding only outside axes 1 and 2."""
    return _block_axes_rows(input, {1, 2})


def as_interleaved_complex_rule(
    x: TensorLayout, *extra: Any, **kwargs: Any
) -> list[AxisAssignment]:
    """Strategies for ``as_interleaved_complex``: sharding only outside the last axis."""
    return _block_axes_rows(x, {x.rank - 1})


def irfft_rule(
    input_tensor: TensorLayout, *extra: Any, **kwargs: Any
) -> list[AxisAssignment]:
    """Strategies for ``irfft``: sharding only outside the last axis."""
    return _block_axes_rows(input_tensor, {input_tensor.rank - 1})


def resize_rule(
    input: TensorLayout, *extra: Any, **kwargs: Any
) -> list[AxisAssignment]:
    """Strategies for the ``resize`` ops: sharding only on the batch axis (0)."""
    return _block_axes_rows(input, set(range(1, input.rank)))


def dequantize_rule(
    encoding: Any, quantized: TensorLayout
) -> list[AxisAssignment]:
    """Linear unary on ``quantized``; ``encoding`` is non-tensor metadata."""
    return linear_unary_rule(quantized)


def qmatmul_rule(
    encoding: Any, config: Any, lhs: TensorLayout, *rhs: TensorLayout
) -> list[AxisAssignment]:
    """Quantized matmul: every tensor Replicated."""
    return replicated_rows(lhs, *rhs)


def masked_scatter_rule(
    input: TensorLayout,
    mask: TensorLayout,
    updates: TensorLayout,
    out_dim: DimLike,
) -> list[AxisAssignment]:
    """Forces every input to Replicated (mask uses absolute positions)."""
    return replicated_rows(input, mask, updates)


def scatter_nd_rule(
    input: TensorLayout, updates: TensorLayout, indices: TensorLayout
) -> list[AxisAssignment]:
    """N-D scatter: everything Replicated."""
    return replicated_rows(input, updates, indices)


def scatter_nd_add_rule(
    input: TensorLayout, updates: TensorLayout, indices: TensorLayout
) -> list[AxisAssignment]:
    """``scatter_nd_add`` shares its strategy with :func:`scatter_nd_rule`."""
    return replicated_rows(input, updates, indices)
