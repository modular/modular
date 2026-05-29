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

"""TensorLayout-based rules for miscellaneous ops."""

from __future__ import annotations

import builtins

from max.experimental.sharding.mappings import PlacementMapping
from max.experimental.sharding.placements import Partial, Placement, Sharded
from max.experimental.sharding.types import TensorLayout
from max.graph.dim import DimLike
from max.graph.shape import ShapeLike

from ._common import RuleSignature, resolve_partials_mapping


def _reject_axes(
    placements: tuple[Placement, ...],
    bad: set[int],
    op_name: str,
) -> None:
    for p in placements:
        if isinstance(p, Sharded) and p.axis in bad:
            raise ValueError(
                f"{op_name}: cannot be sharded along axis {p.axis}."
            )


def band_part_rule(
    x: TensorLayout,
    num_lower: int | None = None,
    num_upper: int | None = None,
    exclude: bool = False,
) -> RuleSignature:
    """band_part is linear. Rejects sharding on last 2 axes (matrix dims)."""
    bad = {builtins.max(0, x.rank - 2), x.rank - 1}
    _reject_axes(x.mapping.to_placements(), bad, "band_part")
    return (x.mapping, num_lower, num_upper, exclude), (x.mapping,)


def fold_rule(
    input: TensorLayout,
    output_size: tuple[DimLike, DimLike],
    kernel_size: tuple[DimLike, DimLike],
    stride: int | tuple[int, int] = 1,
    dilation: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
) -> RuleSignature:
    """Fold is linear. Rejects sharding on axes 1, 2."""
    _reject_axes(input.mapping.to_placements(), {1, 2}, "fold")
    return (
        (input.mapping, output_size, kernel_size, stride, dilation, padding),
        (input.mapping,),
    )


def as_interleaved_complex_rule(x: TensorLayout) -> RuleSignature:
    """Rejects sharding on last axis. Non-linear."""
    s = resolve_partials_mapping(x.mapping)
    _reject_axes(s.to_placements(), {x.rank - 1}, "as_interleaved_complex")
    return (s,), (s,)


def resize_rule(
    input: TensorLayout, size: ShapeLike, *extra: object
) -> RuleSignature:
    """Resize is linear. Only batch-dim sharding allowed.

    Shared by ``ops.resize``, ``ops.resize_nearest``, ``ops.resize_bicubic``;
    their trailing args differ, hence ``*extra``.
    """
    _reject_axes(
        input.mapping.to_placements(),
        set(builtins.range(1, input.rank)),
        "resize",
    )
    return (input.mapping, size, *extra), (input.mapping,)


def resize_linear_rule(
    input: TensorLayout,
    size: ShapeLike,
    coordinate_transform_mode: int = 0,
    antialias: bool = False,
) -> RuleSignature:
    """Resize_linear is linear. Only batch-dim sharding allowed."""
    _reject_axes(
        input.mapping.to_placements(),
        set(builtins.range(1, input.rank)),
        "resize_linear",
    )
    return (
        input.mapping,
        size,
        coordinate_transform_mode,
        antialias,
    ), (input.mapping,)


def irfft_rule(input_tensor: TensorLayout, *extra: object) -> RuleSignature:
    """Sharding rule for irfft. Rejects sharding on the last (FFT) axis."""
    _reject_axes(
        input_tensor.mapping.to_placements(),
        {input_tensor.rank - 1},
        "irfft",
    )
    return (input_tensor.mapping, *extra), (input_tensor.mapping,)


def reject_distributed_rule(
    x: TensorLayout, op_name: str = ""
) -> RuleSignature:
    """Reject any op that does not support distributed tensors."""
    mesh = x.mapping.mesh
    if mesh.num_devices > 1:
        raise ValueError(f"{op_name}: distributed tensors are not supported.")
    return (x.mapping,), (x.mapping,)


# ═══════════════════════════════════════════════════════════════════════
#  Control flow rules
# ═══════════════════════════════════════════════════════════════════════


def cond_rule(
    pred: TensorLayout,
    out_types: object,
    then_fn: object,
    else_fn: object,
) -> RuleSignature:
    """cond: predicate must be fully Replicated."""
    for p in pred.mapping.to_placements():
        if isinstance(p, Sharded):
            raise ValueError(
                "cond: predicate must be Replicated. "
                "A Sharded predicate would cause different branches "
                "on different devices."
            )
        if isinstance(p, Partial):
            raise ValueError(
                "cond: predicate must be Replicated. "
                "A Partial predicate is undefined."
            )
    mesh = pred.mapping.mesh
    out_m = PlacementMapping(mesh, pred.mapping.to_placements())
    return (pred.mapping, out_types, then_fn, else_fn), (out_m,)


def while_loop_rule(
    initial_values: object,
    predicate: object,
    body: object,
) -> RuleSignature:
    """while_loop: rejects distributed initial values (for now)."""
    if isinstance(initial_values, TensorLayout):
        if initial_values.mapping.mesh.num_devices > 1:
            raise ValueError(
                "while_loop: distributed tensors are not supported as "
                "initial values."
            )
        return (
            (initial_values.mapping, predicate, body),
            (initial_values.mapping,),
        )
    if isinstance(initial_values, (list, tuple)):
        suggested = []
        out_mappings = []
        for v in initial_values:
            if isinstance(v, TensorLayout):
                if v.mapping.mesh.num_devices > 1:
                    raise ValueError(
                        "while_loop: distributed tensors are not supported as "
                        "initial values."
                    )
                suggested.append(v.mapping)
                out_mappings.append(v.mapping)
            else:
                suggested.append(v)
        return (
            (type(initial_values)(suggested), predicate, body),
            tuple(out_mappings),
        )
    return (initial_values, predicate, body), ()


def buffer_store_rule(
    destination: TensorLayout,
    source: TensorLayout,
) -> RuleSignature:
    """buffer_store: both tensors must have same placements."""
    return (destination.mapping, source.mapping), (destination.mapping,)


def buffer_store_slice_rule(
    destination: TensorLayout,
    source: TensorLayout,
    indices: object,
) -> RuleSignature:
    """buffer_store_slice: both tensors must have same placements."""
    return (destination.mapping, source.mapping, indices), (
        destination.mapping,
    )
