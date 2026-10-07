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

"""Placement rules for shape-family ops (``reshape``, ``transpose``, ``split``, ``stack``, ``gather``, ...)."""

from __future__ import annotations

import builtins
import functools
import operator
from collections.abc import Iterable, Sequence
from typing import Any

from max.experimental.sharding import (
    Placement,
    Sharded,
    ShardingError,
    Unknown,
)
from max.experimental.sharding.per_shard_dim import (
    global_dim,
    global_shape,
    is_per_shard_dim,
    local_shape_at,
    make_per_shard_dim,
)
from max.experimental.sharding.types import TensorLayout
from max.graph.dim import Dim, DimLike, StaticDim
from max.graph.ops.slice_tensor import SliceIndex, SliceIndices
from max.graph.shape import Shape

from ..action import AxisAssignment, pass_through_rows
from ..cost import P, R

# ── split / slice helpers (only consumers of these are split_rule / slice_tensor_rule) ──


def _expand_ellipsis(
    indices: Sequence[SliceIndex], ndim: int
) -> list[SliceIndex]:
    # ``None`` inserts an output axis without consuming an input one, so it
    # does not count against the input rank the ellipsis has to fill. Mirrors
    # ``ops.slice_tensor``'s own ``len - ellipsis - count(None)``.
    n_explicit = builtins.sum(
        1 for i in indices if i is not Ellipsis and i is not None
    )
    out: list[SliceIndex] = []
    for idx in indices:
        if idx is Ellipsis:
            out.extend([slice(None)] * (ndim - n_explicit))
        else:
            out.append(idx)
    while len(out) < ndim:
        out.append(slice(None))
    return out


def _untouched_axes(indices: SliceIndices, ndim: int) -> dict[int, int]:
    """Maps input axis to output axis for the axes a slice leaves whole.

    A slice expression does not preserve rank, so the two numberings differ:
    ``None`` inserts an output axis without consuming an input one, and an
    integer index consumes an input axis without producing an output one.
    """
    if not isinstance(indices, (list, tuple)):
        return {}
    kept: dict[int, int] = {}
    in_axis = out_axis = 0
    for entry in _expand_ellipsis(indices, ndim):
        if entry is None:
            out_axis += 1
            continue
        if isinstance(entry, int):
            in_axis += 1
            continue
        if entry == slice(None):
            kept[in_axis] = out_axis
        in_axis += 1
        out_axis += 1
    return kept


def _is_minus_one(d: DimLike) -> bool:
    return isinstance(d, StaticDim) and d.dim == -1


def _product(dims: Iterable[DimLike]) -> Dim:
    return functools.reduce(operator.mul, (Dim(d) for d in dims), Dim(1))


def _is_one(d: Dim) -> bool:
    return isinstance(d, StaticDim) and d.dim == 1


def _scaled(size: Dim, num: Dim, den: Dim) -> Dim | None:
    """``size * num // den`` when it is provably exact, else ``None``."""
    if num == den:
        return size
    if isinstance(num, StaticDim) and isinstance(den, StaticDim):
        if isinstance(size, StaticDim):
            total = size.dim * num.dim
            return None if total % den.dim else Dim(total // den.dim)
        if num.dim % den.dim == 0:
            return size * (num.dim // den.dim)
    return None


def _resolve_minus_one(
    src_dims: Sequence[Dim], tgt_dims: Sequence[Dim]
) -> list[Dim]:
    """Replaces a single ``-1`` in ``tgt_dims`` with ``prod(src) // prod(others)``.

    Returns ``tgt_dims`` unchanged when no ``-1`` is present or more
    than one is present.
    """
    pos: int | None = None
    for i, d in enumerate(tgt_dims):
        if _is_minus_one(d):
            if pos is not None:
                return list(tgt_dims)
            pos = i
    if pos is None:
        return list(tgt_dims)
    src_prod = functools.reduce(
        operator.mul, (Dim(d) for d in src_dims), Dim(1)
    )
    other_prod = functools.reduce(
        operator.mul,
        (Dim(d) for i, d in enumerate(tgt_dims) if i != pos),
        Dim(1),
    )
    out = list(tgt_dims)
    # Zero-cell denominator (e.g. static 1 sharded by mesh size 4):
    # leave ``-1`` unresolved so the IR rejects the action cleanly.
    if isinstance(other_prod, StaticDim) and other_prod.dim == 0:
        return list(tgt_dims)
    out[pos] = src_prod // other_prod
    return out


def tile_rule(
    x: TensorLayout, repeats: Iterable[DimLike]
) -> list[AxisAssignment]:
    """Strategies for ``tile``: sharding preserved on axes it does not repeat.

    Tiling a sharded axis locally would give ``[a0 a0 | a1 a1]`` rather than
    ``[a0 a1 a0 a1]``.
    """
    return [
        AxisAssignment((R,), (R,)),
        *(
            AxisAssignment((Sharded(d),), (Sharded(d),))
            for d, r in enumerate(repeats)
            if isinstance(Dim(r), StaticDim) and Dim(r) == 1
        ),
        AxisAssignment((P,), (P,)),
    ]


def permute_rule(x: TensorLayout, dims: Sequence[int]) -> list[AxisAssignment]:
    """Strategies for ``permute``: sharding follows the axis permutation."""
    dims_list = list(dims)
    rows: list[AxisAssignment] = [AxisAssignment((R,), (R,))]
    for in_ax in range(x.rank):
        out_ax = dims_list.index(in_ax)
        rows.append(AxisAssignment((Sharded(in_ax),), (Sharded(out_ax),)))
    rows.append(AxisAssignment((P,), (P,)))
    return rows


def transpose_rule(
    x: TensorLayout, axis_1: int, axis_2: int
) -> list[AxisAssignment]:
    """Strategies for ``transpose``: sharding swaps along with the two axes."""
    n = x.rank
    a1, a2 = axis_1 % n, axis_2 % n
    rows: list[AxisAssignment] = [AxisAssignment((R,), (R,))]
    for in_ax in range(n):
        out_ax = a2 if in_ax == a1 else a1 if in_ax == a2 else in_ax
        rows.append(AxisAssignment((Sharded(in_ax),), (Sharded(out_ax),)))
    rows.append(AxisAssignment((P,), (P,)))
    return rows


def unsqueeze_rule(x: TensorLayout, axis: int) -> list[AxisAssignment]:
    """Strategies for ``unsqueeze``: inserts a size-1 axis; sharding shifts."""
    n = x.rank
    norm = axis if axis >= 0 else axis + n + 1
    rows: list[AxisAssignment] = [AxisAssignment((R,), (R,))]
    for in_ax in range(n):
        out_ax = in_ax if in_ax < norm else in_ax + 1
        rows.append(AxisAssignment((Sharded(in_ax),), (Sharded(out_ax),)))
    rows.append(AxisAssignment((P,), (P,)))
    return rows


def squeeze_rule(x: TensorLayout, axis: int) -> list[AxisAssignment]:
    """Strategies for ``squeeze``: removes a size-1 axis; sharding shifts."""
    n = x.rank
    norm = axis % n
    rows: list[AxisAssignment] = [AxisAssignment((R,), (R,))]
    for in_ax in range(n):
        if in_ax == norm:
            continue
        out_ax = in_ax if in_ax < norm else in_ax - 1
        rows.append(AxisAssignment((Sharded(in_ax),), (Sharded(out_ax),)))
    rows.append(AxisAssignment((P,), (P,)))
    return rows


def flatten_rule(
    x: TensorLayout, start_dim: int = 0, end_dim: int = -1
) -> list[AxisAssignment]:
    """Returns the rows for ``flatten``: only axes outside the flattened range stay sharded."""
    rank = x.rank
    start = start_dim if start_dim >= 0 else start_dim + rank
    end = end_dim if end_dim >= 0 else end_dim + rank
    rows: list[AxisAssignment] = [AxisAssignment((R,), (R,))]
    for in_axis in range(rank):
        if start <= in_axis <= end:
            continue
        out_axis = in_axis if in_axis < start else in_axis - (end - start)
        rows.append(AxisAssignment((Sharded(in_axis),), (Sharded(out_axis),)))
    rows.append(AxisAssignment((P,), (P,)))
    return rows


def _target_shapes(shape: Any, count: int) -> list[Shape]:
    return [local_shape_at(Shape(shape), rank) for rank in range(count)]


def _map_sharded_axes(
    x: TensorLayout,
    target: Shape,
    sources: Sequence[Shape],
    targets: Sequence[Shape],
) -> dict[int, int] | None:
    """Returns the target axis that each sharded axis of ``x`` maps to.

    ``sources`` and ``targets`` hold each device's input and target shape.
    A sharded axis ``k`` maps to target axis ``j`` when ``target`` gives
    ``j`` a size per device and, on every device, the axes before ``j``
    hold as many elements as the axes before ``k``. Each device's block of
    ``k`` is then a block of ``j``. Every other target axis must have the
    same size on every device, unless an ``Unknown`` placement already
    makes sizes differ per device. Returns ``None`` when a sharded axis maps
    to no target axis, or two sharded axes map to the same one.
    """
    placements = x.mapping.placements
    mapped: dict[int, int] = {}
    for k in sorted({p.axis for p in placements if isinstance(p, Sharded)}):
        candidates = [
            j
            for j, d in enumerate(target)
            if is_per_shard_dim(d)
            and all(
                _product(s[:k]) == _product(t[:j])
                for s, t in zip(sources, targets, strict=True)
            )
        ]
        if len(candidates) != 1 or candidates[0] in mapped.values():
            return None
        mapped[k] = candidates[0]
    if not any(isinstance(p, Unknown) for p in placements):
        for axis in range(len(target)):
            if axis not in mapped.values() and any(
                t[axis] != targets[0][axis] for t in targets
            ):
                return None
    return mapped


def reshape_rule(x: TensorLayout, shape: Any) -> list[AxisAssignment]:
    """Returns the rows for reshaping each device's shard to its own target.

    ``shape`` gives each device's sizes: a plain dim is every device's
    size, and a :class:`~max.experimental.sharding.per_shard_dim.PerShardDim`
    gives each device its own. The public ``reshape`` turns a global shape
    into this form with :func:`localize_reshape_target`. The rows keep every
    placement when each device's shard reshapes to its target and every
    sharded axis maps to a target axis (see :func:`_map_sharded_axes`).
    Otherwise, when ``shape`` is a shape for the whole tensor, the rows
    gather the input first.
    """
    target = Shape(shape)
    if sum(1 for d in target if _is_minus_one(d)) > 1:
        raise ValueError(
            f"reshape: at most one -1 dimension is allowed (target {target})."
        )
    sources = [t.shape for t in x.local_types]
    targets = [
        Shape(_resolve_minus_one(source, local))
        for source, local in zip(
            sources, _target_shapes(target, x.mesh.num_devices), strict=True
        )
    ]
    if (
        all(
            _product(s) == _product(t)
            for s, t in zip(sources, targets, strict=True)
        )
        and (mapped := _map_sharded_axes(x, target, sources, targets))
        is not None
    ):
        return pass_through_rows(
            (x,),
            lambda actuals: (
                (Sharded(mapped[actuals[0].axis]),)
                if isinstance(actuals[0], Sharded)
                else actuals
            ),
        )
    whole = list(global_shape(x.shape))
    if not any(is_per_shard_dim(d) for d in target) and _product(whole) == (
        _product(_resolve_minus_one(whole, list(target)))
    ):
        rows = [AxisAssignment((R,), (R,))]
        if P in x.mapping.placements:
            rows.append(AxisAssignment((P,), (P,)))
        return rows
    raise ShardingError(
        f"reshape: no device's shard reshapes to {list(target)} with its "
        f"placement kept, nor does the whole tensor {whole}."
    )


def localize_reshape_target(x: TensorLayout, shape: Any) -> Shape | None:
    """Translates a global ``reshape`` target into each device's sizes.

    Plain dims in ``shape`` are global sizes. A sharded axis ``k`` maps to
    the first target axis ``j`` that has as many elements before it as
    ``k`` has. When a ``-1`` comes before ``j``, the elements after each
    axis are compared instead. Each device's block of ``k`` is then a block
    of ``j``, and ``j`` takes each device's size. For example, rows
    ``(7, 8)`` split 4 and 3 over two devices reshape to ``(7, 4, 2)`` as
    ``(4, 4, 2)`` on the first device and ``(3, 4, 2)`` on the second.

    Args:
        x: The layout of the tensor being reshaped.
        shape: The global target shape.

    Returns:
        The target with each device's sizes, or ``None`` when a sharded
        axis maps to no target axis or a device's size does not convert
        exactly. The caller then gathers the tensor first.

    Raises:
        ValueError: If the target has a known number of elements that
            differs from the tensor's.
    """
    target = Shape(shape)
    sharded = sorted(
        {p.axis for p in x.mapping.placements if isinstance(p, Sharded)}
    )
    if not sharded:
        return target
    minus_ones = [j for j, d in enumerate(target) if _is_minus_one(d)]
    if len(minus_ones) > 1:
        # Left for reshape_rule to reject.
        return target
    minus = minus_ones[0] if minus_ones else None
    source, tgt = list(global_shape(x.shape)), list(global_shape(target))
    if minus is None and all(isinstance(d, StaticDim) for d in (*source, *tgt)):
        before, after = _product(source), _product(tgt)
        if before != after:
            raise ValueError(
                f"reshape: {source} has {before} elements, the target "
                f"{tgt} has {after}."
            )
    mapped: dict[int, int] = {}
    for k in sharded:
        # Count the elements on the side of ``j`` that has no -1.
        j = next(
            (
                j
                for j, d in enumerate(tgt)
                if not _is_one(d)
                and (
                    _product(source[:k]) == _product(tgt[:j])
                    if minus is None or minus >= j
                    else _product(source[k:]) == _product(tgt[j:])
                )
            ),
            None,
        )
        if j is None or j in mapped.values():
            return None
        mapped[k] = j
    shapes: list[list[Dim]] = []
    for local in (t.shape for t in x.local_types):
        out = list(tgt)
        for k, j in mapped.items():
            size = (
                _scaled(
                    Dim(local[k]),
                    _product(source[k + 1 :]),
                    _product(tgt[j + 1 :]),
                )
                if minus is None or minus <= j
                else _scaled(Dim(local[k]), tgt[j], source[k])
            )
            if size is None:
                return None
            out[j] = size
        if minus is not None and minus not in mapped.values():
            # A device with an empty shard cannot infer the -1 itself.
            others = _product(d for i, d in enumerate(tgt) if i != minus)
            size = _scaled(Dim(1), _product(source), others)
            if size is not None:
                out[minus] = size
        shapes.append(out)
    return Shape(
        make_per_shard_dim(
            [s[j] for s in shapes],
            global_dim=None if _is_minus_one(tgt[j]) else tgt[j],
            force_wrap=True,
        )
        if j in mapped.values()
        # A size given per device, as along an Unknown axis, stays each
        # device's own.
        else target[j]
        if is_per_shard_dim(target[j])
        else shapes[0][j]
        for j in range(len(target))
    )


def localize_same_shape_target(x: TensorLayout, shape: Any) -> Shape:
    """Translates a global ``broadcast_to`` or ``rebind`` target into each device's sizes.

    Both ops keep each input axis on the target axis it is right-aligned
    with. Where the input axis is sharded, that target axis takes the
    input's sizes on each device, and its global size must match the
    input's. Every other target axis is whole on each device, so a size
    taken from another tensor's sharded ``shape`` becomes its global size.
    A size with no global value, as along an
    :class:`~max.experimental.sharding.Unknown` placement, stays each
    device's own.

    Args:
        x: The layout of the input tensor.
        shape: The global target shape.

    Returns:
        The target with each device's sizes on the sharded axes.

    Raises:
        ShardingError: If the target gives a sharded axis a size other than
            its global size.
    """
    target = Shape(shape)
    offset = len(target) - x.rank
    out = [
        d._global if is_per_shard_dim(d) and d._global is not None else d
        for d in target
    ]
    for k in sorted(
        {p.axis for p in x.mapping.placements if isinstance(p, Sharded)}
    ):
        j = k + offset
        if j < 0:
            continue
        size, wanted = global_dim(x.shape[k]), global_dim(target[j])
        if (
            isinstance(size, StaticDim)
            and isinstance(wanted, StaticDim)
            and size != wanted
        ):
            raise ShardingError(
                f"target axis {j} ({wanted}) differs from the size of "
                f"sharded input axis {k} ({size})."
            )
        out[j] = (
            target[j]
            if is_per_shard_dim(target[j])
            else make_per_shard_dim(
                [t.shape[k] for t in x.local_types],
                global_dim=size,
                force_wrap=True,
            )
        )
    return Shape(out)


def rebind_rule(
    x: TensorLayout,
    shape: Any,
    message: str = "",
    layout: object = None,
) -> list[AxisAssignment]:
    """Keeps placement unchanged while asserting the supplied local shape."""
    return pass_through_rows((x,), lambda actuals: actuals)


def broadcast_to_rule(
    x: TensorLayout,
    shape: Any,
    out_dims: Iterable[DimLike] | None = None,
) -> list[AxisAssignment]:
    """Returns the rows for ``broadcast_to``: a shard stays on its right-aligned axis."""
    if isinstance(shape, TensorLayout):
        raise NotImplementedError(
            "broadcast_to does not support a tensor-valued shape; pass a "
            "ShapeLike (list of DimLike)."
        )
    targets = _target_shapes(shape, x.mesh.num_devices)
    for source, target in zip(x.local_types, targets, strict=True):
        if len(target) < x.rank or any(
            source_dim != 1 and source_dim != target_dim
            for source_dim, target_dim in zip(
                reversed(source.shape), reversed(target), strict=False
            )
        ):
            raise ValueError(
                "broadcast_to: each rank's input dimension must be either 1 or equal to its target; "
                "use source tensor dimensions or explicitly redistribute first."
            )
    offset = len(targets[0]) - x.rank

    def outputs(actuals: tuple[Placement, ...]) -> tuple[Placement, ...] | None:
        (placement,) = actuals
        if not isinstance(placement, Sharded):
            return actuals
        if any(
            local.shape[placement.axis] != target[placement.axis + offset]
            for local, target in zip(x.local_types, targets, strict=True)
        ):
            return None
        return (Sharded(placement.axis + offset),)

    return pass_through_rows((x,), outputs)


def _concat_stack_rows(
    layouts: tuple[TensorLayout, ...],
    out_axis_for_in: int | None,
    norm: int | None,
) -> list[AxisAssignment]:
    """Rows shared by ``concat``/``stack``: every input wears the same placement."""
    n = len(layouts)
    rank = layouts[0].rank
    rows = [AxisAssignment((R,) * n, (R,))]
    for ax in range(rank):
        if norm is None:
            out_ax = ax
        else:
            out_ax = ax if ax < norm else ax + 1
        rows.append(AxisAssignment((Sharded(ax),) * n, (Sharded(out_ax),)))
    rows.append(AxisAssignment((P,) * n, (P,)))
    return rows


def concat_rule(
    original_vals: Iterable[TensorLayout], axis: int = 0
) -> list[AxisAssignment]:
    """Returns the rows for ``concat``: every input shares one placement."""
    layouts = tuple(original_vals)
    if not layouts:
        raise ValueError("concat: no tensor inputs.")
    return _concat_stack_rows(layouts, out_axis_for_in=None, norm=None)


def stack_rule(
    values: Iterable[TensorLayout], axis: int = 0
) -> list[AxisAssignment]:
    """Returns the rows for ``stack``: a new axis, and one shared placement."""
    layouts = tuple(values)
    if not layouts:
        raise ValueError("stack: no tensor inputs.")
    rank = layouts[0].rank
    norm = axis if axis >= 0 else axis + rank + 1
    return _concat_stack_rows(layouts, out_axis_for_in=None, norm=norm)


def chunk_rule(
    x: TensorLayout, chunks: int, axis: int = 0
) -> list[AxisAssignment]:
    """Strategies for ``chunk``: sharding on any axis except the chunk axis."""
    n = x.rank
    norm = axis % n
    rows: list[AxisAssignment] = [AxisAssignment((R,), (R,) * chunks)]
    for ax in range(n):
        if ax == norm:
            continue
        rows.append(AxisAssignment((Sharded(ax),), (Sharded(ax),) * chunks))
    rows.append(AxisAssignment((P,), (P,) * chunks))
    return rows


def top_k_rule(
    input: TensorLayout, k: int, axis: int = -1
) -> list[AxisAssignment]:
    """Strategies for ``top_k``: sharding on any axis except the reduction axis."""
    n = input.rank
    norm = axis % n
    rows: list[AxisAssignment] = [AxisAssignment((R,), (R, R))]
    for ax in range(n):
        if ax == norm:
            continue
        rows.append(AxisAssignment((Sharded(ax),), (Sharded(ax), Sharded(ax))))
    return rows


def argsort_rule(
    x: TensorLayout, ascending: bool = True
) -> list[AxisAssignment]:
    """Strategies for ``argsort``: Replicated only (sort needs full view)."""
    return [AxisAssignment((R,), (R,))]


def nonzero_rule(x: TensorLayout, out_dim: DimLike) -> list[AxisAssignment]:
    """Strategies for ``nonzero``: Replicated only (data-dependent output shape)."""
    return [AxisAssignment((R,), (R,))]


def repeat_interleave_rule(
    x: TensorLayout,
    repeats: int | TensorLayout,
    axis: int | None = None,
    out_dim: DimLike | None = None,
) -> list[AxisAssignment]:
    """Strategies for ``repeat_interleave``: sharding on any non-repeat axis."""
    rows: list[AxisAssignment] = [AxisAssignment((R,), (R,))]
    if axis is not None:
        n = x.rank
        norm = axis % n
        for ax in range(n):
            if ax == norm:
                continue
            rows.append(AxisAssignment((Sharded(ax),), (Sharded(ax),)))
        rows.append(AxisAssignment((P,), (P,)))
    if isinstance(repeats, TensorLayout):
        rows = [
            AxisAssignment(row.needed_inputs + (R,), row.outputs)
            for row in rows
        ]
    return rows


def pad_rule(
    input: TensorLayout,
    paddings: Iterable[int],
    mode: str = "constant",
    value: TensorLayout | float = 0,
) -> list[AxisAssignment]:
    """Strategies for ``pad``: sharding allowed on unpadded axes only."""
    pads = tuple(paddings)
    padded = {
        i // 2
        for i in range(0, len(pads) - 1, 2)
        if pads[i] != 0 or pads[i + 1] != 0
    }
    linear = mode != "constant" or (
        isinstance(value, (int, float)) and value == 0
    )
    rows: list[AxisAssignment] = [AxisAssignment((R,), (R,))]
    for ax in range(input.rank):
        if ax in padded:
            continue
        rows.append(AxisAssignment((Sharded(ax),), (Sharded(ax),)))
    if linear:
        rows.append(AxisAssignment((P,), (P,)))
    if isinstance(value, TensorLayout):
        rows = [
            AxisAssignment(row.needed_inputs + (R,), row.outputs)
            for row in rows
        ]
    return rows


def slice_tensor_rule(
    x: TensorLayout, indices: SliceIndices
) -> list[AxisAssignment]:
    """Strategies for ``slice_tensor``: sharding allowed on non-sliced axes only."""
    kept = _untouched_axes(indices, x.rank) if indices is not None else {}
    rows: list[AxisAssignment] = [AxisAssignment((R,), (R,))]
    for in_axis, out_axis in sorted(kept.items()):
        rows.append(AxisAssignment((Sharded(in_axis),), (Sharded(out_axis),)))
    rows.append(AxisAssignment((P,), (P,)))
    return rows


def gather_rule(
    input: TensorLayout, indices: TensorLayout, axis: int
) -> list[AxisAssignment]:
    """Strategies for ``gather``: sharding follows either input or indices axes.

    Deliberately does not emit the expert-parallel
    ``(Sharded(a_axis), R) -> Partial(SUM)`` row. That row treats a
    local gather on each rank as if the missing cross-rank entries
    were zero and then sums — only correct when the caller has masked
    indices to each rank's owned slice. Letting the picker pick it
    silently produces wrong results in the common case (e.g. gathering
    a few token positions out of a seq-sharded residual stream). When
    the gather axis is sharded under this rule, the picker must
    ``allgather`` to :class:`Replicated` first. Callers that genuinely
    want EP semantics override ``gather.rule`` with their own rule
    (see :doc:`rules/README` "Adding rows for a custom placement type").
    """
    in_r, idx_r = input.rank, indices.rank
    a_axis = axis % in_r
    rows: list[AxisAssignment] = [AxisAssignment((R, R), (R,))]
    for in_ax in range(in_r):
        if in_ax == a_axis:
            continue
        out_ax = in_ax if in_ax < a_axis else in_ax + idx_r - 1
        rows.append(AxisAssignment((Sharded(in_ax), R), (Sharded(out_ax),)))
    for idx_ax in range(idx_r):
        rows.append(
            AxisAssignment((R, Sharded(idx_ax)), (Sharded(a_axis + idx_ax),))
        )
    rows.append(AxisAssignment((P, R), (P,)))
    return rows


def gather_nd_rule(
    input: TensorLayout, indices: TensorLayout, batch_dims: int = 0
) -> list[AxisAssignment]:
    """Strategies for ``gather_nd``: sharding on batch dims only."""
    rows: list[AxisAssignment] = [AxisAssignment((R, R), (R,))]
    for ax in range(batch_dims):
        rows.append(AxisAssignment((Sharded(ax), Sharded(ax)), (Sharded(ax),)))
    rows.append(AxisAssignment((P, R), (P,)))
    return rows


def _scatter_rows(input: TensorLayout, axis: int) -> list[AxisAssignment]:
    """Builds the menu for ``scatter`` / ``scatter_add``.

    Non-scatter axes are batch-like (``updates``/``indices`` share the
    input's extent there), so all three operands shard together on them.
    Replicating ``updates``/``indices`` against a sharded input would give
    each device the full batch extent against a partial input shard, and the
    scatter kernel would index outside the shard.
    """
    in_r = input.rank
    a_axis = axis % in_r
    rows: list[AxisAssignment] = [AxisAssignment((R, R, R), (R,))]
    for ax in range(in_r):
        if ax == a_axis:
            continue
        rows.append(
            AxisAssignment(
                (Sharded(ax), Sharded(ax), Sharded(ax)), (Sharded(ax),)
            )
        )
    return rows


def scatter_rule(
    input: TensorLayout,
    updates: TensorLayout,
    indices: TensorLayout,
    axis: int = -1,
) -> list[AxisAssignment]:
    """Strategies for ``scatter``: sharding on any axis except the scatter axis."""
    return _scatter_rows(input, axis)


def scatter_add_rule(
    input: TensorLayout,
    updates: TensorLayout,
    indices: TensorLayout,
    axis: int = -1,
) -> list[AxisAssignment]:
    """Strategies for ``scatter_add``: same as ``scatter`` (accumulating variant)."""
    return _scatter_rows(input, axis)


def split_rule(
    x: TensorLayout, split_sizes: Sequence[DimLike], axis: int = 0
) -> list[AxisAssignment]:
    """Strategies for ``split``: sharding on any axis except the split axis."""
    n = x.rank
    norm = axis % n
    count = len(split_sizes)
    rows: list[AxisAssignment] = [AxisAssignment((R,), (R,) * count)]
    for ax in range(n):
        if ax == norm:
            continue
        rows.append(AxisAssignment((Sharded(ax),), (Sharded(ax),) * count))
    rows.append(AxisAssignment((P,), (P,) * count))
    return rows
