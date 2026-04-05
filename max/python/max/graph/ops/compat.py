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
"""Higher-level tensor compatibility helpers built from core graph ops."""

from __future__ import annotations

from collections.abc import Sequence
from functools import reduce
from operator import mul

from ..dim import Dim, DimLike, StaticDim
from ..value import TensorValue, TensorValueLike
from .broadcast_to import broadcast_to
from .reduction import min as reduce_min
from .reshape import reshape
from .slice_tensor import slice_tensor
from .split import split
from .squeeze import squeeze
from .transpose import transpose
from .validation import assert_valid_axis
from .where import where


def _normalize_axis(x: TensorValue, axis: int) -> int:
    assert_valid_axis(x, axis)
    return axis + x.rank if axis < 0 else axis


def amin(
    x: TensorValueLike,
    axis: int | Sequence[int] | None = -1,
) -> TensorValue:
    """Computes the minimum over one or more axes."""
    x = TensorValue(x)
    if axis is None:
        return squeeze(reduce_min(x.reshape([-1]), axis=0), 0)
    if isinstance(axis, int):
        normalized_axis = _normalize_axis(x, axis)
        return squeeze(reduce_min(x, axis=normalized_axis), normalized_axis)

    axes = list(axis)
    if not axes:
        return x

    normalized_axes = [_normalize_axis(x, ax) for ax in axes]
    if len(set(normalized_axes)) != len(normalized_axes):
        raise ValueError(f"Duplicate axes are not allowed: {axis}")

    out = x
    for ax in sorted(normalized_axes, reverse=True):
        out = reduce_min(out, axis=ax)
    for ax in sorted(normalized_axes, reverse=True):
        out = squeeze(out, ax)
    return out


def masked_fill(
    x: TensorValueLike,
    mask: TensorValueLike,
    value: TensorValueLike,
) -> TensorValue:
    """Replaces masked elements with a broadcastable fill value."""
    return where(mask, value, x)


def repeat(x: TensorValueLike, repeats: Sequence[DimLike]) -> TensorValue:
    """Repeats the tensor along each dimension using PyTorch semantics."""
    x = TensorValue(x)
    repeat_dims: list[Dim] = [Dim(r) for r in repeats]

    if len(repeat_dims) < x.rank:
        raise ValueError(
            "repeat expects at least as many repeat dimensions as the input rank"
        )

    input_shape: list[Dim] = [Dim(dim) for dim in x.shape]
    if len(repeat_dims) > x.rank:
        input_shape = [Dim(1)] * (len(repeat_dims) - x.rank) + input_shape
        x = reshape(x, input_shape)

    interleaved_shape: list[Dim] = []
    broadcast_shape: list[Dim] = []
    output_shape: list[Dim] = []
    for repeat_dim, input_dim in zip(repeat_dims, input_shape, strict=True):
        if isinstance(repeat_dim, StaticDim) and int(repeat_dim) < 0:
            raise ValueError(
                f"repeat expects non-negative repeat counts, got {repeat_dim}"
            )
        interleaved_shape.extend([Dim(1), input_dim])
        broadcast_shape.extend([repeat_dim, input_dim])
        output_shape.append(Dim(repeat_dim * input_dim))

    x = reshape(x, interleaved_shape)
    x = broadcast_to(x, broadcast_shape)
    return reshape(x, output_shape)


def flip(x: TensorValueLike, dims: Sequence[int]) -> TensorValue:
    """Reverses the tensor along the requested dimensions."""
    x = TensorValue(x)
    normalized_dims = [_normalize_axis(x, dim) for dim in dims]
    if len(set(normalized_dims)) != len(normalized_dims):
        raise ValueError(f"flip does not allow duplicate dims: {dims}")

    indices = [slice(None)] * x.rank
    for dim in normalized_dims:
        indices[dim] = slice(None, None, -1)
    return slice_tensor(x, tuple(indices))


def swapaxes(x: TensorValueLike, axis0: int, axis1: int) -> TensorValue:
    """Swaps two tensor dimensions."""
    return transpose(x, axis0, axis1)


def unflatten(
    x: TensorValueLike,
    dim: int,
    sizes: Sequence[DimLike],
) -> TensorValue:
    """Expands one dimension into multiple dimensions."""
    x = TensorValue(x)
    axis = _normalize_axis(x, dim)
    size_dims: list[Dim] = [Dim(size) for size in sizes]

    if not size_dims:
        raise ValueError("unflatten expects at least one replacement size")
    if sum(size == Dim(-1) for size in size_dims) > 1:
        raise ValueError("unflatten allows at most one inferred dimension")
    if any(
        isinstance(size, StaticDim) and int(size) < -1 for size in size_dims
    ):
        raise ValueError(
            f"unflatten does not allow sizes below -1: {size_dims}"
        )

    axis_dim = x.shape[axis]
    static_known_sizes: list[StaticDim] = []
    has_inferred_dim = False
    all_non_inferred_static = True
    for size in size_dims:
        if isinstance(size, StaticDim):
            if int(size) == -1:
                has_inferred_dim = True
            else:
                static_known_sizes.append(size)
        else:
            all_non_inferred_static = False

    if isinstance(axis_dim, StaticDim):
        axis_size = int(axis_dim)
        if all_non_inferred_static:
            known_product = reduce(
                mul, (int(size) for size in static_known_sizes), 1
            )
        else:
            known_product = None

        if has_inferred_dim and known_product is not None:
            if known_product == 0 or axis_size % known_product != 0:
                raise ValueError(
                    "unflatten sizes are not compatible with the source dimension"
                )
        elif (
            not has_inferred_dim
            and known_product is not None
            and known_product != axis_size
        ):
            raise ValueError(
                "unflatten sizes must multiply to the source dimension size"
            )

    output_shape: list[Dim] = [Dim(dim) for dim in x.shape[:axis]]
    output_shape.extend(size_dims)
    output_shape.extend(Dim(dim) for dim in x.shape[axis + 1 :])
    return reshape(x, output_shape)


def unbind(x: TensorValueLike, dim: int = 0) -> tuple[TensorValue, ...]:
    """Returns a tuple of slices with the selected dimension removed."""
    x = TensorValue(x)
    axis = _normalize_axis(x, dim)
    axis_dim = x.shape[axis]
    if not isinstance(axis_dim, StaticDim):
        raise ValueError(
            "unbind requires a statically known size along the selected axis"
        )

    return tuple(
        squeeze(part, axis) for part in split(x, [1] * int(axis_dim), axis)
    )
