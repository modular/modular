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
"""Ops for generating random numbers."""

from __future__ import annotations

import weakref
from collections.abc import MutableMapping
from dataclasses import replace

import numpy as np
from max._core.dialects import kgen, rmo
from max.dtype import DType

from .. import dtype_promotion
from ..graph import Graph
from ..type import DeviceRef, TensorType
from ..value import TensorValue, TensorValueLike
from .elementwise import _accum_type

SEEDS: MutableMapping[Graph, TensorValue] = weakref.WeakKeyDictionary()
SeedType = TensorType(DType.int64, [], device=DeviceRef.CPU())


def _rotate_seed(seed: TensorValue):  # noqa: ANN202
    # Let's just get some different random numbers
    # from the initial seed for now.
    return seed + 1


def assert_scalar(value: TensorValueLike) -> None:
    """Raises :class:`ValueError` if value is not a scalar (has non-empty ``shape``)."""
    if isinstance(value, np.ndarray | TensorValue) and value.shape:
        raise ValueError("Expected a scalar value")


def _next_seed():  # noqa: ANN202
    graph = Graph.current
    seed = _peek_seed()
    SEEDS[graph] = _rotate_seed(seed)
    return seed


def _peek_seed():  # noqa: ANN202
    graph = Graph.current
    try:
        return SEEDS[graph]
    except LookupError:
        raise RuntimeError("No seed set! Set with `ops.random.set_seed`.")  # noqa: B904


def set_seed(seed: TensorValueLike | int = 0) -> None:
    """Sets the seed for random numbers generated in the graph.

    Call this once per graph. Subsequent random ops (:func:`gaussian`,
    :func:`uniform`) automatically rotate from this seed, so you never need
    to call :func:`set_seed` again within the same graph.

    A static integer makes output deterministic across every execution of
    the compiled graph. To vary the seed per execution (for example, one
    seed per user request), declare a ``SeedType`` graph input and pass it
    to :func:`set_seed` at build time:

    .. code-block:: python

        with Graph("my_graph", input_types=[ops.random.SeedType]) as g:
            ops.random.set_seed(g.inputs[0].tensor)
            result = ops.random.gaussian(tensor_type)

    The graph is then compiled once and executed many times, binding a
    different seed value to that input on each execution (no recompilation
    needed).

    Args:
        seed: The seed value. Accepts a Python int or a scalar int64
            :class:`~max.graph.TensorValue` (for dynamic graph inputs).
    """
    assert_scalar(seed)
    seed = dtype_promotion._promote_to_strong(
        seed, DType.int64, DeviceRef.CPU()
    )
    if seed.dtype != DType.int64:
        raise TypeError("Seed value must be int64")
    SEEDS[Graph.current] = seed


def gaussian(
    like: TensorType,
    mean: TensorValueLike = 0,
    std: TensorValueLike = 1,
) -> TensorValue:
    """Samples from a Gaussian (normal) distribution with the given mean and standard deviation.

    Output shape and dtype match the ``like`` tensor type. A seed must be
    set on the current graph with :func:`set_seed`.

    Args:
        like: A :class:`~max.graph.TensorType` whose shape, dtype, and device
            determine the output tensor.
        mean: The mean of the Gaussian distribution. Must be a scalar.
            Defaults to 0.
        std: The standard deviation of the Gaussian distribution. Must be a
            scalar. Defaults to 1.

    Returns:
        A symbolic tensor with the same shape and dtype as ``like``, filled
        with values sampled from the specified Gaussian distribution.

    Raises:
        RuntimeError: If no seed has been set with :func:`set_seed`.
        ValueError: If ``mean`` or ``std`` are not scalar values.
    """
    assert_scalar(mean)
    assert_scalar(std)
    # Check whether we have a seed before we add other constants to the graph.
    seed = _next_seed()
    accum_type = _accum_type(like) if like.dtype.is_float() else DType.float32
    random_accum = Graph.current._add_op_generated(
        rmo.MoRandomNormalOp,
        result=replace(like, dtype=accum_type),
        shape=TensorValue(like.shape),
        mean=dtype_promotion._promote_to_strong(
            mean, DType.float32, DeviceRef.CPU()
        ),
        variance=dtype_promotion._promote_to_strong(
            std, DType.float32, DeviceRef.CPU()
        ),
        seed=seed,
        output_param_decls=kgen.ParamDeclArrayAttr([]),
    )[0].tensor
    if not like.dtype.is_float():
        random_accum = round(random_accum)
    return random_accum.cast(like.dtype)


# Alias normal <-> gaussian
normal = gaussian


def uniform(
    like: TensorType,
    range: tuple[TensorValueLike, TensorValueLike] = (0, 1),
) -> TensorValue:
    """Samples uniformly from the half-open interval ``[lower, upper)``.

    Values satisfy ``lower ≤ x < upper``. Output shape and dtype match
    the ``like`` tensor type. A seed must be set on the current graph with
    :func:`set_seed`.

    Args:
        like: A :class:`~max.graph.TensorType` whose shape, dtype, and device
            determine the output tensor.
        range: A tuple ``(lower, upper)`` specifying the half-open interval to
            sample from. Both bounds must be scalars. Defaults to ``(0, 1)``.

    Returns:
        A symbolic tensor with the same shape and dtype as ``like``, filled
        with values sampled uniformly from ``[lower, upper)``.

    Raises:
        RuntimeError: If no seed has been set with :func:`set_seed`.
        ValueError: If ``lower`` or ``upper`` are not scalar values.
    """
    lower, upper = range

    assert_scalar(lower)
    assert_scalar(upper)
    # Check whether we have a seed before we add other constants to the graph.
    seed = _next_seed()
    return Graph.current._add_op_generated(
        rmo.MoRandomUniformOp,
        result=like,
        shape=TensorValue(like.shape),
        lower_bound=dtype_promotion._promote_to_strong(
            lower, like.dtype, DeviceRef.CPU()
        ),
        upper_bound=dtype_promotion._promote_to_strong(
            upper, like.dtype, DeviceRef.CPU()
        ),
        seed=seed,
        output_param_decls=kgen.ParamDeclArrayAttr([]),
    )[0].tensor
