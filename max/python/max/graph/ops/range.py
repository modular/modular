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
"""Op implementation for range."""

from __future__ import annotations

import math

import numpy as np
from max._core.dialects import kgen, mo
from max.dtype import DType

from ...driver import Device
from .. import dtype_promotion
from ..dim import Dim, DimLike, StaticDim
from ..graph import Graph
from ..type import DeviceRef
from ..value import (
    Scalar,
    TensorType,
    TensorValue,
    TensorValueLike,
    _is_scalar,
)
from .cast import cast


def _as_number(value: Scalar) -> Scalar:
    return int(value) if isinstance(value, StaticDim) else value


def _range_length(start: Scalar, stop: Scalar, step: Scalar) -> Dim:
    """Returns the number of values in ``[start, stop)`` spaced by ``step``."""
    start, stop, step = (_as_number(v) for v in (start, stop, step))
    if isinstance(start, Dim) or isinstance(stop, Dim) or isinstance(step, Dim):
        # - Most combinations of scalars will work fine
        # - Specifically mixing float and Dim doesn't work today (but could)
        # - Telling mypy about this case specifically is hard, hence the ignore
        return Dim((stop - start) // step if step != 0 else 0)  # type: ignore
    # TODO(bduke): step == 0 should raise in the range op builder.
    if step == 0:
        return Dim(0)
    # The last value may land short of stop, so the length rounds up.
    if (
        isinstance(start, int | np.integer)
        and isinstance(stop, int | np.integer)
        and isinstance(step, int | np.integer)
    ):
        return Dim(int(-((start - stop) // step)))
    return Dim(math.ceil((stop - start) / step))


def range(
    start: TensorValueLike,
    stop: TensorValueLike,
    step: TensorValueLike = 1,
    out_dim: DimLike | None = None,
    *,
    dtype: DType,
    device: Device | DeviceRef,
) -> TensorValue:
    """Creates a sequence of evenly spaced values from ``start`` to ``stop``.

    The sequence begins at ``start`` and increments by ``step``, stopping
    before ``stop`` (the upper bound is exclusive).

    ``stop - start`` must be zero or have the same sign as ``step``. The
    sequence may end short of ``stop`` when ``step`` doesn't divide
    ``stop - start`` evenly: ``range(0, 5, 2)`` produces ``[0, 2, 4]``.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("range_example") as graph:
            graph.output(ops.range(0, 5, 1, dtype=DType.float32, device=device))

        model = InferenceSession().load(graph)
        result = model.execute()[0]
        # result holds [0.0, 1.0, 2.0, 3.0, 4.0].

    Args:
        start: The first value in the sequence. Must be a scalar value.
        stop: The exclusive upper bound. The sequence stops before this value. Must be a scalar value.
        step: The spacing between consecutive values. Must be non-zero. Defaults to ``1``.
        out_dim: The expected length of the output. Required when dynamic
            scalar :class:`~max.graph.TensorValue` inputs prevent static length
            inference. When omitted, it's computed from scalar literals. A
            static ``out_dim`` must agree with the length computed from
            scalar literals; a symbolic one is bound to that length at runtime.
        dtype: The element type of the result tensor.
        device: The device the result tensor lives on.

    Returns:
        A ``TensorValue`` representing the generated sequence.

    Raises:
        ValueError: If ``out_dim`` is omitted for dynamic scalar inputs, if a
            static ``out_dim`` disagrees with the length of a literal range, if
            any input isn't scalar, or if any input isn't on the CPU.
    """
    device = DeviceRef.from_device(device)

    if _is_scalar(start) and _is_scalar(stop) and _is_scalar(step):
        length = _range_length(start, stop, step)
        out_dim = length if out_dim is None else Dim(out_dim)
        if (
            isinstance(out_dim, StaticDim)
            and isinstance(length, StaticDim)
            and out_dim != length
        ):
            raise ValueError(
                f"range({start}, {stop}, {step}) produces {int(length)}"
                f" values, but out_dim is {int(out_dim)}"
            )
    elif out_dim is None:
        raise ValueError("Dynamic ranges must provide an explicit out_dim")
    out_dim = Dim(out_dim)

    def to_dtype(value: TensorValueLike) -> TensorValue:
        value = dtype_promotion._promote_to_strong(
            value, dtype, DeviceRef.CPU()
        )
        if value.dtype != dtype:
            value = cast(value, dtype)
        return value

    start = to_dtype(start)
    stop = to_dtype(stop)
    step = to_dtype(step)
    assert start.dtype == stop.dtype == step.dtype

    if not start.rank == stop.rank == step.rank == 0:
        raise ValueError("range expected scalar values as inputs!")
    if not start.device == stop.device == step.device == DeviceRef.CPU():
        raise ValueError("Range input values must be on CPU")

    return Graph.current._add_op_generated(
        mo.RangeOp,
        TensorType(dtype, shape=[out_dim], device=device).to_mlir(),
        start,
        stop,
        step,
        kgen.ParamDeclArrayAttr([]),
    )[0].tensor
