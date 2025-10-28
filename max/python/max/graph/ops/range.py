# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from max._core.dialects import kgen, rmo
from max.dtype import DType

from ...driver import Device
from .. import dtype_promotion
from ..dim import Dim, DimLike
from ..graph import Graph
from ..type import DeviceRef
from ..value import TensorType, TensorValue, TensorValueLike, _is_scalar
from .cast import cast


def range(
    start: TensorValueLike,
    stop: TensorValueLike,
    step: TensorValueLike = 1,
    out_dim: DimLike | None = None,
    *,
    dtype: DType,
    device: Device | DeviceRef,
) -> TensorValue:
    """Creates a sequence of numbers. The sequence goes from `start` with
    increments of size `step` up to (but not including) `stop`. All arguments
    are mandatory and must have the same element type.

    Note the following restrictions on input values:
    1. `step` must be non-zero
    2. `stop - start` must be zero or have the same sign as `step`

    Args:
        start: The start of the range to generate.
        stop: The range will be generated up to, but not including, this value.
        step: The step size for the range.
        out_dim: The expected output dimensions returned by the range op.
          These will be assert at graph execution time to be correct.
        device: Device of the result tensor.
        dtype: Data type of the result tensor. If not specified, defaults to
          float32 for numeric inputs or infers from tensor inputs.

    Returns:
        A symbolic tensor value containing the defined range of values.
    """
    device = DeviceRef.from_device(device)

    if out_dim is None:
        if not (_is_scalar(start) and _is_scalar(stop) and _is_scalar(step)):
            raise ValueError("Dynamic ranges must provide an explicit out_dim")
        # - Most combinations of scalars will work fine
        # - Specifically mixing float and Dim doesn't work today (but could)
        # - Telling mypy about this case specifically is hard, hence the ignore
        out_dim = (stop - start) // step if step != 0 else 0  # type: ignore
        if isinstance(out_dim, float):
            out_dim = int(out_dim)
        assert out_dim is not None
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
        rmo.MoRangeOp,
        TensorType(dtype, shape=[out_dim], device=device).to_mlir(),
        start,
        stop,
        step,
        kgen.ParamDeclArrayAttr([]),
    )[0].tensor
