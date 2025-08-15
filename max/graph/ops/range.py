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

from typing import Optional, get_args

from max.dtype import DType
from max.mlir.dialects import rmo

from .. import dtype_promotion
from ..dim import DimLike
from ..graph import Graph
from ..type import DeviceRef
from ..value import Numeric, TensorType, TensorValue, TensorValueLike
from .cast import cast
from .constant import constant


def range(
    start: TensorValueLike,
    stop: TensorValueLike,
    step: TensorValueLike,
    out_dim: Optional[DimLike] = None,
    device: DeviceRef = DeviceRef.CPU(),
    dtype: Optional[DType] = None,
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
    if out_dim is None:
        if (
            isinstance(start, get_args(Numeric))
            and isinstance(stop, get_args(Numeric))
            and isinstance(step, get_args(Numeric))
        ):
            out_dim = (stop - start) // step if step != 0 else 0
        else:
            raise ValueError(
                "range expected out_dim for non-numeric values as inputs!"
            )

    # Determine the dtype: explicit dtype if provided, otherwise infer or default
    if dtype is None:
        if isinstance(start, get_args(Numeric)):
            # For numeric inputs, default to float32 to maintain API simplicity
            dtype = DType.float32
        else:
            # For tensor inputs, infer from the inputs
            start_tensor = TensorValue(start)
            stop_tensor = TensorValue(stop)
            step_tensor = TensorValue(step)
            if start_tensor.dtype == stop_tensor.dtype == step_tensor.dtype:
                dtype = start_tensor.dtype
            else:
                raise ValueError("range expected inputs of the same type!")

    def to_dtype(value: TensorValueLike) -> TensorValue:
        # For numeric inputs, use constant() which allows precision loss
        # This is appropriate for range operations where we're creating indices
        if isinstance(value, get_args(Numeric)):
            return constant(value, dtype, DeviceRef.CPU())

        # For tensor inputs, use the more strict _promote_to_strong approach
        value = dtype_promotion._promote_to_strong(value, dtype, device)
        if value.dtype != dtype:
            value = cast(value, dtype)
        return value

    start = to_dtype(start)
    stop = to_dtype(stop)
    step = to_dtype(step)

    if start.dtype != stop.dtype or stop.dtype != step.dtype:
        raise ValueError("range expected inputs of the same type!")
    if start.rank != 0 or stop.rank != 0 or step.rank != 0:
        raise ValueError("range expected scalar values as inputs!")

    return Graph.current._add_op(
        rmo.mo_range,
        TensorType(dtype, shape=[out_dim], device=device).to_mlir(),
        start,
        stop,
        step,
    )[0].tensor
