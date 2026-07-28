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
"""Op implementation for repeat_interleave."""

import numpy as np
from max.dtype import DType

from ..dim import Dim, DimLike
from ..shape import Shape
from ..type import DeviceRef, TensorType
from ..value import TensorValue, TensorValueLike
from .broadcast_to import broadcast_to
from .constant import constant
from .custom import custom


def _promote_repeats(
    repeats: int | TensorValue,
    input_dim: Dim,
    out_dim: DimLike | None,
) -> tuple[TensorValue, Dim | None]:
    if out_dim is not None:
        out_dim = Dim(out_dim)

    if isinstance(repeats, TensorValue):
        if repeats.rank == 0:
            repeats = broadcast_to(repeats, [1])
        return repeats, out_dim

    if repeats <= 0:
        raise ValueError(
            f"repeats_inteleave: repeat value must be positive, given {repeats=}"
        )

    return constant(
        np.array([repeats]), DType.int64, DeviceRef.CPU()
    ), input_dim * repeats


def repeat_interleave(
    x: TensorValueLike,
    repeats: int | TensorValue,
    axis: int | None = None,
    out_dim: DimLike | None = None,
) -> TensorValue:
    """Repeats each element of a tensor along an axis.

    This op runs on CPU only; a GPU input raises an error.

    The examples below use an
    input containing ``[[1.0, 2.0], [3.0, 4.0]]``:

    .. code-block:: python

        from max.dtype import DType
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("repeat_interleave_example") as graph:
            input = ops.constant(
                [[1.0, 2.0], [3.0, 4.0]], DType.float32, device=device
            )

            # Repeat each row twice.
            rows = ops.repeat_interleave(input, repeats=2, axis=0)
            # [[1, 2], [1, 2], [3, 4], [3, 4]], shape (4, 2)

            # Repeat row 0 twice and row 1 three times.
            repeats = ops.constant([2, 3], DType.int64, device=device)
            uneven_rows = ops.repeat_interleave(
                input, repeats=repeats, axis=0, out_dim=5
            )
            # [[1, 2], [1, 2], [3, 4], [3, 4], [3, 4]], shape (5, 2)
            graph.output(rows, uneven_rows)

    Args:
        x: The input tensor.
        repeats: The number of times to repeat each element. Pass either a
            positive integer or a rank-0/rank-1 integer ``TensorValue``.
        axis: The axis to repeat along. If ``None`` (the default), the input is
            flattened first.
        out_dim: The output size along ``axis``. Required when ``repeats`` is a
            ``TensorValue``.

    Returns:
        A ``TensorValue`` representing the input with its elements interleaved.

    Raises:
        ValueError: If ``repeats`` is non-positive, if ``axis`` is out of range,
            or if the input is on a GPU device.
    """
    x = TensorValue(x)

    if x.device == DeviceRef.GPU():
        raise ValueError("repeat_interleave is not supported on GPU")

    if axis is not None and not -x.rank <= axis < x.rank:
        raise ValueError(
            f"repeat_interleave: {axis=} out of bounds for {x.rank=}"
        )

    # For compatibility with Torch, if `axis` is not passed, flatten the input array and return a flat array.
    if axis is None:
        x = x.reshape([-1])
        axis = 0

    if axis < 0:
        axis += x.rank

    repeats, inferred_size = _promote_repeats(repeats, x.shape[axis], out_dim)

    result_shape = Shape(x.shape)

    if inferred_size is None:
        raise ValueError("out_dim must be provided for TensorValue repeats")

    # Try to infer the output shape if the multiplier along the axis
    # is statically known, otherwise use the provided out_dim.
    result_shape[axis] = inferred_size

    axis_val = constant(axis, DType.int64, DeviceRef.CPU())

    output = custom(
        "repeat_interleave",
        device=x.device,
        values=[x, repeats, axis_val],
        out_types=[TensorType(x.dtype, result_shape, device=x.device)],
    )

    return output[0].tensor
