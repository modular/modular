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
"""Op implementation for repeat_interleave."""

from typing import Optional, Union

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
    repeats: Union[int, TensorValue],
    input_dim: Dim,
    out_dim: Optional[DimLike],
) -> tuple[TensorValue, Optional[Dim]]:
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
    repeats: Union[int, TensorValue],
    axis: Optional[int] = None,
    out_dim: Optional[DimLike] = None,
) -> TensorValue:
    """Repeats elements of a tensor along the given dimension.

    Modeled after :obj:`torch.repeat_interleave`, with the constraint that

    For example, given ``repeats=2`` and the following input:

    .. code-block:: python

        # Input tensor with shape (2, 2)
        input = TensorValue(x)  # Contains [[1.0, 2.0], [3.0, 4.0]]

    ``repeat_interleave`` with ``axis=0``:

    .. code-block:: python

        # Output tensor with shape (4, 2)
        output = repeat_interleave(input, repeats=2, axis=0)
        # Contains [[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0]]

    ``repeat_interleave`` with ``axis=1``:

    .. code-block:: python

        # Output tensor with shape (2, 4)
        output = repeat_interleave(input, repeats=2, axis=1)
        # Contains [[1.0, 1.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0]]

    ``repeat_interleave`` with ``axis=None`` (the default):

    ``repeat_interleave`` with ``repeats=[2, 3]`` and ``axis=0``:

    .. code-block:: python

        repeat_value = TensorValue([2, 3])

        # Output tensor with shape (5, 2)
        output = repeat_interleave(input, repeats=repeat_value, axis=0)
        # Contains [[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0], [3.0, 4.0]]

    .. code-block:: python

        # Output tensor with shape (8,)
        output = repeat_interleave(input, repeats=2)  # axis = None
        # Contains [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]

    Args:
        x:
            The input tensor.
        repeats:
            The number of repetitions for each element.
        axis:
            The dimension along which to repeat values. If axis is not
            specified or None (the default), flatten the input array
            and repeat the flattened values.

    Returns:
        A symbolic tensor with the elements interleaved.

    Raises:
        ValueError: If ``repeats`` non-positive or if ``axis`` is out of range.
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

    axis_val = constant(np.array(axis), DType.int64, DeviceRef.CPU())

    output = custom(
        "repeat_interleave",
        device=x.device,
        values=[x, repeats, axis_val],
        out_types=[TensorType(x.dtype, result_shape, device=x.device)],
    )

    return output[0].tensor
