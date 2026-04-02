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
"""Op implementation for scatter."""

from max._core.dialects import kgen, rmo
from max.dtype import DType

from .. import dtype_promotion
from ..dim import DimLike
from ..graph import Graph
from ..type import DeviceRef
from ..value import TensorValue, TensorValueLike
from .constant import constant
from .nonzero import nonzero
from .transfer_to import transfer_to
from .validation import _check_device_placement, assert_same_device


def scatter(
    input: TensorValueLike,
    updates: TensorValueLike,
    indices: TensorValueLike,
    axis: int = -1,
) -> TensorValue:
    """Creates a new symbolic tensor where the updates are written to input according to indices.

    Args:
        input: The input symbolic tensor to write elements to.
        updates: A symbolic tensor of elements to write to input.
        indices: The positions in input to update.
        axis: The axis along which indices indexes into.

    Returns:
        A new symbolic tensor representing the result of the scatter operation.

    Raises:
        ValueError: If ``axis`` is out of range, if dtypes mismatch, if
            ``indices`` dtype is not int32/int64, or if any input is on a
            non-CPU device and
            ``strict_device_placement=DevicePlacementPolicy.Error``.
    """
    input = TensorValue(input)

    if not (-input.rank <= axis < input.rank):
        raise ValueError(
            f"Invalid axis value {axis}. Axis must be in range [-{input.rank}, {input.rank - 1}]"
        )

    updates = TensorValue(updates)
    if input.dtype != updates.dtype:
        raise ValueError(
            f"The input dtype '{input.dtype}' and updates dtype '{updates.dtype}' must match."
        )

    indices = TensorValue(indices)
    if indices.dtype not in [DType.int32, DType.int64]:
        raise ValueError(
            f"Invalid indices dtype: '{indices.dtype}'. Indices must be of type int32 or int64."
        )

    assert_same_device(input=input, updates=updates, indices=indices)
    old_device = input.device if not input.device.is_cpu() else None
    if old_device is not None:
        _check_device_placement("ops.scatter", "TODO(GEX-2197).")
        input = transfer_to(input, DeviceRef.CPU())
        updates = transfer_to(updates, DeviceRef.CPU())
        indices = transfer_to(indices, DeviceRef.CPU())
    # TODO(GEX-2197): Add GPU kernel support for scatter.
    axis_constant = constant(axis, DType.int64, DeviceRef.CPU())

    result = Graph.current._add_op_generated(
        rmo.MoScatterOp,
        input.type,
        input,
        updates,
        indices,
        axis_constant,
        kgen.ParamDeclArrayAttr([]),
    )[0].tensor
    if old_device is not None:
        return transfer_to(result, old_device)
    return result


def scatter_nd(
    input: TensorValueLike,
    updates: TensorValueLike,
    indices: TensorValueLike,
) -> TensorValue:
    """Creates a new symbolic tensor where the updates are scattered into input at specified indices.

    Args:
        input: The input symbolic tensor to write elements to.
        updates: A symbolic tensor of elements to write to input.
        indices: A tensor of indices specifying where to write updates.
            Shape should be [num_updates, rank] for full indexing or
            [num_updates, k] for partial indexing where k < rank.

    Returns:
        A new symbolic tensor representing the result of the scatter_nd operation.
    """
    input = TensorValue(input)
    updates = TensorValue(updates)
    indices = TensorValue(indices)

    if input.dtype != updates.dtype:
        raise ValueError(
            f"The input dtype ({input.dtype}) and updates dtype"
            f" ({updates.dtype}) must match"
        )

    if indices.dtype not in (DType.int32, DType.int64):
        raise ValueError(
            f"Invalid indices dtype: '{indices.dtype}'. Indices must be of type int32 or int64."
        )

    assert_same_device(input=input, updates=updates, indices=indices)

    return Graph.current._add_op_generated(
        rmo.MoScatterNdOp,
        input.type,
        input,
        updates,
        indices,
        kgen.ParamDeclArrayAttr([]),
    )[0].tensor


def scatter_nd_add(
    input: TensorValueLike,
    updates: TensorValueLike,
    indices: TensorValueLike,
) -> TensorValue:
    """Creates a new symbolic tensor by accumulating updates into input at N-D indices.

    Produces an output tensor by scattering slices from updates into a copy
    of input according to N-dimensional index vectors, summing values at
    duplicate index positions.  Each index vector is the last dimension of
    ``indices`` and selects a slice (or scalar) in input.

    Example for ``input.shape = [4, 2]``, ``indices.shape = [3, 1]``
    (1-D partial indexing, writes whole rows):

    .. code-block:: text

        output[indices[i, 0], :] += updates[i, :]

    Args:
        input: The input symbolic tensor to accumulate into.
        updates: A symbolic tensor of values to add.
        indices: An index tensor whose last dimension is the index vector
            length ``k`` (``k <= input.rank``).

    Returns:
        A new symbolic tensor with the same shape and dtype as input.
    """
    input = TensorValue(input)
    updates = TensorValue(updates)
    indices = TensorValue(indices)

    if input.dtype != updates.dtype:
        raise ValueError(
            f"The input dtype ({input.dtype}) and updates dtype"
            f" ({updates.dtype}) must match"
        )

    if indices.dtype not in (DType.int32, DType.int64):
        raise ValueError(
            f"Invalid indices dtype: '{indices.dtype}'."
            " Indices must be of type int32 or int64."
        )

    assert_same_device(input=input, updates=updates, indices=indices)

    return Graph.current._add_op_generated(
        rmo.MoScatterNdAddOp,
        input.type,
        input,
        updates,
        indices,
        kgen.ParamDeclArrayAttr([]),
    )[0].tensor


def scatter_add(
    input: TensorValueLike,
    updates: TensorValueLike,
    indices: TensorValueLike,
    axis: int = -1,
) -> TensorValue:
    """Creates a new symbolic tensor by accumulating updates into input at indices.

    Produces an output tensor by scattering elements from updates into input
    according to indices, summing values at duplicate indices.  For a 2-D
    input with ``axis=0`` the update rule is:

    .. code-block:: text

        output[indices[i][j]][j] += updates[i][j]

    and with ``axis=1``:

    .. code-block:: text

        output[i][indices[i][j]] += updates[i][j]

    Args:
        input: The input symbolic tensor to accumulate into.
        updates: A symbolic tensor of values to add.
        indices: The positions in input to update.
        axis: The axis along which indices indexes into.

    Returns:
        A new symbolic tensor with the same shape and dtype as input.

    Raises:
        ValueError: If ``axis`` is out of range, if dtypes mismatch, if
            ``indices`` dtype is not int32/int64, or if any input is on a
            non-CPU device and
            ``strict_device_placement=DevicePlacementPolicy.Error``.
    """
    input = TensorValue(input)

    if not (-input.rank <= axis < input.rank):
        raise ValueError(
            f"Invalid axis value {axis}. Axis must be in range"
            f" [-{input.rank}, {input.rank - 1}]"
        )

    updates = TensorValue(updates)
    if input.dtype != updates.dtype:
        raise ValueError(
            f"The input dtype '{input.dtype}' and updates dtype"
            f" '{updates.dtype}' must match."
        )

    indices = TensorValue(indices)
    if indices.dtype not in [DType.int32, DType.int64]:
        raise ValueError(
            f"Invalid indices dtype: '{indices.dtype}'."
            " Indices must be of type int32 or int64."
        )

    assert_same_device(input=input, updates=updates, indices=indices)
    old_device = input.device if not input.device.is_cpu() else None
    if old_device is not None:
        _check_device_placement("ops.scatter_add", "TODO(GEX-2197).")
        input = transfer_to(input, DeviceRef.CPU())
        updates = transfer_to(updates, DeviceRef.CPU())
        indices = transfer_to(indices, DeviceRef.CPU())
    # TODO(GEX-2197): Add GPU kernel support for scatter_add.
    axis_constant = constant(axis, DType.int64, DeviceRef.CPU())

    result = Graph.current._add_op_generated(
        rmo.MoScatterAddOp,
        input.type,
        input,
        updates,
        indices,
        axis_constant,
        kgen.ParamDeclArrayAttr([]),
    )[0].tensor
    if old_device is not None:
        return transfer_to(result, old_device)
    return result


def masked_scatter(
    input: TensorValueLike,
    mask: TensorValueLike,
    updates: TensorValueLike,
    out_dim: DimLike,
) -> TensorValue:
    """Creates a new symbolic tensor where the updates are written to input where mask is true.

    Args:
        input: The input symbolic tensor to write elements to.
        mask: A symbolic tensor of boolean values to update.
        updates: A symbolic tensor of elements to write to input.
        out_dim: The new data-dependent dimension.

    Returns:
        A new symbolic tensor representing the result of the masked_scatter operation.
    """
    input, updates = TensorValue(input), TensorValue(updates)
    mask = dtype_promotion._promote_to_strong(
        mask, DType.bool, input.type.device or DeviceRef.CPU()
    )

    if input.dtype != updates.dtype:
        raise ValueError(
            f"The input dtype ({input.dtype}) and updates dtype"
            f" ({updates.dtype}) must match"
        )

    mask = mask.broadcast_to(input.shape)
    indices = nonzero(mask, out_dim)

    updates = updates.flatten()

    return scatter_nd(input, updates, indices)
