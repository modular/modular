# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for argsort."""

from max.dtype import DType

from ..value import TensorType, TensorValue
from .custom import custom


def argsort(x: TensorValue, ascending: bool = True) -> TensorValue:
    """Returns the indices that would sort a tensor.

    This function returns the indices that would sort the input tensor along
    its first dimension. The returned indices are of type int64.

    Args:
        x: Input tensor to be sorted.
        ascending: If True (default), sort in ascending order. If False, sort in
            descending order.

    Returns:
        A tensor of indices of the same shape as the input tensor.
    """
    if x.rank != 1:
        raise ValueError("argsort only implemented for input tensors of rank 1")
    return custom(
        "mx.argsort",
        x.device,
        [x],
        out_types=[
            TensorType(dtype=DType.int64, shape=x.shape, device=x.device)
        ],
        parameters={
            "ascending": ascending,
        },
    )[0].tensor
