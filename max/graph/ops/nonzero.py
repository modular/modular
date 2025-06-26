# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for nonzero."""

from max.dtype import DType
from max.mlir.dialects import rmo

from ..graph import Graph
from ..type import DeviceRef, DimLike, TensorType
from ..value import TensorValue, TensorValueLike


def nonzero(x: TensorValueLike, out_dim: DimLike) -> TensorValue:
    """Returns the indices of all nozero elements in a tensor.

    Returns a tensor of indices of the nonzero values in the given tensor. The
    return value is a 2D tensor of shape ``[out_dim x rank_in]``, where
    out_dim is the number of nonzero elements in the input tensor, and
    rank_in is the rank of the input tensor. Indices are generated in
    row-major order.

    Args:
        x: The input symbolic tensor.
        out_dim:
            The newly generated dimension that is sized for the number of
            nonzero elements.

    Returns:
        A symbolic tensor of indices
    """
    x = TensorValue(x)

    if x.rank == 0:
        raise ValueError("Scalar inputs not supported")

    # TODO(GEX-2041): Add support for GPU kernel for nonzero and remove manual transfers
    original_device = x.type.device
    x = x.to(DeviceRef.CPU())
    answer = Graph.current._add_op(
        rmo.mo_arg_nonzero,
        TensorType(
            dtype=DType.int64, shape=[out_dim, x.rank], device=x.device
        ).to_mlir(),
        TensorValue(x),
    )[0].tensor
    return answer.to(original_device)
