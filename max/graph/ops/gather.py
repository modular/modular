# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for gather."""

from max import mlir
from max.dtype import DType
from max.mlir.dialects import rmo

from ..graph import Graph
from ..type import DeviceRef, StaticDim, TensorType
from ..value import TensorValue, TensorValueLike
from .constant import constant


def gather(
    input: TensorValueLike, indices: TensorValueLike, axis: int = -1
) -> TensorValue:
    """
    Selects elements out of an input tensor by index.

    Args:
        input: The input symbolic tensor to select elements from.
        indices: A symbolic tensor of index values to use for selection.
        axis: The dimension which ``indices`` indexes from ``input``.
            If negative, indexes relative to the end of the input tensor.
            For instance, ``gather(input, indices, axis=-1)`` will index
            against the last dimension of ``input``.

    Returns:
        A new symbolic tensor representing the result of the gather operation.
    """
    input, indices = TensorValue(input), TensorValue(indices)
    shape = input.shape
    output_shape = [*shape[:axis], *indices.shape, *shape[axis + 1 :]]
    return Graph.current._add_op(
        rmo.mo_gather,
        TensorType(
            input.dtype,
            output_shape,
            # Prefer indices device if input device unset since they're equal.
            input.device if input.device else indices.device,
        ).to_mlir(),
        input,
        indices,
        constant(axis, DType.int64, DeviceRef.CPU()),
    )[0].tensor


def gather_nd(
    input: TensorValueLike,
    indices: TensorValueLike,
    batch_dims: int = 0,
) -> TensorValue:
    """Selects elements out of an input tensor by index.

    Examples:

    >>> input_shape = ["a", "b", "c", "d", "e"]
    >>> indices_shape = ["a", "f", 3]
    >>> input_type = TensorType(DType.bfloat16, input_shape)
    >>> indices_type = TensorType(DType.int32, indices_shape)
    >>> with Graph("gather_nd", input_types=[input_type, indices_type]) as graph:
    ...     input, indices = graph.inputs
    ...     gathered = ops.gather_nd(input, indices, batch_dims=1)
    ...     print(gathered.type)
    TensorType(dtype=DType.bfloat16, shape=["a", "f", "e"])

    In this example
    - batch_dims is 1, so there's 1 shared dimension at the beginning
    - indices has an additional dimension "f" which
    - the last dimension of indices is the index vector; values in
        this vector are interpreted to be indicies into "b", "c", and "d"
    - since batch_dims (1) + index size (3) < input.rank (5), the remaining
        dimensions (in this case "e") are sliced into the output as features

    Args:
        input: The input symbolic tensor to select elements from.
        indices: A symbolic tensor of index values to use for selection.
            The last dimension of this tensor must be static. This dimension
            will be used to index or slice into `input` immediately following
            `batch_dims` initial dimensions. The size of this index dimension
            is the number of dimensions it specifies.
        batch_dims: The number of leading batch dimensions shared by
            `input` and `indices`; 0 by default. `input` and `indices` must
            _exactly_ match up to their first `batch_dims` dimensions. This
            function does not broadcast!

    Returns:
        A new symbolic tensor representing the result of the gather operation.
        The output will have the same DType as `input`, and will have shape
        depending on the inputs, in this order
            - `input.shape[:batch_dims]` -- think of this as the "broadcast"
                dimensions (though note that this function does not broadcast).
                These dimensions must be identical between `input` and `indices`.
            - `indices.shape[batch_dims:-1]` -- the "gather" dimensions; this
                allows multi-dimensional tensors of indices. The last dimension
                is the index vector.
            - `input.shape[batch_dims + indices.shape[-1]:]` -- the "slice"
                dimensions. If `batch_dims` < `input.rank - indices.shape[-1]`
                (again, this last is the index vector), then any following
                dimensions of the inputs are taken entirely as though slicing.
    """
    input, indices = TensorValue(input), TensorValue(indices)
    if not isinstance(indices.shape[-1], StaticDim):
        raise ValueError(f"index last dimension must be static: {indices=}")
    index_size = int(indices.shape[-1])

    if batch_dims < 0:
        raise ValueError(f"batch_dims must be non-negative: {batch_dims=}")
    if batch_dims > indices.rank - 1:
        raise ValueError(f"Not enough dims in {indices=} for {batch_dims=}")
    if batch_dims + index_size > input.rank:
        raise ValueError(
            f"Not enough dims in {input=}: {batch_dims=},"
            f" {index_size=} ({indices=})"
        )

    if input.shape[:batch_dims] != indices.shape[:batch_dims]:
        raise ValueError(
            f"{input=} and {indices=} must match up to {batch_dims=}"
        )

    if indices.dtype.is_float():
        raise ValueError(f"{indices.dtype=} must be an integer type.")

    output_shape = [
        *input.shape[:batch_dims],
        *indices.shape[batch_dims:-1],
        *input.shape[batch_dims + index_size :],
    ]

    return Graph.current._add_op(
        rmo.mo_gather_nd,
        TensorType(input.dtype, output_shape, input.device).to_mlir(),
        input,
        indices,
        mlir.IntegerAttr.get(mlir.IndexType.get(), batch_dims),
    )[0].tensor
