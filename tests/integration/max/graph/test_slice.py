# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import operator

import numpy as np
import pytest
from max.driver import Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, StaticDim, TensorType


@pytest.mark.parametrize(
    ("tensor_type", "indices"),
    [
        # x[1:]
        (TensorType(DType.float32, shape=["dim0"]), (slice(1, None),)),
        (TensorType(DType.float32, shape=["dim0", "dim1"]), (slice(1, None),)),
        # x[:-1]
        (TensorType(DType.float32, shape=["dim0"]), (slice(None, -1))),
        # x[-1:]
        (TensorType(DType.float32, shape=["dim0"]), (slice(-1, None))),
        # x[::2]
        (TensorType(DType.float32, shape=["dim0"]), (slice(None, None, 2),)),
        # x[::-1]
        # TODO(AIPIPE-109): allow negative step after improving rmo.slice.
        # (TensorType(DType.float32, shape=["dim0"]), (slice(None, None, -1),)),
        # x[:, None, :]
        (
            TensorType(DType.float32, shape=["dim0", "dim1"]),
            (slice(None), None, slice(None)),
        ),
        # x[..., None]
        (TensorType(DType.float32, shape=["dim0", "dim1"]), (Ellipsis, None)),
        # x[..., 1]
        (
            TensorType(DType.float32, shape=["dim0", "dim1", "dim2"]),
            (Ellipsis, 1),
        ),
        # x[Ellipsis, 1:]
        (
            TensorType(DType.float32, shape=["dim0", "dim1"]),
            (Ellipsis, slice(1, None)),
        ),
        # x[1, ..., ::-1]
        # TODO(AIPIPE-109): allow negative step after improving rmo.slice.
        # (
        #     TensorType(DType.float32, shape=["dim0", "dim1", "dim2"]),
        #     (1, Ellipsis, slice(None, None, -1)),
        # ),
        # x[:, -1]
        (
            TensorType(DType.float32, shape=["dim0", "dim1", "dim2"]),
            (slice(None), -1),
        ),
    ],
)
def test_slice_numpy(
    session: InferenceSession, tensor_type: TensorType, indices: tuple[slice]
) -> None:
    """Tests end-to-end slice lowering and execution."""
    graph = Graph(
        "slice",
        forward=operator.itemgetter(indices),
        input_types=[tensor_type],
    )

    # Compile and execute the slice graph.
    model = session.load(graph)

    # Compute a random input with shape compatible with tensor_type.
    input_shape = [
        idx * 3 + 7 if not isinstance(dim, StaticDim) else dim.dim
        for idx, dim in enumerate(tensor_type.shape)
    ]
    input_array = np.random.randn(*input_shape).astype(
        tensor_type.dtype.to_numpy()
    )

    # Run the slice graph.
    sliced = model.execute(
        Tensor.from_numpy(input_array).to(model.input_devices[0])
    )[0].to_numpy()

    # Verify that the max.graph slicing matches NumPy.
    expected = input_array[indices]
    np.testing.assert_array_equal(sliced, expected)
