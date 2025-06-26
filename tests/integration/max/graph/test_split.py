# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import numpy as np
import pytest
from max.driver import Tensor
from max.dtype import DType
from max.graph import DeviceRef, Graph, Shape, TensorType, ops


@pytest.mark.parametrize(
    "input_shape,split_sizes,axis",
    [
        ([15], [14, 1], 0),
        ([5, 10, 20], [3, 2], 0),
        ([5, 10, 20], [2, 3, 2, 1, 2], 1),
        ([5, 10, 20], [4, 6, 4, 2, 4], 2),
    ],
)
def test_split(session, input_shape: Shape, split_sizes: list[int], axis: int):
    input = np.random.uniform(size=input_shape).astype(np.float32)

    with Graph(
        "split",
        input_types=[
            TensorType(DType.float32, input_shape, device=DeviceRef.CPU())
        ],
    ) as graph:
        output = ops.split(graph.inputs[0].tensor, split_sizes, axis)
        graph.output(*output)

    model = session.load(graph)
    result = model.execute(Tensor.from_numpy(input).to(model.input_devices[0]))
    np_split_indices = []
    end_index = 0
    for n in split_sizes[:-1]:
        end_index += n
        np_split_indices.append(end_index)

    expected_results = np.split(input, np_split_indices, axis)
    assert len(result) == len(expected_results)
    for actual, expected in zip(result, expected_results):
        np.testing.assert_equal(actual.to_numpy(), expected)
