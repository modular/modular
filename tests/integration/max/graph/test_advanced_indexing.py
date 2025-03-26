# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import math
from dataclasses import dataclass
from typing import List

import numpy as np
import pytest
import torch
from max.dtype import DType
from max.graph import Graph, TensorType, ops

# Test component functions for numpy-like advanced indexing

# The rank of input tensors
RANK = 5

# The dimension sizes of the input tensor
INPUT_DIM_LENGTH = 5


@dataclass
class StandardInputAndIndexTensors:
    """Standardized data generator for advanced indexing unittests.

    This mainly abstracts away generation and some shape inference logic.
    The names `input`, `index`, and `update` correspond to something like:

        input_tensor[:, indexing_tensor1, indexing_tensor2, :, :] ( = optional_update)
    """

    input_tensor_shape: List[int]
    index_tensor_shape: List[int]

    # We expect indexing tensors to be contiguous in the axis they operate on.
    # This represents the first axis
    start_axis: int
    num_indexing_tensors: int

    def max_input_tensor_type(self, dtype=DType.float32) -> TensorType:
        return TensorType(dtype, self.input_tensor_shape)

    def max_index_tensor_type(self, dtype=DType.int32) -> TensorType:
        return TensorType(dtype, self.index_tensor_shape)

    def output_tensor_shape(self) -> List[int]:
        return (
            self.input_tensor_shape[: self.start_axis]
            + self.index_tensor_shape
            + self.input_tensor_shape[
                self.start_axis + self.num_indexing_tensors :
            ]
        )

    def max_output_tensor_type(self, dtype=DType.float32) -> TensorType:
        return TensorType(dtype, self.output_tensor_shape())

    def max_output_tensor_type_unknown_shape(
        self, dtype=DType.float32
    ) -> TensorType:
        shape = self.output_tensor_shape()

        # Replace known integer dims with symbolic unknown values.
        # We expect a runtime shape function to populate these values.
        assert len(shape) <= 26, (
            "Logic below only works for rank under 26, "
            "expect weird dimension names otherwise."
        )
        erased_shape = [chr(ord("a") + i) for i in range(len(shape))]
        return TensorType(dtype, erased_shape)

    def max_update_tensor_type(self, dtype=DType.float32) -> TensorType:
        return TensorType(dtype, self.output_tensor_shape())

    def input_tensor(self, dtype=torch.float32) -> torch.Tensor:
        return (
            torch.arange(math.prod(self.input_tensor_shape))
            .reshape(self.input_tensor_shape)
            .type(dtype)
        )

    def index_tensors(self, dtype=torch.int32) -> List[torch.Tensor]:
        # Make the indices slightly different for variety
        result = [
            torch.arange(np.prod(self.index_tensor_shape))
            .reshape(self.index_tensor_shape)
            .type(dtype)
            for _ in range(self.num_indexing_tensors)
        ]
        for i in range(self.num_indexing_tensors):
            result[i] = (result[i] + i) % INPUT_DIM_LENGTH
        return result

    def update_tensor(self, dtype=torch.float32) -> torch.Tensor:
        return (
            torch.arange(np.prod(self.output_tensor_shape()))
            .reshape(self.output_tensor_shape())
            .type(dtype)
            * -1000
        )


@pytest.mark.parametrize("start_axis", [0, 2])
@pytest.mark.parametrize("num_indexing_tensors", [1, 3])
@pytest.mark.parametrize("indexing_tensor_rank", [1, 2])
@pytest.mark.parametrize("use_unknown_shape", [True, False])
def test_advanced_indexing_get_item(
    session,
    start_axis,
    num_indexing_tensors,
    indexing_tensor_rank,
    use_unknown_shape,
):
    data_generator = StandardInputAndIndexTensors(
        input_tensor_shape=[INPUT_DIM_LENGTH] * RANK,
        index_tensor_shape=list(range(3, 3 + indexing_tensor_rank)),
        start_axis=start_axis,
        num_indexing_tensors=num_indexing_tensors,
    )

    input_types = [data_generator.max_input_tensor_type()] + [
        data_generator.max_index_tensor_type()
    ] * num_indexing_tensors
    with Graph(
        f"advanced_indexing_get_item|start_axis={start_axis}|num_indexing_tensors={num_indexing_tensors}",
        input_types=input_types,
    ) as graph:
        out = ops.custom(
            "advanced_indexing_getitem",
            values=list(graph.inputs),
            out_types=[
                data_generator.max_output_tensor_type()
                if use_unknown_shape
                else data_generator.max_output_tensor_type_unknown_shape()
            ],
            parameters={"start_axis": start_axis},
        )[0]
        graph.output(out.cast(DType.float32))

    model = session.load(graph)

    # Generate input, numbers 0, 1, 2, ... N
    input_tensor = data_generator.input_tensor()
    index_tensors = data_generator.index_tensors()

    # Generate expected. Analogous to something like
    # arr[:, :, index1, index2, :]
    torch_slice = [slice(None, None, None)] * RANK
    for i in range(num_indexing_tensors):
        torch_slice[i + start_axis] = index_tensors[i]
    expected = input_tensor[*torch_slice].numpy()

    graph_inputs = [input_tensor] + index_tensors
    actual = model(*graph_inputs)[0].to_numpy()
    np.testing.assert_equal(actual, expected)


@pytest.mark.parametrize("start_axis", [0, 2])
@pytest.mark.parametrize("num_indexing_tensors", [1, 3])
@pytest.mark.parametrize("indexing_tensor_rank", [1, 2])
def test_advanced_indexing_set_item(
    session, start_axis, num_indexing_tensors, indexing_tensor_rank
):
    # NOTE: it is possible to have multiple redundant indices which map to same memory.
    # This results in undefined behavior in assignment. Make sure the index_tensors are small
    # enough where this is not a problem to keep test reproducible.
    data_generator = StandardInputAndIndexTensors(
        input_tensor_shape=[INPUT_DIM_LENGTH] * RANK,
        index_tensor_shape=list(range(1, 1 + indexing_tensor_rank)),
        start_axis=start_axis,
        num_indexing_tensors=num_indexing_tensors,
    )

    input_types = (
        [data_generator.max_input_tensor_type()]
        + [data_generator.max_update_tensor_type()]
        + [data_generator.max_index_tensor_type()] * num_indexing_tensors
    )
    with Graph(
        f"advanced_indexing_set_item|start_axis={start_axis}|num_indexing_tensors={num_indexing_tensors}",
        input_types=input_types,
    ) as graph:
        out = ops.custom(
            "advanced_indexing_setitem",
            values=list(graph.inputs),
            out_types=[data_generator.max_input_tensor_type()],
            parameters={"start_axis": start_axis},
        )[0]
        graph.output(out.cast(DType.float32))

    model = session.load(graph)

    # Generate input, numbers 0, 1, 2, ... N
    input_tensor = data_generator.input_tensor()
    update_tensor = data_generator.update_tensor()
    index_tensors = data_generator.index_tensors()

    # Generate expected. Analogous to something like
    # arr[:, :, index1, index2, :] = update
    torch_slice = [slice(None, None, None)] * RANK
    for i in range(num_indexing_tensors):
        torch_slice[i + start_axis] = index_tensors[i]

    # PT does it in-place unfortunately, so make a copy
    expected = input_tensor.clone()
    expected[*torch_slice] = update_tensor
    expected = expected.numpy()

    graph_inputs = [input_tensor, update_tensor] + index_tensors
    actual = model(*graph_inputs)[0].to_numpy()
    np.testing.assert_equal(actual, expected)


@pytest.mark.parametrize("start_axis", [0, 2])
@pytest.mark.parametrize("num_indexing_tensors", [1, 3])
@pytest.mark.parametrize("indexing_tensor_rank", [1, 2])
def test_advanced_indexing_set_item_inplace(
    session, start_axis, num_indexing_tensors, indexing_tensor_rank
):
    # NOTE: it is possible to have multiple redundant indices which map to same memory.
    # This results in undefined behavior in assignment. Make sure the index_tensors are small
    # enough where this is not a problem to keep test reproducible.
    data_generator = StandardInputAndIndexTensors(
        input_tensor_shape=[INPUT_DIM_LENGTH] * RANK,
        index_tensor_shape=list(range(1, 1 + indexing_tensor_rank)),
        start_axis=start_axis,
        num_indexing_tensors=num_indexing_tensors,
    )

    input_types = (
        [data_generator.max_input_tensor_type().as_buffer()]
        + [data_generator.max_update_tensor_type()]
        + [data_generator.max_index_tensor_type()] * num_indexing_tensors
    )
    with Graph(
        f"advanced_indexing_set_item_inplace|start_axis={start_axis}|num_indexing_tensors={num_indexing_tensors}",
        input_types=input_types,
    ) as graph:
        ops.inplace_custom(
            "advanced_indexing_setitem_inplace",
            values=list(graph.inputs),
            out_types=[],
            parameters={"start_axis": start_axis},
        )
        graph.output()

    model = session.load(graph)

    # Generate input, numbers 0, 1, 2, ... N
    input_tensor = data_generator.input_tensor()
    update_tensor = data_generator.update_tensor()
    index_tensors = data_generator.index_tensors()

    # Generate expected. Analogous to something like
    # arr[:, :, index1, index2, :] = update
    torch_slice = [slice(None, None, None)] * RANK
    for i in range(num_indexing_tensors):
        torch_slice[i + start_axis] = index_tensors[i]

    # PT does it in-place unfortunately, so make a copy
    expected = input_tensor.clone()
    expected[*torch_slice] = update_tensor
    expected = expected.numpy()

    # This modifies input_tensor in-place
    graph_inputs = [input_tensor, update_tensor] + index_tensors
    model(*graph_inputs)
    np.testing.assert_equal(input_tensor, expected)
