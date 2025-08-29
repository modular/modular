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


import math
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from max.driver import Tensor, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

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

    input_tensor_shape: list[int]
    index_tensor_shape: list[int]

    # We expect indexing tensors to be contiguous in the axis they operate on.
    # This represents the first axis
    start_axis: int
    num_indexing_tensors: int

    def max_input_tensor_type(self, dtype: DType = DType.float32) -> TensorType:
        return TensorType(
            dtype, self.input_tensor_shape, device=DeviceRef.CPU()
        )

    def max_index_tensor_type(self, dtype: DType = DType.int32) -> TensorType:
        return TensorType(
            dtype, self.index_tensor_shape, device=DeviceRef.CPU()
        )

    def output_tensor_shape(self) -> list[int]:
        return (
            self.input_tensor_shape[: self.start_axis]
            + self.index_tensor_shape
            + self.input_tensor_shape[
                self.start_axis + self.num_indexing_tensors :
            ]
        )

    def max_output_tensor_type(
        self, dtype: DType = DType.float32
    ) -> TensorType:
        return TensorType(
            dtype, self.output_tensor_shape(), device=DeviceRef.CPU()
        )

    def max_output_tensor_type_unknown_shape(
        self, dtype: DType = DType.float32
    ) -> TensorType:
        shape = self.output_tensor_shape()

        # Replace known integer dims with symbolic unknown values.
        # We expect a runtime shape function to populate these values.
        assert len(shape) <= 26, (
            "Logic below only works for rank under 26, "
            "expect weird dimension names otherwise."
        )
        erased_shape = [chr(ord("a") + i) for i in range(len(shape))]
        return TensorType(dtype, erased_shape, device=DeviceRef.CPU())

    def max_update_tensor_type(
        self, dtype: DType = DType.float32
    ) -> TensorType:
        return TensorType(
            dtype, self.output_tensor_shape(), device=DeviceRef.CPU()
        )

    def input_tensor(self, dtype: DType = torch.float32) -> torch.Tensor:
        x = (
            torch.arange(math.prod(self.input_tensor_shape))
            .reshape(self.input_tensor_shape)
            .type(dtype)
        )
        return x

    def index_tensors(self, dtype: DType = torch.int32) -> list[torch.Tensor]:
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

    def update_tensor(self, dtype: DType = torch.float32) -> torch.Tensor:
        x = (
            torch.arange(np.prod(self.output_tensor_shape()))
            .reshape(self.output_tensor_shape())
            .type(dtype)
            * -1000
        )
        if accelerator_count() > 0:
            return x.cuda()
        return x


@pytest.mark.parametrize("start_axis", [0, 2])
@pytest.mark.parametrize("num_indexing_tensors", [1, 3])
@pytest.mark.parametrize("indexing_tensor_rank", [1, 2])
@pytest.mark.parametrize("use_unknown_shape", [True, False])
def test_advanced_indexing_get_item(
    session: InferenceSession,
    start_axis: int,
    num_indexing_tensors: int,
    indexing_tensor_rank: int,
    use_unknown_shape: bool,
) -> None:
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
            device=input_types[0].device,
            values=list(graph.inputs),
            out_types=[
                data_generator.max_output_tensor_type()
                if use_unknown_shape
                else data_generator.max_output_tensor_type_unknown_shape()
            ],
            parameters={"start_axis": start_axis},
        )[0].tensor
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
    expected = input_tensor[*torch_slice].cpu().numpy()

    graph_inputs = [input_tensor] + index_tensors
    actual = model(*graph_inputs)[0]
    assert isinstance(actual, Tensor)
    np.testing.assert_equal(actual.to_numpy(), expected)


@pytest.mark.skipif(
    accelerator_count() > 0,
    reason="TODO(GEX-2132): Some combinations lead to `CUDA call failed: CUDA_ERROR_MISALIGNED_ADDRESS (misaligned address)`",
)
@pytest.mark.parametrize("start_axis", [0, 2])
@pytest.mark.parametrize("num_indexing_tensors", [1, 3])
@pytest.mark.parametrize("indexing_tensor_rank", [1, 2])
def test_advanced_indexing_set_item(
    session: InferenceSession,
    start_axis: int,
    num_indexing_tensors: int,
    indexing_tensor_rank: int,
) -> None:
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
            device=input_types[0].device,
            values=list(graph.inputs),
            out_types=[data_generator.max_input_tensor_type()],
            parameters={"start_axis": start_axis},
        )[0].tensor
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
    expected = expected.cpu().numpy()

    graph_inputs = [input_tensor, update_tensor] + index_tensors
    actual = model(*graph_inputs)[0]
    assert isinstance(actual, Tensor)
    np.testing.assert_equal(actual.to_numpy(), expected)


@pytest.mark.skipif(
    accelerator_count() > 0,
    reason="TODO(GEX-2132): Compilation fails in elaboration",
)
@pytest.mark.parametrize("start_axis", [0, 2])
@pytest.mark.parametrize("num_indexing_tensors", [1, 3])
@pytest.mark.parametrize("indexing_tensor_rank", [1, 2])
def test_advanced_indexing_set_item_inplace(
    session: InferenceSession,
    start_axis: int,
    num_indexing_tensors: int,
    indexing_tensor_rank: int,
) -> None:
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
            device=input_types[0].device,
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
    expected = expected.cpu().numpy()

    # This modifies input_tensor in-place
    graph_inputs = [input_tensor, update_tensor] + index_tensors
    model(*graph_inputs)
    np.testing.assert_equal(input_tensor.cpu().numpy(), expected)
