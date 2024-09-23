# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
import pytest
import torch
from conftest import given_input_types
from max.driver import Tensor
from max.dtype import DType
from max.graph import Graph, TensorType


def torch_dtype(dtype: DType):
    return getattr(torch, dtype.name)


@pytest.mark.skip("TODO(GRA-1047)")
@pytest.mark.parametrize(
    "input_type,output_type",
    [(TensorType(DType.float32, ["dim"]), DType.bfloat16)],
)
def test_cast(session, input_type: TensorType, output_type: DType):
    with Graph("cast", input_types=[input_type]) as graph:
        graph.output(graph.inputs[0].cast(output_type))
    model = session.load(graph)

    @given_input_types((input.type for input in graph.inputs))
    def test_correctness(inputs):
        expected = torch.tensor(inputs[0]).type(torch_dtype(output_type))
        result = model.execute(Tensor.from_numpy(inputs[0]))[0]
        print(expected, torch.from_dlpack(result))
        torch.testing.assert_close(torch.from_dlpack(result), expected)

    test_correctness()
