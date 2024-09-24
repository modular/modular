# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Integration tests for mutable ops."""

import pytest
import torch
import numpy as np
from max.dtype import DType
from max.graph import Graph, TensorType, BufferType
from max.graph.ops import load_buffer, store_in_buffer
from max.engine import InferenceSession


def torch_add100(x):
    return torch.add(x, 100)


def torch_multiply(x):
    return torch.mul(x, x)


def torch_add_relu(x):
    relu = torch.nn.ReLU()
    return relu(torch.add(x, 100))


def zeros(shape, dtype):
    return np.zeros([int(d) for d in shape]).astype(dtype.to_numpy())


def ones(shape, dtype):
    return np.ones([int(d) for d in shape]).astype(dtype.to_numpy())


@pytest.fixture(
    params=[
        BufferType(DType.float32, [100, 100]),
        BufferType(DType.float32, [10, 10]),
        BufferType(DType.float32, [100, 40]),
    ]
)
def buffer_type(request):
    return request.param


@pytest.fixture(
    params=[
        TensorType(DType.float32, [10, 10]),
        TensorType(DType.float32, [1, 1]),
        TensorType(DType.float32, [10, 2]),
        TensorType(DType.float32, [5, 5]),
    ]
)
def tensor_type(request):
    return request.param


@pytest.fixture
def buffer_graph(buffer_type) -> Graph:
    graph = Graph("buffer", input_types=[buffer_type])
    return graph


@pytest.fixture
def buffer_tensor_graph(tensor_type, buffer_type) -> Graph:
    graph = Graph(
        "buffer_tensor",
        input_types=[
            tensor_type,
            buffer_type,
        ],
    )
    return graph


def test_load_mutate_store(buffer_graph: Graph, session: InferenceSession):
    with buffer_graph as graph:
        buf = graph.inputs[0]
        x = load_buffer(buf)
        x = x + 100
        store_in_buffer(x, buf)
        graph.output()
        graph._mlir_op.verify()
        compiled = session.load(graph)
        input = zeros(buf.shape, buf.dtype)
        expected = torch_add100(torch.from_numpy(input))
        compiled.execute(input)
    assert np.allclose(input, expected)


def test_store_slice(buffer_tensor_graph: Graph, session: InferenceSession):
    with buffer_tensor_graph as graph:
        tensor = graph.inputs[0]
        buffer = graph.inputs[1]

        load_buffer(buffer)

        buf_idx = [(slice(0, int(d)), "out_dim") for d in tensor.shape]
        # All stores are explicit. This syntax uses store_slice.
        buffer[*buf_idx] = tensor * tensor
        graph.output()

        compiled = session.load(graph)
        input_tensor = ones(tensor.shape, tensor.dtype) + 2
        input_buffer = zeros(buffer.shape, buffer.dtype)
        compiled.execute(input_tensor, input_buffer)

        expected = zeros(buffer.shape, buffer.dtype)
        expected[
            : input_tensor.shape[0], : input_tensor.shape[1]
        ] = torch_multiply(torch.from_numpy(input_tensor))
        assert np.allclose(input_buffer, expected)
