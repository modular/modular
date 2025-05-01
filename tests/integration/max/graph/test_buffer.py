# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Integration tests for mutable ops."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from max.driver import Tensor, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    ops,
)
from max.graph.ops import buffer_load, buffer_store


@pytest.fixture
def custom_ops_path() -> Path:
    return Path(os.environ["CUSTOM_OPS_PATH"])


def torch_add_n(x, n):
    return torch.add(x, n)


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
        BufferType(DType.float32, [100, 100], device=DeviceRef.CPU()),
        BufferType(DType.float32, [10, 10], device=DeviceRef.CPU()),
        BufferType(DType.float32, [100, 40], device=DeviceRef.CPU()),
    ]
)
def buffer_type(request):
    return request.param


@pytest.fixture(
    params=[
        TensorType(DType.float32, [10, 10], device=DeviceRef.CPU()),
        TensorType(DType.float32, [1, 1], device=DeviceRef.CPU()),
        TensorType(DType.float32, [10, 2], device=DeviceRef.CPU()),
        TensorType(DType.float32, [5, 5], device=DeviceRef.CPU()),
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


@pytest.mark.skipif(
    accelerator_count() > 0,
    reason="TODO(GEX-2137): Crashing on gpu",
)
@pytest.mark.parametrize("n", [-9, 9, 100])
def test_load_mutate_store(n, buffer_graph: Graph, session: InferenceSession):
    with buffer_graph as graph:
        input_buffer = graph.inputs[0].buffer
        x = buffer_load(input_buffer)
        x = x + n
        buffer_store(input_buffer, x)
        graph.output()
        graph._mlir_op.verify()
        compiled = session.load(graph)
    input = Tensor.from_numpy(zeros(input_buffer.shape, input_buffer.dtype)).to(
        compiled.input_devices[0]
    )
    expected = torch_add_n(torch.from_dlpack(input), n)
    compiled.execute(input)
    assert np.allclose(input.to_numpy(), expected.cpu().numpy())


@pytest.mark.skipif(
    accelerator_count() > 0,
    reason="TODO(GEX-2137): Crashing on gpu",
)
@pytest.mark.parametrize("n", [-9, 9, 100])
def test_load_mutate_store_ellipsis(
    n, buffer_graph: Graph, session: InferenceSession
):
    with buffer_graph as graph:
        input_buffer = graph.inputs[0].buffer
        input_buffer[...] = input_buffer[...] + n
        graph.output()
        graph._mlir_op.verify()
        compiled = session.load(graph)
    input = Tensor.from_numpy(zeros(input_buffer.shape, input_buffer.dtype)).to(
        compiled.input_devices[0]
    )
    expected = torch_add_n(torch.from_dlpack(input), n)
    compiled.execute(input)
    assert np.allclose(input.to_numpy(), expected.cpu().numpy())


@pytest.mark.skipif(
    accelerator_count() > 0,
    reason="TODO(GEX-2137): Crashing on gpu",
)
@pytest.mark.parametrize("n", [-9, 9, 100])
def test_store_slice_load_slice(
    n, buffer_tensor_graph: Graph, session: InferenceSession
):
    with buffer_tensor_graph as graph:
        tensor = graph.inputs[0].tensor
        buffer = graph.inputs[1].buffer

        buf_idx = [(slice(0, int(d)), d) for d in tensor.shape]
        y = tensor * tensor
        # Store slice.
        buffer[*buf_idx] = y + buffer[*buf_idx]

        graph.output()

        compiled_model = session.load(graph)
    input_tensor = Tensor.from_numpy(ones(tensor.shape, tensor.dtype) + n).to(
        compiled_model.input_devices[0]
    )
    input_buffer = Tensor.from_numpy(zeros(buffer.shape, buffer.dtype) + n).to(
        compiled_model.input_devices[1]
    )
    compiled_model.execute(input_tensor, input_buffer)

    expected = zeros(buffer.shape, buffer.dtype) + n
    expected[: input_tensor.shape[0], : input_tensor.shape[1]] = (
        torch_multiply(torch.from_dlpack(input_tensor).cpu())
        + expected[: input_tensor.shape[0], : input_tensor.shape[1]]
    )
    assert np.allclose(input_buffer.to_numpy(), expected)


@pytest.mark.skipif(
    accelerator_count() > 0,
    reason="TODO(GEX-2136): Graph generating erroneous transfer to cpu for buffer",
)
def test_inplace_user_supplied(custom_ops_path, session: InferenceSession):
    bt = BufferType(DType.float32, [2, 2], device=DeviceRef.CPU())

    with Graph(
        "basic", input_types=[bt], custom_extensions=[custom_ops_path]
    ) as graph:
        buffer: BufferValue = graph.inputs[0].buffer

        # this custom op is equivalent to buffer[0,0] += 1
        ops.inplace_custom("mutable_test_op", values=[buffer])
        ops.inplace_custom("mutable_test_op", values=[buffer])
        buffer[...] = ops.negate(buffer[...])

        graph.output()

    rawbuffer = torch.ones((2, 2), dtype=torch.float32)
    if accelerator_count() > 0:
        rawbuffer = rawbuffer.cuda()

    model = session.load(graph)
    model.execute(Tensor.from_dlpack(rawbuffer))

    actual = np.array([[3, 1], [1, 1]], dtype=np.float32) * -1

    np.testing.assert_equal(rawbuffer.cpu().numpy(), actual)


@pytest.mark.skipif(
    accelerator_count() > 0,
    reason="TODO(GEX-2136): Graph generating erroneous transfer to cpu for buffer",
)
def test_variadic_buffer_handling(
    custom_ops_path: Path, session: InferenceSession
) -> None:
    """Test custom op with variadic buffer inputs."""

    # Build, compile, and execute.
    model = session.load(
        Graph(
            "variadic_buffer_test",
            forward=lambda x, y: ops.inplace_custom(
                "reduce_buffers",
                values=[x, y],
                out_types=[
                    TensorType(DType.float32, [1], device=DeviceRef.CPU())
                ],
            ),
            input_types=[
                BufferType(DType.float32, [2], device=DeviceRef.CPU()),
                BufferType(DType.float32, [2], device=DeviceRef.CPU()),
            ],
            custom_extensions=[custom_ops_path],
        ),
    )
    in1 = Tensor.from_numpy(np.arange(2, dtype=np.float32)).to(
        model.input_devices[0]
    )
    in2 = Tensor.from_numpy(np.arange(2, dtype=np.float32)).to(
        model.input_devices[1]
    )
    output = model.execute(in1, in2)[0]
    assert isinstance(output, Tensor)


def test_inplace_custom(custom_ops_path: Path) -> None:
    tensor_type = TensorType(DType.float32, shape=[4], device=DeviceRef.CPU())
    buffer_type = BufferType(DType.float32, shape=[4], device=DeviceRef.CPU())
    with Graph(
        "buffer_load",
        input_types=[buffer_type, tensor_type],
        custom_extensions=[custom_ops_path],
    ) as graph:
        buffer: BufferValue = graph.inputs[0].buffer
        tensor: TensorValue = graph.inputs[1].tensor

        chain_0 = graph._current_chain

        ops.inplace_custom("foo", values=[buffer])
        chain_1 = graph._current_chain

        ops.buffer_store(buffer, tensor)
        chain_2 = graph._current_chain

        ops.inplace_custom("bar", values=[buffer])
        chain_3 = graph._current_chain

        with pytest.raises(TypeError):
            ops.inplace_custom("baz", values=[tensor])

        graph.output()

        assert chain_0 != chain_1
        assert chain_1 != chain_2
        assert chain_2 != chain_3

        assert 'mo.custom {symbol = "foo"}' in str(graph)
        assert 'mo.custom {symbol = "bar"}' in str(graph)


def test_forward_inplace_custom(custom_ops_path: Path) -> None:
    """Tests that returning the result of an `inplace_custom` op works."""
    M = 42
    N = 37
    graph = Graph(
        "foo",
        forward=lambda x: ops.inplace_custom("foo", values=[x]),
        input_types=[
            BufferType(
                dtype=DType.float32, shape=(M, N), device=DeviceRef.GPU()
            )
        ],
        custom_extensions=[custom_ops_path],
    )


def test_custom_buffer_error(custom_ops_path: Path) -> None:
    """Test that we get an error for passing unchained buffers to custom ops."""
    with Graph(
        "custom_buffer_error",
        input_types=[
            BufferType(DType.float32, shape=[42], device=DeviceRef.CPU())
        ],
        custom_extensions=[custom_ops_path],
    ) as graph:
        buffer = graph.inputs[0]

        with pytest.raises(
            TypeError,
            match=(
                "custom ops that take buffers or opaque values to do in-place "
                "updates should use ops.inplace_custom instead"
            ),
        ):
            _ = ops.custom(
                "bar",
                values=[buffer],
                out_types=[
                    TensorType(DType.uint32, shape=[], device=DeviceRef.CPU())
                ],
            )[0]

        graph.output()
