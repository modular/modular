# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn import LinearV1
from max.nn.sequential import Sequential
from modular_graph_test import are_all_tensor_values, modular_graph_test


def test_sequential__one_linear_layer(session) -> None:
    dtype = DType.float32

    with Graph(
        "sequential_two_linear_layers",
        input_types=[
            TensorType(dtype, ["batch", "input_dim"], device=DeviceRef.CPU()),
            TensorType(
                dtype, ["hidden_dim", "input_dim"], device=DeviceRef.CPU()
            ),
        ],
    ) as graph:
        assert are_all_tensor_values(graph.inputs)
        x, w1 = graph.inputs
        one_layer = Sequential([LinearV1(w1)])

        graph.output(one_layer(x))

        @modular_graph_test(
            session,
            graph,
            max_magnitude=1 / 64,
        )
        def test_correctness(execute, inputs, torch_inputs) -> None:
            result = execute(inputs).to_numpy()
            x, w1 = torch_inputs

            expected = x @ w1.T

            np.testing.assert_allclose(
                result, expected, atol=1e-1, rtol=1e-6, equal_nan=True
            )


def test_sequential__two_linear_layers(session) -> None:
    dtype = DType.float32

    with Graph(
        "sequential_two_linear_layers",
        input_types=[
            TensorType(dtype, ["batch", "input_dim"], device=DeviceRef.CPU()),
            TensorType(
                dtype, ["hidden_dim", "input_dim"], device=DeviceRef.CPU()
            ),
            TensorType(
                dtype, ["input_dim", "hidden_dim"], device=DeviceRef.CPU()
            ),
        ],
    ) as graph:
        assert are_all_tensor_values(graph.inputs)
        x, w1, w2 = graph.inputs
        two_layers = Sequential([LinearV1(w1), LinearV1(w2)])

        graph.output(two_layers(x))

        @modular_graph_test(
            session,
            graph,
            max_magnitude=1 / 64,
        )
        def test_correctness(execute, inputs, torch_inputs) -> None:
            result = execute(inputs).to_numpy()
            x, w1, w2 = torch_inputs

            expected = x @ w1.T
            expected = expected @ w2.T

            np.testing.assert_allclose(
                result, expected, atol=1e-1, rtol=1e-6, equal_nan=True
            )


def test_sequential__two_linear_layers_with_activation(session) -> None:
    dtype = DType.float32

    with Graph(
        "sequential_two_linear_layers",
        input_types=[
            TensorType(dtype, ["batch", "input_dim"], device=DeviceRef.CPU()),
            TensorType(
                dtype, ["hidden_dim", "input_dim"], device=DeviceRef.CPU()
            ),
            TensorType(
                dtype, ["input_dim", "hidden_dim"], device=DeviceRef.CPU()
            ),
        ],
    ) as graph:
        assert are_all_tensor_values(graph.inputs)
        x, w1, w2 = graph.inputs
        two_layer = Sequential([LinearV1(w1), ops.relu, LinearV1(w2)])
        graph.output(two_layer(x))

        @modular_graph_test(
            session,
            graph,
            max_magnitude=1 / 64,
        )
        def test_correctness(execute, inputs, torch_inputs) -> None:
            result = execute(inputs).to_numpy()
            x, w1, w2 = torch_inputs

            expected = x @ w1.T
            expected = np.maximum(expected, 0)
            expected = expected @ w2.T

            np.testing.assert_allclose(
                result, expected, atol=1e-1, rtol=1e-6, equal_nan=True
            )
