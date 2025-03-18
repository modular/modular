# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Integration tests for conditional execution."""

import os
from pathlib import Path

import numpy as np
import pytest
from max.driver import Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    Weight,
    ops,
)


@pytest.fixture
def custom_ops_path() -> Path:
    return Path(os.environ["CUSTOM_OPS_PATH"])


def test_conditional_execution_no_results(session, capfd):
    with Graph(
        "conditional", input_types=(TensorType(DType.bool, []),)
    ) as graph:

        def then_fn():
            ops.print("then")

        def else_fn():
            ops.print("else")

        ops.cond(graph.inputs[0], None, then_fn, else_fn)
        graph.output()

    compiled = session.load(graph)
    compiled.execute(True)

    captured = capfd.readouterr()
    assert "then" in captured.out
    assert "else" not in captured.out

    compiled.execute(False)

    captured = capfd.readouterr()
    assert "then" not in captured.out
    assert "else" in captured.out


def test_conditional_execution_with_results(session):
    with Graph(
        "conditional", input_types=(TensorType(DType.bool, []),)
    ) as graph:
        cond = graph.inputs[0]

        def then_fn():
            return ops.constant(1, DType.int32)

        def else_fn():
            return ops.constant(0, DType.int32)

        result = ops.cond(cond, [TensorType(DType.int32, [])], then_fn, else_fn)
        graph.output(result[0])

    compiled = session.load(graph)
    output = compiled.execute(True)
    assert output[0].to_numpy() == 1

    output = compiled.execute(False)
    assert output[0].to_numpy() == 0


def test_conditional_shape_to_tensor_solo_dim(session):
    input_type = TensorType(dtype=DType.float32, shape=["batch", "channels"])
    with Graph("input_shape", input_types=(input_type,)) as graph:
        shape = graph.inputs[0].shape

        def then_fn():
            return ops.constant(1, DType.int32)

        def else_fn():
            return ops.constant(0, DType.int32)

        result = ops.cond(
            TensorValue(shape[1]) == 3,
            [TensorType(DType.int32, [])],
            then_fn,
            else_fn,
        )
        graph.output(result[0])

    compiled = session.load(graph)

    x = np.ones((7, 3)).astype(np.float32)
    output = compiled.execute(x)

    # Output is only a scalar
    assert output[0].shape == ()
    np.testing.assert_equal(output[0].to_numpy(), np.array([1]))

    x = np.ones((7, 4)).astype(np.float32)
    output = compiled.execute(x)

    # Output is only a scalar
    assert output[0].shape == ()
    np.testing.assert_equal(output[0].to_numpy(), np.array([0]))


def test_conditional_inplace_user_supplied(
    custom_ops_path, session: InferenceSession
):
    bt = BufferType(DType.float32, [2, 2])
    bool_type = TensorType(DType.bool, [])

    with Graph("basic", input_types=[bt, bool_type]) as graph:
        buffer: BufferValue = graph.inputs[0]
        cond = graph.inputs[1]

        def then_fn():
            # this custom op is equivalent to buffer[0,0] += 1
            ops.inplace_custom("mutable_test_op", values=[buffer])
            ops.inplace_custom("mutable_test_op", values=[buffer])
            buffer[...] = ops.negate(buffer[...])

        def else_fn():
            # this custom op is equivalent to buffer[0,0] += 1
            ops.inplace_custom("mutable_test_op", values=[buffer])
            ops.inplace_custom("mutable_test_op", values=[buffer])
            ops.inplace_custom("mutable_test_op", values=[buffer])

        ops.cond(cond, None, then_fn, else_fn)

        graph.output()

    model = session.load(graph, custom_ops_path=custom_ops_path)

    rawbuffer = np.ones((2, 2), dtype=np.float32)
    model.execute(Tensor.from_dlpack(rawbuffer), True)
    actual = np.array([[3, 1], [1, 1]], dtype=np.float32) * -1
    np.testing.assert_equal(rawbuffer, actual)

    rawbuffer = np.ones((2, 2), dtype=np.float32)
    model.execute(Tensor.from_dlpack(rawbuffer), False)
    actual = np.array([[4, 1], [1, 1]], dtype=np.float32)
    np.testing.assert_equal(rawbuffer, actual)


def test_conditional_nested_conditionals(session, capfd):
    with Graph(
        "nested_conditionals",
        input_types=(
            TensorType(DType.bool, []),
            TensorType(DType.bool, []),
        ),
    ) as graph:
        cond_1 = graph.inputs[0]
        cond_2 = graph.inputs[1]

        def true_fn_1():
            ops.print("true_1")

            def true_fn_2():
                ops.print("true_true_2")

            def false_fn_2():
                ops.print("true_false_2")

            ops.cond(cond_2, None, true_fn_2, false_fn_2)

        def false_fn_1():
            ops.print("false_1")

            def true_fn_2():
                ops.print("false_true_2")

            def false_fn_2():
                ops.print("false_false_2")

            ops.cond(cond_2, None, true_fn_2, false_fn_2)

        ops.cond(cond_1, None, true_fn_1, false_fn_1)
        graph.output()

    compiled = session.load(graph)
    compiled.execute(True, True)
    captured = capfd.readouterr()
    assert "true_1" in captured.out
    assert "false_1" not in captured.out
    assert "true_true_2" in captured.out
    assert "true_false_2" not in captured.out
    assert "false_true_2" not in captured.out
    assert "false_false_2" not in captured.out

    compiled.execute(False, True)
    captured = capfd.readouterr()
    assert "true_1" not in captured.out
    assert "false_1" in captured.out
    assert "true_true_2" not in captured.out
    assert "true_false_2" not in captured.out
    assert "false_true_2" in captured.out
    assert "false_false_2" not in captured.out

    compiled.execute(True, False)
    captured = capfd.readouterr()
    assert "true_1" in captured.out
    assert "false_1" not in captured.out
    assert "true_true_2" not in captured.out
    assert "true_false_2" in captured.out
    assert "false_true_2" not in captured.out
    assert "false_false_2" not in captured.out

    compiled.execute(False, False)
    captured = capfd.readouterr()
    assert "true_1" not in captured.out
    assert "false_1" in captured.out
    assert "true_true_2" not in captured.out
    assert "true_false_2" not in captured.out
    assert "false_true_2" not in captured.out
    assert "false_false_2" in captured.out


def test_conditional_with_same_name_weight(session) -> None:
    """Tests adding an external weight to a graph."""
    weight = np.random.uniform(1, 100, size=[5, 10]).astype(np.int64)
    with Graph(
        "graph_with_cond_weights", input_types=[TensorType(DType.bool, [])]
    ) as graph:

        def true_fn():
            w = Weight(
                "random_weight",
                dtype=DType.int64,
                shape=[5, 10],
                device=DeviceRef.CPU(),
            )
            return w * 2

        def false_fn():
            w = Weight(
                "random_weight",
                dtype=DType.int64,
                shape=[5, 10],
                device=DeviceRef.CPU(),
            )
            return w * 3

        graph.output(
            *ops.cond(
                graph.inputs[0],
                [TensorType(DType.int64, [5, 10], device=DeviceRef.CPU())],
                true_fn,
                false_fn,
            )
        )

        compiled = session.load(
            graph, weights_registry={"random_weight": weight}
        )
        output = compiled.execute(True)
        np.testing.assert_array_equal(weight * 2, output[0].to_numpy())

        output = compiled.execute(False)
        np.testing.assert_array_equal(weight * 3, output[0].to_numpy())


def test_conditional_with_diff_names_weights(session) -> None:
    """Tests adding an external weight to a graph."""
    with Graph(
        "graph_with_cond_weights", input_types=[TensorType(DType.bool, [])]
    ) as graph:
        true_weight = np.random.uniform(1, 100, size=[5, 10]).astype(np.int64)
        false_weight = np.random.uniform(1, 100, size=[5, 10]).astype(np.int64)

        def true_fn():
            w = Weight("true_weight", dtype=DType.int64, shape=[5, 10])
            graph.add_weight(w, DeviceRef.CPU())
            return w * 2

        def false_fn():
            w = Weight("false_weight", dtype=DType.int64, shape=[5, 10])
            graph.add_weight(w, DeviceRef.CPU())
            return w * 3

        graph.output(
            *ops.cond(
                graph.inputs[0],
                [TensorType(DType.int64, [5, 10], device=DeviceRef.CPU())],
                true_fn,
                false_fn,
            )
        )

        compiled = session.load(
            graph,
            weights_registry={
                "true_weight": true_weight,
                "false_weight": false_weight,
            },
        )
        output = compiled.execute(True)
        np.testing.assert_array_equal(true_weight * 2, output[0].to_numpy())

        output = compiled.execute(False)
        np.testing.assert_array_equal(false_weight * 3, output[0].to_numpy())


def test_conditional_with_returned_weights(session) -> None:
    """Tests adding an external weight to a graph."""
    with Graph(
        "graph_with_cond_weights", input_types=[TensorType(DType.bool, [])]
    ) as graph:
        weight = np.random.uniform(1, 100, size=[5, 10]).astype(np.int64)
        w = Weight("random_weight", dtype=DType.int64, shape=[5, 10])
        graph.add_weight(w, DeviceRef.CPU())

        graph.output(
            *ops.cond(
                graph.inputs[0],
                [TensorType(DType.int64, [5, 10], device=DeviceRef.CPU())],
                lambda: w,
                lambda: w,
            )
        )
        compiled = session.load(
            graph, weights_registry={"random_weight": weight}
        )
        output = compiled.execute(True)
        np.testing.assert_array_equal(weight, output[0].to_numpy())

        output = compiled.execute(False)
        np.testing.assert_array_equal(weight, output[0].to_numpy())


@pytest.mark.skip(reason="This is causes a crash in the graph compiler.")
def test_cond_returned_diff_weights(session) -> None:
    """Tests adding an external weight to a graph."""
    with Graph(
        "graph_with_cond_weights", input_types=[TensorType(DType.bool, [])]
    ) as graph:
        true_weight = np.random.uniform(1, 100, size=[5, 10]).astype(np.int64)
        false_weight = np.random.uniform(1, 100, size=[5, 10]).astype(np.int64)

        def true_fn():
            w = Weight("true_weight", dtype=DType.int64, shape=[5, 10])
            graph.add_weight(w, DeviceRef.CPU())
            return w

        def false_fn():
            w = Weight("false_weight", dtype=DType.int64, shape=[5, 10])
            graph.add_weight(w, DeviceRef.CPU())
            return w

        results = ops.cond(
            graph.inputs[0],
            [TensorType(DType.int64, [5, 10], device=DeviceRef.CPU())],
            true_fn,
            false_fn,
        )
        result = results[0] * 2
        graph.output(result)
        compiled = session.load(
            graph,
            weights_registry={
                "true_weight": true_weight,
                "false_weight": false_weight,
            },
        )
        output = compiled.execute(True)
        np.testing.assert_array_equal(true_weight * 2, output[0].to_numpy())

        output = compiled.execute(False)
        np.testing.assert_array_equal(false_weight * 2, output[0].to_numpy())
