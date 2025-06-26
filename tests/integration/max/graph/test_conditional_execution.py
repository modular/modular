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
    Weight,
    ops,
)


@pytest.fixture
def custom_ops_path() -> Path:
    return Path(os.environ["CUSTOM_OPS_PATH"])


device_ref = DeviceRef.GPU() if accelerator_count() > 0 else DeviceRef.CPU()


def test_conditional_execution_no_results(session, capfd):
    with Graph(
        "conditional",
        input_types=(TensorType(DType.bool, [], device=device_ref),),
    ) as graph:

        def then_fn():
            ops.print("then")

        def else_fn():
            ops.print("else")

        ops.cond(graph.inputs[0].tensor, None, then_fn, else_fn)
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
        "conditional",
        input_types=(TensorType(DType.bool, [], device=device_ref),),
    ) as graph:
        cond = graph.inputs[0]

        def then_fn():
            return ops.constant(1, DType.int32, device=device_ref)

        def else_fn():
            return ops.constant(0, DType.int32, device=device_ref)

        result = ops.cond(
            cond.tensor,
            [TensorType(DType.int32, [], device=device_ref)],
            then_fn,
            else_fn,
        )
        graph.output(result[0])

    compiled = session.load(graph)
    output = compiled.execute(True)
    assert output[0].to_numpy() == 1

    output = compiled.execute(False)
    assert output[0].to_numpy() == 0


def test_conditional_shape_to_tensor_solo_dim(session):
    input_type = TensorType(
        dtype=DType.float32, shape=["batch", "channels"], device=device_ref
    )
    with Graph("input_shape", input_types=(input_type,)) as graph:
        shape = graph.inputs[0].tensor.shape

        def then_fn():
            return ops.constant(1, DType.int32, device=device_ref)

        def else_fn():
            return ops.constant(0, DType.int32, device=device_ref)

        result = ops.cond(
            TensorValue(shape[1]) == 3,
            [TensorType(DType.int32, [], device=device_ref)],
            then_fn,
            else_fn,
        )
        graph.output(result[0])

    compiled = session.load(graph)

    x = Tensor.from_numpy(np.ones((7, 3)).astype(np.float32)).to(
        compiled.input_devices[0]
    )
    output = compiled.execute(x)

    # Output is only a scalar
    assert output[0].shape == ()
    np.testing.assert_equal(output[0].to_numpy(), np.array([1]))

    x = Tensor.from_numpy(np.ones((7, 4)).astype(np.float32)).to(
        compiled.input_devices[0]
    )
    output = compiled.execute(x)

    # Output is only a scalar
    assert output[0].shape == ()
    np.testing.assert_equal(output[0].to_numpy(), np.array([0]))


@pytest.mark.skip(reason="assert fail")
def test_conditional_inplace_user_supplied(
    custom_ops_path, session: InferenceSession
):
    import torch

    bt = BufferType(DType.float32, [2, 2], device=device_ref)
    bool_type = TensorType(DType.bool, [], device=device_ref)

    with Graph(
        "basic",
        input_types=[bt, bool_type],
        custom_extensions=[custom_ops_path],
    ) as graph:
        buffer: BufferValue = graph.inputs[0].buffer
        cond = graph.inputs[1].tensor

        def then_fn():
            # this custom op is equivalent to buffer[0,0] += 1
            ops.inplace_custom(
                "mutable_test_op", device=buffer.device, values=[buffer]
            )
            ops.inplace_custom(
                "mutable_test_op", device=buffer.device, values=[buffer]
            )
            buffer[...] = ops.negate(buffer[...])

        def else_fn():
            # this custom op is equivalent to buffer[0,0] += 1
            ops.inplace_custom(
                "mutable_test_op", device=buffer.device, values=[buffer]
            )
            ops.inplace_custom(
                "mutable_test_op", device=buffer.device, values=[buffer]
            )
            ops.inplace_custom(
                "mutable_test_op", device=buffer.device, values=[buffer]
            )

        ops.cond(cond, None, then_fn, else_fn)

        graph.output()

    model = session.load(graph)

    rawbuffer = torch.ones((2, 2), dtype=torch.float32)
    if accelerator_count() > 0:
        rawbuffer = rawbuffer.cuda()
    model.execute(Tensor.from_dlpack(rawbuffer), True)
    actual = np.array([[3, 1], [1, 1]], dtype=np.float32) * -1
    np.testing.assert_equal(rawbuffer.cpu().numpy(), actual)

    rawbuffer = torch.ones((2, 2), dtype=torch.float32)
    if accelerator_count() > 0:
        rawbuffer = rawbuffer.cuda()
    model.execute(Tensor.from_dlpack(rawbuffer), False)
    actual = np.array([[4, 1], [1, 1]], dtype=np.float32)
    np.testing.assert_equal(rawbuffer.cpu().numpy(), actual)


def test_conditional_nested_conditionals(session, capfd):
    with Graph(
        "nested_conditionals",
        input_types=(
            TensorType(DType.bool, [], device=device_ref),
            TensorType(DType.bool, [], device=device_ref),
        ),
    ) as graph:
        cond_1 = graph.inputs[0].tensor
        cond_2 = graph.inputs[1].tensor

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
        "graph_with_cond_weights",
        input_types=[TensorType(DType.bool, [], device=device_ref)],
    ) as graph:

        def true_fn():
            w = Weight(
                "random_weight",
                dtype=DType.int64,
                shape=[5, 10],
                device=DeviceRef.CPU(),
            )
            return w.to(device_ref) * 2

        def false_fn():
            w = Weight(
                "random_weight",
                dtype=DType.int64,
                shape=[5, 10],
                device=DeviceRef.CPU(),
            )
            return w.to(device_ref) * 3

        graph.output(
            *ops.cond(
                graph.inputs[0].tensor,
                [TensorType(DType.int64, [5, 10], device=device_ref)],
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
        "graph_with_cond_weights",
        input_types=[TensorType(DType.bool, [], device=device_ref)],
    ) as graph:
        true_weight = np.random.uniform(1, 100, size=[5, 10]).astype(np.int64)
        false_weight = np.random.uniform(1, 100, size=[5, 10]).astype(np.int64)

        def true_fn():
            w = Weight(
                "true_weight",
                dtype=DType.int64,
                shape=[5, 10],
                device=DeviceRef.CPU(),
            )
            graph.add_weight(w)
            return w.to(device_ref) * 2

        def false_fn():
            w = Weight(
                "false_weight",
                dtype=DType.int64,
                shape=[5, 10],
                device=DeviceRef.CPU(),
            )
            graph.add_weight(w)
            return w.to(device_ref) * 3

        graph.output(
            *ops.cond(
                graph.inputs[0].tensor,
                [TensorType(DType.int64, [5, 10], device=device_ref)],
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
        "graph_with_cond_weights",
        input_types=[TensorType(DType.bool, [], device=device_ref)],
    ) as graph:
        weight = np.random.uniform(1, 100, size=[5, 10]).astype(np.int64)
        w = Weight(
            "random_weight",
            dtype=DType.int64,
            shape=[5, 10],
            device=DeviceRef.CPU(),
        )
        graph.add_weight(w)

        graph.output(
            *ops.cond(
                graph.inputs[0].tensor,
                [TensorType(DType.int64, [5, 10], device=device_ref)],
                lambda: w.to(device_ref),
                lambda: w.to(device_ref),
            )
        )
        compiled = session.load(
            graph, weights_registry={"random_weight": weight}
        )
        output = compiled.execute(True)
        np.testing.assert_array_equal(weight, output[0].to_numpy())

        output = compiled.execute(False)
        np.testing.assert_array_equal(weight, output[0].to_numpy())


def test_cond_returned_diff_weights(session) -> None:
    """Tests adding an external weight to a graph."""
    with Graph(
        "graph_with_cond_weights",
        input_types=[TensorType(DType.bool, [], device=device_ref)],
    ) as graph:
        true_weight = np.random.uniform(1, 100, size=[5, 10]).astype(np.int64)
        false_weight = np.random.uniform(1, 100, size=[5, 10]).astype(np.int64)

        def true_fn():
            w = Weight(
                "true_weight",
                dtype=DType.int64,
                shape=[5, 10],
                device=DeviceRef.CPU(),
            )
            graph.add_weight(w)
            return w.to(device_ref)

        def false_fn():
            w = Weight(
                "false_weight",
                dtype=DType.int64,
                shape=[5, 10],
                device=DeviceRef.CPU(),
            )
            graph.add_weight(w)
            return w.to(device_ref)

        results = ops.cond(
            graph.inputs[0].tensor,
            [TensorType(DType.int64, [5, 10], device=device_ref)],
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
