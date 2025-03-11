# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pytest
from max.dtype import DType
from max.graph import Graph, TensorType, ops
from max.nn.layer import Layer, add_layer_hook, clear_hooks


@pytest.fixture(autouse=True)
def clear_hooks_between_tests():
    yield
    clear_hooks()


class OuterLayer(Layer):
    def __init__(self):
        self.inner_layer_1 = InnerLayer()
        self.inner_layer_2 = InnerLayer()

    def __call__(self, input):
        cast_input = input.cast(DType.int32)
        inner_1 = self.inner_layer_1(cast_input)
        inner_2 = self.inner_layer_2(inner_1)
        return inner_2


class InnerLayer(Layer):
    def __call__(self, input):
        return input.transpose(0, 1)


def test_hook_nested_layers(mocker):
    outer_layer = OuterLayer()

    hook = mocker.Mock(side_effect=lambda *args: args[-1])
    add_layer_hook(hook)
    g = Graph(
        "test_hook",
        outer_layer,
        input_types=[TensorType(DType.int64, shape=[5, "dim1", "dim2"])],
    )
    assert hook.call_count == 3

    # Check the arguments passed to the hook.
    # First call should be to the outer_layer.inner_layer_1.
    layer1, args1, kwargs1, outputs1 = hook.call_args_list[0][0]
    assert layer1 is outer_layer.inner_layer_1
    assert len(args1) == 1
    assert args1[0].type == TensorType(DType.int32, shape=[5, "dim1", "dim2"])
    assert len(kwargs1) == 0
    assert outputs1.type == TensorType(DType.int32, shape=["dim1", 5, "dim2"])

    # Second call should be to outer_layer.inner_layer_2.
    layer2, args2, kwargs2, outputs2 = hook.call_args_list[1][0]
    assert layer2 is outer_layer.inner_layer_2
    assert len(args2) == 1
    assert args2[0] is outputs1
    assert len(kwargs2) == 0
    assert outputs2.type == TensorType(DType.int32, shape=[5, "dim1", "dim2"])

    # Third call should be to outer_layer.
    layer3, args3, kwargs3, outputs3 = hook.call_args_list[2][0]
    assert layer3 is outer_layer
    assert len(args3) == 1
    assert args3[0] is g.inputs[0]
    assert len(kwargs2) == 0
    assert outputs3 is outputs2


def test_hook_nested_hooks_returns(mocker):
    # Test multiple hooks that modify the output value.
    inner_layer = InnerLayer()
    hook = mocker.Mock(side_effect=lambda *args: ops.tile(args[-1], [1, 2, 1]))
    hook2 = mocker.Mock(
        side_effect=lambda *args: ops.cast(args[-1], DType.float32)
    )
    add_layer_hook(hook)
    add_layer_hook(hook2)
    with Graph(
        "test_hook",
        input_types=[TensorType(DType.int64, shape=[5, "dim1", "dim2"])],
    ) as g:
        output = inner_layer(g.inputs[0])
        g.output(output)

    # The final output of inner_layer should be the return value of `hook2`.
    assert output.type == TensorType(DType.float32, shape=["dim1", 10, "dim2"])

    # Check that both hooks have been called.
    assert hook.call_count == 1
    assert hook2.call_count == 1

    # Check the args passed to `hook`
    layer1, args1, kwargs1, outputs1 = hook.call_args_list[0][0]
    assert layer1 is inner_layer
    assert len(args1) == 1
    assert args1[0] is g.inputs[0]
    assert len(kwargs1) == 0
    assert outputs1.type == TensorType(DType.int64, shape=["dim1", 5, "dim2"])

    # Check the args passed to `hook2`
    layer2, args2, kwargs2, outputs2 = hook2.call_args_list[0][0]
    assert layer2 is inner_layer
    assert len(args2) == 1
    assert args2[0] is g.inputs[0]
    assert len(kwargs2) == 0
    # The outputs passed to hook2 should be the return value of `hook`.
    assert outputs2.type == TensorType(DType.int64, shape=["dim1", 10, "dim2"])


def test_clear_hooks(mocker):
    hook = mocker.Mock(side_effect=lambda *args: args[-1])
    hook2 = mocker.Mock(side_effect=lambda *args: args[-1])
    add_layer_hook(hook)
    add_layer_hook(hook2)
    clear_hooks()

    outer_layer = OuterLayer()
    _ = Graph(
        "test_hook",
        outer_layer,
        input_types=[TensorType(DType.int64, shape=[5, "dim1", "dim2"])],
    )

    assert hook.call_count == 0
    assert hook2.call_count == 0


class LayerWithArgsKwargs(Layer):
    def __call__(self, arg1, arg2, *args, kwarg1="1", kwarg2="3", **kwargs):
        return arg1


def test_hook_args_kwargs(mocker):
    hook = mocker.Mock(side_effect=lambda *args: args[-1])
    add_layer_hook(hook)
    layer = LayerWithArgsKwargs()

    g = Graph(
        "test_hook",
        layer,
        [TensorType(DType.int64, shape=[5, "dim1", "dim2"])],
        kwarg2="kwarg2 value",
        arg2="arg2 value",
    )

    assert hook.call_count == 1

    _, args, kwargs, _ = hook.call_args_list[0][0]
    assert len(args) == 2
    assert args[0] is g.inputs[0]
    assert args[1] == "arg2 value"
    assert kwargs == {"kwarg2": "kwarg2 value"}


def test_hook_many_args_kwargs(mocker):
    hook = mocker.Mock(side_effect=lambda *args: args[-1])
    add_layer_hook(hook)
    layer = LayerWithArgsKwargs()

    g = Graph(
        "test_hook",
        layer,
        [TensorType(DType.int64, shape=[5, "dim1", "dim2"])],
        None,  # Path - if loading mef from file
        "arg2 value",
        "arg3 value",
        "arg4 value",
        kwarg3="kwarg3 value",
        kwarg2="kwarg2 value",
        kwarg1="kwarg1 value",
    )

    assert hook.call_count == 1

    _, args, kwargs, _ = hook.call_args_list[0][0]
    assert len(args) == 4
    assert args[0] is g.inputs[0]
    assert args[1:] == ("arg2 value", "arg3 value", "arg4 value")
    assert kwargs == {
        "kwarg1": "kwarg1 value",
        "kwarg2": "kwarg2 value",
        "kwarg3": "kwarg3 value",
    }
