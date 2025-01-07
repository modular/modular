# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from max.driver import Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType
from nn.hooks.print_hook import PrintHook
from nn.layer import Layer


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


def test_unnamed_print_hook(session: InferenceSession, capfd) -> None:
    print_hook = PrintHook()
    layer = OuterLayer()

    g = Graph(
        "nested", layer, input_types=[TensorType(DType.float32, (2, 4, 6))]
    )

    model = session.load(g)
    unused_output = model(Tensor.zeros((2, 4, 6), DType.float32))

    print_hook.remove()
    del print_hook  # Trigger print_hook.summarize()

    # The printed tensors should include names of the layer classes.
    captured = capfd.readouterr()
    assert "OuterLayer-input" in captured.out
    assert "OuterLayer-output" in captured.out
    assert "InnerLayer-input" in captured.out
    assert "InnerLayer-output" in captured.out
    assert "InnerLayer2-input" in captured.out
    assert "InnerLayer2-output" in captured.out
    assert "Printed 6 tensors for step 0" in captured.out


def test_named_print_hook(session: InferenceSession, capfd) -> None:
    print_hook = PrintHook()
    layer = OuterLayer()

    print_hook.name_layers(layer)  # Creates names for each layer.

    g = Graph(
        "nested", layer, input_types=[TensorType(DType.float32, (2, 4, 6))]
    )

    model = session.load(g)
    unused_output = model(Tensor.zeros((2, 4, 6), DType.float32))

    print_hook.remove()
    del print_hook  # Trigger print_hook.summarize()

    # The printed tensors should include names created by `name_layers`
    captured = capfd.readouterr()
    assert "model-input" in captured.out
    assert "model-output" in captured.out
    assert "model.inner_layer_1-input" in captured.out
    assert "model.inner_layer_1-output" in captured.out
    assert "model.inner_layer_2-input" in captured.out
    assert "model.inner_layer_2-output" in captured.out
    assert "Printed 6 tensors for step 0" in captured.out
