# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

from collections import OrderedDict

import pytest
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.nn.graph_groups import Logits
from max.nn.hooks.print_hook import PrintHook
from max.nn.layer import Layer


class OuterLayer(Layer):
    def __init__(self) -> None:
        self.inner_layer_1 = InnerLayer()
        self.inner_layer_2 = InnerLayer()

    def __call__(self, input: TensorValue) -> TensorValue:
        cast_input = input.cast(DType.int32)
        inner_1 = self.inner_layer_1(cast_input)
        inner_2 = self.inner_layer_2(inner_1)
        return inner_2


class InnerLayer(Layer):
    def __call__(self, input: TensorValue) -> TensorValue:
        return input.transpose(0, 1)


def test_unnamed_print_hook(
    session: InferenceSession,
    capfd: pytest.CaptureFixture,  # type: ignore[type-arg]
) -> None:
    print_hook = PrintHook()
    layer = OuterLayer()

    g = Graph(
        "nested",
        layer,
        input_types=[
            TensorType(DType.float32, (2, 4, 6), device=DeviceRef.CPU())
        ],
    )

    model = session.load(g)
    model(Buffer.zeros((2, 4, 6), DType.float32))

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


def test_named_print_hook(
    session: InferenceSession,
    capfd: pytest.CaptureFixture,  # type: ignore[type-arg]
) -> None:
    print_hook = PrintHook()
    layer = OuterLayer()

    print_hook.name_layers(layer)  # Creates names for each layer.

    g = Graph(
        "nested",
        layer,
        input_types=[
            TensorType(DType.float32, (2, 4, 6), device=DeviceRef.CPU())
        ],
    )

    model = session.load(g)
    model(Buffer.zeros((2, 4, 6), DType.float32))

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


class NamedOutputs(Layer):
    def __call__(self, input: TensorValue) -> OrderedDict[str, Logits]:
        return OrderedDict(logits=Logits(next_token_logits=input))


def test_print_hook_records_named_outputs_by_path(
    session: InferenceSession,
    capfd: pytest.CaptureFixture[str],
) -> None:
    print_hook = PrintHook()
    layer = NamedOutputs()
    print_hook.name_layers(layer)

    g = Graph(
        "named_outputs",
        layer,
        input_types=[TensorType(DType.float32, (2, 4), device=DeviceRef.CPU())],
    )
    model = session.load(g)
    model(Buffer.zeros((2, 4), DType.float32))

    print_hook.remove()
    del print_hook

    captured = capfd.readouterr()
    assert "model-output_logits.next_token_logits" in captured.out
    assert "Was not able to write outputs" not in captured.out
