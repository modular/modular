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
"""Tests for MissingLayer, the place of a layer another stage builds."""

import pytest
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight
from max.nn import LayerList, MissingLayer, Module


class Scale(Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = Weight("weight", DType.float32, [4], DeviceRef.CPU())

    def __call__(self, x: TensorValue) -> TensorValue:
        return x * self.weight


def test_missing_layers_keep_checkpoint_names() -> None:
    held = range(2, 4)
    layers = LayerList(
        [Scale() if i in held else MissingLayer(i) for i in range(5)]
    )
    assert sorted(layers.state_dict()) == ["2.weight", "3.weight"]


def test_calling_a_missing_layer_names_it() -> None:
    with pytest.raises(RuntimeError, match="Decoder layer 3"):
        MissingLayer(3)()
