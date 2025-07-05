# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from max.dtype import DType
from max.engine.api import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops


@dataclass
class Unity:
    def __call__(self, x: TensorValue) -> TensorValue:
        return ops.constant(1.0, dtype=DType.float32, device=DeviceRef.CPU())


def test_load_rejects_non_contiguous_weights() -> None:
    """Test that InferenceSession.load() raises ValueError for non-contiguous weight registry inputs."""
    graph = Graph(
        "unity",
        forward=Unity(),
        input_types=[
            TensorType(DType.float32, ["batch", "dim"], device=DeviceRef.CPU())
        ],
    )

    session = InferenceSession()

    # Create a non-contiguous weight tensor for the registry
    weight_tensor = torch.randn(4, 4).t()  # transpose makes it non-contiguous
    assert not weight_tensor.is_contiguous()

    # Attempt to load model with non-contiguous weight should raise ValueError
    with pytest.raises(ValueError, match="non-contiguous tensors"):
        session.load(graph, weights_registry={"weight": weight_tensor.numpy()})


def test_execute_rejects_non_contiguous_input() -> None:
    """Test that model.execute() raises ValueError for non-contiguous input tensors."""
    graph = Graph(
        "unity",
        forward=Unity(),
        input_types=[
            TensorType(DType.float32, ["batch", "dim"], device=DeviceRef.CPU())
        ],
    )

    # Create the model (without any weights)
    session = InferenceSession()
    model = session.load(graph)

    # Create a non-contiguous input tensor
    input_tensor = torch.randn(4, 4).t()  # transpose makes it non-contiguous
    assert not input_tensor.is_contiguous()

    # Attempt to execute with non-contiguous input should raise ValueError
    with pytest.raises(ValueError, match="non-contiguous tensors"):
        model.execute(input_tensor)
