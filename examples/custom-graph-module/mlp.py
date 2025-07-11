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

# DOC: max/tutorials/build-an-mlp-block.mdx

from __future__ import annotations

from typing import Any, Callable, Optional

from max import nn
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops


class MLPBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: Optional[list[int]] = None,
        activation: Optional[Callable[[TensorValue], TensorValue]] = None,
    ) -> None:
        super().__init__()

        # Use empty list if no hidden features provided
        hidden_features = hidden_features or []

        # Default to ReLU activation if none provided
        activation = activation or ops.relu

        # Build the sequence of layers
        layers: list[Any] = []
        current_dim = in_features

        # Add hidden layers with their activations
        for i, h_dim in enumerate(hidden_features):
            layers.append(
                nn.Linear(
                    in_dim=current_dim,
                    out_dim=h_dim,
                    dtype=DType.float32,
                    device=DeviceRef.CPU(),
                    has_bias=True,
                    name=f"hidden_{i}",
                )
            )
            layers.append(activation)
            current_dim = h_dim

        # Add the final output layer
        layers.append(
            nn.Linear(
                in_dim=current_dim,
                out_dim=out_features,
                dtype=DType.float32,
                device=DeviceRef.CPU(),
                has_bias=True,
                name="output",
            )
        )

        # Create Sequential module with the layers
        self.model = nn.Sequential(layers)

    def __call__(self, x: TensorValue) -> TensorValue:
        return self.model(x)

    def __repr__(self) -> str:
        # Since Sequential is iterable, get all the layers
        layers = list(self.model)

        # Count different layer types
        linear_count = sum(
            1 for layer in layers if layer.__class__.__name__ == "Linear"
        )
        activation_count = len(layers) - linear_count

        return f"MLPBlock({linear_count} linear layers, {activation_count} activations)"
