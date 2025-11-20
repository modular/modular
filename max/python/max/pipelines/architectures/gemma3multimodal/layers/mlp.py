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

from collections.abc import Sequence

from max.graph import TensorValue, ops
from max.graph.type import DeviceRef
from max.nn import Linear, Module

from ..model_config import VisionConfig


class VisionMLP(Module):
    """MLP for vision encoder with GELU activation."""

    def __init__(
        self,
        config: VisionConfig,
        devices: Sequence[DeviceRef],
        layer_idx: int = 0,
    ):
        """Initialize vision MLP.

        Args:
            config: Vision configuration.
            devices: Devices to place the weights on.
            layer_idx: Index of the layer (for unique naming).
        """
        super().__init__()
        self.config = config
        self.devices = devices
        self.device = devices[0] if devices else DeviceRef.CPU()
        self.layer_idx = layer_idx

        self.fc1 = Linear(
            in_dim=config.hidden_size,
            out_dim=config.intermediate_size,
            dtype=config.dtype,
            device=self.device,
            has_bias=True,
        )

        self.fc2 = Linear(
            in_dim=config.intermediate_size,
            out_dim=config.hidden_size,
            dtype=config.dtype,
            device=self.device,
            has_bias=True,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        """Forward pass for MLP.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        x = self.fc1(x)
        x = ops.gelu(x)
        x = self.fc2(x)
        return x
