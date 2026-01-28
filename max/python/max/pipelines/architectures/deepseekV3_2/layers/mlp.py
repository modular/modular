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
"""Simplified MLP layer for DeepseekV3.2."""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, TensorValueLike, ops
from max.nn.legacy import Float8Config, Linear, Module


class MLP(Module):
    """Simple multi-layer perceptron with three linear layers and SiLU activation.

    Implements a gated MLP:
        output = down_proj(silu(gate_proj(x)) * up_proj(x))

    Args:
        dtype: DType for the layer weights, which should match the input dtype.
        hidden_dim: The last dimension of the layer input.
        feed_forward_length: Size of dimension used to project the inputs.
        device: Device to run the MLP layer on.
        float8_config: Optional Float8Config for float8 quantization.
    """

    def __init__(
        self,
        dtype: DType,
        hidden_dim: int,
        feed_forward_length: int,
        device: DeviceRef,
        float8_config: Float8Config | None = None,
    ) -> None:
        """Initializes the MLP layer."""
        super().__init__()

        self.hidden_dim = hidden_dim
        self.feed_forward_length = feed_forward_length

        self.gate_proj = Linear(
            in_dim=hidden_dim,
            out_dim=feed_forward_length,
            dtype=dtype,
            device=device,
            float8_config=float8_config,
        )
        self.down_proj = Linear(
            in_dim=feed_forward_length,
            out_dim=hidden_dim,
            dtype=dtype,
            device=device,
            float8_config=float8_config,
        )
        self.up_proj = Linear(
            in_dim=hidden_dim,
            out_dim=feed_forward_length,
            dtype=dtype,
            device=device,
            float8_config=float8_config,
        )

    def __call__(self, x: TensorValueLike) -> TensorValue:
        """Applies the MLP transformation to the input.
        Differs from standard MLP in that it uses float32 for intermediate operations.

        Args:
            x: Input tensor of shape ``(..., hidden_dim)``.

        Returns:
            Output tensor of shape ``(..., hidden_dim)`` after applying
            the gated MLP transformation.
        """
        x = TensorValue(x)
        dtype = x.dtype
        gate_out = self.gate_proj(x).cast(DType.float32)
        up_out = self.up_proj(x).cast(DType.float32)
        hidden = (ops.silu(gate_out) * up_out).cast(dtype)
        return self.down_proj(hidden)
