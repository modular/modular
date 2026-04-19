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
"""Simplified MLP layer for DeepseekV3.2."""

from __future__ import annotations

from max.dtype import DType
from max.graph import (
    TensorValue,
    TensorValueLike,
    ops,
)
from max.nn import MLP


class DeepseekV3_2MLP(MLP):
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
        gate_out = self.gate_proj(x).cast(DType.float32)
        up_out = self.up_proj(x).cast(DType.float32)
        hidden = (ops.silu(gate_out) * up_out).cast(x.dtype)
        return self.down_proj(hidden)
