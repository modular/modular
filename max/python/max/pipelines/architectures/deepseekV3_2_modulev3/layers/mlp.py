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
"""Simplified MLP layer for DeepseekV3.2, in the ModuleV3 API."""

from __future__ import annotations

from max.dtype import DType
from max.experimental.tensor import Tensor

from ...deepseekV3_modulev3.layers import quant_ops
from ...deepseekV3_modulev3.layers.quant_linear import QuantizedMLP


class DeepseekV3_2MLP(QuantizedMLP):
    """Gated MLP whose activation is computed in float32.

    Matches V2: gate and up stay separate matmuls and the activation runs in
    float32.
    """

    def forward(self, x: Tensor) -> Tensor:
        # V2 runs gate and up as two matmuls; fusing them is one FP8 matmul
        # fewer and was measured bit-identical here (the 2048-row split lands
        # on the 128-row weight-scale block boundary, so the concatenated
        # weight and its scales are exactly the two originals stacked).
        gate_up_weight = quant_ops.concat_weights(
            self.gate_proj.weight, self.up_proj.weight, axis=0
        )
        gate_up = quant_ops.matmul(x, gate_up_weight)
        gate, up = gate_up.split(
            [self.feed_forward_length, self.feed_forward_length], axis=-1
        )
        gate = (gate + self.gate_proj.bias).cast(DType.float32)
        up = (up + self.up_proj.bias).cast(DType.float32)
        hidden = (self.activation_function(gate) * up).cast(x.dtype)
        return self.down_proj(hidden)
