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
"""Hy3 MoE gate: sigmoid + per-expert correction bias + scaled top-k."""

from __future__ import annotations

from max.dtype import DType
from max.graph import TensorValue, ops
from max.nn.moe import SigmoidTopKRouter


class HYV3TopKRouter(SigmoidTopKRouter):
    """Sigmoid top-k router whose gate projection runs in float32."""

    def _gate_logits(self, hidden_states: TensorValue) -> TensorValue:
        # Compute the router matmul in FP32 to match HF
        # (`F.linear(hidden.float(), weight.float())`); the top-8-of-192
        # decision is sensitive to BF16 matmul rounding.
        hs_fp32 = ops.cast(hidden_states, DType.float32)
        w_fp32 = ops.cast(self.gate_score.weight, DType.float32).to(
            hs_fp32.device
        )
        return hs_fp32 @ ops.transpose(w_fp32, -1, -2)
