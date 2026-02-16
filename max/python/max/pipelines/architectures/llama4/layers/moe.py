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

"""Mixture of Experts Layer."""

from __future__ import annotations

from max.dtype import DType
from max.graph import TensorValue, ops
from max.nn.legacy.moe import MoEGate, StackedMoE


class Llama4MoEGate(MoEGate):
    """Mixture of Experts Gate Layer for Llama-4."""

    def __call__(
        self, hidden_states: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        """Compute expert routing weights and indices for input hidden states.

        Args:
            hidden_states: Input tensor of shape (seq_len, hidden_dim)

        Returns:
            tuple containing:
                - topk_idx: Indices of top-k selected experts of shape (seq_len, num_experts_per_token)
                - topk_weight: Routing weights for selected experts of shape (seq_len, num_experts_per_token)
        """
        # compute gating score
        logits = self.gate_score(hidden_states)
        # (seq_len, num_experts)
        top_idx = ops.squeeze(ops.argmax(logits, axis=-1), axis=1)
        # (seq_len,)
        router_probs = ops.sigmoid(
            ops.max(logits, axis=-1).cast(DType.float32)
        ).cast(hidden_states.dtype)

        return top_idx, router_probs


class Llama4MoE(StackedMoE):
    """Mixture of Experts Layer for Llama-4.

    Inherits from StackedMoE with concatenated gate/up format, silu activation,
    shared experts, and apply_router_weight_first=True.
    """

    pass
