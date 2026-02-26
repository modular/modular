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
"""Mixture of Experts layer for Qwen3-MoE.

Uses the base MoE implementation with individual expert MLPs (no stacked
weights). Weights are loaded directly from the checkpoint per expert.
"""

from __future__ import annotations

from collections.abc import Iterable

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn.moe import MoEGate


class Qwen3MoEGate(MoEGate):
    """Qwen3-MoE gate with top-k routing and softmax normalization.

    Matches HuggingFace Qwen3-MoE: softmax over all experts, then top-k,
    then renormalize the top-k weights to sum to 1.
    """

    def __call__(
        self, hidden_states: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        """Compute expert routing weights and indices for input hidden states.

        Args:
            hidden_states: Input tensor of shape (seq_len, hidden_dim).

        Returns:
            tuple containing:
                - topk_idx: Indices of top-k selected experts of shape
                  (seq_len, num_experts_per_token).
                - topk_weight: Routing weights for selected experts of shape
                  (seq_len, num_experts_per_token).
        """
        router_logits = self.gate_score(hidden_states).cast(DType.float32)
        routing_weights = ops.softmax(router_logits, axis=-1)
        topk_weights, topk_indices = ops.top_k(
            routing_weights, k=self.num_experts_per_token, axis=-1
        )
        denominator = ops.sum(topk_weights, axis=-1) + 1e-20
        topk_weights = topk_weights / denominator
        topk_weights = topk_weights.cast(hidden_states.dtype)
        return topk_indices, topk_weights

    def shard(self, devices: Iterable[DeviceRef]) -> list[Qwen3MoEGate]:
        """Create sharded views of this gate."""
        if not self._sharding_strategy:
            raise ValueError(
                "MoEGate module cannot be sharded because no sharding "
                "strategy was provided."
            )
        gate_score_shards = self.gate_score.shard(devices)
        shards: list[Qwen3MoEGate] = []
        for shard_idx, device in enumerate(devices):
            sharded = Qwen3MoEGate(
                devices=[device],
                hidden_dim=self.hidden_dim,
                num_experts=self.num_experts,
                num_experts_per_token=self.num_experts_per_token,
                dtype=self.dtype,
                is_sharding=True,
            )
            sharded.gate_score = gate_score_shards[shard_idx]
            shards.append(sharded)
        return shards
