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

"""Mixture of Experts Gate Layer for DeepSeek V3.2."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn.legacy.kernels import moe_router_group_limited
from max.nn.legacy.moe import MoEGate
from max.pipelines.architectures.deepseekV3.layers.moe_gate import (
    DeepseekV3TopKRouter,
)


class DeepseekV3_2TopKRouter(DeepseekV3TopKRouter):
    """Mixture of Experts Gate Layer for DeepSeek V3.2.

    Inherits from DeepseekV3TopKRouter and overrides __call__ to use float32
    for gate computation, matching the V3.2 reference implementation.
    """

    def __call__(
        self, hidden_states: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        """Compute expert routing weights and indices for input hidden states.

        Uses float32 for gate computation to match DeepSeek V3.2 reference.

        Args:
            hidden_states: Input tensor of shape (seq_len, hidden_dim)

        Returns:
            tuple containing:
                - topk_idx: Indices of top-k selected experts of shape (seq_len, num_experts_per_token)
                - topk_weight: Routing weights for selected experts of shape (seq_len, num_experts_per_token)
        """
        # Cast to float for gate computation (V3.2 change)
        hidden_states_float = hidden_states.cast(DType.float32)

        # Compute gate score with float inputs
        logits = self.gate_score(hidden_states_float)

        if self.scoring_func == "sigmoid":
            scores = ops.sigmoid(logits.cast(self.correction_bias_dtype))
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        topk_idx, topk_weight = moe_router_group_limited(
            scores,
            self.e_score_correction_bias,
            self.num_experts,
            self.num_experts_per_token,
            self.n_group,
            self.topk_group,
            self.norm_topk_prob,
            self.routed_scaling_factor,
        )
        return topk_idx, topk_weight

    def shard(self, devices: Iterable[DeviceRef]) -> Sequence[MoEGate]:
        """Create sharded views of this MoEGate module across multiple devices.

        Args:
            devices: Iterable of devices to place the shards on.

        Returns:
            List of sharded DeepseekV3_2TopKRouter instances, one for each device."""
        if not self._sharding_strategy:
            raise ValueError(
                "MoEGate module cannot be sharded because no sharding strategy was provided."
            )

        # Get sharded weights
        gate_score_shards = self.gate_score.shard(devices)
        correction_bias_shards = self.e_score_correction_bias.shard(devices)

        shards = []
        for shard_idx, device in enumerate(devices):
            sharded = DeepseekV3_2TopKRouter(
                hidden_dim=self.hidden_dim,
                num_experts=self.num_experts,
                num_experts_per_token=self.num_experts_per_token,
                routed_scaling_factor=self.routed_scaling_factor,
                scoring_func=self.scoring_func,
                topk_method=self.topk_method,
                n_group=self.n_group,
                topk_group=self.topk_group,
                norm_topk_prob=self.norm_topk_prob,
                dtype=self.dtype,
                gate_dtype=self.gate_dtype,
                correction_bias_dtype=self.correction_bias_dtype,
                devices=[device],
            )

            # Replace the weights with sharded versions.
            sharded.gate_score = gate_score_shards[shard_idx]
            sharded.e_score_correction_bias = correction_bias_shards[shard_idx]
            shards.append(sharded)
        return shards
