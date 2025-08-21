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

"""GPT OSS Mixture of Experts Layer."""

from __future__ import annotations

from collections.abc import Iterable
from copy import copy

from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy, TensorValue, Weight, ops
from max.nn.clamp import clamp
from max.nn.kernels import grouped_matmul_ragged, moe_create_indices
from max.nn.layer import LayerList
from max.nn.linear import Linear
from max.nn.moe import MoE, MoEGate

from ..model_config import GptOssConfig


class GptOssMoEGate(MoEGate):
    """GptOss-style Gate module for MoE with bias support."""

    def __init__(
        self,
        devices: list[DeviceRef],
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        dtype: DType,
    ) -> None:
        """
        Args:
            devices: List of devices to use for the MoEGate.
            hidden_dim: The dimension of the hidden state.
            num_experts: The number of experts.
            num_experts_per_token: The number of experts per token.
            dtype: The data type of the MoEGate.
        """
        # Initialize parent class
        super().__init__(
            devices=devices,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            dtype=dtype,
        )

        # Override gate_score with bias-enabled Linear layer
        self.gate_score = Linear(
            in_dim=hidden_dim,
            out_dim=num_experts,
            dtype=dtype,
            device=devices[0],
            has_bias=True,  # Enable bias
        )

    def __call__(
        self, hidden_state: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        """
        Args:
            hidden_state: The hidden state of the model.

        Returns:
            A tuple of the topk indices and scores with softmax applied.
        """
        scores = self.gate_score(hidden_state)
        topk_scores, topk_indices = ops.top_k(
            scores, k=self.num_experts_per_token, axis=-1
        )

        # Apply softmax to top-k scores (matching GptOss behavior)
        topk_scores = ops.softmax(topk_scores)

        return topk_indices, topk_scores


class GptOssMoE(MoE):
    """GptOss-style MoE implementation with custom activation and biases."""

    def __init__(
        self,
        config: GptOssConfig,
    ):
        """
        Args:
            config: The configuration for the GPT OSS Model.
        """
        # Store GptOss-specific parameters
        self.alpha = 1.702
        self.limit = 7.0

        self.config = config
        self._sharding_strategy = None

        # Initialize parent class
        super().__init__(
            devices=config.devices,
            hidden_dim=config.hidden_size,
            num_experts=config.num_local_experts,
            num_experts_per_token=config.num_experts_per_tok,
            moe_dim=config.intermediate_size,
            gate_cls=GptOssMoEGate,
            has_shared_experts=False,
            ep_size=1,
            dtype=config.dtype,
            apply_router_weight_first=False,
        )

    def _init_experts(self) -> None:
        # Instead of creating individual MLP experts, we'll use combined weight tensors
        # This matches how the weights are stored in the checkpoint
        self.experts = LayerList([])  # Empty list to maintain compatibility

        # Create combined weight tensors for all experts
        # Gate and up projections are combined into one tensor
        self._experts_gate_up_proj_weight = Weight(
            "experts.gate_up_proj",
            shape=[self.num_experts, self.hidden_dim, 2 * self.moe_dim],
            dtype=self.dtype,
            device=self.devices[0],
        )

        # Down projection weights
        self._experts_down_proj_weight = Weight(
            "experts.down_proj",
            shape=[self.num_experts, self.moe_dim, self.hidden_dim],
            dtype=self.dtype,
            device=self.devices[0],
        )

        # Bias terms for gate_up projection (combined)
        self._experts_gate_up_proj_bias = Weight(
            "experts.gate_up_proj_bias",
            shape=[self.num_experts, 2 * self.moe_dim],
            dtype=self.dtype,
            device=self.devices[0],
        )

        # Bias terms for down projection
        self._experts_down_proj_bias = Weight(
            "experts.down_proj_bias",
            shape=[self.num_experts, self.hidden_dim],
            dtype=self.dtype,
            device=self.devices[0],
        )

    @property
    def gate_up_proj(self) -> TensorValue:
        # Return the combined gate_up projection weights, transposed for grouped_matmul_ragged
        # grouped_matmul_ragged expects shape [num_experts, out_features, in_features]
        return self._experts_gate_up_proj_weight.transpose(1, 2)

    @property
    def down_proj(self) -> TensorValue:
        # Return the combined down projection weights, transposed for grouped_matmul_ragged
        # grouped_matmul_ragged expects shape [num_experts, out_features, in_features]
        return self._experts_down_proj_weight.transpose(1, 2)

    @property
    def gate_up_proj_bias_stacked(self) -> TensorValue:
        # Return the combined gate_up projection biases
        return self._experts_gate_up_proj_bias

    @property
    def down_proj_bias_stacked(self) -> TensorValue:
        # Return the combined down projection biases
        return self._experts_down_proj_bias

    def __call__(self, x: TensorValue) -> TensorValue:
        """
        Args:
            x: (seq_len, hidden_dim)

        Returns:
            (seq_len, hidden_dim)
        """
        seq_len = x.shape[0]

        # Get the topk experts per token and their weights
        router_idx, router_weight = self.gate(x)
        router_idx = ops.reshape(
            router_idx, [-1]
        )  # (seq_len * n_expert_per_token,)

        (
            token_expert_order,
            expert_start_indices,
            restore_token_order,
            expert_ids,
            expert_usage_stats,
        ) = moe_create_indices(
            ops.cast(router_idx, DType.int32), self.num_experts
        )

        permutated_states = ops.gather(
            x,
            token_expert_order / self.num_experts_per_token,
            axis=0,
        )

        if self.apply_router_weight_first:
            permutated_states = permutated_states * ops.gather(
                router_weight.reshape([-1, 1]), token_expert_order, axis=0
            ).cast(x.dtype)

        # Apply gate_up projection with bias
        gate_up_output = grouped_matmul_ragged(
            permutated_states,
            self.gate_up_proj,
            expert_start_indices,
            expert_ids,
            expert_usage_stats.to(DeviceRef.CPU()),
        )

        # Apply bias based on expert assignment
        # We need to gather the bias for each token based on which expert it was routed to
        # router_idx contains the expert assignment for each token
        expert_assignments = ops.gather(router_idx, token_expert_order, axis=0)
        bias_per_token = ops.gather(
            self.gate_up_proj_bias_stacked, expert_assignments, axis=0
        )
        gate_up_output = gate_up_output + bias_per_token

        # Split gate and up projections
        gate = gate_up_output[:, : self.moe_dim]
        up = gate_up_output[:, self.moe_dim :]

        # Apply clamping (NOTE: This is specific to GptOss)
        gate = ops.min(gate, self.limit)
        up = clamp(up, min=-self.limit, max=self.limit)

        # GptOss-style activation: gate * sigmoid(gate * alpha) * (up + 1)
        glu = gate * ops.sigmoid(gate * self.alpha)
        gated_output = (up + 1.0) * glu

        # Apply down projection
        down_output = grouped_matmul_ragged(
            gated_output,
            self.down_proj,
            expert_start_indices,
            expert_ids,
            expert_usage_stats.to(DeviceRef.CPU()),
        )

        # Apply bias based on expert assignment
        # Use the same expert assignments we calculated earlier
        down_bias_per_token = ops.gather(
            self.down_proj_bias_stacked, expert_assignments, axis=0
        )
        down_output = down_output + down_bias_per_token

        # Reshape and apply routing weights
        down_output = ops.gather(
            down_output, restore_token_order, axis=0
        ).reshape([seq_len, self.num_experts_per_token, -1])

        if not self.apply_router_weight_first:
            # (seq_len, 1, n_expert) @ (seq_len, n_expert, hidden_dim) -> (seq_len, 1, hidden_dim)
            routed_expert_out = (
                ops.unsqueeze(router_weight, axis=1) @ down_output
            )
            routed_expert_out = ops.squeeze(routed_expert_out, axis=1).cast(
                x.dtype
            )
        else:
            routed_expert_out = down_output.transpose(1, 2)
            routed_expert_out = ops.squeeze(
                ops.sum(routed_expert_out, axis=2), axis=2
            ).cast(x.dtype)

        if self.has_shared_experts:
            routed_expert_out += self.shared_experts(x)

        return routed_expert_out

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Get the sharding strategy for the module."""
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Set the sharding strategy for the module."""
        if strategy.is_tensor_parallel:
            self._sharding_strategy = strategy
            self.gate.gate_score.sharding_strategy = ShardingStrategy.replicate(
                strategy.num_devices
            )
            if self.has_shared_experts:
                self.shared_experts.sharding_strategy = strategy

            # Set sharding strategy for the combined expert weights
            self._experts_gate_up_proj_weight.sharding_strategy = (
                ShardingStrategy.rowwise(strategy.num_devices)
            )
            self._experts_down_proj_weight.sharding_strategy = (
                ShardingStrategy.columnwise(strategy.num_devices)
            )
            self._experts_gate_up_proj_bias.sharding_strategy = (
                ShardingStrategy.rowwise(strategy.num_devices)
            )
            self._experts_down_proj_bias.sharding_strategy = (
                ShardingStrategy.replicate(strategy.num_devices)
            )
        else:
            raise ValueError(
                "Only tensor parallel sharding strategy is supported for MoE"
            )

    def shard(self, devices: Iterable[DeviceRef]) -> list[GptOssMoE]:  # type: ignore[override]
        """Create sharded views of this MoE module across multiple devices.

        Args:
            devices: Iterable of devices to place the shards on.

        Returns:
            List of sharded MoE instances, one for each device."""
        if not self._sharding_strategy:
            raise ValueError(
                "MoE module cannot be sharded because no sharding strategy was provided."
            )

        # Get sharded weights
        gate_score_shards = self.gate.gate_score.shard(devices)

        if self.has_shared_experts:
            shared_experts_shards = self.shared_experts.shard(devices)

        # Shard the combined expert weight tensors
        experts_gate_up_proj_shards = self._experts_gate_up_proj_weight.shard(
            devices
        )
        experts_down_proj_shards = self._experts_down_proj_weight.shard(devices)
        experts_gate_up_proj_bias_shards = (
            self._experts_gate_up_proj_bias.shard(devices)
        )
        experts_down_proj_bias_shards = self._experts_down_proj_bias.shard(
            devices
        )

        shards = []
        for shard_idx, device in enumerate(devices):
            new_config = copy(self.config)
            new_config.devices = [device]
            new_config.hidden_size = (
                self.hidden_dim // self._sharding_strategy.num_devices
            )
            new_config.intermediate_size = (
                self.moe_dim // self._sharding_strategy.num_devices
            )
            new_config.num_local_experts = (
                self.num_experts // self._sharding_strategy.num_devices
            )
            new_config.num_experts_per_tok = (
                self.num_experts_per_token
                // self._sharding_strategy.num_devices
            )
            new_config.dtype = self.dtype

            sharded = GptOssMoE(
                config=new_config,
            )

            # Replace the weights with sharded versions.
            sharded.gate.gate_score = gate_score_shards[shard_idx]
            if self.has_shared_experts:
                sharded.shared_experts = shared_experts_shards[shard_idx]

            # Replace the combined expert weights with sharded versions
            sharded._experts_gate_up_proj_weight = experts_gate_up_proj_shards[
                shard_idx
            ]
            sharded._experts_down_proj_weight = experts_down_proj_shards[
                shard_idx
            ]
            sharded._experts_gate_up_proj_bias = (
                experts_gate_up_proj_bias_shards[shard_idx]
            )
            sharded._experts_down_proj_bias = experts_down_proj_bias_shards[
                shard_idx
            ]

            shards.append(sharded)

        return shards
