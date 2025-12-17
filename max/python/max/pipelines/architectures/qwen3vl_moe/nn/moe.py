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
"""Mixture of Experts Layer for Qwen3VL MoE."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy, TensorValue, Weight, ops
from max.nn.kernels import grouped_matmul_ragged, moe_create_indices
from max.nn.moe import MoE, MoEGate


class Qwen3VLMoEGate(MoEGate):
    """Qwen3VL MoE Gate with simple top-k routing and softmax normalization."""

    def __init__(
        self,
        devices: list[DeviceRef],
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        dtype: DType,
        is_sharding: bool = False,
        norm_topk_prob: bool = True,
    ) -> None:
        """
        Args:
            devices: List of devices to use for the MoEGate.
            hidden_dim: The dimension of the hidden state.
            num_experts: The number of experts.
            num_experts_per_token: The number of experts per token.
            dtype: The data type of the MoEGate.
            norm_topk_prob: Whether to normalize top-k probabilities to sum to 1.
        """
        super().__init__(
            devices=devices,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            dtype=dtype,
            is_sharding=is_sharding,
        )
        self.norm_topk_prob = norm_topk_prob

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
        # Compute router logits
        router_logits = self.gate_score(hidden_states).cast(DType.float32)

        # Apply softmax to get routing weights
        routing_weights = ops.softmax(router_logits, axis=-1)

        # Select top-k experts
        topk_weights, topk_indices = ops.top_k(
            routing_weights, k=self.num_experts_per_token, axis=-1
        )

        # Normalize top-k weights to sum to 1
        denominator = ops.sum(topk_weights, axis=-1) + 1e-20

        topk_weights = topk_weights / denominator

        # Cast back to original dtype
        topk_weights = topk_weights.cast(hidden_states.dtype)

        return topk_indices, topk_weights

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Get the sharding strategy for the module."""
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Set the sharding strategy for the module."""
        if strategy.is_replicate:
            self._sharding_strategy = strategy
            self.gate_score.sharding_strategy = ShardingStrategy.replicate(
                strategy.num_devices
            )
        else:
            raise ValueError(
                "Only replicate sharding strategy is supported for MoEGate."
            )

    def shard(self, devices: Iterable[DeviceRef]) -> Sequence[MoEGate]:
        """Create sharded views of this MoEGate module across multiple devices.

        Args:
            devices: Iterable of devices to place the shards on.

        Returns:
            List of sharded Qwen3VLMoEGate instances, one for each device."""
        if not self._sharding_strategy:
            raise ValueError(
                "MoEGate module cannot be sharded because no sharding strategy was provided."
            )

        # Get sharded weights
        gate_score_shards = self.gate_score.shard(devices)

        shards = []
        for shard_idx, device in enumerate(devices):
            sharded = Qwen3VLMoEGate(
                devices=[device],
                hidden_dim=self.hidden_dim,
                num_experts=self.num_experts,
                num_experts_per_token=self.num_experts_per_token,
                dtype=self.dtype,
                is_sharding=True,
                norm_topk_prob=self.norm_topk_prob,
            )

            # Replace the weights with sharded versions.
            sharded.gate_score = gate_score_shards[shard_idx]
            shards.append(sharded)
        return shards


class Qwen3VLMoE(MoE):
    """Qwen3VL MoE implementation with standard gated activation."""

    def __init__(
        self,
        devices: list[DeviceRef],
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        moe_dim: int,
        gate_cls: Callable[..., MoEGate] = Qwen3VLMoEGate,
        dtype: DType = DType.bfloat16,
        norm_topk_prob: bool = True,
        mlp_only_layers: list[int] | None = None,
        is_sharding: bool = False,
    ) -> None:
        """
        Args:
            devices: List of devices to use for the MoE.
            hidden_dim: The dimension of the hidden state.
            num_experts: The number of experts.
            num_experts_per_token: The number of experts per token.
            moe_dim: The intermediate dimension of each expert.
            dtype: The data type of the MoE.
            norm_topk_prob: Whether to normalize top-k probabilities.
            mlp_only_layers: List of layer indices that use MLP instead of MoE (unused here, kept for compatibility).
        """
        super().__init__(
            devices=devices,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            moe_dim=moe_dim,
            gate_cls=gate_cls,
            dtype=dtype,
            has_shared_experts=False,
            shared_experts_dim=0,
            ep_size=1,
            apply_router_weight_first=False,
            ep_batch_manager=None,
            float8_config=None,
            is_sharding=is_sharding,
        )
        self.norm_topk_prob = norm_topk_prob
        self.mlp_only_layers = mlp_only_layers

    def _init_experts(self) -> None:
        """Initialize experts using stacked weight tensors instead of individual MLPs.

        This matches how weights are stored in the checkpoint:
        - layers.{i}.mlp.experts.gate_up_proj with shape [num_experts, hidden_dim, 2*moe_dim]
        - layers.{i}.mlp.experts.down_proj with shape [num_experts, moe_dim, hidden_dim]
        """
        # Create stacked weight tensors for all experts
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

    @property
    def gate_up_proj(self) -> TensorValue:
        """Return combined gate_up projection weights for grouped_matmul_ragged.

        grouped_matmul_ragged expects shape [num_experts, out_features, in_features].
        """
        # Return the combined gate_up projection weights, transposed for grouped_matmul_ragged
        # grouped_matmul_ragged expects shape [num_experts, out_features, in_features]
        return self._experts_gate_up_proj_weight.transpose(1, 2)

    @property
    def down_proj(self) -> TensorValue:
        """Return down projection weights for grouped_matmul_ragged.

        grouped_matmul_ragged expects shape [num_experts, out_features, in_features].
        """
        # Transpose for grouped_matmul_ragged: [num_experts, hidden_dim, moe_dim]
        return self._experts_down_proj_weight.transpose(1, 2)

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Get the sharding strategy for the module."""
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Set the sharding strategy for the module."""
        if strategy.is_tensor_parallel:
            self._sharding_strategy = strategy
            self.gate.sharding_strategy = ShardingStrategy.replicate(
                strategy.num_devices
            )
            # Set sharding strategy for the stacked expert weights
            self._experts_gate_up_proj_weight.sharding_strategy = (
                ShardingStrategy.axiswise(
                    axis=2, num_devices=strategy.num_devices
                )
            )
            self._experts_down_proj_weight.sharding_strategy = (
                ShardingStrategy.axiswise(
                    axis=1, num_devices=strategy.num_devices
                )
            )
        else:
            raise ValueError(
                "Only tensor parallel sharding strategy is supported for Qwen3VLMoE"
            )

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
            ops.cast(
                token_expert_order // self.num_experts_per_token, DType.int32
            ),
            axis=0,
        )
        # Gate + Up projection
        gate_up_projs = grouped_matmul_ragged(
            permutated_states,
            self.gate_up_proj,
            expert_start_indices,
            expert_ids,
            expert_usage_stats.to(DeviceRef.CPU()),
        )

        # Standard gated activation: up * act_fn(gate)
        # Split gate and up projections (interleaved format like GPT-OSS)
        gate = gate_up_projs[:, 0::2]
        up = gate_up_projs[:, 1::2]

        gated_output = up * ops.silu(gate)

        # Down projection
        down_projs = grouped_matmul_ragged(
            gated_output,
            self.down_proj,
            expert_start_indices,
            expert_ids,
            expert_usage_stats.to(DeviceRef.CPU()),
        )

        # Restore token order and reshape
        down_projs = ops.gather(
            down_projs, restore_token_order, axis=0
        ).reshape([seq_len, self.num_experts_per_token, self.hidden_dim])

        # Weighted combination: (seq_len, 1, n_expert) @ (seq_len, n_expert, hidden_dim) -> (seq_len, 1, hidden_dim)
        routed_expert_out = ops.unsqueeze(router_weight, axis=1) @ down_projs
        routed_expert_out = ops.squeeze(routed_expert_out, axis=1).cast(x.dtype)

        return routed_expert_out

    def shard(self, devices: Iterable[DeviceRef]) -> list[Qwen3VLMoE]:
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
        gate_shards = self.gate.shard(devices)

        # Shard the stacked expert weight tensors
        experts_gate_up_proj_shards = self._experts_gate_up_proj_weight.shard(
            devices
        )
        experts_down_proj_shards = self._experts_down_proj_weight.shard(devices)

        shards = []
        num_devices = self._sharding_strategy.num_devices
        sharded_moe_dim = self.moe_dim // num_devices
        for shard_idx, device in enumerate(devices):
            sharded = self.__class__(
                devices=[device],
                hidden_dim=self.hidden_dim,
                num_experts=self.num_experts,
                num_experts_per_token=self.num_experts_per_token,
                moe_dim=sharded_moe_dim,
                gate_cls=self.gate_cls,
                dtype=self.dtype,
                norm_topk_prob=self.norm_topk_prob,
                mlp_only_layers=self.mlp_only_layers,
            )
            # Replace layers and weights with sharded versions.
            sharded.gate = gate_shards[shard_idx]

            # Replace the stacked expert weights with sharded versions
            sharded._experts_gate_up_proj_weight = experts_gate_up_proj_shards[
                shard_idx
            ]
            sharded._experts_down_proj_weight = experts_down_proj_shards[
                shard_idx
            ]

            shards.append(sharded)

        return shards
