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
"""A generalized Mixture of Experts (MoE) module."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable

from max.dtype import DType
from max.graph import BufferValue, DeviceRef, ShardingStrategy, TensorValue, ops

from ..comm import Allreduce
from ..kernels import grouped_matmul_ragged, moe_create_indices
from ..layer import LayerList, Module, Shardable
from ..linear import MLP, Linear


class MoEGate(Module):
    """Gate module for MoE."""

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
        super().__init__()
        self.devices = devices
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token

        self.gate_score = Linear(
            in_dim=hidden_dim,
            out_dim=num_experts,
            dtype=dtype,
            device=devices[0],
        )

    def __call__(
        self, hidden_state: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        """
        Args:
            hidden_state: The hidden state of the model.

        Returns:
            A tuple of the topk indices and scores.
        """
        scores = self.gate_score(hidden_state)
        topk_scores, topk_indices = ops.top_k(
            scores, k=self.num_experts_per_token, axis=-1
        )

        return topk_indices, topk_scores


class MoE(Module, Shardable):
    """Implementation of Mixture of Experts (MoE)."""

    _sharding_strategy: ShardingStrategy | None = None
    """The sharding strategy for the module."""

    def __init__(
        self,
        devices: list[DeviceRef],
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        moe_dim: int,
        gate_cls: Callable[..., MoEGate] = MoEGate,
        has_shared_experts: bool = False,
        shared_experts_dim: int = 0,
        ep_size: int = 1,
        dtype: DType = DType.bfloat16,
        apply_router_weight_first: bool = False,
    ):
        """
        Args:
            devices: List of devices to use for the MoE.
            hidden_dim: The dimension of the hidden state.
            num_experts: The number of experts.
            num_experts_per_token: The number of experts per token.
            moe_dim: The intermediate dimension of each expert.
            gate_cls: The model specific gate implementation.
            has_shared_experts: Whether to use shared experts.
            shared_experts_dim: The dimension of the shared experts.
            ep_size: The expert parallelism size.
            dtype: The data type of the MoE.
            apply_router_weight_first: Whether to apply the router weight first.
        """
        super().__init__()
        self.devices = devices
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.moe_dim = moe_dim
        self.gate_cls = gate_cls
        self.has_shared_experts = has_shared_experts
        self.shared_experts_dim = shared_experts_dim
        self.ep_size = ep_size
        self.dtype = dtype
        self.apply_router_weight_first = apply_router_weight_first
        self.gate = gate_cls(
            devices=devices,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            dtype=dtype,
        )
        self.num_local_experts = num_experts // ep_size

        if has_shared_experts:
            assert shared_experts_dim > 0, (
                "shared_experts_dim must be greater than 0"
            )
            self.shared_experts = MLP(
                dtype=dtype,
                quantization_encoding=None,
                hidden_dim=self.hidden_dim,
                feed_forward_length=self.shared_experts_dim,
                devices=self.devices,
            )

        self._init_experts()

    def _init_experts(self) -> None:
        self.experts: LayerList = LayerList(
            [
                MLP(
                    dtype=self.dtype,
                    quantization_encoding=None,
                    hidden_dim=self.hidden_dim,
                    feed_forward_length=self.moe_dim,
                    devices=self.devices,
                )
                for _ in range(self.num_experts)
            ]
        )

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
                self.shared_experts.gate_proj.sharding_strategy = (
                    ShardingStrategy.rowwise(strategy.num_devices)
                )
                self.shared_experts.up_proj.sharding_strategy = (
                    ShardingStrategy.rowwise(strategy.num_devices)
                )
                self.shared_experts.down_proj.sharding_strategy = (
                    ShardingStrategy.columnwise(strategy.num_devices)
                )

            for expert in self.experts:
                expert.gate_proj.sharding_strategy = ShardingStrategy.rowwise(
                    strategy.num_devices
                )
                expert.up_proj.sharding_strategy = ShardingStrategy.rowwise(
                    strategy.num_devices
                )
                expert.down_proj.sharding_strategy = (
                    ShardingStrategy.columnwise(strategy.num_devices)
                )
        else:
            raise ValueError(
                "Only tensor parallel sharding strategy is supported for MoE"
            )

    def shard(self, devices: Iterable[DeviceRef]) -> list[MoE]:
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

        shared_gate_proj_shards = []
        shared_up_proj_shards = []
        shared_down_proj_shards = []
        if self.has_shared_experts:
            shared_gate_proj_shards = self.shared_experts.gate_proj.shard(
                devices
            )
            shared_up_proj_shards = self.shared_experts.up_proj.shard(devices)
            shared_down_proj_shards = self.shared_experts.down_proj.shard(
                devices
            )

        # Shard each expert's weights
        expert_gate_proj_shards = [
            expert.gate_proj.shard(devices) for expert in self.experts
        ]
        expert_up_proj_shards = [
            expert.up_proj.shard(devices) for expert in self.experts
        ]
        expert_down_proj_shards = [
            expert.down_proj.shard(devices) for expert in self.experts
        ]

        shards = []
        for shard_idx, device in enumerate(devices):
            sharded = MoE(
                devices=[device],
                hidden_dim=self.hidden_dim,
                num_experts=self.num_experts,
                num_experts_per_token=self.num_experts_per_token,
                moe_dim=self.moe_dim // self._sharding_strategy.num_devices,
                gate_cls=self.gate_cls,
                has_shared_experts=self.has_shared_experts,
                shared_experts_dim=self.shared_experts_dim
                // self._sharding_strategy.num_devices,
                ep_size=self.ep_size,
                dtype=self.dtype,
                apply_router_weight_first=self.apply_router_weight_first,
            )

            # Replace the weights with sharded versions.
            sharded.gate.gate_score = gate_score_shards[shard_idx]
            if self.has_shared_experts:
                sharded.shared_experts.gate_proj = shared_gate_proj_shards[
                    shard_idx
                ]
                sharded.shared_experts.up_proj = shared_up_proj_shards[
                    shard_idx
                ]
                sharded.shared_experts.down_proj = shared_down_proj_shards[
                    shard_idx
                ]

            for _expert_idx, (
                expert,
                gate_shards,
                up_shards,
                down_shards,
            ) in enumerate(
                zip(
                    sharded.experts,
                    expert_gate_proj_shards,
                    expert_up_proj_shards,
                    expert_down_proj_shards,
                )
            ):
                expert.gate_proj = gate_shards[shard_idx]
                expert.up_proj = up_shards[shard_idx]
                expert.down_proj = down_shards[shard_idx]

            shards.append(sharded)

        return shards

    @property
    def gate_up_proj(self) -> TensorValue:
        gate_list = [expert.gate_proj.weight for expert in self.experts]
        up_list = [expert.up_proj.weight for expert in self.experts]

        gate_up_list: list[TensorValue] = []
        for tensors in zip(gate_list, up_list):
            gate_up_list.extend(tensors)

        return ops.stack(gate_up_list, axis=0).reshape(
            [self.num_local_experts, -1, self.hidden_dim]
        )

    @property
    def down_proj(self) -> TensorValue:
        down_proj = ops.stack(
            [expert.down_proj.weight for expert in self.experts], axis=0
        )
        return down_proj

    def __call__(self, hidden_state: TensorValue) -> TensorValue:
        """
        Args:
            hidden_state: (seq_len, hidden_dim)

        Returns:
            (seq_len, hidden_dim)
        """
        seq_len = hidden_state.shape[0]

        # Get the topk experts per token and their weights
        router_idx, router_weight = self.gate(hidden_state)
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
            ops.cast(router_idx, DType.uint32), self.num_experts
        )

        permutated_states = ops.gather(
            hidden_state,
            token_expert_order / self.num_experts_per_token,
            axis=0,
        )

        if self.apply_router_weight_first:
            permutated_states = permutated_states * ops.gather(
                router_weight.reshape([-1, 1]), token_expert_order, axis=0
            ).cast(hidden_state.dtype)

        gate_up_projs = grouped_matmul_ragged(
            permutated_states,
            self.gate_up_proj,
            expert_start_indices,
            expert_ids,
            expert_usage_stats.to(DeviceRef.CPU()),
        )

        gate_up_projs = (
            ops.silu(gate_up_projs[:, : self.moe_dim])
            * gate_up_projs[:, self.moe_dim :]
        )

        down_projs = grouped_matmul_ragged(
            gate_up_projs,
            self.down_proj,
            expert_start_indices,
            expert_ids,
            expert_usage_stats.to(DeviceRef.CPU()),
        )

        down_projs = ops.gather(
            down_projs, restore_token_order, axis=0
        ).reshape([seq_len, self.num_experts_per_token, -1])

        if not self.apply_router_weight_first:
            # (seq_len, 1, n_expert) @ (seq_len, n_expert, hidden_dim) -> (seq_len, 1, hidden_dim)
            routed_expert_out = (
                ops.unsqueeze(router_weight, axis=1) @ down_projs
            )
            routed_expert_out = ops.squeeze(routed_expert_out, axis=1).cast(
                hidden_state.dtype
            )
        else:
            routed_expert_out = down_projs.transpose(1, 2)
            routed_expert_out = ops.squeeze(
                ops.sum(routed_expert_out, axis=2), axis=2
            ).cast(hidden_state.dtype)

        if self.has_shared_experts:
            routed_expert_out += self.shared_experts(hidden_state)

        return routed_expert_out


class DistributedTPMoE(MoE):
    """Distributed implementation of Mixture of Experts (MoE) with tensor parallelism."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if not self.devices or len(self.devices) < 2:
            raise ValueError(
                f"Must provide at least 2 devices to `DistributedMoE`, got {self.devices}"
            )
        self.num_devices = len(self.devices)
        self.sharding_strategy = ShardingStrategy.tensor_parallel(
            self.num_devices
        )
        self.allreduce = Allreduce(self.num_devices)

        self.list_of_moes = self.shard(self.devices)

    def __call__(  # type: ignore[override]
        self,
        hidden_states: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
    ) -> list[TensorValue]:
        assert self.devices
        return self.allreduce(
            [
                expert(hidden)
                for expert, hidden in zip(self.list_of_moes, hidden_states)
            ],
            signal_buffers=signal_buffers,
        )
