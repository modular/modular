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

"""Mixture of Experts Layer."""

from __future__ import annotations

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    Weight,
    ops,
)
from max.nn.comm import Allreduce
from max.nn.kernels import grouped_matmul_ragged, moe_create_indices
from max.nn.layer import Module
from max.nn.linear import Linear


class MoE(Module):
    hidden_dim: int
    """Hidden dimension size."""

    top_k: int
    """Number of experts to route each token to."""

    num_experts: int
    """Total number of experts."""

    intermediate_size: int
    """Hidden dimension size for MoE intermediate layer."""

    intermediate_size_mlp: int
    """Hidden dimension size for MoE experts mlp layer."""

    dtype: DType
    """Data type for weights."""

    def __init__(
        self,
        devices: list[DeviceRef],
        hidden_dim: int = 5120,
        top_k: int = 1,
        num_experts: int = 16,
        intermediate_size: int = 8192,
        intermediate_size_mlp: int = 16384,
        dtype: DType = DType.bfloat16,
    ):
        """
        Args:
            devices: The devices to use to run this layer.
            hidden_dim: Hidden dimension size.
            top_k: Number of experts to route each token to.
            num_experts: Total number of experts.
            intermediate_size: Hidden dimension size for MoE intermediate layer.
            intermediate_size_mlp: Hidden dimension size for MoE MLP layer.
            dtype: Data type for weights.
        """
        super().__init__()

        self.top_k = top_k
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.intermediate_size = intermediate_size
        self.intermediate_size_mlp = intermediate_size_mlp
        self.devices = devices

        if self.top_k > 1:
            raise NotImplementedError(
                "Multiple expert routing (top-k > 1) is not yet implemented. "
                "This layer currently only supports single expert routing (top-k = 1) "
                "as used in the Llama-4-Scout-17B-16E-Instruct model."
            )

        # Routed experts weights.
        self.down_proj = Weight(
            name="experts.down_proj",
            shape=(
                self.num_experts,
                self.intermediate_size,
                self.hidden_dim,
            ),
            dtype=dtype,
            device=devices[0],
        )

        self.gate_up_proj = Weight(
            name="experts.gate_up_proj",
            shape=(
                self.num_experts,
                self.hidden_dim,
                self.intermediate_size_mlp,
            ),
            dtype=dtype,
            device=devices[0],
        )

        # Shared experts weights
        self.shared_expert_up_proj = Linear(
            in_dim=self.hidden_dim,
            out_dim=self.intermediate_size,
            dtype=dtype,
            device=devices[0],
        )
        self.shared_expert_down_proj = Linear(
            in_dim=self.intermediate_size,
            out_dim=self.hidden_dim,
            dtype=dtype,
            device=devices[0],
        )
        self.shared_expert_gate_proj = Linear(
            in_dim=self.hidden_dim,
            out_dim=self.intermediate_size,
            dtype=dtype,
            device=devices[0],
        )

        self.router = Linear(
            in_dim=self.hidden_dim,
            out_dim=self.num_experts,
            dtype=dtype,
            device=devices[0],
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        **unused_kwargs,
    ) -> TensorValue:
        hidden_states = ops.reshape(hidden_states, (-1, self.hidden_dim))
        router_logits = self.router(hidden_states)
        # (batch * seq_len, num_experts)
        top_idx = ops.squeeze(ops.argmax(router_logits, axis=-1), axis=1)
        # (batch * seq_len,)
        router_probs = ops.sigmoid(
            ops.max(router_logits, axis=-1).cast(DType.float32)
        ).cast(hidden_states.dtype)
        # (batch * seq_len, 1)

        (
            token_expert_order,
            expert_start_indices,
            restore_token_order,
            expert_ids,
            expert_usage_stats,
        ) = moe_create_indices(
            ops.cast(top_idx, DType.uint32),
            self.num_experts,
        )

        permutated_states = ops.gather(
            router_probs, token_expert_order, axis=0
        ) * ops.gather(hidden_states, token_expert_order, axis=0)
        gate_up_projs = grouped_matmul_ragged(
            permutated_states,
            ops.transpose(self.gate_up_proj, 1, 2),
            expert_start_indices,
            expert_ids,
            expert_usage_stats.to(DeviceRef.CPU()),
        )

        gate_up_projs = (
            ops.silu(gate_up_projs[:, : self.intermediate_size])
            * gate_up_projs[:, self.intermediate_size :]
        )

        down_projs = grouped_matmul_ragged(
            gate_up_projs,
            ops.transpose(self.down_proj, 1, 2),
            expert_start_indices,
            expert_ids,
            expert_usage_stats.to(DeviceRef.CPU()),
        )

        routed_expert_out = ops.gather(down_projs, restore_token_order, axis=0)

        shared_expert_out = self.shared_expert_down_proj(
            ops.silu(self.shared_expert_gate_proj(hidden_states))
            * self.shared_expert_up_proj(hidden_states)
        )
        return routed_expert_out + shared_expert_out


class DistributedMoE(MoE):
    """A distributed Mixture of Experts layer.

    This class has the same state keys as the non-distributed MoE Layer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.devices or len(self.devices) < 2:
            raise ValueError(
                f"Must provide at least 2 devices to `DistributedMoE`, got {self.devices}"
            )
        self.num_devices = len(self.devices)

        # Sharding strategies for the shared expert's weights, similar
        # to the sharding strategy in the DistributedMLP layer.
        def col_sharding_strategy(weight: Weight, i) -> TensorValue:
            col_size = int(weight.shape[1]) // self.num_devices
            return weight[:, i * col_size : (i + 1) * col_size]

        def row_sharding_strategy(weight: Weight, i) -> TensorValue:
            row_size = int(weight.shape[0]) // self.num_devices
            return weight[i * row_size : (i + 1) * row_size, :]

        self.shared_expert_gate_proj.set_sharding(
            ShardingStrategy.rowwise(self.num_devices)
        )
        self.shared_expert_down_proj.set_sharding(
            ShardingStrategy.columnwise(self.num_devices)
        )
        self.shared_expert_up_proj.set_sharding(
            ShardingStrategy.rowwise(self.num_devices)
        )

        # Sharding strategies for the routed experts' weights. The key differences are:
        # 1. All weights are rank-3 tensors
        # 2. Weights are not transposed.
        # 3. gate_proj and up_proj weights are concatenated together
        def down_sharding_strategy(
            weight: Weight, i: int, num_devices: int
        ) -> TensorValue:
            row_size = int(weight.shape[1]) // num_devices
            return weight[:, i * row_size : (i + 1) * row_size, :]

        def gate_up_sharding_strategy(
            weight: Weight, i: int, num_devices: int
        ) -> TensorValue:
            intermediate_size = int(weight.shape[2]) // 2
            col_size = intermediate_size // num_devices
            sharded_gate_proj = weight[:, :, i * col_size : (i + 1) * col_size]
            sharded_up_proj = weight[
                :,
                :,
                intermediate_size + i * col_size : intermediate_size
                + (i + 1) * col_size,
            ]
            return ops.concat((sharded_gate_proj, sharded_up_proj), axis=2)

        self.down_proj.set_sharding_strategy(
            ShardingStrategy(self.num_devices, down_sharding_strategy)
        )
        self.gate_up_proj.set_sharding_strategy(
            ShardingStrategy(self.num_devices, gate_up_sharding_strategy)
        )

        # we clone the router weights for each device
        self.router.set_sharding(ShardingStrategy.replicate(self.num_devices))

        # Create a separate MoE layer for each device.
        kwargs = kwargs.copy()
        kwargs["intermediate_size"] = self.intermediate_size // self.num_devices
        kwargs["intermediate_size_mlp"] = (
            self.intermediate_size_mlp // self.num_devices
        )
        self.moe_layers = []
        for n, device in enumerate(self.devices):
            kwargs["devices"] = [device]
            layer = MoE(*args, **kwargs)

            layer.shared_expert_gate_proj.device = device
            layer.shared_expert_gate_proj.weight = (
                self.shared_expert_gate_proj.weight.shard(n, device)
            )

            layer.shared_expert_down_proj.device = device
            layer.shared_expert_down_proj.weight = (
                self.shared_expert_down_proj.weight.shard(n, device)
            )

            layer.shared_expert_up_proj.device = device
            layer.shared_expert_up_proj.weight = (
                self.shared_expert_up_proj.weight.shard(n, device)
            )

            layer.down_proj = self.down_proj.shard(n, device)
            layer.gate_up_proj = self.gate_up_proj.shard(n, device)

            layer.router.device = device
            layer.router.weight = self.router.weight.shard(n, device)

            self.moe_layers.append(layer)

        self.allreduce = Allreduce(num_accelerators=self.num_devices)

    def __call__(  # type: ignore[override]
        self,
        hidden_states: list[TensorValue],
        signal_buffers: list[BufferValue],
    ) -> list[TensorValue]:
        """Applies a distributed Mixture of Experts layer to the input hidden states.

        Args:
            hidden_states: The input hidden states to the MoE layer.

        Returns:
            A list of output tensors from the MoE layer.
        """
        moe_outs = [
            self.moe_layers[i](hidden_states[i])
            for i in range(self.num_devices)
        ]
        return self.allreduce(moe_outs, signal_buffers)
