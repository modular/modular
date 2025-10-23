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
from max.graph import BufferValue, ShardingStrategy, TensorValue, Weight, ops
from max.nn.comm import Allreduce
from max.nn.moe import MoE, MoEGate


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


class Llama4MoE(MoE):
    """Mixture of Experts Layer for Llama-4. The key difference is that all the
    experts' weights are concatenated together.
    """

    def _init_experts(self) -> None:
        # the routed experts' weights are concatenated together, and
        # stored in a non-transposed format.
        self.gate_up_weight = Weight(
            name="experts.gate_up_proj",
            shape=(
                self.num_experts,
                self.hidden_dim,
                self.moe_dim * 2,
            ),
            dtype=self.dtype,
            device=self.devices[0],
        )

        self.down_weight = Weight(
            name="experts.down_proj",
            shape=(
                self.num_experts,
                self.moe_dim,
                self.hidden_dim,
            ),
            dtype=self.dtype,
            device=self.devices[0],
        )

    @property
    def gate_up_proj(self) -> TensorValue:
        return ops.transpose(self.gate_up_weight, 1, 2)

    @property
    def down_proj(self) -> TensorValue:
        return ops.transpose(self.down_weight, 1, 2)


class DistributedLlama4MoE(Llama4MoE):
    """A distributed Mixture of Experts layer for Llama-4.

    This class implements tensor parallelism for the MoE layer.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not self.devices or len(self.devices) < 2:
            raise ValueError(
                f"Must provide at least 2 devices to `DistributedMoE`, got {self.devices}"
            )
        self.num_devices = len(self.devices)

        self.shared_experts.gate_proj.sharding_strategy = (
            ShardingStrategy.rowwise(self.num_devices)
        )
        self.shared_experts.down_proj.sharding_strategy = (
            ShardingStrategy.columnwise(self.num_devices)
        )
        self.shared_experts.up_proj.sharding_strategy = (
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

        self.down_weight.sharding_strategy = ShardingStrategy(
            self.num_devices, down_sharding_strategy
        )
        self.gate_up_weight.sharding_strategy = ShardingStrategy(
            self.num_devices, gate_up_sharding_strategy
        )

        # we clone the router weights for each device
        self.gate.gate_score.sharding_strategy = ShardingStrategy.replicate(
            self.num_devices
        )

        # Create a separate MoE layer for each device.
        kwargs = kwargs.copy()
        kwargs["moe_dim"] = self.moe_dim // self.num_devices
        self.moe_layers = []

        # Shard weights once for all devices
        shared_gate_proj_weight_shards = (
            self.shared_experts.gate_proj.weight.shard(self.devices)
        )
        shared_down_proj_weight_shards = (
            self.shared_experts.down_proj.weight.shard(self.devices)
        )
        shared_up_proj_weight_shards = self.shared_experts.up_proj.weight.shard(
            self.devices
        )
        down_weight_shards = self.down_weight.shard(self.devices)
        gate_up_weight_shards = self.gate_up_weight.shard(self.devices)
        gate_score_weight_shards = self.gate.gate_score.weight.shard(
            self.devices
        )

        for n, device in enumerate(self.devices):
            kwargs["devices"] = [device]
            layer = Llama4MoE(*args, **kwargs)

            layer.shared_experts.gate_proj.device = device
            layer.shared_experts.gate_proj.weight = (
                shared_gate_proj_weight_shards[n]
            )

            layer.shared_experts.down_proj.device = device
            layer.shared_experts.down_proj.weight = (
                shared_down_proj_weight_shards[n]
            )

            layer.shared_experts.up_proj.device = device
            layer.shared_experts.up_proj.weight = shared_up_proj_weight_shards[
                n
            ]

            layer.down_weight = down_weight_shards[n]
            layer.gate_up_weight = gate_up_weight_shards[n]

            layer.gate.gate_score.device = device
            layer.gate.gate_score.weight = gate_score_weight_shards[n]

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
