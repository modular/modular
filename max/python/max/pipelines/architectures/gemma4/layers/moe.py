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
"""Gemma4 Mixture of Experts."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn.layer import LayerList
from max.nn.moe.moe import MoE, MoEGate, ShardingStrategy
from max.pipelines.architectures.gemma4.layers.rms_norm import Gemma4RMSNorm
from typing_extensions import Self


class Gemma4MoEGate(MoEGate):
    """Gemma4 MoE gate with RMSNorm, learned scale, and per-expert scaling."""

    def __init__(
        self,
        devices: list[DeviceRef],
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        dtype: DType = DType.bfloat16,
        is_sharding: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(
            devices=devices,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            dtype=dtype,
            is_sharding=is_sharding,
        )
        self.scalar_root_size = hidden_dim**-0.5
        self.eps = eps

        # The norm has no weights, so it is the same on every device.
        self.norm = Gemma4RMSNorm(
            dim=hidden_dim, dtype=dtype, eps=eps, with_weight=False
        )
        # scale and per_expert_scale are replicated across devices; shard()
        # replaces them with per-device copies.
        self.scale = Weight(
            name="scale", dtype=dtype, shape=[hidden_dim], device=devices[0]
        )
        self.per_expert_scale = Weight(
            name="per_expert_scale",
            dtype=dtype,
            shape=[num_experts],
            device=devices[0],
        )

    def _set_sharding_strategy(self, strategy: ShardingStrategy) -> None:
        if not strategy.is_replicate:
            raise ValueError(
                "Only replicate sharding strategy is supported for"
                " Gemma4MoEGate."
            )
        super()._set_sharding_strategy(strategy)
        replicate = ShardingStrategy.replicate(strategy.num_devices)
        self.scale.sharding_strategy = replicate
        self.per_expert_scale.sharding_strategy = replicate

    def shard(self, devices: Iterable[DeviceRef]) -> Sequence[Gemma4MoEGate]:
        """Create replicated views of this gate across multiple devices."""
        if not self._sharding_strategy:
            raise ValueError(
                "Gemma4MoEGate cannot be sharded because no sharding strategy"
                " was provided."
            )
        devices = list(devices)
        gate_score_shards = self.gate_score.shard(devices)
        scale_shards = self.scale.shard(devices)
        per_expert_scale_shards = self.per_expert_scale.shard(devices)

        shards: list[Gemma4MoEGate] = []
        for shard_idx, device in enumerate(devices):
            sharded = Gemma4MoEGate(
                devices=[device],
                hidden_dim=self.hidden_dim,
                num_experts=self.num_experts,
                num_experts_per_token=self.num_experts_per_token,
                dtype=self.dtype,
                is_sharding=True,
                eps=self.eps,
            )
            sharded.gate_score = gate_score_shards[shard_idx]
            sharded.scale = scale_shards[shard_idx]
            sharded.per_expert_scale = per_expert_scale_shards[shard_idx]
            shards.append(sharded)
        return shards

    def __call__(
        self, hidden_state: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        hidden_state = self.norm(hidden_state)
        hidden_state = hidden_state * self.scale * self.scalar_root_size

        expert_scores = self.gate_score(hidden_state)
        # HF gemma4 computes the router softmax and top-k renormalization in
        # float32 (Gemma4TextRouter.forward: `softmax(..., dtype=torch.float32)`)
        # for numerical stability across the 128-expert logits.
        router_probs = ops.softmax(ops.cast(expert_scores, DType.float32))

        top_k_weights, top_k_index = ops.top_k(
            router_probs, k=self.num_experts_per_token, axis=-1
        )

        top_k_weights = top_k_weights / ops.sum(top_k_weights, axis=-1)
        top_k_weights = top_k_weights * ops.cast(
            ops.gather(self.per_expert_scale, top_k_index, axis=0),
            DType.float32,
        )

        return top_k_index, ops.cast(top_k_weights, expert_scores.dtype)


class Gemma4MoE(MoE):
    """Gemma4's routed experts, declared as the BF16 checkpoints store them.

    ``experts.gate_up_proj`` is ``[num_experts, 2 * moe_dim, hidden_dim]``,
    each expert's gate rows then its up rows, and ``experts.down_proj`` is
    ``[num_experts, hidden_dim, moe_dim]``: the stacks the grouped matmuls
    read. The NVFP4 checkpoints store each expert on its own, as
    :class:`~max.nn.moe.MoEQuantized` declares them.
    """

    def _init_experts(self) -> None:
        # The expert list stays empty: MoE reads it only to stack per-expert
        # weights, which the properties below replace.
        self._all_experts = []
        self.experts = LayerList([])
        self.gate_up_weight = Weight(
            "experts.gate_up_proj",
            self.dtype,
            [self.num_experts, 2 * self.moe_dim, self.hidden_dim],
            device=self.devices[0],
        )
        self.down_weight = Weight(
            "experts.down_proj",
            self.dtype,
            [self.num_experts, self.hidden_dim, self.moe_dim],
            device=self.devices[0],
        )

    @property
    def gate_up_proj(self) -> TensorValue:
        return self.gate_up_weight

    @property
    def down_proj(self) -> TensorValue:
        return self.down_weight

    def _set_sharding_strategy(self, strategy: ShardingStrategy) -> None:
        if not strategy.is_tensor_parallel:
            raise ValueError("Gemma4MoE supports tensor parallelism only.")
        super()._set_sharding_strategy(strategy)
        # Every device takes a slice of each expert's width: gate_up splits
        # its gate and up rows separately, so they stay paired, and down
        # splits its input columns.
        n = strategy.num_devices
        self.gate_up_weight.sharding_strategy = ShardingStrategy.gate_up(
            n, axis=1
        )
        self.down_weight.sharding_strategy = ShardingStrategy.axiswise(
            axis=2, num_devices=n
        )

    def shard(self, devices: Iterable[DeviceRef]) -> list[Self]:
        """Creates one per-device view of this layer."""
        devices = list(devices)
        gate_up_shards = self.gate_up_weight.shard(devices)
        down_shards = self.down_weight.shard(devices)
        shards = super().shard(devices)
        for shard, gate_up, down in zip(
            shards, gate_up_shards, down_shards, strict=True
        ):
            shard.gate_up_weight = gate_up
            shard.down_weight = down
        return shards
