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

"""Quantize-aware MoE layers."""

from __future__ import annotations

from collections.abc import Callable

from max.driver import CPU, Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.functional_kernels import (
    local_map,
    moe_create_indices,
    shard_and_stack,
)
from max.experimental.nn.common_layers.moe import MoEGate
from max.experimental.nn.sequential import ModuleList
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    Partial,
    PlacementMapping,
    Sharded,
)
from max.experimental.tensor import Tensor
from max.graph import DimLike, TensorValue
from max.nn.comm.ep.ep_kernels import fused_silu as _ep_fused_silu
from max.nn.quant_config import QuantConfig
from typing_extensions import Self

from . import quant_ops
from .quant_linear import QuantizedMLP, tensor_parallel_mlp
from .quant_ops import QuantAwareTensor
from .quant_tensor import FP8BlockTensor


def _mesh(target: Device | DeviceMesh | DeviceMapping) -> DeviceMesh:
    """Resolve a transfer target to a :class:`DeviceMesh`."""
    if isinstance(target, DeviceMesh):
        return target
    if isinstance(target, DeviceMapping):
        return target.mesh
    return DeviceMesh.single(target)


def stack_experts(
    per_expert: list[QuantAwareTensor],
    *,
    shard_axis: int | None,
    mesh: DeviceMesh,
) -> QuantAwareTensor:
    """Stack per-expert weights and optionally scatter across a mesh.

    Args:
        per_expert: One :class:`~.quant_ops.QuantAwareTensor` per global
            expert (homogeneous: all bf16 or all
            :class:`~.quant_tensor.FP8BlockTensor`).
        shard_axis: Tensor axis to shard along for TP; ``None`` for no shard.
        mesh: :class:`~max.experimental.sharding.DeviceMesh` to scatter onto.

    Returns:
        A single stacked :class:`~.quant_ops.QuantAwareTensor`, distributed
        when ``shard_axis`` is provided and ``mesh.num_devices > 1``.
    """
    stacked = quant_ops.stack(per_expert, axis=0)
    if shard_axis is None or mesh.num_devices == 1:
        return stacked
    # Scatter: bf16 path uses Tensor.to(Sharded); FP8 uses FP8BlockTensor.shard.
    if isinstance(stacked, FP8BlockTensor):
        return stacked.shard(shard_axis, mesh)
    assert isinstance(stacked, Tensor)
    return stacked.to(PlacementMapping(mesh, (Sharded(axis=shard_axis),)))


class QuantizedMoE(Module[[Tensor], Tensor]):
    """Mixture of Experts with quantize-aware expert weights."""

    gate: MoEGate
    experts: ModuleList
    shared_experts: QuantizedMLP | None

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        moe_dim: int,
        gate_cls: Callable[..., MoEGate] = MoEGate,
        has_shared_experts: bool = False,
        shared_experts_dim: int = 0,
        apply_router_weight_first: bool = False,
        quant_config: QuantConfig | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.apply_router_weight_first = apply_router_weight_first
        self.moe_dim = moe_dim
        self.quant_config = quant_config

        self.gate = gate_cls(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
        )

        self.shared_experts: QuantizedMLP | None = None
        if has_shared_experts:
            assert shared_experts_dim > 0
            self.shared_experts = QuantizedMLP(
                hidden_dim=hidden_dim,
                feed_forward_length=shared_experts_dim,
                quant_config=quant_config,
            )

        self.experts = ModuleList(
            [
                QuantizedMLP(
                    hidden_dim=hidden_dim,
                    feed_forward_length=moe_dim,
                    quant_config=quant_config,
                )
                for _ in range(num_experts)
            ]
        )

    @property
    def gate_up_proj(self) -> list[QuantAwareTensor]:
        """Per-device ``[gate, up]`` expert-weight bundle."""
        per_expert: list[QuantAwareTensor] = []
        for expert in self.experts:
            assert isinstance(expert, QuantizedMLP)
            per_expert.append(
                quant_ops.concat_weights(
                    expert.gate_proj.weight, expert.up_proj.weight, axis=0
                )
            )
        return [
            stack_experts(
                per_expert,
                shard_axis=None,
                mesh=DeviceMesh.single(self.device),
            )
        ]

    @property
    def down_proj(self) -> list[QuantAwareTensor]:
        """Per-device down-projection weight bundle (bf16 or FP8; one entry)."""
        per_expert: list[QuantAwareTensor] = []
        for expert in self.experts:
            assert isinstance(expert, QuantizedMLP)
            per_expert.append(expert.down_proj.weight)
        return [
            stack_experts(
                per_expert,
                shard_axis=None,
                mesh=DeviceMesh.single(self.device),
            )
        ]

    def _local_expert_matmul(
        self,
        tokens: Tensor,
        gate_up: QuantAwareTensor,
        down: QuantAwareTensor,
        expert_start: Tensor,
        expert_ids: Tensor,
        usage_stats: Tensor,
    ) -> Tensor:
        """Runs local expert matmuls on dispatched tokens."""
        usage_cpu = usage_stats.to(CPU())

        gate_up_out = quant_ops.grouped_matmul(
            tokens,
            gate_up,
            expert_start,
            expert_ids,
            usage_cpu,
        )
        silu_out = Tensor.from_graph_value(
            _ep_fused_silu(TensorValue(gate_up_out), TensorValue(expert_start))
        )
        return quant_ops.grouped_matmul(
            silu_out,
            down,
            expert_start,
            expert_ids,
            usage_cpu,
        )

    def _local_routed_output(
        self,
        permuted_states: Tensor,
        gate_up: QuantAwareTensor,
        down: QuantAwareTensor,
        expert_start: Tensor,
        expert_ids: Tensor,
        usage_stats: Tensor,
        restore_token_order: Tensor,
        router_weight: Tensor,
        dtype: DType,
    ) -> Tensor:
        """Compute a single-device output for the routed experts."""
        down_projs = self._local_expert_matmul(
            permuted_states,
            gate_up,
            down,
            expert_start,
            expert_ids,
            usage_stats,
        )

        # Restore the original token order and weight-combine the per-token
        # expert outputs.
        seq_len = router_weight.shape[0]
        down_projs = F.gather(down_projs, restore_token_order, axis=0).reshape(
            [seq_len, self.num_experts_per_token, -1]
        )
        if not self.apply_router_weight_first:
            out = F.unsqueeze(router_weight, axis=1) @ down_projs
            return F.squeeze(out, axis=1).cast(dtype)
        out = down_projs.transpose(1, 2)
        return F.squeeze(F.sum(out, axis=2), axis=2).cast(dtype)

    def _routed_output_mapping(self, permuted_states: Tensor) -> DeviceMapping:
        """Placement of the routed output.

        This function is overridden by `TensorParallelMoE` to return a `Partial`
        mapping.
        """
        return permuted_states.mapping

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for the MoE layer.

        Args:
            x: ``(seq_len, hidden_dim)``.

        Returns:
            ``(seq_len, hidden_dim)``.
        """
        router_idx, router_weight = self.gate(x)
        router_idx = F.reshape(router_idx, [-1])

        (
            token_expert_order,
            expert_start_indices,
            restore_token_order,
            expert_ids,
            expert_usage_stats,
        ) = moe_create_indices(
            F.cast(router_idx, DType.int32), self.num_experts
        )

        permuted_states = F.gather(
            x,
            F.cast(
                token_expert_order // self.num_experts_per_token, DType.int32
            ),
            axis=0,
        )

        if self.apply_router_weight_first:
            permuted_states = permuted_states * F.gather(
                router_weight.reshape([-1, 1]), token_expert_order, axis=0
            ).cast(x.dtype)

        # Apply the expert matmuls per device, then reassemble.
        # Note: Using `local_map` to run the local experts is not needed for
        # `QuantizedMoE`, but is used in `TensorParallelMoE` to run the sharded
        # experts separately across devices.
        out = local_map(
            self._local_routed_output,
            {
                "permuted_states": permuted_states,
                "gate_up": self.gate_up_proj,
                "down": self.down_proj,
                "expert_start": expert_start_indices,
                "expert_ids": expert_ids,
                "usage_stats": expert_usage_stats,
                "restore_token_order": restore_token_order,
                "router_weight": router_weight,
            },
            {"dtype": x.dtype},
        )
        routed_expert_out = Tensor.from_shard_values(
            [TensorValue(s) for s in out],
            mapping=self._routed_output_mapping(permuted_states),
        )
        if self.shared_experts is not None:
            routed_expert_out = routed_expert_out + self.shared_experts(x)
        return routed_expert_out


class TensorParallelMoE(QuantizedMoE):
    """Quantize-aware MoE with tensor parallelism."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._initial_moe_dim = self.moe_dim
        self.mesh = DeviceMesh.single(self.device)
        if self.shared_experts is not None:
            self.shared_experts = tensor_parallel_mlp(self.shared_experts)
        for n, expert in enumerate(self.experts):
            assert isinstance(expert, QuantizedMLP)
            self.experts[n] = tensor_parallel_mlp(expert)

    def to(self, target: Device | DeviceMesh | DeviceMapping) -> Self:
        """Transfer the MoE layer to the target device mesh."""
        self.mesh = _mesh(target)
        if self.mesh.ndim != 1:
            raise ValueError(
                "Mesh used with TensorParallelMoE must have exactly one device"
                f" axis, but got {self.mesh}"
            )
        if self._initial_moe_dim % self.mesh.num_devices != 0:
            raise ValueError(
                f"moe_dim ({self._initial_moe_dim}) must be divisible by the "
                f"number of devices ({self.mesh.num_devices}) for tensor "
                "parallelism"
            )
        self.moe_dim = self._initial_moe_dim // self.mesh.num_devices

        self.gate.to(target)
        if self.shared_experts is not None:
            self.shared_experts.to(target)

        # With multiple devices, keep expert weights on CPU because they are
        # transferred (sharded) via the ``shard_and_stack`` operation. On a
        # single device, place them directly.
        device = CPU() if self.mesh.num_devices > 1 else self.mesh.devices[0]
        for expert in self.experts:
            expert.to(device)
        return self

    def _shard_stack_tensors(
        self,
        per_expert: list[Tensor],
        axis: int,
        shard_shape: list[DimLike] | None = None,
    ) -> list[Tensor]:
        """Shard each weight in ``per_expert`` along ``axis`` and stack."""
        shards: list[Tensor] = []
        for shard in shard_and_stack(per_expert, self.mesh.devices, axis=axis):
            assert isinstance(shard, Tensor)
            if shard_shape is not None:
                shard = shard.reshape(shard_shape)
            shards.append(shard)
        return shards

    @property
    def gate_up_proj(self) -> list[QuantAwareTensor]:
        """Per-device ``[gate, up]`` weight bundle, sharded along ``moe_dim``."""
        if self.mesh.num_devices == 1:
            return super().gate_up_proj

        # Interleave per-expert gate/up so each device receives a contiguous
        # ``gate||up`` block of its ``moe_dim`` slice.
        interleaved: list[QuantAwareTensor] = []
        for e in self.experts:
            assert isinstance(e, QuantizedMLP)
            interleaved.extend((e.gate_proj.weight, e.up_proj.weight))

        def _combine_gate_up(tensors: list[Tensor]) -> list[Tensor]:
            # The shard is along axis 0, so each leaf's trailing dim is
            # preserved: ``hidden_dim`` for the data leaf, ``hidden_dim //
            # block_k`` for the FP8 block-scale leaf. Read it off the leaf
            # rather than discriminating on a leaf name.
            return self._shard_stack_tensors(
                tensors,
                axis=0,
                shard_shape=[self.num_experts, -1, tensors[0].shape[-1]],
            )

        return quant_ops.combine_quant_per_device(interleaved, _combine_gate_up)

    @property
    def down_proj(self) -> list[QuantAwareTensor]:
        """Per-device down-projection weight bundle, sharded along ``moe_dim``."""
        if self.mesh.num_devices == 1:
            return super().down_proj

        down_list: list[QuantAwareTensor] = []
        for e in self.experts:
            assert isinstance(e, QuantizedMLP)
            down_list.append(e.down_proj.weight)

        distributed = stack_experts(down_list, shard_axis=-1, mesh=self.mesh)
        return list(distributed.local_shards)

    def _routed_output_mapping(self, permuted_states: Tensor) -> DeviceMapping:
        if self.mesh.num_devices == 1:
            return super()._routed_output_mapping(permuted_states)
        return PlacementMapping(self.mesh, (Partial(),))
