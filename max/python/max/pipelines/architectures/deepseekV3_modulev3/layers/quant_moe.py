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
    fused_silu,
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
from max.nn.comm.ep import EPBatchManager
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


def _stack_experts(
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


def _local_expert_matmul(
    tokens: Tensor,
    gate_up: QuantAwareTensor,
    down: QuantAwareTensor,
    expert_start: Tensor,
    expert_ids: Tensor,
    usage_stats: Tensor,
) -> Tensor:
    """Runs local expert matmuls on dispatched tokens."""
    gate_up_out = quant_ops.grouped_matmul(
        tokens,
        gate_up,
        expert_start,
        expert_ids,
        usage_stats,
    )
    silu_out = fused_silu(gate_up_out, expert_start)
    return quant_ops.grouped_matmul(
        silu_out,
        down,
        expert_start,
        expert_ids,
        usage_stats,
    )


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
            _stack_experts(
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
            _stack_experts(
                per_expert,
                shard_axis=None,
                mesh=DeviceMesh.single(self.device),
            )
        ]

    def apply_experts(
        self,
        permuted_states: Tensor,
        gate_up: QuantAwareTensor | list[QuantAwareTensor],
        down: QuantAwareTensor | list[QuantAwareTensor],
        expert_start_indices: Tensor,
        expert_ids: Tensor,
        expert_usage_stats: Tensor,
        restore_token_order: Tensor,
        router_weight: Tensor,
    ) -> Tensor:
        """Compute a single-device output for the routed experts."""
        if isinstance(gate_up, list):
            gate_up = gate_up[0]
        if isinstance(down, list):
            down = down[0]
        dtype = permuted_states.dtype

        down_projs = _local_expert_matmul(
            permuted_states,
            gate_up,
            down,
            expert_start_indices,
            expert_ids,
            expert_usage_stats,
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

        expert_usage_stats = expert_usage_stats.to(CPU())

        routed_expert_out = self.apply_experts(
            permuted_states,
            self.gate_up_proj,
            self.down_proj,
            expert_start_indices,
            expert_ids,
            expert_usage_stats,
            restore_token_order,
            router_weight,
        )

        if self.shared_experts is not None:
            routed_expert_out += self.shared_experts(x)
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

        distributed = _stack_experts(down_list, shard_axis=-1, mesh=self.mesh)
        return list(distributed.local_shards)

    def apply_experts(
        self,
        permuted_states: Tensor,
        gate_up: QuantAwareTensor | list[QuantAwareTensor],
        down: QuantAwareTensor | list[QuantAwareTensor],
        expert_start_indices: Tensor,
        expert_ids: Tensor,
        expert_usage_stats: Tensor,
        restore_token_order: Tensor,
        router_weight: Tensor,
    ) -> Tensor:
        """Compute a single-device output for the routed experts."""
        assert isinstance(gate_up, list)
        assert isinstance(down, list)
        out = local_map(
            super().apply_experts,
            {
                "permuted_states": permuted_states,
                "gate_up": self.gate_up_proj,
                "down": self.down_proj,
                "expert_start_indices": expert_start_indices,
                "expert_ids": expert_ids,
                "expert_usage_stats": expert_usage_stats,
                "restore_token_order": restore_token_order,
                "router_weight": router_weight,
            },
            {},
        )
        out_tensor = Tensor.from_shard_values(
            [TensorValue(s) for s in out],
            mapping=PlacementMapping(self.mesh, (Partial(),)),
        )
        return out_tensor


class ExpertParallelMoE(QuantizedMoE):
    """Quantize-aware MoE with expert parallelism.

    Each device owns ``num_experts / n_devices`` routed experts. Tokens are
    routed per device, dispatched to the device owning their assigned expert,
    computed locally, and combined back at the end.
    """

    def __init__(
        self, *args, ep_batch_manager: EPBatchManager, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.ep_batch_manager = ep_batch_manager
        self.mesh = DeviceMesh.single(self.device)

    def to(self, target: Device | DeviceMesh | DeviceMapping) -> Self:
        """Distribute routed experts round-robin across the mesh devices."""
        mesh = _mesh(target)
        if mesh.ndim != 1:
            raise ValueError(
                "Mesh used with ExpertParallelMoE must have exactly one device"
                f" axis, but got {mesh}"
            )
        if self.num_experts % mesh.num_devices != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by the "
                f"number of devices ({mesh.num_devices}) for expert parallelism"
            )
        self.mesh = mesh

        # Gate and shared experts stay replicated.
        self.gate.to(target)
        if self.shared_experts is not None:
            self.shared_experts.to(target)

        num_local_experts = self.num_experts // mesh.num_devices
        for i in range(mesh.num_devices):
            for j in range(num_local_experts):
                self.experts[i * num_local_experts + j].to(mesh.devices[i])
        return self

    # ----- EP weight stacking ------------------------------------------------

    @property
    def gate_up_proj(self) -> list[QuantAwareTensor]:
        """Per-device stacked ``[gate, up]`` weight bundle for local experts."""
        device_to_idx = {d: i for i, d in enumerate(self.mesh.devices)}
        per_device: list[list[QuantAwareTensor]] = [
            [] for _ in self.mesh.devices
        ]

        config = self.ep_batch_manager.config
        if config.fused_shared_expert and self.shared_experts is not None:
            gate_w = self.shared_experts.gate_proj.weight
            up_w = self.shared_experts.up_proj.weight
            for i in range(self.mesh.num_devices):
                per_device[i].append(
                    quant_ops.concat_weights(
                        gate_w.local_shards[i], up_w.local_shards[i], axis=0
                    )
                )

        for expert in self.experts:
            assert isinstance(expert, QuantizedMLP)
            idx = device_to_idx[expert.device]
            per_device[idx].append(
                quant_ops.concat_weights(
                    expert.gate_proj.weight, expert.up_proj.weight, axis=0
                )
            )
        return [quant_ops.stack(local, axis=0) for local in per_device]

    @property
    def down_proj(self) -> list[QuantAwareTensor]:
        """Per-device stacked down-projection weight bundle for local experts."""
        device_to_idx = {d: i for i, d in enumerate(self.mesh.devices)}
        per_device: list[list[QuantAwareTensor]] = [
            [] for _ in self.mesh.devices
        ]
        if self.ep_batch_manager.config.fused_shared_expert:
            assert self.shared_experts is not None, (
                "Shared experts must present if fused shared expert is enabled"
            )
            for i in range(self.mesh.num_devices):
                per_device[i].append(
                    self.shared_experts.down_proj.weight.local_shards[i]
                )
        for expert in self.experts:
            assert isinstance(expert, QuantizedMLP)
            idx = device_to_idx[expert.device]
            per_device[idx].append(expert.down_proj.weight)
        return [quant_ops.stack(local, axis=0) for local in per_device]

    def forward(self, x: Tensor) -> Tensor:
        """Expert-parallel forward: gate -> dispatch -> local compute -> combine."""
        batch_mgr = self.ep_batch_manager
        config = batch_mgr.config

        # Per-device gate computation (replicated router scores).
        router_idx, router_weight = self.gate(x)
        router_idx = router_idx.cast(DType.int32)

        x_shards = [TensorValue(s) for s in x.local_shards]
        topk_id_shards = [TensorValue(s) for s in router_idx.local_shards]
        router_weight_shards = [
            TensorValue(s) for s in router_weight.local_shards
        ]
        device_ids = [d.id for d in self.mesh.devices]

        # Dispatch tokens to the device owning each routed expert.
        if config.use_allreduce:
            dispatch_results = [
                batch_mgr.ep_dispatch(
                    x_shards[i], topk_id_shards[i], device_ids[i]
                )
                for i in range(self.mesh.num_devices)
            ]
        else:
            dispatch_results = batch_mgr.ep_dispatch_all(
                x_shards, topk_id_shards, device_ids
            )

        # Now each device runs its own experts on the tokens it was sent.
        tokens, expert_start, expert_ids, usage_stats = (
            [Tensor.from_graph_value(v) for v in column]
            for column in zip(*dispatch_results, strict=True)
        )
        for stat in usage_stats:
            assert stat.device.is_host
        down_bundle = local_map(
            _local_expert_matmul,
            {
                "tokens": tokens,
                "gate_up": self.gate_up_proj,
                "down": self.down_proj,
                "expert_start": expert_start,
                "expert_ids": expert_ids,
                "usage_stats": usage_stats,
            },
            {},
        )
        down_shards = [TensorValue(t) for t in down_bundle]

        # Combine expert outputs back to their source devices.
        if config.use_allreduce:
            combine_results = [
                batch_mgr.ep_combine(
                    down_shards[i],
                    router_weight_shards[i],
                    device_ids[i],
                    topk_id_shards[i],
                )
                for i in range(self.mesh.num_devices)
            ]
        else:
            combine_results = batch_mgr.ep_combine_all(
                down_shards, router_weight_shards, device_ids
            )

        # Optional (unfused) shared-expert add, then cast back to input dtype.
        shared_shards: list[TensorValue] | None = None
        if self.shared_experts is not None and not config.fused_shared_expert:
            shared_shards = [
                TensorValue(s) for s in self.shared_experts(x).local_shards
            ]

        # ``ep_combine`` returns each device exactly the tokens it dispatched,
        # so the output placement matches the input's.
        placement = PlacementMapping(self.mesh, x.placements)
        outputs: list[TensorValue] = []
        for i in range(self.mesh.num_devices):
            out = combine_results[i]
            if shared_shards is not None:
                out = out + shared_shards[i]
            outputs.append(out.cast(x_shards[i].dtype))
        return Tensor.from_shard_values(outputs, mapping=placement)
