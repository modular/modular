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
"""A generalized Mixture of Experts (MoE) module.

These classes need to be cleaned up before moving to `max.nn`.
"""

from __future__ import annotations

from collections.abc import Callable

from max.driver import CPU, Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Linear
from max.experimental.nn.common_layers.functional_kernels import (
    Independent,
    fused_silu,
    grouped_matmul_ragged,
    moe_create_indices,
    shard_and_stack,
)
from max.experimental.nn.common_layers.mlp import MLP
from max.experimental.nn.module import Module
from max.experimental.nn.sequential import ModuleList
from max.experimental.realization_context import ensure_context
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    Partial,
    PlacementMapping,
)
from max.experimental.tensor import Tensor
from max.graph import TensorValue
from max.nn.comm.ep import EPBatchManager
from typing_extensions import Self


class MoEGate(Module[[Tensor], tuple[Tensor, Tensor]]):
    """Gate module for MoE."""

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
    ) -> None:
        """Initialize MoE gate.

        Args:
            hidden_dim: The dimension of the hidden state.
            num_experts: The number of experts.
            num_experts_per_token: The number of experts per token.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token

        self.gate_score = Linear(
            in_dim=hidden_dim, out_dim=num_experts, bias=False
        )

    def forward(self, hidden_state: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass for MoE gate.

        Args:
            hidden_state: The hidden state of the model.

        Returns:
            A tuple of the topk indices and scores.
        """
        scores = self.gate_score(hidden_state)
        topk_scores, topk_indices = F.top_k(
            scores, k=self.num_experts_per_token, axis=-1
        )

        return topk_indices, topk_scores


class MoE(Module[[Tensor], Tensor]):
    """Implementation of Mixture of Experts (MoE)."""

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
    ):
        """Initialize MoE layer.

        Args:
            hidden_dim: The dimension of the hidden state.
            num_experts: The number of experts.
            num_experts_per_token: The number of experts per token.
            moe_dim: The intermediate dimension of each expert.
            gate_cls: The model specific gate implementation.
            has_shared_experts: Whether to use shared experts.
            shared_experts_dim: The dimension of the shared experts.
            apply_router_weight_first: Whether to apply the router weight first.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.apply_router_weight_first = apply_router_weight_first
        self.gate = gate_cls(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
        )
        self.moe_dim = moe_dim

        self.shared_experts: MLP | None = None
        if has_shared_experts:
            assert shared_experts_dim > 0, (
                "shared_experts_dim must be greater than 0"
            )
            self.shared_experts = MLP(
                hidden_dim=self.hidden_dim,
                feed_forward_length=shared_experts_dim,
                bias=False,
            )

        self._init_experts()

    def _init_experts(self) -> None:
        self.experts: ModuleList[MLP] = ModuleList(
            [
                MLP(
                    hidden_dim=self.hidden_dim,
                    feed_forward_length=self.moe_dim,
                    bias=False,
                )
                for _ in range(self.num_experts)
            ]
        )

    def to(self, target: Device | DeviceMesh | DeviceMapping) -> Self:
        """Transfer layers with fully replicated placements."""
        mesh = _mesh(target)
        if mesh.num_devices > 1:
            raise ValueError(
                "Cannot transfer MoE Layer to multi-device mesh. Use TensorParallelMoE or ExpertParallelMoE instead."
            )
        super().to(target)
        return self

    @property
    def gate_up_proj(self) -> Tensor:
        """Return the stacked expert gate and up projection weights."""
        gate_list = [expert.gate_proj.weight for expert in self.experts]
        up_list = [expert.up_proj.weight for expert in self.experts]

        gate_up_list: list[Tensor] = []
        for tensors in zip(gate_list, up_list, strict=True):
            gate_up_list.extend(tensors)

        return F.stack(gate_up_list, axis=0).reshape(
            [self.num_experts, -1, self.hidden_dim]
        )

    @property
    def down_proj(self) -> Tensor:
        """Return the stacked expert down projection weights."""
        down_proj = F.stack(
            [expert.down_proj.weight for expert in self.experts], axis=0
        )
        return down_proj

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for MoE layer.

        Args:
            x: (seq_len, hidden_dim)

        Returns:
            Tensor with shape (seq_len, hidden_dim)
        """
        seq_len = x.shape[0]

        # Get the topk experts per token and their weights
        router_idx, router_weight = self.gate(x)
        router_idx = F.reshape(
            router_idx, [-1]
        )  # (seq_len * n_expert_per_token,)

        (
            token_expert_order,
            expert_start_indices,
            restore_token_order,
            expert_ids,
            expert_usage_stats,
        ) = moe_create_indices(
            F.cast(router_idx, DType.int32), self.num_experts
        )

        permutated_states = F.gather(
            x,
            F.cast(
                token_expert_order // self.num_experts_per_token, DType.int32
            ),
            axis=0,
        )

        if self.apply_router_weight_first:
            permutated_states = permutated_states * F.gather(
                router_weight.reshape([-1, 1]), token_expert_order, axis=0
            ).cast(x.dtype)

        gate_up_projs = grouped_matmul_ragged(
            permutated_states,
            self.gate_up_proj,
            expert_start_indices,
            expert_ids,
            expert_usage_stats.to(CPU()),
        )

        gate_up_projs = fused_silu(gate_up_projs, expert_start_indices)

        down_projs = grouped_matmul_ragged(
            gate_up_projs,
            self.down_proj,
            expert_start_indices,
            expert_ids,
            expert_usage_stats.to(CPU()),
        )

        down_projs = F.gather(down_projs, restore_token_order, axis=0).reshape(
            [seq_len, self.num_experts_per_token, -1]
        )

        if not self.apply_router_weight_first:
            # (seq_len, 1, n_expert) @ (seq_len, n_expert, hidden_dim) -> (seq_len, 1, hidden_dim)
            routed_expert_out = F.unsqueeze(router_weight, axis=1) @ down_projs
            routed_expert_out = F.squeeze(routed_expert_out, axis=1).cast(
                x.dtype
            )
        else:
            routed_expert_out = down_projs.transpose(1, 2)
            routed_expert_out = F.squeeze(
                F.sum(routed_expert_out, axis=2), axis=2
            ).cast(x.dtype)

        if self.shared_experts is not None:
            routed_expert_out += self.shared_experts(x)

        return routed_expert_out


class TensorParallelMoE(MoE):
    """MoE layer with tensor parallelism."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._initial_moe_dim = self.moe_dim
        self.mesh = DeviceMesh.single(self.device)

    def to(self, target: Device | DeviceMesh | DeviceMapping) -> Self:
        """Transfer the MoE layer to the target device."""
        self.mesh = _mesh(target)
        if self.mesh.ndim != 1:
            raise ValueError(
                f"Mesh used with TensorParallelMoE must have exactly one device axis, but got {self.mesh}"
            )
        self.moe_dim = self._initial_moe_dim // self.mesh.num_devices

        self.gate.to(target)
        if self.shared_experts is not None:
            self.shared_experts.to(target)

        # If there are multiple devices, keep expert weights on CPU because
        # the weights will be transferred via the `shard_and_stack` operation.
        device = CPU() if self.mesh.num_devices > 1 else self.mesh.devices[0]
        for expert in self.experts:
            expert.to(device)
        return self

    @property
    def gate_up_proj(self) -> Tensor:
        """Return the stacked expert gate and up projection weights."""
        if self.mesh.num_devices == 1:
            return super().gate_up_proj

        gate_list = [expert.gate_proj.weight for expert in self.experts]
        up_list = [expert.up_proj.weight for expert in self.experts]

        gate_up_list: list[Tensor] = []
        for tensors in zip(gate_list, up_list, strict=True):
            gate_up_list.extend(tensors)

        shards: list[TensorValue] = []
        for shard in shard_and_stack(gate_up_list, devices=self.mesh.devices):
            shards.append(
                TensorValue(
                    shard.reshape([self.num_experts, -1, self.hidden_dim])
                )
            )
        return Tensor.from_shard_values(
            shards,
            mapping=PlacementMapping(self.mesh, (Independent(),)),
        )

    @property
    def down_proj(self) -> Tensor:
        """Return the stacked expert down projection weights."""
        if self.mesh.num_devices == 1:
            return super().down_proj
        down_proj_list = [expert.down_proj.weight for expert in self.experts]
        shards: list[TensorValue] = []
        for shard in shard_and_stack(
            down_proj_list, devices=self.mesh.devices, axis=-1
        ):
            shards.append(TensorValue(shard))
        return Tensor.from_shard_values(
            shards,
            mapping=PlacementMapping(self.mesh, (Independent(),)),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for tensor parallel MoE layer.

        This method assumes that `x` is already replicated along the mesh.
        """
        with ensure_context():
            output = super().forward(x)

            # Weights are marked as Independent/Replicated for correct per-device op
            # dispatch, so the output inherits the replicated placement. The outputs
            # are actually partial, so re-tag with the correct placement.
            return Tensor.from_shard_values(
                [TensorValue(s) for s in output.local_shards],
                mapping=PlacementMapping(self.mesh, (Partial(),)),
            )


class ExpertParallelMoE(MoE):
    """MoE layer with expert parallelism."""

    def __init__(
        self, *args, ep_batch_manager: EPBatchManager, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.ep_batch_manager = ep_batch_manager
        self.mesh = DeviceMesh.single(self.device)

    def to(self, target: Device | DeviceMesh | DeviceMapping) -> Self:
        """Transfer the MoE layer to the target device."""
        self.mesh = _mesh(target)
        if self.mesh.ndim != 1:
            raise ValueError(
                f"Mesh used with ExpertParallelMoE must have exactly one device axis, but got {self.mesh}"
            )

        self.gate.to(target)
        if self.shared_experts is not None:
            self.shared_experts.to(target)

        # Move experts to different devices
        num_local_experts = self.num_experts // self.mesh.num_devices
        for i in range(self.mesh.num_devices):
            for j in range(num_local_experts):
                self.experts[i * num_local_experts + j].to(self.mesh.devices[i])

        return self

    @property
    def gate_up_proj(self) -> Tensor:
        """Return the stacked expert gate and up projection weights."""
        # Stack the weights for each expert, on the same device.
        gate_up_list_per_device: list[list[Tensor]] = [
            [] for _ in self.mesh.devices
        ]
        device_to_idx = {
            device: i for i, device in enumerate(self.mesh.devices)
        }

        if self.ep_batch_manager.config.fused_shared_expert:
            assert self.shared_experts is not None, (
                "Shared experts must be present if fused shared expert is enabled"
            )
            for i in range(self.mesh.num_devices):
                gate_up_list_per_device[i].append(
                    self.shared_experts.gate_proj.weight.local_shards[i]
                )
                gate_up_list_per_device[i].append(
                    self.shared_experts.up_proj.weight.local_shards[i]
                )

        for expert in self.experts:
            device_idx = device_to_idx[expert.device]
            gate_up_list_per_device[device_idx].append(expert.gate_proj.weight)
            gate_up_list_per_device[device_idx].append(expert.up_proj.weight)

        gate_up_tensors = []
        num_local_experts = self.num_experts // self.mesh.num_devices
        for i in range(self.mesh.num_devices):
            gate_up = F.stack(gate_up_list_per_device[i], axis=0).reshape(
                [num_local_experts, -1, self.hidden_dim]
            )
            gate_up_tensors.append(TensorValue(gate_up))

        return Tensor.from_shard_values(
            gate_up_tensors,
            mapping=PlacementMapping(self.mesh, (Independent(),)),
        )

    @property
    def down_proj(self) -> Tensor:
        """Return the stacked expert down projection weights."""
        down_proj_list_per_device: list[list[Tensor]] = [
            [] for _ in self.mesh.devices
        ]
        device_to_idx = {
            device: i for i, device in enumerate(self.mesh.devices)
        }
        for expert in self.experts:
            device_idx = device_to_idx[expert.device]
            down_proj_list_per_device[device_idx].append(
                expert.down_proj.weight
            )

        down_proj_tensors = []
        for i in range(self.mesh.num_devices):
            down_proj = F.stack(down_proj_list_per_device[i], axis=0)
            down_proj_tensors.append(TensorValue(down_proj))

        return Tensor.from_shard_values(
            down_proj_tensors,
            mapping=PlacementMapping(self.mesh, (Independent(),)),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for expert parallel MoE layer.

        Mirrors :func:`max.nn.moe.expert_parallel._ep_forward`:
        per-device gate -> cross-device dispatch -> per-device local expert
        compute -> cross-device combine.
        """
        # Per-device gate computation. Returns distributed Tensors.
        router_idx, router_weight = self.gate(x)
        router_idx = router_idx.cast(DType.int32)

        # Per-device shards as raw graph values for the EP batch manager.
        x_shards = [TensorValue(s) for s in x.local_shards]
        topk_id_shards = [TensorValue(s) for s in router_idx.local_shards]
        router_weight_shards = [
            TensorValue(s) for s in router_weight.local_shards
        ]
        device_ids = [d.id for d in self.mesh.devices]

        batch_mgr = self.ep_batch_manager
        config = batch_mgr.config

        # Dispatch tokens to the device that owns their assigned expert.
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

        # Re-package the per-device dispatch outputs into distributed
        # Tensors so we can drive the local expert compute through the
        # standard F.functional dispatch path.
        placement = PlacementMapping(self.mesh, (Independent(),))
        dispatched_tokens = Tensor.from_shard_values(
            [r[0] for r in dispatch_results], mapping=placement
        )
        # Common grouped-matmul metadata across devices: (row_offsets,
        # expert_ids, expert_usage_stats). For BF16 dispatch this is
        # `dispatch_results[i][1:]`.
        meta_tensors = [
            Tensor.from_shard_values(
                [r[j] for r in dispatch_results], mapping=placement
            )
            for j in range(1, len(dispatch_results[0]))
        ]

        # Local expert compute: gate/up grouped matmul, silu*up,
        # down grouped matmul.
        gate_up = grouped_matmul_ragged(
            dispatched_tokens, self.gate_up_proj, *meta_tensors
        )
        silu_out = fused_silu(gate_up, meta_tensors[0])
        down = grouped_matmul_ragged(silu_out, self.down_proj, *meta_tensors)

        # Combine expert outputs back to their source devices.
        down_shards = [TensorValue(s) for s in down.local_shards]
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

        # Optional shared-expert add, then cast back to input dtype.
        shared_shards: list[TensorValue] | None = None
        if self.shared_experts is not None and not config.fused_shared_expert:
            shared_shards = [
                TensorValue(s) for s in self.shared_experts(x).local_shards
            ]

        outputs: list[TensorValue] = []
        for i in range(self.mesh.num_devices):
            out = combine_results[i]
            if shared_shards is not None:
                out = out + shared_shards[i]
            outputs.append(out.cast(x_shards[i].dtype))

        return Tensor.from_shard_values(outputs, mapping=placement)


def _mesh(target: Device | DeviceMesh | DeviceMapping) -> DeviceMesh:
    """Compute the mesh for the target device."""
    if isinstance(target, DeviceMesh):
        return target
    elif isinstance(target, Device):
        return DeviceMesh.single(target)
    elif isinstance(target, DeviceMapping):
        return target.mesh
    else:
        raise ValueError(f"Invalid target device: {target}")
