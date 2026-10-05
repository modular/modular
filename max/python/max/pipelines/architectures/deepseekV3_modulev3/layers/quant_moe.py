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

import contextlib
import os
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from max import tree
from max.driver import CPU, Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.functional_kernels import (
    moe_create_indices,
    moe_finalize,
    shard_and_stack,
)
from max.experimental.nn.common_layers.moe import MoEGate
from max.experimental.nn.sequential import ModuleList
from max.experimental.realization_context import ensure_context
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    Sharded,
)
from max.experimental.tensor import (
    Tensor,
    default_device,
    default_dtype,
    defaults,
)
from max.graph import DimLike, TensorValue
from max.nn.comm.ep import EPBatchManager, EPCommBuffers
from max.nn.moe.expert_parallel import _SHARED_EXPERT_STREAM_ID
from max.nn.quant_config import QuantConfig
from typing_extensions import Self

from . import quant_ops
from .quant_linear import QuantizedMLP, tensor_parallel_mlp
from .quant_ops import (
    EPDispatchPayload,
    ep_requires_dispatch_scales,
    moe_requires_scales_offsets,
    routed_weight_dtype,
)
from .quant_tensor import FP8BlockTensor, NVFP4Tensor, QuantAwareTensor

_M = TypeVar("_M", bound=Module[..., Any])


def _on_device(module: _M, device: int) -> _M:
    """Returns a copy of ``module`` holding only its weights on ``device``."""
    return tree.map(
        lambda t: t.local_shards[device], module, leaf=Tensor, shared=True
    )


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
    if isinstance(stacked, (FP8BlockTensor, NVFP4Tensor)):
        return stacked.shard(shard_axis, mesh)
    assert isinstance(stacked, Tensor)
    return stacked.to(DeviceMapping(mesh, (Sharded(axis=shard_axis),)))


def _local_expert_matmul(
    tokens: QuantAwareTensor,
    gate_up: QuantAwareTensor,
    down: QuantAwareTensor,
    expert_start: Tensor,
    expert_ids: Tensor,
    usage_stats: Tensor | None = None,
    quant_config: QuantConfig | None = None,
    scales_offset: Tensor | None = None,
    estimated_total_m: Tensor | None = None,
) -> Tensor:
    """Runs local expert matmuls on dispatched tokens."""
    down_in = quant_ops.grouped_matmul_silu(
        tokens,
        gate_up,
        down,
        expert_start,
        expert_ids,
        usage_stats,
        quant_config,
        scales_offset=scales_offset,
        estimated_total_m=estimated_total_m,
    )
    return quant_ops.grouped_matmul(
        down_in,
        down,
        expert_start,
        expert_ids,
        usage_stats,
        scales_offset=scales_offset,
        estimated_total_m=estimated_total_m,
    )


def _new_expert(
    hidden_dim: int, moe_dim: int, quant_config: QuantConfig | None
) -> QuantizedMLP:
    """Create a new quantized MLP expert."""
    return QuantizedMLP(
        hidden_dim=hidden_dim,
        feed_forward_length=moe_dim,
        quant_config=quant_config,
    )


class QuantizedMoE(Module[..., Tensor]):
    """Mixture of Experts with quantize-aware expert weights."""

    gate: MoEGate
    experts: ModuleList[QuantizedMLP]
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
        quant_config: QuantConfig | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
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
            shared_experts_quant_config: QuantConfig | None = None
            if quant_config is not None:
                if quant_config.shared_experts_use_quant(
                    routed_weight_dtype(quant_config)
                ):
                    shared_experts_quant_config = quant_config
            self.shared_experts = QuantizedMLP(
                hidden_dim=hidden_dim,
                feed_forward_length=shared_experts_dim,
                quant_config=shared_experts_quant_config,
            )

        _, self.mesh = defaults()
        self.experts = self._init_experts()

    def _init_experts(self) -> ModuleList[QuantizedMLP]:
        return ModuleList(
            [
                _new_expert(self.hidden_dim, self.moe_dim, self.quant_config)
                for _ in range(self.num_experts)
            ]
        )

    def to(self, target: Device | DeviceMesh | DeviceMapping) -> Self:
        """Moves the MoE layer to a single device."""
        super().to(target)
        self.mesh = _mesh(target)
        return self

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
        return [_stack_experts(per_expert, shard_axis=None, mesh=self.mesh)]

    @property
    def down_proj(self) -> list[QuantAwareTensor]:
        """Per-device down-projection weight bundle (bf16 or FP8; one entry)."""
        per_expert: list[QuantAwareTensor] = []
        for expert in self.experts:
            assert isinstance(expert, QuantizedMLP)
            per_expert.append(expert.down_proj.weight)
        return [_stack_experts(per_expert, shard_axis=None, mesh=self.mesh)]

    def _combine_expert_outputs(
        self,
        down_projs: Tensor,
        restore_token_order: Tensor,
        router_weight: Tensor,
        dtype: DType,
    ) -> Tensor:
        """Restores token order and weight-combines the per-token expert outputs."""
        return moe_finalize(
            down_projs, restore_token_order, router_weight, dtype
        )

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
        scales_offset: Tensor | None = None,
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
            quant_config=self.quant_config,
            scales_offset=scales_offset,
        )
        return self._combine_expert_outputs(
            down_projs, restore_token_order, router_weight, dtype
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for the MoE layer.

        Args:
            x: ``(seq_len, hidden_dim)``.

        Returns:
            ``(seq_len, hidden_dim)``.
        """
        router_idx, router_weight = self.gate(x)
        router_idx = F.reshape(router_idx, [-1])

        needs_scales_offset = moe_requires_scales_offsets(self.quant_config)

        # Unpack the common five outputs, then pull the offset only when it was
        # requested — one code path, no fork on the return-tuple shape.
        (
            token_expert_order,
            expert_start_indices,
            restore_token_order,
            expert_ids,
            expert_usage_stats,
            *scales_offset_maybe,
        ) = moe_create_indices(
            F.cast(router_idx, DType.int32),
            self.num_experts,
            needs_scales_offset=needs_scales_offset,
        )
        scales_offset = scales_offset_maybe[0] if needs_scales_offset else None

        permuted_states = F.gather(
            x,
            F.cast(
                token_expert_order // self.num_experts_per_token, DType.int32
            ),
            axis=0,
        )

        routed_expert_out = self.apply_experts(
            permuted_states,
            self.gate_up_proj,
            self.down_proj,
            expert_start_indices,
            expert_ids,
            expert_usage_stats,
            restore_token_order,
            router_weight,
            scales_offset=scales_offset,
        )

        if self.shared_experts is not None:
            routed_expert_out += self.shared_experts(x)
        return routed_expert_out


class TensorParallelMoE(QuantizedMoE):
    """Quantize-aware MoE with tensor parallelism."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._initial_moe_dim = self.moe_dim
        self._set_mesh(self.mesh)
        if self.shared_experts is not None:
            self.shared_experts = tensor_parallel_mlp(self.shared_experts)
        for n, expert in enumerate(self.experts):
            assert isinstance(expert, QuantizedMLP)
            self.experts[n] = tensor_parallel_mlp(expert)

    def _init_experts(self) -> ModuleList[QuantizedMLP]:
        dtype, mesh = defaults()
        # With multiple devices, build the experts on CPU. Experts are sharded,
        # stacked, and moved to the mesh later.
        placement = (
            contextlib.nullcontext()
            if mesh.num_devices == 1
            else default_device(CPU())
        )
        with placement, default_dtype(dtype):
            return super()._init_experts()

    def _set_mesh(self, mesh: DeviceMesh) -> None:
        if mesh.ndim != 1:
            raise ValueError(
                "Mesh used with TensorParallelMoE must have exactly one device"
                f" axis, but got {mesh}"
            )
        if self._initial_moe_dim % mesh.num_devices != 0:
            raise ValueError(
                f"moe_dim ({self._initial_moe_dim}) must be divisible by the "
                f"number of devices ({mesh.num_devices}) for tensor "
                "parallelism"
            )
        self.mesh = mesh
        self.moe_dim = self._initial_moe_dim // mesh.num_devices

    def to(self, target: Device | DeviceMesh | DeviceMapping) -> Self:
        """Moves the MoE layer to a single device."""
        super().to(target)
        self._set_mesh(self.mesh)
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
        gates: list[QuantAwareTensor] = []
        for e in self.experts:
            assert isinstance(e, QuantizedMLP)
            interleaved.extend((e.gate_proj.weight, e.up_proj.weight))
            gates.append(e.gate_proj.weight)

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

        # ``interleaved`` has two entries per expert, but an NVFP4 expert has a
        # single global scale pair shared by its gate and up rows, so those come
        # off the gate weights.
        return quant_ops.combine_quant_per_device(
            interleaved, _combine_gate_up, global_scale_items=gates
        )

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
        scales_offset: Tensor | None = None,
    ) -> Tensor:
        """Compute a Partial-summed output for the routed experts under TP."""
        assert isinstance(gate_up, list)
        assert isinstance(down, list)
        dtype = permuted_states.dtype
        gate_up_t = quant_ops.stack_device_shards(
            gate_up, axis=1, mesh=self.mesh
        )
        down_t = quant_ops.stack_device_shards(down, axis=2, mesh=self.mesh)

        usage_stats = expert_usage_stats
        down_projs = _local_expert_matmul(
            permuted_states,
            gate_up_t,
            down_t,
            expert_start_indices,
            expert_ids,
            usage_stats,
            quant_config=self.quant_config,
            scales_offset=scales_offset,
        )
        # The rule-less combine runs on each device's own expert outputs; its
        # result keeps their placement.
        return F.functional(self._combine_expert_outputs)(
            down_projs, restore_token_order, router_weight, dtype
        ).rebind_mapping(down_projs.mapping)


class ExpertParallelMoE(QuantizedMoE):
    """Quantize-aware MoE with expert parallelism.

    Each device owns ``num_experts / n_devices`` routed experts. Tokens are
    routed per device, dispatched to the device owning their assigned expert,
    computed locally, and combined back at the end.

    Places the experts round-robin across the mesh from
    :func:`~max.experimental.tensor.default_device` at construction; the
    gate and shared experts are replicated.
    """

    def __init__(
        self, *args, ep_batch_manager: EPBatchManager, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.ep_batch_manager = ep_batch_manager

    @property
    def _num_local_experts(self) -> int:
        return self.num_experts // self.mesh.num_devices

    def _init_experts(self) -> ModuleList[QuantizedMLP]:
        dtype, mesh = defaults()
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
        num_local_experts = self.num_experts // mesh.num_devices
        experts: list[QuantizedMLP] = []
        for device in mesh.devices:
            with default_device(device), default_dtype(dtype):
                experts.extend(
                    _new_expert(
                        self.hidden_dim, self.moe_dim, self.quant_config
                    )
                    for _ in range(num_local_experts)
                )
        return ModuleList(experts)

    # ----- EP weight stacking ------------------------------------------------

    @property
    def _uses_fused_swiglu(self) -> bool:
        """Whether the NVFP4 EP gate/up path uses the fused SwiGLU kernel."""
        return (
            self.quant_config is not None
            and self.quant_config.can_use_fused_swiglu
        )

    @property
    def gate_up_proj(self) -> list[QuantAwareTensor]:
        """Per-device stacked ``[gate, up]`` weight bundle for local experts."""
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

        for n, expert in enumerate(self.experts):
            assert isinstance(expert, QuantizedMLP)
            idx = n // self._num_local_experts
            per_device[idx].append(
                quant_ops.concat_weights(
                    expert.gate_proj.weight, expert.up_proj.weight, axis=0
                )
            )
        stacked = [quant_ops.stack(local, axis=0) for local in per_device]
        if self._uses_fused_swiglu:
            permuted: list[QuantAwareTensor] = []
            for w in stacked:
                assert isinstance(w, NVFP4Tensor)
                permuted.append(quant_ops.sigma_permute_gate_up_nvfp4(w))
            return permuted
        return stacked

    @property
    def down_proj(self) -> list[QuantAwareTensor]:
        """Per-device stacked down-projection weight bundle for local experts."""
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
        for n, expert in enumerate(self.experts):
            assert isinstance(expert, QuantizedMLP)
            idx = n // self._num_local_experts
            per_device[idx].append(expert.down_proj.weight)
        return [quant_ops.stack(local, axis=0) for local in per_device]

    # ----- local expert compute ----------------------------------------------

    def _nvfp4_global_input_scale(self) -> Tensor:
        """Scalar max static gate ``input_scale`` across all experts."""
        gate_scales: list[Tensor] = []
        for expert in self.experts:
            assert isinstance(expert.gate_proj.weight, NVFP4Tensor)
            gate_scales.append(expert.gate_proj.weight.input_scale)
        gate_scales = F.stack(gate_scales, axis=0)
        return F.max(gate_scales, axis=0)

    def _local_compute(
        self,
        payload: EPDispatchPayload,
        global_scale: Tensor | None,
        estimated_total_m: Tensor,
    ) -> list[Tensor]:
        """Runs the per-device expert matmuls on dispatched tokens."""
        # The EP dispatch hands back one bundle per device, so each device's
        # expert matmuls run on its own entries.
        tokens = payload.per_device_tokens(
            self.quant_config, nvfp4_global_scale=global_scale
        )
        gate_up = self.gate_up_proj
        down = self.down_proj
        usage_stats = payload.usage_stats
        with ensure_context():
            return [
                _local_expert_matmul(
                    tokens[i],
                    gate_up[i],
                    down[i],
                    payload.expert_start[i],
                    payload.expert_ids[i],
                    usage_stats[i] if usage_stats is not None else None,
                    quant_config=self.quant_config,
                    estimated_total_m=estimated_total_m,
                )
                for i in range(len(tokens))
            ]

    def _compute_shared_experts(
        self, x: Tensor, devices: Sequence[int]
    ) -> list[Tensor]:
        """Runs the shared expert for ``devices``.

        Returns:
            One shared-expert output per entry of ``devices``, in order.
        """
        shared_experts = self.shared_experts
        assert shared_experts is not None
        devices = list(devices)

        def run_shared_experts(*inputs: Tensor) -> list[Tensor]:
            return [
                _on_device(shared_experts, device)(shard)
                for device, shard in zip(devices, inputs, strict=True)
            ]

        inputs = [x.local_shards[device] for device in devices]
        if os.environ.get("MODULAR_OVERLAP_SHARED_EXPERT", "1") == "0":
            outputs = run_shared_experts(*inputs)
        else:
            outputs = F.side_stream(
                inputs,
                run_shared_experts,
                result_types=[shard.type for shard in inputs],
                stream_id=_SHARED_EXPERT_STREAM_ID,
            )

        return outputs

    def forward(self, x: Tensor, comm: EPCommBuffers | None = None) -> Tensor:
        """Expert-parallel forward: gate -> dispatch -> local compute -> combine.

        ``comm`` is optional only to keep the override compatible with the base
        ``forward(self, x)``; the EP path always supplies it.
        """
        assert comm is not None, (
            "ExpertParallelMoE.forward requires comm buffers"
        )
        batch_mgr = self.ep_batch_manager
        batch_mgr.bind_comm_buffers(comm)
        config = batch_mgr.config

        # Per-device gate computation (replicated router scores).
        router_idx, router_weight = self.gate(x)
        router_idx = router_idx.cast(DType.int32)

        x_shards = list(x.local_shards)
        topk_id_shards = list(router_idx.local_shards)
        router_weight_shards = list(router_weight.local_shards)
        device_ids = [d.id for d in self.mesh.devices]

        if ep_requires_dispatch_scales(self.quant_config):
            global_scale = self._nvfp4_global_input_scale()
            input_scales = [
                F.broadcast_to(global_scale, [self.num_experts])
                for _ in self.mesh.devices
            ]
        else:
            global_scale = None
            input_scales = None

        # Dispatch tokens to the device owning each routed expert.
        if config.use_allreduce:
            dispatch_results = [
                batch_mgr.ep_dispatch(
                    x_shards[i],
                    topk_id_shards[i],
                    device_ids[i],
                    input_scales=input_scales[i] if input_scales else None,
                )
                for i in range(self.mesh.num_devices)
            ]
        else:
            dispatch_results = batch_mgr.ep_dispatch_all(
                x_shards, topk_id_shards, device_ids, input_scales=input_scales
            )

        # Under allreduce, combine outputs are per-device partial sums that get
        # summed later, so add the replicated shared expert on one device only.
        shared_by_device: dict[int, Tensor] = {}
        if self.shared_experts is not None and not config.fused_shared_expert:
            devices = (
                [0] if config.use_allreduce else range(self.mesh.num_devices)
            )
            shared_by_device = dict(
                zip(
                    devices,
                    self._compute_shared_experts(x, devices),
                    strict=True,
                )
            )

        # Estimated total token-expert pairs across all devices.
        total_tokens = F.shape_to_tensor(x_shards[0].shape)[0]
        for shard in x_shards[1:]:
            total_tokens = total_tokens + F.shape_to_tensor(shard.shape)[0]
        estimated_total_m = (
            total_tokens * self.num_experts_per_token // config.n_gpus_per_node
        ).cast(DType.uint32)

        # Now each device runs its own experts on the tokens it was sent.
        payload = EPDispatchPayload.from_dispatch(
            dispatch_results, self.quant_config, config
        )
        down_bundle = self._local_compute(
            payload, global_scale, estimated_total_m
        )

        # Combine expert outputs back to their source devices.
        if config.use_allreduce:
            combine_results = [
                batch_mgr.ep_combine(
                    down_bundle[i],
                    router_weight_shards[i],
                    device_ids[i],
                    topk_id_shards[i],
                )
                for i in range(self.mesh.num_devices)
            ]
        else:
            combine_results = batch_mgr.ep_combine_all(
                down_bundle, router_weight_shards, device_ids
            )

        # ``ep_combine`` returns each device exactly the tokens it dispatched,
        # so the output placement matches the input's.
        outputs: list[Tensor] = []
        for i in range(self.mesh.num_devices):
            out = combine_results[i]
            if i in shared_by_device:
                out = out + shared_by_device[i]
            outputs.append(out.cast(x_shards[i].dtype))
        return Tensor.from_shard_values(
            [TensorValue(shard) for shard in outputs],
            mapping=x.mapping,
        )
