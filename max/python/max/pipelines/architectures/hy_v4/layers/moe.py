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
"""Hy4 mixture of experts.

It pairs a noaux_tc router with clamped MXFP8 routed experts and an unclamped
shared expert.
"""

from __future__ import annotations

from typing import TypeVar

from max import functional as F
from max import tree
from max.driver import Device
from max.dtype import DType
from max.nn import Module
from max.nn.activation import GatedActivation
from max.nn.common_layers import quant_ops
from max.nn.common_layers.mesh_axis import TP
from max.nn.common_layers.quant_tensor import MXFP8Activation, MXFP8Tensor
from max.nn.kernels import (
    grouped_quantize_dynamic_block_scaled,
    moe_create_indices,
    moe_finalize,
    moe_router_group_limited,
)
from max.pipelines.architectures.deepseekV3_modulev3.layers.moe_gate import (
    DeepseekV3TopKRouter,
)
from max.tensor import Tensor
from max.tensor.sharding import (
    DeviceMapping,
    DeviceMesh,
    NamedMapping,
    Partial,
)
from max.tensor.sharding.action import PerShard
from typing_extensions import Self

from ..model_config import HYV4Config
from .mlp import HYV4MLP

_M = TypeVar("_M", bound=Module[..., Tensor])


def _on_device(module: _M, rank: int) -> _M:
    """Returns a view of ``module`` holding only device ``rank``'s shards."""
    return tree.map(
        lambda t: t.local_shards[rank], module, leaf=Tensor, shared=True
    )


class HYV4TopKRouter(DeepseekV3TopKRouter):
    """noaux_tc router whose logits are computed in float32.

    The reference casts both the hidden state and the BF16 gate weight to
    float32 before the matmul; a BF16 matmul output would perturb the
    near-tied scores that decide the top-k.
    """

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        """Scores each token's experts and returns the routing weights."""
        weight = self.gate_score.weight.cast(DType.float32)
        logits = F.matmul(
            hidden_states.cast(DType.float32), F.transpose(weight, 0, 1)
        )
        scores = F.sigmoid(logits)
        topk_idx, topk_weight = moe_router_group_limited(
            scores,
            self.e_score_correction_bias,
            self.num_experts,
            self.num_experts_per_token,
            self.n_group,
            self.topk_group,
            self.norm_topk_prob,
            self.routed_scaling_factor,
        )
        return (
            topk_idx.rebind_mapping(scores.mapping),
            topk_weight.rebind_mapping(scores.mapping),
        )


class HYV4Experts(Module[..., Tensor]):
    """Stacked MXFP8 routed experts with the clamped SwiGLU."""

    def __init__(
        self,
        hidden_dim: int,
        moe_dim: int,
        num_experts: int,
        swiglu_limit: float,
    ) -> None:
        super().__init__()
        self.swiglu_limit = swiglu_limit
        # Gate rows then up rows, as the checkpoint stores them.
        self.gate_up_proj = MXFP8Tensor.zeros(
            (num_experts, 2 * moe_dim, hidden_dim)
        )
        self.down_proj = MXFP8Tensor.zeros((num_experts, hidden_dim, moe_dim))

    def to(self, target: Device | DeviceMesh | DeviceMapping) -> Self:
        """Moves the expert weights to ``target`` in place."""
        super().to(target)
        self.gate_up_proj = self.gate_up_proj.to(target)
        self.down_proj = self.down_proj.to(target)
        return self

    def forward(
        self,
        x: Tensor,
        token_order: Tensor,
        top_k: int,
        expert_start_indices: Tensor,
        expert_ids: Tensor,
        scales_offset: Tensor,
        estimated_total_m: Tensor,
    ) -> Tensor:
        """Runs the routed experts over the rows the router sent each token to.

        Slot ``i`` of ``token_order`` reads token row ``token_order[i] //
        top_k`` of ``x``; the quantize does that gather in its loads.
        """
        unit_scales = F.broadcast_to(
            F.constant(1.0, DType.float32, device=expert_ids.device),
            expert_ids.shape,
        )
        data, scales = grouped_quantize_dynamic_block_scaled(
            x,
            row_offsets=expert_start_indices,
            scales_offsets=scales_offset,
            expert_ids=expert_ids,
            sf_tensor=unit_scales,
            sf_vector_size=self.gate_up_proj.block_size[1],
            scales_type=DType.float8_e8m0fnu,
            out_type=DType.float8_e4m3fn,
            indices=token_order,
            indices_divisor=top_k,
        )
        tokens = MXFP8Activation(
            data=data, scales=scales, scales_offset=scales_offset
        )
        # One kernel runs the gate/up matmul, the clamped SwiGLU and the down
        # input's MXFP8 quantize. Rounding the SwiGLU to bfloat16 first
        # quantizes the values a separate SwiGLU kernel would write. The
        # kernel reads gate and up rows interleaved; init runs this
        # weight-only permute once, so loading makes no host copy. Freeing
        # the checkpoint-order copy fragments 256 MB memory-manager pages,
        # so launches set MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM_PAGE_MB=32
        # (see the recipe).
        # TODO: drop the env once the driver can permute in place at load or
        # copy host-to-device with strides; Mach launches do not set it yet.
        gate_up = quant_ops.sigma_permute_gate_up(self.gate_up_proj)
        assert isinstance(gate_up, MXFP8Tensor)
        hidden = quant_ops.grouped_matmul_swiglu_mx_ep(
            tokens,
            gate_up,
            expert_start_indices,
            expert_ids,
            estimated_total_m=estimated_total_m,
            activation=GatedActivation(
                "silu", limit=self.swiglu_limit, round_to_bf16=True
            ),
        )
        return quant_ops.grouped_matmul(
            hidden,
            self.down_proj,
            expert_start_indices,
            expert_ids,
            None,
            estimated_total_m=estimated_total_m,
        )


class HYV4MoE(Module[[Tensor], Tensor]):
    """Routed experts plus one always-on shared expert."""

    def __init__(self, config: HYV4Config) -> None:
        super().__init__()
        if not config.norm_topk_prob:
            # The single-group router kernel always normalizes.
            raise ValueError("Hy4 MoE requires norm_topk_prob")
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.gate = HYV4TopKRouter(
            hidden_dim=config.hidden_size,
            num_experts=config.n_routed_experts,
            num_experts_per_token=config.num_experts_per_tok,
            routed_scaling_factor=config.routed_scaling_factor,
            scoring_func="sigmoid",
            topk_method="noaux_tc",
            n_group=config.n_group,
            topk_group=config.topk_group,
            norm_topk_prob=config.norm_topk_prob,
            correction_bias_dtype=DType.float32,
        )
        self.experts = HYV4Experts(
            config.hidden_size,
            config.moe_intermediate_size,
            config.n_routed_experts,
            config.swiglu_limit,
        )
        self.shared_experts = HYV4MLP(
            config.hidden_size,
            config.moe_intermediate_size * config.n_shared_experts,
            config.quant_config,
        )

    def shard_experts(self) -> None:
        """Splits the routed experts evenly over the mesh (expert parallel).

        The router and the shared expert stay replicated.
        """
        experts = self.experts
        spec = NamedMapping(experts.gate_up_proj.mesh, (TP, None, None))
        experts.gate_up_proj = experts.gate_up_proj.to(spec)
        experts.down_proj = experts.down_proj.to(spec)

    def routed(self, x: Tensor) -> Tensor:
        """The weighted sum of the routed experts, without the shared one."""
        topk_idx, topk_weight = self.gate(x)
        mesh = x.mesh
        if mesh.num_devices == 1:
            return self._expert_sum(x, topk_idx, topk_weight, self.experts)

        # Every device holds all tokens; each runs only its own experts, so
        # the per-device sums add up to the full one.
        # TODO: an all-to-all dispatch would spare each device the router
        # and the all-reduce over every token; it pays off once batches are
        # large enough for that traffic to show.
        def local(
            x: Tensor, topk_idx: Tensor, topk_weight: Tensor, rank: int
        ) -> Tensor:
            return self._expert_sum(
                x,
                topk_idx,
                topk_weight,
                _on_device(self.experts, rank),
                first_expert=rank * self.num_experts // mesh.num_devices,
            )

        partial = F.call_on_mesh(
            local, mesh, out_specs=DeviceMapping(mesh, (Partial(),))
        )(x, topk_idx, topk_weight, PerShard(range(mesh.num_devices)))
        return F.allreduce_sum(partial)

    def _expert_sum(
        self,
        x: Tensor,
        topk_idx: Tensor,
        topk_weight: Tensor,
        experts: HYV4Experts,
        first_expert: int | None = None,
    ) -> Tensor:
        """Sums ``experts`` over the routed rows they own.

        ``first_expert`` is the global id of ``experts``' first expert under
        expert parallelism. ``moe_create_indices`` leaves an id outside the
        local range out of every group, and the restore entries it would
        have used unwritten, so those are masked here. The grouped kernels
        read only the grouped rows, so the order needs no mask.
        """
        ids = F.reshape(topk_idx, [-1])
        if first_expert is not None:
            ids = ids - first_expert
        num_local = int(experts.gate_up_proj.weight.shape[0])
        order, start, restore, expert_ids, _, scales_offset = (
            moe_create_indices(ids, num_local, needs_scales_offset=True)
        )
        slots = F.shape_to_tensor(ids.shape)[0]
        down = experts(
            x,
            order,
            self.top_k,
            start,
            expert_ids,
            scales_offset,
            estimated_total_m=slots.cast(DType.uint32),
        )
        if first_expert is not None:
            foreign = (ids < 0) | (ids >= num_local)
            restore = F.where(foreign, 0, restore)
            topk_weight = F.where(
                foreign.reshape(topk_weight.shape), 0.0, topk_weight
            )
            # A where, not a zero weight: rows past the local ones are never
            # written and may hold NaN.
            local_rows = (
                F.arange(
                    0,
                    slots,
                    out_dim=ids.shape[0],
                    dtype=DType.uint32,
                    device=ids.device,
                )
                < start[num_local]
            )
            down = F.where(local_rows.unsqueeze(1), down, 0.0)
        return moe_finalize(down, restore, topk_weight, x.dtype)

    def forward(self, x: Tensor) -> Tensor:
        """Runs the routed and shared experts over ``x``."""
        return self.routed(x) + self.shared_experts(x)
