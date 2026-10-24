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
"""Hy4 mixture of experts: noaux_tc router, clamped MXFP8 routed experts and
an unclamped shared expert."""

from __future__ import annotations

from typing import TypeVar

from max import tree
from max.driver import Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers import quant_ops
from max.experimental.nn.common_layers.functional_kernels import (
    moe_create_indices,
    moe_finalize,
    moe_router_group_limited,
)
from max.experimental.nn.common_layers.mesh_axis import TP
from max.experimental.nn.common_layers.quant_tensor import MXFP8Tensor
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    NamedMapping,
    Partial,
)
from max.experimental.sharding.action import PerShard
from max.experimental.tensor import Tensor
from max.pipelines.architectures.deepseekV3_modulev3.layers.moe_gate import (
    DeepseekV3TopKRouter,
)
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
        # Gate rows [0, moe_dim) then up rows, as the checkpoint stores them.
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
        expert_start_indices: Tensor,
        expert_ids: Tensor,
        scales_offset: Tensor,
    ) -> Tensor:
        # The block-scaled kernels pick their tile regime from the row count.
        rows = F.shape_to_tensor(x.shape)[0].cast(DType.uint32)

        def matmul(tokens: Tensor, weight: MXFP8Tensor) -> Tensor:
            return quant_ops.grouped_matmul(
                tokens,
                weight,
                expert_start_indices,
                expert_ids,
                None,
                scales_offset=scales_offset,
                estimated_total_m=rows,
            )

        hidden = quant_ops.grouped_silu(
            matmul(x, self.gate_up_proj),
            expert_start_indices,
            swiglu_limit=self.swiglu_limit,
        )
        assert isinstance(hidden, Tensor)
        return matmul(hidden, self.down_proj)


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
        # TODO: an all-to-all dispatch would spare each device the
        # gather and quantize of the rows it skips; it pays off once batches
        # are large enough for that traffic to show.
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
        local range out of every group, and the order and restore entries it
        would have used unwritten, so those are masked here.
        """
        ids = F.reshape(topk_idx, [-1])
        if first_expert is not None:
            ids = ids - first_expert
        num_local = int(experts.gate_up_proj.weight.shape[0])
        order, start, restore, expert_ids, _, scales_offset = (
            moe_create_indices(ids, num_local, needs_scales_offset=True)
        )
        valid_rows: Tensor | None = None
        if first_expert is not None:
            valid_rows = (
                F.arange(
                    0,
                    F.shape_to_tensor(ids.shape)[0],
                    out_dim=ids.shape[0],
                    dtype=DType.uint32,
                    device=ids.device,
                )
                < start[num_local]
            )
            order = F.where(valid_rows, order, 0)
            foreign = (ids < 0) | (ids >= num_local)
            restore = F.where(foreign, 0, restore)
            topk_weight = F.where(
                foreign.reshape(topk_weight.shape), 0.0, topk_weight
            )
        permuted = F.gather(x, F.cast(order // self.top_k, DType.int32), axis=0)
        down = experts(permuted, start, expert_ids, scales_offset)
        if valid_rows is not None:
            # Rows past the local ones are never written by the matmuls.
            down = F.where(valid_rows.unsqueeze(1), down, 0.0)
        return moe_finalize(down, restore, topk_weight, x.dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.routed(x) + self.shared_experts(x)
