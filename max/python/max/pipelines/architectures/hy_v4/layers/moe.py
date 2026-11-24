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

It pairs a noaux_tc router with clamped MXFP8 or block-FP8 routed experts and
an unclamped shared expert.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

from max import functional as F
from max import tree
from max.driver import Device
from max.dtype import DType
from max.nn import Module, ModuleList
from max.nn.activation import GatedActivation
from max.nn.common_layers import quant_ops
from max.nn.common_layers.mesh_axis import TP
from max.nn.common_layers.moe import MoEGate
from max.nn.common_layers.quant_linear import QuantizedMLP
from max.nn.common_layers.quant_tensor import MXFP8Activation, MXFP8Tensor
from max.nn.kernels import (
    grouped_quantize_dynamic_block_scaled,
    moe_create_indices,
    moe_finalize,
    moe_fp8_host_usage_stats,
    moe_router_group_limited,
)
from max.nn.quant_config import QuantConfig
from max.tensor import Tensor, default_device, defaults
from max.tensor.sharding import (
    DeviceMapping,
    DeviceMesh,
    NamedMapping,
    Partial,
)
from max.tensor.sharding.action import PerShard
from typing_extensions import Self

from ..model_config import HYV4Config
from .mlp import HYV4MLP, is_partial

_M = TypeVar("_M", bound=Module[..., Tensor])

# What ``moe_create_indices(..., needs_scales_offset=True)`` returns: token
# order, expert start rows, restore order, expert ids, usage stats and scale
# offsets.
_Indices = tuple[Tensor, ...]


def _on_device(module: _M, rank: int) -> _M:
    """Returns a view of ``module`` holding only device ``rank``'s shards."""
    return tree.map(
        lambda t: t.local_shards[rank], module, leaf=Tensor, shared=True
    )


class HYV4TopKRouter(MoEGate):
    """noaux_tc router whose logits are computed in float32.

    The reference casts both the hidden state and the BF16 gate weight to
    float32 before the matmul; a BF16 matmul output would perturb the
    near-tied scores that decide the top-k. Scores are sigmoids, picked by a
    group-limited top-k with a learned float32 expert correction bias.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        routed_scaling_factor: float,
        n_group: int,
        topk_group: int,
        norm_topk_prob: bool,
    ) -> None:
        """Stores the router dimensions and routing parameters.

        Args:
            hidden_dim: Dimension of the hidden state.
            num_experts: Number of routed experts.
            num_experts_per_token: Top-k experts to select per token.
            routed_scaling_factor: Scaling factor applied to expert weights.
            n_group: Number of expert groups.
            topk_group: Number of expert groups selected per token.
            norm_topk_prob: Whether to normalize the top-k probabilities.
        """
        super().__init__(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
        )
        if num_experts % n_group != 0:
            raise ValueError(
                f"num_experts must be divisible by n_group: {num_experts} % "
                f"{n_group} != 0"
            )
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.e_score_correction_bias = Tensor.zeros(
            [num_experts], dtype=DType.float32
        )

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        """Scores each token's experts and returns the routing weights."""
        weight = self.gate_score.weight.cast(DType.float32)
        logits = F.matmul(
            hidden_states.cast(DType.float32), F.transpose(weight, 0, 1)
        )
        scores = F.sigmoid(logits)
        topk_idx, topk_weight = moe_router_group_limited(
            scores,
            self.e_score_correction_bias.cast(DType.float32),
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

    @property
    def num_experts(self) -> int:
        """The number of experts this module holds."""
        return int(self.gate_up_proj.weight.shape[0])

    def local(self, rank: int) -> Self:
        """A view holding only device ``rank``'s experts."""
        return _on_device(self, rank)

    def grouped(self, x: Tensor, top_k: int, indices: _Indices) -> Tensor:
        """Runs the routed experts over the rows the router sent each token to.

        ``indices`` is what ``moe_create_indices`` returns. Slot ``i`` of its
        order reads token row ``order[i] // top_k`` of ``x``; the quantize
        does that gather in its loads.
        """
        token_order, expert_start_indices, _, expert_ids, _, scales_offset = (
            indices
        )
        estimated_total_m = F.shape_to_tensor(token_order.shape)[0].cast(
            DType.uint32
        )
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


class HYV4BlockFP8Experts(ModuleList[QuantizedMLP]):
    """Block-FP8 routed experts with the clamped SwiGLU, one module each.

    The checkpoint stores each expert's projections apart; the graph stacks
    gate rows before up rows. On several devices each expert is built on the
    device that owns it, the first ``num_experts / devices`` on device 0.
    """

    def __init__(
        self,
        experts: Iterable[QuantizedMLP],
        swiglu_limit: float,
        per_device: int,
    ) -> None:
        super().__init__(experts)
        self.swiglu_limit = swiglu_limit
        self.per_device = per_device

    @classmethod
    def zeros(
        cls,
        hidden_dim: int,
        moe_dim: int,
        num_experts: int,
        swiglu_limit: float,
        quant_config: QuantConfig,
    ) -> HYV4BlockFP8Experts:
        """Builds the experts on the default mesh's devices, all zeros."""
        _, mesh = defaults()
        per_device = num_experts // mesh.num_devices
        experts: list[QuantizedMLP] = []
        for device in mesh.devices:
            with default_device(device):
                experts.extend(
                    QuantizedMLP(hidden_dim, moe_dim, quant_config=quant_config)
                    for _ in range(per_device)
                )
        return cls(experts, swiglu_limit, per_device)

    @property
    def num_experts(self) -> int:
        """The number of experts this module holds."""
        return len(self)

    def local(self, rank: int) -> HYV4BlockFP8Experts:
        """A view holding only device ``rank``'s experts."""
        first = rank * self.per_device
        return HYV4BlockFP8Experts(
            self[first : first + self.per_device],
            self.swiglu_limit,
            self.per_device,
        )

    def grouped(self, x: Tensor, top_k: int, indices: _Indices) -> Tensor:
        """Runs the routed experts over the rows the router sent each token to.

        ``indices`` is what ``moe_create_indices`` returns. Slot ``i`` of its
        order reads token row ``order[i] // top_k`` of ``x``.
        """
        token_order, expert_start_indices, _, expert_ids, usage_stats, _ = (
            indices
        )
        # Every group counts, empty ones too, so shapes give the kernel's
        # host bounds and no per-layer device-to-host copy runs.
        usage_stats = moe_fp8_host_usage_stats(
            usage_stats, token_order.shape[0], expert_ids.shape[0]
        )
        estimated_total_m = F.shape_to_tensor(token_order.shape)[0].cast(
            DType.uint32
        )
        tokens = F.gather(x, (token_order // top_k).cast(DType.int32), axis=0)
        gate_up = quant_ops.stack(
            [
                quant_ops.concat_weights(e.gate_proj.weight, e.up_proj.weight)
                for e in self
            ]
        )
        down = quant_ops.stack([e.down_proj.weight for e in self])
        hidden = quant_ops.grouped_matmul_silu(
            tokens,
            gate_up,
            down,
            expert_start_indices,
            expert_ids,
            usage_stats,
            None,
            estimated_total_m=estimated_total_m,
            swiglu_limit=self.swiglu_limit,
            # As the TRTLLM experts vLLM runs at TP2 on B200: only the
            # input's scales round; the weights keep fp32 scales and the
            # kernel scales the SwiGLU by amax / 448.
            pow2_scales=True,
        )
        return quant_ops.grouped_matmul(
            hidden,
            down,
            expert_start_indices,
            expert_ids,
            usage_stats,
            estimated_total_m=estimated_total_m,
        )


def expert_sum(
    experts: HYV4Experts | HYV4BlockFP8Experts,
    x: Tensor,
    topk_idx: Tensor,
    topk_weight: Tensor,
    top_k: int,
    first_expert: int | None = None,
) -> Tensor:
    """Sums ``experts`` over the routed rows they own.

    ``first_expert`` is the global id of ``experts``' first expert under
    expert parallelism. ``moe_create_indices`` leaves an id outside the
    local range out of every group, and leaves the order slots and restore
    entries it would have used unwritten, so those are masked here.
    """
    ids = F.reshape(topk_idx, [-1])
    if first_expert is not None:
        ids = ids - first_expert
    num_local = experts.num_experts
    indices = moe_create_indices(ids, num_local, needs_scales_offset=True)
    order, start, restore = indices[0], indices[1], indices[2]
    if first_expert is None:
        return moe_finalize(
            experts.grouped(x, top_k, indices), restore, topk_weight, x.dtype
        )
    # Order slots past the local rows are never written and may hold any
    # index; point them at row 0 so a full gather stays in bounds.
    local_rows = (
        F.arange(
            0,
            F.shape_to_tensor(ids.shape)[0],
            out_dim=ids.shape[0],
            dtype=DType.uint32,
            device=ids.device,
        )
        < start[num_local]
    )
    order = F.where(local_rows, order, 0)
    down = experts.grouped(x, top_k, (order, *indices[1:]))
    foreign = (ids < 0) | (ids >= num_local)
    restore = F.where(foreign, 0, restore)
    topk_weight = F.where(foreign.reshape(topk_weight.shape), 0.0, topk_weight)
    # A where, not a zero weight: rows past the local ones are never
    # written and may hold NaN.
    down = F.where(local_rows.unsqueeze(1), down, 0.0)
    return moe_finalize(down, restore, topk_weight, x.dtype)


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
            n_group=config.n_group,
            topk_group=config.topk_group,
            norm_topk_prob=config.norm_topk_prob,
        )
        self.experts: HYV4Experts | HYV4BlockFP8Experts
        if quant_ops.is_fp8_block_quantized(config.quant_config):
            assert config.quant_config is not None
            self.experts = HYV4BlockFP8Experts.zeros(
                config.hidden_size,
                config.moe_intermediate_size,
                config.n_routed_experts,
                config.swiglu_limit,
                config.quant_config,
            )
        else:
            self.experts = HYV4Experts(
                config.hidden_size,
                config.moe_intermediate_size,
                config.n_routed_experts,
                config.swiglu_limit,
            )
        self.shared_experts = HYV4MLP(
            config.hidden_size,
            config.moe_intermediate_size * config.n_shared_experts,
            config.projection_quant_config,
        )

    def shard(self) -> None:
        """Splits the routed experts by expert and the shared expert by width.

        The router stays replicated.
        """
        self.shard_experts()
        self.shared_experts.shard()

    def shard_experts(self) -> None:
        """Splits the routed experts evenly over the mesh (expert parallel).

        Block-FP8 experts are built on their devices already.
        """
        experts = self.experts
        if not isinstance(experts, HYV4Experts):
            return
        spec = NamedMapping(experts.gate_up_proj.mesh, (TP, None, None))
        experts.gate_up_proj = experts.gate_up_proj.to(spec)
        experts.down_proj = experts.down_proj.to(spec)

    def routed(self, x: Tensor) -> Tensor:
        """The weighted sum of the routed experts, without the shared one.

        On several devices it is each device's unreduced sum (``Partial``).
        """
        topk_idx, topk_weight = self.gate(x)
        mesh = x.mesh
        if mesh.num_devices == 1:
            return expert_sum(
                self.experts, x, topk_idx, topk_weight, self.top_k
            )

        # Every device holds all tokens; each runs only its own experts, so
        # the per-device sums add up to the full one.
        # TODO: an all-to-all dispatch would spare each device the router
        # and the all-reduce over every token; it pays off once batches are
        # large enough for that traffic to show.
        def local(
            x: Tensor, topk_idx: Tensor, topk_weight: Tensor, rank: int
        ) -> Tensor:
            return expert_sum(
                self.experts.local(rank),
                x,
                topk_idx,
                topk_weight,
                self.top_k,
                first_expert=rank * self.num_experts // mesh.num_devices,
            )

        return F.call_on_mesh(
            local, mesh, out_specs=DeviceMapping(mesh, (Partial(),))
        )(x, topk_idx, topk_weight, PerShard(range(mesh.num_devices)))

    def forward(self, x: Tensor) -> Tensor:
        """Runs the routed and shared experts over ``x``."""
        routed = self.routed(x)
        shared = self.shared_experts.partial(x)
        if not is_partial(routed):
            return routed + shared
        if not is_partial(shared):
            return F.allreduce_sum(routed) + shared
        # Adding per device keeps the generic add from reducing each term:
        # one all-reduce carries both.
        total = F.call_on_mesh(F.add, x.mesh, out_specs=routed.mapping)(
            routed, shared
        )
        return F.allreduce_sum(total)
