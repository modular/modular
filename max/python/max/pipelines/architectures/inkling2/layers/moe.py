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
"""Inkling 2's sigmoid-routed mixture of experts."""

from __future__ import annotations

import numpy as np
from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef
from max.nn import Module
from max.nn._graph_ops import grouped_quantize_nvfp4_row_scaled
from max.nn.activation import GatedActivation
from max.nn.kernels import (
    block_scales_interleave,
    fused_silu,
    grouped_matmul_block_scaled,
    grouped_matmul_ragged,
    grouped_quantize_dynamic_block_scaled,
    moe_create_indices,
    moe_finalize,
    moe_sigmoid_gemv_router,
)
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping, Partial
from max.tensor.sharding.action import PerShard

from ..model_config import ExpertFormat, Inkling2TextConfig
from .placement import shard_axis

# The fused router runs one thread per expert in a single block.
_ROUTER_WARP = 32
_ROUTER_MAX_EXPERTS = 1024
_ROUTER_EPS = 1e-20

NVFP4_BLOCK = 16
MXFP8_BLOCK = 32
# One 128-row scale tile covers 32 rows x 4 row groups of the SM100 layout.
_SF_TILE_ROWS = 128

_quantize_nvfp4_rows = F.functional(grouped_quantize_nvfp4_row_scaled)


class Inkling2Gate(Module[[Tensor], tuple[Tensor, Tensor]]):
    """Sigmoid router with a selection-only bias and renormalized weights.

    The experts are chosen by ``sigmoid(logits) + bias``; the weights come
    from the unbiased scores, normalized over the chosen experts and scaled by
    ``route_scale``. Logits are float32: a bfloat16 router rounds distinct
    scores into ties before the top-k.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        route_scale: float,
        device: DeviceRef,
    ) -> None:
        self.top_k = top_k
        self.route_scale = route_scale
        self.weight = Tensor.zeros([num_experts, hidden_size])
        self.bias = Tensor.zeros([num_experts], dtype=DType.float32)
        # The fused kernel accumulates the gate GEMV in float32, as the
        # reference does. A matmul would run float32 as TF32 on GPU.
        self.fused = (
            device.is_gpu()
            and num_experts % _ROUTER_WARP == 0
            and num_experts <= _ROUTER_MAX_EXPERTS
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Returns ``[tokens, top_k]`` float32 weights and int32 expert ids."""
        x32 = x.cast(DType.float32)
        weight = self.weight.cast(DType.float32)
        if self.fused:
            indices, weights = moe_sigmoid_gemv_router(
                x32,
                weight,
                self.bias,
                n_experts_per_tok=self.top_k,
                norm_weights=True,
                routed_scaling_factor=self.route_scale,
            )
            return weights, indices.cast(DType.int32)
        # Serves expert counts that are not a multiple of 32, as in the
        # minimized fixture, with an exact float32 product: a float32 matmul
        # would round its operands to TF32 on GPU.
        logits = (x32.unsqueeze(1) * weight).sum(axis=-1).squeeze(-1)
        scores = F.sigmoid(logits)
        _, indices = F.top_k(scores + self.bias, self.top_k, axis=-1)
        indices = indices.cast(DType.int32)
        weights = F.gather_nd(scores, indices.unsqueeze(-1), batch_dims=1)
        weights = weights / (weights.sum(axis=-1) + _ROUTER_EPS)
        return weights * self.route_scale, indices


# TODO(mrajagopalan): swizzle at load time. The weight rules only rename,
# retype and drop tensors, so the graph swizzles the stored scales; the compiler
# hoists it to init, at the cost of a second copy of every scale tensor.
def _interleave_scales(scales: Tensor, block: int) -> Tensor:
    """Lays ``[E, N, K / block]`` block scales out in the SM100 scale tiles."""
    experts, rows, groups = (int(d) for d in scales.shape)
    if rows % _SF_TILE_ROWS:
        raise ValueError(
            f"{rows} expert rows are not whole 128-row scale tiles"
        )
    tiled = block_scales_interleave(
        scales.reshape([experts * rows, groups]), block
    )
    return tiled.reshape(
        [experts, rows // _SF_TILE_ROWS, *(int(d) for d in tiled.shape[1:])]
    )


def _usage_stats(rows: Tensor, num_experts: int) -> Tensor:
    """The grouped kernel's host ``[max rows per expert, experts]`` hint."""
    return F.concat(
        [
            F.shape_to_tensor([rows.shape[0]]).cast(DType.uint32),
            F.constant(
                np.array([num_experts], np.uint32), DType.uint32, device=CPU()
            ),
        ],
        axis=0,
    )


def _expert_stack(shape: list[int], dtype: DType | None = None) -> Tensor:
    """A stack of per-expert tensors, a run of whole experts on each device."""
    return shard_axis(Tensor.zeros(shape, dtype=dtype), 0)


class RoutedExperts(Module[[Tensor, Tensor, Tensor, int], Tensor]):
    """The routed experts as the checkpoint stores them, stacked by expert.

    ``w13_weight`` is ``[experts, 2 * intermediate, hidden]`` with the gate
    and up rows interleaved, and ``w2_weight`` is
    ``[experts, hidden, intermediate]``. A quantized layer stores them as:

    - NVFP4: packed e2m1 bytes with ``float8_e4m3fn`` scales per 16 inputs,
      and a float32 ``*_amax`` per expert and ``*_outer_scale_denominator``
      per stack. An expert's weights are the elements times the block scale
      times ``amax / denominator``.
    - MXFP8: ``float8_e4m3fn`` elements with ue8m0 scales per 32 inputs.

    Activations quantize on the fly into the weights' format.

    Under tensor parallelism each device holds a run of whole experts, since
    an expert's activation scale covers its full input width. Every device
    runs its own experts over all tokens, and the partial sums are reduced
    afterwards.
    """

    w13_weight: Tensor
    w2_weight: Tensor

    def __init__(
        self, text_config: Inkling2TextConfig, expert_format: ExpertFormat
    ) -> None:
        self.format = expert_format
        self.num_experts = text_config.n_routed_experts
        self.top_k = text_config.num_experts_per_tok
        self.activation = GatedActivation(
            "situ",
            alpha=text_config.mlp_gate_softcap,
            limit=text_config.mlp_up_softcap,
        )
        hidden = text_config.hidden_size
        inner = text_config.intermediate_size
        experts = self.num_experts
        stacks = {"w13": (2 * inner, hidden), "w2": (hidden, inner)}
        for name, (rows, inputs) in stacks.items():
            if expert_format == ExpertFormat.BF16:
                weight = _expert_stack([experts, rows, inputs])
            elif expert_format == ExpertFormat.NVFP4:
                weight = _expert_stack(
                    [experts, rows, inputs // 2], DType.uint8
                )
                setattr(
                    self,
                    f"{name}_weight_scale",
                    _expert_stack(
                        [experts, rows, inputs // NVFP4_BLOCK],
                        DType.float8_e4m3fn,
                    ),
                )
                setattr(
                    self,
                    f"{name}_weight_amax",
                    _expert_stack([experts], DType.float32),
                )
                setattr(
                    self,
                    f"{name}_weight_outer_scale_denominator",
                    Tensor.zeros([1], dtype=DType.float32),
                )
            else:
                weight = _expert_stack(
                    [experts, rows, inputs], DType.float8_e4m3fn
                )
                setattr(
                    self,
                    f"{name}_weight_scale",
                    _expert_stack(
                        [experts, rows, inputs // MXFP8_BLOCK],
                        DType.float8_e8m0fnu,
                    ),
                )
            setattr(self, f"{name}_weight", weight)

    def _expert_scales(self, name: str) -> Tensor:
        """The per-expert factor the matmul epilogue multiplies in."""
        if self.format == ExpertFormat.NVFP4:
            amax = getattr(self, f"{name}_weight_amax")
            denominator = getattr(
                self, f"{name}_weight_outer_scale_denominator"
            )
            return amax / denominator
        return F.constant(
            np.ones(int(self.w13_weight.shape[0]), np.float32),
            DType.float32,
            device=self.w13_weight.mapping,
        )

    def _grouped_matmul(
        self,
        name: str,
        rows: Tensor,
        starts: Tensor,
        expert_ids: Tensor,
        scale_offsets: Tensor,
        quantized: tuple[Tensor, Tensor, Tensor | None],
    ) -> Tensor:
        """Multiplies quantized, expert-sorted rows by one weight stack."""
        data, scales, row_scales = quantized
        block = (
            NVFP4_BLOCK if self.format == ExpertFormat.NVFP4 else MXFP8_BLOCK
        )
        return grouped_matmul_block_scaled(
            data,
            getattr(self, f"{name}_weight"),
            scales,
            _interleave_scales(getattr(self, f"{name}_weight_scale"), block),
            starts,
            scale_offsets,
            expert_ids,
            self._expert_scales(name),
            _usage_stats(rows, int(self.w13_weight.shape[0])),
            a_row_scales=row_scales,
        )

    def _quantize(
        self,
        x: Tensor,
        starts: Tensor,
        scale_offsets: Tensor,
        expert_ids: Tensor,
        order: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Quantizes expert-sorted rows, gathering them from ``x`` by ``order``.

        Returns the data, the block scales and, for NVFP4, each row's own
        scale: the checkpoint's format scales every row by its maximum.
        """
        if self.format == ExpertFormat.NVFP4:
            return _quantize_nvfp4_rows(
                x,
                starts,
                scale_offsets,
                indices=order,
                indices_divisor=self.top_k,
            )
        data, scales = grouped_quantize_dynamic_block_scaled(
            x,
            starts,
            scale_offsets,
            expert_ids,
            F.constant(
                np.ones(int(self.w13_weight.shape[0]), np.float32),
                DType.float32,
                device=x.mapping,
            ),
            sf_vector_size=MXFP8_BLOCK,
            scales_type=DType.float8_e8m0fnu,
            out_type=DType.float8_e4m3fn,
            indices=order,
            indices_divisor=self.top_k,
        )
        return data, scales, None

    def forward(
        self, x: Tensor, weights: Tensor, indices: Tensor, first: int
    ) -> Tensor:
        """Returns each token's weighted sum over its experts held here.

        Runs on one device's experts, ``first`` through ``first + n_local``
        by global id. A slot routed to another device's expert adds nothing,
        so the devices' results sum to the whole.
        """
        quantized = self.format != ExpertFormat.BF16
        n_local = int(self.w13_weight.shape[0])
        flat = indices.reshape([-1])
        masked = n_local < self.num_experts
        if masked:
            # Slots of other devices' experts go to a trailing group, past
            # every group a kernel below reads.
            is_local = F.logical_and(
                F.greater_equal(flat, first), F.greater(first + n_local, flat)
            )
            flat = F.where(is_local, flat - first, n_local)
        sorted_slots = moe_create_indices(
            flat,
            n_local + 1 if masked else n_local,
            needs_scales_offset=quantized,
        )
        order, starts, restore, expert_ids, usage = sorted_slots[:5]
        valid_rows = None
        if masked:
            # The kernels below see only the local groups. The trailing
            # group's rows sit past `valid_rows`, which the combine skips.
            valid_rows = starts[n_local : n_local + 1]
            starts = starts[: n_local + 1]
            expert_ids = expert_ids[:n_local]
        if not quantized:
            if masked:
                usage = F.concat(
                    [
                        usage[:1],
                        F.constant(
                            np.array([n_local], np.uint32),
                            DType.uint32,
                            device=usage.mapping,
                        ),
                    ],
                    axis=0,
                )
            gate_up = grouped_matmul_ragged(
                F.gather(x, (order // self.top_k).cast(DType.int32), axis=0),
                self.w13_weight,
                starts,
                expert_ids,
                usage,
            )
            hidden = fused_silu(
                gate_up, starts, activation=self.activation, interleaved=True
            )
            down = grouped_matmul_ragged(
                hidden, self.w2_weight, starts, expert_ids, usage
            )
            return moe_finalize(down, restore, weights, x.dtype, valid_rows)

        scale_offsets = sorted_slots[5]
        if masked:
            scale_offsets = scale_offsets[:n_local]
        slots = order
        gate_up = self._grouped_matmul(
            "w13",
            slots,
            starts,
            expert_ids,
            scale_offsets,
            self._quantize(x, starts, scale_offsets, expert_ids, order),
        )
        hidden = fused_silu(
            gate_up, starts, activation=self.activation, interleaved=True
        )
        down = self._grouped_matmul(
            "w2",
            slots,
            starts,
            expert_ids,
            scale_offsets,
            self._quantize(hidden, starts, scale_offsets, expert_ids),
        )
        return moe_finalize(down, restore, weights, x.dtype, valid_rows)


def _on_each_device(moe: Inkling2MoE, x: Tensor, first: int) -> Tensor:
    """Routes ``x`` and runs this device's experts on every token."""
    weights, indices = moe.gate(x)
    return moe.experts(x, weights, indices, first)


class Inkling2MoE(Module[[Tensor], Tensor]):
    """The gate and routed experts of one layer. There is no shared expert.

    Every device routes the replicated tokens the same way and runs its own
    run of whole experts. The result is each device's partial sum.
    """

    def __init__(
        self,
        *,
        text_config: Inkling2TextConfig,
        expert_format: ExpertFormat,
        device: DeviceRef,
        num_devices: int,
    ) -> None:
        if text_config.n_routed_experts % num_devices:
            raise ValueError(
                f"{text_config.n_routed_experts} experts do not split over "
                f"{num_devices} devices"
            )
        self.gate = Inkling2Gate(
            hidden_size=text_config.hidden_size,
            num_experts=text_config.n_routed_experts,
            top_k=text_config.num_experts_per_tok,
            route_scale=text_config.route_scale,
            device=device,
        )
        self.experts = RoutedExperts(text_config, expert_format)

    def forward(self, x: Tensor) -> Tensor:
        """Returns the ``[tokens, hidden]`` MoE output, a partial sum."""
        mesh = x.mesh
        local = self.experts.num_experts // mesh.num_devices
        return F.call_on_mesh(
            _on_each_device,
            mesh,
            out_specs=DeviceMapping(mesh, (Partial(),)),
        )(self, x, PerShard(range(0, self.experts.num_experts, local)))
