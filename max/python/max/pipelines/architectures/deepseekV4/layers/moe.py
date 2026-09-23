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

"""DeepSeek-V4 MoE, with hash routing on the first ``num_hash_layers`` layers.

Reference: ``inference/model.py`` classes ``Gate``, ``Expert`` and ``MoE``.
Every V4 layer is MoE -- there is no dense prefix -- and every layer has one
shared expert on top of the ``num_experts_per_tok`` routed ones.

Routing has two shapes, and which one a layer uses is not a property of the
gate weight but of the layer index:

* Layers below ``num_hash_layers`` (0, 1 and 2) read their expert indices
  straight out of ``gate.tid2eid[token_id]``. The table is different on each of
  the three layers. They still compute gate scores and still have a
  ``gate.weight``, because the scores supply the routing *weights* -- only the
  choice of expert comes from the table.
* The rest score normally and take the top-k of the bias-shifted scores.

The two are mutually exclusive in the checkpoint: hash layers have no
``gate.bias``, scored layers have no ``gate.tid2eid``.

Three details from ``Gate.forward`` that are each a plausible place to be
wrong:

* The bias shifts the scores *for selection only*. The weights are gathered
  from the unshifted scores.
* The normalization divides by the sum of the gathered weights, which happens
  after the gather, not by the sum over all experts.
* ``routed_scaling_factor`` multiplies after the normalization, so it does not
  cancel.

``config.json`` says ``topk_method: noaux_tc``, which is vestigial: the
reference takes a plain top-k with no group limiting, and ``n_group`` /
``topk_group`` are not in the config at all.
"""

from __future__ import annotations

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn.kernels import (
    block_scales_interleave,
    grouped_matmul_block_scaled,
    moe_create_indices,
    quantize_dynamic_scaled_float8,
)
from max.nn.layer import LayerList, Module
from max.nn.linear import Linear
from max.nn.quant_config import QuantConfig

from ..model_config import DeepseekV4Config
from .quantization import LINEAR_QUANT_BLOCK, linear_for

# e2m1 weights carry one e8m0 scale per 32 elements along K (OCP MXFP4), and
# the SM100 scale-factor atom covers 128 rows -- both fixed by the kernel.
FP4_WEIGHT_BLOCK = 32
SF_ROWS = 128


def sqrt_softplus(x: TensorValue) -> TensorValue:
    """``F.softplus(x).sqrt()`` -- ``scoring_func: sqrtsoftplus``.

    MAX has no ``softplus``, so it is written in the numerically stable form:
    ``max(x, 0) + log1p(exp(-|x|))``. The naive ``log(1 + exp(x))`` overflows
    for large positive scores, which is exactly where a routed expert's score
    lives.
    """
    zero = ops.constant(0.0, DType.float32, x.device)
    softplus = ops.max(x, zero) + ops.log1p(ops.exp(-ops.abs(x)))
    return ops.sqrt(softplus)


class DeepseekV4Expert(Module):
    """SwiGLU FFN. ``w1`` gate, ``w3`` up, ``w2`` down.

    The routing weight is applied *inside*, between the activation and ``w2``.
    Scaling the output instead would be the same algebra but not the same
    arithmetic: the reference multiplies in float32 and casts to the model
    dtype afterwards, so the rounding lands on the scaled value.

    ``fp8`` selects the checkpoint's fp8 projections (the shared expert; the
    routed experts are fp4 and go through :class:`DeepseekV4RoutedExperts`
    when the model is quantized). With ``config.quant_config`` unset both
    forms are the plain ``config.dtype`` linears of the dequantized gates.
    """

    def __init__(
        self, config: DeepseekV4Config, device: DeviceRef, *, fp8: bool = False
    ) -> None:
        super().__init__()
        self.swiglu_limit = config.swiglu_limit

        def make(in_dim: int, out_dim: int) -> Linear:
            if fp8:
                return linear_for(config, in_dim, out_dim, device)
            return Linear(in_dim, out_dim, config.dtype, device)

        self.w1 = make(config.hidden_size, config.moe_intermediate_size)
        self.w2 = make(config.moe_intermediate_size, config.hidden_size)
        self.w3 = make(config.hidden_size, config.moe_intermediate_size)

    def __call__(
        self, x: TensorValue, weights: TensorValue | None = None
    ) -> TensorValue:
        gate = ops.cast(self.w1(x), DType.float32)
        up = ops.cast(self.w3(x), DType.float32)
        if self.swiglu_limit > 0:
            limit = ops.constant(self.swiglu_limit, DType.float32, gate.device)
            # Asymmetric on purpose: the reference clamps ``up`` on both sides
            # but ``gate`` only from above.
            up = ops.min(ops.max(up, -limit), limit)
            gate = ops.min(gate, limit)
        h = ops.silu(gate) * up
        if weights is not None:
            h = weights * h
        return self.w2(ops.cast(h, x.dtype))


class DeepseekV4Gate(Module):
    """Expert selection and routing weights for one layer."""

    def __init__(
        self, config: DeepseekV4Config, layer_idx: int, device: DeviceRef
    ) -> None:
        super().__init__()
        self.is_hash_routed = config.layer_is_hash_routed(layer_idx)
        self.topk = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.route_scale = config.routed_scaling_factor

        self.weight = Weight(
            name="weight",
            dtype=config.dtype,
            shape=(config.n_routed_experts, config.hidden_size),
            device=device,
        )
        if self.is_hash_routed:
            # int64 in the checkpoint. ``inference/model.py:564`` declares
            # int32, but that is the declaration; the stored tensor is I64 in
            # both the full and the minimized model, and reading it as int32
            # gives misaligned indices.
            self.tid2eid = Weight(
                name="tid2eid",
                dtype=DType.int64,
                shape=(config.vocab_size, config.num_experts_per_tok),
                device=device,
            )
        else:
            self.bias = Weight(
                name="bias",
                dtype=DType.float32,
                shape=(config.n_routed_experts,),
                device=device,
            )

    def __call__(
        self, x: TensorValue, token_ids: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        """``[b, s, d]`` and ``[b, s]`` -> weights and indices, both ``[b, s, k]``."""
        scores = sqrt_softplus(
            ops.matmul(
                ops.cast(x, DType.float32),
                ops.transpose(ops.cast(self.weight, DType.float32), 0, 1),
            )
        )
        if self.is_hash_routed:
            indices = ops.gather(self.tid2eid, token_ids, axis=0)
        else:
            _, indices = ops.top_k(scores + self.bias, self.topk, axis=-1)
        indices = ops.cast(indices, DType.int32)

        # Per-token gather along the expert axis, so batch_dims covers both
        # batch and sequence.
        weights = ops.gather_nd(
            scores, ops.unsqueeze(indices, -1), batch_dims=2
        )
        weights = weights / ops.sum(weights, axis=-1)
        return weights * self.route_scale, indices


class _PackedExpertWeight(Module):
    """One stacked expert projection: e2m1 packed two per byte + e8m0/32 scales.

    ``weight`` is ``[E, N, K/2]`` uint8 and ``weight_scale`` ``[E, N, K/32]``
    e8m0, exactly the checkpoint's storage stacked by expert id (the weight
    adapter builds both). The grouped kernel reads them as is once the scales
    are in its interleaved layout, which ``__call__`` produces.
    """

    def __init__(
        self, n_experts: int, out_dim: int, in_dim: int, device: DeviceRef
    ) -> None:
        super().__init__()
        if in_dim % SF_ROWS:
            # The W4A8 TMA copy pads packed rows in 128-element units.
            raise ValueError(f"fp4 expert K={in_dim} is not a multiple of 128")
        self.n_experts = n_experts
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.weight = Weight(
            name="weight",
            dtype=DType.uint8,
            shape=(n_experts, out_dim, in_dim // 2),
            device=device,
        )
        self.weight_scale = Weight(
            name="weight_scale",
            dtype=DType.float8_e8m0fnu,
            shape=(n_experts, out_dim, in_dim // FP4_WEIGHT_BLOCK),
            device=device,
        )

    def __call__(self, device: DeviceRef) -> TensorValue:
        """``[E, N/128, K/128, 32, 4, 4]``: the scales in the kernel's SF-atom layout."""
        scales = self.weight_scale.to(device)
        per_expert = ops.split(scales, [1] * self.n_experts, axis=0)
        return ops.stack(
            [
                block_scales_interleave(
                    ops.reshape(
                        e, [self.out_dim, self.in_dim // FP4_WEIGHT_BLOCK]
                    ),
                    FP4_WEIGHT_BLOCK,
                )
                for e in per_expert
            ],
            axis=0,
        )


def _quantize_rows_for_w4a8(
    x: TensorValue, quant_config: QuantConfig
) -> tuple[TensorValue, TensorValue]:
    """``act_quant(x, 128, "ue8m0")`` laid out for the 32-group W4A8 kernel.

    The reference quantizes an expert's input with one power-of-two scale per
    128 elements (``fp4_gemm``'s A side); the kernel wants one scale per 32.
    Quantizing at 128 and repeating each scale four times along K feeds the
    kernel the identical activation codes and scales, so the result matches
    the reference bit for bit (PROGRESS-194-QUANT §1, route (a)). Quantizing
    at 32 directly changes 27 % of the codes and cannot be gated exactly.
    """
    rows = int(x.shape[0])
    width = int(x.shape[1])
    quantized, scales = quantize_dynamic_scaled_float8(
        ops.cast(x, DType.bfloat16),
        quant_config.input_scale,
        quant_config.weight_scale,
        group_size_or_per_token=LINEAR_QUANT_BLOCK,
        out_type=DType.float8_e4m3fn,
        scales_type=DType.float8_e8m0fnu,
    )
    # scales: [K/128, rows padded to 16] -> [rows, K/128] -> [rows, K/32].
    groups = width // LINEAR_QUANT_BLOCK
    repeat = LINEAR_QUANT_BLOCK // FP4_WEIGHT_BLOCK
    per_row = ops.transpose(scales[:, 0:rows], 0, 1)
    per_row = ops.reshape(
        ops.broadcast_to(ops.unsqueeze(per_row, -1), [rows, groups, repeat]),
        [rows, groups * repeat],
    )
    return quantized, block_scales_interleave(per_row, FP4_WEIGHT_BLOCK)


class DeepseekV4RoutedExperts(Module):
    """The routed experts on the W4A8 grouped kernel (PROBE-B G3-G9).

    Replaces the dense dispatch (every expert sees every token) with one
    grouped GEMM per projection over the tokens sorted by expert. Routing is
    untouched: the gate still supplies ``(weights, indices)``, hash-routed or
    scored, and this only groups the ``[tokens, k]`` slots.

    Token layout: slots sorted by expert are laid into a buffer where each
    expert's group starts at a 128-row boundary (the exclusive prefix sum of
    the group sizes rounded up to 128). The gap rows between a
    group's last token and the next boundary belong to the group as zero rows
    -- the kernel takes contiguous groups, so they cost a little work and no
    correctness -- and the A-scale tiles then need no per-expert offset
    (``a_scale_offsets = 0``). Up to 127 wasted rows per active expert; fine
    for bringup, a grouped-quantize variant that writes the tile layout
    directly is the real fix at 256 experts (DECISIONS D23).
    """

    def __init__(self, config: DeepseekV4Config, device: DeviceRef) -> None:
        super().__init__()
        if config.quant_config is None:
            raise ValueError("native routed experts need config.quant_config")
        self.quant_config = config.quant_config
        self.n_experts = config.n_routed_experts
        self.topk = config.num_experts_per_tok
        self.hidden = config.hidden_size
        self.inter = config.moe_intermediate_size
        self.swiglu_limit = config.swiglu_limit
        self.gate_up_proj = _PackedExpertWeight(
            self.n_experts, 2 * self.inter, self.hidden, device
        )
        self.down_proj = _PackedExpertWeight(
            self.n_experts, self.hidden, self.inter, device
        )

    def _grouped(
        self,
        proj: _PackedExpertWeight,
        x: TensorValue,
        a_offsets: TensorValue,
        expert_ids: TensorValue,
    ) -> TensorValue:
        quantized, a_scales = _quantize_rows_for_w4a8(x, self.quant_config)
        device = x.device
        zeros_u32 = ops.constant(
            np.zeros(self.n_experts, np.uint32), DType.uint32, device
        )
        ones_f32 = ops.constant(
            np.ones(self.n_experts, np.float32), DType.float32, device
        )
        # Host-side usage stats: the kernel only reads the active-expert count
        # here; the max-tokens slot is an estimate the padded buffer bounds.
        usage = ops.constant(
            np.array([int(x.shape[0]), self.n_experts], np.uint32),
            DType.uint32,
            DeviceRef.CPU(),
        )
        return grouped_matmul_block_scaled(
            quantized,
            proj.weight.to(device),
            a_scales,
            proj(device),
            a_offsets,
            zeros_u32,
            expert_ids,
            ones_f32,
            usage,
            out_type=DType.bfloat16,
        )

    def __call__(
        self, x: TensorValue, weights: TensorValue, indices: TensorValue
    ) -> TensorValue:
        """``[tokens, hidden]``, ``[tokens, k]``, ``[tokens, k]`` -> ``[tokens, hidden]``.

        The sum over a token's ``k`` expert outputs, in float32, before the
        shared expert is added (the reference's ``y[idx] += expert(...)``).
        """
        device = x.device
        tokens = int(x.shape[0])
        slots = tokens * self.topk
        padded = slots + SF_ROWS * self.n_experts
        i32 = DType.int32

        router_idx = ops.reshape(indices, [slots])
        order, start, restore, expert_ids, _usage = moe_create_indices(
            router_idx, self.n_experts
        )
        order = ops.cast(order, i32)
        restore = ops.cast(restore, i32)
        start = ops.cast(start, i32)

        # Each group's rows begin at a 128-row boundary: aligned_start[g] is
        # the exclusive prefix sum of the group sizes rounded up to 128, and
        # group g owns [aligned_start[g], aligned_start[g + 1]) including its
        # trailing zero rows. This is the layout MAX's grouped quantize kernel
        # pads to, so the kernel's tile lookup needs no per-group offset.
        counts = start[1 : self.n_experts + 1] - start[0 : self.n_experts]
        rows_128 = ops.constant(SF_ROWS, i32, device)
        aligned_counts = (
            ops.floor_div(
                counts + ops.constant(SF_ROWS - 1, i32, device), rows_128
            )
            * rows_128
        )
        aligned_start = ops.concat(
            [
                ops.constant(np.zeros(1, np.int32), i32, device),
                ops.cumsum(aligned_counts, axis=0),
            ],
            axis=0,
        )
        a_offsets = ops.cast(aligned_start, DType.uint32)

        # Sorted slot i of group g -> padded row aligned_start[g] + (i - start[g]).
        slot_expert = ops.gather(router_idx, order, axis=0)
        slot_rows = ops.range(
            0, slots, 1, out_dim=slots, dtype=i32, device=device
        )
        slot_pad = (
            slot_rows
            + ops.gather(aligned_start[0 : self.n_experts], slot_expert, axis=0)
            - ops.gather(start[0 : self.n_experts], slot_expert, axis=0)
        )

        # Padded row -> token (``tokens`` = the appended zero row).
        slot_token = ops.cast(
            ops.floor_div(order, ops.constant(self.topk, i32, device)), i32
        )
        row_token = ops.scatter(
            ops.constant(np.full(padded, tokens, np.int32), i32, device),
            slot_token,
            slot_pad,
            axis=0,
        )
        x_ext = ops.concat(
            [
                x,
                ops.constant(
                    np.zeros((1, self.hidden), np.float32), x.dtype, device
                ),
            ],
            axis=0,
        )
        x_pad = ops.gather(x_ext, row_token, axis=0)
        slot_weight = ops.gather(ops.reshape(weights, [slots]), order, axis=0)
        row_weight = ops.scatter(
            ops.constant(np.zeros(padded, np.float32), DType.float32, device),
            ops.cast(slot_weight, DType.float32),
            slot_pad,
            axis=0,
        )

        gate_up = self._grouped(self.gate_up_proj, x_pad, a_offsets, expert_ids)
        gate, up = ops.split(gate_up, [self.inter, self.inter], axis=1)
        gate = ops.cast(gate, DType.float32)
        up = ops.cast(up, DType.float32)
        if self.swiglu_limit > 0:
            limit = ops.constant(self.swiglu_limit, DType.float32, device)
            # Asymmetric on purpose: the reference clamps ``up`` on both sides
            # but ``gate`` only from above.
            up = ops.min(ops.max(up, -limit), limit)
            gate = ops.min(gate, limit)
        h = ops.silu(gate) * up * ops.unsqueeze(row_weight, -1)
        down = self._grouped(self.down_proj, h, a_offsets, expert_ids)

        out = ops.gather(ops.gather(down, slot_pad, axis=0), restore, axis=0)
        out = ops.reshape(
            ops.cast(out, DType.float32), [tokens, self.topk, self.hidden]
        )
        return ops.cast(ops.squeeze(ops.sum(out, axis=1), axis=1), x.dtype)


class DeepseekV4MoE(Module):
    """``num_experts_per_tok`` routed experts plus one shared expert."""

    def __init__(
        self, config: DeepseekV4Config, layer_idx: int, device: DeviceRef
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.n_routed_experts = config.n_routed_experts
        self.gate = DeepseekV4Gate(config, layer_idx, device)
        self.native_experts = config.routed_experts_native
        self.experts: Module
        if self.native_experts:
            self.experts = DeepseekV4RoutedExperts(config, device)
        else:
            self.experts = LayerList(
                [
                    DeepseekV4Expert(config, device)
                    for _ in range(config.n_routed_experts)
                ]
            )
        self.shared_experts = DeepseekV4Expert(config, device, fp8=True)

    def __call__(self, x: TensorValue, token_ids: TensorValue) -> TensorValue:
        """``[b, s, hidden]`` in, same out.

        Dense dispatch: every expert sees every token, weighted by zero where
        it was not selected. Per MXSERV-502 kernel performance is out of scope,
        and a gather-based dispatch needs a dynamic shape per expert, which the
        graph cannot express. On the 8-expert minimized model this costs 8/6 of
        the routed work; on the 256-expert model it would cost 42x, so this is
        the piece to replace first if the full checkpoint is ever run.

        The scatter from ``[b, s, k]`` slots to ``[b, s, n_experts]`` sums when
        two slots name the same expert. The reference's ``y[idx] += v`` would
        instead keep only the last of them, but no such row exists: every
        ``tid2eid`` row in the checkpoint holds ``k`` distinct experts.
        """
        weights, indices = self.gate(x, token_ids)
        if self.native_experts:
            assert isinstance(self.experts, DeepseekV4RoutedExperts)
            hidden = int(x.shape[-1])
            batch, seq = x.shape[0], x.shape[1]
            routed = self.experts(
                ops.reshape(x, [batch * seq, hidden]),
                ops.reshape(weights, [batch * seq, self.experts.topk]),
                ops.reshape(indices, [batch * seq, self.experts.topk]),
            )
            y = ops.cast(self.shared_experts(x), DType.float32) + ops.reshape(
                ops.cast(routed, DType.float32), [batch, seq, hidden]
            )
            return ops.cast(y, x.dtype)
        assert isinstance(self.experts, LayerList)
        experts = ops.range(
            0,
            self.n_routed_experts,
            1,
            out_dim=self.n_routed_experts,
            device=indices.device,
            dtype=DType.int32,
        )
        selected = ops.cast(
            ops.unsqueeze(indices, -1) == experts, DType.float32
        )
        per_expert = ops.squeeze(
            ops.sum(ops.unsqueeze(weights, -1) * selected, axis=2), axis=2
        )

        y = ops.cast(self.shared_experts(x), DType.float32)
        for i, expert in enumerate(self.experts):
            y = y + ops.cast(
                expert(x, per_expert[..., i : i + 1]), DType.float32
            )
        return ops.cast(y, x.dtype)
