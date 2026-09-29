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

from collections.abc import Sequence

import numpy as np
from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    Weight,
    ops,
)
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
from .ragged import count_offsets, segment_ids

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
            limit = float(self.swiglu_limit)
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
    x: TensorValue, quant_config: QuantConfig, scale_slot: TensorValue
) -> tuple[TensorValue, TensorValue]:
    """``act_quant(x, 128, "ue8m0")`` laid out for the 32-group W4A8 kernel.

    The reference quantizes an expert's input with one power-of-two scale per
    128 elements (``fp4_gemm``'s A side); the kernel wants one scale per 32.
    Quantizing at 128 and repeating each scale four times along K feeds the
    kernel the identical activation codes and scales, so the result matches
    the reference bit for bit (PROGRESS-194-QUANT §1, route (a)). Quantizing
    at 32 directly changes 27 % of the codes and cannot be gated exactly.

    ``x`` holds one row per slot, packed by expert. The codes stay packed;
    only the scales are laid out by ``scale_slot`` (the slot whose scales
    each scale row holds) into the 128-row-aligned tiles the kernel reads.
    """
    width = int(x.shape[1])
    quantized, scales = quantize_dynamic_scaled_float8(
        ops.cast(x, DType.bfloat16),
        quant_config.input_scale,
        quant_config.weight_scale,
        group_size_or_per_token=LINEAR_QUANT_BLOCK,
        out_type=DType.float8_e4m3fn,
        scales_type=DType.float8_e8m0fnu,
    )
    # scales: [K/128, rows padded to 16] -> [rows, K/128] -> scale rows ->
    # [scale rows, K/32].
    groups = width // LINEAR_QUANT_BLOCK
    repeat = LINEAR_QUANT_BLOCK // FP4_WEIGHT_BLOCK
    per_row = ops.transpose(scales[:, 0 : x.shape[0]], 0, 1)
    per_row = ops.gather(per_row, scale_slot, axis=0)
    rows = scale_slot.shape[0]
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

    Token layout: the activations are the slots sorted by expert, packed,
    one row each. Only the A-side scales are padded: the kernel reads each
    group's scales from 128-row tiles of its own, located per group by
    ``a_scale_offsets`` (``start // 128 + offset`` is the group's first
    tile, the layout ``ep_comm``'s ``pad_expert_offsets`` builds). So the
    quantize and the GEMMs touch the slots only, and the padding -- up to
    127 rows per group, over every group, whether or not it has a token --
    costs one scale gather.
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
        # The global id of this module's first expert; with ``n_experts`` it
        # names the experts it holds. All of them unless expert-parallel.
        self.expert_offset = 0
        self.n_global_experts = self.n_experts

    def shard_experts(self, num_devices: int) -> None:
        """Declares the expert-parallel split: whole experts along axis 0."""
        for proj in (self.gate_up_proj, self.down_proj):
            proj.weight.sharding_strategy = ShardingStrategy.axiswise(
                0, num_devices
            )
            proj.weight_scale.sharding_strategy = ShardingStrategy.axiswise(
                0, num_devices
            )

    def keep_local_experts(self, num_devices: int, rank: int) -> None:
        """Makes this replica hold the ``rank``-th run of :meth:`shard_experts`."""
        local = self.n_global_experts // num_devices
        self.n_experts = local
        self.expert_offset = rank * local
        self.gate_up_proj.n_experts = local
        self.down_proj.n_experts = local

    def _grouped(
        self,
        proj: _PackedExpertWeight,
        x: TensorValue,
        a_offsets: TensorValue,
        scale_offsets: TensorValue,
        scale_slot: TensorValue,
        expert_ids: TensorValue,
    ) -> TensorValue:
        quantized, a_scales = _quantize_rows_for_w4a8(
            x, self.quant_config, scale_slot
        )
        device = x.device
        ones_f32 = ops.constant(
            np.ones(self.n_experts, np.float32), DType.float32, device
        )
        # Host-side usage stats: the kernel only reads the active-expert count
        # here; the max-tokens slot is an estimate the row count bounds.
        # Both are host values; the row count is read off the (symbolic)
        # shape at run time.
        usage = ops.concat(
            [
                ops.cast(ops.shape_to_tensor([x.shape[0]]), DType.uint32),
                ops.constant(
                    np.array([self.n_experts], np.uint32),
                    DType.uint32,
                    DeviceRef.CPU(),
                ),
            ],
            axis=0,
        )
        return grouped_matmul_block_scaled(
            quantized,
            proj.weight.to(device),
            a_scales,
            proj(device),
            a_offsets,
            scale_offsets,
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
        Expert-parallel (:meth:`keep_local_experts`), only the slots routed to
        this module's experts contribute; the rest are zero.
        """
        slots = x.shape[0] * self.topk
        router_idx = ops.reshape(indices, [slots])
        if self.n_experts == self.n_global_experts:
            return self._routed(x, weights, router_idx, self.n_experts, None)
        first = self.expert_offset
        end = self.expert_offset + self.n_experts
        is_local = ops.logical_and(
            ops.greater_equal(router_idx, first), ops.greater(end, router_idx)
        )
        # Other devices' slots form one extra trailing group that no GEMM
        # covers. They cannot simply be left out of range: moe_create_indices
        # skips such ids, leaving their order/restore entries unwritten.
        router_idx = ops.where(
            is_local,
            router_idx - first,
            self.n_experts,
        )
        return self._routed(
            x, weights, router_idx, self.n_experts + 1, is_local
        )

    def _routed(
        self,
        x: TensorValue,
        weights: TensorValue,
        router_idx: TensorValue,
        groups: int,
        is_local: TensorValue | None,
    ) -> TensorValue:
        """:meth:`__call__` over ``router_idx`` in ``[0, groups)``.

        Groups past ``self.n_experts`` are not multiplied; ``is_local`` (per
        slot) masks their rows' output, which the GEMMs leave unwritten.
        """
        device = x.device
        # ``tokens`` is symbolic in the serving graph (the batch's ragged
        # row count), so every length below is a shape expression and every
        # value derived from it is read at run time.
        tokens = x.shape[0]
        slots = tokens * self.topk
        # Scale rows only: every group's scales start on a 128-row tile.
        padded = slots + SF_ROWS * groups
        i32 = DType.int32

        order, start, restore, expert_ids, _usage = moe_create_indices(
            router_idx, groups
        )
        a_offsets = start
        order = ops.cast(order, i32)
        restore = ops.cast(restore, i32)
        start = ops.cast(start, i32)

        # Group g's scales occupy the tiles [aligned_start[g], aligned_start[g
        # + 1]) / 128: the exclusive prefix sum of the group sizes rounded up
        # to 128. The kernel finds that first tile as start[g] // 128 plus the
        # group's scale offset.
        counts = start[1 : groups + 1] - start[0:groups]
        aligned_counts = (counts + (SF_ROWS - 1)) // SF_ROWS * SF_ROWS
        # Kept on device, as are the row maps below: ops.cumsum/ops.scatter
        # run on the host, and those round trips beside the tokens broadcast
        # closed a 2-GPU deadlock.
        aligned_start = count_offsets(aligned_counts)
        scale_offsets = ops.cast(
            aligned_start[0:groups] // SF_ROWS - start[0:groups] // SF_ROWS,
            DType.uint32,
        )
        if groups != self.n_experts:
            a_offsets = a_offsets[0 : self.n_experts + 1]
            scale_offsets = scale_offsets[0 : self.n_experts]
            expert_ids = expert_ids[0 : self.n_experts]

        # Scale row r of group g holds sorted slot start[g] + (r -
        # aligned_start[g]) when that is below start[g + 1]; the rest of the
        # tile is never multiplied into a stored row, so any slot's scales
        # do. Rows past aligned_start[groups] are in no group; clamped to the
        # last one they fail the same test.
        row_group, row_ids = segment_ids(aligned_start, padded, device)
        row_group = ops.min(row_group, groups - 1)
        in_group = row_ids - ops.gather(aligned_start, row_group, axis=0)
        has_slot = in_group < ops.gather(counts, row_group, axis=0)
        scale_slot = ops.where(
            has_slot,
            ops.gather(start, row_group, axis=0) + in_group,
            0,
        )

        slot_token = ops.cast(order // self.topk, i32)
        x_sorted = ops.gather(x, slot_token, axis=0)
        slot_weight = ops.cast(
            ops.gather(ops.reshape(weights, [slots]), order, axis=0),
            DType.float32,
        )

        gate_up = self._grouped(
            self.gate_up_proj,
            x_sorted,
            a_offsets,
            scale_offsets,
            scale_slot,
            expert_ids,
        )
        gate, up = ops.split(gate_up, [self.inter, self.inter], axis=1)
        gate = ops.cast(gate, DType.float32)
        up = ops.cast(up, DType.float32)
        if self.swiglu_limit > 0:
            limit = float(self.swiglu_limit)
            # Asymmetric on purpose: the reference clamps ``up`` on both sides
            # but ``gate`` only from above.
            up = ops.min(ops.max(up, -limit), limit)
            gate = ops.min(gate, limit)
        h = ops.silu(gate) * up * ops.unsqueeze(slot_weight, -1)
        down = self._grouped(
            self.down_proj, h, a_offsets, scale_offsets, scale_slot, expert_ids
        )

        out = ops.gather(down, restore, axis=0)
        out = ops.cast(out, DType.float32)
        if is_local is not None:
            # ``where``, not a multiply: the masked rows hold whatever the
            # GEMM left there, NaN included.
            out = ops.where(
                ops.unsqueeze(is_local, -1),
                out,
                0.0,
            )
        out = ops.reshape(out, [tokens, self.topk, self.hidden])
        routed = ops.squeeze(ops.sum(out, axis=1), axis=1)
        if is_local is not None:
            # Kept float32 for the all-reduce, as the reference's ``y``.
            return routed
        return ops.cast(routed, x.dtype)


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
        # Global ids of the routed experts this module computes.
        self.local_experts = range(config.n_routed_experts)

    def shard_experts(self, num_devices: int) -> None:
        """Declares the reference's expert-parallel split.

        Each device owns a contiguous run of ``n_routed_experts / n`` whole
        experts. The gate (``tid2eid`` included) and the shared expert are
        replicated. The dense experts keep a replicated copy each; a replica
        just does not compute the ones it does not own.
        """
        if self.n_routed_experts % num_devices:
            raise ValueError(
                f"{self.n_routed_experts} routed experts do not split across "
                f"{num_devices} devices"
            )
        if self.native_experts:
            assert isinstance(self.experts, DeepseekV4RoutedExperts)
            self.experts.shard_experts(num_devices)

    def keep_local_experts(self, num_devices: int, rank: int) -> None:
        """Makes this replica compute only its run of :meth:`shard_experts`."""
        local = self.n_routed_experts // num_devices
        self.local_experts = range(rank * local, (rank + 1) * local)
        if self.native_experts:
            assert isinstance(self.experts, DeepseekV4RoutedExperts)
            self.experts.keep_local_experts(num_devices, rank)

    def routed(self, x: TensorValue, token_ids: TensorValue) -> TensorValue:
        """``[b, s, hidden]`` -> float32 ``[b, s, hidden]``, the reference's
        ``y`` before its all-reduce: this module's experts' share only.
        """
        weights, indices = self.gate(x, token_ids)
        hidden = int(x.shape[-1])
        batch, seq = x.shape[0], x.shape[1]
        if self.native_experts:
            assert isinstance(self.experts, DeepseekV4RoutedExperts)
            routed = self.experts(
                ops.reshape(x, [batch * seq, hidden]),
                ops.reshape(weights, [batch * seq, self.experts.topk]),
                ops.reshape(indices, [batch * seq, self.experts.topk]),
            )
            return ops.reshape(
                ops.cast(routed, DType.float32), [batch, seq, hidden]
            )
        assert isinstance(self.experts, LayerList)
        per_expert = self._per_expert_weights(weights, indices)
        y: TensorValue | None = None
        for i in self.local_experts:
            out = ops.cast(
                self.experts[i](x, per_expert[..., i : i + 1]), DType.float32
            )
            y = out if y is None else y + out
        assert y is not None
        return y

    @staticmethod
    def tensor_parallel(
        moes: Sequence[DeepseekV4MoE],
        xs: Sequence[TensorValue],
        token_ids: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
    ) -> list[TensorValue]:
        """:meth:`__call__` expert-parallel; ``moes`` holds a replica each.

        As ``MoE.forward``: every device sums its own experts' outputs, the
        sum is all-reduced in float32, and only then is the shared expert
        added. Adding it before would count it once per device.
        """
        partials = [
            moe.routed(x, tok)
            for moe, x, tok in zip(moes, xs, token_ids, strict=True)
        ]
        reduced = ops.allreduce.sum(partials, signal_buffers)
        return [
            ops.cast(
                y + ops.cast(moe.shared_experts(x), DType.float32), x.dtype
            )
            for moe, x, y in zip(moes, xs, reduced, strict=True)
        ]

    def _per_expert_weights(
        self, weights: TensorValue, indices: TensorValue
    ) -> TensorValue:
        """``[b, s, k]`` slots -> ``[b, s, n_routed_experts]``, zero if unpicked."""
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
        return ops.squeeze(
            ops.sum(ops.unsqueeze(weights, -1) * selected, axis=2), axis=2
        )

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
        per_expert = self._per_expert_weights(weights, indices)

        y = ops.cast(self.shared_experts(x), DType.float32)
        for i, expert in enumerate(self.experts):
            y = y + ops.cast(
                expert(x, per_expert[..., i : i + 1]), DType.float32
            )
        return ops.cast(y, x.dtype)
