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
# Ported to ModuleV3 from
# max/python/max/pipelines/architectures/qwen3_5/layers/gated_deltanet.py
# at main 4df8111afba.
"""Gated DeltaNet, the linear attention of Qwen3.5's linear layers.

State is held in two pools that the kernels update in place at each
request's row:

- ``conv_pool``: the window of recent inputs of the causal conv1d,
  ``[rows, conv_dim, kernel_size - 1]``.
- ``recurrent_pool``: the accumulated key-value memory,
  ``[rows, num_value_heads, key_head_dim, value_head_dim]``.

Under tensor parallelism the heads are split, so ``conv_dim`` and
``num_value_heads`` are each device's share, and each device owns its pools.
A decode step runs the slot-indexed conv and recurrence kernels; a prefill
on SM100+ runs the chunked recurrence instead.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from max import functional as F
from max import tree
from max.driver import CPU, accelerator_api
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.nn import Module
from max.nn._graph_ops.state_space import has_blackwell_tcgen05
from max.nn.common_layers.mesh_axis import TP
from max.nn.kernels import (
    gated_delta_conv1d_fwd,
    gated_delta_conv1d_verify_fwd,
    gated_delta_recurrence_fwd,
    gated_delta_recurrence_verify_ring_fwd,
    kda_chunk,
    kda_chunk_is_registered,
    kda_chunk_rows,
    kda_chunk_rows_servable,
    kda_chunk_supports_head_dims,
)
from max.nn.quant_config import QuantConfig
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping, NamedMapping, Partial
from max.tensor.sharding.action import PerShard

from .linear import Projection, ProjectionTensors, matmul, stacked_matmul
from .norm import rms_norm_scaled
from .sharding import shard_segments

__all__ = [
    "GatedDeltaNet",
    "GatedDeltaReplay",
    "GatedDeltaStateAccess",
    "GatedDeltaVerify",
    "ReplaySink",
]

_OUTPUT_GATE_ACTIVATIONS: dict[str, Callable[[Tensor], Tensor]] = {
    "swish": F.silu,
    "silu": F.silu,
    "sigmoid": F.sigmoid,
}
"""Activations ``output_gate_type`` can name. Qwen3.5 says ``swish``."""


@tree.dataclass(frozen=True)
class GatedDeltaStateAccess:
    """One linear layer's rows of the state pools.

    The pools hold every layer's state; the rows are this layer's,
    ``[batch_size]``, selected before the layers' shared subgraph.
    """

    conv_pool: Tensor
    conv_row_id: Tensor
    recurrent_pool: Tensor
    recurrent_row_id: Tensor
    ring_pool: Tensor | None = None
    """The verify ring, set only inside a speculative verify."""
    ring_row_id: Tensor | None = None


@tree.dataclass(frozen=True)
class GatedDeltaReplay:
    """One layer's per-token inputs to the two state kernels, on one device.

    A speculative rollback reruns the kernels over the accepted rows from
    these, so the arithmetic it repeats is the arithmetic the verify did.
    """

    qkv: Tensor
    conv_weight: Tensor
    decay: Tensor
    beta: Tensor
    conv_output: Tensor
    qkv_window: Tensor | None = None


class ReplaySink:
    """One device's list of a verify's replay inputs.

    A per-device function appends to it; a plain list would be rebuilt on
    the way in, so the appends would land in a copy.
    """

    __slots__ = ("items",)

    def __init__(self, items: list[GatedDeltaReplay]) -> None:
        self.items = items


@dataclass
class GatedDeltaVerify:
    """A speculative verify's view of the linear layers.

    Set around the target's verify only. The verify records each token's
    state update in the ring instead of writing the recurrent pool, and
    leaves the conv window unwritten; the rollback lands both on the
    accepted prefix.
    """

    verify_width: Tensor
    """The host ``[K]`` operand the state kernels read ``K`` from."""
    captures: list[list[GatedDeltaReplay]] = field(default_factory=list)
    """Per device, every linear layer's replay inputs in schedule order."""


def _mixed_case_predicate(
    verify_width: Tensor,
    prefill_case: Tensor,
    total_tokens: Tensor,
    batch_size: Tensor,
) -> Tensor:
    """Returns whether a verify step holds a row longer than its window."""
    k = F.shape_to_tensor(verify_width.shape)[0]
    return F.logical_and(
        F.logical_not(prefill_case), total_tokens > batch_size * (k + 1)
    )


def _prefill_case(verify_width: Tensor, is_prefill: Tensor) -> Tensor:
    """Returns whether a draft-free prefill owns the step."""
    no_drafts = F.equal(F.shape_to_tensor(verify_width.shape)[0], 0)
    return F.logical_and(no_drafts, is_prefill)


def _gated_delta_core(
    x: Tensor,
    in_proj: Sequence[ProjectionTensors],
    in_proj_ba: Sequence[ProjectionTensors] | None,
    conv_weight: Tensor,
    dt_bias: Tensor,
    a_log: Tensor,
    norm_weight: Tensor,
    out_proj: ProjectionTensors,
    state: GatedDeltaStateAccess,
    input_row_offsets: Tensor,
    verify_width: Tensor | None,
    capture: ReplaySink | None,
    *,
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    ssm_dtype: DType,
    norm_eps: float,
    output_gate: Callable[[Tensor], Tensor],
) -> Tensor:
    """Runs one device's heads and returns its ``out_proj`` partial sums."""
    device = x.device
    nv = num_value_heads
    vd = value_head_dim
    key_dim = key_head_dim * num_key_heads
    value_dim = value_head_dim * num_value_heads
    conv_dim = key_dim * 2 + value_dim
    kernel_size = int(conv_weight.shape[-1])

    if in_proj_ba is None:
        qkv, z, b_proj, a_proj = F.split(
            stacked_matmul(x, in_proj), [conv_dim, value_dim, nv, nv], axis=-1
        )
    else:
        qkv, z = F.split(
            stacked_matmul(x, in_proj), [conv_dim, value_dim], axis=-1
        )
        b_proj, a_proj = F.split(
            stacked_matmul(x, in_proj_ba), [nv, nv], axis=-1
        )
    qkv_f32 = qkv.cast(DType.float32)

    dt_bias = dt_bias.cast(DType.float32)
    a_log = a_log.cast(DType.float32)
    decay_rate = F.exp(a_log.cast(ssm_dtype))
    a_float = a_proj.cast(ssm_dtype)
    # A stable softplus: past 20 it is the identity, and exp would overflow.
    x_sp = a_float + dt_bias.cast(ssm_dtype)
    softplus = F.where(x_sp > 20.0, x_sp, F.log(1.0 + F.exp(x_sp)))
    decay = F.exp((-decay_rate * softplus).cast(DType.float32))
    beta = F.sigmoid(b_proj).cast(DType.float32)

    conv_weight = conv_weight.cast(DType.float32).reshape(
        [conv_dim, kernel_size]
    )
    offsets_uint32 = input_row_offsets.cast(DType.uint32)
    offsets_int32 = input_row_offsets.cast(DType.int32)
    conv_slot = state.conv_row_id.cast(DType.uint32)
    rec_slot_uint32 = state.recurrent_row_id.cast(DType.uint32)
    rec_slot_int32 = state.recurrent_row_id.cast(DType.int32)

    qkv_window: Tensor | None = None
    if verify_width is None:
        conv_output = gated_delta_conv1d_fwd(
            qkv_input_ragged=qkv_f32,
            conv_weight=conv_weight,
            conv_state=state.conv_pool,
            slot_idx=conv_slot,
            input_row_offsets=offsets_uint32,
        )
    else:
        conv_output, qkv_window = gated_delta_conv1d_verify_fwd(
            qkv_input_ragged=qkv_f32,
            conv_weight=conv_weight,
            conv_state=state.conv_pool,
            slot_idx=conv_slot,
            input_row_offsets=offsets_uint32,
            verify_width=verify_width,
            rollback=False,
        )
    conv_output = F.silu(conv_output)

    if capture is not None:
        capture.items.append(
            GatedDeltaReplay(
                qkv=qkv_f32,
                conv_weight=conv_weight,
                decay=decay,
                beta=beta,
                conv_output=conv_output,
                qkv_window=qkv_window,
            )
        )

    def chunk_inputs() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # Separate bf16 q/k/v give the chunk op explicit head dims.
        q, k, v = F.split(conv_output, [key_dim, key_dim, value_dim], axis=-1)
        q = q.reshape([-1, num_key_heads, key_head_dim]).cast(DType.bfloat16)
        k = k.reshape([-1, num_key_heads, key_head_dim]).cast(DType.bfloat16)
        v = v.reshape([-1, nv, vd]).cast(DType.bfloat16)
        # The chunk op takes a per-channel gate, so each head's scalar is
        # spread along the key dim. A multiply materializes it: the kernel
        # reads the elements through a TMA descriptor, which a stride-0
        # broadcast view would not back.
        ones_k = F.constant(
            np.ones((1, 1, key_head_dim), dtype=np.float32),
            DType.float32,
            device=device,
        )
        raw_gate = a_float.reshape([-1, nv, 1]) * ones_k
        dt_bias_2d = dt_bias.reshape([nv, 1]) * F.constant(
            np.ones((1, key_head_dim), dtype=np.float32),
            DType.float32,
            device=device,
        )
        return q, k, v, raw_gate, dt_bias_2d

    def prefill_recurrence() -> Tensor:
        q, k, v, raw_gate, dt_bias_2d = chunk_inputs()
        out = kda_chunk(
            q=q,
            k=k,
            v=v,
            raw_gate=raw_gate,
            beta_logits=beta,
            a_log=a_log,
            dt_bias=dt_bias_2d,
            cu_seqlens=offsets_int32,
            state_pool=state.recurrent_pool,
            state_indices=rec_slot_int32,
            output_dtype=DType.float32,
            gate_mode="original",
            beta_mode="probability",
            state_layout="K_FIRST",
            use_computebound=True,
        )
        return out.reshape([-1, value_dim])

    def chunk_long_rows(ring_output: Tensor) -> Tensor:
        assert verify_width is not None
        q, k, v, raw_gate, dt_bias_2d = chunk_inputs()
        out = kda_chunk_rows(
            q=q,
            k=k,
            v=v,
            raw_gate=raw_gate,
            beta_logits=beta,
            a_log=a_log,
            dt_bias=dt_bias_2d,
            cu_seqlens=offsets_int32,
            state_pool=state.recurrent_pool,
            state_indices=rec_slot_int32,
            passthrough=ring_output.reshape([conv_output.shape[0], nv, vd]),
            verify_width=verify_width,
            output_dtype=DType.float32,
            gate_mode="original",
            beta_mode="probability",
            state_layout="K_FIRST",
        )
        return out.reshape([-1, value_dim])

    def decode_recurrence() -> Tensor:
        return gated_delta_recurrence_fwd(
            qkv_conv_output=conv_output,
            decay_per_token=decay,
            beta_per_token=beta,
            recurrent_state=state.recurrent_pool,
            slot_idx=rec_slot_uint32,
            input_row_offsets=offsets_uint32,
        )

    def verify_recurrence(skip_long_rows: bool = False) -> Tensor:
        assert verify_width is not None
        assert state.ring_pool is not None and state.ring_row_id is not None
        return gated_delta_recurrence_verify_ring_fwd(
            qkv_conv_output=conv_output,
            decay_per_token=decay,
            beta_per_token=beta,
            recurrent_state=state.recurrent_pool,
            ring=state.ring_pool,
            slot_idx=rec_slot_uint32,
            ring_slot_idx=state.ring_row_id.cast(DType.uint32),
            input_row_offsets=offsets_uint32,
            verify_width=verify_width,
            skip_long_rows=skip_long_rows,
        )

    # The chunk op's speed is in its tcgen05 scan and chain bodies, so a part
    # without tcgen05 keeps the sequential recurrence for prefill. A build
    # without the closed-source kernels doesn't register the op either.
    chunk_servable = (
        accelerator_api() == "cuda"
        and has_blackwell_tcgen05()
        and kda_chunk_supports_head_dims(key_head_dim, value_head_dim)
        and kda_chunk_is_registered()
    )
    rows_servable = chunk_servable and kda_chunk_rows_servable(
        key_head_dim, value_head_dim
    )
    output_types = [
        TensorType(
            DType.float32,
            [x.shape[0], value_dim],
            DeviceRef.from_device(device),
        )
    ]
    total_tokens = F.shape_to_tensor(x.shape)[0]
    batch_size = F.shape_to_tensor(input_row_offsets.shape)[0] - 1
    # A decode batch has one token per sequence. `F.cond` branches are
    # separate regions, so a device graph captured for decode records only
    # the decode kernels, and not the chunk map's host readback.
    is_prefill = total_tokens > batch_size

    if verify_width is not None and rows_servable:
        # A draft-free prefill has nothing to roll back, so it lands on the
        # forward's prefill kernel. A verify keeps its decode rows on the
        # ring recurrence the fold follows, and commits its prompt rows
        # through the chunked prefill.
        prefill_case = _prefill_case(verify_width, is_prefill)
        (ring_output,) = F.cond(
            prefill_case,
            output_types,
            lambda: [prefill_recurrence()],
            lambda: [verify_recurrence(skip_long_rows=True)],
        )
        (recurrence,) = F.cond(
            _mixed_case_predicate(
                verify_width, prefill_case, total_tokens, batch_size
            ),
            output_types,
            lambda: [chunk_long_rows(ring_output)],
            lambda: [ring_output],
        )
    elif verify_width is not None and chunk_servable:
        (recurrence,) = F.cond(
            _prefill_case(verify_width, is_prefill),
            output_types,
            lambda: [prefill_recurrence()],
            lambda: [verify_recurrence()],
        )
    elif verify_width is not None:
        recurrence = verify_recurrence()
    elif chunk_servable:
        (recurrence,) = F.cond(
            is_prefill,
            output_types,
            lambda: [prefill_recurrence()],
            lambda: [decode_recurrence()],
        )
    else:
        recurrence = decode_recurrence()

    recurrence = F.rebind(
        recurrence,
        [x.shape[0], value_dim],
        "recurrence_output_flat total_seq_len rebind",
    )
    # Normalize, then gate.
    normed = rms_norm_scaled(
        recurrence.reshape([-1, nv, vd]).cast(x.dtype),
        norm_weight,
        eps=norm_eps,
        weight_offset=0.0,
    )
    gate = output_gate(z.reshape([-1, nv, vd]).cast(DType.float32))
    gated = (normed.cast(DType.float32) * gate).cast(x.dtype)
    return matmul(gated.reshape([-1, value_dim]), out_proj)


class _DepthwiseConv(Module[..., Tensor]):
    """The causal conv's ``[conv_dim, 1, kernel_size]`` weight.

    It stays on the host, as stored, for the layer to place each device's
    share of every head-major part.
    """

    def __init__(self, conv_dim: int, kernel_size: int, dtype: DType) -> None:
        self.weight = Tensor.zeros(
            [conv_dim, 1, kernel_size],
            dtype=dtype,
            device=CPU(),
        )

    def forward(self) -> Tensor:
        """Returns the weight."""
        return self.weight


class GatedDeltaNet(Module[..., Tensor]):
    """Gated DeltaNet linear attention, split by head across devices.

    The recurrence maps value head ``v`` onto key head ``v // (nv // nk)``,
    so cutting both head counts by the device count keeps that mapping on
    every device. Returns ``out_proj``'s partial sums, which the block
    reduces.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_key_heads: int,
        num_value_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        conv_kernel_size: int,
        dtype: DType,
        rms_norm_eps: float,
        ssm_dtype: DType = DType.float32,
        quant_config: QuantConfig | None = None,
        output_gate_type: str = "swish",
        scalar_dtypes: Mapping[str, DType] | None = None,
    ) -> None:
        if output_gate_type not in _OUTPUT_GATE_ACTIVATIONS:
            raise ValueError(
                f"unsupported output_gate_type {output_gate_type!r}; expected"
                f" one of {sorted(_OUTPUT_GATE_ACTIVATIONS)}"
            )
        self.num_key_heads = num_key_heads
        self.num_value_heads = num_value_heads
        self.key_head_dim = key_head_dim
        self.value_head_dim = value_head_dim
        self.ssm_dtype = ssm_dtype
        self.rms_norm_eps = rms_norm_eps
        self.output_gate_type = output_gate_type
        scalar_dtypes = dict(scalar_dtypes or {})
        key_dim = key_head_dim * num_key_heads
        value_dim = value_head_dim * num_value_heads
        self.segments = (key_dim, key_dim, value_dim)
        conv_dim = sum(self.segments)

        # A quantized checkpoint keeps in_proj_b and in_proj_a at the compute
        # dtype, so they can't stack with the FP8 projections.
        self.quantized = quant_config is not None
        self.in_proj_qkv = Projection(
            hidden_size,
            conv_dim,
            dtype=dtype,
            quant_config=quant_config,
            on_host=True,
        )
        self.in_proj_z = Projection(
            hidden_size,
            value_dim,
            dtype=dtype,
            quant_config=quant_config,
            sharding=(TP, None),
        )
        self.in_proj_b = Projection(
            hidden_size, num_value_heads, dtype=dtype, sharding=(TP, None)
        )
        self.in_proj_a = Projection(
            hidden_size, num_value_heads, dtype=dtype, sharding=(TP, None)
        )
        self.conv1d = _DepthwiseConv(conv_dim, conv_kernel_size, dtype)

        def per_value_head(name: str) -> Tensor:
            t = Tensor.zeros(
                [num_value_heads],
                dtype=scalar_dtypes.get(name, DType.float32),
            )
            return t.to(NamedMapping(t.mesh, (TP,)))

        self.dt_bias = per_value_head("dt_bias")
        self.A_log = per_value_head("A_log")
        # The gated norm scales by `weight`, not `1 + weight`, and reduces
        # within one head, so it needs no cross-device statistic.
        self.norm = _GatedNormWeight(
            value_head_dim, scalar_dtypes.get("norm.weight", DType.float32)
        )
        self.out_proj = Projection(
            value_dim,
            hidden_size,
            dtype=dtype,
            quant_config=quant_config,
            sharding=(None, TP),
        )
        self.verify: GatedDeltaVerify | None = None
        """Set by a speculative target around its verify, else ``None``."""

    def forward(
        self,
        x: Tensor,
        state: GatedDeltaStateAccess,
        input_row_offsets: Tensor,
    ) -> Tensor:
        """Runs the linear attention over ``x`` and advances ``state``."""
        mesh = x.mesh
        num_devices = mesh.num_devices
        qkv = self.in_proj_qkv.tensors
        qkv = ProjectionTensors(
            weight=shard_segments(qkv.weight, self.segments, mesh),
            quant_config=qkv.quant_config,
            weight_scale=qkv.weight_scale,
            input_scale=qkv.input_scale,
        )
        ba = (self.in_proj_b.tensors, self.in_proj_a.tensors)
        in_proj = (qkv, self.in_proj_z.tensors)
        verify = self.verify
        core = functools.partial(
            _gated_delta_core,
            num_key_heads=self.num_key_heads // num_devices,
            num_value_heads=self.num_value_heads // num_devices,
            key_head_dim=self.key_head_dim,
            value_head_dim=self.value_head_dim,
            ssm_dtype=self.ssm_dtype,
            norm_eps=self.rms_norm_eps,
            output_gate=_OUTPUT_GATE_ACTIVATIONS[self.output_gate_type],
        )
        return F.call_on_mesh(
            core, mesh, out_specs=DeviceMapping(mesh, (Partial(),))
        )(
            x,
            in_proj if self.quantized else (*in_proj, *ba),
            ba if self.quantized else None,
            shard_segments(self.conv1d.weight, self.segments, mesh),
            self.dt_bias,
            self.A_log,
            self.norm.weight,
            self.out_proj.tensors,
            state,
            input_row_offsets,
            None if verify is None else verify.verify_width,
            None
            if verify is None
            else PerShard(ReplaySink(c) for c in verify.captures),
        )


class _GatedNormWeight(Module[..., Tensor]):
    """The gated output norm's ``[value_head_dim]`` weight."""

    def __init__(self, dim: int, dtype: DType) -> None:
        self.weight = Tensor.zeros([dim], dtype=dtype)

    def forward(self) -> Tensor:
        """Returns the weight."""
        return self.weight
