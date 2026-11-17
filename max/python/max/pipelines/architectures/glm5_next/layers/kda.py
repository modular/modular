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
"""Kimi Delta Attention, GLM-5.3-Flash's 34 linear-attention layers.

Three orderings are wrong silently, and this module keeps all three: the conv
channels are q, k, v in that order, the gated norm normalizes before it gates,
and tensor parallelism splits heads *within* each of q, k and v -- which is why
the three conv weights stay separate tensors here and are concatenated per
device, not sharded as one.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Generic, NamedTuple, TypeVar

from max import functional as F
from max import tree
from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef, Dim, DimLike, TensorType
from max.nn import Linear, Module, RMSNorm
from max.nn._graph_ops.state_space.kimi_delta import (
    STATE_LAYOUT,
    KdaGateMode,
    kda_chunk_bf16_launch,
    kda_chunk_predicate,
    kda_chunk_servable,
    kda_gate_mode,
)
from max.nn.common_layers.linear import col_parallel, row_parallel
from max.nn.common_layers.mesh_axis import TP
from max.nn.kernels import (
    gated_delta_conv1d_fwd,
    gated_delta_conv1d_verify_fwd,
    kda_chunk,
    kda_chunk_conv,
    kda_decode,
    kda_verify_ring,
    kda_verify_ring_kgroup_conv,
    kda_verify_ring_kgroup_conv_servable,
)
from max.tensor import Tensor
from max.tensor.sharding import NamedMapping
from max.tensor.sharding.action import PerShard

from ..model_config import Glm5NextConfig
from .kda_replay import KdaReplayInputs

__all__ = [
    "CaptureSink",
    "KdaStateAccess",
    "KdaVerify",
    "KimiDeltaAttention",
    "kda_core",
    "layer_rows",
    "verify_width_operand",
]


@tree.dataclass(frozen=True)
class KdaStateAccess:
    """One KDA layer's view of the two state pools.

    Each pool travels with the ``[num_kda_layers, batch_size]`` rows of every
    layer, and the layer reads its own rows by ``layer``, so the same layer
    body can serve every KDA layer.
    """

    conv_pool: Tensor
    conv_rows: Tensor
    recurrent_pool: Tensor
    recurrent_rows: Tensor
    layer: Tensor
    """int64 CPU scalar: this layer's position among the KDA layers."""
    position: int | None = None
    """``layer`` as a Python int, set when the layers do not share a
    subgraph. The layer then reads its rows by a static slice, which the
    compiler passes as a view of the table rather than a kernel."""


_Captured = TypeVar("_Captured")


class CaptureSink(Generic[_Captured]):
    """One device's list of a verify's replay inputs.

    Handed to a per-device graph function, which appends to it. A list would
    be rebuilt on the way in, so the appends would land in a copy.
    """

    __slots__ = ("items",)

    def __init__(self, items: list[_Captured]) -> None:
        self.items = items


@dataclass(frozen=True)
class KdaVerify:
    """A speculative verify's view of one KDA layer.

    Set on the layer around the target's verify only. The verify records each
    token's state update into the ring instead of writing the recurrent pool,
    and leaves the conv window unwritten; the rollback lands both on the
    accepted prefix from what :attr:`captures` collects.
    """

    ring_pool: Tensor
    """The verify ring, every layer's rows, head-split like the state."""
    ring_rows: Tensor
    """``[num_kda_layers, batch_size]`` ring rows."""
    verify_width: Tensor
    """The speculative graph's host ``[K]`` verify width operand, read by
    its length only."""
    captures: list[list[KdaReplayInputs]]
    """Per-device sink every KDA layer appends its replay inputs to, in
    schedule order."""


def verify_width_operand(num_draft_tokens: DimLike) -> Tensor:
    """Returns the ``[K]`` host operand the state kernels read ``K`` from.

    The kernels read only its length, host metadata, so choosing between
    landing the state on the forward and deferring it to the rollback costs
    no device sync. At a width of zero there is no draft to reject.
    """
    return F.range(
        0,
        Dim(num_draft_tokens),
        1,
        out_dim=num_draft_tokens,
        dtype=DType.int64,
        device=CPU(),
    )


def layer_rows(
    rows: Tensor, layer: Tensor, position: int | None = None
) -> Tensor:
    """Row ``layer`` of a ``[num_layers, batch_size]`` table, as ``[batch]``.

    A static ``position`` takes the row by a static slice instead.
    """
    if position is not None:
        return rows[position]
    return rows[(slice(layer, layer + 1), 1), :].squeeze(0)


_chunk_predicate = F.functional(kda_chunk_predicate)


class _KdaOperands(NamedTuple):
    """The gate and row operands of the recurrence ops, in their dtypes."""

    raw_gate: Tensor
    beta_logits: Tensor
    a_log: Tensor
    dt_bias: Tensor
    cu_seqlens: Tensor
    state_indices: Tensor


def _kda_operands(
    raw_gate: Tensor,
    beta_logits: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    input_row_offsets: Tensor,
    recurrent_rows: Tensor,
    layer: Tensor,
    position: int | None,
    head_shape: list[int],
) -> _KdaOperands:
    """Casts and shapes the operands the recurrence ops share.

    Made outside the callers' conditionals: a weight first read inside one
    branch is defined in that branch's region, and the other branch cannot
    use it.
    """
    return _KdaOperands(
        raw_gate=raw_gate.cast(DType.float32).reshape(head_shape),
        beta_logits=beta_logits.cast(DType.float32),
        a_log=a_log.cast(DType.float32),
        dt_bias=dt_bias.reshape(head_shape[1:]),
        cu_seqlens=input_row_offsets.cast(DType.int32),
        state_indices=layer_rows(recurrent_rows, layer, position).cast(
            DType.int32
        ),
    )


def _split_heads(
    conv_out: Tensor, head_shape: list[int]
) -> tuple[Tensor, Tensor, Tensor]:
    """Splits the conv output into q, k and v, each in ``head_shape``."""
    q, k, v = (
        part.reshape(head_shape)
        for part in conv_out.split([head_shape[1] * head_shape[2]] * 3, axis=-1)
    )
    return q, k, v


def _capture_replay(
    capture: CaptureSink[KdaReplayInputs] | None,
    qkv: Tensor,
    conv_weight: Tensor,
    operands: _KdaOperands,
    qkv_window: Tensor | None,
) -> None:
    """Appends the layer's replay inputs to ``capture``, if any."""
    if capture is not None:
        capture.items.append(
            KdaReplayInputs(
                qkv=qkv,
                conv_weight=conv_weight,
                raw_gate=operands.raw_gate,
                beta_logits=operands.beta_logits,
                qkv_window=qkv_window,
            )
        )


def _chunk_core(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    operands: _KdaOperands,
    recurrent_pool: Tensor,
    gate_mode: KdaGateMode,
    out_shape: list[DimLike],
) -> Tensor:
    """Runs ``kda_chunk`` on bf16 q, k and v; returns the float32 readout."""
    # Both vendors' chunk ops take bf16 q/k/v, so this branch rounds at the op
    # boundary and nowhere else. On NVIDIA, float32 q/k/v would fall back to
    # the L1/L2/L3 pipeline instead of the fused compute-bound kernel.
    chunk_output_dtype, use_computebound = kda_chunk_bf16_launch()
    out = kda_chunk(
        q,
        k,
        v,
        operands.raw_gate,
        operands.beta_logits,
        operands.a_log,
        operands.dt_bias,
        operands.cu_seqlens,
        recurrent_pool,
        operands.state_indices,
        output_dtype=chunk_output_dtype,
        use_computebound=use_computebound,
        gate_mode=gate_mode,
        beta_mode="logits",
        state_layout=STATE_LAYOUT,
    ).cast(DType.float32)
    return F.rebind(out, out_shape, "kda chunk output rebind")


def _gated_norm(
    core: Tensor,
    norm_weight: Tensor,
    norm_eps: float,
    output_gate: Tensor,
    head_shape: list[int],
    dtype: DType,
) -> Tensor:
    """Normalizes, scales, and only then gates the recurrence readout."""
    normed = F.rms_norm(
        core,
        norm_weight.cast(DType.float32),
        norm_eps,
        multiply_before_cast=False,
    ) * F.sigmoid(output_gate.cast(DType.float32).reshape(head_shape))
    return normed.cast(dtype).reshape([-1, head_shape[1] * head_shape[2]])


def _kgroup_conv_verify(
    qkv: Tensor,
    conv_weight: Tensor,
    conv_pool: Tensor,
    conv_slots: Tensor,
    raw_gate: Tensor,
    beta_logits: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    recurrent_pool: Tensor,
    recurrent_rows: Tensor,
    ring_pool: Tensor,
    ring_rows: Tensor,
    input_row_offsets: Tensor,
    layer: Tensor,
    verify_width: Tensor,
    head_shape: list[int],
    gate_mode: KdaGateMode,
    capture: CaptureSink[KdaReplayInputs] | None,
    position: int | None,
) -> Tensor:
    """Runs a verify's conv and recurrence on the K-group conv op.

    The op runs the conv in the ring kernel, so a verify with drafts takes
    one launch for both. A draft-free prefill takes the chunk kernel, whose
    bf16 q, k and v come out of one ``kda_chunk_conv`` launch. Both ops take
    ``qkv`` in the model dtype and widen it themselves, which is exact, so
    the layer casts none of it.
    """
    operands = _kda_operands(
        raw_gate,
        beta_logits,
        a_log,
        dt_bias,
        input_row_offsets,
        recurrent_rows,
        layer,
        position,
        head_shape,
    )
    raw_gate, beta_logits, a_log, dt_bias, cu_seqlens, state_indices = operands
    conv_offsets = input_row_offsets.cast(DType.uint32)
    # Sliced outside the cond, as `kda_core` slices it.
    ring_indices = layer_rows(ring_rows, layer, position)
    out_shape: list[DimLike] = [qkv.shape[0], *head_shape[1:]]

    def ring() -> list[Tensor]:
        out, window = kda_verify_ring_kgroup_conv(
            qkv,
            conv_weight,
            conv_pool,
            conv_slots,
            raw_gate,
            beta_logits,
            a_log,
            dt_bias,
            cu_seqlens,
            recurrent_pool,
            state_indices,
            ring_pool,
            ring_indices,
            verify_width,
        )
        return [F.rebind(out, out_shape, "kda verify output rebind"), window]

    def chunk() -> list[Tensor]:
        q, k, v, window = kda_chunk_conv(
            qkv,
            conv_weight,
            conv_pool,
            conv_slots,
            conv_offsets,
            verify_width,
            num_key_heads=head_shape[1],
            num_value_heads=head_shape[1],
            head_dim=head_shape[2],
        )
        core = _chunk_core(
            q, k, v, operands, recurrent_pool, gate_mode, out_shape
        )
        return [core, window]

    if kda_chunk_servable(head_shape[2], gate_mode, recurrent_pool.dtype):
        # As in `kda_core`: a draft-free prefill takes the chunk kernel.
        no_drafts = F.equal(F.shape_to_tensor(verify_width.shape)[0], 0)
        total_tokens = F.shape_to_tensor(qkv.shape)[0]
        batch_size = F.shape_to_tensor(input_row_offsets.shape)[0] - 1
        device = DeviceRef.from_device(qkv.device)
        window_rows = Dim(conv_slots.shape[0]) * (
            Dim(verify_width.shape[0]) + 1
        )
        core, window = F.cond(
            F.logical_and(
                no_drafts,
                _chunk_predicate(
                    total_tokens, batch_size, head_shape[1], head_shape[2]
                ),
            ),
            [
                TensorType(DType.float32, out_shape, device),
                TensorType(DType.float32, [window_rows, qkv.shape[1]], device),
            ],
            chunk,
            ring,
        )
    else:
        core, window = ring()
    _capture_replay(capture, qkv, conv_weight, operands, window)
    return core


def kda_core(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_conv: Tensor,
    k_conv: Tensor,
    v_conv: Tensor,
    conv_pool: Tensor,
    conv_rows: Tensor,
    raw_gate: Tensor,
    beta_logits: Tensor,
    output_gate: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    norm_weight: Tensor,
    recurrent_pool: Tensor,
    recurrent_rows: Tensor,
    input_row_offsets: Tensor,
    layer: Tensor,
    head_dim: int,
    norm_eps: float,
    gate_mode: KdaGateMode,
    ring_pool: Tensor | None = None,
    ring_rows: Tensor | None = None,
    verify_width: Tensor | None = None,
    capture: CaptureSink[KdaReplayInputs] | None = None,
    pin_decode_fma: bool = False,
    position: int | None = None,
) -> Tensor:
    """Runs one device's heads: conv, recurrence, gated norm.

    Every tensor is one device's, so the recurrence reads that device's own
    pool rows and heads. Returns ``[total_tokens, local_heads * head_dim]`` in
    ``q``'s dtype, the input of the row-parallel ``o_proj``. Both pools are
    mutated in place, except under a verify (``verify_width`` set), which
    records the recurrence into ``ring_pool`` and appends its replay inputs
    to ``capture`` for the rollback. ``pin_decode_fma`` pins the decode
    kernel's fused multiply-adds, which only a test of decode against the
    verify ring needs: a speculative graph never takes the decode branch.
    """
    dtype = q.dtype
    qkv_dim = int(q.shape[-1])
    head_shape = [-1, qkv_dim // head_dim, head_dim]

    # q, k, v in that order: the conv's channel layout.
    qkv = F.concat([q, k, v], axis=-1)
    # The module holds each conv weight as the checkpoint's depthwise
    # `[channels, 1, kernel]`.
    conv_weight = (
        F.concat([q_conv, k_conv, v_conv], axis=0)
        .reshape([-1, q_conv.shape[-1]])
        .cast(DType.float32)
    )
    conv_slots = layer_rows(conv_rows, layer, position).cast(DType.uint32)
    if verify_width is not None and kda_verify_ring_kgroup_conv_servable(
        head_dim, gate_mode, recurrent_pool.dtype
    ):
        assert ring_pool is not None and ring_rows is not None
        core = _kgroup_conv_verify(
            qkv,
            conv_weight,
            conv_pool,
            conv_slots,
            raw_gate,
            beta_logits,
            a_log,
            dt_bias,
            recurrent_pool,
            recurrent_rows,
            ring_pool,
            ring_rows,
            input_row_offsets,
            layer,
            verify_width,
            head_shape,
            gate_mode,
            capture,
            position,
        )
        return _gated_norm(
            core, norm_weight, norm_eps, output_gate, head_shape, dtype
        )
    qkv = qkv.cast(DType.float32)
    qkv_window: Tensor | None = None
    if verify_width is None:
        conv_out = gated_delta_conv1d_fwd(
            qkv_input_ragged=qkv,
            conv_weight=conv_weight,
            conv_state=conv_pool,
            slot_idx=conv_slots,
            input_row_offsets=input_row_offsets.cast(DType.uint32),
            silu=True,
        )
    else:
        conv_out, qkv_window = gated_delta_conv1d_verify_fwd(
            qkv_input_ragged=qkv,
            conv_weight=conv_weight,
            conv_state=conv_pool,
            slot_idx=conv_slots,
            input_row_offsets=input_row_offsets.cast(DType.uint32),
            verify_width=verify_width,
            rollback=False,
            silu=True,
        )
    q, k, v = _split_heads(conv_out, head_shape)

    operands = _kda_operands(
        raw_gate,
        beta_logits,
        a_log,
        dt_bias,
        input_row_offsets,
        recurrent_rows,
        layer,
        position,
        head_shape,
    )
    raw_gate, beta_logits, a_log, dt_bias, cu_seqlens, state_indices = operands
    out_shape: list[DimLike] = [q.shape[0], *head_shape[1:]]
    _capture_replay(capture, qkv, conv_weight, operands, qkv_window)

    def decode() -> Tensor:
        return kda_decode(
            q,
            k,
            v,
            raw_gate,
            beta_logits,
            a_log,
            dt_bias,
            cu_seqlens,
            recurrent_pool,
            state_indices,
            output_dtype=DType.float32,
            gate_mode=gate_mode,
            beta_mode="logits",
            state_layout=STATE_LAYOUT,
            pin_fma=pin_decode_fma,
        )

    def chunk(qkv_io: tuple[Tensor, Tensor, Tensor] | None = None) -> Tensor:
        q_io, k_io, v_io = qkv_io or (
            q.cast(DType.bfloat16),
            k.cast(DType.bfloat16),
            v.cast(DType.bfloat16),
        )
        return _chunk_core(
            q_io, k_io, v_io, operands, recurrent_pool, gate_mode, out_shape
        )

    # Sliced here rather than in the branch: mo-symbolic-optimize merges the
    # layers' slices of one row table into a single split, which must not sit
    # in one layer's branch.
    ring_indices = (
        None if ring_rows is None else layer_rows(ring_rows, layer, position)
    )

    def verify_ring() -> Tensor:
        assert verify_width is not None
        assert ring_pool is not None and ring_indices is not None
        out = kda_verify_ring(
            q,
            k,
            v,
            raw_gate,
            beta_logits,
            a_log,
            dt_bias,
            cu_seqlens,
            recurrent_pool,
            state_indices,
            ring_pool,
            ring_indices,
            verify_width,
            output_dtype=DType.float32,
            gate_mode=gate_mode,
            beta_mode="logits",
            state_layout=STATE_LAYOUT,
            kgroup=True,
        )
        return F.rebind(out, out_shape, "kda verify output rebind")

    # Prefill takes the chunk op where it is built: SM100+ and gfx950, where
    # `test_kda_chunk_prefill.py` grades it against `kda_decode`. A decode
    # batch (one token per sequence) stays on float32 `kda_decode`, and so
    # does the device graph captured for it: `F.cond` branches are separate
    # regions, which also keeps NVIDIA `kda_chunk`'s host readback of
    # `cu_seqlens` out of it. A mixed prefill+decode batch takes the chunk
    # branch whole, decode rows included.
    chunked = kda_chunk_servable(head_dim, gate_mode, recurrent_pool.dtype)
    total_tokens = F.shape_to_tensor(q.shape)[0]
    batch_size = F.shape_to_tensor(input_row_offsets.shape)[0] - 1
    out_types = [
        TensorType(DType.float32, out_shape, DeviceRef.from_device(q.device))
    ]
    if verify_width is None and chunked:
        (core,) = F.cond(
            _chunk_predicate(total_tokens, batch_size, head_shape[1], head_dim),
            out_types,
            lambda: [chunk()],
            lambda: [F.rebind(decode(), out_shape, "kda decode output rebind")],
        )
    elif verify_width is None:
        core = decode()
    elif chunked:
        # A draft-free prefill has nothing to roll back, so it takes the
        # forward's chunked kernel. A verify stays on the sequential
        # recurrence, whose arithmetic the ring records and the fold replays.
        no_drafts = F.equal(F.shape_to_tensor(verify_width.shape)[0], 0)
        # TODO(KERN-3993): every verify computes this bf16 cast, and only the
        # chunk branch reads it. The cast alone cannot move into that branch:
        # mo-hoist-param-exprs then leaves its result on the outer token dim
        # and the graph fails to verify (KERN-3879). Make the conv inside each
        # branch too, as `_kgroup_conv_verify` does. One cast of the whole
        # conv output rides in the q/k/v split's kernel; casting q, k and v
        # apart launches two more kernels per layer.
        conv_io = conv_out.cast(DType.bfloat16)
        (core,) = F.cond(
            F.logical_and(
                no_drafts,
                _chunk_predicate(
                    total_tokens, batch_size, head_shape[1], head_dim
                ),
            ),
            out_types,
            lambda: [chunk(_split_heads(conv_io, head_shape))],
            lambda: [verify_ring()],
        )
    else:
        core = verify_ring()
    return _gated_norm(
        core, norm_weight, norm_eps, output_gate, head_shape, dtype
    )


class KimiDeltaAttention(Module[[Tensor, KdaStateAccess, Tensor], Tensor]):
    """One KDA layer, split by head across the tensor-parallel axis.

    Returns ``o_proj``'s partial sums; the decoder layer reduces them.
    """

    def __init__(self, config: Glm5NextConfig) -> None:
        hidden = config.hidden_size
        self.num_heads = config.linear_num_heads
        self.head_dim = config.linear_head_dim
        self.norm_eps = config.rms_norm_eps
        self.gate_mode = kda_gate_mode(config.linear_lower_bound)
        qkv_dim = self.num_heads * self.head_dim
        kernel = config.linear_conv_kernel_dim

        def head_rows(in_dim: int, out_dim: int) -> Linear:
            return col_parallel(Linear(in_dim, out_dim, bias=False))

        self.q_proj = head_rows(hidden, qkv_dim)
        self.k_proj = head_rows(hidden, qkv_dim)
        self.v_proj = head_rows(hidden, qkv_dim)
        self.q_conv1d = _shard_rows(Tensor.zeros([qkv_dim, 1, kernel]))
        self.k_conv1d = _shard_rows(Tensor.zeros([qkv_dim, 1, kernel]))
        self.v_conv1d = _shard_rows(Tensor.zeros([qkv_dim, 1, kernel]))

        # The low-rank gate projections bottom out at `head_dim`, which is not
        # a head axis, so their down halves are replicated.
        self.f_a_proj = Linear(hidden, self.head_dim, bias=False)
        self.f_b_proj = head_rows(self.head_dim, qkv_dim)
        self.g_a_proj = Linear(hidden, self.head_dim, bias=False)
        self.g_b_proj = head_rows(self.head_dim, qkv_dim)
        self.b_proj = head_rows(hidden, self.num_heads)

        self.dt_bias = _shard_rows(Tensor.zeros([qkv_dim], dtype=DType.float32))
        self.A_log = _shard_rows(
            Tensor.zeros([self.num_heads], dtype=DType.float32)
        )
        # The reference's gated norm multiplies by the weight, not 1 + weight,
        # and holds everything in float32. It runs inside the per-device core
        # below, which reads only its weight and widens it to float32.
        self.o_norm = RMSNorm(self.head_dim, eps=self.norm_eps)
        self.o_proj = row_parallel(Linear(qkv_dim, hidden, bias=False))
        self.verify: KdaVerify | None = None
        """Set by the speculative target around its verify, else ``None``."""
        self.pin_decode_fma = False
        """Whether decode pins its fused multiply-adds so its state matches a
        verify ring's bit for bit; the pinned kernel is slower on AMD."""

    def forward(
        self, x: Tensor, state: KdaStateAccess, input_row_offsets: Tensor
    ) -> Tensor:
        """Runs the delta attention over ``x`` and advances ``state``."""
        head_split = NamedMapping(x.mesh, (None, TP))
        verify = self.verify
        core = functools.partial(
            kda_core,
            head_dim=self.head_dim,
            norm_eps=self.norm_eps,
            gate_mode=self.gate_mode,
            pin_decode_fma=self.pin_decode_fma,
            position=state.position,
        )
        normed = F.call_on_mesh(core, x.mesh, out_specs=head_split)(
            self.q_proj(x),
            self.k_proj(x),
            self.v_proj(x),
            self.q_conv1d,
            self.k_conv1d,
            self.v_conv1d,
            state.conv_pool,
            state.conv_rows,
            self.f_b_proj(self.f_a_proj(x)),
            self.b_proj(x),
            self.g_b_proj(self.g_a_proj(x)),
            self.A_log,
            self.dt_bias,
            self.o_norm.weight,
            state.recurrent_pool,
            state.recurrent_rows,
            input_row_offsets,
            state.layer,
            ring_pool=None if verify is None else verify.ring_pool,
            ring_rows=None if verify is None else verify.ring_rows,
            verify_width=None if verify is None else verify.verify_width,
            capture=(
                None
                if verify is None
                else PerShard(CaptureSink(c) for c in verify.captures)
            ),
        )
        return self.o_proj(normed)


def _shard_rows(t: Tensor) -> Tensor:
    spec: list[str | None] = [None] * t.rank
    spec[0] = TP
    return t.to(NamedMapping(t.mesh, tuple(spec)))
