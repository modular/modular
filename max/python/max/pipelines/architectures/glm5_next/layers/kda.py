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
from typing import Generic, TypeVar

from max import functional as F
from max import tree
from max.driver import CPU, accelerator_api
from max.dtype import DType
from max.graph import DeviceRef, Dim, DimLike, TensorType
from max.nn import Linear, Module, RMSNorm
from max.nn._graph_ops.state_space.kimi_delta import (
    STATE_LAYOUT,
    KdaGateMode,
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
    kda_decode,
    kda_verify_ring,
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


def _qkv_projection(
    x: Tensor, q_weight: Tensor, k_weight: Tensor, v_weight: Tensor
) -> Tensor:
    """Projects one device's q, k and v heads with one matmul.

    Returns ``[total_tokens, 3 * local_heads * head_dim]`` in the conv's
    channel order. Each weight holds only this device's head rows, so the
    stack keeps the per-rank order.
    """
    weight = F.concat([q_weight, k_weight, v_weight])
    return x.cast(weight.dtype) @ weight.T


def kda_core(
    qkv: Tensor,
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
    ``qkv``'s dtype, the input of the row-parallel ``o_proj``. Both pools are
    mutated in place, except under a verify (``verify_width`` set), which
    records the recurrence into ``ring_pool`` and appends its replay inputs
    to ``capture`` for the rollback. ``pin_decode_fma`` pins the decode
    kernel's fused multiply-adds, which only a test of decode against the
    verify ring needs: a speculative graph never takes the decode branch.
    """
    dtype = qkv.dtype
    qkv_dim = int(qkv.shape[-1]) // 3
    head_shape = [-1, qkv_dim // head_dim, head_dim]

    qkv = qkv.cast(DType.float32)
    # The module holds each conv weight as the checkpoint's depthwise
    # `[channels, 1, kernel]`.
    conv_weight = (
        F.concat([q_conv, k_conv, v_conv], axis=0)
        .reshape([-1, q_conv.shape[-1]])
        .cast(DType.float32)
    )
    conv_slots = layer_rows(conv_rows, layer, position).cast(DType.uint32)
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
    q, k, v = (
        part.reshape(head_shape)
        for part in conv_out.split([qkv_dim] * 3, axis=-1)
    )

    raw_gate = raw_gate.cast(DType.float32).reshape(head_shape)
    beta_logits = beta_logits.cast(DType.float32)
    # Read here, outside the conditional below: a weight first read inside
    # one branch is defined in that branch's region, and the other branch
    # cannot use it.
    a_log = a_log.cast(DType.float32)
    dt_bias = dt_bias.reshape(head_shape[1:])
    cu_seqlens = input_row_offsets.cast(DType.int32)
    state_indices = layer_rows(recurrent_rows, layer, position).cast(
        DType.int32
    )
    out_shape: list[DimLike] = [q.shape[0], *head_shape[1:]]
    if capture is not None:
        capture.items.append(
            KdaReplayInputs(
                qkv=qkv,
                conv_weight=conv_weight,
                raw_gate=raw_gate,
                beta_logits=beta_logits,
                qkv_window=qkv_window,
            )
        )

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

    # NVIDIA's chunk kernels take float32 q/k/v and output, as decode does.
    # The gfx950 ones take only bf16, so that branch rounds at the op
    # boundary and nowhere else; its state stays float32 because eligibility
    # requires a float32 pool.
    chunk_dtype = (
        DType.bfloat16 if accelerator_api() == "hip" else DType.float32
    )

    def chunk(qkv_io: tuple[Tensor, Tensor, Tensor] | None = None) -> Tensor:
        q_io, k_io, v_io = qkv_io or (
            q.cast(chunk_dtype),
            k.cast(chunk_dtype),
            v.cast(chunk_dtype),
        )
        out = kda_chunk(
            q_io,
            k_io,
            v_io,
            raw_gate,
            beta_logits,
            a_log,
            dt_bias,
            cu_seqlens,
            recurrent_pool,
            state_indices,
            output_dtype=chunk_dtype,
            gate_mode=gate_mode,
            beta_mode="logits",
            state_layout=STATE_LAYOUT,
        ).cast(DType.float32)
        return F.rebind(out, out_shape, "kda chunk output rebind")

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
        )
        return F.rebind(out, out_shape, "kda verify output rebind")

    # Prefill takes the chunk op where it is built: SM100+ and gfx950, where
    # `test_kda_chunk_prefill.py` grades it against `kda_decode`. A decode
    # batch (one token per sequence) stays on float32 `kda_decode`, and so
    # does the device graph captured for it: `F.cond` branches are separate
    # regions. A mixed prefill+decode batch takes the chunk branch whole,
    # decode rows included.
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
    elif chunked and accelerator_api() == "hip":
        # A draft-free prefill has nothing to roll back, so it takes the
        # forward's chunked kernel. A verify stays on the sequential
        # recurrence, whose arithmetic the ring records and the fold replays.
        # TODO(MAX-907): take kda_chunk here on NVIDIA too. In the fused MTP
        # graph on 2xB200 its prefill moves the first token's top logit ~15
        # bf16 ulps from the plain graph's, which the forward's does not.
        no_drafts = F.equal(F.shape_to_tensor(verify_width.shape)[0], 0)
        # TODO(KERN-3879): the gfx950 bf16 cast is made outside the cond
        # because, made inside a branch, mo-hoist-param-exprs leaves its
        # result on the outer token dim and the graph fails to verify. Every
        # verify computes and drops it. One cast of the whole conv output
        # rides in the q/k/v split's kernel, where casting q, k and v apart
        # launches two more kernels per layer. Move it into the branch once
        # the pass is fixed.
        conv_io = conv_out.cast(chunk_dtype)

        def qkv_io() -> tuple[Tensor, Tensor, Tensor]:
            q_io, k_io, v_io = (
                part.reshape(head_shape)
                for part in conv_io.split([qkv_dim] * 3, axis=-1)
            )
            return q_io, k_io, v_io

        (core,) = F.cond(
            F.logical_and(
                no_drafts,
                _chunk_predicate(
                    total_tokens, batch_size, head_shape[1], head_dim
                ),
            ),
            out_types,
            lambda: [chunk(qkv_io())],
            lambda: [verify_ring()],
        )
    else:
        core = verify_ring()
    # Normalize, scale, and only then gate.
    normed = F.rms_norm(
        core,
        norm_weight.cast(DType.float32),
        norm_eps,
        multiply_before_cast=False,
    ) * F.sigmoid(output_gate.cast(DType.float32).reshape(head_shape))
    return normed.cast(dtype).reshape([-1, qkv_dim])


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
        qkv = F.call_on_mesh(_qkv_projection, x.mesh, out_specs=head_split)(
            x, self.q_proj.weight, self.k_proj.weight, self.v_proj.weight
        )
        normed = F.call_on_mesh(core, x.mesh, out_specs=head_split)(
            qkv,
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
