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

The graph-API layer
(:mod:`max.pipelines.architectures.glm5_next.layers.kimi_delta_attention`)
documents the math and the three orderings that are wrong silently. This module
keeps all three: the conv channels are q, k, v in that order, the gated norm
normalizes before it gates, and tensor parallelism splits heads *within* each
of q, k and v -- which is why the three conv weights stay separate tensors here
and are concatenated per device, not sharded as one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from max import functional as F
from max import tree
from max.driver import accelerator_api
from max.dtype import DType
from max.graph import BufferValue, DimLike, TensorType, TensorValue, ops
from max.nn import Linear, Module, RMSNorm
from max.nn._graph_ops.state_space import (
    gated_delta_conv1d_fwd,
    gated_delta_conv1d_verify_fwd,
    kda_chunk,
    kda_decode,
)
from max.nn._graph_ops.state_space.kimi_delta import (
    STATE_LAYOUT,
    KdaGateMode,
    kda_chunk_predicate,
    kda_chunk_servable,
    kda_gate_mode,
    kda_verify_ring,
)
from max.nn.common_layers.linear import col_parallel, row_parallel
from max.nn.common_layers.mesh_axis import TP
from max.pipelines.architectures.glm5_next.layers.kda import KdaReplayInputs
from max.tensor import Tensor
from max.tensor.sharding import NamedMapping
from max.tensor.sharding.action import PerShard

from ..model_config import Glm5NextConfig

__all__ = ["CaptureSink", "KdaStateAccess", "KdaVerify", "KimiDeltaAttention"]


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
    verify_width: TensorValue
    """The speculative graph's
    :func:`~max.nn._graph_ops.state_space.verify_width_operand`."""
    captures: list[list[KdaReplayInputs]]
    """Per-device sink every KDA layer appends its replay inputs to, in
    schedule order."""


def _layer_rows(rows: TensorValue, layer: TensorValue) -> TensorValue:
    """Row ``layer`` of a ``[num_layers, batch_size]`` table, as ``[batch]``."""
    return ops.squeeze(rows[(slice(layer, layer + 1), 1), :], 0)


def _kda_core_graph(
    q: TensorValue,
    k: TensorValue,
    v: TensorValue,
    q_conv: TensorValue,
    k_conv: TensorValue,
    v_conv: TensorValue,
    conv_pool: BufferValue,
    conv_rows: TensorValue,
    raw_gate: TensorValue,
    beta_logits: TensorValue,
    output_gate: TensorValue,
    a_log: TensorValue,
    dt_bias: TensorValue,
    norm_weight: TensorValue,
    recurrent_pool: BufferValue,
    recurrent_rows: TensorValue,
    input_row_offsets: TensorValue,
    layer: TensorValue,
    head_dim: int,
    norm_eps: float,
    gate_mode: KdaGateMode,
    ring_pool: BufferValue | None = None,
    ring_rows: TensorValue | None = None,
    verify_width: TensorValue | None = None,
    capture: CaptureSink[KdaReplayInputs] | None = None,
    pin_decode_fma: bool = False,
) -> TensorValue:
    """Runs one device's heads: conv, recurrence, gated norm.

    Returns ``[total_tokens, local_heads * head_dim]`` in ``q``'s dtype, the
    input of the row-parallel ``o_proj``. Both pools are mutated in place,
    except under a verify (``verify_width`` set), which records the
    recurrence into ``ring_pool`` and appends its replay inputs to
    ``capture`` for the rollback. ``pin_decode_fma`` pins the decode
    kernel's fused multiply-adds, which only a test of decode against the
    verify ring needs: a speculative graph never takes the decode branch.
    """
    dtype = q.dtype
    qkv_dim = q.shape[-1]
    head_shape = [-1, int(qkv_dim) // head_dim, head_dim]

    # q, k, v in that order: the conv's channel layout. The module holds each
    # conv weight as the checkpoint's depthwise `[channels, 1, kernel]`.
    qkv = ops.concat([q, k, v], axis=-1).cast(DType.float32)
    conv_weight = ops.reshape(
        ops.concat([q_conv, k_conv, v_conv], axis=0), [-1, q_conv.shape[-1]]
    ).cast(DType.float32)
    conv_slots = _layer_rows(conv_rows, layer).cast(DType.uint32)
    qkv_window: TensorValue | None = None
    if verify_width is None:
        conv_out = gated_delta_conv1d_fwd(
            qkv_input_ragged=qkv,
            conv_weight=conv_weight,
            conv_state=conv_pool,
            slot_idx=conv_slots,
            input_row_offsets=input_row_offsets.cast(DType.uint32),
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
        )
    conv_out = ops.silu(conv_out)
    q, k, v = (
        ops.reshape(part, head_shape)
        for part in ops.split(conv_out, [qkv_dim] * 3, axis=-1)
    )

    raw_gate = ops.reshape(raw_gate.cast(DType.float32), head_shape)
    beta_logits = beta_logits.cast(DType.float32)
    dt_bias = ops.reshape(dt_bias, head_shape[1:])
    cu_seqlens = input_row_offsets.cast(DType.int32)
    state_indices = _layer_rows(recurrent_rows, layer).cast(DType.int32)
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

    def decode() -> TensorValue:
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

    def chunk(
        qkv_io: tuple[TensorValue, TensorValue, TensorValue] | None = None,
    ) -> TensorValue:
        # The gfx950 chunk kernels take only bf16 q/k/v and output, so this
        # branch rounds at the op boundary and nowhere else; the state stays
        # float32 because eligibility requires a float32 pool.
        q_io, k_io, v_io = qkv_io or (
            q.cast(DType.bfloat16),
            k.cast(DType.bfloat16),
            v.cast(DType.bfloat16),
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
            output_dtype=DType.bfloat16,
            gate_mode=gate_mode,
            beta_mode="logits",
            state_layout=STATE_LAYOUT,
        ).cast(DType.float32)
        return ops.rebind(out, out_shape, "kda chunk output rebind")

    # Sliced here rather than in the branch: mo-symbolic-optimize merges the
    # layers' slices of one row table into a single split, which must not sit
    # in one layer's branch.
    ring_indices = None if ring_rows is None else _layer_rows(ring_rows, layer)

    def verify_ring() -> TensorValue:
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
        return ops.rebind(out, out_shape, "kda verify output rebind")

    # Only gfx950 takes the chunk op here: its NVIDIA chunk path has no
    # numerics coverage yet, so NVIDIA prefill keeps the sequential
    # recurrence. A decode batch (one token per sequence) stays on float32
    # `kda_decode`, and so does the device graph captured for it: `ops.cond`
    # branches are separate regions. A mixed prefill+decode batch takes the
    # chunk branch whole, decode rows included.
    chunked = accelerator_api() == "hip" and kda_chunk_servable(
        head_dim, gate_mode, recurrent_pool.dtype
    )
    total_tokens = ops.shape_to_tensor(q.shape)[0]
    batch_size = ops.shape_to_tensor(input_row_offsets.shape)[0] - 1
    if verify_width is None and chunked:
        core = ops.cond(
            kda_chunk_predicate(
                total_tokens, batch_size, head_shape[1], head_dim
            ),
            [TensorType(DType.float32, out_shape, q.device)],
            chunk,
            lambda: ops.rebind(decode(), out_shape, "kda decode output rebind"),
        )[0]
    elif verify_width is None:
        core = decode()
    elif chunked:
        # A draft-free prefill has nothing to roll back, so it takes the
        # forward's chunked kernel. A verify stays on the sequential
        # recurrence, whose arithmetic the ring records and the fold replays.
        no_drafts = ops.equal(ops.shape_to_tensor(verify_width.shape)[0], 0)
        # TODO(KERN-3879): the bf16 casts are made outside the cond because,
        # made inside this branch, mo-hoist-param-exprs leaves the first
        # cast's result on the outer token dim and the graph fails to verify.
        # Every verify computes them and drops them; a 2xMI355 trace shows no
        # standalone cast kernel, so they cost at most ~0.5% of kernel time.
        # Move them back once the pass is fixed.
        qkv_io = (
            q.cast(DType.bfloat16),
            k.cast(DType.bfloat16),
            v.cast(DType.bfloat16),
        )
        core = ops.cond(
            ops.logical_and(
                no_drafts,
                kda_chunk_predicate(
                    total_tokens, batch_size, head_shape[1], head_dim
                ),
            ),
            [TensorType(DType.float32, out_shape, q.device)],
            lambda: chunk(qkv_io),
            verify_ring,
        )[0]
    else:
        core = verify_ring()
    # Normalize, scale, and only then gate.
    normed = ops.rms_norm(
        core,
        norm_weight.cast(DType.float32),
        norm_eps,
        multiply_before_cast=False,
    ) * ops.sigmoid(ops.reshape(output_gate.cast(DType.float32), head_shape))
    return ops.reshape(normed.cast(dtype), [-1, qkv_dim])


_kda_core = F.functional(_kda_core_graph)


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
        normed = _kda_core(
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
            self.head_dim,
            self.norm_eps,
            self.gate_mode,
            ring_pool=None if verify is None else verify.ring_pool,
            ring_rows=None if verify is None else verify.ring_rows,
            verify_width=None if verify is None else verify.verify_width,
            capture=(
                None
                if verify is None
                else PerShard(CaptureSink(c) for c in verify.captures)
            ),
            pin_decode_fma=self.pin_decode_fma,
        ).rebind_mapping(head_split)
        return self.o_proj(normed)


def _shard_rows(t: Tensor) -> Tensor:
    spec: list[str | None] = [None] * t.rank
    spec[0] = TP
    return t.to(NamedMapping(t.mesh, tuple(spec)))
