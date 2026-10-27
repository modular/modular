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

from max import functional as F
from max import tree
from max.driver import accelerator_api
from max.dtype import DType
from max.graph import BufferValue, DimLike, TensorType, TensorValue, ops
from max.nn import Linear, Module, RMSNorm
from max.nn.common_layers.linear import col_parallel, row_parallel
from max.nn.common_layers.mesh_axis import TP
from max.nn.legacy.state_space import (
    gated_delta_conv1d_fwd,
    kda_chunk,
    kda_decode,
)
from max.nn.legacy.state_space.kimi_delta import KdaGateMode
from max.pipelines.architectures.glm5_next.layers.kimi_delta_attention import (
    STATE_LAYOUT,
    kda_chunk_predicate,
    kda_chunk_servable,
    kda_gate_mode,
)
from max.tensor import Tensor
from max.tensor.sharding import NamedMapping

from ..model_config import Glm5NextConfig

__all__ = ["KdaStateAccess", "KimiDeltaAttention"]


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
) -> TensorValue:
    """Runs one device's heads: conv, recurrence, gated norm.

    Returns ``[total_tokens, local_heads * head_dim]`` in ``q``'s dtype, the
    input of the row-parallel ``o_proj``. Both pools are mutated in place.
    """
    dtype = q.dtype
    qkv_dim = q.shape[-1]
    head_shape = [-1, int(qkv_dim) // head_dim, head_dim]

    # q, k, v in that order: the conv's channel layout.
    qkv = ops.concat([q, k, v], axis=-1).cast(DType.float32)
    conv_weight = ops.concat([q_conv, k_conv, v_conv], axis=0).cast(
        DType.float32
    )
    conv_out = ops.silu(
        gated_delta_conv1d_fwd(
            qkv_input_ragged=qkv,
            conv_weight=conv_weight,
            conv_state=conv_pool,
            slot_idx=_layer_rows(conv_rows, layer).cast(DType.uint32),
            input_row_offsets=input_row_offsets.cast(DType.uint32),
        )
    )
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
        )

    def chunk() -> TensorValue:
        # The gfx950 chunk kernels take only bf16 q/k/v and output, so this
        # branch rounds at the op boundary and nowhere else; the state stays
        # float32 because eligibility requires a float32 pool.
        out = kda_chunk(
            q.cast(DType.bfloat16),
            k.cast(DType.bfloat16),
            v.cast(DType.bfloat16),
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

    # Only gfx950 takes the chunk op here: this graph has no speculative
    # verify or state replay, but its NVIDIA chunk path has no numerics
    # coverage yet, so NVIDIA prefill keeps the sequential recurrence. A
    # decode batch (one token per sequence) stays on float32 `kda_decode`, and
    # so does the device graph captured for it: `ops.cond` branches are
    # separate regions. A mixed prefill+decode batch takes the chunk branch
    # whole, decode rows included.
    if accelerator_api() == "hip" and kda_chunk_servable(
        head_dim, gate_mode, recurrent_pool.dtype
    ):
        total_tokens = ops.shape_to_tensor(q.shape)[0]
        batch_size = ops.shape_to_tensor(input_row_offsets.shape)[0] - 1
        core = ops.cond(
            kda_chunk_predicate(
                total_tokens, batch_size, head_shape[1], head_dim
            ),
            [TensorType(DType.float32, out_shape, q.device)],
            chunk,
            lambda: ops.rebind(decode(), out_shape, "kda decode output rebind"),
        )[0]
    else:
        core = decode()
    # Normalize, scale, and only then gate.
    normed = ops.rms_norm(
        core, norm_weight, norm_eps, multiply_before_cast=False
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
        self.q_conv1d = _shard_rows(Tensor.zeros([qkv_dim, kernel]))
        self.k_conv1d = _shard_rows(Tensor.zeros([qkv_dim, kernel]))
        self.v_conv1d = _shard_rows(Tensor.zeros([qkv_dim, kernel]))

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
        # below, which reads only its weight.
        self.o_norm = RMSNorm(self.head_dim, eps=self.norm_eps)
        self.o_norm.weight = Tensor.ones([self.head_dim], dtype=DType.float32)
        self.o_proj = row_parallel(Linear(qkv_dim, hidden, bias=False))

    def forward(
        self, x: Tensor, state: KdaStateAccess, input_row_offsets: Tensor
    ) -> Tensor:
        head_split = NamedMapping(x.mesh, (None, TP))
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
        ).rebind_mapping(head_split)
        return self.o_proj(normed)


def _shard_rows(t: Tensor) -> Tensor:
    spec: list[str | None] = [None] * t.rank
    spec[0] = TP
    return t.to(NamedMapping(t.mesh, tuple(spec)))
