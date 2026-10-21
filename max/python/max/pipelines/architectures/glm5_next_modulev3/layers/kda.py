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

from max import tree
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Linear, Module, RMSNorm
from max.experimental.nn.common_layers.linear import col_parallel, row_parallel
from max.experimental.nn.common_layers.mesh_axis import TP
from max.experimental.sharding import NamedMapping
from max.experimental.tensor import Tensor
from max.graph import BufferValue, TensorValue, ops
from max.nn.state_space import gated_delta_conv1d_fwd, kda_decode
from max.nn.state_space.kimi_delta import KDA_GATE_LOWER_BOUND, KdaGateMode
from max.pipelines.architectures.glm5_next.layers.kimi_delta_attention import (
    STATE_LAYOUT,
)

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


def _gate_mode(lower_bound: float) -> KdaGateMode:
    """The kernel's bounded forget gate, which compiles in its lower bound."""
    if lower_bound != KDA_GATE_LOWER_BOUND:
        raise ValueError(
            f"KDA's bounded forget gate is compiled with a lower bound of "
            f"{KDA_GATE_LOWER_BOUND}, but this checkpoint asks for "
            f"{lower_bound}."
        )
    return "safe"


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

    core = kda_decode(
        q,
        k,
        v,
        ops.reshape(raw_gate.cast(DType.float32), head_shape),
        beta_logits.cast(DType.float32),
        a_log,
        ops.reshape(dt_bias, head_shape[1:]),
        input_row_offsets.cast(DType.int32),
        recurrent_pool,
        _layer_rows(recurrent_rows, layer).cast(DType.int32),
        output_dtype=DType.float32,
        gate_mode=gate_mode,
        beta_mode="logits",
        state_layout=STATE_LAYOUT,
    )
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
        self.gate_mode = _gate_mode(config.linear_lower_bound)
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
