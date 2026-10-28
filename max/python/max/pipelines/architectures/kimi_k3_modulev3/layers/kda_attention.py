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
"""Kimi K3's Kimi Delta Attention (ModuleV3), ``kda_decode``-only form.

Direct V3 port of ``kimi_k3/layers/kda_attention.py``'s
``kimi_k3_kda_attention`` orchestration, calling the same ``kda_decode``
custom op via :func:`max.functional.inplace_custom`.

Scope cut, explicit: this ports ``kda_decode`` only, not ``kda_chunk`` (the
prefill-optimized arm selected by ``ops.cond`` in V2) or the speculative-
decoding ``replay_capture`` machinery. V2's own comment states
``kda_decode`` alone is CORRECT for both decode and prefill batches -- just
slower on long prefills (~6.8x at K3's production head count) -- so this is
a performance cut, not a correctness one. On this hardware it isn't even a
cut relative to V2: ``_chunk_prefill_available()`` returns ``False`` on HIP
(no TMA on CDNA), so V2 already falls back to ``kda_decode``-only here too.

Both gate-lower-bound arms V2 supports are kept (not scoped out): the
"original" arm (``gate_lower_bound is None``, the kernel's own per-head
``-exp(A_log) * softplus(pre)``) and the "safe" arm (pre-activated here,
``log_alpha = bound * sigmoid(exp(A_log) * (raw_gate + dt_bias))`` folded
into the kernel's ``A=1`` path). Published K3 sets the bound to -5.0, and
both kernels hardcode -5.0 in their "safe" arm, so any other bound raises
rather than silently running the wrong gate.
"""

from __future__ import annotations

from typing import Any, cast

from max import functional as F
from max.dtype import DType
from max.graph import BufferValue, TensorType, TensorValue, ops
from max.nn import Linear, Module
from max.nn.common_layers.mesh_axis import TP
from max.tensor import Tensor
from max.tensor.sharding import DeviceMesh, NamedMapping

from .norm import KimiRMSNormGated
from .short_conv import kimi_short_conv_qkv
from .tensor_parallel import check_divisible, shard_fused_blocks

_KIMI_O_NORM_EPS = 1e-5
"""``KIMI_RMS_NORM_EPS`` in ``kimi_k3/layers/norm.py`` -- ``o_norm``'s eps,
distinct from MLA's ``q_a_layernorm``/``kv_a_layernorm`` eps."""


def _kda_decode(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    raw_gate: Tensor,
    beta_logits: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    input_row_offsets: Tensor,
    state_pool: Tensor,
    recurrent_row_id: Tensor,
    *,
    gate_mode: str,
    out_type: TensorType,
) -> TensorValue:
    values: list[Any] = [
        TensorValue(q),
        TensorValue(k),
        TensorValue(v),
        TensorValue(raw_gate),
        TensorValue(beta_logits),
        TensorValue(a_log),
        TensorValue(dt_bias),
        ops.cast(TensorValue(input_row_offsets), DType.int32),
        BufferValue(state_pool),
        ops.cast(TensorValue(recurrent_row_id), DType.int32),
    ]
    # Under TP this runs once per device, so the output takes `q`'s device.
    local_out_type = TensorType(
        out_type.dtype, out_type.shape, device=TensorValue(q).device
    )
    return cast(
        TensorValue,
        ops.inplace_custom(
            "kda_decode",
            device=TensorValue(q).device,
            values=values,
            out_types=[local_out_type],
            parameters={
                "gate_mode": gate_mode,
                "beta_mode": "logits",
                "state_layout": "K_FIRST",
            },
        )[0],
    )


kda_decode = F.functional(_kda_decode)


class KimiK3KDAAttention(
    Module[[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor], Tensor]
):
    """One KDA layer, ``kda_decode``-only; see the module docstring."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        conv_kernel_size: int,
        gate_lower_bound: float | None,
    ) -> None:
        """Builds the KDA attention.

        Args:
            hidden_size: ``config.hidden_size``.
            num_heads: ``config.kda_num_heads``.
            head_dim: ``config.kda_head_dim``.
            conv_kernel_size: ``config.kda_conv_kernel_size``; the conv pool's
            history depth is ``conv_kernel_size - 1``.
            gate_lower_bound: ``config.kda_gate_lower_bound``. Published K3 sets
            -5.0; ``None`` selects the kernel's own per-head gate instead.
            Any OTHER value raises -- see the module docstring.
        """
        super().__init__()
        if gate_lower_bound is not None and gate_lower_bound != -5.0:
            raise NotImplementedError(
                "Kimi K3 KDA: the kernel's safe gate hardcodes a lower bound"
                f" of -5.0, but this config declares {gate_lower_bound}."
            )
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.gate_lower_bound = gate_lower_bound
        inner = num_heads * head_dim

        self.q_proj = Linear(in_dim=hidden_size, out_dim=inner, bias=False)
        self.k_proj = Linear(in_dim=hidden_size, out_dim=inner, bias=False)
        self.v_proj = Linear(in_dim=hidden_size, out_dim=inner, bias=False)
        # Fused q/k/v depthwise conv weight: see `fuse_qkv_conv_weight` in
        # the V2 weight adapter, which concatenates the checkpoint's
        # separate per-projection conv weights at load time.
        self.conv_weight = Tensor.zeros([3 * inner, conv_kernel_size])

        self.f_a_proj = Linear(in_dim=hidden_size, out_dim=head_dim, bias=False)
        self.f_b_proj = Linear(in_dim=head_dim, out_dim=inner, bias=False)
        self.a_log = Tensor.zeros([num_heads, head_dim], dtype=DType.float32)
        self.dt_bias = Tensor.zeros([num_heads, head_dim], dtype=DType.float32)
        # fp32: the beta GEMM runs in fp32 and feeds the op without a
        # further cast (see `kimi_k3.KimiK3DecoderLayer`'s weight dtype).
        self.b_proj = Linear(in_dim=hidden_size, out_dim=num_heads, bias=False)
        self.g_proj = Linear(in_dim=hidden_size, out_dim=inner, bias=False)

        self.o_norm = KimiRMSNormGated(head_dim, eps=_KIMI_O_NORM_EPS)
        self.o_proj = Linear(in_dim=inner, out_dim=hidden_size, bias=False)

    def forward(
        self,
        x: Tensor,
        conv_pool: Tensor,
        conv_row_id: Tensor,
        state_pool: Tensor,
        recurrent_row_id: Tensor,
        input_row_offsets: Tensor,
    ) -> Tensor:
        """Runs the KDA attention over ``x``.

        Args:
            x: ``[total_seq_len, hidden_size]``, all sequences packed.
            conv_pool: ``[conv_rows, 3 * inner, conv_kernel_size - 1]``,
                mutated in place at ``conv_row_id``.
            conv_row_id: ``[batch_size]`` conv-pool row of each sequence.
            state_pool: ``[recurrent_rows, num_heads, head_dim, head_dim]``
                recurrent state, ``K_FIRST`` layout, mutated in place at
                ``recurrent_row_id``.
            recurrent_row_id: ``[batch_size]`` recurrent-pool row of each
                sequence.
            input_row_offsets: ``[batch_size + 1]`` exclusive prefix sums.

            Under TP, both pools are sharded on axis 1 like the heads they hold.

        Returns:
            ``[total_seq_len, hidden_size]``.
        """
        total_seq_len = x.shape[0]
        inner = self.num_heads * self.head_dim
        mesh = self.conv_weight.mesh
        tp_active = TP in mesh.axis_names and mesh.axis_size(TP) > 1
        tp_size = mesh.axis_size(TP) if tp_active else 1

        qkv = F.concat([self.q_proj(x), self.k_proj(x), self.v_proj(x)], axis=1)
        qkv = kimi_short_conv_qkv(
            qkv, self.conv_weight, conv_pool, conv_row_id, input_row_offsets
        )
        if tp_active:
            # Custom ops have no sharding rule, so their outputs need their
            # placement restated. `split` keeps the shards local; slicing
            # would gather them.
            qkv = qkv.rebind_mapping(NamedMapping(qkv.mesh, (None, TP)))
        elif mesh.num_devices > 1:
            # Same restatement, for a mesh with no `tp` axis (pure EP).
            # Here q/k/v_proj are replicated rather than sharded, but the
            # custom op's output still needs the mapping restated.
            qkv = qkv.rebind_mapping(NamedMapping(qkv.mesh, (None, None)))
        q_flat, k_flat, v_flat = qkv.split([inner, inner, inner], axis=1)

        def as_heads(flat: Tensor) -> Tensor:
            return F.reshape(
                flat, [1, total_seq_len, self.num_heads, self.head_dim]
            )

        q = as_heads(q_flat)
        k = as_heads(k_flat)
        v = as_heads(v_flat)

        raw_gate = F.reshape(
            self.f_b_proj(self.f_a_proj(x)),
            [1, total_seq_len, self.num_heads, self.head_dim],
        )
        # `kda_decode` takes float32 beta logits.
        b_proj = F.cast(self.b_proj.weight, DType.float32)
        beta_logits = F.reshape(
            F.cast(x, DType.float32) @ b_proj.T,
            [1, total_seq_len, self.num_heads],
        )

        a_log_f32 = F.cast(self.a_log, DType.float32)
        dt_bias_f32 = F.cast(self.dt_bias, DType.float32)
        raw_gate_f32 = F.cast(raw_gate, DType.float32)
        if self.gate_lower_bound is None:
            kernel_a_log = a_log_f32[:, 0]
            kernel_dt_bias = dt_bias_f32
            kernel_raw_gate = raw_gate_f32
            gate_mode = "original"
        else:
            kernel_raw_gate = (raw_gate_f32 + dt_bias_f32) * F.exp(a_log_f32)
            kernel_a_log = F.zeros_like(a_log_f32[:, 0])
            kernel_dt_bias = F.zeros_like(dt_bias_f32)
            gate_mode = "safe"

        # `out_type` is one device's result, so it takes the local head
        # count. `_kda_decode` replaces the device.
        out_type = TensorType(
            x.dtype,
            [1, total_seq_len, self.num_heads // tp_size, self.head_dim],
            device=x.mesh.devices[0] if x.is_distributed else x.device,
        )
        o = kda_decode(
            q,
            k,
            v,
            kernel_raw_gate,
            beta_logits,
            kernel_a_log,
            kernel_dt_bias,
            input_row_offsets,
            state_pool,
            recurrent_row_id,
            gate_mode=gate_mode,
            out_type=out_type,
        )
        if tp_active:
            o = o.rebind_mapping(NamedMapping(o.mesh, (None, None, TP, None)))
        elif mesh.num_devices > 1:
            o = o.rebind_mapping(NamedMapping(o.mesh, (None, None, None, None)))
        o = F.reshape(o, [total_seq_len, self.num_heads, self.head_dim])

        gate = F.reshape(
            self.g_proj(x), [total_seq_len, self.num_heads, self.head_dim]
        )
        normed = self.o_norm(o, gate)
        return self.o_proj(F.reshape(normed, [total_seq_len, inner]))


def tensor_parallel_kda_attention(
    layer: KimiK3KDAAttention, mesh: DeviceMesh
) -> KimiK3KDAAttention:
    """Shards a KDA layer by head along the ``tp`` mesh axis.

    ``f_a_proj`` (the gate's rank-``head_dim`` down projection) and
    ``o_norm`` (which normalizes within a head) stay replicated. ``o_proj``
    is sharded on its input dim, so the caller must allreduce the output.
    """
    tp_size = mesh.axis_size(TP)
    if tp_size == 1:
        return layer
    check_divisible("KDA head count", layer.num_heads, tp_size)

    row_mapping = NamedMapping(mesh, (TP, None))
    layer.q_proj.weight = layer.q_proj.weight.to(row_mapping)
    layer.k_proj.weight = layer.k_proj.weight.to(row_mapping)
    layer.v_proj.weight = layer.v_proj.weight.to(row_mapping)
    layer.f_b_proj.weight = layer.f_b_proj.weight.to(row_mapping)
    layer.b_proj.weight = layer.b_proj.weight.to(row_mapping)
    layer.a_log = layer.a_log.to(row_mapping)
    layer.dt_bias = layer.dt_bias.to(row_mapping)
    layer.g_proj.weight = layer.g_proj.weight.to(row_mapping)
    layer.o_proj.weight = layer.o_proj.weight.to(NamedMapping(mesh, (None, TP)))
    layer.conv_weight = shard_fused_blocks(
        layer.conv_weight, mesh, num_blocks=3, axis=0
    )
    return layer
