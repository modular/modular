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
"""Kimi K3's Kimi Delta Attention (ModuleV3).

Direct V3 port of ``kimi_k3/layers/kda_attention.py``'s
``kimi_k3_kda_attention`` orchestration, calling the same ``kda_decode``
custom op via :func:`max.functional.inplace_custom`.

Every batch takes ``kda_decode``, and the op itself picks the recurrence,
so this layer carries no device knowledge: no accelerator query, no
servability check, no operand casts, and no per-batch conditional. Under
the safe gate every batch outside a verify passes the conv's raw fp32
q/k/v and the raw gate under ``input_mode="conv_raw"``: the op routes
at run time on the shapes it already holds, so a decode-shaped batch runs
the fused prologue -- the SiLU, the model-dtype rounding and the gate's
pre-activation that the graph otherwise runs as its own small elementwise
launches over ``[1, tokens, 12, 128]`` -- while any other batch
activates first inside the op and continues on the chunk pipeline or the
token-serial kernel exactly as before. The routing cannot live at trace
time: a staged graph's symbolic extents never prove a batch shape, so a
static-shape gate left every served decode batch unfused.

On gfx950, under the bounded "safe" gate with bf16 q/k/v and a float32
state pool, a prefill batch runs the chunk-parallel pipeline inside the op
while every other batch -- decode included -- runs the token-sequential
kernel; on any other target, and under the "original" gate, every batch
runs the token-sequential kernel. The K3 memory planner still reserves the
chunk workspaces host-side (see ``memory_planner.py``); KERN-3869 tracks
moving them into graph-planned buffers.

The speculative-decoding VERIFY arm: when a binding is set on the layer
(see :class:`KdaVerifyBinding`), the conv runs
``gated_delta_conv1d_verify_fwd``, which leaves the live window unwritten
and keeps a copy of its input for the conv replay, and the recurrence runs
``kda_verify_ring``, which reads the live pool without writing it and
records each token's update in the ring. A width-0 window (no drafts)
commits both live pools itself, the same arithmetic the plain forward uses,
so one graph serves every step. Outside a verify the layer is unchanged.

Both gate-lower-bound arms V2 supports are kept (not scoped out): the
"original" arm (``gate_lower_bound is None``, the kernel's own per-head
``-exp(A_log) * softplus(pre)``) and the "safe" arm
(``log_alpha = bound * sigmoid(exp(A_log) * (raw_gate + dt_bias))``).
Under ``conv_raw`` the raw operands pass through and the op applies the
pre-activation itself: in the fused kernel's prologue for a decode-shaped
batch, in one activation launch for every other batch. Every activated
safe-gate call -- the verify ring's -- instead takes the pre-activated
gate with zeroed gate parameters: the activated kernels read ``a_log``
per head, so the per-channel packing would both point every head at head
0's coefficient and fail the chunk contract's ``a_log`` extent.
Published K3 sets the bound to -5.0, and both kernels hardcode -5.0 in
their "safe" arm, so any other bound raises rather than silently running
the wrong gate.
"""

from __future__ import annotations

from typing import NamedTuple

from max import functional as F
from max.dtype import DType
from max.graph import BufferValue, TensorType, TensorValue, ops
from max.nn import Linear, Module
from max.nn._graph_ops.state_space import gated_delta_conv1d_verify_fwd
from max.nn._graph_ops.state_space.kimi_delta import STATE_LAYOUT, KdaGateMode
from max.nn.common_layers.mesh_axis import TP
from max.tensor import Tensor
from max.tensor.sharding import DeviceMesh, NamedMapping
from max.tensor.sharding.action import PerShard

from .norm import KimiRMSNormGated, kimi_rms_norm_gated
from .short_conv import kimi_short_conv_qkv
from .tensor_parallel import check_divisible, shard_interleaved_blocks

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
    gate_mode: KdaGateMode,
    out_type: TensorType,
    pin_fma: bool,
    input_mode: str = "activated",
    activation_dtype: DType | None = None,
) -> TensorValue:
    values: list[TensorValue | BufferValue] = [
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
    parameters: dict[str, bool | int | str | DType] = {
        "gate_mode": gate_mode,
        "beta_mode": "logits",
        "state_layout": STATE_LAYOUT,
    }
    if pin_fma:
        # The speculative verify-ring and fold kernels spell every
        # multiply-add as an explicit `fma`; pinning here makes a plain
        # decode land on the same state bit for bit, so a rolled-back verify
        # equals the decode it interrupted.
        parameters["pin_fma"] = True
    if input_mode != "activated":
        if activation_dtype is None:
            raise ValueError(
                "kda_decode: input_mode 'conv_raw' needs activation_dtype,"
                " the model dtype the fused SiLU result rounds to"
            )
        parameters["input_mode"] = input_mode
        parameters["activation_dtype"] = activation_dtype
    out = ops.inplace_custom(
        "kda_decode",
        device=TensorValue(q).device,
        values=values,
        out_types=[local_out_type],
        parameters=parameters,
    )[0]
    assert isinstance(out, TensorValue)
    return out


kda_decode = F.functional(_kda_decode)


def _kda_verify_ring(
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
    ring_pool: Tensor | PerShard[Tensor],
    ring_row_id: Tensor | PerShard[Tensor],
    verify_width: TensorValue,
    *,
    gate_mode: KdaGateMode,
    out_type: TensorType,
) -> TensorValue:
    # The dispatch delivers each device its own entry of a `PerShard`, so
    # what reaches the op body is always one device's tensors.
    assert isinstance(ring_pool, Tensor)
    assert isinstance(ring_row_id, Tensor)
    values: list[TensorValue | BufferValue] = [
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
        BufferValue(ring_pool),
        ops.cast(TensorValue(ring_row_id), DType.int32),
        verify_width,
    ]
    local_out_type = TensorType(
        out_type.dtype, out_type.shape, device=TensorValue(q).device
    )
    out = ops.inplace_custom(
        "kda_verify_ring",
        device=TensorValue(q).device,
        values=values,
        out_types=[local_out_type],
        parameters={
            "gate_mode": gate_mode,
            "beta_mode": "logits",
            "state_layout": STATE_LAYOUT,
        },
    )[0]
    assert isinstance(out, TensorValue)
    return out


kda_verify_ring = F.functional(_kda_verify_ring)


def _kimi_short_conv_qkv_verify(
    x_qkv_ragged: Tensor,
    conv_weight: Tensor,
    conv_pool: Tensor,
    conv_row_id: Tensor,
    input_row_offsets: Tensor,
    verify_width: TensorValue,
) -> tuple[TensorValue, TensorValue]:
    """Runs the verify form of :func:`.short_conv.kimi_short_conv_qkv`.

    Kept here rather than in ``short_conv.py`` so the plain form stays the
    V2 port alone. Same fp32 working dtype and the same raw fp32 output (the
    caller applies ``silu`` per q/k/v slice); the kernel writes the live
    window only at a verify width of zero, and with drafts it returns the
    copy of its input the conv replay replays over instead.
    """
    conv, window = gated_delta_conv1d_verify_fwd(
        qkv_input_ragged=ops.cast(TensorValue(x_qkv_ragged), DType.float32),
        conv_weight=ops.cast(TensorValue(conv_weight), DType.float32),
        conv_state=BufferValue(conv_pool),
        slot_idx=TensorValue(conv_row_id),
        input_row_offsets=TensorValue(input_row_offsets),
        verify_width=verify_width,
        rollback=False,
    )
    return conv, window


kimi_short_conv_qkv_verify = F.functional(_kimi_short_conv_qkv_verify)


def _record_conv_replay(
    binding: KdaVerifyBinding, window: Tensor, conv_weight: Tensor
) -> None:
    """Appends this layer's per-device conv replay inputs to the capture.

    The conv op already ran once per device, so the joined window and the
    weight each hold one shard per device; the replay re-runs each device's
    own shard over its own pool rows. Call order is layer order, which is
    the order the replay's per-layer row tables read.
    """
    windows = window.local_shards
    weights = conv_weight.local_shards
    assert len(windows) == len(weights) == len(binding.captures), (
        f"the verify captured {len(windows)} windows and {len(weights)}"
        f" weights across {len(binding.captures)} devices"
    )
    for device, (shard_window, shard_weight) in enumerate(
        zip(windows, weights, strict=True)
    ):
        binding.captures[device].append(
            KdaConvReplayInputs(
                qkv_window=TensorValue(shard_window),
                conv_weight=TensorValue(shard_weight),
            )
        )


class KdaConvReplayInputs(NamedTuple):
    """One KDA layer's conv inputs the conv replay re-runs.

    Satisfies the conv replay's
    :class:`~max.pipelines.speculative.gated_delta_rollback.ConvReplayInputs`
    protocol; the recurrent replay needs nothing per layer -- the ring's
    records carry the whole update. Graph values rather than the shard
    tensors, so a capture is replay-ready wherever it lands.
    """

    qkv_window: TensorValue
    """``[batch_size * (1 + K), conv_dim]`` copy the verify's conv kept."""

    conv_weight: TensorValue
    """``[conv_dim, conv_kernel_size]`` depthwise weights, fp32 casts."""


class KdaVerifyBinding:
    """One verify's ring, width and capture, bound to a KDA layer.

    Set on each KDA layer by ``kimi_k3_modulev3.spec_state``'s
    ``capturing`` context and unwound on the way out, so a forward outside
    a verify -- the base graph's, or the draft's -- never sees it. A plain
    holder rather than bare module attributes: the ring tensors would be
    swept up as module parameters, while a holder the module tree carries
    as one static value keeps them invisible.

    One binding per layer: the ring row id is the layer's own. The capture
    lists are shared with every other KDA layer's binding, one list per
    device, and each layer appends its own entry in call order.
    """

    __slots__ = ("captures", "ring_pool", "ring_row_id", "verify_width")

    def __init__(
        self,
        verify_width: TensorValue,
        ring_pool: Tensor | PerShard[Tensor],
        ring_row_id: Tensor | PerShard[Tensor],
        captures: list[list[KdaConvReplayInputs]],
    ) -> None:
        self.verify_width = verify_width
        self.ring_pool = ring_pool
        self.ring_row_id = ring_row_id
        self.captures = captures


class KimiK3KDAAttention(
    Module[[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor], Tensor]
):
    """One KDA layer; see the module docstring."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        conv_kernel_size: int,
        gate_lower_bound: float | None,
        *,
        pin_fma: bool = False,
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
            pin_fma: Whether the ``kda_decode`` op pins its multiply-adds as
            explicit ``fma``s, the bit-identity a speculative build's
            verify-ring and fold kernels need; plain serving keeps the
            kernel's default, which only AMD ever unpins.
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
        self.pin_fma = pin_fma
        inner = num_heads * head_dim

        # One merged q/k/v projection: three separate GEMMs followed by an
        # axis-1 concat cannot fuse, one GEMM can. Under TP the merged
        # weight is interleaved per rank exactly like `conv_weight` (see
        # `_fuse_qkv_proj` in the weight adapter), so the `split` below
        # still hands each rank its own q, k and v.
        self.qkv_proj = Linear(
            in_dim=hidden_size, out_dim=3 * inner, bias=False
        )
        # Fused q/k/v depthwise conv weight: see `_fuse_qkv_conv_weight`
        # in the weight adapter, which concatenates the checkpoint's
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

    verify_binding: KdaVerifyBinding | None = None
    """The verify's ring, width and capture, or ``None`` outside a verify.

    Bound per layer by ``spec_state``'s ``capturing`` context; see
    :class:`KdaVerifyBinding`.
    """

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
            A verify's ring (``verify_binding``) is sharded the same way.

        Returns:
            ``[total_seq_len, hidden_size]``.
        """
        binding = self.verify_binding
        total_seq_len = x.shape[0]
        inner = self.num_heads * self.head_dim
        mesh = self.conv_weight.mesh
        tp_active = TP in mesh.axis_names and mesh.axis_size(TP) > 1
        tp_size = mesh.axis_size(TP) if tp_active else 1

        qkv = self.qkv_proj(x)
        if binding is None:
            qkv = kimi_short_conv_qkv(
                qkv, self.conv_weight, conv_pool, conv_row_id, input_row_offsets
            )
        else:
            # The replay re-runs the conv over the same fp32 operands the
            # verify's own cast produced, so it repeats the verify's
            # arithmetic rather than rounding a bf16 weight twice.
            conv_weight = self.conv_weight.cast(DType.float32)
            conv_out, window = kimi_short_conv_qkv_verify(
                qkv,
                conv_weight,
                conv_pool,
                conv_row_id,
                input_row_offsets,
                binding.verify_width,
            )
            _record_conv_replay(binding, window, conv_weight)
            qkv = conv_out
        if tp_active:
            # Custom ops have no sharding rule, so their outputs need their
            # placement restated. `split` keeps the shards local; slicing
            # would gather them.
            qkv = qkv.rebind_mapping(NamedMapping(qkv.mesh, (None, TP)))
        elif mesh.num_devices > 1:
            # Same restatement, for a mesh with no `tp` axis (pure EP).
            # Here `qkv_proj` is replicated rather than sharded, but the
            # custom op's output still needs the mapping restated.
            qkv = qkv.rebind_mapping(NamedMapping(qkv.mesh, (None, None)))
        q_flat, k_flat, v_flat = qkv.split([inner, inner, inner], axis=1)

        # Every safe-gate batch outside a verify passes the conv's raw fp32
        # operands: the op routes at run time on the shapes it holds, so a
        # decode-shaped batch runs the fused prologue while any other batch
        # activates first inside the op (SiLU, model-dtype rounding, gate
        # pre-activation). A trace-time shape gate cannot see a staged
        # batch's symbolic extents. The verify ring takes activated q/k/v and
        # the pre-activated gate, so only a plain decode fuses, and only in a
        # build without speculation (`pin_fma` off): the fused prologue is
        # not byte-identical to the graph's elementwise activation, and a
        # speculative build's rollback must land exactly on what its plain
        # decode would have computed.
        fused_decode = (
            self.gate_lower_bound is not None
            and binding is None
            and not self.pin_fma
        )

        def as_heads(flat: Tensor) -> Tensor:
            heads = flat.reshape(
                [1, total_seq_len, self.num_heads, self.head_dim]
            )
            if fused_decode:
                return heads
            # silu and the cast back to the model dtype run per q/k/v slice
            # (the conv returns fp32 without them, `short_conv.py`), so each
            # fuses with the split that feeds it instead of forming a
            # separate launch over the conv's whole output.
            return F.silu(heads).cast(x.dtype)

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
            gate_mode: KdaGateMode = "original"
        elif fused_decode:
            # Only the conv_raw call folds the safe gate's per-channel
            # pre-activation, so only it passes the raw gate, `dt_bias` and
            # the per-channel `a_log` through unpacked, `a_log` flat for the
            # op's rank-1 slot.
            kernel_a_log = F.reshape(
                a_log_f32, [self.num_heads * self.head_dim]
            )
            kernel_dt_bias = dt_bias_f32
            kernel_raw_gate = raw_gate_f32
            gate_mode = "safe"
        else:
            # Every other safe-gate call runs activated -- the verify ring,
            # or a plain decode that does not fuse -- and the activated
            # kernels read `a_log` per head and apply the pre-activation
            # themselves, so the gate arrives pre-activated and the gate
            # parameters arrive as the identity.
            kernel_raw_gate = (raw_gate_f32 + dt_bias_f32) * F.exp(a_log_f32)
            kernel_a_log = F.zeros_like(a_log_f32[:, 0])
            kernel_dt_bias = F.zeros_like(dt_bias_f32)
            gate_mode = "safe"

        # `out_type` is one device's result, so it takes the local head
        # count: under TP each device runs `num_heads // tp_size` heads,
        # while the pure-EP mesh replicates the layer and runs all of them.
        # `_kda_decode` replaces the device. fp32 because `o_norm` computes
        # in fp32 anyway: a model-dtype output would round `o` once before
        # the norm and need a cast launch to widen it back.
        heads_per_device = self.num_heads // tp_size
        out_type = TensorType(
            DType.float32,
            [1, total_seq_len, heads_per_device, self.head_dim],
            device=x.mesh.devices[0] if x.is_distributed else x.device,
        )
        if binding is None:
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
                pin_fma=self.pin_fma,
                input_mode="conv_raw" if fused_decode else "activated",
                activation_dtype=x.dtype if fused_decode else None,
            )
        else:
            o = kda_verify_ring(
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
                binding.ring_pool,
                binding.ring_row_id,
                binding.verify_width,
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
        # `o_norm` consumes the fp32 recurrence output and rounds once to
        # the model dtype inside the fused norm (via `out_dtype`), so
        # `o_proj` sees the model dtype with no separate cast launch.
        # Called as the function rather than through the module's forward
        # because the output dtype is this layer's per-call choice.
        normed = kimi_rms_norm_gated(
            o, self.o_norm.weight, gate, self.o_norm.eps, out_dtype=x.dtype
        )
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
    layer.qkv_proj.weight = shard_interleaved_blocks(
        layer.qkv_proj.weight, mesh, num_blocks=3, axis=0
    )
    layer.f_b_proj.weight = layer.f_b_proj.weight.to(row_mapping)
    layer.b_proj.weight = layer.b_proj.weight.to(row_mapping)
    layer.a_log = layer.a_log.to(row_mapping)
    layer.dt_bias = layer.dt_bias.to(row_mapping)
    layer.g_proj.weight = layer.g_proj.weight.to(row_mapping)
    layer.o_proj.weight = layer.o_proj.weight.to(NamedMapping(mesh, (None, TP)))
    layer.conv_weight = shard_interleaved_blocks(
        layer.conv_weight, mesh, num_blocks=3, axis=0
    )
    return layer
