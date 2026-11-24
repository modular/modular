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

The ``KimiK3KDAAttention`` orchestration, calling the ``kda_decode``
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
runs the token-sequential kernel. The chunk workspaces are allocated
host-side; KERN-3869 tracks moving them into graph-planned buffers.

The speculative-decoding VERIFY arm: when a binding is set on the layer
(see :class:`KdaVerifyBinding`), the conv runs
``gated_delta_conv1d_verify_fwd``, which leaves the live window unwritten
and keeps a copy of its input for the conv replay, and the recurrence runs
``kda_verify_ring``, which reads the live pool without writing it and
records each token's update in the ring. In a mixed verify, a prompt row
skips both: the conv writes its window, and the recurrence commits its
state on the chunk pipeline, each at the row's last real token, so the
rollback leaves the row alone. A width-0 window (no drafts) commits both
live pools itself, the same arithmetic the plain forward uses, so one graph
serves every step. Outside a verify the layer is unchanged.

Under the fused op's compiled geometry -- 128-wide heads, a 4-tap conv and
the bounded "safe" gate -- the verify arm collapses into one
``kda_verify_ring_conv`` launch instead: the short conv over the raw qkv,
the glue's silu + cast, the gate's pre-activation and the gated o_norm all
run inside the recurrence kernel, which writes the conv window copy the
replay reads as its second output. The ring records and the fold contract
are the unfused ops' own, so the rollback is unchanged; every other config
keeps the four-launch chain, byte for byte.

Both gate-lower-bound arms are kept (not scoped out): the
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

from functools import lru_cache
from typing import NamedTuple

from max import functional as F
from max.driver import accelerator_architecture_name
from max.dtype import DType
from max.graph import BufferValue, Dim, TensorType, TensorValue, ops
from max.nn import Linear, Module
from max.nn._graph_ops.state_space import gated_delta_conv1d_verify_fwd
from max.nn._graph_ops.state_space.kimi_delta import (
    STATE_LAYOUT,
    KdaGateMode,
)
from max.nn._graph_ops.state_space.kimi_delta import (
    kda_verify_ring_conv_packed as kda_verify_ring_conv_op,
)
from max.nn._graph_ops.state_space.kimi_delta import (
    kda_verify_ring_conv_packed_gate_up as kda_verify_ring_conv_gate_up_op,
)
from max.nn.common_layers.mesh_axis import TP
from max.tensor import Tensor
from max.tensor.sharding import DeviceMesh, NamedMapping
from max.tensor.sharding.action import PerShard

from .norm import KimiRMSNormGated, kimi_rms_norm_gated
from .short_conv import kimi_short_conv_qkv
from .tensor_parallel import check_divisible, fuse_rows

_KIMI_O_NORM_EPS = 1e-5
"""``KIMI_RMS_NORM_EPS`` in ``norm.py`` -- ``o_norm``'s eps,
distinct from MLA's ``q_a_layernorm``/``kv_a_layernorm`` eps."""
_FUSED_CONV_ARCH = "gfx950"
"""The one target the fused verify kernel is built for (CDNA4, MI355X)."""
_GATE_UP_HEAD_MULTIPLE = 4
"""The per-device head count the folded op requires a multiple of: it loads
the gate's down projection in 4-element vectors from column
``4 * inner + heads`` of the packed row."""


def _static_extent(dim: Dim) -> int | None:
    """Returns ``dim``'s extent when it is static, else ``None``."""
    try:
        return int(dim)
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=1)
def _on_fused_conv_arch() -> bool:
    """Whether the accelerator is the fused verify kernel's target.

    Asked once per process: the architecture is a process-wide property, and
    the probe builds a device handle, which a verify step must not pay per
    layer.
    """
    return accelerator_architecture_name().startswith(_FUSED_CONV_ARCH)


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
            # A mixed verify's prompt rows commit on the chunk pipeline at
            # their last real token; a decode-only verify is unchanged.
            "chunk_long_rows": True,
        },
    )[0]
    assert isinstance(out, TensorValue)
    return out


kda_verify_ring = F.functional(_kda_verify_ring)


def _kda_verify_ring_conv(
    packed_qkvg: Tensor,
    conv_weight: Tensor,
    conv_pool: Tensor,
    conv_row_id: Tensor,
    raw_gate: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    input_row_offsets: Tensor,
    state_pool: Tensor,
    recurrent_row_id: Tensor,
    ring_pool: Tensor | PerShard[Tensor],
    ring_row_id: Tensor | PerShard[Tensor],
    verify_width: TensorValue,
    norm_weight: Tensor,
    out_type: TensorType,
    *,
    gate_mode: KdaGateMode,
    norm_eps: float,
) -> tuple[TensorValue, TensorValue]:
    """Runs the fused verify op; see ``kimi_delta.kda_verify_ring_conv_packed``.

    A thin bridge onto the shared wrapper rather than a local
    ``ops.inplace_custom`` call, so the operand contract and its checks
    live once; the dispatch splits the ring ``PerShard`` the same way
    ``_kda_verify_ring``'s does. The output type is the caller's
    declaration, as the unfused path's own op call.
    """
    # The dispatch delivers each device its own entry of a `PerShard`, so
    # what reaches the op body is always one device's tensors.
    assert isinstance(ring_pool, Tensor)
    assert isinstance(ring_row_id, Tensor)
    return kda_verify_ring_conv_op(
        TensorValue(packed_qkvg),
        TensorValue(conv_weight),
        BufferValue(conv_pool),
        TensorValue(conv_row_id),
        TensorValue(raw_gate),
        TensorValue(a_log),
        TensorValue(dt_bias),
        TensorValue(input_row_offsets),
        BufferValue(state_pool),
        TensorValue(recurrent_row_id),
        BufferValue(ring_pool),
        TensorValue(ring_row_id),
        verify_width,
        TensorValue(norm_weight),
        out_type=TensorType(
            out_type.dtype,
            out_type.shape,
            device=TensorValue(packed_qkvg).device,
        ),
        norm_eps=norm_eps,
        gate_mode=gate_mode,
        beta_mode="logits",
        state_layout=STATE_LAYOUT,
    )


kda_verify_ring_conv = F.functional(_kda_verify_ring_conv)


def _kda_verify_ring_conv_gate_up(
    packed_qkvg: Tensor,
    conv_weight: Tensor,
    conv_pool: Tensor,
    conv_row_id: Tensor,
    gate_up_weight: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    input_row_offsets: Tensor,
    state_pool: Tensor,
    recurrent_row_id: Tensor,
    ring_pool: Tensor | PerShard[Tensor],
    ring_row_id: Tensor | PerShard[Tensor],
    verify_width: TensorValue,
    norm_weight: Tensor,
    out_type: TensorType,
    *,
    gate_mode: KdaGateMode,
    norm_eps: float,
) -> tuple[TensorValue, TensorValue]:
    """Runs the fused verify op with the gate's up projection folded in.

    :func:`_kda_verify_ring_conv`'s bridge onto
    ``kimi_delta.kda_verify_ring_conv_packed_gate_up``: the raw gate operand
    is replaced by ``f_b_proj``'s weight, and the op reads the gate's down
    projection from the packed row's tail.
    """
    assert isinstance(ring_pool, Tensor)
    assert isinstance(ring_row_id, Tensor)
    return kda_verify_ring_conv_gate_up_op(
        TensorValue(packed_qkvg),
        TensorValue(conv_weight),
        BufferValue(conv_pool),
        TensorValue(conv_row_id),
        TensorValue(gate_up_weight),
        TensorValue(a_log),
        TensorValue(dt_bias),
        TensorValue(input_row_offsets),
        BufferValue(state_pool),
        TensorValue(recurrent_row_id),
        BufferValue(ring_pool),
        TensorValue(ring_row_id),
        verify_width,
        TensorValue(norm_weight),
        out_type=TensorType(
            out_type.dtype,
            out_type.shape,
            device=TensorValue(packed_qkvg).device,
        ),
        norm_eps=norm_eps,
        gate_mode=gate_mode,
        beta_mode="logits",
        state_layout=STATE_LAYOUT,
    )


kda_verify_ring_conv_gate_up = F.functional(_kda_verify_ring_conv_gate_up)


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
    shared-kernel bridge alone. Same fp32 working dtype and the same raw
    fp32 output (the caller applies ``silu`` per q/k/v slice); the kernel
    writes the live window only at a verify width of zero, and with drafts
    it returns the copy of its input the conv replay replays over instead.
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


def _record_conv_replay(binding: KdaVerifyBinding, window: Tensor) -> None:
    """Appends this layer's per-device captured windows to the fold input.

    The verify op already ran once per device -- the conv's or the fused
    one's -- so the joined window holds one shard per device; the fold
    places each device's conv window from its own shard. Call order is
    layer order, which is the order the fold's per-layer row tables read.
    """
    windows = window.local_shards
    assert len(windows) == len(binding.captures), (
        f"the verify captured {len(windows)} windows across"
        f" {len(binding.captures)} devices"
    )
    for device, shard_window in enumerate(windows):
        binding.captures[device].append(
            KdaConvReplayInputs(qkv_window=TensorValue(shard_window))
        )


class KdaConvReplayInputs(NamedTuple):
    """One KDA layer's captured window, the fold's conv-window source.

    The state fold places each request's conv window from this copy in the
    same launch that folds the ring records; the recurrent side needs
    nothing per layer -- the ring's records carry the whole update. Graph
    values rather than the shard tensors, so a capture is fold-ready
    wherever it lands.
    """

    qkv_window: TensorValue
    """``[batch_size * (1 + K), conv_dim]`` copy the verify's conv kept."""


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
        a_log_size: int | None = None,
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
            a_log_size: The length the checkpoint stores ``A_log`` at, which
                may pad the per-head coefficients with zeros; ``None`` is
                ``num_heads``.
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

        # The checkpoint's tensors as it stores them. The graph fuses the
        # projections that read the layer input into one GEMM's weight and
        # packs the three convs into one (`_packed_qkvg_weight`,
        # `_conv_weight`); both depend only on weights, so they run once,
        # at initialization.
        self.q_proj = Linear(in_dim=hidden_size, out_dim=inner, bias=False)
        self.k_proj = Linear(in_dim=hidden_size, out_dim=inner, bias=False)
        self.v_proj = Linear(in_dim=hidden_size, out_dim=inner, bias=False)
        self.g_proj = Linear(in_dim=hidden_size, out_dim=inner, bias=False)
        self.b_proj = Linear(in_dim=hidden_size, out_dim=num_heads, bias=False)
        self.q_conv1d = Tensor.zeros(
            [inner, 1, conv_kernel_size], dtype=DType.float32
        )
        self.k_conv1d = Tensor.zeros(
            [inner, 1, conv_kernel_size], dtype=DType.float32
        )
        self.v_conv1d = Tensor.zeros(
            [inner, 1, conv_kernel_size], dtype=DType.float32
        )
        self.f_a_proj = Linear(in_dim=hidden_size, out_dim=head_dim, bias=False)
        self.f_b_proj = Linear(in_dim=head_dim, out_dim=inner, bias=False)
        self.a_log = Tensor.zeros(
            [a_log_size or num_heads], dtype=DType.float32
        )
        self.dt_bias = Tensor.zeros([inner], dtype=DType.float32)

        self.o_norm = KimiRMSNormGated(head_dim, eps=_KIMI_O_NORM_EPS)
        self.o_proj = Linear(in_dim=inner, out_dim=hidden_size, bias=False)

    verify_binding: KdaVerifyBinding | None = None
    """The verify's ring, width and capture, or ``None`` outside a verify.

    Bound per layer by ``spec_state``'s ``capturing`` context; see
    :class:`KdaVerifyBinding`.
    """

    def _packed_qkvg_weight(self) -> Tensor:
        """The fused input GEMM's weight, per rank ``[q | k | v | g | b | f_a]``.

        The qkv prefix keeps the per-rank order the packed conv expects.
        ``f_a_proj`` is replicated, so every rank's block holds all of it:
        each rank's ``f_b_proj`` slice reads the whole gate row.
        """
        return fuse_rows(
            [
                self.q_proj.weight,
                self.k_proj.weight,
                self.v_proj.weight,
                self.g_proj.weight,
                self.b_proj.weight,
                self.f_a_proj.weight,
            ]
        )

    def _conv_weight(self) -> Tensor:
        """The packed ``[3 * inner, kernel]`` conv weight, per rank ``[q | k | v]``."""
        fused = fuse_rows([self.q_conv1d, self.k_conv1d, self.v_conv1d])
        return F.reshape(
            fused,
            [3 * self.num_heads * self.head_dim, int(self.q_conv1d.shape[-1])],
        )

    def _gate_params(self) -> tuple[Tensor, Tensor]:
        """``a_log`` and ``dt_bias`` per channel, both ``[num_heads, head_dim]``.

        ``a_log`` holds one coefficient per head, past which a checkpoint may
        pad it with zeros; each head's coefficient covers its channels.
        """
        a_log = F.broadcast_to(
            F.reshape(self.a_log[: self.num_heads], [self.num_heads, 1]),
            [self.num_heads, self.head_dim],
        )
        dt_bias = F.reshape(self.dt_bias, [self.num_heads, self.head_dim])
        mesh = self.dt_bias.mesh
        if TP in mesh.axis_names and mesh.axis_size(TP) > 1:
            heads = NamedMapping(mesh, (TP, None))
            a_log = a_log.to(heads)
            dt_bias = dt_bias.to(heads)
        return a_log, dt_bias

    def _fused_verify_eligible(
        self,
        packed_qkvg: Tensor,
        row_offsets: Tensor,
        verify_width: TensorValue,
    ) -> bool:
        """Whether the fused verify op covers this layer's config and batch.

        The op is compiled for K3's geometry alone: 128-wide heads, a 4-tap
        conv, and the bounded "safe" gate whose pre-activation is the piece
        that moves in-kernel -- on the gfx950 target the kernel is built for.
        The qkv family may be bf16 (production) or fp32 (the fp32-exact
        integration tests), and the op binds the o_norm weight and gate to
        that same dtype. Anything else -- the 32-wide head configs, the
        "original" gate, another accelerator -- keeps the unfused verify
        chain, byte for byte.

        The batch must also be a uniform verify: the op holds every row to
        one token and its drafts. A mixed verify carries prompt rows, which
        commit on the chunk pipeline and are skipped by the serial ring; the
        unfused arm runs those, and a symbolic shape is left to it rather
        than guessed at.
        """
        if not (
            self.gate_lower_bound is not None
            and self.head_dim == 128
            and int(self.q_conv1d.shape[-1]) == 4
            and packed_qkvg.dtype in (DType.bfloat16, DType.float32)
            and _on_fused_conv_arch()
        ):
            return False
        total = _static_extent(packed_qkvg.shape[0])
        batch = _static_extent(row_offsets.shape[0])
        drafts = _static_extent(verify_width.shape[0])
        if total is None or batch is None:
            # A served graph declares its token and offset extents
            # symbolically, and the verify's own window is still uniform
            # there: the op checks it at graph build and refuses a wider
            # batch rather than place a wrong window, so a symbolic extent
            # keeps the fused arm.
            return True
        if drafts is None:
            return False
        # A width of zero is the context step, whose arm takes any rows.
        return drafts == 0 or total == (batch - 1) * (drafts + 1)

    def _folds_gate_up(self, mesh_tp: int) -> bool:
        """Whether the fused verify op runs ``f_b_proj`` itself.

        The op reads the gate's down projection from the fused input GEMM's
        tail columns and binds ``f_b_proj``'s weight to the packed row's
        dtype.
        """
        return (
            self.f_b_proj.weight.dtype == self.q_proj.weight.dtype
            and (self.num_heads // mesh_tp) % _GATE_UP_HEAD_MULTIPLE == 0
        )

    def _fused_verify_forward(
        self,
        x: Tensor,
        packed_qkvg: Tensor,
        gate_down: Tensor,
        binding: KdaVerifyBinding,
        conv_pool: Tensor,
        conv_row_id: Tensor,
        state_pool: Tensor,
        recurrent_row_id: Tensor,
        input_row_offsets: Tensor,
    ) -> Tensor:
        """Runs the whole verify chain as the one fused op.

        The op takes the fused input GEMM's packed ``q | k | v | gate | beta``
        output whole and carves the qkv, the norm-gate and the beta views out
        of it itself (see ``kimi_delta.kda_verify_ring_conv_packed``), so no
        block is materialized as a copy ahead of the launch. The raw gate --
        the up projection the GEMM does not cover -- runs inside the op when
        ``_folds_gate_up`` holds, which reads the ``gate_down`` columns at the
        packed row's tail; otherwise it is built here off the ``gate_down``
        columns `packed_qkvg` is split into, and those tail columns sit past
        everything the op views -- it walks its own row stride -- so they cost
        it nothing.
        """
        total_seq_len = x.shape[0]
        # The output type is the caller's declaration, as the unfused path's
        # own op call: under TP the op's output holds the per-device head
        # count, which only the layer knows.
        mesh = self.q_proj.weight.mesh
        tp_active = TP in mesh.axis_names and mesh.axis_size(TP) > 1
        tp_size = mesh.axis_size(TP) if tp_active else 1
        heads_per_device = self.num_heads // tp_size
        out_type = TensorType(
            x.dtype,
            [1, total_seq_len, heads_per_device, self.head_dim],
            device=x.mesh.devices[0] if x.is_distributed else x.device,
        )
        if self._folds_gate_up(tp_size):
            # The op runs `f_b_proj` over the packed row's `gate_down` tail,
            # so the graph launches neither that GEMM nor the copy that made
            # its input contiguous.
            verify_op = kda_verify_ring_conv_gate_up
            gate_operand = self.f_b_proj.weight
        else:
            # The fused op takes the gate at the rank it requires ([1, T, H,
            # K]); it does not add the leading axis itself, so the reshape
            # here builds it.
            verify_op = kda_verify_ring_conv
            gate_operand = F.reshape(
                self.f_b_proj(gate_down),
                [1, total_seq_len, self.num_heads, self.head_dim],
            )
        a_log, dt_bias = self._gate_params()
        normed, window = verify_op(
            packed_qkvg,
            self._conv_weight(),
            conv_pool,
            conv_row_id,
            gate_operand,
            a_log,
            dt_bias,
            input_row_offsets,
            state_pool,
            recurrent_row_id,
            binding.ring_pool,
            binding.ring_row_id,
            binding.verify_width,
            self.o_norm.weight.cast(packed_qkvg.dtype),
            out_type,
            gate_mode="safe",
            norm_eps=self.o_norm.eps,
        )
        _record_conv_replay(binding, window)
        if tp_active:
            normed = normed.rebind_mapping(
                NamedMapping(normed.mesh, (None, None, TP, None))
            )
        elif mesh.num_devices > 1:
            normed = normed.rebind_mapping(
                NamedMapping(normed.mesh, (None, None, None, None))
            )
        # Drop the op's leading axis before the merge, as the unfused path's
        # norm output does: the sharding rule for the 2-D merge expects the
        # sharded head axis adjacent to it, not one axis further out.
        normed = F.reshape(
            normed, [total_seq_len, self.num_heads, self.head_dim]
        )
        return self.o_proj(
            F.reshape(normed, [total_seq_len, self.num_heads * self.head_dim])
        )

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
        mesh = self.q_proj.weight.mesh
        tp_active = TP in mesh.axis_names and mesh.axis_size(TP) > 1
        tp_size = mesh.axis_size(TP) if tp_active else 1

        # One GEMM for every projection that reads the layer input: separate
        # GEMMs cannot share a launch, and at decode row counts the gate's
        # `head_dim`-row down projection was a whole launch of its own. The
        # split sizes are GLOBAL (a distributed `split` sections each rank's
        # own piece), so each rank lands its own q/k/v block, its own g, its
        # own beta heads, and -- off the `tp_size *` width -- the whole gate
        # row, which `f_b_proj` then reduces per head.
        weight = self._packed_qkvg_weight()
        packed_qkvg = x.cast(weight.dtype) @ weight.T
        qkv, gate_pre, beta_pre, gate_down = packed_qkvg.split(
            [3 * inner, inner, self.num_heads, tp_size * self.head_dim],
            axis=1,
        )
        if tp_active:
            # `f_a_proj`'s rows are replicated, so its slice is whole on
            # every device despite the GEMM's sharded columns: restate that
            # placement (the `split` kept the shards local, a slice would
            # have gathered them).
            gate_down = gate_down.rebind_mapping(
                NamedMapping(gate_down.mesh, (None, None))
            )
        if binding is not None and self._fused_verify_eligible(
            packed_qkvg, input_row_offsets, binding.verify_width
        ):
            # The fused verify arm: one launch covers the conv, the glue,
            # the ring and the norm (see the module docstring).
            return self._fused_verify_forward(
                x,
                packed_qkvg,
                gate_down,
                binding,
                conv_pool,
                conv_row_id,
                state_pool,
                recurrent_row_id,
                input_row_offsets,
            )
        conv_weight = self._conv_weight()
        if binding is None:
            qkv = kimi_short_conv_qkv(
                qkv, conv_weight, conv_pool, conv_row_id, input_row_offsets
            )
        else:
            conv_out, window = kimi_short_conv_qkv_verify(
                qkv,
                conv_weight,
                conv_pool,
                conv_row_id,
                input_row_offsets,
                binding.verify_width,
            )
            _record_conv_replay(binding, window)
            qkv = conv_out
        if tp_active:
            # Custom ops have no sharding rule, so their outputs need their
            # placement restated. `split` keeps the shards local; slicing
            # would gather them.
            qkv = qkv.rebind_mapping(NamedMapping(qkv.mesh, (None, TP)))
        elif mesh.num_devices > 1:
            # Same restatement, for a mesh with no `tp` axis (pure EP).
            # Here the packed weight is replicated rather than sharded, but
            # the custom op's output still needs the mapping restated.
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
            self.f_b_proj(gate_down),
            [1, total_seq_len, self.num_heads, self.head_dim],
        )
        # `kda_decode` takes float32 beta logits; the fused GEMM's
        # model-dtype beta block widens here.
        beta_logits = F.reshape(
            F.cast(beta_pre, DType.float32),
            [1, total_seq_len, self.num_heads],
        )

        a_log, dt_bias = self._gate_params()
        a_log_f32 = F.cast(a_log, DType.float32)
        dt_bias_f32 = F.cast(dt_bias, DType.float32)
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
            gate_pre, [total_seq_len, self.num_heads, self.head_dim]
        )
        # `o_norm` consumes the fp32 recurrence output and rounds once to
        # the model dtype inside the fused norm (via `out_dtype`), so
        # `o_proj` sees the model dtype with no separate cast launch.
        # Called as the function rather than through the module's forward
        # because the output dtype is this layer's per-call choice.
        normed = kimi_rms_norm_gated(
            o,
            self.o_norm.weight.cast(x.dtype),
            gate,
            self.o_norm.eps,
            out_dtype=x.dtype,
        )
        return self.o_proj(F.reshape(normed, [total_seq_len, inner]))


def tensor_parallel_kda_attention(
    layer: KimiK3KDAAttention, mesh: DeviceMesh
) -> KimiK3KDAAttention:
    """Shards a KDA layer by head along the ``tp`` mesh axis.

    ``f_a_proj`` (the gate's rank-``head_dim`` down projection) stays
    replicated, because every rank's ``f_b_proj`` slice needs the full
    ``head_dim`` gate row; so do ``o_norm``, which normalizes within a head,
    and ``a_log`` and ``dt_bias``, which the graph shards once it lays them
    out per channel. ``o_proj`` is sharded on its input dim, so the caller
    must allreduce the output.
    """
    tp_size = mesh.axis_size(TP)
    if tp_size == 1:
        return layer
    check_divisible("KDA head count", layer.num_heads, tp_size)

    row_mapping = NamedMapping(mesh, (TP, None))
    for proj in (
        layer.q_proj,
        layer.k_proj,
        layer.v_proj,
        layer.g_proj,
        layer.b_proj,
        layer.f_b_proj,
    ):
        proj.weight = proj.weight.to(row_mapping)
    conv_mapping = NamedMapping(mesh, (TP, None, None))
    layer.q_conv1d = layer.q_conv1d.to(conv_mapping)
    layer.k_conv1d = layer.k_conv1d.to(conv_mapping)
    layer.v_conv1d = layer.v_conv1d.to(conv_mapping)
    layer.o_proj.weight = layer.o_proj.weight.to(NamedMapping(mesh, (None, TP)))
    return layer
