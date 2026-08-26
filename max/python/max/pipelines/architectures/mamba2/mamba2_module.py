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
"""Mamba2 prefill + step modules for the MAX-graph pipeline.

Mirrors :mod:`max.pipelines.architectures.mamba.mamba_module`'s
``MambaPrefill`` / ``MambaStep`` split for the Mamba2 SSD architecture:

* :class:`Mamba2Prefill` stacks :class:`Mamba2Block` over a full-prompt
  sequence and returns ``(logits, ssm_state_0, ..., ssm_state_{N-1})``.
  The :func:`ssd_chunk_scan_combined` op exposes its post-last-chunk SSM
  state as ``final_state``; the prefill module forwards each layer's
  final state so :class:`Mamba2Model` can seed the SSM cache and the
  subsequent step kernel picks up where prefill left off. Conv state
  is still zero-initialised at the prefill→step boundary (the
  ``causal_conv1d`` wrapper does not surface its rolling tail), which
  is tracked as a separate follow-up.
* :class:`Mamba2Step` consumes the cached per-layer ``(conv_state,
  ssm_state)`` tuples and runs a single-token update via the existing
  :func:`selective_scan_update` decode kernel. The kernel's
  ``(batch, dim, dstate)`` state layout is reached by flattening
  ``(nheads, head_dim) -> d_inner`` at the call site, and the
  per-head Mamba2 ``A`` / ``dt_bias`` / ``D`` vectors are broadcast to
  per-channel ``(d_inner,)`` / ``(d_inner, d_state)`` before calling.

Both modules share the same weight namespace as the standalone
:class:`Mamba2Block` so the existing weight adapter (item 4) loads
unchanged.
"""

from __future__ import annotations

from typing import cast as typing_cast

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.embedding import Embedding
from max.experimental.nn.linear import Linear
from max.experimental.nn.norm import RMSNorm, rms_norm
from max.experimental.nn.sequential import ModuleList
from max.experimental.tensor import Tensor

from ..mamba.functional_ops import (
    causal_conv1d,
    causal_conv1d_update,
    rms_norm_fused_residual,
    selective_scan_update,
)
from .functional_ops import ssd_chunk_scan_combined
from .model_config import Mamba2Config


def _silu(x: Tensor) -> Tensor:
    """SiLU as a plain ``Tensor -> Tensor`` callable (matches mamba2.py)."""
    return typing_cast(Tensor, F.silu(x))


def _softplus(x: Tensor) -> Tensor:
    """Stable softplus: ``relu(x) + log(1 + exp(-|x|))``.

    The naive ``log(1 + exp(x))`` overflows for large positive x in float32.
    The absolute-value form is bit-equivalent under exact arithmetic and
    stays finite for any real input — matches the reference Mamba2 path
    (``F.softplus(dt + dt_bias)``).
    """
    return typing_cast(
        Tensor,
        F.relu(x) + F.log(F.exp(-F.abs(x)) + 1.0),
    )


class _Mamba2MixerForGraph(Module[[Tensor], Tensor]):
    """Internal mixer that exposes ``prefill()`` and ``step()`` methods.

    Mirrors the weight layout and forward of :class:`Mamba2Mixer`
    (``mamba2.mamba2``) so the same weight-adapter state dict loads here
    unchanged. The difference is structural: this version splits prefill
    and step into separate callables so :class:`Mamba2Prefill` and
    :class:`Mamba2Step` can compile independently.
    """

    in_proj: Linear
    out_proj: Linear
    conv1d_weight: Tensor
    A_log: Tensor
    dt_bias: Tensor
    D: Tensor
    norm: RMSNorm

    def __init__(self, config: Mamba2Config) -> None:
        d_model = config.d_model
        d_inner = config.d_inner
        nheads = config.nheads
        ngroups = config.ngroups
        d_state = config.d_state
        headdim = config.headdim
        d_conv = config.d_conv
        conv_dim = config.conv_dim
        d_in_proj = config.d_in_proj

        self._d_model = d_model
        self._d_state = d_state
        self._d_conv = d_conv
        self._d_inner = d_inner
        self._headdim = headdim
        self._nheads = nheads
        self._ngroups = ngroups
        self._chunk_size = config.chunk_size
        self._conv_dim = conv_dim
        self._d_in_proj = d_in_proj
        self._rms_norm_eps = config.rms_norm_eps
        self._use_conv_bias = config.use_conv_bias

        self.in_proj = Linear(d_model, d_in_proj, bias=config.use_bias)
        self.out_proj = Linear(d_inner, d_model, bias=config.use_bias)

        self.conv1d_weight = Tensor.zeros([conv_dim, d_conv])
        if config.use_conv_bias:
            self.conv1d_bias: Tensor | None = Tensor.zeros([conv_dim])
        else:
            self.conv1d_bias = None

        self.A_log = Tensor.zeros([nheads])
        self.dt_bias = Tensor.zeros([nheads])
        self.D = Tensor.zeros([nheads])
        self.norm = RMSNorm(d_inner, eps=config.rms_norm_eps)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("Use prefill() or step() directly")

    def _get_A(self) -> Tensor:
        """Compute ``A = -exp(A_log)`` (per-head scalar of shape ``(nheads,)``)."""
        return -F.exp(self.A_log)

    # -- prefill --------------------------------------------------------

    def prefill(self, u: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run the Mamba2 SSD mixer over a full prefill sequence.

        Args:
            u: ``(batch, seqlen, d_model)`` — token-major hidden states.

        Returns:
            ``(output, final_conv_state, final_ssm_state)`` where:

            * ``output`` is ``(batch, seqlen, d_model)`` — mixer output.
            * ``final_conv_state`` is ``(batch, conv_dim, d_conv-1)`` — the
              last ``d_conv-1`` pre-conv xBC tokens, in the layout the
              decode kernel's rolling conv state expects.
            * ``final_ssm_state`` is
              ``(batch, nheads, head_dim, d_state)`` — the post-last-chunk
              SSM state surfaced by ``ssd_chunk_scan_combined``. Caller
              writes both into the per-slot cache so step-mode decode
              picks up where prefill left off.
        """
        batch = u.shape[0]
        seqlen = u.shape[1]
        d_inner = self._d_inner
        nheads = self._nheads
        ngroups = self._ngroups
        d_state = self._d_state
        headdim = self._headdim
        d_conv = self._d_conv

        zxbcdt = self.in_proj(u)
        z_raw, xBC_raw, dt_raw = F.split(
            zxbcdt,
            [d_inner, d_inner + 2 * ngroups * d_state, nheads],
            axis=-1,
        )
        z = typing_cast(Tensor, z_raw)
        xBC = typing_cast(Tensor, xBC_raw)
        dt = typing_cast(Tensor, dt_raw)

        # Causal conv1d wrapper expects (batch, channels, seqlen).
        xBC_bcs = xBC.permute([0, 2, 1])

        # Slice the last ``d_conv - 1`` pre-conv tokens off the input as the
        # rolling conv state for the next step. The decode kernel
        # (``causal_conv1d_update``) expects ``[batch, conv_dim, d_conv-1]``
        # of *pre-conv, pre-activation* history — exactly a tail slice of
        # the prefill conv input. Padding for ``seqlen < d_conv-1`` is not
        # required: the embedding context always has at least ``d_conv``
        # tokens for any non-degenerate prompt.
        final_conv_state = xBC_bcs[:, :, seqlen - (d_conv - 1) :]

        xBC_bcs = causal_conv1d(
            xBC_bcs,
            self.conv1d_weight,
            bias=self.conv1d_bias,
            activation="silu",
        )
        xBC_post = xBC_bcs.permute([0, 2, 1])  # (B, L, conv_dim)

        x_raw, B_raw, C_raw = F.split(
            xBC_post,
            [d_inner, ngroups * d_state, ngroups * d_state],
            axis=-1,
        )
        x = typing_cast(Tensor, x_raw)
        B = typing_cast(Tensor, B_raw)
        C = typing_cast(Tensor, C_raw)

        x4 = F.reshape(x, [batch, seqlen, nheads, headdim])
        B4 = F.reshape(B, [batch, seqlen, ngroups, d_state])
        C4 = F.reshape(C, [batch, seqlen, ngroups, d_state])
        # Discretization gate: reference Mamba2 applies softplus(dt + dt_bias)
        # before the SSD scan so dt is strictly positive. The raw additive
        # form was a known-incorrect approximation.
        dt_with_bias = _softplus(dt + self.dt_bias)

        A = self._get_A()

        # The SSD kernel requires ``seqlen % chunk_size == 0``. Pad with
        # zeros along the seqlen axis so any prompt length is admissible.
        # Padding ``dt_with_bias`` with zeros makes the SSM scan a no-op
        # at the padded positions (X_disc = x*dt = 0, A_disc = A*dt = 0,
        # decay = exp(0) = 1, so ``final_state`` is exactly the state at
        # the real last token). Y at padded positions is discarded by the
        # post-op slice below; final_state passes through untouched.
        chunk_size = self._chunk_size
        padded_seqlen = ((seqlen + chunk_size - 1) // chunk_size) * chunk_size
        pad_len = padded_seqlen - seqlen

        zero_scalar = typing_cast(
            Tensor, F.constant(0.0, dtype=x4.dtype, device=x4.device)
        )
        x_pad = F.broadcast_to(zero_scalar, [batch, pad_len, nheads, headdim])
        dt_pad = F.broadcast_to(zero_scalar, [batch, pad_len, nheads])
        B_pad = F.broadcast_to(zero_scalar, [batch, pad_len, ngroups, d_state])
        C_pad = F.broadcast_to(zero_scalar, [batch, pad_len, ngroups, d_state])
        x4_p = F.concat([x4, x_pad], axis=1)
        dt_p = F.concat([dt_with_bias, dt_pad], axis=1)
        B4_p = F.concat([B4, B_pad], axis=1)
        C4_p = F.concat([C4, C_pad], axis=1)

        y, final_state = ssd_chunk_scan_combined(
            x=x4_p.__tensorvalue__(),
            dt=dt_p.__tensorvalue__(),
            A=A.__tensorvalue__(),
            B=B4_p.__tensorvalue__(),
            C=C4_p.__tensorvalue__(),
            chunk_size=chunk_size,
        )
        y_padded = Tensor.from_graph_value(y)
        # Slice y back to the original seqlen so downstream shapes
        # (gate, RMSNorm, out_proj) see the real length.
        y_tensor = y_padded[:, :seqlen, :, :]
        final_state_tensor = Tensor.from_graph_value(final_state)

        # D skip-connection: reference adds ``D * x`` to y per head before the
        # gate/norm. ``D`` is per-head ``(nheads,)``; broadcast over ``(B, L,
        # H, P)``. The kernel does not consume ``D`` yet; applying here keeps
        # the per-head semantic correct.
        D_b = F.reshape(self.D, [1, 1, nheads, 1])
        y_with_D = y_tensor + D_b * x4

        y_flat = F.reshape(y_with_D, [batch, seqlen, d_inner])

        # Gate then RMSNorm then out_proj — matches the reference's
        # ``RMSNormGated(..., norm_before_gate=False)`` semantics, which is
        # algorithmically ``rms_norm(y * silu(z))``. The reference fuses
        # the gate * norm * silu pass for performance and upcasts to fp32
        # internally; we keep the unfused form here and accept the small
        # bf16 precision delta.
        z_gate = _silu(z)
        y_gated = y_flat * z_gate
        y_norm = self.norm(y_gated)
        out = typing_cast(Tensor, self.out_proj(y_norm))
        return out, final_conv_state, final_state_tensor

    # -- step -----------------------------------------------------------

    def step(
        self,
        u: Tensor,
        conv_state: Tensor,
        ssm_state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Single-token Mamba2 SSD update using cached states.

        Args:
            u: ``(batch, d_model)`` — newest token hidden state.
            conv_state: ``(batch, conv_dim, d_conv - 1)`` — rolling
                pre-conv history for the xBC slot.
            ssm_state: ``(batch, nheads, head_dim, d_state)`` — per-head
                SSM state. Reshaped to ``(batch, d_inner, d_state)``
                before calling :func:`selective_scan_update`, then
                reshaped back on return so the cache layout stays the
                Mamba2-native per-head one.

        Returns:
            ``(output, updated_conv_state, updated_ssm_state)`` with the
            same shapes as the inputs (mixer output is
            ``(batch, d_model)``).
        """
        batch = u.shape[0]
        d_inner = self._d_inner
        nheads = self._nheads
        ngroups = self._ngroups
        d_state = self._d_state
        headdim = self._headdim
        conv_dim = self._conv_dim

        # 1) in_proj on a single token.
        zxbcdt = self.in_proj(u)
        z_raw, xBC_raw, dt_raw = F.split(
            zxbcdt,
            [d_inner, d_inner + 2 * ngroups * d_state, nheads],
            axis=-1,
        )
        z = typing_cast(Tensor, z_raw)  # (B, d_inner)
        xBC = typing_cast(Tensor, xBC_raw)  # (B, conv_dim)
        dt = typing_cast(Tensor, dt_raw)  # (B, nheads)

        # 2) Causal conv1d update. Wrapper expects
        # (batch, channels, 1) input.
        xBC_3d = F.reshape(xBC, [batch, conv_dim, 1])
        x_conv, updated_conv = causal_conv1d_update(
            xBC_3d,
            conv_state,
            self.conv1d_weight,
            bias=self.conv1d_bias,
            activation="silu",
        )
        # Drop the singleton seq dim -> (B, conv_dim).
        x_flat_conv = F.reshape(x_conv, [batch, conv_dim])

        # 3) Split conv-updated xBC into x, B, C.
        x_raw, B_raw, C_raw = F.split(
            x_flat_conv,
            [d_inner, ngroups * d_state, ngroups * d_state],
            axis=-1,
        )
        x_flat = typing_cast(Tensor, x_raw)  # (B, d_inner)
        B_lin = typing_cast(Tensor, B_raw)  # (B, ngroups*d_state)
        C_lin = typing_cast(Tensor, C_raw)  # (B, ngroups*d_state)

        B_g = F.reshape(B_lin, [batch, ngroups, d_state])
        C_g = F.reshape(C_lin, [batch, ngroups, d_state])

        # 4) Broadcast per-head Mamba2 weights to the per-channel layout
        #    that ``selective_scan_update`` expects.
        #
        #    The kernel signature is documented in
        #    ``selective_scan_ops.mojo``: state (batch, dim, dstate),
        #    A (dim, dstate), dt (batch, dim), D (dim,), z (batch, dim).
        #    Mamba2 stores ``A_log`` / ``dt_bias`` / ``D`` as
        #    ``(nheads,)`` and the SSM state as
        #    ``(batch, nheads, head_dim, d_state)``. With
        #    ``ngroups == nheads`` (enforced upstream), the kernel's
        #    ``group_size = dim // n_groups`` works out to ``headdim``
        #    which is exactly what we want.
        #
        #    Broadcast pattern:
        #      A:       (nheads,)        -> (nheads, 1, 1)   -> (d_inner, d_state)
        #      dt_bias: (nheads,)        -> (1, nheads, 1)   -> (B, d_inner)
        #      D:       (nheads,)        -> (1, nheads, 1)   -> (B, d_inner)  [bias slot]
        #      dt:      (B, nheads)      -> (B, nheads, 1)   -> (B, d_inner)
        #    The repeats are along the per-head ``headdim`` axis.
        dt_3d = F.reshape(dt, [batch, nheads, 1])
        dt_full = F.reshape(
            dt_3d.broadcast_to([batch, nheads, headdim]),
            [batch, d_inner],
        )

        A_per_head = self._get_A()  # (nheads,)
        A_2d = F.reshape(A_per_head, [nheads, 1, 1])
        A_full = F.reshape(
            A_2d.broadcast_to([nheads, headdim, d_state]),
            [d_inner, d_state],
        )

        D_3d = F.reshape(self.D, [1, nheads, 1])
        D_full = F.reshape(
            D_3d.broadcast_to([batch, nheads, headdim]),
            [batch, d_inner],
        ).cast(self.D.dtype)
        # ``selective_scan_update`` consumes D as ``(dim,)``; the kernel
        # iterates over the batch dim itself so for batch=1 we can just
        # take the first row. For batch>1 every row is identical (the
        # bias is data-independent) so a single-row slice is correct.
        D_vec = D_full[0]  # (d_inner,)

        dt_bias_3d = F.reshape(self.dt_bias, [1, nheads, 1])
        dt_bias_full = F.reshape(
            dt_bias_3d.broadcast_to([batch, nheads, headdim]),
            [batch, d_inner],
        )
        dt_bias_vec = dt_bias_full[0]  # (d_inner,)

        # 5) Flatten the per-head SSM state to the kernel layout.
        ssm_state_2d = F.reshape(ssm_state, [batch, d_inner, d_state])

        # 6) Run the existing decode kernel.
        updated_ssm_2d, y_flat = selective_scan_update(
            state=ssm_state_2d,
            x=x_flat,
            dt=dt_full,
            A=A_full,
            B=B_g,
            C=C_g,
            D=D_vec,
            z=z,
            dt_bias=dt_bias_vec,
            dt_softplus=True,
        )

        # 7) Reshape SSM state back to the cache's per-head layout.
        updated_ssm = F.reshape(
            updated_ssm_2d, [batch, nheads, headdim, d_state]
        )

        # 8) Norm + out_proj. The mixer's ``norm`` weight lives over
        #    ``d_inner``; we use the unfused path since we already have
        #    the gate baked into the scan (the kernel applies ``z`` as
        #    a gate when supplied).
        y_norm = self.norm(y_flat)
        out = typing_cast(Tensor, self.out_proj(y_norm))
        return out, updated_conv, updated_ssm


class _Mamba2LayerForGraph(Module[[Tensor], Tensor]):
    """Pre-norm + mixer pair for a single Mamba2 block in graph land."""

    norm: RMSNorm
    mixer: _Mamba2MixerForGraph

    def __init__(self, config: Mamba2Config) -> None:
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.mixer = _Mamba2MixerForGraph(config)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("Use Mamba2Prefill / Mamba2Step")


class _Mamba2Base(Module[[Tensor, Tensor], tuple[Tensor, ...]]):
    """Shared weight structure for prefill and step graph modules."""

    embedding: Embedding
    layers: ModuleList[_Mamba2LayerForGraph]
    norm: RMSNorm

    def __init__(self, config: Mamba2Config) -> None:
        self.embedding = Embedding(config.padded_vocab_size, dim=config.d_model)
        self.layers = ModuleList(
            [_Mamba2LayerForGraph(config) for _ in range(config.n_layer)]
        )
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self._num_layers = config.n_layer
        self._residual_in_fp32 = config.residual_in_fp32

    def forward(self, tokens: Tensor, aux: Tensor) -> tuple[Tensor, ...]:
        raise NotImplementedError("Use Mamba2Prefill or Mamba2Step")

    def _apply_layer_norm(
        self,
        h: Tensor,
        residual: Tensor,
        norm_weight: Tensor,
        eps: float,
        layer_idx: int,
    ) -> tuple[Tensor, Tensor]:
        """Norm with fused residual after the first layer (mirrors Phase-1)."""
        if layer_idx == 0:
            return rms_norm(h, norm_weight, eps), h
        if self._residual_in_fp32:
            h_fp32 = h.cast(DType.float32)
            res_fp32 = residual.cast(DType.float32)
            h_normed, residual = rms_norm_fused_residual(
                h_fp32, res_fp32, norm_weight, eps
            )
            return h_normed.cast(h.dtype), residual.cast(h.dtype)
        return rms_norm_fused_residual(h, residual, norm_weight, eps)


class Mamba2Prefill(_Mamba2Base):
    """Prefill graph: full-sequence forward returning logits + per-layer states.

    Output layout is ``(logits, conv_0, ssm_0, conv_1, ssm_1, ...)`` matching the
    Mamba1 interleaved convention. The model wrapper writes both halves into the
    per-slot cache so step-mode decode resumes from the exact state the prefill
    recurrence ended on. Conv state is sliced from the last ``d_conv-1`` pre-conv
    xBC tokens; SSM state is surfaced by ``ssd_chunk_scan_combined.final_state``.
    """

    def forward(
        self, tokens: Tensor, input_row_offsets: Tensor
    ) -> tuple[Tensor, ...]:
        h = self.embedding(tokens)
        # Add a synthetic batch axis so the mixer's (batch, seqlen, ...)
        # shape contract holds: Mamba1 used a flat (seqlen, hidden)
        # layout because its functional ops are batch-1 implicit; the
        # Mamba2 SSD wrapper is explicitly batched.
        seqlen = h.shape[0]
        hidden = h.shape[1]
        h = F.reshape(h, [1, seqlen, hidden])

        layer_states: list[Tensor] = []
        residual = h  # placeholder; overwritten by _apply_layer_norm.
        for i in range(self._num_layers):
            layer = self.layers[i]
            h_normed, residual = self._apply_layer_norm(
                h, residual, layer.norm.weight, layer.norm.eps, i
            )
            h, conv_state, ssm_state = layer.mixer.prefill(h_normed)
            layer_states.append(conv_state)
            layer_states.append(ssm_state)

        h = h + residual
        h = rms_norm(h, self.norm.weight, self.norm.eps)

        # Drop the synthetic batch axis -> (seqlen, hidden).
        h = F.reshape(h, [seqlen, hidden])

        # Gather last-token hidden states per batch element.
        last_indices = input_row_offsets[1:] - 1
        last_h = F.gather(h, last_indices, axis=0)

        # Tied lm_head via embedding.weight.T.
        logits = (last_h @ self.embedding.weight.T).cast(DType.float32)
        return (logits, *layer_states)


class Mamba2Step(_Mamba2Base):
    """Step graph for Mamba2: single-token update with cached states.

    Forward signature: ``(tokens, *layer_states) -> (logits, *updated_states)``
    matching :class:`max.pipelines.architectures.mamba.mamba_module.MambaStep`.
    The layer_states list is laid out as
    ``[conv_0, ssm_0, conv_1, ssm_1, ...]`` per Mamba1 convention.
    """

    def forward(
        self, tokens: Tensor, *layer_states: Tensor
    ) -> tuple[Tensor, ...]:
        num_layers = self._num_layers
        if len(layer_states) != 2 * num_layers:
            raise ValueError(
                f"Mamba2Step expected {2 * num_layers} layer state "
                f"tensors, got {len(layer_states)}"
            )
        conv_states = [layer_states[2 * i] for i in range(num_layers)]
        ssm_states = [layer_states[2 * i + 1] for i in range(num_layers)]

        h = self.embedding(tokens)
        # (batch, d_model) -> (batch, 1, d_model) so the in_proj path is
        # symmetric with the prefill mixer's token-major layout.
        batch = h.shape[0]
        hidden = h.shape[1]
        h = F.reshape(h, [batch, 1, hidden])

        updated: list[Tensor] = []
        residual = h
        for i in range(num_layers):
            layer = self.layers[i]
            h_normed, residual = self._apply_layer_norm(
                h, residual, layer.norm.weight, layer.norm.eps, i
            )
            # The mixer's step path consumes (batch, d_model); drop the
            # synthetic seq axis.
            h_normed_2d = F.reshape(h_normed, [batch, hidden])
            out_2d, conv_s, ssm_s = layer.mixer.step(
                h_normed_2d, conv_states[i], ssm_states[i]
            )
            # Re-add the synthetic seq axis for the residual chain.
            h = F.reshape(out_2d, [batch, 1, hidden])
            updated.append(conv_s)
            updated.append(ssm_s)

        h = h + residual
        h = rms_norm(h, self.norm.weight, self.norm.eps)
        h_2d = F.reshape(h, [batch, hidden])

        logits = (h_2d @ self.embedding.weight.T).cast(DType.float32)
        return (logits, *updated)
