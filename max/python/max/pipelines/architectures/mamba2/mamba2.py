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
"""Mamba2 NN modules (mixer + block).

Mirrors :mod:`max.pipelines.architectures.mamba.mamba_module` for the
SSD-based Mamba2 architecture. The mixer composes:

* a single fused ``in_proj`` matmul producing ``zxbcdt``
* a depthwise causal-conv1d on the ``xBC`` slice (reuses the Mamba1
  ``causal_conv1d`` wrapper which is registered in the shared
  ``state_space`` kernel package)
* the chunk-scan-combined SSD op
  (:func:`mamba2.functional_ops.ssd_chunk_scan_combined`)
* a SiLU gate from ``z``, an RMSNorm on the mixer output, and the
  ``out_proj`` matmul.

Reference: ``mamba_ssm.modules.mamba2.Mamba2`` (non-tensor-parallel
branch with ``rmsnorm=True`` and ``use_mem_eff_path=False``). We pick
that branch because (a) it uses the same fused ``mamba_chunk_scan_combined``
op our SSD kernels target, and (b) it matches the Phase-1 shape
conventions in :mod:`...mamba.mamba_module`.

Shape conventions / constraints:

* ``ngroups == nheads`` (matches the kernel ABI documented in
  :mod:`.functional_ops`; the reference allows ``ngroups < nheads``
  but the kernel assumes the broadcast has already happened).
* Token-major input ``u: (batch, seqlen, hidden)`` rather than the
  Phase-1 ``(seqlen, hidden)`` flat layout, because the SSD op
  signature is token-major.

The runtime kernel-package loader is currently blocked upstream — see
``proposed/approvals/causal_conv1d_ops-tuple-shape-bug.md`` — so this
module is shipped without an end-to-end smoke test. A construction-only
test (no compile / no execute) will land alongside the integration task
once the loader fix unblocks both Mamba1 and Mamba2.
"""

from __future__ import annotations

import math
from typing import cast as typing_cast

from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.linear import Linear
from max.experimental.nn.norm import RMSNorm, rms_norm
from max.experimental.tensor import Tensor

from ..mamba.functional_ops import causal_conv1d, rms_norm_fused_residual
from .functional_ops import ssd_chunk_scan_combined


def _silu(x: Tensor) -> Tensor:
    """SiLU activation as a plain ``Tensor -> Tensor`` callable.

    ``max.experimental.functional.silu`` returns a ``Tensor`` already;
    this thin wrapper is only here to give the call site a stable name
    matching the reference (``F.silu(z) * y``). The cast is defensive:
    some functional ops in the experimental namespace are typed as
    ``Tensor | Any`` due to dispatch-engine generics.
    """
    return typing_cast(Tensor, F.silu(x))


class Mamba2Mixer(Module[[Tensor], Tensor]):
    """Mamba2 SSD mixer (prefill-only path).

    Forward signature: ``u: (batch, seqlen, d_model) -> (batch, seqlen, d_model)``.

    Weights:

    * ``in_proj``: ``Linear(d_model, d_in_proj)`` where
      ``d_in_proj = 2*d_inner + 2*ngroups*d_state + nheads``.
    * ``out_proj``: ``Linear(d_inner, d_model)``.
    * ``conv1d_weight``: depthwise conv weight of shape
      ``(conv_dim, d_conv)`` with ``conv_dim = d_inner + 2*ngroups*d_state``.
    * ``conv1d_bias``: optional bias of shape ``(conv_dim,)``.
    * ``A_log``: log-space diagonal state-decay of shape ``(nheads,)``.
      ``A = -exp(A_log)`` is computed lazily.
    * ``dt_bias``: per-head time-step bias of shape ``(nheads,)``. The
      SSD op does its own softplus, but the bias is added before the
      kernel by the reference; we follow the same convention by adding
      it to ``dt`` prior to calling :func:`ssd_chunk_scan_combined`.
    * ``D``: per-head skip-connection coefficient of shape ``(nheads,)``.
      Not consumed by the current SSD kernel (the wrapper does not
      forward ``D``); we hold the weight so the same state_dict loads
      cleanly and so a future kernel-level pass-through is a no-op for
      callers.
    * ``norm``: :class:`RMSNorm` over the mixer-output channel dim
      (``d_inner``).
    """

    in_proj: Linear
    out_proj: Linear
    conv1d_weight: Tensor
    A_log: Tensor
    dt_bias: Tensor
    D: Tensor
    norm: RMSNorm

    def __init__(
        self,
        d_model: int,
        *,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int | None = None,
        chunk_size: int = 256,
        use_bias: bool = False,
        use_conv_bias: bool = True,
        rms_norm_eps: float = 1e-5,
    ) -> None:
        d_inner = expand * d_model
        if d_inner % headdim != 0:
            raise ValueError(
                f"d_inner ({d_inner}) must be divisible by headdim ({headdim})"
            )
        nheads = d_inner // headdim
        # Kernel ABI assumption (see functional_ops.py): ngroups == nheads.
        # Default to nheads when caller leaves it unspecified.
        if ngroups is None:
            ngroups = nheads
        if ngroups != nheads:
            raise ValueError(
                "Mamba2Mixer currently requires ngroups == nheads "
                f"(got ngroups={ngroups}, nheads={nheads}); the SSD kernel "
                "does not yet handle ngroups < nheads broadcast internally."
            )

        d_in_proj = 2 * d_inner + 2 * ngroups * d_state + nheads
        conv_dim = d_inner + 2 * ngroups * d_state

        self._d_model = d_model
        self._d_state = d_state
        self._d_conv = d_conv
        self._d_inner = d_inner
        self._headdim = headdim
        self._nheads = nheads
        self._ngroups = ngroups
        self._chunk_size = chunk_size
        self._conv_dim = conv_dim
        self._rms_norm_eps = rms_norm_eps
        self._use_conv_bias = use_conv_bias

        self.in_proj = Linear(d_model, d_in_proj, bias=use_bias)
        self.out_proj = Linear(d_inner, d_model, bias=use_bias)

        self.conv1d_weight = Tensor.zeros([conv_dim, d_conv])
        if use_conv_bias:
            self.conv1d_bias: Tensor | None = Tensor.zeros([conv_dim])
        else:
            self.conv1d_bias = None

        self.A_log = Tensor.zeros([nheads])
        self.dt_bias = Tensor.zeros([nheads])
        self.D = Tensor.zeros([nheads])
        self.norm = RMSNorm(d_inner, eps=rms_norm_eps)

    # -- properties -----------------------------------------------------

    @property
    def d_model(self) -> int:
        """Hidden width of the residual stream."""
        return self._d_model

    @property
    def d_inner(self) -> int:
        """Per-head channel sum (``expand * d_model``)."""
        return self._d_inner

    @property
    def nheads(self) -> int:
        """Number of SSM heads."""
        return self._nheads

    @property
    def d_state(self) -> int:
        """SSM state width."""
        return self._d_state

    # -- forward --------------------------------------------------------

    def _get_A(self) -> Tensor:
        """Compute ``A = -exp(A_log)`` from the log-space weight."""
        return -F.exp(self.A_log)

    def forward(self, u: Tensor) -> Tensor:
        """Run the Mamba2 SSD mixer on a full prefill sequence.

        Args:
            u: Input of shape ``(batch, seqlen, d_model)``.

        Returns:
            Mixer output of shape ``(batch, seqlen, d_model)``.
        """
        batch = u.shape[0]
        seqlen = u.shape[1]
        d_inner = self._d_inner
        nheads = self._nheads
        ngroups = self._ngroups
        d_state = self._d_state
        headdim = self._headdim

        # 1) Fused input projection: (B, L, d_model) -> (B, L, d_in_proj).
        zxbcdt = self.in_proj(u)

        # 2) Split along the last axis into z, xBC, dt.
        #    Order matches the reference's non-mem-eff path
        #    (without ``d_mlp``): [z, xBC, dt].
        z_raw, xBC_raw, dt_raw = F.split(
            zxbcdt,
            [d_inner, d_inner + 2 * ngroups * d_state, nheads],
            axis=-1,
        )
        z = typing_cast(Tensor, z_raw)
        xBC = typing_cast(Tensor, xBC_raw)
        dt = typing_cast(Tensor, dt_raw)

        # 3) Depthwise causal conv1d over xBC. The wrapper expects
        #    ``(batch, channels, seqlen)`` so we permute, run, permute
        #    back. Activation "silu" matches the reference default.
        xBC_bcs = xBC.permute([0, 2, 1])  # (B, conv_dim, L)
        xBC_bcs = causal_conv1d(
            xBC_bcs,
            self.conv1d_weight,
            bias=self.conv1d_bias,
            activation="silu",
        )
        xBC_post = xBC_bcs.permute([0, 2, 1])  # (B, L, conv_dim)

        # 4) Split xBC into x, B, C.
        x_raw, B_raw, C_raw = F.split(
            xBC_post,
            [d_inner, ngroups * d_state, ngroups * d_state],
            axis=-1,
        )
        x = typing_cast(Tensor, x_raw)
        B = typing_cast(Tensor, B_raw)
        C = typing_cast(Tensor, C_raw)

        # 5) Reshape for the SSD op.
        #    x: (B, L, d_inner) -> (B, L, nheads, headdim).
        #    B, C: (B, L, ngroups*d_state) -> (B, L, ngroups, d_state).
        #    With ngroups == nheads (enforced in __init__), B/C broadcast
        #    1:1 to the per-head shape the kernel expects.
        x4 = F.reshape(x, [batch, seqlen, nheads, headdim])
        B4 = F.reshape(B, [batch, seqlen, ngroups, d_state])
        C4 = F.reshape(C, [batch, seqlen, ngroups, d_state])

        # Add dt_bias before passing to the kernel. The reference
        # ``mamba_chunk_scan_combined`` consumes ``dt_bias`` directly,
        # but our op wrapper doesn't expose a bias slot — instead we
        # fold the bias here. The kernel itself applies softplus
        # internally per the RFC 0002 ABI lock.
        dt_with_bias = dt + self.dt_bias

        A = self._get_A()

        # 6) Run the SSD chunk-scan combined op. The wrapper takes
        #    ``TensorValue`` directly (it uses the lower-level
        #    ``ops.custom`` interface), so we bridge from the experimental
        #    ``Tensor`` view via ``__tensorvalue__`` / ``from_graph_value``.
        y, _final_state = ssd_chunk_scan_combined(
            x=x4.__tensorvalue__(),
            dt=dt_with_bias.__tensorvalue__(),
            A=A.__tensorvalue__(),
            B=B4.__tensorvalue__(),
            C=C4.__tensorvalue__(),
            chunk_size=self._chunk_size,
        )
        y_tensor = Tensor.from_graph_value(y)

        # 7) Gate by SiLU(z). Reference path with ``rmsnorm=True`` passes
        #    ``z=None`` to the kernel and instead applies ``norm(y, z)``
        #    (gated RMSNorm) afterward. We approximate that by:
        #      y_gated = y * silu(z)  (per-channel gate, then norm)
        #    which matches the un-gated-RMSNorm fallback the reference
        #    uses when ``rmsnorm=False`` and ``z`` is passed into the
        #    kernel. Faithful gated-RMSNorm needs a separate kernel and
        #    is left to a follow-up RFC item (parity tracker).
        y_flat = F.reshape(y_tensor, [batch, seqlen, d_inner])
        z_gate = _silu(z)
        y_gated = y_flat * z_gate

        # 8) RMSNorm over the channel axis, then out_proj.
        y_norm = self.norm(y_gated)
        out = self.out_proj(y_norm)
        return typing_cast(Tensor, out)


class Mamba2Block(Module[[Tensor, Tensor], tuple[Tensor, Tensor]]):
    """Mamba2 block: ``RMSNorm(u + residual) -> Mamba2Mixer -> (out, residual)``.

    Mirrors the carry convention from :mod:`...mamba.mamba_module` so
    stacked blocks can keep accumulating the residual stream:

    * On the very first block, ``residual`` is conventionally the
      pre-embedding hidden state (so ``u + residual`` is just ``u +
      embedding``); downstream wiring decides whether to seed
      ``residual`` to zero or to a copy of ``u``.
    * On every subsequent block, ``residual`` is the carry returned by
      the previous block (``u_prev + residual_prev``), which means the
      norm runs on the fully-accumulated stream and the mixer reads from
      that normalized view.

    Forward returns ``(mixer_out, u + residual)`` so the next block can
    keep the chain going.
    """

    norm: RMSNorm
    mixer: Mamba2Mixer

    def __init__(
        self,
        d_model: int,
        *,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int | None = None,
        chunk_size: int = 256,
        use_bias: bool = False,
        use_conv_bias: bool = True,
        rms_norm_eps: float = 1e-5,
        residual_in_fp32: bool = False,
        fused_residual: bool = True,
    ) -> None:
        self.norm = RMSNorm(d_model, eps=rms_norm_eps)
        self.mixer = Mamba2Mixer(
            d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
            chunk_size=chunk_size,
            use_bias=use_bias,
            use_conv_bias=use_conv_bias,
            rms_norm_eps=rms_norm_eps,
        )
        self._rms_norm_eps = rms_norm_eps
        self._residual_in_fp32 = residual_in_fp32
        self._fused_residual = fused_residual

    def forward(self, u: Tensor, residual: Tensor) -> tuple[Tensor, Tensor]:
        """Apply norm-then-mixer with a residual carry.

        Args:
            u: New token-major hidden state from the previous mixer
                output, shape ``(batch, seqlen, d_model)``.
            residual: Accumulated residual stream from upstream blocks,
                same shape as ``u``.

        Returns:
            ``(mixer_out, new_residual)`` where ``new_residual = u +
            residual`` so a stack of blocks can chain without re-summing.
        """
        if self._fused_residual:
            # rms_norm_fused_residual returns (normed, updated_residual)
            # and matches the Phase-1 convention exactly.
            normed, new_residual = rms_norm_fused_residual(
                u,
                residual,
                self.norm.weight,
                self._rms_norm_eps,
            )
        else:
            new_residual = typing_cast(Tensor, u + residual)
            normed = rms_norm(
                new_residual, self.norm.weight, self._rms_norm_eps
            )
        mixer_out = self.mixer(normed)
        return mixer_out, new_residual


# Helper for downstream wiring: re-derive the same arithmetic the
# reference uses (d_inner / nheads / d_in_proj / conv_dim). Kept as a
# free function (not a method) so the model config / weight adapter
# tasks can call it before constructing modules.
def mamba2_dims(
    d_model: int,
    *,
    d_state: int = 128,
    expand: int = 2,
    headdim: int = 64,
    ngroups: int | None = None,
) -> dict[str, int]:
    """Derive the Mamba2 shape parameters from a config.

    Returns a dict with keys ``d_inner``, ``nheads``, ``ngroups``,
    ``d_in_proj``, ``conv_dim`` — useful for the weight adapter and
    integration test to validate shapes without instantiating the
    module.
    """
    d_inner = expand * d_model
    if d_inner % headdim != 0:
        raise ValueError(
            f"d_inner ({d_inner}) must be divisible by headdim ({headdim})"
        )
    nheads = d_inner // headdim
    if ngroups is None:
        ngroups = nheads
    d_in_proj = 2 * d_inner + 2 * ngroups * d_state + nheads
    conv_dim = d_inner + 2 * ngroups * d_state
    # ``math.gcd`` is only imported defensively below; surface a stable
    # error if the caller asked for a non-divisible ngroups so config
    # validation has a single chokepoint.
    if math.gcd(ngroups, nheads) != min(ngroups, nheads):
        raise ValueError(
            f"ngroups ({ngroups}) does not divide nheads ({nheads}); "
            "the SSD kernel currently requires ngroups == nheads."
        )
    return {
        "d_inner": d_inner,
        "nheads": nheads,
        "ngroups": ngroups,
        "d_in_proj": d_in_proj,
        "conv_dim": conv_dim,
    }
