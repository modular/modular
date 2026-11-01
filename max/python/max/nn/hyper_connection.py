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

"""Provides a manifold-constrained hyper-connection layer."""

from __future__ import annotations

from max import functional as F
from max.driver import accelerator_api
from max.dtype import DType
from max.functional import random
from max.nn.legacy.kernels import mhc_pre_supports
from max.tensor import Tensor

from .common_layers.functional_kernels import hyper_connection_gates, mhc_pre
from .module import Module
from .norm import RMSNorm


# `...` because a parameter list cannot mark `forward`'s `norm` optional:
# `[Tensor, RMSNorm | None]` would make every caller pass it.
class HyperConnection(Module[..., tuple[Tensor, Tensor, Tensor]]):
    r"""Mixes ``hc_mult`` parallel residual streams around a transformer sublayer.

    Manifold-Constrained Hyper-Connections (mHC) generalize the residual
    connection: instead of one stream carrying the residual between sublayers,
    ``hc_mult`` streams run in parallel, and a learned gate decides how they
    collapse into the sublayer's input and how the sublayer's output is written
    back. Reference: Xie et al. 2026, section 2.2 equation 8.

    A single projection of the normalized streams produces the three gates:

    - ``pre``: how the parallel streams collapse into the single sequence the
      sublayer consumes. Applied here, producing ``collapsed``.
    - ``post``: where the sublayer's output lands across the streams, in
      ``[0, 2]``. Returned for the caller to apply.
    - ``comb``: an ``hc_mult x hc_mult`` mixer over the streams, projected onto
      the doubly-stochastic manifold by Sinkhorn-Knopp. Returned for the caller
      to apply.

    A decoder layer holds two of these, one for the attention site and one for
    the feed-forward site. The caller drives the surrounding residual update::

        post, comb, collapsed = hc(hidden_streams)
        sublayer_out = sublayer(collapsed)
        hidden_streams = comb @ hidden_streams + post.unsqueeze(-1) * sublayer_out

    The gate math always runs in float32. ``collapsed`` is cast back to the
    input's dtype. Where :func:`~max.nn.legacy.kernels.mhc_pre_supports` accepts the
    shape and device, one fused op computes the whole site.

    Args:
        hidden_size: Width of a single residual stream.
        hc_mult: Number of parallel residual streams. Must be a power of two
            whose square fits in one warp.
        hc_sinkhorn_iters: Sinkhorn-Knopp iterations used to project ``comb``.
        hc_eps: Epsilon guarding the Sinkhorn divisions.
        rms_norm_eps: Epsilon of the unweighted RMSNorm applied to the streams
            before the projection.
        hc_fn_dtype: Storage dtype of ``hc_fn``, float32 or bfloat16. The
            fused op reads a bfloat16 projection in one pass instead of
            three; elsewhere it is widened to float32 in the graph.
    """

    hc_fn: Tensor
    """Stream projection of shape ``[2 * hc_mult + hc_mult ** 2, hc_mult * hidden_size]``."""

    hc_base: Tensor
    """Per-output bias of shape ``[2 * hc_mult + hc_mult ** 2]``, in ``pre``, ``post``, ``comb`` order."""

    hc_scale: Tensor
    """Per-output scale of shape ``[3]``, in ``pre``, ``post``, ``comb`` order."""

    def __init__(
        self,
        hidden_size: int,
        hc_mult: int,
        *,
        hc_sinkhorn_iters: int = 20,
        hc_eps: float = 1e-6,
        rms_norm_eps: float = 1e-6,
        hc_fn_dtype: DType = DType.float32,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        self.rms_norm_eps = rms_norm_eps

        mix_width = 2 * hc_mult + hc_mult**2
        self.hc_fn = random.normal(
            [mix_width, hc_mult * hidden_size], dtype=hc_fn_dtype
        )
        self.hc_base = Tensor.zeros([mix_width], dtype=DType.float32)
        self.hc_scale = Tensor.ones([3], dtype=DType.float32)

    def __rich_repr__(self):
        """Yields fields for the rich debug repr."""
        yield "hidden_size", self.hidden_size
        yield "hc_mult", self.hc_mult
        yield "hc_sinkhorn_iters", self.hc_sinkhorn_iters, 20

    def forward(
        self, hidden_streams: Tensor, norm: RMSNorm | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Computes the mHC gates and collapses the streams for the sublayer.

        Args:
            hidden_streams: The parallel residual streams, shaped
                ``[total_seq_len, hc_mult, hidden_size]``. The batch and
                sequence axes are folded into the leading axis.
            norm: The sublayer's input norm, applied to ``collapsed``. The
                fused op applies a plain :class:`RMSNorm` itself.

        Returns:
            A tuple ``(post, comb, collapsed)``. ``post`` is float32
            ``[total_seq_len, hc_mult]``, ``comb`` is float32
            ``[total_seq_len, hc_mult, hc_mult]``, and ``collapsed`` is
            ``[total_seq_len, hidden_size]`` in ``hidden_streams``' dtype,
            normalized by ``norm`` if given.

        Raises:
            ValueError: If ``hidden_streams`` is not rank 3 or its trailing two
                dimensions do not match ``hc_mult`` and ``hidden_size``.
        """
        if hidden_streams.rank != 3:
            raise ValueError(
                "expected hidden_streams of rank 3 "
                "[total_seq_len, hc_mult, hidden_size], got rank "
                f"{hidden_streams.rank}"
            )
        if hidden_streams.shape[1] != self.hc_mult:
            raise ValueError(
                f"expected hidden_streams with {self.hc_mult} streams, got"
                f" {hidden_streams.shape[1]}"
            )
        if hidden_streams.shape[2] != self.hidden_size:
            raise ValueError(
                f"expected hidden_streams of width {self.hidden_size}, got"
                f" {hidden_streams.shape[2]}"
            )

        if accelerator_api() == "hip" and mhc_pre_supports(
            self.hc_mult,
            self.hidden_size,
            hidden_streams.dtype,
            self.hc_fn.dtype,
            hidden_streams.mesh.devices[0],
        ):
            gamma: Tensor | None = None
            norm_eps: float | None = None
            # The op normalizes as RMSNorm does: it rounds the normalized row
            # to the streams' dtype, then scales it by the weight.
            if (
                type(norm) is RMSNorm
                and norm.weight.dtype == hidden_streams.dtype
            ):
                gamma, norm_eps, norm = norm.weight, norm.eps, None
            # The op runs once per device; its rows keep the streams'
            # placement.
            post, comb, collapsed, _ = (
                out.rebind_mapping(hidden_streams.mapping)
                if hidden_streams.is_distributed
                else out
                for out in mhc_pre(
                    hidden_streams,
                    self.hc_fn,
                    self.hc_base,
                    self.hc_scale,
                    hc_sinkhorn_iters=self.hc_sinkhorn_iters,
                    rms_eps=self.rms_norm_eps,
                    hc_eps=self.hc_eps,
                    gamma=gamma,
                    norm_eps=norm_eps,
                )
            )
            if norm is not None:
                collapsed = norm(collapsed)
            return post, comb, collapsed

        rows = hidden_streams.shape[0]
        streams = hidden_streams.cast(DType.float32)
        hc_fn = self.hc_fn
        if hc_fn.dtype != DType.float32:
            hc_fn = hc_fn.cast(DType.float32)

        flat = streams.reshape([rows, self.hc_mult * self.hidden_size])
        # Unweighted RMSNorm, applied to the projection rather than its input:
        # the same linear map, one scale per row instead of
        # ``hc_mult * hidden_size``, and ``flat`` keeps its input precision
        # into the fp32 matmul, which may run in TF32.
        rms = F.rsqrt((flat * flat).mean(axis=-1) + self.rms_norm_eps)

        mix = (flat @ hc_fn.T) * rms
        # The gates are computed row by row, so on a mesh each device's rows
        # keep the projection's placement.
        pre, post, comb = (
            gate.rebind_mapping(mix.mapping) if mix.is_distributed else gate
            for gate in hyper_connection_gates(
                mix,
                self.hc_base,
                self.hc_scale,
                hc_mult=self.hc_mult,
                hc_eps=self.hc_eps,
                hc_sinkhorn_iters=self.hc_sinkhorn_iters,
            )
        )

        collapsed = (pre.unsqueeze(-1) * streams).sum(axis=1).squeeze(1)
        comb = comb.reshape([rows, self.hc_mult, self.hc_mult])
        collapsed = collapsed.cast(hidden_streams.dtype)
        if norm is not None:
            collapsed = norm(collapsed)
        return post, comb, collapsed
