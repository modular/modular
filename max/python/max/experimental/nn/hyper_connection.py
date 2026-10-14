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

from max.dtype import DType
from max.experimental import functional as F
from max.experimental import random
from max.experimental.tensor import Tensor

from .common_layers.functional_kernels import hyper_connection_gates
from .module import Module


class HyperConnection(Module[[Tensor], tuple[Tensor, Tensor, Tensor]]):
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
    input's dtype.

    Args:
        hidden_size: Width of a single residual stream.
        hc_mult: Number of parallel residual streams. Must be a power of two
            whose square fits in one warp.
        hc_sinkhorn_iters: Sinkhorn-Knopp iterations used to project ``comb``.
        hc_eps: Epsilon guarding the Sinkhorn divisions.
        rms_norm_eps: Epsilon of the unweighted RMSNorm applied to the streams
            before the projection.
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
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        self.rms_norm_eps = rms_norm_eps

        mix_width = 2 * hc_mult + hc_mult**2
        self.hc_fn = random.normal(
            [mix_width, hc_mult * hidden_size], dtype=DType.float32
        )
        self.hc_base = Tensor.zeros([mix_width], dtype=DType.float32)
        self.hc_scale = Tensor.ones([3], dtype=DType.float32)

    def __rich_repr__(self):
        """Yields fields for the rich debug repr."""
        yield "hidden_size", self.hidden_size
        yield "hc_mult", self.hc_mult
        yield "hc_sinkhorn_iters", self.hc_sinkhorn_iters, 20

    def forward(self, hidden_streams: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Computes the mHC gates and collapses the streams for the sublayer.

        Args:
            hidden_streams: The parallel residual streams, shaped
                ``[total_seq_len, hc_mult, hidden_size]``. The batch and
                sequence axes are folded into the leading axis.

        Returns:
            A tuple ``(post, comb, collapsed)``. ``post`` is float32
            ``[total_seq_len, hc_mult]``, ``comb`` is float32
            ``[total_seq_len, hc_mult, hc_mult]``, and ``collapsed`` is
            ``[total_seq_len, hidden_size]`` in ``hidden_streams``' dtype.

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

        rows = hidden_streams.shape[0]
        streams = hidden_streams.cast(DType.float32)

        flat = streams.reshape([rows, self.hc_mult * self.hidden_size])
        # Unweighted RMSNorm, applied to the projection rather than its input:
        # the same linear map, one scale per row instead of
        # ``hc_mult * hidden_size``, and ``flat`` keeps its input precision
        # into the fp32 matmul, which may run in TF32.
        rms = F.rsqrt((flat * flat).mean(axis=-1) + self.rms_norm_eps)

        pre, post, comb = hyper_connection_gates(
            (flat @ self.hc_fn.T) * rms,
            self.hc_base,
            self.hc_scale,
            hc_mult=self.hc_mult,
            hc_eps=self.hc_eps,
            hc_sinkhorn_iters=self.hc_sinkhorn_iters,
        )

        collapsed = (pre.unsqueeze(-1) * streams).sum(axis=1).squeeze(1)
        return (
            post,
            comb.reshape([rows, self.hc_mult, self.hc_mult]),
            collapsed.cast(hidden_streams.dtype),
        )
