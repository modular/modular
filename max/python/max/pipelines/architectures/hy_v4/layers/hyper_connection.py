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

"""Identity hyper-connections (iHC) for Hy4 (ModuleV3).

Hy4 keeps ``hc_mult`` residual streams like DeepSeek-V4's mHC but drops the
``comb`` mixer and its Sinkhorn projection: each stream is carried forward
unchanged and only receives the sublayer output scaled by ``post``. That is
why this is not :class:`max.nn.HyperConnection`, whose fused gate
kernel always computes ``comb``.
"""

from __future__ import annotations

from max import functional as F
from max.dtype import DType
from max.nn import Module
from max.nn._graph_ops import ihc_pre_supports
from max.nn.kernels.hyper_connection import ihc_pre
from max.nn.norm import RMSNorm
from max.tensor import Tensor

from ..model_config import HYV4Config

__all__ = ["HYV4HyperConnection", "HYV4HyperHead"]


def _collapse_norm(
    streams: Tensor,
    fn: Tensor,
    base: Tensor,
    scale: Tensor,
    norm: RMSNorm,
    *,
    hc_magnitude: float,
    hc_eps: float,
    rms_norm_eps: float,
) -> tuple[Tensor, Tensor]:
    """Returns ``(norm(sum_j pre_j * streams[:, j]), gates)``.

    ``gates`` is float32 ``[rows, len(fn)]``: ``pre``, then ``post`` when
    ``fn`` has ``2 * hc_mult`` rows. Where :func:`ihc_pre_supports` accepts
    the site and ``norm`` is a plain RMSNorm, the fused op reads the streams
    in their own dtype and applies the norm; elsewhere the same math runs in
    graph ops.
    """
    rows, hc_mult, hidden = streams.shape
    if type(norm) is RMSNorm and ihc_pre_supports(
        int(hc_mult),
        int(hidden),
        int(fn.shape[0]),
        streams.dtype,
        streams.mesh.devices[0],
    ):
        # The op runs per device; under tensor parallelism every device
        # computes the same replicated result.
        x, gates = ihc_pre(
            streams,
            fn,
            base,
            scale,
            norm.weight.cast(streams.dtype),
            hc_magnitude=hc_magnitude,
            hc_eps=hc_eps,
            rms_eps=rms_norm_eps,
            norm_eps=norm.eps,
        )
        return x.rebind_mapping(streams.mapping), gates.rebind_mapping(
            streams.mapping
        )
    flat = streams.cast(DType.float32).reshape([rows, hc_mult * hidden])
    rms = F.rsqrt((flat * flat).mean(axis=-1) + rms_norm_eps)
    mixes = (flat @ fn.T) * rms
    bands = [
        F.sigmoid(band * scale[i] + bias)
        for i, (band, bias) in enumerate(
            zip(
                mixes.split(int(hc_mult), axis=-1),
                base.split(int(hc_mult)),
                strict=True,
            )
        )
    ]
    gates = (
        F.concat([bands[0], *(hc_magnitude * b for b in bands[1:])], axis=-1)
        + hc_eps
    )
    pre = gates[:, : int(hc_mult)].unsqueeze(-1)
    x = (pre * streams.cast(DType.float32)).sum(axis=1).squeeze(1)
    return norm(x.cast(streams.dtype)), gates


class HYV4HyperConnection(Module[[Tensor, RMSNorm], tuple[Tensor, Tensor]]):
    """One iHC site, wrapped around a decoder sublayer.

    Weights are ``fn [2 * hc_mult, hc_mult * hidden_size]``,
    ``base [2 * hc_mult]`` and ``scale [2]``, all float32; the first half of
    ``fn`` and ``base`` drives ``pre``, the second half ``post``. A site is
    used as::

        post, x = hc(streams, norm)
        y = sublayer(x)
        streams = hc.write_back(streams, y, post)

    Args:
        config: The model config; ``hc_mult``, ``hidden_size``,
            ``hc_magnitude``, ``hc_eps`` and ``rms_norm_eps`` are read from it.
    """

    fn: Tensor
    base: Tensor
    scale: Tensor

    def __init__(self, config: HYV4Config) -> None:
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_magnitude = config.hc_magnitude
        self.hc_eps = config.hc_eps
        self.rms_norm_eps = config.rms_norm_eps
        self.fn = Tensor.zeros(
            [2 * self.hc_mult, self.hc_mult * config.hidden_size],
            dtype=DType.float32,
        )
        self.base = Tensor.zeros([2 * self.hc_mult], dtype=DType.float32)
        self.scale = Tensor.zeros([2], dtype=DType.float32)

    def forward(self, streams: Tensor, norm: RMSNorm) -> tuple[Tensor, Tensor]:
        """Computes the gates and the normalized sublayer input.

        Args:
            streams: ``[total_seq_len, hc_mult, hidden_size]``.
            norm: The sublayer's input norm, applied to the collapsed
                streams.

        Returns:
            ``(post, x)``: ``post`` is float32 ``[total_seq_len, hc_mult]``
            and ``x`` is ``norm`` of the collapse, ``[total_seq_len,
            hidden_size]`` in ``streams``' dtype.
        """
        x, gates = _collapse_norm(
            streams,
            self.fn,
            self.base,
            self.scale,
            norm,
            hc_magnitude=self.hc_magnitude,
            hc_eps=self.hc_eps,
            rms_norm_eps=self.rms_norm_eps,
        )
        return gates[:, self.hc_mult :], x

    def write_back(self, streams: Tensor, y: Tensor, post: Tensor) -> Tensor:
        """Adds the sublayer output into every stream, scaled by ``post``.

        Args:
            streams: The streams passed to :meth:`forward`.
            y: The sublayer output, ``[total_seq_len, hidden_size]``.
            post: The gate :meth:`forward` returned.

        Returns:
            The updated streams, same shape and dtype as ``streams``.
        """
        updated = post.unsqueeze(-1) * y.cast(DType.float32).unsqueeze(
            1
        ) + streams.cast(DType.float32)
        return updated.cast(streams.dtype)


class HYV4HyperHead(Module[[Tensor, RMSNorm], Tensor]):
    """Collapses the residual streams into one normalized sequence.

    Weights are ``hc_fn [hc_mult, hc_mult * hidden_size]``,
    ``hc_base [hc_mult]`` and ``hc_scale [1]``, all float32.

    Args:
        config: The model config; ``hc_mult``, ``hidden_size``, ``hc_eps``
            and ``rms_norm_eps`` are read from it.
    """

    hc_fn: Tensor
    hc_base: Tensor
    hc_scale: Tensor

    def __init__(self, config: HYV4Config) -> None:
        super().__init__()
        self.hc_eps = config.hc_eps
        self.rms_norm_eps = config.rms_norm_eps
        hc_mult = config.hc_mult
        self.hc_fn = Tensor.zeros(
            [hc_mult, hc_mult * config.hidden_size], dtype=DType.float32
        )
        self.hc_base = Tensor.zeros([hc_mult], dtype=DType.float32)
        self.hc_scale = Tensor.zeros([1], dtype=DType.float32)

    def forward(self, streams: Tensor, norm: RMSNorm) -> Tensor:
        """Returns ``norm`` of the collapsed sequence, in ``streams``' dtype.

        Args:
            streams: ``[total_seq_len, hc_mult, hidden_size]``.
            norm: The model's final norm.
        """
        x, _ = _collapse_norm(
            streams,
            self.hc_fn,
            self.hc_base,
            self.hc_scale,
            norm,
            hc_magnitude=1.0,
            hc_eps=self.hc_eps,
            rms_norm_eps=self.rms_norm_eps,
        )
        return x
