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
why this is not :class:`max.experimental.nn.HyperConnection`, whose fused gate
kernel always computes ``comb``.
"""

from __future__ import annotations

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.tensor import Tensor

from ..model_config import HYV4Config

__all__ = ["HYV4HyperConnection", "HYV4HyperHead"]


def _stream_mixes(streams: Tensor, fn: Tensor, rms_norm_eps: float) -> Tensor:
    """Projects the flattened float32 streams, scaled by their inverse RMS."""
    rows, hc_mult, hidden = streams.shape
    flat = streams.reshape([rows, hc_mult * hidden])
    rms = F.rsqrt((flat * flat).mean(axis=-1) + rms_norm_eps)
    return (flat @ fn.T) * rms


def _collapse(pre: Tensor, streams: Tensor) -> Tensor:
    return (pre.unsqueeze(-1) * streams).sum(axis=1).squeeze(1)


class HYV4HyperConnection(Module[[Tensor], tuple[Tensor, Tensor]]):
    """One iHC site, wrapped around a decoder sublayer.

    Weights are ``fn [2 * hc_mult, hc_mult * hidden_size]``,
    ``base [2 * hc_mult]`` and ``scale [2]``, all float32; the first half of
    ``fn`` and ``base`` drives ``pre``, the second half ``post``. A site is
    used as::

        post, x = hc(streams)
        y = sublayer(norm(x))
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

    def forward(self, streams: Tensor) -> tuple[Tensor, Tensor]:
        """Computes the gates and reads the sublayer input out of the streams.

        Args:
            streams: ``[total_seq_len, hc_mult, hidden_size]``.

        Returns:
            ``(post, x)``: ``post`` is float32 ``[total_seq_len, hc_mult]``
            and ``x`` is ``[total_seq_len, hidden_size]`` in ``streams``'
            dtype.
        """
        streams_f32 = streams.cast(DType.float32)
        pre_mix, post_mix = _stream_mixes(
            streams_f32, self.fn, self.rms_norm_eps
        ).split(self.hc_mult, axis=-1)
        pre_base, post_base = self.base.split(self.hc_mult)
        pre = F.sigmoid(pre_mix * self.scale[0] + pre_base) + self.hc_eps
        post = (
            self.hc_magnitude * F.sigmoid(post_mix * self.scale[1] + post_base)
            + self.hc_eps
        )
        return post, _collapse(pre, streams_f32).cast(streams.dtype)

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


class HYV4HyperHead(Module[[Tensor], Tensor]):
    """Collapses the residual streams into one sequence before the final norm.

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

    def forward(self, streams: Tensor) -> Tensor:
        """Returns the collapsed sequence, in ``streams``' dtype.

        Args:
            streams: ``[total_seq_len, hc_mult, hidden_size]``.
        """
        streams_f32 = streams.cast(DType.float32)
        mixes = _stream_mixes(streams_f32, self.hc_fn, self.rms_norm_eps)
        pre = F.sigmoid(mixes * self.hc_scale + self.hc_base) + self.hc_eps
        return _collapse(pre, streams_f32).cast(streams.dtype)
