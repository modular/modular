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
"""RoPE for DeepSeek-V4 on ModuleV3, ragged (one table row per token).

No mscale, and two schedules: compressed layers ``compress_rope_theta`` with
YaRN, window-only layers base theta with YaRN off. Rotation is over
interleaved pairs of the trailing ``qk_rope_head_dim`` dims. See the
ModuleV2 implementation (``deepseekV4/layers/rope.py`` at dfc8109fc0c).

The table is built with the same CPU graph ops, in the same order, as the V2
``YarnRotaryEmbedding`` it subclasses there, so it is bit-identical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.nn.rope_scaling import YarnScalingParams
from max.tensor import Tensor
from max.tensor.sharding import DeviceMesh

from ..model_config import DeepseekV4Config
from .kernels import rope_ragged

__all__ = ["DeepseekV4RotaryEmbedding", "apply_rope_tail", "rope_for_layer"]


def _cpu_f32(value: float) -> Tensor:
    return F.constant(value, DType.float32, device=CPU())


def _correction_dim(
    num_rotations: float, dim: int, base: float, max_position: int
) -> Tensor:
    return (
        _cpu_f32(float(dim))
        * F.log(
            _cpu_f32(float(max_position))
            / (_cpu_f32(num_rotations) * 2 * math.pi)
        )
    ) / (2 * F.log(_cpu_f32(float(base))))


@dataclass(frozen=True)
class DeepseekV4RotaryEmbedding:
    """One layer's cos/sin table; ``scaling_params=None`` is the YaRN-off one."""

    theta: float
    rope_dim: int
    max_seq_len: int
    scaling_params: YarnScalingParams | None = None

    def _inv_freqs(self) -> Tensor:
        n = self.rope_dim
        if self.scaling_params is None:
            iota = F.range(0, n, 2, dtype=DType.float64, device=CPU())
            return (1.0 / (self.theta ** (iota / n))).cast(DType.float32)

        sp = self.scaling_params
        half = n // 2
        iota = F.range(0, n, 2, out_dim=half, dtype=DType.float64, device=CPU())
        freq_base = self.theta ** (iota / float(n))
        freq_extra = (1.0 / freq_base).cast(DType.float32)
        freq_inter = (1.0 / (sp.factor * freq_base)).cast(DType.float32)

        orig = sp.original_max_position_embeddings
        low = F.floor(_correction_dim(sp.beta_fast, n, self.theta, orig))
        high = F.trunc(_correction_dim(sp.beta_slow, n, self.theta, orig)) + 1
        low = F.elementwise_max(low, 0)
        high = F.elementwise_min(high, n - 1)

        diff = high - low
        diff = F.where(diff == 0, _cpu_f32(0.001), diff)
        ramp = (
            F.range(0, half, dtype=DType.float32, device=CPU()) - low
        ) / diff
        ramp = F.elementwise_min(F.elementwise_max(ramp, 0), 1)
        mask = 1.0 - ramp.cast(DType.float32)
        return freq_inter * (1 - mask) + freq_extra * mask

    def freqs_cis_base(self, mesh: DeviceMesh) -> Tensor:
        """Returns the cos/sin table for every position, whole on each device.

        The table is ``[max_seq_len, rope_dim // 2, 2]`` and unit-magnitude,
        replicated on every device of ``mesh``.
        """
        t = F.range(
            0,
            self.max_seq_len,
            1,
            out_dim=self.max_seq_len,
            dtype=DType.float32,
            device=CPU(),
        )
        freqs = F.outer(t, self._inv_freqs())
        return F.stack([F.cos(freqs), F.sin(freqs)], axis=-1).to(mesh)


def rope_for_layer(
    config: DeepseekV4Config, layer_idx: int, max_seq_len: int
) -> DeepseekV4RotaryEmbedding:
    """The table this layer's attention, compressor and indexer share."""
    compressed = config.layer_compress_ratio(layer_idx) != 0
    scaling = config.rope_scaling or {}
    return DeepseekV4RotaryEmbedding(
        theta=config.compress_rope_theta if compressed else config.rope_theta,
        rope_dim=config.qk_rope_head_dim,
        max_seq_len=max_seq_len,
        scaling_params=YarnScalingParams(
            factor=float(scaling["factor"]),
            beta_fast=float(scaling["beta_fast"]),
            beta_slow=float(scaling["beta_slow"]),
            original_max_position_embeddings=int(
                scaling["original_max_position_embeddings"]
            ),
            truncate=False,
        )
        if compressed
        else None,
    )


def apply_rope_tail(
    x: Tensor,
    freqs_cis: Tensor,
    rope_head_dim: int,
    *,
    inverse: bool = False,
) -> Tensor:
    """Rotates the trailing ``rope_head_dim`` dims of ``x``, leaving the rest.

    Args:
        x: ``[T, ..., width]``, token axis first.
        freqs_cis: ``[T, rope_head_dim // 2, 2]``, one row per token.
        rope_head_dim: Width of the rotated tail.
        inverse: Rotate by the conjugate (the attention output's un-rotation).
    """
    width = int(x.shape[-1])
    if x.rank >= 2 and width % 8 == 0 and rope_head_dim % 8 == 0:
        return _rope_tail_kernel(x, freqs_cis, rope_head_dim, inverse)
    return _rope_tail_ops(x, freqs_cis, rope_head_dim, inverse)


def _rope_tail_kernel(
    x: Tensor, freqs_cis: Tensor, rope_head_dim: int, inverse: bool
) -> Tensor:
    """One ``rope_ragged`` over a single batch at position 0: row i, token i."""
    width = int(x.shape[-1])
    tokens = x.shape[0]
    table = freqs_cis.reshape([tokens, rope_head_dim])
    if inverse:
        conj = np.tile(np.array([1.0, -1.0], np.float32), rope_head_dim // 2)
        table = table * F.constant(conj, table.dtype, device=table.mesh)
    offsets = F.constant(
        np.array([0, np.iinfo(np.uint32).max], np.uint32),
        DType.uint32,
        device=x.mesh,
    )
    start = F.constant(np.zeros(1, np.uint32), DType.uint32, device=x.mesh)
    out = rope_ragged(
        x.reshape([tokens, -1, width]),
        offsets,
        start,
        table,
        interleaved=True,
    )
    return out.reshape(x.shape)


def _rope_tail_ops(
    x: Tensor, freqs_cis: Tensor, rope_head_dim: int, inverse: bool
) -> Tensor:
    # Positive bounds on the last axis, deliberately: a
    # negative bound on a concat result can come back rotated.
    width = int(x.shape[-1])
    head = x[..., : width - rope_head_dim]
    tail = x[..., width - rope_head_dim : width]

    pairs = tail.reshape([*tail.shape[:-1], rope_head_dim // 2, 2])
    real = pairs[..., 0]
    imag = pairs[..., 1]

    bcast = [freqs_cis.shape[0], *([1] * (pairs.rank - 3)), freqs_cis.shape[1]]
    cos = freqs_cis[..., 0].reshape(bcast)
    sin = freqs_cis[..., 1].reshape(bcast)
    if inverse:
        sin = -sin

    out_real = real * cos - imag * sin
    out_imag = real * sin + imag * cos
    rotated = F.stack([out_real, out_imag], axis=-1).reshape(tail.shape)
    return F.concat([head, rotated.cast(tail.dtype)], axis=-1)
