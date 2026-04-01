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

import math

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Linear, Module
from max.experimental.tensor import Tensor


class TimestepEmbedder(Module[[Tensor], Tensor]):
    def __init__(
        self,
        out_size: int,
        mid_size: int | None = None,
        frequency_embedding_size: int = 256,
    ) -> None:
        if mid_size is None:
            mid_size = out_size

        self.linear_1 = Linear(frequency_embedding_size, mid_size, bias=True)
        self.linear_2 = Linear(mid_size, out_size, bias=True)
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        t: Tensor,
        dim: int,
        max_period: float = 10000.0,
    ) -> Tensor:
        half = dim // 2
        freqs = F.arange(0, half, dtype=DType.float32, device=t.device)
        freqs = F.exp((-math.log(max_period) * freqs) / float(half))

        args = F.cast(t, DType.float32)[:, None] * freqs[None, :]
        embedding = F.concat([F.cos(args), F.sin(args)], axis=-1)

        if dim % 2:
            # Avoid Tensor.zeros in the graph path; broadcast a scalar zero like
            # other normalization blocks (see LayerNorm gamma/beta patterns).
            zero = F.reshape(
                F.constant(0.0, embedding.dtype, device=t.device),
                (1, 1),
            )
            zeros_col = F.broadcast_to(zero, (embedding.shape[0], 1))
            embedding = F.concat([embedding, zeros_col], axis=-1)

        return embedding

    def forward(self, t: Tensor) -> Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = F.cast(t_freq, self.linear_1.weight.dtype)
        t_emb = self.linear_2(F.silu(self.linear_1(t_freq)))
        return t_emb


def _get_1d_rope_interleaved(
    dim: int,
    pos: Tensor,
    theta: float = 10000.0,
) -> Tensor:
    """Compute 1-D RoPE in [cos, sin] interleaved pair format."""
    half = dim // 2
    freq_exp = F.arange(0, half, dtype=DType.float32, device=pos.device) / half
    freq = 1.0 / (theta**freq_exp)
    freqs = F.outer(pos, freq)
    paired = F.stack([F.cos(freqs), F.sin(freqs)], axis=2)
    return F.reshape(paired, [freqs.shape[0], dim])


class RopeEmbedder(Module[[Tensor], Tensor]):
    def __init__(
        self,
        theta: float = 256.0,
        axes_dims: tuple[int, ...] = (32, 48, 48),
    ) -> None:
        self.theta = theta
        self.axes_dims = axes_dims

    def forward(self, ids: Tensor) -> Tensor:
        pos = ids.cast(DType.float32)
        parts = []
        for i in range(len(self.axes_dims)):
            parts.append(
                _get_1d_rope_interleaved(
                    self.axes_dims[i],
                    pos[:, i],
                    theta=self.theta,
                )
            )
        return F.concat(parts, axis=-1)
