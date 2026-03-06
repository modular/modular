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

from ...flux2.layers.embeddings import get_1d_rotary_pos_embed


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
            zeros = Tensor.zeros(
                [embedding.shape[0], 1],
                dtype=embedding.dtype,
                device=t.device,
            )
            embedding = F.concat([embedding, zeros], axis=-1)

        return embedding

    def forward(self, t: Tensor) -> Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = F.cast(t_freq, self.linear_1.weight.dtype)
        t_emb = self.linear_2(F.silu(self.linear_1(t_freq)))
        return t_emb


class RopeEmbedder(Module[[Tensor], tuple[Tensor, Tensor]]):
    def __init__(
        self,
        theta: float = 256.0,
        axes_dims: tuple[int, ...] = (32, 48, 48),
    ) -> None:
        self.theta = theta
        self.axes_dims = axes_dims

    def forward(self, ids: Tensor) -> tuple[Tensor, Tensor]:
        if ids.rank != 2:
            raise ValueError(f"Expected 2D ids tensor, got rank={ids.rank}")

        if int(ids.shape[-1]) != len(self.axes_dims):
            raise ValueError(
                "ids last dimension must match axes_dims length "
                f"({len(self.axes_dims)}), got {ids.shape[-1]}"
            )

        pos = ids.cast(DType.float32)
        cos_out = []
        sin_out = []
        for i in range(len(self.axes_dims)):
            cos_i, sin_i = get_1d_rotary_pos_embed(
                self.axes_dims[i],
                pos[:, i],
                theta=self.theta,
                use_real=True,
                repeat_interleave_real=True,
            )
            cos_out.append(cos_i)
            sin_out.append(sin_i)

        return F.concat(cos_out, axis=-1), F.concat(sin_out, axis=-1)
