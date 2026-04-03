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

from __future__ import annotations

import math

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn.layer import Module
from max.nn.linear import Linear


class TimestepEmbedder(Module):
    def __init__(
        self,
        out_size: int,
        mid_size: int | None = None,
        frequency_embedding_size: int = 256,
        *,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        """Initialize TimestepEmbedder."""
        super().__init__()
        if mid_size is None:
            mid_size = out_size

        self.frequency_embedding_size = frequency_embedding_size

        self.linear_1 = Linear(
            in_dim=frequency_embedding_size,
            out_dim=mid_size,
            dtype=dtype,
            device=device,
            has_bias=True,
        )
        self.linear_2 = Linear(
            in_dim=mid_size,
            out_dim=out_size,
            dtype=dtype,
            device=device,
            has_bias=True,
        )

    @staticmethod
    def timestep_embedding(
        t: TensorValue,
        dim: int,
        max_period: float = 10000.0,
    ) -> TensorValue:
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = ops.range(0, half, dtype=DType.float32, device=t.device)
        freqs = ops.exp((-math.log(max_period) * freqs) / float(half))

        args = ops.cast(t, DType.float32)[:, None] * freqs[None, :]
        embedding = ops.concat([ops.cos(args), ops.sin(args)], axis=-1)

        if dim % 2:
            zero = ops.reshape(
                ops.constant(0.0, DType.float32, device=t.device),
                (1, 1),
            )
            zeros_col = ops.broadcast_to(zero, (embedding.shape[0], 1))
            embedding = ops.concat([embedding, zeros_col], axis=-1)

        return embedding

    def __call__(self, t: TensorValue) -> TensorValue:
        """Embed timesteps."""
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = ops.cast(t_freq, self.linear_1.weight.dtype)
        t_emb = self.linear_2(ops.silu(self.linear_1(t_freq)))
        return t_emb


class RopeEmbedder(Module):
    def __init__(
        self,
        theta: float = 256.0,
        axes_dims: tuple[int, ...] = (32, 48, 48),
    ) -> None:
        """Initialize RopeEmbedder."""
        super().__init__()
        self.theta = theta
        self.axes_dims = axes_dims

    def __call__(self, ids: TensorValue) -> TensorValue:
        """Compute interleaved [cos, sin] rotary position embeddings."""
        pos = ops.cast(ids, DType.float32)
        parts = []
        for i in range(len(self.axes_dims)):
            dim = self.axes_dims[i]
            half = dim // 2
            freq_exp = (
                ops.range(
                    0,
                    half,
                    dtype=DType.float32,
                    device=pos.device,
                )
                / half
            )
            freq = 1.0 / (self.theta**freq_exp)
            freqs = ops.outer(pos[:, i], freq)
            paired = ops.stack([ops.cos(freqs), ops.sin(freqs)], axis=2)
            parts.append(ops.reshape(paired, [freqs.shape[0], dim]))

        return ops.concat(parts, axis=-1)
