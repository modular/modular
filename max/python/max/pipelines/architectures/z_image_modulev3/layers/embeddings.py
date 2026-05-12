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


def apply_rotary_emb(
    x: Tensor,
    freqs_cis: tuple[Tensor, Tensor],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
    sequence_dim: int = 2,
) -> Tensor:
    """Apply rotary position embeddings to input tensor.

    Args:
        x: Input tensor of shape [B, H, S, D] (if sequence_dim=2) or [B, S, H, D] (if sequence_dim=1).
        freqs_cis: Tuple of (cos, sin) frequency tensors of shape [S, D].
        use_real: If True, use real-valued RoPE. If False, raises NotImplementedError.
        use_real_unbind_dim: Dimension to unbind for real-valued RoPE.
            -1: Used for Flux, CogVideoX, Hunyuan-Dit (reshape to [..., D//2, 2])
            -2: Used for Stable Audio, OmniGen, CogView4, Cosmos (reshape to [..., 2, D//2])
        sequence_dim: Dimension index for sequence length (1 or 2).

    Returns:
        Tensor with rotary embeddings applied.

    Raises:
        NotImplementedError: If use_real is False.
    """
    if not use_real:
        raise NotImplementedError("Only use_real=True is supported")

    cos, sin = freqs_cis  # [S, D]
    # Expand cos/sin to match x shape based on sequence_dim
    if sequence_dim == 2:
        # x: [B, H, S, D], need cos/sin: [1, 1, S, D]
        cos = F.unsqueeze(F.unsqueeze(cos, 0), 0)
        sin = F.unsqueeze(F.unsqueeze(sin, 0), 0)
    elif sequence_dim == 1:
        # x: [B, S, H, D], need cos/sin: [1, S, 1, D]
        cos = F.unsqueeze(F.unsqueeze(cos, 0), 2)
        sin = F.unsqueeze(F.unsqueeze(sin, 0), 2)
    else:
        raise ValueError(f"`sequence_dim={sequence_dim}` but should be 1 or 2.")

    input_dtype = x.dtype

    if use_real_unbind_dim == -1:
        # Used for Flux, CogVideoX, Hunyuan-Dit
        # Reshape x: [..., D] -> [..., D//2, 2]
        x_shape = list(x.shape)
        new_shape = x_shape[:-1] + [x_shape[-1] // 2, 2]
        x_reshaped = F.reshape(x, new_shape)

        # Split into real and imaginary parts: [..., D//2, 2] -> 2 x [..., D//2]
        x_real = x_reshaped[..., 0]  # [..., D//2]
        x_imag = x_reshaped[..., 1]  # [..., D//2]

        # Create rotated version: stack([-x_imag, x_real], dim=-1) then flatten
        # This creates [..., D//2, 2]
        x_rotated_real = -x_imag
        x_rotated_imag = x_real
        x_rotated_stacked = F.stack(
            [x_rotated_real, x_rotated_imag], axis=-1
        )  # [..., D//2, 2]

        # Flatten back to [..., D]
        x_rotated = F.reshape(x_rotated_stacked, x_shape)

    elif use_real_unbind_dim == -2:
        # Used for Stable Audio, OmniGen, CogView4, Cosmos
        # Reshape x: [..., D] -> [..., 2, D//2]
        x_shape = list(x.shape)
        new_shape = x_shape[:-1] + [2, x_shape[-1] // 2]
        x_reshaped = F.reshape(x, new_shape)

        # Split into real and imaginary parts: [..., 2, D//2] -> 2 x [..., D//2]
        x_real = x_reshaped[..., 0, :]  # [..., D//2]
        x_imag = x_reshaped[..., 1, :]  # [..., D//2]

        # Create rotated version: cat([-x_imag, x_real], dim=-1)
        x_rotated = F.concat([-x_imag, x_real], axis=-1)  # [..., D]

    else:
        raise ValueError(
            f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2."
        )

    # Apply rotation: x * cos + x_rotated * sin
    # Cast to float32 for computation, then back to input dtype
    x_float = F.cast(x, DType.float32)
    x_rotated_float = F.cast(x_rotated, DType.float32)
    cos_float = F.cast(cos, DType.float32)
    sin_float = F.cast(sin, DType.float32)

    out = x_float * cos_float + x_rotated_float * sin_float
    out = F.cast(out, input_dtype)

    return out


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Tensor | int,
    theta: float = 10000.0,
    use_real: bool = True,
    repeat_interleave_real: bool = True,
) -> tuple[Tensor, Tensor]:
    """Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    Args:
        dim: Dimension of the embedding (must be even).
        pos: Position indices tensor of shape [S] or scalar.
        theta: Base frequency for the sinusoidal encoding.
        use_real: If True, use real-valued RoPE.
        repeat_interleave_real: If True, repeat cos/sin along the embedding dimension. If False, raises NotImplementedError.

    Returns:
        Tuple of (cos, sin) tensors for rotary position embedding.

    Raises:
        NotImplementedError: If use_real is False or repeat_interleave_real is False.
    """
    assert dim % 2 == 0, f"dim must be even, got {dim}"

    if not use_real:
        raise NotImplementedError("Only use_real=True is supported")

    if not repeat_interleave_real:
        raise NotImplementedError(
            "Only repeat_interleave_real=True is supported"
        )

    # Create frequency bands: [dim/2]
    device = pos.device if isinstance(pos, Tensor) else None
    freq_exponent = (
        F.arange(0, dim, 2, dtype=DType.float32, device=device) / dim
    )
    freq = 1.0 / (theta**freq_exponent)

    # Compute outer product: [S, dim/2]
    freqs = F.outer(pos, freq)

    # Compute cos and sin: [S, dim/2]
    cos_emb = F.cos(freqs)
    sin_emb = F.sin(freqs)

    # Repeat interleave: [S, dim/2] -> [S, dim]
    # repeat_interleave not supported on GPU, use stack + reshape instead
    # PyTorch: repeat_interleave(2, dim=1) makes [a, b, c] -> [a, a, b, b, c, c]
    # MAX equivalent: stack([x, x], axis=2) + reshape
    cos_stacked = F.stack([cos_emb, cos_emb], axis=2)  # [S, dim/2, 2]
    sin_stacked = F.stack([sin_emb, sin_emb], axis=2)  # [S, dim/2, 2]

    freqs_cos = F.reshape(
        cos_stacked,
        [cos_emb.shape[0], cos_stacked.shape[1] * cos_stacked.shape[2]],
    )  # [S, dim]
    freqs_sin = F.reshape(
        sin_stacked,
        [sin_emb.shape[0], sin_stacked.shape[1] * sin_stacked.shape[2]],
    )  # [S, dim]

    return freqs_cos, freqs_sin


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
