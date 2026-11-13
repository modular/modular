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
"""Mistral Large 4 rope: YaRN inverse frequencies times a learned LeRoPE term.

Two things make this rope different from every other rope in the repository:

* LeRoPE. One global learned multiplier per frequency,
  ``inv_freq[i] * exp(lerope.alpha[i])``, held in a single ``lerope.alpha``
  tensor of width ``qk_rope_head_dim / 2``. It scales the *finished* YaRN
  frequencies: the ramp bounds are computed from the unmodified frequencies,
  never from the scaled ones.
* No magnitude scaling. ``yarn.apply_scale`` is false, so ``attention_factor``
  is 1.0 and ``cos``/``sin`` are not rescaled; the attention scale stays an
  explicit ``qk_head_dim ** -0.5``. The DeepSeek ``mscale`` path is dead here.

The frequency table is materialized inside the graph from the ``alpha`` weight,
so a reloaded weight cannot leave a stale table behind. It is built at the
replicated mapping, so every rank computes its own copy and nothing is
shipped between devices; the compiler hoists the arithmetic to the one-time
init.
"""

from __future__ import annotations

import numpy as np
from max import functional as F
from max.dtype import DType
from max.nn import Module
from max.tensor import Tensor
from numpy.typing import NDArray

from ..model_config import yarn_correction_range


def yarn_inv_freq(
    *,
    theta: float,
    factor: float,
    beta_fast: float,
    beta_slow: float,
    dim: int,
    original_max_position_embeddings: int,
) -> NDArray[np.floating]:
    """Returns YaRN's inverse frequencies, before any LeRoPE term.

    Mirrors ``transformers.modeling_rope_utils._compute_yarn_parameters`` at
    ``dim = qk_rope_head_dim``: the correction range is found on the
    unmodified frequencies and truncated to integers, and the extrapolated and
    interpolated frequencies are blended along the ramp. ``dim // 2`` angles.

    Args:
        theta: RoPE theta.
        factor: YaRN scaling factor.
        beta_fast: The fast-rotation beta (``yarn.beta`` in ``params.json``).
        beta_slow: The slow-rotation beta (``yarn.alpha`` in ``params.json``).
        dim: Rope width in dimensions (``qk_rope_head_dim``).
        original_max_position_embeddings: The pre-training window.

    Returns:
        A ``float32`` array of ``dim // 2`` inverse frequencies.

    Raises:
        ValueError: If the truncated ramp is empty.
    """
    low, high = yarn_correction_range(
        beta_fast=beta_fast,
        beta_slow=beta_slow,
        dim=dim,
        base=theta,
        original_max_position_embeddings=original_max_position_embeddings,
    )

    i = np.arange(0, dim, 2, dtype=np.float32)
    freq_base = np.float32(theta) ** (i / np.float32(dim))
    freq_extra = np.float32(1.0) / freq_base
    freq_inter = np.float32(1.0) / (np.float32(factor) * freq_base)
    ramp = np.clip(
        (np.arange(dim // 2, dtype=np.float32) - low) / float(high - low),
        0.0,
        1.0,
    )
    extrapolation = np.float32(1.0) - ramp
    return (
        freq_inter * (1 - extrapolation) + freq_extra * extrapolation
    ).astype(np.float32)


def lerope_inv_freq(
    alpha: NDArray[np.floating],
    *,
    theta: float,
    factor: float,
    beta_fast: float,
    beta_slow: float,
    original_max_position_embeddings: int,
) -> NDArray[np.floating]:
    """Returns ``yarn_inv_freq * exp(alpha)``.

    Args:
        alpha: The checkpoint's ``lerope.alpha``, one entry per angle.
        theta: RoPE theta.
        factor: YaRN scaling factor.
        beta_fast: The fast-rotation beta.
        beta_slow: The slow-rotation beta.
        original_max_position_embeddings: The pre-training window.

    Returns:
        A ``float32`` array of ``len(alpha)`` scaled inverse frequencies.
    """
    base = yarn_inv_freq(
        theta=theta,
        factor=factor,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
        dim=2 * int(alpha.shape[0]),
        original_max_position_embeddings=original_max_position_embeddings,
    )
    return (base * np.exp(alpha.astype(np.float32))).astype(np.float32)


class LeRope(Module[..., Tensor]):
    """The checkpoint's single global ``lerope.alpha`` frequency multiplier.

    One learned tensor for the whole model, ``qk_rope_head_dim / 2`` wide,
    BF16 as stored. It multiplies the finished YaRN inverse frequencies.
    """

    alpha: Tensor
    """The learned multiplier, ``[dim // 2]``."""

    def __init__(self, dim: int) -> None:
        """Allocates a zero ``alpha``, which leaves the rope unchanged.

        Args:
            dim: The rope width; ``alpha`` holds one entry per rotated pair.
        """
        super().__init__()
        # exp(0) = 1, so LeRoPE is the identity until the checkpoint loads.
        self.alpha = Tensor.zeros((dim // 2,), dtype=DType.bfloat16)


class MistralLarge4RotaryEmbedding(Module[..., Tensor]):
    """Rotary table built from the unmodified YaRN frequencies plus LeRoPE.

    ``freqs_cis(alpha)`` is ``[2 * max_seq_len, dim]`` in the layout the
    kernels read: the ``dim // 2`` cos values of a row followed by its
    ``dim // 2`` sin values, with pairs ``(d, d + 1)`` rotating together
    (GPT-J interleaved).
    """

    dim: int
    max_seq_len: int
    theta: float
    factor: float
    beta_fast: float
    beta_slow: float
    original_max_position_embeddings: int

    def __init__(
        self,
        *,
        dim: int,
        max_seq_len: int,
        theta: float,
        factor: float,
        beta_fast: float,
        beta_slow: float,
        original_max_position_embeddings: int,
    ) -> None:
        """Stores the rope geometry.

        Args:
            dim: Rope width in dimensions (``qk_rope_head_dim``).
            max_seq_len: Longest position the table covers.
            theta: RoPE theta.
            factor: YaRN scaling factor.
            beta_fast: The fast-rotation beta.
            beta_slow: The slow-rotation beta.
            original_max_position_embeddings: The pre-training window.
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.factor = factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.original_max_position_embeddings = original_max_position_embeddings

    def base_inv_freq(self) -> NDArray[np.floating]:
        """The frequency table with LeRoPE disabled (the drafter's rope)."""
        return yarn_inv_freq(
            theta=self.theta,
            factor=self.factor,
            beta_fast=self.beta_fast,
            beta_slow=self.beta_slow,
            dim=self.dim,
            original_max_position_embeddings=(
                self.original_max_position_embeddings
            ),
        )

    @property
    def freqs_cis(self) -> Tensor:
        """The rotary table with LeRoPE disabled (the drafter's rope)."""
        return self.freqs_cis_from(
            Tensor.zeros((self.dim // 2,), dtype=DType.bfloat16)
        )

    def freqs_cis_from(self, alpha: Tensor) -> Tensor:
        """Returns the rotary table, ``[2 * max_seq_len, dim]``.

        The table is replicated across the mesh: every rank rotates the same
        positions, and each rank's query heads only differ in which queries
        they hold. It is therefore computed at the replicated mapping, so each
        rank evaluates its own copy of a table the arithmetic hoists to the
        one-time init -- rather than built on one rank and transferred, which
        would lower to a host-enqueued cross-device wait
        (``mgp.device_wait``) around a peer copy and make the whole decode step
        uncapturable.

        Args:
            alpha: The ``lerope.alpha`` multiplier, ``[dim // 2]``. Passing
                zeros gives plain YaRN.
        """
        mapping = alpha.mapping
        inv_freq = F.constant(
            self.base_inv_freq(), DType.float32, device=mapping
        ) * F.exp(F.cast(alpha, DType.float32))
        # Two windows: a sequence this long needs every row the kernels index.
        positions = F.arange(
            0, 2 * self.max_seq_len, 1, out_dim=2 * self.max_seq_len,
            device=mapping, dtype=DType.float32,
        )  # fmt: skip
        freqs = F.outer(positions, inv_freq)
        return F.reshape(
            F.stack([F.cos(freqs), F.sin(freqs)], axis=-1),
            [2 * self.max_seq_len, self.dim],
        )
