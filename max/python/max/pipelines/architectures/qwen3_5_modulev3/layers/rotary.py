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
# Ported to ModuleV3 from
# max/python/max/pipelines/architectures/qwen3_5/layers/text_rotary.py
# at main 4df8111afba.
"""Qwen3.5's rotary tables: the static one and the per-token M-RoPE one."""

from __future__ import annotations

import numpy as np
from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.tensor import Tensor

__all__ = ["mrope_freqs_cis", "rope_freqs_cis"]


def _inv_freqs(theta: float, rotary_dim: int) -> Tensor:
    """Returns the ``[rotary_dim // 2]`` inverse frequencies, on the host.

    Computed in float64, which keeps the exponential from overflowing.
    """
    iota = F.arange(0, rotary_dim, step=2, dtype=DType.float64, device=CPU())
    return F.cast(1.0 / (theta ** (iota / rotary_dim)), DType.float32)


def rope_freqs_cis(theta: float, rotary_dim: int, max_seq_len: int) -> Tensor:
    """Returns the static ``[2 * max_seq_len, rotary_dim]`` cos/sin table.

    Row ``p`` holds position ``p``'s ``(cos, sin)`` pairs. It is built on the
    host, which places it once at model init.
    """
    t = F.arange(0, max_seq_len * 2, device=CPU(), dtype=DType.float32)
    freqs = F.outer(t, _inv_freqs(theta, rotary_dim))
    table = F.stack([F.cos(freqs), F.sin(freqs)], axis=-1)
    return table.reshape([table.shape[0], rotary_dim])


def _axis_per_frequency(
    mrope_section: list[int], num_frequencies: int
) -> np.ndarray:
    """Returns which position axis each rotary frequency reads.

    Interleaved M-RoPE spreads the temporal, height and width axes across the
    frequencies round-robin: height takes every third frequency from offset 1,
    width every third from offset 2, and temporal keeps the rest.
    """
    axis_of = np.zeros(num_frequencies, dtype=np.int64)
    for axis, offset in ((1, 1), (2, 2)):
        axis_of[offset : mrope_section[axis] * 3 : 3] = axis
    return axis_of


def mrope_freqs_cis(
    position_ids: Tensor,
    theta: float,
    rotary_dim: int,
    mrope_section: list[int],
) -> Tensor:
    """Returns the ``[total_seq_len, rotary_dim]`` per-token cos/sin table.

    The axis each frequency reads is fixed, so the merge of the three axes is
    a constant mask and the table stays on the accelerator.

    Args:
        position_ids: ``[3, total_seq_len]`` temporal, height and width
            positions, one column per token of the ragged batch, replicated
            on the mesh.
        theta: The rotary base.
        rotary_dim: How many lanes of each head rotate.
        mrope_section: Frequencies per axis, which sum to ``rotary_dim // 2``.

    Raises:
        ValueError: If ``mrope_section`` doesn't name three axes that cover
            the frequencies.
    """
    num_frequencies = rotary_dim // 2
    if len(mrope_section) != 3 or sum(mrope_section) != num_frequencies:
        raise ValueError(
            f"mrope_section {mrope_section} must name the temporal, height"
            f" and width axes and sum to {num_frequencies}"
        )
    mesh = position_ids.mesh
    positions = position_ids.unsqueeze(-1).cast(DType.float32)
    per_axis = positions * _inv_freqs(theta, rotary_dim).to(mesh)
    axis_of = _axis_per_frequency(mrope_section, num_frequencies)

    def masked(axis: int) -> Tensor:
        return per_axis[axis] * F.constant(
            (axis_of == axis).astype(np.float32), DType.float32, device=mesh
        )

    freqs = masked(0)
    for axis in range(1, len(mrope_section)):
        freqs = freqs + masked(axis)
    table = F.stack([F.cos(freqs), F.sin(freqs)], axis=-1)
    return table.reshape([table.shape[0], rotary_dim])
