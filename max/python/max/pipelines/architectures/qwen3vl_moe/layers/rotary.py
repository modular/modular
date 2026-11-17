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
"""The decoder's per-token interleaved M-RoPE table."""

from __future__ import annotations

from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.tensor import Tensor

__all__ = ["mrope_freqs_cis"]


def _inv_freqs(theta: float, head_dim: int, dtype: DType) -> Tensor:
    """Returns the ``[head_dim // 2]`` inverse frequencies, on the host.

    They are rounded to ``dtype`` and back, as the reference stores them.
    Computed in float64, which keeps the exponential from overflowing.
    """
    iota = F.range(0, head_dim, 2, dtype=DType.float64, device=CPU())
    inv_freqs = F.cast(1.0 / (theta ** (iota / head_dim)), DType.float32)
    return inv_freqs.cast(dtype).cast(DType.float32)


def _take_axis(
    freqs: Tensor, source: Tensor, offset: int, length: int, gather_axis: int
) -> Tensor:
    """Overwrites every third frequency from ``offset`` with ``source``'s.

    Interleaved M-RoPE reads the height and width axes at frequencies
    ``offset, offset + 3, ...`` below ``length``, and the temporal axis at
    the rest.
    """
    step = 3
    indices = F.range(
        offset,
        length,
        step,
        device=CPU(),
        dtype=DType.int64,
        out_dim=(length + 1) // step,
    )
    selected = F.gather(source, indices, axis=gather_axis)
    rows = F.tile(indices.unsqueeze(0), (freqs.shape[0], 1))
    return F.scatter(freqs, selected, rows, axis=1)


def mrope_freqs_cis(
    position_ids: Tensor,
    *,
    theta: float,
    head_dim: int,
    mrope_section: list[int],
    dtype: DType,
) -> Tensor:
    """Returns the ``[total_seq_len, head_dim]`` per-token cos/sin table.

    Args:
        position_ids: ``[3, total_seq_len]`` temporal, height and width
            positions on the host, one column per token of the ragged batch.
        theta: The rotary base.
        head_dim: The attention head width; every lane rotates.
        mrope_section: Frequencies per axis.
        dtype: The table's dtype.
    """
    inv_freqs = _inv_freqs(theta, head_dim, dtype).unsqueeze(0).unsqueeze(0)
    positions = position_ids.unsqueeze(-1).cast(DType.float32)
    freqs = positions * inv_freqs
    table = freqs[0]
    table = _take_axis(table, freqs[1], 1, mrope_section[1] * 3, -1)
    table = _take_axis(table, freqs[2], 2, mrope_section[2] * 3, 1)
    table = F.stack([F.cos(table), F.sin(table)], axis=-1).cast(dtype)
    seq_len, half, pair = table.shape
    return table.reshape([seq_len, half * pair])
