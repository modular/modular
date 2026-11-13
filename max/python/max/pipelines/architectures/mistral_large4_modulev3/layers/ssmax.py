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
"""SSMax: Mistral Large 4's per-(token, head) query scaling.

For every query row at absolute 0-based position ``pos`` and head ``h``::

    q'[t, h, :] = q[t, h, :] * (s[h] * log(pos + 1) + b[h])

``pos`` is the row's own position, never the sequence or cache length, so a
prefill chunk scales each row differently and a decode step rescales the one
row it holds. No head is clamped: the reference lets two layer-0 MLA heads
cross zero inside 1M positions and produce inverted queries.

Placement, both prefill and decode:

* MLA -- on the query after its rope (on the whole 128-wide query on a NoPE
  layer), before the sparse MLA op's absorbed ``q . k_b``.
* Indexer -- on the query after ``wq_b``, before its rope and FP8 quantize.

:func:`ssmax_scale` multiplies a prepared factor onto a query tensor and is
applied outside the fused MLA op, which rotates the query itself. Scaling
before that rotation is the same linear map as scaling after it -- the rotation
is linear and the factor is constant across a head's dimensions -- so the only
difference from the reference is the bf16 rounding order: the measured
approximation, until the fused op applies SSMax after its own rounded
rotation.
"""

from __future__ import annotations

from max import functional as F
from max.dtype import DType
from max.nn import Module
from max.tensor import Tensor


def log_positions(positions: Tensor, dtype: DType) -> Tensor:
    """Returns ``log(pos + 1)`` per row, taken in float32 and rounded to ``dtype``.

    It depends only on the rows, so a graph computes it once for every layer.

    Args:
        positions: ``[rows]`` absolute 0-based positions.
        dtype: The query's dtype, which the reference rounds the log to.

    Returns:
        ``[rows]`` logs in ``dtype``.
    """
    return F.cast(F.log(F.cast(positions, DType.float32) + 1.0), dtype)


def query_factor(
    positions: Tensor, scale: Tensor, bias: Tensor, dtype: DType
) -> Tensor:
    """Returns the ``[rows, heads]`` SSMax factor for ``positions``.

    The rounding chain is the reference's, in ``dtype``: the log is taken in
    float32 and rounded to it, then ``* s``, then ``+ b``.

    Args:
        positions: ``[rows]`` absolute 0-based positions.
        scale: ``[heads]`` learned ``s``.
        bias: ``[heads]`` learned ``b``.
        dtype: The query's dtype, which the reference rounds each step to.

    Returns:
        The per-(row, head) factor, ``[rows, heads]``.
    """
    return factor_from_log(log_positions(positions, dtype), scale, bias, dtype)


def factor_from_log(
    log_pos: Tensor, scale: Tensor, bias: Tensor, dtype: DType
) -> Tensor:
    """Returns :func:`query_factor` from rows' :func:`log_positions`."""
    product = F.cast(F.unsqueeze(log_pos, axis=-1) * scale, dtype)
    return F.cast(product + bias, dtype)


def ssmax_scale(query: Tensor, factor: Tensor) -> Tensor:
    """Scales ``[rows, heads, head_dim]`` queries by a ``[rows, heads]`` factor.

    Args:
        query: ``[rows, heads, head_dim]`` queries.
        factor: ``[rows, heads]`` factor from :func:`query_factor`.

    Returns:
        The scaled query, same shape and dtype as ``query``.
    """
    return query * F.unsqueeze(factor, axis=-1)


class SSMaxScaling(Module[[Tensor, Tensor], Tensor]):
    """Learned per-head scale and bias, applied to a query tensor."""

    s: Tensor
    """The learned ``s``, ``[heads]``."""
    b: Tensor
    """The learned ``b``, ``[heads]``."""

    def __init__(self, num_heads: int) -> None:
        """Allocates the checkpoint's ``ssmax_scaling.{s,b}`` pair.

        Args:
            num_heads: Query heads the scale covers.
        """
        super().__init__()
        # BF16 as the checkpoint stores them; the reference rounds each step of
        # the chain to the query's dtype, which these match bit for bit.
        self.s = Tensor.zeros((num_heads,), dtype=DType.bfloat16)
        self.b = Tensor.zeros((num_heads,), dtype=DType.bfloat16)

    def forward(self, query: Tensor, positions: Tensor) -> Tensor:
        """Scales ``query`` rows by their own positions' factor."""
        return ssmax_scale(
            query,
            query_factor(positions, self.s, self.b, query.dtype),
        )

    def scale_by_log(self, query: Tensor, log_pos: Tensor) -> Tensor:
        """Scales ``query`` as :meth:`forward` does, from :func:`log_positions`.

        Without the log, the factor is cheap enough for the graph compiler to
        compute inside the multiply's own launch.
        """
        return ssmax_scale(
            query, factor_from_log(log_pos, self.s, self.b, query.dtype)
        )
