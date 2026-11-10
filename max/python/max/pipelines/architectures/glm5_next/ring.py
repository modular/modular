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
"""The verify ring's compiled lengths and record stride."""

from __future__ import annotations

from max.dtype import DType

__all__ = [
    "COMPILED_RING_LENS",
    "RING_DTYPE",
    "ring_len_for_window",
    "ring_record_stride",
]


RING_DTYPE = DType.float32
"""Dtype of the ring pool. Float32 keeps the fold bit-exact against a
forward over the accepted prefix alone."""


COMPILED_RING_LENS = (2, 4, 8)
"""Ring lengths the ring kernels are compiled for."""


_RING_RECORD_ALIGN = 32
"""Elements a ring record's stride is a multiple of, one 128-byte line."""


def ring_len_for_window(window: int) -> int:
    """Returns the shortest compiled ring length that fits ``window``.

    Args:
        window: Tokens one verify advances a request by, ``K + 1``.

    Returns:
        The smallest entry of :data:`COMPILED_RING_LENS` that is at least
        ``window``.

    Raises:
        ValueError: If no compiled ring length is long enough.
    """
    for compiled in COMPILED_RING_LENS:
        if window <= compiled:
            return compiled
    raise ValueError(
        f"a {window}-token verify window needs a ring longer than any the "
        f"gated-delta ring kernels are compiled for ({COMPILED_RING_LENS}); "
        "add the length to both the kernel dispatch and COMPILED_RING_LENS"
    )


def ring_record_stride(
    *, record_elements: int, bytes_per_element: int, page_multiple_of: int
) -> int:
    """Returns the padded stride of one ring record.

    A huge block is the least common multiple of every leaf's page, so the
    ring's page is padded to divide ``page_multiple_of``, the live leaves'
    least common multiple, and never enlarges it. The kernels lay a record
    out as ``gated_delta_ring_record_elements`` does, the raw key and then
    each value head's delta row and decay, which ``record_elements`` counts.

    Args:
        record_elements: Elements one record holds before padding.
        bytes_per_element: Page bytes one element of stride costs.
        page_multiple_of: Bytes the ring's page must divide.

    Returns:
        The smallest aligned stride whose page divides ``page_multiple_of``,
        or the aligned record itself when no stride that small does.
    """
    aligned = -(-record_elements // _RING_RECORD_ALIGN) * _RING_RECORD_ALIGN
    for stride in range(
        aligned,
        page_multiple_of // bytes_per_element + 1,
        _RING_RECORD_ALIGN,
    ):
        if page_multiple_of % (stride * bytes_per_element) == 0:
            return stride
    return aligned
