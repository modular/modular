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
"""The HiSparse sizes at the widest MTP widths a ``--hisparse`` export reaches.

The resolve kernel keeps its selection hash in shared memory, so the sizes
``HiSparseParams.for_top_k`` picks must stay inside a block's opt-in budget.
At GLM-5.x's top-k 2048 that bounds the speculative width an export may ask
for, and the boundary is what these tests pin.
"""

from __future__ import annotations

import pytest
from max.nn._graph_ops import HiSparseParams

_TOP_K = 2048
"""GLM-5.x and DeepSeek-V3.2's indexer top-k."""

_PAGE_SIZE = 128
_FULL_LAYERS = 21
"""The full indexer layers of the GLM-5.x checkpoint, one LRU group each."""


def test_the_last_mtp_width_that_fits() -> None:
    """K = 5's hash is the largest that fits shared memory."""
    params = HiSparseParams.for_top_k(
        top_k=_TOP_K,
        query_rows=6,
        page_size=_PAGE_SIZE,
        num_groups=_FULL_LAYERS,
    )
    assert params.hash_capacity == 16384


def test_a_wider_mtp_width_is_refused() -> None:
    """K = 6's hash is twice the budget, so it must not build.

    Raises:
        ValueError: As :meth:`HiSparseParams.for_top_k` does, for a hash the
            resolve kernel cannot hold in shared memory.
    """
    with pytest.raises(ValueError) as exc:
        HiSparseParams.for_top_k(
            top_k=_TOP_K,
            query_rows=7,
            page_size=_PAGE_SIZE,
            num_groups=_FULL_LAYERS,
        )
    message = str(exc.value)
    assert "393216" in message, message
    assert "232448" in message, message
    assert "speculative width 6" in message, message
