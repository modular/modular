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
"""A request that hits ``InsufficientBlocksError`` is retried; it must not add
its prompt to ``input_tokens`` on every attempt."""

from __future__ import annotations

import numpy as np
import pytest
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.kv_cache import InsufficientBlocksError
from max.pipelines.kv_cache.paged_kv_cache.block_manager import BlockManager
from max.pipelines.modeling.types import RequestID


def _ctx(num_tokens: int) -> TextContext:
    return TextContext(
        request_id=RequestID(),
        max_length=4096,
        tokens=TokenBuffer(np.arange(num_tokens, dtype=np.int64)),
    )


def _full_pool() -> tuple[BlockManager, TextContext]:
    """4 blocks of 1 token; a 3-token request holds 3, so another cannot fit."""
    bm = BlockManager(
        total_num_blocks=4, block_size=1, enable_prefix_caching=False
    )
    held = _ctx(3)
    bm.claim(held)
    bm.allocate_new_blocks(held)
    return bm, held


def _counts(bm: BlockManager) -> tuple[int, int, int]:
    m = bm.metrics
    return (m.input_tokens, m.cache_tokens, m.prompt_tokens)


def test_failed_allocs_do_not_count_the_prompt() -> None:
    bm, _ = _full_pool()
    waiting = _ctx(3)
    bm.claim(waiting)
    for _ in range(5):
        with pytest.raises(InsufficientBlocksError):
            bm.allocate_new_blocks(waiting)
    assert _counts(bm) == (3, 0, 3)


def test_a_retried_request_counts_once_it_fits() -> None:
    bm, held = _full_pool()
    waiting = _ctx(3)
    bm.claim(waiting)
    for _ in range(3):
        with pytest.raises(InsufficientBlocksError):
            bm.allocate_new_blocks(waiting)
    bm.release(held)
    bm.allocate_new_blocks(waiting)
    assert _counts(bm) == (6, 0, 6)
