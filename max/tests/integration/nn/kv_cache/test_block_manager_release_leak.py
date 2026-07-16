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

"""Regression tests for the MXSERV-152 leak in ``BlockManager.release``.

``release`` used to reset the per-request ``defaultdict(list)`` maps with
``[request_id] = []``, retaining the request-id key + an empty list per request
forever. It now uses ``.pop(request_id, None)``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
from max.pipelines.context import TextContext
from max.pipelines.kv_cache.connectors.null_connector import NullConnector
from max.pipelines.kv_cache.memory_tier import MemoryTier
from max.pipelines.kv_cache.paged_kv_cache.block_manager import BlockManager
from max.pipelines.modeling.types import RequestID


def _make_ctx(tokens: np.ndarray, request_id: RequestID) -> TextContext:
    return cast(
        TextContext,
        SimpleNamespace(request_id=request_id, tokens=tokens, cache_salt=None),
    )


def _make_block_manager() -> BlockManager:
    return BlockManager(
        device_memory_tier=MemoryTier.MEMORY_TIER_CPU,
        total_num_blocks=256,
        block_size=8,
        connector=cast(object, NullConnector()),  # type: ignore[arg-type]
        enable_prefix_caching=True,
    )


def _track_request(bm: BlockManager, request_id: RequestID) -> None:
    """Fill req_to_hashes (via compute_hashes) and req_to_blocks for a request."""
    bm.compute_hashes_for_request(
        _make_ctx(np.arange(33, dtype=np.int32), request_id)
    )
    bm.req_to_blocks[request_id] = [
        bm.allocate_device_block() for _ in range(2)
    ]


def test_release_deletes_per_request_entries() -> None:
    """release must delete the per-request keys, not reset them to ``[]``."""
    bm = _make_block_manager()
    request_id = RequestID("req-1")
    _track_request(bm, request_id)

    bm.release(request_id)

    assert request_id not in bm.req_to_blocks
    assert request_id not in bm.req_to_hashes


def test_release_does_not_accumulate_across_requests() -> None:
    """After N request lifecycles the per-request maps return to empty."""
    bm = _make_block_manager()
    for i in range(64):
        request_id = RequestID(f"req-{i}")
        _track_request(bm, request_id)
        bm.release(request_id)

    assert len(bm.req_to_blocks) == 0
    assert len(bm.req_to_hashes) == 0


def test_release_is_idempotent() -> None:
    """A repeated release must not raise -- why we use pop, not del."""
    bm = _make_block_manager()
    request_id = RequestID("req-1")
    _track_request(bm, request_id)

    bm.release(request_id)
    bm.release(request_id)  # must not raise

    assert request_id not in bm.req_to_blocks
    assert request_id not in bm.req_to_hashes
