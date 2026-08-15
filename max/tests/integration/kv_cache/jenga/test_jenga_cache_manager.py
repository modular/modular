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
"""Tests for JengaKVCacheManager's runtime input preparation.

``JengaBlockManager`` (see ``test_jenga_block_manager.py``) owns allocation;
this file covers what ``JengaKVCacheManager`` adds on top of it --
``runtime_inputs`` / ``_compute_kv_cache_assignments`` -- since that's where
this PR's two real bugs lived: computing the per-request block count from a
leaf-id dict's *keys* instead of its *values*, and an alloc-time
``num_draft_tokens_per_step`` that silently diverged from ``params``'.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheInputsInterface,
    MHAKVCacheParams,
    MultiKVCacheParams,
)
from max.nn.kv_cache.cache_params import (
    KVCacheParamInterface,
    SpeculativeMethod,
)
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.kv_cache.paged_kv_cache.jenga_cache_manager import (
    JengaKVCacheManager,
)
from max.pipelines.request.base import RequestID

SLIDING = "sliding"
FULL = "full"


def make_leaf(
    *,
    n_kv_heads: int,
    window_size: int | None = None,
    page_size: int = 1,
    speculative_method: SpeculativeMethod | None = None,
    num_draft_tokens: int = 0,
) -> MHAKVCacheParams:
    return MHAKVCacheParams(
        dtype=DType.float32,
        num_layers=1,
        n_kv_heads=n_kv_heads,
        head_dim=1,
        page_size=page_size,
        devices=[DeviceRef.CPU()],
        data_parallel_degree=1,
        window_size=window_size,
        speculative_method=speculative_method,
        num_draft_tokens=num_draft_tokens,
    )


def create_manager(
    params: KVCacheParamInterface, num_huge_blocks: int, max_batch_size: int
) -> JengaKVCacheManager:
    huge_page_bytes = max(
        leaf.bytes_per_page for leaf in params.leaves().values()
    )
    return JengaKVCacheManager.create(
        params=params,
        available_bytes=num_huge_blocks * huge_page_bytes,
        max_batch_size=max_batch_size,
    )


def make_multi_leaf_manager(
    num_huge_blocks: int, max_batch_size: int = 8
) -> JengaKVCacheManager:
    """Two leaves with different bytes-per-page, so they land on different
    per-huge-block ratios -- mirrors gemma4's sliding (ratio 1) vs full
    (ratio 10) shape, just scaled down. n_kv_heads=1 vs 3 with page_size=1,
    head_dim=1, float32 gives 8 vs 24 bytes/page -> ratios 3 and 1.
    """
    params = MultiKVCacheParams.from_params(
        {
            SLIDING: make_leaf(n_kv_heads=1, window_size=4),
            FULL: make_leaf(n_kv_heads=3),
        }
    )
    return create_manager(params, num_huge_blocks, max_batch_size)


def make_single_leaf_manager(
    num_huge_blocks: int,
    max_batch_size: int = 8,
    *,
    speculative_method: SpeculativeMethod | None = None,
    num_draft_tokens: int = 0,
) -> JengaKVCacheManager:
    params = make_leaf(
        n_kv_heads=1,
        speculative_method=speculative_method,
        num_draft_tokens=num_draft_tokens,
    )
    return create_manager(params, num_huge_blocks, max_batch_size)


def make_ctx(num_tokens: int) -> TextContext:
    return TextContext(
        request_id=RequestID(),
        max_length=4096,
        tokens=TokenBuffer(np.arange(num_tokens, dtype=np.int64)),
    )


def get_lut(
    kv_inputs: KVCacheInputsInterface[Buffer, Buffer],
) -> list[list[int]]:
    """Returns the assigned block ids in the runtime LUT, trimming the
    tail padding. Unlike the legacy manager (sentinel = total_num_pages),
    Jenga's null block is index 0 -- real block ids start at 1 -- so
    padding columns fill with 0 and get trimmed the same way here."""
    assert isinstance(kv_inputs, KVCacheInputs)
    raw = kv_inputs.inputs[0].lookup_table.to_numpy().tolist()
    return [[b for b in row if b != 0] for row in raw]


def test_runtime_inputs_boundary_matches_real_allocated_blocks() -> None:
    """The insufficient-blocks check must use the request's real per-leaf
    block count, not something derived from the leaf-id dict's keys.

    Regression test for a bug where ``min(len(bs) for bs in
    get_req_blocks_per_leaf(ctx))`` iterated dict *keys* (leaf-id strings)
    instead of ``.values()``, so the boundary was a bogus constant
    (``len("full_attention.full_group")``) instead of the request's actual
    block count.
    """
    mgr = make_multi_leaf_manager(num_huge_blocks=10)
    ctx = make_ctx(num_tokens=3)
    mgr.claim(ctx)
    mgr.alloc(ctx)

    min_blocks = min(
        len(bs) for bs in mgr.get_req_blocks_per_leaf(ctx).values()
    )
    page_size = mgr.params.page_size

    # Exactly at the allocated capacity: must not raise.
    while ctx.tokens.active_length < min_blocks * page_size:
        ctx.update(0)
    mgr.runtime_inputs([[ctx]])

    # One token past capacity: must raise, and say so.
    ctx.update(0)
    with pytest.raises(ValueError, match="does not have sufficient blocks"):
        mgr.runtime_inputs([[ctx]])


def test_runtime_inputs_lut_and_cache_lengths() -> None:
    """Happy path: runtime_inputs reports the blocks alloc() actually gave."""
    mgr = make_single_leaf_manager(num_huge_blocks=10)
    ctx_a = make_ctx(num_tokens=3)
    ctx_b = make_ctx(num_tokens=2)
    mgr.claim(ctx_a)
    mgr.claim(ctx_b)
    mgr.alloc(ctx_a)
    mgr.alloc(ctx_b)

    kv_inputs = mgr.runtime_inputs([[ctx_a, ctx_b]])
    assert isinstance(kv_inputs, KVCacheInputs)

    (leaf_id,) = mgr._leaf_infos
    assert get_lut(kv_inputs) == [
        mgr.get_req_blocks_per_leaf(ctx_a)[leaf_id],
        mgr.get_req_blocks_per_leaf(ctx_b)[leaf_id],
    ]
    cache_lengths = kv_inputs.inputs[0].cache_lengths.to_numpy().tolist()
    assert cache_lengths == [
        ctx_a.tokens.processed_length,
        ctx_b.tokens.processed_length,
    ]


def test_runtime_inputs_without_alloc_raises() -> None:
    """A request that was only claimed, never allocated, cannot run."""
    mgr = make_single_leaf_manager(num_huge_blocks=10)
    ctx = make_ctx(num_tokens=3)
    mgr.claim(ctx)

    with pytest.raises(ValueError, match="does not have sufficient blocks"):
        mgr.runtime_inputs([[ctx]])


def test_create_logs_the_pool_geometry(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``create()`` logs one summary line plus one line per leaf, so an
    operator can read the huge/little page split straight from server logs.

    Asserted as one golden block (rather than piecemeal substring checks) so
    a failure prints the actual vs. expected geometry side by side.
    """
    with caplog.at_level(logging.INFO, logger="max.pipelines"):
        make_multi_leaf_manager(num_huge_blocks=10)

    expected = (
        "Jenga KV manager: 10 huge pages x 0.02 KiB = 0.23 KiB (per device), page_size 1 tokens\n"
        "\tsliding.sliding_window_group(4): 30 pages of 0.01 KiB  (3 per huge page)\n"
        "\tfull.full_group                : 10 pages of 0.02 KiB  (1 per huge page)"
    )
    assert "\n".join(r.message for r in caplog.records) == expected


def test_num_draft_tokens_per_step_threaded_from_params() -> None:
    """``__init__`` must forward ``params.num_draft_tokens_per_step`` to the
    block manager instead of relying on its hardcoded default.

    Regression test: the constructor used to omit this kwarg entirely, so
    the block manager's default of 1 could silently diverge from whatever
    ``params.num_draft_tokens_per_step`` returns (e.g. dflash's block-draft
    configs, where it equals ``num_draft_tokens``), desyncing alloc-time
    sizing from the runtime_inputs boundary check that reads the live
    property.
    """
    mgr = make_single_leaf_manager(
        num_huge_blocks=10,
        speculative_method="dflash",
        num_draft_tokens=3,
    )
    assert mgr.params.num_draft_tokens_per_step == 3
    assert mgr._num_draft_tokens_per_step == 3
