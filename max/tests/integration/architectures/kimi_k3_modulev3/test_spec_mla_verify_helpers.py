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
"""CPU checks for the draft-aware MLA sizing helpers.

Exercises the code the GPU verify tests cannot: the draft-width resolution
the MLA cache leaf's wiring passes through, the plan-cap formula, and the
block-manager accounting (``_compute_seq_len``) that consumes
``num_draft_tokens`` -- the page-count half of the one-page-short verify
defect.
"""

from __future__ import annotations

import numpy as np
import pytest
from _tp_model_test_utils import spec_decode_context
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import MLAKVCacheParams
from max.pipelines.architectures.kimi_k3_modulev3.dspark.config import (
    K3_BLOCK_SIZE,
)
from max.pipelines.architectures.kimi_k3_modulev3.dspark.mla_extent import (
    spec_block_extent_bound,
    spec_verify_extent,
)
from max.pipelines.architectures.kimi_k3_modulev3.spec_mla import (
    kimi_k3_mla_buffer_size,
    kimi_k3_mla_table_rows,
    kimi_k3_spec_num_draft_tokens,
)
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.kv_cache import KVCacheConfig
from max.pipelines.kv_cache.paged_kv_cache._common import (
    _compute_seq_len,
    cache_valid_length_for_context,
)
from max.pipelines.lib import PipelineConfig
from max.pipelines.speculative.config import SpeculativeConfig
from max.support.math import ceildiv

_PAGE_SIZE = 128
_NUM_DRAFT_TOKENS = 7
_VERIFY_ROWS = 8
_MAX_SEQ_LEN = 16384
_EXTENT_RNG = np.random.default_rng(20261118)
"""One draw shared by the extent tests, so failures reproduce."""


def _pipeline_config(
    speculative: SpeculativeConfig | None,
) -> PipelineConfig:
    # Only the speculative field is read; construct without validation so no
    # model manifest is needed.
    return PipelineConfig.model_construct(speculative=speculative)


def _mid_decode_context(committed: int) -> TextContext:
    """A request with ``committed`` processed tokens and one active row.

    The shape the cache manager sees at a verify step's fetch: the committed
    context is cached, the step's anchor is the one scheduled row, and the
    drafts are merged into the step after scheduling.
    """
    context = TextContext(
        max_length=committed + 100,
        tokens=TokenBuffer(np.full(committed, 3, dtype=np.int64)),
    )
    context.update(5)
    assert context.tokens.processed_length == committed
    assert context.tokens.active_length == 1
    return context


def test_spec_num_draft_tokens() -> None:
    """The helper sizes the DSpark block width and nothing else."""
    assert kimi_k3_spec_num_draft_tokens(_pipeline_config(None)) == 0, (
        "no speculative config must size for zero drafts"
    )

    dspark = SpeculativeConfig(
        speculative_method="dspark", num_speculative_tokens=_NUM_DRAFT_TOKENS
    )
    assert (
        kimi_k3_spec_num_draft_tokens(_pipeline_config(dspark))
        == _NUM_DRAFT_TOKENS
    )

    # Only DSpark block-drafts through this tower; the one-token-per-step
    # methods keep the plain decode-sized leaf rather than failing at KV
    # sizing.
    eagle = SpeculativeConfig(speculative_method="eagle")
    assert kimi_k3_spec_num_draft_tokens(_pipeline_config(eagle)) == 0
    mtp = SpeculativeConfig(speculative_method="mtp")
    assert kimi_k3_spec_num_draft_tokens(_pipeline_config(mtp)) == 0

    unresolved = SpeculativeConfig(speculative_method="dspark")
    with pytest.raises(ValueError, match="checkpoint_draft_width"):
        kimi_k3_spec_num_draft_tokens(_pipeline_config(unresolved))


def test_mla_table_rows() -> None:
    """The rope table's row count: one request's window, page-padded.

    The table is indexed at a single request's absolute positions -- the
    context limit plus its verify rows -- so it needs one copy of that
    window; the plan cap on the prefill arm instead spans every batch
    slot's copy.
    """
    rows = kimi_k3_mla_table_rows(
        max_seq_len=_MAX_SEQ_LEN,
        page_size=_PAGE_SIZE,
        num_draft_tokens=_NUM_DRAFT_TOKENS,
    )
    # 16384 + 8 verify rows crosses a page boundary, so it pads up.
    assert rows == ceildiv(_MAX_SEQ_LEN + _VERIFY_ROWS, _PAGE_SIZE) * _PAGE_SIZE
    assert rows > _MAX_SEQ_LEN

    # Without spec there is one active row past the committed context.
    assert (
        kimi_k3_mla_table_rows(
            max_seq_len=_MAX_SEQ_LEN, page_size=_PAGE_SIZE, num_draft_tokens=0
        )
        == ceildiv(_MAX_SEQ_LEN + 1, _PAGE_SIZE) * _PAGE_SIZE
    )


def test_mla_buffer_size() -> None:
    """The cap covers the rope table on the fold arm and the TG worst case
    on the prefill arm, where decode batches are never budget-charged."""
    # One request at the context limit, its verify rows past it, padded.
    table_need = ceildiv(_MAX_SEQ_LEN + _VERIFY_ROWS, _PAGE_SIZE) * _PAGE_SIZE

    # The decode fold (K3 at TP8 on AMD): an FP8 latent cache at 12 heads
    # per device folds the 8-row verify, so the plan is dead data on
    # verify steps and the cap only has to bound context-encoding windows
    # and the table.
    fold = kimi_k3_mla_buffer_size(
        max_batch_context_length=_MAX_SEQ_LEN,
        max_seq_len=_MAX_SEQ_LEN,
        page_size=_PAGE_SIZE,
        num_draft_tokens=_NUM_DRAFT_TOKENS,
        max_batch_size=4,
        num_q_heads_per_device=12,
        cache_dtype=DType.float8_e4m3fn,
        device_api="hip",
    )
    assert fold == table_need

    # A BF16 latent cache on AMD takes the prefill arm at the SAME head
    # count: the fold is FP8-only there, so the verify reads the plan and
    # every slot's window must fit (the silent-truncation case).
    bf16_latent = kimi_k3_mla_buffer_size(
        max_batch_context_length=_MAX_SEQ_LEN,
        max_seq_len=_MAX_SEQ_LEN,
        page_size=_PAGE_SIZE,
        num_draft_tokens=_NUM_DRAFT_TOKENS,
        max_batch_size=4,
        num_q_heads_per_device=12,
        cache_dtype=DType.bfloat16,
        device_api="hip",
    )
    assert bf16_latent == 4 * table_need

    # The prefill arm (TP2/TP4 latent spec): every slot at the context
    # limit verifying 8 rows, since TG batches are not budget-charged.
    prefill = kimi_k3_mla_buffer_size(
        max_batch_context_length=_MAX_SEQ_LEN,
        max_seq_len=_MAX_SEQ_LEN,
        page_size=_PAGE_SIZE,
        num_draft_tokens=_NUM_DRAFT_TOKENS,
        max_batch_size=4,
        num_q_heads_per_device=48,
        cache_dtype=DType.float8_e4m3fn,
        device_api="hip",
    )
    assert prefill == 4 * table_need

    # Without spec there are no verify rows to make room for.
    plain = kimi_k3_mla_buffer_size(
        max_batch_context_length=_MAX_SEQ_LEN,
        max_seq_len=_MAX_SEQ_LEN,
        page_size=_PAGE_SIZE,
        num_draft_tokens=0,
        max_batch_size=4,
        num_q_heads_per_device=48,
        cache_dtype=DType.bfloat16,
        device_api="hip",
    )
    assert plain == max(
        _MAX_SEQ_LEN, ceildiv(_MAX_SEQ_LEN + 1, _PAGE_SIZE) * _PAGE_SIZE
    )


def test_dspark_leaf_covers_verify_window_across_page() -> None:
    """A dspark-sized leaf reserves a page for the verify rows; a zero draft
    width does not.

    A request with a 125-token cached context adds 8 verify rows (133, past
    the 128-token page boundary). With the draft width threaded through,
    the block manager's per-request length covers that window; with the
    decode-assumption leaf (``num_draft_tokens`` 0) it is one page short --
    the RED case, the ModuleV3 form of the one-page-short defect.
    """
    committed = 125
    context = _mid_decode_context(committed)
    verify_window = committed + _VERIFY_ROWS

    dspark_leaf = MLAKVCacheParams(
        dtype=DType.float8_e4m3fn,
        head_dim=576,
        num_layers=1,
        devices=[DeviceRef.CPU()],
        page_size=_PAGE_SIZE,
        num_q_heads=96,
        speculative_method="dspark",
        num_draft_tokens=_NUM_DRAFT_TOKENS,
    )
    assert dspark_leaf.num_draft_tokens == _NUM_DRAFT_TOKENS
    # DSpark block-drafts the whole width in one forward, so the per-step
    # count selects the block-draft page accounting.
    assert dspark_leaf.num_draft_tokens_per_step == _NUM_DRAFT_TOKENS

    dspark_len = _compute_seq_len(
        context,
        dspark_leaf.num_draft_tokens,
        dspark_leaf.num_draft_tokens_per_step,
    )
    assert ceildiv(dspark_len, _PAGE_SIZE) >= ceildiv(
        verify_window, _PAGE_SIZE
    ), (
        f"a dspark-sized leaf reserved {dspark_len} slots, short of the"
        f" {verify_window}-row verify window"
    )

    decode_assumed_leaf = MLAKVCacheParams(
        dtype=DType.float8_e4m3fn,
        head_dim=576,
        num_layers=1,
        devices=[DeviceRef.CPU()],
        page_size=_PAGE_SIZE,
        num_q_heads=96,
    )
    assert decode_assumed_leaf.num_draft_tokens == 0
    assert decode_assumed_leaf.num_draft_tokens_per_step == 0

    decode_assumed_len = _compute_seq_len(
        context,
        decode_assumed_leaf.num_draft_tokens,
        decode_assumed_leaf.num_draft_tokens_per_step,
    )
    assert ceildiv(decode_assumed_len, _PAGE_SIZE) < ceildiv(
        verify_window, _PAGE_SIZE
    ), (
        f"the decode-assumption leaf reserved {decode_assumed_len} slots and"
        " was NOT one page short of the verify window; the RED case no longer"
        " reproduces the defect"
    )


def _random_batch(
    count: int, *, max_drafts: int, maybe_accepted: bool
) -> list[TextContext]:
    """Random contexts whose windows cross pages and 64-token buckets."""
    return [
        spec_decode_context(
            cached=int(_EXTENT_RNG.integers(1, 600)),
            active=int(_EXTENT_RNG.integers(1, 3)),
            drafts=int(_EXTENT_RNG.integers(0, max_drafts + 1)),
            maybe_accepted=(
                int(_EXTENT_RNG.integers(0, _NUM_DRAFT_TOKENS + 1))
                if maybe_accepted
                else 0
            ),
        )
        for _ in range(count)
    ]


def _align_up(length: int, page: int) -> int:
    return ceildiv(length, page) * page


def test_spec_verify_extent_is_the_plans_own_extent() -> None:
    """The helper computes what the plan kernel computes, term for term.

    The plan sums each request's ``align_up(cache + merged rows, page)``
    over the same runtime tensors; the helper must read the same terms off
    the context -- committed and maybe-accepted rows, this step's active
    rows, the drafts merged after scheduling -- or the host extent drifts
    from what the device plan builds.
    """
    lengths = [125, 128, 129, 3, 200, 260, 512, 0]
    contexts = [
        spec_decode_context(
            cached=lengths[i],
            active=1 + (i % 2),
            drafts=_NUM_DRAFT_TOKENS - (i % 3),
            maybe_accepted=(i % 5),
        )
        for i in range(len(lengths))
    ]
    expected = sum(
        _align_up(
            ctx.tokens.processed_length
            + len(ctx.spec_decoding_state.maybe_accepted_draft_tokens)
            + ctx.tokens.active_length
            + len(ctx.spec_decoding_state.draft_tokens_to_verify),
            _PAGE_SIZE,
        )
        for ctx in contexts
    )
    assert spec_verify_extent(contexts, _PAGE_SIZE) == expected

    # Edge: an empty prompt-chunk batch still plans zero rows' extent.
    assert spec_verify_extent([], _PAGE_SIZE) == 0


def _mla_leaf(num_q_heads: int) -> MLAKVCacheParams:
    """An MLA leaf built the way ``KimiK3Config._attn_kv_params`` builds one.

    The draft width is resolved through ``kimi_k3_spec_num_draft_tokens``
    and stamped through ``KVCacheConfig.to_params`` with the dspark
    method -- the production construction -- rather than a hand-rolled
    ``num_draft_tokens``.
    """
    dspark = SpeculativeConfig(
        speculative_method="dspark", num_speculative_tokens=_NUM_DRAFT_TOKENS
    )
    params = KVCacheConfig().to_params(
        dtype=DType.float32,
        n_kv_heads=1,
        head_dim=16,
        num_layers=1,
        devices=[DeviceRef.CPU()],
        data_parallel_degree=1,
        is_mla=True,
        num_q_heads=num_q_heads,
        speculative_method="dspark",
        num_draft_tokens=kimi_k3_spec_num_draft_tokens(
            _pipeline_config(dspark)
        ),
    )
    assert isinstance(params, MLAKVCacheParams)
    return params


def _max_cache_length(contexts: list[TextContext]) -> int:
    """The leaf's ``max_cache_length`` host input, per the pipeline's own
    replay-characteristics derivation: the widest request's window at the
    leaf's draft width."""
    return max(
        cache_valid_length_for_context(ctx, _NUM_DRAFT_TOKENS)
        for ctx in contexts
    )


def test_capture_bound_covers_both_extents() -> None:
    """``B * align_up(max_cache_length, page)`` is an upper bound.

    That expression is what a captured decode graph freezes, from host
    inputs keyed to the aligned cache-length bucket. Two properties keep a
    replay safe: the bound frozen at a batch's own bucket covers that
    batch's verify extent and drafter block bound, and -- because capture
    probes the largest length first within each key and skips keys already
    captured -- the bound frozen at a LARGER bucket covers every smaller
    batch that replays the same graph. An uncovered replay silently
    truncates the plan (the copy launches only ``length`` rows).
    """
    for _ in range(32):
        count = int(_EXTENT_RNG.integers(1, 9))
        batches = [
            _random_batch(
                count, max_drafts=_NUM_DRAFT_TOKENS, maybe_accepted=True
            )
            for _ in range(2)
        ]
        # The graph is captured at the larger bucket of the pair, as the
        # capture order guarantees, and both batches replay it.
        captured, replayed = sorted(
            batches, key=_max_cache_length, reverse=True
        )
        captured_max = _max_cache_length(captured)
        bound = count * _align_up(captured_max, _PAGE_SIZE)
        for contexts in (captured, replayed):
            verify = spec_verify_extent(contexts, _PAGE_SIZE)
            block = spec_block_extent_bound(
                contexts, _PAGE_SIZE, _NUM_DRAFT_TOKENS
            )
            assert bound >= verify, (
                f"capture bound {bound} below the verify extent {verify}"
                f" for {count} requests at max cache length {captured_max}"
            )
            assert bound >= block, (
                f"capture bound {bound} below the block bound {block} for"
                f" {count} requests at max cache length {captured_max}"
            )


def test_block_extent_bound_covers_the_drivers_block_windows() -> None:
    """The block bound covers the block driver's real block windows.

    The block cache's length after a verify is the pre-iteration length
    plus the commit, and the block queries its own rows on top. ``pre``
    and the window are read off the context's raw fields -- the driver's
    own quantities, not the helper's term function -- and the commit is
    drawn anywhere in the merged window the verify can accept, so every
    partial acceptance is covered, not just the full one; at a full
    accept the bound is tight, not merely safe.
    """
    assert K3_BLOCK_SIZE == _NUM_DRAFT_TOKENS, (
        "the block bound charges the pairing's draft width; a drafter whose"
        " block ran wider would need its own bound"
    )
    for _ in range(64):
        contexts = _random_batch(
            int(_EXTENT_RNG.integers(1, 9)),
            max_drafts=_NUM_DRAFT_TOKENS,
            maybe_accepted=True,
        )
        pre = [
            ctx.tokens.processed_length
            + len(ctx.spec_decoding_state.maybe_accepted_draft_tokens)
            for ctx in contexts
        ]
        windows = [
            ctx.tokens.active_length
            + len(ctx.spec_decoding_state.draft_tokens_to_verify)
            for ctx in contexts
        ]
        commits = [
            int(_EXTENT_RNG.integers(0, window + 1)) for window in windows
        ]
        real = sum(
            _align_up(p + commit + K3_BLOCK_SIZE, _PAGE_SIZE)
            for p, commit in zip(pre, commits, strict=True)
        )
        bound = spec_block_extent_bound(contexts, _PAGE_SIZE, _NUM_DRAFT_TOKENS)
        assert bound >= real, (
            f"block bound {bound} below the block window {real} at commits"
            f" {commits} over pre-iteration lengths {pre}"
        )
        # At a full accept -- commit the whole merged window -- the bound
        # is tight, since the drafter's block width is the pairing's draft
        # width.
        full = sum(
            _align_up(p + window + K3_BLOCK_SIZE, _PAGE_SIZE)
            for p, window in zip(pre, windows, strict=True)
        )
        assert bound == full


def test_dspark_leaf_covers_the_committed_block_window() -> None:
    """The real config path's leaves cover ``pre + commit + 7``.

    The block cache's length after a verify is the pre-iteration length
    plus the commit, and the block queries its seven rows on top. Both
    leaves share one draft width (``MultiKVCacheParams`` refuses a tree
    whose leaves disagree), and at that width each request's
    ``cache_valid_length_for_context`` -- what the leaf's
    ``max_cache_length`` host input holds the max of -- already counts the
    commit (at most the merged window) plus the block's rows.
    """
    target_leaf = _mla_leaf(96)
    draft_leaf = _mla_leaf(64)
    assert target_leaf.num_draft_tokens == draft_leaf.num_draft_tokens, (
        "the pairing's leaves must share one draft width, or the drafter's"
        " host cache length no longer bounds its own block window"
    )
    assert draft_leaf.num_draft_tokens == K3_BLOCK_SIZE
    for _ in range(64):
        ctx = _random_batch(
            1, max_drafts=_NUM_DRAFT_TOKENS, maybe_accepted=True
        )[0]
        pre = ctx.tokens.processed_length + len(
            ctx.spec_decoding_state.maybe_accepted_draft_tokens
        )
        window = ctx.tokens.active_length + len(
            ctx.spec_decoding_state.draft_tokens_to_verify
        )
        commit = int(_EXTENT_RNG.integers(0, window + 1))
        covered = cache_valid_length_for_context(
            ctx, draft_leaf.num_draft_tokens
        )
        assert covered >= pre + commit + K3_BLOCK_SIZE, (
            f"the drafter leaf's cache-length terms ({covered}) fall short"
            f" of the block window {pre + commit + K3_BLOCK_SIZE}"
        )
