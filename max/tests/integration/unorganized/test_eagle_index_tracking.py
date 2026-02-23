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
"""Unit tests for EAGLE index tracking through update_contexts,
_prepare_draft_batch, and reserve_token_space_for_batch.

Verifies that TokenBuffer indices and ``_draft_kv_start_idx`` stay correct
across all acceptance/rejection scenarios. No model loading or GPU required.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from max.interfaces import RequestID
from max.interfaces.context import SamplingParams
from max.interfaces.tokens import TokenBuffer
from max.pipelines.core import reserve_token_space_for_batch
from max.pipelines.core.context import TextContext
from max.pipelines.lib.interfaces import ModelInputs
from max.pipelines.lib.speculative_decoding.base import (
    SpeculativeDecodingMetrics,
)
from max.pipelines.lib.speculative_decoding.eagle import (
    EAGLESpeculativeDecodingPipeline,
)


# ---------------------------------------------------------------------------
# Mock objects
# ---------------------------------------------------------------------------
class MockKVManager:
    def alloc(
        self, ctx: TextContext, replica_idx: int = 0, num_steps: int = 1
    ) -> None:
        pass

    def runtime_inputs(
        self, replica_batches: list[list[TextContext]], num_steps: int
    ) -> list[Any]:
        return []


class MockModel:
    def __init__(self) -> None:
        self.kv_manager = MockKVManager()

    def prepare_initial_token_inputs(self, **kwargs: Any) -> ModelInputs:
        return ModelInputs()


class EAGLEIndexTracker:
    """Minimal mock so we can call real EAGLE methods as unbound."""

    def __init__(self) -> None:
        self._draft_kv_start_idx: dict[RequestID, int] = {}
        self._last_verified_token: dict[RequestID, int] = {}
        self._metrics = SpeculativeDecodingMetrics()

    def update_contexts(
        self,
        context_batch: list[TextContext],
        first_rejected_tokens: npt.NDArray[np.integer[Any]],
        recovered_tokens: npt.NDArray[np.integer[Any]],
        bonus_tokens: npt.NDArray[np.integer[Any]] | None,
        draft_tokens: npt.NDArray[np.integer[Any]],
        num_draft_tokens_generated: int,
        data_parallel_splits: npt.NDArray[np.int64] | None = None,
    ) -> None:
        EAGLESpeculativeDecodingPipeline.update_contexts(
            self,  # type: ignore[arg-type]
            context_batch=context_batch,
            first_rejected_tokens=first_rejected_tokens,
            recovered_tokens=recovered_tokens,
            bonus_tokens=bonus_tokens,
            draft_tokens=draft_tokens,
            num_draft_tokens_generated=num_draft_tokens_generated,
            data_parallel_splits=data_parallel_splits,
        )

    def _seek_processing_position(
        self,
        context: TextContext,
        target_position: int,
    ) -> None:
        EAGLESpeculativeDecodingPipeline._seek_processing_position(
            self,  # type: ignore[arg-type]
            context,
            target_position,
        )

    def prepare_draft_batch(
        self,
        batch: list[TextContext],
        num_steps: int = 3,
        return_n_logits: int = 1,
    ) -> None:
        model = MockModel()
        EAGLESpeculativeDecodingPipeline._prepare_draft_batch(
            self,  # type: ignore[arg-type]
            model=model,  # type: ignore[arg-type]
            batch=batch,
            replica_batches=[batch],
            num_steps=num_steps,
            return_n_logits=return_n_logits,
            hidden_states=None,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
MAX_LENGTH = 10_000


def make_context(prompt_tokens: list[int]) -> TextContext:
    return TextContext(
        max_length=MAX_LENGTH,
        tokens=TokenBuffer(np.array(prompt_tokens, dtype=np.int64)),
        request_id=RequestID(),
        eos_token_ids=set(),
        sampling_params=SamplingParams(),
    )


def assert_state(
    ctx: TextContext,
    *,
    processed: int,
    active: int,
    position: int,
    generated: int,
    label: str = "",
) -> None:
    prefix = f"[{label}] " if label else ""
    assert ctx.tokens.processed_length == processed, (
        f"{prefix}processed_length: expected {processed}, "
        f"got {ctx.tokens.processed_length}"
    )
    assert ctx.tokens.active_length == active, (
        f"{prefix}active_length: expected {active}, "
        f"got {ctx.tokens.active_length}"
    )
    assert ctx.tokens.current_position == position, (
        f"{prefix}current_position: expected {position}, "
        f"got {ctx.tokens.current_position}"
    )
    assert ctx.tokens.generated_length == generated, (
        f"{prefix}generated_length: expected {generated}, "
        f"got {ctx.tokens.generated_length}"
    )


def setup_single_context() -> tuple[TextContext, EAGLEIndexTracker]:
    """Prompt [10,11,12,13] + target token 50 → processed=4, active=1.
    Sets draft_kv_start_idx=4."""
    ctx = make_context([10, 11, 12, 13])
    ctx.update(50)
    tracker = EAGLEIndexTracker()
    tracker._draft_kv_start_idx[ctx.request_id] = 4
    return ctx, tracker


def setup_post_target_context() -> tuple[TextContext, EAGLEIndexTracker]:
    """Same as setup_single_context but without draft_kv_start_idx set
    (simulates first draft call for a new request)."""
    ctx = make_context([10, 11, 12, 13])
    ctx.update(50)
    tracker = EAGLEIndexTracker()
    return ctx, tracker


# ---------------------------------------------------------------------------
# update_contexts: parametrized single-context scenarios
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "first_rejected, bonus, expected",
    [
        pytest.param(
            3,
            np.array([[300]], dtype=np.int64),
            # All accepted + bonus: 4 updates, offset=-3
            dict(
                processed=8, active=4, position=12, generated=5, kv=4, last=300
            ),
            id="all-accepted-bonus",
        ),
        pytest.param(
            3,
            None,
            # All accepted, no bonus (MTP): 3 updates, offset=-3
            dict(
                processed=7, active=4, position=11, generated=4, kv=4, last=102
            ),
            id="all-accepted-no-bonus",
        ),
        pytest.param(
            0,
            np.array([[300]], dtype=np.int64),
            # All rejected: 0 draft + recovered[0,0]=200, offset=0
            dict(
                processed=5, active=1, position=6, generated=2, kv=4, last=200
            ),
            id="all-rejected",
        ),
        pytest.param(
            2,
            np.array([[300]], dtype=np.int64),
            # Partial (2/3): 2 draft + recovered[0,2]=202, offset=-2
            dict(
                processed=7, active=3, position=10, generated=4, kv=4, last=202
            ),
            id="partial-2-of-3",
        ),
    ],
)
class TestUpdateContextsSingle:
    def test_indices(
        self,
        first_rejected: int,
        bonus: npt.NDArray[np.int64] | None,
        expected: dict[str, int],
    ) -> None:
        ctx, tracker = setup_single_context()
        tracker.update_contexts(
            context_batch=[ctx],
            first_rejected_tokens=np.array([first_rejected], dtype=np.int64),
            recovered_tokens=np.array([[200, 201, 202]], dtype=np.int64),
            bonus_tokens=bonus,
            draft_tokens=np.array([[100, 101, 102]], dtype=np.int64),
            num_draft_tokens_generated=3,
        )
        assert_state(
            ctx,
            processed=expected["processed"],
            active=expected["active"],
            position=expected["position"],
            generated=expected["generated"],
        )
        assert tracker._draft_kv_start_idx[ctx.request_id] == expected["kv"]
        assert tracker._last_verified_token[ctx.request_id] == expected["last"]


# ---------------------------------------------------------------------------
# update_contexts: mixed batch (ctx0 all-accepted, ctx1 all-rejected)
# ---------------------------------------------------------------------------
class TestUpdateContextsMixedBatch:
    def test_two_contexts(self) -> None:
        ctx0, _ = setup_single_context()
        ctx1, _ = setup_single_context()
        tracker = EAGLEIndexTracker()
        tracker._draft_kv_start_idx[ctx0.request_id] = 4
        tracker._draft_kv_start_idx[ctx1.request_id] = 4

        tracker.update_contexts(
            context_batch=[ctx0, ctx1],
            first_rejected_tokens=np.array([3, 0], dtype=np.int64),
            recovered_tokens=np.array(
                [[900, 901, 902], [800, 801, 802]], dtype=np.int64
            ),
            bonus_tokens=np.array([[300], [301]], dtype=np.int64),
            draft_tokens=np.array(
                [[100, 101, 102], [200, 201, 202]], dtype=np.int64
            ),
            num_draft_tokens_generated=3,
        )

        assert_state(ctx0, processed=8, active=4, position=12, generated=5)
        assert tracker._last_verified_token[ctx0.request_id] == 300
        assert_state(ctx1, processed=5, active=1, position=6, generated=2)
        assert tracker._last_verified_token[ctx1.request_id] == 800


# ---------------------------------------------------------------------------
# update_contexts: draft_kv_start_idx capping
# ---------------------------------------------------------------------------
class TestDraftKVStartIndexCapping:
    def test_capped_to_processed(self) -> None:
        """Artificially high kv_idx (10) is capped to processed (7)."""
        ctx, _ = setup_single_context()
        tracker = EAGLEIndexTracker()
        tracker._draft_kv_start_idx[ctx.request_id] = 10

        tracker.update_contexts(
            context_batch=[ctx],
            first_rejected_tokens=np.array([2], dtype=np.int64),
            recovered_tokens=np.array([[200, 201, 202]], dtype=np.int64),
            bonus_tokens=np.array([[300]], dtype=np.int64),
            draft_tokens=np.array([[100, 101, 102]], dtype=np.int64),
            num_draft_tokens_generated=3,
        )
        assert tracker._draft_kv_start_idx[ctx.request_id] == 7

    def test_not_capped_when_below(self) -> None:
        ctx, tracker = setup_single_context()
        tracker.update_contexts(
            context_batch=[ctx],
            first_rejected_tokens=np.array([3], dtype=np.int64),
            recovered_tokens=np.array([[200, 201, 202]], dtype=np.int64),
            bonus_tokens=np.array([[300]], dtype=np.int64),
            draft_tokens=np.array([[100, 101, 102]], dtype=np.int64),
            num_draft_tokens_generated=3,
        )
        # processed=8, kv_idx=min(4,8)=4 → unchanged
        assert tracker._draft_kv_start_idx[ctx.request_id] == 4


# ---------------------------------------------------------------------------
# update_contexts: metrics
# ---------------------------------------------------------------------------
class TestMetrics:
    def test_all_accepted_with_bonus(self) -> None:
        ctx, tracker = setup_single_context()
        tracker.update_contexts(
            context_batch=[ctx],
            first_rejected_tokens=np.array([3], dtype=np.int64),
            recovered_tokens=np.array([[200, 201, 202]], dtype=np.int64),
            bonus_tokens=np.array([[300]], dtype=np.int64),
            draft_tokens=np.array([[100, 101, 102]], dtype=np.int64),
            num_draft_tokens_generated=3,
        )
        stats = tracker._metrics.get_stats()
        assert stats["draft_tokens_accepted"] == 3
        assert stats["bonus_tokens_used"] == 1
        assert stats["acceptance_rate"] == 1.0

    def test_partial_rejection(self) -> None:
        ctx, tracker = setup_single_context()
        tracker.update_contexts(
            context_batch=[ctx],
            first_rejected_tokens=np.array([1], dtype=np.int64),
            recovered_tokens=np.array([[200, 201, 202]], dtype=np.int64),
            bonus_tokens=np.array([[300]], dtype=np.int64),
            draft_tokens=np.array([[100, 101, 102]], dtype=np.int64),
            num_draft_tokens_generated=3,
        )
        stats = tracker._metrics.get_stats()
        assert stats["draft_tokens_accepted"] == 1
        assert stats["bonus_tokens_used"] == 0
        assert stats["acceptance_rate"] == pytest.approx(1 / 3)


# ===========================================================================
# _prepare_draft_batch tests
# ===========================================================================


class TestPrepareDraftBatchFullCycle:
    """End-to-end: target → prepare_draft(new) → update(all accepted) →
    prepare_draft(existing) → update(partial). Covers new request,
    existing request after all-accepted, and after partial."""

    def test_two_iterations(self) -> None:
        ctx, tracker = setup_post_target_context()

        # --- prepare_draft_batch (new request, not in _draft_kv_start_idx) ---
        tracker.prepare_draft_batch(batch=[ctx])
        assert_state(ctx, processed=4, active=1, position=5, generated=1)
        assert tracker._draft_kv_start_idx[ctx.request_id] == 4

        # --- update_contexts: all accepted + bonus ---
        tracker.update_contexts(
            context_batch=[ctx],
            first_rejected_tokens=np.array([3], dtype=np.int64),
            recovered_tokens=np.array([[200, 201, 202]], dtype=np.int64),
            bonus_tokens=np.array([[300]], dtype=np.int64),
            draft_tokens=np.array([[100, 101, 102]], dtype=np.int64),
            num_draft_tokens_generated=3,
        )
        assert_state(ctx, processed=8, active=4, position=12, generated=5)
        assert tracker._draft_kv_start_idx[ctx.request_id] == 4

        # --- prepare_draft_batch (existing request) ---
        tracker.prepare_draft_batch(batch=[ctx])
        assert_state(ctx, processed=8, active=1, position=9, generated=5)
        assert tracker._draft_kv_start_idx[ctx.request_id] == 8

        # --- update_contexts: partial acceptance (1 of 3) ---
        tracker.update_contexts(
            context_batch=[ctx],
            first_rejected_tokens=np.array([1], dtype=np.int64),
            recovered_tokens=np.array([[500, 501, 502]], dtype=np.int64),
            bonus_tokens=np.array([[600]], dtype=np.int64),
            draft_tokens=np.array([[400, 401, 402]], dtype=np.int64),
            num_draft_tokens_generated=3,
        )
        assert_state(ctx, processed=10, active=2, position=12, generated=7)
        assert tracker._draft_kv_start_idx[ctx.request_id] == 8
        assert tracker._last_verified_token[ctx.request_id] == 501


class TestPrepareDraftBatchMixedBatch:
    """Two contexts with different rejection histories go through
    _prepare_draft_batch together."""

    def test_two_contexts_different_kv_idx(self) -> None:
        ctx0, _ = setup_single_context()
        ctx1, _ = setup_single_context()
        tracker = EAGLEIndexTracker()
        tracker._draft_kv_start_idx[ctx0.request_id] = 4
        tracker._draft_kv_start_idx[ctx1.request_id] = 4

        # ctx0: all accepted + bonus → kv_idx stays 4, offset=-3
        tracker.update_contexts(
            context_batch=[ctx0],
            first_rejected_tokens=np.array([3], dtype=np.int64),
            recovered_tokens=np.array([[200, 201, 202]], dtype=np.int64),
            bonus_tokens=np.array([[300]], dtype=np.int64),
            draft_tokens=np.array([[100, 101, 102]], dtype=np.int64),
            num_draft_tokens_generated=3,
        )

        # ctx1: partial (2/3) → kv_idx stays 4, offset=-2
        tracker.update_contexts(
            context_batch=[ctx1],
            first_rejected_tokens=np.array([2], dtype=np.int64),
            recovered_tokens=np.array([[200, 201, 202]], dtype=np.int64),
            bonus_tokens=np.array([[300]], dtype=np.int64),
            draft_tokens=np.array([[100, 101, 102]], dtype=np.int64),
            num_draft_tokens_generated=3,
        )

        tracker.prepare_draft_batch(batch=[ctx0, ctx1])

        # ctx0: kv_idx += active(4) → 8, offset reset → active=1
        assert_state(ctx0, processed=8, active=1, position=9, generated=5)
        assert tracker._draft_kv_start_idx[ctx0.request_id] == 8

        # ctx1: kv_idx += active(3) → 7, offset reset → active=1
        assert_state(ctx1, processed=7, active=1, position=8, generated=4)
        assert tracker._draft_kv_start_idx[ctx1.request_id] == 7


# ===========================================================================
# reserve_token_space_for_batch tests
# ===========================================================================


class TestReserveTokenSpace:
    def test_bump_and_restore(self) -> None:
        ctx, _ = setup_single_context()
        with reserve_token_space_for_batch([ctx], num_tokens=3):
            assert_state(ctx, processed=4, active=4, position=8, generated=4)
        assert_state(ctx, processed=4, active=1, position=5, generated=1)

    def test_restore_on_exception(self) -> None:
        ctx, _ = setup_single_context()
        with pytest.raises(RuntimeError):
            with reserve_token_space_for_batch([ctx], num_tokens=5):
                assert ctx.tokens.current_position == 10
                raise RuntimeError("simulated failure")
        assert_state(ctx, processed=4, active=1, position=5, generated=1)

    def test_with_nonzero_processing_offset(self) -> None:
        """Bump interacts correctly with non-zero offset from update_contexts."""
        ctx, tracker = setup_single_context()
        tracker.update_contexts(
            context_batch=[ctx],
            first_rejected_tokens=np.array([3], dtype=np.int64),
            recovered_tokens=np.array([[200, 201, 202]], dtype=np.int64),
            bonus_tokens=np.array([[300]], dtype=np.int64),
            draft_tokens=np.array([[100, 101, 102]], dtype=np.int64),
            num_draft_tokens_generated=3,
        )
        # State: processed=8, active=4, offset=-3

        with reserve_token_space_for_batch([ctx], num_tokens=3):
            assert_state(ctx, processed=8, active=7, position=15, generated=8)
        assert_state(ctx, processed=8, active=4, position=12, generated=5)
