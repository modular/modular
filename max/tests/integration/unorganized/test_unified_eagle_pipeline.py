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

from __future__ import annotations

import numpy as np
from max.interfaces import RequestID
from max.interfaces.context import SamplingParams
from max.interfaces.tokens import TokenBuffer
from max.pipelines.core.context import TextContext
from max.pipelines.lib.speculative_decoding.utils import (
    SpeculativeDecodingMetrics,
)

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


def save_draft_token(ctx: TextContext, new_token: int) -> None:
    ctx.spec_decoding_state.saved_draft_tokens = [new_token]


def load_draft_tokens(
    contexts: list[TextContext],
) -> tuple[np.ndarray, int]:
    drafts = [ctx.spec_decoding_state.saved_draft_tokens for ctx in contexts]
    k = len(drafts[0])
    if k == 0:
        return np.zeros((len(contexts), 0), dtype=np.int64), 0
    return np.stack(drafts), k


def simulate_prefill(
    context_batch: list[TextContext],
    bonus: np.ndarray,
    new_draft_token: np.ndarray,
) -> None:
    for idx, ctx in enumerate(context_batch):
        ctx.update(int(bonus[idx, 0]))
    for idx, ctx in enumerate(context_batch):
        save_draft_token(ctx, int(new_draft_token[idx]))


def simulate_decode(
    context_batch: list[TextContext],
    num_accepted_draft_tokens: np.ndarray,
    next_tokens: np.ndarray,
    next_draft_tokens: np.ndarray,
    draft_tokens: np.ndarray,
    num_draft_tokens_generated: int,
) -> SpeculativeDecodingMetrics:
    """Simulate the unified EAGLE decode step.

    Mirrors the logic in UnifiedEAGLEPipeline.execute (unified_eagle.py):
    for each request, commit the accepted draft tokens, then commit the
    corrected next_token from the target, and save next_draft_tokens.
    """
    for batch_idx, ctx in enumerate(context_batch):
        for token_idx in range(num_accepted_draft_tokens[batch_idx]):
            if not ctx.is_done:
                ctx.update(int(draft_tokens[batch_idx, token_idx]))
        if not ctx.is_done:
            ctx.update(int(next_tokens[batch_idx]))
            ctx.spec_decoding_state.saved_draft_tokens = next_draft_tokens[
                batch_idx
            ].copy()

    return SpeculativeDecodingMetrics(
        num_speculative_tokens=num_draft_tokens_generated,
        draft_tokens_accepted=int(num_accepted_draft_tokens.sum()),
        draft_tokens_generated=num_draft_tokens_generated * len(context_batch),
    )


class TestUnifiedPrefill:
    def test_prefill_single_request(self) -> None:
        ctx = make_context([10, 11, 12, 13])
        simulate_prefill(
            context_batch=[ctx],
            bonus=np.array([[50]], dtype=np.int64),
            new_draft_token=np.array([100], dtype=np.int64),
        )
        assert ctx.tokens[-1] == 50
        assert ctx.tokens.generated_length == 1
        np.testing.assert_array_equal(
            ctx.spec_decoding_state.saved_draft_tokens, [100]
        )

    def test_prefill_batch(self) -> None:
        ctx0 = make_context([10, 11, 12, 13])
        ctx1 = make_context([20, 21])
        simulate_prefill(
            context_batch=[ctx0, ctx1],
            bonus=np.array([[50], [60]], dtype=np.int64),
            new_draft_token=np.array([100, 200], dtype=np.int64),
        )
        assert ctx0.tokens[-1] == 50
        assert ctx0.tokens.generated_length == 1
        np.testing.assert_array_equal(
            ctx0.spec_decoding_state.saved_draft_tokens, [100]
        )
        assert ctx1.tokens[-1] == 60
        assert ctx1.tokens.generated_length == 1
        np.testing.assert_array_equal(
            ctx1.spec_decoding_state.saved_draft_tokens, [200]
        )


class TestUnifiedDecode:
    def _setup_post_prefill(
        self, prompt: list[int], bonus_token: int, draft_token: int
    ) -> TextContext:
        ctx = make_context(prompt)
        ctx.update(bonus_token)
        save_draft_token(ctx, draft_token)
        return ctx

    def test_decode_draft_accepted(self) -> None:
        ctx = self._setup_post_prefill(
            [10, 11, 12, 13], bonus_token=50, draft_token=100
        )
        assert ctx.tokens.generated_length == 1

        draft_tokens, k = load_draft_tokens([ctx])
        assert k == 1
        np.testing.assert_array_equal(draft_tokens, [[100]])

        metrics = simulate_decode(
            context_batch=[ctx],
            num_accepted_draft_tokens=np.array([1], dtype=np.int64),
            next_tokens=np.array([300], dtype=np.int64),
            next_draft_tokens=np.array([[400]], dtype=np.int64),
            draft_tokens=draft_tokens,
            num_draft_tokens_generated=k,
        )
        assert ctx.tokens.generated_length == 3
        assert ctx.tokens[-1] == 300
        np.testing.assert_array_equal(
            ctx.spec_decoding_state.saved_draft_tokens, [400]
        )

    def test_decode_draft_rejected(self) -> None:
        ctx = self._setup_post_prefill(
            [10, 11, 12, 13], bonus_token=50, draft_token=100
        )
        assert ctx.tokens.generated_length == 1

        draft_tokens, k = load_draft_tokens([ctx])

        metrics = simulate_decode(
            context_batch=[ctx],
            num_accepted_draft_tokens=np.array([0], dtype=np.int64),
            next_tokens=np.array([200], dtype=np.int64),
            next_draft_tokens=np.array([[400]], dtype=np.int64),
            draft_tokens=draft_tokens,
            num_draft_tokens_generated=k,
        )
        assert ctx.tokens.generated_length == 2
        assert ctx.tokens[-1] == 200
        np.testing.assert_array_equal(
            ctx.spec_decoding_state.saved_draft_tokens, [400]
        )

    def test_decode_mixed_batch(self) -> None:
        ctx0 = self._setup_post_prefill(
            [10, 11, 12, 13], bonus_token=50, draft_token=100
        )
        ctx1 = self._setup_post_prefill(
            [20, 21], bonus_token=60, draft_token=200
        )

        draft_tokens = np.array([[100], [200]], dtype=np.int64)

        metrics = simulate_decode(
            context_batch=[ctx0, ctx1],
            num_accepted_draft_tokens=np.array([1, 0], dtype=np.int64),
            next_tokens=np.array([300, 700], dtype=np.int64),
            next_draft_tokens=np.array([[400], [500]], dtype=np.int64),
            draft_tokens=draft_tokens,
            num_draft_tokens_generated=1,
        )
        assert ctx0.tokens.generated_length == 3
        assert ctx0.tokens[-1] == 300
        assert ctx1.tokens.generated_length == 2
        assert ctx1.tokens[-1] == 700


class TestUnifiedDraftSaveLoad:
    def test_save_single_draft(self) -> None:
        ctx = make_context([10, 11, 12])
        save_draft_token(ctx, 100)
        np.testing.assert_array_equal(
            ctx.spec_decoding_state.saved_draft_tokens, [100]
        )

    def test_load_saved_draft(self) -> None:
        ctx = make_context([10, 11, 12])
        save_draft_token(ctx, 100)
        draft_tokens, k = load_draft_tokens([ctx])
        assert k == 1
        assert draft_tokens.shape == (1, 1)
        assert draft_tokens[0, 0] == 100

    def test_load_empty_for_prefill(self) -> None:
        ctx = make_context([10, 11, 12])
        draft_tokens, k = load_draft_tokens([ctx])
        assert k == 0
        assert draft_tokens.shape == (1, 0)

    def test_load_batch(self) -> None:
        ctx0 = make_context([10])
        ctx1 = make_context([20])
        save_draft_token(ctx0, 100)
        save_draft_token(ctx1, 200)
        draft_tokens, k = load_draft_tokens([ctx0, ctx1])
        assert k == 1
        assert draft_tokens.shape == (2, 1)
        np.testing.assert_array_equal(draft_tokens[:, 0], [100, 200])


class TestUnifiedMultiIteration:
    def test_prefill_then_two_decodes(self) -> None:
        ctx = make_context([10, 11, 12, 13])

        simulate_prefill(
            context_batch=[ctx],
            bonus=np.array([[50]], dtype=np.int64),
            new_draft_token=np.array([100], dtype=np.int64),
        )
        assert ctx.tokens[-1] == 50
        assert ctx.tokens.generated_length == 1
        np.testing.assert_array_equal(
            ctx.spec_decoding_state.saved_draft_tokens, [100]
        )

        draft_tokens, k = load_draft_tokens([ctx])
        np.testing.assert_array_equal(draft_tokens, [[100]])
        simulate_decode(
            context_batch=[ctx],
            num_accepted_draft_tokens=np.array([1], dtype=np.int64),
            next_tokens=np.array([300], dtype=np.int64),
            next_draft_tokens=np.array([[200]], dtype=np.int64),
            draft_tokens=draft_tokens,
            num_draft_tokens_generated=k,
        )
        assert ctx.tokens.generated_length == 3
        np.testing.assert_array_equal(
            ctx.spec_decoding_state.saved_draft_tokens, [200]
        )

        draft_tokens, k = load_draft_tokens([ctx])
        np.testing.assert_array_equal(draft_tokens, [[200]])
        simulate_decode(
            context_batch=[ctx],
            num_accepted_draft_tokens=np.array([0], dtype=np.int64),
            next_tokens=np.array([500], dtype=np.int64),
            next_draft_tokens=np.array([[600]], dtype=np.int64),
            draft_tokens=draft_tokens,
            num_draft_tokens_generated=k,
        )
        assert ctx.tokens.generated_length == 4

        final_tokens = list(ctx.tokens[: ctx.tokens.current_position])
        assert final_tokens == [10, 11, 12, 13, 50, 100, 300, 500]

        np.testing.assert_array_equal(
            ctx.spec_decoding_state.saved_draft_tokens, [600]
        )


class TestUnifiedMetrics:
    def _setup_post_prefill(self) -> TextContext:
        ctx = make_context([10, 11, 12, 13])
        ctx.update(50)
        save_draft_token(ctx, 100)
        return ctx

    def test_metrics_all_accepted(self) -> None:
        ctx = self._setup_post_prefill()
        metrics = simulate_decode(
            context_batch=[ctx],
            num_accepted_draft_tokens=np.array([1], dtype=np.int64),
            next_tokens=np.array([300], dtype=np.int64),
            next_draft_tokens=np.array([[400]], dtype=np.int64),
            draft_tokens=np.array([[100]], dtype=np.int64),
            num_draft_tokens_generated=1,
        )
        assert metrics.draft_tokens_accepted == 1
        assert metrics.draft_tokens_generated == 1
        assert metrics.acceptance_rate == 1.0

    def test_metrics_all_rejected(self) -> None:
        ctx = self._setup_post_prefill()
        metrics = simulate_decode(
            context_batch=[ctx],
            num_accepted_draft_tokens=np.array([0], dtype=np.int64),
            next_tokens=np.array([200], dtype=np.int64),
            next_draft_tokens=np.array([[400]], dtype=np.int64),
            draft_tokens=np.array([[100]], dtype=np.int64),
            num_draft_tokens_generated=1,
        )
        assert metrics.draft_tokens_accepted == 0
        assert metrics.draft_tokens_generated == 1
        assert metrics.acceptance_rate == 0.0
