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


import threading
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
from max.pipelines.core import StructuredOutputRegionDelimiters, TextContext
from max.pipelines.core.context import FUTURE_TOKEN
from max.pipelines.lib import (
    OverlapTextGenerationPipeline,
    TextGenerationPipeline,
)
from max.pipelines.lib.pipeline_variants.overlap_text_generation import (
    _MAX_GRAPH_CAPTURE_BATCH_SIZE,
    _SYNC_PRIME_CALLBACK_TIMEOUT_S,
    AsyncBatch,
)
from max.pipelines.lib.pipeline_variants.utils import (
    StructuredOutputHelper,
    _count_token_subsequence,
)
from max.pipelines.lib.registry import get_pipeline_for_task
from max.pipelines.modeling.types import (
    PipelineTask,
    RequestID,
    TextGenerationInputs,
    TokenBuffer,
)


@pytest.mark.parametrize(
    "content,special_tags,expected",
    [
        ([5, 1, 2, 5, 3, 5], [5], 3),  # single-token marker (Kimi case)
        ([1, 2, 3], [5], 0),
        ([], [5], 0),
        ([1, 2, 3], [], 0),  # empty tags never match
        ([9, 8, 1, 9, 8], [9, 8], 2),  # multi-token marker
        ([7, 7, 7, 7], [7, 7], 2),  # no double-counting overlaps
    ],
)
def test_count_token_subsequence(
    content: list[int], special_tags: list[int], expected: int
) -> None:
    assert _count_token_subsequence(content, special_tags) == expected


class _FakeMatcher:
    def __init__(self, accepting: bool, stopped: bool) -> None:
        self._a, self._s = accepting, stopped

    def is_accepting(self) -> bool:
        return self._a

    def is_stopped(self) -> bool:
        return self._s


def _fake_ctx(matcher: object, generated: list[int]) -> MagicMock:
    ctx = MagicMock()
    ctx.matcher = matcher
    ctx.grammar_enforced = True
    ctx.tools_forced = True
    ctx.snapshot_grammar_state.return_value.in_thinking_region = False
    ctx.tokens.generated = generated
    return ctx


def test_rejection_diagnostics_reports_mid_tool_call_state() -> None:
    """Diagnostics surface open-section + non-accepting (mid-tool-call) state.

    Logs raw token IDs only (no decoded text); the desyncing batch is
    reconstructable offline from ``committed_token_ids``.
    """
    helper = StructuredOutputHelper(
        enabled=True,
        tool_call_region_delimiters=StructuredOutputRegionDelimiters(
            start_token_ids=[256], end_token_ids=[257]
        ),
    )
    # One section-begin (256), no section-end -> open_sections == 1.
    ctx = _fake_ctx(_FakeMatcher(accepting=False, stopped=False), [256, 10, 11])
    diag = helper._rejection_diagnostics(
        ctx, committed_tokens=[1, 2, 27], committed_idx=2
    )
    assert "matcher_accepting=False" in diag
    assert "open_sections=1" in diag
    assert "reject_idx=2/3" in diag
    assert "committed_token_ids=[1, 2, 27]" in diag


def test_rejection_diagnostics_never_raises() -> None:
    """A diagnostic failure degrades to a placeholder, never crashes."""
    helper = StructuredOutputHelper(enabled=True)
    bad_matcher = MagicMock()
    bad_matcher.is_accepting.side_effect = RuntimeError("boom")
    ctx = _fake_ctx(bad_matcher, [1, 2, 3])
    diag = helper._rejection_diagnostics(
        ctx, committed_tokens=[5], committed_idx=0
    )
    assert diag.startswith("<diagnostics unavailable")


def test_throws_if_num_steps_gt_1() -> None:
    """Overlap pipeline should reject num_steps > 1."""
    pipeline = OverlapTextGenerationPipeline.__new__(
        OverlapTextGenerationPipeline
    )
    pipeline._pipeline_config = MagicMock()
    request_id = RequestID()
    ctx = TextContext(
        request_id=request_id,
        max_length=1000,
        tokens=TokenBuffer(np.array([42, 67, 21])),
    )
    inputs: TextGenerationInputs[TextContext] = TextGenerationInputs(
        batches=[[ctx]],
        num_steps=2,
    )
    with pytest.raises(
        ValueError,
        match=r"num_steps > 1 is not supported by the overlap pipeline",
    ):
        pipeline.execute(inputs)


def test_throws_if_enable_log_probs() -> None:
    pipeline = OverlapTextGenerationPipeline.__new__(
        OverlapTextGenerationPipeline
    )
    request_id = RequestID()
    ctx = TextContext(
        request_id=request_id,
        max_length=1000,
        tokens=TokenBuffer(np.array([42, 67, 21])),
        log_probabilities=1,
    )
    inputs: TextGenerationInputs[TextContext] = TextGenerationInputs(
        batches=[[ctx]],
        num_steps=1,
    )
    with pytest.raises(
        ValueError,
        match=r"Log probabilities are not supported with overlap pipeline",
    ):
        pipeline.execute(inputs)


@pytest.mark.parametrize(
    ("config_max_batch_size", "expected_capture_batch_size"),
    [
        (4096, _MAX_GRAPH_CAPTURE_BATCH_SIZE),
        (32, 32),
    ],
    ids=["capped", "uncapped"],
)
def test_warmup_graph_capture_batch_size(
    config_max_batch_size: int,
    expected_capture_batch_size: int,
) -> None:
    """warmup_graph_capture should cap the batch size at _MAX_GRAPH_CAPTURE_BATCH_SIZE."""
    pipeline = OverlapTextGenerationPipeline.__new__(
        OverlapTextGenerationPipeline
    )
    mock_model = MagicMock()
    mock_model.model = MagicMock()
    mock_model.execute = MagicMock()
    mock_model.max_seq_len = 2048
    pipeline._pipeline_model = mock_model
    pipeline._pipeline_config = MagicMock()
    pipeline._pipeline_config.runtime.max_batch_size = config_max_batch_size
    pipeline._kv_manager = MagicMock()
    mock_kv_params = MagicMock()
    mock_kv_params.page_size = 128
    pipeline._kv_manager.params = mock_kv_params
    pipeline._kv_manager.cache_params.return_value = mock_kv_params
    pipeline._kv_manager._total_num_pages = 100
    pipeline._spec_decode_state = None
    pipeline._kv_manager.num_caches = 1
    pipeline.session = MagicMock()

    with patch(
        "max.pipelines.lib.pipeline_variants.overlap_text_generation"
        ".ServeGraphCaptureRunner"
    ) as MockRunner:
        mock_runner_instance = MockRunner.return_value
        mock_runner_instance.warmup_pre_ready = MagicMock()

        pipeline.warmup_graph_capture()

        call_kwargs = MockRunner.call_args.kwargs
        assert call_kwargs["model"] is mock_model.model
        assert call_kwargs["execute_model"] is mock_model.execute
        assert call_kwargs["session"] is pipeline.session
        assert call_kwargs["kv_params"] is mock_kv_params
        assert callable(call_kwargs["warmup_model_inputs"])
        assert (
            call_kwargs["max_cache_length_upper_bound"]
            == mock_model.max_seq_len
        )
        assert call_kwargs["max_batch_size"] == expected_capture_batch_size
        assert (
            pipeline._max_graph_capture_batch_size
            == expected_capture_batch_size
        )


def _make_pipeline_config_mock(
    *, enable_overlap: bool, pipeline_role: str
) -> MagicMock:
    config = MagicMock()
    config.runtime.enable_overlap_scheduler = enable_overlap
    config.runtime.pipeline_role = pipeline_role
    config.speculative = None
    return config


def test_prefill_only_gets_overlap_pipeline() -> None:
    """Prefill-only DI workers use the overlap pipeline when overlap is enabled."""
    config = _make_pipeline_config_mock(
        enable_overlap=True, pipeline_role="prefill_only"
    )
    result = get_pipeline_for_task(PipelineTask.TEXT_GENERATION, config)
    assert result is OverlapTextGenerationPipeline[TextContext]


def test_decode_only_gets_overlap_pipeline() -> None:
    """Decode-only workers use the overlap pipeline when overlap is enabled."""
    config = _make_pipeline_config_mock(
        enable_overlap=True, pipeline_role="decode_only"
    )
    result = get_pipeline_for_task(PipelineTask.TEXT_GENERATION, config)
    assert result is OverlapTextGenerationPipeline[TextContext]


def test_prefill_and_decode_gets_overlap_pipeline() -> None:
    """Non-DI workers continue to use the overlap pipeline as before."""
    config = _make_pipeline_config_mock(
        enable_overlap=True, pipeline_role="prefill_and_decode"
    )
    result = get_pipeline_for_task(PipelineTask.TEXT_GENERATION, config)
    assert result is OverlapTextGenerationPipeline[TextContext]


def test_async_batch_sync_with_single_step_tokens() -> None:
    """AsyncBatch.sync_and_process_outputs handles single-step token shapes."""

    batch_size = 3

    # Create mock contexts
    contexts = []
    for i in range(batch_size):
        ctx = TextContext(
            request_id=RequestID(f"req_{i}"),
            max_length=1000,
            tokens=TokenBuffer(np.array([42, 67, 21])),
        )
        contexts.append(ctx)

    # Create mock inputs
    mock_inputs = MagicMock()
    mock_inputs.flat_batch = contexts

    # Create single-step tokens buffer [batch_size]
    generated_tokens = np.arange(batch_size, dtype=np.int64)

    # Create mock host buffer
    mock_host_buffer = MagicMock()
    mock_host_buffer.to_numpy.return_value = generated_tokens

    # Create mock event
    mock_event = MagicMock()

    # Create AsyncBatch
    async_batch: AsyncBatch[TextContext] = AsyncBatch(
        inputs=mock_inputs,
        generated_tokens_device=MagicMock(),
        generated_tokens_host=mock_host_buffer,
        copy_event=mock_event,
    )

    # Patch update_context_and_prepare_responses to verify it's called correctly
    with patch(
        "max.pipelines.lib.pipeline_variants.overlap_text_generation"
        ".update_context_and_prepare_responses"
    ) as mock_update:
        mock_update.return_value = {}

        async_batch.sync_and_process_outputs()

        # Verify the function was called with correct arguments
        mock_update.assert_called_once()
        call_args = mock_update.call_args

        # Check positional args - tokens should be reshaped to [batch_size, 1]
        tokens_arg = call_args[0][0]
        assert tokens_arg.shape == (batch_size, 1)

        contexts_arg = call_args[0][1]
        assert contexts_arg == contexts

        # Check keyword args (overlap path always uses single-step [batch, 1] tokens)
        assert call_args[1]["overwrite_future"] is True


class TestUpdateWithFutureTokenStructuredOutput:
    """Tests for update_with_future_token behavior with structured output."""

    def test_skips_fsm_when_matcher_present(self) -> None:
        """update_with_future_token should NOT advance FSM when matcher is present.

        For structured output, the FSM cannot accept placeholder tokens (-999).
        The FSM will be advanced later with the real token.
        """

        ctx = TextContext(
            request_id=RequestID("test"),
            max_length=1000,
            tokens=TokenBuffer(np.array([42, 67, 21])),
        )
        # Set up a mock matcher
        mock_matcher = MagicMock()
        mock_matcher.try_consume_tokens = MagicMock(return_value=1)
        ctx._matcher = mock_matcher

        initial_length = len(ctx.tokens.all)

        # Call update_with_future_token
        ctx.update_with_future_token()

        # Token buffer should have FUTURE_TOKEN appended
        assert len(ctx.tokens.all) == initial_length + 1
        assert ctx.tokens.all[-1] == FUTURE_TOKEN

        # FSM (matcher.try_consume_tokens) should NOT have been called
        mock_matcher.try_consume_tokens.assert_not_called()

        # On the other hand, update_with_future_token should advance FSM when no matcher present.
        # For non-structured output, the standard update() path is used
        ctx = TextContext(
            request_id=RequestID("test"),
            max_length=1000,
            tokens=TokenBuffer(np.array([42, 67, 21])),
        )
        # No matcher set (ctx.matcher is None)
        assert ctx.matcher is None

        initial_length = len(ctx.tokens.all)

        # Call update_with_future_token
        ctx.update_with_future_token()

        # Token buffer should have FUTURE_TOKEN appended
        assert len(ctx.tokens.all) == initial_length + 1
        assert ctx.tokens.all[-1] == FUTURE_TOKEN


class TestSyncAndProcessOutputsStructuredOutput:
    """Tests for sync_and_process_outputs with structured output."""

    def test_advances_fsm_with_real_token(self) -> None:
        """sync_and_process_outputs should advance FSM with real token for structured output.

        When syncing the previous batch, the FSM should be advanced with the
        actual sampled token (not the placeholder).
        """
        batch_size = 2
        real_tokens = np.array([100, 200], dtype=np.int64)

        # Create contexts with mock matchers. ``advance_fsm`` only calls
        # ``try_consume_tokens`` while ``grammar_enforced=True``, so flip
        # it on for this test path.
        contexts = []
        for i in range(batch_size):
            ctx = TextContext(
                request_id=RequestID(f"req_{i}"),
                max_length=1000,
                tokens=TokenBuffer(np.array([42, 67, 21])),
            )
            ctx.grammar_enforced = True
            mock_matcher = MagicMock()
            mock_matcher.try_consume_tokens = MagicMock(return_value=1)
            ctx._matcher = mock_matcher
            contexts.append(ctx)

        # Create mock inputs
        mock_inputs = MagicMock()
        mock_inputs.flat_batch = contexts

        # Create mock host buffer with real tokens
        mock_host_buffer = MagicMock()
        mock_host_buffer.to_numpy.return_value = real_tokens
        mock_host_buffer.shape = real_tokens.shape

        # Create mock event
        mock_event = MagicMock()

        # Create mock structured output helper (required for FSM advancement)
        mock_structured_output = MagicMock()
        mock_structured_output.enabled = True

        # Create AsyncBatch
        async_batch: AsyncBatch[TextContext] = AsyncBatch(
            inputs=mock_inputs,
            generated_tokens_device=MagicMock(),
            generated_tokens_host=mock_host_buffer,
            copy_event=mock_event,
            structured_output=mock_structured_output,
        )

        # Patch update_context_and_prepare_responses
        with patch(
            "max.pipelines.lib.pipeline_variants.overlap_text_generation"
            ".update_context_and_prepare_responses"
        ) as mock_update:
            mock_update.return_value = {}

            async_batch.sync_and_process_outputs()

            # Verify FSM was advanced with real tokens for each context
            for i, ctx in enumerate(contexts):
                assert ctx._matcher is not None
                ctx._matcher.try_consume_tokens.assert_called_once_with(
                    [int(real_tokens[i])]
                )

    def test_updates_bitmask_for_continuing_requests(self) -> None:
        """sync_and_process_outputs should update bitmask for requests continuing to next batch."""
        real_token = np.array([100], dtype=np.int64)

        # Create context with mock matcher
        ctx = TextContext(
            request_id=RequestID("continuing_req"),
            max_length=1000,
            tokens=TokenBuffer(np.array([42, 67, 21])),
        )
        mock_matcher = MagicMock()
        mock_matcher.try_consume_tokens = MagicMock(return_value=1)
        ctx._matcher = mock_matcher

        # Previous batch inputs
        mock_inputs = MagicMock()
        mock_inputs.flat_batch = [ctx]

        # Create mock host buffer
        mock_host_buffer = MagicMock()
        mock_host_buffer.to_numpy.return_value = real_token
        mock_host_buffer.shape = real_token.shape

        # Create mock StructuredOutputHelper
        mock_structured_output = MagicMock()

        # Create AsyncBatch for previous batch
        async_batch: AsyncBatch[TextContext] = AsyncBatch(
            inputs=mock_inputs,
            generated_tokens_device=MagicMock(),
            generated_tokens_host=mock_host_buffer,
            copy_event=MagicMock(),
            structured_output=mock_structured_output,
        )

        # Current batch also contains this request (continuing)
        curr_flat_batch = [ctx]
        bitmask = np.zeros((1, 10), dtype=np.int32)
        mock_sampling_processor = MagicMock()

        with patch(
            "max.pipelines.lib.pipeline_variants.overlap_text_generation"
            ".update_context_and_prepare_responses"
        ) as mock_update:
            mock_update.return_value = {}

            async_batch.sync_and_process_outputs(
                curr_flat_batch=curr_flat_batch,
                bitmask=bitmask,
                sampling_processor=mock_sampling_processor,
            )

            # Verify bitmask was filled for continuing request via StructuredOutputHelper
            mock_structured_output.fill_bitmask.assert_called_once_with(
                ctx,
                bitmask,
                0,  # curr_idx = 0
            )

            # Verify bitmask was transferred to device
            mock_sampling_processor.update_bitmask.assert_called_once_with(
                bitmask
            )

    def test_skips_fsm_for_chunked_prefill(self) -> None:
        """sync_and_process_outputs should skip FSM advancement for actively chunked contexts."""
        real_token = np.array([100], dtype=np.int64)

        # Create context with mock matcher that is actively chunked
        ctx = TextContext(
            request_id=RequestID("chunked_req"),
            max_length=1000,
            tokens=TokenBuffer(np.array([42, 67, 21])),
        )
        ctx.tokens._actively_chunked = True  # Simulate chunked prefill
        mock_matcher = MagicMock()
        mock_matcher.try_consume_tokens = MagicMock(return_value=1)
        ctx._matcher = mock_matcher

        mock_inputs = MagicMock()
        mock_inputs.flat_batch = [ctx]

        mock_host_buffer = MagicMock()
        mock_host_buffer.to_numpy.return_value = real_token
        mock_host_buffer.shape = real_token.shape

        async_batch: AsyncBatch[TextContext] = AsyncBatch(
            inputs=mock_inputs,
            generated_tokens_device=MagicMock(),
            generated_tokens_host=mock_host_buffer,
            copy_event=MagicMock(),
        )

        with patch(
            "max.pipelines.lib.pipeline_variants.overlap_text_generation"
            ".update_context_and_prepare_responses"
        ) as mock_update:
            mock_update.return_value = {}

            async_batch.sync_and_process_outputs()

            # FSM should NOT be advanced for actively chunked context
            mock_matcher.try_consume_tokens.assert_not_called()


class TestAdvanceFsmAndComputeBitmasks:
    """Tests for StructuredOutputHelper.advance_fsm_and_compute_bitmasks."""

    def _make_helper(self, vocab_size: int = 64) -> StructuredOutputHelper:
        return StructuredOutputHelper(enabled=True, vocab_size=vocab_size)

    def _make_context_with_matcher(
        self, always_accept: bool = True
    ) -> tuple[TextContext, MagicMock]:
        ctx = TextContext(
            request_id=RequestID("req"),
            max_length=1000,
            tokens=TokenBuffer(np.array([1, 2, 3])),
        )
        # advance_fsm_and_compute_bitmasks only advances the matcher while
        # ``grammar_enforced=True``. Default the test context to enforced so
        # these tests exercise the matcher-advance path.
        ctx.grammar_enforced = True
        mock_matcher = MagicMock()
        mock_matcher.try_consume_tokens = MagicMock(
            return_value=1 if always_accept else 0
        )
        ctx._matcher = mock_matcher
        return ctx, mock_matcher

    def test_unconstrained_context_bitmask_stays_all_minus_one(self) -> None:
        """Context with no matcher keeps all bitmask slots at -1 (unconstrained)."""
        helper = self._make_helper()
        ctx = TextContext(
            request_id=RequestID("req"),
            max_length=1000,
            tokens=TokenBuffer(np.array([1, 2, 3])),
        )
        assert ctx.matcher is None

        bitmask_out = np.zeros((1, 3, 2), dtype=np.int32)

        with patch("llguidance.numpy.fill_next_token_bitmask") as mock_fill:
            helper.advance_fsm_and_compute_bitmasks(
                context_batch=[ctx],
                accepted_draft_tokens=np.zeros((1, 2), dtype=np.int64),
                num_accepted=np.zeros(1, dtype=np.int32),
                bonus_tokens=np.array([5], dtype=np.int64),
                next_draft_tokens=np.array([[10, 11]], dtype=np.int64),
                bitmask_out=bitmask_out,
            )

        mock_fill.assert_not_called()
        assert np.all(bitmask_out == -1)

    def test_part1_advances_through_accepted_drafts_and_bonus(self) -> None:
        """Part 1 advances FSM through accepted draft tokens then bonus (no rollback)."""
        helper = self._make_helper()
        ctx, mock_matcher = self._make_context_with_matcher()
        bitmask_out = np.full((1, 3, 2), -1, dtype=np.int32)

        with patch("llguidance.numpy.fill_next_token_bitmask"):
            helper.advance_fsm_and_compute_bitmasks(
                context_batch=[ctx],
                accepted_draft_tokens=np.array([[7, 8]], dtype=np.int64),
                num_accepted=np.array([2], dtype=np.int32),
                bonus_tokens=np.array([9], dtype=np.int64),
                next_draft_tokens=np.array([[10, 11]], dtype=np.int64),
                bitmask_out=bitmask_out,
            )

        # Part 1 consumes one token at a time (accepted drafts then bonus) so
        # the enforcement state machine can flip mid-sequence on special
        # tokens. Followed by Part 2 speculatively consuming next_draft_tokens.
        all_calls = mock_matcher.try_consume_tokens.call_args_list
        assert all_calls[:3] == [call([7]), call([8]), call([9])]

    def test_part1_skips_accepted_drafts_when_zero_accepted(self) -> None:
        """When n_accepted=0, only the bonus token is consumed in Part 1."""
        helper = self._make_helper()
        ctx, mock_matcher = self._make_context_with_matcher()
        bitmask_out = np.full((1, 2, 2), -1, dtype=np.int32)

        with patch("llguidance.numpy.fill_next_token_bitmask"):
            helper.advance_fsm_and_compute_bitmasks(
                context_batch=[ctx],
                accepted_draft_tokens=np.array([[7, 8]], dtype=np.int64),
                num_accepted=np.array([0], dtype=np.int32),
                bonus_tokens=np.array([9], dtype=np.int64),
                next_draft_tokens=np.array([[10]], dtype=np.int64),
                bitmask_out=bitmask_out,
            )

        all_calls = mock_matcher.try_consume_tokens.call_args_list
        # No call for either accepted draft token; bonus is consumed first.
        assert call([7]) not in all_calls
        assert call([8]) not in all_calls
        assert all_calls[0] == call([9])

    def test_part2_fills_position_0_bitmask_after_part1(self) -> None:
        """Position 0 of bitmask is filled with current FSM state (after Part 1 advance)."""
        helper = self._make_helper(vocab_size=64)
        ctx, _ = self._make_context_with_matcher()
        bitmask_out = np.full((1, 3, 2), -1, dtype=np.int32)

        with patch("llguidance.numpy.fill_next_token_bitmask") as mock_fill:
            helper.advance_fsm_and_compute_bitmasks(
                context_batch=[ctx],
                accepted_draft_tokens=np.zeros((1, 0), dtype=np.int64),
                num_accepted=np.zeros(1, dtype=np.int32),
                bonus_tokens=np.array([5], dtype=np.int64),
                next_draft_tokens=np.array([[10, 11]], dtype=np.int64),
                bitmask_out=bitmask_out,
            )

        # fill_next_token_bitmask called at least once (position 0)
        assert mock_fill.call_count >= 1
        # First call is for position 0
        first_kwargs = mock_fill.call_args_list[0][1]
        assert first_kwargs["index"] == 0

    def test_part2_speculative_advance_then_rollback(self) -> None:
        """Part 2 speculatively advances through next draft tokens then rolls back."""
        helper = self._make_helper()
        ctx, mock_matcher = self._make_context_with_matcher(always_accept=True)
        bitmask_out = np.full((1, 3, 2), -1, dtype=np.int32)

        with patch("llguidance.numpy.fill_next_token_bitmask"):
            helper.advance_fsm_and_compute_bitmasks(
                context_batch=[ctx],
                accepted_draft_tokens=np.zeros((1, 0), dtype=np.int64),
                num_accepted=np.zeros(1, dtype=np.int32),
                bonus_tokens=np.array([5], dtype=np.int64),
                next_draft_tokens=np.array([[10, 11]], dtype=np.int64),
                bitmask_out=bitmask_out,
            )

        # Both next draft tokens consumed → rollback(2)
        mock_matcher.rollback.assert_called_once_with(2)

    def test_part2_stops_and_no_rollback_when_fsm_rejects_first_draft(
        self,
    ) -> None:
        """When FSM rejects the first next draft token, rollback is not called."""
        helper = self._make_helper()
        ctx, mock_matcher = self._make_context_with_matcher()
        # Bonus token accepted (Part 1), first next_draft rejected (Part 2)
        mock_matcher.try_consume_tokens.side_effect = [1, 0]
        bitmask_out = np.full((1, 3, 2), -1, dtype=np.int32)

        with patch("llguidance.numpy.fill_next_token_bitmask"):
            helper.advance_fsm_and_compute_bitmasks(
                context_batch=[ctx],
                accepted_draft_tokens=np.zeros((1, 0), dtype=np.int64),
                num_accepted=np.zeros(1, dtype=np.int32),
                bonus_tokens=np.array([5], dtype=np.int64),
                next_draft_tokens=np.array([[10, 11]], dtype=np.int64),
                bitmask_out=bitmask_out,
            )

        # No tokens consumed in Part 2 → no rollback
        mock_matcher.rollback.assert_not_called()


class TestBuildBitmaskCallback:
    """Tests for OverlapTextGenerationPipeline._build_bitmask_callback."""

    def test_callback_calls_advance_fsm_and_compute_bitmasks(self) -> None:
        """The returned callback delegates to advance_fsm_and_compute_bitmasks."""
        pipeline = OverlapTextGenerationPipeline.__new__(
            OverlapTextGenerationPipeline
        )
        mock_so = MagicMock()
        pipeline._structured_output = mock_so

        ctx = TextContext(
            request_id=RequestID("r"),
            max_length=100,
            tokens=TokenBuffer(np.array([1])),
        )
        bonus_np = np.array([5], dtype=np.int64)
        num_acc_np = np.zeros(1, dtype=np.int64)
        draft_np = np.zeros((1, 2), dtype=np.int64)
        next_draft_np = np.zeros((1, 2), dtype=np.int64)
        bitmask_np = np.full((1, 3, 2), -1, dtype=np.int32)
        # Bool unpacked output target — populated by the callback after the
        # int32 advance_fsm_and_compute_bitmasks call returns.
        overlap_bool_np = np.zeros((1, 3, 64), dtype=np.bool_)

        done_event = threading.Event()
        callback = pipeline._build_bitmask_callback(
            context_batch=[ctx],
            bonus_tokens_np=bonus_np,
            num_accepted_np=num_acc_np,
            accepted_draft_tokens_np=draft_np,
            next_draft_tokens_np=next_draft_np,
            bitmask_pinned_np=bitmask_np,
            overlap_bool_pinned_np=overlap_bool_np,
            done_event=done_event,
        )

        callback()
        assert done_event.is_set()

        mock_so.advance_fsm_and_compute_bitmasks.assert_called_once_with(
            context_batch=[ctx],
            accepted_draft_tokens=draft_np,
            num_accepted=num_acc_np,
            bonus_tokens=bonus_np,
            next_draft_tokens=next_draft_np,
            bitmask_out=bitmask_np,
        )

    def test_callback_logs_error_on_exception(self) -> None:
        """Callback catches exceptions and logs them instead of propagating."""
        pipeline = OverlapTextGenerationPipeline.__new__(
            OverlapTextGenerationPipeline
        )
        mock_so = MagicMock()
        mock_so.advance_fsm_and_compute_bitmasks.side_effect = RuntimeError(
            "boom"
        )
        pipeline._structured_output = mock_so

        done_event = threading.Event()
        callback = pipeline._build_bitmask_callback(
            context_batch=[],
            bonus_tokens_np=np.array([], dtype=np.int64),
            num_accepted_np=np.array([], dtype=np.int64),
            accepted_draft_tokens_np=np.zeros((0, 0), dtype=np.int64),
            next_draft_tokens_np=np.zeros((0, 0), dtype=np.int64),
            bitmask_pinned_np=np.zeros((0, 0, 0), dtype=np.int32),
            overlap_bool_pinned_np=np.zeros((0, 0, 0), dtype=np.bool_),
            done_event=done_event,
        )

        # Must not raise — exceptions are caught and logged
        callback()
        # ``finally`` in the callback signals even on exception so the
        # next iter's sync_prime can't deadlock.
        assert done_event.is_set()


class TestEnqueueAsyncBitmaskCallback:
    """Tests for OverlapTextGenerationPipeline._enqueue_async_bitmask_callback."""

    def _make_pipeline(
        self, structured_output_enabled: bool = True
    ) -> OverlapTextGenerationPipeline[TextContext]:
        pipeline = OverlapTextGenerationPipeline.__new__(
            OverlapTextGenerationPipeline
        )
        mock_so = MagicMock()
        mock_so.enabled = structured_output_enabled
        pipeline._structured_output = mock_so
        mock_config = MagicMock()
        mock_config.needs_bitmask_constraints = structured_output_enabled
        pipeline._pipeline_config = mock_config
        return pipeline

    def test_returns_false_when_structured_output_disabled(self) -> None:
        """Returns False immediately when structured output is not enabled."""
        pipeline = self._make_pipeline(structured_output_enabled=False)
        pipeline._spec_decode_state = MagicMock()

        result = pipeline._enqueue_async_bitmask_callback(
            context_batch=[],
            num_draft_tokens_to_verify=2,
            next_draft_k=2,
            verify_draft_tokens=True,
        )

        assert result is False

    def test_returns_false_when_spec_decode_state_is_none(self) -> None:
        """Returns False when spec decode state is not initialized."""
        pipeline = self._make_pipeline()
        pipeline._spec_decode_state = None

        result = pipeline._enqueue_async_bitmask_callback(
            context_batch=[],
            num_draft_tokens_to_verify=2,
            next_draft_k=2,
            verify_draft_tokens=True,
        )

        assert result is False

    def test_returns_false_for_prefill_batch(self) -> None:
        """Returns False without enqueueing when verify_draft_tokens=False (prefill)."""
        pipeline = self._make_pipeline()
        mock_spec_state = MagicMock()
        # All persistent pinned buffers present so the prefill-only short-circuit
        # is the one that fires (not the buffer-missing one).
        for name in (
            "persistent_bitmask_pinned",
            "persistent_bonus_tokens_pinned",
            "persistent_num_accepted_pinned",
            "persistent_accepted_draft_tokens_pinned",
            "persistent_next_draft_tokens_pinned",
        ):
            setattr(mock_spec_state, name, MagicMock())
        pipeline._spec_decode_state = mock_spec_state

        result = pipeline._enqueue_async_bitmask_callback(
            context_batch=[],
            num_draft_tokens_to_verify=0,
            next_draft_k=2,
            verify_draft_tokens=False,
        )

        assert result is False

    def test_returns_true_and_enqueues_for_decode_batch(self) -> None:
        """Returns True and dispatches via overlap_state.enqueue_async_callback
        for a decode batch."""
        pipeline = self._make_pipeline()
        # The pipeline's structured_output also drives any nested behavior the
        # callback closure may invoke at construction time.
        pipeline._structured_output.vocab_size = 64

        batch_size = 1
        num_draft = 2
        num_positions = num_draft + 1
        packed_vocab = 4
        vocab_size = 64

        mock_spec_state = MagicMock()
        # Configure each persistent pinned buffer so `to_numpy()` returns a
        # writable numpy array of the right shape/dtype.
        bitmask_pinned = MagicMock()
        bitmask_pinned.to_numpy.return_value = np.full(
            (batch_size, num_positions, packed_vocab),
            -1,
            dtype=np.int32,
        )
        bonus_tokens_pinned = MagicMock()
        bonus_tokens_pinned.to_numpy.return_value = np.array(
            [5], dtype=np.int64
        )
        num_accepted_pinned = MagicMock()
        num_accepted_pinned.to_numpy.return_value = np.zeros(1, dtype=np.int64)
        accepted_draft_tokens_pinned = MagicMock()
        accepted_draft_tokens_pinned.to_numpy.return_value = np.zeros(
            (batch_size, num_draft), dtype=np.int64
        )
        next_draft_tokens_pinned = MagicMock()
        next_draft_tokens_pinned.to_numpy.return_value = np.zeros(
            (batch_size, num_draft), dtype=np.int64
        )

        mock_spec_state.persistent_bitmask_pinned = bitmask_pinned
        mock_spec_state.persistent_bonus_tokens_pinned = bonus_tokens_pinned
        mock_spec_state.persistent_num_accepted_pinned = num_accepted_pinned
        mock_spec_state.persistent_accepted_draft_tokens_pinned = (
            accepted_draft_tokens_pinned
        )
        mock_spec_state.persistent_next_draft_tokens_pinned = (
            next_draft_tokens_pinned
        )
        mock_spec_state.callback_request_ids = []
        mock_spec_state.has_precomputed_bitmask = False

        # Overlap state must exist (structured output is enabled). Stub the
        # pinned_bitmask view so the enqueue path can take its leading rows.
        overlap_pinned = MagicMock()
        overlap_pinned.to_numpy.return_value = np.zeros(
            (batch_size, num_positions, vocab_size), dtype=np.bool_
        )
        mock_overlap_state = MagicMock()
        mock_overlap_state.pinned_bitmask = overlap_pinned
        mock_spec_state.overlap_state = mock_overlap_state

        pipeline._spec_decode_state = mock_spec_state
        pipeline._devices = [MagicMock()]
        pipeline._disable_overlap = False

        rid = RequestID("r")
        ctx = TextContext(
            request_id=rid,
            max_length=100,
            tokens=TokenBuffer(np.array([1])),
        )

        with patch.object(
            pipeline,
            "_build_bitmask_callback",
            return_value=lambda: None,
        ):
            result = pipeline._enqueue_async_bitmask_callback(
                context_batch=[ctx],
                num_draft_tokens_to_verify=num_draft,
                next_draft_k=num_draft,
                verify_draft_tokens=True,
            )

        assert result is True
        # The handoff event is stashed on ``SpecDecodeState`` so the next
        # iter's ``_assign_bitmask_inputs`` can wait on it before
        # ``sync_prime`` even after ``_prev_batch`` is cleared between
        # requests.
        assert isinstance(
            mock_spec_state.last_callback_done_event, threading.Event
        )
        mock_overlap_state.enqueue_async_callback.assert_called_once()
        assert mock_spec_state.has_precomputed_bitmask is True
        assert mock_spec_state.callback_request_ids == [rid]

    def test_snapshot_preserves_multi_context_row_order(self) -> None:
        """``callback_request_ids`` is snapshotted in the exact order
        of ``context_batch``, not sorted. ``_assign_bitmask_inputs``
        on the next iter compares row-for-row against
        ``current_request_ids``, so a sort or any reordering here
        would make the adoption guard reject every identity case.
        Equivalent intent to the deleted
        ``test_enqueue_snapshots_callback_request_ids`` test."""
        pipeline = self._make_pipeline()
        pipeline._structured_output.vocab_size = 64

        batch_size = 3
        num_draft = 2
        num_positions = num_draft + 1
        packed_vocab = 4
        vocab_size = 64

        mock_spec_state = MagicMock()
        bitmask_pinned = MagicMock()
        bitmask_pinned.to_numpy.return_value = np.full(
            (batch_size, num_positions, packed_vocab),
            -1,
            dtype=np.int32,
        )
        bonus_tokens_pinned = MagicMock()
        bonus_tokens_pinned.to_numpy.return_value = np.zeros(
            batch_size, dtype=np.int64
        )
        num_accepted_pinned = MagicMock()
        num_accepted_pinned.to_numpy.return_value = np.zeros(
            batch_size, dtype=np.int64
        )
        accepted_draft_tokens_pinned = MagicMock()
        accepted_draft_tokens_pinned.to_numpy.return_value = np.zeros(
            (batch_size, num_draft), dtype=np.int64
        )
        next_draft_tokens_pinned = MagicMock()
        next_draft_tokens_pinned.to_numpy.return_value = np.zeros(
            (batch_size, num_draft), dtype=np.int64
        )

        mock_spec_state.persistent_bitmask_pinned = bitmask_pinned
        mock_spec_state.persistent_bonus_tokens_pinned = bonus_tokens_pinned
        mock_spec_state.persistent_num_accepted_pinned = num_accepted_pinned
        mock_spec_state.persistent_accepted_draft_tokens_pinned = (
            accepted_draft_tokens_pinned
        )
        mock_spec_state.persistent_next_draft_tokens_pinned = (
            next_draft_tokens_pinned
        )
        mock_spec_state.callback_request_ids = []
        mock_spec_state.has_precomputed_bitmask = False

        overlap_pinned = MagicMock()
        overlap_pinned.to_numpy.return_value = np.zeros(
            (batch_size, num_positions, vocab_size), dtype=np.bool_
        )
        mock_overlap_state = MagicMock()
        mock_overlap_state.pinned_bitmask = overlap_pinned
        mock_spec_state.overlap_state = mock_overlap_state

        pipeline._spec_decode_state = mock_spec_state
        pipeline._devices = [MagicMock()]
        pipeline._disable_overlap = False

        # Deliberately non-sorted ordering so a stray sort would be
        # caught.
        ordered_rids = [
            RequestID("z"),
            RequestID("a"),
            RequestID("m"),
        ]
        contexts = [
            TextContext(
                request_id=rid,
                max_length=100,
                tokens=TokenBuffer(np.array([1])),
            )
            for rid in ordered_rids
        ]

        with patch.object(
            pipeline,
            "_build_bitmask_callback",
            return_value=lambda: None,
        ):
            result = pipeline._enqueue_async_bitmask_callback(
                context_batch=contexts,
                num_draft_tokens_to_verify=num_draft,
                next_draft_k=num_draft,
                verify_draft_tokens=True,
            )

        assert result is True
        assert mock_spec_state.has_precomputed_bitmask is True
        # Snapshot preserves ``context_batch`` row order verbatim.
        assert mock_spec_state.callback_request_ids == ordered_rids


class TestInitializeBitmaskWithGrammar:
    """Tests for initialize_bitmask behavior with grammar field.

    These tests verify that structured output bitmasks are allocated when
    a context has a grammar set (e.g., for tool calls), even if json_schema
    is not set.
    """

    def _create_overlap_pipeline_with_structured_output(
        self, enabled: bool = True
    ) -> OverlapTextGenerationPipeline[TextContext]:
        """Create a mock OverlapTextGenerationPipeline with structured output."""
        pipeline = OverlapTextGenerationPipeline.__new__(
            OverlapTextGenerationPipeline
        )
        mock_structured_output = MagicMock()
        mock_structured_output.enabled = enabled
        mock_structured_output.vocab_size = 32000
        mock_structured_output.allocate_bitmask.return_value = np.zeros(
            (1, 1000), dtype=np.int32
        )
        pipeline._structured_output = mock_structured_output
        return pipeline

    def _create_text_pipeline_with_structured_output(
        self, enabled: bool = True
    ) -> TextGenerationPipeline[TextContext]:
        """Create a mock TextGenerationPipeline with structured output."""
        pipeline = TextGenerationPipeline.__new__(TextGenerationPipeline)
        mock_structured_output = MagicMock()
        mock_structured_output.enabled = enabled
        mock_structured_output.vocab_size = 32000
        mock_structured_output.allocate_bitmask.return_value = np.zeros(
            (1, 1000), dtype=np.int32
        )
        pipeline._structured_output = mock_structured_output
        return pipeline

    def test_allocates_bitmask_when_grammar_only_overlap_pipeline(self) -> None:
        """initialize_bitmask should allocate when grammar is set but json_schema is None.

        This simulates tool call scenarios where grammar is used for constrained
        decoding without a JSON schema.
        """
        pipeline = self._create_overlap_pipeline_with_structured_output()

        # Create context with grammar but no json_schema
        ctx = TextContext(
            request_id=RequestID("grammar_only"),
            max_length=1000,
            tokens=TokenBuffer(np.array([42, 67, 21])),
            grammar="<some llguidance grammar>",
        )
        assert ctx.json_schema is None
        assert ctx.grammar is not None

        result = pipeline.initialize_bitmask([ctx])

        # Should have allocated a bitmask
        assert result is not None

    def test_allocates_bitmask_when_grammar_only_text_pipeline(self) -> None:
        """initialize_bitmask should allocate when grammar is set (TextGenerationPipeline)."""
        pipeline = self._create_text_pipeline_with_structured_output()

        ctx = TextContext(
            request_id=RequestID("grammar_only"),
            max_length=1000,
            tokens=TokenBuffer(np.array([42, 67, 21])),
            grammar="<some llguidance grammar>",
        )

        result = pipeline.initialize_bitmask([ctx])

        assert result is not None

    def test_returns_none_when_both_grammar_and_json_schema_none(self) -> None:
        """initialize_bitmask should return None when neither grammar nor json_schema is set."""
        pipeline = self._create_overlap_pipeline_with_structured_output()

        # Create context with neither grammar nor json_schema
        ctx = TextContext(
            request_id=RequestID("no_constraint"),
            max_length=1000,
            tokens=TokenBuffer(np.array([42, 67, 21])),
        )
        assert ctx.json_schema is None
        assert ctx.grammar is None

        result = pipeline.initialize_bitmask([ctx])

        # Should NOT allocate a bitmask
        assert result is None

    def test_allocates_bitmask_for_grammar_even_when_response_format_schema_disabled(
        self,
    ) -> None:
        """initialize_bitmask should allocate for grammar even when enable_response_format_schema=False.

        Tool calling (grammar) should work regardless of the --enable-structured-output
        flag. Only user-provided json_schema requires the flag.
        """
        # Create pipeline with enabled=True (constrained decoding available)
        # but enable_response_format_schema=False (json_schema not allowed)
        pipeline = self._create_overlap_pipeline_with_structured_output(
            enabled=True
        )

        # Grammar for tool calls should work even without --enable-structured-output
        ctx = TextContext(
            request_id=RequestID("grammar_for_tool_call"),
            max_length=1000,
            tokens=TokenBuffer(np.array([42, 67, 21])),
            grammar="<tool call grammar>",
        )

        result = pipeline.initialize_bitmask([ctx])

        # Should allocate bitmask for tool call grammar
        assert result is not None

    def test_allocates_bitmask_for_heterogeneous_batch_with_grammar(
        self,
    ) -> None:
        """initialize_bitmask should allocate for batch with mixed grammar/no-grammar contexts.

        A batch containing at least one context with grammar should trigger
        bitmask allocation, even if other contexts have no constraints.
        """
        pipeline = self._create_overlap_pipeline_with_structured_output()

        # Context with grammar (e.g., tool call)
        ctx_with_grammar = TextContext(
            request_id=RequestID("with_grammar"),
            max_length=1000,
            tokens=TokenBuffer(np.array([42, 67, 21])),
            grammar="<tool call grammar>",
        )

        # Context without any constraint
        ctx_no_constraint = TextContext(
            request_id=RequestID("no_constraint"),
            max_length=1000,
            tokens=TokenBuffer(np.array([10, 20, 30])),
        )

        batch = [ctx_with_grammar, ctx_no_constraint]

        result = pipeline.initialize_bitmask(batch)

        # Should allocate bitmask for the entire batch
        assert result is not None

    def test_allocates_bitmask_for_heterogeneous_batch_with_json_schema(
        self,
    ) -> None:
        """initialize_bitmask should allocate for batch with mixed json_schema/no-constraint contexts."""
        pipeline = self._create_overlap_pipeline_with_structured_output()

        # Context with json_schema
        ctx_with_schema = TextContext(
            request_id=RequestID("with_schema"),
            max_length=1000,
            tokens=TokenBuffer(np.array([42, 67, 21])),
            json_schema='{"type": "object"}',
        )

        # Context without any constraint
        ctx_no_constraint = TextContext(
            request_id=RequestID("no_constraint"),
            max_length=1000,
            tokens=TokenBuffer(np.array([10, 20, 30])),
        )

        batch = [ctx_with_schema, ctx_no_constraint]

        result = pipeline.initialize_bitmask(batch)

        # Should allocate bitmask for the entire batch
        assert result is not None

    def test_allocates_bitmask_for_batch_with_grammar_and_json_schema(
        self,
    ) -> None:
        """initialize_bitmask should allocate for batch mixing grammar and json_schema contexts."""
        pipeline = self._create_overlap_pipeline_with_structured_output()

        # Context with grammar (tool call)
        ctx_with_grammar = TextContext(
            request_id=RequestID("with_grammar"),
            max_length=1000,
            tokens=TokenBuffer(np.array([42, 67, 21])),
            grammar="<tool call grammar>",
        )

        # Context with json_schema (structured response format)
        ctx_with_schema = TextContext(
            request_id=RequestID("with_schema"),
            max_length=1000,
            tokens=TokenBuffer(np.array([10, 20, 30])),
            json_schema='{"type": "object", "properties": {"name": {"type": "string"}}}',
        )

        # Context with neither
        ctx_freeform = TextContext(
            request_id=RequestID("freeform"),
            max_length=1000,
            tokens=TokenBuffer(np.array([1, 2, 3])),
        )

        batch = [ctx_with_grammar, ctx_with_schema, ctx_freeform]

        result = pipeline.initialize_bitmask(batch)

        # Should allocate bitmask for the entire batch
        assert result is not None

    def test_returns_none_for_all_unconstrained_batch(self) -> None:
        """initialize_bitmask should return None when all contexts are unconstrained."""
        pipeline = self._create_overlap_pipeline_with_structured_output()

        # Multiple contexts, none with grammar or json_schema
        contexts = [
            TextContext(
                request_id=RequestID(f"ctx_{i}"),
                max_length=1000,
                tokens=TokenBuffer(np.array([i, i + 1, i + 2])),
            )
            for i in range(3)
        ]

        # Verify all contexts are unconstrained
        for ctx in contexts:
            assert ctx.json_schema is None
            assert ctx.grammar is None

        result = pipeline.initialize_bitmask(contexts)

        # Should NOT allocate a bitmask
        assert result is None


class TestAssignBitmaskInputs:
    """Tests for OverlapTextGenerationPipeline._assign_bitmask_inputs.

    The PR replaced the prior per-row H2D remap (which physically reordered
    callback-written rows into the next iter's row order on device) with an
    in-graph wait + pinned-source design: the callback writes pinned rows in
    iter-N's order, and ``_assign_bitmask_inputs`` either *adopts* those
    writes when the row layout still matches, or *overwrites* them via
    ``StructuredOutputOverlapState.prime`` when composition / order
    changed. These tests cover the four branches of that decision (adopt,
    reorder-overwrite, missing-matcher overwrite, cold-start overwrite) at
    the abstraction the new code actually exposes -- mocking the overlap
    state's ``get_input_views`` / ``prime`` / flag instead of the old
    per-row inplace_copy_from chain.
    """

    _VOCAB = 64
    _MAX_BATCH = 4
    _K = 2  # num speculative tokens (matches num_draft_tokens_to_verify)
    _NUM_POS = _K + 1

    @staticmethod
    def _make_constrained_ctx(request_id: RequestID) -> TextContext:
        """Create a constrained context with ``ctx.matcher`` set, so the
        adoption guard's ``ctx.matcher is None`` clause does not fire."""
        ctx = TextContext(
            request_id=request_id,
            max_length=1000,
            tokens=TokenBuffer(np.array([1, 2, 3])),
        )
        ctx._matcher = MagicMock()
        return ctx

    @classmethod
    def _make_pipeline(
        cls,
        callback_request_ids: list[RequestID],
        has_precomputed_bitmask: bool,
    ) -> tuple[
        OverlapTextGenerationPipeline[TextContext],
        MagicMock,
        MagicMock,
        MagicMock,
        MagicMock,
    ]:
        """Build a pipeline + spec_state + overlap_state wired with
        mocks for the helpers ``_assign_bitmask_inputs`` consumes.

        Returns ``(pipeline, structured_output, spec_state,
        overlap_state, mock_device)``. The mock_device stands in for
        ``pipeline._devices[0]`` and is returned separately so each
        test can assert that the device's default stream is never
        synchronised from this code path (the design contract is
        that ``_assign_bitmask_inputs`` runs without ever blocking on
        the device).  ``overlap_state.get_input_views`` returns a
        deterministic pair of sentinel objects so the test can verify
        the (pinned, scratch) triple was wired onto ``model_inputs``.
        """
        pipeline: OverlapTextGenerationPipeline[TextContext] = (
            OverlapTextGenerationPipeline.__new__(OverlapTextGenerationPipeline)
        )
        mock_structured_output = MagicMock()
        mock_structured_output.enabled = True
        # ``compute_speculative_bitmasks`` is expected to be called only
        # on the sync-prime branch; configure it to return a
        # state.num_positions-shaped bool array so ``prime`` sees the
        # right shape.
        mock_structured_output.compute_speculative_bitmasks.return_value = (
            np.ones((1, cls._NUM_POS, cls._VOCAB), dtype=np.bool_)
        )
        pipeline._structured_output = mock_structured_output

        mock_overlap_state = MagicMock()
        mock_overlap_state.num_positions = cls._NUM_POS
        mock_overlap_state.vocab_size = cls._VOCAB
        mock_overlap_state.max_batch_size = cls._MAX_BATCH
        # Sentinels so the assertion on ``model_inputs.*`` can compare
        # by identity.
        pinned_view = MagicMock(name="pinned_view")
        scratch_view = MagicMock(name="scratch_view")
        wait_payload = MagicMock(name="wait_payload")
        mock_overlap_state.wait_payload = wait_payload
        mock_overlap_state.get_input_views.return_value = (
            pinned_view,
            scratch_view,
        )

        mock_spec_state = MagicMock()
        mock_spec_state.callback_request_ids = list(callback_request_ids)
        mock_spec_state.has_precomputed_bitmask = has_precomputed_bitmask
        mock_spec_state.overlap_state = mock_overlap_state

        pipeline._spec_decode_state = mock_spec_state
        mock_device = MagicMock()
        pipeline._devices = [mock_device]
        return (
            pipeline,
            mock_structured_output,
            mock_spec_state,
            mock_overlap_state,
            mock_device,
        )

    def test_adopts_callback_when_request_ids_match(self) -> None:
        """When the callback's row order matches and every context has a
        matcher, ``_assign_bitmask_inputs`` reuses the pinned writes:
        ``prime`` is not called and the device default stream is never
        synchronised (it never is, on any branch).
        ``has_precomputed_bitmask`` is cleared because the callback's
        writes have been consumed."""
        rid_a = RequestID("a")
        rid_b = RequestID("b")
        ctx_a = self._make_constrained_ctx(rid_a)
        ctx_b = self._make_constrained_ctx(rid_b)

        pipeline, structured_output, spec_state, overlap_state, mock_device = (
            self._make_pipeline(
                callback_request_ids=[rid_a, rid_b],
                has_precomputed_bitmask=True,
            )
        )

        model_inputs = MagicMock()
        draft_tokens_np = np.zeros((2, self._K), dtype=np.int64)
        pipeline._assign_bitmask_inputs(
            model_inputs=model_inputs,
            context_batch=[ctx_a, ctx_b],
            draft_tokens_np=draft_tokens_np,
            num_draft_tokens_to_verify=self._K,
        )

        structured_output.compute_speculative_bitmasks.assert_not_called()
        overlap_state.prime.assert_not_called()
        mock_device.default_stream.synchronize.assert_not_called()
        assert spec_state.has_precomputed_bitmask is False
        overlap_state.get_input_views.assert_called_once_with(2, self._NUM_POS)
        pinned_view, scratch_view = overlap_state.get_input_views.return_value
        assert model_inputs.pinned_bitmask is pinned_view
        assert model_inputs.device_bitmask_scratch is scratch_view
        assert model_inputs.wait_payload is overlap_state.wait_payload

    def test_sync_prime_when_request_ids_reordered(self) -> None:
        """When the callback batch was ``[a, b]`` but the current batch
        is ``[b, a]``, the row layout no longer matches; overwrite via
        ``prime``. The device default stream must not be synchronised
        (this code path is contractually drain-free)."""
        rid_a = RequestID("a")
        rid_b = RequestID("b")
        ctx_a = self._make_constrained_ctx(rid_a)
        ctx_b = self._make_constrained_ctx(rid_b)

        pipeline, structured_output, spec_state, overlap_state, mock_device = (
            self._make_pipeline(
                callback_request_ids=[rid_a, rid_b],
                has_precomputed_bitmask=True,
            )
        )

        pipeline._assign_bitmask_inputs(
            model_inputs=MagicMock(),
            context_batch=[ctx_b, ctx_a],
            draft_tokens_np=np.zeros((2, self._K), dtype=np.int64),
            num_draft_tokens_to_verify=self._K,
        )

        mock_device.default_stream.synchronize.assert_not_called()
        structured_output.compute_speculative_bitmasks.assert_called_once()
        overlap_state.prime.assert_called_once()
        assert spec_state.has_precomputed_bitmask is False

    def test_sync_prime_when_some_context_missing_matcher(self) -> None:
        """A new context that joined this iter has ``matcher is None``
        but a grammar / schema set -- the adoption guard rejects this
        case because the FSM for the joining context hasn't been
        initialised, so the callback's bitmask is stale for that row.
        Must fall through to the sync-prime path."""
        rid_a = RequestID("a")
        rid_b = RequestID("b")
        ctx_a = self._make_constrained_ctx(rid_a)
        # ctx_b has grammar set but no matcher yet (just joined).
        ctx_b = TextContext(
            request_id=rid_b,
            max_length=1000,
            tokens=TokenBuffer(np.array([1, 2, 3])),
            grammar="root ::= 'x'",
        )
        assert ctx_b.matcher is None and ctx_b.grammar is not None

        (
            pipeline,
            structured_output,
            _spec_state,
            overlap_state,
            mock_device,
        ) = self._make_pipeline(
            callback_request_ids=[rid_a, rid_b],
            has_precomputed_bitmask=True,
        )

        pipeline._assign_bitmask_inputs(
            model_inputs=MagicMock(),
            context_batch=[ctx_a, ctx_b],
            draft_tokens_np=np.zeros((2, self._K), dtype=np.int64),
            num_draft_tokens_to_verify=self._K,
        )

        mock_device.default_stream.synchronize.assert_not_called()
        structured_output.compute_speculative_bitmasks.assert_called_once()
        overlap_state.prime.assert_called_once()

    def test_sync_prime_when_no_callback_at_all(self) -> None:
        """Cold start (e.g. prefill -> first decode): no callback was
        enqueued at the prior iter, so ``has_precomputed_bitmask`` is
        ``False``. ``prime`` runs; the default stream is not
        synchronised (this code path is contractually drain-free)."""
        rid_a = RequestID("a")
        ctx_a = self._make_constrained_ctx(rid_a)

        pipeline, structured_output, spec_state, overlap_state, mock_device = (
            self._make_pipeline(
                callback_request_ids=[],
                has_precomputed_bitmask=False,
            )
        )

        pipeline._assign_bitmask_inputs(
            model_inputs=MagicMock(),
            context_batch=[ctx_a],
            draft_tokens_np=np.zeros((1, self._K), dtype=np.int64),
            num_draft_tokens_to_verify=self._K,
        )

        mock_device.default_stream.synchronize.assert_not_called()
        structured_output.compute_speculative_bitmasks.assert_called_once()
        overlap_state.prime.assert_called_once()
        # Flag is not toggled by ``_assign_bitmask_inputs`` itself when
        # there was no callback to consume.
        assert spec_state.has_precomputed_bitmask is False

    def test_sync_prime_when_composition_changed_with_all_matchers(
        self,
    ) -> None:
        """Composition change with every ctx already holding a matcher
        still falls into sync-prime: ``callback_request_ids ==
        current_request_ids`` is False, which short-circuits the
        adoption guard before the matcher check runs. Verifies the
        pure-composition-change branch independent of the
        ``ctx.matcher is None`` clause."""
        rid_a, rid_b, rid_c = (
            RequestID("a"),
            RequestID("b"),
            RequestID("c"),
        )
        ctx_a = self._make_constrained_ctx(rid_a)
        ctx_b = self._make_constrained_ctx(rid_b)
        ctx_c = self._make_constrained_ctx(rid_c)

        pipeline, structured_output, _spec_state, overlap_state, mock_device = (
            self._make_pipeline(
                # Callback wrote rows for [a, b]; iter-N+1 adds c.
                callback_request_ids=[rid_a, rid_b],
                has_precomputed_bitmask=True,
            )
        )

        pipeline._assign_bitmask_inputs(
            model_inputs=MagicMock(),
            context_batch=[ctx_a, ctx_b, ctx_c],
            draft_tokens_np=np.zeros((3, self._K), dtype=np.int64),
            num_draft_tokens_to_verify=self._K,
        )

        mock_device.default_stream.synchronize.assert_not_called()
        structured_output.compute_speculative_bitmasks.assert_called_once()
        overlap_state.prime.assert_called_once()

    def test_sync_prime_when_callback_ids_disjoint_from_current(self) -> None:
        """All callback rows belong to evicted requests; iter-N+1's
        batch is entirely fresh. Equivalent to the deleted
        ``test_all_missing_falls_back_to_full_sync`` -- there is
        nothing to adopt, so every row is sync-primed.  The default
        stream is never synchronised."""
        ctx_fresh = self._make_constrained_ctx(RequestID("fresh"))

        pipeline, structured_output, _spec_state, overlap_state, mock_device = (
            self._make_pipeline(
                callback_request_ids=[RequestID("evicted")],
                has_precomputed_bitmask=True,
            )
        )

        pipeline._assign_bitmask_inputs(
            model_inputs=MagicMock(),
            context_batch=[ctx_fresh],
            draft_tokens_np=np.zeros((1, self._K), dtype=np.int64),
            num_draft_tokens_to_verify=self._K,
        )

        mock_device.default_stream.synchronize.assert_not_called()
        structured_output.compute_speculative_bitmasks.assert_called_once()
        overlap_state.prime.assert_called_once()

    def test_sync_prime_passes_full_batch_to_compute(self) -> None:
        """The replacement for the deleted per-row remap is a *full*
        sync-prime: ``compute_speculative_bitmasks`` re-computes every
        row in ``context_batch``, never just the subset of rows the
        callback was missing. This is the contract the new code relies
        on -- the in-graph H2D copies the entire leading rectangle
        from pinned into scratch, so any unwritten row would alias
        stale data."""
        rid_a, rid_b = RequestID("a"), RequestID("b")
        ctx_a = self._make_constrained_ctx(rid_a)
        ctx_b = self._make_constrained_ctx(rid_b)

        pipeline, structured_output, _spec_state, _overlap_state, _ = (
            self._make_pipeline(
                callback_request_ids=[rid_a],  # b is "missing".
                has_precomputed_bitmask=True,
            )
        )
        # Match the runtime batch shape (2 rows) so ``prime``'s shape
        # check inside the mock would line up if we wired it through.
        structured_output.compute_speculative_bitmasks.return_value = np.ones(
            (2, self._NUM_POS, self._VOCAB), dtype=np.bool_
        )
        draft_tokens_np = np.zeros((2, self._K), dtype=np.int64)

        pipeline._assign_bitmask_inputs(
            model_inputs=MagicMock(),
            context_batch=[ctx_a, ctx_b],
            draft_tokens_np=draft_tokens_np,
            num_draft_tokens_to_verify=self._K,
        )

        call_kwargs = (
            structured_output.compute_speculative_bitmasks.call_args.kwargs
        )
        # Full batch, not just the missing tail.
        assert call_kwargs["context_batch"] == [ctx_a, ctx_b]
        # Bitmask shape is keyed on overlap_state.num_positions (the
        # captured-graph dim), not on num_draft_tokens_to_verify.
        assert call_kwargs["num_positions"] == self._NUM_POS

    def test_sync_prime_waits_for_unset_callback_event_then_clears(
        self,
    ) -> None:
        """On the sync-prime branch, an in-flight callback's done_event
        is awaited *before* ``prime`` overwrites pinned, then cleared to
        ``None`` so a subsequent callback-less iter doesn't re-wait on
        the consumed event."""
        ctx_a = self._make_constrained_ctx(RequestID("a"))
        pipeline, _structured_output, spec_state, overlap_state, _ = (
            self._make_pipeline(
                callback_request_ids=[],
                has_precomputed_bitmask=False,
            )
        )

        mock_event = MagicMock(name="done_event")
        mock_event.is_set.return_value = False

        # Worker finished within the timeout. Assert the wait happens
        # before ``prime`` so a late worker write can't stomp primed rows.
        def _wait(timeout: float) -> bool:
            assert not overlap_state.prime.called, (
                "prime ran before waiting on the callback done_event"
            )
            return True

        mock_event.wait.side_effect = _wait
        spec_state.last_callback_done_event = mock_event

        pipeline._assign_bitmask_inputs(
            model_inputs=MagicMock(),
            context_batch=[ctx_a],
            draft_tokens_np=np.zeros((1, self._K), dtype=np.int64),
            num_draft_tokens_to_verify=self._K,
        )

        mock_event.wait.assert_called_once_with(
            timeout=_SYNC_PRIME_CALLBACK_TIMEOUT_S
        )
        overlap_state.prime.assert_called_once()
        assert spec_state.last_callback_done_event is None

    def test_sync_prime_skips_wait_when_callback_event_already_set(
        self,
    ) -> None:
        """If the prior callback already signalled completion, the
        sync-prime branch skips ``wait()`` entirely but still clears the
        consumed event."""
        ctx_a = self._make_constrained_ctx(RequestID("a"))
        pipeline, _structured_output, spec_state, overlap_state, _ = (
            self._make_pipeline(
                callback_request_ids=[],
                has_precomputed_bitmask=False,
            )
        )

        mock_event = MagicMock(name="done_event")
        mock_event.is_set.return_value = True
        spec_state.last_callback_done_event = mock_event

        pipeline._assign_bitmask_inputs(
            model_inputs=MagicMock(),
            context_batch=[ctx_a],
            draft_tokens_np=np.zeros((1, self._K), dtype=np.int64),
            num_draft_tokens_to_verify=self._K,
        )

        mock_event.wait.assert_not_called()
        overlap_state.prime.assert_called_once()
        assert spec_state.last_callback_done_event is None

    def test_sync_prime_logs_and_proceeds_when_callback_event_times_out(
        self,
    ) -> None:
        """A worker that died before reaching its ``finally`` never sets
        the event. The bounded wait must time out, log an error, and
        still proceed to ``prime`` (degrade to a noisy race, not a silent
        hang) -- and the consumed event is still cleared."""
        ctx_a = self._make_constrained_ctx(RequestID("a"))
        pipeline, _structured_output, spec_state, overlap_state, _ = (
            self._make_pipeline(
                callback_request_ids=[],
                has_precomputed_bitmask=False,
            )
        )

        mock_event = MagicMock(name="done_event")
        mock_event.is_set.return_value = False
        mock_event.wait.return_value = False  # timed out
        spec_state.last_callback_done_event = mock_event

        with patch(
            "max.pipelines.lib.pipeline_variants.overlap_text_generation.logger"
        ) as mock_logger:
            pipeline._assign_bitmask_inputs(
                model_inputs=MagicMock(),
                context_batch=[ctx_a],
                draft_tokens_np=np.zeros((1, self._K), dtype=np.int64),
                num_draft_tokens_to_verify=self._K,
            )

        mock_event.wait.assert_called_once_with(
            timeout=_SYNC_PRIME_CALLBACK_TIMEOUT_S
        )
        mock_logger.error.assert_called_once()
        overlap_state.prime.assert_called_once()
        assert spec_state.last_callback_done_event is None

    def test_adopt_path_leaves_callback_event_uncleared(self) -> None:
        """The clear-to-``None`` lives inside the sync-prime branch. On
        the adopt path (``can_adopt`` true, ``prime`` skipped) the event
        is intentionally left as-is: ``prime`` is never called there, so
        there is nothing to re-wait on. Locks the subtlety that
        ``last_callback_done_event`` is *not* an
        ``in-flight-iff-non-None`` flag."""
        rid_a = RequestID("a")
        ctx_a = self._make_constrained_ctx(rid_a)
        pipeline, structured_output, spec_state, overlap_state, _ = (
            self._make_pipeline(
                callback_request_ids=[rid_a],
                has_precomputed_bitmask=True,
            )
        )

        mock_event = MagicMock(name="done_event")
        mock_event.is_set.return_value = True
        spec_state.last_callback_done_event = mock_event

        pipeline._assign_bitmask_inputs(
            model_inputs=MagicMock(),
            context_batch=[ctx_a],
            draft_tokens_np=np.zeros((1, self._K), dtype=np.int64),
            num_draft_tokens_to_verify=self._K,
        )

        structured_output.compute_speculative_bitmasks.assert_not_called()
        overlap_state.prime.assert_not_called()
        mock_event.wait.assert_not_called()
        assert spec_state.last_callback_done_event is mock_event
