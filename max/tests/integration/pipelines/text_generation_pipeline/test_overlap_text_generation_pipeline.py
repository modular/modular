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


from unittest.mock import MagicMock, call, patch

import numpy as np
import numpy.typing as npt
import pytest
from max.pipelines.core import TextContext
from max.pipelines.core.context import FUTURE_TOKEN
from max.pipelines.core.exceptions import InputError
from max.pipelines.lib import (
    OverlapTextGenerationPipeline,
    TextGenerationPipeline,
)
from max.pipelines.lib.pipeline_variants.overlap_text_generation import (
    _MAX_GRAPH_CAPTURE_BATCH_SIZE,
    AsyncBatch,
)
from max.pipelines.lib.pipeline_variants.utils import StructuredOutputHelper
from max.pipelines.lib.registry import get_pipeline_for_task
from max.pipelines.modeling.types import (
    PipelineTask,
    RequestID,
    TextGenerationInputs,
    TokenBuffer,
)


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
        match=r"Max num steps > 1 is not supported with the Overlap scheduler\.",
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

        # Check keyword args
        assert call_args[1]["num_steps"] == 1
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
        mock_matcher.consume_token = MagicMock(return_value=True)
        ctx._matcher = mock_matcher

        initial_length = len(ctx.tokens.all)

        # Call update_with_future_token
        ctx.update_with_future_token()

        # Token buffer should have FUTURE_TOKEN appended
        assert len(ctx.tokens.all) == initial_length + 1
        assert ctx.tokens.all[-1] == FUTURE_TOKEN

        # FSM (matcher.consume_token) should NOT have been called
        mock_matcher.consume_token.assert_not_called()

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
        # ``consume_token`` while ``grammar_enforced=True``, so flip it on
        # for this test path.
        contexts = []
        for i in range(batch_size):
            ctx = TextContext(
                request_id=RequestID(f"req_{i}"),
                max_length=1000,
                tokens=TokenBuffer(np.array([42, 67, 21])),
            )
            ctx.grammar_enforced = True
            mock_matcher = MagicMock()
            mock_matcher.consume_token = MagicMock(return_value=True)
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
                ctx._matcher.consume_token.assert_called_once_with(
                    int(real_tokens[i])
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
        mock_matcher.consume_token = MagicMock(return_value=True)
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
        mock_matcher.consume_token = MagicMock(return_value=True)
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
            mock_matcher.consume_token.assert_not_called()


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
        bitmask_bool_np = np.zeros((1, 3, 64), dtype=np.bool_)

        callback = pipeline._build_bitmask_callback(
            context_batch=[ctx],
            bonus_tokens_np=bonus_np,
            num_accepted_np=num_acc_np,
            accepted_draft_tokens_np=draft_np,
            next_draft_tokens_np=next_draft_np,
            bitmask_pinned_np=bitmask_np,
            bitmask_bool_pinned_np=bitmask_bool_np,
        )

        callback()

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

        callback = pipeline._build_bitmask_callback(
            context_batch=[],
            bonus_tokens_np=np.array([], dtype=np.int64),
            num_accepted_np=np.array([], dtype=np.int64),
            accepted_draft_tokens_np=np.zeros((0, 0), dtype=np.int64),
            next_draft_tokens_np=np.zeros((0, 0), dtype=np.int64),
            bitmask_pinned_np=np.zeros((0, 0, 0), dtype=np.int32),
            bitmask_bool_pinned_np=np.zeros((0, 0, 0), dtype=np.bool_),
        )

        # Must not raise — exceptions are caught and logged
        callback()


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
            "persistent_bitmask_bool_pinned",
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
        """Returns True and calls __unsafe_enqueue_py_host_func for a decode batch."""
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
        bitmask_bool_pinned = MagicMock()
        bitmask_bool_pinned.to_numpy.return_value = np.zeros(
            (batch_size, num_positions, vocab_size),
            dtype=np.bool_,
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
        mock_spec_state.persistent_bitmask_bool_pinned = bitmask_bool_pinned
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
        pipeline._spec_decode_state = mock_spec_state

        mock_device = MagicMock()
        pipeline._devices = [mock_device]
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
        # Production code uses getattr(device, "__unsafe_enqueue_py_host_func")
        # to avoid Python name mangling, so the mock records the call under the
        # unmangled name. The test also uses getattr to avoid the same mangling
        # that would occur with a bare identifier inside a class body.
        getattr(
            mock_device, "__unsafe_enqueue_py_host_func"
        ).assert_called_once()
        assert mock_spec_state.has_precomputed_bitmask is True
        assert mock_spec_state.callback_request_ids == [rid]


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


class _MockBuffer:
    """Buffer-like mock that tracks slice/view chains for copy verification.

    Each `.view(dtype, shape)` returns a new `_MockBuffer` preserving the
    current slice offset. Each `[start:stop]` slice records `start` so we
    can recover which row was copied when `inplace_copy_from` fires.
    """

    def __init__(
        self,
        label: str,
        num_elements: int,
        shape: tuple[int, ...],
        copy_log: list[tuple[str, int, str, int]],
        slice_start: int = 0,
    ):
        self.label = label
        self.num_elements = num_elements
        self.shape = shape
        self.slice_start = slice_start
        self.dtype = MagicMock()
        self._copy_log = copy_log

    def view(self, dtype: object, shape: tuple[int, ...]) -> "_MockBuffer":
        new_num = int(np.prod(shape))
        return _MockBuffer(
            label=self.label,
            num_elements=new_num,
            shape=tuple(shape),
            copy_log=self._copy_log,
            slice_start=self.slice_start,
        )

    def __getitem__(self, key: slice) -> "_MockBuffer":
        assert isinstance(key, slice), (
            f"Only slice indexing supported, got {key!r}"
        )
        start = key.start or 0
        stop = key.stop
        length = stop - start
        return _MockBuffer(
            label=self.label,
            num_elements=length,
            shape=(length,),
            copy_log=self._copy_log,
            slice_start=self.slice_start + start,
        )

    def inplace_copy_from(self, src: "_MockBuffer") -> None:
        self._copy_log.append(
            (src.label, src.slice_start, self.label, self.slice_start)
        )

    def to_numpy(self) -> npt.NDArray[np.bool_]:
        # Used for missing-row path: `missing_host.to_numpy()[:] = missing_np`.
        # The data isn't read again; we only assert on the H2D destination.
        return np.zeros(self.shape, dtype=np.bool_)


class TestComputeSpeculativeBitmasksRowRemapping:
    """Tests for _compute_speculative_bitmasks's per-row remapping by request_id.

    Verifies the precomputed-bitmask branch correctly routes callback rows
    (indexed by the previous batch's order) into device rows (indexed by the
    current batch's order), and that contexts missing from the callback batch
    are sync-computed and H2D'd to the correct destination row.
    """

    _VOCAB = 64
    _MAX_BATCH = 4
    _K = 2  # num speculative tokens
    _NUM_POS = _K + 1
    _ROW_SIZE = _NUM_POS * _VOCAB

    @staticmethod
    def _make_constrained_ctx(request_id: RequestID) -> TextContext:
        """Create a constrained context (ctx.matcher is not None) so
        `any_has_constraint=True` and the precomputed branch can fire."""
        ctx = TextContext(
            request_id=request_id,
            max_length=1000,
            tokens=TokenBuffer(np.array([1, 2, 3])),
        )
        ctx._matcher = MagicMock()
        return ctx

    @classmethod
    def _setup_pipeline(
        cls,
        callback_request_ids: list[RequestID],
        has_precomputed_bitmask: bool = True,
    ) -> tuple[
        OverlapTextGenerationPipeline[TextContext],
        MagicMock,
        MagicMock,
        list[tuple[str, int, str, int]],
    ]:
        """Build a pipeline + SpecDecodeState wired with mock buffers.

        Returns (pipeline, mock_structured_output, mock_spec_state, copy_log).
        copy_log is a list of (src_label, src_start, dst_label, dst_start)
        tuples captured by every `inplace_copy_from` call performed by the
        method under test.
        """
        pipeline: OverlapTextGenerationPipeline[TextContext] = (
            OverlapTextGenerationPipeline.__new__(OverlapTextGenerationPipeline)
        )
        mock_structured_output = MagicMock()
        mock_structured_output.enabled = True
        pipeline._structured_output = mock_structured_output
        pipeline.vocab_size = cls._VOCAB
        pipeline._devices = [MagicMock()]

        copy_log: list[tuple[str, int, str, int]] = []

        persistent_bitmask = _MockBuffer(
            label="device",
            num_elements=cls._MAX_BATCH * cls._NUM_POS * cls._VOCAB,
            shape=(cls._MAX_BATCH, cls._NUM_POS, cls._VOCAB),
            copy_log=copy_log,
        )
        persistent_bitmask_bool_pinned = _MockBuffer(
            label="callback_pinned",
            num_elements=cls._MAX_BATCH * cls._NUM_POS * cls._VOCAB,
            shape=(cls._MAX_BATCH, cls._NUM_POS, cls._VOCAB),
            copy_log=copy_log,
        )

        spec_state = MagicMock()
        spec_state.persistent_bitmask = persistent_bitmask
        spec_state.persistent_bitmask_bool_pinned = (
            persistent_bitmask_bool_pinned
        )
        spec_state.callback_request_ids = list(callback_request_ids)
        spec_state.has_precomputed_bitmask = has_precomputed_bitmask
        pipeline._spec_decode_state = spec_state

        return pipeline, mock_structured_output, spec_state, copy_log

    @classmethod
    def _precomputed_copy(
        cls, old_idx: int, new_idx: int
    ) -> tuple[str, int, str, int]:
        """Expected copy_log entry for callback_pinned[old_idx] → device[new_idx]."""
        return (
            "callback_pinned",
            old_idx * cls._ROW_SIZE,
            "device",
            new_idx * cls._ROW_SIZE,
        )

    @classmethod
    def _missing_copy(cls, i: int, new_idx: int) -> tuple[str, int, str, int]:
        """Expected copy_log entry for missing_pinned row i → device[new_idx]."""
        return (
            "missing_pinned",
            i * cls._ROW_SIZE,
            "device",
            new_idx * cls._ROW_SIZE,
        )

    def test_per_row_h2d_identity_mapping(self) -> None:
        """When callback batch and current batch have identical request order,
        each row copies callback_row[i] → device_row[i]."""
        rid_a, rid_b = RequestID("A"), RequestID("B")
        current = [
            self._make_constrained_ctx(rid_a),
            self._make_constrained_ctx(rid_b),
        ]
        pipeline, mock_so, mock_spec_state, copy_log = self._setup_pipeline(
            callback_request_ids=[rid_a, rid_b]
        )

        result = pipeline._compute_speculative_bitmasks(
            context_batch=current,
            draft_tokens_np=np.zeros((2, self._K), dtype=np.int64),
            num_draft_tokens_to_verify=self._K,
        )

        assert result is not None
        assert len(copy_log) == 2
        assert copy_log[0] == self._precomputed_copy(old_idx=0, new_idx=0)
        assert copy_log[1] == self._precomputed_copy(old_idx=1, new_idx=1)
        # Flag cleared so the next call doesn't reuse stale precomputed state.
        assert mock_spec_state.has_precomputed_bitmask is False
        # No sync-compute on the steady-state path.
        mock_so.compute_speculative_bitmasks.assert_not_called()

    def test_per_row_h2d_reordered(self) -> None:
        """Reordered batch: callback[A=0] -> device[A=1], callback[B=1] -> device[B=0]."""
        rid_a, rid_b = RequestID("A"), RequestID("B")
        # B then A in the current batch:
        current = [
            self._make_constrained_ctx(rid_b),
            self._make_constrained_ctx(rid_a),
        ]
        pipeline, mock_so, _, copy_log = self._setup_pipeline(
            callback_request_ids=[rid_a, rid_b]
        )

        pipeline._compute_speculative_bitmasks(
            context_batch=current,
            draft_tokens_np=np.zeros((2, self._K), dtype=np.int64),
            num_draft_tokens_to_verify=self._K,
        )

        assert len(copy_log) == 2
        # B is at callback row 1, current row 0
        assert copy_log[0] == self._precomputed_copy(old_idx=1, new_idx=0)
        # A is at callback row 0, current row 1
        assert copy_log[1] == self._precomputed_copy(old_idx=0, new_idx=1)
        mock_so.compute_speculative_bitmasks.assert_not_called()

    def test_missing_context_sync_computed(self) -> None:
        """A context not in the callback batch (e.g. a returning request)
        has its bitmask computed via compute_speculative_bitmasks and
        H2D'd to the correct destination row, while rows for contexts
        present in the callback batch are still served from callback_pinned."""
        rid_a, rid_c = RequestID("A"), RequestID("C")
        current = [
            self._make_constrained_ctx(rid_a),
            self._make_constrained_ctx(rid_c),
        ]
        pipeline, mock_so, _, copy_log = self._setup_pipeline(
            callback_request_ids=[rid_a]  # only A
        )
        # Mock the sync compute path: return a bitmask for the 1 missing ctx.
        missing_np = np.ones((1, self._NUM_POS, self._VOCAB), dtype=np.bool_)
        mock_so.compute_speculative_bitmasks = MagicMock(
            return_value=missing_np
        )

        def make_missing_pinned(*args, **kwargs) -> _MockBuffer:
            return _MockBuffer(
                label="missing_pinned",
                num_elements=int(np.prod(kwargs["shape"])),
                shape=tuple(kwargs["shape"]),
                copy_log=copy_log,
            )

        with patch(
            "max.pipelines.lib.pipeline_variants."
            "overlap_text_generation.DevicePinnedBuffer",
            side_effect=make_missing_pinned,
        ):
            pipeline._compute_speculative_bitmasks(
                context_batch=current,
                draft_tokens_np=np.zeros((2, self._K), dtype=np.int64),
                num_draft_tokens_to_verify=self._K,
            )

        # First copy: precomputed row for A (callback_pinned[0] → device[0])
        assert copy_log[0] == self._precomputed_copy(old_idx=0, new_idx=0)
        # Sync compute invoked with only the missing context.
        mock_so.compute_speculative_bitmasks.assert_called_once()
        sync_call = mock_so.compute_speculative_bitmasks.call_args
        assert [c.request_id for c in sync_call.kwargs["context_batch"]] == [
            rid_c
        ]
        # Second copy: synced row for C (missing_pinned[0] → device[1])
        assert len(copy_log) == 2
        assert copy_log[1] == self._missing_copy(i=0, new_idx=1)

    def test_all_missing_falls_back_to_full_sync(self) -> None:
        """If the current batch shares no request_ids with the callback batch
        (e.g. callback context was evicted), every row is sync-computed and
        no per-row copies are issued from callback_pinned."""
        current = [self._make_constrained_ctx(RequestID("A"))]
        pipeline, mock_so, _, copy_log = self._setup_pipeline(
            callback_request_ids=[RequestID("STALE")]
        )
        missing_np = np.ones((1, self._NUM_POS, self._VOCAB), dtype=np.bool_)
        mock_so.compute_speculative_bitmasks = MagicMock(
            return_value=missing_np
        )

        def make_missing_pinned(*args, **kwargs) -> _MockBuffer:
            return _MockBuffer(
                label="missing_pinned",
                num_elements=int(np.prod(kwargs["shape"])),
                shape=tuple(kwargs["shape"]),
                copy_log=copy_log,
            )

        with patch(
            "max.pipelines.lib.pipeline_variants."
            "overlap_text_generation.DevicePinnedBuffer",
            side_effect=make_missing_pinned,
        ):
            pipeline._compute_speculative_bitmasks(
                context_batch=current,
                draft_tokens_np=np.zeros((1, self._K), dtype=np.int64),
                num_draft_tokens_to_verify=self._K,
            )

        # No callback_pinned → device copies; only missing_pinned → device.
        assert all(c[0] != "callback_pinned" for c in copy_log), copy_log
        assert len(copy_log) == 1
        assert copy_log[0] == self._missing_copy(i=0, new_idx=0)
        mock_so.compute_speculative_bitmasks.assert_called_once()

    def test_enqueue_snapshots_callback_request_ids(self) -> None:
        """_enqueue_async_bitmask_callback writes the current batch's
        request_ids into spec_state.callback_request_ids before setting
        has_precomputed_bitmask, so the next _compute_speculative_bitmasks
        call can map by request_id."""
        pipeline = OverlapTextGenerationPipeline.__new__(
            OverlapTextGenerationPipeline
        )
        pipeline._structured_output = MagicMock()
        pipeline._structured_output.enabled = True
        pipeline._disable_overlap = False
        pipeline._devices = [MagicMock()]

        spec_state = MagicMock()
        # All four persistent pinned buffers must be non-None or the function
        # short-circuits before our snapshot line.
        for name in (
            "persistent_bitmask_pinned",
            "persistent_bitmask_bool_pinned",
            "persistent_bonus_tokens_pinned",
            "persistent_num_accepted_pinned",
            "persistent_accepted_draft_tokens_pinned",
            "persistent_next_draft_tokens_pinned",
        ):
            buf = MagicMock()
            buf.to_numpy.return_value = np.zeros(
                (self._MAX_BATCH, self._NUM_POS, self._VOCAB),
                dtype=np.int32,
            )
            setattr(spec_state, name, buf)
        spec_state.callback_request_ids = []
        spec_state.has_precomputed_bitmask = False
        pipeline._spec_decode_state = spec_state

        rid_x, rid_y = RequestID("X"), RequestID("Y")
        context_batch = [
            self._make_constrained_ctx(rid_x),
            self._make_constrained_ctx(rid_y),
        ]

        device_mock = MagicMock()
        pipeline._devices = [device_mock]

        with patch.object(
            pipeline,
            "_build_bitmask_callback",
            return_value=lambda: None,
        ):
            result = pipeline._enqueue_async_bitmask_callback(
                context_batch=context_batch,
                num_draft_tokens_to_verify=self._K,
                next_draft_k=self._K,
                verify_draft_tokens=True,
            )

        assert result is True
        assert spec_state.callback_request_ids == [rid_x, rid_y]
        assert spec_state.has_precomputed_bitmask is True


def test_structured_output_helper_raises_input_error_for_json_schema_without_flag() -> (
    None
):
    """Verify that InputError is raised when json_schema is provided without enabling structured output.

    This test ensures that when a user provides a json_schema but the
    --enable-structured-output flag is not set, an InputError is raised.
    This allows the serving layer to return a proper HTTP 400 response
    instead of crashing the server.
    """
    helper = StructuredOutputHelper(
        enabled=True,
        enable_response_format_schema=False,
        vocab_size=1000,
    )
    request_id = RequestID()
    ctx = TextContext(
        request_id=request_id,
        max_length=1000,
        tokens=TokenBuffer(np.array([42, 67, 21])),
        json_schema='{"type": "object", "properties": {"name": {"type": "string"}}}',
    )
    bitmask = np.zeros((1, 32), dtype=np.int32)

    with pytest.raises(
        InputError,
        match=r"json_schema provided but structured output is not enabled\.",
    ):
        helper.update_context(ctx, bitmask, index=0)
