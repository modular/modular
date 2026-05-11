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


from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from max.interfaces import (
    PipelineTask,
    RequestID,
    TextGenerationInputs,
    TokenBuffer,
)
from max.pipelines.core import TextContext
from max.pipelines.core.context import FUTURE_TOKEN
from max.pipelines.lib import (
    OverlapTextGenerationPipeline,
    TextGenerationPipeline,
)
from max.pipelines.lib.pipeline_variants.overlap_text_generation import (
    _MAX_GRAPH_CAPTURE_BATCH_SIZE,
    AsyncBatch,
)
from max.pipelines.lib.registry import get_pipeline_for_task


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

        # Create contexts with mock matchers
        contexts = []
        for i in range(batch_size):
            ctx = TextContext(
                request_id=RequestID(f"req_{i}"),
                max_length=1000,
                tokens=TokenBuffer(np.array([42, 67, 21])),
            )
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

    def test_returns_none_when_structured_output_disabled(self) -> None:
        """initialize_bitmask should return None when structured output is disabled."""
        pipeline = self._create_overlap_pipeline_with_structured_output(
            enabled=False
        )

        # Even with grammar set, should return None if structured output is disabled
        ctx = TextContext(
            request_id=RequestID("grammar_but_disabled"),
            max_length=1000,
            tokens=TokenBuffer(np.array([42, 67, 21])),
            grammar="<some grammar>",
        )

        result = pipeline.initialize_bitmask([ctx])

        assert result is None

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
