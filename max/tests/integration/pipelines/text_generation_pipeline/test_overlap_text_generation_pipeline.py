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
from max.pipelines.lib import OverlapTextGenerationPipeline
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
