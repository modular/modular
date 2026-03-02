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
from max.interfaces import RequestID, TextGenerationInputs, TokenBuffer
from max.pipelines.core import TextContext
from max.pipelines.lib import OverlapTextGenerationPipeline
from max.pipelines.lib.pipeline_variants.overlap_text_generation import (
    _MAX_GRAPH_CAPTURE_BATCH_SIZE,
)


def test_throws_if_num_steps_gt_1() -> None:
    pipeline = OverlapTextGenerationPipeline.__new__(
        OverlapTextGenerationPipeline
    )
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


def test_warmup_graph_capture_caps_batch_size() -> None:
    """warmup_graph_capture should cap the batch size passed to the runner."""
    pipeline = OverlapTextGenerationPipeline.__new__(
        OverlapTextGenerationPipeline
    )
    mock_model = MagicMock()
    mock_model.model = MagicMock()
    mock_model.execute = MagicMock()
    mock_model.max_seq_len = 2048
    pipeline._pipeline_model = mock_model
    pipeline._pipeline_config = MagicMock()
    pipeline._pipeline_config.max_batch_size = 4096

    with patch(
        "max.pipelines.lib.pipeline_variants.overlap_text_generation"
        ".ServeGraphCaptureRunner"
    ) as MockRunner:
        mock_runner_instance = MockRunner.return_value
        mock_runner_instance.warmup_pre_ready = MagicMock()

        pipeline.warmup_graph_capture()

        call_kwargs = MockRunner.call_args.kwargs
        assert call_kwargs["max_batch_size"] == _MAX_GRAPH_CAPTURE_BATCH_SIZE
        assert pipeline._max_graph_capture_batch_size == (
            _MAX_GRAPH_CAPTURE_BATCH_SIZE
        )


def test_warmup_graph_capture_no_cap_when_small() -> None:
    """warmup_graph_capture should not cap when max_batch_size <= 128."""
    pipeline = OverlapTextGenerationPipeline.__new__(
        OverlapTextGenerationPipeline
    )
    mock_model = MagicMock()
    mock_model.model = MagicMock()
    mock_model.execute = MagicMock()
    mock_model.max_seq_len = 2048
    pipeline._pipeline_model = mock_model
    pipeline._pipeline_config = MagicMock()
    pipeline._pipeline_config.max_batch_size = 32

    with patch(
        "max.pipelines.lib.pipeline_variants.overlap_text_generation"
        ".ServeGraphCaptureRunner"
    ) as MockRunner:
        mock_runner_instance = MockRunner.return_value
        mock_runner_instance.warmup_pre_ready = MagicMock()

        pipeline.warmup_graph_capture()

        call_kwargs = MockRunner.call_args.kwargs
        assert call_kwargs["max_batch_size"] == 32
        assert pipeline._max_graph_capture_batch_size == 32
