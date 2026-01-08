# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

import queue
from unittest.mock import Mock

import numpy as np
import pytest
from max.interfaces import (
    GenerationStatus,
    MAXPullQueue,
    MAXPushQueue,
    RequestID,
    SchedulerResult,
    TextGenerationInputs,
    TextGenerationOutput,
    TokenBuffer,
)
from max.pipelines.core import TextAndVisionContext, TextContext
from max.serve.scheduler.config import TokenGenerationSchedulerConfig
from max.serve.scheduler.text_generation_scheduler import (
    TokenGenerationScheduler,
)
from max.support.math import ceildiv

ARBITRARY_TOKEN_ID = 999


def create_mock_pipeline() -> Mock:
    def next_token_behavior(
        inputs: TextGenerationInputs[TextContext],
    ) -> dict[RequestID, TextGenerationOutput]:
        responses: dict[RequestID, TextGenerationOutput] = {}

        for request_id, request in inputs.batch.items():
            request.update(0)

            # Return a valid response.
            responses[request_id] = TextGenerationOutput(
                request_id=request_id,
                tokens=[0, 0],  # Two tokens with ID 0
                final_status=GenerationStatus.ACTIVE,
                log_probabilities=None,
            )

        return responses

    pipeline = Mock()
    pipeline.execute = Mock(side_effect=next_token_behavior)
    pipeline.release = Mock()
    pipeline._pipeline_model = Mock(_lora_manager=None)
    return pipeline


def create_scheduler(
    dp: int = 1,
    max_batch_size: int = 4,
    max_forward_steps_tg: int = 8,
    target_tokens_per_batch_ce: int = 32,
    kvcache_ce_watermark: float = 0.95,
) -> tuple[
    TokenGenerationScheduler,
    MAXPushQueue[TextContext | TextAndVisionContext],
    MAXPullQueue[dict[RequestID, SchedulerResult[TextGenerationOutput]]],
    MAXPushQueue[list[RequestID]],
]:
    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=max_batch_size,
        max_forward_steps_tg=max_forward_steps_tg,
        target_tokens_per_batch_ce=target_tokens_per_batch_ce,
        data_parallel_degree=dp,
        kvcache_ce_watermark=kvcache_ce_watermark,
    )

    request_queue: queue.Queue[TextContext | TextAndVisionContext] = (
        queue.Queue()
    )
    response_queue: queue.Queue[
        dict[RequestID, SchedulerResult[TextGenerationOutput]]
    ] = queue.Queue()
    cancel_queue: queue.Queue[list[RequestID]] = queue.Queue()

    scheduler = TokenGenerationScheduler(
        scheduler_config=scheduler_config,
        pipeline=create_mock_pipeline(),
        request_queue=request_queue,
        response_queue=response_queue,
        cancel_queue=cancel_queue,
        offload_queue_draining=False,
    )

    return (
        scheduler,
        request_queue,
        response_queue,
        cancel_queue,
    )


def create_mock_request(
    seq_len: int = 30,
    start_idx: int = 0,
    is_tg: bool = False,
) -> TextContext:
    tokens = np.ones(seq_len, dtype=np.int64)
    assert len(tokens) == seq_len
    context = TextContext(
        request_id=RequestID(),
        max_length=100,
        tokens=TokenBuffer(tokens),
    )
    assert context.tokens.current_position == seq_len
    context.tokens.skip_processing(start_idx)
    if is_tg:
        context.update(ARBITRARY_TOKEN_ID)
    return context


def test_try_create_ce_batch() -> None:
    scheduler, request_push_socket, _, _ = create_scheduler()

    mock_request = create_mock_request()
    request_push_socket.put_nowait(mock_request)
    scheduler._retrieve_pending_requests()

    batch = scheduler.batch_constructor.construct_batch().batch
    assert len(batch) == 1
    assert mock_request.request_id in batch
    # Cache management is now handled by the paged_manager/pipeline
    assert batch[mock_request.request_id] is not None


def test_try_create_chunked_ce_batch() -> None:
    scheduler, request_push_socket, _, _ = create_scheduler()
    # Configure scheduler for chunked prefill
    scheduler.scheduler_config.enable_chunked_prefill = True
    scheduler.scheduler_config.target_tokens_per_batch_ce = 20

    mock_data = create_mock_request(seq_len=30)
    request_push_socket.put_nowait(mock_data)
    scheduler._retrieve_pending_requests()

    batch = scheduler.batch_constructor.construct_batch().batch
    assert len(batch) == 1
    assert mock_data.request_id in batch
    # Cache management is now handled by the paged_manager/pipeline
    assert batch[mock_data.request_id] is not None
    assert batch[mock_data.request_id].tokens.current_position == 20
    assert batch[mock_data.request_id].tokens.active_length == 20


def test_scheduler_handle_terminated_responses() -> None:
    scheduler, _, _, _ = create_scheduler()
    batch_constructor = scheduler.batch_constructor

    mock_1 = create_mock_request()
    mock_2 = create_mock_request()
    batch_constructor.enqueue_new_request(mock_1)
    batch_constructor.enqueue_new_request(mock_2)
    mock_1.update(ARBITRARY_TOKEN_ID)
    mock_2.update(ARBITRARY_TOKEN_ID)
    batch_executed = [{mock_1.request_id: mock_1, mock_2.request_id: mock_2}]

    resp_1: TextGenerationOutput = Mock(is_done=False, tokens=[Mock()])
    resp_2: TextGenerationOutput = Mock(is_done=True, tokens=[])
    batch_responses = {
        mock_1.request_id: resp_1,
        mock_2.request_id: resp_2,
    }

    chunked_ids = batch_constructor.advance_requests_and_collect_invalid_ids(
        batch_executed
    )
    for request_id in chunked_ids:
        del batch_responses[request_id]

    # Release terminated requests
    num_terminated_reqs = 0
    for request_id, response in batch_responses.items():
        if response.is_done:
            batch_constructor.release_request(request_id)
            num_terminated_reqs += 1

    assert num_terminated_reqs == 1
    assert isinstance(scheduler.pipeline, Mock)
    scheduler.pipeline.release.assert_called_once()


def test_scheduler_handle_chunked_requests() -> None:
    scheduler, _, _, _ = create_scheduler()
    batch_constructor = scheduler.batch_constructor

    req_1 = create_mock_request(is_tg=True)
    # this is a partially encoded request
    req_2 = create_mock_request(seq_len=30, start_idx=20)

    batch_executed = {
        req_1.request_id: req_1,
        req_2.request_id: req_2,
    }
    mock_1: TextGenerationOutput = Mock(is_done=False, tokens=[Mock()])
    mock_2: TextGenerationOutput = Mock(is_done=False, tokens=[])
    batch_responses = {req_1.request_id: mock_1, req_2.request_id: mock_2}

    chunked_ids = batch_constructor.advance_requests_and_collect_invalid_ids(
        [batch_executed]
    )
    for request_id in chunked_ids:
        del batch_responses[request_id]

    assert req_2.request_id not in batch_responses
    assert batch_constructor.all_ce_reqs


def test_handle_cancelled_requests() -> None:
    scheduler, _, _, cancel_push_socket = create_scheduler()
    batch_constructor = scheduler.batch_constructor

    mock_request = create_mock_request(is_tg=True)
    batch_constructor.enqueue_new_request(mock_request)

    cancel_push_socket.put_nowait([mock_request.request_id])

    batch_constructor.release_request(mock_request.request_id)

    assert len(batch_constructor.all_tg_reqs) == 0
    # Cache cleanup is now handled by pipeline.release()
    assert isinstance(scheduler.pipeline, Mock)
    scheduler.pipeline.release.assert_called_once_with(mock_request.request_id)


def test_schedule_ce() -> None:
    scheduler, _, _, _ = create_scheduler()

    mock_request = create_mock_request()
    batch_to_execute: dict[RequestID, TextContext] = {
        mock_request.request_id: mock_request
    }
    inputs = TextGenerationInputs(
        batches=[batch_to_execute],
        num_steps=1,
    )

    scheduler._schedule(inputs)

    assert mock_request.request_id in scheduler.batch_constructor.all_tg_reqs
    assert isinstance(scheduler.pipeline, Mock)
    scheduler.pipeline.execute.assert_called_once()


def test_schedule_ce_with_chunked_prefill() -> None:
    scheduler, request_push_socket, response_pull_socket, _ = create_scheduler()
    batch_constructor = scheduler.batch_constructor

    # Setup scheduler with chunked prefill enabled
    scheduler.scheduler_config.enable_chunked_prefill = True
    scheduler.scheduler_config.target_tokens_per_batch_ce = 20

    mock_request = create_mock_request(seq_len=30)

    request_push_socket.put_nowait(mock_request)
    scheduler._retrieve_pending_requests()
    assert len(batch_constructor.all_ce_reqs) == 1

    batch_to_execute = batch_constructor.construct_batch().batch
    assert len(batch_to_execute) > 0

    inputs = TextGenerationInputs(
        batches=[batch_to_execute],
        num_steps=1,
    )

    scheduler._schedule(inputs)

    assert mock_request.request_id not in batch_constructor.all_tg_reqs

    # Assert that the response socket is not empty.
    with pytest.raises(queue.Empty):
        x = response_pull_socket.get_nowait()
        print(f"There should be no response but mysteriously got: {x}")

    # check req1 is put back in the request queue with the correct current_position and active_length
    assert batch_constructor.all_ce_reqs
    req_id, data = list(batch_constructor.all_ce_reqs.items())[-1]
    assert req_id == mock_request.request_id
    assert data.tokens.processed_length == 20
    assert data.tokens.current_position == 30
    assert data.tokens.active_length == 10


def test_should_schedule_ce_empty_queue() -> None:
    scheduler, _, _, _ = create_scheduler()
    assert not scheduler.batch_constructor.construct_batch().batch


def test_should_schedule_ce_full_batch() -> None:
    scheduler, request_push_socket, _, _ = create_scheduler()
    for _ in range(scheduler.scheduler_config.max_batch_size):
        mock_request = create_mock_request(is_tg=True)
        scheduler.batch_constructor.enqueue_new_request(mock_request)
    mock_request_ce = create_mock_request()
    request_push_socket.put_nowait(mock_request_ce)
    batch = scheduler.batch_constructor.construct_batch().batch
    assert batch
    assert mock_request_ce.request_id not in batch


def test_schedule_tg() -> None:
    scheduler, _, _, _ = create_scheduler()

    mock_request = create_mock_request()
    batch_to_execute: dict[RequestID, TextContext] = {
        mock_request.request_id: mock_request
    }
    inputs = TextGenerationInputs(
        batches=[batch_to_execute],
        num_steps=scheduler.scheduler_config.max_forward_steps_tg,
    )

    scheduler._schedule(inputs)

    assert isinstance(scheduler.pipeline, Mock)
    scheduler.pipeline.execute.assert_called_once()


@pytest.mark.parametrize("dp", [1, 2, 15, 32])
def test_scheduler_dp(dp: int) -> None:
    """Check that DP works in basic cases."""
    scheduler, _, _, _ = create_scheduler(
        dp=dp,
        max_batch_size=512,
        max_forward_steps_tg=10,
        target_tokens_per_batch_ce=8192,
    )
    batch_constructor = scheduler.batch_constructor

    num_reqs = 17
    for _ in range(num_reqs):
        batch_constructor.enqueue_new_request(create_mock_request())
    assert len(batch_constructor.all_ce_reqs) == num_reqs
    assert len(batch_constructor.all_tg_reqs) == 0

    inputs = batch_constructor.construct_batch()
    assert len(inputs.batches) == dp

    # The 17 requests are split among the X replicas.
    bs_high = ceildiv(num_reqs, dp)
    bs_low = num_reqs // dp
    num_bs_high = num_reqs % dp
    for i in range(dp):
        if i < num_bs_high:
            assert len(inputs.batches[i]) == bs_high
        else:
            assert len(inputs.batches[i]) == bs_low

    # Num steps is 1 since we are running CE
    assert inputs.num_steps == 1

    # Generate new tokens for the requests.
    for batch in inputs.batches:
        for req in batch.values():
            req.update(ARBITRARY_TOKEN_ID)
    response: TextGenerationOutput = Mock(
        is_done=False, tokens=[ARBITRARY_TOKEN_ID]
    )
    responses = {req.request_id: response for req in inputs.batch.values()}

    chunked_ids = batch_constructor.advance_requests_and_collect_invalid_ids(
        inputs.batches
    )
    for request_id in chunked_ids:
        del responses[request_id]

    assert len(batch_constructor.all_ce_reqs) == 0
    assert len(batch_constructor.all_tg_reqs) == num_reqs


@pytest.mark.parametrize("req1_is_tg", [True, False])
@pytest.mark.parametrize("req2_is_tg", [True, False])
def test_scheduler_dp2_ce_tg(req1_is_tg: bool, req2_is_tg: bool) -> None:
    """Check that DP takes the min num_steps across all replicas."""
    scheduler, _, _, _ = create_scheduler(
        dp=2,
        max_forward_steps_tg=42,
    )
    batch_constructor = scheduler.batch_constructor

    batch_constructor.enqueue_new_request(create_mock_request(is_tg=req1_is_tg))
    batch_constructor.enqueue_new_request(create_mock_request(is_tg=req2_is_tg))

    inputs = batch_constructor.construct_batch()

    # If both requests are TG, we should have 42 steps.
    # Otherwise, we should have 1 step since we take the min num_steps across all replicas.
    if req1_is_tg and req2_is_tg:
        expected_num_steps = 42
    else:
        expected_num_steps = 1
    assert inputs.num_steps == expected_num_steps

    # There are 2 batches since DP=2
    assert len(inputs.batches) == 2

    # Each batch should have 1 request.
    assert len(inputs.batches[0]) == 1
    assert len(inputs.batches[1]) == 1


def test_scheduler_single_req_with_dp2_should_have_num_steps_of_42() -> None:
    """Check that when there is a single request, we run with num_steps=42.

    We should NOT run with num_steps=1.
    """
    scheduler, _, _, _ = create_scheduler(dp=2, max_forward_steps_tg=42)
    batch_constructor = scheduler.batch_constructor
    batch_constructor.enqueue_new_request(create_mock_request(is_tg=True))
    inputs = batch_constructor.construct_batch()
    assert inputs.num_steps == 42


def test_scheduler_empty_batch() -> None:
    """Check that we do not blow up when there are no requests."""
    scheduler, _, _, _ = create_scheduler(dp=100)
    inputs = scheduler.batch_constructor.construct_batch()
    assert len(inputs.batches) == 100
    assert len(inputs.batch) == 0
    assert inputs.num_steps == 0
