# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import queue
import time
from typing import Union
from unittest.mock import Mock

import numpy as np
import pytest
import zmq
from max.interfaces import (
    GenerationStatus,
    RequestID,
    SchedulerResult,
    TextGenerationInputs,
    TextGenerationOutput,
    msgpack_numpy_decoder,
    msgpack_numpy_encoder,
)
from max.pipelines.core import TextAndVisionContext, TextContext
from max.serve.queue.zmq_queue import (
    ZmqPullSocket,
    ZmqPushSocket,
    generate_zmq_ipc_path,
)
from max.serve.scheduler.text_generation_scheduler import (
    BatchType,
    SchedulerOutput,
    TokenGenerationScheduler,
    TokenGenerationSchedulerConfig,
)


@pytest.fixture(scope="session")
def zmq_ctx():
    return zmq.Context(io_threads=2)


@pytest.fixture
def mock_pipeline():
    def next_token_behavior(
        inputs: TextGenerationInputs[TextContext],
    ) -> dict[RequestID, TextGenerationOutput]:
        responses: dict[RequestID, TextGenerationOutput] = {}

        for request_id, request in inputs.batch.items():
            # Update the InputContext.
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
    pipeline.next_token = Mock(side_effect=next_token_behavior)
    pipeline.release = Mock()
    return pipeline


@pytest.fixture
def scheduler_config():
    return TokenGenerationSchedulerConfig(
        max_batch_size_tg=4,
        max_forward_steps_tg=8,
        max_batch_size_ce=4,
        target_tokens_per_batch_ce=32,
    )


@pytest.fixture(scope="function")
def scheduler(
    mock_pipeline,  # noqa: ANN001
    scheduler_config,  # noqa: ANN001
    zmq_ctx,  # noqa: ANN001
):
    return TokenGenerationScheduler(
        scheduler_config=scheduler_config,
        pipeline=mock_pipeline,
        request_zmq_endpoint=generate_zmq_ipc_path(),
        response_zmq_endpoint=generate_zmq_ipc_path(),
        cancel_zmq_endpoint=generate_zmq_ipc_path(),
        zmq_ctx=zmq_ctx,
    )


def create_mock_request(
    seq_len: int = 30,
    start_idx: int = 0,
) -> TextContext:
    tokens = np.ones(seq_len, dtype=np.int32)
    assert len(tokens) == seq_len
    context = TextContext(
        max_length=100,
        tokens=tokens,
    )
    assert context.active_idx == seq_len
    context.bump_token_indices(start_idx=start_idx)
    return context


def test_should_schedule_ce_empty_queue(scheduler) -> None:  # noqa: ANN001
    assert not scheduler._should_schedule_ce()


def test_should_schedule_ce_full_batch(scheduler, zmq_ctx) -> None:  # noqa: ANN001
    scheduler.active_batch = {}
    for _ in range(scheduler.scheduler_config.max_batch_size_tg):
        mock_request = create_mock_request(seq_len=5, start_idx=0)
        scheduler.active_batch[mock_request.request_id] = mock_request

    # Create a push socket to send data.
    request_push_socket = ZmqPushSocket[
        tuple[str, Union[TextContext, TextAndVisionContext]]
    ](
        zmq_ctx,
        zmq_endpoint=scheduler.request_q.zmq_endpoint,
        serialize=msgpack_numpy_encoder(),
    )
    mock_request = create_mock_request()
    request_push_socket.put((mock_request.request_id, mock_request))
    time.sleep(1)
    assert not scheduler._should_schedule_ce()


def test_try_create_ce_batch(scheduler, zmq_ctx) -> None:  # noqa: ANN001
    # Create a push socket to send data.
    request_push_socket = ZmqPushSocket[
        tuple[str, Union[TextContext, TextAndVisionContext]]
    ](
        zmq_ctx,
        zmq_endpoint=scheduler.request_q.zmq_endpoint,
        serialize=msgpack_numpy_encoder(),
    )

    mock_request = create_mock_request()
    request_push_socket.put_nowait((mock_request.request_id, mock_request))
    time.sleep(1)

    batch = scheduler._create_batch_to_execute().batch_inputs
    assert len(batch) == 1
    assert mock_request.request_id in batch
    # Cache management is now handled by the paged_manager/pipeline
    assert batch[mock_request.request_id] is not None


def test_try_create_chunked_ce_batch(scheduler, zmq_ctx) -> None:  # noqa: ANN001
    # Configure scheduler for chunked prefill
    scheduler.scheduler_config.enable_chunked_prefill = True
    scheduler.scheduler_config.target_tokens_per_batch_ce = 20

    mock_data = create_mock_request(seq_len=30)
    # Create a push socket to send data.
    request_push_socket = ZmqPushSocket[
        tuple[str, Union[TextContext, TextAndVisionContext]]
    ](
        zmq_ctx,
        zmq_endpoint=scheduler.request_q.zmq_endpoint,
        serialize=msgpack_numpy_encoder(),
    )

    request_push_socket.put_nowait((mock_data.request_id, mock_data))
    time.sleep(1)

    batch = scheduler._create_batch_to_execute().batch_inputs
    assert len(batch) == 1
    assert mock_data.request_id in batch
    # Cache management is now handled by the paged_manager/pipeline
    assert batch[mock_data.request_id] is not None
    assert batch[mock_data.request_id].active_idx == 20
    assert batch[mock_data.request_id].active_length == 20


def test_scheduler_handle_terminated_responses(scheduler, zmq_ctx) -> None:  # noqa: ANN001
    mock_1 = create_mock_request()
    mock_2 = create_mock_request()
    batch_executed = {
        mock_1.request_id: mock_1,
        mock_2.request_id: mock_2,
    }
    batch_responses = {mock_1.request_id: Mock(), mock_2.request_id: Mock()}
    batch_responses[mock_1.request_id].is_done = False
    batch_responses[mock_1.request_id].tokens = [Mock()]
    batch_responses[mock_2.request_id].is_done = True  # req2 is terminated
    batch_responses[mock_2.request_id].tokens = []

    scheduler._handle_terminated_responses(batch_executed, batch_responses)

    assert mock_2.request_id not in batch_executed
    # Cache cleanup is now handled by pipeline.release()
    scheduler.pipeline.release.assert_called_once()


def test_scheduler_handle_chunked_requests(scheduler, zmq_ctx) -> None:  # noqa: ANN001
    req_1 = create_mock_request(seq_len=31, start_idx=30)
    req_2 = create_mock_request(
        seq_len=30, start_idx=20
    )  # this is a partially encoded request

    batch_executed = {
        req_1.request_id: req_1,
        req_2.request_id: req_2,
    }
    batch_responses = {req_1.request_id: Mock(), req_2.request_id: Mock()}
    batch_responses[req_1.request_id].is_done = False
    batch_responses[req_2.request_id].is_done = False

    scheduler._handle_chunked_requests(batch_executed, batch_responses)

    assert req_2.request_id not in batch_executed
    assert req_2.request_id not in batch_responses
    assert scheduler.pending_reqs


def test_handle_cancelled_requests(scheduler, zmq_ctx) -> None:  # noqa: ANN001
    mock_request = create_mock_request()
    scheduler.active_batch = {mock_request.request_id: mock_request}

    # Create a response queue endpoint to receive from.
    response_pull_socket = ZmqPullSocket[
        dict[str, SchedulerResult[TextGenerationOutput]]
    ](
        zmq_ctx,
        zmq_endpoint=scheduler.response_q.zmq_endpoint,
        deserialize=msgpack_numpy_decoder(
            dict[str, SchedulerResult[TextGenerationOutput]]
        ),
    )

    cancel_push_socket = ZmqPushSocket[list[str]](
        zmq_ctx,
        zmq_endpoint=scheduler.cancel_q.zmq_endpoint,
        serialize=msgpack_numpy_encoder(),
    )
    cancel_push_socket.put([mock_request.request_id])
    time.sleep(1)

    scheduler._handle_cancelled_requests()

    assert len(scheduler.active_batch) == 0
    # Cache cleanup is now handled by pipeline.release()
    scheduler.pipeline.release.assert_called_once_with(mock_request.request_id)


def test_schedule_ce(scheduler, zmq_ctx) -> None:  # noqa: ANN001
    mock_request = create_mock_request()
    batch_to_execute: dict[str, Union[TextContext, TextAndVisionContext]] = {
        mock_request.request_id: mock_request
    }
    sch_output = SchedulerOutput(
        batch_type=BatchType.ContextEncoding,
        batch_inputs=batch_to_execute,
        num_steps=1,
    )

    # Create a response queue endpoint to receive from.
    response_pull_socket = ZmqPullSocket[
        dict[str, SchedulerResult[TextGenerationOutput]]
    ](
        zmq_ctx,
        zmq_endpoint=scheduler.response_q.zmq_endpoint,
        deserialize=msgpack_numpy_decoder(
            dict[str, SchedulerResult[TextGenerationOutput]]
        ),
    )

    scheduler._schedule_ce(sch_output)

    assert mock_request.request_id in scheduler.active_batch
    scheduler.pipeline.next_token.assert_called_once_with(
        TextGenerationInputs(batch_to_execute, num_steps=1)
    )


def test_schedule_ce_with_chunked_prefill(scheduler, zmq_ctx) -> None:  # noqa: ANN001
    # Setup scheduler with chunked prefill enabled
    scheduler.scheduler_config.enable_chunked_prefill = True
    scheduler.scheduler_config.target_tokens_per_batch_ce = 20

    mock_request = create_mock_request(seq_len=30)
    # Create a push socket to send data.
    request_push_socket = ZmqPushSocket[
        tuple[str, Union[TextContext, TextAndVisionContext]]
    ](
        zmq_ctx,
        zmq_endpoint=scheduler.request_q.zmq_endpoint,
        serialize=msgpack_numpy_encoder(),
    )

    # Create a response queue endpoint to receive from.
    response_pull_socket = ZmqPullSocket[
        dict[str, SchedulerResult[TextGenerationOutput]]
    ](
        zmq_ctx,
        zmq_endpoint=scheduler.response_q.zmq_endpoint,
        deserialize=msgpack_numpy_decoder(
            dict[str, SchedulerResult[TextGenerationOutput]]
        ),
    )

    request_push_socket.put((mock_request.request_id, mock_request))
    time.sleep(1)
    batch_to_execute = scheduler._create_batch_to_execute().batch_inputs
    assert len(batch_to_execute) > 0
    sch_output = SchedulerOutput(
        batch_type=BatchType.ContextEncoding, batch_inputs=batch_to_execute
    )

    scheduler._schedule_ce(sch_output)

    assert mock_request.request_id not in scheduler.active_batch

    # Assert that the response socket is not empty.
    with pytest.raises(queue.Empty):
        response_pull_socket.get_nowait()

    # check req1 is put back in the request queue with the correct active_idx and active_length
    assert scheduler.pending_reqs
    req_id, data = scheduler.pending_reqs.pop()
    assert req_id == mock_request.request_id
    assert data.start_idx == 20
    assert data.active_idx == 30
    assert data.active_length == 10


def test_schedule_mixed_ce_tg(scheduler, zmq_ctx) -> None:  # noqa: ANN001
    # Setup scheduler with chunked prefill enabled
    scheduler.scheduler_config.enable_chunked_prefill = True
    scheduler.scheduler_config.enable_in_flight_batching = True
    scheduler.scheduler_config.target_tokens_per_batch_ce = 20

    mock_request_tg = create_mock_request(seq_len=10)
    assert mock_request_tg.active_idx == 10
    assert mock_request_tg.end_idx == 10

    # Create a push socket to send data.
    request_push_socket = ZmqPushSocket[
        tuple[str, Union[TextContext, TextAndVisionContext]]
    ](
        zmq_ctx,
        zmq_endpoint=scheduler.request_q.zmq_endpoint,
        serialize=msgpack_numpy_encoder(),
    )
    request_push_socket.put_nowait(
        (mock_request_tg.request_id, mock_request_tg)
    )
    time.sleep(1)

    # Create a response queue endpoint to receive from.
    response_pull_socket = ZmqPullSocket[
        dict[str, SchedulerResult[TextGenerationOutput]]
    ](
        zmq_ctx,
        zmq_endpoint=scheduler.response_q.zmq_endpoint,
        deserialize=msgpack_numpy_decoder(
            dict[str, SchedulerResult[TextGenerationOutput]]
        ),
    )

    batch_to_execute = scheduler._create_batch_to_execute().batch_inputs
    assert len(batch_to_execute) == 1
    sch_output = SchedulerOutput(
        batch_type=BatchType.ContextEncoding, batch_inputs=batch_to_execute
    )

    scheduler._schedule_ce(sch_output)
    # req1 has been put in `active_batch`

    mock_request_ce = create_mock_request(seq_len=30)
    request_push_socket.put((mock_request_ce.request_id, mock_request_ce))
    time.sleep(1)
    batch = scheduler._create_batch_to_execute().batch_inputs

    # `batch_to_execute` should contain 1 token from req1 and 19 tokens from req2
    assert len(batch) == 2
    assert mock_request_tg.request_id in batch
    assert mock_request_ce.request_id in batch
    assert batch[mock_request_tg.request_id].active_idx == 11
    assert batch[mock_request_tg.request_id].active_length == 1
    assert batch[mock_request_ce.request_id].active_idx == 19
    assert batch[mock_request_ce.request_id].active_length == 19


def test_schedule_tg(scheduler, zmq_ctx) -> None:  # noqa: ANN001
    mock_request = create_mock_request()
    batch_to_execute: dict[str, Union[TextContext, TextAndVisionContext]] = {
        mock_request.request_id: mock_request
    }
    sch_output = SchedulerOutput(
        batch_inputs=batch_to_execute,
        num_steps=scheduler.scheduler_config.max_forward_steps_tg,
    )

    # Create a response queue endpoint to receive from.
    response_pull_socket = ZmqPullSocket[
        dict[str, SchedulerResult[TextGenerationOutput]]
    ](
        zmq_ctx,
        zmq_endpoint=scheduler.response_q.zmq_endpoint,
        deserialize=msgpack_numpy_decoder(
            dict[str, SchedulerResult[TextGenerationOutput]]
        ),
    )

    scheduler._schedule_tg(sch_output)

    scheduler.pipeline.next_token.assert_called_once_with(
        TextGenerationInputs(
            batch_to_execute,
            num_steps=scheduler.scheduler_config.max_forward_steps_tg,
        )
    )
