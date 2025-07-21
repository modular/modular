# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import queue
import time
from typing import Union
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
import zmq
from max.interfaces import (
    EngineResult,
    GenerationStatus,
    TextGenerationOutput,
)
from max.pipelines.core import (
    TextAndVisionContext,
    TextContext,
    msgpack_numpy_encoder,
)
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
        batch: dict[str, TextContext],
        num_steps=1,  # noqa: ANN001
    ) -> dict[str, TextGenerationOutput]:
        responses: dict[str, TextGenerationOutput] = {}

        for request_id, request in batch.items():
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
    cache_seq_id=0,  # noqa: ANN001
    seq_len=30,  # noqa: ANN001
    start_idx=0,  # noqa: ANN001
) -> TextContext:
    tokens = np.ones(seq_len, dtype=np.int32)
    assert len(tokens) == seq_len
    context = TextContext(
        prompt=tokens.tolist(),
        max_length=100,
        tokens=tokens,
    )
    assert context.active_idx == seq_len
    context.assign_to_cache(cache_seq_id)
    context.bump_token_indices(start_idx=start_idx)
    return context


def test_should_schedule_ce_empty_queue(scheduler) -> None:  # noqa: ANN001
    assert not scheduler._should_schedule_ce()


def test_should_schedule_ce_full_batch(scheduler, zmq_ctx) -> None:  # noqa: ANN001
    scheduler.active_batch = {
        i: create_mock_request(cache_seq_id=i, seq_len=5, start_idx=0)
        for i in range(scheduler.scheduler_config.max_batch_size_tg)
    }
    # Create a push socket to send data.
    request_push_socket = ZmqPushSocket[
        tuple[str, Union[TextContext, TextAndVisionContext]]
    ](
        zmq_ctx,
        scheduler.request_q.zmq_endpoint,
        serialize=msgpack_numpy_encoder(),
    )
    request_push_socket.put(("req1", create_mock_request()))
    time.sleep(1)
    assert not scheduler._should_schedule_ce()


def test_try_create_ce_batch(scheduler, zmq_ctx) -> None:  # noqa: ANN001
    mock_data = MagicMock()
    mock_data.active_length = 10
    # Create a push socket to send data.
    request_push_socket = ZmqPushSocket[
        tuple[str, Union[TextContext, TextAndVisionContext]]
    ](
        zmq_ctx,
        scheduler.request_q.zmq_endpoint,
        serialize=msgpack_numpy_encoder(),
    )

    request_push_socket.put_nowait(("req1", create_mock_request()))
    time.sleep(1)

    batch = scheduler._create_batch_to_execute().batch_inputs
    assert len(batch) == 1
    assert "req1" in batch
    assert batch["req1"].cache_seq_id not in scheduler.available_cache_indices


def test_try_create_chunked_ce_batch(scheduler, zmq_ctx) -> None:  # noqa: ANN001
    # Configure scheduler for chunked prefill
    scheduler.scheduler_config.enable_chunked_prefill = True
    scheduler.scheduler_config.target_tokens_per_batch_ce = 20

    mock_data = create_mock_request(cache_seq_id=0, seq_len=30)
    # Create a push socket to send data.
    request_push_socket = ZmqPushSocket[
        tuple[str, Union[TextContext, TextAndVisionContext]]
    ](
        zmq_ctx,
        scheduler.request_q.zmq_endpoint,
        serialize=msgpack_numpy_encoder(),
    )

    request_push_socket.put_nowait(("req1", create_mock_request()))
    time.sleep(1)

    batch = scheduler._create_batch_to_execute().batch_inputs
    assert len(batch) == 1
    assert "req1" in batch
    assert batch["req1"].cache_seq_id not in scheduler.available_cache_indices
    assert batch["req1"].active_idx == 20
    assert batch["req1"].active_length == 20


def test_scheduler_handle_terminated_responses(scheduler, zmq_ctx) -> None:  # noqa: ANN001
    batch_executed = {
        "req1": create_mock_request(cache_seq_id=0),
        "req2": create_mock_request(cache_seq_id=1),
    }
    batch_responses = {"req1": Mock(), "req2": Mock()}
    batch_responses["req1"].is_done = False
    batch_responses["req1"].tokens = [Mock()]
    batch_responses["req2"].is_done = True  # req2 is terminated
    batch_responses["req2"].tokens = []

    scheduler._handle_terminated_responses(batch_executed, batch_responses)

    assert "req2" not in batch_executed
    assert 1 in scheduler.available_cache_indices
    scheduler.pipeline.release.assert_called_once()


def test_scheduler_handle_chunked_requests(scheduler, zmq_ctx) -> None:  # noqa: ANN001
    req_1 = create_mock_request(cache_seq_id=0, seq_len=31, start_idx=30)
    req_2 = create_mock_request(
        cache_seq_id=0, seq_len=30, start_idx=20
    )  # this is a partially encoded request

    batch_executed = {
        "req1": req_1,
        "req2": req_2,
    }
    batch_responses = {"req1": Mock(), "req2": Mock()}
    batch_responses["req1"].is_done = False
    batch_responses["req2"].is_done = False

    scheduler._handle_chunked_requests(batch_executed, batch_responses)

    assert "req2" not in batch_executed
    assert "req2" not in batch_responses
    assert scheduler.pending_reqs


def test_handle_cancelled_requests(scheduler, zmq_ctx) -> None:  # noqa: ANN001
    mock_request = create_mock_request(cache_seq_id=0)
    scheduler.active_batch = {"req1": mock_request}
    scheduler.available_cache_indices = set()

    # Create a response queue endpoint to receive from.
    response_pull_socket = ZmqPullSocket[
        dict[str, EngineResult[TextGenerationOutput]]
    ](
        zmq_ctx,
        scheduler.response_q.zmq_endpoint,
    )

    cancel_push_socket = ZmqPushSocket[list[str]](
        zmq_ctx,
        scheduler.cancel_q.zmq_endpoint,
        serialize=msgpack_numpy_encoder(),
    )
    cancel_push_socket.put(["req1"])
    time.sleep(1)

    scheduler._handle_cancelled_requests()

    assert len(scheduler.active_batch) == 0
    assert 0 in scheduler.available_cache_indices
    scheduler.pipeline.release.assert_called_once_with(mock_request)


def test_schedule_ce(scheduler, zmq_ctx) -> None:  # noqa: ANN001
    mock_request = create_mock_request(cache_seq_id=0)
    batch_to_execute: dict[str, Union[TextContext, TextAndVisionContext]] = {
        "req1": mock_request
    }
    sch_output = SchedulerOutput(
        batch_type=BatchType.ContextEncoding,
        batch_inputs=batch_to_execute,
        num_steps=1,
    )

    # Create a response queue endpoint to receive from.
    response_pull_socket = ZmqPullSocket[
        dict[str, EngineResult[TextGenerationOutput]]
    ](zmq_ctx, scheduler.response_q.zmq_endpoint)

    scheduler._schedule_ce(sch_output)

    assert "req1" in scheduler.active_batch
    scheduler.pipeline.next_token.assert_called_once_with(
        batch_to_execute,
        num_steps=1,
    )


def test_schedule_ce_with_chunked_prefill(scheduler, zmq_ctx) -> None:  # noqa: ANN001
    # Setup scheduler with chunked prefill enabled
    scheduler.scheduler_config.enable_chunked_prefill = True
    scheduler.scheduler_config.target_tokens_per_batch_ce = 20

    mock_request = create_mock_request(cache_seq_id=0, seq_len=30)
    # Create a push socket to send data.
    request_push_socket = ZmqPushSocket[
        tuple[str, Union[TextContext, TextAndVisionContext]]
    ](
        zmq_ctx,
        scheduler.request_q.zmq_endpoint,
        serialize=msgpack_numpy_encoder(),
    )

    # Create a response queue endpoint to receive from.
    response_pull_socket = ZmqPullSocket[
        dict[str, EngineResult[TextGenerationOutput]]
    ](zmq_ctx, scheduler.response_q.zmq_endpoint)

    request_push_socket.put(("req1", mock_request))
    time.sleep(1)
    batch_to_execute = scheduler._create_batch_to_execute().batch_inputs
    assert len(batch_to_execute) > 0
    sch_output = SchedulerOutput(
        batch_type=BatchType.ContextEncoding, batch_inputs=batch_to_execute
    )

    scheduler._schedule_ce(sch_output)

    assert "req1" not in scheduler.active_batch

    # Assert that the response socket is not empty.
    with pytest.raises(queue.Empty):
        response_pull_socket.get_nowait()

    # check req1 is put back in the request queue with the correct active_idx and active_length
    assert scheduler.pending_reqs
    req_id, data = scheduler.pending_reqs.pop()
    assert req_id == "req1"
    assert data.start_idx == 20
    assert data.active_idx == 30
    assert data.active_length == 10


def test_schedule_mixed_ce_tg(scheduler, zmq_ctx) -> None:  # noqa: ANN001
    # Setup scheduler with chunked prefill enabled
    scheduler.scheduler_config.enable_chunked_prefill = True
    scheduler.scheduler_config.enable_in_flight_batching = True
    scheduler.scheduler_config.target_tokens_per_batch_ce = 20

    mock_request_tg = create_mock_request(cache_seq_id=0, seq_len=10)
    assert mock_request_tg.active_idx == 10
    assert mock_request_tg.end_idx == 10

    # Create a push socket to send data.
    request_push_socket = ZmqPushSocket[
        tuple[str, Union[TextContext, TextAndVisionContext]]
    ](
        zmq_ctx,
        scheduler.request_q.zmq_endpoint,
        serialize=msgpack_numpy_encoder(),
    )
    request_push_socket.put_nowait(("req1", mock_request_tg))
    time.sleep(1)

    # Create a response queue endpoint to receive from.
    response_pull_socket = ZmqPullSocket[
        dict[str, EngineResult[TextGenerationOutput]]
    ](zmq_ctx, scheduler.response_q.zmq_endpoint)

    batch_to_execute = scheduler._create_batch_to_execute().batch_inputs
    assert len(batch_to_execute) == 1
    sch_output = SchedulerOutput(
        batch_type=BatchType.ContextEncoding, batch_inputs=batch_to_execute
    )

    scheduler._schedule_ce(sch_output)
    # req1 has been put in `active_batch`

    mock_request_ce = create_mock_request(cache_seq_id=1, seq_len=30)
    request_push_socket.put(("req2", mock_request_ce))
    time.sleep(1)
    batch = scheduler._create_batch_to_execute().batch_inputs

    # `batch_to_execute` should contain 1 token from req1 and 19 tokens from req2
    assert len(batch) == 2
    assert "req1" in batch
    assert "req2" in batch
    assert batch["req1"].active_idx == 11
    assert batch["req1"].active_length == 1
    assert batch["req2"].active_idx == 19
    assert batch["req2"].active_length == 19


def test_schedule_tg(scheduler, zmq_ctx) -> None:  # noqa: ANN001
    mock_request = create_mock_request(cache_seq_id=0)
    batch_to_execute: dict[str, Union[TextContext, TextAndVisionContext]] = {
        "req1": mock_request
    }
    sch_output = SchedulerOutput(
        batch_inputs=batch_to_execute,
        num_steps=scheduler.scheduler_config.max_forward_steps_tg,
    )

    # Create a response queue endpoint to receive from.
    response_pull_socket = ZmqPullSocket[
        dict[str, EngineResult[TextGenerationOutput]]
    ](zmq_ctx, scheduler.response_q.zmq_endpoint)

    scheduler._schedule_tg(sch_output)

    scheduler.pipeline.next_token.assert_called_once_with(
        batch_to_execute,
        num_steps=scheduler.scheduler_config.max_forward_steps_tg,
    )
