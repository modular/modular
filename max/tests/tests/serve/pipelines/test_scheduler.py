# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import time
from queue import Queue
from typing import cast
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
from max.pipelines.context import InputContext, TextContext
from max.serve.pipelines.scheduler import (
    BatchType,
    SchedulerOutput,
    TextGenerationResponse,
    TokenGenerationScheduler,
    TokenGenerationSchedulerConfig,
)


@pytest.fixture
def mock_pipeline():
    def next_token_behavior(
        batch: dict[str, TextContext], num_steps=1
    ) -> dict[str, TextGenerationResponse]:
        responses = {}

        for request_id, request in batch.items():
            responses[request_id] = Mock()
            responses[request_id].tokens = []
            responses[request_id].is_done = False
            for _ in range(num_steps):
                request.update(new_token=1)

        return cast(dict[str, TextGenerationResponse], responses)

    pipeline = Mock()
    pipeline.next_token = Mock(side_effect=next_token_behavior)
    pipeline.release = Mock()
    return pipeline


@pytest.fixture
def mock_process_control():
    pc = Mock()
    pc.is_canceled = Mock(return_value=False)
    pc.beat = Mock()
    return pc


@pytest.fixture
def scheduler_config():
    return TokenGenerationSchedulerConfig(
        max_batch_size_tg=4,
        max_forward_steps_tg=8,
        target_tokens_per_batch_tg=32,
        max_batch_size_ce=4,
        max_forward_steps_ce=8,
        target_tokens_per_batch_ce=32,
        batch_timeout=0.1,
    )


@pytest.fixture
def queues():
    # we use a regular queue here because the multiprocessing.Queue has some lag
    return {"REQUEST": Queue(), "RESPONSE": Queue(), "CANCEL": Queue()}


@pytest.fixture
def scheduler(mock_pipeline, mock_process_control, scheduler_config, queues):
    return TokenGenerationScheduler(
        process_control=mock_process_control,
        scheduler_config=scheduler_config,
        pipeline=mock_pipeline,
        queues=queues,
    )


def create_mock_request(
    cache_seq_id=0,
    seq_len=30,
    start_idx=0,
) -> TextContext:
    tokens = np.ones(seq_len)
    context = TextContext(
        cache_seq_id=cache_seq_id,
        prompt=tokens.tolist(),
        max_length=None,
        tokens=tokens,
    )
    context.bump_token_indices(start_idx=start_idx)
    return context


def test_should_schedule_ce_empty_queue(scheduler):
    assert not scheduler._should_schedule_ce()


def test_should_schedule_ce_full_batch(scheduler):
    scheduler.active_batch = {
        i: Mock() for i in range(scheduler.scheduler_config.max_batch_size_tg)
    }
    scheduler.request_q.put(("req1", Mock()))
    assert not scheduler._should_schedule_ce()


def test_should_schedule_ce_empty_batch(scheduler):
    scheduler.request_q.put(("req1", Mock()))
    scheduler.ce_batch_start_time = (
        time.monotonic() - scheduler.scheduler_config.batch_timeout - 0.1
    )
    assert scheduler._should_schedule_ce()


def test_should_schedule_ce_timeout(scheduler):
    scheduler.request_q.put(("req1", Mock()))
    scheduler.active_batch = {"existing": Mock()}
    scheduler.ce_batch_start_time = (
        time.monotonic() - scheduler.scheduler_config.batch_timeout - 0.1
    )
    assert scheduler._should_schedule_ce()


def test_should_schedule_ce_timeout_not_reached(scheduler):
    scheduler.request_q.put(("req1", Mock()))
    scheduler.active_batch = {"existing": Mock()}
    scheduler.ce_batch_start_time = time.monotonic()
    assert not scheduler._should_schedule_ce()


def test_try_create_ce_batch(scheduler):
    mock_data = MagicMock()
    mock_data.active_length = 10
    scheduler.request_q.put(("req1", mock_data))

    batch = scheduler._try_create_ce_batch().batch_inputs
    assert len(batch) == 1
    assert "req1" in batch
    assert batch["req1"].cache_seq_id not in scheduler.available_cache_indices


def test_try_create_chunked_ce_batch(scheduler):
    # Configure scheduler for chunked prefill
    scheduler.scheduler_config.enable_chunked_prefill = True
    scheduler.scheduler_config.max_forward_steps_ce = 1
    scheduler.scheduler_config.target_tokens_per_batch_ce = 20

    mock_data = create_mock_request(cache_seq_id=0, seq_len=30)
    scheduler.request_q.put(("req1", mock_data))

    batch = scheduler._try_create_ce_batch().batch_inputs
    assert len(batch) == 1
    assert "req1" in batch
    assert batch["req1"].cache_seq_id not in scheduler.available_cache_indices
    assert batch["req1"].active_idx == 20
    assert batch["req1"].active_length == 20


def test_handle_terminated_responses(scheduler):
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


def test_handle_chunked_requests(scheduler):
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
    assert not scheduler.request_q.empty()


def test_handle_cancelled_requests(scheduler):
    mock_request = create_mock_request(cache_seq_id=0)
    scheduler.active_batch = {"req1": mock_request}
    scheduler.available_cache_indices = set()
    scheduler.cancel_q.put(["req1"])

    scheduler._handle_cancelled_requests()

    assert len(scheduler.active_batch) == 0
    assert 0 in scheduler.available_cache_indices
    scheduler.pipeline.release.assert_called_once_with(mock_request)


def test_schedule_ce(scheduler):
    mock_request = create_mock_request(cache_seq_id=0)
    batch_to_execute: dict[str, InputContext] = {"req1": mock_request}
    sch_output = SchedulerOutput(
        batch_type=BatchType.ContextEncoding,
        batch_inputs=batch_to_execute,
        num_steps=scheduler.scheduler_config.max_forward_steps_ce,
    )

    scheduler._schedule_ce(sch_output)

    assert scheduler.ce_batch_start_time is None
    assert "req1" in scheduler.active_batch
    scheduler.pipeline.next_token.assert_called_once_with(
        batch_to_execute,
        num_steps=scheduler.scheduler_config.max_forward_steps_ce,
    )


def test_schedule_ce_with_chunked_prefill(scheduler):
    # Setup scheduler with chunked prefill enabled
    scheduler.scheduler_config.enable_chunked_prefill = True
    scheduler.scheduler_config.max_forward_steps_ce = 1
    scheduler.scheduler_config.target_tokens_per_batch_ce = 20

    mock_request = create_mock_request(cache_seq_id=0, seq_len=30)
    scheduler.request_q.put(("req1", mock_request))
    batch_to_execute = scheduler._try_create_ce_batch().batch_inputs
    sch_output = SchedulerOutput(
        batch_type=BatchType.ContextEncoding, batch_inputs=batch_to_execute
    )

    scheduler._schedule_ce(sch_output)

    assert "req1" not in scheduler.active_batch
    assert scheduler.response_q.empty()

    # check req1 is put back in the request queue with the correct active_idx and active_length
    assert not scheduler.request_q.empty()
    req_id, data = scheduler.request_q.get_nowait()
    assert req_id == "req1"
    assert data.start_idx == 20
    assert data.active_idx == 30
    assert data.active_length == 10


def test_schedule_mixed_ce_tg(scheduler):
    # Setup scheduler with chunked prefill enabled
    scheduler.scheduler_config.enable_chunked_prefill = True
    scheduler.scheduler_config.enable_in_flight_batching = True
    scheduler.scheduler_config.max_forward_steps_ce = 1
    scheduler.scheduler_config.target_tokens_per_batch_ce = 20

    mock_request_tg = create_mock_request(cache_seq_id=0, seq_len=10)
    scheduler.request_q.put(("req1", mock_request_tg))
    batch_to_execute = scheduler._try_create_ce_batch().batch_inputs
    sch_output = SchedulerOutput(
        batch_type=BatchType.ContextEncoding, batch_inputs=batch_to_execute
    )

    scheduler._schedule_ce(sch_output)
    # req1 has been put in `active_batch`

    mock_request_ce = create_mock_request(cache_seq_id=1, seq_len=30)
    scheduler.request_q.put(("req2", mock_request_ce))
    batch = scheduler._try_create_ce_batch().batch_inputs

    # `batch_to_execute` should contain 1 token from req1 and 19 tokens from req2
    assert len(batch) == 2
    assert "req1" in batch
    assert "req2" in batch
    assert batch["req1"].active_idx == 11
    assert batch["req1"].active_length == 1
    assert batch["req2"].active_idx == 19
    assert batch["req2"].active_length == 19


def test_schedule_tg(scheduler):
    mock_request = create_mock_request(cache_seq_id=0)
    batch_to_execute: dict[str, InputContext] = {"req1": mock_request}
    sch_output = SchedulerOutput(
        batch_inputs=batch_to_execute,
        num_steps=scheduler.scheduler_config.max_forward_steps_tg,
    )

    scheduler._schedule_tg(sch_output)

    scheduler.pipeline.next_token.assert_called_once_with(
        batch_to_execute,
        num_steps=scheduler.scheduler_config.max_forward_steps_tg,
    )


def test_run_basic_flow(scheduler):
    # Setup mock data
    mock_request = create_mock_request(cache_seq_id=0, seq_len=10)
    scheduler.request_q.put(("req1", mock_request))

    # Mock is_canceled to return True after first iteration
    scheduler.pc.is_canceled.side_effect = [False, True]

    scheduler.run()

    scheduler.pc.beat.assert_called()
    scheduler.pipeline.next_token.assert_called()
