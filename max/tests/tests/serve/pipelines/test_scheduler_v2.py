# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import time
from queue import Queue
from unittest.mock import MagicMock, Mock

import pytest
from max.serve.pipelines.scheduler_v2 import (
    TokenGenerationSchedulerConfig,
    TokenGenerationSchedulerV2,
)
from max.serve.scheduler.queues import STOP_STREAM


@pytest.fixture
def mock_pipeline():
    pipeline = Mock()
    pipeline.next_token = Mock()
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
    return TokenGenerationSchedulerV2(
        process_control=mock_process_control,
        scheduler_config=scheduler_config,
        pipeline=mock_pipeline,
        queues=queues,
    )


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


def test_create_ce_batch(scheduler):
    mock_data = MagicMock()
    mock_data.seq_len = 10
    scheduler.request_q.put(("req1", mock_data))

    batch = scheduler._create_ce_batch(1)
    assert len(batch) == 1
    assert "req1" in batch
    assert batch["req1"].cache_seq_id not in scheduler.available_cache_indices


def test_handle_terminated_responses(scheduler):
    batch_executed = {
        "req1": MagicMock(cache_seq_id=0),
        "req2": MagicMock(cache_seq_id=1),
    }
    batch_responses = [{"req1": Mock()}]  # req2 is terminated

    scheduler._handle_terminated_responses(batch_executed, batch_responses)

    assert "req2" not in batch_executed
    assert batch_responses[0]["req2"] == STOP_STREAM
    assert 1 in scheduler.available_cache_indices
    scheduler.pipeline.release.assert_called_once()


def test_handle_cancelled_requests(scheduler):
    mock_request = MagicMock(cache_seq_id=0)
    scheduler.active_batch = {"req1": mock_request}
    scheduler.available_cache_indices = set()
    scheduler.cancel_q.put(["req1"])

    scheduler._handle_cancelled_requests()

    assert len(scheduler.active_batch) == 0
    assert 0 in scheduler.available_cache_indices
    scheduler.pipeline.release.assert_called_once_with(mock_request)


def test_schedule_ce(scheduler):
    mock_request = MagicMock(cache_seq_id=0)
    batch_to_execute = {"req1": mock_request}
    mock_response = [{"req1": Mock()}]
    scheduler.pipeline.next_token.return_value = mock_response

    scheduler._schedule_ce(batch_to_execute)

    assert scheduler.ce_batch_start_time is None
    assert "req1" in scheduler.active_batch
    scheduler.pipeline.next_token.assert_called_once_with(
        batch_to_execute,
        num_steps=scheduler.scheduler_config.max_forward_steps_ce,
    )


def test_schedule_tg(scheduler):
    mock_request = MagicMock(cache_seq_id=0)
    batch_to_execute = {"req1": mock_request}
    mock_response = [{"req1": Mock()}]
    scheduler.pipeline.next_token.return_value = mock_response

    scheduler._schedule_tg(batch_to_execute)

    scheduler.pipeline.next_token.assert_called_once_with(
        batch_to_execute,
        num_steps=scheduler.scheduler_config.max_forward_steps_tg,
    )


def test_run_basic_flow(scheduler):
    # Setup mock data
    mock_request = MagicMock(cache_seq_id=0, seq_len=10)
    scheduler.request_q.put(("req1", mock_request))
    scheduler.pipeline.next_token.return_value = [{"req1": Mock()}]

    # Mock is_canceled to return True after first iteration
    scheduler.pc.is_canceled.side_effect = [False, True]

    scheduler.run()

    scheduler.pc.beat.assert_called()
    scheduler.pipeline.next_token.assert_called()
