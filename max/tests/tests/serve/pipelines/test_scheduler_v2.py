# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import time
from functools import partial
from queue import Queue
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.pipelines.kv_cache import (
    KVCacheParams,
    KVCacheStrategy,
    PagedKVCacheManager,
)
from max.serve.pipelines.scheduler_v2 import (
    TokenGenerationSchedulerConfig,
    TokenGenerationSchedulerV2,
)


@pytest.fixture
def mock_pipeline():
    def next_token_behavior(batch, num_steps=1):
        responses = {}

        print(f"batch: {batch}")
        for request_id, request in batch.items():
            responses[request_id] = Mock()
            responses[request_id].tokens = []
            responses[request_id].is_done = False
            for _ in range(num_steps):
                # Simulate chunked prefill behavior
                if request.active_idx < request.current_length:
                    request.start_idx = request.active_idx
                    request.active_idx = request.current_length
                    request.active_length = (
                        request.active_idx - request.start_idx
                    )
                # Simulate token generation behavior
                else:
                    request.start_idx = request.active_idx
                    request.active_idx += 1
                    request.current_length += 1
                    request.active_length = 1

                responses[request_id].tokens.append(Mock())

        return responses

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
    return TokenGenerationSchedulerV2(
        process_control=mock_process_control,
        scheduler_config=scheduler_config,
        pipeline=mock_pipeline,
        queues=queues,
    )


def create_mock_request(
    seq_len=30,
    start_idx=0,
    active_idx=None,
    active_length=None,
    current_length=None,
    cache_seq_id=None,
):
    """Create a mock request with common attributes.

    Args:
        seq_len (int): Total sequence length
        start_idx (int): Starting index
        active_idx (int, optional): Current active index. Defaults to seq_len
        active_length (int, optional): Active length. Defaults to seq_len - start_idx
        current_length (int, optional): Current length. Defaults to seq_len
        cache_seq_id (int, optional): Cache sequence ID. Defaults to None
    """
    mock_data = MagicMock()
    mock_data.tokens = {}
    mock_data.active_length = active_length
    mock_data.start_idx = start_idx
    mock_data.active_idx = active_idx if active_idx is not None else seq_len
    mock_data.active_length = (
        active_length if active_length is not None else seq_len - start_idx
    )
    mock_data.current_length = (
        current_length if current_length is not None else seq_len
    )
    mock_data.cache_seq_id = cache_seq_id
    mock_data.next_tokens = np.ones(mock_data.active_length)

    def bump_token_indices(self, start_idx=0, active_idx=0, end_idx=0):
        self.start_idx += start_idx
        self.active_idx += active_idx
        self.end_idx += end_idx
        self.active_length = self.active_idx - self.start_idx

    mock_data.bump_token_indices.side_effect = partial(
        bump_token_indices, mock_data
    )

    return mock_data


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

    batch = scheduler._try_create_ce_batch()
    assert len(batch) == 1
    assert "req1" in batch
    assert batch["req1"].cache_seq_id not in scheduler.available_cache_indices


def test_try_create_chunked_ce_batch(scheduler):
    # Configure scheduler for chunked prefill
    scheduler.scheduler_config.enable_chunked_prefill = True
    scheduler.scheduler_config.max_forward_steps_ce = 1
    scheduler.scheduler_config.target_tokens_per_batch_ce = 20

    mock_data = create_mock_request(seq_len=30)
    scheduler.request_q.put(("req1", mock_data))

    batch = scheduler._try_create_ce_batch()
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
    batch_to_execute = {"req1": mock_request}

    scheduler._schedule_ce(batch_to_execute)

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
    batch_to_execute = scheduler._try_create_ce_batch()

    scheduler._schedule_ce(batch_to_execute)

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
    batch_to_execute = scheduler._try_create_ce_batch()

    scheduler._schedule_ce(batch_to_execute)
    # req1 has been put in `active_batch`

    mock_request_ce = create_mock_request(cache_seq_id=1, seq_len=30)
    scheduler.request_q.put(("req2", mock_request_ce))
    batch = scheduler._try_create_ce_batch()

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
    batch_to_execute = {"req1": mock_request}

    scheduler._schedule_tg(batch_to_execute)

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


# Tests for scheduler with paged manager


def create_paged_manager(
    num_blocks: int,
    max_batch_size: int,
    max_seq_len: int,
    page_size: int,
) -> PagedKVCacheManager:
    # Setting kv_heads, head_dim, and num_layers to 1 so it is easy to compute
    # memory usage. Now we know each block is 1 byte.
    NUM_KV_HEADS = 1
    HEAD_DIM = 1
    NUM_LAYERS = 1

    kv_params = KVCacheParams(
        dtype=DType.float32,
        n_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        cache_strategy=KVCacheStrategy.PAGED,
        page_size=page_size,
    )

    session = InferenceSession()

    cache_memory = (
        2
        * NUM_LAYERS
        * NUM_KV_HEADS
        * HEAD_DIM
        * page_size
        * num_blocks
        * kv_params.dtype.size_in_bytes
    )
    kv_manager = PagedKVCacheManager(
        params=kv_params,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        num_layers=NUM_LAYERS,
        devices=[CPU()],
        session=session,
        cache_memory=cache_memory,
        page_size=page_size,
        enable_runtime_checks=True,
    )

    assert len(kv_manager.available_blocks) == num_blocks
    return kv_manager


def test_schedule_paged_manager_exceed_max_seq_len(
    scheduler_config, mock_process_control, mock_pipeline, queues
):
    # Create a paged manager that has one slot
    max_seq_len = 2048
    paged_manager = create_paged_manager(
        num_blocks=16,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        page_size=128,
    )

    # Create a scheduler with a paged manager
    scheduler = TokenGenerationSchedulerV2(
        process_control=mock_process_control,
        scheduler_config=scheduler_config,
        pipeline=mock_pipeline,
        queues=queues,
        paged_manager=paged_manager,
    )

    # Check that we would exceed max_seq_len during TG step
    prompt_len = 2045
    num_steps = scheduler_config.max_forward_steps_tg
    assert num_steps == 8
    assert prompt_len + num_steps > max_seq_len

    # Run CE on 2045 tokens for this req in paged manager
    cache_seq_id = 0
    paged_manager.external_claim([cache_seq_id])
    seq_ids_and_prompts = {cache_seq_id: np.ones(prompt_len)}
    paged_manager.fetch(seq_ids_and_prompts)
    seq_ids_and_new_tokens = {cache_seq_id: np.ones(1)}
    paged_manager.step(seq_ids_and_new_tokens)

    # Create a mock request and add it to the active batch
    # This request has already encoded its prompt and is ready for its first TG step
    mock_request = create_mock_request(
        cache_seq_id=cache_seq_id, seq_len=prompt_len + 1, start_idx=prompt_len
    )
    assert mock_request.active_length == 1
    scheduler.active_batch["req1"] = mock_request

    # Try to construct TG batch and make sure it is non-empty
    batch_to_execute = scheduler._create_tg_batch()
    assert len(batch_to_execute) == 1
