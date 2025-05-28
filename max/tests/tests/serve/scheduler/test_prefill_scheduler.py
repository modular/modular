# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import tempfile
import threading
import time
import uuid
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest
import zmq
from max.driver import CPU, Tensor
from max.nn.kv_cache import KVTransferEngine
from max.pipelines.core import TextContext, TextGenerationResponse
from max.serve.scheduler.prefill_scheduler import (
    PrefillScheduler,
    PrefillSchedulerConfig,
)


def _generate_zmq_ipc_path() -> str:
    base_rpc_path = tempfile.gettempdir()
    return f"ipc://{base_rpc_path}/{uuid.uuid4()}"


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
def paged_manager():
    device = CPU()
    total_num_pages = 1
    elts_per_page = 128
    num_elts = total_num_pages * elts_per_page

    paged_manager = Mock()
    paged_manager.external_claim = Mock()
    paged_manager.prefetch = Mock(return_value=True)
    paged_manager.device_tensors = [
        Tensor.from_numpy(np.arange(num_elts, dtype=np.int8)).to(device)
    ]
    paged_manager.total_num_pages = total_num_pages
    return paged_manager


@pytest.fixture
def scheduler_config():
    return PrefillSchedulerConfig(
        max_batch_size_ce=16,
        target_tokens_per_batch_ce=128,
        batch_timeout=None,
        enable_chunked_prefill=True,
    )


@pytest.fixture
def prefill_zmq_endpoint():
    return _generate_zmq_ipc_path()


@pytest.fixture
def decode_zmq_endpoint():
    return _generate_zmq_ipc_path()


@pytest.fixture
def zmq_ctx() -> zmq.Context:
    return zmq.Context(io_threads=2)


def remote_agent_peer(agent_md):
    zmq_ctx = zmq.Context(io_threads=1)
    socket = zmq_ctx.socket(zmq.REP)
    socket.bind(f"ipc://{tempfile.gettempdir()}/transfer_engine")
    _ = socket.recv_pyobj()
    socket.send_pyobj(agent_md)


@pytest.fixture
def scheduler(
    mock_pipeline,
    mock_process_control,
    scheduler_config,
    prefill_zmq_endpoint,
    decode_zmq_endpoint,
    paged_manager,
    zmq_ctx,
):
    # Create peer transfer agent
    device = CPU()
    total_num_pages = 1
    elts_per_page = 128
    num_elts = total_num_pages * elts_per_page

    blocks_1 = Tensor.from_numpy(np.arange(num_elts, dtype=np.int8) + 10).to(
        device
    )

    peer_agent = KVTransferEngine(
        name="dummy_agent",
        listen_port=8057,
        tensor=blocks_1,
        total_num_pages=total_num_pages,
    )

    # Create a Thread to send remote agent metadata
    thread = threading.Thread(
        target=remote_agent_peer, args=(peer_agent.metadata,)
    )
    thread.start()

    scheduler = PrefillScheduler(
        process_control=mock_process_control,
        scheduler_config=scheduler_config,
        pipeline=mock_pipeline,
        prefill_zmq_endpoint=prefill_zmq_endpoint,
        decode_zmq_endpoint=decode_zmq_endpoint,
        paged_manager=paged_manager,
        zmq_ctx=zmq_ctx,
    )

    # Ensure agent is registered appropriately.
    assert "dummy_agent" in scheduler.transfer_engine.remote_connections

    # Block until remote agent peer thread resolves.
    thread.join()

    return scheduler


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


def test_prefill_scheduler_create_batch(
    scheduler,
    prefill_zmq_endpoint,
    decode_zmq_endpoint,
    zmq_ctx,
):
    # Create push socket to send to.
    prefill_push_socket = zmq_ctx.socket(zmq.constants.PUSH)
    prefill_push_socket.setsockopt(zmq.constants.RCVHWM, 0)
    prefill_push_socket.setsockopt(zmq.constants.RCVBUF, -1)
    prefill_push_socket.bind(prefill_zmq_endpoint)

    # Create pull socket to receive from.
    decode_pull_socket = zmq_ctx.socket(zmq.constants.PULL)
    decode_pull_socket.setsockopt(zmq.constants.RCVHWM, 0)
    decode_pull_socket.setsockopt(zmq.constants.RCVBUF, -1)
    decode_pull_socket.connect(decode_zmq_endpoint)

    # Create mock requests.
    request_1_id = "test_request_1"
    request_1 = create_mock_request(cache_seq_id=0, seq_len=15, start_idx=0)

    request_2_id = "req2"
    request_2 = create_mock_request(cache_seq_id=0, seq_len=15, start_idx=0)

    # Send to Prefill Pull Socket
    prefill_push_socket.send_pyobj((request_1_id, request_1))
    prefill_push_socket.send_pyobj((request_2_id, request_2))
    time.sleep(5)

    # Get batch from scheduler, test that the batch includes two requests.
    scheduler.update_batch()
    assert len(scheduler.active_batch) == 2

    # Send back each item to the decode queue.
    for request_id, context in scheduler.active_batch.items():
        scheduler.push_to_decode_socket(request_id, context)
    time.sleep(5)

    # Retrieve from the Decode Queue
    recv_1_id, _ = decode_pull_socket.recv_pyobj(flags=zmq.NOBLOCK)
    assert recv_1_id == request_1_id

    recv_2_id, _ = decode_pull_socket.recv_pyobj(flags=zmq.NOBLOCK)
    assert recv_2_id == request_2_id

    # Pre-empt new request
    prefill_push_socket.send_pyobj((request_1_id, request_1))
    prefill_push_socket.send_pyobj((request_2_id, request_2))
    time.sleep(5)

    # Retrieve first request
    req_1_id, req_1_data = scheduler.pull_from_prefill_socket()
    scheduler.return_to_prefill_queue(req_1_id, req_1_data)
    time.sleep(2)

    # Retrieve the same request again.
    req_1_id, req_1_data = scheduler.pull_from_prefill_socket()
    assert req_1_id == request_1_id
