# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import tempfile
import time
import uuid
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest
import zmq
from max.pipelines.core import TextContext, TextGenerationResponse
from max.serve.scheduler.decode_scheduler import (
    DecodeScheduler,
    DecodeSchedulerConfig,
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
    paged_manager = Mock()
    paged_manager.external_claim = Mock()
    paged_manager.prefetch = Mock(return_value=True)
    return paged_manager


@pytest.fixture
def scheduler_config():
    return DecodeSchedulerConfig(
        max_batch_size_tg=16,
        max_forward_steps_tg=8,
    )


@pytest.fixture
def prefill_zmq_endpoint():
    return _generate_zmq_ipc_path()


@pytest.fixture
def decode_zmq_endpoint():
    return _generate_zmq_ipc_path()


@pytest.fixture
def request_zmq_endpoint():
    return _generate_zmq_ipc_path()


@pytest.fixture
def response_zmq_endpoint():
    return _generate_zmq_ipc_path()


@pytest.fixture
def cancel_zmq_endpoint():
    return _generate_zmq_ipc_path()


@pytest.fixture
def zmq_ctx() -> zmq.Context:
    return zmq.Context(io_threads=2)


@pytest.fixture
def scheduler(
    mock_pipeline,
    mock_process_control,
    scheduler_config,
    prefill_zmq_endpoint,
    decode_zmq_endpoint,
    request_zmq_endpoint,
    response_zmq_endpoint,
    cancel_zmq_endpoint,
    paged_manager,
    zmq_ctx,
):
    return DecodeScheduler(
        process_control=mock_process_control,
        scheduler_config=scheduler_config,
        pipeline=mock_pipeline,
        prefill_zmq_endpoint=prefill_zmq_endpoint,
        decode_zmq_endpoint=decode_zmq_endpoint,
        response_zmq_endpoint=response_zmq_endpoint,
        request_zmq_endpoint=request_zmq_endpoint,
        cancel_zmq_endpoint=cancel_zmq_endpoint,
        paged_manager=paged_manager,
        zmq_ctx=zmq_ctx,
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


def test_decode_scheduler(
    scheduler,
    prefill_zmq_endpoint,
    decode_zmq_endpoint,
    response_zmq_endpoint,
    request_zmq_endpoint,
    cancel_zmq_endpoint,
    zmq_ctx,
):
    # Create push socket to send to.
    decode_push_socket = zmq_ctx.socket(zmq.constants.PUSH)
    decode_push_socket.setsockopt(zmq.constants.RCVHWM, 0)
    decode_push_socket.setsockopt(zmq.constants.RCVBUF, -1)
    decode_push_socket.bind(decode_zmq_endpoint)

    # Create pull socket to receive from.
    request_push_socket = zmq_ctx.socket(zmq.constants.PUSH)
    request_push_socket.setsockopt(zmq.constants.RCVHWM, 0)
    request_push_socket.setsockopt(zmq.constants.RCVBUF, -1)
    request_push_socket.bind(request_zmq_endpoint)

    # Create pull socket to emulate prefill worker.
    prefill_pull_socket = zmq_ctx.socket(zmq.constants.PULL)
    prefill_pull_socket.setsockopt(zmq.constants.RCVHWM, 0)
    prefill_pull_socket.setsockopt(zmq.constants.RCVBUF, -1)
    prefill_pull_socket.connect(prefill_zmq_endpoint)

    # Create mock requests.
    request_1_id = "test_request_1"
    request_1 = create_mock_request(cache_seq_id=0, seq_len=15, start_idx=0)

    request_2_id = "req2"
    request_2 = create_mock_request(cache_seq_id=0, seq_len=15, start_idx=0)

    # Emulate forwarding request through to prefill worker.
    request_push_socket.send_pyobj((request_1_id, request_1))
    request_push_socket.send_pyobj((request_2_id, request_2))
    time.sleep(5)

    # Grab and Send through to Prefill Node.
    scheduler.reserve_memory_and_send_to_prefill()
    time.sleep(5)

    # Ensure that the prefill node got the details.
    recv_1_id, _ = prefill_pull_socket.recv_pyobj(flags=zmq.NOBLOCK)
    assert recv_1_id == request_1_id
    recv_2_id, _ = prefill_pull_socket.recv_pyobj(flags=zmq.NOBLOCK)
    assert recv_2_id == request_2_id
    time.sleep(5)
