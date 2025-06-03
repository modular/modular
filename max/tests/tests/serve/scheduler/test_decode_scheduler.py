# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import asyncio
import threading
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest
import zmq
from max.driver import CPU, Tensor
from max.nn.kv_cache import KVTransferEngine
from max.pipelines.core import TextContext, TextGenerationResponse
from max.serve.kvcache_agent.dispatcher_base import MessageType, ReplyContext
from max.serve.kvcache_agent.dispatcher_factory import (
    DispatcherConfig,
    DispatcherFactory,
    TransportFactory,
    TransportType,
)
from max.serve.queue.zmq_queue import ZmqPushSocket, generate_zmq_ipc_path
from max.serve.scheduler.base import PrefillRequest, PrefillResponse
from max.serve.scheduler.decode_scheduler import (
    DecodeScheduler,
    DecodeSchedulerConfig,
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
    return DecodeSchedulerConfig(
        max_batch_size_tg=16,
        max_forward_steps_tg=8,
    )


@pytest.fixture
def request_zmq_endpoint():
    return generate_zmq_ipc_path()


@pytest.fixture
def response_zmq_endpoint():
    return generate_zmq_ipc_path()


@pytest.fixture
def cancel_zmq_endpoint():
    return generate_zmq_ipc_path()


@pytest.fixture
def transfer_engine_zmq_endpoint():
    return generate_zmq_ipc_path()


@pytest.fixture
def zmq_ctx() -> zmq.Context:
    return zmq.Context(io_threads=2)


def remote_agent_peer(transfer_engine_zmq_endpoint, agent_md):
    zmq_ctx = zmq.Context(io_threads=1)
    socket = zmq_ctx.socket(zmq.REQ)
    socket.connect(transfer_engine_zmq_endpoint)
    socket.send_pyobj(agent_md)
    _ = socket.recv_pyobj()


@pytest.fixture
def prefill_address():
    return generate_zmq_ipc_path()


@pytest.fixture
def decode_address():
    return generate_zmq_ipc_path()


@pytest.fixture
def decode_dispatcher_factory(prefill_address, decode_address):
    """Create a dispatcher factory for decode service."""
    config = DispatcherConfig(
        transport=TransportType.DYNAMIC_ZMQ,
        transport_config=TransportFactory.DynamicZmqTransportConfig(
            bind_address=decode_address,
            default_destination_address=prefill_address,
        ),
    )
    return DispatcherFactory(config)


@pytest.fixture
def prefill_dispatcher_factory(prefill_address, decode_address):
    """Create a dispatcher factory for prefill service."""
    config = DispatcherConfig(
        transport=TransportType.DYNAMIC_ZMQ,
        transport_config=TransportFactory.DynamicZmqTransportConfig(
            bind_address=prefill_address,
            default_destination_address=decode_address,
        ),
    )
    return DispatcherFactory(config)


async def setup_scheduler(
    mock_pipeline,
    mock_process_control,
    scheduler_config,
    request_zmq_endpoint,
    response_zmq_endpoint,
    cancel_zmq_endpoint,
    transfer_engine_zmq_endpoint,
    paged_manager,
    zmq_ctx,
    decode_dispatcher_factory,
):
    """Setup function to create scheduler at the start."""
    decode_service = decode_dispatcher_factory.create_service(zmq_ctx)
    await decode_service.start()

    decode_client = decode_dispatcher_factory.create_client(zmq_ctx)
    decode_client.start()

    # Create peer transfer agent
    device = CPU()
    total_num_pages = 1
    elts_per_page = 128
    num_elts = total_num_pages * elts_per_page

    blocks_1 = Tensor.from_numpy(np.arange(num_elts, dtype=np.int8) + 10).to(
        device
    )

    dummy_agent = KVTransferEngine(
        name="dummy_agent",
        listen_port=8058,
        tensor=blocks_1,
        total_num_pages=total_num_pages,
    )

    # Create a Thread to send remote agent metadata
    thread = threading.Thread(
        target=remote_agent_peer,
        args=(
            transfer_engine_zmq_endpoint,
            dummy_agent.metadata,
        ),
    )
    thread.start()

    scheduler = DecodeScheduler(
        process_control=mock_process_control,
        scheduler_config=scheduler_config,
        pipeline=mock_pipeline,
        request_zmq_endpoint=request_zmq_endpoint,
        response_zmq_endpoint=response_zmq_endpoint,
        cancel_zmq_endpoint=cancel_zmq_endpoint,
        paged_manager=paged_manager,
        transfer_engine_zmq_endpoint=transfer_engine_zmq_endpoint,
        zmq_ctx=zmq_ctx,
        dispatcher_client=decode_client,
    )

    # Ensure agent is registered appropriately.
    assert "dummy_agent" in scheduler.transfer_engine.remote_connections

    # Block until remote agent peer thread resolves.
    thread.join()

    return scheduler, decode_service, decode_client


def create_mock_request(
    cache_seq_id=0,
    seq_len=30,
    start_idx=0,
) -> TextContext:
    tokens = np.ones(seq_len, dtype=np.int32)
    context = TextContext(
        cache_seq_id=cache_seq_id,
        prompt=tokens.tolist(),
        max_length=100,
        tokens=tokens,
    )
    context.bump_token_indices(start_idx=start_idx)
    return context


@pytest.mark.asyncio
async def test_decode_scheduler(
    mock_pipeline,
    mock_process_control,
    scheduler_config,
    request_zmq_endpoint,
    response_zmq_endpoint,
    cancel_zmq_endpoint,
    transfer_engine_zmq_endpoint,
    paged_manager,
    zmq_ctx,
    decode_dispatcher_factory,
    prefill_dispatcher_factory,
):
    # Create scheduler at the start
    scheduler, decode_service, decode_client = await setup_scheduler(
        mock_pipeline,
        mock_process_control,
        scheduler_config,
        request_zmq_endpoint,
        response_zmq_endpoint,
        cancel_zmq_endpoint,
        transfer_engine_zmq_endpoint,
        paged_manager,
        zmq_ctx,
        decode_dispatcher_factory,
    )

    try:
        prefill_service = prefill_dispatcher_factory.create_service(zmq_ctx)
        await prefill_service.start()

        prefill_client = prefill_dispatcher_factory.create_client(zmq_ctx)
        prefill_client.start()

        # Give services time to start up
        await asyncio.sleep(0.1)

        # Track received prefill requests on prefill side
        received_prefill_requests: list[PrefillRequest] = []

        # Register handler for prefill requests on prefill client
        @prefill_client.request_handler(MessageType.PREFILL_REQUEST)
        def handle_prefill_request(
            payload: PrefillRequest, reply_context: ReplyContext
        ) -> None:
            received_prefill_requests.append(payload)
            # Send back a mock response
            prefill_client.send_reply(
                MessageType.PREFILL_RESPONSE,
                PrefillResponse(id=payload.id, context=payload.context),
                reply_context,
            )

        request_push_socket = ZmqPushSocket[tuple[str, TextContext]](
            zmq_ctx, request_zmq_endpoint
        )

        # Create mock requests
        request_1_id = "test_request_1"
        request_1 = create_mock_request(cache_seq_id=0, seq_len=15, start_idx=0)

        request_2_id = "req2"
        request_2 = create_mock_request(cache_seq_id=0, seq_len=15, start_idx=0)

        # Send requests to decode scheduler via request socket
        request_push_socket.put((request_1_id, request_1))
        request_push_socket.put((request_2_id, request_2))

        # Trigger the scheduler to reserve memory and send to prefill
        scheduler.reserve_memory_and_send_to_prefill()

        await asyncio.sleep(0.1)

        # Ensure that the prefill node got the details.
        assert len(received_prefill_requests) == 2, (
            f"Expected 2 requests, got {len(received_prefill_requests)}"
        )

        # Verify both requests are present by ID (order-independent)
        received_ids = {req.id for req in received_prefill_requests}
        expected_ids = {request_1_id, request_2_id}
        assert received_ids == expected_ids, (
            f"Expected IDs {expected_ids}, got {received_ids}"
        )

        # Verify that the requests are present by ID
        requests_by_id = {req.id: req for req in received_prefill_requests}
        assert request_1_id in requests_by_id
        assert request_2_id in requests_by_id
    finally:
        request_push_socket.close()
        await decode_service.stop()
        await prefill_service.stop()
        decode_client.stop()
        prefill_client.stop()
