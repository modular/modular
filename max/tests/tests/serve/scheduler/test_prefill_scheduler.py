# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import asyncio
import tempfile
import threading
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest
import zmq
from max.driver import CPU, Tensor
from max.nn.kv_cache import KVTransferEngine
from max.pipelines.core import TextContext, TextGenerationResponse
from max.serve.kvcache_agent.dispatcher_base import MessageType
from max.serve.kvcache_agent.dispatcher_factory import (
    DispatcherConfig,
    DispatcherFactory,
    TransportFactory,
    TransportType,
)
from max.serve.queue.zmq_queue import generate_zmq_ipc_path
from max.serve.scheduler.base import PrefillRequest, PrefillResponse
from max.serve.scheduler.prefill_scheduler import (
    PrefillScheduler,
    PrefillSchedulerConfig,
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
    return PrefillSchedulerConfig(
        max_batch_size_ce=16,
        target_tokens_per_batch_ce=128,
        enable_chunked_prefill=True,
    )


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
def prefill_address():
    return generate_zmq_ipc_path()


@pytest.fixture
def decode_address():
    return generate_zmq_ipc_path()


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


async def setup_scheduler(
    mock_pipeline,
    mock_process_control,
    scheduler_config,
    paged_manager,
    zmq_ctx,
    prefill_dispatcher_factory,
):
    prefill_service = prefill_dispatcher_factory.create_service(zmq_ctx)
    await prefill_service.start()

    prefill_client = prefill_dispatcher_factory.create_client(zmq_ctx)
    prefill_client.start()

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
        paged_manager=paged_manager,
        zmq_ctx=zmq_ctx,
        dispatcher_client=prefill_client,
    )

    # Ensure agent is registered appropriately.
    assert "dummy_agent" in scheduler.transfer_engine.remote_connections

    # Block until remote agent peer thread resolves.
    thread.join()

    return scheduler, prefill_service, prefill_client


def create_mock_request(
    cache_seq_id=0,
    seq_len=30,
    start_idx=0,
) -> TextContext:
    tokens = np.ones(seq_len, dtype=np.int32)
    context = TextContext(
        prompt=tokens.tolist(),
        max_length=100,
        tokens=tokens,
    )
    context.assign_to_cache(cache_seq_id)
    context.bump_token_indices(start_idx=start_idx)
    return context


@pytest.mark.asyncio
async def test_prefill_scheduler_create_batch(
    mock_pipeline,
    mock_process_control,
    scheduler_config,
    paged_manager,
    zmq_ctx,
    prefill_dispatcher_factory,
    decode_dispatcher_factory,
):
    # Create scheduler at the start
    scheduler, prefill_service, prefill_client = await setup_scheduler(
        mock_pipeline,
        mock_process_control,
        scheduler_config,
        paged_manager,
        zmq_ctx,
        prefill_dispatcher_factory,
    )

    try:
        decode_service = decode_dispatcher_factory.create_service(zmq_ctx)
        await decode_service.start()

        decode_client = decode_dispatcher_factory.create_client(zmq_ctx)
        decode_client.start()

        # Give services time to start up
        await asyncio.sleep(0.5)

        # Track received responses on decode side
        received_responses: list[PrefillResponse] = []

        # Register handler for prefill responses on decode client
        @decode_client.reply_handler(MessageType.PREFILL_RESPONSE)
        def handle_prefill_response(payload: PrefillResponse) -> None:
            received_responses.append(payload)

        # Create mock requests
        request_1_id = "test_request_1"
        request_1 = create_mock_request(cache_seq_id=0, seq_len=15, start_idx=0)

        request_2_id = "req2"
        request_2 = create_mock_request(cache_seq_id=0, seq_len=15, start_idx=0)

        # Send prefill requests through decode_client
        decode_client.send(
            MessageType.PREFILL_REQUEST,
            PrefillRequest(
                id=request_1_id,
                context=request_1,
                transfer_engine_name="dummy_agent",
                block_ids=[1],
            ),
        )
        decode_client.send(
            MessageType.PREFILL_REQUEST,
            PrefillRequest(
                id=request_2_id,
                context=request_2,
                transfer_engine_name="dummy_agent",
                block_ids=[2],
            ),
        )

        await asyncio.sleep(0.5)

        # Get batch from scheduler, test that the batch includes two requests
        scheduler.update_batch()
        assert len(scheduler.active_batch) == 2

        # Process the batch
        scheduler.schedule()

        await asyncio.sleep(0.5)

        # Ensure that the prefill node got the details.
        assert len(received_responses) == 2, (
            f"Expected 2 requests, got {len(received_responses)}"
        )

        # Verify both requests are present by ID (order-independent)
        received_ids = {req.id for req in received_responses}
        expected_ids = {request_1_id, request_2_id}
        assert received_ids == expected_ids, (
            f"Expected IDs {expected_ids}, got {received_ids}"
        )

        # Verify that the requests are present by ID
        requests_by_id = {req.id: req for req in received_responses}
        assert request_1_id in requests_by_id
        assert request_2_id in requests_by_id

        # Pre-empt new request
        decode_client.send(
            MessageType.PREFILL_REQUEST,
            PrefillRequest(
                id=request_1_id,
                context=request_1,
                transfer_engine_name="dummy_agent",
                block_ids=[1],
            ),
        )
        decode_client.send(
            MessageType.PREFILL_REQUEST,
            PrefillRequest(
                id=request_2_id,
                context=request_2,
                transfer_engine_name="dummy_agent",
                block_ids=[2],
            ),
        )

        await asyncio.sleep(0.5)

        # Retrieve first request
        prefill_data = scheduler.get_prefill_request()
        scheduler.return_to_prefill_queue(prefill_data)

        # Retrieve the same request again.
        prefill_data = scheduler.get_prefill_request()
        assert prefill_data.id == request_1_id
    finally:
        await prefill_service.stop()
        await decode_service.stop()
        prefill_client.stop()
        decode_client.stop()
