# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import asyncio
from typing import Union, cast
from unittest.mock import Mock

import numpy as np
import pytest
import zmq
from max.driver import CPU, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.nn.kv_cache import (
    KVCacheParams,
    KVCacheStrategy,
    KVTransferEngine,
    KVTransferEngineMetadata,
    PagedKVCacheManager,
)
from max.pipelines.core import TextContext, TextGenerationResponse
from max.serve.kvcache_agent.dispatcher_base import MessageType
from max.serve.kvcache_agent.dispatcher_factory import (
    DispatcherConfig,
    DispatcherFactory,
    TransportFactory,
    TransportType,
)
from max.serve.kvcache_agent.dispatcher_transport import TransportMessage
from max.serve.queue.zmq_queue import generate_zmq_ipc_path
from max.serve.scheduler.base import PrefillRequest, PrefillResponse
from max.serve.scheduler.prefill_scheduler import (
    PrefillScheduler,
    PrefillSchedulerConfig,
)

_SCHEDULER_TESTS_PORT = 8057


def get_unique_port():
    global _SCHEDULER_TESTS_PORT
    _SCHEDULER_TESTS_PORT += 1
    return _SCHEDULER_TESTS_PORT


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


def create_paged_manager(
    session: InferenceSession,
    max_seq_len: int = 2048,
    page_size: int = 128,
) -> PagedKVCacheManager:
    # Setting kv_heads, head_dim, and num_layers to 1 for simplicity
    NUM_KV_HEADS = 1
    HEAD_DIM = 4
    NUM_LAYERS = 1
    MAX_LEN = 2048
    num_blocks = max_seq_len // page_size

    dtype = DType.float32

    cache_memory = (
        2
        * NUM_LAYERS
        * NUM_KV_HEADS
        * HEAD_DIM
        * page_size
        * num_blocks
        * dtype.size_in_bytes
    )

    kv_params = KVCacheParams(
        dtype=dtype,
        n_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        cache_strategy=KVCacheStrategy.PAGED,
        page_size=page_size,
    )

    return PagedKVCacheManager(
        params=kv_params,
        max_batch_size=4,
        max_seq_len=max_seq_len,
        num_layers=NUM_LAYERS,
        devices=[CPU()],
        session=session,
        cache_memory=cache_memory,
        page_size=page_size,
        enable_runtime_checks=True,
    )


@pytest.fixture
def paged_manager(session):
    """Create a real PagedKVCacheManager for the prefill scheduler."""
    return create_paged_manager(session, max_seq_len=2048, page_size=128)


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


@pytest.fixture
def prefill_address():
    return generate_zmq_ipc_path()


@pytest.fixture
def decode_address():
    return generate_zmq_ipc_path()


@pytest.fixture(scope="session")
def session():
    return InferenceSession()


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
    return DispatcherFactory[
        Union[PrefillResponse, PrefillRequest, KVTransferEngineMetadata]
    ](
        config,
        transport_payload_type=TransportMessage[
            Union[PrefillRequest, PrefillResponse, KVTransferEngineMetadata]
        ],
    )


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
    return DispatcherFactory[
        Union[PrefillRequest, PrefillResponse, KVTransferEngineMetadata]
    ](
        config,
        transport_payload_type=TransportMessage[
            Union[PrefillRequest, PrefillResponse, KVTransferEngineMetadata]
        ],
    )


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
        listen_port=get_unique_port(),
        tensor=blocks_1,
        total_num_pages=total_num_pages,
    )

    scheduler = PrefillScheduler(
        process_control=mock_process_control,
        scheduler_config=scheduler_config,
        pipeline=mock_pipeline,
        paged_manager=paged_manager,
        zmq_ctx=zmq_ctx,
        dispatcher_client=prefill_client,
    )

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


@pytest.mark.skip("E2EOPT-318 - Flaky due to Bad File Descriptor")
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


@pytest.mark.skip("E2EOPT-318 - Flaky due to Bad File Descriptor")
@pytest.mark.asyncio
async def test_prefill_scheduler_remote_agent_registration(
    session,
    mock_pipeline,
    mock_process_control,
    scheduler_config,
    paged_manager,
    zmq_ctx,
    prefill_dispatcher_factory,
    decode_dispatcher_factory,
):
    """Test remote agent registration by sending TransferEngineRequest from decode dispatcher."""
    # Create scheduler at the start
    scheduler, prefill_service, prefill_client = await setup_scheduler(
        mock_pipeline,
        mock_process_control,
        scheduler_config,
        paged_manager,
        zmq_ctx,
        prefill_dispatcher_factory,
    )

    # Initialize variable for cleanup
    decode_transfer_engine = None

    try:
        decode_service = decode_dispatcher_factory.create_service(zmq_ctx)
        await decode_service.start()

        decode_client = decode_dispatcher_factory.create_client(zmq_ctx)
        decode_client.start()

        # Give services time to start up
        await asyncio.sleep(0.5)

        # Track received transfer engine responses on decode side
        received_transfer_responses: list = []

        # Register handler for transfer engine responses on decode client
        @decode_client.reply_handler(MessageType.TRANSFER_ENGINE_RESPONSE)
        def handle_transfer_engine_response(payload) -> None:
            received_transfer_responses.append(payload)

        decode_paged_manager = create_paged_manager(
            session=session,
            max_seq_len=2048,
            page_size=128,
        )

        # Create decode transfer engine using paged manager blocks
        decode_transfer_engine = KVTransferEngine(
            name="decode_agent_test",
            listen_port=get_unique_port(),
            tensor=decode_paged_manager.device_tensors[0],
            total_num_pages=decode_paged_manager.total_num_pages,
        )

        # Get the metadata from the actual transfer engine
        decode_transfer_metadata = decode_transfer_engine.metadata

        # Verify scheduler initially has no remote connections
        assert len(scheduler.transfer_engine.remote_connections) == 0

        # Send transfer engine request from decode client to prefill scheduler
        decode_client.send(
            MessageType.TRANSFER_ENGINE_REQUEST,
            decode_transfer_metadata,
        )

        await asyncio.sleep(0.5)

        # Verify that the scheduler received and processed the transfer engine request
        # The scheduler should have added the remote connection
        assert len(scheduler.transfer_engine.remote_connections) == 1
        assert (
            "decode_agent_test" in scheduler.transfer_engine.remote_connections
        )

        # Verify the metadata is correctly stored
        stored_metadata = scheduler.transfer_engine.remote_connections[
            "decode_agent_test"
        ]
        assert stored_metadata.name == "decode_agent_test"
        assert (
            stored_metadata.bytes_per_page
            == decode_transfer_engine.bytes_per_page
        )
        assert stored_metadata.base_addr == decode_transfer_engine.base_addr
        assert stored_metadata.memory_type == decode_transfer_engine.memory_type

        # Verify that the prefill scheduler sent back a transfer engine response
        assert len(received_transfer_responses) == 1

        # Verify the response contains the prefill scheduler's transfer engine metadata
        response_metadata = received_transfer_responses[0]
        assert response_metadata.name.startswith("prefill_agent_")
        assert (
            response_metadata.total_num_pages == paged_manager.total_num_pages
        )

    finally:
        await prefill_service.stop()
        await decode_service.stop()
        prefill_client.stop()
        decode_client.stop()
        del decode_transfer_engine
