# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import asyncio
import time
from typing import Callable, Union, cast
from unittest.mock import Mock

import numpy as np
import pytest
import zmq
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.nn.kv_cache import KVCacheParams, KVCacheStrategy, load_kv_manager
from max.pipelines.core import (
    InputContext,
    TextContext,
    TextGenerationResponse,
    TextResponse,
)
from max.serve.kvcache_agent.dispatcher_factory import (
    DispatcherConfig,
    DispatcherFactory,
    TransportFactory,
    TransportType,
)
from max.serve.kvcache_agent.dispatcher_transport import TransportMessage
from max.serve.queue.zmq_queue import (
    ZmqPullSocket,
    ZmqPushSocket,
    generate_zmq_ipc_path,
)
from max.serve.scheduler import PrefillRequest, PrefillResponse
from max.serve.scheduler.decode_scheduler import (
    DecodeScheduler,
    DecodeSchedulerConfig,
)
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
def decode_paged_manager():
    params = KVCacheParams(
        dtype=DType.float16,
        n_kv_heads=32,
        head_dim=128,
        cache_strategy=KVCacheStrategy.PAGED,
        enable_prefix_caching=False,
        enable_kvcache_swapping_to_host=False,
        page_size=128,
    )
    return load_kv_manager(
        params=params,
        max_batch_size=16,
        max_seq_len=512,
        num_layers=32,
        devices=[CPU()],
        session=InferenceSession(),
        available_cache_memory=500 * 2**20,
        page_size=128,
    )


@pytest.fixture
def decode_request_zmq_path():
    return generate_zmq_ipc_path()


@pytest.fixture
def decode_response_zmq_path():
    return generate_zmq_ipc_path()


@pytest.fixture
def decode_cancel_zmq_path():
    return generate_zmq_ipc_path()


@pytest.fixture
def decode_client_zmq_ctx():
    return zmq.Context()


@pytest.fixture
def decode_dispatch_endpoint():
    return "tcp://127.0.0.1:5555"


@pytest.fixture
def decode_dispatcher_factory(decode_dispatch_endpoint):
    config = DispatcherConfig(
        transport=TransportType.DYNAMIC_ZMQ,
        transport_config=TransportFactory.DynamicZmqTransportConfig(
            bind_address=decode_dispatch_endpoint,
            instance_id="decode_service",
        ),
    )
    return DispatcherFactory[
        TransportMessage[Union[PrefillRequest, PrefillResponse]]
    ](
        config,
        transport_payload_type=TransportMessage[
            Union[PrefillRequest, PrefillResponse]
        ],
    )


@pytest.fixture
def decode_scheduler(
    mock_pipeline,
    mock_process_control,
    decode_paged_manager,
    decode_client_zmq_ctx,
    decode_request_zmq_path,
    decode_response_zmq_path,
    decode_cancel_zmq_path,
    decode_dispatcher_factory,
) -> Callable[[], DecodeScheduler]:
    def create_scheduler() -> DecodeScheduler:
        # Create dispatcher client
        decode_client = decode_dispatcher_factory.create_client(
            decode_client_zmq_ctx
        )

        # Initialize scheduler config
        config = DecodeSchedulerConfig(
            max_batch_size_tg=16,
            max_forward_steps_tg=8,
        )

        return DecodeScheduler(
            process_control=mock_process_control,
            pipeline=mock_pipeline,
            scheduler_config=config,
            paged_manager=decode_paged_manager,
            request_zmq_endpoint=decode_request_zmq_path,
            response_zmq_endpoint=decode_response_zmq_path,
            cancel_zmq_endpoint=decode_cancel_zmq_path,
            zmq_ctx=decode_client_zmq_ctx,
            dispatcher_client=decode_client,
        )

    return create_scheduler


@pytest.fixture
def prefill_client_zmq_ctx():
    return zmq.Context()


@pytest.fixture
def prefill_dispatch_endpoint():
    return "tcp://127.0.0.1:5556"


@pytest.fixture
def prefill_dispatcher_factory(prefill_dispatch_endpoint):
    config = DispatcherConfig(
        transport=TransportType.DYNAMIC_ZMQ,
        transport_config=TransportFactory.DynamicZmqTransportConfig(
            bind_address=prefill_dispatch_endpoint,
            instance_id="prefill_service",
        ),
    )
    return DispatcherFactory[
        TransportMessage[Union[PrefillRequest, PrefillResponse]]
    ](
        config,
        transport_payload_type=TransportMessage[
            Union[PrefillRequest, PrefillResponse]
        ],
    )


@pytest.fixture
def prefill_paged_manager():
    device = CPU()
    total_num_pages = 1
    elts_per_page = 128
    num_elts = total_num_pages * elts_per_page

    params = KVCacheParams(
        dtype=DType.float16,
        n_kv_heads=1,
        head_dim=64,
        cache_strategy=KVCacheStrategy.PAGED,
        enable_prefix_caching=False,
        enable_kvcache_swapping_to_host=False,
        page_size=128,
    )

    manager = load_kv_manager(
        params=params,
        max_batch_size=16,
        max_seq_len=512,
        num_layers=1,
        devices=[device],
        session=InferenceSession(),
        available_cache_memory=500 * 2**20,
        page_size=elts_per_page,
    )

    return manager


@pytest.fixture
def prefill_scheduler(
    mock_pipeline,
    mock_process_control,
    prefill_paged_manager,
    prefill_client_zmq_ctx,
    prefill_dispatcher_factory,
) -> Callable[[], PrefillScheduler]:
    def create_scheduler() -> PrefillScheduler:
        # Create dispatcher client
        prefill_client = prefill_dispatcher_factory.create_client(
            prefill_client_zmq_ctx
        )

        # Initialize scheduler config
        config = PrefillSchedulerConfig(
            max_batch_size_ce=16,
            target_tokens_per_batch_ce=128,
            enable_chunked_prefill=True,
        )

        return PrefillScheduler(
            process_control=mock_process_control,
            pipeline=mock_pipeline,
            scheduler_config=config,
            paged_manager=prefill_paged_manager,
            zmq_ctx=prefill_client_zmq_ctx,
            dispatcher_client=prefill_client,
        )

    return create_scheduler


@pytest.mark.skip("NOT YET Functional")
@pytest.mark.timeout(30)
@pytest.mark.asyncio
async def test_transfer_between_prefill_and_decode_scheduler(
    prefill_scheduler,
    decode_scheduler,
    decode_request_zmq_path,
    decode_response_zmq_path,
    decode_cancel_zmq_path,
    decode_dispatcher_factory,
    prefill_dispatcher_factory,
):
    # Create separate ZMQ contexts for each service
    decode_zmq_ctx = zmq.Context()
    prefill_zmq_ctx = zmq.Context()

    # Create and start decode service thread
    decode_service = decode_dispatcher_factory.create_service(decode_zmq_ctx)
    await decode_service.start()

    # Create and start prefill service thread
    prefill_service = prefill_dispatcher_factory.create_service(prefill_zmq_ctx)
    await prefill_service.start()

    # Create request push socket
    request_push_socket = ZmqPushSocket[tuple[str, InputContext]](
        decode_zmq_ctx, zmq_endpoint=decode_request_zmq_path
    )

    # Create response pull socket
    response_pull_socket = ZmqPullSocket[tuple[str, TextResponse]](
        decode_zmq_ctx, zmq_endpoint=decode_response_zmq_path
    )

    # Create cancel push socket
    cancel_push_socket = ZmqPushSocket[tuple[str, InputContext]](
        decode_zmq_ctx, zmq_endpoint=decode_cancel_zmq_path
    )

    # Create a mock text context
    mock_request_1 = TextContext(
        cache_seq_id=0,
        prompt=[1, 2, 3, 4, 5],  # Sample token sequence
        max_length=None,
        tokens=np.array([1, 2, 3, 4, 5], dtype=np.int32),
    )

    # Create a thread to manage decode scheduler tests
    def run_decode_scheduler_tests():
        # Create decode scheduler
        scheduler_instance = decode_scheduler()
        print("Created decode scheduler")

    # Create a thread to manage prefill decode scheduler tests
    def run_prefill_scheduler_tests():
        # We are sleeping here to give the decode scheduler time to bind to the transfer engine endpoint
        # This is temporary until we have dynamic remote transfer engine registration within each scheduler
        time.sleep(15)

        # Create prefill scheduler
        scheduler_instance = prefill_scheduler()
        print("Created prefill scheduler")

    # Run both schedulers concurrently
    loop = asyncio.get_event_loop()
    decode_task = loop.run_in_executor(None, run_decode_scheduler_tests)
    prefill_task = loop.run_in_executor(None, run_prefill_scheduler_tests)

    # Wait for both tasks to complete
    await asyncio.gather(decode_task, prefill_task)

    await decode_service.stop()
    await prefill_service.stop()

    # Clean up ZMQ contexts
    decode_zmq_ctx.term()
    prefill_zmq_ctx.term()
