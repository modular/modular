# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import asyncio
import time
from datetime import datetime
from multiprocessing import Process, Queue
from typing import Callable, Union
from unittest.mock import Mock

import numpy as np
import pytest
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.interfaces import (
    InputContext,
    RequestID,
    SchedulerResult,
    TextGenerationOutput,
    msgpack_numpy_decoder,
    msgpack_numpy_encoder,
)
from max.nn.kv_cache import (
    KVCacheParams,
    KVCacheStrategy,
    KVTransferEngineMetadata,
    load_kv_manager,
)
from max.pipelines.core import TextContext
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
from max.serve.scheduler.decode_scheduler import DecodeScheduler
from max.serve.scheduler.prefill_scheduler import PrefillScheduler
from max.serve.scheduler.text_batch_constructor import (
    TokenGenerationSchedulerConfig,
)


@pytest.fixture
def mock_pipeline():
    def next_token_behavior(
        batch: dict[str, TextContext],
        num_steps: int = 1,
    ) -> dict[str, TextGenerationOutput]:
        from max.interfaces import GenerationStatus

        responses = {}

        for request_id, request in batch.items():
            tokens = []
            for _ in range(num_steps):
                request.update(new_token=1)
                tokens.append(1)

            responses[request_id] = TextGenerationOutput(
                request_id=request_id,
                tokens=tokens,
                final_status=GenerationStatus.ACTIVE,
                log_probabilities=None,
            )

        return responses

    pipeline = Mock()
    pipeline.next_token = Mock(side_effect=next_token_behavior)
    pipeline.release = Mock()
    return pipeline


@pytest.fixture
def decode_paged_manager():
    params = KVCacheParams(
        dtype=DType.float16,
        n_kv_heads=8,
        head_dim=128,
        cache_strategy=KVCacheStrategy.PAGED,
        enable_prefix_caching=True,
        enable_kvcache_swapping_to_host=False,
        page_size=128,
    )
    return load_kv_manager(
        params=params,
        max_batch_size=4,
        max_seq_len=512,
        num_layers=1,
        devices=[CPU()],
        session=InferenceSession(devices=[CPU()]),
        available_cache_memory=500 * 2**24,
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
def decode_dispatch_endpoint() -> str:
    return "tcp://127.0.0.1:5555"


@pytest.fixture
def decode_dispatcher_factory(
    decode_dispatch_endpoint,  # noqa: ANN001
    prefill_dispatch_endpoint,  # noqa: ANN001
):
    config = DispatcherConfig(
        transport=TransportType.DYNAMIC_ZMQ,
        transport_config=TransportFactory.DynamicZmqTransportConfig(
            bind_address=decode_dispatch_endpoint,
            instance_id="decode_service",
            default_destination_address=prefill_dispatch_endpoint,
        ),
    )
    return DispatcherFactory[
        TransportMessage[
            Union[PrefillRequest, PrefillResponse, KVTransferEngineMetadata]
        ]
    ](
        config,
        transport_payload_type=TransportMessage[
            Union[PrefillRequest, PrefillResponse, KVTransferEngineMetadata]
        ],
    )


@pytest.fixture
def decode_scheduler(
    mock_pipeline,  # noqa: ANN001
    decode_paged_manager,  # noqa: ANN001
    decode_request_zmq_path,  # noqa: ANN001
    decode_response_zmq_path,  # noqa: ANN001
    decode_cancel_zmq_path,  # noqa: ANN001
    decode_dispatcher_factory,  # noqa: ANN001
) -> Callable[[], DecodeScheduler]:
    def create_scheduler() -> DecodeScheduler:
        # Create dispatcher client
        decode_client = decode_dispatcher_factory.create_client()
        decode_client.start()

        # Initialize scheduler config
        config = TokenGenerationSchedulerConfig(
            max_batch_size_tg=4,
            max_forward_steps_tg=8,
            max_batch_size_ce=4,
        )

        return DecodeScheduler(
            pipeline=mock_pipeline,
            scheduler_config=config,
            paged_manager=decode_paged_manager,
            request_zmq_endpoint=decode_request_zmq_path,
            response_zmq_endpoint=decode_response_zmq_path,
            cancel_zmq_endpoint=decode_cancel_zmq_path,
            dispatcher_client=decode_client,
        )

    return create_scheduler


@pytest.fixture
def prefill_dispatch_endpoint() -> str:
    return "tcp://127.0.0.1:5556"


@pytest.fixture
def prefill_dispatcher_factory(
    prefill_dispatch_endpoint,  # noqa: ANN001
    decode_dispatch_endpoint,  # noqa: ANN001
):
    config = DispatcherConfig(
        transport=TransportType.DYNAMIC_ZMQ,
        transport_config=TransportFactory.DynamicZmqTransportConfig(
            bind_address=prefill_dispatch_endpoint,
            instance_id="prefill_service",
            default_destination_address=decode_dispatch_endpoint,
        ),
    )
    return DispatcherFactory[
        TransportMessage[
            Union[PrefillRequest, PrefillResponse, KVTransferEngineMetadata]
        ]
    ](
        config,
        transport_payload_type=TransportMessage[
            Union[PrefillRequest, PrefillResponse, KVTransferEngineMetadata]
        ],
    )


@pytest.fixture
def prefill_paged_manager():
    params = KVCacheParams(
        dtype=DType.float16,
        n_kv_heads=8,
        head_dim=128,
        cache_strategy=KVCacheStrategy.PAGED,
        enable_prefix_caching=True,
        enable_kvcache_swapping_to_host=False,
        page_size=128,
    )

    manager = load_kv_manager(
        params=params,
        max_batch_size=4,
        max_seq_len=512,
        num_layers=1,
        devices=[CPU()],
        session=InferenceSession(devices=[CPU()]),
        available_cache_memory=500 * 2**24,
        page_size=128,
    )

    return manager


@pytest.fixture
def prefill_scheduler(
    mock_pipeline,  # noqa: ANN001
    prefill_paged_manager,  # noqa: ANN001
    prefill_dispatcher_factory,  # noqa: ANN001
) -> Callable[[], PrefillScheduler]:
    def create_scheduler() -> PrefillScheduler:
        # Create dispatcher client
        prefill_client = prefill_dispatcher_factory.create_client()
        prefill_client.start()

        # Initialize scheduler config
        config = TokenGenerationSchedulerConfig(
            max_batch_size_ce=4,
            target_tokens_per_batch_ce=128,
            enable_chunked_prefill=True,
            max_batch_size_tg=4,
            max_forward_steps_tg=10,
        )

        return PrefillScheduler(
            pipeline=mock_pipeline,
            scheduler_config=config,
            paged_cache=prefill_paged_manager,
            dispatcher_client=prefill_client,
        )

    return create_scheduler


@pytest.mark.skip(reason="May time out")
@pytest.mark.asyncio
async def test_transfer_between_prefill_and_decode_scheduler(
    prefill_scheduler,  # noqa: ANN001
    decode_scheduler,  # noqa: ANN001
    decode_request_zmq_path,  # noqa: ANN001
    decode_response_zmq_path,  # noqa: ANN001
    decode_cancel_zmq_path,  # noqa: ANN001
    decode_dispatcher_factory,  # noqa: ANN001
    prefill_dispatcher_factory,  # noqa: ANN001
) -> None:
    # Create request push socket
    request_push_socket = ZmqPushSocket[tuple[str, InputContext]](
        zmq_endpoint=decode_request_zmq_path,
        serialize=msgpack_numpy_encoder(),
    )

    # Create response pull socket
    response_pull_socket = ZmqPullSocket[
        dict[RequestID, SchedulerResult[TextGenerationOutput]]
    ](
        zmq_endpoint=decode_response_zmq_path,
        deserialize=msgpack_numpy_decoder(
            dict[RequestID, SchedulerResult[TextGenerationOutput]]
        ),
    )

    # Create cancel push socket
    cancel_push_socket = ZmqPushSocket[list[str]](
        zmq_endpoint=decode_cancel_zmq_path,
        serialize=msgpack_numpy_encoder(),
    )

    # Create queues for assertion results
    decode_queue: Queue[Union[bool, Exception]] = Queue()
    prefill_queue: Queue[Union[bool, Exception]] = Queue()

    def run_decode_scheduler_tests(decode_result_queue) -> None:  # noqa: ANN001
        asyncio.run(_run_decode_scheduler_tests(decode_result_queue))

    async def _run_decode_scheduler_tests(decode_result_queue) -> None:  # noqa: ANN001
        try:
            # Create and start decode service thread
            print(
                f"decode worker {datetime.now().strftime('%H:%M:%S.%f')[:-3]}: creating dispatcher service"
            )
            decode_service = decode_dispatcher_factory.create_service()
            print(
                f"decode service {datetime.now().strftime('%H:%M:%S.%f')[:-3]}: pull endpoint {decode_service.local_pull_socket.zmq_endpoint}"
            )
            await decode_service.start()
            print(
                f"decode worker {datetime.now().strftime('%H:%M:%S.%f')[:-3]}: decode dispatcher service started"
            )

            # Create decode scheduler
            print(
                f"decode worker {datetime.now().strftime('%H:%M:%S.%f')[:-3]}: initializing scheduler"
            )
            scheduler_instance = decode_scheduler()
            print(
                f"decode worker {datetime.now().strftime('%H:%M:%S.%f')[:-3]}: scheduler initialized successfully"
            )

            async def _verify_request_forwarded_to_prefill(
                request_id: str,
            ) -> None:
                # Try and Forward the First Request to the Prefill Scheduler
                i = 0
                sent = False
                while i < 5:
                    print(
                        f"decode worker {datetime.now().strftime('%H:%M:%S.%f')[:-3]}: reserving memory for {request_id} and sending to prefill"
                    )
                    scheduler_instance.reserve_memory_and_send_to_prefill()

                    if request_id in scheduler_instance.reserved_cache_indices:
                        sent = True
                        print(
                            f"decode worker {datetime.now().strftime('%H:%M:%S.%f')[:-3]}: memory reserved for {request_id} and sent to prefill"
                        )
                        break

                    print(
                        f"decode worker {datetime.now().strftime('%H:%M:%S.%f')[:-3]}: {request_id} not available, waiting for 1s"
                    )
                    await asyncio.sleep(1)
                    i += 1

                if not sent:
                    raise RuntimeError(
                        f"{request_id} not received on decode worker, and not sent to prefill"
                    )

            async def _verify_transfer_engine_registered() -> None:
                # Check if the prefill scheduler, sent back transfer metadata, then
                # registered with the Decode scheduler
                engine_registered = False
                i = 0
                while i < 5:
                    if (
                        len(
                            scheduler_instance.transfer_engine.remote_connections
                        )
                        == 1
                    ):
                        engine_registered = True
                        print(
                            f"decode worker {datetime.now().strftime('%H:%M:%S.%f')[:-3]}: remote transfer engine registered with decode scheduler"
                        )
                        break
                    await asyncio.sleep(1)
                    i += 1

                if not engine_registered:
                    raise RuntimeError(
                        "no remote transfer engine registered with decode scheduler"
                    )

            async def _verify_prefill_executed_and_transfer_complete(
                request_id: str,
            ) -> None:
                # Check if the Prefill scheduler, executed prefill, initiated the transfer
                # and replied
                transfer_complete = False
                i = 0
                while i < 5:
                    scheduler_instance.update_batch()
                    if request_id in scheduler_instance.active_batch:
                        transfer_complete = True
                        print(
                            f"decode worker {datetime.now().strftime('%H:%M:%S.%f')[:-3]}: scheduler received a prefill response for {request_id} and transfer"
                        )
                        break
                    print(
                        f"decode worker {datetime.now().strftime('%H:%M:%S.%f')[:-3]}: scheduler waiting for transfer for {request_id} to be completed"
                    )
                    await asyncio.sleep(1)
                    i += 1

                if not transfer_complete:
                    raise RuntimeError(
                        f"decode scheduler did not receive a reply and complete transfer for {request_id}"
                    )

            # Verify the first request
            await _verify_request_forwarded_to_prefill("request_1")
            await _verify_transfer_engine_registered()
            await _verify_prefill_executed_and_transfer_complete("request_1")

            # # Verify the second request
            await asyncio.sleep(5)
            await _verify_request_forwarded_to_prefill("request_2")
            await _verify_transfer_engine_registered()
            await _verify_prefill_executed_and_transfer_complete("request_2")

            # Signal success
            print(
                f"decode worker {datetime.now().strftime('%H:%M:%S.%f')[:-3]}: all tests successful"
            )
            decode_result_queue.put_nowait(True)
        except Exception as e:
            # Send the exception back to the main process
            decode_result_queue.put_nowait(e)

        print(
            f"decode worker {datetime.now().strftime('%H:%M:%S.%f')[:-3]}: exiting"
        )

    def run_prefill_scheduler_tests() -> None:
        asyncio.run(_run_prefill_scheduler_tests())

    async def _run_prefill_scheduler_tests() -> None:
        try:
            # Create and start prefill service thread
            print(
                f"prefill worker {datetime.now().strftime('%H:%M:%S.%f')[:-3]}: creating dispatcher service"
            )
            prefill_service = prefill_dispatcher_factory.create_service()
            await prefill_service.start()
            print(
                f"prefill worker {datetime.now().strftime('%H:%M:%S.%f')[:-3]}: prefill dispatcher service started"
            )

            # Create prefill scheduler
            print(
                f"prefill worker {datetime.now().strftime('%H:%M:%S.%f')[:-3]}: initializing scheduler"
            )
            scheduler_instance = prefill_scheduler()
            print(
                f"prefill worker {datetime.now().strftime('%H:%M:%S.%f')[:-3]}: scheduler initialized successfully"
            )

            # Check if the new request has been added to the batch
            async def _verify_prefill_request_received(request_id: str) -> None:
                i = 0
                received = False
                while i < 5:
                    print(
                        f"prefill worker {datetime.now().strftime('%H:%M:%S.%f')[:-3]}: checking prefill request queue for {request_id}"
                    )
                    scheduler_instance.update_batch()

                    if request_id in scheduler_instance.active_batch:
                        received = True
                        print(
                            f"prefill worker {datetime.now().strftime('%H:%M:%S.%f')[:-3]}: {request_id} added to active prefill batch"
                        )

                        # Check that prefill_request has an appropriate dst_idx
                        prefill_request = scheduler_instance.pending_transfers[
                            request_id
                        ]
                        assert len(prefill_request.block_ids) > 0, (
                            f"prefill_request.block_ids must be longer than 0: {prefill_request.block_ids}"
                        )

                        break

                    print(
                        f"prefill worker {datetime.now().strftime('%H:%M:%S.%f')[:-3]}: {request_id} not in active prefill batch, waiting 1s"
                    )
                    await asyncio.sleep(1)
                    i += 1

                if not received:
                    raise RuntimeError(
                        f"{request_id} not received on the prefill scheduler"
                    )

            async def _verify_transfer_engine_registered() -> None:
                # Wait a few seconds to ensure everything is processed.
                transfer_engine_registered = False
                i = 0
                while i < 5:
                    if (
                        len(
                            scheduler_instance.transfer_engine.remote_connections
                        )
                        > 0
                    ):
                        transfer_engine_registered = True
                        break

                    await asyncio.sleep(1)
                    i += 1

                if not transfer_engine_registered:
                    raise RuntimeError(
                        "remote transfer engine does not appear to be registered with the prefill scheduler"
                    )
                print(
                    f"prefill worker {datetime.now().strftime('%H:%M:%S.%f')[:-3]}: transfer engine registered"
                )

            async def _verify_transfer_initiated(request_id: str) -> None:
                # Run Prefill and Initiate the Transfer
                print(
                    f"prefill worker {datetime.now().strftime('%H:%M:%S.%f')[:-3]}: executing prefill for {request_id} and initiating transfer"
                )
                scheduler_instance.schedule()
                assert len(scheduler_instance.active_batch) == 0
                assert request_id in scheduler_instance.active_transfers
                print(
                    f"prefill worker {datetime.now().strftime('%H:%M:%S.%f')[:-3]}: executed prefill for {request_id} and transfer in progress"
                )

            async def _verify_transfer_cleanup(request_id: str) -> None:
                # Check that the transfer completes successfully and cleansup.
                i = 0
                transfer_complete = False
                while i < 5:
                    print(
                        f"prefill worker {datetime.now().strftime('%H:%M:%S.%f')[:-3]}: checking that transfer for {request_id} completes"
                    )
                    scheduler_instance.cleanup_active_transfers()
                    if request_id not in scheduler_instance.active_transfers:
                        transfer_complete = True
                        break

                    await asyncio.sleep(1)
                    i += 1

                if not transfer_complete:
                    raise RuntimeError(
                        "prefill scheduler never completed transfer successfully"
                    )

                print(
                    f"prefill worker {datetime.now().strftime('%H:%M:%S.%f')[:-3]}: transfer for {request_id} appears to complete successfully"
                )

            # Verify Request 1
            await _verify_prefill_request_received("request_1")
            await _verify_transfer_engine_registered()
            await _verify_transfer_initiated("request_1")
            await _verify_transfer_cleanup("request_1")

            # Verify Request 2
            # Note we dont have to test the transfer engine is re-registered.
            await asyncio.sleep(5)
            await _verify_prefill_request_received("request_2")
            await _verify_transfer_initiated("request_2")
            await _verify_transfer_cleanup("request_2")

            print(
                f"prefill worker {datetime.now().strftime('%H:%M:%S.%f')[:-3]}: all tests successful"
            )
            prefill_queue.put(True)
        except Exception as e:
            # Send the exception back to the main process
            prefill_queue.put(e)

        print(
            f"prefill worker {datetime.now().strftime('%H:%M:%S.%f')[:-3]}: exiting"
        )

    # Run both schedulers concurrently in separate processes
    decode_process = Process(
        target=run_decode_scheduler_tests, args=(decode_queue,)
    )
    prefill_process = Process(target=run_prefill_scheduler_tests)

    print(
        f"test process: starting decode process at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}"
    )
    decode_process.start()
    print(
        f"test process: decode process started at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}"
    )
    print(
        f"test process: starting prefill process at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}"
    )
    prefill_process.start()
    print(
        f"test process: prefill process started at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}"
    )

    # Create a mock text context
    mock_request_1 = TextContext(
        max_length=100,
        tokens=np.array([1, 2, 3, 4, 5], dtype=np.int32),
    )

    # Send the request to the decode scheduler
    print(
        f"test process: sending request to decode worker at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}"
    )
    request_push_socket.put_nowait(("request_1", mock_request_1))
    print(
        f"test process: sent request to decode worker successfully at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}"
    )

    # Wait and just process the first request
    time.sleep(5)

    mock_request_2 = TextContext(
        max_length=100,
        tokens=np.array([1, 2, 3], dtype=np.int32),
    )

    # Send the request to the decode scheduler
    print(
        f"test process: sending request to decode worker at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}"
    )
    request_push_socket.put_nowait(("request_2", mock_request_2))
    print(
        f"test process: sent request to decode worker successfully at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}"
    )

    # Wait for both processes to complete
    print(
        f"test process: waiting for decode process to complete at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}"
    )
    decode_process.join()
    print(
        f"test process: decode process complete at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}"
    )
    prefill_process.join()

    # Check for any exceptions from the subprocesses
    decode_result = decode_queue.get_nowait()
    prefill_result = prefill_queue.get_nowait()

    if isinstance(decode_result, Exception):
        raise decode_result
    if isinstance(prefill_result, Exception):
        raise prefill_result
