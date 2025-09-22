# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import queue
import time
from typing import Callable, TypeVar, cast

from max.driver import CPU, Device
from max.interfaces import RequestID, SchedulerResult, TextGenerationOutput
from max.nn.kv_cache.paged_cache.transfer_engine import KVTransferEngineMetadata
from max.pipelines.core import TextContext
from max.serve.config import generate_zmq_ipc_path
from max.serve.queue.zmq_queue import ClientIdentity
from max.serve.scheduler.base import PrefillRequest, PrefillResponse
from max.serve.scheduler.decode_scheduler import (
    DecodeScheduler,
    TokenGenerationSchedulerConfig,
)
from max.serve.scheduler.di_dispatchers import (
    DecodeDispatcherClientV2,
    PrefillDispatcherServerV2,
    ReplyType,
    RequestType,
)
from max.serve.scheduler.prefill_scheduler import PrefillScheduler
from tests.serve.scheduler.common import (
    FakeTokenGeneratorPipeline,
    PagedKVCacheManager,
    create_paged_manager,
    create_text_context,
)

TIMEOUT = 1.0
T = TypeVar("T")


def blocking_recv(fn: Callable[[], T], timeout: float = TIMEOUT) -> T:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            return fn()
        except queue.Empty:
            time.sleep(0.001)
    raise queue.Empty()


class BasicDispatcherServer(PrefillDispatcherServerV2):
    def __init__(self, bind_addr: str):
        self.bind_addr = bind_addr
        super().__init__(bind_addr=bind_addr)

    def recv_request_nowait(self) -> tuple[RequestType, ClientIdentity]:
        return blocking_recv(super().recv_request_nowait)


class BasicDispatcherClient(DecodeDispatcherClientV2):
    def __init__(self, bind_addr: str, default_dest_addr: str | None):
        self.bind_addr = bind_addr
        super().__init__(
            bind_addr=bind_addr,
            default_dest_addr=default_dest_addr,
        )

    def recv_reply_nowait(self) -> ReplyType:
        return blocking_recv(super().recv_reply_nowait)


def create_di_scheduler(
    max_seq_len: int = 2048,
    num_blocks: int = 9999,
    max_batch_size: int = 512,
    page_size: int = 128,
    max_forward_steps_tg: int = 10,
    target_tokens_per_batch_ce: int = 8192,
    enable_prefix_caching: bool = False,
    enable_in_flight_batching: bool = False,
    enable_chunked_prefill: bool = True,
    enable_kvcache_swapping_to_host: bool = False,
    device: Device = CPU(),
) -> tuple[DecodeScheduler, PrefillScheduler]:
    def _create_paged_manager() -> PagedKVCacheManager:
        return create_paged_manager(
            num_blocks=num_blocks,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            page_size=page_size,
            enable_prefix_caching=enable_prefix_caching,
            enable_kvcache_swapping_to_host=enable_kvcache_swapping_to_host,
            device=device,
        )

    # Create a scheduler with a paged manager
    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size_tg=max_batch_size,
        max_forward_steps_tg=max_forward_steps_tg,
        max_batch_size_ce=max_batch_size,
        target_tokens_per_batch_ce=target_tokens_per_batch_ce,
        enable_chunked_prefill=enable_chunked_prefill,
        enable_in_flight_batching=enable_in_flight_batching,
    )

    # Use queue.Queue to simulate the ZMQ queues.
    request_queue: queue.Queue[tuple[RequestID, TextContext]] = queue.Queue()
    response_queue: queue.Queue[
        dict[RequestID, SchedulerResult[TextGenerationOutput]]
    ] = queue.Queue()
    cancel_queue: queue.Queue[list[RequestID]] = queue.Queue()

    paged_manager_prefill = _create_paged_manager()
    paged_manager_decode = _create_paged_manager()
    server_addr = generate_zmq_ipc_path()
    client_addr = generate_zmq_ipc_path()
    dispatcher_server = BasicDispatcherServer(
        bind_addr=server_addr,
    )
    dispatcher_client = BasicDispatcherClient(
        bind_addr=client_addr,
        default_dest_addr=server_addr,
    )

    decode_scheduler = DecodeScheduler(
        pipeline=FakeTokenGeneratorPipeline(
            paged_manager_decode, start_token_id=42
        ),
        scheduler_config=scheduler_config,
        paged_manager=paged_manager_decode,
        request_queue=request_queue,
        response_queue=response_queue,
        cancel_queue=cancel_queue,
        dispatcher=dispatcher_client,
    )

    prefill_scheduler = PrefillScheduler(
        pipeline=FakeTokenGeneratorPipeline(
            paged_manager_prefill, start_token_id=99
        ),
        scheduler_config=scheduler_config,
        paged_cache=paged_manager_prefill,
        dispatcher=dispatcher_server,
    )

    return decode_scheduler, prefill_scheduler


def create_default_di_scheduler_and_submit_one_request() -> tuple[
    DecodeScheduler, PrefillScheduler, TextContext
]:
    decode, prefill = create_di_scheduler()
    ctx = create_text_context(prompt_len=100, max_seq_len=105)
    request_queue: queue.Queue = cast(queue.Queue, decode.request_queue)
    request_queue.put((ctx.request_id, ctx))
    return decode, prefill, ctx


def test_decode_sends_request_to_prefill() -> None:
    decode, prefill, _ = create_default_di_scheduler_and_submit_one_request()

    # Send request from decode -> prefill
    decode.reserve_memory_and_send_to_prefill()

    # Check that prefill received the transfer engine metadata
    decode_metadata, client_identity = prefill.dispatcher.recv_request_nowait()
    assert isinstance(decode_metadata, KVTransferEngineMetadata)
    assert decode_metadata.name == decode.transfer_engine.name

    # Check that prefill received the request
    prefill_request, client_identity2 = prefill.dispatcher.recv_request_nowait()
    assert isinstance(prefill_request, PrefillRequest)
    ctx2 = prefill_request.context
    assert client_identity2 == client_identity
    assert ctx2.start_idx == 0
    assert ctx2.active_length == 100


def test_prefill_sends_new_token_to_decode() -> None:
    decode, prefill, ctx = create_default_di_scheduler_and_submit_one_request()

    # Send request from decode -> prefill
    decode.reserve_memory_and_send_to_prefill()

    # Received the request and execute prefill with num_steps=1, generating token 99
    # Send response from prefill -> decode
    prefill.run_iteration()

    # Check that decode received the response
    prefill_metadata = decode.dispatcher.recv_reply_nowait()
    assert isinstance(prefill_metadata, KVTransferEngineMetadata)
    prefill_response = decode.dispatcher.recv_reply_nowait()
    assert isinstance(prefill_response, PrefillResponse)
    assert prefill_response.id == ctx.request_id
    assert prefill_response.generated_token_id == 99


def test_one_req_end_to_end() -> None:
    decode, prefill, ctx = create_default_di_scheduler_and_submit_one_request()
    req_id = ctx.request_id

    # Send request from decode -> prefill
    decode.run_iteration()
    # Execute prefill with num_steps=1, generating token 99
    # Send response from prefill -> decode
    prefill.run_iteration()
    # Stream token 99 to frontend
    # Execute decode with num_steps=4, generating token 42, 43, 44, 45
    # Stream tokens 42, 43, 44, 45 to frontend
    decode.run_iteration()

    # Hacky cast to get the response queue
    response_q = cast(queue.Queue, decode.response_queue)

    # Check that the first token is 99
    output1 = response_q.get()
    assert len(output1) == 1
    sch_output1 = output1[req_id]
    assert not sch_output1.is_done
    single_token = sch_output1.result
    assert isinstance(single_token, TextGenerationOutput)
    assert single_token.request_id == req_id
    assert single_token.tokens == [99]

    # Check that the rest of the tokens are 42, 43, 44, 45
    output2 = response_q.get()
    assert len(output2) == 1
    sch_output2 = output2[req_id]
    assert sch_output2.is_done
    rest_of_tokens = sch_output2.result
    assert isinstance(rest_of_tokens, TextGenerationOutput)
    assert rest_of_tokens.request_id == req_id
    assert rest_of_tokens.tokens == [42, 43, 44, 45]
