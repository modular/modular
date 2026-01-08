# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import queue
import time
from collections.abc import Callable
from typing import TypeVar, cast

import numpy as np
from max.driver import CPU, Device
from max.interfaces import (
    RequestID,
    SchedulerResult,
    TextGenerationOutput,
    TokenBuffer,
)
from max.kv_cache.paged_cache.transfer_engine import KVTransferEngineMetadata
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
    def __init__(self, bind_addr: str):
        self.bind_addr = bind_addr
        super().__init__(bind_addr=bind_addr)

    def recv_reply_nowait(self) -> ReplyType:
        return blocking_recv(super().recv_reply_nowait)


def create_text_context(
    target_endpoint: str,
    prompt_len: int,
    output_len: int | None = None,
) -> TextContext:
    tokens = TokenBuffer(np.ones(prompt_len, dtype=np.int64))
    if output_len is not None:
        max_length = prompt_len + output_len
    else:
        max_length = 2048
    return TextContext(
        request_id=RequestID(),
        max_length=max_length,
        tokens=tokens,
        target_endpoint=target_endpoint,
    )


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
    dp: int = 1,
    device: Device = CPU(),
) -> tuple[DecodeScheduler, PrefillScheduler, str]:
    def _create_paged_manager() -> PagedKVCacheManager:
        return create_paged_manager(
            num_blocks=num_blocks,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            page_size=page_size,
            enable_prefix_caching=enable_prefix_caching,
            enable_kvcache_swapping_to_host=enable_kvcache_swapping_to_host,
            dp=dp,
            device=device,
        )

    # Create a scheduler with a paged manager
    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=max_batch_size,
        max_forward_steps_tg=max_forward_steps_tg,
        target_tokens_per_batch_ce=target_tokens_per_batch_ce,
        max_seq_len=max_seq_len,
        enable_chunked_prefill=enable_chunked_prefill,
        enable_in_flight_batching=enable_in_flight_batching,
        data_parallel_degree=dp,
    )

    # Use queue.Queue to simulate the ZMQ queues.
    request_queue: queue.Queue[TextContext] = queue.Queue()
    response_queue: queue.Queue[
        dict[RequestID, SchedulerResult[TextGenerationOutput]]
    ] = queue.Queue()
    cancel_queue: queue.Queue[list[RequestID]] = queue.Queue()

    paged_manager_prefill = _create_paged_manager()
    paged_manager_decode = _create_paged_manager()
    server_addr = generate_zmq_ipc_path()
    client_addr = generate_zmq_ipc_path()
    dispatcher_server = BasicDispatcherServer(bind_addr=server_addr)
    dispatcher_client = BasicDispatcherClient(bind_addr=client_addr)

    decode_scheduler = DecodeScheduler(
        pipeline=FakeTokenGeneratorPipeline(
            paged_manager_decode, max_seq_len=max_seq_len, start_token_id=42
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
            paged_manager_prefill, max_seq_len=max_seq_len, start_token_id=99
        ),
        scheduler_config=scheduler_config,
        paged_cache=paged_manager_prefill,
        dispatcher=dispatcher_server,
    )

    return decode_scheduler, prefill_scheduler, server_addr


def create_default_di_scheduler_and_submit_one_request() -> tuple[
    DecodeScheduler, PrefillScheduler, TextContext
]:
    decode, prefill, server_addr = create_di_scheduler()
    ctx = create_text_context(
        target_endpoint=server_addr, prompt_len=100, output_len=5
    )
    request_queue: queue.Queue = cast(queue.Queue, decode.request_queue)
    request_queue.put(ctx)
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
    assert ctx2.tokens.processed_length == 0
    assert ctx2.tokens.active_length == 100


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


def test_di_with_dp2_requests_distributed_to_different_replicas() -> None:
    """Test that with DP=2, requests are distributed to different replicas."""
    decode, prefill, server_addr = create_di_scheduler(dp=2)

    # Create and submit two requests
    ctx1 = create_text_context(target_endpoint=server_addr, prompt_len=1111)
    ctx2 = create_text_context(target_endpoint=server_addr, prompt_len=1111)
    ctx3 = create_text_context(target_endpoint=server_addr, prompt_len=1111)
    request_queue: queue.Queue = cast(queue.Queue, decode.request_queue)
    request_queue.put(ctx1)
    request_queue.put(ctx2)
    request_queue.put(ctx3)

    # Send requests from decode -> prefill
    decode.reserve_memory_and_send_to_prefill()

    # Check that prefill received the transfer engine metadata
    decode_metadata, _ = prefill.dispatcher.recv_request_nowait()
    assert isinstance(decode_metadata, KVTransferEngineMetadata)

    # Check that first request was assigned to replica 0
    prefill_request1, _ = prefill.dispatcher.recv_request_nowait()
    assert isinstance(prefill_request1, PrefillRequest)
    assert prefill_request1.dst_replica_idx == 0
    assert prefill_request1.dst_block_ids == [0, 1, 2, 3, 4, 5, 6, 7, 8]

    # Check that second request was assigned to replica 1
    prefill_request2, _ = prefill.dispatcher.recv_request_nowait()
    assert isinstance(prefill_request2, PrefillRequest)
    assert prefill_request2.dst_replica_idx == 1
    assert prefill_request2.dst_block_ids == [0, 1, 2, 3, 4, 5, 6, 7, 8]

    # Check that third request was assigned to replica 0
    prefill_request3, _ = prefill.dispatcher.recv_request_nowait()
    assert isinstance(prefill_request3, PrefillRequest)
    assert prefill_request3.dst_replica_idx == 0
    assert prefill_request3.dst_block_ids == [9, 10, 11, 12, 13, 14, 15, 16, 17]


def test_di_with_dp2_end_to_end() -> None:
    """Test end-to-end DI flow with DP=2."""
    decode, prefill, server_addr = create_di_scheduler(dp=2)

    # Create and submit two requests
    ctx1 = create_text_context(
        target_endpoint=server_addr, prompt_len=100, output_len=5
    )
    ctx2 = create_text_context(
        target_endpoint=server_addr, prompt_len=100, output_len=5
    )
    req_id1 = ctx1.request_id
    req_id2 = ctx2.request_id
    request_queue: queue.Queue = cast(queue.Queue, decode.request_queue)
    response_q = cast(queue.Queue, decode.response_queue)
    request_queue.put(ctx1)
    request_queue.put(ctx2)

    # Send requests from decode -> prefill
    decode.run_iteration()
    # Execute prefill, generating tokens 99 and 100 for the two requests respectively
    prefill.run_iteration()
    # Stream tokens to frontend and execute decode
    decode.run_iteration()

    # Collect all outputs from the queue - there should be 4 total:
    # 2 prefill responses and 2 decode responses
    req1_outputs: list[TextGenerationOutput] = []
    req2_outputs: list[TextGenerationOutput] = []
    for output in response_q.queue:
        for req_id, sch_result in output.items():
            result = sch_result.result
            assert isinstance(result, TextGenerationOutput)
            if req_id == req_id1:
                req1_outputs.append(result)
            elif req_id == req_id2:
                req2_outputs.append(result)
            else:
                raise ValueError(f"Unexpected request ID: {req_id}")

    # Check req1: first token from prefill (99), then decode tokens
    assert len(req1_outputs) == 2
    assert req1_outputs[0].tokens == [99]  # From prefill
    assert req1_outputs[1].tokens == [42, 43, 44, 45]  # From decode

    # Check req2: first token from prefill (100), then decode tokens
    assert len(req2_outputs) == 2
    assert req2_outputs[0].tokens == [100]  # From prefill
    assert req2_outputs[1].tokens == [46, 47, 48, 49]  # From decode
