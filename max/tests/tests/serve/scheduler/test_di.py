# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
from unittest.mock import MagicMock

import numpy as np
from max.driver import CPU, Device
from max.interfaces import (
    GenerationStatus,
    RequestID,
    SchedulerResult,
    TextGenerationInputs,
    TextGenerationOutput,
    TokenBuffer,
)
from max.kv_cache.paged_kv_cache.transfer_engine import KVTransferEngineMetadata
from max.nn.kv_cache import KVConnectorType
from max.pipelines.core import TextContext
from max.pipelines.core.context import FUTURE_TOKEN
from max.pipelines.lib import OverlapTextGenerationPipeline
from max.serve.config import generate_zmq_ipc_path
from max.serve.scheduler.base import (
    CancelRequest,
    PrefillRequest,
    PrefillResponse,
    SchedulerProgress,
)
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
from max.serve.worker_interface.zmq_queue import ClientIdentity
from tests.serve.scheduler.common import (
    FakeOverlapPipeline,
    FakeTokenGeneratorPipeline,
    PagedKVCacheManager,
    create_kv_cache,
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
    kv_connector: KVConnectorType | None = None,
    dp: int = 1,
    device: Device = CPU(),
    overlap_prefill: bool = False,
) -> tuple[DecodeScheduler, PrefillScheduler, str]:
    """Creates a DecodeScheduler and PrefillScheduler pair for testing.

    Args:
        overlap_prefill: When True, the PrefillScheduler uses FakeOverlapPipeline,
            which mimics the one-batch output lag of OverlapTextGenerationPipeline.
    """

    def _create_kv_cache() -> PagedKVCacheManager:
        return create_kv_cache(
            num_blocks=num_blocks,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            page_size=page_size,
            enable_prefix_caching=enable_prefix_caching,
            kv_connector=kv_connector,
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

    kv_cache_prefill = _create_kv_cache()
    kv_cache_decode = _create_kv_cache()
    server_addr = generate_zmq_ipc_path()
    client_addr = generate_zmq_ipc_path()
    dispatcher_server = BasicDispatcherServer(bind_addr=server_addr)
    dispatcher_client = BasicDispatcherClient(bind_addr=client_addr)

    decode_scheduler = DecodeScheduler(
        pipeline=FakeTokenGeneratorPipeline(
            kv_cache_decode, max_seq_len=max_seq_len, start_token_id=42
        ),
        scheduler_config=scheduler_config,
        kv_cache=kv_cache_decode,
        request_queue=request_queue,
        response_queue=response_queue,
        cancel_queue=cancel_queue,
        dispatcher=dispatcher_client,
    )

    prefill_pipeline = (
        FakeOverlapPipeline(
            kv_cache_prefill, max_seq_len=max_seq_len, start_token_id=99
        )
        if overlap_prefill
        else FakeTokenGeneratorPipeline(
            kv_cache_prefill, max_seq_len=max_seq_len, start_token_id=99
        )
    )

    prefill_scheduler = PrefillScheduler(
        pipeline=prefill_pipeline,
        scheduler_config=scheduler_config,
        kv_cache=kv_cache_prefill,
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
    request_queue: queue.Queue = cast(queue.Queue, decode.request_queue)  # type: ignore[type-arg]
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
    response_q = cast(queue.Queue, decode.response_queue)  # type: ignore[type-arg]

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
    request_queue: queue.Queue = cast(queue.Queue, decode.request_queue)  # type: ignore[type-arg]
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
    request_queue: queue.Queue = cast(queue.Queue, decode.request_queue)  # type: ignore[type-arg]
    response_q = cast(queue.Queue, decode.response_queue)  # type: ignore[type-arg]
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


def test_overlap_di_schedule_filters_stale_responses() -> None:
    """Verify schedule() drops responses for request IDs not in batch_constructor."""
    decode, prefill, server_addr = create_di_scheduler(max_forward_steps_tg=1)
    ctx = create_text_context(
        target_endpoint=server_addr, prompt_len=100, output_len=10
    )
    req_id = ctx.request_id
    request_queue: queue.Queue = cast(queue.Queue, decode.request_queue)  # type: ignore[type-arg]
    response_q = cast(queue.Queue, decode.response_queue)  # type: ignore[type-arg]
    request_queue.put(ctx)

    # Send to prefill, execute prefill, then run decode
    # (streams prefill token + generates 1 decode token).
    decode.run_iteration()
    prefill.run_iteration()
    decode.run_iteration()

    # Drain both responses (prefill token, first decode token).
    assert response_q.qsize() == 2
    response_q.get()
    response_q.get()

    # Patch execute to inject a stale response for a fabricated request ID.
    stale_id = RequestID()
    original_execute = decode.pipeline.execute

    def patched_execute(
        inputs: TextGenerationInputs[TextContext],
    ) -> dict[RequestID, TextGenerationOutput]:
        responses = original_execute(inputs)
        responses[stale_id] = TextGenerationOutput(
            request_id=stale_id,
            tokens=[999],
            final_status=GenerationStatus.ACTIVE,
        )
        return responses

    decode.pipeline.execute = patched_execute  # type: ignore[method-assign]

    # Run another decode iteration; the stale response should be filtered.
    decode.run_iteration()

    output = response_q.get()
    assert req_id in output
    assert stale_id not in output


def test_overlap_di_has_pending_outputs_prevents_no_progress() -> None:
    """Verify run_iteration() returns MADE_PROGRESS when the batch is empty,
    but the overlap pipeline reports pending outputs."""
    decode, _prefill, _server_addr = create_di_scheduler()

    # Simulate overlap pipeline behavior without a real OverlapPipeline.
    mock_pipeline = MagicMock(spec=OverlapTextGenerationPipeline)
    mock_pipeline.has_pending_outputs.return_value = True
    mock_pipeline.execute.return_value = {}
    decode.pipeline = mock_pipeline

    result = decode.run_iteration()
    assert result == SchedulerProgress.MADE_PROGRESS


def test_prefill_reqs_per_replica_decremented_on_completion() -> None:
    """prefill_reqs_per_replica must return to [0, 0] after requests complete
    end-to-end with DP=2.

    Regression: check_for_completed_transfers popped from prefill_reqs
    without decrementing prefill_reqs_per_replica, causing the counter to
    drift and degrade DP replica load balancing.
    """
    decode, prefill, server_addr = create_di_scheduler(dp=2)
    request_queue: queue.Queue = cast(queue.Queue, decode.request_queue)  # type: ignore[type-arg]

    # Submit 2 requests
    ctx1 = create_text_context(
        target_endpoint=server_addr, prompt_len=100, output_len=5
    )
    ctx2 = create_text_context(
        target_endpoint=server_addr, prompt_len=100, output_len=5
    )
    request_queue.put(ctx1)
    request_queue.put(ctx2)

    # Run end-to-end
    decode.run_iteration()
    prefill.run_iteration()
    decode.run_iteration()

    # Both requests should have been popped from prefill_reqs
    assert decode.prefill_reqs == {}

    # prefill_reqs_per_replica must be back to zero for both replicas
    assert decode.prefill_reqs_per_replica == [0, 0], (
        f"prefill_reqs_per_replica not decremented on normal completion: "
        f"{decode.prefill_reqs_per_replica}"
    )


def test_cancel_pending_prefill_releases_decode_kv_blocks() -> None:
    """Cancelling a request pending prefill must release its KV cache blocks
    on the decode side.

    Regression: _handle_cancelled_requests removed the request from
    prefill_reqs but never called kv_cache.release, permanently leaking
    the blocks allocated before sending to prefill.
    """
    decode, _, ctx = create_default_di_scheduler_and_submit_one_request()
    req_id = ctx.request_id
    cancel_queue: queue.Queue = cast(queue.Queue, decode.cancel_queue)  # type: ignore[type-arg]
    response_q = cast(queue.Queue, decode.response_queue)  # type: ignore[type-arg]

    # Record baseline KV usage.
    pages_before = decode.kv_cache.get_num_used_pages(replica_idx=0)

    # Send to prefill -> allocates KV blocks on decode
    decode.run_iteration()

    pages_after_send = decode.kv_cache.get_num_used_pages(replica_idx=0)
    assert pages_after_send > pages_before, (
        "Expected KV blocks to be allocated after sending to prefill"
    )

    # Cancel before prefill runs
    cancel_queue.put([req_id])
    decode.run_iteration()

    # Drain the cancelled response
    assert not response_q.empty()
    batch = response_q.get()
    assert req_id in batch
    assert batch[req_id].result is None  # cancelled

    # KV blocks must be released back to pool
    pages_after_cancel = decode.kv_cache.get_num_used_pages(replica_idx=0)
    assert pages_after_cancel == pages_before, (
        f"KV blocks leaked after cancel: had {pages_before} before, "
        f"{pages_after_cancel} after cancel (expected {pages_before}). "
        f"Delta = {pages_after_cancel - pages_before} pages leaked."
    )


def test_stale_prefill_response_after_cancel_does_not_crash() -> None:
    """A PrefillResponse arriving after the request was cancelled must be
    silently discarded, not raise KeyError.

    Regression: handle_prefill_response accessed self.prefill_reqs[request_id]
    without checking membership, crashing when the request had already been
    cancelled and removed in a prior iteration.
    """
    decode, prefill, ctx = create_default_di_scheduler_and_submit_one_request()
    req_id = ctx.request_id
    cancel_queue: queue.Queue = cast(queue.Queue, decode.cancel_queue)  # type: ignore[type-arg]

    # Send to prefill
    decode.run_iteration()

    # Cancel before prefill runs
    cancel_queue.put([req_id])
    decode.run_iteration()

    # Prefill runs now and sends a PrefillResponse (has not seen cancel)
    prefill.run_iteration()

    # Decode receives the stale PrefillResponse
    # It must not crash and must discard it
    decode.run_iteration()

    assert req_id not in decode.prefill_reqs
    assert req_id not in decode.inflight_transfers
    assert not decode.batch_constructor.contains(req_id)


def test_prefix_caching_marks_cached_blocks_in_prefill_request() -> None:
    """Sending the same prompt twice marks already cached blocks as -1 in dst_block_ids."""
    page_size = 32
    prompt_len = 256  # 8 full pages
    decode, prefill, server_addr = create_di_scheduler(
        page_size=page_size,
        enable_prefix_caching=True,
        max_forward_steps_tg=10,
    )
    request_queue: queue.Queue = cast(queue.Queue, decode.request_queue)  # type: ignore[type-arg]

    # Request 1 -> run end-to-end so blocks are committed to prefix cache
    ctx1 = create_text_context(
        target_endpoint=server_addr, prompt_len=prompt_len, output_len=5
    )
    request_queue.put(ctx1)
    decode.run_iteration()
    prefill.run_iteration()
    decode.run_iteration()

    # Request 2 -> identical prompt tokens
    ctx2 = create_text_context(
        target_endpoint=server_addr, prompt_len=prompt_len, output_len=5
    )
    request_queue.put(ctx2)
    decode.reserve_memory_and_send_to_prefill()

    # Drain transfer engine metadata then read the PrefillRequest for the second request
    while True:
        msg, _ = prefill.dispatcher.recv_request_nowait()
        if isinstance(msg, PrefillRequest):
            break

    # The leading pages should be marked -1 (prefix-cached on decode)
    dst_ids = msg.dst_block_ids
    num_cached = dst_ids.count(-1)

    assert num_cached > 0, "Expected some blocks to be prefix-cached"
    # All -1 entries must be at front
    assert dst_ids[:num_cached] == [-1] * num_cached
    # Remaining entries must be valid non-negative block indices
    assert all(idx >= 0 for idx in dst_ids[num_cached:])


def test_prefix_caching_prefill_skips_cached_blocks_in_transfer() -> None:
    """Prefill strips cached blocks from the NIXL transfer so only new pages are sent."""
    page_size = 32
    prompt_len = 256  # 8 full pages
    decode, prefill, server_addr = create_di_scheduler(
        page_size=page_size,
        enable_prefix_caching=True,
        max_forward_steps_tg=10,
    )
    request_queue: queue.Queue = cast(queue.Queue, decode.request_queue)  # type: ignore[type-arg]
    num_total_pages = prompt_len // page_size

    # Request 1 -> end-to-end
    ctx1 = create_text_context(
        target_endpoint=server_addr, prompt_len=prompt_len, output_len=5
    )
    request_queue.put(ctx1)
    decode.run_iteration()
    prefill.run_iteration()
    decode.run_iteration()
    prefill.run_iteration()

    # Request 2 -> same prompt
    ctx2 = create_text_context(
        target_endpoint=server_addr, prompt_len=prompt_len, output_len=5
    )
    request_queue.put(ctx2)
    decode.run_iteration()  # sends to prefill
    prefill.run_iteration()  # executes prefill + initiates transfer

    # Transfer should only cover the non-cached pages
    assert len(prefill.active_transfers) == 1
    _, _, transfer_data = next(iter(prefill.active_transfers.values()))
    transferred_pages = len(transfer_data.src_idxs)
    assert transferred_pages < num_total_pages, (
        f"Expected fewer than {num_total_pages} pages transferred, "
        f"got {transferred_pages}"
    )


def test_completed_transfer_is_fully_cleaned_up() -> None:
    """After one request completes end-to-end, no transfer state remains on prefill or decode."""
    decode, prefill, _ = create_default_di_scheduler_and_submit_one_request()

    # Full end-to-end flow
    decode.run_iteration()
    prefill.run_iteration()
    decode.run_iteration()
    prefill.run_iteration()

    # Decode side -> no inflight transfers or pending prefill requests
    assert decode.inflight_transfers == {}
    assert decode.prefill_reqs == {}

    # Prefill side -> no active transfers or inflight send transfers.
    assert prefill.active_transfers == {}
    assert prefill.transfer_engine.inflight_send_transfers == {}

    # Prefill's KV cache should have released all pages
    assert prefill.kv_cache.get_num_used_pages(replica_idx=0) == 0


def test_multiple_requests_all_transfers_cleaned_up() -> None:
    """Multiple concurrent requests all have their transfer state cleaned up."""
    decode, prefill, server_addr = create_di_scheduler()
    request_queue: queue.Queue = cast(queue.Queue, decode.request_queue)  # type: ignore[type-arg]

    # Submit 3 requests
    for _ in range(3):
        ctx = create_text_context(
            target_endpoint=server_addr, prompt_len=100, output_len=5
        )
        request_queue.put(ctx)

    # Run full end-to-end
    decode.run_iteration()
    prefill.run_iteration()
    decode.run_iteration()

    # Both sides need to poll for transfer completion
    for _ in range(5):
        decode.run_iteration()
        prefill.run_iteration()

    assert decode.inflight_transfers == {}
    assert decode.prefill_reqs == {}
    assert prefill.active_transfers == {}
    assert prefill.transfer_engine.inflight_send_transfers == {}
    assert prefill.kv_cache.get_num_used_pages(replica_idx=0) == 0


def test_cancel_request_mid_prefill_produces_no_decode_output() -> None:
    """A request cancelled while prefill is in-flight does not enter the decode batch."""
    decode, prefill, ctx = create_default_di_scheduler_and_submit_one_request()
    req_id = ctx.request_id
    cancel_queue: queue.Queue = cast(queue.Queue, decode.cancel_queue)  # type: ignore[type-arg]
    response_q = cast(queue.Queue, decode.response_queue)  # type: ignore[type-arg]

    # Send to prefill
    decode.run_iteration()

    # Cancel before the decode scheduler sees prefill response
    cancel_queue.put([req_id])

    prefill.run_iteration()

    # Decode processes cancel + in-flight prefill response
    # Request should not enter the decode batch
    decode.run_iteration()

    # Request must not be in decode batch or prefill_reqs
    assert req_id not in decode.prefill_reqs
    assert not decode.batch_constructor.contains(req_id)

    # The final response should be the cancelled sentinel
    all_outputs = []
    while not response_q.empty():
        batch = response_q.get()
        if req_id in batch:
            all_outputs.append(batch[req_id])

    assert len(all_outputs) >= 1
    assert all_outputs[-1].is_done
    assert all_outputs[-1].result is None


def test_cancel_request_before_prefill_executes() -> None:
    """Cancelling before prefill runs sends a CancelRequest and produces no token output."""
    decode, prefill, ctx = create_default_di_scheduler_and_submit_one_request()
    req_id = ctx.request_id
    cancel_queue: queue.Queue = cast(queue.Queue, decode.cancel_queue)  # type: ignore[type-arg]
    response_q = cast(queue.Queue, decode.response_queue)  # type: ignore[type-arg]

    # Send to prefill
    decode.run_iteration()

    # Cancel immediately (before prefill has run)
    cancel_queue.put([req_id])
    decode.run_iteration()

    # Decode should have emitted a cancelled response
    assert not response_q.empty()
    batch = response_q.get()
    assert req_id in batch
    assert batch[req_id].is_done
    assert batch[req_id].result is None

    found_cancel = False
    for _ in range(10):
        try:
            msg, _ = prefill.dispatcher.recv_request_nowait()
            if isinstance(msg, CancelRequest) and msg.id == req_id:
                found_cancel = True
                break
        except queue.Empty:
            break
    assert found_cancel, "Expected a CancelRequest to be sent to prefill"

    # Run prefill...cancel should cause it to discard the result
    prefill.run_iteration()

    # Verify no PrefillResponse was sent back to decode.
    try:
        reply = blocking_recv(decode.dispatcher.recv_reply_nowait, timeout=0.1)
        if isinstance(reply, KVTransferEngineMetadata):
            try:
                reply2 = blocking_recv(
                    decode.dispatcher.recv_reply_nowait, timeout=0.1
                )
                assert not isinstance(reply2, PrefillResponse), (
                    "Did not expect a PrefillResponse after cancellation"
                )
            except queue.Empty:
                pass  # Valid (no PrefillResponse)
    except queue.Empty:
        pass  # Valid


def test_chunked_prefill_completes_across_multiple_iterations() -> None:
    """A prompt exceeding the CE token budget is chunked and completes correctly."""
    target_tokens_per_batch_ce = 128
    page_size = 32
    prompt_len = 512  # 4x the token budget
    decode, prefill, server_addr = create_di_scheduler(
        target_tokens_per_batch_ce=target_tokens_per_batch_ce,
        page_size=page_size,
        enable_chunked_prefill=True,
        max_forward_steps_tg=10,
        max_batch_size=target_tokens_per_batch_ce,
    )
    request_queue: queue.Queue = cast(queue.Queue, decode.request_queue)  # type: ignore[type-arg]
    response_q = cast(queue.Queue, decode.response_queue)  # type: ignore[type-arg]

    ctx = create_text_context(
        target_endpoint=server_addr, prompt_len=prompt_len, output_len=5
    )
    req_id = ctx.request_id
    request_queue.put(ctx)

    # Send to prefill
    decode.run_iteration()

    # Run prefill iterations until it sends a response to decode
    prefill_progress_count = 0
    for _ in range(20):
        result = prefill.run_iteration()
        if result == SchedulerProgress.MADE_PROGRESS:
            prefill_progress_count += 1
        else:
            break

    assert prefill_progress_count > 1, (
        f"Expected multiple prefill iterations for chunked prefill, "
        f"got {prefill_progress_count}"
    )

    # Decode receives the response and runs decode
    decode.run_iteration()

    # Verify tokens
    outputs: list[TextGenerationOutput] = []
    while not response_q.empty():
        batch = response_q.get()
        if req_id in batch:
            result = batch[req_id].result
            if isinstance(result, TextGenerationOutput):
                outputs.append(result)

    assert len(outputs) >= 2, f"Expected at least 2 outputs, got {len(outputs)}"
    # First output is the prefill-generated token
    assert len(outputs[0].tokens) == 1
    # Last output should be done
    assert outputs[-1].is_done


def test_chunked_prefill_with_multiple_requests() -> None:
    """Multiple requests with prompts exceeding the token budget all complete correctly."""
    target_tokens_per_batch_ce = 128
    page_size = 32
    prompt_len = 384  # 3x the budget
    decode, prefill, server_addr = create_di_scheduler(
        target_tokens_per_batch_ce=target_tokens_per_batch_ce,
        page_size=page_size,
        enable_chunked_prefill=True,
        max_forward_steps_tg=10,
        max_batch_size=target_tokens_per_batch_ce,
    )
    request_queue: queue.Queue = cast(queue.Queue, decode.request_queue)  # type: ignore[type-arg]
    response_q = cast(queue.Queue, decode.response_queue)  # type: ignore[type-arg]

    ctx1 = create_text_context(
        target_endpoint=server_addr, prompt_len=prompt_len, output_len=5
    )
    ctx2 = create_text_context(
        target_endpoint=server_addr, prompt_len=prompt_len, output_len=5
    )
    req_id1 = ctx1.request_id
    req_id2 = ctx2.request_id
    request_queue.put(ctx1)
    request_queue.put(ctx2)

    # Send to prefill
    decode.run_iteration()

    # Run prefill until no progress
    for _ in range(30):
        if prefill.run_iteration() == SchedulerProgress.NO_PROGRESS:
            break

    decode.run_iteration()

    # Collect outputs per request
    req1_tokens: list[int] = []
    req2_tokens: list[int] = []
    req1_done = False
    req2_done = False
    while not response_q.empty():
        batch = response_q.get()
        for rid, sch_result in batch.items():
            result = sch_result.result
            if isinstance(result, TextGenerationOutput):
                if rid == req_id1:
                    req1_tokens.extend(result.tokens)
                    req1_done = req1_done or sch_result.is_done
                elif rid == req_id2:
                    req2_tokens.extend(result.tokens)
                    req2_done = req2_done or sch_result.is_done

    # Both requests should have generated tokens and completed
    assert len(req1_tokens) > 0, "Request 1 produced no tokens"
    assert len(req2_tokens) > 0, "Request 2 produced no tokens"
    assert req1_done, "Request 1 did not complete"
    assert req2_done, "Request 2 did not complete"


def test_kv_backpressure_stops_sending_to_prefill() -> None:
    """When decode KV utilization >= 90%, new requests stay in pending_reqs."""
    # 10 blocks, page_size=128. A 1152-token prompt needs 9 pages
    # After the first request: 9/10 = 90% utilization, which is NOT < 0.9
    # Therefore, backpressure gate blocks the second request
    decode, _, server_addr = create_di_scheduler(num_blocks=10, page_size=128)
    request_queue: queue.Queue = cast(queue.Queue, decode.request_queue)  # type: ignore[type-arg]

    ctx1 = create_text_context(
        target_endpoint=server_addr, prompt_len=1152, output_len=5
    )
    ctx2 = create_text_context(
        target_endpoint=server_addr, prompt_len=1152, output_len=5
    )
    request_queue.put(ctx1)
    request_queue.put(ctx2)

    decode.reserve_memory_and_send_to_prefill()

    assert len(decode.prefill_reqs) == 1, "Only one request should be sent"
    assert len(decode.pending_reqs) == 1, "Second request should be held back"


def test_decode_kv_blocks_freed_after_request_completes() -> None:
    """Decode-side KV pages return to zero after a request finishes end-to-end."""
    decode, prefill, _ = create_default_di_scheduler_and_submit_one_request()

    assert decode.kv_cache.get_num_used_pages(replica_idx=0) == 0

    # Send to prefill, allocates KV blocks on decode
    decode.reserve_memory_and_send_to_prefill()
    pages_allocated = decode.kv_cache.get_num_used_pages(replica_idx=0)
    assert pages_allocated > 0, (
        "Expected KV pages allocated after sending to prefill"
    )

    # Complete full lifecycle
    prefill.run_iteration()
    decode.run_iteration()

    assert decode.kv_cache.get_num_used_pages(replica_idx=0) == 0, (
        "KV pages not freed after request completed"
    )


# ---------------------------------------------------------------------------
# Two-phase prefill tests (overlap scheduling + DI)
# ---------------------------------------------------------------------------


def test_overlap_prefill_two_phase_execute_sends_real_token() -> None:
    """PrefillScheduler with OverlapTextGenerationPipeline must:

    1. Not send PrefillResponse in iteration 1 (real token deferred by one batch).
    2. Keep the scheduler alive in iteration 2 via has_pending_outputs() guard
       even when the incoming batch is empty (last-batch flush).
    3. Send PrefillResponse with the real generated token (never FUTURE_TOKEN).
    """
    decode, prefill, server_addr = create_di_scheduler(overlap_prefill=True)
    assert isinstance(prefill.pipeline, FakeOverlapPipeline)

    ctx = create_text_context(
        target_endpoint=server_addr, prompt_len=100, output_len=5
    )
    request_queue: queue.Queue = cast(queue.Queue, decode.request_queue)  # type: ignore[type-arg]
    request_queue.put(ctx)

    # Iteration 1: decode sends PrefillRequest; prefill launches CE batch.
    # No PrefillResponse yet — real token deferred by one batch.
    decode.run_iteration()
    prefill.run_iteration()

    # Pipeline must report pending outputs (drain not yet run).
    assert prefill.pipeline.has_pending_outputs()

    # Iteration 2: empty batch flush. has_pending_outputs() guard keeps the
    # scheduler alive; real token is already in context.tokens[-1]; PrefillResponse sent.
    prefill.run_iteration()

    # Pipeline's pending outputs should now be drained.
    assert not prefill.pipeline.has_pending_outputs()

    # PrefillResponse must have arrived with the real token.
    metadata = decode.dispatcher.recv_reply_nowait()
    assert isinstance(metadata, KVTransferEngineMetadata)
    response = decode.dispatcher.recv_reply_nowait()
    assert isinstance(response, PrefillResponse)
    assert response.id == ctx.request_id
    assert response.generated_token_id == 99  # match against test sentinel


def test_overlap_prefill_multiple_requests_deferred_and_resolved() -> None:
    """Multiple CE completions in a single batch must all be deferred and
    resolved correctly across the two-phase boundary.

    Iteration 1: two requests complete CE together — both deferred.
    Iteration 2: empty flush — both resolved, both PrefillResponses carry the
    real generated token (never FUTURE_TOKEN).
    """
    decode, prefill, server_addr = create_di_scheduler(overlap_prefill=True)
    assert isinstance(prefill.pipeline, FakeOverlapPipeline)

    request_queue: queue.Queue = cast(queue.Queue, decode.request_queue)  # type: ignore[type-arg]
    ctx1 = create_text_context(
        target_endpoint=server_addr, prompt_len=100, output_len=5
    )
    ctx2 = create_text_context(
        target_endpoint=server_addr, prompt_len=100, output_len=5
    )
    request_queue.put(ctx1)
    request_queue.put(ctx2)

    # Iteration 1: decode sends both requests; prefill runs CE for both.
    # Both deferred — no PrefillResponses yet.
    decode.run_iteration()
    prefill.run_iteration()
    assert prefill.pipeline.has_pending_outputs()

    # Iteration 2: empty batch flush resolves both deferred requests.
    prefill.run_iteration()
    assert not prefill.pipeline.has_pending_outputs()

    # Both PrefillResponses must arrive with real tokens (not FUTURE_TOKEN).
    # The engine-registration handshake (one KVTransferEngineMetadata) arrived
    # during iter 1; drain it once before checking the per-request responses.
    handshake = decode.dispatcher.recv_reply_nowait()
    assert isinstance(handshake, KVTransferEngineMetadata)

    received_ids = set()
    for _ in range(2):
        response = decode.dispatcher.recv_reply_nowait()
        assert isinstance(response, PrefillResponse)
        assert response.generated_token_id != FUTURE_TOKEN
        received_ids.add(response.id)

    assert received_ids == {ctx1.request_id, ctx2.request_id}


def test_overlap_prefill_staggered_requests_across_batches() -> None:
    """Deferred and newly arriving requests coexist correctly across batches.

    Iteration 1: req1 completes CE — deferred.
    Iteration 2: req2 arrives and completes CE — req1 resolved, req2 deferred.
    Iteration 3: empty flush — req2 resolved.
    """
    decode, prefill, server_addr = create_di_scheduler(overlap_prefill=True)
    assert isinstance(prefill.pipeline, FakeOverlapPipeline)

    request_queue: queue.Queue = cast(queue.Queue, decode.request_queue)  # type: ignore[type-arg]
    ctx1 = create_text_context(
        target_endpoint=server_addr, prompt_len=100, output_len=5
    )
    ctx2 = create_text_context(
        target_endpoint=server_addr, prompt_len=100, output_len=5
    )

    # Iteration 1: only req1 in flight.
    request_queue.put(ctx1)
    decode.run_iteration()
    prefill.run_iteration()
    assert prefill.pipeline.has_pending_outputs()

    # Iteration 2: req2 arrives; req1 resolves, req2 defers.
    request_queue.put(ctx2)
    decode.run_iteration()
    prefill.run_iteration()
    assert prefill.pipeline.has_pending_outputs()

    # req1's PrefillResponse must be available now.
    # Note: decode.run_iteration() in iter 2 already consumed the
    # engine-registration KVTransferEngineMetadata, so only PrefillResponse
    # remains in the queue.
    response1 = decode.dispatcher.recv_reply_nowait()
    assert isinstance(response1, PrefillResponse)
    assert response1.id == ctx1.request_id
    assert response1.generated_token_id != FUTURE_TOKEN

    # Iteration 3: empty flush resolves req2.
    prefill.run_iteration()
    assert not prefill.pipeline.has_pending_outputs()

    response2 = decode.dispatcher.recv_reply_nowait()
    assert isinstance(response2, PrefillResponse)
    assert response2.id == ctx2.request_id
    assert response2.generated_token_id != FUTURE_TOKEN


def test_overlap_prefill_cancel_between_defer_and_resolve() -> None:
    """A request cancelled after deferral but before resolution must not
    send FUTURE_TOKEN to decode and must not crash the scheduler.

    The cancel arrives after iteration 1 (request is in _pending_first_token)
    but before iteration 2 (the flush that would call
    initiate_transfer_and_send_reply). The outstanding_cancelled_requests
    guard in initiate_transfer_and_send_reply silently discards the result.
    """
    decode, prefill, server_addr = create_di_scheduler(overlap_prefill=True)
    assert isinstance(prefill.pipeline, FakeOverlapPipeline)

    request_queue: queue.Queue = cast(queue.Queue, decode.request_queue)  # type: ignore[type-arg]
    ctx = create_text_context(
        target_endpoint=server_addr, prompt_len=100, output_len=5
    )
    request_queue.put(ctx)

    # Iteration 1: request completes CE and is deferred.
    decode.run_iteration()
    prefill.run_iteration()
    assert prefill.pipeline.has_pending_outputs()

    # Cancel the request while it sits in _pending_first_token.
    prefill.handle_cancel_request(CancelRequest(id=ctx.request_id))

    # Iteration 2: flush runs; cancel guard suppresses the transfer.
    # Must not raise and must not send any reply to decode.
    prefill.run_iteration()
    assert not prefill.pipeline.has_pending_outputs()

    # Only the engine-registration handshake may arrive; no PrefillResponse
    # should be sent for the cancelled request.
    while True:
        try:
            msg = decode.dispatcher.recv_reply_nowait()
            assert not isinstance(msg, PrefillResponse), (
                "Cancelled request must not produce a PrefillResponse"
            )
        except queue.Empty:
            break
