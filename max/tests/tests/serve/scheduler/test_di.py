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
from dataclasses import dataclass
from typing import Any, TypeVar
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
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
from max.pipelines.lib.config.speculative_config import SpeculativeConfig
from max.pipelines.lib.pipeline_variants.utils import (
    update_spec_decode_context_and_prepare_responses,
)
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
    DecodeDispatcherClient,
    PrefillDispatcherServer,
    ReplyType,
    RequestType,
)
from max.serve.scheduler.prefill_scheduler import (
    PrefillScheduler,
    load_prefill_scheduler,
)
from max.serve.worker_interface.zmq_queue import (
    ClientIdentity,
    generate_zmq_ipc_path,
)
from tests.serve.scheduler.common import (
    FakeOverlapPipeline,
    FakeTokenGeneratorPipeline,
    PagedKVCacheManager,
    create_kv_cache,
)

TIMEOUT = 1.0
_T = TypeVar("_T")


@dataclass
class DIQueues:
    request_queue: queue.Queue[TextContext]
    response_queue: queue.Queue[
        dict[RequestID, SchedulerResult[TextGenerationOutput]]
    ]
    cancel_queue: queue.Queue[list[RequestID]]


def blocking_recv(fn: Callable[[], _T], timeout: float = TIMEOUT) -> _T:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            return fn()
        except queue.Empty:
            time.sleep(0.001)
    raise queue.Empty()


class BasicDispatcherServer(PrefillDispatcherServer):
    def __init__(self, bind_addr: str):
        self.bind_addr = bind_addr
        super().__init__(bind_addr=bind_addr)

    def recv_request_nowait(self) -> tuple[RequestType, ClientIdentity]:
        return blocking_recv(super().recv_request_nowait)


class BasicDispatcherClient(DecodeDispatcherClient):
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
    overlap_decode: bool = False,
    spec_decode_prefill: bool = False,
    num_speculative_tokens: int = 2,
    prefill_is_mla: bool = False,
    prefill_tp_per_replica: int = 1,
    prefill_dp: int | None = None,
    decode_dp: int | None = None,
) -> tuple[DecodeScheduler, PrefillScheduler, str, DIQueues]:
    """Creates a DecodeScheduler and PrefillScheduler pair for testing.

    Args:
        overlap_prefill: Use FakeOverlapPipeline in overlap mode (one-batch
            output lag). Combine with spec_decode_prefill for the overlap +
            spec decode path (num_speculative_tokens=1).
        overlap_decode: Use FakeOverlapPipeline on decode.
        spec_decode_prefill: Populate draft tokens during prefill CE.
            Combined with overlap_prefill=False, opts out of the two-phase
            path (disable_overlap=True), matching num_speculative_tokens > 1.
        num_speculative_tokens: Number of draft tokens per step when
            spec_decode_prefill is True.
        prefill_is_mla: When True, the prefill KV cache uses ``is_mla=True``
            (num_kv_heads=1 replicated across TP) so PrefillScheduler flattens
            the engine shape from [dp][tp] to [dp*tp][1].
        prefill_tp_per_replica: Number of simulated TP shards per prefill
            model replica.
        prefill_dp: Override ``dp`` for the prefill worker only (defaults to
            ``dp``). Lets the test set prefill and decode to asymmetric
            topologies while keeping the same total device count.
        decode_dp: Override ``dp`` for the decode worker only (defaults to
            ``dp``).
    """

    effective_prefill_dp = prefill_dp if prefill_dp is not None else dp
    effective_decode_dp = decode_dp if decode_dp is not None else dp

    def _create_prefill_kv_cache() -> PagedKVCacheManager:
        return create_kv_cache(
            num_blocks=num_blocks,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            page_size=page_size,
            enable_prefix_caching=enable_prefix_caching,
            kv_connector=kv_connector,
            dp=effective_prefill_dp,
            device=device,
            is_mla=prefill_is_mla,
            tp_per_replica=prefill_tp_per_replica,
        )

    def _create_decode_kv_cache() -> PagedKVCacheManager:
        return create_kv_cache(
            num_blocks=num_blocks,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            page_size=page_size,
            enable_prefix_caching=enable_prefix_caching,
            kv_connector=kv_connector,
            dp=effective_decode_dp,
            device=device,
            # Decode must use the same KV layout as prefill so bytes_per_page
            # matches at connect() time. For heterogeneous MLA DI the model
            # is MLA on both sides; decode's TP=1 naturally so no flatten.
            is_mla=prefill_is_mla,
        )

    def _make_scheduler_config(
        dp_value: int,
    ) -> TokenGenerationSchedulerConfig:
        return TokenGenerationSchedulerConfig(
            max_batch_size=max_batch_size,
            max_forward_steps_tg=max_forward_steps_tg,
            target_tokens_per_batch_ce=target_tokens_per_batch_ce,
            max_seq_len=max_seq_len,
            enable_chunked_prefill=enable_chunked_prefill,
            enable_in_flight_batching=enable_in_flight_batching,
            data_parallel_degree=dp_value,
            num_speculative_tokens=num_speculative_tokens
            if spec_decode_prefill
            else 0,
        )

    # For heterogeneous-MLA DI (prefill DP=1/TP>1, decode DP=N/TP=1), the
    # engine auto-detects the shape mismatch via replicate_kv_across_tp on
    # both sides' metadata at connect() time — no config flag needed.
    prefill_scheduler_config = _make_scheduler_config(effective_prefill_dp)
    decode_scheduler_config = _make_scheduler_config(effective_decode_dp)

    # Use queue.Queue to simulate the ZMQ queues.
    request_queue: queue.Queue[TextContext] = queue.Queue()
    response_queue: queue.Queue[
        dict[RequestID, SchedulerResult[TextGenerationOutput]]
    ] = queue.Queue()
    cancel_queue: queue.Queue[list[RequestID]] = queue.Queue()

    kv_cache_prefill = _create_prefill_kv_cache()
    kv_cache_decode = _create_decode_kv_cache()
    server_addr = generate_zmq_ipc_path()
    client_addr = generate_zmq_ipc_path()
    dispatcher_server = BasicDispatcherServer(bind_addr=server_addr)
    dispatcher_client = BasicDispatcherClient(bind_addr=client_addr)

    decode_pipeline = (
        FakeOverlapPipeline(
            kv_cache_decode, max_seq_len=max_seq_len, start_token_id=42
        )
        if overlap_decode
        else FakeTokenGeneratorPipeline(
            kv_cache_decode, max_seq_len=max_seq_len, start_token_id=42
        )
    )

    decode_scheduler = DecodeScheduler(
        pipeline=decode_pipeline,
        scheduler_config=decode_scheduler_config,
        kv_cache=kv_cache_decode,
        request_queue=request_queue,
        response_queue=response_queue,
        cancel_queue=cancel_queue,
        dispatcher=dispatcher_client,
    )

    prefill_pipeline: FakeTokenGeneratorPipeline
    if spec_decode_prefill or overlap_prefill:
        prefill_pipeline = FakeOverlapPipeline(
            kv_cache_prefill,
            max_seq_len=max_seq_len,
            start_token_id=99,
            num_speculative_tokens=(
                num_speculative_tokens if spec_decode_prefill else 0
            ),
            disable_overlap=not overlap_prefill,
        )
    else:
        prefill_pipeline = FakeTokenGeneratorPipeline(
            kv_cache_prefill, max_seq_len=max_seq_len, start_token_id=99
        )

    prefill_scheduler = PrefillScheduler(
        pipeline=prefill_pipeline,
        scheduler_config=prefill_scheduler_config,
        kv_cache=kv_cache_prefill,
        dispatcher=dispatcher_server,
    )

    return (
        decode_scheduler,
        prefill_scheduler,
        server_addr,
        DIQueues(
            request_queue=request_queue,
            response_queue=response_queue,
            cancel_queue=cancel_queue,
        ),
    )


def create_default_di_scheduler_and_submit_one_request() -> tuple[
    DecodeScheduler, PrefillScheduler, DIQueues, TextContext
]:
    decode, prefill, server_addr, q = create_di_scheduler()
    ctx = create_text_context(
        target_endpoint=server_addr, prompt_len=100, output_len=5
    )
    q.request_queue.put(ctx)
    return decode, prefill, q, ctx


def test_decode_sends_request_to_prefill() -> None:
    decode, prefill, _q, _ = (
        create_default_di_scheduler_and_submit_one_request()
    )

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
    decode, prefill, _q, ctx = (
        create_default_di_scheduler_and_submit_one_request()
    )

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
    decode, prefill, q, ctx = (
        create_default_di_scheduler_and_submit_one_request()
    )
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

    # Check that the first token is 99
    output1 = q.response_queue.get()
    assert len(output1) == 1
    sch_output1 = output1[req_id]
    assert not sch_output1.is_done
    single_token = sch_output1.result
    assert isinstance(single_token, TextGenerationOutput)
    assert single_token.request_id == req_id
    assert single_token.tokens == [99]

    # Check that the rest of the tokens are 42, 43, 44, 45
    output2 = q.response_queue.get()
    assert len(output2) == 1
    sch_output2 = output2[req_id]
    assert sch_output2.is_done
    rest_of_tokens = sch_output2.result
    assert isinstance(rest_of_tokens, TextGenerationOutput)
    assert rest_of_tokens.request_id == req_id
    assert rest_of_tokens.tokens == [42, 43, 44, 45]


def test_heterogeneous_mla_prefill_tp2_to_decode_dp2_end_to_end() -> None:
    """
    Both engines are constructed with their natural ``[dp][tp]`` layout.
    At connect(), ``resolve_peer_view`` picks ``flatten_local=True`` on
    the prefill side so the KVTransferEngine's effective view matches
    the decode side's DP=2/TP=1 shape. This test drives a full request
    through the pair and asserts the KV transfer completes and tokens
    land on the decode side.
    """
    decode, prefill, server_addr, q = create_di_scheduler(
        prefill_is_mla=True,
        prefill_tp_per_replica=2,
        prefill_dp=1,
        decode_dp=2,
    )

    # Prefill engine keeps its natural DP=1/TP=2 shape and advertises
    # replicate_kv_across_tp so peers can flatten at connect() time.
    assert prefill.transfer_engine.dp == 1
    assert prefill.transfer_engine.tp == 2
    assert prefill.transfer_engine.replicate_kv_across_tp is True
    # Decode engine is natural DP=2/TP=1; replicate is coerced off by TP=1.
    assert decode.transfer_engine.dp == 2
    assert decode.transfer_engine.tp == 1
    assert decode.transfer_engine.replicate_kv_across_tp is False

    ctx = create_text_context(
        target_endpoint=server_addr, prompt_len=100, output_len=5
    )
    q.request_queue.put(ctx)
    req_id = ctx.request_id

    # decode → prefill (exchanges transfer-engine metadata + request).
    decode.run_iteration()
    # prefill executes CE, initiates NIXL transfer to decode, and replies.
    prefill.run_iteration()
    # decode picks up the transferred KV + the first generated token, then
    # runs token generation.
    decode.run_iteration()

    # First response: prefill-generated token.
    output1 = q.response_queue.get()
    sch_output1 = output1[req_id]
    assert not sch_output1.is_done
    single_token = sch_output1.result
    assert isinstance(single_token, TextGenerationOutput)
    assert single_token.tokens == [99]

    # Second response: decode-generated tokens.
    output2 = q.response_queue.get()
    sch_output2 = output2[req_id]
    assert sch_output2.is_done
    rest_of_tokens = sch_output2.result
    assert isinstance(rest_of_tokens, TextGenerationOutput)
    assert rest_of_tokens.tokens == [42, 43, 44, 45]

    # One more prefill iteration drains cleanup_active_transfers — verify the
    # flattened-engine send transfer was released symmetrically.
    prefill.run_iteration()
    assert prefill.active_transfers == {}
    assert prefill.transfer_engine.inflight_send_transfers == {}


def test_heterogeneous_mla_prefill_src_replica_maps_to_first_shard() -> None:
    """
    With prefill DP=1/TP=2 (MLA), the PrefillScheduler should translate the
    model src_replica_idx=0 (only one model replica) to engine replica idx 0
    (the first of the 2 flattened TP shards). We check this by inspecting
    the TransferReqData produced by initiating a transfer.
    """
    decode, prefill, server_addr, q = create_di_scheduler(
        prefill_is_mla=True,
        prefill_tp_per_replica=2,
        prefill_dp=1,
        decode_dp=2,
    )

    ctx = create_text_context(
        target_endpoint=server_addr, prompt_len=100, output_len=5
    )
    q.request_queue.put(ctx)

    # Drive decode → prefill message exchange.
    decode.run_iteration()
    prefill.run_iteration()

    # After prefill.run_iteration(), the transfer request is recorded in
    # active_transfers. Grab it and assert src_replica_idx was translated.
    assert len(prefill.active_transfers) == 1
    active = next(iter(prefill.active_transfers.values()))
    # Scheduler-level (model) replica is always 0 for DP=1 prefill.
    assert active.replica_idx == 0
    # Engine-level src_replica_idx is the RR-selected flattened index.
    # For the first request (counter=0 → offset 0), it's the first TP shard.
    assert active.transfer.src_replica_idx == 0


def test_di_with_dp2_requests_distributed_to_different_replicas() -> None:
    """Test that with DP=2, requests are distributed to different replicas."""
    decode, prefill, server_addr, q = create_di_scheduler(dp=2)

    # Create and submit two requests
    ctx1 = create_text_context(target_endpoint=server_addr, prompt_len=1111)
    ctx2 = create_text_context(target_endpoint=server_addr, prompt_len=1111)
    ctx3 = create_text_context(target_endpoint=server_addr, prompt_len=1111)
    q.request_queue.put(ctx1)
    q.request_queue.put(ctx2)
    q.request_queue.put(ctx3)

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
    decode, prefill, server_addr, q = create_di_scheduler(dp=2)

    # Create and submit two requests
    ctx1 = create_text_context(
        target_endpoint=server_addr, prompt_len=100, output_len=5
    )
    ctx2 = create_text_context(
        target_endpoint=server_addr, prompt_len=100, output_len=5
    )
    req_id1 = ctx1.request_id
    req_id2 = ctx2.request_id
    q.request_queue.put(ctx1)
    q.request_queue.put(ctx2)

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
    for output in q.response_queue.queue:
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
    decode, prefill, server_addr, q = create_di_scheduler(
        max_forward_steps_tg=1
    )
    ctx = create_text_context(
        target_endpoint=server_addr, prompt_len=100, output_len=10
    )
    req_id = ctx.request_id
    q.request_queue.put(ctx)

    # Send to prefill, execute prefill, then run decode
    # (streams prefill token + generates 1 decode token).
    decode.run_iteration()
    prefill.run_iteration()
    decode.run_iteration()

    # Drain both responses (prefill token, first decode token).
    assert q.response_queue.qsize() == 2
    q.response_queue.get()
    q.response_queue.get()

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

    output = q.response_queue.get()
    assert req_id in output
    assert stale_id not in output


def test_overlap_di_has_pending_outputs_prevents_no_progress() -> None:
    """Verify run_iteration() returns MADE_PROGRESS when the batch is empty,
    but the overlap pipeline reports pending outputs."""
    decode, _prefill, _server_addr, _q = create_di_scheduler()

    # Simulate overlap pipeline behavior without a real OverlapPipeline.
    mock_pipeline = MagicMock(spec=OverlapTextGenerationPipeline)
    mock_pipeline.has_pending_outputs.return_value = True
    mock_pipeline.execute.return_value = {}
    mock_pipeline.spec_decode_metrics.return_value = None
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
    decode, prefill, server_addr, q = create_di_scheduler(dp=2)

    # Submit 2 requests
    ctx1 = create_text_context(
        target_endpoint=server_addr, prompt_len=100, output_len=5
    )
    ctx2 = create_text_context(
        target_endpoint=server_addr, prompt_len=100, output_len=5
    )
    q.request_queue.put(ctx1)
    q.request_queue.put(ctx2)

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
    decode, _, q, ctx = create_default_di_scheduler_and_submit_one_request()
    req_id = ctx.request_id

    # Record baseline KV usage.
    pages_before = decode.kv_cache.get_num_used_pages(replica_idx=0)

    # Send to prefill -> allocates KV blocks on decode
    decode.run_iteration()

    pages_after_send = decode.kv_cache.get_num_used_pages(replica_idx=0)
    assert pages_after_send > pages_before, (
        "Expected KV blocks to be allocated after sending to prefill"
    )

    # Cancel before prefill runs
    q.cancel_queue.put([req_id])
    decode.run_iteration()

    # Drain the cancelled response
    assert not q.response_queue.empty()
    batch = q.response_queue.get()
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
    decode, prefill, q, ctx = (
        create_default_di_scheduler_and_submit_one_request()
    )
    req_id = ctx.request_id

    # Send to prefill
    decode.run_iteration()

    # Cancel before prefill runs
    q.cancel_queue.put([req_id])
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
    decode, prefill, server_addr, q = create_di_scheduler(
        page_size=page_size,
        enable_prefix_caching=True,
        max_forward_steps_tg=10,
    )

    # Request 1 -> run end-to-end so blocks are committed to prefix cache
    ctx1 = create_text_context(
        target_endpoint=server_addr, prompt_len=prompt_len, output_len=5
    )
    q.request_queue.put(ctx1)
    decode.run_iteration()
    prefill.run_iteration()
    decode.run_iteration()

    # Request 2 -> identical prompt tokens
    ctx2 = create_text_context(
        target_endpoint=server_addr, prompt_len=prompt_len, output_len=5
    )
    q.request_queue.put(ctx2)
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
    decode, prefill, server_addr, q = create_di_scheduler(
        page_size=page_size,
        enable_prefix_caching=True,
        max_forward_steps_tg=10,
    )
    num_total_pages = prompt_len // page_size

    # Request 1 -> end-to-end
    ctx1 = create_text_context(
        target_endpoint=server_addr, prompt_len=prompt_len, output_len=5
    )
    q.request_queue.put(ctx1)
    decode.run_iteration()
    prefill.run_iteration()
    decode.run_iteration()
    prefill.run_iteration()

    # Request 2 -> same prompt
    ctx2 = create_text_context(
        target_endpoint=server_addr, prompt_len=prompt_len, output_len=5
    )
    q.request_queue.put(ctx2)
    decode.run_iteration()  # sends to prefill
    prefill.run_iteration()  # executes prefill + initiates transfer

    # Transfer should only cover the non-cached pages
    assert len(prefill.active_transfers) == 1
    active = next(iter(prefill.active_transfers.values()))
    transferred_pages = len(active.transfer.src_idxs)
    assert transferred_pages < num_total_pages, (
        f"Expected fewer than {num_total_pages} pages transferred, "
        f"got {transferred_pages}"
    )


def test_completed_request_cleans_up_all_state() -> None:
    """After one request completes end-to-end, all transfer state and KV pages
    are released on both decode and prefill sides."""
    decode, prefill, _q, _ = (
        create_default_di_scheduler_and_submit_one_request()
    )

    # Initially no KV pages allocated on decode
    assert decode.kv_cache.get_num_used_pages(replica_idx=0) == 0

    # Send to prefill -> allocates decode KV blocks
    decode.run_iteration()
    assert decode.kv_cache.get_num_used_pages(replica_idx=0) > 0, (
        "Expected KV pages allocated after sending to prefill"
    )

    # Complete full lifecycle
    prefill.run_iteration()
    decode.run_iteration()
    prefill.run_iteration()

    # Transfer state fully cleaned up on both sides
    assert decode.inflight_transfers == {}
    assert decode.prefill_reqs == {}
    assert prefill.active_transfers == {}
    assert prefill.transfer_engine.inflight_send_transfers == {}

    # Both KV caches released
    assert decode.kv_cache.get_num_used_pages(replica_idx=0) == 0, (
        "Decode KV pages not freed after request completed"
    )
    assert prefill.kv_cache.get_num_used_pages(replica_idx=0) == 0


def test_multiple_requests_all_transfers_cleaned_up() -> None:
    """Multiple concurrent requests all have their transfer state cleaned up."""
    decode, prefill, server_addr, q = create_di_scheduler()

    # Submit 3 requests
    for _ in range(3):
        ctx = create_text_context(
            target_endpoint=server_addr, prompt_len=100, output_len=5
        )
        q.request_queue.put(ctx)

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
    decode, prefill, q, ctx = (
        create_default_di_scheduler_and_submit_one_request()
    )
    req_id = ctx.request_id

    # Send to prefill
    decode.run_iteration()

    # Cancel before the decode scheduler sees prefill response
    q.cancel_queue.put([req_id])

    prefill.run_iteration()

    # Decode processes cancel + in-flight prefill response
    # Request should not enter the decode batch
    decode.run_iteration()

    # Request must not be in decode batch or prefill_reqs
    assert req_id not in decode.prefill_reqs
    assert not decode.batch_constructor.contains(req_id)

    # The final response should be the cancelled sentinel
    all_outputs = []
    while not q.response_queue.empty():
        batch = q.response_queue.get()
        if req_id in batch:
            all_outputs.append(batch[req_id])

    assert len(all_outputs) >= 1
    assert all_outputs[-1].is_done
    assert all_outputs[-1].result is None


def test_cancel_request_before_prefill_executes() -> None:
    """Cancelling before prefill runs sends a CancelRequest and produces no token output."""
    decode, prefill, q, ctx = (
        create_default_di_scheduler_and_submit_one_request()
    )
    req_id = ctx.request_id

    # Send to prefill
    decode.run_iteration()

    # Cancel immediately (before prefill has run)
    q.cancel_queue.put([req_id])
    decode.run_iteration()

    # Decode should have emitted a cancelled response
    assert not q.response_queue.empty()
    batch = q.response_queue.get()
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
    decode, prefill, server_addr, q = create_di_scheduler(
        target_tokens_per_batch_ce=target_tokens_per_batch_ce,
        page_size=page_size,
        enable_chunked_prefill=True,
        max_forward_steps_tg=10,
        max_batch_size=target_tokens_per_batch_ce,
    )

    ctx = create_text_context(
        target_endpoint=server_addr, prompt_len=prompt_len, output_len=5
    )
    req_id = ctx.request_id
    q.request_queue.put(ctx)

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
    while not q.response_queue.empty():
        batch = q.response_queue.get()
        if req_id in batch:
            output = batch[req_id].result
            if isinstance(output, TextGenerationOutput):
                outputs.append(output)

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
    decode, prefill, server_addr, q = create_di_scheduler(
        target_tokens_per_batch_ce=target_tokens_per_batch_ce,
        page_size=page_size,
        enable_chunked_prefill=True,
        max_forward_steps_tg=10,
        max_batch_size=target_tokens_per_batch_ce,
    )

    ctx1 = create_text_context(
        target_endpoint=server_addr, prompt_len=prompt_len, output_len=5
    )
    ctx2 = create_text_context(
        target_endpoint=server_addr, prompt_len=prompt_len, output_len=5
    )
    req_id1 = ctx1.request_id
    req_id2 = ctx2.request_id
    q.request_queue.put(ctx1)
    q.request_queue.put(ctx2)

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
    while not q.response_queue.empty():
        batch = q.response_queue.get()
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
    decode, _, server_addr, q = create_di_scheduler(
        num_blocks=10, page_size=128
    )

    ctx1 = create_text_context(
        target_endpoint=server_addr, prompt_len=1152, output_len=5
    )
    ctx2 = create_text_context(
        target_endpoint=server_addr, prompt_len=1152, output_len=5
    )
    q.request_queue.put(ctx1)
    q.request_queue.put(ctx2)

    decode.reserve_memory_and_send_to_prefill()

    assert len(decode.prefill_reqs) == 1, "Only one request should be sent"
    assert len(decode.pending_reqs) == 1, "Second request should be held back"


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
    decode, prefill, server_addr, q = create_di_scheduler(overlap_prefill=True)
    assert isinstance(prefill.pipeline, FakeOverlapPipeline)

    ctx = create_text_context(
        target_endpoint=server_addr, prompt_len=100, output_len=5
    )
    q.request_queue.put(ctx)

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
    decode, prefill, server_addr, q = create_di_scheduler(overlap_prefill=True)
    assert isinstance(prefill.pipeline, FakeOverlapPipeline)

    ctx1 = create_text_context(
        target_endpoint=server_addr, prompt_len=100, output_len=5
    )
    ctx2 = create_text_context(
        target_endpoint=server_addr, prompt_len=100, output_len=5
    )
    q.request_queue.put(ctx1)
    q.request_queue.put(ctx2)

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
    decode, prefill, server_addr, q = create_di_scheduler(overlap_prefill=True)
    assert isinstance(prefill.pipeline, FakeOverlapPipeline)

    ctx1 = create_text_context(
        target_endpoint=server_addr, prompt_len=100, output_len=5
    )
    ctx2 = create_text_context(
        target_endpoint=server_addr, prompt_len=100, output_len=5
    )

    # Iteration 1: only req1 in flight.
    q.request_queue.put(ctx1)
    decode.run_iteration()
    prefill.run_iteration()
    assert prefill.pipeline.has_pending_outputs()

    # Iteration 2: req2 arrives; req1 resolves, req2 defers.
    q.request_queue.put(ctx2)
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
    decode, prefill, server_addr, q = create_di_scheduler(overlap_prefill=True)
    assert isinstance(prefill.pipeline, FakeOverlapPipeline)

    ctx = create_text_context(
        target_endpoint=server_addr, prompt_len=100, output_len=5
    )
    q.request_queue.put(ctx)

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


# E2E tests for DI with overlap scheduling on decode, prefill, or both


def test_overlap_di_prefill_lag_e2e_token_reaches_frontend() -> None:
    """With overlap on prefill, the deferred first token correctly reaches
    the frontend response queue through the full decode -> prefill -> decode path."""
    decode, prefill, server_addr, q = create_di_scheduler(overlap_prefill=True)
    ctx = create_text_context(
        target_endpoint=server_addr, prompt_len=100, output_len=5
    )
    req_id = ctx.request_id
    q.request_queue.put(ctx)

    # Decode sends to prefill, prefill CE (defers), flush (resolves + sends)
    decode.run_iteration()
    prefill.run_iteration()
    prefill.run_iteration()

    decode.run_iteration()

    assert not q.response_queue.empty()
    first_batch = q.response_queue.get()
    assert req_id in first_batch
    result = first_batch[req_id].result
    assert isinstance(result, TextGenerationOutput)
    assert result.tokens == [99]  # match prefill start_token_id=99
    assert FUTURE_TOKEN not in result.tokens


def test_overlap_di_e2e_correct_token_streaming_order() -> None:
    """Tokens stream in exact order with overlap on decode alone and on both
    sides simultaneously, the 1 batch lag must not drop or reorder tokens."""
    for overlap_prefill in (False, True):
        decode, prefill, server_addr, q = create_di_scheduler(
            overlap_prefill=overlap_prefill,
            overlap_decode=True,
            max_forward_steps_tg=1,
        )
        ctx = create_text_context(
            target_endpoint=server_addr, prompt_len=100, output_len=5
        )
        req_id = ctx.request_id
        q.request_queue.put(ctx)

        decode.run_iteration()
        prefill.run_iteration()
        if overlap_prefill:
            prefill.run_iteration()  # flush deferred prefill token
        for _ in range(8):
            decode.run_iteration()

        all_tokens: list[int] = []
        is_done = False
        while not q.response_queue.empty():
            batch = q.response_queue.get()
            if req_id in batch:
                sch_result = batch[req_id]
                result = sch_result.result
                if isinstance(result, TextGenerationOutput):
                    all_tokens.extend(result.tokens)
                if sch_result.is_done:
                    is_done = True

        # match 99 from prefill
        # match 42 to 45 from decode
        assert all_tokens == [99, 42, 43, 44, 45]
        assert is_done


def test_overlap_di_both_sides_multiple_concurrent_requests() -> None:
    """Multiple concurrent requests with overlap on both sides: all complete
    with exact expected tokens and no FUTURE_TOKEN sentinel leaks."""
    decode, prefill, server_addr, q = create_di_scheduler(
        overlap_prefill=True, overlap_decode=True, max_forward_steps_tg=1
    )

    ctx1 = create_text_context(
        target_endpoint=server_addr, prompt_len=100, output_len=5
    )
    ctx2 = create_text_context(
        target_endpoint=server_addr, prompt_len=100, output_len=5
    )
    req_id1, req_id2 = ctx1.request_id, ctx2.request_id
    q.request_queue.put(ctx1)
    q.request_queue.put(ctx2)

    # Both requests through prefill (overlap) and decode (overlap).
    decode.run_iteration()
    prefill.run_iteration()
    prefill.run_iteration()
    for _ in range(8):
        decode.run_iteration()
        prefill.run_iteration()

    tokens1: list[int] = []
    tokens2: list[int] = []
    done1, done2 = False, False
    while not q.response_queue.empty():
        batch = q.response_queue.get()
        for rid, sch_result in batch.items():
            result = sch_result.result
            if isinstance(result, TextGenerationOutput):
                if rid == req_id1:
                    tokens1.extend(result.tokens)
                elif rid == req_id2:
                    tokens2.extend(result.tokens)
            if sch_result.is_done:
                if rid == req_id1:
                    done1 = True
                elif rid == req_id2:
                    done2 = True

    # Must complete with 5 tokens each (1 prefill + 4 decode)
    assert len(tokens1) == 5
    assert len(tokens2) == 5
    # Prefill start_token_id=99, increments per request as ctx1=99, ctx2=100
    assert tokens1[0] == 99
    assert tokens2[0] == 100
    assert done1 and done2
    assert FUTURE_TOKEN not in tokens1 + tokens2


def test_overlap_di_both_sides_kv_cache_fully_released() -> None:
    """After all requests complete with overlap on both sides, all KV cache
    pages on both decode and prefill are fully released — no resource leaks."""
    decode, prefill, server_addr, q = create_di_scheduler(
        overlap_prefill=True, overlap_decode=True, max_forward_steps_tg=1
    )

    num_requests = 3
    for _ in range(num_requests):
        ctx = create_text_context(
            target_endpoint=server_addr, prompt_len=100, output_len=5
        )
        q.request_queue.put(ctx)

    decode.run_iteration()
    prefill.run_iteration()
    prefill.run_iteration()
    for _ in range(8):
        decode.run_iteration()
        prefill.run_iteration()

    done_count = 0
    while not q.response_queue.empty():
        batch = q.response_queue.get()
        for sch_result in batch.values():
            if sch_result.is_done:
                done_count += 1
    assert done_count == num_requests

    # All KV pages must be released on both sides
    assert decode.kv_cache.get_num_used_pages(replica_idx=0) == 0
    assert prefill.kv_cache.get_num_used_pages(replica_idx=0) == 0
    # No lingering transfer state
    assert decode.inflight_transfers == {}
    assert prefill.active_transfers == {}


def test_overlap_di_both_sides_staggered_arrivals_e2e() -> None:
    """Staggered request arrivals with overlap on both sides.

    req1 arrives first, goes through prefill, and starts decoding. While
    req1 is mid-decode, req2 arrives and goes through prefill. Both must
    complete with correct tokens and no drops or misordering.
    """
    decode, prefill, server_addr, q = create_di_scheduler(
        overlap_prefill=True, overlap_decode=True, max_forward_steps_tg=1
    )

    ctx1 = create_text_context(
        target_endpoint=server_addr, prompt_len=100, output_len=5
    )
    ctx2 = create_text_context(
        target_endpoint=server_addr, prompt_len=100, output_len=5
    )
    req_id1, req_id2 = ctx1.request_id, ctx2.request_id

    # Phase 1: req1 through prefill and to decode
    q.request_queue.put(ctx1)
    decode.run_iteration()
    prefill.run_iteration()
    prefill.run_iteration()  # flush deferred prefill token
    decode.run_iteration()  # receives PrefillResponse, start decode
    decode.run_iteration()  # first decode step for req1

    # Phase 2: req2 arrives while req1 is mid-decode
    q.request_queue.put(ctx2)
    decode.run_iteration()  # sends req2 to prefill + continues req1 decode
    prefill.run_iteration()
    prefill.run_iteration()  # flush deferred prefill token for req2

    # Run remaining iterations until both complete
    for _ in range(8):
        decode.run_iteration()
        prefill.run_iteration()

    tokens1: list[int] = []
    tokens2: list[int] = []
    done1, done2 = False, False
    while not q.response_queue.empty():
        batch = q.response_queue.get()
        for rid, sch_result in batch.items():
            result = sch_result.result
            if isinstance(result, TextGenerationOutput):
                if rid == req_id1:
                    tokens1.extend(result.tokens)
                elif rid == req_id2:
                    tokens2.extend(result.tokens)
            if sch_result.is_done:
                if rid == req_id1:
                    done1 = True
                elif rid == req_id2:
                    done2 = True

    assert done1 and done2
    # Must complete with 5 tokens each (1 prefill + 4 decode)
    assert len(tokens1) == 5
    assert len(tokens2) == 5
    # Prefill start_token_id=99, increments per request as ctx1=99, ctx2=100
    assert tokens1[0] == 99
    assert tokens2[0] == 100
    assert FUTURE_TOKEN not in tokens1 + tokens2


def test_overlap_di_both_sides_minimal_output() -> None:
    """output_len=1: the request terminates as soon as possible.

    With overlap on both sides, the decode side token is deferred by one
    batch. The termination signal (MAXIMUM_LENGTH) must cross the lag
    boundary correctly.

    In DI, the decode side always runs at least one step after prefill
    (even if prefill already hit max_length), so we expect 2 tokens:
    1 from prefill + 1 from decode's first step.
    """
    decode, prefill, server_addr, q = create_di_scheduler(
        overlap_prefill=True, overlap_decode=True, max_forward_steps_tg=1
    )

    ctx = create_text_context(
        target_endpoint=server_addr, prompt_len=100, output_len=1
    )
    req_id = ctx.request_id
    q.request_queue.put(ctx)

    # Prefill with overlap (defer + flush).
    decode.run_iteration()
    prefill.run_iteration()
    prefill.run_iteration()

    # Decode iterations — the request should terminate quickly.
    for _ in range(8):
        decode.run_iteration()

    all_tokens: list[int] = []
    is_done = False
    while not q.response_queue.empty():
        batch = q.response_queue.get()
        if req_id in batch:
            sch_result = batch[req_id]
            result = sch_result.result
            if isinstance(result, TextGenerationOutput):
                all_tokens.extend(result.tokens)
            if sch_result.is_done:
                is_done = True

    assert is_done
    # match 99 as prefill start_token_id
    # match 42 as decode start_token_id
    assert all_tokens == [99, 42]
    assert FUTURE_TOKEN not in all_tokens


# Spec decode + disable_overlap=True: covers the PrefillScheduler's
# synchronous branch. Production Eagle currently always runs with overlap
# enabled (see the overlap + spec decode section below), so this section
# tests the scheduler path only and can be removed if the sync path is
# retired.


def test_spec_decode_prefill_sends_token_and_draft_tokens() -> None:
    """When the prefill pipeline uses unified Eagle (spec decode), the
    PrefillResponse must carry both the generated token and draft tokens."""
    num_spec_tokens = 3
    decode, prefill, server_addr, q = create_di_scheduler(
        spec_decode_prefill=True,
        num_speculative_tokens=num_spec_tokens,
    )
    ctx = create_text_context(
        target_endpoint=server_addr, prompt_len=100, output_len=5
    )
    q.request_queue.put(ctx)

    # Send request from decode -> prefill
    decode.run_iteration()
    # Execute prefill (spec decode pipeline) and send response
    prefill.run_iteration()

    # Read the PrefillResponse from the dispatcher
    prefill_metadata = decode.dispatcher.recv_reply_nowait()
    assert isinstance(prefill_metadata, KVTransferEngineMetadata)
    prefill_response = decode.dispatcher.recv_reply_nowait()
    assert isinstance(prefill_response, PrefillResponse)
    assert prefill_response.id == ctx.request_id
    assert prefill_response.generated_token_id == 99
    # Draft tokens must be present and have the right length
    assert prefill_response.draft_tokens is not None
    assert len(prefill_response.draft_tokens) == num_spec_tokens


def test_spec_decode_prefill_end_to_end() -> None:
    """End-to-end DI flow with speculative decoding on the prefill side.

    Verifies that:
    1. Prefill generates a token and transfers to decode
    2. Decode receives the token and draft tokens
    3. The full request completes normally
    """
    num_spec_tokens = 2
    decode, prefill, server_addr, q = create_di_scheduler(
        spec_decode_prefill=True,
        num_speculative_tokens=num_spec_tokens,
    )
    ctx = create_text_context(
        target_endpoint=server_addr, prompt_len=100, output_len=5
    )
    req_id = ctx.request_id
    q.request_queue.put(ctx)

    # Full lifecycle: decode sends to prefill, prefill executes, decode runs
    decode.run_iteration()
    prefill.run_iteration()
    decode.run_iteration()

    # First response should be the prefill token (99)
    output1 = q.response_queue.get()
    assert len(output1) == 1
    sch_output1 = output1[req_id]
    assert not sch_output1.is_done
    single_token = sch_output1.result
    assert isinstance(single_token, TextGenerationOutput)
    assert single_token.tokens == [99]

    # Second response should be decode tokens
    output2 = q.response_queue.get()
    assert len(output2) == 1
    sch_output2 = output2[req_id]
    assert sch_output2.is_done
    rest_of_tokens = sch_output2.result
    assert isinstance(rest_of_tokens, TextGenerationOutput)
    assert rest_of_tokens.tokens == [42, 43, 44, 45]


def test_spec_decode_prefill_does_not_accumulate_pending_first_token() -> None:
    """Regression: with spec decode (overlap disabled), requests must NOT
    accumulate in _pending_first_token. The synchronous path should be used.

    Before the fix, the PrefillScheduler always used the two-phase path
    for OverlapTextGenerationPipeline, causing requests to get stuck in
    _pending_first_token when overlap was disabled.
    """
    decode, prefill, server_addr, q = create_di_scheduler(
        spec_decode_prefill=True,
    )

    # Submit two requests sequentially
    ctx1 = create_text_context(
        target_endpoint=server_addr, prompt_len=100, output_len=5
    )
    ctx2 = create_text_context(
        target_endpoint=server_addr, prompt_len=100, output_len=5
    )
    q.request_queue.put(ctx1)
    q.request_queue.put(ctx2)

    # Decode sends both requests to prefill
    decode.run_iteration()

    # Prefill executes first batch
    prefill.run_iteration()

    # Overlap disabled here → synchronous path, never populates
    # _pending_first_token. Overlap + spec decode is covered separately
    # in test_overlap_spec_decode_prefill_uses_two_phase_path.
    assert len(prefill._pending_first_token) == 0

    # Both requests should have initiated transfers
    assert len(prefill.active_transfers) == 2


def test_spec_decode_prefill_decode_receives_draft_tokens() -> None:
    """The decode scheduler correctly restores draft tokens from PrefillResponse
    onto the context's spec_decoding_state."""
    num_spec_tokens = 3
    decode, prefill, server_addr, q = create_di_scheduler(
        spec_decode_prefill=True,
        num_speculative_tokens=num_spec_tokens,
    )
    ctx = create_text_context(
        target_endpoint=server_addr, prompt_len=100, output_len=5
    )
    req_id = ctx.request_id
    q.request_queue.put(ctx)

    # Send to prefill and execute prefill
    decode.run_iteration()
    prefill.run_iteration()

    # Manually receive the PrefillResponse and feed it to the decode scheduler
    # (mirrors test_prefill_sends_new_token_to_decode pattern)
    prefill_metadata = decode.dispatcher.recv_reply_nowait()
    assert isinstance(prefill_metadata, KVTransferEngineMetadata)
    prefill_response = decode.dispatcher.recv_reply_nowait()
    assert isinstance(prefill_response, PrefillResponse)

    assert prefill_response.draft_tokens is not None
    assert len(prefill_response.draft_tokens) == num_spec_tokens

    # Feed the response into the decode scheduler
    decode.handle_prefill_response(prefill_response)

    # The context in prefill_reqs should now have draft tokens set
    assert req_id in decode.prefill_reqs
    pending = decode.prefill_reqs[req_id]
    assert (
        len(pending.context.spec_decoding_state.draft_tokens_to_verify)
        == num_spec_tokens
    )


def test_load_prefill_scheduler_rejects_standalone_spec_decode() -> None:
    """load_prefill_scheduler must raise for standalone speculative decoding."""
    pipeline = MagicMock()
    pipeline.kv_manager = MagicMock()
    config = MagicMock()
    config.speculative = SpeculativeConfig(speculative_method="standalone")

    with pytest.raises(ValueError, match="Standalone speculative decoding"):
        load_prefill_scheduler(pipeline, config, MagicMock())


def test_load_prefill_scheduler_accepts_eagle_spec_decode() -> None:
    """load_prefill_scheduler returns a PrefillScheduler for eagle spec decode.

    PrefillScheduler is patched so the test stays a unit — it would otherwise
    need a full KV cache, transfer engine, and NIXL agent.
    """
    pipeline = MagicMock()
    pipeline.kv_manager = MagicMock()
    config = MagicMock()
    config.speculative = SpeculativeConfig(speculative_method="eagle")
    sentinel = MagicMock(name="PrefillScheduler")

    with (
        patch(
            "max.serve.scheduler.prefill_scheduler.PrefillScheduler",
            return_value=sentinel,
        ) as prefill_scheduler_cls,
        patch("max.serve.scheduler.prefill_scheduler.PrefillDispatcherServer"),
        patch(
            "max.serve.scheduler.prefill_scheduler."
            "TokenGenerationSchedulerConfig.from_pipeline_config",
        ) as from_pipeline_config,
    ):
        result = load_prefill_scheduler(pipeline, config, MagicMock())

    assert result is sentinel
    prefill_scheduler_cls.assert_called_once()
    from_pipeline_config.assert_called_once_with(config)


def test_load_prefill_scheduler_accepts_mtp_spec_decode() -> None:
    """load_prefill_scheduler accepts mtp spec decode just like eagle."""
    pipeline = MagicMock()
    pipeline.kv_manager = MagicMock()
    config = MagicMock()
    config.speculative = SpeculativeConfig(speculative_method="mtp")

    with (
        patch("max.serve.scheduler.prefill_scheduler.PrefillScheduler"),
        patch("max.serve.scheduler.prefill_scheduler.PrefillDispatcherServer"),
        patch(
            "max.serve.scheduler.prefill_scheduler."
            "TokenGenerationSchedulerConfig.from_pipeline_config",
        ),
    ):
        load_prefill_scheduler(pipeline, config, MagicMock())


# Overlap + speculative decoding on the prefill side: CE deferral and
# draft-token publication must coexist (num_speculative_tokens=1 path).


def test_overlap_spec_decode_prefill_uses_two_phase_path() -> None:
    """With overlap enabled, spec decode requests flow through the two-phase
    path (deferred CE → flush), not the synchronous shortcut."""
    decode, prefill, server_addr, q = create_di_scheduler(
        spec_decode_prefill=True,
        overlap_prefill=True,
    )
    assert isinstance(prefill.pipeline, FakeOverlapPipeline)
    assert not prefill.pipeline._disable_overlap

    ctx = create_text_context(
        target_endpoint=server_addr, prompt_len=100, output_len=5
    )
    q.request_queue.put(ctx)

    # Iter 1: CE deferred; _pending_first_token tracks the request.
    decode.run_iteration()
    prefill.run_iteration()

    assert len(prefill._pending_first_token) == 1
    assert ctx.request_id in prefill._pending_first_token
    assert prefill.pipeline.has_pending_outputs()


def test_overlap_spec_decode_prefill_response_carries_draft_tokens() -> None:
    """After the two-phase flush, the PrefillResponse carries both the real
    generated token and the populated draft tokens."""
    num_spec_tokens = 1
    decode, prefill, server_addr, q = create_di_scheduler(
        spec_decode_prefill=True,
        overlap_prefill=True,
        num_speculative_tokens=num_spec_tokens,
    )
    assert isinstance(prefill.pipeline, FakeOverlapPipeline)
    ctx = create_text_context(
        target_endpoint=server_addr, prompt_len=100, output_len=5
    )
    q.request_queue.put(ctx)

    # Iter 1: defer. Iter 2: flush + send PrefillResponse.
    decode.run_iteration()
    prefill.run_iteration()
    assert prefill.pipeline.has_pending_outputs()
    prefill.run_iteration()
    assert not prefill.pipeline.has_pending_outputs()

    metadata = decode.dispatcher.recv_reply_nowait()
    assert isinstance(metadata, KVTransferEngineMetadata)
    response = decode.dispatcher.recv_reply_nowait()
    assert isinstance(response, PrefillResponse)
    assert response.id == ctx.request_id
    assert response.generated_token_id != FUTURE_TOKEN
    assert response.draft_tokens is not None
    assert len(response.draft_tokens) == num_spec_tokens


def test_overlap_spec_decode_end_to_end() -> None:
    """Full DI lifecycle with overlap + spec decode on the prefill side."""
    decode, prefill, server_addr, q = create_di_scheduler(
        spec_decode_prefill=True,
        overlap_prefill=True,
        num_speculative_tokens=1,
    )
    ctx = create_text_context(
        target_endpoint=server_addr, prompt_len=100, output_len=5
    )
    req_id = ctx.request_id
    q.request_queue.put(ctx)

    decode.run_iteration()
    prefill.run_iteration()  # defer
    prefill.run_iteration()  # flush
    decode.run_iteration()

    output1 = q.response_queue.get()
    sch1 = output1[req_id]
    assert not sch1.is_done
    tok1 = sch1.result
    assert isinstance(tok1, TextGenerationOutput)
    assert tok1.tokens == [99]

    output2 = q.response_queue.get()
    sch2 = output2[req_id]
    assert sch2.is_done
    tok2 = sch2.result
    assert isinstance(tok2, TextGenerationOutput)
    assert tok2.tokens == [42, 43, 44, 45]


def test_overlap_spec_decode_cancel_between_defer_and_resolve() -> None:
    """Cancelling an overlap + spec-decode request between CE deferral and
    flush emits no PrefillResponse and does not crash."""
    decode, prefill, server_addr, q = create_di_scheduler(
        spec_decode_prefill=True,
        overlap_prefill=True,
        num_speculative_tokens=1,
    )
    assert isinstance(prefill.pipeline, FakeOverlapPipeline)
    ctx = create_text_context(
        target_endpoint=server_addr, prompt_len=100, output_len=5
    )
    q.request_queue.put(ctx)

    decode.run_iteration()
    prefill.run_iteration()
    assert prefill.pipeline.has_pending_outputs()

    # Cancel while request sits in _pending_first_token.
    prefill.handle_cancel_request(CancelRequest(id=ctx.request_id))

    prefill.run_iteration()
    assert not prefill.pipeline.has_pending_outputs()

    while True:
        try:
            msg = decode.dispatcher.recv_reply_nowait()
            assert not isinstance(msg, PrefillResponse), (
                "Cancelled overlap+spec-decode request must not produce a "
                "PrefillResponse"
            )
        except queue.Empty:
            break


def test_update_spec_decode_skips_draft_tokens_when_is_done() -> None:
    """update_spec_decode_context_and_prepare_responses skips draft tokens
    when ctx.is_done=True (MAXIMUM_LENGTH).

    Regression test for the 1p1d overlap + Eagle3 CE→TG handoff bug. On the
    prefill pod, max_gen_tokens=1 arises naturally when an accumulated prompt
    reaches max_seq_len-1 tokens (e.g. long multi-turn sessions). The overlap
    pipeline calls update_with_future_token() during Phase 1, which advances
    current_position to max_length and sets status=MAXIMUM_LENGTH (is_done=True).
    The done context produces no further TG steps on the decode pod, so draft
    tokens are neither generated nor sent. The prefill and decode schedulers
    gate their draft-token checks on not context.is_done to handle this case.
    """
    prompt_len = 10
    ctx = create_text_context(
        target_endpoint="ipc:///tmp/test",
        prompt_len=prompt_len,
        output_len=1,
    )
    assert ctx.max_length == prompt_len + 1

    # Phase 1: overlap pipeline appends FUTURE_TOKEN and advances position
    # to max_length, setting is_done=True via MAXIMUM_LENGTH.
    ctx.update_with_future_token()
    assert ctx.is_done, (
        "Expected is_done=True after update_with_future_token on max_gen_tokens=1 context"
    )

    real_draft_tokens = [10, 11, 12]

    update_spec_decode_context_and_prepare_responses(
        draft_tokens=np.array([[42, 42, 42]], dtype=np.int32),
        next_draft_tokens=np.array([real_draft_tokens], dtype=np.int32),
        num_accepted_draft_tokens=np.array([0], dtype=np.int32),
        next_tokens=np.array([99], dtype=np.int32),
        context_batch=[ctx],
        max_seq_len=2048,
    )

    assert ctx.spec_decoding_state.draft_tokens_to_verify == [], (
        "draft_tokens_to_verify must be empty when ctx.is_done=True; "
        "done contexts produce no TG steps so draft tokens are not sent"
    )


# ---------------------------------------------------------------------------
# Stall watchdog tests
# ---------------------------------------------------------------------------


def test_stall_watchdog_fires_when_prefill_stalled() -> None:
    """When requests are stuck in prefill longer than the timeout,
    run_iteration raises SystemExit(1) to enable a pod restart."""
    decode, _, server_addr, q = create_di_scheduler()
    ctx = create_text_context(target_endpoint=server_addr, prompt_len=100)
    q.request_queue.put(ctx)

    # Move request into prefill_reqs without running prefill (simulates NIXL stall).
    decode.reserve_memory_and_send_to_prefill()
    assert len(decode.prefill_reqs) == 1

    # Backdate last activity to simulate a long stall.
    decode._last_batch_activity = time.monotonic() - 9999
    decode.scheduler_config.decode_stall_timeout_s = 1.0

    with pytest.raises(SystemExit) as exc_info:
        decode.run_iteration()
    assert exc_info.value.code == 1


def test_stall_watchdog_no_fire_within_timeout() -> None:
    """Watchdog does not fire when the stall duration is below the threshold."""
    decode, _, server_addr, q = create_di_scheduler()
    ctx = create_text_context(target_endpoint=server_addr, prompt_len=100)
    q.request_queue.put(ctx)

    decode.reserve_memory_and_send_to_prefill()
    assert len(decode.prefill_reqs) == 1

    # Last activity was just now.
    decode._last_batch_activity = time.monotonic()
    decode.scheduler_config.decode_stall_timeout_s = 9999.0

    result = decode.run_iteration()

    assert result == SchedulerProgress.NO_PROGRESS


def test_stall_watchdog_resets_on_batch_activity() -> None:
    """_last_batch_activity is updated whenever a non-empty batch is produced."""
    decode, prefill, server_addr, q = create_di_scheduler()
    ctx = create_text_context(
        target_endpoint=server_addr, prompt_len=100, output_len=5
    )
    q.request_queue.put(ctx)

    decode.run_iteration()  # sends to prefill
    prefill.run_iteration()  # executes prefill, sends PrefillResponse

    t_before = time.monotonic()
    decode.run_iteration()  # receives response, produces decode batch

    assert decode._last_batch_activity >= t_before


def test_stall_watchdog_clock_resets_while_idle() -> None:
    """Clock resets when the queue is empty, so idle warmup time before the
    first request does not count toward the stall threshold.

    Without this reset, initializing `_last_batch_activity` at scheduler
    startup causes the watchdog to fire immediately on the first request if
    startup takes longer than the timeout (e.g. dataset loading before
    benchmark start)."""
    decode, _, server_addr, q = create_di_scheduler()

    # Simulate a stale clock from a long idle period (e.g. pre-benchmark warmup).
    decode._last_batch_activity = time.monotonic() - 9999
    decode.scheduler_config.decode_stall_timeout_s = 60.0

    # With no pending requests, run_iteration resets the clock.
    decode.run_iteration()

    # Now submit the first request — stall duration is measured from the reset,
    # not from scheduler init, so the watchdog must not fire.
    ctx = create_text_context(target_endpoint=server_addr, prompt_len=100)
    q.request_queue.put(ctx)
    decode.reserve_memory_and_send_to_prefill()
    assert len(decode.prefill_reqs) == 1

    result = decode.run_iteration()

    assert result == SchedulerProgress.NO_PROGRESS


def test_stall_watchdog_default_is_disabled() -> None:
    """The stall timeout defaults to None (disabled) when the env var is unset."""
    decode, _, _, _ = create_di_scheduler()
    assert decode.scheduler_config.decode_stall_timeout_s is None


# ---------------------------------------------------------------------------
# Per-request TTL eviction (decode_request_ttl_s)
# ---------------------------------------------------------------------------


def test_decode_request_ttl_default_is_disabled() -> None:
    """The per-request TTL defaults to None when the env var is unset."""
    decode, _, _, _ = create_di_scheduler()
    assert decode.scheduler_config.decode_request_ttl_s is None


def test_decode_request_ttl_propagates_from_pipeline_config() -> None:
    """``decode_request_ttl_s`` flows through ``from_pipeline_config``."""
    pipeline_config = MagicMock()
    pipeline_config.runtime.max_batch_size = 1
    pipeline_config.runtime.max_num_steps = 1
    pipeline_config.runtime.max_batch_input_tokens = 8192
    pipeline_config.runtime.max_batch_total_tokens = 8192
    pipeline_config.runtime.enable_chunked_prefill = True
    pipeline_config.runtime.enable_in_flight_batching = False
    pipeline_config.runtime.kvcache_ce_watermark = 0.95
    pipeline_config.runtime.decode_stall_timeout_s = None
    pipeline_config.runtime.decode_request_ttl_s = 42.0
    pipeline_config.model.max_length = 2048
    pipeline_config.model.data_parallel_degree = 1
    pipeline_config.speculative = None

    config = TokenGenerationSchedulerConfig.from_pipeline_config(
        pipeline_config
    )

    assert config.decode_request_ttl_s == 42.0


def test_decode_run_iteration_evicts_stuck_prefill_request_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run_iteration`` evicts a request stuck in ``prefill_reqs`` past TTL."""
    decode, prefill, server_addr, q = create_di_scheduler()
    decode.scheduler_config.decode_request_ttl_s = 30.0

    ctx = create_text_context(
        target_endpoint=server_addr, prompt_len=100, output_len=5
    )
    q.request_queue.put(ctx)
    req_id = ctx.request_id

    pages_before = decode.kv_cache.get_num_used_pages(replica_idx=0)

    # Send to prefill but never run prefill, so PrefillResponse never arrives.
    decode.run_iteration()
    assert req_id in decode.prefill_reqs
    assert decode.prefill_reqs_per_replica[0] == 1

    # Patch only affects time.monotonic() in _evict_expired_requests; the
    # already-set PendingPrefill.sent_at uses the originally-captured ref.
    real_now = time.monotonic()
    monkeypatch.setattr(time, "monotonic", lambda: real_now + 1000.0)

    decode.run_iteration()

    assert req_id not in decode.prefill_reqs
    assert decode.prefill_reqs_per_replica[0] == 0
    assert decode.kv_cache.get_num_used_pages(replica_idx=0) == pages_before

    saw_cancel_response = False
    while not q.response_queue.empty():
        batch = q.response_queue.get_nowait()
        if req_id in batch and batch[req_id].result is None:
            saw_cancel_response = True
    assert saw_cancel_response

    # Drain the prefill dispatcher: handshake + PrefillRequest + CancelRequest.
    saw_cancel_to_prefill = False
    for _ in range(3):
        try:
            msg, _identity = prefill.dispatcher.recv_request_nowait()
        except queue.Empty:
            break
        if isinstance(msg, CancelRequest) and msg.id == req_id:
            saw_cancel_to_prefill = True
    assert saw_cancel_to_prefill


# ---------------------------------------------------------------------------
# maxserve.num_requests_queued gauge (MXSERV-29).
#
# The gauge is a synchronous OTel Gauge: ``BatchMetrics.publish_metrics``
# samples ``num_pending_reqs`` once per scheduler iteration and replaces
# the previous reading. The ``num_pending_reqs`` value the DecodeScheduler
# passes into ``log_metrics`` is ``len(pending_reqs) + len(prefill_reqs)``,
# so the gauge mirrors the local pending set on the decode side.
# ---------------------------------------------------------------------------


class _GaugeRecorder:
    """Captures METRICS.reqs_queued snapshots from BatchMetrics."""

    def __init__(self) -> None:
        self.values: list[int] = []

    @property
    def last(self) -> int | None:
        return self.values[-1] if self.values else None

    def reqs_queued(self, value: int) -> None:
        self.values.append(value)

    def __getattr__(self, _name: str) -> object:
        return MagicMock()


def _patch_decode_metrics(recorder: _GaugeRecorder) -> Any:
    """Patches the METRICS binding used by ``BatchMetrics.publish_metrics``."""
    return patch("max.serve.scheduler.utils.METRICS", recorder)


def test_decode_publish_metrics_emits_pending_snapshot() -> None:
    """Each decode iteration that runs a batch publishes the current
    pending depth (len(pending_reqs) + len(prefill_reqs)) as a gauge.
    """
    recorder = _GaugeRecorder()
    with _patch_decode_metrics(recorder):
        decode, prefill, _q, _ctx = (
            create_default_di_scheduler_and_submit_one_request()
        )

        # Drive the request all the way through: decode drains and
        # forwards to prefill, prefill runs, decode picks up the
        # response and runs TG. Each non-empty batch on either side
        # publishes a snapshot; the final value must read 0.
        decode.run_iteration()
        prefill.run_iteration()
        decode.run_iteration()

        assert recorder.last == 0
        assert (
            len(decode.pending_reqs) + len(decode.prefill_reqs) == recorder.last
        )
