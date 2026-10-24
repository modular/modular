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

import dataclasses
import time
from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock, call, patch

import numpy as np
import pytest
from max.nn.kv_cache import KVCacheGroupId, PagedKVLeafRegion
from max.pipelines.context import (
    GenerationStatus,
    TextContext,
    TextGenerationOutput,
    TokenBuffer,
)
from max.pipelines.kv_cache import InsufficientBlocksError
from max.pipelines.kv_cache.connectors.null_connector import NullConnector
from max.pipelines.kv_cache.kv_connector import (
    BlockCount,
    CompletedTransfer,
    KVLoadFailed,
    KVLookup,
    KVTransfer,
)
from max.pipelines.kv_cache.paged_kv_cache import (
    PrefixCacheHits,
    jenga_block_manager,
)
from max.pipelines.kv_cache.paged_kv_cache._common import compute_block_hashes
from max.pipelines.kv_cache.paged_kv_cache.block_manager import BlockManager
from max.pipelines.kv_cache.paged_kv_cache.cache_manager import (
    PagedKVCacheManager,
)
from max.pipelines.kv_cache.paged_kv_cache.jenga_block_manager import (
    JengaBlockManager,
    KVLeafInfo,
    create_groups,
    create_pools,
)
from max.pipelines.modeling.types import (
    Pipeline,
    RequestID,
    TextGenerationInputs,
)
from max.serve.scheduler.batch_constructor import text_batch_constructor
from max.serve.scheduler.batch_constructor.text_batch_constructor import (
    MAX_PENDING_LOOKUPS_ENV_VAR,
    PreemptionReason,
    TextBatchConstructor,
)
from max.serve.scheduler.batch_constructor.token_budget import RequestType
from max.serve.scheduler.config import TokenGenerationSchedulerConfig

ARBITRARY_TOKEN_ID = 999


@pytest.fixture
def pipeline() -> Pipeline[
    TextGenerationInputs[TextContext], TextGenerationOutput
]:
    pipeline = Mock()
    pipeline.release = Mock()
    return pipeline


def create_mock_lora_manager(max_num_loras: int = 2) -> Mock:
    """Create a mock LoRA manager for testing."""
    manager = Mock()
    manager.max_num_loras = max_num_loras
    active_loras: set[str] = set()
    all_loras: set[str] = set()
    manager._active_loras = active_loras
    manager._all_loras = all_loras

    def is_lora(model_name: str | None) -> bool:
        return bool(model_name and model_name.startswith("lora_"))

    def is_active_lora(model_name: str | None) -> bool:
        return model_name in manager._active_loras if model_name else False

    def activate_adapter(model_name: str) -> None:
        if len(manager._active_loras) >= max_num_loras:
            raise RuntimeError("Cannot activate more LoRAs than max_num_loras")
        manager._active_loras.add(model_name)
        manager._all_loras.add(model_name)

    manager.is_lora = Mock(side_effect=is_lora)
    manager.is_active_lora = Mock(side_effect=is_active_lora)
    manager.activate_adapter = Mock(side_effect=activate_adapter)

    return manager


def set_mock_kv_usage(cache: Mock, used_fraction: float) -> None:
    """Point a mock cache's device block count at a given usage fraction.

    ``_identify_priority`` reads ``pressure_pct()``, so a mock that reaches
    it needs real numbers rather than an auto-created ``Mock``.
    """
    used = round(used_fraction * 100)
    cache.block_count = Mock(
        return_value=BlockCount(free=100 - used, total=100)
    )
    cache.pressure_pct = Mock(return_value=float(used))


def create_mock_kv_cache() -> Mock:
    """Create a mock paged KV cache manager with minimal interface."""
    cache = Mock()
    cache.chunk_alignment_tokens = 0

    cache.alloc = Mock(return_value=CompletedTransfer())
    cache.claim = Mock()
    cache.release = Mock()
    cache.contains = Mock(return_value=False)
    cache.pending_transfers_exist = Mock(return_value=False)
    cache.starts_lookup = Mock(return_value=False)
    cache.awaits_lookup = Mock(return_value=False)
    cache.hand_back_lookup = Mock()
    set_mock_kv_usage(cache, 0.0)

    return cache


def create_mock_pipeline_with_lora(lora_manager: Mock) -> Mock:
    """Create a mock pipeline with LoRA support."""

    def next_token_behavior(
        inputs: TextGenerationInputs[TextContext],
    ) -> dict[RequestID, TextGenerationOutput]:
        responses: dict[RequestID, TextGenerationOutput] = {}

        for request in inputs.flat_batch:
            request_id = request.request_id
            request.update(0)

            responses[request_id] = TextGenerationOutput(
                request_id=request_id,
                tokens=[0, 0],
                final_status=GenerationStatus.ACTIVE,
                log_probabilities=None,
            )

        return responses

    pipeline = Mock()
    pipeline.execute = Mock(side_effect=next_token_behavior)
    pipeline.release = Mock()
    pipeline._pipeline_model = Mock()
    pipeline._pipeline_model._lora_manager = lora_manager

    return pipeline


def create_lora_context(
    seq_len: int = 30, model_name: str | None = None, is_tg: bool = False
) -> TextContext:
    """Create a TextContext with optional LoRA model name."""
    tokens = np.ones(seq_len, dtype=np.int64)
    context = TextContext(
        request_id=RequestID(),
        max_length=100,
        tokens=TokenBuffer(tokens),
    )
    if model_name:
        context.model_name = model_name
    if is_tg:
        context.update(ARBITRARY_TOKEN_ID)
    return context


def has_request(batch: list[TextContext], request_id: RequestID) -> bool:
    return any(ctx.request_id == request_id for ctx in batch)


def test_text_batch_constructor__batch_construction_without_chunked_prefill_no_preemption(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=5,
        max_batch_total_tokens=None,
        enable_in_flight_batching=False,
        enable_chunked_prefill=False,
        target_tokens_per_batch_ce=30,
    )

    kv_cache = Mock()
    kv_cache.chunk_alignment_tokens = 0
    kv_cache.alloc = Mock()
    kv_cache.alloc.return_value = CompletedTransfer()
    kv_cache.claim = Mock()
    kv_cache.contains = Mock()
    set_mock_kv_usage(kv_cache, 0.0)

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
    )

    # Enqueue 6 CE requests, at 9 tokens each
    # Each have plenty of room for max length
    contexts = {}
    for _ in range(6):
        context = TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(9, dtype=np.int64)),
            max_length=100,
        )
        contexts[context.request_id] = context
        batch_constructor.enqueue_new_request(context)

    assert batch_constructor._identify_priority(0) == RequestType.CE
    inputs = batch_constructor.construct_batch()
    # 9 * 4 = 36 tokens, since no max_batch_total_tokens is set, we should have 4 requests in the batch
    assert len(inputs.batches[0]) == 4
    # since this is CE, we should have 1 step

    # test that we have 2 requests remaining in the queue
    assert len(batch_constructor.replicas[0].ce_reqs) == 2

    # test that 2 of the requests finished
    request_ids = list(contexts.keys())
    responses = {
        request_ids[0]: TextGenerationOutput(
            request_id=request_ids[0],
            final_status=GenerationStatus.END_OF_SEQUENCE,
            tokens=[0],
        ),
        request_ids[1]: TextGenerationOutput(
            request_id=request_ids[1],
            final_status=GenerationStatus.ACTIVE,
            tokens=[1],
        ),
        request_ids[2]: TextGenerationOutput(
            request_id=request_ids[2],
            final_status=GenerationStatus.END_OF_SEQUENCE,
            tokens=[2],
        ),
    }

    # Update a token for each request in the batch
    for batch in inputs.batches:
        for context in batch:
            context.update(0)

    batch_constructor.advance_requests(inputs)

    for request_id, response in responses.items():
        if response.is_done:
            batch_constructor.release_request(request_id)

    # 4 completed CE, 2 were completed, and 2 moved to TG
    assert len(batch_constructor.replicas[0].tg_reqs) == 2
    # There are 2 requests remaining in the CE queue
    assert len(batch_constructor.replicas[0].ce_reqs) == 2

    assert batch_constructor._identify_priority(0) == RequestType.CE

    inputs = batch_constructor.construct_batch()
    assert len(inputs.batches[0]) == 2

    for batch in inputs.batches:
        for context in batch:
            context.update(0)

    batch_constructor.advance_requests(inputs)

    assert len(batch_constructor.replicas[0].ce_reqs) == 0
    assert len(batch_constructor.replicas[0].tg_reqs) == 4

    # Assume that we have 4 requests remaining in the queue
    # And none of the requests have a max length, therefore we use the default
    assert batch_constructor._identify_priority(0) == RequestType.TG
    inputs = batch_constructor.construct_batch()
    assert len(inputs.batches[0]) == 4


def test_text_batch_constructor__batch_construction_no_requests(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=5,
        max_batch_total_tokens=None,
        enable_in_flight_batching=False,
        enable_chunked_prefill=False,
        target_tokens_per_batch_ce=30,
    )

    kv_cache = Mock()
    kv_cache.chunk_alignment_tokens = 0
    kv_cache.alloc = Mock()
    kv_cache.alloc.return_value = CompletedTransfer()
    kv_cache.claim = Mock()
    kv_cache.contains = Mock()
    set_mock_kv_usage(kv_cache, 0.0)

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
    )
    inputs = batch_constructor.construct_batch()
    assert len(inputs.batches) == 1
    assert len(inputs.batches[0]) == 0


def test_text_batch_constructor__structured_output_enabled_mirrors_bitmask_constraints(
    pipeline: Mock,
) -> None:
    """``structured_output_enabled`` forwards
    ``PipelineConfig.needs_bitmask_constraints`` -- on for either
    ``--enable-structured-output`` or a grammar-capable tool parser with a
    ``--tool-call-policy`` other than ``force_unconstrained`` -- and stays off
    if the pipeline exposes no ``pipeline_config`` at all."""
    batch_constructor = TextBatchConstructor(
        scheduler_config=TokenGenerationSchedulerConfig(
            max_batch_size=5,
            max_batch_total_tokens=None,
            enable_in_flight_batching=False,
            enable_chunked_prefill=False,
            target_tokens_per_batch_ce=30,
        ),
        pipeline=pipeline,
        kv_cache=Mock(),
    )

    pipeline.pipeline_config = Mock(needs_bitmask_constraints=True)
    assert batch_constructor.structured_output_enabled is True

    pipeline.pipeline_config.needs_bitmask_constraints = False
    assert batch_constructor.structured_output_enabled is False

    del pipeline.pipeline_config
    assert batch_constructor.structured_output_enabled is False


def test_text_batch_constructor__batch_construction_no_room_in_cache(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=5,
        max_batch_total_tokens=None,
        enable_in_flight_batching=False,
        enable_chunked_prefill=False,
        target_tokens_per_batch_ce=30,
    )
    kv_cache = Mock()
    kv_cache.chunk_alignment_tokens = 0
    kv_cache.alloc = Mock(side_effect=InsufficientBlocksError)
    kv_cache.claim = Mock()
    kv_cache.contains = Mock()
    kv_cache.pending_transfers_exist = Mock(return_value=False)
    kv_cache.num_req_blocks = Mock(return_value=0)
    kv_cache.get_req_blocks_per_leaf = Mock(return_value={"full": []})
    set_mock_kv_usage(kv_cache, 0.0)

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
    )

    for _ in range(2):
        context = TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(9, dtype=np.int64)),
            max_length=100,
        )
        batch_constructor.enqueue_new_request(context)

    # With no TG, no active batch, and no in-flight KV transfers, there is
    # nothing that will free blocks — InsufficientBlocksError propagates.
    with pytest.raises(InsufficientBlocksError):
        batch_constructor.construct_batch()


def test_text_batch_constructor__insufficient_blocks_defers_then_retries(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """A CE request deferred by InsufficientBlocksError is admitted once
    blocks free up: something is in flight, so the first call defers
    (presence, not magnitude -- it doesn't matter whether that in-flight
    signal is "enough"); the second call's alloc succeeds outright."""
    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=5,
        max_batch_total_tokens=None,
        enable_in_flight_batching=False,
        enable_chunked_prefill=False,
        target_tokens_per_batch_ce=30,
    )
    kv_cache = Mock()
    kv_cache.chunk_alignment_tokens = 0
    kv_cache.alloc = Mock(
        side_effect=[
            InsufficientBlocksError("insufficient blocks"),
            CompletedTransfer(),
            CompletedTransfer(),
        ]
    )
    kv_cache.claim = Mock()
    kv_cache.contains = Mock()
    kv_cache.pending_transfers_exist = Mock(return_value=False)
    set_mock_kv_usage(kv_cache, 0.0)

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
        get_inflight_kv_transfer_count=lambda replica_idx: 1,
    )

    for _ in range(2):
        context = TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(9, dtype=np.int64)),
            max_length=100,
        )
        batch_constructor.enqueue_new_request(context)

    # First call: alloc fails, but something is in flight → defer.
    inputs = batch_constructor.construct_batch()
    assert len(inputs.batches[0]) == 0
    assert len(batch_constructor.replicas[0].ce_reqs) == 2

    # Second call: alloc succeeds → both requests admitted.
    inputs = batch_constructor.construct_batch()
    assert len(inputs.batches[0]) == 2
    assert len(batch_constructor.replicas[0].ce_reqs) == 0


def _presence_test_setup(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
    *,
    inflight_count: int = 0,
    is_tg: bool = False,
) -> TextBatchConstructor:
    """Builds a TextBatchConstructor with one request (CE, or TG when
    is_tg) whose alloc always fails, backed by an in-flight tracker
    reporting the given count."""
    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=5,
        max_batch_total_tokens=None,
        enable_in_flight_batching=False,
        enable_chunked_prefill=False,
        target_tokens_per_batch_ce=30,
    )
    kv_cache = Mock()
    kv_cache.chunk_alignment_tokens = 0
    kv_cache.alloc = Mock(
        side_effect=InsufficientBlocksError("insufficient blocks")
    )
    kv_cache.claim = Mock()
    kv_cache.contains = Mock()
    kv_cache.pending_transfers_exist = Mock(return_value=False)
    kv_cache.num_req_blocks = Mock(return_value=0)
    kv_cache.get_req_blocks_per_leaf = Mock(return_value={"full": []})
    set_mock_kv_usage(kv_cache, 0.0)

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
        get_inflight_kv_transfer_count=lambda replica_idx: inflight_count,
    )
    context = (
        create_lora_context(seq_len=9, is_tg=True)
        if is_tg
        else TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(9, dtype=np.int64)),
            max_length=100,
        )
    )
    batch_constructor.enqueue_new_request(context)
    return batch_constructor


def test_text_batch_constructor__ce_insufficient_blocks_fatal_when_nothing_inflight(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """No other work and nothing in flight -- genuinely stuck, must raise."""
    batch_constructor = _presence_test_setup(pipeline)
    with pytest.raises(InsufficientBlocksError):
        batch_constructor.construct_batch()


def _fatal_test_batch_constructor(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
    *,
    inflight_count: int = 0,
    pending_transfers_exist: bool = False,
) -> TextBatchConstructor:
    """Builds a bare TextBatchConstructor for calling
    _is_insufficient_blocks_fatal directly, without going through
    construct_batch()."""
    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=5,
        max_batch_total_tokens=None,
        enable_in_flight_batching=False,
        enable_chunked_prefill=False,
        target_tokens_per_batch_ce=30,
    )
    kv_cache = Mock()
    kv_cache.chunk_alignment_tokens = 0
    kv_cache.pending_transfers_exist = Mock(
        return_value=pending_transfers_exist
    )
    return TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
        get_inflight_kv_transfer_count=lambda replica_idx: inflight_count,
    )


def test_text_batch_constructor__is_insufficient_blocks_fatal_false_with_other_work(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """With other runnable work present, never fatal regardless of what's
    in flight."""
    batch_constructor = _fatal_test_batch_constructor(pipeline)
    assert not batch_constructor._is_insufficient_blocks_fatal(
        replica_idx=0, no_other_work=False
    )


def test_text_batch_constructor__is_insufficient_blocks_fatal_true_when_nothing_inflight(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """No other work and nothing in flight -- genuinely stuck, fatal."""
    batch_constructor = _fatal_test_batch_constructor(pipeline)
    assert batch_constructor._is_insufficient_blocks_fatal(
        replica_idx=0, no_other_work=True
    )


def test_text_batch_constructor__is_insufficient_blocks_fatal_false_when_anything_inflight(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """No other work, but something is in flight -- not fatal, no matter
    how little: a single in-flight transfer might still resolve, and
    sizing "enough" against a snapshot would only ever undercount what
    could actually come back."""
    batch_constructor = _fatal_test_batch_constructor(
        pipeline, inflight_count=1
    )
    assert not batch_constructor._is_insufficient_blocks_fatal(
        replica_idx=0, no_other_work=True
    )


def test_text_batch_constructor__is_insufficient_blocks_fatal_false_when_pending_transfer_exists(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """A pending device-side transfer (e.g. a D2H offload) alone is enough
    to defer, even with no external in-flight signal."""
    batch_constructor = _fatal_test_batch_constructor(
        pipeline, pending_transfers_exist=True
    )
    assert not batch_constructor._is_insufficient_blocks_fatal(
        replica_idx=0, no_other_work=True
    )


def test_text_batch_constructor__tg_insufficient_blocks_fatal_when_nothing_inflight(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """Same presence check on the TG path: raises once candidate_ids is
    exhausted and nothing at all is in flight."""
    batch_constructor = _presence_test_setup(pipeline, is_tg=True)
    with pytest.raises(InsufficientBlocksError):
        batch_constructor.construct_batch()


def test_text_batch_constructor__tg_insufficient_blocks_preempts_ce_block_holder(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """A decode request that can't get a block reclaims from a chunked
    prefill parked in ce_reqs (which keeps its blocks between chunks)
    instead of raising a fatal InsufficientBlocksError: the parked
    prefill's blocks are reclaimable, so this is not a genuine OOM."""
    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=5,
        max_batch_total_tokens=None,
        enable_in_flight_batching=False,
        enable_chunked_prefill=True,
        target_tokens_per_batch_ce=30,
    )
    holder = TextContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.ones(50, dtype=np.int64)),
        max_length=100,
    )
    released: list[RequestID] = []

    def release(ctx: TextContext) -> None:
        released.append(ctx.request_id)

    def alloc(ctx: TextContext) -> CompletedTransfer:
        # Blocks free up only once the parked prefill is preempted.
        if not released:
            raise InsufficientBlocksError("insufficient blocks")
        return CompletedTransfer()

    kv_cache = Mock()
    kv_cache.chunk_alignment_tokens = 0
    kv_cache.alloc = Mock(side_effect=alloc)
    kv_cache.claim = Mock()
    kv_cache.release = Mock(side_effect=release)
    kv_cache.contains = Mock(
        side_effect=lambda ctx: (
            ctx.request_id == holder.request_id and not released
        )
    )
    kv_cache.pending_transfers_exist = Mock(return_value=False)
    kv_cache.num_req_blocks = Mock(
        side_effect=lambda ctx: 1 if ctx.request_id == holder.request_id else 0
    )
    kv_cache.get_req_blocks_per_leaf = Mock(
        side_effect=lambda ctx: {
            "full": [0] if ctx.request_id == holder.request_id else []
        }
    )
    # A full cache forces TG priority, so the parked prefill is never
    # popped by _add_ce_requests before the decode request's alloc fails.
    set_mock_kv_usage(kv_cache, 1.0)

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
        get_inflight_kv_transfer_count=lambda replica_idx: 0,
    )
    tg_context = create_lora_context(seq_len=9, is_tg=True)
    batch_constructor.enqueue_new_request(tg_context, replica_idx=0)
    batch_constructor.enqueue_new_request(holder, replica_idx=0)

    inputs = batch_constructor.construct_batch()

    assert has_request(inputs.batches[0], tg_context.request_id)
    assert released == [holder.request_id]
    assert batch_constructor.total_preemption_count == 1
    # The preempted prefill is requeued, not dropped.
    assert holder.request_id in batch_constructor.replicas[0].ce_reqs


def test_text_batch_constructor__batch_construction_with_chunked_prefill_and_preemption(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=5,
        max_batch_total_tokens=None,
        enable_in_flight_batching=False,
        enable_chunked_prefill=True,
        target_tokens_per_batch_ce=30,
    )
    kv_cache = Mock()
    kv_cache.chunk_alignment_tokens = 0
    kv_cache.alloc = Mock()
    kv_cache.alloc.return_value = CompletedTransfer()
    kv_cache.claim = Mock()
    kv_cache.contains = Mock()
    kv_cache.pending_transfers_exist = Mock(return_value=False)
    set_mock_kv_usage(kv_cache, 0.0)

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
    )

    contexts = {}
    for _ in range(8):
        context = TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(9, dtype=np.int64)),
            max_length=100,
        )
        contexts[context.request_id] = context
        batch_constructor.enqueue_new_request(context)

    assert batch_constructor._identify_priority(0) == RequestType.CE
    inputs = batch_constructor.construct_batch()
    assert len(inputs.batches[0]) == 4
    # The last request should be chunked
    assert inputs.batches[0][-1].tokens.generated_length == 0

    # Update a token for each request in the batch
    for batch in inputs.batches:
        for context in batch:
            context.update(0)

    batch_constructor.advance_requests(inputs)

    # There should now be 3 requests in TG, and 7 in CE
    assert len(batch_constructor.replicas[0].tg_reqs) == 3
    assert len(batch_constructor.replicas[0].ce_reqs) == 5

    # We should still be prioritizing CE
    assert batch_constructor._identify_priority(0) == RequestType.CE

    inputs = batch_constructor.construct_batch()
    # We only grab 2 new CE requests here, because we have 3 TG requests outstanding.
    # Since max_batch_size is 5, we can only have 5 requests outstanding at a time.
    assert len(inputs.batches[0]) == 2
    assert inputs.batches[0][-1].tokens.generated_length == 0

    for batch in inputs.batches:
        for context in batch:
            context.update(0)

    batch_constructor.advance_requests(inputs)

    assert len(batch_constructor.replicas[0].ce_reqs) == 3
    assert len(batch_constructor.replicas[0].tg_reqs) == 5

    # We still prioritize CE, but return an empty batch
    assert batch_constructor._identify_priority(0) == RequestType.CE

    # Since we generate an empty CE batch, we then fill with TG requests
    inputs = batch_constructor.construct_batch()
    assert len(inputs.batches[0]) == 5

    # Last Ce Batch
    assert batch_constructor._identify_priority(0) == RequestType.CE
    inputs = batch_constructor.construct_batch()
    # Since we already have 5 CE request outstanding, we cannot grab any new CE requests.
    assert len(inputs.batches[0]) == 5

    for batch in inputs.batches:
        for context in batch:
            context.update(0)

    batch_constructor.advance_requests(inputs)

    assert len(batch_constructor.replicas[0].ce_reqs) == 3
    assert len(batch_constructor.replicas[0].tg_reqs) == 5

    # Test for Pre-emption
    # The first item won't have enough space, so we will pre-empt the last one
    # The first item will have 2 alloc calls, failing with InsufficientBlocksError on the first,
    # then succeeding and returning 0 (no prefix cache skip) for the remaining calls.
    kv_cache.alloc.side_effect = [
        InsufficientBlocksError(),
        CompletedTransfer(),
        CompletedTransfer(),
        CompletedTransfer(),
        CompletedTransfer(),
        CompletedTransfer(),
    ]

    last_request_id = list(batch_constructor.replicas[0].tg_reqs.keys())[-1]
    assert batch_constructor._identify_priority(0) == RequestType.CE
    assert len(batch_constructor.replicas[0].ce_reqs) == 3
    inputs = batch_constructor.construct_batch()
    assert len(inputs.batches[0]) == 4
    assert all(
        context.request_id != last_request_id for context in inputs.batches[0]
    )

    # We've pre-empted the last request, so it should be in the CE queue
    assert len(batch_constructor.replicas[0].ce_reqs) == 4
    assert last_request_id in batch_constructor.replicas[0].ce_reqs
    assert len(batch_constructor.replicas[0].tg_reqs) == 4

    # Test that we can release the request
    batch_constructor.release_request(last_request_id)
    assert last_request_id not in batch_constructor.replicas[0].ce_reqs
    assert last_request_id not in batch_constructor.replicas[0].tg_reqs
    assert len(batch_constructor.replicas[0].ce_reqs) == 3
    assert len(batch_constructor.replicas[0].tg_reqs) == 4


def test_text_batch_constructor__batch_construction_with_chunked_prefill_and_inflight_batching(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=10,
        max_batch_total_tokens=None,
        enable_in_flight_batching=True,
        enable_chunked_prefill=True,
        target_tokens_per_batch_ce=30,
    )
    kv_cache = Mock()
    kv_cache.chunk_alignment_tokens = 0
    kv_cache.alloc = Mock()
    kv_cache.alloc.return_value = CompletedTransfer()
    kv_cache.claim = Mock()
    kv_cache.contains = Mock()
    set_mock_kv_usage(kv_cache, 0.0)

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
    )

    for _ in range(8):
        context = TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(9, dtype=np.int64)),
            max_length=100,
        )
        batch_constructor.enqueue_new_request(context)

    # With inflight batching, we should prioritize CE ONLY when we have no TG requests
    assert batch_constructor._identify_priority(0) == RequestType.CE
    inputs = batch_constructor.construct_batch()
    assert len(inputs.batches[0]) == 4
    assert inputs.batches[0][-1].tokens.generated_length == 0

    # Update a token for each request in the batch
    for batch in inputs.batches:
        for context in batch:
            context.update(0)

    batch_constructor.advance_requests(inputs)

    # There should now be 3 requests in TG, and 7 in CE
    assert len(batch_constructor.replicas[0].tg_reqs) == 3
    assert len(batch_constructor.replicas[0].ce_reqs) == 5

    # We should now prioritize TG
    assert batch_constructor._identify_priority(0) == RequestType.TG
    inputs = batch_constructor.construct_batch()

    # We should have 5 requests
    assert len(inputs.batches[0]) == 7
    # Last item should be chunked, with a length of 3
    assert inputs.batches[0][-1].tokens.generated_length == 0

    for batch in inputs.batches:
        for context in batch:
            context.update(0)

    batch_constructor.advance_requests(inputs)


def test_text_batch_constructor__batch_construction_without_chunked_prefill_and_inflight_batching(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=10,
        max_batch_total_tokens=None,
        enable_in_flight_batching=True,
        enable_chunked_prefill=False,
        target_tokens_per_batch_ce=30,
    )
    kv_cache = Mock()
    kv_cache.chunk_alignment_tokens = 0
    kv_cache.alloc = Mock()
    kv_cache.alloc.return_value = CompletedTransfer()
    kv_cache.claim = Mock()
    kv_cache.contains = Mock()
    set_mock_kv_usage(kv_cache, 0.0)

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
    )

    for _ in range(8):
        context = TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(9, dtype=np.int64)),
            max_length=100,
        )
        batch_constructor.enqueue_new_request(context)

    assert batch_constructor._identify_priority(0) == RequestType.CE
    inputs = batch_constructor.construct_batch()
    assert len(inputs.batches[0]) == 4
    assert inputs.batches[0][-1].tokens.generated_length == 0

    # Update a token for each request in the batch
    for batch in inputs.batches:
        for context in batch:
            context.update(0)

    batch_constructor.advance_requests(inputs)

    assert len(batch_constructor.replicas[0].ce_reqs) == 4
    assert len(batch_constructor.replicas[0].tg_reqs) == 4

    assert batch_constructor._identify_priority(0) == RequestType.TG
    inputs = batch_constructor.construct_batch()
    assert len(inputs.batches[0]) == 7
    for i in range(len(inputs.batches[0])):
        if i < 4:
            # The first four requests are TG, and should not need CE
            assert inputs.batches[0][i].tokens.generated_length != 0
        else:
            # The second four requests are CE, and should need CE
            assert inputs.batches[0][i].tokens.generated_length == 0

    for batch in inputs.batches:
        for context in batch:
            context.update(0)

    batch_constructor.advance_requests(inputs)

    assert len(batch_constructor.replicas[0].ce_reqs) == 1


def _three_decode_rows(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
    width: int = 0,
    enable_chunked_prefill: bool = True,
) -> TextBatchConstructor:
    batch_constructor = TextBatchConstructor(
        scheduler_config=TokenGenerationSchedulerConfig(
            max_batch_size=10,
            max_batch_total_tokens=None,
            enable_in_flight_batching=True,
            enable_chunked_prefill=enable_chunked_prefill,
            target_tokens_per_batch_ce=30,
            spec_decode_mixed_verify_width=width,
        ),
        pipeline=pipeline,
        kv_cache=create_mock_kv_cache(),
    )
    for _ in range(3):
        batch_constructor.enqueue_new_request(
            TextContext(
                request_id=RequestID(),
                tokens=TokenBuffer(np.ones(5, dtype=np.int64)),
                max_length=100,
            )
        )
    inputs = batch_constructor.construct_batch()
    for context in inputs.batches[0]:
        context.update(0)
    batch_constructor.advance_requests(inputs)
    assert len(batch_constructor.replicas[0].tg_reqs) == 3
    return batch_constructor


@pytest.mark.parametrize(("width", "prompt_chunk"), [(0, 27), (1, 23), (2, 19)])
def test_text_batch_constructor__mixed_step_reserves_draft_tokens(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
    width: int,
    prompt_chunk: int,
) -> None:
    """A mixed step verifies ``width`` drafts on every row, so the prompt
    gets what the 3 decode rows and its own drafts leave."""
    batch_constructor = _three_decode_rows(pipeline, width)

    prompt = TextContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.ones(50, dtype=np.int64)),
        max_length=100,
    )
    batch_constructor.enqueue_new_request(prompt)
    inputs = batch_constructor.construct_batch()

    assert len(inputs.batches[0]) == 4
    assert inputs.batches[0][-1].request_id == prompt.request_id
    assert prompt.tokens.active_length == prompt_chunk


@pytest.mark.parametrize("width", [1, 2, 3])
def test_text_batch_constructor__mixed_step_drafts_fit_budget(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
    width: int,
) -> None:
    """Prompt rows backfilled into a mixed step carry ``width`` dummy drafts
    too, so the merged verify forward must stay within the budget."""
    batch_constructor = _three_decode_rows(pipeline, width)
    for length in (5, 50):
        batch_constructor.enqueue_new_request(
            TextContext(
                request_id=RequestID(),
                tokens=TokenBuffer(np.ones(length, dtype=np.int64)),
                max_length=100,
            )
        )
    rows = batch_constructor.construct_batch().batches[0]

    assert len(rows) == 5
    merged = sum(ctx.tokens.active_length for ctx in rows) + width * len(rows)
    assert merged <= 30


@pytest.mark.parametrize("width", [1, 2, 3])
def test_text_batch_constructor__mixed_step_dp_prompt_replica_fits_budget(
    width: int,
) -> None:
    """Decode rows on replica 0 make the whole DP step verify, so the
    prompt-only replica 1 carries ``width`` dummy drafts per row too."""
    # The pipeline fixture has a LoRA manager, which DP > 1 refuses.
    pipeline = Mock(spec=["release"])
    batch_constructor = TextBatchConstructor(
        scheduler_config=TokenGenerationSchedulerConfig(
            max_batch_size=10,
            max_batch_total_tokens=None,
            enable_in_flight_batching=True,
            enable_chunked_prefill=True,
            target_tokens_per_batch_ce=30,
            data_parallel_degree=2,
            spec_decode_mixed_verify_width=width,
        ),
        pipeline=pipeline,
        kv_cache=create_mock_kv_cache(),
    )
    for _ in range(3):
        decode = TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(5, dtype=np.int64)),
            max_length=100,
        )
        decode.update(ARBITRARY_TOKEN_ID)
        batch_constructor.enqueue_new_request(decode, replica_idx=0)
    batch_constructor.enqueue_new_request(
        TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(50, dtype=np.int64)),
            max_length=100,
        ),
        replica_idx=1,
    )
    inputs = batch_constructor.construct_batch()

    assert [len(rows) for rows in inputs.batches] == [3, 1]
    for rows in inputs.batches:
        merged = sum(ctx.tokens.active_length for ctx in rows) + width * len(
            rows
        )
        assert merged <= 30


def test_text_batch_constructor__mixed_step_defers_unchunkable_prompt(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """Without chunked prefill, a prompt too long for a mixed step's room
    waits for the next step, which prefills it alone without drafts."""
    batch_constructor = _three_decode_rows(
        pipeline, width=1, enable_chunked_prefill=False
    )
    prompt = TextContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.ones(25, dtype=np.int64)),
        max_length=100,
    )
    batch_constructor.enqueue_new_request(prompt)

    inputs = batch_constructor.construct_batch()
    rows = inputs.batches[0]
    assert len(rows) == 3
    assert sum(ctx.tokens.active_length for ctx in rows) + len(rows) <= 30
    assert prompt.request_id in batch_constructor.replicas[0].ce_reqs

    for context in rows:
        context.update(0)
    batch_constructor.advance_requests(inputs)
    rows = batch_constructor.construct_batch().batches[0]
    assert [ctx.request_id for ctx in rows] == [prompt.request_id]
    assert prompt.tokens.active_length == 25


@pytest.mark.parametrize(
    ("used_fraction", "admitted"), [(0.5, True), (0.9, True), (0.95, False)]
)
def test_text_batch_constructor__inflight_backfill_kv_gate(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
    used_fraction: float,
    admitted: bool,
) -> None:
    """The in-flight backfill admits a new prompt only at or below the
    TG-priority KV threshold (90% by default)."""
    batch_constructor = _three_decode_rows(pipeline)
    set_mock_kv_usage(cast(Mock, batch_constructor.kv_cache), used_fraction)
    prompt = TextContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.ones(5, dtype=np.int64)),
        max_length=100,
    )
    batch_constructor.enqueue_new_request(prompt)
    rows = batch_constructor.construct_batch().batches[0]

    assert has_request(rows, prompt.request_id) == admitted
    assert len(rows) == 3 + admitted
    assert (prompt.request_id in batch_constructor.replicas[0].ce_reqs) != (
        admitted
    )


def test_text_batch_constructor__inflight_backfill_kv_gate_continues_chunked_prefill(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """Above the threshold, a prefill that already holds KV keeps going, and
    a new prompt queued behind it waits."""
    batch_constructor = _three_decode_rows(pipeline)
    kv_cache = cast(Mock, batch_constructor.kv_cache)
    chunked = TextContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.ones(50, dtype=np.int64)),
        max_length=100,
    )
    batch_constructor.enqueue_new_request(chunked)
    inputs = batch_constructor.construct_batch()
    for context in inputs.batches[0]:
        context.update(0)
    batch_constructor.advance_requests(inputs)
    assert next(iter(batch_constructor.replicas[0].ce_reqs)) == (
        chunked.request_id
    )

    new = TextContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.ones(5, dtype=np.int64)),
        max_length=100,
    )
    batch_constructor.enqueue_new_request(new)
    set_mock_kv_usage(kv_cache, 0.95)
    kv_cache.contains = Mock(
        side_effect=lambda ctx: ctx.request_id == chunked.request_id
    )
    kv_cache.num_req_blocks = Mock(
        side_effect=lambda ctx: 2 if ctx.request_id == chunked.request_id else 0
    )
    rows = batch_constructor.construct_batch().batches[0]

    assert has_request(rows, chunked.request_id)
    assert not has_request(rows, new.request_id)
    assert new.request_id in batch_constructor.replicas[0].ce_reqs


def test_text_batch_constructor__inflight_backfill_kv_gate_holds_a_claim_without_blocks(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """Above the threshold, being claimed is not the same as holding KV.

    A request readmitted from a connector lookup cordon is claimed but its
    alloc has taken nothing yet, so letting it through would draw the
    blocks the running generations need next. It goes back to the head of
    the queue rather than being allocated, and keeps its claim, because
    releasing it would free nothing and throw away its answer and the
    counts that bound how often it asks again. The answer stays under the
    hand back rule that applies to any request passed over.
    """
    batch_constructor = _three_decode_rows(pipeline)
    kv_cache = cast(Mock, batch_constructor.kv_cache)
    readmitted = TextContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.ones(5, dtype=np.int64)),
        max_length=100,
    )
    batch_constructor.enqueue_new_request(readmitted)
    set_mock_kv_usage(kv_cache, 0.95)
    kv_cache.contains = Mock(
        side_effect=lambda ctx: ctx.request_id == readmitted.request_id
    )
    kv_cache.num_req_blocks = Mock(return_value=0)
    kv_cache.awaits_lookup = Mock(
        side_effect=lambda ctx: ctx.request_id == readmitted.request_id
    )
    kv_cache.alloc.reset_mock()

    rows = batch_constructor.construct_batch().batches[0]

    assert not has_request(rows, readmitted.request_id)
    assert next(iter(batch_constructor.replicas[0].ce_reqs)) == (
        readmitted.request_id
    )
    kv_cache.release.assert_not_called()
    kv_cache.hand_back_lookup.assert_called_once_with(readmitted)
    assert all(
        alloc_call.args[0] is not readmitted
        for alloc_call in kv_cache.alloc.call_args_list
    )


def test_text_batch_constructor__kv_gate_spares_prefill_only_step(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """With no decode rows to drain, in-flight batching still runs prefill
    above the threshold, as the production path does."""
    kv_cache = create_mock_kv_cache()
    set_mock_kv_usage(kv_cache, 0.95)
    batch_constructor = TextBatchConstructor(
        scheduler_config=TokenGenerationSchedulerConfig(
            max_batch_size=10,
            max_batch_total_tokens=None,
            enable_in_flight_batching=True,
            enable_chunked_prefill=True,
            target_tokens_per_batch_ce=30,
        ),
        pipeline=pipeline,
        kv_cache=kv_cache,
    )
    prompt = TextContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.ones(5, dtype=np.int64)),
        max_length=100,
    )
    batch_constructor.enqueue_new_request(prompt)
    rows = batch_constructor.construct_batch().batches[0]

    assert [ctx.request_id for ctx in rows] == [prompt.request_id]


def test_text_batch_constructor__advance_requests_requeues_every_chunked_request(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """A chunked request is requeued wherever it sits in the batch, not just last.

    The CE token budget can only chunk the last request admitted to a batch, so
    the batch a per-request prefill cap would produce is built by hand here.
    """
    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=5,
        max_batch_total_tokens=None,
        enable_in_flight_batching=False,
        enable_chunked_prefill=True,
        target_tokens_per_batch_ce=30,
    )
    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=create_mock_kv_cache(),
    )

    contexts = [
        TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(9, dtype=np.int64)),
            max_length=100,
        )
        for _ in range(3)
    ]
    mid_prefill_a, mid_prefill_b, finished = contexts
    for context in contexts:
        batch_constructor.enqueue_new_request(context, replica_idx=0)

    replica = batch_constructor.replicas[0]
    # Stands in for _add_ce_requests draining the queue into the batch.
    replica.ce_reqs.clear()
    finished.update(ARBITRARY_TOKEN_ID)

    batch_constructor.advance_requests(
        TextGenerationInputs(batches=[[mid_prefill_a, mid_prefill_b, finished]])
    )

    assert list(replica.ce_reqs) == [
        mid_prefill_a.request_id,
        mid_prefill_b.request_id,
    ]
    assert list(replica.tg_reqs) == [finished.request_id]


def _prefill_cap_constructor(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
    max_request_input_tokens: int,
) -> TextBatchConstructor:
    return TextBatchConstructor(
        scheduler_config=TokenGenerationSchedulerConfig(
            max_batch_size=5,
            max_batch_total_tokens=None,
            enable_in_flight_batching=False,
            enable_chunked_prefill=True,
            target_tokens_per_batch_ce=1000,
            max_request_input_tokens=max_request_input_tokens,
        ),
        pipeline=pipeline,
        kv_cache=create_mock_kv_cache(),
    )


def _one_long_then_short_requests() -> tuple[TextContext, list[TextContext]]:
    long_request = TextContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.ones(1000, dtype=np.int64)),
        max_length=2000,
    )
    short_requests = [
        TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
            max_length=100,
        )
        for _ in range(3)
    ]
    return long_request, short_requests


def test_text_batch_constructor__prefill_cap_leaves_room_for_short_requests(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    batch_constructor = _prefill_cap_constructor(pipeline, 256)
    long_request, short_requests = _one_long_then_short_requests()
    for context in [long_request, *short_requests]:
        batch_constructor.enqueue_new_request(context, replica_idx=0)

    batch = batch_constructor.construct_batch().batches[0]

    assert long_request.tokens.active_length == 256
    assert [ctx.request_id for ctx in batch] == [
        long_request.request_id,
        *(ctx.request_id for ctx in short_requests),
    ]


def test_text_batch_constructor__long_request_takes_the_whole_batch_uncapped(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    batch_constructor = _prefill_cap_constructor(pipeline, 0)
    long_request, short_requests = _one_long_then_short_requests()
    for context in [long_request, *short_requests]:
        batch_constructor.enqueue_new_request(context, replica_idx=0)

    batch = batch_constructor.construct_batch().batches[0]

    assert long_request.tokens.active_length == 1000
    assert [ctx.request_id for ctx in batch] == [long_request.request_id]
    assert list(batch_constructor.replicas[0].ce_reqs) == [
        ctx.request_id for ctx in short_requests
    ]


def test_single_lora_scheduling() -> None:
    """Test scheduling a single LoRA request in CE batch."""
    lora_manager = create_mock_lora_manager(max_num_loras=2)
    pipeline = create_mock_pipeline_with_lora(lora_manager)
    kv_cache = create_mock_kv_cache()

    config = TokenGenerationSchedulerConfig(
        max_batch_size=4,
        target_tokens_per_batch_ce=100,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=config,
        pipeline=pipeline,
        kv_cache=kv_cache,
    )

    ctx = create_lora_context(model_name="lora_model1")
    batch_constructor.enqueue_new_request(ctx)

    output = batch_constructor.construct_batch()

    assert len(output.batches[0]) == 1
    assert has_request(output.batches[0], ctx.request_id)
    lora_manager.activate_adapter.assert_called_once_with("lora_model1")
    assert "lora_model1" in lora_manager._active_loras


def test_multi_lora_within_budget() -> None:
    """Test scheduling multiple LoRA requests within budget."""
    lora_manager = create_mock_lora_manager(max_num_loras=3)
    pipeline = create_mock_pipeline_with_lora(lora_manager)
    kv_cache = create_mock_kv_cache()

    config = TokenGenerationSchedulerConfig(
        max_batch_size=4,
        target_tokens_per_batch_ce=200,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=config,
        pipeline=pipeline,
        kv_cache=kv_cache,
    )

    ctx1 = create_lora_context(model_name="lora_model1")
    ctx2 = create_lora_context(model_name="lora_model2")
    ctx3 = create_lora_context(model_name="lora_model3")

    batch_constructor.enqueue_new_request(ctx1)
    batch_constructor.enqueue_new_request(ctx2)
    batch_constructor.enqueue_new_request(ctx3)

    output = batch_constructor.construct_batch()
    assert len(output.batches[0]) == 3
    assert has_request(output.batches[0], ctx1.request_id)
    assert has_request(output.batches[0], ctx2.request_id)
    assert has_request(output.batches[0], ctx3.request_id)
    assert len(lora_manager._active_loras) == 3


def test_lora_preemption_over_budget() -> None:
    """Test that LoRA requests are deferred when over budget during CE."""
    lora_manager = create_mock_lora_manager(max_num_loras=2)
    pipeline = create_mock_pipeline_with_lora(lora_manager)
    kv_cache = create_mock_kv_cache()

    config = TokenGenerationSchedulerConfig(
        max_batch_size=5,
        target_tokens_per_batch_ce=200,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=config,
        pipeline=pipeline,
        kv_cache=kv_cache,
    )

    ctx_lora1 = create_lora_context(model_name="lora_model1")
    ctx_lora2 = create_lora_context(model_name="lora_model2")
    ctx_lora3 = create_lora_context(model_name="lora_model3")
    ctx_base = create_lora_context(model_name=None)

    batch_constructor.enqueue_new_request(ctx_lora1)
    batch_constructor.enqueue_new_request(ctx_lora2)
    batch_constructor.enqueue_new_request(ctx_lora3)
    batch_constructor.enqueue_new_request(ctx_base)

    output = batch_constructor.construct_batch()

    assert len(output.batches[0]) == 3
    assert has_request(output.batches[0], ctx_base.request_id)
    assert has_request(output.batches[0], ctx_lora1.request_id)
    assert has_request(output.batches[0], ctx_lora2.request_id)
    assert ctx_lora3.request_id not in output.batches[0]

    assert ctx_lora3.request_id in batch_constructor.all_ce_reqs


def test_age_based_scheduling_with_lora() -> None:
    """Test that age-based scheduling is maintained with LoRA constraints."""
    lora_manager = create_mock_lora_manager(max_num_loras=2)
    pipeline = create_mock_pipeline_with_lora(lora_manager)
    kv_cache = create_mock_kv_cache()

    config = TokenGenerationSchedulerConfig(
        max_batch_size=4,
        target_tokens_per_batch_ce=40,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=config,
        pipeline=pipeline,
        kv_cache=kv_cache,
    )

    lora_manager._active_loras.add("lora_model2")

    ctx_inactive = create_lora_context(model_name="lora_model1")
    ctx_base = create_lora_context(model_name=None)
    ctx_active = create_lora_context(model_name="lora_model2")

    batch_constructor.enqueue_new_request(ctx_inactive)
    batch_constructor.enqueue_new_request(ctx_base)
    batch_constructor.enqueue_new_request(ctx_active)

    output = batch_constructor.construct_batch()

    assert len(output.batches[0]) == 2
    assert has_request(output.batches[0], ctx_inactive.request_id)
    assert has_request(output.batches[0], ctx_base.request_id)


def test_tg_batch_with_active_loras() -> None:
    """Test that TG batch correctly handles requests with active LoRAs."""
    lora_manager = create_mock_lora_manager(max_num_loras=2)
    pipeline = create_mock_pipeline_with_lora(lora_manager)
    kv_cache = create_mock_kv_cache()

    config = TokenGenerationSchedulerConfig(
        max_batch_size=5,
        target_tokens_per_batch_ce=100,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=config,
        pipeline=pipeline,
        kv_cache=kv_cache,
    )

    lora_manager._active_loras.add("lora_model1")
    lora_manager._active_loras.add("lora_model2")

    ctx_active1 = create_lora_context(model_name="lora_model1", is_tg=True)
    ctx_active2 = create_lora_context(model_name="lora_model2", is_tg=True)
    ctx_base = create_lora_context(model_name=None, is_tg=True)

    batch_constructor.enqueue_new_request(ctx_active1)
    batch_constructor.enqueue_new_request(ctx_active2)
    batch_constructor.enqueue_new_request(ctx_base)

    output = batch_constructor.construct_batch()

    assert len(output.batches[0])
    assert has_request(output.batches[0], ctx_active1.request_id)
    assert has_request(output.batches[0], ctx_active2.request_id)
    assert has_request(output.batches[0], ctx_base.request_id)


def test_ce_lora_activation_within_budget() -> None:
    """Test that LoRAs are activated during CE when within budget."""
    lora_manager = create_mock_lora_manager(max_num_loras=3)
    pipeline = create_mock_pipeline_with_lora(lora_manager)
    kv_cache = create_mock_kv_cache()

    config = TokenGenerationSchedulerConfig(
        max_batch_size=4,
        target_tokens_per_batch_ce=100,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=config,
        pipeline=pipeline,
        kv_cache=kv_cache,
    )

    ctx_lora1 = create_lora_context(model_name="lora_model1")
    ctx_lora2 = create_lora_context(model_name="lora_model2")

    batch_constructor.enqueue_new_request(ctx_lora1)
    batch_constructor.enqueue_new_request(ctx_lora2)

    output = batch_constructor.construct_batch()

    assert len(output.batches[0]) == 2
    assert has_request(output.batches[0], ctx_lora1.request_id)
    assert has_request(output.batches[0], ctx_lora2.request_id)

    assert "lora_model1" in lora_manager._active_loras
    assert "lora_model2" in lora_manager._active_loras


def test_tg_pure_age_based_preemption() -> None:
    """Test that preemption is purely age-based for KV cache constraints."""
    lora_manager = create_mock_lora_manager(max_num_loras=3)
    pipeline = create_mock_pipeline_with_lora(lora_manager)
    kv_cache = create_mock_kv_cache()

    kv_cache.alloc = Mock(
        side_effect=[
            CompletedTransfer(),
            InsufficientBlocksError,
            InsufficientBlocksError,
        ]
    )

    config = TokenGenerationSchedulerConfig(
        max_batch_size=4,
        target_tokens_per_batch_ce=100,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=config,
        pipeline=pipeline,
        kv_cache=kv_cache,
    )

    lora_manager._active_loras.add("lora_model1")
    lora_manager._active_loras.add("lora_model2")

    ctx1 = create_lora_context(model_name="lora_model1", is_tg=True)
    ctx2 = create_lora_context(model_name="lora_model2", is_tg=True)
    ctx3 = create_lora_context(model_name=None, is_tg=True)

    batch_constructor.enqueue_new_request(ctx1)
    batch_constructor.enqueue_new_request(ctx2)
    batch_constructor.enqueue_new_request(ctx3)

    output = batch_constructor.construct_batch()

    assert len(output.batches[0]) == 1
    assert has_request(output.batches[0], ctx1.request_id)
    pipeline.release.assert_called()


def test_lora_swapping_ce_to_tg() -> None:
    """Test LoRA remains active when moving from CE to TG."""
    lora_manager = create_mock_lora_manager(max_num_loras=2)
    pipeline = create_mock_pipeline_with_lora(lora_manager)
    kv_cache = create_mock_kv_cache()

    config = TokenGenerationSchedulerConfig(
        max_batch_size=4,
        target_tokens_per_batch_ce=100,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=config,
        pipeline=pipeline,
        kv_cache=kv_cache,
    )

    ctx = create_lora_context(model_name="lora_model1")
    batch_constructor.enqueue_new_request(ctx)

    batch_constructor.construct_batch()
    assert "lora_model1" in lora_manager._active_loras

    ctx.update(29)
    batch_constructor.enqueue_new_request(ctx)

    ctx2 = create_lora_context(model_name="lora_model2")
    batch_constructor.enqueue_new_request(ctx2)

    batch_constructor.construct_batch()
    assert "lora_model2" in lora_manager._active_loras

    ctx2.update(29)
    batch_constructor.enqueue_new_request(ctx2)

    tg_output = batch_constructor.construct_batch()

    assert has_request(tg_output.batches[0], ctx.request_id)
    assert has_request(tg_output.batches[0], ctx2.request_id)


def test_mixed_requests_scheduling() -> None:
    """Test scheduling with mixed LoRA and base model requests."""
    lora_manager = create_mock_lora_manager(max_num_loras=1)
    pipeline = create_mock_pipeline_with_lora(lora_manager)
    kv_cache = create_mock_kv_cache()

    config = TokenGenerationSchedulerConfig(
        max_batch_size=4,
        target_tokens_per_batch_ce=100,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=config,
        pipeline=pipeline,
        kv_cache=kv_cache,
    )

    ctx_lora1 = create_lora_context(model_name="lora_model1")
    ctx_lora2 = create_lora_context(model_name="lora_model2")
    ctx_base1 = create_lora_context(model_name=None)
    ctx_base2 = create_lora_context(model_name=None)

    batch_constructor.enqueue_new_request(ctx_lora1)
    batch_constructor.enqueue_new_request(ctx_lora2)
    batch_constructor.enqueue_new_request(ctx_base1)
    batch_constructor.enqueue_new_request(ctx_base2)

    output = batch_constructor.construct_batch()

    assert len(output.batches[0]) == 3
    assert has_request(output.batches[0], ctx_base1.request_id)
    assert has_request(output.batches[0], ctx_base2.request_id)
    assert has_request(output.batches[0], ctx_lora1.request_id) or (
        has_request(output.batches[0], ctx_lora2.request_id)
    )

    assert len(lora_manager._active_loras) == 1


def test_text_batch_constructor__load_based_replica_assignment_with_kv_cache() -> (
    None
):
    """Test that load-based assignment distributes requests evenly across replicas.

    This is the core test to catch bugs like [2,1,1,1,1,1,1,0] instead of [1,1,1,1,1,1,1,1].
    """
    data_parallel_degree = 8
    num_requests = 8

    # Create a pipeline without LoRA support
    pipeline = Mock(spec=["release"])
    pipeline.release = Mock()

    # Create paged cache
    kv_cache = create_mock_kv_cache()
    kv_cache.num_replicas = data_parallel_degree

    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=10,
        target_tokens_per_batch_ce=1000,
        data_parallel_degree=data_parallel_degree,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
    )

    # Enqueue requests - with load-based assignment, all should go to least loaded
    for _ in range(num_requests):
        context = TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
            max_length=100,
        )
        batch_constructor.enqueue_new_request(context)

    # Count requests per replica
    requests_per_replica = [
        len(batch_constructor.replicas[i].ce_reqs)
        for i in range(data_parallel_degree)
    ]

    # With load-based assignment, distribution should be balanced
    # Each replica should have 1 request (8 requests, 8 replicas)
    expected_distribution = [1, 1, 1, 1, 1, 1, 1, 1]
    assert requests_per_replica == expected_distribution, (
        f"Expected distribution {expected_distribution}, got {requests_per_replica}"
    )


def test_text_batch_constructor__data_parallel_explicit_replica_assignment() -> (
    None
):
    """Test explicit replica_idx assignment used by decode_scheduler.

    This tests the code path where replica_idx is explicitly passed, ensuring
    requests go to the correct replica.
    """
    data_parallel_degree = 8

    # Create a pipeline without LoRA support
    pipeline = Mock(spec=["release"])
    pipeline.release = Mock()

    # Create paged cache (required but not used for explicit assignment)
    kv_cache = create_mock_kv_cache()
    kv_cache.num_replicas = data_parallel_degree
    kv_cache.get_replica_request_count = Mock(return_value=0)

    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=10,
        target_tokens_per_batch_ce=1000,
        data_parallel_degree=data_parallel_degree,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
    )

    # Enqueue one request to each replica explicitly
    for replica_idx in range(data_parallel_degree):
        context = TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
            max_length=100,
        )
        batch_constructor.enqueue_new_request(context, replica_idx=replica_idx)

    # Count requests per replica
    requests_per_replica = [
        len(batch_constructor.replicas[i].ce_reqs)
        for i in range(data_parallel_degree)
    ]

    # Each replica should have exactly 1 request
    expected_distribution = [1, 1, 1, 1, 1, 1, 1, 1]
    assert requests_per_replica == expected_distribution, (
        f"Expected distribution {expected_distribution}, got {requests_per_replica}"
    )


def test_text_batch_constructor__load_based_handles_imbalance() -> None:
    """Test that load-based assignment prioritizes least loaded replicas.

    This test creates an imbalanced load scenario and verifies that new
    requests are assigned to the replica with the fewest active requests.
    """
    data_parallel_degree = 4

    pipeline = Mock(spec=["release"])
    pipeline.release = Mock()

    # Create paged cache
    kv_cache = create_mock_kv_cache()
    kv_cache.num_replicas = data_parallel_degree

    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=10,
        target_tokens_per_batch_ce=1000,
        data_parallel_degree=data_parallel_degree,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
    )

    # Create an imbalanced initial load: [5, 2, 8, 1]
    # Replica 0: 5 requests, Replica 1: 2 requests, Replica 2: 8 requests, Replica 3: 1 request
    for replica_idx, count in enumerate([5, 2, 8, 1]):
        for _ in range(count):
            context = TextContext(
                request_id=RequestID(),
                tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
                max_length=100,
            )
            batch_constructor.enqueue_new_request(
                context, replica_idx=replica_idx
            )

    # Track request counts before adding new requests
    requests_before = [
        len(batch_constructor.replicas[i].ce_reqs)
        for i in range(data_parallel_degree)
    ]
    assert requests_before == [5, 2, 8, 1], (
        f"Initial load should be [5, 2, 8, 1], got {requests_before}"
    )

    # Enqueue 4 new requests without specifying replica_idx
    for _ in range(4):
        context = TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
            max_length=100,
        )
        batch_constructor.enqueue_new_request(context)

    # Count requests per replica after adding new requests
    requests_after = [
        len(batch_constructor.replicas[i].ce_reqs)
        for i in range(data_parallel_degree)
    ]

    # Replica 3 had the lowest load (1), so it should receive the first new request → [5, 2, 8, 2]
    # Replica 1 now has the lowest load (2), so it should receive the second new request → [5, 3, 8, 2]
    # Replica 3 now tied for lowest (2), so it should receive the third new request → [5, 3, 8, 3]
    # Replica 1 now tied for lowest (3), so it should receive the fourth new request → [5, 4, 8, 3]
    assert requests_after == [5, 4, 8, 3]


def test_batch_scheduling_strategy__per_replica_default() -> None:
    """Test PER_REPLICA strategy (default) allows independent replica decisions."""
    from max.serve.scheduler.batch_constructor.text_batch_constructor import (
        BatchSchedulingStrategy,
    )

    data_parallel_degree = 3
    pipeline = Mock(spec=["release"])
    pipeline.release = Mock()
    kv_cache = create_mock_kv_cache()

    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=10,
        target_tokens_per_batch_ce=100,
        data_parallel_degree=data_parallel_degree,
        enable_in_flight_batching=False,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
        batch_scheduling_strategy=BatchSchedulingStrategy.PER_REPLICA,
    )

    # Replica 0: 2 CE requests (should prioritize CE)
    # Replica 1: 2 TG requests (should prioritize TG)
    # Replica 2: 1 CE + 1 TG (should prioritize CE with enable_in_flight_batching=False)

    # Add CE requests to replica 0
    for _ in range(2):
        ctx = TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
            max_length=100,
        )
        batch_constructor.enqueue_new_request(ctx, replica_idx=0)

    # Add TG requests to replica 1
    for _ in range(2):
        ctx = TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
            max_length=100,
        )
        ctx.update(ARBITRARY_TOKEN_ID)
        batch_constructor.enqueue_new_request(ctx, replica_idx=1)

    # Add mixed requests to replica 2
    ctx_ce = TextContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
        max_length=100,
    )
    batch_constructor.enqueue_new_request(ctx_ce, replica_idx=2)

    ctx_tg = TextContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
        max_length=100,
    )
    ctx_tg.update(ARBITRARY_TOKEN_ID)
    batch_constructor.enqueue_new_request(ctx_tg, replica_idx=2)

    # Verify each replica identifies priority independently
    assert batch_constructor._identify_priority(0) == RequestType.CE
    assert batch_constructor._identify_priority(1) == RequestType.TG
    assert batch_constructor._identify_priority(2) == RequestType.CE

    # Construct batch
    inputs = batch_constructor.construct_batch()

    # Replica 0 should have CE batch
    assert len(inputs.batches[0]) == 2
    assert all(ctx.tokens.generated_length == 0 for ctx in inputs.batches[0])

    # Replica 1 should have TG batch
    assert len(inputs.batches[1]) == 2
    assert all(ctx.tokens.generated_length > 0 for ctx in inputs.batches[1])

    # Replica 2 should have CE batch (prioritizes CE when enable_in_flight_batching=False)
    assert len(inputs.batches[2]) == 1
    assert inputs.batches[2][0].tokens.generated_length == 0


def test_batch_scheduling_strategy__prefill_first() -> None:
    """Test PREFILL_FIRST strategy forces all replicas to prioritize CE."""
    from max.serve.scheduler.batch_constructor.text_batch_constructor import (
        BatchSchedulingStrategy,
    )

    data_parallel_degree = 3
    pipeline = Mock(spec=["release"])
    pipeline.release = Mock()
    kv_cache = create_mock_kv_cache()

    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=10,
        target_tokens_per_batch_ce=100,
        data_parallel_degree=data_parallel_degree,
        enable_in_flight_batching=False,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
        batch_scheduling_strategy=BatchSchedulingStrategy.PREFILL_FIRST,
    )

    # Replica 0: 2 CE requests
    # Replica 1: 2 TG requests
    # Replica 2: 1 CE + 1 TG

    for _ in range(2):
        ctx = TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
            max_length=100,
        )
        batch_constructor.enqueue_new_request(ctx, replica_idx=0)

    for _ in range(2):
        ctx = TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
            max_length=100,
        )
        ctx.update(ARBITRARY_TOKEN_ID)
        batch_constructor.enqueue_new_request(ctx, replica_idx=1)

    ctx_ce = TextContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
        max_length=100,
    )
    batch_constructor.enqueue_new_request(ctx_ce, replica_idx=2)

    ctx_tg = TextContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
        max_length=100,
    )
    ctx_tg.update(ARBITRARY_TOKEN_ID)
    batch_constructor.enqueue_new_request(ctx_tg, replica_idx=2)

    # Construct batch
    inputs = batch_constructor.construct_batch()

    # All replicas should prioritize CE since PREFILL_FIRST and CE work exists
    # Replica 0: CE batch
    assert len(inputs.batches[0]) == 2
    assert all(ctx.tokens.generated_length == 0 for ctx in inputs.batches[0])

    # Replica 1: Should be empty or have TG (no CE requests)
    assert len(inputs.batches[1]) == 0

    # Replica 2: CE batch
    assert len(inputs.batches[2]) == 1
    assert inputs.batches[2][0].tokens.generated_length == 0


def test_batch_scheduling_strategy__decode_first() -> None:
    """Test DECODE_FIRST strategy forces all replicas to prioritize TG."""
    from max.serve.scheduler.batch_constructor.text_batch_constructor import (
        BatchSchedulingStrategy,
    )

    data_parallel_degree = 3
    pipeline = Mock(spec=["release"])
    pipeline.release = Mock()
    kv_cache = create_mock_kv_cache()

    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=10,
        target_tokens_per_batch_ce=100,
        data_parallel_degree=data_parallel_degree,
        enable_in_flight_batching=True,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
        batch_scheduling_strategy=BatchSchedulingStrategy.DECODE_FIRST,
    )

    # Replica 0: 2 CE requests
    # Replica 1: 2 TG requests
    # Replica 2: 1 CE + 1 TG

    for _ in range(2):
        ctx = TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
            max_length=100,
        )
        batch_constructor.enqueue_new_request(ctx, replica_idx=0)

    for _ in range(2):
        ctx = TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
            max_length=100,
        )
        ctx.update(ARBITRARY_TOKEN_ID)
        batch_constructor.enqueue_new_request(ctx, replica_idx=1)

    ctx_ce = TextContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
        max_length=100,
    )
    batch_constructor.enqueue_new_request(ctx_ce, replica_idx=2)

    ctx_tg = TextContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
        max_length=100,
    )
    ctx_tg.update(ARBITRARY_TOKEN_ID)
    batch_constructor.enqueue_new_request(ctx_tg, replica_idx=2)

    # Construct batch
    inputs = batch_constructor.construct_batch()

    # All replicas should prioritize TG since DECODE_FIRST and TG work exists
    # Replica 0: Should be empty (no TG requests)
    assert len(inputs.batches[0]) == 0

    # Replica 1: TG batch
    assert len(inputs.batches[1]) == 2
    assert all(ctx.tokens.generated_length > 0 for ctx in inputs.batches[1])

    # Replica 2: TG batch (with possible CE fill due to enable_in_flight_batching)
    assert len(inputs.batches[2]) >= 1
    assert inputs.batches[2][0].tokens.generated_length > 0


def test_batch_scheduling_strategy__decode_first_idle_replica_does_not_starve_sibling() -> (
    None
):
    """Test an idle replica can't veto a sibling's real CE request (SERVOPT-1560).

    Regression test for a livelock: DECODE_FIRST computed a batch-wide
    priority override from the set of every replica's _identify_priority()
    result. A replica with zero CE *and* zero TG requests used to default to
    RequestType.TG (indistinguishable from a replica that genuinely needs
    TG), so its phantom vote alone could force priority_override=TG for
    every replica -- including a sibling with a real, ready CE request and
    zero TG requests, whose CE admission would then be skipped every single
    iteration forever, with no exception and no error log.
    """
    from max.serve.scheduler.batch_constructor.text_batch_constructor import (
        BatchSchedulingStrategy,
    )

    data_parallel_degree = 2
    pipeline = Mock(spec=["release"])
    pipeline.release = Mock()
    kv_cache = create_mock_kv_cache()

    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=10,
        target_tokens_per_batch_ce=100,
        data_parallel_degree=data_parallel_degree,
        enable_in_flight_batching=False,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
        batch_scheduling_strategy=BatchSchedulingStrategy.DECODE_FIRST,
    )

    # Replica 0: completely idle (no CE, no TG requests).
    # Replica 1: one real, ready CE request, no TG requests -- matches the
    # exact captured state at max-concurrency=1: ce_reqs=[0, 1] tg_reqs=[0, 0].
    ctx_ce = TextContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
        max_length=100,
    )
    batch_constructor.enqueue_new_request(ctx_ce, replica_idx=1)

    # An idle replica has no preference at all, not a TG preference.
    assert batch_constructor._identify_priority(0) is None
    assert batch_constructor._identify_priority(1) == RequestType.CE

    inputs = batch_constructor.construct_batch()

    # Replica 0 has nothing to do either way.
    assert len(inputs.batches[0]) == 0

    # Replica 1's CE request must be admitted -- it must not be starved by
    # replica 0's idleness being mistaken for a TG override.
    assert len(inputs.batches[1]) == 1
    assert inputs.batches[1][0].tokens.generated_length == 0


def test_ce_preempts_pending_tg_emits_iteration_stealing_metric() -> None:
    """A CE batch that admits work while TG requests are pending on the
    same replica steals the whole iteration from TG (the CE case's TG
    fallback only runs when CE admits nothing). Verify the
    iteration-stealing counter and gauge fire in that case.
    """
    from max.serve.scheduler.batch_constructor.text_batch_constructor import (
        BatchSchedulingStrategy,
    )

    pipeline = Mock(spec=["release"])
    pipeline.release = Mock()
    kv_cache = create_mock_kv_cache()

    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=10,
        target_tokens_per_batch_ce=100,
        data_parallel_degree=1,
        enable_in_flight_batching=False,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
        batch_scheduling_strategy=BatchSchedulingStrategy.PREFILL_FIRST,
    )

    # One TG request already in flight, plus a fresh CE request --
    # PREFILL_FIRST forces CE priority for the whole replica, so CE wins
    # this iteration and the TG request gets skipped entirely.
    ctx_tg = TextContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
        max_length=100,
    )
    ctx_tg.update(ARBITRARY_TOKEN_ID)
    batch_constructor.enqueue_new_request(ctx_tg, replica_idx=0)

    ctx_ce = TextContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
        max_length=100,
    )
    batch_constructor.enqueue_new_request(ctx_ce, replica_idx=0)

    with patch(
        "max.serve.scheduler.batch_constructor.text_batch_constructor.METRICS"
    ) as mock_metrics:
        inputs = batch_constructor.construct_batch()

    assert len(inputs.batches[0]) == 1
    assert inputs.batches[0][0].tokens.generated_length == 0
    mock_metrics.di_ce_preempted_tg_iteration_count.assert_called_once()
    mock_metrics.di_ce_preempted_tg_pending_count.assert_called_once_with(1)


def test_batch_scheduling_strategy__balanced_majority_ce() -> None:
    """Test BALANCED strategy prioritizes CE when CE is the majority."""
    from max.serve.scheduler.batch_constructor.text_batch_constructor import (
        BatchSchedulingStrategy,
    )

    data_parallel_degree = 3
    pipeline = Mock(spec=["release"])
    pipeline.release = Mock()
    kv_cache = create_mock_kv_cache()

    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=10,
        target_tokens_per_batch_ce=100,
        data_parallel_degree=data_parallel_degree,
        enable_in_flight_batching=False,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
        batch_scheduling_strategy=BatchSchedulingStrategy.BALANCED,
    )

    # Replica 0: CE priority (2 CE requests)
    # Replica 1: CE priority (2 CE requests)
    # Replica 2: TG priority (2 TG requests)
    # Majority: CE (2 CE vs 1 TG)

    for replica_idx in [0, 1]:
        for _ in range(2):
            ctx = TextContext(
                request_id=RequestID(),
                tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
                max_length=100,
            )
            batch_constructor.enqueue_new_request(ctx, replica_idx=replica_idx)

    for _ in range(2):
        ctx = TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
            max_length=100,
        )
        ctx.update(ARBITRARY_TOKEN_ID)
        batch_constructor.enqueue_new_request(ctx, replica_idx=2)

    # Verify individual priorities
    assert batch_constructor._identify_priority(0) == RequestType.CE
    assert batch_constructor._identify_priority(1) == RequestType.CE
    assert batch_constructor._identify_priority(2) == RequestType.TG

    # Construct batch - should prioritize CE globally
    inputs = batch_constructor.construct_batch()

    # Replicas 0 and 1 should have CE batches
    assert len(inputs.batches[0]) == 2
    assert all(ctx.tokens.generated_length == 0 for ctx in inputs.batches[0])

    assert len(inputs.batches[1]) == 2
    assert all(ctx.tokens.generated_length == 0 for ctx in inputs.batches[1])

    # Replica 2 should be empty (forced to CE but has no CE requests)
    assert len(inputs.batches[2]) == 0


def test_batch_scheduling_strategy__balanced_majority_tg() -> None:
    """Test BALANCED strategy prioritizes TG when TG is the majority."""
    from max.serve.scheduler.batch_constructor.text_batch_constructor import (
        BatchSchedulingStrategy,
    )

    data_parallel_degree = 3
    pipeline = Mock(spec=["release"])
    pipeline.release = Mock()
    kv_cache = create_mock_kv_cache()

    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=10,
        target_tokens_per_batch_ce=100,
        data_parallel_degree=data_parallel_degree,
        enable_in_flight_batching=True,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
        batch_scheduling_strategy=BatchSchedulingStrategy.BALANCED,
    )

    # Replica 0: CE priority (2 CE requests)
    # Replica 1: TG priority (2 TG requests)
    # Replica 2: TG priority (2 TG requests)
    # Majority: TG (1 CE vs 2 TG)

    for _ in range(2):
        ctx = TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
            max_length=100,
        )
        batch_constructor.enqueue_new_request(ctx, replica_idx=0)

    for replica_idx in [1, 2]:
        for _ in range(2):
            ctx = TextContext(
                request_id=RequestID(),
                tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
                max_length=100,
            )
            ctx.update(ARBITRARY_TOKEN_ID)
            batch_constructor.enqueue_new_request(ctx, replica_idx=replica_idx)

    # Verify individual priorities
    assert batch_constructor._identify_priority(0) == RequestType.CE
    assert batch_constructor._identify_priority(1) == RequestType.TG
    assert batch_constructor._identify_priority(2) == RequestType.TG

    # Construct batch - should prioritize TG globally
    inputs = batch_constructor.construct_batch()

    # Replica 0 should be empty (forced to TG but has no TG requests)
    assert len(inputs.batches[0]) == 0

    # Replicas 1 and 2 should have TG batches
    assert len(inputs.batches[1]) == 2
    assert all(ctx.tokens.generated_length > 0 for ctx in inputs.batches[1])

    assert len(inputs.batches[2]) == 2
    assert all(ctx.tokens.generated_length > 0 for ctx in inputs.batches[2])


def test_batch_scheduling_strategy__balanced_tie_defaults_to_tg() -> None:
    """Test BALANCED strategy defaults to TG when CE and TG counts are equal."""
    from max.serve.scheduler.batch_constructor.text_batch_constructor import (
        BatchSchedulingStrategy,
    )

    data_parallel_degree = 4
    pipeline = Mock(spec=["release"])
    pipeline.release = Mock()
    kv_cache = create_mock_kv_cache()

    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=10,
        target_tokens_per_batch_ce=100,
        data_parallel_degree=data_parallel_degree,
        enable_in_flight_batching=True,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
        batch_scheduling_strategy=BatchSchedulingStrategy.BALANCED,
    )

    # Replica 0: CE priority
    # Replica 1: CE priority
    # Replica 2: TG priority
    # Replica 3: TG priority
    # Tie: 2 CE vs 2 TG -> should default to TG

    for replica_idx in [0, 1]:
        ctx = TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
            max_length=100,
        )
        batch_constructor.enqueue_new_request(ctx, replica_idx=replica_idx)

    for replica_idx in [2, 3]:
        ctx = TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
            max_length=100,
        )
        ctx.update(ARBITRARY_TOKEN_ID)
        batch_constructor.enqueue_new_request(ctx, replica_idx=replica_idx)

    # Verify individual priorities
    assert batch_constructor._identify_priority(0) == RequestType.CE
    assert batch_constructor._identify_priority(1) == RequestType.CE
    assert batch_constructor._identify_priority(2) == RequestType.TG
    assert batch_constructor._identify_priority(3) == RequestType.TG

    # Construct batch - should default to TG on tie
    inputs = batch_constructor.construct_batch()

    # Replicas 0 and 1 should be empty (forced to TG but have no TG requests)
    assert len(inputs.batches[0]) == 0
    assert len(inputs.batches[1]) == 0

    # Replicas 2 and 3 should have TG batches
    assert len(inputs.batches[2]) == 1
    assert inputs.batches[2][0].tokens.generated_length > 0

    assert len(inputs.batches[3]) == 1
    assert inputs.batches[3][0].tokens.generated_length > 0


def test_batch_scheduling_strategy__all_replicas_empty() -> None:
    """Test that all strategies handle the case where all replicas are empty."""
    from max.serve.scheduler.batch_constructor.text_batch_constructor import (
        BatchSchedulingStrategy,
    )

    data_parallel_degree = 2
    pipeline = Mock(spec=["release"])
    pipeline.release = Mock()
    kv_cache = create_mock_kv_cache()

    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=10,
        target_tokens_per_batch_ce=100,
        data_parallel_degree=data_parallel_degree,
    )

    for strategy in [
        BatchSchedulingStrategy.PER_REPLICA,
        BatchSchedulingStrategy.PREFILL_FIRST,
        BatchSchedulingStrategy.DECODE_FIRST,
        BatchSchedulingStrategy.BALANCED,
    ]:
        batch_constructor = TextBatchConstructor(
            scheduler_config=scheduler_config,
            pipeline=pipeline,
            kv_cache=kv_cache,
            batch_scheduling_strategy=strategy,
        )

        inputs = batch_constructor.construct_batch()

        # All batches should be empty
        assert len(inputs.batches) == data_parallel_degree
        assert all(len(batch) == 0 for batch in inputs.batches)


# ---------------------------------------------------------------------------
# DP-balanced CE scheduling (_plan_ce_step) tests
# ---------------------------------------------------------------------------


def create_dp_balance_constructor(
    dp: int = 2,
    timeout_ms: float = 10_000.0,
    threshold: float = 0.8,
    enable_dynamic_chunk_size: bool = True,
    hit_counts: list[PrefixCacheHits] | None = None,
    max_batch_total_tokens: int | None = None,
    prefill_schedule_interval: int = 1,
    enable_in_flight_batching: bool = False,
    spec_decode_mixed_verify_width: int = 0,
    chunked_prefill_min_chunk_size: int = 0,
) -> TextBatchConstructor:
    """A DP constructor with the CE balancer on and a stubbed cache probe."""
    pipeline = Mock(spec=["release"])
    pipeline.release = Mock()
    kv_cache = create_mock_kv_cache()
    kv_cache.params.page_size = 16
    kv_cache.get_prefix_cache_hit_counts = Mock(
        return_value=(
            hit_counts if hit_counts is not None else [PrefixCacheHits()] * dp
        )
    )
    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=10,
        target_tokens_per_batch_ce=100,
        data_parallel_degree=dp,
        dp_ce_balance_timeout_ms=timeout_ms,
        dp_ce_balance_threshold=threshold,
        dp_ce_balance_enable_dynamic_chunk_size=enable_dynamic_chunk_size,
        max_batch_total_tokens=max_batch_total_tokens,
        prefill_schedule_interval=prefill_schedule_interval,
        enable_in_flight_batching=enable_in_flight_batching,
        spec_decode_mixed_verify_width=spec_decode_mixed_verify_width,
        chunked_prefill_min_chunk_size=chunked_prefill_min_chunk_size,
    )
    return TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
    )


def test_dp_ce_balance__disabled_binds_on_arrival() -> None:
    """timeout_ms=-1 (default) disables pooling: arrival binds immediately."""
    batch_constructor = create_dp_balance_constructor(timeout_ms=-1.0)
    ctx = create_lora_context()
    batch_constructor.enqueue_new_request(ctx)
    assert not batch_constructor._ce_pending
    assert any(
        ctx.request_id in replica.ce_reqs
        for replica in batch_constructor.replicas
    )


def test_dp_ce_balance__pools_new_requests_and_binds_when_fleet_idle() -> None:
    batch_constructor = create_dp_balance_constructor()
    ctx = create_lora_context()
    batch_constructor.enqueue_new_request(ctx)

    # Pooled: tracked by the constructor but bound to no replica queue.
    assert batch_constructor.contains(ctx.request_id)
    assert ctx.request_id in batch_constructor._ce_pending
    assert all(not replica.ce_reqs for replica in batch_constructor.replicas)

    # The fleet has nothing else to run, so the planner must not defer: the
    # request binds and is scheduled this very step.
    inputs = batch_constructor.construct_batch()
    assert has_request(inputs.batches[0] + inputs.batches[1], ctx.request_id)
    assert not batch_constructor._ce_pending


def test_dp_ce_balance__binding_records_pool_wait() -> None:
    batch_constructor = create_dp_balance_constructor()
    ctx = create_lora_context()
    batch_constructor.enqueue_new_request(ctx)
    batch_constructor._ce_pending[ctx.request_id].entered_at -= 0.5

    with patch(
        "max.serve.scheduler.batch_constructor.text_batch_constructor.METRICS"
    ) as mock_metrics:
        batch_constructor.construct_batch()

    mock_metrics.ce_pool_wait_time.assert_called_once()
    (wait_ms,) = mock_metrics.ce_pool_wait_time.call_args.args
    assert 500 <= wait_ms < 1000


def test_dp_ce_balance__pooled_request_prefers_replica_with_cached_prefix() -> (
    None
):
    hit_counts = [PrefixCacheHits(), PrefixCacheHits(device_blocks=4)]
    batch_constructor = create_dp_balance_constructor(hit_counts=hit_counts)
    ctx = create_lora_context(seq_len=96)
    batch_constructor.enqueue_new_request(ctx)

    # Weighted at post-prefix-cache length: 96 tokens raw, minus 4 blocks
    # (x 16-token pages) resident on replica 1.
    assert batch_constructor._ce_pending[ctx.request_id].weights == [96, 32]

    inputs = batch_constructor.construct_batch()
    assert has_request(inputs.batches[1], ctx.request_id)


def test_dp_ce_balance__legacy_manager_prices_each_replicas_device_run() -> (
    None
):
    page_size = 16
    block_manager = BlockManager(
        total_num_blocks=32,
        block_size=page_size,
        enable_prefix_caching=True,
        num_replicas=2,
    )
    ctx = create_lora_context(seq_len=96)
    hashes = compute_block_hashes(
        ctx,
        [],
        page_size,
        block_manager.kv_hash_algo,
        block_manager.kv_hash_seed,
    )
    # Unequal device runs: replica 1 holds a block more than replica 0.
    for replica_idx, depth in ((0, 1), (1, 2)):
        pool = block_manager.device_block_pools[replica_idx]
        for block_hash in hashes[:depth]:
            block, _ = pool.alloc_block()
            pool.commit_into_prefix_cache(block_hash, block)
    cache = cast(
        PagedKVCacheManager,
        SimpleNamespace(
            params=SimpleNamespace(enable_prefix_caching=True),
            _replica=[SimpleNamespace(block_manager=block_manager)] * 2,
        ),
    )

    hits = PagedKVCacheManager.get_prefix_cache_hit_counts(cache, ctx)

    assert hits == [
        PrefixCacheHits(device_blocks=1),
        PrefixCacheHits(device_blocks=2),
    ]
    batch_constructor = create_dp_balance_constructor()
    _mock_kv_cache_of(batch_constructor).get_prefix_cache_hit_counts = Mock(
        side_effect=lambda c: PagedKVCacheManager.get_prefix_cache_hit_counts(
            cache, c
        )
    )
    batch_constructor.enqueue_new_request(ctx)
    assert batch_constructor._ce_pending[ctx.request_id].weights == [80, 64]


def test_dp_ce_balance__defers_lone_unexpired_ce_when_tg_available() -> None:
    batch_constructor = create_dp_balance_constructor(threshold=0.8)
    tg_ctx = create_lora_context(is_tg=True)
    batch_constructor.enqueue_new_request(tg_ctx, replica_idx=0)
    ce_ctx = create_lora_context(seq_len=50)
    batch_constructor.enqueue_new_request(ce_ctx, replica_idx=0)
    # Deadline budget left, as if the request had been pooled on arrival.
    batch_constructor._ce_arrival[ce_ctx.request_id] = time.monotonic()

    # Occupancy would be 50/(2*50) = 0.5 < 0.8 with no partner CE anywhere,
    # so replica 0's CE work is held and it runs TG instead.
    inputs = batch_constructor.construct_batch()
    assert batch_constructor._ce_deferred_replicas == {0}
    assert has_request(inputs.batches[0], tg_ctx.request_id)
    assert not has_request(inputs.batches[0], ce_ctx.request_id)


def test_dp_ce_balance__expired_ce_runs_despite_imbalance() -> None:
    batch_constructor = create_dp_balance_constructor(timeout_ms=10_000.0)
    tg_ctx = create_lora_context(is_tg=True)
    batch_constructor.enqueue_new_request(tg_ctx, replica_idx=0)
    ce_ctx = create_lora_context(seq_len=50)
    batch_constructor.enqueue_new_request(ce_ctx, replica_idx=0)
    batch_constructor._ce_arrival[ce_ctx.request_id] = time.monotonic() - 60.0

    # Same imbalance as the deferral test, but the deadline is blown: the CE
    # work joins the floor and runs.
    inputs = batch_constructor.construct_batch()
    assert batch_constructor._ce_deferred_replicas == set()
    assert has_request(inputs.batches[0], ce_ctx.request_id)


def test_dp_ce_balance__no_deferral_without_tg_work() -> None:
    batch_constructor = create_dp_balance_constructor()
    ce_ctx = create_lora_context(seq_len=50)
    batch_constructor.enqueue_new_request(ce_ctx, replica_idx=0)
    batch_constructor._ce_arrival[ce_ctx.request_id] = time.monotonic()

    # Replica 0 has no TG to run instead; deferring would idle it, so its CE
    # work is not deferrable even with deadline budget left.
    inputs = batch_constructor.construct_batch()
    assert batch_constructor._ce_deferred_replicas == set()
    assert has_request(inputs.batches[0], ce_ctx.request_id)


def test_dp_ce_balance__balanced_ce_across_replicas_schedules() -> None:
    batch_constructor = create_dp_balance_constructor(threshold=0.8)
    ce_ctxs = []
    for replica_idx in range(2):
        tg_ctx = create_lora_context(is_tg=True)
        batch_constructor.enqueue_new_request(tg_ctx, replica_idx=replica_idx)
        ce_ctx = create_lora_context(seq_len=50)
        batch_constructor.enqueue_new_request(ce_ctx, replica_idx=replica_idx)
        batch_constructor._ce_arrival[ce_ctx.request_id] = time.monotonic()
        ce_ctxs.append(ce_ctx)

    # 50 tokens on each rank is a perfectly balanced step: occupancy 1.0
    # meets the threshold and everything runs, nothing is deferred.
    inputs = batch_constructor.construct_batch()
    assert batch_constructor._ce_deferred_replicas == set()
    assert has_request(inputs.batches[0], ce_ctxs[0].request_id)
    assert has_request(inputs.batches[1], ce_ctxs[1].request_id)


def test_dp_ce_balance__pool_fills_step_when_no_replica_has_tg() -> None:
    batch_constructor = create_dp_balance_constructor()
    ce_ctxs = [create_lora_context(seq_len=20) for _ in range(8)]
    for ctx in ce_ctxs:
        batch_constructor.enqueue_new_request(ctx)

    # A prefill-only worker never has TG work. Once the replicas are level,
    # no single equal-sized request strictly improves occupancy, but holding
    # the rest back would only idle the fleet: every request runs now.
    inputs = batch_constructor.construct_batch()
    assert not batch_constructor._ce_pending
    assert [len(batch) for batch in inputs.batches] == [4, 4]


def test_dp_ce_balance__pool_waits_for_occupancy_gain_when_tg_runs() -> None:
    batch_constructor = create_dp_balance_constructor()
    for replica_idx in range(2):
        tg_ctx = create_lora_context(is_tg=True)
        batch_constructor.enqueue_new_request(tg_ctx, replica_idx=replica_idx)
    for _ in range(4):
        batch_constructor.enqueue_new_request(create_lora_context(seq_len=20))

    # With TG to run instead, pooled work still binds only where it improves
    # occupancy: one request per replica levels the step, the rest wait.
    batch_constructor.construct_batch()
    assert len(batch_constructor._ce_pending) == 2


def test_dp_ce_balance__release_pooled_request() -> None:
    batch_constructor = create_dp_balance_constructor()
    ctx = create_lora_context()
    batch_constructor.enqueue_new_request(ctx)
    assert batch_constructor.contains(ctx.request_id)

    # Releasing a pooled request (e.g. client cancellation) must work even
    # though it was never bound to a replica or claimed in the KV cache.
    batch_constructor.release_request(ctx.request_id)
    assert not batch_constructor.contains(ctx.request_id)
    assert isinstance(batch_constructor.pipeline, Mock)
    batch_constructor.pipeline.release.assert_called_once_with(ctx.request_id)

    inputs = batch_constructor.construct_batch()
    assert all(len(batch) == 0 for batch in inputs.batches)


def _add_deferrable_ce(
    batch_constructor: TextBatchConstructor, replica_idx: int, seq_len: int
) -> TextContext:
    """A TG request plus an unexpired mid-CE request pinned to a replica."""
    tg_ctx = create_lora_context(is_tg=True)
    batch_constructor.enqueue_new_request(tg_ctx, replica_idx=replica_idx)
    ce_ctx = create_lora_context(seq_len=seq_len)
    batch_constructor.enqueue_new_request(ce_ctx, replica_idx=replica_idx)
    batch_constructor._ce_arrival[ce_ctx.request_id] = time.monotonic()
    return ce_ctx


def test_dp_ce_balance__reduces_chunk_size_to_balance_level() -> None:
    batch_constructor = create_dp_balance_constructor(threshold=0.9)
    heavy_ce = _add_deferrable_ce(batch_constructor, replica_idx=0, seq_len=96)
    light_ce = _add_deferrable_ce(batch_constructor, replica_idx=1, seq_len=60)

    # Occupancy (96+60)/(2*96) = 0.81 misses the 0.9 threshold, but both
    # replicas have CE work and the balance level (60) is at least half the
    # 100-token chunk target, so the step runs at a 60-token chunk size:
    # the heavy request is chunked at its quota and only its excess defers.
    # The light replica already sits at the level, so it is not cut and
    # keeps no quota (0 = uncapped).
    inputs = batch_constructor.construct_batch()
    assert batch_constructor._ce_step_quota == [60, 0]
    assert batch_constructor._ce_deferred_replicas == set()
    assert has_request(inputs.batches[0], heavy_ce.request_id)
    assert heavy_ce.tokens.active_length == 60
    assert has_request(inputs.batches[1], light_ce.request_id)


def test_dp_ce_balance__dynamic_chunk_size_disabled_holds_step() -> None:
    batch_constructor = create_dp_balance_constructor(
        threshold=0.9, enable_dynamic_chunk_size=False
    )
    heavy_ce = _add_deferrable_ce(batch_constructor, replica_idx=0, seq_len=96)
    light_ce = _add_deferrable_ce(batch_constructor, replica_idx=1, seq_len=60)

    # Same step as above, but with dynamic chunk sizing off it is held whole.
    inputs = batch_constructor.construct_batch()
    assert batch_constructor._ce_step_quota is None
    assert batch_constructor._ce_deferred_replicas == {0, 1}
    assert not has_request(inputs.batches[0], heavy_ce.request_id)
    assert not has_request(inputs.batches[1], light_ce.request_id)


def test_dp_ce_balance__no_chunk_size_reduction_below_half_target() -> None:
    batch_constructor = create_dp_balance_constructor()
    _add_deferrable_ce(batch_constructor, replica_idx=0, seq_len=96)
    _add_deferrable_ce(batch_constructor, replica_idx=1, seq_len=30)

    # The balance level (30) is under half the 100-token chunk target:
    # chunks that small cost more in extra steps than the imbalance they
    # avoid, so the work is held instead.
    batch_constructor.construct_batch()
    assert batch_constructor._ce_step_quota is None
    assert batch_constructor._ce_deferred_replicas == {0, 1}


def test_dp_ce_balance__quota_never_below_floor() -> None:
    batch_constructor = create_dp_balance_constructor(dp=3, threshold=0.9)
    # Replica 0's CE deadline is blown: it is floor work that runs to the
    # full chunk budget, and the balance level cannot drop below it.
    tg_ctx = create_lora_context(is_tg=True)
    batch_constructor.enqueue_new_request(tg_ctx, replica_idx=0)
    expired_ce = create_lora_context(seq_len=70)
    batch_constructor.enqueue_new_request(expired_ce, replica_idx=0)
    batch_constructor._ce_arrival[expired_ce.request_id] = (
        time.monotonic() - 60.0
    )
    heavy_ce = _add_deferrable_ce(batch_constructor, replica_idx=1, seq_len=96)
    _add_deferrable_ce(batch_constructor, replica_idx=2, seq_len=55)

    # Occupancy (70+96+55)/(3*96) = 0.77 misses the threshold. The lightest
    # replica sits at 55, but the cut stops at the 70-token floor, and only
    # the replica above it is cut.
    inputs = batch_constructor.construct_batch()
    assert batch_constructor._ce_step_quota == [0, 70, 0]
    assert batch_constructor._ce_deferred_replicas == set()
    assert has_request(inputs.batches[1], heavy_ce.request_id)
    assert heavy_ce.tokens.active_length == 70


@pytest.mark.parametrize(
    ("heavy_len", "light_len", "heavy_chunk"), [(96, 40, 40), (80, 60, 80)]
)
def test_dp_ce_balance__min_chunk_size_bounds_the_cut(
    heavy_len: int, light_len: int, heavy_chunk: int
) -> None:
    """The min chunk size bounds both the balance level and the remainder.

    A 40-token level is under half the 100-token target but not under the
    30-token min chunk, so the heavy replica is cut to it. Cutting 80 tokens
    to 60 would leave 20, which the budget's min-chunk floor would refuse or
    move, so that step runs uncut.
    """
    batch_constructor = create_dp_balance_constructor(
        threshold=0.9, chunked_prefill_min_chunk_size=30
    )
    heavy_ce = _add_deferrable_ce(batch_constructor, 0, heavy_len)
    _add_deferrable_ce(batch_constructor, 1, light_len)

    batch_constructor.construct_batch()

    cut = heavy_chunk if heavy_chunk < heavy_len else 0
    assert batch_constructor._ce_step_quota == [cut, 0]
    assert heavy_ce.tokens.active_length == heavy_chunk


def test_dp_ce_balance__pooled_request_binds_at_its_cut_size() -> None:
    """A pooled request heavier than the step binds, cut to the balance level.

    Priced at its full 96 tokens it would lower the step's occupancy and wait
    until it expired to run alone. Priced at the 25-token cut it balances the
    step, which still needs the quota that makes the cut.
    """
    batch_constructor = create_dp_balance_constructor(
        dp=4, chunked_prefill_min_chunk_size=25
    )
    for replica_idx in range(3):
        _add_deferrable_ce(batch_constructor, replica_idx, 25)
    pooled_ctx = create_lora_context(seq_len=96)
    batch_constructor.enqueue_new_request(pooled_ctx)

    inputs = batch_constructor.construct_batch()

    assert has_request(inputs.batches[3], pooled_ctx.request_id)
    assert pooled_ctx.tokens.active_length == 25
    assert batch_constructor._ce_step_quota == [0, 0, 0, 25]


def test_dp_ce_balance__quota_caps_prefill_under_in_flight_batching() -> None:
    """The CE quota holds on a mixed step without truncating its decode rows."""
    batch_constructor = create_dp_balance_constructor(
        threshold=0.9,
        enable_in_flight_batching=True,
        spec_decode_mixed_verify_width=2,
    )
    heavy_ce = _add_deferrable_ce(batch_constructor, replica_idx=0, seq_len=96)
    light_ce = _add_deferrable_ce(batch_constructor, replica_idx=1, seq_len=60)

    inputs = batch_constructor.construct_batch()

    assert batch_constructor._ce_step_quota == [60, 0]
    # Each replica runs its decode row and its prompt.
    assert [len(batch) for batch in inputs.batches] == [2, 2]
    assert heavy_ce.tokens.active_length == 60
    assert light_ce.tokens.active_length == 60


@pytest.mark.parametrize("timeout_ms", [0.0, 10_000.0])
def test_dp_ce_balance__bind_tie_goes_to_the_fewest_queued_requests(
    timeout_ms: float,
) -> None:
    """A bind tied on CE tokens goes to the replica with the fewest requests.

    With the new request, replicas 0-2 tie at 70 tokens. Counting decode and
    prompt requests (2, 3, 1) picks replica 2; counting either alone picks
    replica 0 or 1. Timeout 0 binds it as expired, a deadline from the pool.
    """
    batch_constructor = create_dp_balance_constructor(
        dp=4, timeout_ms=timeout_ms
    )
    for replica_idx, num_tg, seq_lens in (
        (0, 0, (25, 25)),
        (1, 2, (50,)),
        (2, 0, (50,)),
        (3, 0, (100,)),
    ):
        for _ in range(num_tg):
            batch_constructor.enqueue_new_request(
                create_lora_context(is_tg=True), replica_idx=replica_idx
            )
        for seq_len in seq_lens:
            batch_constructor.enqueue_new_request(
                create_lora_context(seq_len=seq_len), replica_idx=replica_idx
            )
    ctx = create_lora_context(seq_len=20)
    batch_constructor.enqueue_new_request(ctx)

    inputs = batch_constructor.construct_batch()

    assert has_request(inputs.batches[2], ctx.request_id)


def _long_ce_context(seq_len: int) -> TextContext:
    """A CE context whose prompt may exceed ``create_lora_context``'s cap."""
    return TextContext(
        request_id=RequestID(),
        max_length=seq_len + 64,
        tokens=TokenBuffer(np.ones(seq_len, dtype=np.int64)),
    )


def _mock_kv_cache_of(batch_constructor: TextBatchConstructor) -> Mock:
    """The stubbed cache behind a ``create_dp_balance_constructor``."""
    kv_cache = batch_constructor.kv_cache
    assert isinstance(kv_cache, Mock)
    return kv_cache


def _stub_per_request_hits(
    batch_constructor: TextBatchConstructor,
    hits: dict[RequestID, list[PrefixCacheHits]],
) -> None:
    """Varies the cache probe per request, unlike the shared stub.

    A request with no entry probes as uncached on every replica.
    """
    uncached = [PrefixCacheHits()] * batch_constructor.num_replicas
    _mock_kv_cache_of(batch_constructor).get_prefix_cache_hit_counts = Mock(
        side_effect=lambda ctx: hits.get(ctx.request_id, uncached)
    )


def test_dp_ce_balance__expired_bind_prices_the_whole_replica_queue() -> None:
    """A replica taking an expired bind is priced with its queue, not the bind.

    The expired request joins a replica that already holds a deferrable tail,
    and the replica runs both: the bind makes it non-deferrable, and
    ``_add_ce_requests`` pops the tail first. Pricing the replica at the
    expired request alone leaves it looking like it still has room for
    another chunk, so later pooled work is weighed against a step that does
    not exist.
    """
    batch_constructor = create_dp_balance_constructor()
    tg_ctx = create_lora_context(is_tg=True)
    batch_constructor.enqueue_new_request(tg_ctx, replica_idx=0)
    # A tail that fills replica 0's whole 100-token chunk budget on its own.
    tail_ctx = _long_ce_context(100)
    batch_constructor.enqueue_new_request(tail_ctx, replica_idx=0)
    batch_constructor._ce_arrival[tail_ctx.request_id] = time.monotonic()
    batch_constructor.enqueue_new_request(
        create_lora_context(is_tg=True), replica_idx=1
    )

    # Both pooled requests are almost entirely cached on replica 0 (192 of
    # 200 tokens), so both price cheapest there.
    expired_ctx = _long_ce_context(200)
    rider_ctx = _long_ce_context(200)
    _stub_per_request_hits(
        batch_constructor,
        {
            ctx.request_id: [
                PrefixCacheHits(device_blocks=12),
                PrefixCacheHits(),
            ]
            for ctx in (expired_ctx, rider_ctx)
        },
    )
    batch_constructor.enqueue_new_request(expired_ctx)
    batch_constructor._ce_arrival[expired_ctx.request_id] = (
        time.monotonic() - 60.0
    )
    batch_constructor.enqueue_new_request(rider_ctx)

    inputs = batch_constructor.construct_batch()

    # Replica 0 is full at its chunk budget, so the rider goes to the idle
    # replica. Priced at the expired request's 8 tokens instead, replica 0
    # still looks open, the rider is compared against that phantom step, and
    # it is dropped for failing to improve an occupancy it never had.
    assert has_request(inputs.batches[1], rider_ctx.request_id)
    assert has_request(inputs.batches[0], tail_ctx.request_id)
    assert batch_constructor._ce_deferred_replicas == set()


def test_dp_ce_balance__expired_request_waits_when_nothing_can_seat_it() -> (
    None
):
    """An expired request stays pooled when no replica can admit it.

    Binding it to a replica that seats no prefill this step runs nothing and
    costs the request its place in the pool: it can no longer be placed by
    price next step, and it is stuck behind that replica's decode queue.
    """
    batch_constructor = create_dp_balance_constructor()
    for replica_idx in range(2):
        _saturate_decode(batch_constructor, replica_idx=replica_idx)

    ce_ctx = create_lora_context(seq_len=48)
    batch_constructor.enqueue_new_request(ce_ctx)
    batch_constructor._ce_arrival[ce_ctx.request_id] = time.monotonic() - 60.0

    batch_constructor.construct_batch()

    assert ce_ctx.request_id in batch_constructor._ce_pending
    assert ce_ctx.request_id not in batch_constructor._bound_requests


def test_dp_ce_balance__mid_prefill_work_is_priced_by_its_live_window() -> None:
    """Once a chunk has run, the remaining window is exact and wins.

    ``kv_cache.alloc`` advances the active window past the prefix-cache hit,
    so from the first chunk onward ``active_length`` is already post-cache.
    Holding on to the estimate taken at bind time would keep pricing the
    request at its original prompt for the rest of its prefill.
    """
    batch_constructor = create_dp_balance_constructor(threshold=0.9)
    for replica_idx, seq_len in ((0, 160), (1, 60)):
        batch_constructor.enqueue_new_request(
            create_lora_context(is_tg=True), replica_idx=replica_idx
        )
        ce_ctx = _long_ce_context(seq_len)
        batch_constructor.enqueue_new_request(ce_ctx, replica_idx=replica_idx)
        batch_constructor._ce_arrival[ce_ctx.request_id] = time.monotonic()
        if replica_idx == 0:
            # 100 tokens already processed: 60 of real work left, exactly
            # matching replica 1.
            ce_ctx.tokens.skip_processing(100)

    batch_constructor.construct_batch()

    # Priced live the step is 60/60: occupancy 1.0 clears the 0.9 threshold
    # and no chunk-size reduction is needed. Priced at the bind-time 160 it
    # reads as 100/60 and both replicas are cut to a 60-token chunk.
    assert batch_constructor._ce_step_quota is None
    assert batch_constructor._ce_deferred_replicas == set()


def test_dp_ce_balance__binding_does_not_reprobe_the_prefix_cache() -> None:
    """The planner already knows a pooled request's weight; binding reuses it.

    The probe hashes the request's whole prompt, so re-running it at the bind
    doubles that cost on the scheduler's hot path for no new information.
    """
    batch_constructor = create_dp_balance_constructor()
    ce_ctx = create_lora_context(seq_len=48)
    batch_constructor.enqueue_new_request(ce_ctx)

    inputs = batch_constructor.construct_batch()

    assert has_request(inputs.batches[0], ce_ctx.request_id)
    probe = _mock_kv_cache_of(batch_constructor).get_prefix_cache_hit_counts
    assert probe.call_count == 1


def test_dp_ce_balance__off_cadence_step_plans_nothing() -> None:
    """A step that cannot admit prefill anywhere must not plan any.

    ``_add_ce_requests`` returns immediately while the prefill cadence is
    closed, so every replica seats nothing. Planning against it would bind
    pooled work and set quotas for a step that runs no CE at all.
    """
    batch_constructor = create_dp_balance_constructor(
        prefill_schedule_interval=2
    )
    batch_constructor.enqueue_new_request(
        create_lora_context(is_tg=True), replica_idx=0
    )
    # Opens on phase 0, closed on the next step.
    batch_constructor.construct_batch()

    ce_ctx = create_lora_context(seq_len=48)
    batch_constructor.enqueue_new_request(ce_ctx)
    batch_constructor._ce_arrival[ce_ctx.request_id] = time.monotonic() - 60.0
    batch_constructor.construct_batch()

    assert not batch_constructor._prefill_interval_open
    assert ce_ctx.request_id in batch_constructor._ce_pending
    assert batch_constructor._ce_deferred_replicas == set()
    assert batch_constructor._ce_step_quota is None


def test_dp_ce_balance__bound_requests_are_priced_post_prefix_cache() -> None:
    """Bound-but-unrun CE work is priced the way the pool is priced.

    ``active_length`` stays at the full prompt until ``kv_cache.alloc``
    advances the window past the prefix-cache hit, and the per-replica total
    is clamped to the chunk target -- so a single long, almost-entirely-cached
    request reads as a full chunk and hides a perfectly balanced step.
    """
    # 96 of replica 0's 156 prompt tokens are already cached: 60 tokens of
    # real work, exactly matching replica 1.
    batch_constructor = create_dp_balance_constructor(
        threshold=0.9,
        hit_counts=[PrefixCacheHits(device_blocks=6), PrefixCacheHits()],
    )
    for replica_idx, seq_len in ((0, 156), (1, 60)):
        tg_ctx = create_lora_context(is_tg=True)
        batch_constructor.enqueue_new_request(tg_ctx, replica_idx=replica_idx)
        ce_ctx = _long_ce_context(seq_len)
        batch_constructor.enqueue_new_request(ce_ctx, replica_idx=replica_idx)
        batch_constructor._ce_arrival[ce_ctx.request_id] = time.monotonic()

    batch_constructor.construct_batch()

    # Priced post-cache the step is 60/60: occupancy 1.0 clears the 0.9
    # threshold and no chunk-size reduction is needed. Priced pre-cache it
    # reads as 100/60 and the chunk size is cut to 60 for no reason.
    assert batch_constructor._ce_step_quota is None
    assert batch_constructor._ce_deferred_replicas == set()


def test_dp_ce_balance__pool_wait_restarts_when_a_preempted_request_returns() -> (
    None
):
    batch_constructor = create_dp_balance_constructor()
    ctx = create_lora_context(seq_len=50)
    batch_constructor.enqueue_new_request(ctx, replica_idx=0)
    # Arrival is the deferral deadline and survives the re-pool on purpose.
    batch_constructor._ce_arrival[ctx.request_id] = time.monotonic() - 60.0

    batch_constructor._preempt_request(
        ctx, 0, reason=PreemptionReason.KV_CACHE_MEMORY
    )
    with patch(
        "max.serve.scheduler.batch_constructor.text_batch_constructor.METRICS"
    ) as mock_metrics:
        batch_constructor.construct_batch()

    # Only the second stay in the pool counts, not the first wait plus the
    # encoding that ran before the preemption.
    mock_metrics.ce_pool_wait_time.assert_called_once()
    (wait_ms,) = mock_metrics.ce_pool_wait_time.call_args.args
    assert 0 <= wait_ms < 1000


def test_dp_ce_balance__a_cordon_wait_survives_the_return_to_the_pool() -> None:
    """A request preempted back to the pool keeps the wait it started.

    Without DP balancing it is rebound and keeps it, so with balancing the
    same preemption must not drop the wait or restart it at a later cordon.
    """
    batch_constructor = create_dp_balance_constructor()
    ctx = create_lora_context(seq_len=50)
    batch_constructor.enqueue_new_request(ctx, replica_idx=0)
    batch_constructor._ce_arrival[ctx.request_id] = time.monotonic()
    batch_constructor._cordon_waits[ctx.request_id] = (
        text_batch_constructor._CordonWait(
            since=time.monotonic() - 0.5, pass_idx=0
        )
    )

    batch_constructor._preempt_request(
        ctx, 0, reason=PreemptionReason.KV_CACHE_MEMORY
    )
    assert ctx.request_id in batch_constructor._ce_pending
    with patch(
        "max.serve.scheduler.batch_constructor.text_batch_constructor.METRICS"
    ) as mock_metrics:
        inputs = batch_constructor.construct_batch()

    assert has_request(inputs.batches[0] + inputs.batches[1], ctx.request_id)
    mock_metrics.connector_cordon_wait_time.assert_called_once()
    (wait_ms,) = mock_metrics.connector_cordon_wait_time.call_args.args
    assert 500 <= wait_ms < 1000


def test_dp_ce_balance__releasing_a_pooled_request_drops_its_cordon_wait() -> (
    None
):
    batch_constructor = create_dp_balance_constructor()
    ctx = create_lora_context(seq_len=50)
    batch_constructor.enqueue_new_request(ctx, replica_idx=0)
    batch_constructor._ce_arrival[ctx.request_id] = time.monotonic()
    batch_constructor._cordon_waits[ctx.request_id] = (
        text_batch_constructor._CordonWait(since=time.monotonic(), pass_idx=0)
    )
    batch_constructor._preempt_request(
        ctx, 0, reason=PreemptionReason.KV_CACHE_MEMORY
    )

    batch_constructor.release_request(ctx.request_id)

    assert not batch_constructor._cordon_waits


def test_dp_ce_balance__preempted_request_returns_to_the_pool() -> None:
    """A preempted pool-managed request must not poison its old replica.

    ``_return_to_request_queue`` resets the request but keeps its original
    arrival on record. Left bound, that stale arrival reads as expired, and
    because a replica is deferrable only when none of its CE work has
    expired, one preempted request makes the whole replica non-deferrable
    for as long as it sits there.
    """
    batch_constructor = create_dp_balance_constructor(threshold=0.8)
    tg_ctx = create_lora_context(is_tg=True)
    batch_constructor.enqueue_new_request(tg_ctx, replica_idx=0)
    fresh_ctx = create_lora_context(seq_len=50)
    batch_constructor.enqueue_new_request(fresh_ctx, replica_idx=0)
    batch_constructor._ce_arrival[fresh_ctx.request_id] = time.monotonic()
    preempted_ctx = create_lora_context(seq_len=50)
    batch_constructor.enqueue_new_request(preempted_ctx, replica_idx=0)
    batch_constructor._ce_arrival[preempted_ctx.request_id] = (
        time.monotonic() - 60.0
    )

    batch_constructor._preempt_request(
        preempted_ctx, 0, reason=PreemptionReason.KV_CACHE_MEMORY
    )

    assert preempted_ctx.request_id in batch_constructor._ce_pending
    assert preempted_ctx.request_id not in batch_constructor._bound_requests
    assert preempted_ctx.request_id not in batch_constructor.replicas[0].ce_reqs
    assert batch_constructor.contains(preempted_ctx.request_id)

    # Unbound again, it rebinds where it now prices cheapest -- replica 1,
    # which is idle -- instead of queueing behind replica 0's tail and
    # dragging that tail into the floor with it.
    inputs = batch_constructor.construct_batch()
    assert has_request(inputs.batches[1], preempted_ctx.request_id)
    assert has_request(inputs.batches[0], fresh_ctx.request_id)

    # The re-pool moved the request's bookkeeping rather than duplicating it,
    # so a full admit/preempt/complete cycle leaves nothing behind.
    batch_constructor.release_request(preempted_ctx.request_id)
    assert not batch_constructor.contains(preempted_ctx.request_id)
    for tracker in (
        batch_constructor._bound_requests,
        batch_constructor._ce_pending,
        batch_constructor._ce_arrival,
        batch_constructor._request_id_to_lora_name,
    ):
        assert preempted_ctx.request_id not in tracker


def _saturate_decode(
    batch_constructor: TextBatchConstructor, replica_idx: int
) -> None:
    """Fills a replica's decode queue to ``max_batch_size``.

    ``_add_ce_requests`` reserves a batch slot for every queued generation,
    so at this depth the replica can seat no prefill at all.
    """
    for _ in range(batch_constructor.scheduler_config.max_batch_size):
        batch_constructor.enqueue_new_request(
            create_lora_context(is_tg=True), replica_idx=replica_idx
        )


def test_dp_ce_balance__pool_skips_replica_with_no_admission_seat() -> None:
    """A pooled request must bind where admission can actually seat it.

    The planner prices a replica by its queued CE tokens, but admission also
    reserves a batch slot for each of that replica's queued generations. A
    replica whose decode queue already fills ``max_batch_size`` seats no
    prefill, and ``_construct_replica_batch`` quietly runs decode instead --
    so binding by price alone parks the request on a replica that cannot run
    it, and the committed two-rank step executes as a one-rank step.
    """
    # Replica 0 holds part of the prompt, so it prices cheapest -- but its
    # decode queue leaves no room to admit the request.
    batch_constructor = create_dp_balance_constructor(
        hit_counts=[PrefixCacheHits(device_blocks=2), PrefixCacheHits()]
    )
    _saturate_decode(batch_constructor, replica_idx=0)
    batch_constructor.enqueue_new_request(
        create_lora_context(is_tg=True), replica_idx=1
    )

    ce_ctx = create_lora_context(seq_len=48)
    batch_constructor.enqueue_new_request(ce_ctx)
    batch_constructor._ce_arrival[ce_ctx.request_id] = time.monotonic() - 60.0

    inputs = batch_constructor.construct_batch()

    assert has_request(inputs.batches[1], ce_ctx.request_id)


def test_dp_ce_balance__saturated_decode_replica_is_not_a_ce_partner() -> None:
    """CE work admission cannot seat must not pass as a balancing partner.

    Replica 0's queued CE request keeps its replica out of the deferrable
    set and puts it in the step's floor, which reads as a partner for
    replica 1's deferrable tail and commits a balanced-looking step. Only
    replica 1 ever runs, so the step lands at half the occupancy planned;
    with no real partner the tail should wait for one instead.
    """
    batch_constructor = create_dp_balance_constructor()
    _saturate_decode(batch_constructor, replica_idx=0)
    batch_constructor.enqueue_new_request(
        create_lora_context(seq_len=50), replica_idx=0
    )
    tail_ctx = _add_deferrable_ce(batch_constructor, replica_idx=1, seq_len=50)

    inputs = batch_constructor.construct_batch()

    assert batch_constructor._ce_deferred_replicas == {1}
    assert not has_request(inputs.batches[1], tail_ctx.request_id)


def test_dp_ce_balance__budget_rejected_request_keeps_its_place() -> None:
    """A request admission could not seat retries in place, not in the pool.

    Pooling is for a replica giving a request up: the binding is what went
    wrong, so the next step places it afresh. A request that merely did not
    fit this step is already bound where it belongs, and pooling it costs it
    the head of its replica's queue -- the next step rebinds it behind
    whatever is queued there, turning a FIFO retry into a rotation.
    """
    # Both requests price cheapest on replica 0, so both bind there, and the
    # 100-token total-context budget fits only the first of the two.
    batch_constructor = create_dp_balance_constructor(
        hit_counts=[PrefixCacheHits(device_blocks=5), PrefixCacheHits()],
        max_batch_total_tokens=100,
    )
    contexts = []
    for _ in range(2):
        ctx = create_lora_context(seq_len=96)
        # Mid-prefill: a small active window against a full 96-token context.
        ctx.tokens.chunk(16)
        batch_constructor.enqueue_new_request(ctx)
        batch_constructor._ce_arrival[ctx.request_id] = time.monotonic() - 60.0
        contexts.append(ctx)
    admitted, rejected = contexts

    inputs = batch_constructor.construct_batch()

    assert has_request(inputs.batches[0], admitted.request_id)
    assert rejected.request_id not in batch_constructor._ce_pending
    replica = batch_constructor.replicas[0]
    assert next(iter(replica.ce_reqs)) == rejected.request_id
    assert (
        batch_constructor._bound_requests[rejected.request_id].replica_idx == 0
    )


# ---------------------------------------------------------------------------
# Async KV onload cordon / re-admit (SERVOPT-1036) tests
# ---------------------------------------------------------------------------


class _IncompleteOnload:
    """A ``KVConnectorTransfer`` whose onload has not yet landed.

    Models the ``rust_tiered`` connector's async handle: ``is_complete`` stays
    ``False`` until the test flips it (via ``synchronize``), so the batch
    constructor cordons the request instead of scheduling it. Setting
    ``fails`` makes every poll raise, as a copy that failed does.
    """

    def __init__(self) -> None:
        self.complete = False
        self.fails = False

    @property
    @property
    def g0_blocks_per_leaf(self) -> Mapping[str, Sequence[int]]:
        return {}

    def is_complete(self) -> bool:
        if self.fails:
            raise KVLoadFailed("memory transfer failed")
        return self.complete

    def synchronize(self) -> None:
        self.complete = True


def _make_cordon_kv_cache(alloc: Mock) -> Mock:
    """A minimal KV cache mock for the cordon path with a custom ``alloc``."""
    kv_cache = Mock()
    kv_cache.chunk_alignment_tokens = 0
    kv_cache.alloc = alloc
    kv_cache.claim = Mock()
    kv_cache.contains = Mock(return_value=False)
    kv_cache.pending_transfers_exist = Mock(return_value=False)
    kv_cache.starts_lookup = Mock(return_value=False)
    kv_cache.awaits_lookup = Mock(return_value=False)
    kv_cache.hand_back_lookup = Mock()
    set_mock_kv_usage(kv_cache, 0.0)
    return kv_cache


def _ce_ctx() -> TextContext:
    return TextContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.ones(9, dtype=np.int64)),
        max_length=100,
    )


def _cordon_config() -> TokenGenerationSchedulerConfig:
    return TokenGenerationSchedulerConfig(
        max_batch_size=5,
        max_batch_total_tokens=None,
        enable_in_flight_batching=False,
        enable_chunked_prefill=False,
        target_tokens_per_batch_ce=30,
    )


def test_text_batch_constructor__cordons_request_with_incomplete_onload(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """A CE request whose KV onload is still in flight is held out of the batch.

    The request is allocated (its blocks pinned) but cordoned: it must not be
    scheduled until the H2D lands, so the GPU can run other ready work meanwhile.
    """
    onload = _IncompleteOnload()
    kv_cache = _make_cordon_kv_cache(Mock(return_value=onload))
    batch_constructor = TextBatchConstructor(
        scheduler_config=_cordon_config(),
        pipeline=pipeline,
        kv_cache=kv_cache,
    )
    ctx = _ce_ctx()
    batch_constructor.enqueue_new_request(ctx)

    inputs = batch_constructor.construct_batch()

    assert len(inputs.batches[0]) == 0
    assert ctx.request_id in batch_constructor._onloading_reqs
    assert ctx.request_id not in batch_constructor.replicas[0].ce_reqs


def test_text_batch_constructor__readmits_request_when_onload_completes(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """A cordoned request is re-admitted and scheduled once its onload lands."""
    onload = _IncompleteOnload()
    # First alloc cordons (incomplete); the re-admit pass re-allocs and the
    # prefix is now device-resident, so the second alloc completes immediately.
    kv_cache = _make_cordon_kv_cache(
        Mock(side_effect=[onload, CompletedTransfer()])
    )
    batch_constructor = TextBatchConstructor(
        scheduler_config=_cordon_config(),
        pipeline=pipeline,
        kv_cache=kv_cache,
    )
    ctx = _ce_ctx()
    batch_constructor.enqueue_new_request(ctx)

    # Iteration 1: cordoned, empty batch.
    inputs = batch_constructor.construct_batch()
    assert len(inputs.batches[0]) == 0
    assert ctx.request_id in batch_constructor._onloading_reqs

    # The onload lands; the next iteration re-admits and schedules it.
    onload.synchronize()
    inputs = batch_constructor.construct_batch()
    assert has_request(inputs.batches[0], ctx.request_id)
    assert ctx.request_id not in batch_constructor._onloading_reqs


def test_text_batch_constructor__admits_with_the_connector_lookup_deferred(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """Admission lets the cache manager answer a connector lookup off the batch.

    The manager starts one only when told the request can be held out, so
    the flag is what keeps the lookup off the scheduler thread. The request
    is cordoned on the unanswered lookup like an onload, and the alloc after
    re-admission carries the flag too, so the manager finishes the reuse from
    that answer rather than asking again.
    """
    lookup = _IncompleteOnload()
    alloc = Mock(side_effect=[lookup, CompletedTransfer()])
    kv_cache = _make_cordon_kv_cache(alloc)
    batch_constructor = TextBatchConstructor(
        scheduler_config=_cordon_config(),
        pipeline=pipeline,
        kv_cache=kv_cache,
    )
    ctx = _ce_ctx()
    batch_constructor.enqueue_new_request(ctx)

    inputs = batch_constructor.construct_batch()
    assert len(inputs.batches[0]) == 0
    assert ctx.request_id in batch_constructor._onloading_reqs
    alloc.assert_called_once_with(ctx, defer_lookup=True)

    lookup.synchronize()
    inputs = batch_constructor.construct_batch()
    assert has_request(inputs.batches[0], ctx.request_id)
    assert alloc.call_args_list == [call(ctx, defer_lookup=True)] * 2


def test_text_batch_constructor__readmits_a_sweep_in_arrival_order(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """Requests whose onloads land in the same sweep keep their arrival order.

    Connector lookups answer in bursts, so several cordoned requests often
    come back at once, and each one going to the front of the queue must
    not reverse them.
    """
    first_onload = _IncompleteOnload()
    second_onload = _IncompleteOnload()
    kv_cache = _make_cordon_kv_cache(
        Mock(
            side_effect=[
                first_onload,
                second_onload,
                CompletedTransfer(),
                CompletedTransfer(),
            ]
        )
    )
    batch_constructor = TextBatchConstructor(
        scheduler_config=_cordon_config(),
        pipeline=pipeline,
        kv_cache=kv_cache,
    )
    first = _ce_ctx()
    second = _ce_ctx()
    batch_constructor.enqueue_new_request(first)
    batch_constructor.enqueue_new_request(second)
    batch_constructor.construct_batch()
    assert list(batch_constructor._onloading_reqs) == [
        first.request_id,
        second.request_id,
    ]

    first_onload.synchronize()
    second_onload.synchronize()
    inputs = batch_constructor.construct_batch()

    assert [ctx.request_id for ctx in inputs.batches[0]] == [
        first.request_id,
        second.request_id,
    ]


class _DeferredLookup:
    """A connector lookup that answers on poll ``answers_on``, if ever.

    It answers a miss unless ``hit`` is set.
    """

    def __init__(
        self, num_blocks: int, answers_on: int | None, hit: bool = False
    ) -> None:
        self._num_blocks = num_blocks
        self._answers_on = answers_on
        self._hit = hit
        self.polls = 0

    def is_complete(self) -> bool:
        self.polls += 1
        return self._answers_on is not None and self.polls >= self._answers_on

    def masks(self) -> Mapping[str, Sequence[bool]]:
        if self._answers_on is None or self.polls < self._answers_on:
            raise RuntimeError("the lookup has not answered")
        return {"full": [self._hit] * self._num_blocks}


class _DeferringConnector(NullConnector):
    """Defers every lookup and records each one handed back.

    ``answers_on`` is the poll on which each lookup answers, never if None,
    and ``hits`` makes each answer a hit over every block it was asked.
    """

    def __init__(
        self, answers_on: int | None = None, hits: bool = False
    ) -> None:
        self.answers_on = answers_on
        self.hits = hits
        self.started: list[KVLookup] = []
        self.abandoned: list[KVLookup] = []

    @property
    def name(self) -> str:
        return "DeferringConnector"

    def start_lookup(
        self,
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
        hint: bytes | None = None,
    ) -> KVLookup:
        lookup = _DeferredLookup(
            len(block_hashes), self.answers_on, hit=self.hits
        )
        self.started.append(lookup)
        return lookup

    def defers_lookup(self, replica_idx: int = 0) -> bool:
        return True

    def abandon_lookup(self, lookup: KVLookup) -> None:
        self.abandoned.append(lookup)


class _AnsweredLookup:
    """A connector lookup that has already answered a hit over every block."""

    def __init__(self, num_blocks: int) -> None:
        self._num_blocks = num_blocks

    def is_complete(self) -> bool:
        return True

    def masks(self) -> Mapping[str, Sequence[bool]]:
        return {"full": [True] * self._num_blocks}


class _OnloadingConnector(NullConnector):
    """Holds every block it is asked about and onloads without landing.

    ``defers`` says whether its lookups are handle carried, as dKV's are,
    though each one has answered by the time the manager first polls it.
    """

    def __init__(self, defers: bool) -> None:
        self.defers = defers
        self.onloads: list[_IncompleteOnload] = []

    def lookup(
        self,
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
        hint: bytes | None = None,
    ) -> Mapping[str, Sequence[bool]]:
        return {"full": [True] * len(block_hashes)}

    def start_lookup(
        self,
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
        hint: bytes | None = None,
    ) -> KVLookup:
        return _AnsweredLookup(len(block_hashes))

    def defers_lookup(self, replica_idx: int = 0) -> bool:
        return self.defers

    def load(
        self,
        block_ids: Mapping[str, Sequence[int]],
        block_hashes: Mapping[str, Sequence[bytes]],
        replica_idx: int = 0,
        hint: bytes | None = None,
    ) -> KVTransfer:
        onload = _IncompleteOnload()
        self.onloads.append(onload)
        return onload


def _batch_constructor_over_jenga(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
    connector: NullConnector,
    scheduler_config: TokenGenerationSchedulerConfig | None = None,
    num_blocks: int = 16,
) -> tuple[TextBatchConstructor, JengaBlockManager]:
    """A batch constructor whose cache calls reach a real Jenga manager."""
    leaf_infos = {"full": KVLeafInfo(1, KVCacheGroupId.full())}
    pools = create_pools(leaf_infos, num_blocks)
    manager = JengaBlockManager(
        pools=pools,
        groups=create_groups(leaf_infos, pools, 1, enable_prefix_caching=True),
        leaves={
            "full": PagedKVLeafRegion(
                leaf_id="full",
                group_id=KVCacheGroupId.full(),
                bytes_per_page=1,
                page_size=1,
            )
        },
        block_size=1,
        enable_prefix_caching=True,
        max_num_input_tokens=None,
        num_draft_tokens=0,
        num_draft_tokens_per_step=0,
        connector=connector,
    )
    kv_cache = _make_cordon_kv_cache(Mock(side_effect=manager.alloc))
    kv_cache.claim = Mock(side_effect=manager.claim)
    kv_cache.contains = Mock(side_effect=manager.contains)
    kv_cache.release = Mock(side_effect=manager.release)
    kv_cache.starts_lookup = Mock(side_effect=manager.starts_lookup)
    kv_cache.start_lookup = Mock(side_effect=manager.start_lookup)
    kv_cache.awaits_lookup = Mock(side_effect=manager.awaits_lookup)
    kv_cache.hand_back_lookup = Mock(side_effect=manager.hand_back_lookup)
    kv_cache.num_req_blocks = Mock(side_effect=manager.num_req_blocks)
    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config or _cordon_config(),
        pipeline=pipeline,
        kv_cache=kv_cache,
    )
    return batch_constructor, manager


def test_text_batch_constructor__release_hands_back_a_cordoned_lookup(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """Cancelling a request held on a connector lookup hands the lookup back.

    The request holds no blocks while it waits, so the cordon is the only
    thing that knows about it, and releasing it through the batch
    constructor has to reach the cache manager that owns the handle.
    """
    connector = _DeferringConnector()
    batch_constructor, manager = _batch_constructor_over_jenga(
        pipeline, connector
    )
    ctx = _ce_ctx()
    batch_constructor.enqueue_new_request(ctx)
    batch_constructor.construct_batch()
    assert ctx.request_id in batch_constructor._onloading_reqs
    (lookup,) = connector.started

    batch_constructor.release_request(ctx.request_id)

    assert connector.abandoned == [lookup]
    assert not manager.contains(ctx)
    assert not batch_constructor.contains(ctx.request_id)


def test_text_batch_constructor__cordons_a_lookup_that_answers_during_admission(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """A lookup answering between alloc and the admission poll is cordoned.

    The request holds no pages until the alloc after the answer, so letting
    the admission poll see the answer would run it with nothing allocated.
    It is cordoned for one sweep instead, then allocated and scheduled.
    """
    connector = _DeferringConnector(answers_on=2)
    batch_constructor, manager = _batch_constructor_over_jenga(
        pipeline, connector
    )
    ctx = _ce_ctx()
    batch_constructor.enqueue_new_request(ctx)

    inputs = batch_constructor.construct_batch()

    assert len(inputs.batches[0]) == 0
    assert ctx.request_id in batch_constructor._onloading_reqs
    assert manager.num_req_blocks(ctx) == 0

    inputs = batch_constructor.construct_batch()

    assert has_request(inputs.batches[0], ctx.request_id)
    assert manager.num_req_blocks(ctx) > 0


def _ce_ctx_of(num_tokens: int) -> TextContext:
    return TextContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.ones(num_tokens, dtype=np.int64)),
        max_length=100,
    )


def test_text_batch_constructor__starts_only_the_lookups_one_pass_could_run(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """A deep queue of fresh prompts starts no more lookups than fit a batch.

    Each lookup is a leased read the store stages behind the others, so a
    pass charges every one it starts the tokens its prompt would take from
    the batch and stops once that is spent, as the blocking lookup did by
    admitting the request. The rest stay queued in order with nothing asked.
    """
    connector = _DeferringConnector()
    batch_constructor, manager = _batch_constructor_over_jenga(
        pipeline, connector
    )
    ctxs = [_ce_ctx_of(20) for _ in range(4)]
    for ctx in ctxs:
        batch_constructor.enqueue_new_request(ctx)

    inputs = batch_constructor.construct_batch()

    assert len(inputs.batches[0]) == 0
    assert len(connector.started) == 2, "30 batch tokens cover two prompts"
    assert list(batch_constructor._onloading_reqs) == [
        ctxs[0].request_id,
        ctxs[1].request_id,
    ]
    assert list(batch_constructor.replicas[0].ce_reqs) == [
        ctxs[2].request_id,
        ctxs[3].request_id,
    ]
    assert manager._state_of(ctxs[2]).pending_lookup is None
    assert not manager.contains(ctxs[3])


def test_text_batch_constructor__bounds_the_unanswered_lookups_per_replica(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A replica never has more unanswered lookups out than its bound.

    The bound holds across passes while the lookups stay out, and a request
    past it waits at the head of the queue with nothing asked. Handing one
    back frees exactly one place.
    """
    monkeypatch.setenv(MAX_PENDING_LOOKUPS_ENV_VAR, "2")
    connector = _DeferringConnector()
    batch_constructor, manager = _batch_constructor_over_jenga(
        pipeline, connector
    )
    ctxs = [_ce_ctx() for _ in range(4)]
    for ctx in ctxs:
        batch_constructor.enqueue_new_request(ctx)

    batch_constructor.construct_batch()
    batch_constructor.construct_batch()

    assert len(connector.started) == 2, "the batch budget alone allows four"
    assert list(batch_constructor.replicas[0].ce_reqs) == [
        ctxs[2].request_id,
        ctxs[3].request_id,
    ]
    assert manager._state_of(ctxs[2]).pending_lookup is None

    batch_constructor.release_request(ctxs[0].request_id)
    batch_constructor.construct_batch()

    assert len(connector.started) == 3
    assert list(batch_constructor.replicas[0].ce_reqs) == [ctxs[3].request_id]


@pytest.mark.parametrize("defers", [False, True])
def test_text_batch_constructor__onloads_do_not_count_as_unanswered_lookups(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
    monkeypatch: pytest.MonkeyPatch,
    defers: bool,
) -> None:
    """A request cordoned on an onload never counts against the lookup bound.

    Its lookup answered inside the alloc, in place for a connector like the
    tiered one and on the first poll for a handle carried one, and the alloc
    went on to post the onload. So every request is onloaded at once rather
    than waiting at the head of the queue behind a bound it does not use.
    """
    monkeypatch.setenv(MAX_PENDING_LOOKUPS_ENV_VAR, "2")
    connector = _OnloadingConnector(defers=defers)
    batch_constructor, _ = _batch_constructor_over_jenga(pipeline, connector)
    ctxs = [_ce_ctx_of(3) for _ in range(4)]
    for ctx in ctxs:
        batch_constructor.enqueue_new_request(ctx)

    batch_constructor.construct_batch()

    assert len(connector.onloads) == 4
    assert list(batch_constructor._onloading_reqs) == [
        ctx.request_id for ctx in ctxs
    ]
    assert batch_constructor._pending_lookup_count(0) == 0
    assert not batch_constructor.replicas[0].ce_reqs


def test_text_batch_constructor__a_bound_of_zero_runs_lookups_in_place(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no lookup allowed out, every request is looked up synchronously.

    Nothing is held out on a lookup and nothing waits at the head of the
    queue for a place that never frees.
    """
    monkeypatch.setenv(MAX_PENDING_LOOKUPS_ENV_VAR, "0")
    connector = _DeferringConnector()
    batch_constructor, _ = _batch_constructor_over_jenga(pipeline, connector)
    ctxs = [_ce_ctx_of(3) for _ in range(2)]
    for ctx in ctxs:
        batch_constructor.enqueue_new_request(ctx)

    inputs = batch_constructor.construct_batch()

    assert not connector.started
    assert not batch_constructor._onloading_reqs
    assert all(has_request(inputs.batches[0], ctx.request_id) for ctx in ctxs)


def test_text_batch_constructor__rejects_a_negative_lookup_bound(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MAX_PENDING_LOOKUPS_ENV_VAR, "-1")

    with pytest.raises(ValueError, match=MAX_PENDING_LOOKUPS_ENV_VAR):
        _batch_constructor_over_jenga(pipeline, _DeferringConnector())


@pytest.mark.parametrize(
    ("reason", "num_tokens", "only_first_looks_up"),
    [
        ("budget", 20, False),
        ("max_pending", 9, False),
        ("held", 9, True),
    ],
)
def test_text_batch_constructor__counts_each_lookup_admission_stop(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
    monkeypatch: pytest.MonkeyPatch,
    reason: str,
    num_tokens: int,
    only_first_looks_up: bool,
) -> None:
    """A pass that stops admission on a deferred lookup says what stopped it.

    ``budget`` is the pass's share of lookup tokens spent, ``max_pending``
    the replica's bound on unanswered lookups reached, and ``held`` a prompt
    that arrived behind one still out on its lookup.
    """
    monkeypatch.setenv(MAX_PENDING_LOOKUPS_ENV_VAR, "2")
    connector = _DeferringConnector()
    batch_constructor, manager = _batch_constructor_over_jenga(
        pipeline, connector
    )
    ctxs = [_ce_ctx_of(num_tokens) for _ in range(4)]
    if only_first_looks_up:
        kv_cache = cast(Mock, batch_constructor.kv_cache)
        kv_cache.starts_lookup = Mock(
            side_effect=lambda ctx: (
                ctx is ctxs[0] and manager.starts_lookup(ctx)
            )
        )
    for ctx in ctxs:
        batch_constructor.enqueue_new_request(ctx)

    with patch(
        "max.serve.scheduler.batch_constructor.text_batch_constructor.METRICS"
    ) as mock_metrics:
        batch_constructor.construct_batch()

    assert mock_metrics.connector_lookup_admission_stop.call_args_list == [
        call(reason)
    ]


def test_text_batch_constructor__records_the_cordon_wait_once_at_admission(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cordoned prompt's wait is recorded when it is admitted, and only then.

    It runs from the cordon to the admission, and a cordon that clears on
    the next pass still reads as the one pass it cost.
    """
    clock = [10.0]
    monkeypatch.setattr(
        text_batch_constructor,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0]),
    )
    onload = _IncompleteOnload()
    kv_cache = _make_cordon_kv_cache(
        Mock(side_effect=[onload, CompletedTransfer()])
    )
    batch_constructor = TextBatchConstructor(
        scheduler_config=_cordon_config(),
        pipeline=pipeline,
        kv_cache=kv_cache,
    )
    ctx = _ce_ctx()
    batch_constructor.enqueue_new_request(ctx)

    with patch(
        "max.serve.scheduler.batch_constructor.text_batch_constructor.METRICS"
    ) as mock_metrics:
        batch_constructor.construct_batch()
        mock_metrics.connector_cordon_wait_time.assert_not_called()

        clock[0] += 0.25
        onload.synchronize()
        inputs = batch_constructor.construct_batch()
        assert has_request(inputs.batches[0], ctx.request_id)
        batch_constructor.construct_batch()

    mock_metrics.connector_cordon_wait_time.assert_called_once_with(
        pytest.approx(250.0)
    )
    mock_metrics.connector_cordon_passes.assert_called_once_with(1)


def test_text_batch_constructor__an_uncordoned_admission_records_no_wait(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    batch_constructor = TextBatchConstructor(
        scheduler_config=_cordon_config(),
        pipeline=pipeline,
        kv_cache=create_mock_kv_cache(),
    )
    ctx = _ce_ctx()
    batch_constructor.enqueue_new_request(ctx)

    with patch(
        "max.serve.scheduler.batch_constructor.text_batch_constructor.METRICS"
    ) as mock_metrics:
        inputs = batch_constructor.construct_batch()

    assert has_request(inputs.batches[0], ctx.request_id)
    mock_metrics.connector_cordon_wait_time.assert_not_called()
    mock_metrics.connector_cordon_passes.assert_not_called()


def test_text_batch_constructor__the_cordon_wait_spans_lookup_and_onload(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prompt cordoned on its lookup and then on the onload the answer
    starts reports one wait, from the lookup's cordon to its admission.
    """
    clock = [0.0]
    monkeypatch.setattr(
        text_batch_constructor,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0]),
    )
    connector = _DeferringOnloadingConnector()
    batch_constructor, _ = _batch_constructor_over_jenga(pipeline, connector)
    ctx = _ce_ctx()
    batch_constructor.enqueue_new_request(ctx)

    admitted_on = None
    with patch(
        "max.serve.scheduler.batch_constructor.text_batch_constructor.METRICS"
    ) as mock_metrics:
        for pass_idx in range(6):
            clock[0] = pass_idx * 0.1
            inputs = batch_constructor.construct_batch()
            if has_request(inputs.batches[0], ctx.request_id):
                admitted_on = pass_idx
                break

    assert admitted_on is not None and admitted_on >= 2, (
        "admitted before both the lookup and the onload cordoned it"
    )
    assert len(connector.started) == 1
    mock_metrics.connector_cordon_wait_time.assert_called_once_with(
        pytest.approx(admitted_on * 100.0)
    )
    mock_metrics.connector_cordon_passes.assert_called_once_with(admitted_on)


def test_text_batch_constructor__release_drops_a_cordon_wait(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    connector = _DeferringConnector()
    batch_constructor, _ = _batch_constructor_over_jenga(pipeline, connector)
    ctx = _ce_ctx()
    batch_constructor.enqueue_new_request(ctx)
    batch_constructor.construct_batch()
    assert ctx.request_id in batch_constructor._cordon_waits

    batch_constructor.release_request(ctx.request_id)

    assert not batch_constructor._cordon_waits


def _claim_tracking_kv_cache(
    alloc: Callable[..., KVTransfer], holder: TextContext
) -> tuple[Mock, list[RequestID]]:
    """A cache mock where only ``holder`` holds blocks, until it is released.

    Returns the mock and the ids it released, in order.
    """
    claimed: set[RequestID] = set()
    released: list[RequestID] = []

    def claim(ctx: TextContext, replica_idx: int = 0) -> None:
        claimed.add(ctx.request_id)

    def release(ctx: TextContext) -> None:
        claimed.discard(ctx.request_id)
        released.append(ctx.request_id)

    kv_cache = _make_cordon_kv_cache(Mock(side_effect=alloc))
    kv_cache.claim = Mock(side_effect=claim)
    kv_cache.release = Mock(side_effect=release)
    kv_cache.contains = Mock(side_effect=lambda ctx: ctx.request_id in claimed)
    kv_cache.num_req_blocks = Mock(
        side_effect=lambda ctx: (
            1
            if ctx.request_id == holder.request_id
            and holder.request_id in claimed
            else 0
        )
    )
    claimed.add(holder.request_id)
    return kv_cache, released


def test_text_batch_constructor__ce_insufficient_blocks_preempts_a_parked_prefill(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """A prompt that cannot get blocks reclaims them from a parked prefill.

    A prefill between chunks holds its blocks in the queue, where neither the
    other work nor the in flight check sees them, so a fresh prompt ahead of
    it with no decode work, nothing in the batch and nothing in flight is
    not out of memory. The parked prefill is preempted the way the decode
    path reclaims blocks and the prompt runs on the next pass.
    """
    holder = _ce_ctx_of(50)
    fresh = _ce_ctx()

    def alloc(ctx: TextContext, defer_lookup: bool = False) -> KVTransfer:
        if ctx is fresh and holder.request_id not in released:
            raise InsufficientBlocksError("insufficient blocks")
        return CompletedTransfer()

    kv_cache, released = _claim_tracking_kv_cache(alloc, holder)
    batch_constructor = TextBatchConstructor(
        scheduler_config=_cordon_config(),
        pipeline=pipeline,
        kv_cache=kv_cache,
    )
    batch_constructor.enqueue_new_request(holder)
    batch_constructor.enqueue_new_request(fresh)
    batch_constructor.replicas[0].ce_reqs.move_to_end(
        fresh.request_id, last=False
    )

    inputs = batch_constructor.construct_batch()

    assert len(inputs.batches[0]) == 0
    assert holder.request_id in released
    assert batch_constructor.total_preemption_count == 1
    assert list(batch_constructor.replicas[0].ce_reqs) == [
        fresh.request_id,
        holder.request_id,
    ]

    inputs = batch_constructor.construct_batch()

    assert has_request(inputs.batches[0], fresh.request_id)


def test_text_batch_constructor__ce_insufficient_blocks_waits_on_a_transfer_in_flight(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """A prompt that cannot get blocks keeps a parked prefill while work lands.

    Something in flight can still free blocks or turn into runnable work, so
    the shortage is not fatal and the parked prefill keeps the KV it already
    computed. The prompt waits at the head of the queue and retries, as it
    did before parked prefills could be preempted.
    """
    holder = _ce_ctx_of(50)
    fresh = _ce_ctx()

    def alloc(ctx: TextContext, defer_lookup: bool = False) -> KVTransfer:
        if ctx is fresh:
            raise InsufficientBlocksError("insufficient blocks")
        return CompletedTransfer()

    kv_cache, released = _claim_tracking_kv_cache(alloc, holder)
    kv_cache.pending_transfers_exist = Mock(return_value=True)
    batch_constructor = TextBatchConstructor(
        scheduler_config=_cordon_config(),
        pipeline=pipeline,
        kv_cache=kv_cache,
    )
    batch_constructor.enqueue_new_request(holder)
    batch_constructor.enqueue_new_request(fresh)
    batch_constructor.replicas[0].ce_reqs.move_to_end(
        fresh.request_id, last=False
    )

    inputs = batch_constructor.construct_batch()

    assert len(inputs.batches[0]) == 0
    assert holder.request_id not in released
    assert batch_constructor.total_preemption_count == 0
    assert list(batch_constructor.replicas[0].ce_reqs) == [
        fresh.request_id,
        holder.request_id,
    ]


def test_text_batch_constructor__readmits_behind_a_prefill_under_way(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """A request back from its lookup does not jump a prefill under way.

    The prefill holds blocks for work already paid for. Readmitted ahead of
    it, a request too big for what is left would fail its alloc at the head
    of the queue every pass and end up preempting the prefill. Behind it,
    the prefill runs its next chunk and the request waits its turn.
    """
    holder = _ce_ctx_of(50)
    holder.tokens.skip_processing(10)
    readmitted = _ce_ctx()
    lookup = _IncompleteOnload()

    def alloc(ctx: TextContext, defer_lookup: bool = False) -> KVTransfer:
        if ctx is readmitted:
            if not lookup.complete:
                return lookup
            raise InsufficientBlocksError("insufficient blocks")
        return CompletedTransfer()

    kv_cache, released = _claim_tracking_kv_cache(alloc, holder)
    kv_cache.awaits_lookup = Mock(
        side_effect=lambda ctx: ctx is readmitted and not lookup.complete
    )
    batch_constructor = TextBatchConstructor(
        scheduler_config=_cordon_config(),
        pipeline=pipeline,
        kv_cache=kv_cache,
    )
    batch_constructor.enqueue_new_request(readmitted)
    batch_constructor.construct_batch()
    assert readmitted.request_id in batch_constructor._onloading_reqs
    batch_constructor.enqueue_new_request(holder)

    lookup.synchronize()
    inputs = batch_constructor.construct_batch()

    assert [ctx.request_id for ctx in inputs.batches[0]] == [holder.request_id]
    assert released == []
    assert batch_constructor.total_preemption_count == 0
    assert readmitted.request_id in batch_constructor.replicas[0].ce_reqs


def test_text_batch_constructor__hands_back_a_lookup_left_behind_a_prefill(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """An answered lookup is not left to lapse behind a prefill under way.

    The request comes back from its lookup behind the prefill, whose next
    chunk spends the pass. Its lease would wait out every remaining chunk,
    so it is handed back and asked again once the request reaches the
    head of the queue.
    """
    holder = _ce_ctx_of(50)
    holder.tokens.skip_processing(10)
    readmitted = _ce_ctx()
    lookup = _IncompleteOnload()

    def alloc(ctx: TextContext, defer_lookup: bool = False) -> KVTransfer:
        if ctx is readmitted and not lookup.complete:
            return lookup
        return CompletedTransfer()

    kv_cache, _ = _claim_tracking_kv_cache(alloc, holder)
    kv_cache.awaits_lookup = Mock(
        side_effect=lambda ctx: ctx is readmitted and not lookup.complete
    )
    batch_constructor = TextBatchConstructor(
        scheduler_config=_cordon_config(),
        pipeline=pipeline,
        kv_cache=kv_cache,
    )
    batch_constructor.enqueue_new_request(readmitted)
    batch_constructor.construct_batch()
    assert batch_constructor._onloading_reqs[
        readmitted.request_id
    ].awaiting_lookup
    kv_cache.hand_back_lookup.assert_not_called()
    batch_constructor.enqueue_new_request(holder)

    lookup.synchronize()
    inputs = batch_constructor.construct_batch()

    assert [ctx.request_id for ctx in inputs.batches[0]] == [holder.request_id]
    assert readmitted.request_id in batch_constructor.replicas[0].ce_reqs
    kv_cache.hand_back_lookup.assert_called_once_with(readmitted)

    batch_constructor.construct_batch()

    kv_cache.hand_back_lookup.assert_called_once()


def test_text_batch_constructor__keeps_a_fresh_lookup_left_behind_a_prefill(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """A lookup the cache manager keeps as fresh is loaded once its turn comes.

    The request comes back from its lookup behind a prefill under way that
    spends the pass. The manager keeps the answer while it can still be
    loaded, so the request stays eligible for hand back and runs on the
    next pass with that answer instead of asking again.
    """
    holder = _ce_ctx_of(50)
    holder.tokens.skip_processing(10)
    readmitted = _ce_ctx()
    lookup = _IncompleteOnload()

    def alloc(ctx: TextContext, defer_lookup: bool = False) -> KVTransfer:
        if ctx is readmitted and not lookup.complete:
            return lookup
        return CompletedTransfer()

    kv_cache, _ = _claim_tracking_kv_cache(alloc, holder)
    kv_cache.awaits_lookup = Mock(
        side_effect=lambda ctx: ctx is readmitted and not lookup.complete
    )
    kv_cache.hand_back_lookup = Mock(return_value=False)
    batch_constructor = TextBatchConstructor(
        scheduler_config=_cordon_config(),
        pipeline=pipeline,
        kv_cache=kv_cache,
    )
    batch_constructor.enqueue_new_request(readmitted)
    batch_constructor.construct_batch()
    batch_constructor.enqueue_new_request(holder)
    lookup.synchronize()

    inputs = batch_constructor.construct_batch()

    assert [ctx.request_id for ctx in inputs.batches[0]] == [holder.request_id]
    kv_cache.hand_back_lookup.assert_called_once_with(readmitted)
    assert readmitted.request_id in batch_constructor._readmitted_lookups

    inputs = batch_constructor.construct_batch()

    assert has_request(inputs.batches[0], readmitted.request_id)
    kv_cache.hand_back_lookup.assert_called_once()
    assert readmitted.request_id not in batch_constructor._readmitted_lookups


def test_text_batch_constructor__hands_back_a_kept_lookup_once_it_grows_stale(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """A lookup kept as fresh on one pass is handed back on a later one.

    The request comes back from its lookup behind two prefills under way,
    each of which spends a pass. The cache manager keeps the answer on the
    first pass, so the request stays eligible, and hands it back on the
    second once it has grown too old to load, which ends its eligibility.
    """
    holders = [_ce_ctx_of(50) for _ in range(2)]
    for holder in holders:
        holder.tokens.skip_processing(10)
    readmitted = _ce_ctx()
    lookup = _IncompleteOnload()

    def alloc(ctx: TextContext, defer_lookup: bool = False) -> KVTransfer:
        if ctx is readmitted and not lookup.complete:
            return lookup
        return CompletedTransfer()

    kv_cache, _ = _claim_tracking_kv_cache(alloc, holders[0])
    kv_cache.awaits_lookup = Mock(
        side_effect=lambda ctx: ctx is readmitted and not lookup.complete
    )
    kv_cache.hand_back_lookup = Mock(side_effect=[False, True])
    batch_constructor = TextBatchConstructor(
        scheduler_config=_cordon_config(),
        pipeline=pipeline,
        kv_cache=kv_cache,
    )
    batch_constructor.enqueue_new_request(readmitted)
    batch_constructor.construct_batch()
    for holder in holders:
        batch_constructor.enqueue_new_request(holder)
    lookup.synchronize()

    inputs = batch_constructor.construct_batch()

    assert [ctx.request_id for ctx in inputs.batches[0]] == [
        holders[0].request_id
    ]
    kv_cache.hand_back_lookup.assert_called_once_with(readmitted)
    assert readmitted.request_id in batch_constructor._readmitted_lookups

    inputs = batch_constructor.construct_batch()

    assert has_request(inputs.batches[0], holders[1].request_id)
    assert not has_request(inputs.batches[0], readmitted.request_id)
    assert kv_cache.hand_back_lookup.call_args_list == [call(readmitted)] * 2
    assert readmitted.request_id not in batch_constructor._readmitted_lookups

    inputs = batch_constructor.construct_batch()

    assert has_request(inputs.batches[0], readmitted.request_id)
    assert kv_cache.hand_back_lookup.call_count == 2


def test_text_batch_constructor__cordons_a_stale_lookup_it_asks_again(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An answer gone stale by the time it is consumed is asked again.

    The request is readmitted once its lookup answers, but its lease may
    lapse before the load reaches the store. Its alloc asks again instead
    of loading, so the request is cordoned on the new lookup and counts
    against the bound on lookups out, as a fresh one would.
    """
    connector = _DeferringConnector(answers_on=2, hits=True)
    batch_constructor, _ = _batch_constructor_over_jenga(pipeline, connector)
    ctx = _ce_ctx()
    batch_constructor.enqueue_new_request(ctx)
    batch_constructor.construct_batch()
    (first,) = connector.started
    monkeypatch.setattr(jenga_block_manager, "_LOOKUP_HAND_BACK_AGE_S", 0.0)

    inputs = batch_constructor.construct_batch()

    assert not has_request(inputs.batches[0], ctx.request_id)
    assert connector.abandoned == [first]
    assert len(connector.started) == 2
    assert batch_constructor._onloading_reqs[ctx.request_id].awaiting_lookup
    assert batch_constructor._pending_lookup_count(0) == 1


def test_text_batch_constructor__a_stale_lookup_waits_for_room_to_ask_again(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale answer is asked again only while the bound has room.

    Asking again starts another leased read, so with the replica's one place
    taken by another request's lookup, the request waits at the head of its
    queue with its answer kept and nothing asked.
    """
    connector = _DeferringConnector(answers_on=3, hits=True)
    batch_constructor, manager = _batch_constructor_over_jenga(
        pipeline, connector
    )
    ctx = _ce_ctx()
    batch_constructor.enqueue_new_request(ctx)
    batch_constructor.construct_batch()
    connector.answers_on = None
    other = _ce_ctx()
    batch_constructor.enqueue_new_request(other)
    batch_constructor.construct_batch()
    assert batch_constructor._pending_lookup_count(0) == 2
    # What MAX_SERVE_MAX_PENDING_CONNECTOR_LOOKUPS=1 sets, lowered only now
    # so that both lookups could start.
    batch_constructor.max_pending_lookups = 1
    monkeypatch.setattr(jenga_block_manager, "_LOOKUP_HAND_BACK_AGE_S", 0.0)

    inputs = batch_constructor.construct_batch()

    assert not has_request(inputs.batches[0], ctx.request_id)
    assert len(connector.started) == 2
    assert connector.abandoned == []
    assert list(batch_constructor.replicas[0].ce_reqs)[:1] == [ctx.request_id]
    assert manager._state_of(ctx).pending_lookup is not None


class _AnswersOnSecondPoll:
    """A lookup cordon that reads answered from its second poll on.

    The first poll is the admission's, which never sees an answer.
    """

    def __init__(self, state: dict[str, str]) -> None:
        self._state = state
        self._polls = 0

    def is_complete(self) -> bool:
        self._polls += 1
        if self._polls > 1:
            self._state["lookup"] = "answered"
            return True
        return False

    def synchronize(self) -> None:
        return


def _handing_back_kv_cache(
    readmitted: TextContext,
) -> tuple[Mock, list[RequestID]]:
    """A cache mock where ``readmitted`` defers its lookup and can lose it.

    Every hand back resets the lookup, so the next alloc starts another,
    and is recorded. Returns the mock and the hand backs.
    """
    state = {"lookup": "none"}
    handed_back: list[RequestID] = []
    claimed: set[RequestID] = set()

    def alloc(ctx: TextContext, defer_lookup: bool = False) -> KVTransfer:
        if ctx is readmitted and state["lookup"] == "none":
            state["lookup"] = "pending"
            return _AnswersOnSecondPoll(state)
        if ctx is readmitted:
            state["lookup"] = "consumed"
        return CompletedTransfer()

    def hand_back(ctx: TextContext) -> bool:
        if ctx is readmitted and state["lookup"] == "answered":
            state["lookup"] = "none"
            handed_back.append(ctx.request_id)
        return True

    kv_cache = create_mock_kv_cache()
    kv_cache.alloc = Mock(side_effect=alloc)
    kv_cache.claim = Mock(
        side_effect=lambda ctx, replica_idx=0: claimed.add(ctx.request_id)
    )
    kv_cache.contains = Mock(side_effect=lambda ctx: ctx.request_id in claimed)
    kv_cache.num_req_blocks = Mock(return_value=0)
    kv_cache.starts_lookup = Mock(
        side_effect=lambda ctx: ctx is readmitted and state["lookup"] == "none"
    )
    kv_cache.awaits_lookup = Mock(
        side_effect=lambda ctx: (
            ctx is readmitted and state["lookup"] == "pending"
        )
    )
    kv_cache.hand_back_lookup = Mock(side_effect=hand_back)
    return kv_cache, handed_back


def test_text_batch_constructor__an_off_cadence_readmission_keeps_its_lookup(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """A lookup that answers on a step admitting no prefill is kept for it.

    With prefill on every other step and decode work queued, the lookup is
    started on an open step and answers on the closed one after it.
    Handing it back there would start it over on the next open step,
    forever, so it waits for the next open step and runs.
    """
    readmitted = _ce_ctx()
    kv_cache, handed_back = _handing_back_kv_cache(readmitted)
    batch_constructor = TextBatchConstructor(
        scheduler_config=TokenGenerationSchedulerConfig(
            max_batch_size=10,
            target_tokens_per_batch_ce=20,
            enable_chunked_prefill=False,
            enable_in_flight_batching=False,
            prefill_schedule_interval=2,
        ),
        pipeline=pipeline,
        kv_cache=kv_cache,
    )
    batch_constructor.enqueue_new_request(create_lora_context(is_tg=True))
    batch_constructor.enqueue_new_request(readmitted)

    ran = False
    for _ in range(6):
        inputs = batch_constructor.construct_batch()
        if has_request(inputs.batches[0], readmitted.request_id):
            ran = True
            break

    assert ran
    assert handed_back == []


def test_text_batch_constructor__hands_back_on_the_first_open_step_after(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """A lookup readmitted on a closed step is handed back on the next open one.

    The lookup answers on the step after the open one that started it,
    which admits no prefill, and the request lands behind a long prefill
    under way. The next open step spends its budget on that prefill and
    stops before the request, so the answer is handed back there rather
    than left to lapse behind every remaining chunk.
    """
    readmitted = _ce_ctx()
    kv_cache, handed_back = _handing_back_kv_cache(readmitted)
    batch_constructor = TextBatchConstructor(
        scheduler_config=TokenGenerationSchedulerConfig(
            max_batch_size=10,
            target_tokens_per_batch_ce=20,
            enable_chunked_prefill=False,
            enable_in_flight_batching=False,
            prefill_schedule_interval=2,
        ),
        pipeline=pipeline,
        kv_cache=kv_cache,
    )
    batch_constructor.enqueue_new_request(create_lora_context(is_tg=True))
    batch_constructor.enqueue_new_request(readmitted)
    for _ in range(4):
        batch_constructor.construct_batch()
        if readmitted.request_id in batch_constructor._onloading_reqs:
            break
    assert readmitted.request_id in batch_constructor._onloading_reqs
    holder = _ce_ctx_of(50)
    holder.tokens.skip_processing(10)
    batch_constructor.enqueue_new_request(holder)

    batch_constructor.construct_batch()

    assert list(batch_constructor.replicas[0].ce_reqs)[:2] == [
        holder.request_id,
        readmitted.request_id,
    ]
    assert handed_back == []

    inputs = batch_constructor.construct_batch()

    assert has_request(inputs.batches[0], holder.request_id)
    assert not has_request(inputs.batches[0], readmitted.request_id)
    assert handed_back == [readmitted.request_id]


def test_text_batch_constructor__a_backfill_out_of_budget_keeps_a_lookup(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """A backfill with no budget left puts an answered lookup back as it is.

    In-flight batching backfills prefill after decode, here a verify
    window as wide as the budget, has spent the batch. Releasing a request popped with nothing left would throw its answer
    away and clear the counts that bound how often it asks again, so it
    waits at the head of its queue holding both, and the hand back rule
    applies to it as it does to any request passed over.
    """
    readmitted = _ce_ctx()
    kv_cache, handed_back = _handing_back_kv_cache(readmitted)
    batch_constructor = TextBatchConstructor(
        scheduler_config=TokenGenerationSchedulerConfig(
            max_batch_size=5,
            max_batch_total_tokens=None,
            enable_in_flight_batching=True,
            enable_chunked_prefill=False,
            target_tokens_per_batch_ce=5,
        ),
        pipeline=pipeline,
        kv_cache=kv_cache,
    )
    batch_constructor.enqueue_new_request(readmitted)
    batch_constructor.construct_batch()
    assert readmitted.request_id in batch_constructor._onloading_reqs
    # A real manager reads an answered lookup still out as awaited too.
    kv_cache.awaits_lookup = Mock(side_effect=lambda ctx: ctx is readmitted)
    decode = create_lora_context(is_tg=True)
    decode.tokens.rewind_processing(4)
    batch_constructor.enqueue_new_request(decode)

    inputs = batch_constructor.construct_batch()

    assert not has_request(inputs.batches[0], readmitted.request_id)
    assert call(readmitted) not in kv_cache.release.call_args_list
    assert next(iter(batch_constructor.replicas[0].ce_reqs)) == (
        readmitted.request_id
    )
    assert handed_back == [readmitted.request_id]


def test_text_batch_constructor__a_decode_priority_pass_keeps_its_lookup(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """A pass that runs only decode keeps the lookup it readmits.

    Above the KV pressure threshold the pass admits no prefill at all, so
    nothing was passed over and there is nothing to hand back.
    """
    readmitted = _ce_ctx()
    kv_cache, handed_back = _handing_back_kv_cache(readmitted)
    batch_constructor = TextBatchConstructor(
        scheduler_config=_cordon_config(),
        pipeline=pipeline,
        kv_cache=kv_cache,
    )
    batch_constructor.enqueue_new_request(readmitted)
    batch_constructor.construct_batch()
    assert readmitted.request_id in batch_constructor._onloading_reqs
    batch_constructor.enqueue_new_request(create_lora_context(is_tg=True))
    set_mock_kv_usage(kv_cache, 0.95)

    inputs = batch_constructor.construct_batch()

    assert not has_request(inputs.batches[0], readmitted.request_id)
    assert readmitted.request_id in batch_constructor.replicas[0].ce_reqs
    assert handed_back == []
    assert readmitted.request_id in batch_constructor._readmitted_lookups

    batch_constructor.release_request(readmitted.request_id)

    assert readmitted.request_id not in batch_constructor._readmitted_lookups


def test_text_batch_constructor__keeps_a_lookup_its_readmission_loads(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """A lookup that answers and runs on the same pass is never handed back."""
    connector = _DeferringConnector(answers_on=2, hits=True)
    batch_constructor, _ = _batch_constructor_over_jenga(pipeline, connector)
    ctx = _ce_ctx()
    batch_constructor.enqueue_new_request(ctx)
    batch_constructor.construct_batch()

    inputs = batch_constructor.construct_batch()

    assert has_request(inputs.batches[0], ctx.request_id)
    assert connector.abandoned == []
    assert ctx.cached_prefix_external_length > 0


class _LandsOnThirdPoll:
    """An onload the manager and the admission poll see in flight."""

    def __init__(self) -> None:
        self._polls = 0

    def is_complete(self) -> bool:
        self._polls += 1
        return self._polls >= 3

    def synchronize(self) -> None:
        return


class _DeferringOnloadingConnector(_DeferringConnector):
    """Answers every lookup a hit and onloads it over one more pass."""

    def __init__(self) -> None:
        super().__init__(answers_on=2, hits=True)

    def load(
        self,
        block_ids: Mapping[str, Sequence[int]],
        block_hashes: Mapping[str, Sequence[bytes]],
        replica_idx: int = 0,
        hint: bytes | None = None,
    ) -> KVTransfer:
        return _LandsOnThirdPoll()


@pytest.mark.parametrize("onloads", [False, True])
def test_text_batch_constructor__a_lookup_keeps_its_turn_in_arrival_order(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
    monkeypatch: pytest.MonkeyPatch,
    onloads: bool,
) -> None:
    """Prompts that arrive behind a request out on a lookup do not pass it.

    Every other prompt is looked up in place and one is always queued behind
    the request, and each pass takes 2.4 s. Admitted while the request
    waited, they would start prefills it is readmitted behind, so its
    answer would age past the hand back age and it would ask again and lose
    its turn once more. With the answer a hit, the onload it starts holds
    the request out one more pass, and that must not let them by either.
    """
    clock = [0.0]
    monkeypatch.setattr(
        jenga_block_manager, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    connector = (
        _DeferringOnloadingConnector()
        if onloads
        else _DeferringConnector(answers_on=2)
    )
    batch_constructor, manager = _batch_constructor_over_jenga(
        pipeline,
        connector,
        scheduler_config=TokenGenerationSchedulerConfig(
            max_batch_size=5,
            max_batch_total_tokens=None,
            enable_in_flight_batching=False,
            enable_chunked_prefill=True,
            target_tokens_per_batch_ce=30,
        ),
        num_blocks=1024,
    )
    cold = TextContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.zeros(100, dtype=np.int64)),
        max_length=200,
    )
    kv_cache = cast(Mock, batch_constructor.kv_cache)
    kv_cache.starts_lookup = Mock(
        side_effect=lambda ctx: ctx is cold and manager.starts_lookup(ctx)
    )
    kv_cache.alloc = Mock(
        side_effect=lambda ctx, defer_lookup=False: manager.alloc(
            ctx, defer_lookup=defer_lookup and ctx is cold
        )
    )
    batch_constructor.enqueue_new_request(cold)
    later: list[TextContext] = []
    ran: list[RequestID] = []
    for _ in range(12):
        queued = batch_constructor.replicas[0].ce_reqs.values()
        if not any(
            ctx.tokens.processed_length == 0 and ctx is not cold
            for ctx in queued
        ):
            prompt = TextContext(
                request_id=RequestID(),
                tokens=TokenBuffer(
                    np.full(100, len(later) + 1, dtype=np.int64)
                ),
                max_length=200,
            )
            later.append(prompt)
            batch_constructor.enqueue_new_request(prompt)
        clock[0] += 2.4

        inputs = batch_constructor.construct_batch()

        for ctx in inputs.batches[0]:
            if ctx.request_id not in ran:
                ran.append(ctx.request_id)
            if ctx.tokens.actively_chunked:
                ctx.tokens.advance_chunk()
            else:
                ctx.update(ARBITRARY_TOKEN_ID)
        batch_constructor.advance_requests(inputs)
        for ctx in inputs.batches[0]:
            if ctx.tokens.generated_length > 0 and ctx is not cold:
                batch_constructor.release_request(ctx.request_id)
        if cold.request_id in ran:
            break

    assert cold.request_id in ran
    assert ran.index(cold.request_id) == 0, "prompts that arrived later ran"
    assert len(connector.started) == 1
    assert manager._state_of(cold).connector_lookup_hand_backs == 0


class _StuckOnloadConnector(_DeferringConnector):
    """Answers every lookup a hit whose onload never settles."""

    def __init__(self) -> None:
        super().__init__(answers_on=2, hits=True)

    def load(
        self,
        block_ids: Mapping[str, Sequence[int]],
        block_hashes: Mapping[str, Sequence[bytes]],
        replica_idx: int = 0,
        hint: bytes | None = None,
    ) -> KVTransfer:
        return _IncompleteOnload()


def test_text_batch_constructor__a_stuck_onload_holds_prompts_only_so_long(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An onload a lookup started holds the prompts behind it for a bound.

    The bound runs from the first cordon on the lookup rather than from the
    onload that followed it, so the whole wait is bounded. Past it a prompt
    that needs no lookup is admitted, and a transfer that never settles
    holds no request but its own.
    """
    clock = [0.0]
    fake_time = SimpleNamespace(monotonic=lambda: clock[0])
    monkeypatch.setattr(jenga_block_manager, "time", fake_time)
    monkeypatch.setattr(text_batch_constructor, "time", fake_time)
    batch_constructor, manager = _batch_constructor_over_jenga(
        pipeline, _StuckOnloadConnector(), num_blocks=64
    )
    cold = _ce_ctx()
    kv_cache = cast(Mock, batch_constructor.kv_cache)
    kv_cache.starts_lookup = Mock(
        side_effect=lambda ctx: ctx is cold and manager.starts_lookup(ctx)
    )
    kv_cache.alloc = Mock(
        side_effect=lambda ctx, defer_lookup=False: manager.alloc(
            ctx, defer_lookup=defer_lookup and ctx is cold
        )
    )
    behind = TextContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.full(9, 2, dtype=np.int64)),
        max_length=100,
    )
    batch_constructor.enqueue_new_request(cold)
    batch_constructor.enqueue_new_request(behind)
    batch_constructor.construct_batch()
    clock[0] = 1.0
    batch_constructor.construct_batch()
    assert not batch_constructor._onloading_reqs[
        cold.request_id
    ].awaiting_lookup

    clock[0] = 4.9
    inputs = batch_constructor.construct_batch()

    assert not has_request(inputs.batches[0], behind.request_id)

    clock[0] = 5.0
    inputs = batch_constructor.construct_batch()

    assert has_request(inputs.batches[0], behind.request_id)
    assert cold.request_id in batch_constructor._onloading_reqs


def test_text_batch_constructor__a_re_ask_keeps_the_hold_from_the_first_cordon(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asking a lookup again does not restart the hold on the prompts behind.

    The answer goes stale before the request is readmitted, so its alloc
    asks again at 4 s. The hold still ends 5 s after the first cordon, or a
    request that keeps asking again could hold the prompts behind it for
    as long as it keeps asking.
    """
    clock = [0.0]
    fake_time = SimpleNamespace(monotonic=lambda: clock[0])
    monkeypatch.setattr(jenga_block_manager, "time", fake_time)
    monkeypatch.setattr(text_batch_constructor, "time", fake_time)
    connector = _DeferringConnector(answers_on=2, hits=True)
    batch_constructor, manager = _batch_constructor_over_jenga(
        pipeline, connector, num_blocks=64
    )
    cold = _ce_ctx()
    kv_cache = cast(Mock, batch_constructor.kv_cache)
    kv_cache.starts_lookup = Mock(
        side_effect=lambda ctx: ctx is cold and manager.starts_lookup(ctx)
    )
    kv_cache.alloc = Mock(
        side_effect=lambda ctx, defer_lookup=False: manager.alloc(
            ctx, defer_lookup=defer_lookup and ctx is cold
        )
    )
    behind = TextContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.full(9, 2, dtype=np.int64)),
        max_length=100,
    )
    batch_constructor.enqueue_new_request(cold)
    batch_constructor.enqueue_new_request(behind)
    batch_constructor.construct_batch()
    connector.answers_on = None
    monkeypatch.setattr(jenga_block_manager, "_LOOKUP_HAND_BACK_AGE_S", 0.0)

    clock[0] = 4.0
    inputs = batch_constructor.construct_batch()

    assert len(connector.started) == 2
    assert batch_constructor._onloading_reqs[cold.request_id].awaiting_lookup
    assert not has_request(inputs.batches[0], behind.request_id)

    clock[0] = 5.0
    inputs = batch_constructor.construct_batch()

    assert has_request(inputs.batches[0], behind.request_id)
    assert cold.request_id in batch_constructor._onloading_reqs


class _ScheduledConnector(_DeferringConnector):
    """Answers each lookup on its own poll, the first a hit if ``first_hits``.

    ``answers`` is the poll each lookup answers on, in the order they start,
    never if None. An onload lands on its third poll.
    """

    def __init__(self, answers: Sequence[int | None], first_hits: bool) -> None:
        super().__init__()
        self._answers = list(answers)
        self._first_hits = first_hits

    def start_lookup(
        self,
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
        hint: bytes | None = None,
    ) -> KVLookup:
        lookup = _DeferredLookup(
            len(block_hashes),
            self._answers[len(self.started)],
            hit=self._first_hits and not self.started,
        )
        self.started.append(lookup)
        return lookup

    def load(
        self,
        block_ids: Mapping[str, Sequence[int]],
        block_hashes: Mapping[str, Sequence[bytes]],
        replica_idx: int = 0,
        hint: bytes | None = None,
    ) -> KVTransfer:
        return _LandsOnThirdPoll()


def _run_until(
    batch_constructor: TextBatchConstructor,
    target: TextContext,
    passes: int,
) -> list[RequestID]:
    """Runs passes until ``target`` is scheduled, returning who ran, in order.

    Each scheduled prompt runs its chunk, and one that finishes is released.
    """
    ran: list[RequestID] = []
    for _ in range(passes):
        inputs = batch_constructor.construct_batch()
        for ctx in inputs.batches[0]:
            if ctx.request_id not in ran:
                ran.append(ctx.request_id)
            if ctx.tokens.actively_chunked:
                ctx.tokens.advance_chunk()
            else:
                ctx.update(ARBITRARY_TOKEN_ID)
        batch_constructor.advance_requests(inputs)
        for ctx in inputs.batches[0]:
            if ctx.tokens.generated_length > 0 and ctx is not target:
                batch_constructor.release_request(ctx.request_id)
        if target.request_id in ran:
            break
    return ran


def _chunking_config() -> TokenGenerationSchedulerConfig:
    return TokenGenerationSchedulerConfig(
        max_batch_size=5,
        max_batch_total_tokens=None,
        enable_in_flight_batching=False,
        enable_chunked_prefill=True,
        target_tokens_per_batch_ce=30,
    )


@pytest.mark.parametrize("hit", [False, True])
def test_text_batch_constructor__a_later_lookup_answering_first_waits_its_turn(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
    hit: bool,
) -> None:
    """A later prompt whose lookup answers first still runs after the earlier.

    Its lookup starts while the earlier one is out, so both are asked at
    once, but the answer waits at the head of the queue until the earlier
    request is admitted, rather than starting a prefill the earlier request
    would be readmitted behind. With the earlier answer a hit, the onload
    it starts holds the later prompt back too, so a long prefill cannot
    push the onloaded request behind every one of its chunks.
    """
    connector = _ScheduledConnector(answers=[4, 2], first_hits=hit)
    batch_constructor, manager = _batch_constructor_over_jenga(
        pipeline,
        connector,
        scheduler_config=_chunking_config(),
        num_blocks=1024,
    )
    earlier = _ce_ctx()
    later = TextContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.full(100, 2, dtype=np.int64)),
        max_length=200,
    )
    batch_constructor.enqueue_new_request(earlier)
    batch_constructor.enqueue_new_request(later)

    batch_constructor.construct_batch()

    assert len(connector.started) == 2, "the later lookup runs alongside"
    assert manager.num_req_blocks(later) == 0

    ran = _run_until(batch_constructor, earlier, passes=8)

    assert ran[:1] == [earlier.request_id], "a later prompt ran first"
    assert len(connector.started) == 2


def test_text_batch_constructor__readmits_lookups_in_arrival_order(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Answers that come back together go to the queue in arrival order.

    The earlier request asked again after its later neighbor started its
    lookup, so it is cordoned behind it, and both answer on the same pass.
    Readmitted in cordon order, the later one would wait at the head of the
    queue for the earlier one stuck behind it, and neither would run.
    """
    connector = _ScheduledConnector(answers=[2, 3, 2], first_hits=False)
    batch_constructor, _ = _batch_constructor_over_jenga(
        pipeline, connector, num_blocks=64
    )
    earlier = _ce_ctx()
    later = TextContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.full(9, 2, dtype=np.int64)),
        max_length=100,
    )
    batch_constructor.enqueue_new_request(earlier)
    batch_constructor.enqueue_new_request(later)
    batch_constructor.construct_batch()
    age = jenga_block_manager._LOOKUP_HAND_BACK_AGE_S
    monkeypatch.setattr(jenga_block_manager, "_LOOKUP_HAND_BACK_AGE_S", 0.0)
    batch_constructor.construct_batch()
    assert list(batch_constructor._onloading_reqs) == [
        later.request_id,
        earlier.request_id,
    ]
    monkeypatch.setattr(jenga_block_manager, "_LOOKUP_HAND_BACK_AGE_S", age)

    inputs = batch_constructor.construct_batch()

    assert [ctx.request_id for ctx in inputs.batches[0]] == [
        earlier.request_id,
        later.request_id,
    ]


def test_text_batch_constructor__a_lookup_holds_only_prompts_behind_it(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """A prompt that arrived before a request out on a lookup is not held.

    It was preempted back to the head of the queue after the request was
    cordoned, so it is ahead of the request in arrival order and runs.
    """
    batch_constructor, manager = _batch_constructor_over_jenga(
        pipeline, _DeferringConnector()
    )
    earlier = TextContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.full(9, 2, dtype=np.int64)),
        max_length=100,
    )
    cold = _ce_ctx()
    kv_cache = cast(Mock, batch_constructor.kv_cache)
    kv_cache.starts_lookup = Mock(
        side_effect=lambda ctx: ctx is cold and manager.starts_lookup(ctx)
    )
    kv_cache.alloc = Mock(
        side_effect=lambda ctx, defer_lookup=False: manager.alloc(
            ctx, defer_lookup=defer_lookup and ctx is cold
        )
    )
    batch_constructor.enqueue_new_request(earlier)
    batch_constructor.enqueue_new_request(cold)
    inputs = batch_constructor.construct_batch()
    assert has_request(inputs.batches[0], earlier.request_id)
    assert cold.request_id in batch_constructor._onloading_reqs
    batch_constructor._preempt_request(
        earlier, 0, reason=PreemptionReason.KV_CACHE_MEMORY
    )

    inputs = batch_constructor.construct_batch()

    assert has_request(inputs.batches[0], earlier.request_id)


def test_text_batch_constructor__a_readmitted_lookup_stays_behind_an_earlier_prompt(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """A later prompt back from its lookup does not pass an earlier one.

    The earlier prompt is preempted back to the queue while two later ones
    are out on lookups, and the latest answers first. Readmitted ahead of
    the earlier prompt, it would wait at the head of the queue for the
    other lookup and keep the earlier prompt out until the hold ran out.
    """
    connector = _ScheduledConnector(answers=[None, 2], first_hits=False)
    batch_constructor, manager = _batch_constructor_over_jenga(
        pipeline, connector, num_blocks=64
    )
    earlier = TextContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.full(9, 2, dtype=np.int64)),
        max_length=100,
    )
    first = _ce_ctx()
    latest = TextContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.full(9, 3, dtype=np.int64)),
        max_length=100,
    )
    kv_cache = cast(Mock, batch_constructor.kv_cache)
    kv_cache.starts_lookup = Mock(
        side_effect=lambda ctx: (
            ctx is not earlier and manager.starts_lookup(ctx)
        )
    )
    kv_cache.alloc = Mock(
        side_effect=lambda ctx, defer_lookup=False: manager.alloc(
            ctx, defer_lookup=defer_lookup and ctx is not earlier
        )
    )
    batch_constructor.enqueue_new_request(earlier)
    batch_constructor.enqueue_new_request(first)
    batch_constructor.enqueue_new_request(latest)
    inputs = batch_constructor.construct_batch()
    assert has_request(inputs.batches[0], earlier.request_id)
    assert list(batch_constructor._onloading_reqs) == [
        first.request_id,
        latest.request_id,
    ]
    batch_constructor._preempt_request(
        earlier, 0, reason=PreemptionReason.KV_CACHE_MEMORY
    )

    inputs = batch_constructor.construct_batch()

    assert latest.request_id not in batch_constructor._onloading_reqs
    assert [ctx.request_id for ctx in inputs.batches[0]] == [earlier.request_id]
    assert list(batch_constructor.replicas[0].ce_reqs) == [latest.request_id]


def _preempt_to_the_pool_warm_on_replica_0(
    batch_constructor: TextBatchConstructor,
    earlier: TextContext,
) -> None:
    """Makes a prompt pool-managed and preempts it back to the DP pool.

    Its arrival has expired, so the next pass binds it first, and its
    cached prefix on replica 0 makes that the cheaper replica to bind to.
    """
    _stub_per_request_hits(
        batch_constructor,
        {
            earlier.request_id: [
                PrefixCacheHits(device_blocks=2),
                PrefixCacheHits(),
            ]
        },
    )
    batch_constructor._ce_arrival[earlier.request_id] = time.monotonic() - 60.0
    batch_constructor._preempt_request(
        earlier, 0, reason=PreemptionReason.KV_CACHE_MEMORY
    )
    assert earlier.request_id in batch_constructor._ce_pending


def _only_starts_a_lookup(
    batch_constructor: TextBatchConstructor, holder: TextContext
) -> None:
    """Has ``holder`` alone start a connector lookup, one that never answers."""
    kv_cache = _mock_kv_cache_of(batch_constructor)
    lookup = _IncompleteOnload()
    kv_cache.num_req_blocks = Mock(return_value=0)
    kv_cache.starts_lookup = Mock(side_effect=lambda ctx: ctx is holder)
    kv_cache.awaits_lookup = Mock(side_effect=lambda ctx: ctx is holder)
    kv_cache.alloc = Mock(
        side_effect=lambda ctx, defer_lookup=False: (
            lookup if ctx is holder else CompletedTransfer()
        )
    )


@pytest.mark.parametrize("bound", [4, 1])
def test_dp_ce_balance__charges_a_lookup_its_post_cache_weight(
    monkeypatch: pytest.MonkeyPatch, bound: int
) -> None:
    """A pass charges each lookup its prompt's post-cache weight.

    Both prompts are three times the step's budget, but the connector holds
    all but 44 tokens of each, so one pass could run both. Charged their
    whole prompts, the first lookup would spend the pass, and the second
    would wait at the head of the queue and end the pass there, keeping the
    prefill under way behind them out of the batch. The bound on unanswered
    lookups still stops the second when it allows only one.
    """
    monkeypatch.setenv(MAX_PENDING_LOOKUPS_ENV_VAR, str(bound))
    batch_constructor = create_dp_balance_constructor()
    first = _long_ce_context(300)
    second = _long_ce_context(300)
    under_way = create_lora_context(seq_len=10)
    lookup_ids = {first.request_id, second.request_id}
    _stub_per_request_hits(
        batch_constructor,
        {
            request_id: [PrefixCacheHits(external_blocks=16), PrefixCacheHits()]
            for request_id in lookup_ids
        },
    )
    kv_cache = _mock_kv_cache_of(batch_constructor)
    kv_cache.num_req_blocks = Mock(return_value=0)
    kv_cache.starts_lookup = Mock(
        side_effect=lambda ctx: ctx.request_id in lookup_ids
    )
    kv_cache.awaits_lookup = Mock(
        side_effect=lambda ctx: ctx.request_id in lookup_ids
    )
    kv_cache.alloc = Mock(
        side_effect=lambda ctx, defer_lookup=False: (
            _IncompleteOnload()
            if ctx.request_id in lookup_ids
            else CompletedTransfer()
        )
    )
    # The second prompt is held behind the first's lookup, so it asks here.
    kv_cache.start_lookup = Mock(side_effect=lambda ctx: _IncompleteOnload())
    for ctx in (first, second, under_way):
        batch_constructor.enqueue_new_request(ctx, replica_idx=0)
    under_way.tokens.skip_processing(2)

    inputs = batch_constructor.construct_batch()

    if bound == 1:
        assert list(batch_constructor._onloading_reqs) == [first.request_id]
        assert not inputs.batches[0]
        assert list(batch_constructor.replicas[0].ce_reqs) == [
            second.request_id,
            under_way.request_id,
        ]
    else:
        assert list(batch_constructor._onloading_reqs) == [
            first.request_id,
            second.request_id,
        ]
        assert [ctx.request_id for ctx in inputs.batches[0]] == [
            under_way.request_id
        ]


@pytest.mark.parametrize("later_under_way", [False, True])
def test_dp_ce_balance__a_rebound_prompt_keeps_its_turn_under_a_lookup_hold(
    later_under_way: bool,
) -> None:
    """A preempted prompt bound again goes ahead of the prompts held after it.

    It was bound before the request out on a lookup, so the hold does not
    keep it waiting. Bound again at the tail of the queue, it would sit
    behind a later prompt the hold keeps at the head, which ends every pass
    there until the hold runs out or a readmission on the replica sorts the
    queue. A later prefill already under way is not held and keeps its
    place.
    """
    batch_constructor = create_dp_balance_constructor()
    earlier = create_lora_context(seq_len=40)
    holder = create_lora_context(seq_len=10)
    later = create_lora_context(seq_len=10)
    _only_starts_a_lookup(batch_constructor, holder)
    for ctx in (earlier, holder, later):
        batch_constructor.enqueue_new_request(ctx, replica_idx=0)
    inputs = batch_constructor.construct_batch()
    assert [ctx.request_id for ctx in inputs.batches[0]] == [earlier.request_id]
    assert holder.request_id in batch_constructor._lookup_holds
    _preempt_to_the_pool_warm_on_replica_0(batch_constructor, earlier)
    if later_under_way:
        later.tokens.skip_processing(2)

    inputs = batch_constructor.construct_batch()

    if later_under_way:
        assert [ctx.request_id for ctx in inputs.batches[0]] == [
            later.request_id,
            earlier.request_id,
        ]
    else:
        assert [ctx.request_id for ctx in inputs.batches[0]] == [
            earlier.request_id
        ]
        assert list(batch_constructor.replicas[0].ce_reqs) == [later.request_id]


def test_dp_ce_balance__a_rebound_prompt_queues_last_without_a_lookup_hold() -> (
    None
):
    """With no lookup hold a preempted prompt bound again still queues last.

    Nothing waits on bind order without a connector lookup out, so the
    prompts queued on the replica keep the order they have always had.
    """
    batch_constructor = create_dp_balance_constructor()
    earlier = create_lora_context(seq_len=40)
    later = create_lora_context(seq_len=10)
    batch_constructor.enqueue_new_request(earlier, replica_idx=0)
    batch_constructor.construct_batch()
    batch_constructor.enqueue_new_request(later, replica_idx=0)
    _preempt_to_the_pool_warm_on_replica_0(batch_constructor, earlier)

    inputs = batch_constructor.construct_batch()

    assert [ctx.request_id for ctx in inputs.batches[0]] == [
        later.request_id,
        earlier.request_id,
    ]


def test_dp_ce_balance__a_hold_on_another_replica_leaves_a_rebind_last() -> (
    None
):
    """A lookup hold on one replica does not reorder a rebind on another.

    Bind order only matters where a hold keeps prompts waiting, so the
    rebound prompt on a replica with no hold of its own queues last.
    """
    batch_constructor = create_dp_balance_constructor()
    earlier = create_lora_context(seq_len=40)
    holder = create_lora_context(seq_len=10)
    later = create_lora_context(seq_len=10)
    _only_starts_a_lookup(batch_constructor, holder)
    batch_constructor.enqueue_new_request(earlier, replica_idx=0)
    batch_constructor.enqueue_new_request(holder, replica_idx=1)
    batch_constructor.construct_batch()
    assert batch_constructor._lookup_holds[holder.request_id].replica_idx == 1
    batch_constructor.enqueue_new_request(later, replica_idx=0)
    _preempt_to_the_pool_warm_on_replica_0(batch_constructor, earlier)

    inputs = batch_constructor.construct_batch()

    assert [ctx.request_id for ctx in inputs.batches[0]] == [
        later.request_id,
        earlier.request_id,
    ]


def test_dp_ce_balance__an_expired_hold_leaves_a_rebind_last() -> None:
    """A hold past ``_LOOKUP_HOLD_S`` does not reorder a rebind.

    It no longer keeps any prompt waiting, so the rebound prompt queues
    last as it does with no hold.
    """
    batch_constructor = create_dp_balance_constructor()
    earlier = create_lora_context(seq_len=40)
    holder = create_lora_context(seq_len=10)
    later = create_lora_context(seq_len=10)
    _only_starts_a_lookup(batch_constructor, holder)
    for ctx in (earlier, holder, later):
        batch_constructor.enqueue_new_request(ctx, replica_idx=0)
    batch_constructor.construct_batch()
    hold = batch_constructor._lookup_holds[holder.request_id]
    batch_constructor._lookup_holds[holder.request_id] = dataclasses.replace(
        hold, since=hold.since - text_batch_constructor._LOOKUP_HOLD_S
    )
    _preempt_to_the_pool_warm_on_replica_0(batch_constructor, earlier)

    inputs = batch_constructor.construct_batch()

    assert [ctx.request_id for ctx in inputs.batches[0]] == [
        later.request_id,
        earlier.request_id,
    ]


def test_dp_ce_balance__a_rebind_stays_behind_an_earlier_queued_prompt() -> (
    None
):
    """A rebound prompt goes only ahead of prompts bound after it.

    The first prompt was bound before it and preempted back to the head of
    the queue on its pinned replica, so it keeps its place in front.
    """
    batch_constructor = create_dp_balance_constructor()
    first = create_lora_context(seq_len=10)
    earlier = create_lora_context(seq_len=40)
    holder = create_lora_context(seq_len=10)
    later = create_lora_context(seq_len=10)
    _only_starts_a_lookup(batch_constructor, holder)
    for ctx in (first, earlier, holder, later):
        batch_constructor.enqueue_new_request(ctx, replica_idx=0)
    inputs = batch_constructor.construct_batch()
    assert [ctx.request_id for ctx in inputs.batches[0]] == [
        first.request_id,
        earlier.request_id,
    ]
    _preempt_to_the_pool_warm_on_replica_0(batch_constructor, earlier)
    batch_constructor._preempt_request(
        first, 0, reason=PreemptionReason.KV_CACHE_MEMORY
    )
    assert next(iter(batch_constructor.replicas[0].ce_reqs)) == first.request_id

    inputs = batch_constructor.construct_batch()

    assert [ctx.request_id for ctx in inputs.batches[0]] == [
        first.request_id,
        earlier.request_id,
    ]
    assert list(batch_constructor.replicas[0].ce_reqs) == [later.request_id]


def test_dp_ce_balance__a_held_rebind_stays_behind_a_later_lookup() -> None:
    """A rebound prompt the hold keeps waiting too does not move up.

    It was bound after the request out on a lookup, so at the head of the
    queue it would end the pass with nothing to do. A later prompt behind
    it that starts its own lookup would then never get to ask, and its
    lookup would wait on the first one's instead of running beside it.
    """
    batch_constructor = create_dp_balance_constructor()
    holder = create_lora_context(seq_len=10)
    earlier = create_lora_context(seq_len=40)
    later = create_lora_context(seq_len=10)
    kv_cache = _mock_kv_cache_of(batch_constructor)
    lookup = _IncompleteOnload()
    kv_cache.num_req_blocks = Mock(return_value=0)
    kv_cache.starts_lookup = Mock(
        side_effect=lambda ctx: ctx is holder or ctx is later
    )
    kv_cache.awaits_lookup = Mock(
        side_effect=lambda ctx: ctx is holder or ctx is later
    )
    kv_cache.start_lookup = Mock(return_value=_IncompleteOnload())
    kv_cache.alloc = Mock(
        side_effect=lambda ctx, defer_lookup=False: (
            lookup if ctx is holder else CompletedTransfer()
        )
    )
    batch_constructor.enqueue_new_request(holder, replica_idx=0)
    batch_constructor.enqueue_new_request(earlier, replica_idx=1)
    inputs = batch_constructor.construct_batch()
    assert [ctx.request_id for ctx in inputs.batches[1]] == [earlier.request_id]
    assert holder.request_id in batch_constructor._lookup_holds
    batch_constructor.enqueue_new_request(later, replica_idx=0)
    _stub_per_request_hits(
        batch_constructor,
        {
            earlier.request_id: [
                PrefixCacheHits(device_blocks=2),
                PrefixCacheHits(),
            ]
        },
    )
    batch_constructor._ce_arrival[earlier.request_id] = time.monotonic() - 60.0
    batch_constructor._preempt_request(
        earlier, 1, reason=PreemptionReason.KV_CACHE_MEMORY
    )
    assert earlier.request_id in batch_constructor._ce_pending

    batch_constructor.construct_batch()

    assert later.request_id in batch_constructor._onloading_reqs
    assert list(batch_constructor.replicas[0].ce_reqs) == [earlier.request_id]


def test_dp_ce_balance__a_rebind_keeps_the_held_prompts_in_order() -> None:
    """A rebound prompt goes ahead of every held prompt, not just the first.

    The held prompts behind it keep the order they had.
    """
    batch_constructor = create_dp_balance_constructor()
    earlier = create_lora_context(seq_len=40)
    holder = create_lora_context(seq_len=10)
    first_later = create_lora_context(seq_len=10)
    second_later = create_lora_context(seq_len=10)
    _only_starts_a_lookup(batch_constructor, holder)
    for ctx in (earlier, holder, first_later, second_later):
        batch_constructor.enqueue_new_request(ctx, replica_idx=0)
    batch_constructor.construct_batch()
    _preempt_to_the_pool_warm_on_replica_0(batch_constructor, earlier)
    ce_reqs = batch_constructor.replicas[0].ce_reqs

    batch_constructor._plan_ce_step()

    assert list(ce_reqs) == [
        earlier.request_id,
        first_later.request_id,
        second_later.request_id,
    ]


def test_dp_ce_balance__a_rebind_puts_earlier_queued_prompts_in_bind_order() -> (
    None
):
    """An earlier prompt queued behind a held one moves up with the rebind.

    It was bound back while no hold was live, so it queued last behind a
    later prompt the hold now keeps waiting. Neither it nor the rebound
    prompt is held, so both go ahead of the held prompt in bind order, the
    rebound prompt first since it was bound first.
    """
    batch_constructor = create_dp_balance_constructor()
    earlier = create_lora_context(seq_len=40)
    holder = create_lora_context(seq_len=10)
    later = create_lora_context(seq_len=10)
    first = create_lora_context(seq_len=10)
    _only_starts_a_lookup(batch_constructor, holder)
    for ctx in (earlier, holder, later):
        batch_constructor.enqueue_new_request(ctx, replica_idx=0)
    batch_constructor.construct_batch()
    batch_constructor.enqueue_new_request(first, replica_idx=0)
    # Both were bound before the hold, the rebound prompt first, and a
    # prompt bound again from the pool keeps its place in bind order.
    batch_constructor._bind_seq[earlier.request_id] = -2
    batch_constructor._bind_seq[first.request_id] = -1
    _preempt_to_the_pool_warm_on_replica_0(batch_constructor, earlier)

    inputs = batch_constructor.construct_batch()

    assert [ctx.request_id for ctx in inputs.batches[0]] == [
        earlier.request_id,
        first.request_id,
    ]
    assert list(batch_constructor.replicas[0].ce_reqs) == [later.request_id]


def test_dp_ce_balance__a_rebind_stays_behind_a_later_landed_onload() -> None:
    """A rebound prompt does not pass a later prompt that holds pages.

    The later prompt was bound after the request out on a lookup, but its
    onload landed, so it holds every page its next forward needs and the
    hold does not keep it waiting. A rebound prompt still has to allocate
    its whole prompt, so it stays behind it, as a readmission orders them.
    """
    batch_constructor = create_dp_balance_constructor()
    earlier = create_lora_context(seq_len=40)
    holder = create_lora_context(seq_len=10)
    loaded = create_lora_context(seq_len=10)
    kv_cache = _mock_kv_cache_of(batch_constructor)
    lookup = _IncompleteOnload()
    onload = _IncompleteOnload()
    kv_cache.contains = Mock(side_effect=lambda ctx: ctx is loaded)
    kv_cache.num_req_blocks = Mock(
        side_effect=lambda ctx: 2 if ctx is loaded else 0
    )
    kv_cache.starts_lookup = Mock(side_effect=lambda ctx: ctx is holder)
    kv_cache.awaits_lookup = Mock(side_effect=lambda ctx: ctx is holder)

    def alloc(ctx: TextContext, defer_lookup: bool = False) -> KVTransfer:
        if ctx is holder:
            return lookup
        if ctx is loaded and not onload.complete:
            return onload
        return CompletedTransfer()

    kv_cache.alloc = Mock(side_effect=alloc)
    for ctx in (earlier, holder, loaded):
        batch_constructor.enqueue_new_request(ctx, replica_idx=0)
    inputs = batch_constructor.construct_batch()
    assert [ctx.request_id for ctx in inputs.batches[0]] == [earlier.request_id]
    assert loaded.request_id in batch_constructor._onloading_reqs
    assert holder.request_id in batch_constructor._lookup_holds
    _preempt_to_the_pool_warm_on_replica_0(batch_constructor, earlier)
    onload.synchronize()

    inputs = batch_constructor.construct_batch()

    assert [ctx.request_id for ctx in inputs.batches[0]] == [
        loaded.request_id,
        earlier.request_id,
    ]


def test_text_batch_constructor__readmits_a_landed_onload_ahead_of_a_prefill(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """A request whose onload landed goes back to the head of its queue.

    It already holds the pages its next forward needs, so it cannot fail
    its alloc at the head of the queue the way a request back from a lookup
    can, and a prefill under way has no reason to pass it.
    """
    holder = _ce_ctx_of(50)
    holder.tokens.skip_processing(10)
    onloaded = _ce_ctx()
    onload = _IncompleteOnload()

    def alloc(ctx: TextContext, defer_lookup: bool = False) -> KVTransfer:
        if ctx is onloaded and not onload.complete:
            return onload
        return CompletedTransfer()

    kv_cache, _ = _claim_tracking_kv_cache(alloc, holder)
    batch_constructor = TextBatchConstructor(
        scheduler_config=_cordon_config(),
        pipeline=pipeline,
        kv_cache=kv_cache,
    )
    batch_constructor.enqueue_new_request(onloaded)
    batch_constructor.construct_batch()
    assert onloaded.request_id in batch_constructor._onloading_reqs
    batch_constructor.enqueue_new_request(holder)

    onload.synchronize()
    inputs = batch_constructor.construct_batch()

    assert [ctx.request_id for ctx in inputs.batches[0]] == [
        onloaded.request_id,
        holder.request_id,
    ]


def test_text_batch_constructor__readmits_a_failed_onload_behind_a_prefill(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """A request whose onload failed does not jump a prefill under way.

    Its next alloc rolls the failed prefix back and takes the whole prompt
    again, so it could fail at the head of the queue every pass while the
    prefill behind it never runs to free anything. It waits behind it.
    """
    holder = _ce_ctx_of(50)
    holder.tokens.skip_processing(10)
    onloaded = _ce_ctx()
    onload = _IncompleteOnload()

    def alloc(ctx: TextContext, defer_lookup: bool = False) -> KVTransfer:
        if ctx is onloaded and not onload.fails:
            return onload
        return CompletedTransfer()

    kv_cache, _ = _claim_tracking_kv_cache(alloc, holder)
    batch_constructor = TextBatchConstructor(
        scheduler_config=_cordon_config(),
        pipeline=pipeline,
        kv_cache=kv_cache,
    )
    batch_constructor.enqueue_new_request(onloaded)
    batch_constructor.construct_batch()
    assert onloaded.request_id in batch_constructor._onloading_reqs
    batch_constructor.enqueue_new_request(holder)

    onload.fails = True
    inputs = batch_constructor.construct_batch()

    assert [ctx.request_id for ctx in inputs.batches[0]] == [holder.request_id]
    assert onloaded.request_id in batch_constructor.replicas[0].ce_reqs


def test_text_batch_constructor__a_re_ask_waits_for_room_under_the_bound(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """A lookup asked again past a shrunken device hit waits for the bound.

    Asking again starts another leased read, so with the replica's one
    place taken, the request waits at the head of its queue with its
    answer kept and nothing asked, rather than loading across the gap.
    """
    connector = _ScheduledConnector(answers=[2, None], first_hits=True)
    batch_constructor, manager = _batch_constructor_over_jenga(
        pipeline, connector, num_blocks=64
    )
    device = TextContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.ones(3, dtype=np.int64)),
        max_length=100,
    )
    manager.claim(device)
    manager.alloc(device)
    device.update(ARBITRARY_TOKEN_ID)
    manager.step(device)
    manager.release(device)
    ctx = _ce_ctx()
    other = TextContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.full(9, 2, dtype=np.int64)),
        max_length=100,
    )
    batch_constructor.enqueue_new_request(ctx)
    batch_constructor.enqueue_new_request(other)
    batch_constructor.construct_batch()
    pending = manager._state_of(ctx).pending_lookup
    assert pending is not None and pending.first_block > 0
    manager.pools[0].reset_prefix_cache()
    # What MAX_SERVE_MAX_PENDING_CONNECTOR_LOOKUPS=1 sets, lowered only now
    # so that both lookups could start.
    batch_constructor.max_pending_lookups = 1

    inputs = batch_constructor.construct_batch()

    assert not has_request(inputs.batches[0], ctx.request_id)
    assert len(connector.started) == 2
    assert connector.abandoned == []
    assert list(batch_constructor.replicas[0].ce_reqs)[:1] == [ctx.request_id]
    assert manager._state_of(ctx).pending_lookup is not None


def test_text_batch_constructor__readmits_request_whose_onload_failed(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """A failed onload brings its request back rather than the worker down.

    The request is re-admitted like one whose onload landed. Its next
    ``alloc`` is where the cache manager rolls the onloaded prefix back, so
    the second ``alloc`` here returning a complete transfer stands in for the
    recompute.
    """
    onload = _IncompleteOnload()
    alloc = Mock(side_effect=[onload, CompletedTransfer()])
    kv_cache = _make_cordon_kv_cache(alloc)
    batch_constructor = TextBatchConstructor(
        scheduler_config=_cordon_config(),
        pipeline=pipeline,
        kv_cache=kv_cache,
    )
    ctx = _ce_ctx()
    batch_constructor.enqueue_new_request(ctx)
    batch_constructor.construct_batch()
    assert ctx.request_id in batch_constructor._onloading_reqs

    onload.fails = True
    inputs = batch_constructor.construct_batch()

    assert has_request(inputs.batches[0], ctx.request_id)
    assert ctx.request_id not in batch_constructor._onloading_reqs
    assert alloc.call_count == 2


def test_text_batch_constructor__cordons_onload_that_fails_at_admission(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """A copy that has already failed when admission polls it is cordoned.

    Scheduling it would run a forward over pages the copy never filled. The
    cordon holds it until the sweep sees the failure and re-admits it.
    """
    onload = _IncompleteOnload()
    onload.fails = True
    alloc = Mock(side_effect=[onload, CompletedTransfer()])
    kv_cache = _make_cordon_kv_cache(alloc)
    batch_constructor = TextBatchConstructor(
        scheduler_config=_cordon_config(),
        pipeline=pipeline,
        kv_cache=kv_cache,
    )
    ctx = _ce_ctx()
    batch_constructor.enqueue_new_request(ctx)

    inputs = batch_constructor.construct_batch()
    assert len(inputs.batches[0]) == 0
    assert ctx.request_id in batch_constructor._onloading_reqs

    inputs = batch_constructor.construct_batch()
    assert has_request(inputs.batches[0], ctx.request_id)
    assert alloc.call_count == 2


def test_text_batch_constructor__oom_deferred_while_onload_in_flight(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """OOM does not crash the server while a cordoned onload is in flight.

    The first request is cordoned (onload in flight); the second hits
    ``InsufficientBlocksError``. A's cordoned onload is enough on its own
    (presence, not magnitude) for B to defer rather than raise.
    """
    onload = _IncompleteOnload()
    kv_cache = _make_cordon_kv_cache(
        Mock(
            side_effect=[
                onload,
                InsufficientBlocksError("insufficient blocks"),
            ]
        )
    )
    kv_cache.num_req_blocks = Mock(return_value=5)
    kv_cache.get_req_blocks_per_leaf = Mock(
        return_value={"full": [0, 1, 2, 3, 4]}
    )
    batch_constructor = TextBatchConstructor(
        scheduler_config=_cordon_config(),
        pipeline=pipeline,
        kv_cache=kv_cache,
    )
    ctx_a = _ce_ctx()
    ctx_b = _ce_ctx()
    batch_constructor.enqueue_new_request(ctx_a)
    batch_constructor.enqueue_new_request(ctx_b)

    # Must not raise: A cordons, B's OOM defers (A's cordoned blocks cover it).
    inputs = batch_constructor.construct_batch()

    assert len(inputs.batches[0]) == 0
    assert ctx_a.request_id in batch_constructor._onloading_reqs
    assert ctx_b.request_id in batch_constructor.replicas[0].ce_reqs


# ---------------------------------------------------------------------------
# Prefill schedule interval (_is_prefill_interval_open) tests
# ---------------------------------------------------------------------------


def _interval_constructor(
    interval: int = 1,
    dp: int = 1,
    in_flight_batching: bool = False,
    coalesce_min_pending: int = 0,
    coalesce_max_held_steps: int = 0,
    chunked: bool = False,
) -> TextBatchConstructor:
    """A constructor that admits exactly one prefill per open step.

    With chunking off the CE target is a soft limit, so the first 30-token
    prefill is admitted whole and ends the pass, leaving the rest queued.
    """
    pipeline = Mock(spec=["release"])
    pipeline.release = Mock()
    kv_cache = create_mock_kv_cache()
    kv_cache.num_replicas = dp
    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=10,
        target_tokens_per_batch_ce=20,
        enable_chunked_prefill=chunked,
        data_parallel_degree=dp,
        enable_in_flight_batching=in_flight_batching,
        prefill_coalesce_min_pending=coalesce_min_pending,
        prefill_coalesce_max_held_steps=coalesce_max_held_steps,
        prefill_schedule_interval=interval,
    )
    return TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
    )


def _enqueue_ce(
    batch_constructor: TextBatchConstructor,
    count: int = 1,
    replica_idx: int = 0,
) -> None:
    for _ in range(count):
        batch_constructor.enqueue_new_request(
            create_lora_context(), replica_idx=replica_idx
        )


def _enqueue_tg(
    batch_constructor: TextBatchConstructor, replica_idx: int = 0
) -> None:
    batch_constructor.enqueue_new_request(
        create_lora_context(is_tg=True), replica_idx=replica_idx
    )


def _has_prefill(batch: list[TextContext]) -> bool:
    return any(ctx.tokens.generated_length == 0 for ctx in batch)


def test_prefill_schedule_interval__admits_prefill_only_on_cadence_steps() -> (
    None
):
    batch_constructor = _interval_constructor(interval=4)
    _enqueue_tg(batch_constructor)
    _enqueue_ce(batch_constructor, count=2)

    admitted = [
        _has_prefill(batch_constructor.construct_batch().batches[0])
        for _ in range(5)
    ]
    assert admitted == [True, False, False, False, True]


def test_prefill_schedule_interval__admits_prefill_when_nothing_decodes() -> (
    None
):
    batch_constructor = _interval_constructor(interval=4)
    _enqueue_ce(batch_constructor)

    # No TG work anywhere: holding would idle the GPU for nothing.
    assert _has_prefill(batch_constructor.construct_batch().batches[0])


def test_prefill_schedule_interval__one_matches_the_flag_being_absent() -> None:
    # An absent flag is exactly an interval of 1.
    absent = TokenGenerationSchedulerConfig(
        max_batch_size=10, target_tokens_per_batch_ce=20
    )
    assert absent.prefill_schedule_interval == 1

    batch_constructor = _interval_constructor(interval=1)
    _enqueue_tg(batch_constructor)
    _enqueue_ce(batch_constructor, count=3)

    steps: list[tuple[int, bool]] = []
    for _ in range(4):
        batch = batch_constructor.construct_batch().batches[0]
        steps.append((len(batch), _has_prefill(batch)))

    # Three queued prefills, one per step, then a pure decode step.
    assert steps == [(1, True), (1, True), (1, True), (1, False)]


def test_prefill_schedule_interval__dp_replicas_prefill_on_the_same_step() -> (
    None
):
    dp = 4
    batch_constructor = _interval_constructor(interval=4, dp=dp)
    for replica_idx in range(dp):
        _enqueue_tg(batch_constructor, replica_idx=replica_idx)

    # Step 0 is pure decode; each replica's prefill then arrives on a
    # different step, and all of them wait for the shared cadence step.
    batch_constructor.construct_batch()
    for step in range(1, 5):
        _enqueue_ce(batch_constructor, replica_idx=step - 1)
        inputs = batch_constructor.construct_batch()
        admitted = [_has_prefill(inputs.batches[i]) for i in range(dp)]
        assert admitted == [step == 4] * dp


def test_prefill_schedule_interval__holds_in_flight_batching_backfill() -> None:
    batch_constructor = _interval_constructor(
        interval=4, in_flight_batching=True
    )
    _enqueue_tg(batch_constructor)
    _enqueue_ce(batch_constructor, count=2)

    # The backfill path never consults _identify_priority, so this covers
    # the _add_ce_requests guard on its own.
    admitted = [
        _has_prefill(batch_constructor.construct_batch().batches[0])
        for _ in range(5)
    ]
    assert admitted == [True, False, False, False, True]


def test_prefill_schedule_interval__composes_with_prefill_coalescing() -> None:
    batch_constructor = _interval_constructor(
        interval=4, in_flight_batching=True, coalesce_min_pending=2
    )
    _enqueue_tg(batch_constructor)
    _enqueue_ce(batch_constructor)

    # Cadence open, but one pending prefill is below the coalesce threshold.
    assert not _has_prefill(batch_constructor.construct_batch().batches[0])

    # A second pending opens the coalesce gate; the cadence still holds.
    _enqueue_ce(batch_constructor)
    for _ in range(3):
        assert not _has_prefill(batch_constructor.construct_batch().batches[0])

    assert _has_prefill(batch_constructor.construct_batch().batches[0])


def test_prefill_schedule_interval__idle_replica_stays_silent_when_held() -> (
    None
):
    """A held step must not turn an idle replica into a TG vote.

    SERVOPT-1560: a batch-wide override assembled from idle replicas'
    priorities starves a sibling's real CE request.
    """
    batch_constructor = _interval_constructor(interval=4, dp=2)
    _enqueue_tg(batch_constructor, replica_idx=0)
    _enqueue_ce(batch_constructor, replica_idx=0)

    # construct_batch decides the cadence for the step it is starting, so it
    # takes two passes to reach one that is held: step 0 is open, step 1 is not.
    batch_constructor.construct_batch()
    batch_constructor.construct_batch()
    assert not batch_constructor._prefill_interval_open

    assert batch_constructor._identify_priority(0) == RequestType.TG
    assert batch_constructor._identify_priority(1) is None


# ---------------------------------------------------------------------------
# Prefill coalescing (_should_backfill_ce) tests
# ---------------------------------------------------------------------------


def test_prefill_coalescing__threshold_counts_the_group_not_each_replica() -> (
    None
):
    """MXSERV-497: prefills spread thin across replicas still clear the bar.

    One eager step serves the whole DP group, so what the threshold has to
    price is how many prefills that one step absorbs in total.
    """
    dp = 4
    batch_constructor = _interval_constructor(
        dp=dp, in_flight_batching=True, coalesce_min_pending=4
    )
    for replica_idx in range(dp):
        _enqueue_tg(batch_constructor, replica_idx=replica_idx)
        _enqueue_ce(batch_constructor, replica_idx=replica_idx)

    # One prefill per replica: each is below a threshold of 4 on its own, the
    # group is exactly at it. Counting per replica would hold all four.
    inputs = batch_constructor.construct_batch()
    assert [_has_prefill(inputs.batches[i]) for i in range(dp)] == [True] * dp
    assert batch_constructor._prefill_coalesce_held_steps == 0


def test_prefill_coalescing__max_held_steps_releases_before_the_threshold() -> (
    None
):
    batch_constructor = _interval_constructor(
        in_flight_batching=True,
        coalesce_min_pending=10,
        coalesce_max_held_steps=2,
    )
    _enqueue_tg(batch_constructor)
    _enqueue_ce(batch_constructor)

    # A single pending prefill never reaches a threshold of 10, so the
    # separate two-step deadline is the only thing that can release it.
    admitted = [
        _has_prefill(batch_constructor.construct_batch().batches[0])
        for _ in range(3)
    ]
    assert admitted == [False, False, True]


def test_prefill_coalescing__max_held_steps_zero_keeps_one_number() -> None:
    batch_constructor = _interval_constructor(
        in_flight_batching=True, coalesce_min_pending=3
    )
    _enqueue_tg(batch_constructor)
    _enqueue_ce(batch_constructor)

    # Unset, the threshold is still the deadline: one pending prefill waits
    # exactly three steps, as it did before the deadline was split out.
    admitted = [
        _has_prefill(batch_constructor.construct_batch().batches[0])
        for _ in range(4)
    ]
    assert admitted == [False, False, False, True]


def test_prefill_coalescing__no_decode_work_anywhere_does_not_hold() -> None:
    """A step with no decode work runs prefill through the CE branch anyway.

    Counting those as held would spend the deadline on steps that admitted
    prefill, releasing the next genuinely held one early.
    """
    batch_constructor = _interval_constructor(
        in_flight_batching=True, coalesce_min_pending=3
    )
    _enqueue_ce(batch_constructor, count=2)

    for step in range(3):
        inputs = batch_constructor.construct_batch()
        assert _has_prefill(inputs.batches[0]) == (step < 2)
        assert batch_constructor._prefill_coalesce_held_steps == 0


def test_prefill_coalescing__deadline_releases_on_a_cadence_step() -> None:
    """The held-steps clock must not tick while the cadence is shut.

    Ticking there lets the deadline expire on a step that admits nothing, and
    the reset then keeps the two gates permanently out of phase: coalescing
    opens every deadline+1 steps, the cadence every interval steps, and with
    the wrong offset they never coincide. The threshold here is far above the
    queue depth so only the deadline can release.
    """
    interval, deadline = 4, 3
    batch_constructor = _interval_constructor(
        interval=interval,
        in_flight_batching=True,
        coalesce_min_pending=10,
        coalesce_max_held_steps=deadline,
    )
    _enqueue_tg(batch_constructor)
    _enqueue_ce(batch_constructor)

    admitted = [
        _has_prefill(batch_constructor.construct_batch().batches[0])
        for _ in range(interval * (deadline + 1) + 1)
    ]
    assert any(admitted)


def test_prefill_coalescing__deferred_replicas_do_not_clear_the_threshold() -> (
    None
):
    """Deferred CE work cannot be admitted, so it must not clear the gate.

    Counting it would let prefills that provably cannot run this step clear
    the threshold and reset the held-steps clock.
    """
    batch_constructor = _interval_constructor(
        dp=2, in_flight_batching=True, coalesce_min_pending=2
    )
    _enqueue_tg(batch_constructor, replica_idx=0)
    _enqueue_tg(batch_constructor, replica_idx=1)
    _enqueue_ce(batch_constructor, count=2, replica_idx=1)

    batch_constructor._ce_deferred_replicas = {1}

    assert not batch_constructor._should_backfill_ce()
    assert batch_constructor._prefill_coalesce_held_steps == 0


def test_prefill_schedule_interval__holds_mid_prefill_continuations() -> None:
    """The cadence holds continuations; coalescing releases on them.

    The budget emits one chunk per step, so a chunked prefill needs one
    cadence-open step per chunk. Exempting continuations would let a long
    prefill run every step and defeat the cadence entirely.
    """
    batch_constructor = _interval_constructor(interval=2, chunked=True)
    _enqueue_tg(batch_constructor)
    _enqueue_ce(batch_constructor)  # 30 tokens, 20-token budget -> 2 chunks

    admitted = []
    for _ in range(3):
        inputs = batch_constructor.construct_batch()
        batch = inputs.batches[0]
        admitted.append(_has_prefill(batch))
        # Stand in for execution: advancing past the processed chunk is what
        # gives the request a non-zero processed_length, i.e. what makes it a
        # continuation when advance_requests returns it to the CE queue.
        for context in batch:
            if context.tokens.actively_chunked:
                context.tokens.advance_chunk()
        batch_constructor.advance_requests(inputs)

    # One chunk per open step; the continuation waits out the closed step.
    assert admitted == [True, False, True]
