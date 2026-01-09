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

from unittest.mock import Mock

import numpy as np
import pytest
from max.interfaces import (
    GenerationStatus,
    Pipeline,
    RequestID,
    TextGenerationInputs,
    TextGenerationOutput,
    TokenBuffer,
)
from max.kv_cache.paged_cache.block_utils import InsufficientBlocksError
from max.pipelines.core import TextContext
from max.serve.scheduler.batch_constructor.text_batch_constructor import (
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


def create_mock_paged_cache() -> Mock:
    """Create a mock paged KV cache manager with minimal interface."""
    cache = Mock()
    cache.max_seq_len = 2048
    cache.page_size = 16
    cache.total_num_pages = 128
    cache.free_blocks_pct = 0.5

    cache.alloc = Mock()
    cache.claim = Mock()
    cache.release = Mock()
    cache.contains = Mock(return_value=False)
    cache.get_or_recommend_replica = Mock(return_value=0)
    cache.get_pct_used_blocks_after_allocation = Mock(return_value=0.94)

    return cache


def create_mock_pipeline_with_lora(lora_manager: Mock) -> Mock:
    """Create a mock pipeline with LoRA support."""

    def next_token_behavior(
        inputs: TextGenerationInputs[TextContext],
    ) -> dict[RequestID, TextGenerationOutput]:
        responses: dict[RequestID, TextGenerationOutput] = {}

        for request_id, request in inputs.batch.items():
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


def test_text_batch_constructor__batch_construction_without_chunked_prefill_no_preemption(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=5,
        max_batch_context_length=None,
        max_forward_steps_tg=10,
        enable_in_flight_batching=False,
        enable_chunked_prefill=False,
        target_tokens_per_batch_ce=30,
    )

    paged_cache = Mock()
    paged_cache.alloc = Mock()
    paged_cache.alloc.return_value = True
    paged_cache.claim = Mock()
    paged_cache.contains = Mock()
    paged_cache.get_pct_used_blocks_after_allocation = Mock()
    paged_cache.get_pct_used_blocks_after_allocation.return_value = 0.0

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        paged_cache=paged_cache,
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
    # 9 * 4 = 36 tokens, since no max_batch_context_length is set, we should have 4 requests in the batch
    assert len(inputs.batches[0]) == 4
    # since this is CE, we should have 1 step
    assert inputs.num_steps == 1

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
        for context in batch.values():
            context.update(0)

    chunked_request_ids = (
        batch_constructor.advance_requests_and_collect_invalid_ids(
            inputs.batches
        )
    )
    assert len(chunked_request_ids) == 0

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
    assert inputs.num_steps == 1

    for batch in inputs.batches:
        for context in batch.values():
            context.update(0)

    chunked_request_ids = (
        batch_constructor.advance_requests_and_collect_invalid_ids(
            inputs.batches
        )
    )
    assert len(chunked_request_ids) == 0

    assert len(batch_constructor.replicas[0].ce_reqs) == 0
    assert len(batch_constructor.replicas[0].tg_reqs) == 4

    # Assume that we have 4 requests remaining in the queue
    # And none of the requests have a max length, therefore we use the default
    assert batch_constructor._identify_priority(0) == RequestType.TG
    inputs = batch_constructor.construct_batch()
    assert len(inputs.batches[0]) == 4
    assert inputs.num_steps == 10


def test_text_batch_constructor__batch_construction_no_requests(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=5,
        max_batch_context_length=None,
        max_forward_steps_tg=10,
        enable_in_flight_batching=False,
        enable_chunked_prefill=False,
        target_tokens_per_batch_ce=30,
    )

    paged_cache = Mock()
    paged_cache.alloc = Mock()
    paged_cache.alloc.return_value = True
    paged_cache.claim = Mock()
    paged_cache.contains = Mock()
    paged_cache.get_pct_used_blocks_after_allocation = Mock()
    paged_cache.get_pct_used_blocks_after_allocation.return_value = 0.0

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        paged_cache=paged_cache,
    )
    inputs = batch_constructor.construct_batch()
    assert len(inputs.batches) == 1
    assert len(inputs.batches[0]) == 0
    assert inputs.num_steps == 0


def test_text_batch_constructor__batch_construction_no_room_in_cache(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=5,
        max_batch_context_length=None,
        max_forward_steps_tg=10,
        enable_in_flight_batching=False,
        enable_chunked_prefill=False,
        target_tokens_per_batch_ce=30,
    )
    paged_cache = Mock()
    paged_cache.alloc = Mock()
    paged_cache.alloc.return_value = False
    paged_cache.alloc.side_effect = InsufficientBlocksError
    paged_cache.claim = Mock()
    paged_cache.contains = Mock()
    paged_cache.get_pct_used_blocks_after_allocation = Mock()
    paged_cache.get_pct_used_blocks_after_allocation.return_value = 0.0

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        paged_cache=paged_cache,
    )

    contexts = {}
    for _ in range(2):
        context = TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(9, dtype=np.int64)),
            max_length=100,
        )
        contexts[context.request_id] = context
        batch_constructor.enqueue_new_request(context)

    with pytest.raises(InsufficientBlocksError):
        inputs = batch_constructor.construct_batch()


def test_text_batch_constructor__batch_construction_with_chunked_prefill_and_preemption(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=5,
        max_batch_context_length=None,
        max_forward_steps_tg=10,
        enable_in_flight_batching=False,
        enable_chunked_prefill=True,
        target_tokens_per_batch_ce=30,
        kvcache_ce_watermark=0.95,
    )
    paged_cache = Mock()
    paged_cache.alloc = Mock()
    paged_cache.alloc.return_value = True
    paged_cache.claim = Mock()
    paged_cache.contains = Mock()
    paged_cache.get_pct_used_blocks_after_allocation = Mock()
    paged_cache.get_pct_used_blocks_after_allocation.return_value = 0.0

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        paged_cache=paged_cache,
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
    assert list(inputs.batches[0].values())[-1].tokens.generated_length == 0

    # Update a token for each request in the batch
    for batch in inputs.batches:
        for context in batch.values():
            context.update(0)

    chunked_request_ids = (
        batch_constructor.advance_requests_and_collect_invalid_ids(
            inputs.batches
        )
    )
    assert len(chunked_request_ids) == 1
    assert (
        chunked_request_ids[0]
        == list(inputs.batches[0].values())[-1].request_id
    )

    # There should now be 3 requests in TG, and 7 in CE
    assert len(batch_constructor.replicas[0].tg_reqs) == 3
    assert len(batch_constructor.replicas[0].ce_reqs) == 5

    # We should still be prioritizing CE
    assert batch_constructor._identify_priority(0) == RequestType.CE

    inputs = batch_constructor.construct_batch()
    # We only grab 2 new CE requests here, because we have 3 TG requests outstanding.
    # Since max_batch_size is 5, we can only have 5 requests outstanding at a time.
    assert len(inputs.batches[0]) == 2
    assert list(inputs.batches[0].values())[-1].tokens.generated_length == 0

    for batch in inputs.batches:
        for context in batch.values():
            context.update(0)

    chunked_request_ids = (
        batch_constructor.advance_requests_and_collect_invalid_ids(
            inputs.batches
        )
    )
    assert len(chunked_request_ids) == 0

    assert len(batch_constructor.replicas[0].ce_reqs) == 3
    assert len(batch_constructor.replicas[0].tg_reqs) == 5

    paged_cache.get_pct_used_blocks_after_allocation.return_value = 0.96

    # We still prioritize CE, but return an empty batch
    assert batch_constructor._identify_priority(0) == RequestType.CE

    # Since we generate an empty CE batch, we then fill with TG requests
    inputs = batch_constructor.construct_batch()
    assert len(inputs.batches[0]) == 5

    # Last Ce Batch
    paged_cache.get_pct_used_blocks_after_allocation.return_value = 0.0
    assert batch_constructor._identify_priority(0) == RequestType.CE
    inputs = batch_constructor.construct_batch()
    # Since we already have 5 CE request outstanding, we cannot grab any new CE requests.
    assert len(inputs.batches[0]) == 5

    for batch in inputs.batches:
        for context in batch.values():
            context.update(0)

    chunked_request_ids = (
        batch_constructor.advance_requests_and_collect_invalid_ids(
            inputs.batches
        )
    )
    assert len(chunked_request_ids) == 0

    assert len(batch_constructor.replicas[0].ce_reqs) == 3
    assert len(batch_constructor.replicas[0].tg_reqs) == 5

    # Test for Pre-emption
    # The first item won't have enough space, so we will pre-empt the last one
    # The first item will have 2 alloc calls, failing with InsufficientBlocksError on the first,
    # then succeeding and returning None for the remaining calls.
    paged_cache.alloc.side_effect = [
        InsufficientBlocksError(),
        None,
        None,
        None,
        None,
        None,
    ]

    last_request_id = list(batch_constructor.replicas[0].tg_reqs.keys())[-1]
    assert batch_constructor._identify_priority(0) == RequestType.CE
    assert len(batch_constructor.replicas[0].ce_reqs) == 3
    inputs = batch_constructor.construct_batch()
    assert len(inputs.batches[0]) == 4
    assert last_request_id not in inputs.batches[0]

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
        max_batch_context_length=None,
        max_forward_steps_tg=10,
        enable_in_flight_batching=True,
        enable_chunked_prefill=True,
        target_tokens_per_batch_ce=30,
        kvcache_ce_watermark=0.95,
    )
    paged_cache = Mock()
    paged_cache.alloc = Mock()
    paged_cache.alloc.return_value = True
    paged_cache.claim = Mock()
    paged_cache.contains = Mock()
    paged_cache.get_pct_used_blocks_after_allocation = Mock()
    paged_cache.get_pct_used_blocks_after_allocation.return_value = 0.0

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        paged_cache=paged_cache,
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
    assert list(inputs.batches[0].values())[-1].tokens.generated_length == 0

    # Update a token for each request in the batch
    for batch in inputs.batches:
        for context in batch.values():
            context.update(0)

    chunked_request_ids = (
        batch_constructor.advance_requests_and_collect_invalid_ids(
            inputs.batches
        )
    )
    assert len(chunked_request_ids) == 1
    assert (
        chunked_request_ids[0]
        == list(inputs.batches[0].values())[-1].request_id
    )

    # There should now be 3 requests in TG, and 7 in CE
    assert len(batch_constructor.replicas[0].tg_reqs) == 3
    assert len(batch_constructor.replicas[0].ce_reqs) == 5

    # We should now prioritize TG
    assert batch_constructor._identify_priority(0) == RequestType.TG
    inputs = batch_constructor.construct_batch()

    # We should have 5 requests
    assert len(inputs.batches[0]) == 7
    # Last item should be chunked, with a length of 3
    assert list(inputs.batches[0].values())[-1].tokens.generated_length == 0

    for batch in inputs.batches:
        for context in batch.values():
            context.update(0)

    chunked_request_ids = (
        batch_constructor.advance_requests_and_collect_invalid_ids(
            inputs.batches
        )
    )
    assert len(chunked_request_ids) == 1


def test_text_batch_constructor__batch_construction_without_chunked_prefill_and_inflight_batching(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=10,
        max_batch_context_length=None,
        max_forward_steps_tg=10,
        enable_in_flight_batching=True,
        enable_chunked_prefill=False,
        target_tokens_per_batch_ce=30,
    )
    paged_cache = Mock()
    paged_cache.alloc = Mock()
    paged_cache.alloc.return_value = True
    paged_cache.claim = Mock()
    paged_cache.contains = Mock()
    paged_cache.get_pct_used_blocks_after_allocation = Mock()
    paged_cache.get_pct_used_blocks_after_allocation.return_value = 0.0

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        paged_cache=paged_cache,
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
    assert list(inputs.batches[0].values())[-1].tokens.generated_length == 0

    # Update a token for each request in the batch
    for batch in inputs.batches:
        for context in batch.values():
            context.update(0)

    chunked_request_ids = (
        batch_constructor.advance_requests_and_collect_invalid_ids(
            inputs.batches
        )
    )
    assert len(chunked_request_ids) == 0

    assert len(batch_constructor.replicas[0].ce_reqs) == 4
    assert len(batch_constructor.replicas[0].tg_reqs) == 4

    assert batch_constructor._identify_priority(0) == RequestType.TG
    inputs = batch_constructor.construct_batch()
    assert len(inputs.batches[0]) == 7
    for i in range(len(inputs.batches[0])):
        if i < 4:
            # The first four requests are TG, and should not need CE
            assert (
                list(inputs.batches[0].values())[i].tokens.generated_length != 0
            )
        else:
            # The second four requests are CE, and should need CE
            assert (
                list(inputs.batches[0].values())[i].tokens.generated_length == 0
            )

    for batch in inputs.batches:
        for context in batch.values():
            context.update(0)

    chunked_request_ids = (
        batch_constructor.advance_requests_and_collect_invalid_ids(
            inputs.batches
        )
    )
    assert len(chunked_request_ids) == 0

    assert len(batch_constructor.replicas[0].ce_reqs) == 1


def test_single_lora_scheduling() -> None:
    """Test scheduling a single LoRA request in CE batch."""
    lora_manager = create_mock_lora_manager(max_num_loras=2)
    pipeline = create_mock_pipeline_with_lora(lora_manager)
    paged_cache = create_mock_paged_cache()

    config = TokenGenerationSchedulerConfig(
        max_batch_size=4,
        max_forward_steps_tg=1,
        target_tokens_per_batch_ce=100,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=config,
        pipeline=pipeline,
        paged_cache=paged_cache,
    )

    ctx = create_lora_context(model_name="lora_model1")
    batch_constructor.enqueue_new_request(ctx)

    output = batch_constructor.construct_batch()

    assert len(output.batches[0]) == 1
    assert ctx.request_id in output.batches[0]
    lora_manager.activate_adapter.assert_called_once_with("lora_model1")
    assert "lora_model1" in lora_manager._active_loras


def test_multi_lora_within_budget() -> None:
    """Test scheduling multiple LoRA requests within budget."""
    lora_manager = create_mock_lora_manager(max_num_loras=3)
    pipeline = create_mock_pipeline_with_lora(lora_manager)
    paged_cache = create_mock_paged_cache()

    config = TokenGenerationSchedulerConfig(
        max_batch_size=4,
        max_forward_steps_tg=1,
        target_tokens_per_batch_ce=200,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=config,
        pipeline=pipeline,
        paged_cache=paged_cache,
    )

    ctx1 = create_lora_context(model_name="lora_model1")
    ctx2 = create_lora_context(model_name="lora_model2")
    ctx3 = create_lora_context(model_name="lora_model3")

    batch_constructor.enqueue_new_request(ctx1)
    batch_constructor.enqueue_new_request(ctx2)
    batch_constructor.enqueue_new_request(ctx3)

    output = batch_constructor.construct_batch()
    assert len(output.batches[0]) == 3
    assert ctx1.request_id in output.batches[0]
    assert ctx2.request_id in output.batches[0]
    assert ctx3.request_id in output.batches[0]
    assert len(lora_manager._active_loras) == 3


def test_lora_preemption_over_budget() -> None:
    """Test that LoRA requests are deferred when over budget during CE."""
    lora_manager = create_mock_lora_manager(max_num_loras=2)
    pipeline = create_mock_pipeline_with_lora(lora_manager)
    paged_cache = create_mock_paged_cache()

    config = TokenGenerationSchedulerConfig(
        max_batch_size=5,
        max_forward_steps_tg=1,
        target_tokens_per_batch_ce=200,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=config,
        pipeline=pipeline,
        paged_cache=paged_cache,
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
    assert ctx_base.request_id in output.batches[0]
    assert ctx_lora1.request_id in output.batches[0]
    assert ctx_lora2.request_id in output.batches[0]
    assert ctx_lora3.request_id not in output.batches[0]

    assert ctx_lora3.request_id in batch_constructor.all_ce_reqs


def test_age_based_scheduling_with_lora() -> None:
    """Test that age-based scheduling is maintained with LoRA constraints."""
    lora_manager = create_mock_lora_manager(max_num_loras=2)
    pipeline = create_mock_pipeline_with_lora(lora_manager)
    paged_cache = create_mock_paged_cache()

    config = TokenGenerationSchedulerConfig(
        max_batch_size=4,
        max_forward_steps_tg=1,
        target_tokens_per_batch_ce=40,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=config,
        pipeline=pipeline,
        paged_cache=paged_cache,
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
    assert ctx_inactive.request_id in output.batches[0]
    assert ctx_base.request_id in output.batches[0]


def test_tg_batch_with_active_loras() -> None:
    """Test that TG batch correctly handles requests with active LoRAs."""
    lora_manager = create_mock_lora_manager(max_num_loras=2)
    pipeline = create_mock_pipeline_with_lora(lora_manager)
    paged_cache = create_mock_paged_cache()

    config = TokenGenerationSchedulerConfig(
        max_batch_size=5,
        max_forward_steps_tg=1,
        target_tokens_per_batch_ce=100,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=config,
        pipeline=pipeline,
        paged_cache=paged_cache,
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
    assert ctx_active1.request_id in output.batches[0]
    assert ctx_active2.request_id in output.batches[0]
    assert ctx_base.request_id in output.batches[0]


def test_ce_lora_activation_within_budget() -> None:
    """Test that LoRAs are activated during CE when within budget."""
    lora_manager = create_mock_lora_manager(max_num_loras=3)
    pipeline = create_mock_pipeline_with_lora(lora_manager)
    paged_cache = create_mock_paged_cache()

    config = TokenGenerationSchedulerConfig(
        max_batch_size=4,
        max_forward_steps_tg=1,
        target_tokens_per_batch_ce=100,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=config,
        pipeline=pipeline,
        paged_cache=paged_cache,
    )

    ctx_lora1 = create_lora_context(model_name="lora_model1")
    ctx_lora2 = create_lora_context(model_name="lora_model2")

    batch_constructor.enqueue_new_request(ctx_lora1)
    batch_constructor.enqueue_new_request(ctx_lora2)

    output = batch_constructor.construct_batch()

    assert len(output.batches[0]) == 2
    assert ctx_lora1.request_id in output.batches[0]
    assert ctx_lora2.request_id in output.batches[0]

    assert "lora_model1" in lora_manager._active_loras
    assert "lora_model2" in lora_manager._active_loras


def test_tg_pure_age_based_preemption() -> None:
    """Test that preemption is purely age-based for KV cache constraints."""
    lora_manager = create_mock_lora_manager(max_num_loras=3)
    pipeline = create_mock_pipeline_with_lora(lora_manager)
    paged_cache = create_mock_paged_cache()

    paged_cache.alloc = Mock(
        side_effect=[None, InsufficientBlocksError, InsufficientBlocksError]
    )

    config = TokenGenerationSchedulerConfig(
        max_batch_size=4,
        max_forward_steps_tg=1,
        target_tokens_per_batch_ce=100,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=config,
        pipeline=pipeline,
        paged_cache=paged_cache,
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
    assert ctx1.request_id in output.batches[0]
    pipeline.release.assert_called()


def test_lora_swapping_ce_to_tg() -> None:
    """Test LoRA remains active when moving from CE to TG."""
    lora_manager = create_mock_lora_manager(max_num_loras=2)
    pipeline = create_mock_pipeline_with_lora(lora_manager)
    paged_cache = create_mock_paged_cache()

    config = TokenGenerationSchedulerConfig(
        max_batch_size=4,
        max_forward_steps_tg=1,
        target_tokens_per_batch_ce=100,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=config,
        pipeline=pipeline,
        paged_cache=paged_cache,
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

    assert ctx.request_id in tg_output.batches[0]
    assert ctx2.request_id in tg_output.batches[0]


def test_mixed_requests_scheduling() -> None:
    """Test scheduling with mixed LoRA and base model requests."""
    lora_manager = create_mock_lora_manager(max_num_loras=1)
    pipeline = create_mock_pipeline_with_lora(lora_manager)
    paged_cache = create_mock_paged_cache()

    config = TokenGenerationSchedulerConfig(
        max_batch_size=4,
        max_forward_steps_tg=1,
        target_tokens_per_batch_ce=100,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=config,
        pipeline=pipeline,
        paged_cache=paged_cache,
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
    assert ctx_base1.request_id in output.batches[0]
    assert ctx_base2.request_id in output.batches[0]
    assert (ctx_lora1.request_id in output.batches[0]) or (
        ctx_lora2.request_id in output.batches[0]
    )

    assert len(lora_manager._active_loras) == 1
