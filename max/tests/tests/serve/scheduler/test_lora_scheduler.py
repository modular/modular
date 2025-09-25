# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Tests for LoRA scheduling logic in the text batch constructor."""

from __future__ import annotations

from typing import Optional
from unittest.mock import Mock

import numpy as np
from max.interfaces import (
    GenerationStatus,
    RequestID,
    TextGenerationInputs,
    TextGenerationOutput,
)
from max.nn.kv_cache import PagedKVCacheManager
from max.pipelines.core import TextContext
from max.serve.scheduler.text_batch_constructor import (
    TextBatchConstructor,
    TokenGenerationSchedulerConfig,
)


def create_mock_lora_manager(max_num_loras: int = 2) -> Mock:
    """Create a mock LoRA manager for testing."""
    manager = Mock()
    manager.max_num_loras = max_num_loras
    active_loras: set[str] = set()
    all_loras: set[str] = set()
    manager._active_loras = active_loras
    manager._all_loras = all_loras

    def is_lora(model_name: Optional[str]) -> bool:
        return bool(model_name and model_name.startswith("lora_"))

    def is_active_lora(model_name: Optional[str]) -> bool:
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
    """Create a mock paged KV cache manager."""
    cache = Mock(spec=PagedKVCacheManager)
    cache.max_seq_len = 2048
    cache.page_size = 16
    cache.total_num_pages = 128
    cache.free_blocks_pct = 0.5

    # Mock prefetch to always succeed
    cache.prefetch = Mock(return_value=True)
    cache.external_claim = Mock()
    cache.release = Mock()

    return cache


def create_mock_pipeline_with_lora(lora_manager: Mock) -> Mock:
    """Create a mock pipeline with LoRA support."""

    def next_token_behavior(
        inputs: TextGenerationInputs[TextContext],
    ) -> dict[RequestID, TextGenerationOutput]:
        responses: dict[RequestID, TextGenerationOutput] = {}

        for request_id, request in inputs.batch.items():
            # Update the InputContext
            request.update(0)

            # Return a valid response
            responses[request_id] = TextGenerationOutput(
                request_id=request_id,
                tokens=[0, 0],  # Two tokens with ID 0
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
    seq_len: int = 30, start_idx: int = 0, model_name: Optional[str] = None
) -> TextContext:
    """Create a TextContext with optional LoRA model name."""
    tokens = np.ones(seq_len, dtype=np.int32)
    context = TextContext(
        max_length=100,
        tokens=tokens,
    )
    if model_name:
        context.model_name = model_name
    context.update(start_idx)
    return context


def test_single_lora_scheduling() -> None:
    """Test scheduling a single LoRA request in CE batch."""
    lora_manager = create_mock_lora_manager(max_num_loras=2)
    pipeline = create_mock_pipeline_with_lora(lora_manager)
    paged_cache = create_mock_paged_cache()

    config = TokenGenerationSchedulerConfig(
        max_batch_size_tg=4,
        max_forward_steps_tg=1,
        max_batch_size_ce=4,
        target_tokens_per_batch_ce=100,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=config,
        pipeline=pipeline,
        paged_cache=paged_cache,
    )

    # Add a LoRA request
    ctx = create_lora_context(model_name="lora_model1")
    batch_constructor.ce_reqs[ctx.request_id] = ctx

    # Construct batch
    output = batch_constructor._try_create_ce_batch()

    # Verify the request was scheduled and LoRA was activated
    assert len(output.batch) == 1
    assert ctx.request_id in output.batch
    lora_manager.activate_adapter.assert_called_once_with("lora_model1")
    assert "lora_model1" in lora_manager._active_loras


def test_multi_lora_within_budget() -> None:
    """Test scheduling multiple LoRA requests within budget."""
    lora_manager = create_mock_lora_manager(max_num_loras=3)
    pipeline = create_mock_pipeline_with_lora(lora_manager)
    paged_cache = create_mock_paged_cache()

    config = TokenGenerationSchedulerConfig(
        max_batch_size_tg=4,
        max_forward_steps_tg=1,
        max_batch_size_ce=4,
        target_tokens_per_batch_ce=200,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=config,
        pipeline=pipeline,
        paged_cache=paged_cache,
    )

    # Add multiple LoRA requests
    ctx1 = create_lora_context(seq_len=20, model_name="lora_model1")
    ctx2 = create_lora_context(seq_len=20, model_name="lora_model2")
    ctx3 = create_lora_context(seq_len=20, model_name="lora_model3")

    batch_constructor.ce_reqs[ctx1.request_id] = ctx1
    batch_constructor.ce_reqs[ctx2.request_id] = ctx2
    batch_constructor.ce_reqs[ctx3.request_id] = ctx3

    # Construct batch
    output = batch_constructor._try_create_ce_batch()

    # All should be scheduled since we're within budget
    assert len(output.batch) == 3
    assert ctx1.request_id in output.batch
    assert ctx2.request_id in output.batch
    assert ctx3.request_id in output.batch
    assert len(lora_manager._active_loras) == 3


def test_lora_preemption_over_budget() -> None:
    """Test that LoRA requests are deferred when over budget during CE."""
    lora_manager = create_mock_lora_manager(max_num_loras=2)
    pipeline = create_mock_pipeline_with_lora(lora_manager)
    paged_cache = create_mock_paged_cache()

    config = TokenGenerationSchedulerConfig(
        max_batch_size_tg=5,
        max_forward_steps_tg=1,
        max_batch_size_ce=5,
        target_tokens_per_batch_ce=200,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=config,
        pipeline=pipeline,
        paged_cache=paged_cache,
    )

    # Don't pre-activate LoRAs - let the scheduler handle it
    # Add requests: 3 LoRAs (exceeds max_num_loras=2), 1 base
    ctx_lora1 = create_lora_context(seq_len=20, model_name="lora_model1")
    ctx_lora2 = create_lora_context(seq_len=20, model_name="lora_model2")
    ctx_lora3 = create_lora_context(seq_len=20, model_name="lora_model3")
    ctx_base = create_lora_context(seq_len=20, model_name=None)

    batch_constructor.ce_reqs[ctx_lora1.request_id] = ctx_lora1
    batch_constructor.ce_reqs[ctx_lora2.request_id] = ctx_lora2
    batch_constructor.ce_reqs[ctx_lora3.request_id] = ctx_lora3
    batch_constructor.ce_reqs[ctx_base.request_id] = ctx_base

    # Construct batch
    output = batch_constructor._try_create_ce_batch()

    # Only 2 LoRAs can be activated (max_num_loras=2), plus base
    assert len(output.batch) == 3
    assert ctx_base.request_id in output.batch  # Base always scheduled
    # First 2 LoRAs should be scheduled
    assert ctx_lora1.request_id in output.batch
    assert ctx_lora2.request_id in output.batch
    # Third LoRA should be deferred
    assert ctx_lora3.request_id not in output.batch

    # Deferred request should be back in ce_reqs
    assert ctx_lora3.request_id in batch_constructor.ce_reqs


def test_age_based_scheduling_with_lora() -> None:
    """Test that age-based scheduling is maintained with LoRA constraints."""
    lora_manager = create_mock_lora_manager(max_num_loras=2)
    pipeline = create_mock_pipeline_with_lora(lora_manager)
    paged_cache = create_mock_paged_cache()

    config = TokenGenerationSchedulerConfig(
        max_batch_size_tg=4,
        max_forward_steps_tg=1,
        max_batch_size_ce=2,  # Small batch to test ordering
        target_tokens_per_batch_ce=40,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=config,
        pipeline=pipeline,
        paged_cache=paged_cache,
    )

    # Pre-activate one LoRA
    lora_manager._active_loras.add("lora_model2")

    # Add requests in order: inactive, base, active
    ctx_inactive = create_lora_context(seq_len=20, model_name="lora_model1")
    ctx_base = create_lora_context(seq_len=20, model_name=None)
    ctx_active = create_lora_context(seq_len=20, model_name="lora_model2")

    # Add in order - age-based scheduling should respect this order
    batch_constructor.ce_reqs[ctx_inactive.request_id] = ctx_inactive
    batch_constructor.ce_reqs[ctx_base.request_id] = ctx_base
    batch_constructor.ce_reqs[ctx_active.request_id] = ctx_active

    # Construct batch
    output = batch_constructor._try_create_ce_batch()

    # Should schedule first two by age: inactive (will activate) and base
    assert len(output.batch) == 2
    assert ctx_inactive.request_id in output.batch
    assert ctx_base.request_id in output.batch
    # Active LoRA added last should not be in batch due to batch size limit


def test_tg_batch_with_active_loras() -> None:
    """Test that TG batch correctly handles requests with active LoRAs."""
    lora_manager = create_mock_lora_manager(max_num_loras=2)
    pipeline = create_mock_pipeline_with_lora(lora_manager)
    paged_cache = create_mock_paged_cache()

    config = TokenGenerationSchedulerConfig(
        max_batch_size_tg=5,
        max_forward_steps_tg=1,
        max_batch_size_ce=4,
        target_tokens_per_batch_ce=100,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=config,
        pipeline=pipeline,
        paged_cache=paged_cache,
    )

    # Pre-activate two LoRAs (simulating they were activated during CE)
    lora_manager._active_loras.add("lora_model1")
    lora_manager._active_loras.add("lora_model2")

    # Add TG requests: only with active LoRAs and base model
    ctx_active1 = create_lora_context(
        seq_len=30, start_idx=29, model_name="lora_model1"
    )
    ctx_active2 = create_lora_context(
        seq_len=30, start_idx=29, model_name="lora_model2"
    )
    ctx_base = create_lora_context(seq_len=30, start_idx=29, model_name=None)

    batch_constructor.tg_reqs[ctx_active1.request_id] = ctx_active1
    batch_constructor.tg_reqs[ctx_active2.request_id] = ctx_active2
    batch_constructor.tg_reqs[ctx_base.request_id] = ctx_base

    # Construct TG batch
    output = batch_constructor._create_tg_batch()

    # All requests should be in batch (all have active LoRAs or are base)
    assert len(output.batch) == 3
    assert ctx_active1.request_id in output.batch
    assert ctx_active2.request_id in output.batch
    assert ctx_base.request_id in output.batch


def test_ce_lora_activation_within_budget() -> None:
    """Test that LoRAs are activated during CE when within budget."""
    lora_manager = create_mock_lora_manager(max_num_loras=3)
    pipeline = create_mock_pipeline_with_lora(lora_manager)
    paged_cache = create_mock_paged_cache()

    config = TokenGenerationSchedulerConfig(
        max_batch_size_tg=4,
        max_forward_steps_tg=1,
        max_batch_size_ce=4,
        target_tokens_per_batch_ce=100,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=config,
        pipeline=pipeline,
        paged_cache=paged_cache,
    )

    # Add CE requests with different LoRAs (within budget)
    ctx_lora1 = create_lora_context(seq_len=30, model_name="lora_model1")
    ctx_lora2 = create_lora_context(seq_len=30, model_name="lora_model2")

    batch_constructor.ce_reqs[ctx_lora1.request_id] = ctx_lora1
    batch_constructor.ce_reqs[ctx_lora2.request_id] = ctx_lora2

    # Construct CE batch
    output = batch_constructor._try_create_ce_batch()

    # Both should be scheduled since we're within budget (max_num_loras=3)
    assert len(output.batch) == 2
    assert ctx_lora1.request_id in output.batch
    assert ctx_lora2.request_id in output.batch

    # Both LoRAs should have been activated
    assert "lora_model1" in lora_manager._active_loras
    assert "lora_model2" in lora_manager._active_loras


def test_tg_pure_age_based_preemption() -> None:
    """Test that preemption is purely age-based for KV cache constraints."""
    lora_manager = create_mock_lora_manager(max_num_loras=3)
    pipeline = create_mock_pipeline_with_lora(lora_manager)
    paged_cache = create_mock_paged_cache()

    # Mock prefetch to fail after first request
    call_count = [0]

    def prefetch_behavior(ctx: TextContext, num_steps: int) -> bool:
        call_count[0] += 1
        return call_count[0] <= 1

    paged_cache.maybe_reserve = Mock(side_effect=prefetch_behavior)

    config = TokenGenerationSchedulerConfig(
        max_batch_size_tg=4,
        max_forward_steps_tg=1,
        max_batch_size_ce=4,
        target_tokens_per_batch_ce=100,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=config,
        pipeline=pipeline,
        paged_cache=paged_cache,
    )

    # Pre-activate LoRAs (simulating they were activated during CE)
    lora_manager._active_loras.add("lora_model1")
    lora_manager._active_loras.add("lora_model2")

    # Add TG requests with active LoRAs
    ctx1 = create_lora_context(
        seq_len=30, start_idx=29, model_name="lora_model1"
    )
    ctx2 = create_lora_context(
        seq_len=30, start_idx=29, model_name="lora_model2"
    )
    ctx3 = create_lora_context(seq_len=30, start_idx=29, model_name=None)

    batch_constructor.tg_reqs[ctx1.request_id] = ctx1
    batch_constructor.tg_reqs[ctx2.request_id] = ctx2
    batch_constructor.tg_reqs[ctx3.request_id] = ctx3

    # Construct TG batch
    output = batch_constructor._create_tg_batch()

    # Only first request should be scheduled (due to our mock)
    assert len(output.batch) == 1

    # The newest request should have been preempted
    pipeline.release.assert_called()


def test_lora_swapping_ce_to_tg() -> None:
    """Test LoRA remains active when moving from CE to TG."""
    lora_manager = create_mock_lora_manager(max_num_loras=2)
    pipeline = create_mock_pipeline_with_lora(lora_manager)
    paged_cache = create_mock_paged_cache()

    config = TokenGenerationSchedulerConfig(
        max_batch_size_tg=4,
        max_forward_steps_tg=1,
        max_batch_size_ce=4,
        target_tokens_per_batch_ce=100,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=config,
        pipeline=pipeline,
        paged_cache=paged_cache,
    )

    # Schedule a LoRA request in CE
    ctx = create_lora_context(seq_len=30, model_name="lora_model1")
    batch_constructor.ce_reqs[ctx.request_id] = ctx

    batch_constructor._try_create_ce_batch()
    assert "lora_model1" in lora_manager._active_loras

    # Move to TG (simulate completion of CE)
    ctx.update(29)
    batch_constructor.tg_reqs[ctx.request_id] = ctx

    # Add another CE request with different LoRA (within budget)
    ctx2 = create_lora_context(seq_len=30, model_name="lora_model2")
    batch_constructor.ce_reqs[ctx2.request_id] = ctx2

    # Process CE request first to activate second LoRA
    batch_constructor._try_create_ce_batch()
    assert "lora_model2" in lora_manager._active_loras

    # Now move second request to TG
    ctx2.update(29)
    batch_constructor.tg_reqs[ctx2.request_id] = ctx2

    # Construct TG batch with both active LoRAs
    tg_output = batch_constructor._create_tg_batch()

    # Both requests should be in batch (both LoRAs are active)
    assert ctx.request_id in tg_output.batch
    assert ctx2.request_id in tg_output.batch


def test_mixed_requests_scheduling() -> None:
    """Test scheduling with mixed LoRA and base model requests."""
    lora_manager = create_mock_lora_manager(max_num_loras=1)
    pipeline = create_mock_pipeline_with_lora(lora_manager)
    paged_cache = create_mock_paged_cache()

    config = TokenGenerationSchedulerConfig(
        max_batch_size_tg=4,
        max_forward_steps_tg=1,
        max_batch_size_ce=4,
        target_tokens_per_batch_ce=100,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=config,
        pipeline=pipeline,
        paged_cache=paged_cache,
    )

    # Add mixed requests
    ctx_lora1 = create_lora_context(seq_len=20, model_name="lora_model1")
    ctx_lora2 = create_lora_context(seq_len=20, model_name="lora_model2")
    ctx_base1 = create_lora_context(seq_len=20, model_name=None)
    ctx_base2 = create_lora_context(seq_len=20, model_name=None)

    batch_constructor.ce_reqs[ctx_lora1.request_id] = ctx_lora1
    batch_constructor.ce_reqs[ctx_lora2.request_id] = ctx_lora2
    batch_constructor.ce_reqs[ctx_base1.request_id] = ctx_base1
    batch_constructor.ce_reqs[ctx_base2.request_id] = ctx_base2

    # Construct batch
    output = batch_constructor._try_create_ce_batch()

    # One LoRA, both base requests should be scheduled
    # Second LoRA should be preempted due to budget
    assert len(output.batch) == 3
    assert ctx_base1.request_id in output.batch
    assert ctx_base2.request_id in output.batch
    # One of the LoRA requests should be scheduled
    assert (ctx_lora1.request_id in output.batch) or (
        ctx_lora2.request_id in output.batch
    )

    # One LoRA should be active
    assert len(lora_manager._active_loras) == 1
