# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from dataclasses import dataclass
from queue import Queue
from unittest.mock import Mock

import numpy as np
import pytest
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.pipelines import TokenGenerator
from max.pipelines.context import TextContext
from max.pipelines.interfaces import (
    TextGenerationResponse,
    TextGenerationStatus,
    TextResponse,
)
from max.pipelines.kv_cache import (
    KVCacheParams,
    KVCacheStrategy,
    PagedKVCacheManager,
)
from max.serve.pipelines.model_worker import ProcessControl
from max.serve.pipelines.scheduler import (
    BatchType,
    TokenGenerationScheduler,
    TokenGenerationSchedulerConfig,
)
from max.support.math import ceildiv


def create_process_control() -> ProcessControl:
    pc = Mock()
    pc.is_canceled = Mock(return_value=False)
    pc.beat = Mock()
    return pc


def create_queues() -> dict[str, Queue]:
    return {"REQUEST": Queue(), "RESPONSE": Queue(), "CANCEL": Queue()}


def create_text_context(
    cache_seq_id: int,
    prompt_len: int,
    max_seq_len: int,
) -> TextContext:
    tokens = np.ones(prompt_len)

    return TextContext(
        cache_seq_id=cache_seq_id,
        prompt=tokens.tolist(),
        max_length=max_seq_len,
        tokens=tokens,
    )


def create_paged_manager(
    num_blocks: int,
    max_batch_size: int,
    max_seq_len: int,
    page_size: int,
    enable_prefix_caching: bool = False,
) -> PagedKVCacheManager:
    # Setting kv_heads, head_dim, and num_layers to 1 so it is easy to compute
    # memory usage. Now we know each block is 1 byte.
    NUM_KV_HEADS = 1
    HEAD_DIM = 1
    NUM_LAYERS = 1

    kv_params = KVCacheParams(
        dtype=DType.float32,
        n_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        cache_strategy=KVCacheStrategy.PAGED,
        page_size=page_size,
        enable_prefix_caching=enable_prefix_caching,
    )

    session = InferenceSession()

    cache_memory = (
        2
        * NUM_LAYERS
        * NUM_KV_HEADS
        * HEAD_DIM
        * page_size
        * num_blocks
        * kv_params.dtype.size_in_bytes
    )
    kv_manager = PagedKVCacheManager(
        params=kv_params,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        num_layers=NUM_LAYERS,
        devices=[CPU()],
        session=session,
        cache_memory=cache_memory,
        page_size=page_size,
        enable_runtime_checks=True,
    )

    assert kv_manager.total_num_pages == num_blocks
    return kv_manager


def create_paged_scheduler(
    max_seq_len=2048,
    num_blocks=16,
    max_batch_size=16,
    max_batch_size_tg=16,
    max_batch_size_ce=16,
    page_size=128,
    max_forward_steps_tg=10,
    max_forward_steps_ce=1,
    target_tokens_per_batch_tg=None,
    target_tokens_per_batch_ce=8192,
    batch_timeout=None,
    enable_prefix_caching=False,
    enable_in_flight_batching=True,
    enable_chunked_prefill=True,
) -> tuple[
    PagedKVCacheManager,
    TokenGenerationScheduler,
    TokenGenerationSchedulerConfig,
    FakeTokenGeneratorPipeline,
]:
    # Create a paged manager that has one slot
    paged_manager = create_paged_manager(
        num_blocks=num_blocks,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        page_size=page_size,
        enable_prefix_caching=enable_prefix_caching,
    )

    # Create a scheduler with a paged manager
    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size_tg=max_batch_size_tg,
        max_forward_steps_tg=max_forward_steps_tg,
        target_tokens_per_batch_tg=target_tokens_per_batch_tg,
        max_batch_size_ce=max_batch_size_ce,
        max_forward_steps_ce=max_forward_steps_ce,
        target_tokens_per_batch_ce=target_tokens_per_batch_ce,
        batch_timeout=batch_timeout,
        enable_chunked_prefill=enable_chunked_prefill,
        enable_in_flight_batching=enable_in_flight_batching,
    )
    token_pipeline = FakeTokenGeneratorPipeline(paged_manager)
    scheduler = TokenGenerationScheduler(
        process_control=create_process_control(),
        scheduler_config=scheduler_config,
        pipeline=token_pipeline,
        queues=create_queues(),  # type: ignore
        paged_manager=paged_manager,
    )
    return paged_manager, scheduler, scheduler_config, token_pipeline


def trim_prompts(
    batch: dict[str, TextContext], seq_ids_and_prompts: dict[int, np.ndarray]
) -> None:
    # This should really be done in the kv cache manager once we pass the TextContext
    # in as a argument...
    for context in batch.values():
        untrimmed_length = len(context.next_tokens)
        trimmed_length = len(seq_ids_and_prompts[context.cache_seq_id])
        bump_length = untrimmed_length - trimmed_length
        if bump_length > 0:
            context.bump_token_indices(
                start_idx=bump_length,
            )


class FakeTokenGeneratorPipeline(TokenGenerator):
    def __init__(self, kv_manager: PagedKVCacheManager):
        self.kv_manager = kv_manager
        self.prev_num_steps: int = 0

    def next_token(
        self, batch: dict[str, TextContext], num_steps=1
    ) -> dict[str, TextGenerationResponse]:
        # Truncate num steps based on the max seq len
        for context in batch.values():
            assert context.max_length is not None
            num_available_steps = context.compute_num_available_steps(
                context.max_length
            )
            assert num_available_steps > 0
            num_steps = min(num_steps, num_available_steps)
        self.prev_num_steps = num_steps

        # Claim cache rows for context.
        for context in batch.values():
            if not self.kv_manager.contains(context.cache_seq_id):
                self.kv_manager.external_claim([context.cache_seq_id])

        # Fetch and trim the prompts
        seq_ids_and_prompts = {
            context.cache_seq_id: context.next_tokens
            for context in batch.values()
        }
        self.kv_manager.fetch(seq_ids_and_prompts, num_steps=num_steps)
        trim_prompts(batch, seq_ids_and_prompts)

        # Generate the responses
        responses = {}
        for req_id, context in batch.items():
            resp = TextGenerationResponse([], TextGenerationStatus.ACTIVE)
            for _ in range(num_steps):
                context.update(new_token=1)

                if context.current_length == context.max_length:
                    resp.update_status(TextGenerationStatus.MAXIMUM_LENGTH)

                if resp.is_done:
                    break

            for token, _ in context.outstanding_completion_tokens():
                resp.append_token(TextResponse(token))

            responses[req_id] = resp

        # Step the kv cache manager
        seq_ids_and_new_tokens = {
            context.cache_seq_id: np.ones(num_steps)
            for context in batch.values()
        }
        self.kv_manager.step(seq_ids_and_new_tokens)

        return responses

    def release(self, context: TextContext):
        self.kv_manager.release(context.cache_seq_id)


@dataclass
class BatchInfo:
    batch_type: BatchType
    batch_size: int
    terminated: int
    num_steps: int
    tokens_to_encode: int

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BatchInfo):
            return False
        # All empty batches are equivalent
        if self.batch_size == 0 and other.batch_size == 0:
            return True
        return (
            self.batch_type,
            self.batch_size,
            self.terminated,
            self.num_steps,
            self.tokens_to_encode,
        ) == (
            other.batch_type,
            other.batch_size,
            other.terminated,
            other.num_steps,
            other.tokens_to_encode,
        )

    @classmethod
    def empty(cls) -> BatchInfo:
        return BatchInfo(BatchType.TokenGeneration, 0, 0, 0, 0)


def create_batch_and_execute(
    scheduler: TokenGenerationScheduler,
) -> BatchInfo:
    batch_to_execute = scheduler._create_batch_to_execute()
    batch_size = batch_to_execute.batch_size
    batch_type = batch_to_execute.batch_type
    tokens_to_encode = batch_to_execute.tokens_to_encode

    if batch_to_execute.batch_size == 0:
        return BatchInfo.empty()

    scheduler._schedule(batch_to_execute)
    terminated_reqs = batch_to_execute.num_terminated

    # get prev num steps
    assert isinstance(scheduler.pipeline, FakeTokenGeneratorPipeline)
    num_steps = scheduler.pipeline.prev_num_steps

    return BatchInfo(
        batch_type=batch_type,
        batch_size=batch_size,
        terminated=terminated_reqs,
        num_steps=num_steps,
        tokens_to_encode=tokens_to_encode,
    )


def run_until_completion(
    scheduler: TokenGenerationScheduler,
) -> list[BatchInfo]:
    # prevent infinite loop in case of bug, and excessive prints
    max_num_steps = 20
    batch_infos = []

    for _ in range(max_num_steps):
        batch_info = create_batch_and_execute(scheduler)
        batch_infos.append(batch_info)
        if batch_info.batch_size == 0:
            break
    return batch_infos


def enqueue_request(
    scheduler: TokenGenerationScheduler,
    prompt_len: int,
    max_seq_len: int,
):
    seq_id = scheduler.available_cache_indices.pop()
    context = create_text_context(
        cache_seq_id=seq_id,
        prompt_len=prompt_len,
        max_seq_len=max_seq_len,
    )
    req_id = f"req{seq_id}"
    assert context.active_length == prompt_len
    scheduler.request_q.put((req_id, context))


CE = BatchType.ContextEncoding
TG = BatchType.TokenGeneration


@pytest.mark.parametrize("num_reqs", [1, 2, 3])
def test_tg_request_exceed_max_seq_len(num_reqs):
    max_seq_len = 2048
    page_size = 128
    num_blocks = max_seq_len / page_size * num_reqs
    _, scheduler, scheduler_config, _ = create_paged_scheduler(
        max_seq_len=max_seq_len,
        max_batch_size=100,
        num_blocks=num_blocks,
        page_size=page_size,
    )

    # Check that we would exceed max_seq_len during TG step
    prompt_len = 2040
    num_steps = scheduler_config.max_forward_steps_tg
    assert num_steps == 10
    assert prompt_len + num_steps > max_seq_len

    # Check that we would run out of blocks if we try to run TG with num_steps = 10
    assert num_reqs * (prompt_len + num_steps) > num_blocks * page_size

    # Create a few requests with 2040 tokens
    for _ in range(num_reqs):
        enqueue_request(scheduler, prompt_len, max_seq_len=max_seq_len)

    expected = [
        # batch_type, batch_size, terminated, num_steps, tokens_to_encode
        BatchInfo(CE, num_reqs, 0, 1, num_reqs * prompt_len),
        BatchInfo(TG, num_reqs, num_reqs, 8, num_reqs * 1),
        BatchInfo.empty(),
    ]
    actual = run_until_completion(scheduler)
    assert actual == expected


def test_basic_chunked_prefill():
    max_seq_len = 99999  # unbounded length
    target_tokens_per_batch_ce = 1000
    max_forward_steps_tg = 10
    page_size = 128
    prompt_len = 9123
    output_tokens = 43
    num_blocks = ceildiv(prompt_len + output_tokens, page_size)
    _, scheduler, _, _ = create_paged_scheduler(
        max_seq_len=max_seq_len,
        num_blocks=num_blocks,
        target_tokens_per_batch_ce=target_tokens_per_batch_ce,
        max_forward_steps_tg=max_forward_steps_tg,
        page_size=page_size,
        enable_chunked_prefill=True,
    )

    enqueue_request(
        scheduler, prompt_len=prompt_len, max_seq_len=prompt_len + output_tokens
    )

    expected = [
        # batch_type, batch_size, terminated, num_steps, tokens_to_encode
        # chunked prefill causes some requests to appear to be "terminated"
        BatchInfo(CE, 1, 1, 1, 1000),
        BatchInfo(CE, 1, 1, 1, 1000),
        BatchInfo(CE, 1, 1, 1, 1000),
        BatchInfo(CE, 1, 1, 1, 1000),
        BatchInfo(CE, 1, 1, 1, 1000),
        BatchInfo(CE, 1, 1, 1, 1000),
        BatchInfo(CE, 1, 1, 1, 1000),
        BatchInfo(CE, 1, 1, 1, 1000),
        BatchInfo(CE, 1, 1, 1, 1000),
        BatchInfo(CE, 1, 0, 1, 123),
        BatchInfo(TG, 1, 0, 10, 1),
        BatchInfo(TG, 1, 0, 10, 1),
        BatchInfo(TG, 1, 0, 10, 1),
        BatchInfo(TG, 1, 0, 10, 1),
        BatchInfo(TG, 1, 1, 3, 1),
        BatchInfo.empty(),
    ]
    actual = run_until_completion(scheduler)
    assert actual == expected
