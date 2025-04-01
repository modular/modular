# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from dataclasses import dataclass
from queue import Queue
from unittest.mock import Mock
from uuid import uuid4

import numpy as np
import pytest
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.pipelines import TokenGenerator
from max.pipelines.context import InputContext, TextContext
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


def rand(length: int) -> np.ndarray:
    return np.random.randint(0, 256, size=length)


def create_text_context(
    prompt_len: int,
    max_seq_len: int,
    shared_prefix: np.ndarray | None = None,
) -> TextContext:
    if shared_prefix is None:
        tokens = np.ones(prompt_len)
    else:
        rem_tokens = prompt_len - len(shared_prefix)
        assert rem_tokens >= 0
        tokens = np.concatenate([shared_prefix, rand(rem_tokens)])

    return TextContext(
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
    enable_kvcache_swapping_to_host: bool = False,
) -> PagedKVCacheManager:
    # Setting kv_heads, head_dim, and num_layers to 1 so it is easy to compute
    # memory usage. Now we know each block is 1 byte.
    NUM_KV_HEADS = 1
    HEAD_DIM = 1
    NUM_LAYERS = 1

    dtype = DType.float32

    cache_memory = (
        2
        * NUM_LAYERS
        * NUM_KV_HEADS
        * HEAD_DIM
        * page_size
        * num_blocks
        * dtype.size_in_bytes
    )

    # CPU swap space is 100x the device cache memory
    GiB = 1024 * 1024 * 1024
    host_kvcache_swap_space_gb = 100 * cache_memory / GiB

    kv_params = KVCacheParams(
        dtype=dtype,
        n_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        cache_strategy=KVCacheStrategy.PAGED,
        page_size=page_size,
        enable_prefix_caching=enable_prefix_caching,
        enable_kvcache_swapping_to_host=enable_kvcache_swapping_to_host,
        host_kvcache_swap_space_gb=host_kvcache_swap_space_gb,
    )

    session = InferenceSession()

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
    num_blocks=9999,
    max_batch_size=512,
    page_size=128,
    max_forward_steps_tg=10,
    target_tokens_per_batch_tg=None,
    target_tokens_per_batch_ce=8192,
    batch_timeout=None,
    enable_prefix_caching=False,
    enable_in_flight_batching=False,
    enable_chunked_prefill=True,
    enable_kvcache_swapping_to_host=False,
) -> TokenGenerationScheduler:
    # Create a paged manager that has one slot
    paged_manager = create_paged_manager(
        num_blocks=num_blocks,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        page_size=page_size,
        enable_prefix_caching=enable_prefix_caching,
        enable_kvcache_swapping_to_host=enable_kvcache_swapping_to_host,
    )

    # Create a scheduler with a paged manager
    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size_tg=max_batch_size,
        max_forward_steps_tg=max_forward_steps_tg,
        target_tokens_per_batch_tg=target_tokens_per_batch_tg,
        max_batch_size_ce=max_batch_size,
        max_forward_steps_ce=1,
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
    return scheduler


class FakeTokenGeneratorPipeline(TokenGenerator):
    def __init__(self, kv_manager: PagedKVCacheManager):
        self.kv_manager = kv_manager
        self.prev_num_steps: int = 0

    def next_token(
        self, batch: dict[str, TextContext], num_steps: int = 1
    ) -> dict[str, TextGenerationResponse]:
        max_seq_len = self.kv_manager.max_seq_len
        # Truncate num steps based on the max seq len
        for context in batch.values():
            num_available_steps = context.compute_num_available_steps(
                max_seq_len
            )
            assert num_available_steps > 0
            num_steps = min(num_steps, num_available_steps)
        self.prev_num_steps = num_steps

        # Claim cache rows for context.
        for context in batch.values():
            if not self.kv_manager.contains(context.cache_seq_id):
                self.kv_manager.external_claim([context.cache_seq_id])

        ctxs: list[InputContext] = list(batch.values())

        self.kv_manager.fetch(ctxs, num_steps=num_steps)

        # Generate the responses
        responses = {}
        for req_id, context in batch.items():
            resp = TextGenerationResponse([], TextGenerationStatus.ACTIVE)
            for _ in range(num_steps):
                context.update(new_token=rand(1)[0])

                if context.current_length == context.max_length:
                    resp.update_status(TextGenerationStatus.MAXIMUM_LENGTH)

                if resp.is_done:
                    break

            for token, _ in context.outstanding_completion_tokens():
                resp.append_token(TextResponse(token))

            responses[req_id] = resp

        # Step the kv cache manager
        self.kv_manager.step(ctxs)

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

    def __repr__(self) -> str:
        return (
            f"BatchInfo("
            f"{self.batch_type.concise_name()}, "
            f"{self.batch_size}, "
            f"{self.terminated}, "
            f"{self.num_steps}, "
            f"{self.tokens_to_encode})"
        )


def create_batch_and_execute(
    scheduler: TokenGenerationScheduler,
) -> BatchInfo:
    batch_to_execute = scheduler._create_batch_to_execute()
    batch_size = batch_to_execute.batch_size
    batch_type = batch_to_execute.batch_type
    tokens_to_encode = batch_to_execute.tokens_to_encode
    num_steps = batch_to_execute.num_steps
    if batch_to_execute.batch_size == 0:
        return BatchInfo.empty()

    scheduler._schedule(batch_to_execute)
    terminated_reqs = batch_to_execute.num_terminated

    assert isinstance(scheduler.pipeline, FakeTokenGeneratorPipeline)

    # Pipelines should use whatever num_steps that the scheduler computed.
    # It should not need to truncate it.
    assert scheduler.pipeline.prev_num_steps == num_steps

    return BatchInfo(
        batch_type=batch_type,
        batch_size=batch_size,
        terminated=terminated_reqs,
        num_steps=num_steps,
        tokens_to_encode=tokens_to_encode,
    )


def run_until_completion(
    scheduler: TokenGenerationScheduler,
    max_num_iters: int = 50,
    output_list: list | None = None,
) -> list[BatchInfo]:
    if output_list is None:
        batch_infos = []
    else:
        batch_infos = output_list

    for _ in range(max_num_iters):
        batch_info = create_batch_and_execute(scheduler)
        batch_infos.append(batch_info)
        if batch_info.batch_size == 0:
            break
    return batch_infos


def enqueue_request(
    scheduler: TokenGenerationScheduler,
    prompt_len: int,
    max_seq_len: int,
    shared_prefix: np.ndarray | None = None,
):
    context = create_text_context(
        prompt_len=prompt_len,
        max_seq_len=max_seq_len,
        shared_prefix=shared_prefix,
    )
    req_id = f"req{uuid4()}"
    assert context.active_length == prompt_len
    scheduler.request_q.put((req_id, context))


def enqueue_request_with_prompt(
    scheduler: TokenGenerationScheduler,
    tokens: np.ndarray,
    max_seq_len: int,
):
    context = TextContext(
        prompt=tokens.tolist(),
        max_length=max_seq_len,
        tokens=tokens,
    )
    req_id = f"req{uuid4()}"
    scheduler.request_q.put((req_id, context))


CE = BatchType.ContextEncoding
TG = BatchType.TokenGeneration


@pytest.mark.parametrize("num_reqs", [1, 2, 3])
def test_tg_request_exceed_max_seq_len(num_reqs):
    max_seq_len = 2048
    page_size = 128
    num_blocks = max_seq_len / page_size * num_reqs
    scheduler = create_paged_scheduler(
        max_seq_len=max_seq_len,
        max_batch_size=100,
        num_blocks=num_blocks,
        page_size=page_size,
    )

    # Check that we would exceed max_seq_len during TG step
    prompt_len = 2040
    num_steps = scheduler.scheduler_config.max_forward_steps_tg
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
    assert len(actual) == len(expected) and actual == expected


def test_basic_chunked_prefill():
    max_seq_len = 99999  # unbounded length
    target_tokens_per_batch_ce = 1000
    max_forward_steps_tg = 10
    page_size = 128
    prompt_len = 9123
    output_tokens = 43
    num_blocks = ceildiv(prompt_len + output_tokens, page_size)
    scheduler = create_paged_scheduler(
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
    assert len(actual) == len(expected) and actual == expected


def test_num_prompts_100_prompt_len_500_output_tokens_16():
    num_prompts = 100
    prompt_len = 500
    output_tokens = 16

    scheduler = create_paged_scheduler(
        enable_chunked_prefill=True,
        enable_in_flight_batching=False,
    )

    for _ in range(num_prompts):
        enqueue_request(
            scheduler,
            prompt_len=prompt_len,
            max_seq_len=prompt_len + output_tokens,
        )

    # We will schedule 8192 / 500 = 16.38 CE req per batch due to target_tokens_per_batch_ce.
    # This is rounded up to 17 due to chunked prefill.
    expected = [
        # batch_type, batch_size, terminated, num_steps, tokens_to_encode
        BatchInfo(CE, 17, 1, 1, 8192),
        BatchInfo(CE, 17, 1, 1, 8192),
        BatchInfo(CE, 18, 1, 1, 8192),
        BatchInfo(CE, 17, 1, 1, 8192),
        BatchInfo(CE, 17, 1, 1, 8192),
        BatchInfo(CE, 18, 1, 1, 8192),
        BatchInfo(CE, 2, 0, 1, 848),
        BatchInfo(TG, 100, 0, 10, 100),
        BatchInfo(TG, 100, 100, 6, 100),
        BatchInfo(TG, 0, 0, 0, 0),
    ]
    actual = run_until_completion(scheduler)
    assert len(actual) == len(expected) and actual == expected


def test_num_prompts_100_prompt_len_500_output_tokens_16_prefix_len_384():
    num_prompts = 100
    prompt_len = 500
    output_tokens = 16
    prefix_len = 384

    scheduler = create_paged_scheduler(
        enable_chunked_prefill=True,
        enable_in_flight_batching=False,
        enable_prefix_caching=True,
    )

    # set seed for reproducibility
    np.random.seed(42)
    shared_prefix = rand(prefix_len)

    for _ in range(num_prompts):
        enqueue_request(
            scheduler,
            prompt_len=prompt_len,
            max_seq_len=prompt_len + output_tokens,
            shared_prefix=shared_prefix,
        )

    # We predict approx 384 tokens to be cache hit.
    # This means we encode 500 - 384 = 116 tokens per CE batch.
    # Hence, we will schedule approx 8192 / 116 = 70.62 CE req per batch.
    # This is rounded up to 71 due to chunked prefill.
    expected = [
        # batch_type, batch_size, terminated, num_steps, tokens_to_encode
        BatchInfo(CE, 17, 1, 1, 8192),
        BatchInfo(CE, 71, 1, 1, 8192),
        BatchInfo(CE, 14, 0, 1, 1552),
        BatchInfo(TG, 100, 0, 10, 100),
        BatchInfo(TG, 100, 100, 6, 100),
        BatchInfo(TG, 0, 0, 0, 0),
    ]
    actual = run_until_completion(scheduler)
    assert len(actual) == len(expected) and actual == expected


def test_num_prompts_100_prompt_len_500_output_tokens_16_prefix_len_200():
    num_prompts = 100
    prompt_len = 500
    output_tokens = 16
    prefix_len = 200

    scheduler = create_paged_scheduler(
        enable_chunked_prefill=True,
        enable_in_flight_batching=False,
        enable_prefix_caching=True,
    )

    # set seed for reproducibility
    np.random.seed(42)
    shared_prefix = rand(prefix_len)

    for _ in range(num_prompts):
        enqueue_request(
            scheduler,
            prompt_len=prompt_len,
            max_seq_len=prompt_len + output_tokens,
            shared_prefix=shared_prefix,
        )

    # We predict 200 tokens to be cache hit.
    # This means we encode 500 - 200 = 300 tokens per CE request.
    # Hence, we will schedule approx 8192 / 300 = 27.31 CE req per batch.
    # This is rounded up to 28 due to chunked prefill.
    # The first batch doesn't get cache hits so it is smaller.
    expected = [
        # batch_type, batch_size, terminated, num_steps, tokens_to_encode
        BatchInfo(CE, 17, 1, 1, 8192),
        BatchInfo(CE, 28, 1, 1, 8192),
        BatchInfo(CE, 28, 1, 1, 8192),
        BatchInfo(CE, 28, 1, 1, 8192),
        BatchInfo(CE, 3, 0, 1, 616),
        BatchInfo(TG, 100, 0, 10, 100),
        BatchInfo(TG, 100, 100, 6, 100),
        BatchInfo(TG, 0, 0, 0, 0),
    ]
    actual = run_until_completion(scheduler)
    assert len(actual) == len(expected) and actual == expected


def test_num_prompts_100_prompt_len_500_output_tokens_16_prefix_len_64():
    num_prompts = 100
    prompt_len = 500
    output_tokens = 16
    prefix_len = 64

    scheduler = create_paged_scheduler(
        enable_chunked_prefill=True,
        enable_in_flight_batching=False,
        enable_prefix_caching=True,
    )

    # set seed for reproducibility
    np.random.seed(42)
    shared_prefix = rand(prefix_len)

    for _ in range(num_prompts):
        enqueue_request(
            scheduler,
            prompt_len=prompt_len,
            max_seq_len=prompt_len + output_tokens,
            shared_prefix=shared_prefix,
        )

    # We predict 64 tokens to be cache hit.
    # This means we encode 500 - 64 = 436 tokens per CE request.
    # Hence, we will schedule approx 8192 / 436 = 18.79 CE req per batch.
    # This is rounded up to 19 due to chunked prefill.
    expected = [
        # batch_type, batch_size, terminated, num_steps, tokens_to_encode
        BatchInfo(CE, 17, 1, 1, 8192),
        BatchInfo(CE, 20, 1, 1, 8192),
        BatchInfo(CE, 19, 1, 1, 8192),
        BatchInfo(CE, 20, 1, 1, 8192),
        BatchInfo(CE, 20, 1, 1, 8192),
        BatchInfo(CE, 9, 0, 1, 3716),
        BatchInfo(TG, 100, 0, 10, 100),
        BatchInfo(TG, 100, 100, 6, 100),
        BatchInfo(TG, 0, 0, 0, 0),
    ]
    actual = run_until_completion(scheduler)
    assert len(actual) == len(expected) and actual == expected


def test_num_prompts_10_prompt_len_100_output_tokens_100_prefix_len_64_low_mem_basic():
    num_prompts = 10
    prompt_len = 100
    output_tokens = 100
    prefix_len = 64

    page_size = 10
    num_blocks = 50  # this is enough for 500 tokens

    scheduler = create_paged_scheduler(
        max_seq_len=num_blocks * page_size,
        page_size=page_size,
        num_blocks=num_blocks,
        enable_chunked_prefill=False,
        enable_in_flight_batching=False,
        enable_prefix_caching=False,
    )

    # set seed for reproducibility
    np.random.seed(42)
    shared_prefix = rand(prefix_len)

    for _ in range(num_prompts):
        enqueue_request(
            scheduler,
            prompt_len=prompt_len,
            max_seq_len=prompt_len + output_tokens,
            shared_prefix=shared_prefix,
        )

    expected = [
        # batch_type, batch_size, terminated, num_steps, tokens_to_encode
        #
        # Can only schedule 5 of 10 reqs bc of 500 token limit due to limited blocks.
        BatchInfo(CE, 5, 0, 1, 500),
        # To schedule a tg iteration, we need to preempt a request (bs 5->4)
        BatchInfo(TG, 4, 0, 10, 4),
        BatchInfo(TG, 4, 0, 10, 4),
        BatchInfo(TG, 3, 0, 10, 3),
        BatchInfo(TG, 3, 0, 10, 3),
        BatchInfo(TG, 3, 0, 10, 3),
        BatchInfo(TG, 3, 0, 10, 3),
        BatchInfo(TG, 2, 0, 10, 2),
        BatchInfo(TG, 2, 0, 10, 2),
        BatchInfo(TG, 2, 0, 10, 2),
        BatchInfo(TG, 2, 2, 10, 2),
        # This encodes more than 3*100 tokens since we are re-encoding some previously
        # preempted requests that have already generated some tokens.
        BatchInfo(CE, 3, 0, 1, 383),
        BatchInfo(TG, 3, 0, 10, 3),
        # ...
    ]
    actual = run_until_completion(scheduler, max_num_iters=len(expected))
    assert len(actual) == len(expected) and actual == expected


def test_num_prompts_10_prompt_len_100_output_tokens_100_prefix_len_64_low_mem_prefix_caching():
    num_prompts = 10
    prompt_len = 100
    output_tokens = 100
    prefix_len = 64

    page_size = 10
    num_blocks = 50  # this is enough for 500 tokens

    scheduler = create_paged_scheduler(
        max_seq_len=num_blocks * page_size,
        page_size=page_size,
        num_blocks=num_blocks,
        enable_chunked_prefill=True,
        enable_in_flight_batching=False,
        enable_prefix_caching=True,
    )

    # set seed for reproducibility
    np.random.seed(42)
    shared_prefix = rand(prefix_len)

    for _ in range(num_prompts):
        enqueue_request(
            scheduler,
            prompt_len=prompt_len,
            max_seq_len=prompt_len + output_tokens,
            shared_prefix=shared_prefix,
        )

    expected = [
        # batch_type, batch_size, terminated, num_steps, tokens_to_encode
        #
        # Can only schedule 5 of 10 reqs bc of 500 token limit due to limited blocks
        BatchInfo(CE, 5, 0, 1, 500),
        # Due to shared prefix, we can use same first 6 blocks for all 10 reqs!
        # This means we use 6 blocks + 4 * n_req == 100 blocks.
        # This means we can schedule the remaining 5 reqs :D.
        BatchInfo(CE, 5, 0, 1, 180),
        # Because we are so constrained on memory, we see many preemptions :(.
        # To run TG on 8 reqs, we need 8 blocks. To free up 8 blocks, we preempt
        # 2 reqs since each req has 4 uncommitted blocks to release.
        BatchInfo(TG, 8, 0, 10, 8),
        BatchInfo(TG, 7, 0, 10, 7),
        BatchInfo(TG, 6, 0, 10, 6),
        BatchInfo(TG, 5, 0, 10, 5),
        BatchInfo(TG, 4, 0, 10, 4),
        BatchInfo(TG, 4, 0, 10, 4),
        BatchInfo(TG, 4, 0, 10, 4),
        BatchInfo(TG, 3, 0, 10, 3),
        BatchInfo(TG, 3, 0, 10, 3),
        BatchInfo(TG, 3, 3, 10, 3),
        BatchInfo(CE, 5, 0, 1, 339),
        BatchInfo(TG, 4, 0, 10, 4),
        BatchInfo(TG, 4, 0, 10, 4),
        BatchInfo(TG, 3, 1, 10, 3),
        BatchInfo(CE, 3, 0, 1, 107),
        BatchInfo(TG, 4, 0, 10, 4),
        BatchInfo(TG, 3, 0, 10, 3),
        BatchInfo(TG, 3, 1, 10, 3),
        BatchInfo(CE, 3, 0, 1, 108),
        BatchInfo(TG, 4, 1, 10, 4),
        # Notice how 22 << 100 tokens of original prompt.
        # Prefix caching allows us to encode fewer number of tokens after preemption.
        BatchInfo(CE, 1, 0, 1, 22),
        BatchInfo(TG, 4, 0, 10, 4),
        BatchInfo(TG, 4, 1, 10, 4),
        BatchInfo(TG, 3, 0, 10, 3),
        BatchInfo(TG, 3, 0, 10, 3),
        BatchInfo(TG, 3, 0, 10, 3),
        BatchInfo(TG, 3, 0, 10, 3),
        BatchInfo(TG, 3, 1, 10, 3),
        BatchInfo(TG, 2, 0, 10, 2),
        BatchInfo(TG, 2, 1, 10, 2),
        BatchInfo(TG, 1, 1, 8, 1),
        BatchInfo(TG, 0, 0, 0, 0),
    ]
    actual = run_until_completion(scheduler)
    assert len(actual) == len(expected) and actual == expected


def test_num_prompts_100_prompt_len_500_output_tokens_16_in_flight_batching():
    num_prompts = 100
    prompt_len = 500
    output_tokens = 16

    scheduler = create_paged_scheduler(
        enable_chunked_prefill=True,
        enable_in_flight_batching=True,
    )

    for _ in range(num_prompts):
        enqueue_request(
            scheduler,
            prompt_len=prompt_len,
            max_seq_len=prompt_len + output_tokens,
        )

    # With inflight batching, the CE batches become bigger and bigger since they
    # now include TG requests.
    expected = [
        # batch_type, batch_size, terminated, num_steps, tokens_to_encode
        BatchInfo(CE, 17, 1, 1, 8192),
        BatchInfo(CE, 33, 1, 1, 8192),
        BatchInfo(CE, 50, 1, 1, 8192),
        BatchInfo(CE, 66, 1, 1, 8192),
        BatchInfo(CE, 82, 1, 1, 8192),
        BatchInfo(CE, 98, 1, 1, 8192),
        BatchInfo(CE, 100, 0, 1, 1188),
        BatchInfo(TG, 100, 32, 10, 100),
        BatchInfo(TG, 68, 68, 6, 68),
        BatchInfo(TG, 0, 0, 0, 0),
    ]
    actual = run_until_completion(scheduler)
    assert len(actual) == len(expected) and actual == expected


def test_tg_preemption_basic():
    num_prompts = 2
    prompt_len = 10
    output_tokens = 100
    page_size = 10
    num_blocks = 11  # enough for 110 tokens or exactly 1 request
    scheduler = create_paged_scheduler(
        enable_chunked_prefill=False,
        enable_in_flight_batching=False,
        num_blocks=num_blocks,
        max_batch_size=999,
        page_size=page_size,
    )

    for _ in range(num_prompts):
        enqueue_request(
            scheduler,
            prompt_len=prompt_len,
            max_seq_len=prompt_len + output_tokens,
        )

    expected = [
        # batch_type, batch_size, terminated, num_steps, tokens_to_encode
        BatchInfo(CE, 2, 0, 1, 20),  # Schedule req 0 and 1
        BatchInfo(TG, 2, 0, 10, 2),
        BatchInfo(TG, 2, 0, 10, 2),
        BatchInfo(TG, 2, 0, 10, 2),
        BatchInfo(TG, 2, 0, 10, 2),
        BatchInfo(TG, 1, 0, 10, 1),  # Run out of blocks so we preempt req 1
        BatchInfo(TG, 1, 0, 10, 1),
        BatchInfo(TG, 1, 0, 10, 1),
        BatchInfo(TG, 1, 0, 10, 1),
        BatchInfo(TG, 1, 0, 10, 1),
        BatchInfo(TG, 1, 1, 10, 1),  # Req 0 finishes
        # Req 1 begins again. We run CE on all orig prompt tokens and newly generated tokens.
        BatchInfo(CE, 1, 0, 1, 51),
        BatchInfo(TG, 1, 0, 10, 1),
        BatchInfo(TG, 1, 0, 10, 1),
        BatchInfo(TG, 1, 0, 10, 1),
        BatchInfo(TG, 1, 0, 10, 1),
        BatchInfo(TG, 1, 0, 10, 1),
        BatchInfo(TG, 1, 1, 9, 1),
        BatchInfo(TG, 0, 0, 0, 0),
    ]
    actual = run_until_completion(scheduler)
    assert len(actual) == len(expected) and actual == expected


def test_oom():
    num_prompts = 2
    # one req is 110 tokens
    prompt_len = 10
    output_tokens = 100
    # this can hold 100 tokens, but is not enough for even 1 request
    page_size = 10
    num_blocks = 10
    scheduler = create_paged_scheduler(
        enable_chunked_prefill=False,
        enable_in_flight_batching=False,
        num_blocks=num_blocks,
        max_batch_size=999,
        page_size=page_size,
    )

    for _ in range(num_prompts):
        enqueue_request(
            scheduler,
            prompt_len=prompt_len,
            max_seq_len=prompt_len + output_tokens,
        )

    actual: list[BatchInfo] = []
    with pytest.raises(RuntimeError) as e:
        run_until_completion(scheduler, output_list=actual)

    expected = [
        # batch_type, batch_size, terminated, num_steps, tokens_to_encode
        BatchInfo(CE, 2, 0, 1, 20),  # Schedule req 0 and 1
        BatchInfo(TG, 2, 0, 10, 2),
        BatchInfo(TG, 2, 0, 10, 2),
        BatchInfo(TG, 2, 0, 10, 2),
        BatchInfo(TG, 2, 0, 10, 2),
        BatchInfo(TG, 1, 0, 10, 1),  # Preempt req 1 (bs 2->1)
        BatchInfo(TG, 1, 0, 10, 1),
        BatchInfo(TG, 1, 0, 10, 1),
        BatchInfo(TG, 1, 0, 10, 1),
        BatchInfo(TG, 1, 0, 10, 1),
        # Can't schedule req 1 and there are no other reqs to preempt, give up!
    ]
    # The error message should be informative:
    assert (
        "Insufficient KV pages to run token generation on a single request with 101 tokens.\n"
        "The KVCache has 10 pages with page size 10. This is only enough to support 100 tokens.\n"
        "You must restart your process and set a lower max seq len to prevent a single request from using the entire KV cache."
        in str(e.value)
    )
    assert len(actual) == len(expected) and actual == expected


def test_dont_oom_during_cow():
    # this can hold 512 tokens
    page_size = 128
    num_blocks = 3
    scheduler = create_paged_scheduler(
        enable_chunked_prefill=True,
        enable_in_flight_batching=False,
        enable_prefix_caching=True,
        num_blocks=num_blocks,
        page_size=page_size,
        max_batch_size=999,
        target_tokens_per_batch_ce=200,
    )

    shared_prefix = rand(64)

    # Request A needs 3 blocks
    enqueue_request(
        scheduler,
        prompt_len=300,
        max_seq_len=300 + 16,
        shared_prefix=shared_prefix,
    )

    batch_info = create_batch_and_execute(scheduler)
    assert batch_info == BatchInfo(CE, 1, 1, 1, 200)

    # Request B needs 1 block
    enqueue_request(
        scheduler,
        prompt_len=64,
        max_seq_len=64 + 16,
        shared_prefix=shared_prefix,
    )

    # Note that request A and request B share some common prefix tokens.
    # Request B will want to COW which requires allocating a new block.
    # However since A is using all of the blocks, B will be unable to COW.
    actual = run_until_completion(scheduler)

    expected = [
        # batch_type, batch_size, terminated, num_steps, tokens_to_encode
        BatchInfo(CE, 1, 0, 1, 100),
        BatchInfo(TG, 1, 0, 10, 1),
        BatchInfo(TG, 1, 1, 6, 1),
        BatchInfo(CE, 1, 0, 1, 1),
        BatchInfo(TG, 1, 0, 10, 1),
        BatchInfo(TG, 1, 1, 6, 1),
        BatchInfo(TG, 0, 0, 0, 0),
    ]
    assert len(actual) == len(expected) and actual == expected


@pytest.mark.parametrize("enable_kvcache_swapping_to_host", [True, False])
def test_paging_to_host(enable_kvcache_swapping_to_host: bool):
    num_prompts = 3
    prompt_len = 550
    page_size = 128
    num_new_tokens = 3
    # We only have 5 gpu blocks which is only enough for 1 request.
    num_gpu_blocks = 5
    scheduler = create_paged_scheduler(
        enable_chunked_prefill=False,
        enable_in_flight_batching=False,
        enable_prefix_caching=True,
        num_blocks=num_gpu_blocks,
        page_size=page_size,
        max_batch_size=999,
        target_tokens_per_batch_ce=200,
        enable_kvcache_swapping_to_host=enable_kvcache_swapping_to_host,
        max_seq_len=prompt_len + num_new_tokens,
    )

    prompts = [rand(prompt_len) for _ in range(num_prompts)]

    # Submit reqs for the first time
    for prompt in prompts:
        enqueue_request_with_prompt(
            scheduler,
            tokens=prompt,
            max_seq_len=prompt_len + num_new_tokens,
        )

    # Submit same reqs again to try to get cache hits
    for prompt in prompts:
        enqueue_request_with_prompt(
            scheduler,
            tokens=prompt,
            max_seq_len=prompt_len + num_new_tokens,
        )

    actual = run_until_completion(scheduler)

    if enable_kvcache_swapping_to_host:
        # When paging to host is enabled, our effective cache size increases so
        # we can get cache hits on the latter CE iterations.
        expected = [
            # batch_type, batch_size, terminated, num_steps, tokens_to_encode
            BatchInfo(CE, 1, 0, 1, 550),
            BatchInfo(TG, 1, 1, 3, 1),
            # d2h copies. device blocks evicted and then offloaded to cpu!
            BatchInfo(CE, 1, 0, 1, 550),
            BatchInfo(TG, 1, 1, 3, 1),
            BatchInfo(CE, 1, 0, 1, 550),
            BatchInfo(TG, 1, 1, 3, 1),
            BatchInfo(CE, 1, 0, 1, 38),  # h2d copies. cpu cache hit!
            BatchInfo(TG, 1, 1, 3, 1),
            BatchInfo(CE, 1, 0, 1, 38),
            BatchInfo(TG, 1, 1, 3, 1),
            BatchInfo(CE, 1, 0, 1, 38),
            BatchInfo(TG, 1, 1, 3, 1),
            BatchInfo(TG, 0, 0, 0, 0),
        ]
    else:
        # When paging to host is disabled, we can't get cache hits because all
        # of the GPU blocks are evicted and discarded.
        expected = [
            # batch_type, batch_size, terminated, num_steps, tokens_to_encode
            BatchInfo(CE, 1, 0, 1, 550),
            BatchInfo(TG, 1, 1, 3, 1),
            # device blocks evicted but not offloaded :(
            BatchInfo(CE, 1, 0, 1, 550),
            BatchInfo(TG, 1, 1, 3, 1),
            BatchInfo(CE, 1, 0, 1, 550),
            BatchInfo(TG, 1, 1, 3, 1),
            BatchInfo(CE, 1, 0, 1, 550),  # no cache hits :(
            BatchInfo(TG, 1, 1, 3, 1),
            BatchInfo(CE, 1, 0, 1, 550),
            BatchInfo(TG, 1, 1, 3, 1),
            BatchInfo(CE, 1, 0, 1, 550),
            BatchInfo(TG, 1, 1, 3, 1),
            BatchInfo(TG, 0, 0, 0, 0),
        ]

    assert len(actual) == len(expected) and actual == expected


@pytest.mark.parametrize(
    "num_prompts, input_tokens, output_tokens, max_forward_steps_tg, target_tokens_per_batch_ce, enable_chunked_prefill, enable_prefix_caching",
    [
        (1, 1, 1, 10, 1, True, True),
        (1, 60, 95, 100, 30, True, False),
        (2, 511, 1, 10, 500, False, True),
        (2, 512, 1, 10, 1000, False, False),
        (30, 256, 16, 5, 33, False, True),
        (30, 256, 16, 100, 33, True, True),
        (100, 256, 1024, 1000, 8192, True, False),
    ],
)
def test_misc_sch_configs(
    num_prompts,
    input_tokens,
    output_tokens,
    max_forward_steps_tg,
    target_tokens_per_batch_ce,
    enable_chunked_prefill,
    enable_prefix_caching,
):
    max_seq_len = input_tokens + output_tokens
    page_size = 128
    num_blocks = ceildiv(max_seq_len, page_size) * max(16, num_prompts)
    max_batch_size = ceildiv(num_prompts, 3)
    scheduler = create_paged_scheduler(
        max_seq_len=max_seq_len,
        page_size=page_size,
        max_batch_size=max_batch_size,
        num_blocks=num_blocks,
        max_forward_steps_tg=max_forward_steps_tg,
        target_tokens_per_batch_ce=target_tokens_per_batch_ce,
        enable_chunked_prefill=enable_chunked_prefill,
        enable_prefix_caching=enable_prefix_caching,
    )

    prefix_len = ceildiv(input_tokens, 2)
    np.random.seed(42)
    shared_prefix = rand(prefix_len)

    for _ in range(num_prompts):
        enqueue_request(
            scheduler,
            input_tokens,
            max_seq_len,
            shared_prefix=shared_prefix,
        )

    # make sure that we terminated within 1000 iterations
    actual = run_until_completion(scheduler, max_num_iters=1000)
    assert actual[-1] == BatchInfo.empty()
