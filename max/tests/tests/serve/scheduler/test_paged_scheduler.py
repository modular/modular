# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import numpy as np
import pytest
from max.driver import CPU
from max.support.math import ceildiv
from tests.serve.scheduler.common import (
    CE,
    TG,
    BatchInfo,
    create_paged_scheduler,
    enqueue_request,
    rand,
    run_until_completion,
)


@pytest.mark.parametrize("num_reqs", [1, 2, 3])
def test_paged_scheduler_tg_request_exceed_max_seq_len(
    num_reqs: int,
) -> None:
    max_seq_len = 2048
    page_size = 128
    num_blocks = int(max_seq_len / page_size * num_reqs)
    scheduler, request_queue = create_paged_scheduler(
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
        enqueue_request(request_queue, prompt_len, max_seq_len=max_seq_len)

    expected = [
        # batch_type, batch_size, terminated, num_steps, input_tokens
        BatchInfo(CE, num_reqs, 0, 1, num_reqs * prompt_len),
        BatchInfo(TG, num_reqs, num_reqs, 8, num_reqs * 1),
        BatchInfo.empty(),
    ]
    actual = run_until_completion(scheduler)
    assert len(actual) == len(expected) and actual == expected
    del request_queue


def test_paged_scheduler_basic_chunked_prefill() -> None:
    max_seq_len = 99999  # unbounded length
    target_tokens_per_batch_ce = 1000
    max_forward_steps_tg = 10
    page_size = 128
    prompt_len = 9123
    output_tokens = 43
    num_blocks = ceildiv(prompt_len + output_tokens, page_size)
    scheduler, request_queue = create_paged_scheduler(
        max_seq_len=max_seq_len,
        num_blocks=num_blocks,
        target_tokens_per_batch_ce=target_tokens_per_batch_ce,
        max_forward_steps_tg=max_forward_steps_tg,
        page_size=page_size,
        enable_chunked_prefill=True,
    )

    enqueue_request(
        request_queue,
        prompt_len=prompt_len,
        max_seq_len=prompt_len + output_tokens,
    )

    expected = [
        # batch_type, batch_size, terminated, num_steps, input_tokens
        BatchInfo(CE, 1, 0, 1, 1000),
        BatchInfo(CE, 1, 0, 1, 1000),
        BatchInfo(CE, 1, 0, 1, 1000),
        BatchInfo(CE, 1, 0, 1, 1000),
        BatchInfo(CE, 1, 0, 1, 1000),
        BatchInfo(CE, 1, 0, 1, 1000),
        BatchInfo(CE, 1, 0, 1, 1000),
        BatchInfo(CE, 1, 0, 1, 1000),
        BatchInfo(CE, 1, 0, 1, 1000),
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


def test_paged_scheduler_basic_small_batch_size() -> None:
    prompt_len = 100
    output_tokens = 13
    max_batch_size = 13
    num_requests = 40
    scheduler, request_queue = create_paged_scheduler(
        max_batch_size=max_batch_size,
    )

    for _ in range(num_requests):
        enqueue_request(
            request_queue,
            prompt_len=prompt_len,
            max_seq_len=prompt_len + output_tokens,
        )

    expected = [
        # batch_type, batch_size, terminated, num_steps, input_tokens
        BatchInfo(CE, 13, 0, 1, 1300),
        BatchInfo(TG, 13, 0, 10, 13),
        BatchInfo(TG, 13, 13, 3, 13),
        BatchInfo(CE, 13, 0, 1, 1300),
        BatchInfo(TG, 13, 0, 10, 13),
        BatchInfo(TG, 13, 13, 3, 13),
        BatchInfo(CE, 13, 0, 1, 1300),
        BatchInfo(TG, 13, 0, 10, 13),
        BatchInfo(TG, 13, 13, 3, 13),
        BatchInfo(CE, 1, 0, 1, 100),
        BatchInfo(TG, 1, 0, 10, 1),
        BatchInfo(TG, 1, 1, 3, 1),
        BatchInfo(TG, 0, 0, 0, 0),
    ]
    actual = run_until_completion(scheduler)
    assert len(actual) == len(expected) and actual == expected


def test_paged_scheduler_basic_small_batch_size_with_chunked_prefill() -> None:
    prompt_len = 1500
    output_tokens = 13
    max_batch_size = 13
    num_requests = 40
    scheduler, request_queue = create_paged_scheduler(
        max_batch_size=max_batch_size,
        enable_chunked_prefill=True,
    )

    for _ in range(num_requests):
        enqueue_request(
            request_queue,
            prompt_len=prompt_len,
            max_seq_len=prompt_len + output_tokens,
        )

    expected = [
        # batch_type, batch_size, terminated, num_steps, input_tokens
        BatchInfo(CE, 6, 0, 1, 8192),
        BatchInfo(CE, 6, 0, 1, 8192),
        BatchInfo(CE, 3, 0, 1, 3116),
        BatchInfo(TG, 13, 0, 10, 13),
        BatchInfo(TG, 13, 13, 3, 13),
        BatchInfo(CE, 6, 0, 1, 8192),
        BatchInfo(CE, 6, 0, 1, 8192),
        BatchInfo(CE, 3, 0, 1, 3116),
        BatchInfo(TG, 13, 0, 10, 13),
        BatchInfo(TG, 13, 13, 3, 13),
        BatchInfo(CE, 6, 0, 1, 8192),
        BatchInfo(CE, 6, 0, 1, 8192),
        BatchInfo(CE, 3, 0, 1, 3116),
        BatchInfo(TG, 13, 0, 10, 13),
        BatchInfo(TG, 13, 13, 3, 13),
        BatchInfo(CE, 1, 0, 1, 1500),
        BatchInfo(TG, 1, 0, 10, 1),
        BatchInfo(TG, 1, 1, 3, 1),
        BatchInfo(TG, 0, 0, 0, 0),
    ]
    actual = run_until_completion(scheduler)
    assert len(actual) == len(expected) and actual == expected


def test_paged_scheduler_num_prompts_100_prompt_len_500_output_tokens_16() -> (
    None
):
    num_prompts = 100
    prompt_len = 500
    output_tokens = 16

    scheduler, request_queue = create_paged_scheduler(
        enable_chunked_prefill=True,
        enable_in_flight_batching=False,
    )

    for _ in range(num_prompts):
        enqueue_request(
            request_queue,
            prompt_len=prompt_len,
            max_seq_len=prompt_len + output_tokens,
        )

    # We will schedule 8192 / 500 = 16.38 CE req per batch due to target_tokens_per_batch_ce.
    # This is rounded up to 17 due to chunked prefill.
    expected = [
        # batch_type, batch_size, terminated, num_steps, input_tokens
        BatchInfo(CE, 17, 0, 1, 8192),
        BatchInfo(CE, 17, 0, 1, 8192),
        BatchInfo(CE, 18, 0, 1, 8192),
        BatchInfo(CE, 17, 0, 1, 8192),
        BatchInfo(CE, 17, 0, 1, 8192),
        BatchInfo(CE, 18, 0, 1, 8192),
        BatchInfo(CE, 2, 0, 1, 848),
        BatchInfo(TG, 100, 0, 10, 100),
        BatchInfo(TG, 100, 100, 6, 100),
        BatchInfo(TG, 0, 0, 0, 0),
    ]
    actual = run_until_completion(scheduler)
    assert len(actual) == len(expected) and actual == expected


def test_paged_scheduler_num_prompts_100_prompt_len_500_output_tokens_16_prefix_len_384() -> (
    None
):
    num_prompts = 100
    prompt_len = 500
    output_tokens = 16
    prefix_len = 384

    scheduler, request_queue = create_paged_scheduler(
        enable_chunked_prefill=True,
        enable_in_flight_batching=False,
        enable_prefix_caching=True,
    )

    # set seed for reproducibility
    np.random.seed(42)
    shared_prefix = rand(prefix_len)

    for _ in range(num_prompts):
        enqueue_request(
            request_queue,
            prompt_len=prompt_len,
            max_seq_len=prompt_len + output_tokens,
            shared_prefix=shared_prefix,
        )

    # We predict approx 384 tokens to be cache hit.
    # This means we encode 500 - 384 = 116 tokens per CE batch.
    # Hence, we will schedule approx 8192 / 116 = 70.62 CE req per batch.
    # This is rounded up to 71 due to chunked prefill.
    expected = [
        # batch_type, batch_size, terminated, num_steps, input_tokens
        BatchInfo(CE, 17, 0, 1, 8192),
        BatchInfo(CE, 71, 0, 1, 8192),
        BatchInfo(CE, 14, 0, 1, 1552),
        BatchInfo(TG, 100, 0, 10, 100),
        BatchInfo(TG, 100, 100, 6, 100),
        BatchInfo(TG, 0, 0, 0, 0),
    ]
    actual = run_until_completion(scheduler)
    assert len(actual) == len(expected) and actual == expected


def test_paged_scheduler_num_prompts_100_prompt_len_500_output_tokens_16_prefix_len_200() -> (
    None
):
    num_prompts = 100
    prompt_len = 500
    output_tokens = 16
    prefix_len = 200

    scheduler, request_queue = create_paged_scheduler(
        enable_chunked_prefill=True,
        enable_in_flight_batching=False,
        enable_prefix_caching=True,
    )

    # set seed for reproducibility
    np.random.seed(42)
    shared_prefix = rand(prefix_len)

    for _ in range(num_prompts):
        enqueue_request(
            request_queue,
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
        # batch_type, batch_size, terminated, num_steps, input_tokens
        BatchInfo(CE, 17, 0, 1, 8192),
        BatchInfo(CE, 23, 0, 1, 8192),
        BatchInfo(CE, 23, 0, 1, 8192),
        BatchInfo(CE, 23, 0, 1, 8192),
        BatchInfo(CE, 18, 0, 1, 6608),
        BatchInfo(TG, 100, 0, 10, 100),
        BatchInfo(TG, 100, 100, 6, 100),
        BatchInfo(TG, 0, 0, 0, 0),
    ]
    actual = run_until_completion(scheduler)
    assert len(actual) == len(expected) and actual == expected


def test_paged_scheduler_num_prompts_100_prompt_len_500_output_tokens_16_prefix_len_64() -> (
    None
):
    num_prompts = 100
    prompt_len = 500
    output_tokens = 16
    prefix_len = 64

    scheduler, request_queue = create_paged_scheduler(
        enable_chunked_prefill=True,
        enable_in_flight_batching=False,
        enable_prefix_caching=True,
    )

    # set seed for reproducibility
    np.random.seed(42)
    shared_prefix = rand(prefix_len)

    for _ in range(num_prompts):
        enqueue_request(
            request_queue,
            prompt_len=prompt_len,
            max_seq_len=prompt_len + output_tokens,
            shared_prefix=shared_prefix,
        )

    # We predict 64 tokens to be cache hit.
    # This means we encode 500 - 64 = 436 tokens per CE request.
    # Hence, we will schedule approx 8192 / 436 = 18.79 CE req per batch.
    # This is rounded up to 19 due to chunked prefill.
    expected = [
        # batch_type, batch_size, terminated, num_steps, input_tokens
        BatchInfo(CE, 17, 0, 1, 8192),
        BatchInfo(CE, 17, 0, 1, 8192),
        BatchInfo(CE, 18, 0, 1, 8192),
        BatchInfo(CE, 17, 0, 1, 8192),
        BatchInfo(CE, 17, 0, 1, 8192),
        BatchInfo(CE, 18, 0, 1, 8192),
        BatchInfo(CE, 2, 0, 1, 848),
        BatchInfo(TG, 100, 0, 10, 100),
        BatchInfo(TG, 100, 100, 6, 100),
        BatchInfo(TG, 0, 0, 0, 0),
    ]
    actual = run_until_completion(scheduler)
    assert len(actual) == len(expected) and actual == expected


def test_paged_scheduler__num_prompts_10_prompt_len_100_output_tokens_100_prefix_len_64_low_mem_basic() -> (
    None
):
    num_prompts = 10
    prompt_len = 100
    output_tokens = 100
    prefix_len = 64

    page_size = 10
    num_blocks = 50  # this is enough for 500 tokens

    scheduler, request_queue = create_paged_scheduler(
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
            request_queue,
            prompt_len=prompt_len,
            max_seq_len=prompt_len + output_tokens,
            shared_prefix=shared_prefix,
        )

    expected = [
        # batch_type, batch_size, terminated, num_steps, input_tokens
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


def test_num_prompts_10_prompt_len_100_output_tokens_100_prefix_len_64_low_mem_prefix_caching() -> (
    None
):
    num_prompts = 10
    prompt_len = 100
    output_tokens = 100
    prefix_len = 64

    page_size = 10
    num_blocks = 50  # this is enough for 500 tokens

    scheduler, request_queue = create_paged_scheduler(
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
            request_queue,
            prompt_len=prompt_len,
            max_seq_len=prompt_len + output_tokens,
            shared_prefix=shared_prefix,
        )

    expected = [
        # batch_type, batch_size, terminated, num_steps, input_tokens
        #
        # Can only schedule 5 of 10 reqs bc of 500 token limit due to limited blocks
        BatchInfo(CE, 5, 0, 1, 500),
        # Due to shared prefix, we can use same first 6 blocks for all 10 reqs!
        # This means we use 6 blocks + 4 * n_req == 100 blocks.
        # This means we can schedule the remaining 5 reqs :D.
        BatchInfo(CE, 5, 0, 1, 200),
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
        BatchInfo(CE, 5, 0, 1, 355),
        BatchInfo(TG, 4, 0, 10, 4),
        BatchInfo(TG, 4, 0, 10, 4),
        BatchInfo(TG, 3, 1, 10, 3),
        BatchInfo(CE, 3, 0, 1, 115),
        BatchInfo(TG, 4, 0, 10, 4),
        BatchInfo(TG, 3, 0, 10, 3),
        BatchInfo(TG, 3, 1, 10, 3),
        BatchInfo(CE, 3, 0, 1, 116),
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


def test_paged_scheduler_num_prompts_100_prompt_len_500_output_tokens_16_in_flight_batching() -> (
    None
):
    num_prompts = 100
    prompt_len = 500
    output_tokens = 16

    scheduler, request_queue = create_paged_scheduler(
        enable_in_flight_batching=True,
    )

    for _ in range(num_prompts):
        enqueue_request(
            request_queue,
            prompt_len=prompt_len,
            max_seq_len=prompt_len + output_tokens,
        )

    # With inflight batching, the CE batches become bigger and bigger since they
    # now include TG requests.
    expected = [
        # batch_type, batch_size, terminated, num_steps, input_tokens
        BatchInfo(CE, 17, 0, 1, 8192),
        BatchInfo(CE, 33, 0, 1, 8192),
        BatchInfo(CE, 50, 0, 1, 8192),
        BatchInfo(CE, 66, 0, 1, 8192),
        BatchInfo(CE, 82, 0, 1, 8192),
        BatchInfo(CE, 98, 0, 1, 8192),
        BatchInfo(CE, 100, 0, 1, 1188),
        BatchInfo(TG, 100, 32, 10, 100),
        BatchInfo(TG, 68, 68, 6, 68),
        BatchInfo(TG, 0, 0, 0, 0),
    ]
    actual = run_until_completion(scheduler)
    assert len(actual) == len(expected) and actual == expected


def test_paged_scheduler_tg_preemption_basic() -> None:
    num_prompts = 2
    prompt_len = 10
    output_tokens = 100
    page_size = 10
    num_blocks = 11  # enough for 110 tokens or exactly 1 request
    scheduler, request_queue = create_paged_scheduler(
        enable_chunked_prefill=False,
        enable_in_flight_batching=False,
        num_blocks=num_blocks,
        max_batch_size=999,
        page_size=page_size,
    )

    for _ in range(num_prompts):
        enqueue_request(
            request_queue,
            prompt_len=prompt_len,
            max_seq_len=prompt_len + output_tokens,
        )

    expected = [
        # batch_type, batch_size, terminated, num_steps, input_tokens
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


def test_paged_scheduler_oom_ce() -> None:
    prompt_len = 200
    output_tokens = 1
    page_size = 10
    num_blocks = 10
    scheduler, request_queue = create_paged_scheduler(
        num_blocks=num_blocks,
        page_size=page_size,
    )

    enqueue_request(
        request_queue,
        prompt_len=prompt_len,
        max_seq_len=prompt_len + output_tokens,
    )

    actual: list[BatchInfo] = []
    with pytest.raises(RuntimeError) as e:
        run_until_completion(scheduler, output_list=actual)

    # The error message should be informative:
    assert (
        "Insufficient KV pages for a single request with 200 tokens.\n"
        "The KVCache has 10 blocks with block size 10. This is only enough to support 100 tokens.\n"
        "You must restart your process and set a lower max seq len to prevent a single request from using the entire KV cache."
        in str(e.value)
    )


def test_paged_scheduler_oom_tg() -> None:
    num_prompts = 2
    # one req is 110 tokens
    prompt_len = 10
    output_tokens = 100
    # this can hold 100 tokens, but is not enough for even 1 request
    page_size = 10
    num_blocks = 10
    scheduler, request_queue = create_paged_scheduler(
        enable_chunked_prefill=False,
        enable_in_flight_batching=False,
        num_blocks=num_blocks,
        max_batch_size=999,
        page_size=page_size,
    )

    for _ in range(num_prompts):
        enqueue_request(
            request_queue,
            prompt_len=prompt_len,
            max_seq_len=prompt_len + output_tokens,
        )

    actual: list[BatchInfo] = []
    with pytest.raises(RuntimeError) as e:
        run_until_completion(scheduler, output_list=actual)

    expected = [
        # batch_type, batch_size, terminated, num_steps, input_tokens
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
        "Insufficient KV pages for a single request with 101 tokens.\n"
        "The KVCache has 10 blocks with block size 10. This is only enough to support 100 tokens.\n"
        "You must restart your process and set a lower max seq len to prevent a single request from using the entire KV cache."
        in str(e.value)
    )
    assert len(actual) == len(expected) and actual == expected


def test_paged_scheduler_paging_to_host_on_cpu_raises() -> None:
    with pytest.raises(ValueError) as e:
        create_paged_scheduler(
            enable_kvcache_swapping_to_host=True,
            enable_prefix_caching=True,
            device=CPU(),
        )
    assert (
        "Host device detected. Paging to host is not supported when executing on CPU."
        in str(e.value)
    )


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
def test_paged_scheduler_misc_sch_configs(
    num_prompts: int,
    input_tokens: int,
    output_tokens: int,
    max_forward_steps_tg: int,
    target_tokens_per_batch_ce: int,
    enable_chunked_prefill: bool,
    enable_prefix_caching: bool,
) -> None:
    max_seq_len = input_tokens + output_tokens
    page_size = 128
    num_blocks = ceildiv(max_seq_len, page_size) * max(16, num_prompts)
    max_batch_size = ceildiv(num_prompts, 3)
    scheduler, request_queue = create_paged_scheduler(
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
            request_queue,
            input_tokens,
            max_seq_len,
            shared_prefix=shared_prefix,
        )

    # make sure that we terminated within 1000 iterations
    actual = run_until_completion(scheduler, max_num_iters=1000)
    assert actual[-1] == BatchInfo.empty()
