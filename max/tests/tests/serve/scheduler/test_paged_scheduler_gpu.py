# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import pytest
from max.driver import Accelerator
from tests.serve.scheduler.common import (
    CE,
    TG,
    BatchInfo,
    assert_batch_info_equal,
    create_paged_scheduler,
    enqueue_request_with_prompt,
    rand,
    run_until_completion,
)


@pytest.mark.parametrize("enable_kvcache_swapping_to_host", [True, False])
def test_paged_scheduler_paging_to_host(
    enable_kvcache_swapping_to_host: bool,
) -> None:
    num_prompts = 3
    prompt_len = 550
    page_size = 128
    num_new_tokens = 3
    # We only have 5 gpu blocks which is only enough for 1 request.
    num_gpu_blocks = 5
    device = Accelerator(0)
    scheduler, request_queue = create_paged_scheduler(
        enable_chunked_prefill=False,
        enable_in_flight_batching=False,
        enable_prefix_caching=True,
        num_blocks=num_gpu_blocks,
        page_size=page_size,
        max_batch_size=999,
        target_tokens_per_batch_ce=200,
        enable_kvcache_swapping_to_host=enable_kvcache_swapping_to_host,
        max_seq_len=prompt_len + num_new_tokens,
        device=device,
    )

    prompts = [rand(prompt_len) for _ in range(num_prompts)]

    # Submit reqs for the first time
    for prompt in prompts:
        enqueue_request_with_prompt(
            request_queue,
            tokens=prompt,
            max_seq_len=prompt_len + num_new_tokens,
        )

    # Submit same reqs again to try to get cache hits
    for prompt in prompts:
        enqueue_request_with_prompt(
            request_queue,
            tokens=prompt,
            max_seq_len=prompt_len + num_new_tokens,
        )

    actual = run_until_completion(scheduler)

    # fmt: off
    if enable_kvcache_swapping_to_host:
        # When paging to host is enabled, our effective cache size increases so
        # we can get cache hits on the latter CE iterations.
        expected = [
            BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=550, cached_toks=0),
            BatchInfo(TG, batch_size=1, terminated=1, steps=3, preempted=0, input_toks=1, cached_toks=550),
            # d2h copies. device blocks evicted and then offloaded to cpu!
            BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=550, cached_toks=0),
            BatchInfo(TG, batch_size=1, terminated=1, steps=3, preempted=0, input_toks=1, cached_toks=550),
            BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=550, cached_toks=0),
            BatchInfo(TG, batch_size=1, terminated=1, steps=3, preempted=0, input_toks=1, cached_toks=550),
            # h2d copies. cpu cache hit!
            BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=38, cached_toks=512),
            BatchInfo(TG, batch_size=1, terminated=1, steps=3, preempted=0, input_toks=1, cached_toks=550),
            BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=38, cached_toks=512),
            BatchInfo(TG, batch_size=1, terminated=1, steps=3, preempted=0, input_toks=1, cached_toks=550),
            BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=38, cached_toks=512),
            BatchInfo(TG, batch_size=1, terminated=1, steps=3, preempted=0, input_toks=1, cached_toks=550),
            BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0)
        ]
    else:
        # When paging to host is disabled, we can't get cache hits because all
        # of the GPU blocks are evicted and discarded.
        expected = [
            BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=550, cached_toks=0),
            BatchInfo(TG, batch_size=1, terminated=1, steps=3, preempted=0, input_toks=1, cached_toks=550),
            # device blocks evicted but not offloaded :(
            BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=550, cached_toks=0),
            BatchInfo(TG, batch_size=1, terminated=1, steps=3, preempted=0, input_toks=1, cached_toks=550),
            BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=550, cached_toks=0),
            BatchInfo(TG, batch_size=1, terminated=1, steps=3, preempted=0, input_toks=1, cached_toks=550),
            # no cache hits :(
            BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=550, cached_toks=0),
            BatchInfo(TG, batch_size=1, terminated=1, steps=3, preempted=0, input_toks=1, cached_toks=550),
            BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=550, cached_toks=0),
            BatchInfo(TG, batch_size=1, terminated=1, steps=3, preempted=0, input_toks=1, cached_toks=550),
            BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=550, cached_toks=0),
            BatchInfo(TG, batch_size=1, terminated=1, steps=3, preempted=0, input_toks=1, cached_toks=550),
            BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
        ]
    # fmt: on

    assert_batch_info_equal(actual, expected)
