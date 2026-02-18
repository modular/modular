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

import numpy as np
import pytest
from max.driver import CPU
from max.interfaces import BatchType
from max.kv_cache import InsufficientBlocksError
from max.support.math import ceildiv
from tests.serve.scheduler.common import (
    CE,
    TG,
    BatchInfo,
    assert_batch_info_equal,
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
        # For now, I am going to ignore kvcache watermark, and make its own test.
        kvcache_ce_watermark=1.0,
    )

    # Check that we would exceed max_seq_len during TG step
    prompt_len = 2040
    num_steps = scheduler.scheduler_config.max_forward_steps_tg
    assert num_steps == 10
    assert prompt_len + num_steps > max_seq_len

    # Check that we would run out of blocks if we try to run TG with num_steps = 10
    assert num_reqs * (prompt_len + num_steps) > num_blocks * page_size

    print(f"supported tokens: {num_blocks * page_size}")

    # Create a few requests with 2040 tokens
    for _ in range(num_reqs):
        enqueue_request(request_queue, prompt_len, max_seq_len=max_seq_len)

    # fmt: off
    if num_reqs == 1:
        expected = [
            BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=2040, cached_toks=0),
            BatchInfo(TG, batch_size=1, terminated=1, steps=8, preempted=0, input_toks=1, cached_toks=2040),
            BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
        ]
    elif num_reqs == 2:
        expected = [
            BatchInfo(CE, batch_size=2, terminated=0, steps=1, preempted=0, input_toks=4080, cached_toks=0),
            BatchInfo(TG, batch_size=2, terminated=2, steps=8, preempted=0, input_toks=2, cached_toks=4080),
            BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
        ]
    else:
        # We can support 6144 tokens, therefore
        expected = [
            BatchInfo(CE, batch_size=3, terminated=0, steps=1, preempted=0, input_toks=6120, cached_toks=0),
            BatchInfo(TG, batch_size=3, terminated=3, steps=8, preempted=0, input_toks=3, cached_toks=6120),
            BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
        ]
    # fmt: on

    actual = run_until_completion(scheduler)
    assert_batch_info_equal(actual, expected)


@pytest.mark.parametrize("num_reqs", [1, 2, 3])
def test_paged_scheduler_tg_request_exceed_max_seq_len_with_watermark(
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
        # For now, I am going to ignore kvcache watermark, and make its own test.
        kvcache_ce_watermark=0.95,
    )

    # Check that we would exceed max_seq_len during TG step
    prompt_len = 2040
    num_steps = scheduler.scheduler_config.max_forward_steps_tg
    assert num_steps == 10
    assert prompt_len + num_steps > max_seq_len

    # Check that we would run out of blocks if we try to run TG with num_steps = 10
    assert num_reqs * (prompt_len + num_steps) > num_blocks * page_size

    print(f"supported tokens: {num_blocks * page_size}")

    # Create a few requests with 2040 tokens
    for _ in range(num_reqs):
        enqueue_request(request_queue, prompt_len, max_seq_len=max_seq_len)

    # fmt: off
    if num_reqs == 1:
        expected = [
            BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=2040, cached_toks=0),
            BatchInfo(TG, batch_size=1, terminated=1, steps=8, preempted=0, input_toks=1, cached_toks=2040),
            BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
        ]
    elif num_reqs == 2:
        expected = [
            BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=2040, cached_toks=0),
            BatchInfo(TG, batch_size=1, terminated=1, steps=8, preempted=0, input_toks=1, cached_toks=2040),
            BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=2040, cached_toks=0),
            BatchInfo(TG, batch_size=1, terminated=1, steps=8, preempted=0, input_toks=1, cached_toks=2040),
            BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
        ]
    else:
        # We can support 6144 tokens, therefore
        expected = [
            # 4080 = 0.66%, therefore we can add a second.
            BatchInfo(CE, batch_size=2, terminated=0, steps=1, preempted=0, input_toks=4080, cached_toks=0),
            BatchInfo(TG, batch_size=2, terminated=2, steps=8, preempted=0, input_toks=2, cached_toks=4080),
            BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=2040, cached_toks=0),
            BatchInfo(TG, batch_size=1, terminated=1, steps=8, preempted=0, input_toks=1, cached_toks=2040),
            BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
        ]
    # fmt: on

    actual = run_until_completion(scheduler)
    assert_batch_info_equal(actual, expected)


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

    # fmt: off
    expected = [
        BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1000, cached_toks=0),
        BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1000, cached_toks=1000),
        BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1000, cached_toks=2000),
        BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1000, cached_toks=3000),
        BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1000, cached_toks=4000),
        BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1000, cached_toks=5000),
        BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1000, cached_toks=6000),
        BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1000, cached_toks=7000),
        BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1000, cached_toks=8000),
        BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=123, cached_toks=9000),
        BatchInfo(TG, batch_size=1, terminated=0, steps=10, preempted=0, input_toks=1, cached_toks=9123),
        BatchInfo(TG, batch_size=1, terminated=0, steps=10, preempted=0, input_toks=1, cached_toks=9133),
        BatchInfo(TG, batch_size=1, terminated=0, steps=10, preempted=0, input_toks=1, cached_toks=9143),
        BatchInfo(TG, batch_size=1, terminated=0, steps=10, preempted=0, input_toks=1, cached_toks=9153),
        BatchInfo(TG, batch_size=1, terminated=1, steps=3, preempted=0, input_toks=1, cached_toks=9163),
        BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
    ]
    # fmt: on
    actual = run_until_completion(scheduler)
    assert_batch_info_equal(actual, expected)


def test_basic_ce_scheduling() -> None:
    num_prompts = 3
    prompt_len = 10
    output_tokens = 10
    page_size = 20
    num_blocks = 3  # Budget of 60 tokens total
    scheduler, request_queue = create_paged_scheduler(
        enable_chunked_prefill=False,
        enable_in_flight_batching=False,
        num_blocks=num_blocks,
        max_batch_size=999,
        page_size=page_size,
        kvcache_ce_watermark=0.95,
    )

    for _ in range(num_prompts):
        enqueue_request(
            request_queue,
            prompt_len=prompt_len,
            max_seq_len=prompt_len + output_tokens,
        )

    # fmt: off
    expected = [
        # Schedule first two requests
        BatchInfo(CE, batch_size=2, terminated=0, steps=1, preempted=0, input_toks=20, cached_toks=0),
        # There is sufficient space in the KVCache for TG to complete
        BatchInfo(TG, batch_size=2, terminated=2, steps=10, preempted=0, input_toks=2, cached_toks=20),
        # Schedule the final CE req
        BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=10, cached_toks=0),
        BatchInfo(TG, batch_size=1, terminated=1, steps=10, preempted=0, input_toks=1, cached_toks=10),
        BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
    ]
    # fmt: on
    actual = run_until_completion(scheduler)
    assert_batch_info_equal(actual, expected)


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

    # fmt: off
    expected = [
        BatchInfo(CE, batch_size=13, terminated=0, steps=1, preempted=0, input_toks=1300, cached_toks=0),
        BatchInfo(TG, batch_size=13, terminated=0, steps=10, preempted=0, input_toks=13, cached_toks=1300),
        BatchInfo(TG, batch_size=13, terminated=13, steps=3, preempted=0, input_toks=13, cached_toks=1430),
        BatchInfo(CE, batch_size=13, terminated=0, steps=1, preempted=0, input_toks=1300, cached_toks=0),
        BatchInfo(TG, batch_size=13, terminated=0, steps=10, preempted=0, input_toks=13, cached_toks=1300),
        BatchInfo(TG, batch_size=13, terminated=13, steps=3, preempted=0, input_toks=13, cached_toks=1430),
        BatchInfo(CE, batch_size=13, terminated=0, steps=1, preempted=0, input_toks=1300, cached_toks=0),
        BatchInfo(TG, batch_size=13, terminated=0, steps=10, preempted=0, input_toks=13, cached_toks=1300),
        BatchInfo(TG, batch_size=13, terminated=13, steps=3, preempted=0, input_toks=13, cached_toks=1430),
        BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=100, cached_toks=0),
        BatchInfo(TG, batch_size=1, terminated=0, steps=10, preempted=0, input_toks=1, cached_toks=100),
        BatchInfo(TG, batch_size=1, terminated=1, steps=3, preempted=0, input_toks=1, cached_toks=110),
        BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
    ]
    # fmt: on
    actual = run_until_completion(scheduler)
    assert_batch_info_equal(actual, expected)


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

    # fmt: off
    expected = [
        BatchInfo(CE, batch_size=6, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=0),
        BatchInfo(CE, batch_size=6, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=692),
        BatchInfo(CE, batch_size=3, terminated=0, steps=1, preempted=0, input_toks=3116, cached_toks=1384),
        BatchInfo(TG, batch_size=13, terminated=0, steps=10, preempted=0, input_toks=13, cached_toks=19500),
        BatchInfo(TG, batch_size=13, terminated=13, steps=3, preempted=0, input_toks=13, cached_toks=19630),
        BatchInfo(CE, batch_size=6, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=0),
        BatchInfo(CE, batch_size=6, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=692),
        BatchInfo(CE, batch_size=3, terminated=0, steps=1, preempted=0, input_toks=3116, cached_toks=1384),
        BatchInfo(TG, batch_size=13, terminated=0, steps=10, preempted=0, input_toks=13, cached_toks=19500),
        BatchInfo(TG, batch_size=13, terminated=13, steps=3, preempted=0, input_toks=13, cached_toks=19630),
        BatchInfo(CE, batch_size=6, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=0),
        BatchInfo(CE, batch_size=6, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=692),
        BatchInfo(CE, batch_size=3, terminated=0, steps=1, preempted=0, input_toks=3116, cached_toks=1384),
        BatchInfo(TG, batch_size=13, terminated=0, steps=10, preempted=0, input_toks=13, cached_toks=19500),
        BatchInfo(TG, batch_size=13, terminated=13, steps=3, preempted=0, input_toks=13, cached_toks=19630),
        BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=1500, cached_toks=0),
        BatchInfo(TG, batch_size=1, terminated=0, steps=10, preempted=0, input_toks=1, cached_toks=1500),
        BatchInfo(TG, batch_size=1, terminated=1, steps=3, preempted=0, input_toks=1, cached_toks=1510),
        BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
    ]
    # fmt: on
    actual = run_until_completion(scheduler)
    assert_batch_info_equal(actual, expected)


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
    # fmt: off
    expected = [
        BatchInfo(CE, batch_size=17, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=0),
        BatchInfo(CE, batch_size=17, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=192),
        BatchInfo(CE, batch_size=18, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=384),
        BatchInfo(CE, batch_size=17, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=76),
        BatchInfo(CE, batch_size=17, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=268),
        BatchInfo(CE, batch_size=18, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=460),
        BatchInfo(CE, batch_size=2, terminated=0, steps=1, preempted=0, input_toks=848, cached_toks=152),
        BatchInfo(TG, batch_size=100, terminated=0, steps=10, preempted=0, input_toks=100, cached_toks=50000),
        BatchInfo(TG, batch_size=100, terminated=100, steps=6, preempted=0, input_toks=100, cached_toks=51000),
        BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
    ]
    # fmt: on
    actual = run_until_completion(scheduler)
    assert_batch_info_equal(actual, expected)


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
    # fmt: off
    expected = [
        BatchInfo(CE, batch_size=17, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=0),
        BatchInfo(CE, batch_size=71, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=27264),
        BatchInfo(CE, batch_size=14, terminated=0, steps=1, preempted=0, input_toks=1552, cached_toks=5448),
        BatchInfo(TG, batch_size=100, terminated=0, steps=10, preempted=0, input_toks=100, cached_toks=50000),
        BatchInfo(TG, batch_size=100, terminated=100, steps=6, preempted=0, input_toks=100, cached_toks=51000),
        BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
    ]
    # fmt: on
    actual = run_until_completion(scheduler)
    assert_batch_info_equal(actual, expected)


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
    # fmt: off
    expected = [
        BatchInfo(CE, batch_size=17, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=0),
        BatchInfo(CE, batch_size=23, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=3008),
        BatchInfo(CE, batch_size=23, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=3016),
        BatchInfo(CE, batch_size=23, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=3024),
        BatchInfo(CE, batch_size=18, terminated=0, steps=1, preempted=0, input_toks=6608, cached_toks=2392),
        BatchInfo(TG, batch_size=100, terminated=0, steps=10, preempted=0, input_toks=100, cached_toks=50000),
        BatchInfo(TG, batch_size=100, terminated=100, steps=6, preempted=0, input_toks=100, cached_toks=51000),
        BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
    ]
    # fmt: on
    actual = run_until_completion(scheduler)
    assert_batch_info_equal(actual, expected)


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
    # fmt: off
    expected = [
        BatchInfo(CE, batch_size=17, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=0),
        BatchInfo(CE, batch_size=17, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=192),
        BatchInfo(CE, batch_size=18, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=384),
        BatchInfo(CE, batch_size=17, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=76),
        BatchInfo(CE, batch_size=17, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=268),
        BatchInfo(CE, batch_size=18, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=460),
        BatchInfo(CE, batch_size=2, terminated=0, steps=1, preempted=0, input_toks=848, cached_toks=152),
        BatchInfo(TG, batch_size=100, terminated=0, steps=10, preempted=0, input_toks=100, cached_toks=50000),
        BatchInfo(TG, batch_size=100, terminated=100, steps=6, preempted=0, input_toks=100, cached_toks=51000),
        BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
    ]
    # fmt: on
    actual = run_until_completion(scheduler)
    assert_batch_info_equal(actual, expected)


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
        kvcache_ce_watermark=0.95,
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

    # fmt: off
    expected = [
        # Can only schedule 4 of 10 reqs due to free block pct heuristic.
        BatchInfo(CE, batch_size=4, terminated=0, steps=1, preempted=0, input_toks=400, cached_toks=0),
        BatchInfo(TG, batch_size=4, terminated=0, steps=10, preempted=0, input_toks=4, cached_toks=400),
        BatchInfo(TG, batch_size=4, terminated=0, steps=10, preempted=0, input_toks=4, cached_toks=440),
        BatchInfo(TG, batch_size=3, terminated=0, steps=10, preempted=1, input_toks=3, cached_toks=360),
        BatchInfo(TG, batch_size=3, terminated=0, steps=10, preempted=0, input_toks=3, cached_toks=390),
        BatchInfo(TG, batch_size=3, terminated=0, steps=10, preempted=0, input_toks=3, cached_toks=420),
        BatchInfo(TG, batch_size=3, terminated=0, steps=10, preempted=0, input_toks=3, cached_toks=450),
        BatchInfo(TG, batch_size=2, terminated=0, steps=10, preempted=0, input_toks=2, cached_toks=320),
        BatchInfo(TG, batch_size=2, terminated=0, steps=10, preempted=1, input_toks=2, cached_toks=340),
        BatchInfo(TG, batch_size=2, terminated=0, steps=10, preempted=0, input_toks=2, cached_toks=360),
        BatchInfo(TG, batch_size=2, terminated=2, steps=10, preempted=0, input_toks=2, cached_toks=380),
        # This encodes more than 3*100 tokens since we are re-encoding some previously
        # preempted requests that have already generated some tokens.
        BatchInfo(CE, batch_size=3, terminated=0, steps=1, preempted=0, input_toks=382, cached_toks=0),
        BatchInfo(TG, batch_size=3, terminated=0, steps=10, preempted=0, input_toks=3, cached_toks=382),
        # ...
    ]
    # fmt: on
    actual = run_until_completion(scheduler, max_num_iters=len(expected))
    assert_batch_info_equal(actual, expected)


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
        kvcache_ce_watermark=0.95,
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
    # fmt: off
    expected = [
        # Can only schedule 4 of 10 reqs bc of 500 token limit + limit on free KVCache size
	BatchInfo(CE, batch_size=4, terminated=0, steps=1, preempted=0, input_toks=400, cached_toks=0),
        # Due to shared prefix, we can use same first 6 blocks for all 10 reqs!
        # this means we use 6 blocks + 4 * n_req <= 50 blocks.
        # This means we can schedule another 4 reqs:
        #   4 reqs * 4 blocks = 160 tokens --> pct used = 0.76

        # Because we are so constrained on memory, we see many preemptions :(.
        # To run TG on 8 reqs, we need 8 blocks. To free up 8 blocks, we preempt
        # 2 reqs since each req has 4 uncommitted blocks to release.
        BatchInfo(CE, batch_size=6, terminated=0, steps=1, preempted=0, input_toks=240, cached_toks=360),
        BatchInfo(TG, batch_size=8, terminated=0, steps=10, preempted=1, input_toks=8, cached_toks=800),
        BatchInfo(TG, batch_size=7, terminated=0, steps=10, preempted=2, input_toks=7, cached_toks=770),
        BatchInfo(TG, batch_size=6, terminated=0, steps=10, preempted=1, input_toks=6, cached_toks=720),
        BatchInfo(TG, batch_size=5, terminated=0, steps=10, preempted=1, input_toks=5, cached_toks=650),
        BatchInfo(TG, batch_size=4, terminated=0, steps=10, preempted=0, input_toks=4, cached_toks=560),
        BatchInfo(TG, batch_size=4, terminated=0, steps=10, preempted=1, input_toks=4, cached_toks=600),
        BatchInfo(TG, batch_size=4, terminated=0, steps=10, preempted=0, input_toks=4, cached_toks=640),
        BatchInfo(TG, batch_size=3, terminated=0, steps=10, preempted=1, input_toks=3, cached_toks=510),
        BatchInfo(TG, batch_size=3, terminated=0, steps=10, preempted=0, input_toks=3, cached_toks=540),
        BatchInfo(TG, batch_size=3, terminated=3, steps=10, preempted=0, input_toks=3, cached_toks=570),
        BatchInfo(CE, batch_size=4, terminated=0, steps=1, preempted=0, input_toks=304, cached_toks=260),
        BatchInfo(TG, batch_size=4, terminated=0, steps=10, preempted=0, input_toks=4, cached_toks=564),
        BatchInfo(TG, batch_size=4, terminated=0, steps=10, preempted=0, input_toks=4, cached_toks=604),
        BatchInfo(TG, batch_size=3, terminated=1, steps=10, preempted=1, input_toks=3, cached_toks=503),
        BatchInfo(CE, batch_size=2, terminated=0, steps=1, preempted=0, input_toks=73, cached_toks=180),
        BatchInfo(TG, batch_size=4, terminated=0, steps=10, preempted=0, input_toks=4, cached_toks=585),
        BatchInfo(TG, batch_size=3, terminated=0, steps=10, preempted=1, input_toks=3, cached_toks=504),
        BatchInfo(TG, batch_size=3, terminated=1, steps=10, preempted=0, input_toks=3, cached_toks=534),
        BatchInfo(CE, batch_size=2, terminated=0, steps=1, preempted=0, input_toks=73, cached_toks=150),
        BatchInfo(TG, batch_size=4, terminated=1, steps=10, preempted=0, input_toks=4, cached_toks=586),
        BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=41, cached_toks=60),
        BatchInfo(TG, batch_size=4, terminated=0, steps=10, preempted=0, input_toks=4, cached_toks=526),
        BatchInfo(TG, batch_size=4, terminated=1, steps=10, preempted=0, input_toks=4, cached_toks=566),
        BatchInfo(TG, batch_size=3, terminated=0, steps=10, preempted=0, input_toks=3, cached_toks=404),
        BatchInfo(TG, batch_size=3, terminated=0, steps=10, preempted=0, input_toks=3, cached_toks=434),
        BatchInfo(TG, batch_size=3, terminated=0, steps=10, preempted=0, input_toks=3, cached_toks=464),
        BatchInfo(TG, batch_size=3, terminated=0, steps=10, preempted=0, input_toks=3, cached_toks=494),
        BatchInfo(TG, batch_size=3, terminated=1, steps=10, preempted=0, input_toks=3, cached_toks=524),
        BatchInfo(TG, batch_size=2, terminated=0, steps=10, preempted=0, input_toks=2, cached_toks=352),
        BatchInfo(TG, batch_size=2, terminated=1, steps=10, preempted=0, input_toks=2, cached_toks=372),
        BatchInfo(TG, batch_size=1, terminated=1, steps=9, preempted=0, input_toks=1, cached_toks=191),
        BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
    ]
    # fmt: on
    actual = run_until_completion(scheduler)
    assert_batch_info_equal(actual, expected)


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
    # fmt: off
    expected = [
        BatchInfo(CE, batch_size=17, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=0),
        BatchInfo(CE, batch_size=33, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=8192),
        BatchInfo(CE, batch_size=50, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=16384),
        BatchInfo(CE, batch_size=66, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=24576),
        BatchInfo(CE, batch_size=82, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=32768),
        BatchInfo(CE, batch_size=98, terminated=0, steps=1, preempted=0, input_toks=8192, cached_toks=40960),
        BatchInfo(CE, batch_size=100, terminated=0, steps=1, preempted=0, input_toks=1188, cached_toks=49152),
        BatchInfo(TG, batch_size=100, terminated=32, steps=10, preempted=0, input_toks=100, cached_toks=50340),
        BatchInfo(TG, batch_size=68, terminated=68, steps=6, preempted=0, input_toks=68, cached_toks=34844),
        BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0)
    ]
    # fmt: on
    actual = run_until_completion(scheduler)
    assert_batch_info_equal(actual, expected)


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
        kvcache_ce_watermark=0.95,
        max_seq_len=110,
    )

    for _ in range(num_prompts):
        enqueue_request(
            request_queue,
            prompt_len=prompt_len,
            max_seq_len=prompt_len + output_tokens,
        )

    # fmt: off
    expected = [
        # Schedule req 0 and 1 again
        BatchInfo(CE, batch_size=2, terminated=0, steps=1, preempted=0, input_toks=20, cached_toks=0),
        BatchInfo(TG, batch_size=2, terminated=0, steps=10, preempted=0, input_toks=2, cached_toks=20),
        BatchInfo(TG, batch_size=2, terminated=0, steps=10, preempted=0, input_toks=2, cached_toks=40),
        BatchInfo(TG, batch_size=2, terminated=0, steps=10, preempted=0, input_toks=2, cached_toks=60),
        BatchInfo(TG, batch_size=2, terminated=0, steps=10, preempted=0, input_toks=2, cached_toks=80),
        BatchInfo(TG, batch_size=1, terminated=0, steps=10, preempted=0, input_toks=1, cached_toks=50),
        # Run out of blocks so we preempt req 1
        BatchInfo(TG, batch_size=1, terminated=0, steps=10, preempted=1, input_toks=1, cached_toks=60),
        BatchInfo(TG, batch_size=1, terminated=0, steps=10, preempted=0, input_toks=1, cached_toks=70),
        BatchInfo(TG, batch_size=1, terminated=0, steps=10, preempted=0, input_toks=1, cached_toks=80),
        BatchInfo(TG, batch_size=1, terminated=0, steps=10, preempted=0, input_toks=1, cached_toks=90),
        # Req 0 finishes
        BatchInfo(TG, batch_size=1, terminated=1, steps=10, preempted=0, input_toks=1, cached_toks=100),
        # Req 1 begins again. We run CE on all orig prompt tokens and newly generated tokens.
        BatchInfo(CE, batch_size=1, terminated=0, steps=1, preempted=0, input_toks=51, cached_toks=0),
        BatchInfo(TG, batch_size=1, terminated=0, steps=10, preempted=0, input_toks=1, cached_toks=51),
        BatchInfo(TG, batch_size=1, terminated=0, steps=10, preempted=0, input_toks=1, cached_toks=61),
        BatchInfo(TG, batch_size=1, terminated=0, steps=10, preempted=0, input_toks=1, cached_toks=71),
        BatchInfo(TG, batch_size=1, terminated=0, steps=10, preempted=0, input_toks=1, cached_toks=81),
        BatchInfo(TG, batch_size=1, terminated=0, steps=10, preempted=0, input_toks=1, cached_toks=91),
        # Req 1 finishes
        BatchInfo(TG, batch_size=1, terminated=1, steps=9, preempted=0, input_toks=1, cached_toks=101),
        BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
    ]
    # fmt: on
    actual = run_until_completion(scheduler)
    assert_batch_info_equal(actual, expected)


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
    with pytest.raises(InsufficientBlocksError) as e:
        run_until_completion(scheduler, output_list=actual)

    # The error message should be informative:
    assert (
        "Insufficient KV pages for a single request with 200 tokens.\n"
        "The KVCache has 10 pages with page size 10. This is only enough to support 100 tokens.\n"
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
    with pytest.raises(InsufficientBlocksError) as e:
        run_until_completion(scheduler, output_list=actual)

    # fmt: off
    expected = [
        # Schedule req 0 and 1
        BatchInfo(CE, batch_size=2, terminated=0, steps=1, preempted=0, input_toks=20, cached_toks=0),
        BatchInfo(TG, batch_size=2, terminated=0, steps=10, preempted=0, input_toks=2, cached_toks=20),
        BatchInfo(TG, batch_size=2, terminated=0, steps=10, preempted=0, input_toks=2, cached_toks=40),
        BatchInfo(TG, batch_size=2, terminated=0, steps=10, preempted=0, input_toks=2, cached_toks=60),
        BatchInfo(TG, batch_size=2, terminated=0, steps=10, preempted=0, input_toks=2, cached_toks=80),
        # Preempt req 1 (bs 2->1)
        BatchInfo(TG, batch_size=1, terminated=0, steps=10, preempted=1, input_toks=1, cached_toks=50),
        BatchInfo(TG, batch_size=1, terminated=0, steps=10, preempted=0, input_toks=1, cached_toks=60),
        BatchInfo(TG, batch_size=1, terminated=0, steps=10, preempted=0, input_toks=1, cached_toks=70),
        BatchInfo(TG, batch_size=1, terminated=0, steps=10, preempted=0, input_toks=1, cached_toks=80),
        BatchInfo(TG, batch_size=1, terminated=0, steps=10, preempted=0, input_toks=1, cached_toks=90),
        # Can't schedule req 1 and there are no other reqs to preempt, give up!
    ]
    # fmt: on
    # The error message should be informative:
    assert (
        "Insufficient KV pages for a single request with 101 tokens.\n"
        "The KVCache has 10 pages with page size 10. This is only enough to support 100 tokens.\n"
        "You must restart your process and set a lower max seq len to prevent a single request from using the entire KV cache."
        in str(e.value)
    )
    assert_batch_info_equal(actual, expected)


def test_paged_scheduler_max_batch_total_tokens_ce() -> None:
    max_batch_total_tokens = 1000
    prompt_len = 60
    page_size = 128
    scheduler, request_queue = create_paged_scheduler(
        max_seq_len=max_batch_total_tokens,
        max_batch_total_tokens=max_batch_total_tokens,
        target_tokens_per_batch_ce=max_batch_total_tokens,
        enable_chunked_prefill=True,
        page_size=page_size,
    )

    for _ in range(20):
        enqueue_request(request_queue, prompt_len=prompt_len, max_seq_len=67)

    actual = run_until_completion(scheduler)
    # fmt: off
    #
    #
    expected = [
        # CE batch is limited by the page-aligned total-context budget.
        BatchInfo(CE, batch_size=7, terminated=0, steps=1, preempted=0, input_toks=420, cached_toks=0),
        BatchInfo(CE, batch_size=7, terminated=0, steps=1, preempted=0, input_toks=420, cached_toks=0),
        BatchInfo(CE, batch_size=6, terminated=0, steps=1, preempted=0, input_toks=360, cached_toks=0),
        BatchInfo(TG, batch_size=20, terminated=20, steps=7, preempted=0, input_toks=20, cached_toks=1200),
        # The entire batch is completed, leading to a remaining empty batch.
        BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
    ]
    # fmt: on
    assert_batch_info_equal(actual, expected)
    for batch in actual:
        if batch.batch_type == BatchType.CE:
            aligned_len = ceildiv(prompt_len, page_size) * page_size
            assert batch.batch_size * aligned_len <= max_batch_total_tokens


def test_paged_scheduler_max_batch_total_tokens_tg() -> None:
    max_batch_total_tokens = 1000
    prompt_len = 30
    page_size = 128
    scheduler, request_queue = create_paged_scheduler(
        max_seq_len=max_batch_total_tokens,
        max_batch_total_tokens=max_batch_total_tokens,
        target_tokens_per_batch_ce=max_batch_total_tokens,
        enable_chunked_prefill=True,
        page_size=page_size,
    )

    for _ in range(20):
        enqueue_request(request_queue, prompt_len=prompt_len, max_seq_len=67)

    actual = run_until_completion(scheduler)
    # fmt: off
    expected = [
        BatchInfo(CE, batch_size=7, terminated=0, steps=1, preempted=0, input_toks=210, cached_toks=0),
        BatchInfo(CE, batch_size=7, terminated=0, steps=1, preempted=0, input_toks=210, cached_toks=0),
        BatchInfo(CE, batch_size=6, terminated=0, steps=1, preempted=0, input_toks=180, cached_toks=0),
        BatchInfo(TG, batch_size=20, terminated=0, steps=10, preempted=0, input_toks=20, cached_toks=600),
        BatchInfo(TG, batch_size=20, terminated=0, steps=10, preempted=0, input_toks=20, cached_toks=800),
        BatchInfo(TG, batch_size=20, terminated=0, steps=10, preempted=0, input_toks=20, cached_toks=1000),
        BatchInfo(TG, batch_size=20, terminated=20, steps=7, preempted=0, input_toks=20, cached_toks=1200),
        BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
    ]
    # fmt: on
    assert_batch_info_equal(actual, expected)
    for batch in actual:
        steps = batch.steps
        batch_size = batch.batch_size
        per_req_aligned_len = ceildiv(prompt_len + steps, page_size) * page_size
        if batch.batch_type == BatchType.CE:
            assert batch_size * per_req_aligned_len <= max_batch_total_tokens


def test_paged_scheduler_dp8() -> None:
    # Each replica has a max batch size of 4
    # Across all replicas the aggregate max batch size is 8 * 4 = 32
    scheduler, request_queue = create_paged_scheduler(dp=8, max_batch_size=4)

    for _ in range(50):
        enqueue_request(request_queue, prompt_len=12, max_seq_len=24)

    actual = run_until_completion(scheduler)
    # fmt: off
    expected = [
        BatchInfo(CE, batch_size=32, terminated=0, steps=1, preempted=0, input_toks=384, cached_toks=0),
        BatchInfo(TG, batch_size=32, terminated=0, steps=10, preempted=0, input_toks=32, cached_toks=384),
        BatchInfo(TG, batch_size=32, terminated=32, steps=2, preempted=0, input_toks=32, cached_toks=704),
        BatchInfo(CE, batch_size=18, terminated=0, steps=1, preempted=0, input_toks=216, cached_toks=0),
        BatchInfo(TG, batch_size=18, terminated=0, steps=10, preempted=0, input_toks=18, cached_toks=216),
        BatchInfo(TG, batch_size=18, terminated=18, steps=2, preempted=0, input_toks=18, cached_toks=396),
        BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
    ]
    # fmt: on
    assert_batch_info_equal(actual, expected)


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
