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

"""Scheduler coverage for a hybrid Jenga KV cache."""

from __future__ import annotations

import queue

import pytest
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.nn.kv_cache import MHAKVCacheParams, MultiKVCacheParams
from max.pipelines.context import TextContext
from max.pipelines.kv_cache import (
    PagedKVCacheManager,
    PagedKVCacheManagerInterface,
)
from max.pipelines.kv_cache.paged_kv_cache.jenga_cache_manager import (
    JengaKVCacheManager,
)
from max.serve.scheduler.config import TokenGenerationSchedulerConfig
from max.serve.scheduler.text_generation_scheduler import (
    TokenGenerationScheduler,
)
from tests.serve.scheduler.common import (
    CE,
    TG,
    BatchInfo,
    FakeTokenGeneratorPipeline,
    assert_batch_info_equal,
    enqueue_request,
    run_until_completion,
)


def create_scheduler(
    is_jenga: bool,
) -> tuple[TokenGenerationScheduler, queue.Queue[TextContext]]:
    session = InferenceSession(devices=[CPU()])
    page_size = 10
    max_batch_size = 512
    max_seq_len = 1000
    params = MultiKVCacheParams.from_params(
        {
            "sliding": MHAKVCacheParams(
                dtype=DType.float32,
                num_layers=1,
                n_kv_heads=1,
                head_dim=16,
                page_size=page_size,
                window_size=25,
                devices=[DeviceRef.CPU()],
            ),
            "full": MHAKVCacheParams(
                dtype=DType.float32,
                num_layers=1,
                n_kv_heads=1,
                head_dim=1,
                page_size=page_size,
                devices=[DeviceRef.CPU()],
            ),
        }
    )
    bytes_per_leaf = {leaf.bytes_per_page for leaf in params.leaves().values()}
    huge_page_bytes = max(bytes_per_leaf)
    avail_bytes = 500 * huge_page_bytes
    if is_jenga:
        kv_cache: PagedKVCacheManagerInterface = JengaKVCacheManager.create(
            params=params,
            available_bytes=avail_bytes,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )
    else:
        kv_cache = PagedKVCacheManager(
            params=params,
            total_num_pages=avail_bytes // sum(bytes_per_leaf),
            session=session,
            enable_runtime_checks=True,
            max_batch_size=max_batch_size,
        )
    request_queue: queue.Queue[TextContext] = queue.Queue()
    scheduler = TokenGenerationScheduler(
        scheduler_config=TokenGenerationSchedulerConfig(
            max_batch_size=512,
            target_tokens_per_batch_ce=8192,
            max_seq_len=max_seq_len,
            enable_chunked_prefill=True,
            enable_in_flight_batching=False,
        ),
        pipeline=FakeTokenGeneratorPipeline(kv_cache, max_seq_len),
        kv_cache=kv_cache,
        request_queue=request_queue,
        response_queue=queue.Queue(),
        cancel_queue=queue.Queue(),
    )
    return scheduler, request_queue


@pytest.mark.parametrize("is_jenga", [True, False])
def test_scheduler_is_jenga_isl_250_osl_6(is_jenga: bool) -> None:
    scheduler, request_queue = create_scheduler(is_jenga)
    num_requests = 50
    isl = 250
    osl = 6
    for _ in range(num_requests):
        enqueue_request(request_queue, prompt_len=isl, max_seq_len=isl + osl)

    # fmt: off
    jenga_expected = [
        BatchInfo(CE, batch_size=18, terminated=0, steps=1, preempted=0, input_toks=4500, cached_toks=0),
        BatchInfo(CE, batch_size=15, terminated=0, steps=1, preempted=0, input_toks=3750, cached_toks=0),
        BatchInfo(CE, batch_size=13, terminated=0, steps=1, preempted=0, input_toks=3250, cached_toks=0),
        BatchInfo(CE, batch_size=4, terminated=0, steps=1, preempted=0, input_toks=1000, cached_toks=0),
        BatchInfo(TG, batch_size=50, terminated=0, steps=1, preempted=0, input_toks=50, cached_toks=12500),
        BatchInfo(TG, batch_size=50, terminated=0, steps=1, preempted=0, input_toks=50, cached_toks=12550),
        BatchInfo(TG, batch_size=50, terminated=0, steps=1, preempted=0, input_toks=50, cached_toks=12600),
        BatchInfo(TG, batch_size=50, terminated=0, steps=1, preempted=0, input_toks=50, cached_toks=12650),
        BatchInfo(TG, batch_size=50, terminated=50, steps=1, preempted=0, input_toks=50, cached_toks=12700),
        BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
    ]
    legacy_expected = [
        BatchInfo(CE, batch_size=18, terminated=0, steps=1, preempted=0, input_toks=4500, cached_toks=0),
        BatchInfo(TG, batch_size=18, terminated=0, steps=1, preempted=0, input_toks=18, cached_toks=4500),
        BatchInfo(TG, batch_size=18, terminated=0, steps=1, preempted=0, input_toks=18, cached_toks=4518),
        BatchInfo(TG, batch_size=18, terminated=0, steps=1, preempted=0, input_toks=18, cached_toks=4536),
        BatchInfo(TG, batch_size=18, terminated=0, steps=1, preempted=0, input_toks=18, cached_toks=4554),
        BatchInfo(TG, batch_size=18, terminated=18, steps=1, preempted=0, input_toks=18, cached_toks=4572),
        BatchInfo(CE, batch_size=18, terminated=0, steps=1, preempted=0, input_toks=4500, cached_toks=0),
        BatchInfo(TG, batch_size=18, terminated=0, steps=1, preempted=0, input_toks=18, cached_toks=4500),
        BatchInfo(TG, batch_size=18, terminated=0, steps=1, preempted=0, input_toks=18, cached_toks=4518),
        BatchInfo(TG, batch_size=18, terminated=0, steps=1, preempted=0, input_toks=18, cached_toks=4536),
        BatchInfo(TG, batch_size=18, terminated=0, steps=1, preempted=0, input_toks=18, cached_toks=4554),
        BatchInfo(TG, batch_size=18, terminated=18, steps=1, preempted=0, input_toks=18, cached_toks=4572),
        BatchInfo(CE, batch_size=14, terminated=0, steps=1, preempted=0, input_toks=3500, cached_toks=0),
        BatchInfo(TG, batch_size=14, terminated=0, steps=1, preempted=0, input_toks=14, cached_toks=3500),
        BatchInfo(TG, batch_size=14, terminated=0, steps=1, preempted=0, input_toks=14, cached_toks=3514),
        BatchInfo(TG, batch_size=14, terminated=0, steps=1, preempted=0, input_toks=14, cached_toks=3528),
        BatchInfo(TG, batch_size=14, terminated=0, steps=1, preempted=0, input_toks=14, cached_toks=3542),
        BatchInfo(TG, batch_size=14, terminated=14, steps=1, preempted=0, input_toks=14, cached_toks=3556),
        BatchInfo(TG, batch_size=0, terminated=0, steps=0, preempted=0, input_toks=0, cached_toks=0),
    ]
    # fmt: on
    assert_batch_info_equal(
        run_until_completion(scheduler),
        jenga_expected if is_jenga else legacy_expected,
    )
