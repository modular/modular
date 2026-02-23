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
from collections.abc import Generator
from dataclasses import dataclass

import numpy as np
import pytest
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.interfaces import (
    AudioGenerationInputs,
    AudioGenerationOutput,
    BatchType,
    GenerationStatus,
    MAXPushQueue,
    RequestID,
    SchedulerResult,
    TokenBuffer,
)
from max.kv_cache import PagedKVCacheManager
from max.nn.legacy.kv_cache import KVCacheParams
from max.pipelines.core import TTSContext
from max.pipelines.lib.audio_generator_pipeline import (
    AudioGeneratorPipelineType,
)
from max.serve.scheduler import AudioGenerationScheduler
from max.serve.scheduler.audio_generation_scheduler import (
    AudioGenerationSchedulerConfig,
    AudioGenerationSchedulerOutput,
)


def rand(length: int) -> np.ndarray:
    return np.random.randint(0, 256, size=length)


def create_text_context(
    prompt_len: int,
    max_seq_len: int,
    shared_prefix: np.ndarray | None = None,
) -> TTSContext:
    if shared_prefix is None:
        tokens = np.ones(prompt_len, dtype=np.int64)
    else:
        rem_tokens = prompt_len - len(shared_prefix)
        assert rem_tokens >= 0
        tokens = np.concatenate([shared_prefix, rand(rem_tokens)])

    return TTSContext(
        request_id=RequestID(),
        max_length=max_seq_len,
        tokens=TokenBuffer(tokens),
        streaming=False,
    )


def create_kv_cache(
    num_blocks: int,
    max_batch_size: int,
    max_seq_len: int,
    page_size: int,
    enable_prefix_caching: bool = False,
    enable_kvcache_swapping_to_host: bool = False,
) -> PagedKVCacheManager:
    dtype = DType.float32

    kv_params = KVCacheParams(
        dtype=dtype,
        num_layers=1,
        n_kv_heads=1,
        head_dim=1,
        cache_strategy="paged",
        page_size=page_size,
        enable_prefix_caching=enable_prefix_caching,
        enable_kvcache_swapping_to_host=enable_kvcache_swapping_to_host,
        host_kvcache_swap_space_gb=999,
        devices=[DeviceRef.CPU()],
        data_parallel_degree=1,
    )

    session = InferenceSession(devices=[CPU()])

    # CPU swap space is 100x the device cache memory
    num_host_pages = num_blocks * 100 if enable_kvcache_swapping_to_host else 0
    kv_manager = PagedKVCacheManager(
        params=kv_params,
        total_num_pages=num_blocks,
        total_num_host_pages=num_host_pages,
        session=session,
        enable_runtime_checks=True,
        max_batch_size=max_batch_size,
    )

    assert kv_manager.get_num_pages(replica_idx=0) == num_blocks
    return kv_manager


def create_paged_scheduler(
    max_seq_len: int = 2048,
    num_blocks: int = 9999,
    max_batch_size: int = 512,
    page_size: int = 128,
    max_forward_steps_tg: int = 10,
    target_tokens_per_batch_ce: int = 8192,
    enable_prefix_caching: bool = False,
    enable_in_flight_batching: bool = False,
    enable_kvcache_swapping_to_host: bool = False,
    max_queue_size_tg: int | None = None,
    min_batch_size_tg: int | None = None,
    ce_delay_ms: float = 0.0,
    enable_prioritize_first_decode: bool = False,
) -> tuple[AudioGenerationScheduler, MAXPushQueue[TTSContext]]:
    max_kv_slots = max_batch_size
    if max_queue_size_tg is not None:
        max_kv_slots = max(max_kv_slots, max_queue_size_tg)
    kv_cache = create_kv_cache(
        num_blocks=num_blocks,
        max_batch_size=max_kv_slots,
        max_seq_len=max_seq_len,
        page_size=page_size,
        enable_prefix_caching=enable_prefix_caching,
        enable_kvcache_swapping_to_host=enable_kvcache_swapping_to_host,
    )

    scheduler_config = AudioGenerationSchedulerConfig(
        max_batch_size=max_batch_size,
        max_forward_steps_tg=max_forward_steps_tg,
        target_tokens_per_batch_ce=target_tokens_per_batch_ce,
        max_seq_len=max_seq_len,
        enable_in_flight_batching=enable_in_flight_batching,
        max_queue_size_tg=max_queue_size_tg,
        min_batch_size_tg=min_batch_size_tg,
        ce_delay_ms=ce_delay_ms,
        enable_prioritize_first_decode=enable_prioritize_first_decode,
    )
    token_pipeline = FakeAudioGeneratorPipeline(
        kv_cache,
        max_num_steps=max_forward_steps_tg,
        max_seq_len=max_seq_len,
    )

    request_queue: queue.Queue[TTSContext] = queue.Queue()
    response_queue: queue.Queue[
        dict[RequestID, SchedulerResult[AudioGenerationOutput]]
    ] = queue.Queue()
    cancel_queue: queue.Queue[list[RequestID]] = queue.Queue()

    scheduler = AudioGenerationScheduler(
        scheduler_config=scheduler_config,
        pipeline=token_pipeline,
        request_queue=request_queue,
        response_queue=response_queue,
        cancel_queue=cancel_queue,
        kv_cache=kv_cache,
    )

    return scheduler, request_queue


class FakeAudioGeneratorPipeline(AudioGeneratorPipelineType):
    def __init__(
        self,
        kv_cache: PagedKVCacheManager,
        max_num_steps: int,
        max_seq_len: int,
    ) -> None:
        self.kv_cache = kv_cache
        self.max_num_steps = max_num_steps
        self.max_seq_len = max_seq_len
        self._prev_num_steps: int | None = None

    def execute(
        self, inputs: AudioGenerationInputs[TTSContext]
    ) -> dict[RequestID, AudioGenerationOutput]:
        needs_ce = (
            next(iter(inputs.batch.values())).tokens.generated_length == 0
        )

        if needs_ce:
            num_tokens = 1
        else:
            num_tokens = self.max_num_steps

        # Truncate num steps based on the max seq len
        for context in inputs.batch.values():
            num_available_steps = context.compute_num_available_steps(
                self.max_seq_len
            )
            assert num_available_steps > 0
            num_tokens = min(num_tokens, num_available_steps)

        self._prev_num_steps = num_tokens

        ctxs: list[TTSContext] = list(inputs.batch.values())

        for context in ctxs:
            self.kv_cache.alloc(context, replica_idx=0, num_steps=num_tokens)
        self.kv_cache.runtime_inputs([ctxs], num_steps=num_tokens)

        # Generate the responses
        responses = {}
        for context in inputs.batch.values():
            req_id = context.request_id
            resp = AudioGenerationOutput(
                GenerationStatus.ACTIVE, steps_executed=num_tokens
            )
            for _ in range(num_tokens):
                context.update(new_token=rand(1)[0])

                if len(context.tokens) == context.max_length:
                    resp = AudioGenerationOutput(
                        GenerationStatus.MAXIMUM_LENGTH,
                        steps_executed=num_tokens,
                    )

                    # Pretend that the audio generation is done immediately when
                    # text generation is done.
                    context.audio_generation_status = (
                        GenerationStatus.MAXIMUM_LENGTH
                    )

                if resp.is_done:
                    break

            responses[req_id] = resp

        # Step the kv cache manager
        self.kv_cache.step([ctxs])

        return responses

    def release(self, request_id: RequestID) -> None:
        pass


@dataclass(eq=True)
class BatchInfo:
    batch_type: BatchType
    batch_size: int
    terminated: int
    num_steps: int
    input_tokens: int

    @classmethod
    def empty(cls) -> BatchInfo:
        return BatchInfo(BatchType.TG, 0, 0, 0, 0)

    def __repr__(self) -> str:
        return (
            f"BatchInfo("
            f"{self.batch_type.value}, "
            f"{self.batch_size}, "
            f"{self.terminated}, "
            f"{self.num_steps}, "
            f"{self.input_tokens})"
        )


def create_batch_and_execute(
    scheduler: AudioGenerationScheduler,
    batch_generator: Generator[AudioGenerationSchedulerOutput, None, None],
) -> BatchInfo:
    batch = next(batch_generator)

    batch_size = batch.batch_size
    batch_type = batch.batch_type
    input_tokens = batch.input_tokens
    if batch.batch_size == 0:
        return BatchInfo.empty()

    scheduler._schedule(batch)
    terminated_reqs = batch.num_terminated

    assert isinstance(scheduler.pipeline, FakeAudioGeneratorPipeline)

    num_steps = scheduler._prev_num_steps
    return BatchInfo(
        batch_type=batch_type,
        batch_size=batch_size,
        terminated=terminated_reqs,
        num_steps=num_steps,
        input_tokens=input_tokens,
    )


def run_until_completion(
    scheduler: AudioGenerationScheduler,
    max_num_iters: int = 50,
    output_list: list | None = None,
) -> list[BatchInfo]:
    if output_list is None:
        batch_infos = []
    else:
        batch_infos = output_list

    def create_batch_generator_non_empty(
        batch_generator: Generator[AudioGenerationSchedulerOutput, None, None],
    ) -> Generator[AudioGenerationSchedulerOutput, None, None]:
        """Generator that discards empty batches"""
        empty_count = 0
        for batch in batch_generator:
            if batch.batch_size > 0:
                empty_count = 0
                yield batch
            else:
                empty_count += 1
                # If we seen 10 empty batches in a row, we are done
                if empty_count > 10:
                    yield batch

    batch_generator = create_batch_generator_non_empty(
        scheduler._create_batch_generator()
    )
    for _ in range(max_num_iters):
        batch_info = create_batch_and_execute(scheduler, batch_generator)
        batch_infos.append(batch_info)
        if batch_info.batch_size == 0:
            break
    return batch_infos


def enqueue_request(
    queue: MAXPushQueue[TTSContext],
    prompt_len: int,
    max_seq_len: int,
    shared_prefix: np.ndarray | None = None,
) -> None:
    context = create_text_context(
        prompt_len=prompt_len,
        max_seq_len=max_seq_len,
        shared_prefix=shared_prefix,
    )
    assert context.tokens.active_length == prompt_len
    queue.put_nowait(context)


def enqueue_request_with_prompt(
    queue: MAXPushQueue[TTSContext],
    tokens: np.ndarray,
    max_seq_len: int,
) -> None:
    context = TTSContext(
        request_id=RequestID(),
        max_length=max_seq_len,
        tokens=TokenBuffer(tokens),
        streaming=False,
    )

    queue.put_nowait(context)


CE = BatchType.CE
TG = BatchType.TG


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


def test_paged_scheduler_num_prompts_100_prompt_len_500_output_tokens_16() -> (
    None
):
    num_prompts = 100
    prompt_len = 500
    output_tokens = 16

    scheduler, push_socket = create_paged_scheduler(
        enable_in_flight_batching=False,
    )

    for _ in range(num_prompts):
        enqueue_request(
            push_socket,
            prompt_len=prompt_len,
            max_seq_len=prompt_len + output_tokens,
        )

    # We will schedule 8192 / 500 = 16.38 CE req per batch due to target_tokens_per_batch_ce.
    # This is rounded up to 17 due to chunked prefill.
    expected = [
        # batch_type, batch_size, terminated, num_steps, input_tokens
        BatchInfo(CE, 17, 0, 1, 8500),
        BatchInfo(CE, 17, 0, 1, 8500),
        BatchInfo(CE, 17, 0, 1, 8500),
        BatchInfo(CE, 17, 0, 1, 8500),
        BatchInfo(CE, 17, 0, 1, 8500),
        BatchInfo(CE, 15, 0, 1, 7500),
        BatchInfo(TG, 100, 0, 10, 100),
        BatchInfo(TG, 100, 100, 10, 100),
        BatchInfo(TG, 0, 0, 0, 0),
    ]
    actual = run_until_completion(scheduler)
    assert len(actual) == len(expected) and actual == expected


@pytest.mark.parametrize("enable_prioritize_first_decode", [True, False])
def test_paged_scheduler_num_prompts_100_prompt_len_500_output_tokens_16_prefix_len_384(
    enable_prioritize_first_decode,  # noqa: ANN001
) -> None:
    num_prompts = 100
    prompt_len = 500
    output_tokens = 16
    prefix_len = 384

    scheduler, request_queue = create_paged_scheduler(
        enable_prefix_caching=True,
        enable_prioritize_first_decode=enable_prioritize_first_decode,
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

    if enable_prioritize_first_decode:
        # As you can see, the TG batch that follows each CE batch has exactly
        # the same number of requests.
        expected = [
            # batch_type, batch_size, terminated, num_steps, input_tokens
            BatchInfo(CE, 17, 0, 1, 8500),
            BatchInfo(TG, 17, 0, 10, 17),
            BatchInfo(CE, 71, 0, 1, 8236),
            BatchInfo(TG, 71, 0, 10, 71),
            BatchInfo(CE, 12, 0, 1, 1392),
            BatchInfo(TG, 12, 0, 10, 12),
            BatchInfo(TG, 100, 100, 10, 100),
            BatchInfo(TG, 0, 0, 0, 0),
        ]
    else:
        expected = [
            # batch_type, batch_size, terminated, num_steps, input_tokens
            BatchInfo(CE, 17, 0, 1, 8500),
            BatchInfo(CE, 71, 0, 1, 8236),
            BatchInfo(CE, 12, 0, 1, 1392),
            BatchInfo(TG, 100, 0, 10, 100),
            BatchInfo(TG, 100, 100, 10, 100),
            BatchInfo(TG, 0, 0, 0, 0),
        ]
    actual = run_until_completion(scheduler)
    assert len(actual) == len(expected) and actual == expected


@pytest.mark.parametrize(
    "max_queue_size_tg",
    [
        # Defaults to max_batch_size
        None,
        # Infinite queue size
        999,
    ],
)
def test_paged_scheduler_max_queue_size_tg(
    max_queue_size_tg,  # noqa: ANN001
) -> None:
    num_prompts = 100
    prompt_len = 500
    output_tokens = 16

    scheduler, request_queue = create_paged_scheduler(
        max_batch_size=32,
        max_queue_size_tg=max_queue_size_tg,
    )

    # set seed for reproducibility
    np.random.seed(42)

    for _ in range(num_prompts):
        enqueue_request(
            request_queue,
            prompt_len=prompt_len,
            max_seq_len=prompt_len + output_tokens,
        )

    if max_queue_size_tg is None:
        # Notice that max_queue_size_tg defaults to max_batch_size. This causes
        # CE batch size to be limited if it causes the resultant number of decode
        # requests to exceed 32.
        expected = [
            # batch_type, batch_size, terminated, num_steps, input_tokens
            BatchInfo(CE, 17, 0, 1, 8500),
            # CE batch size is limited to 15 here!
            BatchInfo(CE, 15, 0, 1, 7500),
            BatchInfo(TG, 32, 0, 10, 32),
            BatchInfo(TG, 32, 32, 10, 32),
            BatchInfo(CE, 17, 0, 1, 8500),
            BatchInfo(CE, 15, 0, 1, 7500),
            BatchInfo(TG, 32, 0, 10, 32),
            BatchInfo(TG, 32, 32, 10, 32),
            BatchInfo(CE, 17, 0, 1, 8500),
            BatchInfo(CE, 15, 0, 1, 7500),
            BatchInfo(TG, 32, 0, 10, 32),
            BatchInfo(TG, 32, 32, 10, 32),
            BatchInfo(CE, 4, 0, 1, 2000),
            BatchInfo(TG, 4, 0, 10, 4),
            BatchInfo(TG, 4, 4, 10, 4),
            BatchInfo(TG, 0, 0, 0, 0),
        ]
    else:
        # CE batch sizes are not limited as max_queue_size_tg is very large!
        # Notice that we don't run TG until all CE is done!
        expected = [
            # batch_type, batch_size, terminated, num_steps, input_tokens
            BatchInfo(CE, 17, 0, 1, 8500),
            BatchInfo(CE, 17, 0, 1, 8500),
            BatchInfo(CE, 17, 0, 1, 8500),
            BatchInfo(CE, 17, 0, 1, 8500),
            BatchInfo(CE, 17, 0, 1, 8500),
            BatchInfo(CE, 15, 0, 1, 7500),
            BatchInfo(TG, 32, 0, 10, 32),
            BatchInfo(TG, 32, 32, 10, 32),
            BatchInfo(TG, 32, 0, 10, 32),
            BatchInfo(TG, 32, 32, 10, 32),
            BatchInfo(TG, 32, 0, 10, 32),
            BatchInfo(TG, 32, 32, 10, 32),
            BatchInfo(TG, 4, 0, 10, 4),
            BatchInfo(TG, 4, 4, 10, 4),
            BatchInfo(TG, 0, 0, 0, 0),
        ]
    actual = run_until_completion(scheduler)
    assert len(actual) == len(expected) and actual == expected


@pytest.mark.parametrize(
    "min_batch_size_tg, max_batch_size, max_queue_size_tg",
    [
        (None, 50, None),
        (50, 50, 50),
        (25, 50, 999),
        (50, 50, 999),
        (75, 50, 999),
        (999, 50, 999),
    ],
)
def test_paged_scheduler_tg_batching(
    min_batch_size_tg,  # noqa: ANN001
    max_batch_size,  # noqa: ANN001
    max_queue_size_tg,  # noqa: ANN001
) -> None:
    num_prompts = 128
    prompt_len = 500
    output_tokens = 16

    scheduler, request_queue = create_paged_scheduler(
        min_batch_size_tg=min_batch_size_tg,
        max_batch_size=max_batch_size,
        max_queue_size_tg=max_queue_size_tg,
        target_tokens_per_batch_ce=16384,
    )

    # set seed for reproducibility
    np.random.seed(42)

    for _ in range(num_prompts):
        enqueue_request(
            request_queue,
            prompt_len=prompt_len,
            max_seq_len=prompt_len + output_tokens,
        )

    key = (min_batch_size_tg, max_batch_size, max_queue_size_tg)
    if key == (None, 50, None) or key == (50, 50, 50):
        # Run CE util we reach exactly 50 requests on decode queue
        expected = [
            # batch_type, batch_size, terminated, num_steps, input_tokens
            BatchInfo(CE, 33, 0, 1, 16500),
            BatchInfo(CE, 17, 0, 1, 8500),
            # 50/50 requests encoded! Time for TG
            BatchInfo(TG, 50, 0, 10, 50),
            BatchInfo(TG, 50, 50, 10, 50),
            BatchInfo(CE, 33, 0, 1, 16500),
            BatchInfo(CE, 17, 0, 1, 8500),
            BatchInfo(TG, 50, 0, 10, 50),
            BatchInfo(TG, 50, 50, 10, 50),
            BatchInfo(CE, 28, 0, 1, 14000),
            BatchInfo(TG, 28, 0, 10, 28),
            BatchInfo(TG, 28, 28, 10, 28),
            BatchInfo(TG, 0, 0, 0, 0),
        ]
    elif key == (25, 50, 999):
        # Run CE until we reach at least 25 requests on decode queue
        expected = [
            BatchInfo(CE, 33, 0, 1, 16500),
            # 33/25 requests! Time for TG
            BatchInfo(TG, 33, 0, 10, 33),
            BatchInfo(TG, 33, 33, 10, 33),
            BatchInfo(CE, 33, 0, 1, 16500),
            BatchInfo(TG, 33, 0, 10, 33),
            BatchInfo(TG, 33, 33, 10, 33),
            BatchInfo(CE, 33, 0, 1, 16500),
            BatchInfo(TG, 33, 0, 10, 33),
            BatchInfo(TG, 33, 33, 10, 33),
            BatchInfo(CE, 29, 0, 1, 14500),
            BatchInfo(TG, 29, 0, 10, 29),
            BatchInfo(TG, 29, 29, 10, 29),
            BatchInfo(TG, 0, 0, 0, 0),
        ]
    elif key == (50, 50, 999):
        # Run CE until we reach at least 50 requests on decode queue
        expected = [
            BatchInfo(CE, 33, 0, 1, 16500),
            BatchInfo(CE, 33, 0, 1, 16500),
            # 66/50 requests encoded! Time for TG
            BatchInfo(TG, 50, 0, 10, 50),
            BatchInfo(TG, 50, 50, 10, 50),
            BatchInfo(CE, 33, 0, 1, 16500),
            BatchInfo(CE, 29, 0, 1, 14500),
            BatchInfo(TG, 50, 0, 10, 50),
            BatchInfo(TG, 50, 50, 10, 50),
            BatchInfo(TG, 28, 0, 10, 28),
            BatchInfo(TG, 28, 28, 10, 28),
            BatchInfo(TG, 0, 0, 0, 0),
        ]
    elif key == (75, 50, 999):
        # Run CE until we reach at least 75 requests on decode queue
        expected = [
            BatchInfo(CE, 33, 0, 1, 16500),
            BatchInfo(CE, 33, 0, 1, 16500),
            BatchInfo(CE, 33, 0, 1, 16500),
            # 99/75 requests encoded! Time for TG
            BatchInfo(TG, 50, 0, 10, 50),
            BatchInfo(TG, 50, 50, 10, 50),
            BatchInfo(CE, 29, 0, 1, 14500),
            BatchInfo(TG, 50, 0, 10, 50),
            BatchInfo(TG, 50, 50, 10, 50),
            BatchInfo(TG, 28, 0, 10, 28),
            BatchInfo(TG, 28, 28, 10, 28),
            BatchInfo(TG, 0, 0, 0, 0),
        ]
    elif key == (999, 50, 999):
        # Super aggressively prioritize CE
        expected = [
            BatchInfo(CE, 33, 0, 1, 16500),
            BatchInfo(CE, 33, 0, 1, 16500),
            BatchInfo(CE, 33, 0, 1, 16500),
            BatchInfo(CE, 29, 0, 1, 14500),
            # Encoded all of the requests! Time for TG
            BatchInfo(TG, 50, 0, 10, 50),
            BatchInfo(TG, 50, 50, 10, 50),
            BatchInfo(TG, 50, 0, 10, 50),
            BatchInfo(TG, 50, 50, 10, 50),
            BatchInfo(TG, 28, 0, 10, 28),
            BatchInfo(TG, 28, 28, 10, 28),
            BatchInfo(TG, 0, 0, 0, 0),
        ]

    actual = run_until_completion(scheduler)
    assert len(actual) == len(expected) and actual == expected
