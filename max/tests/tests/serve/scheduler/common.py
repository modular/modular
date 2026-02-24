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
from dataclasses import dataclass

import numpy as np
from max.driver import CPU, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.interfaces import (
    BatchType,
    GenerationStatus,
    MAXPushQueue,
    Pipeline,
    RequestID,
    SchedulerResult,
    TextGenerationInputs,
    TextGenerationOutput,
    TokenBuffer,
)
from max.kv_cache import PagedKVCacheManager
from max.nn.legacy.kv_cache import KVCacheParams
from max.pipelines.core import TextContext
from max.serve.scheduler.config import TokenGenerationSchedulerConfig
from max.serve.scheduler.text_generation_scheduler import (
    TokenGenerationScheduler,
)


def rand(length: int) -> np.ndarray:
    return np.random.randint(0, 256, size=length)


def create_text_context(
    prompt_len: int,
    max_seq_len: int,
    shared_prefix: np.ndarray | None = None,
) -> TextContext:
    if shared_prefix is None:
        tokens = np.ones(prompt_len, dtype=np.int64)
    else:
        rem_tokens = prompt_len - len(shared_prefix)
        assert rem_tokens >= 0
        tokens = np.concatenate([shared_prefix, rand(rem_tokens)])

    return TextContext(
        request_id=RequestID(),
        max_length=max_seq_len,
        tokens=TokenBuffer(tokens),
    )


def create_kv_cache(
    num_blocks: int,
    max_batch_size: int,
    max_seq_len: int,
    page_size: int,
    enable_prefix_caching: bool = False,
    enable_kvcache_swapping_to_host: bool = False,
    dp: int = 1,
    device: Device = CPU(),
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
        data_parallel_degree=dp,
        devices=[DeviceRef.from_device(device) for i in range(dp)],
    )

    session = InferenceSession(devices=[device])

    # CPU swap space is 100x the device cache memory
    num_blocks = num_blocks
    num_host_pages = num_blocks * 100 if enable_kvcache_swapping_to_host else 0
    kv_manager = PagedKVCacheManager(
        params=kv_params,
        total_num_pages=num_blocks,
        total_num_host_pages=num_host_pages,
        session=session,
        enable_runtime_checks=True,
        max_batch_size=max_batch_size,
    )

    assert all(
        kv_manager.get_num_pages(replica_idx=replica_idx) == num_blocks
        for replica_idx in range(dp)
    )
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
    enable_chunked_prefill: bool = True,
    enable_kvcache_swapping_to_host: bool = False,
    max_batch_total_tokens: int | None = None,
    dp: int = 1,
    device: Device = CPU(),
    kvcache_ce_watermark: float = 1.0,
) -> tuple[
    TokenGenerationScheduler,
    MAXPushQueue[TextContext],
]:
    # Create a paged manager that has one slot
    kv_cache = create_kv_cache(
        num_blocks=num_blocks,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        page_size=page_size,
        enable_prefix_caching=enable_prefix_caching,
        enable_kvcache_swapping_to_host=enable_kvcache_swapping_to_host,
        dp=dp,
        device=device,
    )

    # Create a scheduler with a paged manager
    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=max_batch_size,
        max_forward_steps_tg=max_forward_steps_tg,
        target_tokens_per_batch_ce=target_tokens_per_batch_ce,
        max_seq_len=max_seq_len,
        enable_chunked_prefill=enable_chunked_prefill,
        enable_in_flight_batching=enable_in_flight_batching,
        max_batch_total_tokens=max_batch_total_tokens,
        data_parallel_degree=dp,
        kvcache_ce_watermark=kvcache_ce_watermark,
    )
    token_pipeline = FakeTokenGeneratorPipeline(kv_cache, max_seq_len)
    request_queue: queue.Queue[TextContext] = queue.Queue()
    response_queue: queue.Queue[
        dict[RequestID, SchedulerResult[TextGenerationOutput]]
    ] = queue.Queue()
    cancel_queue: queue.Queue[list[RequestID]] = queue.Queue()
    scheduler = TokenGenerationScheduler(
        scheduler_config=scheduler_config,
        pipeline=token_pipeline,
        kv_cache=kv_cache,
        request_queue=request_queue,
        response_queue=response_queue,
        cancel_queue=cancel_queue,
    )

    return (scheduler, request_queue)


class FakeTokenGeneratorPipeline(
    Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput]
):
    def __init__(
        self,
        kv_manager: PagedKVCacheManager,
        max_seq_len: int,
        start_token_id: int = 42,
    ) -> None:
        self.kv_manager = kv_manager
        self.token_id = start_token_id
        self.max_seq_len = max_seq_len

    def execute(
        self, inputs: TextGenerationInputs[TextContext]
    ) -> dict[RequestID, TextGenerationOutput]:
        max_seq_len = self.max_seq_len
        # Truncate num steps based on the max seq len
        num_steps = inputs.num_steps
        for context in inputs.flat_batch:
            num_available_steps = context.compute_num_available_steps(
                max_seq_len
            )
            assert num_available_steps > 0
            num_steps = min(num_steps, num_available_steps)

        # Claim cache rows for context.
        for replica_idx, batch in enumerate(inputs.batches):
            for context in batch:
                if not self.kv_manager.contains(
                    context.request_id, replica_idx=replica_idx
                ):
                    self.kv_manager.claim(
                        context.request_id, replica_idx=replica_idx
                    )

        for replica_idx, batch in enumerate(inputs.batches):
            for ctx in batch:
                self.kv_manager.alloc(
                    ctx, replica_idx=replica_idx, num_steps=num_steps
                )
        self.kv_manager.runtime_inputs(inputs.batches, num_steps=num_steps)

        # Generate the responses
        responses = {}
        for context in inputs.flat_batch:
            req_id = context.request_id
            for _ in range(num_steps):
                context.update(new_token=self.token_id)
                self.token_id += 1

                if len(context.tokens) == context.max_length:
                    context.status = GenerationStatus.MAXIMUM_LENGTH

                if context.is_done:
                    break

            output = context.to_generation_output()
            if output.tokens:
                responses[req_id] = output

        # Step the kv cache manager
        self.kv_manager.step(inputs.batches)

        return responses

    def release(self, request_id: RequestID) -> None:
        # No-op. Previously the pipeline was responsible for calling kv.release().
        # but now the whole lifecycle is managed by the scheduler.
        pass


@dataclass(eq=True)
class BatchInfo:
    batch_type: BatchType
    """Type of the batch, either CE or TG"""

    batch_size: int
    """Batch size. This is the number of requests in the batch."""

    terminated: int
    """Number of requests that were terminated after this iteration in the batch."""

    steps: int
    """Number of steps to execute for."""

    preempted: int = -1
    """Number of requests that were preempted while scheduling this batch."""

    input_toks: int = -1
    """Total number of input tokens across all requests in the batch."""

    cached_toks: int = -1
    """Total number of cached context tokens across all requests in the batch."""

    @classmethod
    def empty(cls) -> BatchInfo:
        return BatchInfo(
            BatchType.TG,
            batch_size=0,
            terminated=0,
            steps=0,
            preempted=0,
            input_toks=0,
            cached_toks=0,
        )

    def __repr__(self) -> str:
        return (
            f"BatchInfo("
            f"{self.batch_type.value}, "
            f"batch_size={self.batch_size}, "
            f"terminated={self.terminated}, "
            f"steps={self.steps}, "
            f"preempted={self.preempted}, "
            f"input_toks={self.input_toks}, "
            f"cached_toks={self.cached_toks}"
            f")"
        )


def pretty_format_batch_info_list(batch_info_list: list[BatchInfo]) -> str:
    """Pretty format a list of BatchInfo for printing to the console."""
    return "[\n\t" + "\n\t".join([f"{x}," for x in batch_info_list]) + "\n]"


def assert_batch_info_equal(
    actual: list[BatchInfo], expected: list[BatchInfo]
) -> None:
    """Assert that two lists of BatchInfo are equal.

    When the lists are unequal, this method ensures that the output dumped to the
    console is easily copy-pastable into the test code.

    This method is preferred over `assert actual == expected`.

    If we naively compare the lists via above method, the output is very
    verbose and cluttered. The assert dumps the contents of `expected` which is
    unnecessary since it is present in the code. Pytest also often elides some
    elements of the list, preventing us from copy-pasting the list into the test code.
    """

    if len(actual) != len(expected):
        # Save lengths to local variable so pytest does not try to print `actual` / `expected`.
        len_actual = len(actual)
        len_expected = len(expected)
        raise AssertionError(
            f"Lengths of actual and expected batch infos do not match: {len_actual} != {len_expected}. Actual:\n"
            f"{pretty_format_batch_info_list(actual)}\nExpected:\n"
            f"{pretty_format_batch_info_list(expected)}"
        )
    for i in range(len(actual)):
        if actual[i] != expected[i]:
            raise AssertionError(
                f"Batch info at index {i} does not match: {actual[i]} != {expected[i]}. Actual:\n"
                f"{pretty_format_batch_info_list(actual)}\nExpected:\n"
                f"{pretty_format_batch_info_list(expected)}"
            )


def create_batch_and_execute(scheduler: TokenGenerationScheduler) -> BatchInfo:
    scheduler._retrieve_pending_requests()
    batch_constructor = scheduler.batch_constructor

    num_preempted_before = scheduler.batch_constructor.total_preemption_count
    inputs = batch_constructor.construct_batch()
    num_preempted_after = scheduler.batch_constructor.total_preemption_count

    num_preempted = num_preempted_after - num_preempted_before
    batch_size = len(inputs.flat_batch)
    batch_type = inputs.batch_type
    input_tokens = inputs.input_tokens
    num_steps = inputs.num_steps
    batch_context_length = sum(
        context.tokens.processed_length for context in inputs.flat_batch
    )

    if batch_size == 0:
        return BatchInfo.empty()

    num_terminated_reqs = scheduler._schedule(inputs)
    assert isinstance(scheduler.pipeline, FakeTokenGeneratorPipeline)

    return BatchInfo(
        batch_type=batch_type,
        batch_size=batch_size,
        terminated=num_terminated_reqs,
        steps=num_steps,
        preempted=num_preempted,
        input_toks=input_tokens,
        cached_toks=batch_context_length,
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
    queue: MAXPushQueue[TextContext],
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
    queue: MAXPushQueue[TextContext],
    tokens: np.ndarray,
    max_seq_len: int,
) -> None:
    context = TextContext(
        request_id=RequestID(),
        max_length=max_seq_len,
        tokens=TokenBuffer(tokens),
    )

    queue.put_nowait(context)


CE = BatchType.CE
TG = BatchType.TG
