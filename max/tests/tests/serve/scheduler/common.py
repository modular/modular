# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import queue
from dataclasses import dataclass

import numpy as np
from max.driver import CPU, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.interfaces import (
    BatchType,
    GenerationStatus,
    MAXPushQueue,
    Pipeline,
    RequestID,
    SchedulerResult,
    TextGenerationInputs,
    TextGenerationOutput,
)
from max.kv_cache import PagedKVCacheManager
from max.nn.kv_cache import KVCacheParams, KVCacheStrategy
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
        tokens = np.ones(prompt_len, dtype=np.int32)
    else:
        rem_tokens = prompt_len - len(shared_prefix)
        assert rem_tokens >= 0
        tokens = np.concatenate([shared_prefix, rand(rem_tokens)])

    return TextContext(
        request_id=RequestID(),
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
    dp: int = 1,
    device: Device = CPU(),
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
    ) * dp

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
        data_parallel_degree=dp,
        n_devices=dp,
    )

    session = InferenceSession(devices=[device])

    kv_manager = PagedKVCacheManager(
        params=kv_params,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        num_layers=NUM_LAYERS,
        devices=[device] * dp,
        session=session,
        available_cache_memory=cache_memory,
        page_size=page_size,
        enable_runtime_checks=True,
    )

    assert kv_manager.total_num_pages == num_blocks * dp
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
    max_batch_context_length: int | None = None,
    dp: int = 1,
    device: Device = CPU(),
) -> tuple[
    TokenGenerationScheduler,
    MAXPushQueue[TextContext],
]:
    # Create a paged manager that has one slot
    paged_manager = create_paged_manager(
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
        max_batch_size_tg=max_batch_size,
        max_forward_steps_tg=max_forward_steps_tg,
        max_batch_size_ce=max_batch_size,
        target_tokens_per_batch_ce=target_tokens_per_batch_ce,
        max_seq_len=max_seq_len,
        enable_chunked_prefill=enable_chunked_prefill,
        enable_in_flight_batching=enable_in_flight_batching,
        max_batch_context_length=max_batch_context_length,
        data_parallel_degree=dp,
    )
    token_pipeline = FakeTokenGeneratorPipeline(paged_manager)
    request_queue: queue.Queue[TextContext] = queue.Queue()
    response_queue: queue.Queue[
        dict[RequestID, SchedulerResult[TextGenerationOutput]]
    ] = queue.Queue()
    cancel_queue: queue.Queue[list[RequestID]] = queue.Queue()
    scheduler = TokenGenerationScheduler(
        scheduler_config=scheduler_config,
        pipeline=token_pipeline,
        paged_manager=paged_manager,
        request_queue=request_queue,
        response_queue=response_queue,
        cancel_queue=cancel_queue,
        offload_queue_draining=False,
    )

    return (scheduler, request_queue)


class FakeTokenGeneratorPipeline(
    Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput]
):
    def __init__(
        self, kv_manager: PagedKVCacheManager, start_token_id: int = 42
    ) -> None:
        self.kv_manager = kv_manager
        self.token_id = start_token_id

    def execute(
        self, inputs: TextGenerationInputs[TextContext]
    ) -> dict[RequestID, TextGenerationOutput]:
        max_seq_len = self.kv_manager.max_seq_len
        # Truncate num steps based on the max seq len
        for context in inputs.batch.values():
            num_available_steps = context.compute_num_available_steps(
                max_seq_len
            )
            assert num_available_steps > 0
            num_steps = min(inputs.num_steps, num_available_steps)

        # Claim cache rows for context.
        for _, context in inputs.batch.items():
            if not self.kv_manager.contains(context.request_id):
                self.kv_manager.external_claim(context.request_id)

        ctxs: list[TextContext] = list(inputs.batch.values())

        self.kv_manager.fetch(ctxs, num_steps=num_steps)

        # Generate the responses
        responses = {}
        for req_id, context in inputs.batch.items():
            for _ in range(num_steps):
                context.update(new_token=self.token_id)
                self.token_id += 1

                if context.current_length == context.max_length:
                    context.status = GenerationStatus.MAXIMUM_LENGTH

                if context.is_done:
                    break

            responses[req_id] = context.to_generation_output()

        # Step the kv cache manager
        self.kv_manager.step(ctxs)

        return responses

    def release(self, request_id: RequestID) -> None:
        self.kv_manager.release(request_id)


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
            f"{pretty_format_batch_info_list(actual)}"
        )
    for i in range(len(actual)):
        if actual[i] != expected[i]:
            raise AssertionError(
                f"Batch info at index {i} does not match: {actual[i]} != {expected[i]}. Actual:\n"
                f"{pretty_format_batch_info_list(actual)}"
            )


def create_batch_and_execute(scheduler: TokenGenerationScheduler) -> BatchInfo:
    scheduler._retrieve_pending_requests()
    batch_constructor = scheduler.batch_constructor

    num_preempted_before = scheduler.batch_constructor.total_preemption_count
    inputs = batch_constructor.construct_batch()
    num_preempted_after = scheduler.batch_constructor.total_preemption_count

    num_preempted = num_preempted_after - num_preempted_before
    batch_size = len(inputs.batch)
    batch_type = inputs.batch_type
    input_tokens = inputs.input_tokens
    num_steps = inputs.num_steps
    batch_context_length = sum(
        context.start_idx for context in inputs.batch.values()
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
    assert context.active_length == prompt_len
    queue.put_nowait(context)


def enqueue_request_with_prompt(
    queue: MAXPushQueue[TextContext],
    tokens: np.ndarray,
    max_seq_len: int,
) -> None:
    context = TextContext(
        request_id=RequestID(),
        max_length=max_seq_len,
        tokens=tokens,
    )

    queue.put_nowait(context)


CE = BatchType.CE
TG = BatchType.TG
