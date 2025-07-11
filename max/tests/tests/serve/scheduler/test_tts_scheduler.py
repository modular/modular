# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import tempfile
import time
from collections.abc import Generator
from dataclasses import dataclass
from queue import Queue
from uuid import uuid4

import numpy as np
import pytest
import zmq
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.interfaces import AudioGenerationResponse, GenerationStatus
from max.nn.kv_cache import KVCacheParams, KVCacheStrategy, PagedKVCacheManager
from max.pipelines.core import (
    AudioGenerator,
    TextContext,
    TTSContext,
    msgpack_numpy_encoder,
)
from max.serve.queue.zmq_queue import ZmqPullSocket, ZmqPushSocket
from max.serve.scheduler import AudioGenerationScheduler
from max.serve.scheduler.audio_generation_scheduler import (
    AudioGenerationSchedulerConfig,
    AudioGenerationSchedulerOutput,
)
from max.serve.scheduler.text_generation_scheduler import BatchType


def create_queues() -> dict[str, Queue]:
    return {"REQUEST": Queue(), "RESPONSE": Queue(), "CANCEL": Queue()}


def rand(length: int) -> np.ndarray:
    return np.random.randint(0, 256, size=length)


def create_text_context(
    prompt_len: int,
    max_seq_len: int,
    shared_prefix: np.ndarray | None = None,
) -> TTSContext:
    if shared_prefix is None:
        tokens = np.ones(prompt_len, dtype=np.int32)
    else:
        rem_tokens = prompt_len - len(shared_prefix)
        assert rem_tokens >= 0
        tokens = np.concatenate([shared_prefix, rand(rem_tokens)])

    return TTSContext(
        prompt=tokens.tolist(),
        max_length=max_seq_len,
        tokens=tokens,
        streaming=False,
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


def request_zmq_endpoint() -> str:
    return f"ipc://{tempfile.gettempdir()}/{uuid4()}"


def response_zmq_endpoint() -> str:
    return f"ipc://{tempfile.gettempdir()}/{uuid4()}"


def cancel_zmq_endpoint() -> str:
    return f"ipc://{tempfile.gettempdir()}/{uuid4()}"


@pytest.fixture(scope="session")
def zmq_ctx():
    return zmq.Context(io_threads=2)


def create_paged_scheduler(
    zmq_ctx,
    request_zmq_endpoint,
    response_zmq_endpoint,
    cancel_zmq_endpoint,
    max_seq_len=2048,
    num_blocks=9999,
    max_batch_size=512,
    page_size=128,
    max_forward_steps_tg=10,
    target_tokens_per_batch_tg=None,
    target_tokens_per_batch_ce=8192,
    enable_prefix_caching=False,
    enable_in_flight_batching=False,
    enable_kvcache_swapping_to_host=False,
    max_queue_size_tg=None,
    min_batch_size_tg=None,
    ce_delay_ms=0.0,
    enable_prioritize_first_decode=False,
) -> AudioGenerationScheduler:
    # Create a paged manager that has one slot
    paged_manager = create_paged_manager(
        num_blocks=num_blocks,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        page_size=page_size,
        enable_prefix_caching=enable_prefix_caching,
        enable_kvcache_swapping_to_host=enable_kvcache_swapping_to_host,
    )

    scheduler_config = AudioGenerationSchedulerConfig(
        max_batch_size_tg=max_batch_size,
        max_forward_steps_tg=max_forward_steps_tg,
        target_tokens_per_batch_tg=target_tokens_per_batch_tg,
        max_batch_size_ce=max_batch_size,
        max_forward_steps_ce=1,
        target_tokens_per_batch_ce=target_tokens_per_batch_ce,
        enable_in_flight_batching=enable_in_flight_batching,
        max_queue_size_tg=max_queue_size_tg,
        min_batch_size_tg=min_batch_size_tg,
        ce_delay_ms=ce_delay_ms,
        enable_prioritize_first_decode=enable_prioritize_first_decode,
    )
    token_pipeline = FakeAudioGeneratorPipeline(
        paged_manager, max_num_steps=max_forward_steps_tg
    )
    scheduler = AudioGenerationScheduler(
        scheduler_config=scheduler_config,
        pipeline=token_pipeline,
        request_zmq_endpoint=request_zmq_endpoint,
        response_zmq_endpoint=response_zmq_endpoint,
        cancel_zmq_endpoint=cancel_zmq_endpoint,
        zmq_ctx=zmq_ctx,
        paged_manager=paged_manager,
    )
    return scheduler


class FakeAudioGeneratorPipeline(AudioGenerator):
    def __init__(
        self, paged_manager: PagedKVCacheManager, max_num_steps: int
    ) -> None:
        self.paged_manager = paged_manager
        self.max_num_steps = max_num_steps
        self._prev_num_steps: int | None = None

    def next_chunk(
        self, batch: dict[str, TTSContext]
    ) -> dict[str, AudioGenerationResponse]:
        is_ce = next(iter(batch.values())).is_ce

        if is_ce:
            num_tokens = 1
        else:
            num_tokens = self.max_num_steps

        # Truncate num steps based on the max seq len
        for context in batch.values():
            num_available_steps = context.compute_num_available_steps(
                self.paged_manager.max_seq_len
            )
            assert num_available_steps > 0
            num_tokens = min(num_tokens, num_available_steps)

        self._prev_num_steps = num_tokens

        ctxs: list[TTSContext] = list(batch.values())

        self.paged_manager.fetch(ctxs, num_steps=num_tokens)

        # Generate the responses
        responses = {}
        for req_id, context in batch.items():
            resp = AudioGenerationResponse(GenerationStatus.ACTIVE)
            for _ in range(num_tokens):
                context.update(new_token=rand(1)[0])

                if context.current_length == context.max_length:
                    resp = AudioGenerationResponse(
                        GenerationStatus.MAXIMUM_LENGTH
                    )

                    # Pretend that the audio generation is done immediately when
                    # text generation is done.
                    context.update_audio_generation_status(
                        GenerationStatus.MAXIMUM_LENGTH
                    )

                if resp.is_done:
                    break

            responses[req_id] = resp

        # Step the kv cache manager
        self.paged_manager.step(ctxs)

        return responses

    def release(self, _: TTSContext) -> None:
        pass

    @property
    def decoder_sample_rate(self) -> int:
        return 999

    @property
    def prev_num_steps(self) -> int:
        assert self._prev_num_steps is not None
        return self._prev_num_steps


@dataclass
class BatchInfo:
    batch_type: BatchType
    batch_size: int
    terminated: int
    num_steps: int
    input_tokens: int

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
            self.input_tokens,
        ) == (
            other.batch_type,
            other.batch_size,
            other.terminated,
            other.num_steps,
            other.input_tokens,
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

    num_steps = scheduler.pipeline.prev_num_steps
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
    socket: ZmqPushSocket[tuple[str, TTSContext]],
    prompt_len: int,
    max_seq_len: int,
    shared_prefix: np.ndarray | None = None,
) -> None:
    context = create_text_context(
        prompt_len=prompt_len,
        max_seq_len=max_seq_len,
        shared_prefix=shared_prefix,
    )
    req_id = f"req{uuid4()}"
    assert context.active_length == prompt_len
    socket.put_nowait((req_id, context))


def enqueue_request_with_prompt(
    socket: ZmqPushSocket[tuple[str, TTSContext]],
    tokens: np.ndarray,
    max_seq_len: int,
) -> None:
    context = TextContext(
        prompt=tokens.tolist(),
        max_length=max_seq_len,
        tokens=tokens,
    )
    req_id = f"req{uuid4()}"

    socket.put_nowait((req_id, context))


CE = BatchType.ContextEncoding
TG = BatchType.TokenGeneration


@pytest.mark.parametrize("num_reqs", [1, 2, 3])
def test_paged_scheduler_tg_request_exceed_max_seq_len(
    num_reqs,
    zmq_ctx,
) -> None:
    max_seq_len = 2048
    page_size = 128
    num_blocks = max_seq_len / page_size * num_reqs
    scheduler = create_paged_scheduler(
        zmq_ctx=zmq_ctx,
        response_zmq_endpoint=response_zmq_endpoint(),
        request_zmq_endpoint=request_zmq_endpoint(),
        cancel_zmq_endpoint=cancel_zmq_endpoint(),
        max_seq_len=max_seq_len,
        max_batch_size=100,
        num_blocks=num_blocks,
        page_size=page_size,
    )

    push_socket = ZmqPushSocket[tuple[str, TTSContext]](
        zmq_ctx,
        scheduler.request_q.zmq_endpoint,
        serialize=msgpack_numpy_encoder(),
    )

    # Create this so the schedule process has a client to send to.
    _ = ZmqPullSocket[list[dict[str, AudioGenerationResponse]]](
        zmq_ctx, scheduler.response_q.zmq_endpoint
    )

    # Check that we would exceed max_seq_len during TG step
    prompt_len = 2040
    num_steps = scheduler.scheduler_config.max_forward_steps_tg
    assert num_steps == 10
    assert prompt_len + num_steps > max_seq_len

    # Create a few requests with 2040 tokens
    for _ in range(num_reqs):
        enqueue_request(push_socket, prompt_len, max_seq_len=max_seq_len)
    time.sleep(1)

    expected = [
        # batch_type, batch_size, terminated, num_steps, input_tokens
        BatchInfo(CE, num_reqs, 0, 1, num_reqs * prompt_len),
        BatchInfo(TG, num_reqs, num_reqs, 8, num_reqs * 1),
        BatchInfo.empty(),
    ]
    actual = run_until_completion(scheduler)
    assert len(actual) == len(expected) and actual == expected
    del push_socket


def test_paged_scheduler_num_prompts_100_prompt_len_500_output_tokens_16(
    zmq_ctx,
) -> None:
    num_prompts = 100
    prompt_len = 500
    output_tokens = 16

    scheduler = create_paged_scheduler(
        zmq_ctx=zmq_ctx,
        response_zmq_endpoint=response_zmq_endpoint(),
        request_zmq_endpoint=request_zmq_endpoint(),
        cancel_zmq_endpoint=cancel_zmq_endpoint(),
        enable_in_flight_batching=False,
    )
    push_socket = ZmqPushSocket[tuple[str, TTSContext]](
        zmq_ctx,
        scheduler.request_q.zmq_endpoint,
        serialize=msgpack_numpy_encoder(),
    )

    _ = ZmqPullSocket[list[dict[str, AudioGenerationResponse]]](
        zmq_ctx, scheduler.response_q.zmq_endpoint
    )

    for _ in range(num_prompts):
        enqueue_request(
            push_socket,
            prompt_len=prompt_len,
            max_seq_len=prompt_len + output_tokens,
        )
    time.sleep(1)

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
    zmq_ctx, enable_prioritize_first_decode
) -> None:
    num_prompts = 100
    prompt_len = 500
    output_tokens = 16
    prefix_len = 384

    scheduler = create_paged_scheduler(
        zmq_ctx=zmq_ctx,
        response_zmq_endpoint=response_zmq_endpoint(),
        request_zmq_endpoint=request_zmq_endpoint(),
        cancel_zmq_endpoint=cancel_zmq_endpoint(),
        enable_prefix_caching=True,
        enable_prioritize_first_decode=enable_prioritize_first_decode,
    )

    push_socket = ZmqPushSocket[tuple[str, TTSContext]](
        zmq_ctx,
        scheduler.request_q.zmq_endpoint,
        serialize=msgpack_numpy_encoder(),
    )

    _ = ZmqPullSocket[list[dict[str, AudioGenerationResponse]]](
        zmq_ctx, scheduler.response_q.zmq_endpoint
    )

    _ = ZmqPullSocket[list[str]](
        zmq_ctx,
        scheduler.cancel_q.zmq_endpoint,
    )

    # set seed for reproducibility
    np.random.seed(42)
    shared_prefix = rand(prefix_len)

    for _ in range(num_prompts):
        enqueue_request(
            push_socket,
            prompt_len=prompt_len,
            max_seq_len=prompt_len + output_tokens,
            shared_prefix=shared_prefix,
        )
    time.sleep(1)

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
    zmq_ctx,
    max_queue_size_tg,
) -> None:
    num_prompts = 100
    prompt_len = 500
    output_tokens = 16

    scheduler = create_paged_scheduler(
        zmq_ctx=zmq_ctx,
        response_zmq_endpoint=response_zmq_endpoint(),
        request_zmq_endpoint=request_zmq_endpoint(),
        cancel_zmq_endpoint=cancel_zmq_endpoint(),
        max_batch_size=32,
        max_queue_size_tg=max_queue_size_tg,
    )

    push_socket = ZmqPushSocket[tuple[str, TTSContext]](
        zmq_ctx,
        scheduler.request_q.zmq_endpoint,
        serialize=msgpack_numpy_encoder(),
    )

    _ = ZmqPullSocket[list[dict[str, AudioGenerationResponse]]](
        zmq_ctx, scheduler.response_q.zmq_endpoint
    )

    _ = ZmqPullSocket[list[str]](
        zmq_ctx,
        scheduler.cancel_q.zmq_endpoint,
    )

    # set seed for reproducibility
    np.random.seed(42)

    for _ in range(num_prompts):
        enqueue_request(
            push_socket,
            prompt_len=prompt_len,
            max_seq_len=prompt_len + output_tokens,
        )
    time.sleep(1)

    if max_queue_size_tg is None:
        # Notice that max_queue_size_tg defaults to max_batch_size_tg. This causes
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
    zmq_ctx,
    min_batch_size_tg,
    max_batch_size,
    max_queue_size_tg,
) -> None:
    num_prompts = 128
    prompt_len = 500
    output_tokens = 16

    scheduler = create_paged_scheduler(
        zmq_ctx=zmq_ctx,
        response_zmq_endpoint=response_zmq_endpoint(),
        request_zmq_endpoint=request_zmq_endpoint(),
        cancel_zmq_endpoint=cancel_zmq_endpoint(),
        min_batch_size_tg=min_batch_size_tg,
        max_batch_size=max_batch_size,
        max_queue_size_tg=max_queue_size_tg,
        target_tokens_per_batch_ce=16384,
    )

    push_socket = ZmqPushSocket[tuple[str, TTSContext]](
        zmq_ctx,
        scheduler.request_q.zmq_endpoint,
        serialize=msgpack_numpy_encoder(),
    )

    _ = ZmqPullSocket[list[dict[str, AudioGenerationResponse]]](
        zmq_ctx, scheduler.response_q.zmq_endpoint
    )

    _ = ZmqPullSocket[list[str]](
        zmq_ctx,
        scheduler.cancel_q.zmq_endpoint,
    )

    # set seed for reproducibility
    np.random.seed(42)

    for _ in range(num_prompts):
        enqueue_request(
            push_socket,
            prompt_len=prompt_len,
            max_seq_len=prompt_len + output_tokens,
        )
    time.sleep(1)

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
