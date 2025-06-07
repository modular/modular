# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from queue import Queue
from unittest.mock import Mock
from uuid import uuid4

import numpy as np
import pytest
import zmq
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.nn.kv_cache import KVCacheParams, KVCacheStrategy, PagedKVCacheManager
from max.pipelines.core import (
    AudioGenerationResponse,
    AudioGenerator,
    TextContext,
    TextGenerationStatus,
    TTSContext,
    msgpack_numpy_encoder,
)
from max.serve.process_control import ProcessControl
from max.serve.queue.zmq_queue import ZmqPullSocket, ZmqPushSocket
from max.serve.scheduler import (
    AudioGenerationScheduler,
    TokenGenerationSchedulerConfig,
)
from max.serve.scheduler.text_generation_scheduler import BatchType


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


def request_zmq_endpoint():
    return f"ipc://{tempfile.gettempdir()}/{uuid4()}"


def response_zmq_endpoint():
    return f"ipc://{tempfile.gettempdir()}/{uuid4()}"


def cancel_zmq_endpoint():
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
    batch_timeout=None,
    enable_prefix_caching=False,
    enable_in_flight_batching=False,
    enable_kvcache_swapping_to_host=False,
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

    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size_tg=max_batch_size,
        max_forward_steps_tg=max_forward_steps_tg,
        target_tokens_per_batch_tg=target_tokens_per_batch_tg,
        max_batch_size_ce=max_batch_size,
        max_forward_steps_ce=1,
        target_tokens_per_batch_ce=target_tokens_per_batch_ce,
        batch_timeout=batch_timeout,
        enable_in_flight_batching=enable_in_flight_batching,
    )
    token_pipeline = FakeAudioGeneratorPipeline(paged_manager)
    scheduler = AudioGenerationScheduler(
        process_control=create_process_control(),
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
    def __init__(self, paged_manager: PagedKVCacheManager):
        self.paged_manager = paged_manager

    def next_chunk(
        self, batch: dict[str, TTSContext], num_tokens: int = 1
    ) -> dict[str, AudioGenerationResponse]:
        # Truncate num steps based on the max seq len
        for context in batch.values():
            num_available_steps = context.compute_num_available_steps(
                self.paged_manager.max_seq_len
            )
            assert num_available_steps > 0
            num_tokens = min(num_tokens, num_available_steps)
        self.prev_num_steps = num_tokens

        ctxs: list[TTSContext] = list(batch.values())

        self.paged_manager.fetch(ctxs, num_steps=num_tokens)

        # Generate the responses
        responses = {}
        for req_id, context in batch.items():
            resp = AudioGenerationResponse(TextGenerationStatus.ACTIVE)
            for _ in range(num_tokens):
                context.update(new_token=rand(1)[0])

                if context.current_length == context.max_length:
                    resp = AudioGenerationResponse(
                        TextGenerationStatus.MAXIMUM_LENGTH
                    )

                if resp.is_done:
                    break

            responses[req_id] = resp

        # Step the kv cache manager
        self.paged_manager.step(ctxs)

        return responses

    def release(self, _: TTSContext):
        pass

    @property
    def decoder_sample_rate(self) -> int:
        return 999


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
) -> BatchInfo:
    batch = scheduler._create_batch()

    batch_size = batch.batch_size
    batch_type = batch.batch_type
    input_tokens = batch.input_tokens
    num_steps = batch.num_steps
    if batch.batch_size == 0:
        return BatchInfo.empty()

    scheduler._schedule(batch)
    terminated_reqs = batch.num_terminated

    assert isinstance(scheduler.pipeline, FakeAudioGeneratorPipeline)

    # Pipelines should use whatever num_steps that the scheduler computed.
    # It should not need to truncate it.
    assert scheduler.pipeline.prev_num_steps == num_steps

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

    for _ in range(max_num_iters):
        batch_info = create_batch_and_execute(scheduler)
        batch_infos.append(batch_info)
        if batch_info.batch_size == 0:
            break
    return batch_infos


def enqueue_request(
    socket: ZmqPushSocket[tuple[str, TTSContext]],
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
    socket.put_nowait((req_id, context))


def enqueue_request_with_prompt(
    socket: ZmqPushSocket[tuple[str, TTSContext]],
    tokens: np.ndarray,
    max_seq_len: int,
):
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
):
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
):
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


def test_paged_scheduler_num_prompts_100_prompt_len_500_output_tokens_16_prefix_len_384(
    zmq_ctx,
):
    num_prompts = 100
    prompt_len = 500
    output_tokens = 16
    prefix_len = 384

    scheduler = create_paged_scheduler(
        zmq_ctx=zmq_ctx,
        response_zmq_endpoint=response_zmq_endpoint(),
        request_zmq_endpoint=request_zmq_endpoint(),
        cancel_zmq_endpoint=cancel_zmq_endpoint(),
        enable_in_flight_batching=False,
        enable_prefix_caching=True,
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

    # We predict approx 384 tokens to be cache hit.
    # This means we encode 500 - 384 = 116 tokens per CE batch.
    # Hence, we will schedule approx 8192 / 116 = 70.62 CE req per batch.
    # This is rounded up to 71 due to chunked prefill.
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


def test_paged_scheduler_num_prompts_100_prompt_len_500_output_tokens_16_in_flight_batching(
    zmq_ctx,
):
    num_prompts = 100
    prompt_len = 500
    output_tokens = 16

    scheduler = create_paged_scheduler(
        zmq_ctx=zmq_ctx,
        request_zmq_endpoint=request_zmq_endpoint(),
        response_zmq_endpoint=response_zmq_endpoint(),
        cancel_zmq_endpoint=cancel_zmq_endpoint(),
        enable_in_flight_batching=True,
    )

    push_socket = ZmqPushSocket[tuple[str, TTSContext]](
        zmq_ctx,
        scheduler.request_q.zmq_endpoint,
        serialize=msgpack_numpy_encoder(),
    )

    response_pull_socket = ZmqPullSocket[
        list[dict[str, AudioGenerationResponse]]
    ](zmq_ctx, scheduler.response_q.zmq_endpoint)

    _ = ZmqPullSocket[list[str]](zmq_ctx, scheduler.cancel_q.zmq_endpoint)

    for _ in range(num_prompts):
        enqueue_request(
            push_socket,
            prompt_len=prompt_len,
            max_seq_len=prompt_len + output_tokens,
        )
    time.sleep(1)

    # With inflight batching, the CE batches become bigger and bigger since they
    # now include TG requests.
    expected = [
        # batch_type, batch_size, terminated, num_steps, input_tokens
        BatchInfo(CE, 17, 0, 1, 8500),
        BatchInfo(CE, 34, 0, 1, 8517),
        BatchInfo(CE, 51, 0, 1, 8534),
        BatchInfo(CE, 68, 0, 1, 8551),
        BatchInfo(CE, 85, 0, 1, 8568),
        BatchInfo(CE, 100, 0, 1, 7585),
        BatchInfo(TG, 100, 17, 10, 100),
        BatchInfo(TG, 83, 83, 10, 83),
        BatchInfo(TG, 0, 0, 0, 0),
    ]
    actual = run_until_completion(scheduler)
    assert len(actual) == len(expected) and actual == expected
