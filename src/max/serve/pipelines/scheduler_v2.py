# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import logging
import queue
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Queue
from typing import Any, Mapping, Optional, Tuple, TypeVar

import numpy as np
from max.pipelines import TokenGenerator
from max.pipelines.interfaces import EmbeddingsGenerator
from max.pipelines.kv_cache.paged_cache import PagedKVCacheManager
from max.profiler import traced
from max.serve.scheduler.process_control import ProcessControl
from max.serve.scheduler.queues import STOP_STREAM

logger = logging.getLogger("max.serve")

BatchReqId = TypeVar("BatchReqId")
BatchReqInput = TypeVar("BatchReqInput")
BatchInputs = dict[BatchReqId, BatchReqInput]


class BatchType(Enum):
    ContextEncoding = 1
    TokenGeneration = 2

    def concise_name(self):
        if self == BatchType.ContextEncoding:
            return "CE"
        else:
            assert self == BatchType.TokenGeneration
            return "TG"


class RequestDeque:
    """A wrapper around the multiprocessing queue that allows us to add
    requests to the front of the queue.
    """

    def __init__(self, queue: Queue):
        self.queue = queue
        self.evicted: list[BatchInputs] = []

    def empty(self) -> bool:
        return self.queue.empty() and len(self.evicted) == 0

    def get_nowait(self) -> Any:
        if self.evicted:
            return self.evicted.pop()
        return self.queue.get_nowait()

    def put_front_nowait(self, item):
        self.evicted.append(item)

    def put(self, item):
        self.queue.put(item)

    def qsize(self) -> Optional[int]:
        try:
            return len(self.evicted) + self.queue.qsize()
        except NotImplementedError:
            return None


class Scheduler(ABC):
    """Abstract base class defining the interface for schedulers."""

    @abstractmethod
    def run(self):
        """The main scheduler loop that creates and executes batches.

        This method should implement the core scheduling logic including:
        - Batch creation and management
        - Request scheduling
        - Error handling
        """
        pass


@dataclass
class TokenGenerationSchedulerConfig:
    """Scheduler configuration."""

    # The maximum number of requests that can be in the token generation batch.
    max_batch_size_tg: int
    # The number of tokens to generate for each request in the token generation iteration.
    max_forward_steps_tg: int
    # The target total number of tokens to generate in the token generation batch.
    target_tokens_per_batch_tg: Optional[int]

    # The maximum number of requests that can be in the context encoding batch.
    max_batch_size_ce: int
    # The number of tokens to encode for each request in the context encoding iteration.
    max_forward_steps_ce: int
    # The target total number of tokens to encode in the context encoding batch.
    target_tokens_per_batch_ce: Optional[int]

    # The maximum amount of time to wait before creating a context encoding batch.
    batch_timeout: Optional[float]

    # Enables chunked prefill, where the scheduler splits requests into chunks to
    # ensure each batch contains exactly `target_tokens_per_batch_ce` tokens.
    enable_chunked_prefill: bool = True

    # When enabled, prioritizes token generation by batching it with context
    # encoding requests. Requires chunked prefill.
    enable_in_flight_batching: bool = False

    def __post_init__(self) -> None:
        if (
            self.enable_chunked_prefill
            and self.target_tokens_per_batch_ce is None
        ):
            msg = "Need set `target_tokens_per_batch_ce` for the scheduler to enable chunked prefill."
            raise ValueError(msg)

        if self.enable_chunked_prefill and self.max_forward_steps_ce > 1:
            self.max_forward_steps_ce = 1
            logger.info(
                "Chunked prefill does not support multistep inference, overriding max_forward_steps_ce to 1."
            )

        if self.enable_in_flight_batching and not self.enable_chunked_prefill:
            msg = " Requires chunked prefill for in-flight batching."
            raise ValueError(msg)


class TokenGenerationSchedulerV2(Scheduler):
    def __init__(
        self,
        process_control: ProcessControl,
        scheduler_config: TokenGenerationSchedulerConfig,
        pipeline: TokenGenerator,
        queues: Mapping[str, Queue],
        paged_manager: Optional[PagedKVCacheManager] = None,
    ):
        self.scheduler_config = scheduler_config
        self.pipeline = pipeline

        # Multiprocessing resources.
        self.pc = process_control

        self.request_q = RequestDeque(queues["REQUEST"])
        self.response_q = queues["RESPONSE"]
        self.cancel_q = queues["CANCEL"]

        # Initialize Scheduler state.
        self.active_batch: BatchInputs = {}
        self.available_cache_indices = set(
            range(self.scheduler_config.max_batch_size_tg)
        )
        self.ce_batch_start_time: Optional[float] = None

        # Optional reference to the paged kv cache manager.
        # Note that the paged manager is shared with the model worker thread.
        # Care must be taken to ensure no race conditions.
        self.paged_manager = paged_manager
        self.total_preemption_count = 0
        self.last_preemption_logging_time: float = 0.0

        # TODO health check

    @traced
    def _should_schedule_ce(self) -> bool:
        # No CE to schedule if queue is empty
        if self.request_q.empty():
            return False

        # At this point there are incoming requests, we start the batch timer if not yet
        if self.ce_batch_start_time is None:
            self.ce_batch_start_time = time.monotonic()

        # If TG batch is full then no reason to schedule CE
        if len(self.active_batch) >= self.scheduler_config.max_batch_size_tg:
            return False

        # If TG batch is empty then schedule CE
        if len(self.active_batch) == 0:
            return True

        # If batch timeout is set
        if self.scheduler_config.batch_timeout:
            # If batch timeout is reached then schedule CE
            if (
                self.ce_batch_start_time is not None
                and time.monotonic()
                >= self.ce_batch_start_time
                + self.scheduler_config.batch_timeout
            ):
                return True
            else:
                # TODO(SI-808): The The multiprocessing.Queue implementation isn't portable.
                # Its Queue.qsize() method throws NotImplementedError exception on MacOS and Graviton
                # messages_needed = self.max_batch_size_tg - len(
                #     self.continuous_batch
                # )
                # if self.request_q.qsize() >= messages_needed:
                #     # If there are enough request to fill the TG batch then schedule CE
                #     return True
                # else:
                #     # If not enough requests then hold off the CE and continue with TG
                #     return False

                return False

        return True

    @traced
    def _construct_fetch_input(self, batch) -> dict[int, np.ndarray]:
        """Construct input to the `fetch` method of paged manager"""
        seq_ids_and_prompts = {}
        counter = -1
        for data in batch.values():
            seq_id = data.cache_seq_id
            # we want to assign a unique unused id for seqs not in cache
            if seq_id is None:
                seq_id = counter
                counter -= 1
            seq_ids_and_prompts[seq_id] = data.next_tokens
        return seq_ids_and_prompts

    @traced
    def _try_create_ce_batch(self) -> BatchInputs:
        """Try to create a context encoding batch"""
        max_batch_size_to_create = min(
            self.scheduler_config.max_batch_size_ce,
            self.scheduler_config.max_batch_size_tg - len(self.active_batch),
        )

        ce_batch: BatchInputs = {}
        total_seq_len = 0

        if self.scheduler_config.enable_in_flight_batching:
            if self.active_batch:
                ce_batch = self._create_tg_batch()
                total_seq_len += len(ce_batch)

        for _ in range(max_batch_size_to_create):
            if (
                self.scheduler_config.target_tokens_per_batch_ce is not None
                and total_seq_len
                >= self.scheduler_config.target_tokens_per_batch_ce
            ):
                break

            try:
                req_id, data = self.request_q.get_nowait()
                # This is a partly encoded request if the start_idx !=0.
                # In this scenario, we already allocated a cache slot to it.
                if data.start_idx == 0:
                    data.cache_seq_id = None
            except queue.Empty:
                break

            has_insufficient_kv_blocks = False
            if self.paged_manager:
                seq_ids_and_prompts = self._construct_fetch_input(
                    self.active_batch | ce_batch | {req_id: data}
                )
                # This hardcoded value was chosen empirically to prevent
                # excessive preemption on sharegpt.
                # TODO: we should look at what vLLM does and decide a more
                # intelligent policy for this.
                headroom_for_num_tg_iterations = 5
                num_steps = (
                    self.scheduler_config.max_forward_steps_ce
                    + headroom_for_num_tg_iterations
                    * self.scheduler_config.max_forward_steps_tg
                )
                # Add this additional request to the ce batch if it leaves
                # sufficient kv blocks to run several token gen steps without
                # causing evictions.
                has_insufficient_kv_blocks = not self.paged_manager.can_fetch(
                    seq_ids_and_prompts, num_steps=num_steps
                )

            if has_insufficient_kv_blocks:
                # we cannot schedule this request so return it to the head of
                # the request queue
                self.request_q.put_front_nowait((req_id, data))
                break

            if self._exceeds_batch_token_limit(
                total_seq_len, data.active_length
            ):
                if self.scheduler_config.enable_chunked_prefill:
                    # We can only schedule part of the prompt.
                    # We achieve this by setting the active_idx of the context class.
                    assert (
                        self.scheduler_config.target_tokens_per_batch_ce
                        is not None
                    )

                    token_num_diff = (
                        total_seq_len
                        + data.active_length
                        - self.scheduler_config.target_tokens_per_batch_ce
                    )
                    data.bump_token_indices(active_idx=-token_num_diff)

                    if data.cache_seq_id is None:
                        data.cache_seq_id = self.available_cache_indices.pop()
                        logger.debug(
                            f"Request {req_id} is chunked to len {data.active_length}."
                        )
                    else:
                        logger.debug(
                            f"Request {req_id} is chunked again to len {data.active_length}."
                        )
                    ce_batch[req_id] = data

                elif (
                    self.scheduler_config.target_tokens_per_batch_ce is None
                    or data.active_length
                    > self.scheduler_config.target_tokens_per_batch_ce
                ):
                    # we need to schedule this request even it exceeds the
                    # `target_tokens_per_batch_ce` limit; otherwise, it will
                    # never be scheduled.
                    data.cache_seq_id = self.available_cache_indices.pop()
                    ce_batch[req_id] = data
                else:
                    # we cannot schedule this request so return it to the head of
                    # the request queue
                    self.request_q.put_front_nowait((req_id, data))

                # break because we exceeded the target tokens per batch limit
                break

            seq_len = data.active_length
            if self.paged_manager is not None:
                cached_tokens = self.paged_manager.get_num_cached_tokens(
                    data.next_tokens
                )
                seq_len -= cached_tokens
            total_seq_len += seq_len
            # We will allocate cache if this is a new request
            if data.cache_seq_id is None:
                data.cache_seq_id = self.available_cache_indices.pop()
            else:
                logger.debug(
                    f"Chunked request {req_id} with len {seq_len} is already allocated to cache slot #{data.cache_seq_id}"
                )
            ce_batch[req_id] = data

        return ce_batch

    @traced
    def _preempt_request(self):
        """Preempts the most recently received request from active batch"""
        assert self.active_batch
        # dicts in python pop the last inserted item
        # this corresponds to the most recently received request
        req_id, data = self.active_batch.popitem()
        self.available_cache_indices.add(data.cache_seq_id)
        self.pipeline.release(data)
        data.reset()
        self.request_q.put_front_nowait((req_id, data))

        # Limit logging about preemptions to at most once per second
        current_time = time.monotonic()
        self.total_preemption_count += 1
        if current_time - self.last_preemption_logging_time > 1:
            self.last_preemption_logging_time = current_time
            logger.info(
                f"Preempted a request due to lack of KV pages. This can affect the end-to-end performance. Consider increasing device-memory-utilization to provide more KV cache memory. Total preemption count: {self.total_preemption_count}."
            )

    @traced
    def _create_tg_batch(self) -> BatchInputs:
        """Creates a non empty token generation batch"""
        assert self.active_batch

        # If we are not using paged attention, we can always schedule the active
        # batch since we reserved blocks for all active requests previously
        if self.paged_manager is None:
            return self.active_batch

        # Keep preempting requests until we can schedule the entire active batch
        initial_active_batch_size = len(self.active_batch)
        while self.active_batch:
            seq_ids_and_prompts = self._construct_fetch_input(self.active_batch)
            if self.paged_manager.can_fetch(
                seq_ids_and_prompts,
                num_steps=self.scheduler_config.max_forward_steps_tg,
            ):
                return self.active_batch
            self._preempt_request()

        # We failed to construct a TG batch.
        # This should literally never happen unless the user sets an absurdly
        # large max seq len or the KV cache is so small.
        raise RuntimeError(
            f"Insufficient KV pages to run token generation with batch size "
            f"of one even after preempting {initial_active_batch_size - 1} requests."
            f"You must restart your process and set a lower max seq len to prevent "
            f"a single request from using the entire KV cache."
        )

    @traced
    def _create_batch_to_execute(self) -> Tuple[BatchInputs, BatchType]:
        """Creates a batch to execute"""
        if self._should_schedule_ce():
            ce_batch = self._try_create_ce_batch()
            if ce_batch:
                return ce_batch, BatchType.ContextEncoding
            # failed to create a CE batch, try to create a TG batch instead

        # if there are no active requests, we can't create a TG batch
        if not self.active_batch:
            return {}, BatchType.TokenGeneration

        tg_batch = self._create_tg_batch()
        return tg_batch, BatchType.TokenGeneration

    def _log_metrics(
        self,
        batch_type: BatchType,
        batch_size: int,
        terminated_reqs: int,
        num_prompt_tokens: int,
        batch_creation_time_s: float,
        batch_execution_time_s: float,
    ) -> None:
        assert batch_size > 0
        num_steps = (
            self.scheduler_config.max_forward_steps_ce
            if batch_type == BatchType.ContextEncoding
            else self.scheduler_config.max_forward_steps_tg
        )
        num_generated_tokens = batch_size * num_steps

        # Number of pending requests is unknown if qsize is not supported
        maybe_pending_reqs: Optional[int] = self.request_q.qsize()
        pending_reqs = "?"
        if maybe_pending_reqs is not None:
            pending_reqs = str(maybe_pending_reqs)

        def format_throughput(tps: float) -> str:
            if tps >= 1_000:
                return f"{tps / 1e3:.1f}K tok/s"
            return f"{tps:.1f} tok/s"

        def format_latency(s: float) -> str:
            if s >= 1:
                return f"{s:.1f}s"
            ms = s * 1e3
            if ms >= 1:
                return f"{ms:.1f}ms"
            us = ms * 1e3
            if us >= 1:
                return f"{us:.1f}us"
            ns = us * 1e3
            return f"{ns:.1f}ns"

        # Format latency and throughput metrics
        prompt_throughput_str = format_throughput(
            num_prompt_tokens / batch_execution_time_s
        )
        generation_throughput_str = format_throughput(
            num_generated_tokens / batch_execution_time_s
        )
        batch_creation_latency_str = format_latency(batch_creation_time_s)
        batch_execution_latency_str = format_latency(batch_execution_time_s)

        if self.paged_manager is None:
            logger.debug(
                f"Executed {batch_type.concise_name()} batch with {batch_size} reqs | "
                f"Terminated: {terminated_reqs} reqs, "
                f"Pending: {pending_reqs} reqs | "
                f"Prompt Tput: {prompt_throughput_str}, "
                f"Generation Tput: {generation_throughput_str} | "
                f"Batch creation: {batch_creation_latency_str}, "
                f"Execution: {batch_execution_latency_str}"
            )
            return

        # KVCache specific metrics
        used_blocks = self.paged_manager.get_num_used_blocks()
        total_blocks = self.paged_manager.total_num_pages
        used_pct = used_blocks / total_blocks
        cache_hit_rate = self.paged_manager.cache_hit_rate()

        logger.debug(
            f"Executed {batch_type.concise_name()} batch with {batch_size} reqs | "
            f"Terminated: {terminated_reqs} reqs, "
            f"Pending: {pending_reqs} reqs | "
            f"Prompt Tput: {prompt_throughput_str}, "
            f"Generation Tput: {generation_throughput_str} | "
            f"Batch creation: {batch_creation_latency_str}, "
            f"Execution: {batch_execution_latency_str} | "
            f"KVCache usage: {used_pct:.1%} of {total_blocks} blocks, "
            f"Cache hit rate: {cache_hit_rate:.1%}, "
            f"All Preemptions: {self.total_preemption_count} reqs"
        )

    def run(self):
        """The Scheduler loop that creates batches and schedules them on GPU"""
        i = 0
        while i % 10 or not self.pc.is_canceled():
            self.pc.beat()
            i += 1
            try:
                # Construct the batch to execute
                t0 = time.monotonic()
                batch_to_execute, batch_type = self._create_batch_to_execute()
                t1 = time.monotonic()
                batch_creation_time_s = t1 - t0

                # If the batch is empty, skip
                if len(batch_to_execute) == 0:
                    continue

                # Records the batch size and number of prompt tokens for logging.
                # This needs to be done before scheduling the batch as the batch
                # input and prompts are mutated in place while executing the
                # batch.
                batch_size = len(batch_to_execute)
                num_prompt_tokens = sum(
                    data.active_length for data in batch_to_execute.values()
                )

                # Schedule the batch
                t0 = time.monotonic()
                if batch_type == BatchType.ContextEncoding:
                    self._schedule_ce(batch_to_execute)
                else:
                    assert batch_type == BatchType.TokenGeneration
                    self._schedule_tg(batch_to_execute)
                t1 = time.monotonic()
                batch_execution_time_s = t1 - t0

                # Log batch metrics
                terminated_reqs = batch_size - len(batch_to_execute)
                self._log_metrics(
                    batch_type,
                    batch_size,
                    terminated_reqs,
                    num_prompt_tokens,
                    batch_creation_time_s,
                    batch_execution_time_s,
                )

                # occasionally handle cancelled requests
                if i % 20 == 0:
                    self._handle_cancelled_requests()

            except Exception as e:
                logger.exception("An error occurred during scheduling ")
                # TODO try to recover
                raise e

    @traced
    def _handle_terminated_responses(
        self,
        batch_executed: dict[str, Any],
        batch_responses: list[dict[str, Any]],
    ):
        """Task that handles responses"""
        if batch_responses is None:
            return

        already_terminated = set()
        for batch_response in batch_responses:
            terminated = batch_executed.keys() - batch_response.keys()
            for req_id in terminated:
                if req_id in already_terminated:
                    continue

                self.pipeline.release(batch_executed[req_id])
                cache_id = batch_executed[req_id].cache_seq_id
                self.available_cache_indices.add(cache_id)
                del batch_executed[req_id]
                batch_response[req_id] = STOP_STREAM
                already_terminated.add(req_id)
                if req_id in self.active_batch:
                    del self.active_batch[req_id]

    @traced
    def _exceeds_batch_token_limit(
        self, total_seq_len: int, new_seq_len: int
    ) -> bool:
        """Check if adding a new sequence would exceed the target tokens per batch limit.

        Args:
            total_seq_len: Current total sequence length in the batch
            new_seq_len: Length of the new sequence to be added
        """
        if not self.scheduler_config.target_tokens_per_batch_ce:
            return False

        return (
            total_seq_len + new_seq_len
        ) > self.scheduler_config.target_tokens_per_batch_ce

    @traced
    def _handle_chunked_requests(
        self,
        batch_executed: dict[str, Any],
        batch_responses: list[dict[str, Any]],
    ):
        """Handle chunked resquests"""

        # Only the last request in a batch could be chunked. We discard its response
        # and put it back into the request quene if it is chunked.
        last_req = list(batch_executed.values())[-1]
        if last_req.active_idx - last_req.start_idx > 1:
            req_id, data = batch_executed.popitem()
            self.request_q.put_front_nowait((req_id, data))

            for batch_response in batch_responses:
                batch_response.pop(req_id, None)

    @traced
    def _handle_cancelled_requests(self):
        try:
            while not self.cancel_q.empty():
                try:
                    for req_id in self.cancel_q.get_nowait():
                        if req_id not in self.active_batch:
                            continue
                        self.pipeline.release(self.active_batch[req_id])
                        self.available_cache_indices.add(
                            self.active_batch[req_id].cache_seq_id
                        )
                        del self.active_batch[req_id]

                        self.response_q.put_nowait([{req_id: STOP_STREAM}])
                except queue.Empty:
                    break
        except Exception:
            logger.exception(
                "An error occurred while handling cancelled requests"
            )

    @traced
    def _schedule_ce(self, batch_to_execute):
        # we about to execute the batch, reset the CE batch timer
        self.ce_batch_start_time = None

        # execute the batch
        batch_responses = self.pipeline.next_token(
            batch_to_execute,
            num_steps=self.scheduler_config.max_forward_steps_ce,
        )
        # put the unfinished request back into the quene, and delete its responses
        self._handle_chunked_requests(batch_to_execute, batch_responses)
        # remove terminated requests from the batch
        self._handle_terminated_responses(batch_to_execute, batch_responses)
        # add the encoded requests to the continuous batch
        for req_id in batch_to_execute:
            self.active_batch[req_id] = batch_to_execute[req_id]
        # send the responses to the API process
        if any(batch_responses):
            self.response_q.put_nowait(batch_responses)

    @traced
    def _schedule_tg(self, batch_to_execute):
        # execute the batch
        batch_responses = self.pipeline.next_token(
            batch_to_execute,
            num_steps=self.scheduler_config.max_forward_steps_tg,
        )
        # remove terminated requests from the batch
        self._handle_terminated_responses(batch_to_execute, batch_responses)
        # send the responses to the API process
        self.response_q.put_nowait(batch_responses)


@dataclass
class EmbeddingsSchedulerConfig:
    """Embeddings Scheduler configuration."""

    # The maximum number of requests that can be in the encode batch.
    max_batch_size: int


class EmbeddingsScheduler(Scheduler):
    def __init__(
        self,
        process_control: ProcessControl,
        scheduler_config: EmbeddingsSchedulerConfig,
        pipeline: EmbeddingsGenerator,
        queues: Mapping[str, Queue],
    ):
        self.scheduler_config = scheduler_config
        self.pipeline = pipeline

        # Multiprocessing resources.
        self.pc = process_control

        self.request_q = queues["REQUEST"]
        self.response_q = queues["RESPONSE"]
        self.cancel_q = queues["CANCEL"]

    @traced
    def _create_batch_to_execute(self):
        max_batch_size_to_create = self.scheduler_config.max_batch_size

        batch = {}
        try:
            while max_batch_size_to_create > 0:
                req_id, data = self.request_q.get_nowait()
                batch[req_id] = data
                max_batch_size_to_create -= 1
        except queue.Empty:
            pass

        return batch

    def run(self):
        """The Scheduler loop that creates batches and schedules them on GPU"""
        i = 0
        while i % 10 or not self.pc.is_canceled():
            self.pc.beat()
            i += 1
            try:
                batch_to_execute = self._create_batch_to_execute()
                if len(batch_to_execute) == 0:
                    continue

                self._schedule_encode(batch_to_execute)
            except Exception as e:
                logger.exception("An error occurred during scheduling ")
                # TODO try to recover
                raise e

    @traced
    def _handle_terminated_responses(
        self,
        batch_executed: dict[str, Any],
        batch_response: dict[str, Any],
    ):
        """Task that handles responses"""
        already_terminated = set()
        terminated = batch_executed.keys() - batch_response.keys()
        for req_id in terminated:
            if req_id in already_terminated:
                continue
            del batch_executed[req_id]
            batch_response[req_id] = STOP_STREAM
            already_terminated.add(req_id)

    @traced
    def _schedule_encode(self, batch_to_execute):
        # execute the batch
        batch_responses = self.pipeline.encode(batch_to_execute)
        # remove terminated requests from the batch
        self._handle_terminated_responses(batch_to_execute, batch_responses)
        # send the responses to the API process
        self.response_q.put_nowait([batch_responses])
