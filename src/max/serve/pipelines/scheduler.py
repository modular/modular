# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import logging
import queue
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional

from max.pipelines import TokenGenerator
from max.pipelines.context import InputContext
from max.pipelines.interfaces import (
    EmbeddingsGenerator,
    TextGenerationResponse,
    TextResponse,
)
from max.pipelines.kv_cache.paged_cache import PagedKVCacheManager
from max.profiler import traced
from max.serve.scheduler.max_queue import MaxQueue
from max.serve.scheduler.process_control import ProcessControl
from max.serve.scheduler.queues import STOP_STREAM
from max.serve.telemetry.metrics import METRICS
from max.support.human_readable_formatter import to_human_readable_latency

logger = logging.getLogger("max.serve")


# TODO: This will be deleted after E2EOPT-113
def _trim_prompt(data: InputContext, new_length: int) -> None:
    untrimmed_length = data.active_length
    trimmed_length = new_length
    bump_length = untrimmed_length - trimmed_length
    assert bump_length >= 0
    if bump_length > 0:
        data.bump_token_indices(start_idx=bump_length)


class BatchType(Enum):
    ContextEncoding = 1
    TokenGeneration = 2

    def concise_name(self):
        if self == BatchType.ContextEncoding:
            return "CE"
        else:
            assert self == BatchType.TokenGeneration
            return "TG"


class SchedulerOutput:
    def __init__(
        self,
        batch_type: BatchType = BatchType.TokenGeneration,
        num_steps: int = 1,
        batch_inputs: dict[str, InputContext] = {},
        prompt_tokens: Optional[int] = None,
        tokens_to_encode: Optional[int] = None,
    ):
        self.batch_type = batch_type
        self.num_steps = num_steps
        self.batch_inputs = batch_inputs
        self.batch_size = len(batch_inputs)
        self.prompt_tokens = (
            prompt_tokens if prompt_tokens is not None else len(batch_inputs)
        )
        self.uncached_prompt_tokens = (
            tokens_to_encode
            if tokens_to_encode is not None
            else len(batch_inputs)
        )
        self.cached_prompt_tokens = (
            self.prompt_tokens - self.uncached_prompt_tokens
        )

    @property
    def tokens_to_encode(self) -> int:
        return self.uncached_prompt_tokens

    @property
    def cache_hit_rate(self) -> float:
        if self.prompt_tokens == 0:
            return 0.0
        return self.cached_prompt_tokens / self.prompt_tokens

    @property
    def num_terminated(self) -> int:
        # this is the difference between the number of request in the batch before
        # and after the batch was scheduled.
        return self.batch_size - len(self.batch_inputs)

    def __repr__(self) -> str:
        return (
            f"SchedulerOutput("
            f"batch_type={self.batch_type.concise_name()}, "
            f"batch_size={self.batch_size}, "
            f"tokens_to_encode={self.tokens_to_encode}, "
            f"cache_hit_rate={self.cache_hit_rate})"
        )


class RequestDeque(MaxQueue):
    """A wrapper around the multiprocessing queue that allows us to add
    requests to the front of the queue.
    """

    def __init__(self, queue: MaxQueue):
        self.queue = queue
        self.preempted: list[tuple[str, InputContext]] = []

    def put_front_nowait(self, item: tuple[str, InputContext]):
        """A new method that allows us to add requests to the front of the queue."""
        self.preempted.append(item)

    def put(self, *args, **kwargs) -> None:
        return self.queue.put(*args, **kwargs)

    def put_nowait(self, item: tuple[str, InputContext]) -> None:
        return self.queue.put_nowait(item)

    def get(self, *args, **kwargs) -> tuple[str, InputContext]:
        if self.preempted:
            return self.preempted.pop()
        return self.queue.get(*args, **kwargs)

    @traced
    def get_nowait(self) -> tuple[str, InputContext]:
        if self.preempted:
            return self.preempted.pop()
        return self.queue.get_nowait()

    def qsize(self) -> int:
        return len(self.preempted) + self.queue.qsize()

    def empty(self) -> bool:
        return self.qsize() == 0


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

        if self.max_forward_steps_ce > 1:
            self.max_forward_steps_ce = 1
            logger.info(
                "Prefill does not support multistep inference, overriding max_forward_steps_ce to 1."
            )
            self.max_forward_steps_ce = 1

        if self.enable_in_flight_batching and not self.enable_chunked_prefill:
            msg = "Requires chunked prefill for in-flight batching."
            raise ValueError(msg)


class TokenGenerationScheduler(Scheduler):
    def __init__(
        self,
        process_control: ProcessControl,
        scheduler_config: TokenGenerationSchedulerConfig,
        pipeline: TokenGenerator,
        queues: Mapping[str, MaxQueue],
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
        self.active_batch: dict[str, InputContext] = {}
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

        # If there are less than 10% free blocks, prioritize TG over CE
        if (
            self.paged_manager is not None
            and self.paged_manager.free_blocks_pct < 0.1
        ):
            return False

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
                messages_needed = self.scheduler_config.max_batch_size_tg - len(
                    self.active_batch
                )
                if self.request_q.qsize() >= messages_needed:
                    # If there are enough request to fill the TG batch then schedule CE
                    return True
                else:
                    # If not enough requests then hold off the CE and continue with TG
                    return False

        return True

    @traced
    def _maybe_chunk_prefill_request(
        self, data: InputContext, tot_tokens_to_encode: int
    ) -> int:
        """Chunks a prefill request if it exceeds the target tokens per batch."""
        if not (
            self.scheduler_config.enable_chunked_prefill
            and self.scheduler_config.target_tokens_per_batch_ce is not None
        ):
            return 0

        tokens_to_encode = data.active_length
        if (
            tot_tokens_to_encode + tokens_to_encode
            <= self.scheduler_config.target_tokens_per_batch_ce
        ):
            return 0

        # We can only schedule part of the prompt.
        # We achieve this by decreasing the active_idx of the context class.
        token_num_diff = (
            tot_tokens_to_encode
            + tokens_to_encode
            - self.scheduler_config.target_tokens_per_batch_ce
        )
        tokens_to_encode -= token_num_diff
        assert tokens_to_encode > 0
        assert token_num_diff > 0
        data.bump_token_indices(active_idx=-token_num_diff)
        return token_num_diff

    @traced
    def _try_create_ce_batch(self) -> SchedulerOutput:
        """Try to create a context encoding batch"""
        max_batch_size_to_create = min(
            self.scheduler_config.max_batch_size_ce,
            self.scheduler_config.max_batch_size_tg - len(self.active_batch),
        )

        # If there are already active TG requests, we want to be more conservative
        # about scheduling new CE requests if we are constrained by KV Cache space.
        num_steps_with_headroom = self.scheduler_config.max_forward_steps_ce
        if len(self.active_batch) > 0:
            # TODO: E2EOPT-77. we should look at what vLLM does and decide a more intelligent
            # policy for this.
            # This hardcoded value was chosen empirically to prevent excessive
            # preemption on sharegpt.
            headroom_for_num_tg_iterations = 5
            num_steps_with_headroom += (
                headroom_for_num_tg_iterations
                * self.scheduler_config.max_forward_steps_tg
            )

        ce_batch: dict[str, InputContext] = {}
        tot_tokens_to_encode = 0
        tot_prompt_tokens = 0

        if self.scheduler_config.enable_in_flight_batching:
            if self.active_batch:
                tg_batch = self._create_tg_batch()
                ce_batch = tg_batch.batch_inputs
                tot_tokens_to_encode = tg_batch.tokens_to_encode
                tot_prompt_tokens = tg_batch.prompt_tokens
            for data in ce_batch.values():
                # active length should be 1 for TG requests
                assert data.active_length == 1

        for _ in range(max_batch_size_to_create):
            if (
                self.scheduler_config.target_tokens_per_batch_ce is not None
                and tot_tokens_to_encode
                >= self.scheduler_config.target_tokens_per_batch_ce
            ):
                break

            try:
                req_id, data = self.request_q.get_nowait()
                # Unfortunately, when we create a new context we set the cache_seq_id
                # to be the req idx in tokenizer.py. We probably should not do
                # this. (TODO: E2EOPT-138)
                #
                # We want to ignore the existing cache_seq_id, UNLESS this request
                # is a partially encoded request due to chunked prefill.
                if data.start_idx == 0:
                    data.unassign_from_cache()
                # Lets assign a new cache slot to this request if it doesn't have one yet.
                if not data.is_assigned_to_cache:
                    data.assign_to_cache(self.available_cache_indices.pop())
                    if self.paged_manager is not None:
                        self.paged_manager.external_claim([data.cache_seq_id])
            except queue.Empty:
                break

            orig_prompt_length = data.active_length
            cache_seq_id = data.cache_seq_id
            num_steps = self.scheduler_config.max_forward_steps_ce

            if self.paged_manager is not None:
                max_seq_len = self.paged_manager.max_seq_len
                num_available_steps = data.compute_num_available_steps(
                    max_seq_len
                )
                num_steps = min(num_steps, num_available_steps)
                prompt = data.next_tokens

                # Lookup blocks to reuse from prefix cache
                prompt = self.paged_manager.reuse_blocks_from_prefix_cache(
                    cache_seq_id, prompt, num_steps=num_steps
                )
                _trim_prompt(data, len(prompt))

            # Chunk the request if it exceeds the token budget
            tokens_trimmed = self._maybe_chunk_prefill_request(
                data, tot_tokens_to_encode
            )
            orig_prompt_length -= tokens_trimmed
            prompt = data.next_tokens

            if self.paged_manager is not None:
                # Allocate new blocks for shortened prompt
                scheduled = self.paged_manager.allocate_new_blocks(
                    cache_seq_id, prompt, num_steps=num_steps
                )

                # We were able to schedule this request
                if not scheduled:
                    self.available_cache_indices.add(data.cache_seq_id)
                    self.pipeline.release(data)
                    data.reset()
                    self.request_q.put_front_nowait((req_id, data))
                    break

            # Schedule the requests as it fits in KVCache and token limit
            tot_tokens_to_encode += data.active_length
            tot_prompt_tokens += orig_prompt_length
            ce_batch[req_id] = data

        return SchedulerOutput(
            batch_type=BatchType.ContextEncoding,
            batch_inputs=ce_batch,
            prompt_tokens=tot_prompt_tokens,
            tokens_to_encode=tot_tokens_to_encode,
            num_steps=self.scheduler_config.max_forward_steps_ce,
        )

    @traced
    def _preempt_request(self, req_id: Any, data: InputContext):
        """Preempts the most recently received request from active batch"""
        # dicts in python pop the last inserted item
        # this corresponds to the most recently received request
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
    def _create_tg_batch(self) -> SchedulerOutput:
        """Creates a non empty token generation batch"""
        assert self.active_batch

        # If we are not using paged attention, we can always schedule the active
        # batch since we reserved blocks for all active requests previously
        if self.paged_manager is None:
            return SchedulerOutput(
                batch_type=BatchType.TokenGeneration,
                batch_inputs=self.active_batch.copy(),
                num_steps=self.scheduler_config.max_forward_steps_tg,
            )

        num_steps = self.scheduler_config.max_forward_steps_tg
        max_seq_len = self.paged_manager.max_seq_len

        # Assume this is sorted by request arrival time where the leftmost request
        # is the oldest and the rightmost request is the newest.
        candidate_reqs = deque(
            (req_id, data) for req_id, data in self.active_batch.items()
        )
        _, first_req_data = candidate_reqs[0]
        self.active_batch.clear()
        while len(candidate_reqs) > 0:
            # Get the oldest request
            req_id, data = candidate_reqs.popleft()
            cache_seq_id = data.cache_seq_id
            prompt = data.next_tokens

            # Determine the number of steps to schedule based on the max_seq_len
            # of the pipeline model.
            num_available_steps = data.compute_num_available_steps(max_seq_len)
            num_steps = min(num_steps, num_available_steps)

            scheduled = False
            while not scheduled:
                # If this is the only request, we should not exceed the max_length
                # specified in its request parameter.
                if (
                    len(self.active_batch) == 0
                    and len(candidate_reqs) == 0
                    and data.max_length is not None
                ):
                    num_available_steps = data.compute_num_available_steps(
                        data.max_length
                    )
                    num_steps = min(num_steps, num_available_steps)

                # Attempt to schedule this request
                prompt = self.paged_manager.reuse_blocks_from_prefix_cache(
                    cache_seq_id, prompt, num_steps=num_steps
                )
                _trim_prompt(data, len(prompt))
                scheduled = self.paged_manager.allocate_new_blocks(
                    cache_seq_id, prompt, num_steps=num_steps
                )

                # We were able to schedule this request
                if scheduled:
                    break

                # We were not able to schedule this request but there is nothing
                # to preempt
                if len(candidate_reqs) == 0:
                    break

                # We were unable to schedule this request so we will try again
                # after preempting the newest request
                req_id_preempt, data_preempt = candidate_reqs.pop()
                self._preempt_request(req_id_preempt, data_preempt)

                # Reset the prompt in case it was trimmed
                prompt = data.next_tokens

            # If we still can't schedule the request, we preempt it
            if not scheduled:
                self._preempt_request(req_id, data)
                break

            # Add the request to the batch
            self.active_batch[req_id] = data

        # We successfully created a TG batch
        if len(self.active_batch) > 0:
            return SchedulerOutput(
                batch_type=BatchType.TokenGeneration,
                batch_inputs=self.active_batch.copy(),
                num_steps=num_steps,
            )

        # We have utterly failed to construct a TG batch.
        # This should literally never happen unless the user sets an absurdly
        # large max seq len or the KV cache is very small.
        current_len = first_req_data.current_length
        page_size = self.paged_manager.page_size
        total_num_blocks = self.paged_manager.total_num_pages
        max_seq_len = total_num_blocks * page_size
        msg = (
            f"Insufficient KV pages to run token generation on a single request with {current_len} tokens.\n"
            f"The KVCache has {total_num_blocks} pages with page size {page_size}. This is only enough to support {max_seq_len} tokens.\n"
            "You must restart your process and set a lower max seq len to prevent a single request from using the entire KV cache."
        )
        raise RuntimeError(msg)

    def _create_batch_to_execute(
        self,
    ) -> SchedulerOutput:
        """Creates a batch to execute"""
        if self._should_schedule_ce():
            ce_batch = self._try_create_ce_batch()
            if ce_batch.batch_size > 0:
                return ce_batch
            # failed to create a CE batch, try to create a TG batch instead

        # if there are no active requests, we can't create a TG batch
        if not self.active_batch:
            return SchedulerOutput(
                batch_type=BatchType.TokenGeneration,
                batch_inputs={},
                num_steps=0,
            )

        tg_batch = self._create_tg_batch()
        return tg_batch

    def _log_metrics(
        self,
        sch_output: SchedulerOutput,
        batch_creation_time_s: float,
        batch_execution_time_s: float,
    ) -> None:
        batch_size = sch_output.batch_size
        batch_type = sch_output.batch_type
        assert batch_size > 0
        terminated_reqs = sch_output.num_terminated
        num_steps = (
            self.scheduler_config.max_forward_steps_ce
            if batch_type == BatchType.ContextEncoding
            else self.scheduler_config.max_forward_steps_tg
        )
        num_generated_tokens = batch_size * num_steps

        # Number of pending requests is unknown if qsize is not supported
        pending_reqs = self.request_q.qsize()

        def to_human_readable_throughput(tps: float) -> str:
            if tps >= 1_000:
                return f"{tps / 1e3:.1f}K tok/s"
            return f"{tps:.1f} tok/s"

        # Format latency and throughput metrics
        num_prompt_tokens = sch_output.prompt_tokens
        prompt_throughput_str = to_human_readable_throughput(
            num_prompt_tokens / batch_execution_time_s
        )
        generation_throughput_str = to_human_readable_throughput(
            num_generated_tokens / batch_execution_time_s
        )
        batch_creation_latency_str = to_human_readable_latency(
            batch_creation_time_s
        )
        batch_execution_latency_str = to_human_readable_latency(
            batch_execution_time_s
        )

        # Prompt cache hit info
        target_tokens = (
            self.scheduler_config.target_tokens_per_batch_ce
            if batch_type == BatchType.ContextEncoding
            else self.scheduler_config.target_tokens_per_batch_tg
        )
        target_tokens_str = f"{target_tokens}" if target_tokens else "INF"
        prompt_tokens = sch_output.prompt_tokens
        uncached_prompt_tokens = sch_output.uncached_prompt_tokens

        if self.paged_manager is None:
            assert prompt_tokens == uncached_prompt_tokens
            logger.debug(
                f"Executed {batch_type.concise_name()} batch with {batch_size} reqs | "
                f"Terminated: {terminated_reqs} reqs, "
                f"Pending: {pending_reqs} reqs | "
                f"Target: {prompt_tokens}/{target_tokens_str} toks | "
                f"Prompt Tput: {prompt_throughput_str}, "
                f"Generation Tput: {generation_throughput_str} | "
                f"Batch creation: {batch_creation_latency_str}, "
                f"Execution: {batch_execution_latency_str}"
            )
            return

        # KVCache specific metrics
        used_pct = self.paged_manager.used_blocks_pct
        cache_hit_rate = sch_output.cache_hit_rate
        total_blocks = self.paged_manager.total_num_pages

        cow_str = ""
        if self.paged_manager.enable_prefix_caching:
            cow_blocks_copied = self.paged_manager.cow_blocks_copied
            self.paged_manager.reset_cow_blocks_copied()
            cow_str = f"COW: {cow_blocks_copied} blocks copied, "

        logger.debug(
            f"Executed {batch_type.concise_name()} batch with {batch_size} reqs | "
            f"Terminated: {terminated_reqs} reqs, "
            f"Pending: {pending_reqs} reqs | "
            f"Target: {uncached_prompt_tokens}/{target_tokens_str} toks | "
            f"Prompt Tput: {prompt_throughput_str}, "
            f"Generation Tput: {generation_throughput_str} | "
            f"Batch creation: {batch_creation_latency_str}, "
            f"Execution: {batch_execution_latency_str} | "
            f"KVCache usage: {used_pct:.1%} of {total_blocks} blocks, "
            f"Cache hit rate: {cache_hit_rate:.1%} of {prompt_tokens} toks, "
            f"{cow_str}"
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
                batch_to_execute = self._create_batch_to_execute()
                t1 = time.monotonic()
                batch_creation_time_s = t1 - t0

                # If the batch is empty, skip
                batch_size = batch_to_execute.batch_size
                if batch_size == 0:
                    continue

                # Schedule the batch
                t0 = time.monotonic()
                self._schedule(batch_to_execute)
                t1 = time.monotonic()
                batch_execution_time_s = t1 - t0

                # Log batch metrics
                self._log_metrics(
                    batch_to_execute,
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
        batch_responses: dict[str, TextGenerationResponse],
    ):
        """Task that handles responses"""
        if batch_responses is None:
            return

        for request_id, response in batch_responses.items():
            if response.is_done:
                # Release from cache
                cache_id = batch_executed[request_id].cache_seq_id
                self.pipeline.release(batch_executed[request_id])
                self.available_cache_indices.add(cache_id)
                del batch_executed[request_id]

                # Remove from active batch
                if request_id in self.active_batch:
                    del self.active_batch[request_id]

    @traced
    def _handle_chunked_requests(
        self,
        batch_executed: dict[str, Any],
        batch_responses: dict[str, TextGenerationResponse],
    ):
        """Handle chunked requests"""

        # Only the last request in a batch could be chunked. We discard its response
        # and put it back into the request queue if it is chunked.
        last_req = list(batch_executed.values())[-1]
        if last_req.active_idx - last_req.start_idx > 1:
            req_id, data = batch_executed.popitem()
            self.request_q.put_front_nowait((req_id, data))

            batch_responses.pop(req_id)

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
    def _schedule_ce(self, sch_output: SchedulerOutput):
        batch_to_execute = sch_output.batch_inputs

        # we about to execute the batch, reset the CE batch timer
        self.ce_batch_start_time = None

        # execute the batch
        batch_responses = self.pipeline.next_token(
            batch_to_execute,
            num_steps=sch_output.num_steps,
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
            # Convert this to list[dict[str, Any]]
            responses: list[dict[str, TextResponse]] = [{}]
            for request_id, response in batch_responses.items():
                # This will just ensure that there is always a response for each token
                # We add one here, as we need to send a stop sentinel
                while (
                    len(response.tokens) + (1 if response.is_done else 0)
                ) > len(responses):
                    responses.append({})

                for token_idx, text_response in enumerate(response.tokens):
                    responses[token_idx][request_id] = text_response

                if response.is_done:
                    responses[len(response.tokens)][request_id] = STOP_STREAM

            self.response_q.put_nowait(responses)

    @traced
    def _schedule_tg(self, sch_output: SchedulerOutput):
        batch_to_execute = sch_output.batch_inputs

        METRICS.batch_size(len(batch_to_execute))
        # execute the batch
        batch_responses = self.pipeline.next_token(
            batch_to_execute,
            num_steps=sch_output.num_steps,
        )
        # remove terminated requests from the batch
        self._handle_terminated_responses(batch_to_execute, batch_responses)
        # send the responses to the API process
        responses: list[dict[str, TextResponse]] = [{}]
        for request_id, response in batch_responses.items():
            while (len(response.tokens) + (1 if response.is_done else 0)) > len(
                responses
            ):
                responses.append({})

            for token_idx, text_response in enumerate(response.tokens):
                responses[token_idx][request_id] = text_response

            if response.is_done:
                responses[len(response.tokens)][request_id] = STOP_STREAM

        self.response_q.put_nowait(responses)

    def _schedule(self, sch_output: SchedulerOutput):
        assert sch_output.batch_size > 0
        if sch_output.batch_type == BatchType.ContextEncoding:
            self._schedule_ce(sch_output)
        else:
            assert sch_output.batch_type == BatchType.TokenGeneration
            self._schedule_tg(sch_output)


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
        queues: Mapping[str, MaxQueue],
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
