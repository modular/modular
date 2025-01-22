# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
import queue
import time
from dataclasses import dataclass
from multiprocessing import Queue
from typing import Optional, Sequence, TypeVar

from max.serve.pipelines.llm import TokenGeneratorPipelineConfig

BatchReqId = TypeVar("BatchReqId")
BatchReqInput = TypeVar("BatchReqInput")
BatchInputs = dict[BatchReqId, BatchReqInput]


@dataclass
class SchedulerOutput:
    requests: BatchInputs
    num_steps: int


class Scheduler:
    """Main scheduler class responsible for tracking active batches in the
    serving layer and scheduling batches to be executed.
    """

    def __init__(self, request_q: Queue, config: TokenGeneratorPipelineConfig):
        # a queue from which we retrieve requests
        self.request_q = request_q

        # the maximum number of requests to include in a single batch
        self.tg_max_batch_size = config.token_generation.size

        # target number of tokens per batch
        self.tg_target_tokens_per_batch = (
            config.token_generation.target_sum_seq_len
        )

        # max number of multistep steps to conduct for token generation
        self.tg_max_forward_steps = config.token_generation.max_forward_steps

        if config.context_encoding:
            self.ce_max_batch_size = config.context_encoding.size
            self.ce_target_tokens_per_batch = (
                config.context_encoding.target_sum_seq_len
            )
            self.ce_max_forward_steps = (
                config.context_encoding.max_forward_steps
            )
            self.ce_timeout: Optional[float] = config.context_encoding.timeout
        else:
            self.ce_max_batch_size = self.tg_max_batch_size
            self.ce_target_tokens_per_batch = self.tg_target_tokens_per_batch
            self.ce_max_forward_steps = self.tg_max_forward_steps
            self.ce_timeout = None

        # the last time we context-encoded
        self.last_ce_time: float = time.monotonic()

        # collection of blocks that have _some_ entry in the KVCache manager
        self.active_requests: dict[int, BatchInputs] = {}

        # the set of available cache indices
        # TODO remove this and track the cache indices exclusively in the kv manager
        self.available_cache_indices = set(range(self.tg_max_batch_size))

    def _should_do_ce(self):
        if self.request_q.empty():
            # nothing to ce
            return False

        if len(self.active_requests) == 0:
            return True

        if len(self.active_requests) >= self.tg_max_batch_size:
            return False

        if self.ce_timeout is not None:
            if time.monotonic() - self.last_ce_time > self.ce_timeout:
                return True
            else:
                return False

        return True

    def schedule(self) -> SchedulerOutput:
        """Returns a prepared batch of requests to be executed."""
        # TODO remove context encoding vs token generation nuance once we have in-flight batching.
        # TODO move num_steps adjustment (e.g. trim to max_seq_len) to this
        # method instead of the token generator pipeline.
        if self._should_do_ce():
            return self._schedule_ce()

        return self._schedule_token_generation()

    def _schedule_ce(self) -> SchedulerOutput:
        sum_seq_len = 0
        requests: dict[int, BatchInputs] = {}
        max_batch_size_to_execute = min(
            self.ce_max_batch_size,
            self.tg_max_batch_size - len(self.active_requests),
        )
        while len(requests) < max_batch_size_to_execute and (
            self.ce_target_tokens_per_batch is None
            or sum_seq_len < self.ce_target_tokens_per_batch
        ):
            try:
                # try to retrieve a request from the queue
                req_id, data = self.request_q.get_nowait()
            except queue.Empty:
                break

            data.cache_seq_id = self.available_cache_indices.pop()
            requests[req_id] = data

            # we'll add new tokens for the cache for each token in the prompt
            # in addition to `num_steps - 1` new tokens while doing multistep
            # inference
            num_new_tokens = data.seq_len + self.ce_max_forward_steps - 1
            sum_seq_len += num_new_tokens

        self.last_ce_time = time.monotonic()
        return SchedulerOutput(
            requests=requests, num_steps=self.ce_max_forward_steps
        )

    def _schedule_token_generation(self) -> SchedulerOutput:
        return SchedulerOutput(
            requests=self.active_requests, num_steps=self.tg_max_forward_steps
        )

    def step(self, active_requests: dict[int, BatchInputs]):
        for req_id, request in active_requests.items():
            if req_id in self.active_requests:
                continue

            self.active_requests[req_id] = request

    def contains(self, req_id: int) -> bool:
        return req_id in self.active_requests

    def get_request(self, req_id: int) -> BatchInputs:
        return self.active_requests[req_id]

    def release(self, req_ids: Sequence[int]) -> None:
        """Releases the cache indices for the given requests."""
        for req_id in req_ids:
            if req_id not in self.active_requests:
                raise ValueError(
                    f"Request {req_id} not found in active requests"
                )

            request = self.active_requests[req_id]
            self.available_cache_indices.add(request.cache_seq_id)  # type: ignore
            del self.active_requests[req_id]
