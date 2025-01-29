# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import logging
import queue
import time
from dataclasses import dataclass
from multiprocessing import Queue
from typing import Any, Mapping, Optional, TypeVar

from max.pipelines import TokenGenerator
from max.profiler import traced
from max.serve.scheduler.process_control import ProcessControl
from max.serve.scheduler.queues import STOP_STREAM

logger = logging.getLogger(__name__)

BatchReqId = TypeVar("BatchReqId")
BatchReqInput = TypeVar("BatchReqInput")
BatchInputs = dict[BatchReqId, BatchReqInput]


@dataclass
class SchedulerConfig:
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


class SchedulerV2:
    def __init__(
        self,
        process_control: ProcessControl,
        scheduler_config: SchedulerConfig,
        pipeline: TokenGenerator,
        queues: Mapping[str, Queue],
    ):
        self.scheduler_config = scheduler_config
        self.pipeline = pipeline

        # Multiprocessing resources.
        self.pc = process_control

        self.request_q = queues["REQUEST"]
        self.response_q = queues["RESPONSE"]
        self.cancel_q = queues["CANCEL"]

        # Initialize Scheduler state.
        self.active_batch: BatchInputs = {}
        self.available_cache_indices = set(
            range(self.scheduler_config.max_batch_size_tg)
        )
        self.ce_batch_start_time: Optional[float] = None

        # TODO health check

    @traced
    def _should_schedule_ce(self):
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
    def _create_batch_to_execute(self):
        """Creates a batch to execute"""

        # if we should schedule CE then create a CE batch
        if self._should_schedule_ce():
            max_batch_size_to_create = min(
                self.scheduler_config.max_batch_size_ce,
                self.scheduler_config.max_batch_size_tg
                - len(self.active_batch),
            )
            ce_batch = self._create_ce_batch(max_batch_size_to_create)
            return ce_batch, True

        # if we execute TG then return the continuous batch
        return self.active_batch, False

    @traced
    def _create_ce_batch(self, max_batch_size_to_create: int):
        batch = {}
        sum_seq_len = 0
        try:
            while max_batch_size_to_create > 0:
                req_id, data = self.request_q.get_nowait()
                data.cache_seq_id = self.available_cache_indices.pop()
                batch[req_id] = data

                # if the batch has hit the target token budget, break early
                sum_seq_len += getattr(data, "seq_len", 0)
                if (
                    self.scheduler_config.target_tokens_per_batch_ce is not None
                    and sum_seq_len
                    > self.scheduler_config.target_tokens_per_batch_ce
                ):
                    break

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
                batch_to_execute, is_ce = self._create_batch_to_execute()
                if len(batch_to_execute) == 0:
                    continue

                if is_ce:
                    self._schedule_ce(batch_to_execute)
                else:
                    self._schedule_tg(batch_to_execute)

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

    @traced
    def _handle_cancelled_requests(self):
        try:
            while not self.cancel_q.empty():
                try:
                    req_id = self.cancel_q.get_nowait()
                    if req_id not in self.active_batch:
                        continue

                    self.pipeline.release(self.active_batch[req_id])
                    self.available_cache_indices.add(
                        self.active_batch[req_id].cache_seq_id
                    )
                    del self.active_batch[req_id]
                except queue.Empty:
                    break
        except Exception:
            logger.exception(
                "An error occurred while handling cancelled requests"
            )

    @traced
    def _schedule_ce(self, batch_to_execute):
        logger.debug(
            "Scheduling CE batch with BS: %d",
            len(batch_to_execute),
        )
        # we about to execute the batch, reset the CE batch timer
        self.ce_batch_start_time = None

        # execute the batch
        batch_responses = self.pipeline.next_token(
            batch_to_execute,
            num_steps=self.scheduler_config.max_forward_steps_ce,
        )
        # remove terminated requests from the batch
        self._handle_terminated_responses(batch_to_execute, batch_responses)
        # add the encoded requests to the continuous batch
        for req_id in batch_to_execute:
            self.active_batch[req_id] = batch_to_execute[req_id]
        # send the responses to the API process
        self.response_q.put_nowait(batch_responses)

    @traced
    def _schedule_tg(self, batch_to_execute):
        logger.debug("Scheduling TG with BS: %d", len(batch_to_execute))
        # execute the batch
        batch_responses = self.pipeline.next_token(
            batch_to_execute,
            num_steps=self.scheduler_config.max_forward_steps_tg,
        )
        # remove terminated requests from the batch
        self._handle_terminated_responses(batch_to_execute, batch_responses)
        # send the responses to the API process
        self.response_q.put_nowait(batch_responses)
