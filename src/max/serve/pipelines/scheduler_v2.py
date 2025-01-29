# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import logging
import math
import queue
import time
from multiprocessing import Queue
from multiprocessing.synchronize import Event
from typing import Any, Mapping, Optional, TypeVar

from max.pipelines import PipelinesFactory, TokenGenerator
from max.profiler import Tracer, traced
from max.serve.pipelines.llm import TokenGeneratorPipelineConfig
from max.serve.scheduler.queues import STOP_STREAM
from max.serve.telemetry.metrics import METRICS
from max.serve.telemetry.stopwatch import record_ms

logger = logging.getLogger(__name__)

BatchReqId = TypeVar("BatchReqId")
BatchReqInput = TypeVar("BatchReqInput")
BatchInputs = dict[BatchReqId, BatchReqInput]


class SchedulerV2:
    def __init__(
        self,
        model_factory: PipelinesFactory,
        pipeline_config: TokenGeneratorPipelineConfig,
        queues: Mapping[str, Queue],
        events: Mapping[str, Event],
    ):
        # Multiprocessing resources.
        self.started = events["STARTED"]
        self.stopped = events["STOPPED"]
        self.shutdown = events["SHUTDOWN"]

        self.request_q = queues["REQUEST"]
        self.response_q = queues["RESPONSE"]
        self.cancel_q = queues["CANCEL"]

        # Initialize token generator.
        with record_ms(METRICS.model_load_time), Tracer("model_factory"):
            model = model_factory()
            assert isinstance(model, TokenGenerator)
            self.model: TokenGenerator = model
        config = pipeline_config
        self.max_batch_size_tg = config.token_generation.size
        self.max_forward_steps_tg = config.token_generation.max_forward_steps
        self.target_tokens_per_batch_tg = (
            config.token_generation.target_sum_seq_len
        )
        if config.context_encoding:
            self.max_batch_size_ce = config.context_encoding.size
            self.max_forward_steps_ce = (
                config.context_encoding.max_forward_steps
            )
            self.target_tokens_per_batch_ce = (
                config.context_encoding.target_sum_seq_len
            )
            if math.isclose(config.context_encoding.timeout, 0.0):
                self.batch_timeout = None
            else:
                self.batch_timeout = config.context_encoding.timeout
        else:
            self.max_batch_size_ce = self.max_batch_size_tg
            self.max_forward_steps_ce = self.max_forward_steps_tg
            self.target_tokens_per_batch_ce = self.target_tokens_per_batch_tg
            self.batch_timeout = None
        logger.info("Token generators loaded!")

        # Initialize Scheduler state.
        self.active_batch: BatchInputs = {}
        self.available_cache_indices = set(range(self.max_batch_size_tg))
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
        if len(self.active_batch) >= self.max_batch_size_tg:
            return False

        # If TG batch is empty then schedule CE
        if len(self.active_batch) == 0:
            return True

        # If batch timeout is set
        if self.batch_timeout:
            # If batch timeout is reached then schedule CE
            if (
                self.ce_batch_start_time is not None
                and time.monotonic()
                >= self.ce_batch_start_time + self.batch_timeout
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
                self.max_batch_size_ce,
                self.max_batch_size_tg - len(self.active_batch),
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
                    self.target_tokens_per_batch_ce is not None
                    and sum_seq_len > self.target_tokens_per_batch_ce
                ):
                    break

                max_batch_size_to_create -= 1
        except queue.Empty:
            pass

        return batch

    def run(self):
        """The Scheduler loop that creates batches and schedules them on GPU"""
        i = 0
        while i % 10 or not self.shutdown.is_set():
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

                self.model.release(batch_executed[req_id])
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

                    self.model.release(self.active_batch[req_id])
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
        batch_responses = self.model.next_token(
            batch_to_execute,
            num_steps=self.max_forward_steps_ce,
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
        batch_responses = self.model.next_token(
            batch_to_execute,
            num_steps=self.max_forward_steps_tg,
        )
        # remove terminated requests from the batch
        self._handle_terminated_responses(batch_to_execute, batch_responses)
        # send the responses to the API process
        self.response_q.put_nowait(batch_responses)
