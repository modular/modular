# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

import logging
import time
from typing import Union

from max.interfaces import (
    Pipeline,
    SchedulerResult,
    TextGenerationInputs,
    TextGenerationOutput,
    msgpack_numpy_decoder,
    msgpack_numpy_encoder,
)
from max.nn.kv_cache import PagedKVCacheManager
from max.pipelines.core import TextAndVisionContext, TextContext
from max.pipelines.lib import PipelineConfig
from max.pipelines.lib.pipeline import get_paged_manager
from max.profiler import Tracer
from max.serve.config import Settings
from max.serve.queue.zmq_queue import ZmqPullSocket, ZmqPushSocket

from .base import Scheduler
from .text_batch_constructor import (
    SchedulerOutput,
    TextBatchConstructor,
    TokenGenerationSchedulerConfig,
)
from .utils import (
    SchedulerLogger,
    maybe_restore_chunked_request,
    release_cancelled_requests,
    release_terminated_requests,
)

logger = logging.getLogger("max.serve")


class TokenGenerationScheduler(Scheduler):
    def __init__(
        self,
        scheduler_config: TokenGenerationSchedulerConfig,
        pipeline: Pipeline[
            TextGenerationInputs[Union[TextContext, TextAndVisionContext]],
            TextGenerationOutput,
        ],
        *,
        request_zmq_endpoint: str,
        response_zmq_endpoint: str,
        cancel_zmq_endpoint: str,
        paged_manager: PagedKVCacheManager | None = None,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.pipeline = pipeline

        self.request_q = ZmqPullSocket[
            tuple[str, Union[TextContext, TextAndVisionContext]]
        ](
            zmq_endpoint=request_zmq_endpoint,
            deserialize=msgpack_numpy_decoder(
                tuple[str, Union[TextContext, TextAndVisionContext]]
            ),
        )
        self.response_q = ZmqPushSocket[
            dict[str, SchedulerResult[TextGenerationOutput]]
        ](
            zmq_endpoint=response_zmq_endpoint,
            serialize=msgpack_numpy_encoder(),
        )
        self.cancel_q = ZmqPullSocket[list[str]](
            zmq_endpoint=cancel_zmq_endpoint,
            deserialize=msgpack_numpy_decoder(list[str]),
        )

        self.batch_constructor = TextBatchConstructor(
            scheduler_config=scheduler_config,
            pipeline=pipeline,
            paged_cache=paged_manager,
        )
        self.scheduler_logger = SchedulerLogger()

    def _retrieve_pending_requests(self) -> None:
        self.batch_constructor.ce_reqs |= dict(self.request_q.drain_nowait())

    def run_iteration(self) -> None:
        """The Scheduler routine that creates batches and schedules them on GPU"""
        # Drain the request queue and add to CE requests
        self._retrieve_pending_requests()

        # Construct the batch to execute
        t0 = time.monotonic()
        batch_to_execute = self.batch_constructor.construct_batch()
        t1 = time.monotonic()
        batch_creation_time_s = t1 - t0

        # If the batch is empty, skip
        batch_size = batch_to_execute.batch_size
        if batch_size == 0:
            return

        # Schedule the batch
        t0 = time.monotonic()
        with Tracer(f"_schedule({batch_to_execute})"):
            self._schedule(batch_to_execute)
        t1 = time.monotonic()
        batch_execution_time_s = t1 - t0

        # Log batch metrics
        self.scheduler_logger.log_metrics(
            sch_config=self.scheduler_config,
            sch_output=batch_to_execute,
            paged_cache=self.batch_constructor.paged_cache,
            batch_creation_time_s=batch_creation_time_s,
            batch_execution_time_s=batch_execution_time_s,
            num_pending_reqs=len(self.batch_constructor.ce_reqs),
            total_preemption_count=self.batch_constructor.total_preemption_count,
        )

        self._handle_cancelled_requests()

    def _handle_cancelled_requests(self) -> None:
        release_cancelled_requests(
            self.cancel_q,
            self.response_q,
            self.batch_constructor.tg_reqs,
            self.pipeline,
        )

    def _schedule(self, sch_output: SchedulerOutput) -> None:
        assert sch_output.batch_size > 0
        batch_to_execute = sch_output.batch_inputs

        # execute the batch
        responses = self.pipeline.execute(
            TextGenerationInputs(
                batches=[batch_to_execute], num_steps=sch_output.num_steps
            )
        )

        # If there is a chunked request, we put it back into the request queue
        maybe_restore_chunked_request(
            batch_to_execute,
            responses,
            self.batch_constructor.ce_reqs,
        )

        # add the encoded requests to the continuous batch
        self.batch_constructor.tg_reqs |= batch_to_execute

        # remove terminated requests from the batch
        release_terminated_requests(
            sch_output,
            responses,
            self.pipeline,
            self.batch_constructor.tg_reqs,
        )

        # send the responses to the API process
        self.response_q.put_nowait(
            {
                req_id: SchedulerResult.create(response)
                for req_id, response in responses.items()
            }
        )


def load_text_generation_scheduler(
    settings: Settings,
    pipeline: Pipeline[
        TextGenerationInputs[Union[TextContext, TextAndVisionContext]],
        TextGenerationOutput,
    ],
    pipeline_config: PipelineConfig,
) -> TokenGenerationScheduler:
    # Create Scheduler Config.
    scheduler_config = TokenGenerationSchedulerConfig.from_pipeline_config(
        pipeline_config
    )

    # Retrieve Paged Manager
    paged_manager = get_paged_manager(pipeline)

    # Return Scheduler
    return TokenGenerationScheduler(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        paged_manager=paged_manager,
        request_zmq_endpoint=settings.request_zmq_endpoint,
        response_zmq_endpoint=settings.response_zmq_endpoint,
        cancel_zmq_endpoint=settings.cancel_zmq_endpoint,
    )
