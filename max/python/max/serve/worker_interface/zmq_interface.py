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

import logging
from typing import Any

from max.interfaces import (
    BaseContext,
    EmbeddingsContext,
    PipelineOutput,
    PipelineTask,
    RequestID,
    SchedulerResult,
    TextGenerationContext,
)
from max.interfaces.queue import MAXPullQueue, MAXPushQueue
from max.serve.worker_interface.worker_interface import ModelWorkerInterface
from max.serve.worker_interface.zmq_queue import ZmqConfig

logger = logging.getLogger("max.serve")


class ZmqModelWorkerInterface(ModelWorkerInterface):
    def __init__(
        self,
        pipeline_task: PipelineTask,
        context_type: type[TextGenerationContext] | type[EmbeddingsContext],
    ) -> None:
        response_type = pipeline_task.output_type

        self.request_queue_config = ZmqConfig[BaseContext](context_type)
        self.response_queue_config = ZmqConfig[
            dict[RequestID, SchedulerResult[PipelineOutput]]
        ](response_type)
        self.cancel_queue_config = ZmqConfig[list[RequestID]](list[RequestID])

    def api_worker_queues(
        self,
    ) -> tuple[
        MAXPushQueue[Any],
        MAXPullQueue[dict[RequestID, SchedulerResult[Any]]],
        MAXPushQueue[list[RequestID]],
    ]:
        return (
            self.request_queue_config.push(),
            self.response_queue_config.pull(),
            self.cancel_queue_config.push(),
        )

    def model_worker_queues(
        self,
    ) -> tuple[
        MAXPullQueue[Any],
        MAXPushQueue[dict[RequestID, SchedulerResult[Any]]],
        MAXPullQueue[list[RequestID]],
    ]:
        return (
            self.request_queue_config.pull(),
            self.response_queue_config.push(),
            self.cancel_queue_config.pull(),
        )
