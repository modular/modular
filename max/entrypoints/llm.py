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

"""A high level interface for interacting with LLMs built from MAX pipelines"""

from __future__ import annotations

import asyncio
import dataclasses
import queue
import threading
from collections.abc import Awaitable, Callable, Mapping, Sequence
from threading import Thread
from typing import TypeVar, cast

import tqdm
from max.interfaces import RequestID, SamplingParams, TextGenerationRequest
from max.pipelines.lib import PIPELINE_REGISTRY, PipelineConfig
from max.serve.config import Settings
from max.serve.pipelines.llm import TokenGeneratorPipeline
from max.serve.pipelines.model_worker import start_model_worker
from max.serve.pipelines.telemetry_worker import start_telemetry_consumer
from max.serve.process_control import ProcessControl
from max.serve.queue.lora_queue import LoRAQueue
from max.serve.scheduler.queues import SchedulerZmqConfigs

T = TypeVar("T")
U = TypeVar("U")


@dataclasses.dataclass
class _Request:
    id: RequestID
    prompts: Sequence[str]
    max_new_tokens: int | None
    use_tqdm: bool


@dataclasses.dataclass
class _Response:
    complete_texts: Sequence[str]


# For now, the LLM class only supports the direct token generation use case.
# Long term, there are multiple other potential use cases to support.
# This class loosely mirrors vllm.LLM for offline inference: https://docs.vllm.ai/en/stable/dev/offline_inference/llm.html
class LLM:
    """A high level interface for interacting with LLMs."""

    _pc: ProcessControl
    _async_runner: Thread
    _request_queue: queue.Queue[_Request]
    _pending_requests: dict[RequestID, queue.Queue[_Response]]

    def __init__(self, pipeline_config: PipelineConfig) -> None:
        settings = Settings(MAX_SERVE_OFFLINE_INFERENCE=True)
        self._pc = ProcessControl(threading, "LLM")
        self._request_queue = queue.Queue()
        self._pending_requests = {}
        self._async_runner = Thread(
            target=_run_async_worker,
            args=(
                self._pc,
                pipeline_config,
                self._request_queue,
                self._pending_requests,
                settings,
            ),
        )
        self._async_runner.start()
        # TODO: set a timeout on wait
        self._pc.started_event.wait()

    def __del__(self) -> None:
        self._pc.set_canceled()
        self._async_runner.join()

    def generate(
        self,
        prompts: str | Sequence[str],
        max_new_tokens: int | None = 100,
        use_tqdm: bool = True,
    ) -> Sequence[str]:
        """Generates text completions for the given prompts.

        This method is thread safe and may be used on the same LLM instance
        from multiple threads concurrently with no external synchronization.

        Args:
            prompts: The input string or list of strings to generate completions for.
            max_new_tokens: The maximum number of tokens to generate in the response.
            use_tqdm: Whether to display a progress bar during generation.

        Returns:
            A list of generated text completions corresponding to each input prompt.

        Raises:
            ValueError: If prompts is empty or contains invalid data.
            RuntimeError: If the model fails to generate completions.
        """
        if isinstance(prompts, str):
            # Handle the edge case where the user passes in a single string
            prompts = (prompts,)

        request = _Request(
            id=RequestID(),
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            use_tqdm=use_tqdm,
        )
        response_queue: queue.Queue[_Response] = queue.Queue()
        self._pending_requests[request.id] = response_queue

        try:
            self._request_queue.put_nowait(request)
            return response_queue.get().complete_texts
        finally:
            # Clean up the pending request mapping
            self._pending_requests.pop(request.id, None)


def _run_async_worker(
    pc: ProcessControl,
    pipeline_config: PipelineConfig,
    request_queue: queue.Queue[_Request],
    pending_requests: Mapping[RequestID, queue.Queue[_Response]],
    settings: Settings,
) -> None:
    asyncio.run(
        _async_worker(
            pc,
            pipeline_config,
            request_queue,
            pending_requests,
            settings,
        )
    )


async def _async_map(
    f: Callable[[T], Awaitable[U]],
    seq: Sequence[T],
    /,
    *,
    use_tqdm: bool = False,
) -> list[U]:
    outputs: list[U | None] = [None] * len(seq)

    async def task_wrapper(i: int) -> None:
        outputs[i] = await f(seq[i])
        if use_tqdm:
            pbar.update(1)

    if use_tqdm:
        with tqdm.tqdm(total=len(seq)) as pbar:
            await asyncio.gather(*map(task_wrapper, range(len(seq))))
    else:
        await asyncio.gather(*map(task_wrapper, range(len(seq))))
    return cast("list[U]", outputs)


async def _async_worker(
    pc: ProcessControl,
    pipeline_config: PipelineConfig,
    request_queue: queue.Queue[_Request],
    pending_requests: Mapping[RequestID, queue.Queue[_Response]],
    settings: Settings,
) -> None:
    tokenizer, model_factory = PIPELINE_REGISTRY.retrieve_factory(
        pipeline_config
    )
    model_name = pipeline_config.model_config.model_path

    # Start the model worker process.
    # Create dynamic and continuous batching workers and associated queues
    # to feed the model worker process.
    pipeline_task = PIPELINE_REGISTRY.retrieve_pipeline_task(pipeline_config)
    lora_queue: LoRAQueue | None = (
        LoRAQueue(pipeline_config.zmq_endpoint_base)
        if pipeline_config.lora_config
        else None
    )
    # Create Queues
    scheduler_zmq_configs = SchedulerZmqConfigs(pipeline_task)
    async with (
        start_telemetry_consumer(settings) as metric_client,
        start_model_worker(
            model_factory=model_factory,  # type: ignore
            pipeline_config=pipeline_config,
            settings=settings,
            metric_client=metric_client,
            scheduler_zmq_configs=scheduler_zmq_configs,
        ) as worker_monitor,
        TokenGeneratorPipeline(
            model_name=model_name,
            tokenizer=tokenizer,
            lora_queue=lora_queue,
            worker_monitor=worker_monitor,
            scheduler_zmq_configs=scheduler_zmq_configs,
        ) as pipeline,
    ):
        pc.set_started()
        while True:
            pc.beat()
            if pc.is_canceled():
                break
            try:
                request = request_queue.get(timeout=0.3)
            except queue.Empty:
                continue

            # Lambda to do a full text generation for a request.
            async def all_tokens(prompt: str) -> str:
                sampling_params = SamplingParams(
                    max_new_tokens=request.max_new_tokens  # noqa: B023
                )
                gen_request = TextGenerationRequest(
                    request_id=RequestID(),
                    model_name=model_name,
                    prompt=prompt,
                    sampling_params=sampling_params,
                )

                # Generate this request until complete
                tokens = await pipeline.all_tokens(gen_request)
                return "".join(
                    t.decoded_token if t.decoded_token is not None else ""
                    for t in tokens
                )

            responses = await _async_map(
                all_tokens, request.prompts, use_tqdm=request.use_tqdm
            )

            # Put the response in the specific queue for this request ID
            if response_queue := pending_requests.get(request.id):
                response_queue.put(_Response(complete_texts=responses))

        pc.set_completed()
