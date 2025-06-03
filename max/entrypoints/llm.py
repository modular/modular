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
import queue
import threading
import uuid
from collections.abc import Awaitable, Sequence
from queue import Queue
from threading import Thread
from typing import Callable, Optional, TypeVar, cast

import tqdm
from max.pipelines.lib import PIPELINE_REGISTRY, PipelineConfig
from max.serve.config import Settings
from max.serve.kvcache_agent.dispatcher_factory import DispatcherFactory
from max.serve.pipelines.llm import (
    TokenGeneratorPipeline,
    TokenGeneratorRequest,
    batch_config_from_pipeline_config,
)
from max.serve.pipelines.model_worker import start_model_worker
from max.serve.pipelines.telemetry_worker import start_telemetry_consumer
from max.serve.process_control import ProcessControl

T = TypeVar("T")
U = TypeVar("U")
RequestQueue = Queue[tuple[Sequence[str], Optional[int], bool]]
ResponseQueue = Queue[list[str]]


# For now, the LLM class only supports the direct token generation use case.
# Long term, there are multiple other potential use cases to support.
# This class loosely mirrors vllm.LLM for offline inference: https://docs.vllm.ai/en/stable/dev/offline_inference/llm.html
class LLM:
    """A high level interface for interacting with LLMs."""

    _pc: ProcessControl
    _async_runner: Thread
    _request_queue: RequestQueue
    _response_queue: ResponseQueue

    def __init__(self, pipeline_config: PipelineConfig) -> None:
        settings = Settings(MAX_SERVE_OFFLINE_INFERENCE=True)
        self._pc = ProcessControl(threading, "LLM")
        self._request_queue = Queue()
        self._response_queue = Queue()
        self._async_runner = Thread(
            target=_run_async_worker,
            args=(
                self._pc,
                pipeline_config,
                self._request_queue,
                self._response_queue,
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
    ) -> list[str]:
        """Generates text completions for the given prompts.

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

        self._request_queue.put((prompts, max_new_tokens, use_tqdm))
        return self._response_queue.get()


def _run_async_worker(
    pc: ProcessControl,
    pipeline_config: PipelineConfig,
    request_queue: RequestQueue,
    response_queue: ResponseQueue,
    settings: Settings,
) -> None:
    asyncio.run(
        _async_worker(
            pc, pipeline_config, request_queue, response_queue, settings
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
    request_queue: RequestQueue,
    response_queue: ResponseQueue,
    settings: Settings,
) -> None:
    tokenizer, model_factory = PIPELINE_REGISTRY.retrieve_factory(
        pipeline_config
    )
    batch_config = batch_config_from_pipeline_config(pipeline_config)
    model_name = pipeline_config.model_config.model_path
    dispatcher_factory = DispatcherFactory(settings.dispatcher_config)

    # Start the model worker process.
    # Create dynamic and continuous batching workers and associated queues
    # to feed the model worker process.
    async with (
        start_telemetry_consumer(settings) as metric_client,
        start_model_worker(
            model_factory=model_factory,
            batch_config=batch_config,
            settings=settings,
            metric_client=metric_client,
            dispatcher_factory=dispatcher_factory,
        ) as engine_queue,
        TokenGeneratorPipeline(
            model_name=model_name,
            tokenizer=tokenizer,
            engine_queue=engine_queue,
        ) as pipeline,
    ):
        pc.set_started()
        while True:
            pc.beat()
            if pc.is_canceled():
                break

            try:
                (prompts, max_new_tokens, use_tqdm) = request_queue.get(
                    timeout=0.3
                )
            except queue.Empty:
                continue

            # Lambda to do a full text generation for a request.
            async def all_tokens(prompt: str) -> str:
                request = TokenGeneratorRequest(
                    id=str(uuid.uuid4()),
                    index=0,
                    model_name=model_name,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                )

                # Generate this request until complete
                tokens = await pipeline.all_tokens(request)
                return "".join(t.decoded_token for t in tokens)

            responses = await _async_map(all_tokens, prompts, use_tqdm=use_tqdm)

            response_queue.put(responses)

        pc.set_completed()
