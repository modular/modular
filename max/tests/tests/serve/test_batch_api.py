# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import asyncio
import multiprocessing

import pytest
from max.serve.pipelines.echo_gen import (
    EchoPipelineTokenizer,
    EchoTokenGenerator,
)
from max.serve.pipelines.llm import (
    TokenGeneratorPipeline,
    TokenGeneratorPipelineConfig,
)
from max.serve.scheduler.process_control import ProcessControl
from max.serve.scheduler.queues import EngineQueue


@pytest.fixture(params=[4, 8, 16, 32])
def num_requests(request):
    return request.param


@pytest.mark.skip(reason="TODO(ylou): Fix this after submitting!!!")
@pytest.mark.asyncio
async def test_batched_requests_pipeline(num_requests):
    config = TokenGeneratorPipelineConfig.dynamic_homogenous(batch_size=1)

    # Submit num_requests to the pipeline which will batch and execute them.
    # Verify results afterwards.
    # This matches vLLM's benchmark_throughput method
    ctx = multiprocessing.get_context("spawn")
    pc = ProcessControl(ctx, "test")
    async with TokenGeneratorPipeline(
        "echo", EchoPipelineTokenizer(), EngineQueue(ctx, pc)
    ) as pipeline:
        request_params = []
        request_tasks = []

        echo_gen = EchoTokenGenerator()

        async def _batch_execute(batch):
            return echo_gen.next_token(batch)[0]

        # model_tasks = start_model_testing_tasks(
        #     pipeline.token_gen_queue, _batch_execute, False
        # )

        for i in range(num_requests):
            request_id = str(i)
            request_prompt = (
                f"This is a prompt for request number {request_id}."
            )
            request = await pipeline.create_request(
                id=str(i), model_name="test", prompt=request_prompt
            )
            request_params.append(request)
            request_task = pipeline.all_tokens(request)
            request_tasks.append(request_task)

        task_results = await asyncio.gather(*request_tasks)
        assert task_results is not None
        assert len(task_results) == num_requests
        for i, result in enumerate(task_results):
            result_str = "".join(result)
            assert result_str[::-1] == request_params[i].prompt
