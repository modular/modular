# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test serving a Llama 3 model on the CPU."""

import asyncio
import json

import pytest
from async_asgi_testclient import TestClient
from max.driver import DeviceSpec
from max.pipelines import PipelineConfig, PipelineEngine, SupportedEncoding
from max.pipelines.kv_cache import KVCacheStrategy
from max.serve.schemas.openai import (  # type: ignore
    CreateChatCompletionResponse,
    CreateCompletionResponse,
)
from test_common.evaluate import PROMPTS

MAX_READ_SIZE = 10 * 1024


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline_config",
    [
        PipelineConfig(
            huggingface_repo_id="HuggingFaceTB/SmolLM-135M",
            max_length=512,
            max_new_tokens=3,
            device_specs=[DeviceSpec.cpu()],
            quantization_encoding=SupportedEncoding.float32,
            cache_strategy=KVCacheStrategy.CONTINUOUS,
            max_batch_size=16,
            engine=PipelineEngine.MAX,
        )
    ],
    indirect=True,
)
async def test_tinyllama_serve_cpu(app):
    async with TestClient(app, timeout=720.0) as client:
        # Test with streaming set to False
        raw_response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "HuggingFaceTB/SmolLM-135M",
                "messages": [{"role": "user", "content": "tell me a joke"}],
                "stream": False,
            },
        )
        # This is not a streamed completion - There is no [DONE] at the end.
        response = CreateChatCompletionResponse.model_validate(
            raw_response.json()
        )

        assert len(response.choices) == 1
        assert response.choices[0].finish_reason == "stop"

        # Test a few prompts, in different formats
        prompt_num = [
            ("Hello world", 1),
            (["Hello world"], 1),
            (["Hello world", "hello there"], 2),
            ([1, 2, 3], 1),
            ([[1, 2, 3]], 1),
            ([[1, 2, 3], [4, 5, 6]], 2),
        ]
        for prompt, n_prompts in prompt_num:
            # Completions endpoint instead of chat completions
            raw_response = await client.post(
                "/v1/completions",
                json={
                    "model": "HuggingFaceTB/SmolLM-135M",
                    "prompt": prompt,
                },
            )
            response = CreateCompletionResponse.model_validate(
                raw_response.json()
            )
            assert len(response.choices) == n_prompts
            assert response.choices[0].finish_reason == "stop"

    def openai_completion_request(content):
        """Create the json request for /v1/completion (not chat)."""
        return {
            "model": "HuggingFaceTB/SmolLM-135M",
            "prompt": content,
            "temperature": 0.7,
        }

    async def main_stream(client, msg: str):
        print(f"Generated request with prompt :{msg}")
        r = await client.post(
            "/v1/completions",
            json=openai_completion_request(msg)
            | {
                "stream": True,
            },
            stream=True,
        )
        response_text = ""
        async for response in r.iter_content(MAX_READ_SIZE):
            response = response.decode("utf-8").strip()
            if response.startswith("data: [DONE]"):
                break
            try:
                data = json.loads(response[len("data: ") :])
                content = data["choices"][0]["text"]
                response_text += content
            except Exception as e:
                # Just suppress the exception as it might be a ping message.
                print(f"Exception {e} at '{response}'")
        return response_text

    tasks = []
    resp = []
    async with TestClient(app, timeout=720.0) as client:
        for prompt in PROMPTS[1:]:
            tasks.append(asyncio.create_task(main_stream(client, prompt)))

        for task in tasks:
            resp.append(await task)
