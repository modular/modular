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
"""Test serving a Llama 3 model on the CPU."""

import asyncio
import json
from typing import Any

import pytest
from async_asgi_testclient import TestClient
from fastapi import FastAPI
from max.driver import DeviceSpec
from max.pipelines import PipelineConfig, SupportedEncoding
from max.pipelines.lib import KVCacheConfig, MAXModelConfig
from max.serve.schemas.openai import (
    CreateChatCompletionResponse,
    CreateCompletionResponse,
)
from test_common.test_data import DEFAULT_PROMPTS

MAX_READ_SIZE = 10 * 1024

MODEL_NAME = "modularai/SmolLM-135M-Instruct-FP32"

pipeline_config = PipelineConfig(
    model=MAXModelConfig(
        model_path=MODEL_NAME,
        device_specs=[DeviceSpec.cpu()],
        quantization_encoding=SupportedEncoding.float32,
        allow_safetensors_weights_fp32_bf6_bidirectional_cast=True,
        kv_cache=KVCacheConfig(cache_strategy="paged"),
        max_length=512,
    ),
    max_batch_size=16,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline_config",
    [pipeline_config],
    indirect=True,
)
async def test_tinyllama_serve_v1_chat_completions_cpu(app: FastAPI) -> None:
    async with TestClient(app, timeout=720.0) as client:
        # Test with streaming set to False
        raw_response = await client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "tell me a joke"}],
                "stream": False,
                "max_tokens": 3,
            },
        )
        # This is not a streamed completion - There is no [DONE] at the end.
        response = CreateChatCompletionResponse.model_validate(
            raw_response.json()
        )

        assert len(response.choices) == 1
        assert response.choices[0].finish_reason == "length"

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
                json={"model": MODEL_NAME, "prompt": prompt},
            )
            response2 = CreateCompletionResponse.model_validate(
                raw_response.json()
            )
            assert len(response2.choices) == n_prompts
            assert response2.choices[0].finish_reason in ["length", "stop"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline_config",
    [pipeline_config],
    indirect=True,
)
async def test_tinyllama_serve_v1_completions_cpu(app: FastAPI) -> None:
    def openai_completion_request(content: str) -> dict[str, Any]:
        """Create the json request for /v1/completion (not chat)."""
        return {
            "model": MODEL_NAME,
            "prompt": content,
            "temperature": 0.7,
            "max_tokens": 3,
        }

    async def main_stream(client: TestClient, msg: str) -> str:
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
        for prompt in DEFAULT_PROMPTS[1:]:
            tasks.append(asyncio.create_task(main_stream(client, prompt)))

        for task in tasks:
            resp.append(await task)
