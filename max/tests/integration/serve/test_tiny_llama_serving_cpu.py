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
from evaluate_llama import (
    PROMPTS,
    NumpyDecoder,
    SupportedTestModels,
    find_runtime_path,
)
from llama3 import Llama3Tokenizer, SupportedEncodings
from max.driver import DeviceSpec
from max.serve.mocks.mock_api_requests import simple_openai_request
from max.serve.schemas.openai import CreateChatCompletionResponse

from .params import ModelParams

pytestmark = pytest.mark.skip("TODO(ylou): Fix!!")

MAX_READ_SIZE = 10 * 1024


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tinyllama_model",
    [
        ModelParams(
            weight_path="tiny_llama.gguf",
            max_length=512,
            max_new_tokens=10,
            device_spec=DeviceSpec.cpu(),
            encoding=SupportedEncodings.float32,
        )
    ],
    indirect=True,
)
async def test_tinyllama_serve_cpu(app):
    async with TestClient(app) as client:
        raw_response = await client.post(
            "/v1/chat/completions", json=simple_openai_request()
        )
        # This is not a streamed completion - There is no [DONE] at the end.
        response = CreateChatCompletionResponse.model_validate_json(
            raw_response.json()
        )

        assert len(response.choices) == 1
        assert response.choices[0].finish_reason == "stop"


@pytest.mark.xfail(reason="SI-667")
@pytest.mark.parametrize(
    "tinyllama_model",
    [
        ModelParams(
            weight_path="tiny_llama.gguf",
            max_length=512,
            max_new_tokens=10,
            device_spec=DeviceSpec.cpu(),
            encoding=SupportedEncodings.float32,
        ),
    ],
    indirect=True,
)
@pytest.mark.asyncio
async def test_tinyllama_serve_cpu_stream(
    app, testdata_directory, tinyllama_model
):
    NUM_TASKS = 16
    model_encoding = SupportedTestModels.TINY_LLAMA_BF16
    golden_data_path = find_runtime_path(
        model_encoding.golden_data_fname(),
        testdata_directory,
    )
    expected_results = NumpyDecoder().decode(golden_data_path.read_text())
    prompts = [p["prompt"] for p in expected_results]
    values = [p["values"] for p in expected_results]
    tokens = []
    for v in values:
        t = []
        for e in v:
            t.append(e["next_token"])
        tokens.append(t)

    inference_config = model_encoding.build_config(testdata_directory)
    tokenizer = Llama3Tokenizer(inference_config)
    expected_response = [
        await tokenizer.decode(tinyllama_model, x) for x in tokens
    ]

    def openai_completion_request(content):
        """Create the json request for /v1/completion (not chat)."""
        return {
            "model": "gpt-3.5-turbo",
            "prompt": content,
            "temperature": 0.7,
        }

    async def main_stream(client, msg: str, expected):
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
        # NOTE: intentionally don't compare text produces since this test uses
        # TinyLlama, whose weights are random, causing numerical issues.
        return response_text

    tasks = []
    resp = []
    async with TestClient(app, timeout=5.0) as client:
        for i in range(NUM_TASKS):
            # we skip the first prompt as it is longer than 512
            data_idx = 1 + (i % (len(PROMPTS) - 1))
            msg = prompts[data_idx]
            expected = expected_response[data_idx]
            tasks.append(
                asyncio.create_task(main_stream(client, msg, expected))
            )
        for t in tasks:
            resp.append(await t)
