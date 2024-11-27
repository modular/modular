# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test serving a Llama 3 model on the GPU."""

import asyncio
import json

import pytest
from async_asgi_testclient import TestClient
from evaluate_llama import SupportedTestModels
from max.driver import DeviceSpec

from max.pipelines import SupportedEncoding, TextTokenizer
from max.serve.mocks.mock_api_requests import simple_openai_request
from max.serve.schemas.openai import CreateChatCompletionResponse  # type: ignore
from test_common.evaluate import PROMPTS
from test_common.numpy_encoder import NumpyDecoder
from test_common.path import find_runtime_path

from .params import ModelParams

MAX_READ_SIZE = 10 * 1024


@pytest.mark.skip(
    reason=(
        "app fixture crashes when reused with 'is bound to a different event"
        " loop'"
    )
)
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tinyllama_model",
    [
        ModelParams(
            weight_path="tiny_llama_bf16.gguf",
            max_length=512,
            max_new_tokens=10,
            device_spec=DeviceSpec.cuda(),
            encoding=SupportedEncoding.bfloat16,
        )
    ],
    indirect=True,
)
async def test_tinyllama_serve_gpu(app):
    # Arbitrary - just demonstrate we can submit multiple async
    # requests and collect the results later
    N_REQUESTS = 3

    async with TestClient(app) as client:
        tasks = [
            client.post("/v1/chat/completions", json=simple_openai_request())
            for _ in range(N_REQUESTS)
        ]

        responses = await asyncio.gather(*tasks)

        for raw_response in responses:
            print(raw_response)

            # This is not a streamed completion - There is no [DONE] at the end.
            response = CreateChatCompletionResponse.parse_raw(
                raw_response.json()
            )
            print(response)

            assert len(response.choices) == 1
            assert response.choices[0].finish_reason == "stop"


@pytest.mark.skip("TODO(ylou): Fix!!")
@pytest.mark.parametrize(
    "tinyllama_model",
    [
        ModelParams(
            weight_path="tiny_llama_bf16.gguf",
            max_length=512,
            max_new_tokens=10,
            device_spec=DeviceSpec.cuda(),
            encoding=SupportedEncoding.bfloat16,
        ),
    ],
    indirect=True,
)
@pytest.mark.asyncio
async def test_tinyllama_serve_gpu_stream(
    app, testdata_directory, tinyllama_model
):
    NUM_TASKS = 16
    model_encoding = SupportedTestModels.get("tinyllama", "bfloat16")
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
    tokenizer = TextTokenizer(inference_config)
    expected_response = [
        await tokenizer.decode(tinyllama_model, x)  # type: ignore
        for x in tokens
    ]

    def openai_completion_request(content):
        """Create the json request for /v1/completion (not chat)."""
        return {
            "model": "gpt-3.5-turbo",
            "prompt": content,
            "temperature": 0.7,
        }

    async def main_stream(client, msg: str, expected):
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
        for t in tasks:  # type: ignore
            resp.append(await t)  # type: ignore
