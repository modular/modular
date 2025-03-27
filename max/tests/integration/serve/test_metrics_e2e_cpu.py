# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test that metrics are collected correctly during a serve request."""

import logging

logging.basicConfig(
    level=logging.DEBUG,
)

import pytest
from async_asgi_testclient import TestClient
from max.driver import DeviceSpec
from max.pipelines import PipelineConfig, SupportedEncoding
from max.pipelines.kv_cache import KVCacheStrategy
from max.serve.schemas.openai import (  # type: ignore
    CreateChatCompletionResponse,
)


@pytest.mark.asyncio()
@pytest.mark.parametrize(
    "pipeline_config",
    [
        PipelineConfig(
            model_path="HuggingFaceTB/SmolLM-135M",
            max_length=512,
            max_new_tokens=3,
            device_specs=[DeviceSpec.cpu()],
            quantization_encoding=SupportedEncoding.float32,
            cache_strategy=KVCacheStrategy.CONTINUOUS,
            max_batch_size=16,
        )
    ],
    indirect=True,
)
async def test_metrics_e2e(app):
    async with TestClient(app, timeout=720.0) as client:
        # Endpoint exists
        raw_response = await client.get("/metrics")

        assert raw_response.status_code == 200

        # There shouldn't be any maxserve_ metrics at this point except for the model load since the server is just started up.
        assert "maxserve_pipeline_load_total" in raw_response.text
        assert (
            "maxserve_request_time_milliseconds_bucket" not in raw_response.text
        )
        # # HELP maxserve_pipeline_load_total Count of pipelines loaded for each model
        # # TYPE maxserve_pipeline_load_total counter
        # # maxserve_pipeline_load_total{model="modularai/llama-3.1"} 1.0
        assert raw_response.text.count("maxserve") == 3

        # Make a request
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
        # Endpoint exists
        raw_response = await client.get("/metrics")

        assert raw_response.status_code == 200
        assert "maxserve_num_input_tokens_total" in raw_response.text
        assert (
            'maxserve_pipeline_load_total{model="HuggingFaceTB/SmolLM-135M"} 1.0'
            in raw_response.text
        )
        assert "maxserve_request_time_milliseconds_bucket" in raw_response.text
        assert (
            "maxserve_time_to_first_token_milliseconds_bucket"
            in raw_response.text
        )
        assert "maxserve_num_output_tokens_total" in raw_response.text
