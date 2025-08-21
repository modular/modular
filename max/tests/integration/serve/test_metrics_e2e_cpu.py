# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test that metrics are collected correctly during a serve request."""

import time

import pytest
import requests
from async_asgi_testclient import TestClient
from fastapi import FastAPI
from max.driver import DeviceSpec
from max.nn.kv_cache import KVCacheStrategy
from max.pipelines import PipelineConfig, SupportedEncoding
from max.serve.config import MetricLevel, MetricRecordingMethod
from max.serve.schemas.openai import CreateChatCompletionResponse

MODEL_NAME = "modularai/SmolLM-135M-Instruct-FP32"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline_config",
    [
        PipelineConfig(
            model_path=MODEL_NAME,
            max_length=512,
            max_new_tokens=3,
            device_specs=[DeviceSpec.cpu()],
            quantization_encoding=SupportedEncoding.float32,
            cache_strategy=KVCacheStrategy.PAGED,
            max_batch_size=16,
            allow_safetensors_weights_fp32_bf6_bidirectional_cast=True,
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "settings_config,",
    [
        {
            "MAX_SERVE_USE_HEARTBEAT": True,
            "MAX_SERVE_METRIC_RECORDING_METHOD": MetricRecordingMethod.PROCESS,
            "MAX_SERVE_METRIC_LEVEL": MetricLevel.DETAILED,
            # This ensures that batch size is sent immediately and not buffered
            "MAX_SERVE_DETAILED_METRIC_BUFFER_FACTOR": 0,
        }
    ],
    indirect=True,
)
async def test_metrics_e2e_v1(app: FastAPI) -> None:
    # Method 2: Using client tuple (host, port)
    async with TestClient(app, timeout=720.0) as client:
        # Endpoint exists
        # use requests since TestClient will return 404 for /metrics endpoint as metrics server is not mounted on FastAPI
        response = requests.get("http://localhost:8001/metrics", timeout=1)
        assert response.status_code == 200

        # There shouldn't be any maxserve_ metrics at this point except for the model load since the server is just started up.
        assert "maxserve_model_load_time_milliseconds_bucket" in response.text
        assert "maxserve_request_time_milliseconds_bucket" not in response.text

        # Make a request
        raw_response = await client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "tell me a joke"}],
                "stream": False,
            },
        )
        # This is not a streamed completion - There is no [DONE] at the end.
        response = CreateChatCompletionResponse.model_validate(
            raw_response.json()
        )

        response = requests.get("http://localhost:8001/metrics", timeout=1)

        # Ensure enough time for request to complete
        time.sleep(5)

        assert response.status_code == 200
        assert "maxserve_num_input_tokens_total" in response.text
        assert (
            f'maxserve_pipeline_load_total{{model="{MODEL_NAME}"}} 1.0'
            in response.text
        )
        assert "maxserve_request_time_milliseconds_bucket" in response.text
        assert (
            "maxserve_time_to_first_token_milliseconds_bucket" in response.text
        )
        assert "maxserve_num_output_tokens_total" in response.text
        assert "maxserve_batch_size" in response.text
        assert "maxserve_cache_hit_rate" in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline_config",
    [
        PipelineConfig(
            model_path=MODEL_NAME,
            max_length=512,
            max_new_tokens=3,
            device_specs=[DeviceSpec.cpu()],
            quantization_encoding=SupportedEncoding.float32,
            cache_strategy=KVCacheStrategy.PAGED,
            max_batch_size=16,
            allow_safetensors_weights_fp32_bf6_bidirectional_cast=True,
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "settings_config",
    [
        {
            "MAX_SERVE_USE_HEARTBEAT": True,
            "MAX_SERVE_METRIC_LEVEL": MetricLevel.DETAILED,
            "MAX_SERVE_METRIC_RECORDING_METHOD": MetricRecordingMethod.PROCESS,
            # This ensures that batch size is sent immediately and not buffered
            "MAX_SERVE_DETAILED_METRIC_BUFFER_FACTOR": 0,
        }
    ],
    indirect=True,
)
async def test_metrics_e2e_v0(app: FastAPI) -> None:
    async with TestClient(app, timeout=720.0) as client:
        # Endpoint exists
        raw_response = requests.get("http://localhost:8001/metrics", timeout=1)

        assert raw_response.status_code == 200

        assert "maxserve_pipeline_load_total" in raw_response.text

        # Make a request
        raw_response = await client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "tell me a joke"}],
                "stream": False,
            },
        )
        # This is not a streamed completion - There is no [DONE] at the end.
        response = CreateChatCompletionResponse.model_validate(
            raw_response.json()
        )

        # Endpoint exists
        raw_response = requests.get("http://localhost:8001/metrics", timeout=1)

        assert raw_response.status_code == 200
        assert "maxserve_num_input_tokens_total" in raw_response.text
        assert (
            f'maxserve_pipeline_load_total{{model="{MODEL_NAME}"}}'
            in raw_response.text
        )
        assert "maxserve_request_time_milliseconds_bucket" in raw_response.text
        assert (
            "maxserve_time_to_first_token_milliseconds_bucket"
            in raw_response.text
        )
        assert "maxserve_num_output_tokens_total" in raw_response.text
        assert "maxserve_batch_size" in raw_response.text
        assert "maxserve_cache_hit_rate" in raw_response.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline_config",
    [
        PipelineConfig(
            model_path=MODEL_NAME,
            max_length=512,
            max_new_tokens=3,
            device_specs=[DeviceSpec.cpu()],
            quantization_encoding=SupportedEncoding.float32,
            cache_strategy=KVCacheStrategy.PAGED,
            max_batch_size=16,
            allow_safetensors_weights_fp32_bf6_bidirectional_cast=True,
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "settings_config",
    [
        {
            "MAX_SERVE_DISABLE_TELEMETRY": True,
            "MAX_SERVE_METRIC_RECORDING_METHOD": MetricRecordingMethod.PROCESS,
        }
    ],
    indirect=True,
)
async def test_metrics_e2e_validate_disable_works_v1(app: FastAPI) -> None:
    async with TestClient(app, timeout=720.0) as client:
        # Endpoint won't exist
        with pytest.raises(requests.exceptions.ConnectionError):
            response = requests.get("http://localhost:8001/metrics", timeout=1)
