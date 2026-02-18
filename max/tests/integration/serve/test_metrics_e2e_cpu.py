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
"""Test that metrics are collected correctly during a serve request."""

import time

import hf_repo_lock
import pytest
import requests
from async_asgi_testclient import TestClient
from fastapi import FastAPI
from max.driver import DeviceSpec
from max.pipelines import PipelineConfig, SupportedEncoding
from max.serve.config import MetricLevel, MetricRecordingMethod
from max.serve.schemas.openai import CreateChatCompletionResponse

MODEL_NAME = "modularai/SmolLM-135M-Instruct-FP32"


def assert_metrics(
    expected_metrics: list[str],
    absent_metrics: list[str] | None,
    timeout: float = 10.0,
    poll_interval: float = 0.5,
) -> None:
    """Poll metrics endpoint until expected metrics are present and absent metrics are not."""
    deadline = time.time() + timeout
    response = requests.Response()
    while time.time() < deadline:
        response = requests.get("http://localhost:8001/metrics", timeout=1)
        if response.status_code == 200:
            if all(metric in response.text for metric in expected_metrics):
                if absent_metrics:
                    for metric in absent_metrics:
                        assert metric not in response.text, (
                            f"Metric {metric} should not be present"
                        )
                return
        time.sleep(poll_interval)
    raise AssertionError(
        f"Metrics not found within {timeout}s: "
        f"{[m for m in expected_metrics if m not in response.text]}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline_config",
    [
        PipelineConfig(
            model_path=MODEL_NAME,
            huggingface_model_revision=hf_repo_lock.revision_for_hf_repo(
                MODEL_NAME
            ),
            max_length=512,
            device_specs=[DeviceSpec.cpu()],
            quantization_encoding=SupportedEncoding.float32,
            cache_strategy="paged",
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
        # Wait for the model load metric to be available (metrics propagate async)
        # There shouldn't be any request metrics yet since the server is just started up.
        assert_metrics(
            expected_metrics=["maxserve_model_load_time_milliseconds_bucket"],
            absent_metrics=["maxserve_request_time_milliseconds_bucket"],
        )

        # Make a few requests
        for _ in range(5):
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
            CreateChatCompletionResponse.model_validate(raw_response.json())

        # Wait for request metrics to propagate
        assert_metrics(
            expected_metrics=[
                "maxserve_num_input_tokens_total",
                "maxserve_request_time_milliseconds_bucket",
                "maxserve_time_to_first_token_milliseconds_bucket",
                "maxserve_num_output_tokens_total",
                "maxserve_batch_size",
                "maxserve_cache_hit_rate",
                f'maxserve_pipeline_load_total{{model="{MODEL_NAME}"}} 1.0',
            ],
            absent_metrics=None,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline_config",
    [
        PipelineConfig(
            model_path=MODEL_NAME,
            huggingface_model_revision=hf_repo_lock.revision_for_hf_repo(
                MODEL_NAME
            ),
            max_length=512,
            device_specs=[DeviceSpec.cpu()],
            quantization_encoding=SupportedEncoding.float32,
            cache_strategy="paged",
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
        # Wait for the pipeline load metric to be available (metrics propagate async)
        assert_metrics(
            expected_metrics=["maxserve_pipeline_load_total"],
            absent_metrics=None,
        )

        # Make a few requests
        for _ in range(5):
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
            CreateChatCompletionResponse.model_validate(raw_response.json())

        # Wait for request metrics to propagate
        assert_metrics(
            expected_metrics=[
                "maxserve_num_input_tokens_total",
                "maxserve_request_time_milliseconds_bucket",
                "maxserve_time_to_first_token_milliseconds_bucket",
                "maxserve_num_output_tokens_total",
                "maxserve_batch_size",
                "maxserve_cache_hit_rate",
                f'maxserve_pipeline_load_total{{model="{MODEL_NAME}"}} 1.0',
            ],
            absent_metrics=None,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline_config",
    [
        PipelineConfig(
            model_path=MODEL_NAME,
            huggingface_model_revision=hf_repo_lock.revision_for_hf_repo(
                MODEL_NAME
            ),
            max_length=512,
            device_specs=[DeviceSpec.cpu()],
            quantization_encoding=SupportedEncoding.float32,
            cache_strategy="paged",
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
