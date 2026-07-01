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

import re
import time

import hf_repo_lock
import pytest
import requests
from async_asgi_testclient import TestClient
from fastapi import FastAPI
from max.driver import DeviceSpec
from max.pipelines import PipelineConfig
from max.pipelines.lib import KVCacheConfig, MAXModelConfig
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig
from max.serve.config import MetricLevel, MetricRecordingMethod
from max.serve.schemas.openai import CreateChatCompletionResponse

MODEL_NAME = "modularai/SmolLM-135M-Instruct-FP32"
MODEL_REVISION = hf_repo_lock.revision_for_hf_repo(MODEL_NAME)
assert MODEL_REVISION is not None


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


def _series_value(
    metrics_text: str, name: str, *, component: str
) -> float | None:
    """Return the value of a Prometheus series for a given component.

    Matches a line like ``name{...,component="<component>",...} <value>``
    regardless of label ordering, and returns the parsed float value.
    """
    pattern = re.compile(
        rf'^{re.escape(name)}\{{[^}}]*component="{re.escape(component)}"[^}}]*\}}\s+(\S+)$',
        re.MULTILINE,
    )
    match = pattern.search(metrics_text)
    return float(match.group(1)) if match else None


def _metric_total(metrics_text: str, name: str) -> float:
    """Sum the value of every Prometheus series for ``name``.

    Handles both label-free lines (``name <value>``) and labeled lines
    (``name{...} <value>``) so the total is independent of label
    cardinality. Intended for counters and histogram ``_sum`` series, not
    cumulative ``_bucket`` series.
    """
    total = 0.0
    for line in metrics_text.splitlines():
        series, sep, value = line.partition(" ")
        if not sep:
            continue
        series_name = series.split("{", 1)[0]
        if series_name == name:
            total += float(value)
    return total


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline_config",
    [
        PipelineConfig(
            models=ModelManifest(
                {
                    "main": MAXModelConfig(
                        model_path=MODEL_NAME,
                        huggingface_model_revision=MODEL_REVISION,
                        device_specs=[DeviceSpec.cpu()],
                        quantization_encoding="float32",
                        kv_cache=KVCacheConfig(),
                        max_length=512,
                    )
                }
            ),
            runtime=PipelineRuntimeConfig(max_batch_size=16),
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
            expected_metrics=[
                "maxserve_model_load_time_milliseconds_bucket",
                # Per-phase startup breakdown on the same metric, split by
                # the component tag.
                'component="total"',
                'component="compile"',
            ],
            absent_metrics=["maxserve_request_time_milliseconds_bucket"],
        )

        # The histogram must carry the real measured duration, not 0.
        metrics_text = requests.get(
            "http://localhost:8001/metrics", timeout=1
        ).text
        total = _series_value(
            metrics_text,
            "maxserve_model_load_time_milliseconds_sum",
            component="total",
        )
        assert total is not None, (
            "maxserve_model_load_time_milliseconds_sum{component='total'} "
            "not found"
        )
        assert total > 0.0, (
            f"model_load_time total sum should be > 0, got {total}"
        )

        # Make a few requests, summing the prompt tokens the API reports.
        # The API usage field is the ground truth for input-token counts.
        num_requests = 5
        expected_input_tokens = 0
        for _ in range(num_requests):
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
            parsed = CreateChatCompletionResponse.model_validate(
                raw_response.json()
            )
            assert parsed.usage is not None
            expected_input_tokens += parsed.usage.prompt_tokens

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

        # Once all requests land in the histogram, assert the counter matches
        # the API's prompt-token total. The exact-count check catches both a
        # 2x counter and a symmetric counter+histogram double-emit.
        deadline = time.time() + 10.0
        input_counter = input_hist_sum = hist_count = 0.0
        while time.time() < deadline:
            metrics_text = requests.get(
                "http://localhost:8001/metrics", timeout=1
            ).text
            hist_count = _metric_total(
                metrics_text, "maxserve_input_tokens_per_request_tokens_count"
            )
            if hist_count == num_requests:
                input_counter = _metric_total(
                    metrics_text, "maxserve_num_input_tokens_total"
                )
                input_hist_sum = _metric_total(
                    metrics_text,
                    "maxserve_input_tokens_per_request_tokens_sum",
                )
                break
            time.sleep(0.5)

        assert hist_count == num_requests, (
            f"expected {num_requests} requests recorded in the input-token "
            f"histogram, got {hist_count}"
        )
        assert input_counter == expected_input_tokens, (
            f"input counter ({input_counter}) must equal the API-reported "
            f"prompt-token total ({expected_input_tokens}); a larger value "
            "means the counter is emitted more than once per request"
        )
        assert input_hist_sum == expected_input_tokens, (
            f"input histogram sum ({input_hist_sum}) must equal the "
            f"API-reported prompt-token total ({expected_input_tokens})"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline_config",
    [
        PipelineConfig(
            models=ModelManifest(
                {
                    "main": MAXModelConfig(
                        model_path=MODEL_NAME,
                        huggingface_model_revision=MODEL_REVISION,
                        device_specs=[DeviceSpec.cpu()],
                        quantization_encoding="float32",
                        kv_cache=KVCacheConfig(),
                        max_length=512,
                    )
                }
            ),
            runtime=PipelineRuntimeConfig(max_batch_size=16),
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
            models=ModelManifest(
                {
                    "main": MAXModelConfig(
                        model_path=MODEL_NAME,
                        huggingface_model_revision=MODEL_REVISION,
                        device_specs=[DeviceSpec.cpu()],
                        quantization_encoding="float32",
                        kv_cache=KVCacheConfig(),
                        max_length=512,
                    )
                }
            ),
            runtime=PipelineRuntimeConfig(max_batch_size=16),
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
    async with TestClient(app, timeout=720.0):
        # Endpoint won't exist
        with pytest.raises(requests.exceptions.ConnectionError):
            requests.get("http://localhost:8001/metrics", timeout=1)
