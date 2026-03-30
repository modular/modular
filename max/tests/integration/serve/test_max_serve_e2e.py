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
"""End-to-end tests for MAX serve.

This test suite validates server functionality by launching max serve in a
subprocess and testing various API endpoints.
"""

import asyncio
import logging
import multiprocessing
import time
from collections.abc import AsyncGenerator
from multiprocessing.context import SpawnProcess

import httpx
import pytest
import pytest_asyncio

logger = logging.getLogger(__name__)

PORT = 8000
METRICS_PORT = 8001
MODEL = "modularai/SmolLM-135M-Instruct-FP32"
BASE_URL = f"http://127.0.0.1:{PORT}"
HEALTH_URL = f"{BASE_URL}/health"
CHAT_COMPLETIONS_URL = f"{BASE_URL}/v1/chat/completions"


def serve_main() -> None:
    """Entrypoint to run max serve in subprocess.

    This function configures and launches the serve API server with model worker.
    It blocks in uvloop.run(server.serve()) until the server is shut down.
    """
    from max.driver import DeviceSpec
    from max.entrypoints.cli.serve.serve_api_and_model_worker import (
        serve_api_server_and_model_worker,
    )
    from max.pipelines import PipelineConfig
    from max.pipelines.lib.config.model_config import MAXModelConfig
    from max.serve.config import Settings

    settings = Settings(
        port=PORT,
        metrics_port=METRICS_PORT,
    )
    # Configure pipeline with GGUF model for fast loading on CPU
    pipeline_config = PipelineConfig(
        model=MAXModelConfig(
            model_path=MODEL,
            device_specs=[DeviceSpec.cpu()],
            quantization_encoding="float32",
        ),
    )
    # Launch server (blocks until shutdown)
    serve_api_server_and_model_worker(settings, pipeline_config)


async def wait_for_server_ready(
    proc: SpawnProcess, url: str, timeout: float
) -> None:
    """Poll server health endpoint until ready or timeout.

    Args:
        proc: Server process to monitor
        url: Health check URL (e.g., "http://127.0.0.1:8000/health")
        timeout: Maximum seconds to wait for server readiness

    Raises:
        RuntimeError: If server process exits before becoming ready
        TimeoutError: If server doesn't become ready within timeout
    """
    start = time.monotonic()
    async with httpx.AsyncClient() as client:
        while time.monotonic() - start < timeout:
            try:
                response = await client.get(url, timeout=1.0)
                if response.status_code == 200:
                    elapsed = time.monotonic() - start
                    logger.info(f"Server ready after {elapsed:.1f}s")
                    return
            except (httpx.RequestError, httpx.TimeoutException):
                pass

            await asyncio.sleep(1)

            if not proc.is_alive():
                raise RuntimeError(
                    f"Server process exited with code {proc.exitcode}"
                )

    raise TimeoutError(
        f"Server at {url} did not become ready within {timeout}s"
    )


@pytest_asyncio.fixture(scope="module")
async def max_serve_server() -> AsyncGenerator[str, None]:
    """Pytest fixture that launches max serve and waits for it to be ready.

    This fixture is module-scoped, meaning the server is started once and
    shared across all tests in this file for efficiency.

    Yields:
        Base URL of the running server (e.g., "http://127.0.0.1:8000")

    Note:
        - Uses non-daemon Process so the server can spawn child processes
        - Timeout is generous to allow for model download and compilation
        - Automatically shuts down server after all tests complete
    """
    # Use spawn method to ensure clean process separation
    ctx = multiprocessing.get_context("spawn")
    server_process = ctx.Process(target=serve_main)
    server_process.start()

    try:
        # Huge timeout for model download + compile (and ASAN CI is super slow)
        await wait_for_server_ready(server_process, HEALTH_URL, timeout=900)

        # Server is ready, yield control to test
        yield BASE_URL

    finally:
        # Cleanup: terminate server process
        if server_process.is_alive():
            server_process.terminate()
            server_process.join(timeout=10)
        # If server is not shut down 10s after SIGTERM, we have a bug
        # FIXME SERVSYS-1197: assert fails 0.6% of the time in CI
        # assert not server_process.is_alive(), (
        #     "Server process failed to shut down"
        # )


@pytest.mark.asyncio
async def test_chat_completions(max_serve_server: str) -> None:
    """Test basic chat completions endpoint.

    This test validates:
    1. Server accepts chat completion requests
    2. Response structure matches OpenAI API format
    3. Generated text is present in response
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{max_serve_server}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 10,
            },
        )

    # Verify response structure
    assert response.status_code == 200, (
        f"Expected status 200, got {response.status_code}: {response.text}"
    )

    data = response.json()
    assert "choices" in data, f"Missing 'choices' in response: {data}"
    assert len(data["choices"]) > 0, "Expected at least one choice in response"
    assert "message" in data["choices"][0], (
        f"Missing 'message' in choice: {data['choices'][0]}"
    )
    assert "content" in data["choices"][0]["message"], (
        f"Missing 'content' in message: {data['choices'][0]['message']}"
    )

    content = data["choices"][0]["message"]["content"]
    assert len(content) > 0, "Expected non-empty generated content"
    logger.info(f"Text generation successful. Response: {content}")


@pytest.mark.asyncio
async def test_health_endpoint(max_serve_server: str) -> None:
    """Test health check endpoint returns 200 OK."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(f"{max_serve_server}/health")

    assert response.status_code == 200, (
        f"Expected status 200, got {response.status_code}"
    )
