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
"""End-to-end test for ``MAXModelWorker`` against a real small model.

Deploys the worker on the local runtime, which spawns a ``max.serve``
model-worker subprocess for ``modularai/SmolLM-135M-Instruct-FP32`` on CPU, and
streams generated token ids through the ``decode`` worker method. This
exercises the full path: ``PipelineConfig`` build, ``retrieve_factory``
resolution, subprocess spawn, ZMQ proxy, and per-request streaming.
"""

from __future__ import annotations

import hf_repo_lock
import numpy as np
import pytest
from max.driver import DeviceSpec
from max.experimental.cascade import GenerateRequest, LocalRuntime
from max.experimental.cascade.core.pipeline_method import _pipeline_method_scope
from max.experimental.cascade.workers.max_model_worker import MAXModelWorker
from max.pipelines.lib import PipelineConfig, generate_local_model_path
from max.pipelines.lib.config.model_config import MAXModelConfig
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig
from transformers import AutoTokenizer

REPO_ID = "modularai/SmolLM-135M-Instruct-FP32"
REVISION = hf_repo_lock.revision_for_hf_repo(REPO_ID)


def _model_path() -> str:
    """Resolve a cached local path for the test model, else the repo id."""
    assert REVISION is not None
    try:
        return generate_local_model_path(REPO_ID, REVISION)
    except FileNotFoundError:
        return REPO_ID


def _cpu_pipeline_config(model_path: str) -> PipelineConfig:
    """Build a fresh CPU float32 config, as the entrypoint/CLI would.

    ``MAXModelWorker`` resolves the config in place on deploy, so each test
    gets its own instance.
    """
    return PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path=model_path,
                    device_specs=[DeviceSpec.cpu()],
                    max_length=512,
                    quantization_encoding="float32",
                )
            }
        ),
        runtime=PipelineRuntimeConfig(
            max_batch_size=8,
            enable_chunked_prefill=True,
            enable_in_flight_batching=False,
        ),
    )


@pytest.fixture(scope="module")
def model_path() -> str:
    return _model_path()


@pytest.mark.asyncio
async def test_decode_streams_requested_token_count(model_path: str) -> None:
    # A short prompt of valid, low token ids; content is irrelevant since
    # ``ignore_eos`` forces exactly ``num_tokens`` to be generated.
    prompt = np.array([1, 2, 3, 4], dtype=np.int64)
    worker = MAXModelWorker(_cpu_pipeline_config(model_path))

    async with LocalRuntime() as rt, _pipeline_method_scope():
        proxy = await rt.deploy(worker)

        # One deploy (one model-worker subprocess), several decode requests.
        for num_tokens in (4, 8):
            req = GenerateRequest(num_tokens=num_tokens, ignore_eos=True)
            chunks = [chunk async for chunk in await proxy.decode(req, prompt)]
            generated = (
                np.concatenate(chunks)
                if chunks
                else np.array([], dtype=np.int32)
            )
            assert generated.dtype == np.int32
            assert len(generated) == num_tokens


@pytest.mark.asyncio
async def test_decode_stops_on_eos(model_path: str) -> None:
    prompt = np.array([1, 2, 3, 4], dtype=np.int64)
    worker = MAXModelWorker(_cpu_pipeline_config(model_path))

    async with LocalRuntime() as rt, _pipeline_method_scope():
        proxy = await rt.deploy(worker)
        req = GenerateRequest(num_tokens=16, ignore_eos=False)
        chunks = [chunk async for chunk in await proxy.decode(req, prompt)]
        generated = (
            np.concatenate(chunks) if chunks else np.array([], dtype=np.int32)
        )
    # With EOS honored, generation may stop early but never exceeds the cap.
    assert generated.dtype == np.int32
    assert 0 <= len(generated) <= 16


@pytest.mark.asyncio
async def test_greedy_decode_answers_capital_of_france(model_path: str) -> None:
    # A true end-to-end LLM check: tokenize a factual question with the chat
    # template, greedily decode (temperature 0), and confirm the model answers
    # "Paris".
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    prompt = np.asarray(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )["input_ids"],
        dtype=np.int64,
    )

    worker = MAXModelWorker(_cpu_pipeline_config(model_path))

    async with LocalRuntime() as rt, _pipeline_method_scope():
        proxy = await rt.deploy(worker)
        req = GenerateRequest(num_tokens=20, temperature=0.0)
        chunks = [chunk async for chunk in await proxy.decode(req, prompt)]

    generated = (
        np.concatenate(chunks) if chunks else np.array([], dtype=np.int32)
    )
    answer = tokenizer.decode(generated, skip_special_tokens=True)
    assert "paris" in answer.lower(), f"unexpected answer: {answer!r}"
