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
"""Fast (no-download) tests for the cascade ``MAXModelWorker`` worker.

Construction metadata and the ``echo`` debugging method run without a model,
GPU, or network. End-to-end generation against a real model lives in
``test_max_model_worker_integration.py``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest
from max.driver import DeviceSpec
from max.experimental.cascade.interfaces.textgen import GenerateRequest
from max.experimental.cascade.workers.max_model_worker import (
    MAXModelWorker,
    _sampling_params_input,
)
from max.pipelines.lib import PipelineConfig
from max.pipelines.lib.config.model_config import MAXModelConfig
from max.pipelines.lib.model_manifest import ModelManifest


def _config(device_specs: list[DeviceSpec]) -> PipelineConfig:
    """Build an unresolved config for construction-only (no-download) tests."""
    return PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path="fake/model", device_specs=device_specs
                )
            }
        )
    )


# ``@worker_method`` types every method for the proxy call path (returning a
# ``Coroutine[..., ResultIter]``). Calling a streaming method directly on a
# worker instance instead returns the async iterator at runtime, so ``echo`` --
# which never touches the model proxy -- can be exercised without a GPU by
# casting back to the runtime shape.
def _echo(
    worker: MAXModelWorker, tokens: npt.NDArray[np.int32]
) -> AsyncIterator[np.int32]:
    return cast("AsyncIterator[np.int32]", worker.echo(tokens))


def test_construction_stores_config() -> None:
    config = _config([DeviceSpec.accelerator()])
    worker = MAXModelWorker(config)
    assert worker.pipeline_config is config
    assert worker.max_length is None
    assert worker._proxy is None


def test_accelerator_config_sets_gpu_hint() -> None:
    worker = MAXModelWorker(_config([DeviceSpec.accelerator()]))
    assert worker.deploy_hints == ["gpu"]


def test_cpu_config_sets_cpu_hint() -> None:
    worker = MAXModelWorker(_config([DeviceSpec.cpu()]))
    assert worker.deploy_hints == ["cpu"]


@pytest.mark.asyncio
async def test_echo_streams_tokens() -> None:
    worker = MAXModelWorker(_config([DeviceSpec.cpu()]))
    tokens = np.array([5, 6, 7], dtype=np.int32)
    echoed = [int(token) async for token in _echo(worker, tokens)]
    assert echoed == [5, 6, 7]


@pytest.mark.asyncio
async def test_echo_empty() -> None:
    worker = MAXModelWorker(_config([DeviceSpec.cpu()]))
    tokens = np.array([], dtype=np.int32)
    echoed = [token async for token in _echo(worker, tokens)]
    assert echoed == []


def test_sampling_params_input_forwards_all_fields() -> None:
    req = GenerateRequest(
        num_tokens=5,
        min_new_tokens=2,
        ignore_eos=True,
        temperature=0.5,
        top_k=7,
        top_p=0.9,
        min_p=0.1,
        thinking_temperature=0.3,
        seed=1234,
        frequency_penalty=0.25,
        presence_penalty=-0.5,
        repetition_penalty=1.1,
        stop=["END", "STOP"],
        stop_token_ids=[1, 2, 3],
    )
    spi = _sampling_params_input(req)
    assert spi.max_new_tokens == 5
    assert spi.min_new_tokens == 2
    assert spi.ignore_eos is True
    assert spi.temperature == 0.5
    assert spi.top_k == 7
    assert spi.top_p == 0.9
    assert spi.min_p == 0.1
    assert spi.thinking_temperature == 0.3
    assert spi.seed == 1234
    assert spi.frequency_penalty == 0.25
    assert spi.presence_penalty == -0.5
    assert spi.repetition_penalty == 1.1
    assert spi.stop == ["END", "STOP"]
    assert spi.stop_token_ids == [1, 2, 3]


def test_sampling_params_input_defaults_are_none() -> None:
    # ``None`` fields let SamplingParams fall back to model / class defaults.
    spi = _sampling_params_input(GenerateRequest(num_tokens=8))
    assert spi.max_new_tokens == 8
    assert spi.min_new_tokens == 0
    assert spi.ignore_eos is False
    assert spi.top_k is None
    assert spi.top_p is None
    assert spi.min_p is None
    assert spi.thinking_temperature is None
    assert spi.seed is None
    assert spi.frequency_penalty is None
    assert spi.presence_penalty is None
    assert spi.repetition_penalty is None
    assert spi.stop is None
    assert spi.stop_token_ids is None
