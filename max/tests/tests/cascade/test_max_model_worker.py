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
from max.experimental.cascade.workers.max_model_worker import MAXModelWorker
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
