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
"""Tests for how ``serve`` sets up the worker context."""

from __future__ import annotations

import sys
import types

import pytest
from max.experimental.cascade import LocalRuntime, Runtime, Worker
from max.experimental.cascade.deployment.context_config import ContextConfig
from max.experimental.cascade.deployment.routing import HybridRuntime
from max.experimental.cascade.interfaces.pipeline import CascadePipeline
from max.experimental.cascade.serve import main
from max.pipelines.lib import PipelineArgs


class _Pipeline(CascadePipeline):
    def __init__(self, hint: str) -> None:
        self.first = Worker(deploy_hints=[hint])
        self.second = Worker(deploy_hints=["cpu"])


class _Stop(Exception):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("hint", "pools"),
    [("rust", {"cpu", "gpu", "rust"}), ("gpu", {"cpu", "gpu"})],
)
async def test_serve_starts_rust_pool_for_rust_workers(
    monkeypatch: pytest.MonkeyPatch, hint: str, pools: set[str]
) -> None:
    stub = types.ModuleType("local_rust_runtime")
    monkeypatch.setattr(stub, "RustRuntime", LocalRuntime, raising=False)
    monkeypatch.setitem(sys.modules, "local_rust_runtime", stub)
    runtimes: list[Runtime] = []

    async def build_pipeline(args: PipelineArgs) -> _Pipeline:
        return _Pipeline(hint)

    async def build_router(
        pipeline: _Pipeline, runtime: Runtime, emit: bool
    ) -> None:
        runtimes.append(runtime)
        raise _Stop

    monkeypatch.setattr(main, "build_pipeline", build_pipeline)
    monkeypatch.setattr(main, "build_router", build_router)
    with pytest.raises(_Stop):
        await main.serve(
            PipelineArgs(),
            ContextConfig(local_cpu_workers=0),
        )

    (runtime,) = runtimes
    assert isinstance(runtime, HybridRuntime)
    assert set(runtime._runtimes) == pools


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("local_cpu_workers", "local_gpu_workers", "sized"),
    [(2, None, True), (2, 0, False), (0, None, False)],
)
async def test_serve_sizes_gpu_workers_only_when_unset_and_needed(
    monkeypatch: pytest.MonkeyPatch,
    local_cpu_workers: int,
    local_gpu_workers: int | None,
    sized: bool,
) -> None:
    calls: list[PipelineArgs] = []

    def count_unique_device_specs(args: PipelineArgs) -> int:
        calls.append(args)
        return 0

    async def build_pipeline(args: PipelineArgs) -> _Pipeline:
        raise _Stop

    monkeypatch.setattr(
        main, "count_unique_device_specs", count_unique_device_specs
    )
    monkeypatch.setattr(main, "build_pipeline", build_pipeline)
    with pytest.raises(_Stop):
        await main.serve(
            PipelineArgs(),
            ContextConfig(
                local_cpu_workers=local_cpu_workers,
                local_gpu_workers=local_gpu_workers,
            ),
        )
    assert bool(calls) == sized
