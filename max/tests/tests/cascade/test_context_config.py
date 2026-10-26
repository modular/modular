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
"""Tests for the serve-CLI :class:`ContextConfig`.

Verifies the transport factories build the right runtimes, that
``build_context`` assembles a device-routed hybrid pool from the worker-count
and remote-endpoint fields, and that ``open_context`` yields a runtime that
runs the dummy text-gen pipeline end to end over multi-process workers.
"""

from __future__ import annotations

import os
import sys
import types

import pytest
from max.experimental.cascade import (
    ChatMessage,
    GenAIRequest,
    GenAITextChunk,
    LocalRuntime,
    TextGenOptions,
    Worker,
    worker_method,
)
from max.experimental.cascade.core.pipeline_method import (
    _pipeline_method_scope,
)
from max.experimental.cascade.deployment.context_config import (
    CONTEXT_FACTORIES,
    ContextConfig,
    GrpcContextFactory,
    HttpContextFactory,
    Transport,
)
from max.experimental.cascade.deployment.routing import (
    HybridRuntime,
    RoundRobinPool,
)
from max.experimental.cascade.grpc_runtime import (
    GrpcRuntimeClient,
    SubprocGrpcRuntimeClient,
)
from max.experimental.cascade.http_runtime import (
    HttpRuntimeProxy,
    SubprocHttpRuntime,
)
from max.experimental.cascade.pipelines.dummy_textgen import (
    build_dummy_textgen_pipeline,
)


class _PidWorker(Worker):
    """Worker that reports the OS PID of the process it runs in."""

    def __init__(self, hint: str = "cpu") -> None:
        super().__init__(deploy_hints=[hint])

    @worker_method()
    async def pid(self) -> int:
        """Return the PID of the worker process handling this call."""
        return os.getpid()


def test_default_config() -> None:
    """Defaults match the prototype: 2 local CPU workers over HTTP."""
    config = ContextConfig()
    assert config.transport == "http"
    assert config.local_cpu_workers == 2
    assert config.local_gpu_workers is None
    assert config.remote_cpu_workers == ()
    assert config.remote_gpu_workers == ()


def test_default_context_pools() -> None:
    """The default context has only a subprocess-backed ``cpu`` pool."""
    runtime = ContextConfig().build_context()
    assert isinstance(runtime, HybridRuntime)
    assert set(runtime._runtimes) == {"cpu"}
    assert isinstance(runtime._runtimes["cpu"], RoundRobinPool)


def test_transport_factories_registered() -> None:
    """Every declared transport has a factory."""
    assert set(CONTEXT_FACTORIES) == {"http", "grpc"}


def test_http_factory_builds_runtimes() -> None:
    """The HTTP factory launches subprocess locals and connects remotes."""
    factory = HttpContextFactory()
    local = factory.local_runtimes(2)
    assert len(local) == 2
    assert all(isinstance(rt, SubprocHttpRuntime) for rt in local)
    remote = factory.remote_runtimes(
        ["http://host-a:9001", "http://host-b:9001"]
    )
    assert len(remote) == 2
    assert all(isinstance(rt, HttpRuntimeProxy) for rt in remote)


def test_grpc_factory_strips_scheme() -> None:
    """The gRPC factory launches subprocess locals and strips ``grpc://``."""
    factory = GrpcContextFactory()
    local = factory.local_runtimes(1)
    assert isinstance(local[0], SubprocGrpcRuntimeClient)
    (remote,) = factory.remote_runtimes(["grpc://host-a:9001"])
    assert isinstance(remote, GrpcRuntimeClient)
    assert remote.target == "host-a:9001"


def test_build_context_pools_by_device() -> None:
    """Configured worker counts produce one pool per used device class."""
    config = ContextConfig(
        transport="grpc", local_cpu_workers=2, local_gpu_workers=1
    )
    runtime = config.build_context()
    assert isinstance(runtime, HybridRuntime)
    assert set(runtime._runtimes) == {"cpu", "gpu"}


@pytest.mark.asyncio
async def test_zero_workers_run_every_worker_in_process() -> None:
    """With no CPU workers, cpu and gpu workers run here."""
    config = ContextConfig(transport="grpc", local_cpu_workers=0)
    async with config.open_context() as runtime, _pipeline_method_scope():
        assert isinstance(runtime, HybridRuntime)
        assert set(runtime._runtimes) == {"cpu", "gpu"}
        assert isinstance(runtime._runtimes["cpu"], LocalRuntime)
        assert runtime._runtimes["gpu"] is runtime._runtimes["cpu"]
        for hint in ("cpu", "gpu"):
            proxy = await runtime.deploy(_PidWorker(hint))
            assert await (await proxy.pid()) == os.getpid()


def test_zero_local_workers_with_remote_workers_use_remote_pools() -> None:
    config = ContextConfig(
        transport="grpc",
        local_cpu_workers=0,
        local_gpu_workers=0,
        remote_cpu_workers=("a:1",),
        remote_gpu_workers=("b:1",),
    )
    runtime = config.build_context()
    assert isinstance(runtime, HybridRuntime)
    assert set(runtime._runtimes) == {"cpu", "gpu"}
    for pool in runtime._runtimes.values():
        assert isinstance(pool, RoundRobinPool)


@pytest.mark.asyncio
async def test_rust_pool_runs_rust_workers_in_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``rust_pool`` adds a ``rust`` pool built from ``local_rust_runtime``."""
    stub = types.ModuleType("local_rust_runtime")
    monkeypatch.setattr(stub, "RustRuntime", LocalRuntime, raising=False)
    monkeypatch.setitem(sys.modules, "local_rust_runtime", stub)
    config = ContextConfig(local_cpu_workers=0)
    async with (
        config.open_context(rust_pool=True) as runtime,
        _pipeline_method_scope(),
    ):
        assert isinstance(runtime, HybridRuntime)
        assert set(runtime._runtimes) == {"cpu", "gpu", "rust"}
        rust_pool = runtime._runtimes["rust"]
        assert isinstance(rust_pool, LocalRuntime)
        assert rust_pool is not runtime._runtimes["cpu"]
        proxy = await runtime.deploy(_PidWorker("rust"))
        assert await (await proxy.pid()) == os.getpid()
        mixed = _PidWorker("gpu")
        mixed.deploy_hints.append("rust")
        await runtime.deploy(mixed)
        assert list(rust_pool._workers) == ["_PidWorker-0", "_PidWorker-1"]


@pytest.mark.parametrize(
    "config",
    [
        ContextConfig(),
        ContextConfig(local_cpu_workers=0, remote_cpu_workers=("a:1",)),
    ],
)
def test_rust_pool_requires_zero_cpu_workers(config: ContextConfig) -> None:
    with pytest.raises(ValueError, match="--cascade-local-cpu-workers 0"):
        config.build_context(rust_pool=True)


@pytest.mark.parametrize(
    "config",
    [
        ContextConfig(local_cpu_workers=0, local_gpu_workers=1),
        ContextConfig(local_cpu_workers=0, remote_gpu_workers=("b:1",)),
    ],
)
def test_no_cpu_workers_rejects_gpu_workers(config: ContextConfig) -> None:
    with pytest.raises(ValueError, match="GPU workers can't run"):
        config.build_context()


def test_build_context_rejects_negative_counts() -> None:
    """Negative worker counts are rejected."""
    config = ContextConfig(local_cpu_workers=-1)
    with pytest.raises(ValueError, match=">= 0"):
        config.build_context()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["grpc", "http"])
async def test_open_context_runs_pipeline(transport: Transport) -> None:
    """``open_context`` runs the dummy pipeline across cpu + gpu worker pools."""
    config = ContextConfig(
        transport=transport, local_cpu_workers=1, local_gpu_workers=1
    )
    async with config.open_context() as runtime:
        pipeline = await build_dummy_textgen_pipeline()
        await pipeline.deploy(runtime)

        req = GenAIRequest(
            messages=[ChatMessage.text("user", "hello, ")],
            text=TextGenOptions(num_tokens=5),
        )
        tokens = [token async for token in pipeline.generate(req)]

    assert tokens == [GenAITextChunk(text="A", num_tokens=1)] * 5


@pytest.mark.asyncio
async def test_cpu_pool_spreads_calls_across_worker_processes() -> None:
    """A cpu-hinted worker fans out across every local cpu worker process.

    This is the property the serve CLI relies on to parallelize the non-GPU
    pipeline stages (tokenization/detokenization) across multiple CPU workers:
    deploying one worker onto an ``N``-way cpu pool round-robins its calls over
    ``N`` distinct worker processes.
    """
    n_workers = 3
    config = ContextConfig(local_cpu_workers=n_workers, local_gpu_workers=0)
    async with config.open_context() as runtime, _pipeline_method_scope():
        proxy = await runtime.deploy(_PidWorker())
        pids = {await (await proxy.pid()) for _ in range(4 * n_workers)}

    # Every cpu worker process handled at least one call.
    assert len(pids) == n_workers


@pytest.mark.asyncio
async def test_open_context_runs_pipeline_multi_cpu_worker() -> None:
    """The dummy pipeline serves text with multiple cpu (tokenizer) workers.

    Mirrors the serve CLI's default topology for a real model: the cpu-hinted
    tokenizer is replicated across several cpu worker processes while the
    gpu-hinted transformer runs on its own worker. Several requests are served
    to exercise the round-robin dispatch across the cpu workers.
    """
    config = ContextConfig(
        transport="http", local_cpu_workers=3, local_gpu_workers=1
    )
    async with config.open_context() as runtime:
        pipeline = await build_dummy_textgen_pipeline()
        await pipeline.deploy(runtime)

        req = GenAIRequest(
            messages=[ChatMessage.text("user", "hello, ")],
            text=TextGenOptions(num_tokens=5),
        )
        for _ in range(2 * 3):
            tokens = [token async for token in pipeline.generate(req)]
            assert tokens == [GenAITextChunk(text="A", num_tokens=1)] * 5
