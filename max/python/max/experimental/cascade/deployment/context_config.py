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
"""Config-driven construction of a hybrid Cascade deployment context.

The "context" a worker deploys into is a
:class:`~max.experimental.cascade.core.Runtime`. :class:`ContextConfig` builds a
device-routed :class:`~max.experimental.cascade.routing.HybridRuntime`:
``cpu``-hinted workers deploy into a round-robin pool of ``cpu`` worker
runtimes and ``gpu``-hinted workers into a pool of ``gpu`` worker runtimes. Each
pool combines locally-launched worker subprocesses (``local_*_workers``) with
connections to already-running remote workers (``remote_*_workers``), over the
selected ``transport``. With no CPU workers, local or remote, every worker
runs in this process instead.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Literal

from max.experimental.cascade.core import Runtime
from max.experimental.cascade.core.local_runtime import LocalRuntime
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
from max.experimental.cascade.telemetry import (
    OTEL_EXPORTER_OTLP_METRICS_ENDPOINT,
)
from pydantic import BaseModel, Field

Transport = Literal["http", "grpc"]


class ContextFactory:
    """Build local and remote worker runtimes for one transport family."""

    def local_runtimes(self, count: int) -> list[Runtime]:
        """Build ``count`` locally-launched worker-subprocess runtimes."""
        raise NotImplementedError

    def remote_runtimes(self, endpoints: Sequence[str]) -> list[Runtime]:
        """Build client runtimes for already-running remote workers."""
        raise NotImplementedError


class HttpContextFactory(ContextFactory):
    """Build pickle-over-HTTP worker runtimes.

    Local development and testing only: the HTTP server is unauthenticated
    and unpickles request bodies, so never point ``http://`` worker URLs at
    a server reachable by untrusted peers.
    """

    def local_runtimes(self, count: int) -> list[Runtime]:
        """Launch ``count`` HTTP worker subprocesses."""
        return [SubprocHttpRuntime() for _ in range(count)]

    def remote_runtimes(self, endpoints: Sequence[str]) -> list[Runtime]:
        """Connect to remote HTTP workers by ``http://`` / ``unix://`` URL."""
        runtimes: list[Runtime] = []
        for endpoint in endpoints:
            address = endpoint.strip()
            if not address:
                raise ValueError("Worker endpoint cannot be empty")
            runtimes.append(HttpRuntimeProxy(address))
        return runtimes


class GrpcContextFactory(ContextFactory):
    """Build gRPC worker runtimes."""

    def local_runtimes(self, count: int) -> list[Runtime]:
        """Launch ``count`` gRPC worker subprocesses."""
        return [SubprocGrpcRuntimeClient() for _ in range(count)]

    def remote_runtimes(self, endpoints: Sequence[str]) -> list[Runtime]:
        """Connect to remote gRPC workers by target (``grpc://`` optional)."""
        runtimes: list[Runtime] = []
        for endpoint in endpoints:
            target = endpoint.strip()
            if not target:
                raise ValueError("Worker endpoint cannot be empty")
            runtimes.append(GrpcRuntimeClient(target.removeprefix("grpc://")))
        return runtimes


CONTEXT_FACTORIES: dict[Transport, ContextFactory] = {
    "http": HttpContextFactory(),
    "grpc": GrpcContextFactory(),
}


class ContextConfig(BaseModel):
    """Configuration for building a hybrid Cascade deployment context."""

    transport: Transport = "http"
    local_cpu_workers: int = 2
    local_gpu_workers: int | None = Field(
        default=None,
        description=(
            "Number of local GPU worker subprocesses. When unset, the server "
            "starts one per distinct GPU set. Must be unset or 0 when there "
            "are no CPU workers, since every worker then runs in the server "
            "process."
        ),
    )
    remote_cpu_workers: tuple[str, ...] = Field(default_factory=tuple)
    remote_gpu_workers: tuple[str, ...] = Field(default_factory=tuple)
    otlp_metrics_endpoint: str | None = Field(
        default=None,
        description=(
            "OTLP endpoint to export metrics to, for example a Datadog Agent "
            "OTLP receiver. Overrides the standard "
            "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT / "
            "OTEL_EXPORTER_OTLP_ENDPOINT variables, which are used when this "
            "is unset. Metrics are always served on /metrics; they are also "
            "exported over OTLP when one of these resolves."
        ),
    )

    def export_otlp_endpoint(self) -> None:
        """Publishes ``otlp_metrics_endpoint`` into the environment.

        The OTLP exporters resolve their endpoint from the standard
        OpenTelemetry variables, so a setting given on the command line has to
        reach them that way. A no-op when the field is unset, leaving any
        variable already set in place.

        Call before :py:func:`~..telemetry.configure_telemetry`, which
        reads the result.
        """
        if self.otlp_metrics_endpoint:
            os.environ[OTEL_EXPORTER_OTLP_METRICS_ENDPOINT] = (
                self.otlp_metrics_endpoint
            )

    def build_context(self, rust_pool: bool = False) -> Runtime:
        """Build the configured device-routed hybrid runtime.

        Args:
            rust_pool: Add a ``rust`` pool for ``rust``-hinted workers. Needs
                zero CPU workers, local or remote.

        Raises:
            ValueError: If worker counts are negative, GPU workers are set
                while there are no CPU workers, or ``rust_pool`` is set while
                there are CPU workers.
        """
        local_gpu_workers = self.local_gpu_workers or 0
        if self.local_cpu_workers < 0 or local_gpu_workers < 0:
            raise ValueError("Worker counts must be >= 0")

        # With no CPU workers, the tokenizer and parser run in this process,
        # and a GPU worker in another process can't read their results, so
        # every worker runs here and the GPU worker count has no say.
        # TODO(SERVSYS-1335): Remove this in-process mode once workers in
        # other processes can read results from this process.
        if not (self.local_cpu_workers or self.remote_cpu_workers):
            if local_gpu_workers or self.remote_gpu_workers:
                raise ValueError(
                    "GPU workers can't run in other processes when CPU "
                    "workers run in the server process "
                    "(--cascade-local-cpu-workers 0)."
                )
            runtimes: dict[str, Runtime] = {}
            if rust_pool:
                runtimes["rust"] = importlib.import_module(
                    "local_rust_runtime"
                ).RustRuntime()
            local = LocalRuntime()
            runtimes |= {"cpu": local, "gpu": local}
            return HybridRuntime(runtimes)
        if rust_pool:
            raise ValueError(
                "Rust workers need every worker in the server process: set "
                "--cascade-local-cpu-workers 0, with no remote CPU workers."
            )

        factory = CONTEXT_FACTORIES[self.transport]
        cpu_runtimes = factory.local_runtimes(
            self.local_cpu_workers
        ) + factory.remote_runtimes(self.remote_cpu_workers)
        gpu_runtimes = factory.local_runtimes(
            local_gpu_workers
        ) + factory.remote_runtimes(self.remote_gpu_workers)

        pools: dict[str, Runtime] = {}
        if cpu_runtimes:
            pools["cpu"] = RoundRobinPool(cpu_runtimes)
        if gpu_runtimes:
            pools["gpu"] = RoundRobinPool(gpu_runtimes)
        return HybridRuntime(pools)

    @asynccontextmanager
    async def open_context(
        self, rust_pool: bool = False
    ) -> AsyncIterator[Runtime]:
        """Open the configured hybrid runtime, owning its lifecycle."""
        async with self.build_context(rust_pool) as runtime:
            yield runtime
