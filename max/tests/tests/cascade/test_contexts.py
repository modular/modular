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
"""Smoke tests for the in-process LocalRuntime."""

from __future__ import annotations

import pytest
from max.experimental.cascade import LocalRuntime, Worker, worker_method


class _CPUEchoWorker(Worker):
    def __init__(self, prefix: str) -> None:
        super().__init__(deploy_hints=["cpu"])
        self.prefix = prefix

    @worker_method()
    async def echo(self, value: str) -> str:
        return f"{self.prefix}:{value}"


@pytest.mark.asyncio
async def test_local_runtime_deploys_worker() -> None:
    async with LocalRuntime().open() as runtime:
        proxy = await runtime.deploy(_CPUEchoWorker("local"))
        assert await proxy.echo("ok") == "local:ok"
