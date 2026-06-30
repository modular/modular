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
"""End-to-end tests parameterized over every Cascade :py:class:`Runtime`.

Runs the same suite (scalar call, streaming call, dummy text-gen pipeline,
varying generation lengths) against:

* :py:class:`LocalRuntime` -- in-process baseline.
* :py:class:`SubprocHttpRuntime` -- uvicorn HTTP server in a child process,
  :py:class:`HttpRuntimeProxy` client over a unix domain socket.

The dummy text-gen pipeline chains
``tokenizer.encode -> transformer.decode -> tokenizer.decode_streaming``,
which forces the runtime to forward intermediate result IDs between workers
(p2p) rather than round-tripping every value through the client.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import socket
import sys
import uuid
from collections.abc import AsyncIterator, Callable

import pytest
from max.experimental.cascade import (
    GenerateRequest,
    LocalRuntime,
    Runtime,
    Worker,
    worker_method,
)
from max.experimental.cascade.core.pipeline_method import (
    _pipeline_method_scope,
)
from max.experimental.cascade.grpc_runtime import SubprocGrpcRuntimeClient
from max.experimental.cascade.grpc_runtime import server as grpc_server
from max.experimental.cascade.grpc_runtime.client import GrpcRuntimeClient
from max.experimental.cascade.http_runtime import SubprocHttpRuntime
from max.experimental.cascade.http_runtime import server as http_server
from max.experimental.cascade.http_runtime.client import HttpRuntimeProxy
from max.experimental.cascade.http_runtime.subproc import (
    _wait_until_alive as _http_wait_until_alive,
)
from max.serve.process_control import subprocess_manager
from max.tests.tests.cascade.dummy_textgen import (
    build_dummy_textgen_pipeline,
)

# Each entry is a zero-arg factory whose ``async with`` yields a connected
# :py:class:`Runtime`. Parameterizing the fixture by factory (rather than by
# instance) keeps each test case lifecycle-isolated.
_RUNTIME_FACTORIES: list[Callable[[], Runtime]] = [
    LocalRuntime,
    SubprocGrpcRuntimeClient,
    SubprocHttpRuntime,
]


@pytest.fixture(
    params=_RUNTIME_FACTORIES,
    ids=lambda factory: factory.__name__,
)
async def runtime(
    request: pytest.FixtureRequest,
) -> AsyncIterator[Runtime]:
    """Yield an opened runtime for each factory under test."""
    factory: Callable[[], Runtime] = request.param
    async with factory() as rt:
        yield rt


class _Echo(Worker):
    """Minimal worker covering scalar and streaming worker methods."""

    @worker_method()
    async def add(self, a: int, b: int) -> int:
        return a + b

    @worker_method()
    async def count(self, n: int) -> AsyncIterator[int]:
        for i in range(n):
            yield i


class _Failing(Worker):
    """Worker whose methods fail in various ways."""

    @worker_method()
    async def raise_value_error(self, message: str) -> str:
        raise ValueError(message)

    @worker_method()
    async def exit_hard(self) -> str:
        # Ensure exit happens after server is setup, instead of
        # racing between server shutdown and `sys.exit(1)`.
        await asyncio.sleep(3)
        sys.exit(1)

    @worker_method()
    async def slow_ok(self) -> str:
        await asyncio.sleep(10)
        return "done"


@pytest.mark.asyncio
async def test_scalar_call(runtime: Runtime) -> None:
    """A coroutine worker method round-trips its scalar return value.

    Two awaits per terminal call: the first gets the :py:class:`Result`
    handle, the second resolves it. Code inside a ``@pipeline_method``
    body skips the second await -- the consuming worker method's
    ``MaybeAsync`` argument resolution does it automatically.
    """
    async with _pipeline_method_scope():
        echo = await runtime.deploy(_Echo())

        add_handle = await echo.add(2, 3)
        assert await add_handle == 5

        add_handle = await echo.add(10, -4)
        assert await add_handle == 6


@pytest.mark.asyncio
async def test_streaming_call(runtime: Runtime) -> None:
    """An async-generator worker method streams items inline."""
    async with _pipeline_method_scope():
        echo = await runtime.deploy(_Echo())
        items = [item async for item in await echo.count(5)]
        assert items == [0, 1, 2, 3, 4]


@pytest.mark.asyncio
async def test_textgen_pipeline(runtime: Runtime) -> None:
    """Dummy text-gen pipeline runs end-to-end, exercising p2p forwarding."""
    pipeline = await build_dummy_textgen_pipeline()
    await pipeline.deploy(runtime)

    req = GenerateRequest(num_tokens=5)
    # ``generate_text`` is decorated with ``@pipeline_method`` so it owns
    # its own scope; the test doesn't need one here.
    tokens = [token async for token in pipeline.generate_text(req, "hello, ")]

    assert len(tokens) == 5
    assert all(token == "A" for token in tokens)


@pytest.mark.asyncio
async def test_textgen_different_lengths(runtime: Runtime) -> None:
    """Repeated generations with different lengths reuse the deployed workers."""
    pipeline = await build_dummy_textgen_pipeline()
    await pipeline.deploy(runtime)

    for num_tokens in [1, 3, 10]:
        request = GenerateRequest(num_tokens=num_tokens)
        tokens = [
            token async for token in pipeline.generate_text(request, "test")
        ]
        assert len(tokens) == num_tokens


# ---------------------------------------------------------------------------
# Cross-runtime data forwarding
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grpc_cross_runtime_result_forwarding() -> None:
    """Two GrpcRuntimeClient instances on separate unix sockets communicate peer-to-peer.

    Each server is a standalone subprocess; the clients are the bare
    :py:class:`GrpcRuntimeClient` (no :py:class:`SubprocGrpcRuntimeClient`
    wrapper). A ``Result`` produced on server A is forwarded as an arg to a
    worker on server B; server B dials server A's socket directly to fetch
    the value via :py:func:`dial` (an unentered ``GrpcRuntimeClient``).
    """
    sock_a = f"/tmp/max-test-{uuid.uuid4().hex[:8]}.sock"
    sock_b = f"/tmp/max-test-{uuid.uuid4().hex[:8]}.sock"
    bind_a, target_a = f"unix://{sock_a}", f"unix:{sock_a}"
    bind_b, target_b = f"unix://{sock_b}", f"unix:{sock_b}"

    try:
        async with (
            subprocess_manager("gRPC server A") as proc_a,
            subprocess_manager("gRPC server B") as proc_b,
        ):
            proc_a.start(grpc_server.serve, bind_a)
            proc_b.start(grpc_server.serve, bind_b)

            async with (
                GrpcRuntimeClient(target_a) as rt_a,
                GrpcRuntimeClient(target_b) as rt_b,
                _pipeline_method_scope(),
            ):
                worker_a = await rt_a.deploy(_Echo())
                worker_b = await rt_b.deploy(_Echo())

                result_on_a = await worker_a.add(10, 5)

                # Server B's result store is empty — it cannot fetch a result
                # that lives on server A.
                with pytest.raises(RuntimeError):
                    await rt_b.get_result(result_on_a.result_id)

                # Pass the Result from server A as an arg to a worker on server B.
                # Server B dials server A's unix socket to resolve it.
                result_on_b = await worker_b.add(result_on_a, 3)
                assert await result_on_b == 18
    finally:
        for path in (sock_a, sock_b):
            with contextlib.suppress(FileNotFoundError):
                os.unlink(path)


@pytest.mark.asyncio
async def test_http_cross_runtime_result_forwarding() -> None:
    """Two HttpRuntimeProxy instances on separate TCP ports communicate peer-to-peer.

    Each server is a standalone subprocess bound to a free loopback port;
    the clients are the bare :py:class:`HttpRuntimeProxy` (no
    :py:class:`SubprocHttpRuntime` wrapper). A ``Result`` produced on server A
    is forwarded as an arg to a worker on server B; server B dials server A's
    port to fetch the value (``SubprocHttpRuntime.__reduce__`` serializes to
    just the address, so the ``Result`` is a plain URL + result_id pair).
    """

    def _free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    addr_a = f"http://127.0.0.1:{_free_port()}"
    addr_b = f"http://127.0.0.1:{_free_port()}"

    async with (
        subprocess_manager("HTTP server A") as proc_a,
        subprocess_manager("HTTP server B") as proc_b,
    ):
        proc_a.start(http_server.serve, addr_a)
        proc_b.start(http_server.serve, addr_b)

        # HttpRuntimeProxy.__aenter__ opens the session but does not wait for
        # the server — poll /alive manually before deploying workers.
        async with (
            HttpRuntimeProxy(addr_a) as rt_a,
            HttpRuntimeProxy(addr_b) as rt_b,
            _pipeline_method_scope(),
        ):
            async with rt_a.session() as s:
                await _http_wait_until_alive(s, rt_a._base_url)
            async with rt_b.session() as s:
                await _http_wait_until_alive(s, rt_b._base_url)

            worker_a = await rt_a.deploy(_Echo())
            worker_b = await rt_b.deploy(_Echo())

            result_on_a = await worker_a.add(10, 5)

            # Server B's result store is empty — it cannot fetch a result
            # that lives on server A.
            with pytest.raises(KeyError):
                await rt_b.get_result(result_on_a.result_id)

            # Pass the Result from server A as an arg to a worker on server B.
            # Server B dials server A's TCP port to resolve it.
            result_on_b = await worker_b.add(result_on_a, 3)
            assert await result_on_b == 18


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_exception(runtime: Runtime) -> None:
    """An exception raised inside a worker method propagates to the caller."""
    async with _pipeline_method_scope():
        worker = await runtime.deploy(_Failing())
        handle = await worker.raise_value_error("kaboom")
        # The HTTP protocol can raise specific errors since it is mostly for
        # Python testing, but the GrpcRuntimeClient raises a generic
        # RuntimeError because it has to serialize errors from
        # workers implemented in other languages.
        with pytest.raises((ValueError, RuntimeError), match="kaboom"):
            await handle


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runtime_cls",
    [SubprocHttpRuntime, SubprocGrpcRuntimeClient],
    ids=lambda c: c.__name__,
)
async def test_worker_sys_exit(runtime_cls: type) -> None:
    """A subprocess worker that calls ``sys.exit(1)`` surfaces an error.

    Only tested against subprocess runtimes because ``sys.exit`` in-process
    raises :py:class:`SystemExit` (a :py:class:`BaseException`) which tears
    through the local task group destructively. In a subprocess runtime the
    child dies and the transport layer reports the failure as a normal
    exception.
    """
    async with runtime_cls() as rt, _pipeline_method_scope():
        worker = await rt.deploy(_Failing())
        with pytest.raises(Exception):
            handle = await worker.exit_hard()
            await handle


@pytest.mark.asyncio
async def test_pipeline_exit_cancels_remote_work(runtime: Runtime) -> None:
    """Exiting the pipeline scope cancels in-flight work and cleans up results.

    Schedules a long-running worker call, then raises to force the
    pipeline scope closed. After the scope exits, the result handle
    should be invalidated -- fetching it must fail, proving the runtime
    released the in-flight task and its result buffer.

    """
    handle = None

    with pytest.raises(RuntimeError, match="orchestrator error"):
        async with _pipeline_method_scope():
            worker = await runtime.deploy(_Failing())
            handle = await worker.slow_ok()
            # Simulate the pipeline orchestrator failing somehow
            raise RuntimeError("orchestrator error")

    assert handle is not None
    # Give the server time to detect the client disconnect
    await asyncio.sleep(1)
    # The remote result should be deleted now
    # The HTTP protocol can raise specific errors since it is mostly for
    # Python testing, but the GrpcRuntimeClient raises a generic
    # RuntimeError because it has to serialize errors from
    # workers implemented in other languages.
    with pytest.raises((KeyError, RuntimeError), match="Unknown result id"):
        await handle
