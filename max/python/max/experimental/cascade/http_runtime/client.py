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
"""Cascade HTTP transport: client-side :py:class:`Runtime` proxy.

Pickle-over-HTTP analogue of the gRPC transport. Each
:py:class:`Runtime` wire primitive maps to one HTTP endpoint.

:py:class:`HttpRuntimeProxy` is a :py:class:`Runtime` backed by a remote
server at a given ``address``. Picklable: ``__getstate__`` returns only
the address, so :py:class:`Result` / :py:class:`ResultIter` handles
carrying an :py:class:`HttpRuntimeProxy` round-trip cleanly across the
wire.

Session model
-------------

The proxy owns a single :py:class:`aiohttp.ClientSession` opened on
``__aenter__``, shared across every RPC the proxy initiates
(``deploy_worker``, ``call_method``, ``get_result``, ``stream_result``,
``get_metrics``). One TCP / unix connection, pooled.
"""

from __future__ import annotations

import asyncio
import pickle
import struct
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlparse

import aiohttp
from max.experimental.cascade.core import Runtime, Worker


async def _read_framed(stream: aiohttp.StreamReader) -> AsyncIterator[bytes]:
    """Yield envelopes off a 4-byte length-prefixed byte stream."""
    while True:
        try:
            header = await stream.readexactly(4)
        except asyncio.IncompleteReadError as exc:
            if not exc.partial:
                return  # 0 bytes left = clean EOF
            raise
        (length,) = struct.unpack(">I", header)
        yield await stream.readexactly(length)


def _connector_for(address: str) -> aiohttp.BaseConnector:
    """Build the right aiohttp connector for an ``http://`` or ``unix://`` URL."""
    parsed = urlparse(address)
    if parsed.scheme == "http":
        return aiohttp.TCPConnector()
    if parsed.scheme == "unix":
        if not parsed.path:
            raise ValueError(f"unix:// address requires a path: {address!r}")
        return aiohttp.UnixConnector(path=parsed.path)
    raise ValueError(f"Unsupported address scheme: {address!r}")


def _base_url_for(address: str) -> str:
    """Return the URL base aiohttp uses to form per-request URLs."""
    parsed = urlparse(address)
    if parsed.scheme == "http":
        return address.rstrip("/")
    if parsed.scheme == "unix":
        # aiohttp's UnixConnector dispatches by socket path; the host
        # portion of the URL is only used to satisfy URL parsing.
        return "http://unix"
    raise ValueError(f"Unsupported address scheme: {address!r}")


class HttpRuntimeProxy(Runtime):
    """Client-side :py:class:`Runtime` backed by an HTTP Cascade server.

    ``address`` is either ``http://host:port`` or ``unix:///path/to.sock``.
    """

    def __init__(self, address: str) -> None:
        super().__init__()
        self.address = address
        self._base_url = _base_url_for(address)
        self._session: aiohttp.ClientSession | None = None

    # -- lifecycle --------------------------------------------------------

    async def __aenter__(self) -> HttpRuntimeProxy:
        await super().__aenter__()
        self._session = await self.enter_async_context(
            aiohttp.ClientSession(
                connector=_connector_for(self.address),
                connector_owner=True,
            )
        )
        return self

    @asynccontextmanager
    async def session(self) -> AsyncIterator[aiohttp.ClientSession]:
        """Yield a connected :py:class:`aiohttp.ClientSession`.

        Inside an ``async with`` block on this proxy the pooled session
        opened by :py:meth:`__aenter__` is reused. Outside that context
        -- e.g. after :py:class:`HttpRuntimeProxy` has been unpickled
        from a wire-side handle and is used directly without being
        explicitly entered -- this yields a fresh single-use session and
        tears it down on exit. Concurrent users still see a single
        in-flight HTTP request per call.
        """
        if self._session is not None:
            yield self._session
            return
        async with aiohttp.ClientSession(
            connector=_connector_for(self.address),
            connector_owner=True,
        ) as session:
            yield session

    # -- pickle ----------------------------------------------------------

    def __getstate__(self) -> object:
        # Only the address travels; the live session is recreated on the
        # other side of the wire when the proxy is re-entered.
        return {"address": self.address}

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.__init__(state["address"])  # type: ignore[misc]

    # -- Runtime wire primitives -----------------------------------------

    async def deploy_worker(self, worker: Worker) -> str:
        """Pickle the worker and register it on the server."""
        # Workers don't carry :py:class:`Result` handles in their state,
        # so plain pickle is fine here.
        async with (
            self.session() as session,
            session.put(
                f"{self._base_url}/worker",
                data=pickle.dumps(worker),
                headers={"Content-Type": "application/pickle"},
            ) as response,
        ):
            response.raise_for_status()
            return (await response.read()).decode()

    @asynccontextmanager
    async def call_method(
        self,
        worker_id: str,
        func: str,
        args: Sequence[object],
        kwargs: Mapping[str, object],
    ) -> AsyncIterator[str]:
        """Open a streaming POST that holds the call alive for the scope.

        The server emits the bound ``result_id`` as the first line and
        then holds the response open until the client disconnects. Exit
        of this context manager closes the response, which is what
        signals the server to cancel the in-flight task and release the
        result buffer.
        """
        async with (
            self.session() as session,
            session.post(
                f"{self._base_url}/worker/{worker_id}/{func}",
                data=pickle.dumps((args, kwargs)),
                headers={"Content-Type": "application/pickle"},
            ) as response,
        ):
            response.raise_for_status()
            result_id_line = await response.content.readline()
            yield result_id_line.decode().rstrip("\n")
            # exiting this context deletes the remote result

    async def get_result(self, result_id: str) -> object:
        """Fetch a single result via the proxy's session."""
        async with (
            self.session() as session,
            session.get(f"{self._base_url}/result/{result_id}") as response,
        ):
            response.raise_for_status()
            payload = await response.read()
        ok, value = pickle.loads(payload)
        if ok:
            return value
        raise value

    async def stream_result(self, result_id: str) -> AsyncIterator[object]:
        """Stream a result via the proxy's session."""
        async with (
            self.session() as session,
            session.get(
                f"{self._base_url}/result/{result_id}/stream"
            ) as response,
        ):
            response.raise_for_status()
            async for payload in _read_framed(response.content):
                ok, value = pickle.loads(payload)
                if ok:
                    yield value
                else:
                    raise value

    async def get_metrics(self) -> str:
        """Fetch Prometheus exposition text from the server."""
        async with (
            self.session() as session,
            session.get(f"{self._base_url}/metrics") as response,
        ):
            response.raise_for_status()
            return await response.text()
