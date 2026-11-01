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
"""Drives an ASGI app through a client that hangs up mid-response.

``TestClient`` reads a response to its end, so it cannot disconnect once the
body has started.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

from fastapi import FastAPI


async def disconnect_after_first_chunk(app: FastAPI, path: str) -> None:
    """Sends a GET to ``path`` and disconnects once a body chunk arrives.

    Args:
        app: The app to call.
        path: The path to request.
    """
    first_chunk_sent = asyncio.Event()
    is_requested = False

    async def receive() -> dict[str, object]:
        nonlocal is_requested
        if not is_requested:
            is_requested = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await first_chunk_sent.wait()
        return {"type": "http.disconnect"}

    async def send(message: Mapping[str, object]) -> None:
        if message["type"] == "http.response.body" and message.get("body"):
            first_chunk_sent.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=30)
