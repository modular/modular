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
"""Cancels a non-streaming request when its client disconnects."""

from __future__ import annotations

import logging

from fastapi import Request

logger = logging.getLogger("max.serve")

# nginx's "client closed request"; nobody reads it, but access logs and the
# orchestrator's 499 then agree on what happened.
CLIENT_CLOSED_REQUEST = 499


class ClientDisconnected(Exception):
    """The client hung up before its non-streaming response was ready.

    The ``request_session`` middleware answers it with a 499, so a route
    needs no handler of its own.
    """


async def raise_on_disconnect(request: Request) -> None:
    """Raises :class:`ClientDisconnected` once the client hangs up.

    Starlette cancels a streaming response when its client disconnects, but
    never a plain endpoint, so without this watch a non-streaming request
    keeps its batch slot and KV blocks until it reaches ``max_tokens``. Run in
    a :class:`~max.support._taskgroups.CancelGroup` beside the completion, the
    raise cancels the completion, which unwinds the model worker stream and
    sends the cancel that releases the request whether it is still queued or
    already decoding.

    Start the watch only after the route has read the request body: it reads
    and discards every message until the disconnect, so started earlier it
    would swallow the body.

    Args:
        request: The request whose client to watch.

    Raises:
        ClientDisconnected: When the client disconnects.
    """
    # Not request.is_disconnected(): behind a BaseHTTPMiddleware (the
    # request_session middleware) its non-blocking receive drops the
    # disconnect message, so it never reports one. A blocking receive does.
    try:
        while (await request.receive())["type"] != "http.disconnect":
            pass
    except Exception:
        # A failed watch must not abandon a live request.
        logger.debug(
            "Disconnect watch failed for request %s",
            request.state.request_id,
            exc_info=True,
        )
        return
    raise ClientDisconnected
