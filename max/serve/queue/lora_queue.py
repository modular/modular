# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

"""LoRA queue implementation for managing LoRA adapter loading/unloading."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import queue
from collections.abc import Generator

import msgspec
from max.interfaces import LoRARequest, LoRAResponse, RequestID
from max.serve.queue.zmq_queue import ZmqPullSocket, ZmqPushSocket

logger = logging.getLogger("max.serve")


class LoRAQueue:
    """Queue for managing LoRA adapter load/unload/list requests."""

    def __init__(
        self,
        request_zmq_endpoint: str,
        response_zmq_endpoint: str,
    ):
        self._request_socket = ZmqPushSocket[tuple[RequestID, LoRARequest]](
            endpoint=request_zmq_endpoint,
            serialize=msgspec.msgpack.Encoder().encode,
        )
        self._response_socket = ZmqPullSocket[tuple[RequestID, LoRAResponse]](
            endpoint=response_zmq_endpoint,
            deserialize=msgspec.msgpack.Decoder(
                type=tuple[RequestID, LoRAResponse]
            ).decode,
        )

        self.pending_out_queues: dict[
            RequestID, asyncio.Queue[LoRAResponse]
        ] = {}

    @contextlib.contextmanager
    def open_channel(
        self, req_id: RequestID, request: LoRARequest
    ) -> Generator[asyncio.Queue[LoRAResponse], None, None]:
        try:
            if req_id in self.pending_out_queues:
                raise RuntimeError(
                    f"Detected multiple requests with `req_id` set to {req_id}. "
                    "This WILL lead to unexpected behavior! "
                    "Please ensure that the `req_id` is unique for each request."
                )

            out_queue: asyncio.Queue[LoRAResponse] = asyncio.Queue()
            self.pending_out_queues[req_id] = out_queue

            # put_nowait will fail if the request_push_socket is unavailable
            # this will immediately trigger the finally block, resulting in
            # the request being purged, and returned without result.
            self._request_socket.put_nowait((req_id, request))
            yield out_queue
        finally:
            del self.pending_out_queues[req_id]

    async def get_response(
        self, req_id: RequestID, request: LoRARequest
    ) -> LoRAResponse:
        with self.open_channel(req_id, request) as queue:
            return await queue.get()

    async def response_worker(self) -> None:
        """
        Continuously processes responses from the remote worker process.

        This method runs in a loop, pulling responses from the response socket and routing them
        to the appropriate pending queues.

        """
        try:
            while True:
                try:
                    request_id, response = self._response_socket.get_nowait()
                    if request_id in self.pending_out_queues:
                        await self.pending_out_queues[request_id].put(response)
                except queue.Empty:
                    await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            raise
        finally:
            logger.debug("Terminating response worker [self=%s]", os.getpid())
