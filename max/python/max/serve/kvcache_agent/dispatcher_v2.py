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
from __future__ import annotations

import queue
from typing import Any, Generic, TypeVar

from max.serve.queue.zmq_queue import ZmqDealerSocket, ZmqRouterSocket

Request = TypeVar("Request")
Reply = TypeVar("Reply")

DispatcherServerV2 = ZmqRouterSocket[Request, Reply]


class DispatcherClientV2(Generic[Request, Reply]):
    def __init__(
        self,
        *,
        bind_addr: str,
        default_dest_addr: str | None,
        request_type: Any,
        reply_type: Any,
    ):
        self._request_type = request_type
        self._reply_type = reply_type

        self._default_dest_address = default_dest_addr
        self._dealers: dict[str, ZmqDealerSocket[Request, Reply]] = {}

    def send_request_nowait(
        self, request: Request, dest_addr: str | None = None
    ) -> None:
        dest_addr = dest_addr or self._default_dest_address
        if dest_addr is None:
            raise ValueError("dest_addr is required")

        if dest_addr not in self._dealers:
            self._dealers[dest_addr] = ZmqDealerSocket[Request, Reply](
                endpoint=dest_addr,
                request_type=self._request_type,
                reply_type=self._reply_type,
            )
        dealer = self._dealers[dest_addr]
        dealer.send_request_nowait(request)

    def recv_reply_nowait(self) -> Reply:
        for dealer in self._dealers.values():
            try:
                reply = dealer.recv_reply_nowait()
            except queue.Empty:
                continue
            return reply
        raise queue.Empty()
