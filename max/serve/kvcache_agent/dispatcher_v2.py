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
from typing import Any, Generic, NewType, TypeVar, Union, cast

import zmq
from max.serve.queue.zmq_queue import (
    ZmqDealerSocket,
    ZmqRouterSocket,
    msgpack_numpy_decoder,
    msgpack_numpy_encoder,
)

Request = TypeVar("Request")
Reply = TypeVar("Reply")


ClientIdentity = NewType("ClientIdentity", bytes)


class DispatcherServerV2(Generic[Request, Reply]):
    def __init__(
        self,
        bind_addr: str,
        request_type: Any,
        reply_type: Any,
    ):
        self.bind_addr = bind_addr

        self._request_type = request_type
        self._reply_type = reply_type

        self._router = ZmqRouterSocket[Union[Request, Reply]](
            bind_addr,
            serialize=msgpack_numpy_encoder(),
            deserialize=msgpack_numpy_decoder(Union[request_type, reply_type]),
        )

    def recv_request_nowait(self) -> tuple[Request, ClientIdentity]:
        identity, request = self._router.recv_multipart_nowait()
        if not isinstance(request, self._request_type):
            raise ValueError(
                f"Received request {request} is not of type {self._request_type}"
            )
        return cast(Request, request), ClientIdentity(identity)

    def send_reply_nowait(self, reply: Reply, identity: ClientIdentity) -> None:
        self._router.send_multipart(identity, reply, flags=zmq.NOBLOCK)


class DispatcherClientV2(Generic[Request, Reply]):
    def __init__(
        self,
        bind_addr: str,
        default_dest_addr: str | None,
        request_type: Any,
        reply_type: Any,
    ):
        self.bind_addr = bind_addr

        self._request_type = request_type
        self._reply_type = reply_type

        self._default_dest_address = default_dest_addr
        self._dealers: dict[str, ZmqDealerSocket[Union[Request, Reply]]] = {}

    def send_request_nowait(
        self, request: Request, dest_addr: str | None = None
    ) -> None:
        dest_addr = dest_addr or self._default_dest_address
        if dest_addr is None:
            raise ValueError("dest_addr is required")

        if dest_addr not in self._dealers:
            self._dealers[dest_addr] = ZmqDealerSocket[Union[Request, Reply]](
                dest_addr,
                bind=False,
                serialize=msgpack_numpy_encoder(),
                deserialize=msgpack_numpy_decoder(
                    Union[self._request_type, self._reply_type]
                ),
            )
        dealer = self._dealers[dest_addr]
        dealer.send_pyobj(request, flags=zmq.NOBLOCK)

    def recv_reply_nowait(self) -> Reply:
        for dealer in self._dealers.values():
            try:
                reply = dealer.recv_pyobj_nowait()
            except queue.Empty:
                continue
            if not isinstance(reply, self._reply_type):
                raise ValueError(
                    f"Received reply {reply} is not of type {self._reply_type}"
                )
            return cast(Reply, reply)
        raise queue.Empty()
