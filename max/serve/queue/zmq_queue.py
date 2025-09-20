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

"""Multi-process queue based on ZeroMQ. Tested for SPSC case."""

from __future__ import annotations

import logging
import queue
import tempfile
import uuid
import weakref
from typing import Any, Callable, Generic, NewType, TypeVar

import psutil
import zmq
from max.interfaces import msgpack_numpy_decoder, msgpack_numpy_encoder
from max.interfaces.queue import MAXPullQueue, MAXPushQueue

logger = logging.getLogger("max.serve")

T = TypeVar("T")

Request = TypeVar("Request")
Reply = TypeVar("Reply")


def generate_zmq_ipc_path() -> str:
    """Generate a unique ZMQ IPC path."""
    base_rpc_path = tempfile.gettempdir()
    return f"ipc://{base_rpc_path}/{uuid.uuid4()}"


def _is_valid_zmq_address(address: str) -> bool:
    """
    Check if a ZMQ address is valid.
    """
    # Check for supported protocols
    if not address.startswith(("tcp://", "ipc://", "inproc://")):
        return False

    # Protocol-specific validation
    if address.startswith("tcp://"):
        # TCP requires host:port format
        parts = address[6:].split(":")
        if len(parts) != 2:
            return False
        # Check if port is numeric
        try:
            port = int(parts[1])
            return 1 <= port <= 65535
        except ValueError:
            return False
    elif address.startswith("ipc://"):
        # IPC requires a path after the protocol
        return len(address) > 6
    elif address.startswith("inproc://"):
        # inproc requires a name after the protocol
        return len(address) > 9

    return True


# Adapted from:
#  - vllm: https://github.com/vllm-project/vllm/blob/46c759c165a5a985ce62f019bf684e4a6109e41c/vllm/utils.py#L2093
#  - sglang: https://github.com/sgl-project/sglang/blob/efc52f85e2d5c9b31545d4092f2b361b6ff04d67/python/sglang/srt/utils.py#L783
def _open_zmq_socket(path: str, mode: int) -> zmq.Socket[bytes]:
    """Open a ZMQ socket with the proper bind/connect semantics."""
    mem = psutil.virtual_memory()

    # Grab the singleton global zmq ctx
    zmq_ctx = zmq.Context.instance(io_threads=2)
    socket = zmq_ctx.socket(mode)

    # Calculate buffer size based on system memory
    GIB = 1024**3
    total_mem_gb = mem.total / GIB
    available_mem_gb = mem.available / GIB
    # For systems with substantial memory (>32GB total, >16GB available):
    # - Set a large 0.5GB buffer to improve throughput
    # For systems with less memory:
    if total_mem_gb > 32 and available_mem_gb > 16:
        buf_size = int(0.5 * GIB)
    else:
        buf_size = -1

    # Configure socket options based on type
    if mode == zmq.PULL:
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.RCVBUF, buf_size)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(path)
    elif mode == zmq.PUSH:
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.SNDBUF, buf_size)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(path)
    elif mode == zmq.ROUTER:
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.RCVBUF, buf_size)
        socket.setsockopt(zmq.SNDBUF, buf_size)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
        socket.bind(path)
    elif mode == zmq.DEALER:
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.RCVBUF, buf_size)
        socket.setsockopt(zmq.SNDBUF, buf_size)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(path)
    else:
        raise ValueError(f"Unknown Socket Mode: {mode}")

    return socket


def _put_helper(func: Callable[[], Any]) -> None:
    while True:
        try:
            func()

            # Exit since we succeeded
            break
        except zmq.ZMQError as e:
            # If we get EAGAIN, we just try again.
            # This could be due to:
            #   - the pull socket not being opened yet
            #   - a full queue
            if e.errno == zmq.EAGAIN:
                continue
            raise RuntimeError("Failed to put message on ZMQ socket") from e


def _get_helper(func: Callable[[], Any]) -> Any:
    try:
        msg = func()
    except zmq.ZMQError as e:
        if e.errno == zmq.EAGAIN:
            raise queue.Empty() from e
        raise RuntimeError("Failed to get message on ZMQ socket") from e
    return msg


class ZmqConfig(Generic[T]):
    def __init__(self, payload_type: Any) -> None:
        self._payload_type = payload_type
        self._endpoint = generate_zmq_ipc_path()

    def push(self) -> ZmqPushSocket[T]:
        return ZmqPushSocket(
            endpoint=self._endpoint, payload_type=self._payload_type
        )

    def pull(self) -> ZmqPullSocket[T]:
        return ZmqPullSocket(
            endpoint=self._endpoint, payload_type=self._payload_type
        )

    def pair(self) -> tuple[ZmqPushSocket[T], ZmqPullSocket[T]]:
        return self.push(), self.pull()


class ZmqSocket:
    def __init__(
        self,
        *,
        endpoint: str,
        mode: int,
    ) -> None:
        if not _is_valid_zmq_address(endpoint):
            raise ValueError(f"Invalid endpoint: {endpoint}")
        self._endpoint = endpoint
        self._socket = _open_zmq_socket(endpoint, mode)
        self._finalize = weakref.finalize(self, self.close)
        self._is_closed = False

    def close(self) -> None:
        """Clean up resources during garbage collection."""
        if not self._is_closed:
            self._is_closed = True
            self._socket.close()


class ZmqPushSocket(Generic[T], ZmqSocket, MAXPushQueue[T]):
    def __init__(self, *, endpoint: str, payload_type: Any) -> None:
        self._serialize = msgpack_numpy_encoder()
        super().__init__(endpoint=endpoint, mode=zmq.PUSH)

    def put_nowait(self, msg: T) -> None:
        if self._is_closed:
            raise RuntimeError("Socket is closed")
        serialized_msg = self._serialize(msg)
        _put_helper(
            lambda: self._socket.send(serialized_msg, flags=zmq.NOBLOCK)
        )


class ZmqPullSocket(Generic[T], ZmqSocket, MAXPullQueue[T]):
    def __init__(self, *, endpoint: str, payload_type: Any) -> None:
        self._deserialize = msgpack_numpy_decoder(payload_type)
        super().__init__(endpoint=endpoint, mode=zmq.PULL)

    def get_nowait(self) -> T:
        if self._is_closed:
            raise RuntimeError("Socket is closed")
        serialized_msg = _get_helper(
            lambda: self._socket.recv(flags=zmq.NOBLOCK)
        )
        msg = self._deserialize(serialized_msg)
        return msg


ClientIdentity = NewType("ClientIdentity", bytes)


class ZmqRouterSocket(Generic[Request, Reply], ZmqSocket):
    def __init__(
        self, *, endpoint: str, request_type: Any, reply_type: Any
    ) -> None:
        self._endpoint = endpoint
        self._serialize = msgpack_numpy_encoder()
        self._deserialize = msgpack_numpy_decoder(request_type)
        super().__init__(endpoint=endpoint, mode=zmq.ROUTER)

    def send_reply_nowait(self, msg: Reply, identity: ClientIdentity) -> None:
        if self._is_closed:
            raise RuntimeError("Socket is closed")
        serialized_msg = self._serialize(msg)
        _put_helper(
            lambda: self._socket.send_multipart(
                [identity, serialized_msg], flags=zmq.NOBLOCK
            )
        )

    def recv_request_nowait(self) -> tuple[Request, ClientIdentity]:
        if self._is_closed:
            raise RuntimeError("Socket is closed")
        identity, serialized_msg = _get_helper(
            lambda: self._socket.recv_multipart(flags=zmq.NOBLOCK)
        )
        msg = self._deserialize(serialized_msg)
        return msg, ClientIdentity(identity)


class ZmqDealerSocket(Generic[Request, Reply], ZmqSocket):
    def __init__(
        self, *, endpoint: str, request_type: Any, reply_type: Any
    ) -> None:
        self._endpoint = endpoint
        self._serialize = msgpack_numpy_encoder()
        self._deserialize = msgpack_numpy_decoder(reply_type)
        super().__init__(endpoint=endpoint, mode=zmq.DEALER)

    def send_request_nowait(self, msg: Request) -> None:
        if self._is_closed:
            raise RuntimeError("Socket is closed")
        serialized_msg = self._serialize(msg)
        _put_helper(
            lambda: self._socket.send(serialized_msg, flags=zmq.NOBLOCK)
        )

    def recv_reply_nowait(self) -> Reply:
        if self._is_closed:
            raise RuntimeError("Socket is closed")
        serialized_msg = _get_helper(
            lambda: self._socket.recv(flags=zmq.NOBLOCK)
        )
        msg = self._deserialize(serialized_msg)
        return msg
