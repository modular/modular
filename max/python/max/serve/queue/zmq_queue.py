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
from collections.abc import Callable
from typing import Any, Generic, NewType, TypeVar

import psutil
import zmq
from max.interfaces import msgpack_numpy_decoder, msgpack_numpy_encoder
from max.interfaces.queue import MAXPullQueue, MAXPushQueue

logger = logging.getLogger("max.serve")

T = TypeVar("T")

Request = TypeVar("Request")
Reply = TypeVar("Reply")

DEFAULT_MSGPACK_NUMPY_ENCODER = msgpack_numpy_encoder(
    use_shared_memory=True,
    shared_memory_threshold=0,
)


def generate_zmq_ipc_path() -> str:
    """Generate a unique ZMQ IPC path."""
    base_rpc_path = tempfile.gettempdir()
    # The full UUID is 36 chars (8-4-4-4-12 hex)
    # However, this may cause the full path to be too long for ZMQ if you append
    # additional characters to it. As such, we truncate the UUID to 18 chars.
    # The chances of collision are still very low, because we don't really make
    # that many ZMQs anyways.
    short_uuid = uuid.uuid4().hex[:18]
    return f"ipc://{base_rpc_path}/{short_uuid}"


def _validate_zmq_address(address: str) -> None:
    """
    Check if a ZMQ address is valid.
    """
    # Check for supported protocols
    if not address.startswith(("tcp://", "ipc://", "inproc://")):
        raise ValueError(
            f"ZMQ address must start with tcp://, ipc://, or inproc://. Found: {address}"
        )

    # Protocol-specific validation
    if address.startswith("tcp://"):
        # TCP requires host:port format
        parts = address[6:].split(":")
        if len(parts) != 2:
            raise ValueError(
                f"ZMQ tcp address must be in the format tcp://host:port. Found: {address}"
            )
        try:
            port = int(parts[1])
        except ValueError:
            raise ValueError(
                f"ZMQ tcp port must be a number. Found: {parts[1]}"
            ) from None
        if not (1 <= port <= 65535):
            raise ValueError(
                f"ZMQ tcp port must be between 1 and 65535. Found: {port}"
            )
    elif address.startswith("ipc://"):
        # On linux, IPC_PATH_MAX_LEN is 107.
        # This is the length of `char sun_path[108]` field of `struct sockaddr_un`
        # subtracted by 1 for the null terminator.
        length = len(address) - len("ipc://")
        if length > zmq.IPC_PATH_MAX_LEN:
            raise ValueError(
                f"ZMQ IPC path is too long: {address}.\n"
                f"The maximum length is {zmq.IPC_PATH_MAX_LEN} characters. Found {length} characters."
            )
        if length == 0:
            raise ValueError(
                f"ZMQ IPC requires a path after the protocol. Found: {address}"
            )
    elif address.startswith("inproc://"):
        length = len(address) - len("inproc://")
        if length == 0:
            raise ValueError(
                f"ZMQ inproc requires a name after the protocol. Found: {address}"
            )


# Adapted from:
#  - vllm: https://github.com/vllm-project/vllm/blob/46c759c165a5a985ce62f019bf684e4a6109e41c/vllm/utils.py#L2093
#  - sglang: https://github.com/sgl-project/sglang/blob/efc52f85e2d5c9b31545d4092f2b361b6ff04d67/python/sglang/srt/utils.py#L783
def _open_zmq_socket(path: str, mode: int) -> zmq.Socket[bytes]:
    """Open a ZMQ socket with the proper bind/connect semantics."""
    mem = psutil.virtual_memory()

    # Grab the singleton global zmq ctx
    # https://zguide.zeromq.org/docs/chapter2/
    # "one I/O thread per gigabyte of data per second"
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
        _validate_zmq_address(endpoint)
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
        self._serialize = DEFAULT_MSGPACK_NUMPY_ENCODER
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
        self._serialize = DEFAULT_MSGPACK_NUMPY_ENCODER
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
        self._serialize = DEFAULT_MSGPACK_NUMPY_ENCODER
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
