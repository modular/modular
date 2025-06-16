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
import pickle
import queue
import tempfile
import uuid
import weakref
from collections import deque
from typing import Any, Callable, Generic, Optional, TypeVar

import psutil
import zmq
import zmq.asyncio
import zmq.constants
from max.profiler import traced

logger = logging.getLogger("max.serve")

T = TypeVar("T")


def generate_zmq_ipc_path() -> str:
    """Generate a unique ZMQ IPC path."""
    base_rpc_path = tempfile.gettempdir()
    return f"ipc://{base_rpc_path}/{uuid.uuid4()}"


def generate_zmq_inproc_endpoint() -> str:
    """Generate a unique ZMQ inproc endpoint."""
    return f"inproc://{uuid.uuid4()}"


def is_valid_zmq_address(address: str) -> bool:
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
def _open_zmq_socket(
    zmq_ctx: zmq.Context,
    path: str,
    mode: int,
    bind: bool = True,
) -> zmq.Socket:
    """Open a ZMQ socket with the proper bind/connect semantics."""
    mem = psutil.virtual_memory()
    socket = zmq_ctx.socket(mode)

    # Calculate buffer size based on system memory
    GIB = 1024**3
    total_mem_gb = mem.total / GIB
    available_mem_gb = mem.available / GIB
    # For systems with substantial memory (>32GB total, >16GB available):
    # - Set a large 0.5GB buffer to improve throughput
    # For systems with less memory:
    # - Use system default (-1) to avoid excessive memory consumption
    if total_mem_gb > 32 and available_mem_gb > 16:
        buf_size = int(0.5 * GIB)
    else:
        buf_size = -1

    # Configure socket options based on type
    if mode == zmq.constants.PULL:
        socket.setsockopt(zmq.constants.RCVHWM, 0)
        socket.setsockopt(zmq.constants.RCVBUF, buf_size)
        socket.setsockopt(zmq.constants.LINGER, 0)
        socket.connect(path)
    elif mode == zmq.constants.PUSH:
        socket.setsockopt(zmq.constants.SNDHWM, 0)
        socket.setsockopt(zmq.constants.SNDBUF, buf_size)
        socket.setsockopt(zmq.constants.LINGER, 0)
        socket.bind(path)
    elif mode == zmq.constants.ROUTER:
        socket.setsockopt(zmq.constants.RCVHWM, 0)
        socket.setsockopt(zmq.constants.SNDHWM, 0)
        socket.setsockopt(zmq.constants.RCVBUF, buf_size)
        socket.setsockopt(zmq.constants.SNDBUF, buf_size)
        socket.setsockopt(zmq.constants.LINGER, 0)
        socket.setsockopt(zmq.constants.ROUTER_MANDATORY, 1)
        if bind:
            socket.bind(path)
        else:
            socket.connect(path)
    elif mode == zmq.constants.DEALER:
        socket.setsockopt(zmq.constants.RCVHWM, 0)
        socket.setsockopt(zmq.constants.SNDHWM, 0)
        socket.setsockopt(zmq.constants.RCVBUF, buf_size)
        socket.setsockopt(zmq.constants.SNDBUF, buf_size)
        socket.setsockopt(zmq.constants.LINGER, 0)
        if bind:
            socket.bind(path)
        else:
            socket.connect(path)
    else:
        raise ValueError(f"Unknown Socket Mode: {mode}")

    return socket


class ZmqPushSocket(Generic[T]):
    def __init__(
        self,
        zmq_ctx: zmq.Context,
        zmq_endpoint: Optional[str] = None,
        serialize: Callable[[Any], bytes] = pickle.dumps,
    ):
        self.zmq_endpoint = (
            zmq_endpoint
            if zmq_endpoint is not None
            else generate_zmq_ipc_path()
        )
        self.push_socket = _open_zmq_socket(
            zmq_ctx, self.zmq_endpoint, mode=zmq.PUSH
        )
        self.serialize = serialize
        self._closed = False
        self._finalize = weakref.finalize(self, self._cleanup)

    def _cleanup(self) -> None:
        if not self.push_socket.closed:
            self.push_socket.close()

    def close(self) -> None:
        """Explicitly close the ZMQ socket."""
        if not self._closed:
            self._closed = True
            if not self.push_socket.closed:
                self.push_socket.close()
            # Cancel the weakref finalizer since we've manually cleaned up
            self._finalize.detach()

    def put_nowait(
        self,
        msg: Any,
        **kwargs,
    ):
        self.put(msg, flags=zmq.NOBLOCK, **kwargs)

    def put(
        self,
        msg: Any,
        **kwargs,
    ) -> None:
        if self._closed:
            raise RuntimeError("Socket is closed")

        while True:
            try:
                serialized_msg = self.serialize(msg)
            except Exception as e:
                logger.exception(f"Failed to serialize message: {e}")
                raise

            try:
                self.push_socket.send(serialized_msg, **kwargs)

                # Exit since we succeeded
                break
            except zmq.ZMQError as e:
                # If we get EAGAIN, we just try again.
                # This could be due to:
                #   - the pull socket not being opened yet
                #   - a full queue
                if e.errno == zmq.EAGAIN:
                    continue

                # Unknown error, log it and let caller handle it
                logger.exception(
                    f"Failed to send message on ZMQ socket for unknown reason: {e}"
                )
                raise


class ZmqPullSocket(Generic[T]):
    def __init__(
        self,
        zmq_ctx: zmq.Context,
        zmq_endpoint: Optional[str] = None,
        deserialize=pickle.loads,
    ):
        self.zmq_endpoint = (
            zmq_endpoint
            if zmq_endpoint is not None
            else generate_zmq_ipc_path()
        )
        self.pull_socket = _open_zmq_socket(
            zmq_ctx, self.zmq_endpoint, mode=zmq.PULL
        )
        self.deserialize = deserialize
        self.local_queue: deque[T] = deque()
        self._closed = False
        self._finalize = weakref.finalize(self, self._cleanup)

    def _cleanup(self) -> None:
        if not self.pull_socket.closed:
            self.pull_socket.close()

    def close(self) -> None:
        """Explicitly close the ZMQ socket."""
        if not self._closed:
            self._closed = True
            if not self.pull_socket.closed:
                self.pull_socket.close()
            # Cancel the weakref finalizer since we've manually cleaned up
            self._finalize.detach()

    def put_front_nowait(self, item: T):
        """A new method that allows us to add requests to the front of the queue."""
        if self._closed:
            raise RuntimeError("Socket is closed")
        self.local_queue.append(item)

    def _pull_from_socket(self, **kwargs) -> T:
        if self._closed:
            raise RuntimeError("Socket is closed")

        try:
            msg = self.pull_socket.recv(**kwargs)
        except zmq.ZMQError as e:
            if e.errno == zmq.EAGAIN:
                raise queue.Empty()

            logger.exception(
                f"Failed to receive message on ZMQ socket for unknown reason: {e}"
            )
            raise

        try:
            return self.deserialize(msg)
        except Exception as e:
            raise

    def get(self, **kwargs) -> T:
        if self._closed:
            raise RuntimeError("Socket is closed")

        if self.local_queue:
            return self.local_queue.pop()

        return self._pull_from_socket(**kwargs)

    @traced
    def get_nowait(self, **kwargs) -> T:
        return self.get(flags=zmq.NOBLOCK, **kwargs)

    def qsize(self) -> int:
        """Return the size of the queue by repeatedly polling the ZmqSocket and
        adding the items to the local queue."""
        if self._closed:
            return len(self.local_queue)

        while True:
            try:
                item = self._pull_from_socket(flags=zmq.NOBLOCK)
                self.local_queue.appendleft(item)
            except queue.Empty:
                break

        return len(self.local_queue)

    def empty(self) -> bool:
        return self.qsize() == 0


class ZmqRouterSocket(Generic[T]):
    """ZMQ ROUTER socket for N:1 communication patterns with identity-based routing."""

    def __init__(
        self,
        zmq_ctx: zmq.Context,
        zmq_endpoint: str,
        bind: bool = True,
        serialize: Callable[[Any], bytes] = pickle.dumps,
        deserialize: Callable[[Any], Any] = pickle.loads,
    ):
        self.zmq_endpoint = zmq_endpoint
        self.router_socket = _open_zmq_socket(
            zmq_ctx, self.zmq_endpoint, mode=zmq.ROUTER, bind=bind
        )
        self._closed = False
        self._finalize = weakref.finalize(self, self._cleanup)
        self.serialize = serialize
        self.deserialize = deserialize

    def _cleanup(self) -> None:
        if not self.router_socket.closed:
            self.router_socket.close()

    def close(self) -> None:
        """Explicitly close the ZMQ socket."""
        if not self._closed:
            self._closed = True
            if not self.router_socket.closed:
                self.router_socket.close()
            self._finalize.detach()

    def send_multipart(
        self, identity: bytes, message: T, flags: int = 0
    ) -> None:
        """Send a message to a specific identity."""
        if self._closed:
            raise RuntimeError("Socket is closed")

        try:
            serialized_msg = self.serialize(message)
        except Exception as e:
            logger.exception(f"Failed to serialize message: {e}")
            raise

        try:
            self.router_socket.send_multipart(
                [identity, serialized_msg], flags=flags
            )
        except zmq.ZMQError as e:
            if e.errno != zmq.EAGAIN:
                logger.exception(f"Failed to send multipart message: {e}")
            raise

    def recv_multipart(self, flags: int = 0) -> tuple[bytes, T]:
        """Receive a message with sender identity."""
        if self._closed:
            raise RuntimeError("Socket is closed")

        try:
            identity, message_data = self.router_socket.recv_multipart(
                flags=flags
            )
        except zmq.ZMQError as e:
            if e.errno == zmq.EAGAIN:
                raise queue.Empty()
            logger.exception(f"Failed to receive multipart message: {e}")
            raise

        try:
            message = self.deserialize(message_data)
            return identity, message
        except Exception as e:
            logger.exception(f"Failed to deserialize message: {e}")
            raise

    def recv_multipart_nowait(self) -> tuple[bytes, T]:
        """Non-blocking receive."""
        return self.recv_multipart(flags=zmq.NOBLOCK)


class ZmqDealerSocket(Generic[T]):
    """ZMQ DEALER socket for 1:N communication patterns."""

    def __init__(
        self,
        zmq_ctx: zmq.Context,
        zmq_endpoint: str,
        bind: bool = False,
        serialize: Callable[[Any], bytes] = pickle.dumps,
        deserialize: Callable[[Any], Any] = pickle.loads,
    ):
        self.zmq_endpoint = zmq_endpoint
        self.dealer_socket = _open_zmq_socket(
            zmq_ctx, self.zmq_endpoint, mode=zmq.DEALER, bind=bind
        )
        self.serialize = serialize
        self.deserialize = deserialize
        self._closed = False
        self._finalize = weakref.finalize(self, self._cleanup)

    def _cleanup(self) -> None:
        if not self.dealer_socket.closed:
            self.dealer_socket.close()

    def close(self) -> None:
        """Explicitly close the ZMQ socket."""
        if not self._closed:
            self._closed = True
            if not self.dealer_socket.closed:
                self.dealer_socket.close()
            self._finalize.detach()

    def send_pyobj(self, message: T, flags: int = 0) -> None:
        """Send a message."""
        if self._closed:
            raise RuntimeError("Socket is closed")

        try:
            serialized_msg = self.serialize(message)
        except Exception as e:
            logger.exception(f"Failed to serialized message: {e}")
            raise

        try:
            self.dealer_socket.send(serialized_msg, flags=flags)
        except zmq.ZMQError as e:
            if e.errno != zmq.EAGAIN:
                logger.exception(f"Failed to send message: {e}")
            raise

    def recv_pyobj(self, flags: int = 0) -> T:
        """Receive a message."""
        if self._closed:
            raise RuntimeError("Socket is closed")

        try:
            message = self.dealer_socket.recv(flags=flags)
        except zmq.ZMQError as e:
            if e.errno == zmq.EAGAIN:
                raise queue.Empty()
            logger.exception(f"Failed to receive message: {e}")
            raise

        try:
            return self.deserialize(message)
        except Exception as e:
            logger.exception(f"Failed to deserialize message: {e}")
            raise

    def recv_pyobj_nowait(self) -> T:
        """Non-blocking receive."""
        return self.recv_pyobj(flags=zmq.NOBLOCK)
