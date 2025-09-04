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
from typing import Any, Callable, Generic, Optional, TypeVar, cast

import psutil
import zmq
import zmq.constants
from max.interfaces import msgpack_numpy_decoder, msgpack_numpy_encoder
from max.interfaces.queue import MAXPushQueue

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
    path: str,
    mode: int,
    bind: bool = True,
) -> zmq.Socket:
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


class ZmqPushSocket(Generic[T], MAXPushQueue[T]):
    """
    ZeroMQ-based push socket implementation using PUSH socket.

    This class implements the MAXPushQueue protocol using ZeroMQ PUSH sockets
    for efficient inter-process communication. It supports only push (producer)
    operations for adding items to the queue with lazy socket initialization.

    The socket uses a single ZMQ endpoint where the PUSH socket binds to
    send messages, following standard ZeroMQ PUSH semantics.
    """

    def __init__(
        self,
        *,
        endpoint: Optional[str] = None,
        serialize: Callable[[T], bytes] = pickle.dumps,
        lazy: bool = True,
    ) -> None:
        """
        Initialize ZmqPushSocket with lazy PUSH socket initialization.

        Args:
            endpoint: ZMQ endpoint for the push socket. If None, generates unique IPC path.
            serialize: Function to serialize messages before sending.
            lazy: If True (default), socket initialization is deferred until first use.
                  If False, socket is initialized immediately during construction.
        """
        # Initialize endpoint
        self._endpoint = (
            endpoint if endpoint is not None else generate_zmq_ipc_path()
        )

        # Validate endpoint
        if not is_valid_zmq_address(self._endpoint):
            raise ValueError(f"Invalid endpoint: {self._endpoint}")

        # Store serialization function
        self._serialize = serialize

        # Initialize socket (lazily)
        self._push_socket: Optional[zmq.Socket] = None

        # State management
        self._closed = False
        self._finalize = weakref.finalize(self, self._cleanup)

        if not lazy:
            self.initialize_socket()

    def initialize_socket(self) -> zmq.Socket:
        """
        Initialize the push socket if needed and return it.

        This allows external users to initialize the socket before use,
        which can be useful for setup or testing scenarios. The socket
        is lazily initialized on first call.

        Returns:
            The initialized ZMQ push socket.
        """
        if self._push_socket is None:
            self._push_socket = _open_zmq_socket(self._endpoint, mode=zmq.PUSH)
        return self._push_socket

    def _cleanup(self) -> None:
        """Clean up resources during garbage collection."""
        if self._push_socket is not None and not self._push_socket.closed:
            self._push_socket.close()

    def close(self) -> None:
        """Explicitly close the socket."""
        if not self._closed:
            self._closed = True

            if self._push_socket is not None and not self._push_socket.closed:
                self._push_socket.close()

            # Cancel the weakref finalizer since we've manually cleaned up
            self._finalize.detach()

    def put_nowait(
        self,
        item: T,
    ) -> None:
        if self._closed:
            raise RuntimeError("Socket is closed")

        push_socket = self.initialize_socket()

        while True:
            try:
                serialized_msg = self._serialize(item)
            except Exception as e:
                logger.exception(f"Failed to serialize message: {e}")
                raise

            try:
                push_socket.send(serialized_msg, flags=zmq.NOBLOCK)

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
    """
    ZeroMQ-based pull socket implementation using PULL socket.

    This class provides a consumer interface for receiving messages from ZMQ PUSH
    sockets using ZeroMQ PULL semantics for efficient inter-process communication.
    It supports only pull (consumer) operations for retrieving items from the queue
    with lazy socket initialization.

    The socket uses a single ZMQ endpoint where the PULL socket connects to
    receive messages, following standard ZeroMQ PULL semantics.
    """

    def __init__(
        self,
        *,
        endpoint: Optional[str] = None,
        deserialize: Callable[[bytes], T] = pickle.loads,
        lazy: bool = True,
    ) -> None:
        """
        Initialize ZmqPullSocket with lazy PULL socket initialization.

        Args:
            endpoint: ZMQ endpoint for the pull socket. If None, generates unique IPC path.
            deserialize: Function to deserialize received messages.
            lazy: If True (default), socket initialization is deferred until first use.
                  If False, socket is initialized immediately during construction.
        """
        # Initialize endpoint
        self._endpoint = (
            endpoint if endpoint is not None else generate_zmq_ipc_path()
        )

        # Validate endpoint
        if not is_valid_zmq_address(self._endpoint):
            raise ValueError(f"Invalid endpoint: {self._endpoint}")

        # Store serialization function
        self._deserialize = deserialize

        # Initialize socket (lazily)
        self._pull_socket: Optional[zmq.Socket] = None

        # State management
        self._closed = False
        self._finalize = weakref.finalize(self, self._cleanup)

        if not lazy:
            self.initialize_socket()

    def initialize_socket(self) -> zmq.Socket:
        """
        Initialize the pull socket if needed and return it.

        This allows external users to initialize the socket before use,
        which can be useful for setup or testing scenarios. The socket
        is lazily initialized on first call.

        Returns:
            The initialized ZMQ pull socket.
        """
        if self._pull_socket is None:
            self._pull_socket = _open_zmq_socket(self._endpoint, mode=zmq.PULL)
        return self._pull_socket

    def _cleanup(self) -> None:
        """Clean up resources during garbage collection."""
        if self._pull_socket is not None and not self._pull_socket.closed:
            self._pull_socket.close()

    def close(self) -> None:
        """Explicitly close the socket."""
        if not self._closed:
            self._closed = True

            if self._pull_socket is not None and not self._pull_socket.closed:
                self._pull_socket.close()

            # Cancel the weakref finalizer since we've manually cleaned up
            self._finalize.detach()

    def get_nowait(self) -> T:
        if self._closed:
            raise RuntimeError("Socket is closed")

        pull_socket = self.initialize_socket()

        try:
            msg = pull_socket.recv(flags=zmq.NOBLOCK)
        except zmq.ZMQError as e:
            if e.errno == zmq.constants.EAGAIN:
                raise queue.Empty()  # noqa: B904

            logger.exception(
                f"Failed to receive message on ZMQ socket for unknown reason: {e}"
            )
            raise

        try:
            return self._deserialize(msg)
        except Exception as e:
            logger.exception(f"Failed to deserialize message: {e}")
            raise


class ZmqRouterSocket(Generic[T]):
    """ZMQ ROUTER socket for N:1 communication patterns with identity-based routing."""

    def __init__(
        self,
        zmq_endpoint: str,
        bind: bool = True,
        serialize: Callable[[Any], bytes] = pickle.dumps,
        deserialize: Callable[[Any], Any] = pickle.loads,
    ) -> None:
        self.zmq_endpoint = zmq_endpoint
        self.router_socket = _open_zmq_socket(
            self.zmq_endpoint, mode=zmq.constants.ROUTER, bind=bind
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
            if e.errno != zmq.constants.EAGAIN:
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
            if e.errno == zmq.constants.EAGAIN:
                raise queue.Empty()  # noqa: B904
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
        return self.recv_multipart(flags=zmq.constants.NOBLOCK)


class ZmqDealerSocket(Generic[T]):
    """ZMQ DEALER socket for 1:N communication patterns."""

    def __init__(
        self,
        zmq_endpoint: str,
        bind: bool = False,
        serialize: Callable[[Any], bytes] = pickle.dumps,
        deserialize: Callable[[Any], Any] = pickle.loads,
    ) -> None:
        self.zmq_endpoint = zmq_endpoint
        self.dealer_socket = _open_zmq_socket(
            self.zmq_endpoint, mode=zmq.constants.DEALER, bind=bind
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
            if e.errno != zmq.constants.EAGAIN:
                logger.exception(f"Failed to send message: {e}")
            raise

    def recv_pyobj(self, flags: int = 0) -> T:
        """Receive a message."""
        if self._closed:
            raise RuntimeError("Socket is closed")

        try:
            message = self.dealer_socket.recv(flags=flags)
        except zmq.ZMQError as e:
            if e.errno == zmq.constants.EAGAIN:
                raise queue.Empty()  # noqa: B904
            logger.exception(f"Failed to receive message: {e}")
            raise

        try:
            return self.deserialize(message)
        except Exception as e:
            logger.exception(f"Failed to deserialize message: {e}")
            raise

    def recv_pyobj_nowait(self) -> T:
        """Non-blocking receive."""
        return self.recv_pyobj(flags=zmq.constants.NOBLOCK)


def create_zmq_push_pull_queues(
    payload_type: type[T],
    endpoint: Optional[str] = None,
    use_pickle: bool = False,
    lazy: bool = True,
) -> tuple[ZmqPushSocket[T], ZmqPullSocket[T]]:
    """
    Factory method to create a matched pair of ZMQ push and pull sockets.

    This factory creates both a ZmqPushSocket and ZmqPullSocket that share the same
    endpoint, allowing them to communicate with each other. The push socket can
    send messages that the pull socket will receive. Both sockets support lazy
    initialization for efficient resource management.

    Args:
        payload_type: Type of the payload that will be sent through the queue.
        endpoint: ZMQ endpoint for both sockets. If None, generates unique IPC path.
        use_pickle: Whether to use pickle for serialization. If False, uses msgpack.
        lazy: If True (default), socket initialization is deferred until first use.
              If False, both sockets are initialized immediately during construction.

    Returns:
        A tuple containing (ZmqPushSocket, ZmqPullSocket) configured to communicate
        with each other using the same endpoint.

    Example:
        >>> push_socket, pull_socket = create_zmq_push_pull_queues(str)
        >>> push_socket.put_nowait("hello")
        >>> message = pull_socket.get_nowait()
        >>> print(message)  # "hello"
    """
    # Use the same endpoint for both sockets so they can communicate
    actual_endpoint = (
        endpoint if endpoint is not None else generate_zmq_ipc_path()
    )

    if use_pickle:
        serialize = cast(Callable[[T], bytes], pickle.dumps)
        deserialize = cast(Callable[[bytes], T], pickle.loads)
    else:
        serialize = msgpack_numpy_encoder()
        deserialize = msgpack_numpy_decoder(payload_type)

    push_socket = ZmqPushSocket[T](
        endpoint=actual_endpoint,
        serialize=serialize,
        lazy=lazy,
    )

    pull_socket = ZmqPullSocket[T](
        endpoint=actual_endpoint,
        deserialize=deserialize,
        lazy=lazy,
    )

    return push_socket, pull_socket
