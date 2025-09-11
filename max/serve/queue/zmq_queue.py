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
import time
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


def _wait_for_peer_connection(
    socket: zmq.Socket[bytes], timeout: float, expected_event: int
) -> None:
    """
    Wait for peer connection using ZMQ socket monitoring.

    Args:
        socket: The ZMQ socket to monitor
        timeout: Timeout in seconds to wait for peer connection
        expected_event: Expected ZMQ event (zmq.EVENT_CONNECTED or zmq.EVENT_ACCEPTED)

    Raises:
        TimeoutError: If peer doesn't connect within the timeout period
        RuntimeError: If monitoring fails
    """
    # Create monitoring endpoint and socket
    monitor_endpoint = f"inproc://monitor-{uuid.uuid4()}"
    monitor_socket = None

    try:
        # Enable monitoring on the socket
        socket.monitor(monitor_endpoint, expected_event)

        # Create monitoring socket to receive events
        monitor_socket = zmq.Context.instance().socket(zmq.PAIR)
        monitor_socket.connect(monitor_endpoint)

        # Set timeout for monitoring socket
        monitor_socket.setsockopt(
            zmq.RCVTIMEO, int(timeout * 1000)
        )  # Convert to milliseconds

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Receive monitoring event
                event = monitor_socket.recv_multipart(zmq.NOBLOCK)
                if len(event) >= 2:
                    # Parse event - first part contains event data
                    event_data = event[0]
                    if len(event_data) >= 4:
                        # Extract event type from the binary data (first 2 bytes, little-endian)
                        event_type = int.from_bytes(
                            event_data[:2], byteorder="little"
                        )

                        if event_type == expected_event:
                            # Peer connected successfully
                            return
                        elif event_type == zmq.EVENT_DISCONNECTED:
                            # Peer disconnected - continue waiting for reconnection
                            continue

            except zmq.Again:
                # No events available, continue polling
                time.sleep(0.001)  # Small sleep to avoid busy waiting
                continue

        # Timeout reached without successful connection
        raise TimeoutError(f"Peer did not connect within {timeout} seconds")

    except Exception as e:
        if isinstance(e, TimeoutError):
            raise
        raise RuntimeError(f"Socket monitoring failed: {e}") from e
    finally:
        # Clean up monitoring
        try:
            if monitor_socket is not None:
                monitor_socket.close()
            socket.disable_monitor()
        except Exception as cleanup_error:
            logger.warning(
                f"Failed to cleanup socket monitoring: {cleanup_error}"
            )


# Adapted from:
#  - vllm: https://github.com/vllm-project/vllm/blob/46c759c165a5a985ce62f019bf684e4a6109e41c/vllm/utils.py#L2093
#  - sglang: https://github.com/sgl-project/sglang/blob/efc52f85e2d5c9b31545d4092f2b361b6ff04d67/python/sglang/srt/utils.py#L783
def _open_zmq_socket(
    path: str,
    mode: int,
    bind: bool = True,
) -> zmq.Socket[bytes]:
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
        if bind:
            socket.bind(path)
        else:
            socket.connect(path)
    elif mode == zmq.DEALER:
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.RCVBUF, buf_size)
        socket.setsockopt(zmq.SNDBUF, buf_size)
        socket.setsockopt(zmq.LINGER, 0)
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
        peer_timeout: Optional[float] = None,
    ) -> None:
        """
        Initialize ZmqPushSocket with lazy PUSH socket initialization.

        Args:
            endpoint: ZMQ endpoint for the push socket. If None, generates unique IPC path.
            serialize: Function to serialize messages before sending.
            lazy: If True (default), socket initialization is deferred until first use.
                  If False, socket is initialized immediately during construction.
            peer_timeout: Optional timeout in seconds to wait for peer connection during
                         socket initialization. If specified, raises TimeoutError if no
                         peer connects within the timeout period.
        """
        # Initialize endpoint
        self._endpoint = (
            endpoint if endpoint is not None else generate_zmq_ipc_path()
        )

        # Validate endpoint
        if not is_valid_zmq_address(self._endpoint):
            raise ValueError(f"Invalid endpoint: {self._endpoint}")

        # Store serialization function and peer timeout
        self._serialize = serialize
        self._peer_timeout = peer_timeout

        # Initialize socket (lazily)
        self._push_socket: Optional[zmq.Socket[bytes]] = None

        # State management
        self._closed = False
        self._finalize = weakref.finalize(self, self._cleanup)

        if not lazy:
            self.initialize_socket()

    def initialize_socket(self) -> zmq.Socket[bytes]:
        """
        Initialize the push socket if needed and return it.

        This allows external users to initialize the socket before use,
        which can be useful for setup or testing scenarios. The socket
        is lazily initialized on first call.

        Returns:
            The initialized ZMQ push socket.

        Raises:
            TimeoutError: If peer_timeout is specified and no peer connects within that time.
        """
        if self._push_socket is None:
            self._push_socket = _open_zmq_socket(self._endpoint, mode=zmq.PUSH)

            # If peer timeout is specified, wait for peer connection
            if self._peer_timeout is not None:
                try:
                    _wait_for_peer_connection(
                        self._push_socket,
                        self._peer_timeout,
                        zmq.EVENT_ACCEPTED,
                    )
                except TimeoutError as e:
                    # Clean up socket on timeout
                    self._push_socket.close()
                    self._push_socket = None
                    raise TimeoutError(
                        f"PUSH socket peer connection timeout: {e}"
                    ) from e

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
        peer_timeout: Optional[float] = None,
    ) -> None:
        """
        Initialize ZmqPullSocket with lazy PULL socket initialization.

        Args:
            endpoint: ZMQ endpoint for the pull socket. If None, generates unique IPC path.
            deserialize: Function to deserialize received messages.
            lazy: If True (default), socket initialization is deferred until first use.
                  If False, socket is initialized immediately during construction.
            peer_timeout: Optional timeout in seconds to wait for peer connection during
                         socket initialization. If specified, raises TimeoutError if no
                         peer connects within the timeout period.
        """
        # Initialize endpoint
        self._endpoint = (
            endpoint if endpoint is not None else generate_zmq_ipc_path()
        )

        # Validate endpoint
        if not is_valid_zmq_address(self._endpoint):
            raise ValueError(f"Invalid endpoint: {self._endpoint}")

        # Store deserialization function and peer timeout
        self._deserialize = deserialize
        self._peer_timeout = peer_timeout

        # Initialize socket (lazily)
        self._pull_socket: Optional[zmq.Socket[bytes]] = None

        # State management
        self._closed = False
        self._finalize = weakref.finalize(self, self._cleanup)

        if not lazy:
            self.initialize_socket()

    def initialize_socket(self) -> zmq.Socket[bytes]:
        """
        Initialize the pull socket if needed and return it.

        This allows external users to initialize the socket before use,
        which can be useful for setup or testing scenarios. The socket
        is lazily initialized on first call.

        Returns:
            The initialized ZMQ pull socket.

        Raises:
            TimeoutError: If peer_timeout is specified and no peer connects within that time.
        """
        if self._pull_socket is None:
            self._pull_socket = _open_zmq_socket(self._endpoint, mode=zmq.PULL)

            # If peer timeout is specified, wait for peer connection
            if self._peer_timeout is not None:
                try:
                    _wait_for_peer_connection(
                        self._pull_socket,
                        self._peer_timeout,
                        zmq.EVENT_CONNECTED,
                    )
                except TimeoutError as e:
                    # Clean up socket on timeout
                    self._pull_socket.close()
                    self._pull_socket = None
                    raise TimeoutError(
                        f"PULL socket peer connection timeout: {e}"
                    ) from e

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
            if e.errno == zmq.EAGAIN:
                raise queue.Empty() from None

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
        lazy: bool = True,
        peer_timeout: Optional[float] = None,
    ) -> None:
        """
        Initialize ZmqRouterSocket with optional lazy initialization and peer timeout.

        Args:
            zmq_endpoint: ZMQ endpoint for the router socket.
            bind: If True (default), bind to the endpoint. If False, connect to it.
            serialize: Function to serialize messages before sending.
            deserialize: Function to deserialize received messages.
            lazy: If True (default), socket initialization is deferred until first use.
                  If False, socket is initialized immediately during construction.
            peer_timeout: Optional timeout in seconds to wait for peer connection during
                         socket initialization. If specified, raises TimeoutError if no
                         peer connects within the timeout period.
        """
        self.zmq_endpoint = zmq_endpoint
        self.bind = bind
        self.serialize = serialize
        self.deserialize = deserialize
        self._peer_timeout = peer_timeout

        # Initialize socket (lazily)
        self.router_socket: Optional[zmq.Socket[bytes]] = None

        # State management
        self._closed = False
        self._finalize = weakref.finalize(self, self._cleanup)

        if not lazy:
            self.initialize_socket()

    def initialize_socket(self) -> zmq.Socket[bytes]:
        """
        Initialize the router socket if needed and return it.

        This allows external users to initialize the socket before use,
        which can be useful for setup or testing scenarios. The socket
        is lazily initialized on first call.

        Returns:
            The initialized ZMQ router socket.

        Raises:
            TimeoutError: If peer_timeout is specified and no peer connects within that time.
        """
        if self.router_socket is None:
            self.router_socket = _open_zmq_socket(
                self.zmq_endpoint, mode=zmq.ROUTER, bind=self.bind
            )

            # If peer timeout is specified, wait for peer connection
            if self._peer_timeout is not None:
                try:
                    expected_event = (
                        zmq.EVENT_ACCEPTED if self.bind else zmq.EVENT_CONNECTED
                    )
                    _wait_for_peer_connection(
                        self.router_socket,
                        self._peer_timeout,
                        expected_event,
                    )
                except TimeoutError as e:
                    # Clean up socket on timeout
                    self.router_socket.close()
                    self.router_socket = None
                    raise TimeoutError(
                        f"ROUTER socket peer connection timeout: {e}"
                    ) from e

        return self.router_socket

    def _cleanup(self) -> None:
        if self.router_socket is not None and not self.router_socket.closed:
            self.router_socket.close()

    def close(self) -> None:
        """Explicitly close the ZMQ socket."""
        if not self._closed:
            self._closed = True
            if self.router_socket is not None and not self.router_socket.closed:
                self.router_socket.close()
            self._finalize.detach()

    def send_multipart(
        self, identity: bytes, message: T, flags: int = 0
    ) -> None:
        """Send a message to a specific identity."""
        if self._closed:
            raise RuntimeError("Socket is closed")

        router_socket = self.initialize_socket()

        try:
            serialized_msg = self.serialize(message)
        except Exception as e:
            logger.exception(f"Failed to serialize message: {e}")
            raise

        try:
            router_socket.send_multipart(
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

        router_socket = self.initialize_socket()

        try:
            identity, message_data = router_socket.recv_multipart(flags=flags)
        except zmq.ZMQError as e:
            if e.errno == zmq.EAGAIN:
                raise queue.Empty() from None
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
        zmq_endpoint: str,
        bind: bool = False,
        serialize: Callable[[Any], bytes] = pickle.dumps,
        deserialize: Callable[[Any], Any] = pickle.loads,
        lazy: bool = True,
        peer_timeout: Optional[float] = None,
    ) -> None:
        """
        Initialize ZmqDealerSocket with optional lazy initialization and peer timeout.

        Args:
            zmq_endpoint: ZMQ endpoint for the dealer socket.
            bind: If False (default), connect to the endpoint. If True, bind to it.
            serialize: Function to serialize messages before sending.
            deserialize: Function to deserialize received messages.
            lazy: If True (default), socket initialization is deferred until first use.
                  If False, socket is initialized immediately during construction.
            peer_timeout: Optional timeout in seconds to wait for peer connection during
                         socket initialization. If specified, raises TimeoutError if no
                         peer connects within the timeout period.
        """
        self.zmq_endpoint = zmq_endpoint
        self.bind = bind
        self.serialize = serialize
        self.deserialize = deserialize
        self._peer_timeout = peer_timeout

        # Initialize socket (lazily)
        self.dealer_socket: Optional[zmq.Socket[bytes]] = None

        # State management
        self._closed = False
        self._finalize = weakref.finalize(self, self._cleanup)

        if not lazy:
            self.initialize_socket()

    def initialize_socket(self) -> zmq.Socket[bytes]:
        """
        Initialize the dealer socket if needed and return it.

        This allows external users to initialize the socket before use,
        which can be useful for setup or testing scenarios. The socket
        is lazily initialized on first call.

        Returns:
            The initialized ZMQ dealer socket.

        Raises:
            TimeoutError: If peer_timeout is specified and no peer connects within that time.
        """
        if self.dealer_socket is None:
            self.dealer_socket = _open_zmq_socket(
                self.zmq_endpoint, mode=zmq.DEALER, bind=self.bind
            )

            # If peer timeout is specified, wait for peer connection
            if self._peer_timeout is not None:
                try:
                    expected_event = (
                        zmq.EVENT_ACCEPTED if self.bind else zmq.EVENT_CONNECTED
                    )
                    _wait_for_peer_connection(
                        self.dealer_socket,
                        self._peer_timeout,
                        expected_event,
                    )
                except TimeoutError as e:
                    # Clean up socket on timeout
                    self.dealer_socket.close()
                    self.dealer_socket = None
                    raise TimeoutError(
                        f"DEALER socket peer connection timeout: {e}"
                    ) from e

        return self.dealer_socket

    def _cleanup(self) -> None:
        if self.dealer_socket is not None and not self.dealer_socket.closed:
            self.dealer_socket.close()

    def close(self) -> None:
        """Explicitly close the ZMQ socket."""
        if not self._closed:
            self._closed = True
            if self.dealer_socket is not None and not self.dealer_socket.closed:
                self.dealer_socket.close()
            self._finalize.detach()

    def send_pyobj(self, message: T, flags: int = 0) -> None:
        """Send a message."""
        if self._closed:
            raise RuntimeError("Socket is closed")

        dealer_socket = self.initialize_socket()

        try:
            serialized_msg = self.serialize(message)
        except Exception as e:
            logger.exception(f"Failed to serialized message: {e}")
            raise

        try:
            dealer_socket.send(serialized_msg, flags=flags)
        except zmq.ZMQError as e:
            if e.errno != zmq.EAGAIN:
                logger.exception(f"Failed to send message: {e}")
            raise

    def recv_pyobj(self, flags: int = 0) -> T:
        """Receive a message."""
        if self._closed:
            raise RuntimeError("Socket is closed")

        dealer_socket = self.initialize_socket()

        try:
            message = dealer_socket.recv(flags=flags)
        except zmq.ZMQError as e:
            if e.errno == zmq.EAGAIN:
                raise queue.Empty() from None
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


def create_zmq_push_pull_queues(
    payload_type: type[T],
    endpoint: Optional[str] = None,
    use_pickle: bool = False,
    lazy: bool = True,
    peer_timeout: Optional[float] = None,
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
        peer_timeout: Optional timeout in seconds to wait for peer connection during
                     socket initialization. If specified, raises TimeoutError if no
                     peer connects within the timeout period.

    Returns:
        A tuple containing (ZmqPushSocket, ZmqPullSocket) configured to communicate
        with each other using the same endpoint.

    Example:
        >>> push_socket, pull_socket = create_zmq_push_pull_queues(str)
        >>> push_socket.put_nowait("hello")
        >>> message = pull_socket.get_nowait()
        >>> print(message)  # "hello"

        >>> # With peer timeout - wait up to 5 seconds for peer connection
        >>> push_socket, pull_socket = create_zmq_push_pull_queues(str, peer_timeout=5.0)
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
        peer_timeout=peer_timeout,
    )

    pull_socket = ZmqPullSocket[T](
        endpoint=actual_endpoint,
        deserialize=deserialize,
        lazy=lazy,
        peer_timeout=peer_timeout,
    )

    return push_socket, pull_socket
