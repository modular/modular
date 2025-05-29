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

import logging
import queue
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import zmq
from max.serve.kvcache_agent.dispatcher_base import ReplyContext
from max.serve.queue.zmq_queue import (
    ZmqDealerSocket,
    ZmqPullSocket,
    ZmqPushSocket,
    ZmqRouterSocket,
)

logger = logging.getLogger(__name__)


@dataclass
class TransportMessage:
    """
    Core message structure for transport layer communication.

    Contains all necessary metadata for routing, correlation, and payload delivery
    across different transport mechanisms (ZMQ, HTTP, gRPC).
    """

    message_id: str  # Unique identifier for this message
    message_type: str  # Type of message
    payload: Any  # Message payload data
    source_id: Optional[str] = None  # Identifier of the sending instance
    destination_address: Optional[str] = None  # Target address for routing
    correlation_id: Optional[str] = None  # ID for request-response correlation
    is_reply: bool = False  # Whether this message is a reply to another message
    timestamp: float = field(
        default_factory=time.time
    )  # Message creation timestamp


@dataclass
class TransportMessageEnvelope:
    """
    Container for an inbound message and its optional reply context.

    Wraps received messages with their associated reply context, enabling
    transport-agnostic reply handling for request-response patterns.
    """

    message: TransportMessage  # The received message
    reply_context: Optional[ReplyContext] = None  # Context for sending replies

    def can_reply(self) -> bool:
        """
        Check if this message can be replied to.
        """
        return self.reply_context is not None


class DispatcherTransport(ABC):
    """
    Abstract base class for transport layer implementations.

    Provides a unified interface for different transport mechanisms (ZMQ, HTTP, gRPC)
    with support for send, receive, and reply operations.
    """

    @abstractmethod
    async def start(self) -> None:
        """
        Initialize and start the transport.
        """
        pass

    @abstractmethod
    async def send_message(
        self,
        message: TransportMessage,
        destination_address: Optional[str] = None,
    ) -> None:
        """
        Send a message to the specified destination.

        Args:
            message: Message to send
            destination_address: Target address (required for some transports)
        """
        pass

    @abstractmethod
    async def send_reply(
        self,
        message: TransportMessage,
        reply_context: Optional[ReplyContext] = None,
    ) -> None:
        """
        Send a reply message using the provided reply context.
        """
        pass

    @abstractmethod
    async def receive_message(self) -> Optional[TransportMessageEnvelope]:
        """
        Receive an inbound message with optional reply context.
        """
        pass

    @abstractmethod
    def get_address(self) -> str:
        """
        Get the address of the transport.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Close all transport connections and clean up resources.
        """
        pass


class StaticZmqTransport(DispatcherTransport):
    """
    Static ZMQ transport for 1:1 point-to-point communication using PUSH/PULL sockets.

    Simple transport implementation with no routing or identities. Both send_message
    and send_reply use PUSH socket, receive_message uses PULL socket. Suitable for
    direct peer-to-peer communication patterns.
    """

    def __init__(
        self,
        zmq_ctx: zmq.Context,
        send_endpoint: str,
        recv_endpoint: str,
    ):
        """
        Initialize static ZMQ transport with PUSH/PULL sockets.
        """
        self._pull_socket = ZmqPullSocket[Any](zmq_ctx, recv_endpoint)
        self._push_socket = ZmqPushSocket[Any](zmq_ctx, send_endpoint)
        self._running = False
        self._lock = threading.RLock()

        logger.debug(
            f"StaticZmqTransport initialized: send={send_endpoint}, recv={recv_endpoint}"
        )

    async def start(self) -> None:
        """Start the transport by marking it as running."""
        self._running = True
        logger.debug("StaticZmqTransport started")

    async def send_message(
        self,
        message: TransportMessage,
        _: Optional[str] = None,
    ) -> None:
        """
        Send a message via PUSH socket.
        """
        if not self._running:
            raise RuntimeError("Transport not started")

        try:
            with self._lock:
                self._push_socket.put_nowait(message)
                logger.debug(
                    f"Sent message {message.message_id} type={message.message_type}"
                )
        except Exception as e:
            logger.exception(
                f"Failed to send message {message.message_id}: {e}"
            )

    async def send_reply(
        self,
        message: TransportMessage,
        _: Optional[ReplyContext] = None,
    ) -> None:
        """
        Send a reply message via PUSH socket.
        """
        if not self._running:
            raise RuntimeError("Transport not started")

        try:
            with self._lock:
                self._push_socket.put_nowait(message)
                logger.debug(
                    f"Sent reply message {message.message_id} type={message.message_type}"
                )
        except Exception as e:
            logger.exception(
                f"Failed to send reply message {message.message_id}: {e}"
            )

    async def receive_message(self) -> Optional[TransportMessageEnvelope]:
        """
        Receive a message via PULL socket.
        """
        if not self._running:
            raise RuntimeError("Transport not started")

        try:
            msg = self._pull_socket.get_nowait()
            logger.debug(
                f"Received message {msg.message_id} type={msg.message_type}"
            )
            return TransportMessageEnvelope(message=msg, reply_context=None)
        except queue.Empty:
            return None
        except Exception as e:
            logger.exception(f"Failed to receive message: {e}")
        return None

    def get_address(self) -> str:
        """
        Get the address of the transport.
        """
        return self._pull_socket.zmq_endpoint

    async def close(self) -> None:
        """Close the transport and clean up ZMQ sockets."""
        self._running = False

        # Close push socket
        try:
            self._push_socket.close()
            logger.debug("Closed PUSH socket")
        except Exception as e:
            logger.exception(f"Error closing PUSH socket: {e}")
            raise

        # Close pull socket
        try:
            self._pull_socket.close()
            logger.debug("Closed PULL socket")
        except Exception as e:
            logger.exception(f"Error closing PULL socket: {e}")
            raise

        logger.debug("StaticZmqTransport closed")


class DynamicZmqTransport(DispatcherTransport):
    """
    Dynamic ZMQ transport designed for N:M communication patterns with explicit addressing.
    """

    def __init__(
        self,
        zmq_ctx: zmq.Context,
        bind_address: str,
        instance_id: str,
    ):
        """
        Initialize dynamic ZMQ transport with ROUTER/DEALER sockets.
        """
        self._zmq_ctx = zmq_ctx
        self._bind_address = bind_address
        self._instance_id = instance_id

        # Receiving socket (ROUTER for handling multiple connections)
        self._router_socket: Optional[ZmqRouterSocket[Any]] = None

        # Sending sockets (DEALER for each destination)
        self._dealer_connections: dict[str, ZmqDealerSocket[Any]] = {}

        # Correlation tracking (correlation_id -> request_info)
        self._pending_requests: dict[str, dict[str, Any]] = {}

        self._running = False
        self._lock = threading.RLock()

        logger.debug(
            f"DynamicZmqTransport initialized: instance_id={self._instance_id}, bind_address={bind_address}"
        )

    async def start(self) -> None:
        """
        Start the transport by creating and binding ROUTER socket.
        """
        try:
            self._router_socket = ZmqRouterSocket[Any](
                self._zmq_ctx, self._bind_address
            )
            self._running = True
            logger.debug(
                f"DynamicZmqTransport started: instance_id={self._instance_id}, bind_address={self._bind_address}"
            )
        except Exception as e:
            logger.exception(f"Failed to start DynamicZmqTransport: {e}")
            raise

    async def send_message(
        self,
        message: TransportMessage,
        destination_address: Optional[str] = None,
    ) -> None:
        """
        Send message to explicit destination address.
        """
        if not self._running:
            raise RuntimeError("Transport not started")

        if destination_address is None:
            logger.error("destination_address is required")
            return

        # Generate correlation ID for request-response tracking
        if message.correlation_id is None:
            message.correlation_id = str(uuid.uuid4())

        try:
            with self._lock:
                dealer_socket = self._dealer_connections.get(
                    destination_address
                )
                if not dealer_socket:
                    if not await self._connect_to_address(destination_address):
                        logger.error(
                            f"Failed to establish connection to: {destination_address}"
                        )
                        return
                    dealer_socket = self._dealer_connections.get(
                        destination_address
                    )

                if dealer_socket:
                    message.source_id = self._instance_id

                    self._pending_requests[message.correlation_id] = {
                        "destination": destination_address,
                        "timestamp": time.time(),
                        "message_id": message.message_id,
                    }

                    dealer_socket.send_pyobj(message, flags=zmq.NOBLOCK)
                    logger.debug(
                        f"Sent message {message.message_id} type={message.message_type} to {destination_address}"
                    )
                else:
                    logger.error(
                        f"Failed to establish connection to: {destination_address}"
                    )
                    return
        except zmq.Again:
            with self._lock:
                self._pending_requests.pop(message.correlation_id, None)
            logger.error(f"Send buffer full for message {message.message_id}")
        except Exception as e:
            with self._lock:
                self._pending_requests.pop(message.correlation_id, None)
            logger.exception(
                f"Failed to send message {message.message_id}: {e}"
            )

    async def send_reply(
        self,
        message: TransportMessage,
        reply_context: Optional[ReplyContext] = None,
    ) -> None:
        """
        Send a reply message using the provided reply context.
        """
        if not self._running:
            raise RuntimeError("Transport not started")

        if reply_context is None:
            logger.error("reply_context is required")
            return

        message.is_reply = True
        message.correlation_id = reply_context.correlation_id

        try:
            with self._lock:
                if self._router_socket:
                    zmq_identity = reply_context.metadata.get("zmq_identity")
                    if not zmq_identity:
                        logger.error("ZMQ identity not found in reply context")
                        return

                    self._router_socket.send_multipart(
                        zmq_identity, message, flags=zmq.NOBLOCK
                    )
                    logger.debug(
                        f"Sent reply message {message.message_id} type={message.message_type} for correlation {message.correlation_id}"
                    )
                else:
                    logger.error("ROUTER socket not available")
                    return
        except zmq.Again:
            logger.error(f"Send buffer full for reply {message.message_id}")
        except Exception as e:
            logger.exception(
                f"Failed to send reply message {message.message_id}: {e}"
            )

    async def receive_message(self) -> Optional[TransportMessageEnvelope]:
        """
        Receive messages from both ROUTER (new requests) and DEALER (replies) sockets.
        """
        if not self._running:
            raise RuntimeError("Transport not started")

        try:
            # First, check ROUTER socket for incoming requests
            if self._router_socket:
                try:
                    identity, message = (
                        self._router_socket.recv_multipart_nowait()
                    )

                    # Create reply context with routing information
                    reply_context = ReplyContext(
                        reply_address=self._bind_address,
                        correlation_id=message.correlation_id
                        or str(uuid.uuid4()),
                        metadata={"zmq_identity": identity},
                    )

                    logger.debug(
                        f"Received request message {message.message_id} type={message.message_type}"
                    )
                    return TransportMessageEnvelope(
                        message=message, reply_context=reply_context
                    )
                except queue.Empty:
                    pass  # No message available
                except Exception as e:
                    logger.exception(f"Failed to receive message: {e}")

            # Then, check DEALER sockets for replies to our requests
            with self._lock:
                for destination_address, dealer_socket in list(
                    self._dealer_connections.items()
                ):
                    try:
                        message = dealer_socket.recv_pyobj_nowait()

                        if (
                            message.correlation_id
                            and message.correlation_id in self._pending_requests
                        ):
                            self._pending_requests.pop(
                                message.correlation_id, None
                            )

                        logger.debug(
                            f"Received reply message {message.message_id} type={message.message_type}"
                        )
                        return TransportMessageEnvelope(
                            message=message, reply_context=None
                        )
                    except queue.Empty:
                        pass  # No message available
                    except Exception as e:
                        logger.exception(f"Failed to receive message: {e}")
                        continue
        except Exception as e:
            logger.exception(f"Failed to receive message: {e}")
        return None

    async def _connect_to_address(self, destination_address: str) -> bool:
        """
        Create a DEALER connection to the destination address.
        """
        try:
            if not destination_address or not self._is_valid_zmq_address(
                destination_address
            ):
                logger.error(
                    f"Invalid ZMQ address format: {destination_address}"
                )
                return False

            dealer_socket = ZmqDealerSocket[Any](
                self._zmq_ctx, destination_address, bind=False
            )
            self._dealer_connections[destination_address] = dealer_socket
            logger.debug(f"Created DEALER connection to: {destination_address}")
            return True
        except Exception as e:
            logger.exception(
                f"Failed to create DEALER connection to {destination_address}: {e}"
            )
        return False

    def get_address(self) -> str:
        """
        Get the address of the transport.
        """
        return self._bind_address

    def _is_valid_zmq_address(self, address: str) -> bool:
        """
        Validate ZMQ address format.
        """
        return address.startswith(("tcp://", "ipc://", "inproc://"))

    async def close(self) -> None:
        """
        Close all connections and clean up resources.

        Closes all DEALER connections and the ROUTER socket, then clears
        internal state tracking.
        """
        self._running = False
        with self._lock:
            # Close all DEALER connections
            for destination_address, socket in self._dealer_connections.items():
                try:
                    socket.close()
                    logger.debug(
                        f"Closed DEALER connection to {destination_address}"
                    )
                except Exception as e:
                    logger.exception(
                        f"Error closing DEALER connection to {destination_address}: {e}"
                    )
                    raise
            self._dealer_connections.clear()
            self._pending_requests.clear()

            # Close ROUTER socket
            if self._router_socket:
                try:
                    self._router_socket.close()
                    self._router_socket = None
                    logger.debug(
                        f"Closed ROUTER socket for {self._bind_address}"
                    )
                except Exception as e:
                    logger.exception(f"Error closing ROUTER socket: {e}")
                    raise

        logger.debug(
            f"DynamicZmqTransport closed: instance_id={self._instance_id}"
        )
