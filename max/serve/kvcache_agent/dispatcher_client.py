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
from typing import Any, Callable, Optional, TypeVar, cast

import zmq
from max.serve.kvcache_agent.dispatcher_base import MessageType, ReplyContext
from max.serve.kvcache_agent.dispatcher_service import DispatcherMessage
from max.serve.queue.zmq_queue import (
    ZmqPullSocket,
    ZmqPushSocket,
    is_valid_zmq_address,
)

logger = logging.getLogger(__name__)

DispatcherMessagePayload = TypeVar("DispatcherMessagePayload")


class DispatcherClient:
    """
    Client for communicating with dispatcher service.

    Provides typed message sending/receiving with automatic handler registration
    and reply context management for request-response patterns.
    """

    def __init__(
        self,
        zmq_ctx: zmq.Context,
        send_endpoint: str,
        recv_endpoint: str,
    ) -> None:
        """Initialize dispatcher client with ZMQ sockets for communication."""
        self.pull_socket = ZmqPullSocket[DispatcherMessage](
            zmq_ctx, recv_endpoint
        )
        self.push_socket = ZmqPushSocket[DispatcherMessage](
            zmq_ctx, send_endpoint
        )

        # Request handlers
        self._request_handlers: dict[
            MessageType, Callable[[Any, ReplyContext], None]
        ] = {}
        self._reply_handlers: dict[MessageType, Callable[[Any], None]] = {}
        self._general_handlers: dict[MessageType, Callable[[Any], None]] = {}

        self._running = False
        self._thread: Optional[threading.Thread] = None

        logger.debug(
            f"DispatcherClient initialized: send={send_endpoint}, recv={recv_endpoint}"
        )

    def request_handler(
        self, message_type: MessageType
    ) -> Callable[
        [Callable[[DispatcherMessagePayload, ReplyContext], None]],
        Callable[[DispatcherMessagePayload, ReplyContext], None],
    ]:
        """Decorator for registering a request handler that can send replies."""

        def decorator(
            func: Callable[[DispatcherMessagePayload, ReplyContext], None],
        ) -> Callable[[DispatcherMessagePayload, ReplyContext], None]:
            self.register_request_handler(message_type, func)
            return func

        return decorator

    def reply_handler(
        self, message_type: MessageType
    ) -> Callable[
        [Callable[[DispatcherMessagePayload], None]],
        Callable[[DispatcherMessagePayload], None],
    ]:
        """Decorator for registering a reply handler (no reply context)."""

        def decorator(
            func: Callable[[DispatcherMessagePayload], None],
        ) -> Callable[[DispatcherMessagePayload], None]:
            self.register_reply_handler(message_type, func)
            return func

        return decorator

    def handler(
        self, message_type: MessageType
    ) -> Callable[
        [Callable[[DispatcherMessagePayload], None]],
        Callable[[DispatcherMessagePayload], None],
    ]:
        """Decorator for registering a general handler for simple message processing (no reply capability)."""

        def decorator(
            func: Callable[[DispatcherMessagePayload], None],
        ) -> Callable[[DispatcherMessagePayload], None]:
            self.register_handler(message_type, func)
            return func

        return decorator

    def register_request_handler(
        self,
        message_type: MessageType,
        handler: Callable[[DispatcherMessagePayload, ReplyContext], None],
    ) -> None:
        """Register a handler function for request messages."""
        if message_type in self._request_handlers:
            raise ValueError(
                f"Request handler for message type {message_type} already registered"
            )
        self._request_handlers[message_type] = cast(
            Callable[[DispatcherMessagePayload, ReplyContext], None], handler
        )
        logger.debug(
            f"Registered request handler for message type: {message_type}"
        )

    def register_reply_handler(
        self,
        message_type: MessageType,
        handler: Callable[[DispatcherMessagePayload], None],
    ) -> None:
        """Register a handler function for reply messages."""
        if message_type in self._reply_handlers:
            raise ValueError(
                f"Reply handler for message type {message_type} already registered"
            )
        self._reply_handlers[message_type] = cast(
            Callable[[DispatcherMessagePayload], None], handler
        )
        logger.debug(
            f"Registered reply handler for message type: {message_type}"
        )

    def register_handler(
        self,
        message_type: MessageType,
        handler: Callable[[DispatcherMessagePayload], None],
    ) -> None:
        """Register a general handler function for simple message processing (no reply capability)."""
        if message_type in self._general_handlers:
            raise ValueError(
                f"General handler for message type {message_type} already registered"
            )
        self._general_handlers[message_type] = cast(
            Callable[[DispatcherMessagePayload], None], handler
        )
        logger.debug(
            f"Registered general handler for message type: {message_type}"
        )

    def send(
        self,
        message_type: MessageType,
        payload: Any,
        destination_address: Optional[str] = None,
    ) -> None:
        """Send a new message to the specified destination address."""
        if destination_address is not None and not is_valid_zmq_address(
            destination_address
        ):
            logger.error(f"Invalid ZMQ address format: {destination_address}")
            return

        try:
            dispatcher_message = DispatcherMessage(
                message_type, payload, destination_address=destination_address
            )
            self.push_socket.put_nowait(dispatcher_message)
            logger.debug(
                f"Sent message: {message_type} to {destination_address}"
            )
        except Exception as e:
            logger.exception(f"Failed to send message {message_type}: {e}")

    def send_reply(
        self,
        message_type: MessageType,
        payload: Any,
        reply_context: ReplyContext,
    ) -> None:
        """Send a reply message using the provided reply context."""
        try:
            dispatcher_message = DispatcherMessage(
                message_type, payload, reply_context=reply_context
            )
            self.push_socket.put_nowait(dispatcher_message)
            logger.debug(f"Sent reply: {message_type}")
        except Exception as e:
            logger.exception(f"Failed to send reply {message_type}: {e}")

    def start(self) -> None:
        """Start the client and begin listening for incoming messages."""
        if self._running:
            logger.warning("DispatcherClient already running")
            return

        try:
            self._running = True
            self._thread = threading.Thread(
                target=self._listen_loop, daemon=True
            )
            self._thread.start()
            logger.debug("DispatcherClient started")
        except Exception as e:
            logger.exception(f"Failed to start dispatcher client: {e}")
            raise

    def stop(self) -> None:
        """Stop the client and clean up resources."""
        self._running = False

        # Close pull socket
        try:
            self.pull_socket.close()
            logger.debug("Pull socket closed")
        except Exception as e:
            logger.exception(f"Failed to close pull socket: {e}")

        # Close push socket
        try:
            self.push_socket.close()
            logger.debug("Push socket closed")
        except Exception as e:
            logger.exception(f"Failed to close push socket: {e}")

        # Join thread
        try:
            if self._thread:
                self._thread.join(timeout=1.0)
                self._thread = None
        except Exception as e:
            logger.exception(f"Failed to join thread: {e}")

        logger.debug("DispatcherClient stopped")

    def _listen_loop(self) -> None:
        """Listen for incoming messages and dispatch to registered handlers."""
        while self._running:
            try:
                message = self.pull_socket.get_nowait()
                if not isinstance(message, DispatcherMessage):
                    logger.error(
                        f"Expected DispatcherMessage, got {type(message)}"
                    )
                    continue

                # Always call general handler if registered
                general_handler = self._general_handlers.get(
                    message.message_type
                )
                if general_handler:
                    try:
                        general_handler(message.payload)
                        logger.debug(
                            f"Called general handler for: {message.message_type}"
                        )
                    except Exception as handler_exc:
                        logger.error(
                            f"General handler failed for {message.message_type}: {handler_exc}"
                        )

                # This is a request (has reply_context) - call request handler
                if message.reply_context is not None:
                    request_handler = self._request_handlers.get(
                        message.message_type
                    )
                    if request_handler:
                        try:
                            request_handler(
                                message.payload, message.reply_context
                            )
                            logger.debug(
                                f"Called request handler for: {message.message_type}"
                            )
                        except Exception as handler_exc:
                            logger.error(
                                f"Request handler failed for {message.message_type}: {handler_exc}"
                            )
                else:
                    # This is a reply message (no reply_context) - call reply handler
                    reply_handler = self._reply_handlers.get(
                        message.message_type
                    )
                    if reply_handler:
                        try:
                            reply_handler(message.payload)
                            logger.debug(
                                f"Called reply handler for: {message.message_type}"
                            )
                        except Exception as handler_exc:
                            logger.error(
                                f"Reply handler failed for {message.message_type}: {handler_exc}"
                            )
            except queue.Empty:
                time.sleep(0.001)
            except Exception as e:
                logger.exception(f"Failed to receive message: {e}")
                time.sleep(0.001)
