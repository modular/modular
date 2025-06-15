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

import asyncio
import logging
import pickle
import queue
import uuid
from typing import Any, Callable, Generic, Optional, TypeVar

import msgspec
import zmq
from max.serve.kvcache_agent.dispatcher_base import MessageType, ReplyContext
from max.serve.kvcache_agent.dispatcher_transport import (
    DispatcherTransport,
    TransportMessage,
)
from max.serve.queue.zmq_queue import ZmqPullSocket, ZmqPushSocket

logger = logging.getLogger(__name__)

DispatcherMessagePayload = TypeVar("DispatcherMessagePayload")


class DispatcherMessage(
    msgspec.Struct,
    Generic[DispatcherMessagePayload],
    tag=True,
    kw_only=True,
    omit_defaults=True,
):
    """
    Message envelope for communication with the dispatcher service.

    Contains message payload, routing information, and optional reply context
    for request-response patterns between clients and dispatcher.
    """

    message_type: MessageType
    payload: DispatcherMessagePayload
    destination_address: Optional[str] = None
    reply_context: Optional[ReplyContext] = None


class DispatcherService(Generic[DispatcherMessagePayload]):
    """
    Dispatcher service that bridges local client communication with remote transport.

    Routes messages between local PUSH/PULL sockets and remote transport layer,
    handling both request forwarding and reply routing.
    """

    def __init__(
        self,
        zmq_ctx: zmq.Context,
        send_endpoint: str,
        recv_endpoint: str,
        transport: DispatcherTransport[DispatcherMessagePayload],
        serialize: Callable[[Any], bytes] = pickle.dumps,
        deserialize: Callable[[Any], Any] = pickle.loads,
    ):
        """Initialize dispatcher service with local sockets and remote transport."""
        self.transport = transport

        self.local_pull_socket = ZmqPullSocket[
            DispatcherMessage[DispatcherMessagePayload]
        ](zmq_ctx, recv_endpoint, deserialize=deserialize)
        self.local_push_socket = ZmqPushSocket[
            DispatcherMessage[DispatcherMessagePayload]
        ](zmq_ctx, send_endpoint, serialize=serialize)

        self._running = False
        self._tasks: list[asyncio.Task] = []

        logger.debug(
            f"DispatcherService initialized: send={send_endpoint}, recv={recv_endpoint}"
        )

    async def start(self) -> None:
        """Start the dispatcher service and begin message routing loops."""
        try:
            await self.transport.start()
            self._running = True

            self._tasks = [
                asyncio.create_task(self._local_to_transport_loop()),
                asyncio.create_task(self._transport_to_local_loop()),
            ]
            logger.debug("DispatcherService started")
        except Exception as e:
            logger.exception(f"Failed to start dispatcher service: {e}")
            raise

    async def stop(self) -> None:
        """Stop the dispatcher service and clean up resources."""
        self._running = False

        # Give tasks a chance to finish gracefully before cancelling
        try:
            done, pending = await asyncio.wait(self._tasks, timeout=1.0)

            # If any tasks are still pending after timeout, cancel them
            if pending:
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

            self._tasks.clear()
        except Exception as e:
            logger.exception(f"Failed to stop dispatcher service: {e}")
            raise

        # Close the transport
        try:
            await self.transport.close()
            logger.debug("Transport closed")
        except Exception as e:
            logger.exception(f"Failed to close transport: {e}")
            raise

        # Close local pull socket
        try:
            self.local_pull_socket.close()
        except Exception as e:
            logger.exception(f"Failed to close local pull socket: {e}")
            raise

        # Close local push socket
        try:
            self.local_push_socket.close()
            logger.debug("Sockets closed")
        except Exception as e:
            logger.exception(f"Failed to close local push socket: {e}")
            raise

        logger.debug("DispatcherService stopped")

    async def _local_to_transport_loop(self) -> None:
        """
        Forward messages from local sockets to transport.

        Processes DispatcherMessage instances from local sockets and forwards them
        via transport as TransportMessage instances.
        """
        while self._running:
            try:
                msg = self.local_pull_socket.get_nowait()
                if msg:
                    if not isinstance(msg, DispatcherMessage):
                        logger.error(
                            f"Expected DispatcherMessage, got {type(msg)}: {msg}"
                        )
                        continue

                    transport_message = TransportMessage(
                        message_id=str(uuid.uuid4()),
                        message_type=msg.message_type.value,
                        payload=msg.payload,
                    )

                    if msg.reply_context is not None:
                        # This is a reply, use reply_context for routing
                        await self.transport.send_reply(
                            transport_message, msg.reply_context
                        )
                        logger.debug(
                            f"Forwarded reply message {transport_message.message_id}"
                        )
                    else:
                        # This is a new request, use destination_address
                        await self.transport.send_message(
                            transport_message, msg.destination_address
                        )
                        logger.debug(
                            f"Forwarded request message {transport_message.message_id}"
                        )
                else:
                    await asyncio.sleep(0.001)
            except queue.Empty:
                await asyncio.sleep(0.001)
            except Exception as e:
                logger.exception(
                    f"Failed to forward local message to transport: {e}"
                )
                await asyncio.sleep(0.001)

    async def _transport_to_local_loop(self) -> None:
        """
        Forward messages from transport to local sockets.

        Receives TransportMessageEnvelope instances from transport and converts them
        to DispatcherMessage instances for local socket consumption.
        """
        while self._running:
            try:
                received = await self.transport.receive_message()
                if received:
                    # Convert transport message type string back to enum
                    try:
                        message_type = MessageType(
                            received.message.message_type
                        )
                    except ValueError:
                        logger.error(
                            f"Unknown message type: {received.message.message_type}"
                        )
                        continue

                    dispatcher_message = DispatcherMessage(
                        message_type=message_type,
                        payload=received.message.payload,
                        destination_address=received.reply_context.reply_address
                        if received.reply_context
                        else None,
                        reply_context=received.reply_context,
                    )

                    self.local_push_socket.put_nowait(dispatcher_message)
                    logger.debug(
                        f"Forwarded transport message {received.message.message_id} to local"
                    )
                else:
                    await asyncio.sleep(0.001)
            except Exception as e:
                logger.exception(
                    f"Failed to forward transport message to local: {e}"
                )
                await asyncio.sleep(0.001)
