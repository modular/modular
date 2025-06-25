# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import asyncio
from typing import Any, Union

import pytest
import zmq
from max.serve.kvcache_agent.dispatcher_base import MessageType, ReplyContext
from max.serve.kvcache_agent.dispatcher_factory import (
    DispatcherConfig,
    DispatcherFactory,
    TransportFactory,
    TransportType,
)
from max.serve.kvcache_agent.dispatcher_transport import TransportMessage
from max.serve.queue.zmq_queue import generate_zmq_inproc_endpoint


@pytest.mark.asyncio
async def test_dispatcher_factory_dynamic_zmq() -> None:
    """Test DispatcherFactory creation from JSON configuration with dynamic transport."""
    try:
        config = DispatcherConfig(
            transport=TransportType.DYNAMIC_ZMQ,
            transport_config=TransportFactory.DynamicZmqTransportConfig(
                bind_address=generate_zmq_inproc_endpoint(),
                instance_id="test_factory",
            ),
        )
        factory = DispatcherFactory[int](
            config, transport_payload_type=TransportMessage[int]
        )

        # Test creating server and client
        zmq_ctx = zmq.Context()
        server = factory.create_service(zmq_ctx)
        client = factory.create_client(zmq_ctx)

        assert server is not None
        assert client is not None

        # Clean up
        await server.transport.close()

    finally:
        pass


@pytest.mark.asyncio
async def test_end_to_end_communication_with_config() -> None:
    """Test end-to-end communication using configuration-based factory."""
    zmq_ctx = zmq.Context()

    try:
        # Create server factory from config
        server_config = DispatcherConfig(
            transport=TransportType.DYNAMIC_ZMQ,
            transport_config=TransportFactory.DynamicZmqTransportConfig(
                bind_address=generate_zmq_inproc_endpoint(),
                instance_id="test_server",
            ),
        )

        server_factory = DispatcherFactory[dict[str, Union[int, str]]](
            server_config,
            transport_payload_type=TransportMessage[dict[str, Union[int, str]]],
        )
        server_dispatcher = server_factory.create_service(zmq_ctx)
        server_client = server_factory.create_client(zmq_ctx)

        # Create client factory from config
        client_config = DispatcherConfig(
            transport=TransportType.DYNAMIC_ZMQ,
            transport_config=TransportFactory.DynamicZmqTransportConfig(
                bind_address=generate_zmq_inproc_endpoint(),
                instance_id="test_client",
            ),
        )

        client_factory = DispatcherFactory[dict[str, Union[int, str]]](
            client_config,
            transport_payload_type=TransportMessage[dict[str, Union[int, str]]],
        )
        client_dispatcher = client_factory.create_service(zmq_ctx)
        client_app = client_factory.create_client(zmq_ctx)

        # Set up message handling
        received_requests: list[Any] = []
        received_replies: list[Any] = []

        @server_client.request_handler(MessageType.PREFILL_REQUEST)
        def handle_request(payload: Any, reply_context: ReplyContext) -> None:
            received_requests.append(payload)
            reply_payload = {
                "result": payload["value"] * 2,
                "status": "processed",
            }
            server_client.send_reply(
                MessageType.PREFILL_RESPONSE, reply_payload, reply_context
            )

        @client_app.reply_handler(MessageType.PREFILL_RESPONSE)
        def handle_reply(payload: Any) -> None:
            received_replies.append(payload)

        # Start everything
        await server_dispatcher.start()
        await client_dispatcher.start()
        server_client.start()
        client_app.start()

        # Allow time for services to fully initialize
        await asyncio.sleep(0.5)

        # Send message from client to server
        test_payload = {"request": "config_test", "value": 21}
        client_app.send(
            MessageType.PREFILL_REQUEST,
            test_payload,
            destination_address=server_dispatcher.transport.get_address(),
        )

        await asyncio.sleep(0.5)

        # Verify communication worked
        assert len(received_requests) == 1
        assert received_requests[0] == test_payload
        assert len(received_replies) == 1
        assert received_replies[0]["result"] == 42
        assert received_replies[0]["status"] == "processed"

    finally:
        # Allow pending operations to complete before shutdown
        await asyncio.sleep(0.5)
        # Clean up and close ZMQ context
        await server_dispatcher.stop()
        await client_dispatcher.stop()
        server_client.stop()
        client_app.stop()
        # Allow cleanup before terminating ZMQ context
        await asyncio.sleep(0.5)
        zmq_ctx.term()
