# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import asyncio
import time
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
from max.serve.queue.zmq_queue import generate_zmq_inproc_endpoint


@pytest.mark.asyncio
async def test_dispatcher_client_to_service_communication():
    """Test proper client-server communication through dispatcher services."""
    zmq_ctx = zmq.Context()

    try:
        # Create dispatcher configs
        client_config = DispatcherConfig(
            transport=TransportType.DYNAMIC_ZMQ,
            transport_config=TransportFactory.DynamicZmqTransportConfig(
                bind_address=generate_zmq_inproc_endpoint(),
                instance_id="client",
            ),
        )
        server_bind_address = generate_zmq_inproc_endpoint()
        server_config = DispatcherConfig(
            transport=TransportType.DYNAMIC_ZMQ,
            transport_config=TransportFactory.DynamicZmqTransportConfig(
                bind_address=server_bind_address,
                instance_id="server",
            ),
        )

        # Create factories
        client_factory = DispatcherFactory[dict[str, Union[str, int]]](
            client_config
        )
        server_factory = DispatcherFactory[dict[str, Union[str, int]]](
            server_config
        )

        # Create dispatcher services and clients using factories
        client_dispatcher_service = client_factory.create_service(zmq_ctx)
        client_app = client_factory.create_client(zmq_ctx)

        server_dispatcher_service = server_factory.create_service(zmq_ctx)
        server_app = server_factory.create_client(zmq_ctx)

        received_requests: list[Any] = []
        received_replies: list[Any] = []

        # Server handler - receives requests and sends replies
        @server_app.request_handler(MessageType.PREFILL_REQUEST)
        def handle_request(payload: Any, reply_context: ReplyContext) -> None:
            received_requests.append(payload)
            reply_payload = {
                "result": payload["value"] * 2,
                "status": "processed",
            }
            server_app.send_reply(
                MessageType.PREFILL_RESPONSE, reply_payload, reply_context
            )

        # Client handler - receives replies
        @client_app.reply_handler(MessageType.PREFILL_RESPONSE)
        def handle_reply(payload: Any) -> None:
            received_replies.append(payload)

        # Start services and clients
        await client_dispatcher_service.start()
        await server_dispatcher_service.start()
        client_app.start()
        server_app.start()

        # Allow time for services to fully initialize
        await asyncio.sleep(0.5)

        # Test message from client to server
        test_payload = {"request": "test_data", "value": 42}

        # Send message from client to server
        client_app.send(
            MessageType.PREFILL_REQUEST,
            test_payload,
            destination_address=server_bind_address,
        )

        await asyncio.sleep(0.5)

        # Verify request was received by server
        assert len(received_requests) == 1, (
            f"Expected 1 request, got {len(received_requests)}"
        )
        assert received_requests[0] == test_payload, "Request payload mismatch"

        # Verify reply was received by client
        assert len(received_replies) == 1, (
            f"Expected 1 reply, got {len(received_replies)}"
        )
        assert received_replies[0]["result"] == 84, "Reply result mismatch"
        assert received_replies[0]["status"] == "processed", (
            "Reply status mismatch"
        )

    finally:
        # Allow pending operations to complete before shutdown
        await asyncio.sleep(0.5)
        await client_dispatcher_service.stop()
        await server_dispatcher_service.stop()
        client_app.stop()
        server_app.stop()
        # Allow cleanup before terminating ZMQ context
        await asyncio.sleep(0.5)
        zmq_ctx.term()


@pytest.mark.asyncio
async def test_dispatcher_request_reply_pattern():
    """Test request-reply pattern through dispatcher service and client."""
    zmq_ctx = zmq.Context()

    try:
        # Create dispatcher configs
        instance_a_config = DispatcherConfig(
            transport=TransportType.DYNAMIC_ZMQ,
            transport_config=TransportFactory.DynamicZmqTransportConfig(
                bind_address=generate_zmq_inproc_endpoint(),
                instance_id="instance_a",
            ),
        )
        instance_b_bind_address = generate_zmq_inproc_endpoint()
        instance_b_config = DispatcherConfig(
            transport=TransportType.DYNAMIC_ZMQ,
            transport_config=TransportFactory.DynamicZmqTransportConfig(
                bind_address=instance_b_bind_address,
                instance_id="instance_b",
            ),
        )

        # Create factories
        instance_a_factory = DispatcherFactory[dict[str, Union[str, int]]](
            instance_a_config
        )
        instance_b_factory = DispatcherFactory[dict[str, Union[str, int]]](
            instance_b_config
        )

        # Create dispatcher services and clients using factories
        instance_a_dispatcher = instance_a_factory.create_service(zmq_ctx)
        instance_a_client = instance_a_factory.create_client(zmq_ctx)

        instance_b_dispatcher = instance_b_factory.create_service(zmq_ctx)
        instance_b_client = instance_b_factory.create_client(zmq_ctx)

        # Set up reply tracking
        instance_b_received_requests = []
        instance_a_received_replies = []

        @instance_b_client.request_handler(MessageType.PREFILL_REQUEST)
        def handle_request(payload: Any, reply_context: ReplyContext) -> None:
            instance_b_received_requests.append(payload)
            # Send reply back
            reply_payload = {
                "result": payload["value"] * 2,
                "status": "success",
            }
            instance_b_client.send_reply(
                MessageType.PREFILL_RESPONSE, reply_payload, reply_context
            )

        @instance_a_client.reply_handler(MessageType.PREFILL_RESPONSE)
        def handle_reply(payload: Any) -> None:
            instance_a_received_replies.append(payload)

        # Start everything
        await instance_a_dispatcher.start()
        await instance_b_dispatcher.start()
        instance_a_client.start()
        instance_b_client.start()

        # Allow time for services to fully initialize
        await asyncio.sleep(0.5)

        # Send request from client to server
        request_payload = {"value": 21, "operation": "double"}
        instance_a_client.send(
            MessageType.PREFILL_REQUEST,
            request_payload,
            destination_address=instance_b_bind_address,
        )

        await asyncio.sleep(0.5)

        # Verify request was received and processed
        assert len(instance_b_received_requests) == 1, (
            f"Expected 1 request, got {len(instance_b_received_requests)}"
        )
        assert instance_b_received_requests[0]["value"] == 21
        assert instance_b_received_requests[0]["operation"] == "double"

        # Verify reply was received
        assert len(instance_a_received_replies) == 1, (
            f"Expected 1 reply, got {len(instance_a_received_replies)}"
        )
        assert instance_a_received_replies[0]["result"] == 42  # 21 * 2
        assert instance_a_received_replies[0]["status"] == "success"

    finally:
        # Allow pending operations to complete before shutdown
        await asyncio.sleep(0.5)
        await instance_a_dispatcher.stop()
        await instance_b_dispatcher.stop()
        instance_a_client.stop()
        instance_b_client.stop()
        # Allow cleanup before terminating ZMQ context
        await asyncio.sleep(0.5)
        zmq_ctx.term()


@pytest.mark.asyncio
async def test_multiple_clients_one_server_dispatcher():
    """Test multiple clients communicating with one server through dispatchers."""
    zmq_ctx = zmq.Context()

    try:
        # Server setup using factory
        server_bind_address = generate_zmq_inproc_endpoint()
        server_config = DispatcherConfig(
            transport=TransportType.DYNAMIC_ZMQ,
            transport_config=TransportFactory.DynamicZmqTransportConfig(
                bind_address=server_bind_address,
                instance_id="server",
            ),
        )
        server_factory = DispatcherFactory[dict[str, Union[str, int]]](
            server_config
        )
        server_dispatcher = server_factory.create_service(zmq_ctx)
        server_app = server_factory.create_client(zmq_ctx)

        # Multiple clients setup using factories
        num_clients = 3
        clients = []
        client_dispatchers = []
        client_configs = []
        client_factories = []

        for i in range(num_clients):
            # Create dispatcher config and factory for each client
            client_config = DispatcherConfig(
                transport=TransportType.DYNAMIC_ZMQ,
                transport_config=TransportFactory.DynamicZmqTransportConfig(
                    bind_address=generate_zmq_inproc_endpoint(),
                    instance_id=f"client_{i}",
                ),
            )
            client_factory = DispatcherFactory[dict[str, Union[str, int]]](
                client_config
            )
            client_dispatcher = client_factory.create_service(zmq_ctx)
            client_app = client_factory.create_client(zmq_ctx)

            clients.append(client_app)
            client_dispatchers.append(client_dispatcher)
            client_configs.append(client_config)
            client_factories.append(client_factory)

        # Track received messages
        server_requests: list[Any] = []
        client_replies: list[list[Any]] = [[] for _ in range(num_clients)]

        # Server handler
        @server_app.request_handler(MessageType.PREFILL_REQUEST)
        def handle_server_request(
            payload: Any, reply_context: ReplyContext
        ) -> None:
            server_requests.append(payload)
            reply_payload = {
                "client_id": payload["client_id"],
                "processed_value": payload["value"] + 100,
                "server_response": f"Processed by server for client {payload['client_id']}",
            }
            server_app.send_reply(
                MessageType.PREFILL_RESPONSE, reply_payload, reply_context
            )

        # Client handlers
        for i, client in enumerate(clients):

            def make_handler(client_idx):
                def handle_client_reply(payload: Any) -> None:
                    client_replies[client_idx].append(payload)

                return handle_client_reply

            client.register_reply_handler(
                MessageType.PREFILL_RESPONSE, make_handler(i)
            )

        # Start everything
        await server_dispatcher.start()
        server_app.start()

        for dispatcher in client_dispatchers:
            await dispatcher.start()
        for client in clients:
            client.start()

        # Allow time for services to fully initialize
        await asyncio.sleep(0.5)

        # Each client sends a request
        for i, client in enumerate(clients):
            request_payload = {
                "client_id": i,
                "value": i * 10,
                "message": f"Request from client {i}",
            }
            client.send(
                MessageType.PREFILL_REQUEST,
                request_payload,
                destination_address=server_bind_address,
            )

        await asyncio.sleep(0.5)

        # Verify server received all requests
        assert len(server_requests) == num_clients, (
            f"Expected {num_clients} requests, got {len(server_requests)}"
        )

        # Verify each client received exactly one reply
        for i in range(num_clients):
            assert len(client_replies[i]) == 1, (
                f"Client {i} should receive 1 reply, got {len(client_replies[i])}"
            )
            reply = client_replies[i][0]
            assert reply["client_id"] == i, f"Reply should be for client {i}"
            assert reply["processed_value"] == i * 10 + 100, (
                f"Processed value mismatch for client {i}"
            )

    finally:
        # Allow pending operations to complete before shutdown
        await asyncio.sleep(0.5)
        await server_dispatcher.stop()
        server_app.stop()

        for dispatcher in client_dispatchers:
            await dispatcher.stop()
        for client in clients:
            client.stop()

        # Allow cleanup before terminating ZMQ context
        await asyncio.sleep(0.5)
        zmq_ctx.term()


@pytest.mark.asyncio
async def test_composable_handlers():
    """Test that both general and specific handlers are called for the same message."""
    zmq_ctx = zmq.Context()

    try:
        # Create dispatcher configs
        instance_a_config = DispatcherConfig(
            transport=TransportType.DYNAMIC_ZMQ,
            transport_config=TransportFactory.DynamicZmqTransportConfig(
                bind_address=generate_zmq_inproc_endpoint(),
                instance_id="instance_a",
            ),
        )
        instance_b_bind_address = generate_zmq_inproc_endpoint()
        instance_b_config = DispatcherConfig(
            transport=TransportType.DYNAMIC_ZMQ,
            transport_config=TransportFactory.DynamicZmqTransportConfig(
                bind_address=instance_b_bind_address,
                instance_id="instance_b",
            ),
        )

        # Create factories
        instance_a_factory = DispatcherFactory[dict[str, Union[str, int]]](
            instance_a_config
        )
        instance_b_factory = DispatcherFactory[dict[str, Union[str, int]]](
            instance_b_config
        )

        # Create dispatcher services and clients using factories
        instance_a_dispatcher = instance_a_factory.create_service(zmq_ctx)
        instance_a_client = instance_a_factory.create_client(zmq_ctx)

        instance_b_dispatcher = instance_b_factory.create_service(zmq_ctx)
        instance_b_client = instance_b_factory.create_client(zmq_ctx)

        # Track messages with BOTH general and specific handlers
        instance_a_general_messages = []
        instance_a_reply_messages = []
        instance_b_general_messages = []
        instance_b_request_messages = []

        # General handlers (for logging/monitoring)
        @instance_a_client.handler(MessageType.PREFILL_RESPONSE)
        def log_a_messages(payload: Any) -> None:
            instance_a_general_messages.append(("general_reply", payload))

        @instance_b_client.handler(MessageType.PREFILL_REQUEST)
        def log_b_messages(payload: Any) -> None:
            instance_b_general_messages.append(("general_request", payload))

        # Specific handlers (for business logic)
        @instance_a_client.reply_handler(MessageType.PREFILL_RESPONSE)
        def handle_a_replies(payload: Any) -> None:
            instance_a_reply_messages.append(("specific_reply", payload))

        @instance_b_client.request_handler(MessageType.PREFILL_REQUEST)
        def handle_b_requests(
            payload: Any, reply_context: ReplyContext
        ) -> None:
            instance_b_request_messages.append(("specific_request", payload))
            reply_payload = {
                "result": payload["value"] * 3,
                "status": "processed",
            }
            instance_b_client.send_reply(
                MessageType.PREFILL_RESPONSE, reply_payload, reply_context
            )

        # Start everything
        await instance_a_dispatcher.start()
        await instance_b_dispatcher.start()
        instance_a_client.start()
        instance_b_client.start()

        # Allow time for services to fully initialize
        await asyncio.sleep(0.5)

        # Send request
        request_payload = {"value": 10, "operation": "triple"}
        instance_a_client.send(
            MessageType.PREFILL_REQUEST,
            request_payload,
            destination_address=instance_b_bind_address,
        )

        await asyncio.sleep(0.5)

        # Verify BOTH general and specific handlers were called for the request
        assert len(instance_b_general_messages) == 1, (
            f"Expected 1 general message at B, got {len(instance_b_general_messages)}"
        )
        assert instance_b_general_messages[0] == (
            "general_request",
            request_payload,
        )

        assert len(instance_b_request_messages) == 1, (
            f"Expected 1 specific message at B, got {len(instance_b_request_messages)}"
        )
        assert instance_b_request_messages[0] == (
            "specific_request",
            request_payload,
        )

        # Verify BOTH general and specific handlers were called for the reply
        assert len(instance_a_general_messages) == 1, (
            f"Expected 1 general message at A, got {len(instance_a_general_messages)}"
        )
        assert instance_a_general_messages[0][0] == "general_reply"
        assert instance_a_general_messages[0][1]["result"] == 30  # 10 * 3
        assert instance_a_general_messages[0][1]["status"] == "processed"

        assert len(instance_a_reply_messages) == 1, (
            f"Expected 1 specific message at A, got {len(instance_a_reply_messages)}"
        )
        assert instance_a_reply_messages[0][0] == "specific_reply"
        assert instance_a_reply_messages[0][1]["result"] == 30  # 10 * 3
        assert instance_a_reply_messages[0][1]["status"] == "processed"

    finally:
        # Allow pending operations to complete before shutdown
        await asyncio.sleep(0.5)
        await instance_a_dispatcher.stop()
        await instance_b_dispatcher.stop()
        instance_a_client.stop()
        instance_b_client.stop()
        # Allow cleanup before terminating ZMQ context
        await asyncio.sleep(0.5)
        zmq_ctx.term()


@pytest.mark.asyncio
async def test_error_handling_and_resilience():
    """Test error handling and system resilience."""
    zmq_ctx = zmq.Context()

    try:
        # Create dispatcher configs
        server_bind_address = generate_zmq_inproc_endpoint()
        server_config = DispatcherConfig(
            transport=TransportType.DYNAMIC_ZMQ,
            transport_config=TransportFactory.DynamicZmqTransportConfig(
                bind_address=server_bind_address,
                instance_id="server",
            ),
        )
        client_config = DispatcherConfig(
            transport=TransportType.DYNAMIC_ZMQ,
            transport_config=TransportFactory.DynamicZmqTransportConfig(
                bind_address=generate_zmq_inproc_endpoint(),
                instance_id="client",
            ),
        )

        # Create factories
        server_factory = DispatcherFactory[dict[str, Union[str, int]]](
            server_config
        )
        client_factory = DispatcherFactory[dict[str, Union[str, int]]](
            client_config
        )

        # Create dispatcher services and clients using factories
        server_dispatcher = server_factory.create_service(zmq_ctx)
        server_app = server_factory.create_client(zmq_ctx)

        client_dispatcher = client_factory.create_service(zmq_ctx)
        client_app = client_factory.create_client(zmq_ctx)

        # Track messages
        successful_requests = []
        error_requests = []
        replies_received = []

        # Server handler that sometimes fails
        @server_app.request_handler(MessageType.PREFILL_REQUEST)
        def handle_request(payload: Any, reply_context: ReplyContext) -> None:
            if payload.get("should_fail", False):
                error_requests.append(payload)
                # Simulate handler error - don't send reply
                raise ValueError(
                    f"Simulated error for request {payload.get('id')}"
                )
            else:
                successful_requests.append(payload)
                reply_payload = {
                    "result": "success",
                    "request_id": payload.get("id"),
                }
                server_app.send_reply(
                    MessageType.PREFILL_RESPONSE, reply_payload, reply_context
                )

        @client_app.reply_handler(MessageType.PREFILL_RESPONSE)
        def handle_reply(payload: Any) -> None:
            replies_received.append(payload)

        # Start everything
        await server_dispatcher.start()
        await client_dispatcher.start()
        server_app.start()
        client_app.start()

        # Allow time for services to fully initialize
        await asyncio.sleep(0.5)

        # Send mix of successful and failing requests
        test_requests = [
            {"id": 1, "should_fail": False},
            {"id": 2, "should_fail": True},
            {"id": 3, "should_fail": False},
            {"id": 4, "should_fail": True},
            {"id": 5, "should_fail": False},
        ]

        for request in test_requests:
            client_app.send(
                MessageType.PREFILL_REQUEST,
                request,
                destination_address=server_bind_address,
            )

        await asyncio.sleep(0.5)

        # Verify error handling
        assert len(successful_requests) == 3, (
            f"Expected 3 successful requests, got {len(successful_requests)}"
        )
        assert len(error_requests) == 2, (
            f"Expected 2 error requests, got {len(error_requests)}"
        )
        assert len(replies_received) == 3, (
            f"Expected 3 replies, got {len(replies_received)}"
        )

        # Verify only successful requests got replies
        reply_ids = {reply["request_id"] for reply in replies_received}
        expected_reply_ids = {1, 3, 5}
        assert reply_ids == expected_reply_ids, (
            f"Expected reply IDs {expected_reply_ids}, got {reply_ids}"
        )

    finally:
        # Allow pending operations to complete before shutdown
        await asyncio.sleep(0.5)
        await server_dispatcher.stop()
        await client_dispatcher.stop()
        server_app.stop()
        client_app.stop()
        # Allow cleanup before terminating ZMQ context
        await asyncio.sleep(0.5)
        zmq_ctx.term()


@pytest.mark.asyncio
async def test_high_throughput_performance():
    """Test high throughput message processing."""
    zmq_ctx = zmq.Context()

    try:
        # Create dispatcher configs
        server_bind_address = generate_zmq_inproc_endpoint()
        server_config = DispatcherConfig(
            transport=TransportType.DYNAMIC_ZMQ,
            transport_config=TransportFactory.DynamicZmqTransportConfig(
                bind_address=server_bind_address,
                instance_id="server",
            ),
        )
        client_config = DispatcherConfig(
            transport=TransportType.DYNAMIC_ZMQ,
            transport_config=TransportFactory.DynamicZmqTransportConfig(
                bind_address=generate_zmq_inproc_endpoint(),
                instance_id="client",
            ),
        )

        # Create factories
        server_factory = DispatcherFactory[dict[str, Union[str, int]]](
            server_config
        )
        client_factory = DispatcherFactory[dict[str, Union[str, int]]](
            client_config
        )

        # Create dispatcher services and clients using factories
        server_dispatcher = server_factory.create_service(zmq_ctx)
        server_app = server_factory.create_client(zmq_ctx)

        client_dispatcher = client_factory.create_service(zmq_ctx)
        client_app = client_factory.create_client(zmq_ctx)

        # Track performance
        requests_processed = 0
        replies_received = 0
        start_time = None

        @server_app.request_handler(MessageType.PREFILL_REQUEST)
        def handle_request(payload: Any, reply_context: ReplyContext) -> None:
            nonlocal requests_processed
            requests_processed += 1
            reply_payload = {
                "result": payload["value"] * 2,
                "id": payload["id"],
            }
            server_app.send_reply(
                MessageType.PREFILL_RESPONSE, reply_payload, reply_context
            )

        @client_app.reply_handler(MessageType.PREFILL_RESPONSE)
        def handle_reply(payload: Any) -> None:
            nonlocal replies_received
            replies_received += 1

        # Start everything
        await server_dispatcher.start()
        await client_dispatcher.start()
        server_app.start()
        client_app.start()

        # Allow time for services to fully initialize
        await asyncio.sleep(0.5)

        # Send high volume of messages
        num_messages = 1000
        start_time = time.time()

        for i in range(num_messages):
            request_payload = {"id": i, "value": i}
            client_app.send(
                MessageType.PREFILL_REQUEST,
                request_payload,
                destination_address=server_bind_address,
            )

        # Wait for processing
        timeout = 30.0  # 30 second timeout
        elapsed = 0.0
        while (
            requests_processed < num_messages or replies_received < num_messages
        ) and elapsed < timeout:
            await asyncio.sleep(0.5)
            elapsed = time.time() - start_time

        end_time = time.time()
        duration = end_time - start_time

        # Verify all messages were processed
        assert requests_processed == num_messages, (
            f"Expected {num_messages} requests processed, got {requests_processed}"
        )
        assert replies_received == num_messages, (
            f"Expected {num_messages} replies received, got {replies_received}"
        )

        # Calculate throughput
        throughput = num_messages / duration
        print(
            f"High throughput test: {num_messages} messages in {duration:.2f}s = {throughput:.1f} msg/s"
        )

        # Expect reasonable throughput (adjust based on system capabilities)
        assert throughput > 100, f"Throughput too low: {throughput:.1f} msg/s"

    finally:
        # Allow pending operations to complete before shutdown
        await asyncio.sleep(0.5)
        await server_dispatcher.stop()
        await client_dispatcher.stop()
        server_app.stop()
        client_app.stop()
        # Allow cleanup before terminating ZMQ context
        await asyncio.sleep(0.5)
        zmq_ctx.term()


@pytest.mark.asyncio
async def test_connection_failure_recovery():
    """Test connection failure and recovery scenarios."""
    zmq_ctx = zmq.Context()

    try:
        # Create dispatcher config
        client_config = DispatcherConfig(
            transport=TransportType.DYNAMIC_ZMQ,
            transport_config=TransportFactory.DynamicZmqTransportConfig(
                bind_address=generate_zmq_inproc_endpoint(),
                instance_id="client",
            ),
        )

        # Create factory
        client_factory = DispatcherFactory[dict[str, Union[str, int]]](
            client_config
        )

        # Create dispatcher service and client using factory
        client_dispatcher = client_factory.create_service(zmq_ctx)
        client_app = client_factory.create_client(zmq_ctx)

        # Start client
        await client_dispatcher.start()
        client_app.start()

        # Allow time for services to fully initialize
        await asyncio.sleep(0.5)

        # Try to send to non-existent server (should not crash)
        invalid_endpoint = "tcp://127.0.0.1:99999"  # Invalid port

        try:
            client_app.send(
                MessageType.PREFILL_REQUEST,
                {"test": "message"},
                destination_address=invalid_endpoint,
            )
            # Should not crash, just log error
            await asyncio.sleep(0.5)
            # connection failed gracefully
        except Exception as e:
            # Should not reach here - errors should be handled internally
            pytest.fail(f"Unexpected exception: {e}")
    finally:
        # Allow pending operations to complete before shutdown
        await asyncio.sleep(0.5)
        await client_dispatcher.stop()
        client_app.stop()
        # Allow cleanup before terminating ZMQ context
        await asyncio.sleep(0.5)
        zmq_ctx.term()


@pytest.mark.asyncio
async def test_handler_exception_isolation():
    """Test that handler exceptions don't crash the process and other messages are still processed."""
    zmq_ctx = zmq.Context()

    try:
        # Create dispatcher configs
        server_bind_address = generate_zmq_inproc_endpoint()
        server_config = DispatcherConfig(
            transport=TransportType.DYNAMIC_ZMQ,
            transport_config=TransportFactory.DynamicZmqTransportConfig(
                bind_address=server_bind_address,
                instance_id="server",
            ),
        )
        client_config = DispatcherConfig(
            transport=TransportType.DYNAMIC_ZMQ,
            transport_config=TransportFactory.DynamicZmqTransportConfig(
                bind_address=generate_zmq_inproc_endpoint(),
                instance_id="client",
            ),
        )

        # Create factories
        server_factory = DispatcherFactory[dict[str, Union[str, int]]](
            server_config
        )
        client_factory = DispatcherFactory[dict[str, Union[str, int]]](
            client_config
        )

        # Create dispatcher services and clients using factories
        server_dispatcher = server_factory.create_service(zmq_ctx)
        server_app = server_factory.create_client(zmq_ctx)

        client_dispatcher = client_factory.create_service(zmq_ctx)
        client_app = client_factory.create_client(zmq_ctx)

        # Track messages
        received_requests: list[Any] = []
        received_replies: list[Any] = []
        exception_count = 0

        # Server handler that throws exception on specific requests
        @server_app.request_handler(MessageType.PREFILL_REQUEST)
        def handle_request_with_exception(
            payload: Any, reply_context: ReplyContext
        ) -> None:
            nonlocal exception_count
            received_requests.append(payload)

            if payload.get("should_throw", False):
                exception_count += 1
                raise RuntimeError("Intentional test exception")

            # Send normal reply for non-exception requests
            reply_payload = {
                "result": payload["value"] * 2,
                "status": "success",
            }
            server_app.send_reply(
                MessageType.PREFILL_RESPONSE, reply_payload, reply_context
            )

        # Client handler for replies
        @client_app.reply_handler(MessageType.PREFILL_RESPONSE)
        def handle_reply(payload: Any) -> None:
            received_replies.append(payload)

        # Start everything
        await server_dispatcher.start()
        await client_dispatcher.start()
        server_app.start()
        client_app.start()

        # Allow time for services to fully initialize
        await asyncio.sleep(0.5)

        # Send request that should cause exception
        exception_request = {"value": 10, "should_throw": True}
        client_app.send(
            MessageType.PREFILL_REQUEST,
            exception_request,
            destination_address=server_bind_address,
        )

        # Wait a bit for the exception to be processed
        await asyncio.sleep(0.5)

        # Send normal request to verify the process is still working
        normal_request = {"value": 20, "should_throw": False}
        client_app.send(
            MessageType.PREFILL_REQUEST,
            normal_request,
            destination_address=server_bind_address,
        )

        # Give time for message processing
        await asyncio.sleep(0.5)

        # Verify both requests were received
        assert len(received_requests) == 2, (
            f"Expected 2 requests, got {len(received_requests)}"
        )
        assert exception_count == 1, (
            f"Expected 1 exception, got {exception_count}"
        )

        # Verify only the normal request got a reply (exception request should not reply)
        assert len(received_replies) == 1, (
            f"Expected 1 reply, got {len(received_replies)}"
        )
        assert received_replies[0]["result"] == 40, (
            "Normal request should have been processed correctly"
        )

    finally:
        # Allow pending operations to complete before shutdown
        await asyncio.sleep(0.5)
        await server_dispatcher.stop()
        await client_dispatcher.stop()
        server_app.stop()
        client_app.stop()
        # Allow cleanup before terminating ZMQ context
        await asyncio.sleep(0.5)
        zmq_ctx.term()


@pytest.mark.asyncio
async def test_no_handler_registered():
    """Test that messages with no registered handler are handled gracefully."""
    zmq_ctx = zmq.Context()

    try:
        # Create dispatcher configs
        server_bind_address = generate_zmq_inproc_endpoint()
        server_config = DispatcherConfig(
            transport=TransportType.DYNAMIC_ZMQ,
            transport_config=TransportFactory.DynamicZmqTransportConfig(
                bind_address=server_bind_address,
                instance_id="server",
            ),
        )
        client_config = DispatcherConfig(
            transport=TransportType.DYNAMIC_ZMQ,
            transport_config=TransportFactory.DynamicZmqTransportConfig(
                bind_address=generate_zmq_inproc_endpoint(),
                instance_id="client",
            ),
        )

        # Create factories
        server_factory = DispatcherFactory[dict[str, Union[str, int]]](
            server_config
        )
        client_factory = DispatcherFactory[dict[str, Union[str, int]]](
            client_config
        )

        # Create dispatcher services and clients using factories
        server_dispatcher = server_factory.create_service(zmq_ctx)
        server_app = server_factory.create_client(zmq_ctx)

        client_dispatcher = client_factory.create_service(zmq_ctx)
        client_app = client_factory.create_client(zmq_ctx)

        # Intentionally DO NOT register any handlers for server
        # This will test what happens when a message is received with no handler

        # Start everything
        await server_dispatcher.start()
        await client_dispatcher.start()
        server_app.start()
        client_app.start()

        # Allow time for services to fully initialize
        await asyncio.sleep(0.5)

        # Send request to server which has no handler registered
        request_payload = {"value": 42, "operation": "test"}
        client_app.send(
            MessageType.PREFILL_REQUEST,
            request_payload,
            destination_address=server_bind_address,
        )

        # Give time for message processing
        await asyncio.sleep(0.2)

        # Send another message to verify the system is still responsive
        client_app.send(
            MessageType.PREFILL_REQUEST,
            {"value": 123, "operation": "test2"},
            destination_address=server_bind_address,
        )

        await asyncio.sleep(0.5)

        # If we reach here without exceptions, the test passes
        # The system should handle unhandled messages gracefully

    finally:
        # Allow pending operations to complete before shutdown
        await asyncio.sleep(0.5)
        await server_dispatcher.stop()
        await client_dispatcher.stop()
        server_app.stop()
        client_app.stop()
        # Allow cleanup before terminating ZMQ context
        await asyncio.sleep(0.5)
        zmq_ctx.term()


@pytest.mark.asyncio
async def test_invalid_destination_address():
    """Test that sending to invalid addresses is handled gracefully."""
    zmq_ctx = zmq.Context()

    try:
        # Create dispatcher config
        client_config = DispatcherConfig(
            transport=TransportType.DYNAMIC_ZMQ,
            transport_config=TransportFactory.DynamicZmqTransportConfig(
                bind_address=generate_zmq_inproc_endpoint(),
                instance_id="client",
            ),
        )

        # Create factory
        client_factory = DispatcherFactory[dict[str, Union[str, int]]](
            client_config
        )

        # Create dispatcher service and client using factory
        client_dispatcher = client_factory.create_service(zmq_ctx)
        client_app = client_factory.create_client(zmq_ctx)

        # Start dispatcher and client
        await client_dispatcher.start()
        client_app.start()

        # Allow time for services to fully initialize
        await asyncio.sleep(0.5)

        # Test various invalid addresses
        invalid_addresses = [
            "invalid://bad.address",  # Invalid protocol
            "tcp://",  # Incomplete address
            "not-an-address-at-all",  # Completely malformed
            "",  # Empty string
            "tcp://nonexistent.host:99999",  # Non-existent host
        ]

        for invalid_address in invalid_addresses:
            # Send request to invalid address
            request_payload = {"value": 42, "target": invalid_address}
            client_app.send(
                MessageType.PREFILL_REQUEST,
                request_payload,
                destination_address=invalid_address,
            )

            # Give time for error handling
            await asyncio.sleep(0.05)

        # If we reach here without exceptions, the test passes
        # The system should handle invalid addresses gracefully

    finally:
        # Allow pending operations to complete before shutdown
        await asyncio.sleep(0.5)
        await client_dispatcher.stop()
        client_app.stop()
        # Allow cleanup before terminating ZMQ context
        await asyncio.sleep(0.5)
        zmq_ctx.term()


@pytest.mark.asyncio
async def test_duplicate_handler_registration():
    """Test that registering duplicate handlers raises ValueError."""
    zmq_ctx = zmq.Context()

    try:
        # Create dispatcher config
        client_config = DispatcherConfig(
            transport=TransportType.DYNAMIC_ZMQ,
            transport_config=TransportFactory.DynamicZmqTransportConfig(
                bind_address=generate_zmq_inproc_endpoint(),
                instance_id="client",
            ),
        )

        # Create factory
        client_factory = DispatcherFactory[dict[str, Union[str, int]]](
            client_config
        )

        # Create client
        client_app = client_factory.create_client(zmq_ctx)

        # Register first handler
        @client_app.request_handler(MessageType.PREFILL_REQUEST)
        def first_handler(payload: Any, reply_context: ReplyContext) -> None:
            pass

        # Attempt to register second handler for same message type - should raise ValueError
        with pytest.raises(
            ValueError,
            match="Request handler for message type .* already registered",
        ):

            @client_app.request_handler(MessageType.PREFILL_REQUEST)
            def second_handler(
                payload: Any, reply_context: ReplyContext
            ) -> None:
                pass

        # Test duplicate reply handler registration
        @client_app.reply_handler(MessageType.PREFILL_RESPONSE)
        def first_reply_handler(payload: Any) -> None:
            pass

        with pytest.raises(
            ValueError,
            match="Reply handler for message type .* already registered",
        ):

            @client_app.reply_handler(MessageType.PREFILL_RESPONSE)
            def second_reply_handler(payload: Any) -> None:
                pass

        # Test duplicate general handler registration
        @client_app.handler(MessageType.PREFILL_REQUEST)
        def first_general_handler(payload: Any) -> None:
            pass

        with pytest.raises(
            ValueError,
            match="General handler for message type .* already registered",
        ):

            @client_app.handler(MessageType.PREFILL_REQUEST)
            def second_general_handler(payload: Any) -> None:
                pass

    finally:
        client_app.stop()
        # Allow cleanup before terminating ZMQ context
        await asyncio.sleep(0.5)
        zmq_ctx.term()
