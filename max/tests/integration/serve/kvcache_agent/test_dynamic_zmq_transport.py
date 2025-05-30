# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import asyncio
from typing import Any

import pytest
import zmq
from max.serve.kvcache_agent.dispatcher_base import MessageType
from max.serve.kvcache_agent.dispatcher_transport import (
    DynamicZmqTransport,
    TransportMessage,
)
from max.serve.queue.zmq_queue import generate_zmq_inproc_endpoint


def create_test_message(
    message_id: str, payload: dict[str, Any]
) -> TransportMessage:
    """Create a test message with the given ID and payload."""
    return TransportMessage(
        message_id=message_id,
        message_type=MessageType.PREFILL_REQUEST.value,
        payload=payload,
    )


@pytest.mark.asyncio
async def test_simple_send_receive():
    """Test basic send and receive between two transports."""
    zmq_ctx = zmq.Context()

    try:
        # Create sender and receiver
        sender = DynamicZmqTransport(
            zmq_ctx, generate_zmq_inproc_endpoint(), "sender"
        )
        receiver = DynamicZmqTransport(
            zmq_ctx, generate_zmq_inproc_endpoint(), "receiver"
        )

        # Start both transports
        await sender.start()
        await receiver.start()

        # Send a message from sender to receiver
        message = create_test_message("test-msg-1", {"content": "hello world"})
        await sender.send_message(
            message, destination_address=receiver.get_address()
        )

        # Receive the message
        received = None
        for _ in range(10):
            received = await receiver.receive_message()
            if received:
                break
            await asyncio.sleep(0.1)

        # Verify the message was received correctly
        assert received is not None, "Message was not received"
        assert received.message.payload["content"] == "hello world"
        assert received.message.source_id == "sender"
        assert received.reply_context is not None, (
            "Reply context should be available"
        )

    finally:
        # Allow pending operations to complete before shutdown
        await asyncio.sleep(0.1)
        await sender.close()
        await receiver.close()
        # Allow cleanup before terminating ZMQ context
        await asyncio.sleep(0.1)
        zmq_ctx.term()


@pytest.mark.asyncio
async def test_request_reply_pattern():
    """Test request-reply communication pattern."""
    zmq_ctx = zmq.Context()

    try:
        # Create client and server
        client = DynamicZmqTransport(
            zmq_ctx, generate_zmq_inproc_endpoint(), "client"
        )
        server = DynamicZmqTransport(
            zmq_ctx, generate_zmq_inproc_endpoint(), "server"
        )

        await client.start()
        await server.start()

        # Client sends request to server
        request = create_test_message("request-1", {"question": "What is 2+2?"})
        await client.send_message(
            request, destination_address=server.get_address()
        )

        # Server receives request
        received_request = None
        for _ in range(10):
            received_request = await server.receive_message()
            if received_request:
                break
            await asyncio.sleep(0.1)

        assert received_request is not None, "Request was not received"
        assert received_request.message.payload["question"] == "What is 2+2?"

        # Server sends reply
        reply = TransportMessage(
            message_id="reply-1",
            message_type=MessageType.PREFILL_RESPONSE.value,
            payload={"answer": "4"},
        )
        await server.send_reply(reply, received_request.reply_context)

        # Client receives reply
        received_reply = None
        for _ in range(10):
            received_reply = await client.receive_message()
            if received_reply:
                break
            await asyncio.sleep(0.1)

        assert received_reply is not None, "Reply was not received"
        assert received_reply.message.payload["answer"] == "4"
        assert received_reply.message.is_reply is True

    finally:
        # Allow pending operations to complete before shutdown
        await asyncio.sleep(0.1)
        await client.close()
        await server.close()
        # Allow cleanup before terminating ZMQ context
        await asyncio.sleep(0.1)
        zmq_ctx.term()


@pytest.mark.asyncio
async def test_multiple_clients_one_server():
    """Test multiple clients sending to one server."""
    zmq_ctx = zmq.Context()

    try:
        # Create one server and three clients
        server = DynamicZmqTransport(
            zmq_ctx, generate_zmq_inproc_endpoint(), "server"
        )

        clients = []
        for i in range(3):
            client_endpoint = generate_zmq_inproc_endpoint()
            client = DynamicZmqTransport(
                zmq_ctx, client_endpoint, f"client_{i}"
            )
            clients.append(client)

        # Start all transports
        await server.start()
        for client in clients:
            await client.start()

        # Each client sends a message to the server
        for i, client in enumerate(clients):
            message = create_test_message(
                f"msg-{i}", {"client_id": i, "data": f"message from client {i}"}
            )
            await client.send_message(
                message, destination_address=server.get_address()
            )

        # Server receives all messages
        received_messages = []
        for _ in range(30):
            received = await server.receive_message()
            if received:
                received_messages.append(received)
                if len(received_messages) == 3:
                    break
            await asyncio.sleep(0.1)

        # Verify all messages were received
        assert len(received_messages) == 3, (
            f"Expected 3 messages, got {len(received_messages)}"
        )

        # Verify each client sent a message
        client_ids = {
            msg.message.payload["client_id"] for msg in received_messages
        }
        assert client_ids == {0, 1, 2}, (
            f"Expected client IDs {{0, 1, 2}}, got {client_ids}"
        )

    finally:
        # Allow pending operations to complete before shutdown
        await asyncio.sleep(0.1)
        await server.close()
        for client in clients:
            await client.close()
        # Allow cleanup before terminating ZMQ context
        await asyncio.sleep(0.1)
        zmq_ctx.term()


@pytest.mark.asyncio
async def test_multiple_clients_multiple_servers():
    """Test N:M communication with proper message routing and replies."""
    zmq_ctx = zmq.Context()

    try:
        # Create 2 servers
        servers = []
        server_endpoints = []
        for i in range(2):
            server = DynamicZmqTransport(
                zmq_ctx, generate_zmq_inproc_endpoint(), f"server_{i}"
            )
            servers.append(server)
            server_endpoints.append(server.get_address())

        # Create 3 clients
        clients = []
        client_endpoints = []
        for i in range(3):
            client = DynamicZmqTransport(
                zmq_ctx, generate_zmq_inproc_endpoint(), f"client_{i}"
            )
            clients.append(client)

        # Start all transports
        for server in servers:
            await server.start()
        for client in clients:
            await client.start()

        # Each client sends a message to a specific server (round-robin)
        sent_requests = []
        for client_idx, client in enumerate(clients):
            server_idx = client_idx % len(servers)
            target_server = server_endpoints[server_idx]

            message = create_test_message(
                f"request_from_client_{client_idx}",
                {
                    "client_id": client_idx,
                    "target_server": server_idx,
                    "question": f"Hello from client {client_idx}",
                },
            )

            await client.send_message(
                message, destination_address=target_server
            )
            sent_requests.append((client_idx, server_idx, message.message_id))

        # Servers receive requests and send replies
        server_received: list[list[Any]] = [[] for _ in range(len(servers))]

        # Collect all requests on servers
        for _ in range(50):
            for server_idx, server in enumerate(servers):
                received = await server.receive_message()
                if received:
                    server_received[server_idx].append(received)

            # Check if we've received all expected messages
            total_received = sum(len(msgs) for msgs in server_received)
            if total_received == len(sent_requests):
                break
            await asyncio.sleep(0.1)

        # Verify each server received the correct messages
        assert len(server_received[0]) == 2, (
            f"Server 0 should receive 2 messages, got {len(server_received[0])}"
        )  # clients 0, 2
        assert len(server_received[1]) == 1, (
            f"Server 1 should receive 1 message, got {len(server_received[1])}"
        )  # client 1

        # Servers send replies back
        for server_idx, received_messages in enumerate(server_received):
            for received_msg in received_messages:
                client_id = received_msg.message.payload["client_id"]

                reply = TransportMessage(
                    message_id=f"reply_from_server_{server_idx}_to_client_{client_id}",
                    message_type=MessageType.PREFILL_RESPONSE.value,
                    payload={
                        "server_id": server_idx,
                        "original_client": client_id,
                        "answer": f"Hello back from server {server_idx}",
                    },
                )

                await servers[server_idx].send_reply(
                    reply, received_msg.reply_context
                )

        # Clients receive replies
        client_replies: list[list[Any]] = [[] for _ in range(len(clients))]

        for _ in range(50):
            for client_idx, client in enumerate(clients):
                received = await client.receive_message()
                if received:
                    client_replies[client_idx].append(received)

            # Check if we've received all expected replies
            total_replies = sum(len(replies) for replies in client_replies)
            if total_replies == len(sent_requests):
                break
            await asyncio.sleep(0.1)

        # Verify routing correctness
        for client_idx, replies in enumerate(client_replies):
            assert len(replies) == 1, (
                f"Client {client_idx} should receive exactly 1 reply, got {len(replies)}"
            )

            reply_envelope = replies[0]
            assert reply_envelope.message.is_reply is True, (
                "Message should be marked as reply"
            )
            assert (
                reply_envelope.message.payload["original_client"] == client_idx
            ), f"Reply should be for client {client_idx}"

            # Verify the reply came from the correct server
            expected_server = client_idx % len(servers)
            actual_server = reply_envelope.message.payload["server_id"]
            assert actual_server == expected_server, (
                f"Client {client_idx} should get reply from server {expected_server}, got {actual_server}"
            )

        print(
            f"Successfully routed {len(sent_requests)} requests and {sum(len(replies) for replies in client_replies)} replies"
        )
        print(f" Server 0 handled {len(server_received[0])} requests")
        print(f" Server 1 handled {len(server_received[1])} requests")

    finally:
        # Allow pending operations to complete before shutdown
        await asyncio.sleep(0.1)
        for server in servers:
            await server.close()
        for client in clients:
            await client.close()
        # Allow cleanup before terminating ZMQ context
        await asyncio.sleep(0.1)
        zmq_ctx.term()


@pytest.mark.asyncio
async def test_connection_failure():
    """Test behavior when connecting to non-existent endpoint."""
    zmq_ctx = zmq.Context()

    try:
        bad_endpoint = (
            generate_zmq_inproc_endpoint()
        )  # No one is listening here

        sender = DynamicZmqTransport(
            zmq_ctx, generate_zmq_inproc_endpoint(), "sender"
        )
        await sender.start()

        # Try to send to non-existent endpoint
        message = create_test_message(
            "test-msg", {"content": "this should fail"}
        )

        # This should raise an exception or handle gracefully
        try:
            await sender.send_message(message, destination_address=bad_endpoint)
            # If we get here, the implementation handles it gracefully
        except (RuntimeError, ValueError) as e:
            # Expected behavior - connection should fail
            assert "Failed to establish connection" in str(e)

    finally:
        # Allow pending operations to complete before shutdown
        await asyncio.sleep(0.1)
        await sender.close()
        # Allow cleanup before terminating ZMQ context
        await asyncio.sleep(0.1)
        zmq_ctx.term()


@pytest.mark.asyncio
async def test_high_throughput():
    """Test sending many messages quickly."""
    zmq_ctx = zmq.Context()

    try:
        sender = DynamicZmqTransport(
            zmq_ctx, generate_zmq_inproc_endpoint(), "sender"
        )
        receiver = DynamicZmqTransport(
            zmq_ctx, generate_zmq_inproc_endpoint(), "receiver"
        )

        await sender.start()
        await receiver.start()

        # Send 1000 messages quickly
        num_messages = 1000
        for i in range(num_messages):
            message = create_test_message(f"msg-{i}", {"sequence": i})
            await sender.send_message(
                message, destination_address=receiver.get_address()
            )

        # Receive all messages
        received_count = 0
        for _ in range(1000):
            received = await receiver.receive_message()
            if received:
                received_count += 1
                if received_count == num_messages:
                    break
            else:
                # Allow time for message processing
                await asyncio.sleep(0.1)

        assert received_count == num_messages, (
            f"Expected {num_messages} messages, got {received_count}"
        )

    finally:
        # Allow pending operations to complete before shutdown
        await asyncio.sleep(0.1)
        await sender.close()
        await receiver.close()
        # Allow cleanup before terminating ZMQ context
        await asyncio.sleep(0.1)
        zmq_ctx.term()


@pytest.mark.asyncio
async def test_no_destination_address_provided():
    """Test behavior when no destination address is provided and no default is set."""
    zmq_ctx = zmq.Context()

    try:
        # Create transport with NO default destination address
        transport = DynamicZmqTransport(
            zmq_ctx=zmq_ctx,
            bind_address=generate_zmq_inproc_endpoint(),
            instance_id="test_transport",
            default_destination_address=None,  # Explicitly set to None
        )

        await transport.start()

        # Send message without destination_address parameter (should use None)
        message_1 = create_test_message(
            "test-msg-1", {"test_case": "no_destination_param"}
        )
        await transport.send_message(
            message_1
        )  # No destination_address provided

        # Send message with explicit None destination_address
        message_2 = create_test_message(
            "test-msg-2", {"test_case": "explicit_none_destination"}
        )
        await transport.send_message(message_2, destination_address=None)

        # Send message with empty string destination_address
        message_3 = create_test_message(
            "test-msg-3", {"test_case": "empty_string_destination"}
        )
        await transport.send_message(message_3, destination_address="")

        await asyncio.sleep(0.1)

        # If we reach here without exceptions, the transport handled the errors gracefully
        # The transport should log errors but not crash

        # Verify transport can still send to a valid destination after errors
        # Create a receiver to test that the transport still works
        receiver = DynamicZmqTransport(
            zmq_ctx=zmq_ctx,
            bind_address=generate_zmq_inproc_endpoint(),
            instance_id="receiver",
        )
        await receiver.start()

        # Send a valid message to verify the transport is still functional
        valid_message = create_test_message(
            "test-msg-4", {"test_case": "valid_destination"}
        )
        await transport.send_message(
            valid_message, destination_address=receiver.get_address()
        )

        # Verify the valid message was received
        received = None
        for _ in range(10):
            received = await receiver.receive_message()
            if received:
                break
            await asyncio.sleep(0.1)

        assert received is not None, (
            "Valid message should have been received after error cases"
        )
        assert received.message.payload["test_case"] == "valid_destination"

        await receiver.close()

    finally:
        # Allow pending operations to complete before shutdown
        await asyncio.sleep(0.1)
        await transport.close()
        # Allow cleanup before terminating ZMQ context
        await asyncio.sleep(0.1)
        zmq_ctx.term()


@pytest.mark.asyncio
async def test_default_destination_address():
    """Test behavior when default destination address is set and used."""
    zmq_ctx = zmq.Context()

    try:
        # Create receiver first to get its address
        receiver = DynamicZmqTransport(
            zmq_ctx=zmq_ctx,
            bind_address=generate_zmq_inproc_endpoint(),
            instance_id="receiver",
        )
        await receiver.start()

        # Create sender with default destination address set to receiver
        sender = DynamicZmqTransport(
            zmq_ctx=zmq_ctx,
            bind_address=generate_zmq_inproc_endpoint(),
            instance_id="sender",
            default_destination_address=receiver.get_address(),
        )
        await sender.start()

        # Send message without destination_address (should use default)
        message_1 = create_test_message(
            "test-msg-1", {"test_case": "use_default"}
        )
        await sender.send_message(message_1)  # No destination_address provided

        # Send message with explicit None (should use default)
        message_2 = create_test_message(
            "test-msg-2", {"test_case": "explicit_none_use_default"}
        )
        await sender.send_message(message_2, destination_address=None)

        # Receive both messages
        received_messages = []
        for _ in range(20):
            received = await receiver.receive_message()
            if received:
                received_messages.append(received)
                if len(received_messages) == 2:
                    break
            await asyncio.sleep(0.1)

        # Verify both messages were received using the default destination
        assert len(received_messages) == 2, (
            f"Expected 2 messages, got {len(received_messages)}"
        )

        test_cases = {
            msg.message.payload["test_case"] for msg in received_messages
        }
        expected_cases = {"use_default", "explicit_none_use_default"}
        assert test_cases == expected_cases, (
            f"Expected {expected_cases}, got {test_cases}"
        )

    finally:
        # Allow pending operations to complete before shutdown
        await asyncio.sleep(0.1)
        await sender.close()
        await receiver.close()
        # Allow cleanup before terminating ZMQ context
        await asyncio.sleep(0.1)
        zmq_ctx.term()
