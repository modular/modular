# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import asyncio
import multiprocessing
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
from max.serve.kvcache_agent.dispatcher_transport import TransportMessage
from max.serve.process_control import ProcessControl, ProcessMonitor
from max.serve.queue.zmq_queue import generate_zmq_ipc_path


def instance_a_service_process_fn(
    pc: ProcessControl,
    factory: DispatcherFactory[dict[str, Union[str, int]]],
) -> None:
    """Process function for instance A dispatcher service."""
    try:
        pc.set_started()
        pc.beat()

        # Create ZMQ context for this process
        zmq_ctx = zmq.Context()

        # Create dispatcher service
        instance_a_dispatcher = factory.create_service(zmq_ctx)

        async def run_instance_a_service():
            await instance_a_dispatcher.start()

            # Keep running and processing messages
            while not pc.is_canceled():
                pc.beat()
                await asyncio.sleep(0.5)

            # Cleanup
            await instance_a_dispatcher.stop()
            zmq_ctx.term()

        # Run the async event loop
        asyncio.run(run_instance_a_service())

    except Exception as e:
        print(f"Instance A service process error: {e}")
        raise
    finally:
        pc.set_completed()


def instance_b_service_process_fn(
    pc: ProcessControl,
    factory: DispatcherFactory[dict[str, Union[str, int]]],
) -> None:
    """Process function for instance B dispatcher service."""
    try:
        pc.set_started()
        pc.beat()

        # Create ZMQ context for this process
        zmq_ctx = zmq.Context()

        # Create dispatcher service
        instance_b_dispatcher = factory.create_service(zmq_ctx)

        async def run_instance_b_service():
            await instance_b_dispatcher.start()

            # Keep running and processing messages
            while not pc.is_canceled():
                pc.beat()
                await asyncio.sleep(0.5)

            # Cleanup
            await instance_b_dispatcher.stop()
            zmq_ctx.term()

        # Run the async event loop
        asyncio.run(run_instance_b_service())

    except Exception as e:
        print(f"Instance B service process error: {e}")
        raise
    finally:
        pc.set_completed()


@pytest.mark.skip(reason="E2EOPT-296 Reenable once not flaky")
@pytest.mark.asyncio
async def test_single_request_reply():
    """Test single request-reply pattern with dispatcher services running in separate processes."""
    mp_context = multiprocessing.get_context("spawn")

    # Create IPC endpoints for dynamic ZMQ communication between dispatchers
    instance_a_address = generate_zmq_ipc_path()
    instance_b_address = generate_zmq_ipc_path()

    # Create dispatcher configs with dynamic ZMQ transport
    instance_a_config = DispatcherConfig(
        transport=TransportType.DYNAMIC_ZMQ,
        transport_config=TransportFactory.DynamicZmqTransportConfig(
            bind_address=instance_a_address,
            default_destination_address=instance_b_address,
        ),
    )

    instance_b_config = DispatcherConfig(
        transport=TransportType.DYNAMIC_ZMQ,
        transport_config=TransportFactory.DynamicZmqTransportConfig(
            bind_address=instance_b_address,
            default_destination_address=instance_a_address,
        ),
    )

    # Create factories
    instance_a_factory = DispatcherFactory[dict[str, Union[str, int]]](
        instance_a_config,
        transport_payload_type=TransportMessage[dict[str, Union[str, int]]],
    )
    instance_b_factory = DispatcherFactory[dict[str, Union[str, int]]](
        instance_b_config,
        transport_payload_type=TransportMessage[dict[str, Union[str, int]]],
    )

    # Create process controls
    pc_a = ProcessControl(mp_context, "instance_a")
    pc_b = ProcessControl(mp_context, "instance_b")

    # Create processes for dispatcher services
    process_a = mp_context.Process(
        target=instance_a_service_process_fn,
        args=(pc_a, instance_a_factory),
        name="instance_a_single_request_reply_service_process",
    )

    process_b = mp_context.Process(
        target=instance_b_service_process_fn,
        args=(pc_b, instance_b_factory),
        name="instance_b_single_request_reply_service_process",
    )

    # Create process monitors
    monitor_a = ProcessMonitor(pc_a, process_a, poll_s=0.01, max_time_s=10.0)
    monitor_b = ProcessMonitor(pc_b, process_b, poll_s=0.01, max_time_s=10.0)

    # Create ZMQ context for main process (clients)
    zmq_ctx = zmq.Context()

    try:
        # Start both service processes
        process_a.start()
        process_b.start()

        # Wait for both processes to start
        started_a = await monitor_a.until_started()
        started_b = await monitor_b.until_started()

        assert started_a, "Instance A service process failed to start"
        assert started_b, "Instance B service process failed to start"

        # Wait for both processes to become healthy
        healthy_a = await monitor_a.until_healthy()
        healthy_b = await monitor_b.until_healthy()

        assert healthy_a, "Instance A service process is not healthy"
        assert healthy_b, "Instance B service process is not healthy"

        # Create clients in main process
        instance_a_client = instance_a_factory.create_client(zmq_ctx)
        instance_b_client = instance_b_factory.create_client(zmq_ctx)

        # Set up message handling
        received_replies: list[Any] = []
        received_requests: list[Any] = []

        # Set up reply handler for instance A client
        @instance_a_client.reply_handler(MessageType.PREFILL_RESPONSE)
        def handle_reply(payload: Any) -> None:
            received_replies.append(payload)

        # Set up request handler for instance B client
        @instance_b_client.request_handler(MessageType.PREFILL_REQUEST)
        def handle_request(payload: Any, reply_context: ReplyContext) -> None:
            received_requests.append(payload)
            # Send reply back
            reply_payload = {
                "result": payload["value"] * 2,
                "status": "success",
            }
            instance_b_client.send_reply(
                MessageType.PREFILL_RESPONSE, reply_payload, reply_context
            )

        # Start clients
        instance_a_client.start()
        instance_b_client.start()

        # Allow time for clients to fully initialize
        await asyncio.sleep(0.5)

        # Send request from instance A client to instance B
        request_payload = {"value": 21, "operation": "double"}
        instance_a_client.send(
            MessageType.PREFILL_REQUEST,
            request_payload,
            destination_address=instance_b_address,
        )

        # Give some time for the message exchange to complete
        await asyncio.sleep(0.5)

        # Verify request was received by instance B
        try:
            received_request = received_requests[0]
            assert received_request["value"] == 21
            assert received_request["operation"] == "double"
        except Exception as e:
            pytest.fail(f"No request received by instance B: {e}")

        # Verify reply was received by instance A
        try:
            received_reply = received_replies[0]
            assert received_reply["result"] == 42  # 21 * 2
            assert received_reply["status"] == "success"
        except Exception as e:
            pytest.fail(f"No reply received by instance A: {e}")

        # Signal service processes to stop
        pc_a.set_canceled()
        pc_b.set_canceled()

        # Wait for processes to complete
        completed_a = await monitor_a.until_completed()
        completed_b = await monitor_b.until_completed()

        assert completed_a, (
            "Instance A service process did not complete gracefully"
        )
        assert completed_b, (
            "Instance B service process did not complete gracefully"
        )

    finally:
        # Allow pending operations to complete before shutdown
        await asyncio.sleep(0.5)
        # Clean up and close ZMQ context
        await monitor_a.shutdown()
        await monitor_b.shutdown()
        instance_a_client.stop()
        instance_b_client.stop()
        # Allow cleanup before terminating ZMQ context
        await asyncio.sleep(0.5)
        zmq_ctx.term()


@pytest.mark.skip(reason="E2EOPT-296 Reenable once not flaky")
@pytest.mark.asyncio
async def test_multiple_request_reply():
    """Test multiple request-reply exchanges between cross-process dispatcher services."""
    mp_context = multiprocessing.get_context("spawn")

    # Create IPC endpoints for dynamic ZMQ communication between dispatchers
    instance_a_address = generate_zmq_ipc_path()
    instance_b_address = generate_zmq_ipc_path()

    # Create dispatcher configs with dynamic ZMQ transport
    instance_a_config = DispatcherConfig(
        transport=TransportType.DYNAMIC_ZMQ,
        transport_config=TransportFactory.DynamicZmqTransportConfig(
            bind_address=instance_a_address,
            default_destination_address=instance_b_address,
        ),
    )

    instance_b_config = DispatcherConfig(
        transport=TransportType.DYNAMIC_ZMQ,
        transport_config=TransportFactory.DynamicZmqTransportConfig(
            bind_address=instance_b_address,
            default_destination_address=instance_a_address,
        ),
    )

    # Create factories
    instance_a_factory = DispatcherFactory[dict[str, Union[str, int]]](
        instance_a_config,
        transport_payload_type=TransportMessage[dict[str, Union[str, int]]],
    )
    instance_b_factory = DispatcherFactory[dict[str, Union[str, int]]](
        instance_b_config,
        transport_payload_type=TransportMessage[dict[str, Union[str, int]]],
    )

    # Create process controls
    pc_a = ProcessControl(mp_context, "instance_a")
    pc_b = ProcessControl(mp_context, "instance_b")

    # Create processes for dispatcher services
    process_a = mp_context.Process(
        target=instance_a_service_process_fn,
        args=(pc_a, instance_a_factory),
        name="instance_a_multi_request_reply_service_process",
    )

    process_b = mp_context.Process(
        target=instance_b_service_process_fn,
        args=(pc_b, instance_b_factory),
        name="instance_b_multi_request_reply_service_process",
    )

    # Create process monitors
    monitor_a = ProcessMonitor(pc_a, process_a, poll_s=0.01, max_time_s=10.0)
    monitor_b = ProcessMonitor(pc_b, process_b, poll_s=0.01, max_time_s=10.0)

    # Create ZMQ context for main process (clients)
    zmq_ctx = zmq.Context()

    try:
        # Start both service processes
        process_a.start()
        process_b.start()

        # Wait for both processes to start
        started_a = await monitor_a.until_started()
        started_b = await monitor_b.until_started()

        assert started_a, "Instance A service process failed to start"
        assert started_b, "Instance B service process failed to start"

        # Wait for both processes to become healthy
        healthy_a = await monitor_a.until_healthy()
        healthy_b = await monitor_b.until_healthy()

        assert healthy_a, "Instance A service process is not healthy"
        assert healthy_b, "Instance B service process is not healthy"

        # Create clients in main process
        instance_a_client = instance_a_factory.create_client(zmq_ctx)
        instance_b_client = instance_b_factory.create_client(zmq_ctx)

        # Set up message handling
        received_replies: list[Any] = []
        received_requests: list[Any] = []

        # Set up reply handler for instance A client
        @instance_a_client.reply_handler(MessageType.PREFILL_RESPONSE)
        def handle_reply(payload: Any) -> None:
            received_replies.append(payload)

        # Set up request handler for instance B client
        @instance_b_client.request_handler(MessageType.PREFILL_REQUEST)
        def handle_request(payload: Any, reply_context: ReplyContext) -> None:
            received_requests.append(payload)
            # Send reply back with triple the value
            reply_payload = {
                "result": payload["value"] * 3,
                "status": "success",
                "id": payload.get("id", -1),
            }
            instance_b_client.send_reply(
                MessageType.PREFILL_RESPONSE, reply_payload, reply_context
            )

        # Start clients
        instance_a_client.start()
        instance_b_client.start()

        # Allow time for clients to fully initialize
        await asyncio.sleep(0.5)

        # Send multiple requests from instance A client to instance B
        for i in range(3):
            request_payload = {"value": 10 + i, "operation": "triple", "id": i}
            instance_a_client.send(
                MessageType.PREFILL_REQUEST,
                request_payload,
                destination_address=instance_b_address,
            )

        # Give time for all message exchanges to complete
        await asyncio.sleep(0.5)

        # Verify we received 3 requests and 3 replies
        assert len(received_requests) == 3, (
            f"Expected 3 requests, got {len(received_requests)}"
        )
        assert len(received_replies) == 3, (
            f"Expected 3 replies, got {len(received_replies)}"
        )

        # Verify request contents
        for i, request in enumerate(received_requests):
            assert request["value"] == 10 + i, f"Request {i} has wrong value"
            assert request["operation"] == "triple", (
                f"Request {i} has wrong operation"
            )
            assert request["id"] == i, f"Request {i} has wrong id"

        # Verify reply contents
        for reply in received_replies:
            expected_result = (10 + reply["id"]) * 3
            assert reply["result"] == expected_result, (
                f"Reply {reply['id']} has wrong result"
            )
            assert reply["status"] == "success", (
                f"Reply {reply['id']} has wrong status"
            )

        # Signal service processes to stop
        pc_a.set_canceled()
        pc_b.set_canceled()

        # Wait for processes to complete
        completed_a = await monitor_a.until_completed()
        completed_b = await monitor_b.until_completed()

        assert completed_a, (
            "Instance A service process did not complete gracefully"
        )
        assert completed_b, (
            "Instance B service process did not complete gracefully"
        )

    finally:
        # Allow pending operations to complete before shutdown
        await asyncio.sleep(0.5)
        # Clean up and close ZMQ context
        await monitor_a.shutdown()
        await monitor_b.shutdown()
        instance_a_client.stop()
        instance_b_client.stop()
        # Allow cleanup before terminating ZMQ context
        await asyncio.sleep(0.5)
        zmq_ctx.term()


@pytest.mark.skip(reason="E2EOPT-296 Reenable once not flaky")
@pytest.mark.asyncio
async def test_bidirectional_communication():
    """Test that both processes can send requests to each other simultaneously."""
    mp_context = multiprocessing.get_context("spawn")

    # Create IPC endpoints for dynamic ZMQ communication between dispatchers
    instance_a_address = generate_zmq_ipc_path()
    instance_b_address = generate_zmq_ipc_path()

    # Create dispatcher configs with dynamic ZMQ transport
    instance_a_config = DispatcherConfig(
        transport=TransportType.DYNAMIC_ZMQ,
        transport_config=TransportFactory.DynamicZmqTransportConfig(
            bind_address=instance_a_address,
            default_destination_address=instance_b_address,
        ),
    )

    instance_b_config = DispatcherConfig(
        transport=TransportType.DYNAMIC_ZMQ,
        transport_config=TransportFactory.DynamicZmqTransportConfig(
            bind_address=instance_b_address,
            default_destination_address=instance_a_address,
        ),
    )

    # Create factories
    instance_a_factory = DispatcherFactory[dict[str, Union[str, int]]](
        instance_a_config,
        transport_payload_type=TransportMessage[dict[str, Union[str, int]]],
    )
    instance_b_factory = DispatcherFactory[dict[str, Union[str, int]]](
        instance_b_config,
        transport_payload_type=TransportMessage[dict[str, Union[str, int]]],
    )

    # Create process controls
    pc_a = ProcessControl(mp_context, "instance_a")
    pc_b = ProcessControl(mp_context, "instance_b")

    # Create processes for dispatcher services
    process_a = mp_context.Process(
        target=instance_a_service_process_fn,
        args=(pc_a, instance_a_factory),
        name="instance_a_bidirectional_service_process",
    )

    process_b = mp_context.Process(
        target=instance_b_service_process_fn,
        args=(pc_b, instance_b_factory),
        name="instance_b_bidirectional_service_process",
    )

    # Create process monitors
    monitor_a = ProcessMonitor(pc_a, process_a, poll_s=0.01, max_time_s=10.0)
    monitor_b = ProcessMonitor(pc_b, process_b, poll_s=0.01, max_time_s=10.0)

    # Create ZMQ context for main process (clients)
    zmq_ctx = zmq.Context()

    try:
        # Start both service processes
        process_a.start()
        process_b.start()

        # Wait for both processes to start and become healthy
        started_a = await monitor_a.until_started()
        started_b = await monitor_b.until_started()
        assert started_a and started_b, "Service processes failed to start"

        healthy_a = await monitor_a.until_healthy()
        healthy_b = await monitor_b.until_healthy()
        assert healthy_a and healthy_b, "Service processes are not healthy"

        # Create clients in main process
        instance_a_client = instance_a_factory.create_client(zmq_ctx)
        instance_b_client = instance_b_factory.create_client(zmq_ctx)

        # Set up message handling for both clients
        a_received_replies: list[Any] = []
        a_received_requests: list[Any] = []
        b_received_replies: list[Any] = []
        b_received_requests: list[Any] = []

        # Set up handlers for instance A
        @instance_a_client.reply_handler(MessageType.PREFILL_RESPONSE)
        def handle_a_reply(payload: Any) -> None:
            a_received_replies.append(payload)

        @instance_a_client.request_handler(MessageType.PREFILL_REQUEST)
        def handle_a_request(payload: Any, reply_context: ReplyContext) -> None:
            a_received_requests.append(payload)
            reply_payload = {
                "result": f"A processed: {payload['value']}",
                "status": "success",
                "source": "A",
            }
            instance_a_client.send_reply(
                MessageType.PREFILL_RESPONSE, reply_payload, reply_context
            )

        # Set up handlers for instance B
        @instance_b_client.reply_handler(MessageType.PREFILL_RESPONSE)
        def handle_b_reply(payload: Any) -> None:
            b_received_replies.append(payload)

        @instance_b_client.request_handler(MessageType.PREFILL_REQUEST)
        def handle_b_request(payload: Any, reply_context: ReplyContext) -> None:
            b_received_requests.append(payload)
            reply_payload = {
                "result": f"B processed: {payload['value']}",
                "status": "success",
                "source": "B",
            }
            instance_b_client.send_reply(
                MessageType.PREFILL_RESPONSE, reply_payload, reply_context
            )

        # Start clients
        instance_a_client.start()
        instance_b_client.start()

        # Allow time for clients to fully initialize
        await asyncio.sleep(0.5)

        # Send requests from A to B
        for i in range(2):
            request_payload = {"value": f"A-to-B-{i}", "sender": "A"}
            instance_a_client.send(
                MessageType.PREFILL_REQUEST,
                request_payload,
                destination_address=instance_b_address,
            )

        # Send requests from B to A
        for i in range(2):
            request_payload = {"value": f"B-to-A-{i}", "sender": "B"}
            instance_b_client.send(
                MessageType.PREFILL_REQUEST,
                request_payload,
                destination_address=instance_a_address,
            )

        # Give time for all message exchanges to complete
        await asyncio.sleep(0.5)

        # Verify A received requests from B and sent replies
        assert len(a_received_requests) == 2, (
            f"A should have received 2 requests, got {len(a_received_requests)}"
        )
        assert len(a_received_replies) == 2, (
            f"A should have received 2 replies, got {len(a_received_replies)}"
        )

        # Verify B received requests from A and sent replies
        assert len(b_received_requests) == 2, (
            f"B should have received 2 requests, got {len(b_received_requests)}"
        )
        assert len(b_received_replies) == 2, (
            f"B should have received 2 replies, got {len(b_received_replies)}"
        )

        # Verify content of replies
        for reply in a_received_replies:
            assert reply["source"] == "B", "A should receive replies from B"
            assert "B processed:" in reply["result"], (
                "Reply should show B processed the request"
            )

        for reply in b_received_replies:
            assert reply["source"] == "A", "B should receive replies from A"
            assert "A processed:" in reply["result"], (
                "Reply should show A processed the request"
            )

        # Signal service processes to stop
        pc_a.set_canceled()
        pc_b.set_canceled()

        # Wait for processes to complete
        completed_a = await monitor_a.until_completed()
        completed_b = await monitor_b.until_completed()
        assert completed_a and completed_b, (
            "Service processes did not complete gracefully"
        )

    finally:
        # Allow pending operations to complete before shutdown
        await asyncio.sleep(0.5)
        # Clean up and close ZMQ context
        await monitor_a.shutdown()
        await monitor_b.shutdown()
        instance_a_client.stop()
        instance_b_client.stop()
        # Allow cleanup before terminating ZMQ context
        await asyncio.sleep(0.5)
        zmq_ctx.term()


@pytest.mark.skip(reason="E2EOPT-296 Reenable once not flaky")
@pytest.mark.asyncio
async def test_high_throughput_cross_process():
    """Test high throughput message processing across processes."""
    mp_context = multiprocessing.get_context("spawn")

    # Create IPC endpoints for dynamic ZMQ communication between dispatchers
    instance_a_address = generate_zmq_ipc_path()
    instance_b_address = generate_zmq_ipc_path()

    # Create dispatcher configs with dynamic ZMQ transport
    instance_a_config = DispatcherConfig(
        transport=TransportType.DYNAMIC_ZMQ,
        transport_config=TransportFactory.DynamicZmqTransportConfig(
            bind_address=instance_a_address,
            default_destination_address=instance_b_address,
        ),
    )

    instance_b_config = DispatcherConfig(
        transport=TransportType.DYNAMIC_ZMQ,
        transport_config=TransportFactory.DynamicZmqTransportConfig(
            bind_address=instance_b_address,
            default_destination_address=instance_a_address,
        ),
    )

    # Create factories
    instance_a_factory = DispatcherFactory[dict[str, Union[str, int]]](
        instance_a_config,
        transport_payload_type=TransportMessage[dict[str, Union[str, int]]],
    )
    instance_b_factory = DispatcherFactory[dict[str, Union[str, int]]](
        instance_b_config,
        transport_payload_type=TransportMessage[dict[str, Union[str, int]]],
    )

    # Create process controls
    pc_a = ProcessControl(mp_context, "instance_a")
    pc_b = ProcessControl(mp_context, "instance_b")

    # Create processes for dispatcher services
    process_a = mp_context.Process(
        target=instance_a_service_process_fn,
        args=(pc_a, instance_a_factory),
        name="instance_a_high_throughput_service_process",
    )

    process_b = mp_context.Process(
        target=instance_b_service_process_fn,
        args=(pc_b, instance_b_factory),
        name="instance_b_high_throughput_service_process",
    )

    # Create process monitors
    monitor_a = ProcessMonitor(pc_a, process_a, poll_s=0.01, max_time_s=30.0)
    monitor_b = ProcessMonitor(pc_b, process_b, poll_s=0.01, max_time_s=30.0)

    # Create ZMQ context for main process (clients)
    zmq_ctx = zmq.Context()

    try:
        # Start both service processes
        process_a.start()
        process_b.start()

        # Wait for both processes to start and become healthy
        started_a = await monitor_a.until_started()
        started_b = await monitor_b.until_started()
        assert started_a and started_b, "Service processes failed to start"

        healthy_a = await monitor_a.until_healthy()
        healthy_b = await monitor_b.until_healthy()
        assert healthy_a and healthy_b, "Service processes are not healthy"

        # Create clients in main process
        instance_a_client = instance_a_factory.create_client(zmq_ctx)
        instance_b_client = instance_b_factory.create_client(zmq_ctx)

        # Track performance metrics
        requests_processed = 0
        replies_received = 0
        start_time = None

        # Set up reply handler for instance A client
        @instance_a_client.reply_handler(MessageType.PREFILL_RESPONSE)
        def handle_reply(payload: Any) -> None:
            nonlocal replies_received
            replies_received += 1

        # Set up request handler for instance B client
        @instance_b_client.request_handler(MessageType.PREFILL_REQUEST)
        def handle_request(payload: Any, reply_context: ReplyContext) -> None:
            nonlocal requests_processed
            requests_processed += 1

            # Send reply back with processed data
            reply_payload = {
                "result": payload["value"] * 2,
                "id": payload["id"],
                "processed_by": "instance_b",
            }
            instance_b_client.send_reply(
                MessageType.PREFILL_RESPONSE, reply_payload, reply_context
            )

        # Start clients
        instance_a_client.start()
        instance_b_client.start()

        # Allow time for clients to fully initialize
        await asyncio.sleep(0.5)

        # Send high volume of messages
        num_messages = 1000
        start_time = time.time()

        print(f"Starting high throughput test with {num_messages} messages...")

        # Send messages in batches to avoid overwhelming the system
        batch_size = 50
        for batch_start in range(0, num_messages, batch_size):
            batch_end = min(batch_start + batch_size, num_messages)

            # Send batch of messages
            for i in range(batch_start, batch_end):
                request_payload = {
                    "id": i,
                    "value": i,
                    "batch": batch_start // batch_size,
                    "timestamp": time.time(),
                }
                instance_a_client.send(
                    MessageType.PREFILL_REQUEST,
                    request_payload,
                    destination_address=instance_b_address,
                )

        end_time = time.time()
        duration = end_time - start_time

        # Allow time for messages to be processed
        await asyncio.sleep(0.5)

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
            f"High throughput cross-process test: {num_messages} messages in {duration:.2f}s = {throughput:.1f} msg/s"
        )

        # Expect high throughput for cross-process communication
        assert throughput > 500, (
            f"Cross-process throughput too low: {throughput:.1f} msg/s"
        )

        # Verify both processes are still healthy after high load
        healthy_a_after = await monitor_a.until_healthy()
        healthy_b_after = await monitor_b.until_healthy()
        assert healthy_a_after and healthy_b_after, (
            "Processes should remain healthy after high throughput test"
        )

        # Signal service processes to stop
        pc_a.set_canceled()
        pc_b.set_canceled()

        # Wait for processes to complete
        completed_a = await monitor_a.until_completed()
        completed_b = await monitor_b.until_completed()
        assert completed_a and completed_b, (
            "Service processes did not complete gracefully"
        )

    finally:
        # Allow pending operations to complete before shutdown
        await asyncio.sleep(0.5)
        # Clean up and close ZMQ context
        await monitor_a.shutdown()
        await monitor_b.shutdown()
        instance_a_client.stop()
        instance_b_client.stop()
        # Allow cleanup before terminating ZMQ context
        await asyncio.sleep(0.5)
        zmq_ctx.term()
