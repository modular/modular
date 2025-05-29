# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import asyncio

import pytest
import zmq
from max.serve.kvcache_agent.dispatcher_base import MessageType
from max.serve.kvcache_agent.dispatcher_transport import (
    StaticZmqTransport,
    TransportMessage,
)
from max.serve.queue.zmq_queue import generate_zmq_inproc_endpoint


@pytest.mark.asyncio
async def test_static_zmq_transport_send_and_reply():
    zmq_ctx = zmq.Context()
    send_socket_path = generate_zmq_inproc_endpoint()
    recv_socket_path = generate_zmq_inproc_endpoint()

    # Create two transports, each connected to the other
    transport_a = StaticZmqTransport(
        zmq_ctx,
        send_endpoint=send_socket_path,
        recv_endpoint=recv_socket_path,
    )
    transport_b = StaticZmqTransport(
        zmq_ctx,
        send_endpoint=recv_socket_path,
        recv_endpoint=send_socket_path,
    )

    await asyncio.gather(transport_a.start(), transport_b.start())

    # Prepare a message from A to B
    payload = {"value": 123}
    message = TransportMessage(
        message_id="msg-1",
        message_type=MessageType.PREFILL_REQUEST.value,
        payload=payload,
    )

    # Send message from A to B
    await transport_a.send_message(message)

    # Receive message on B
    for _ in range(20):
        received_msg = await transport_b.receive_message()
        if received_msg:
            break
        await asyncio.sleep(0.1)

    assert received_msg is not None, "Transport B did not receive the message"
    assert received_msg.message.payload == payload
    assert (
        received_msg.message.message_type == MessageType.PREFILL_REQUEST.value
    )

    # Prepare a reply from B to A
    reply_payload = {"value": 456}
    reply_message = TransportMessage(
        message_id="msg-1-reply",
        message_type=MessageType.PREFILL_RESPONSE.value,
        payload=reply_payload,
    )
    await transport_b.send_reply(reply_message)

    # Receive reply on A
    for _ in range(20):
        received_msg = await transport_a.receive_message()
        if received_msg:
            break
        await asyncio.sleep(0.1)
    assert received_msg is not None, "Transport A did not receive the reply"
    assert received_msg.message.payload == reply_payload
    assert (
        received_msg.message.message_type == MessageType.PREFILL_RESPONSE.value
    )

    # Cleanup
    await asyncio.gather(transport_a.close(), transport_b.close())
    zmq_ctx.term()
