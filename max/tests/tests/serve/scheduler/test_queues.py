# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pickle
import uuid

import numpy as np
import pytest
import zmq
from max.pipelines.core import (
    TextContext,
    msgpack_numpy_decoder,
    msgpack_numpy_encoder,
)
from max.serve.queue.zmq_queue import (
    ZmqPullSocket,
    ZmqPushSocket,
    generate_zmq_ipc_path,
)


@pytest.fixture(scope="session")
def zmq_ctx():
    return zmq.Context(io_threads=2)


def test_serialization_and_deserialization_through_queue_with_pickle(
    zmq_ctx,
) -> None:
    test_address = generate_zmq_ipc_path()
    push_socket = ZmqPushSocket[tuple[int, TextContext]](
        zmq_ctx, test_address, serialize=pickle.dumps
    )
    pull_socket = ZmqPullSocket[tuple[int, TextContext]](
        zmq_ctx, test_address, deserialize=pickle.loads
    )

    context = (
        1,
        TextContext(prompt="hello!", max_length=15, tokens=np.ones(5)),
    )

    push_socket.put(context)
    received_context = pull_socket.get()

    assert context == received_context


def test_serialization_and_deserialization_through_queue_with_msgpack(
    zmq_ctx,
) -> None:
    test_address = generate_zmq_ipc_path()
    push_socket = ZmqPushSocket[tuple[str, TextContext]](
        zmq_ctx, test_address, serialize=msgpack_numpy_encoder()
    )

    pull_socket = ZmqPullSocket[tuple[str, TextContext]](
        zmq_ctx,
        test_address,
        deserialize=msgpack_numpy_decoder(tuple[str, TextContext]),
    )

    context = (
        str(uuid.uuid4()),
        TextContext(prompt="hello!", max_length=15, tokens=np.ones(5)),
    )

    push_socket.put(context)
    received_context = pull_socket.get()

    assert context == received_context
