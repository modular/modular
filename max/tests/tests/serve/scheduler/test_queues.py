# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pickle
import time
import uuid

import numpy as np
from max.interfaces import (
    SharedMemoryArray,
    msgpack_eq,
    msgpack_numpy_decoder,
    msgpack_numpy_encoder,
)
from max.pipelines.core.context import TextAndVisionContext, TextContext
from max.serve.queue.zmq_queue import (
    ZmqPullSocket,
    ZmqPushSocket,
    generate_zmq_ipc_path,
)


def test_serialization_and_deserialization_through_queue_with_pickle() -> None:
    test_address = generate_zmq_ipc_path()
    push_socket = ZmqPushSocket[tuple[int, TextContext]](
        zmq_endpoint=test_address, serialize=pickle.dumps
    )
    pull_socket = ZmqPullSocket[tuple[int, TextContext]](
        zmq_endpoint=test_address, deserialize=pickle.loads
    )

    context = (1, TextContext(max_length=15, tokens=np.ones(5, dtype=np.int32)))

    push_socket.put_nowait(context)
    time.sleep(1)
    received_context = pull_socket.get_nowait()

    assert context[0] == received_context[0]
    assert msgpack_eq(context[1], received_context[1])


def test_serialization_and_deserialization_through_queue_with_msgpack() -> None:
    test_address = generate_zmq_ipc_path()
    push_socket = ZmqPushSocket[tuple[str, TextContext]](
        zmq_endpoint=test_address, serialize=msgpack_numpy_encoder()
    )

    pull_socket = ZmqPullSocket[tuple[str, TextContext]](
        zmq_endpoint=test_address,
        deserialize=msgpack_numpy_decoder(tuple[str, TextContext]),
    )

    context = (
        str(uuid.uuid4()),
        TextContext(max_length=15, tokens=np.ones(5, dtype=np.int32)),
    )

    push_socket.put_nowait(context)
    time.sleep(1)
    received_context = pull_socket.get_nowait()

    assert context[0] == received_context[0]
    assert msgpack_eq(context[1], received_context[1])


def test_vision_context_shared_memory_fallback(mocker) -> None:  # noqa: ANN001
    """Test that vision context serialization falls back gracefully when shared memory is exhausted."""

    # Create realistic vision context with InternVL-sized image
    shape = (10, 32, 32, 3, 14, 14)
    img = np.random.rand(*shape).astype(np.float32)

    context = TextAndVisionContext(
        request_id="test-request",
        max_length=50,
        tokens=np.array([0, 1, 2, 3, 4]),
        pixel_values=(img,),  # Only one image supported
    )

    # Test the encoder directly
    encoder = msgpack_numpy_encoder(use_shared_memory=True)

    # Test 1: Fallback case - when shared memory allocation fails
    mocker.patch(
        "max.interfaces.utils.shared_memory.ndarray_to_shared_memory",
        return_value=None,
    )

    # Encode with fallback
    encoded_data = encoder(("test_req_id", context))
    # Decode to verify
    decoded = msgpack_numpy_decoder(tuple[str, TextAndVisionContext])(
        encoded_data
    )
    req_id, decoded_context = decoded

    assert req_id == "test_req_id"
    # In fallback case, images should be numpy arrays after round-trip
    assert isinstance(decoded_context.pixel_values[0], np.ndarray)
    assert np.allclose(decoded_context.pixel_values[0], img)

    # Verify original context wasn't modified
    assert isinstance(context.pixel_values[0], np.ndarray)

    # Test 2: Success case - when shared memory allocation succeeds
    mock_shm = SharedMemoryArray(
        name="test_shm_123", shape=shape, dtype="float32"
    )
    mocker.patch(
        "max.interfaces.utils.shared_memory.ndarray_to_shared_memory",
        return_value=mock_shm,
    )

    # Create a new context for second test
    context2 = TextAndVisionContext(
        request_id="test-request-2",
        max_length=50,
        tokens=np.array([0, 1, 2, 3, 4]),
        pixel_values=(img,),
    )

    # Encode with shared memory
    encoded_data2 = encoder(("test_req_id_2", context2))

    # Verify original context wasn't modified
    assert isinstance(context2.pixel_values[0], np.ndarray)

    # The encoded data should contain shared memory references
    # We can verify this by checking the encoded bytes contain the __shm__ marker
    assert b"__shm__" in encoded_data2
