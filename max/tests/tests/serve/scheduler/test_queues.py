# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pickle
import queue
import time
import uuid

import numpy as np
import pytest
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
    create_zmq_push_pull_queues,
    generate_zmq_ipc_path,
)


def test_serialization_and_deserialization_through_queue_with_pickle() -> None:
    test_address = generate_zmq_ipc_path()
    push_socket = ZmqPushSocket[tuple[int, TextContext]](
        endpoint=test_address, serialize=pickle.dumps, lazy=False
    )
    pull_socket = ZmqPullSocket[tuple[int, TextContext]](
        endpoint=test_address, deserialize=pickle.loads, lazy=False
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
        endpoint=test_address, serialize=msgpack_numpy_encoder(), lazy=False
    )

    pull_socket = ZmqPullSocket[tuple[str, TextContext]](
        endpoint=test_address,
        deserialize=msgpack_numpy_decoder(tuple[str, TextContext]),
        lazy=False,
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


def test_zmq_push_pull_queue_basic_functionality() -> None:
    """Test basic put_nowait and get_nowait functionality."""
    push_queue, pull_queue = create_zmq_push_pull_queues(
        payload_type=int,
        lazy=False,
    )

    time.sleep(1)
    push_queue.put_nowait(42)
    # Give it some time to send appropriately.
    time.sleep(1)
    result = pull_queue.get_nowait()
    assert result == 42


def test_zmq_push_pull_queue_with_complex_data() -> None:
    """Test queue with complex data structures using pickle serialization."""

    context = TextContext(max_length=15, tokens=np.array([1, 1, 1, 1, 1]))
    test_data = ("test_id", context)

    push_queue, pull_queue = create_zmq_push_pull_queues(
        lazy=False, payload_type=tuple[str, TextContext]
    )

    push_queue.put_nowait(test_data)
    time.sleep(1)
    result = pull_queue.get_nowait()

    assert result[0] == test_data[0]
    assert msgpack_eq(result[1], test_data[1])


def test_zmq_push_pull_queue_with_custom_serialization() -> None:
    """Test queue with custom msgpack serialization."""
    context = TextContext(max_length=10, tokens=np.array([1, 2, 3, 4, 5]))
    test_data = (str(uuid.uuid4()), context)

    push_queue, pull_queue = create_zmq_push_pull_queues(
        lazy=False, payload_type=tuple[str, TextContext]
    )

    try:
        push_queue.put_nowait(test_data)
        time.sleep(1)
        result = pull_queue.get_nowait()

        assert result[0] == test_data[0]
        assert msgpack_eq(result[1], test_data[1])
    finally:
        push_queue.close()
        pull_queue.close()


def test_zmq_push_pull_queue_empty_queue_raises_exception() -> None:
    """Test that get_nowait raises queue.Empty when queue is empty."""
    push_queue, pull_queue = create_zmq_push_pull_queues(
        lazy=False, payload_type=str
    )

    with pytest.raises(queue.Empty):
        pull_queue.get_nowait()


def test_zmq_push_pull_queue_multiple_items() -> None:
    """Test queue with multiple items maintains order (FIFO)."""
    test_items = ["first", "second", "third", "fourth"]

    push_queue, pull_queue = create_zmq_push_pull_queues(
        lazy=False, payload_type=str
    )
    # Put all items
    for item in test_items:
        push_queue.put_nowait(item)
        time.sleep(1)

    # Get all items and verify order
    results = []
    for _ in test_items:
        results.append(pull_queue.get_nowait())

    assert results == test_items


def test_zmq_push_pull_queue_closed_state() -> None:
    """Test that operations fail when queue is closed."""
    push_queue, pull_queue = create_zmq_push_pull_queues(
        lazy=False, payload_type=str
    )
    push_queue.close()
    pull_queue.close()

    with pytest.raises(RuntimeError, match="Socket is closed"):
        push_queue.put_nowait("sample_str")

    with pytest.raises(RuntimeError, match="Socket is closed"):
        pull_queue.get_nowait()


def test_zmq_push_pull_queue_endpoint_validation() -> None:
    """Test that invalid endpoints raise ValueError."""
    with pytest.raises(ValueError, match="Invalid endpoint"):
        ZmqPushSocket(endpoint="invalid://endpoint")

    with pytest.raises(ValueError, match="Invalid endpoint"):
        ZmqPullSocket(endpoint="")


def test_zmq_push_pull_queue_with_vision_context() -> None:
    """Test queue with complex vision context data."""
    # Create vision context with image data
    shape = (2, 3, 224, 224)
    img = np.random.rand(*shape).astype(np.float32)

    context = TextAndVisionContext(
        request_id="test-vision-request",
        max_length=50,
        tokens=np.array([0, 1, 2, 3, 4]),
        pixel_values=(img,),
    )

    test_data = ("vision_test", context)

    push_queue, pull_queue = create_zmq_push_pull_queues(
        lazy=False, payload_type=tuple[str, TextAndVisionContext]
    )

    push_queue.put_nowait(test_data)
    time.sleep(1)
    result = pull_queue.get_nowait()

    assert result[0] == test_data[0]
    assert result[1].request_id == test_data[1].request_id
    assert np.array_equal(result[1].tokens, test_data[1].tokens)
    assert np.allclose(result[1].pixel_values[0], test_data[1].pixel_values[0])
