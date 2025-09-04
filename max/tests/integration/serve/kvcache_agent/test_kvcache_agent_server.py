# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pickle
import time
from collections.abc import Generator

import grpc
import pytest
from max.serve.kvcache_agent.kvcache_agent import (
    KVCacheAgentServer,
    KVCacheAgentServerConfig,
    KVCacheChangeMessage,
)
from max.serve.kvcache_agent.kvcache_agent_service_v1_pb2 import (  # type: ignore
    MemoryTier,
    SubscriptionRequest,
    UpdateType,
)
from max.serve.kvcache_agent.kvcache_agent_service_v1_pb2_grpc import (
    KVCacheAgentServiceStub,
)
from max.serve.queue.zmq_queue import (
    ZmqPushSocket,
    generate_zmq_inproc_endpoint,
)


@pytest.fixture
def zmq_endpoint() -> str:
    return generate_zmq_inproc_endpoint()


@pytest.fixture
def zmq_push_socket(
    zmq_endpoint: str,
) -> Generator[ZmqPushSocket[KVCacheChangeMessage], None, None]:
    push_socket = ZmqPushSocket[KVCacheChangeMessage](
        endpoint=zmq_endpoint,
        serialize=pickle.dumps,
        lazy=False,
    )
    yield push_socket
    push_socket._cleanup()


@pytest.fixture(scope="module")
def server_config() -> KVCacheAgentServerConfig:
    """Fixture that provides server configuration for tests."""
    return KVCacheAgentServerConfig(host="localhost", port=50052, num_workers=2)


@pytest.fixture
def server(
    server_config: KVCacheAgentServerConfig, zmq_endpoint: str
) -> Generator[KVCacheAgentServer, None, None]:
    """Fixture that provides a running server instance for tests using ZMQ."""
    server = KVCacheAgentServer(server_config, zmq_endpoint)
    server.start()
    time.sleep(0.1)
    yield server
    server.stop(grace=1)


@pytest.fixture
def stub(server_config: KVCacheAgentServerConfig) -> KVCacheAgentServiceStub:
    """Fixture that provides a gRPC client stub connected to the test server."""
    channel = grpc.insecure_channel(
        f"{server_config.host}:{server_config.port}"
    )
    return KVCacheAgentServiceStub(channel)


@pytest.mark.skip("GENAI-233")
def test_server_initialization(
    server_config: KVCacheAgentServerConfig, zmq_endpoint: str
) -> None:
    """Test that the server initializes correctly with ZMQ."""
    server = KVCacheAgentServer(server_config, zmq_endpoint)
    assert not server._started

    server.start()
    assert server._started

    server.stop()
    assert not server._started


@pytest.mark.skip("GENAI-233")
def test_smoke(
    server: KVCacheAgentServer,
    zmq_push_socket: ZmqPushSocket[KVCacheChangeMessage],
    stub: KVCacheAgentServiceStub,
) -> None:
    """Smoke test using ZMQ for event delivery."""
    zmq_push_socket.put_nowait(
        KVCacheChangeMessage(
            cache_id="id1",
            memory_tier=MemoryTier.MEMORY_TIER_GPU,
            update_type=UpdateType.UPDATE_TYPE_ADDED,
        )
    )
    zmq_push_socket.put_nowait(
        KVCacheChangeMessage(
            cache_id="id2",
            memory_tier=MemoryTier.MEMORY_TIER_CPU,
            update_type=UpdateType.UPDATE_TYPE_ADDED,
        )
    )

    responses = stub.SubscribeToUpdates(SubscriptionRequest())

    response = next(responses)
    assert response.update_type == UpdateType.UPDATE_TYPE_ADDED
    assert response.memory_tier == MemoryTier.MEMORY_TIER_GPU
    assert response.cache_ids == ["id1"]

    response = next(responses)
    assert response.update_type == UpdateType.UPDATE_TYPE_ADDED
    assert response.memory_tier == MemoryTier.MEMORY_TIER_CPU
    assert response.cache_ids == ["id2"]

    zmq_push_socket.put_nowait(
        KVCacheChangeMessage(
            cache_id="id1",
            memory_tier=MemoryTier.MEMORY_TIER_GPU,
            update_type=UpdateType.UPDATE_TYPE_REMOVED,
        )
    )

    response = next(responses)
    assert response.update_type == UpdateType.UPDATE_TYPE_REMOVED
    assert response.memory_tier == MemoryTier.MEMORY_TIER_GPU
    assert response.cache_ids == ["id1"]


@pytest.mark.skip("GENAI-233")
def test_multiple_subscribers(
    server: KVCacheAgentServer,
    zmq_push_socket: ZmqPushSocket[KVCacheChangeMessage],
    stub: KVCacheAgentServiceStub,
) -> None:
    """Test that multiple subscribers receive updates using ZMQ."""

    responses1 = stub.SubscribeToUpdates(SubscriptionRequest())
    responses2 = stub.SubscribeToUpdates(SubscriptionRequest())

    zmq_push_socket.put_nowait(
        KVCacheChangeMessage(
            cache_id="id1",
            memory_tier=MemoryTier.MEMORY_TIER_GPU,
            update_type=UpdateType.UPDATE_TYPE_ADDED,
        )
    )

    response1 = next(responses1)
    assert response1.update_type == UpdateType.UPDATE_TYPE_ADDED

    response2 = next(responses2)
    assert response2.update_type == UpdateType.UPDATE_TYPE_ADDED
