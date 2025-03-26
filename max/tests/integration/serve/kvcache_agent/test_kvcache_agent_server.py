# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import multiprocessing
import time

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


@pytest.fixture(scope="module")
def server_config():
    """Fixture that provides server configuration for tests."""
    return KVCacheAgentServerConfig(host="localhost", port=50052, num_workers=2)


@pytest.fixture
def queue():
    """Fixture that provides a queue for tests."""
    return multiprocessing.Queue()


@pytest.fixture
def server(server_config, queue):
    """Fixture that provides a running server instance for tests."""
    server = KVCacheAgentServer(server_config, queue)
    server.start()

    # Give the server time to start up
    time.sleep(0.1)

    yield server

    # Teardown
    server.stop(grace=1)


@pytest.fixture
def stub(server_config):
    """Fixture that provides a gRPC client stub connected to the test server."""
    channel = grpc.insecure_channel(
        f"{server_config.host}:{server_config.port}"
    )
    return KVCacheAgentServiceStub(channel)


def test_server_initialization(server_config, queue):
    """Test that the server initializes correctly."""
    server = KVCacheAgentServer(server_config, queue)
    assert not server._started

    server.start()
    assert server._started

    server.stop()
    assert not server._started


def test_smoke(server, queue, stub):
    """Smoke test."""
    queue.put(
        KVCacheChangeMessage(
            cache_id="id1",
            memory_tier=MemoryTier.MEMORY_TIER_GPU,
            update_type=UpdateType.UPDATE_TYPE_ADDED,
        )
    )
    queue.put(
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

    queue.put(
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


def test_multiple_subscribers(server, queue, stub):
    """Test that multiple subscribers receive updates."""

    responses1 = stub.SubscribeToUpdates(SubscriptionRequest())
    responses2 = stub.SubscribeToUpdates(SubscriptionRequest())

    queue.put(
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
