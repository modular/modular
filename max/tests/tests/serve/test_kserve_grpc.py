# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# type: ignore

import max.serve.grpc_serve.grpc_predict_v2_pb2 as pb2
import max.serve.grpc_serve.grpc_serve as max_grpc
import pytest
from max.serve.grpc_serve.grpc_predict_v2_pb2_grpc import (
    GRPCInferenceServiceStub,
    add_GRPCInferenceServiceServicer_to_server,
)
from max.serve.pipelines.performance_fake import get_performance_fake


@pytest.fixture(scope="module")
def grpc_add_to_server():
    return add_GRPCInferenceServiceServicer_to_server


@pytest.fixture(scope="module")
def grpc_servicer(fixture_tokenizer):
    model_name = "perf-fake"
    tokenizer = fixture_tokenizer
    pipeline = get_performance_fake("no-op")
    service = max_grpc.MaxDirectInferenceService(
        model_name, pipeline, tokenizer, 16
    )
    return service


@pytest.fixture(scope="module")
def grpc_stub_cls(grpc_channel):
    return GRPCInferenceServiceStub


@pytest.mark.asyncio
async def test_server_live(grpc_stub):
    request = pb2.ServerLiveRequest()
    response: pb2.ServerLiveResponse = grpc_stub.ServerLive(request)
    assert response.live == True


@pytest.mark.asyncio
async def test_server_ready(grpc_stub):
    request = pb2.ServerReadyRequest()
    response: pb2.ServerReadyResponse = grpc_stub.ServerReady(request)
    assert response.ready == True


@pytest.mark.asyncio
async def test_model_ready(grpc_stub):
    request = pb2.ModelReadyRequest(name="perf-fake", version="0")
    response: pb2.ModelReadyResponse = grpc_stub.ModelReady(request)
    assert response.ready == True


@pytest.mark.asyncio
async def test_server_metadata(grpc_stub):
    request = pb2.ServerMetadataRequest()
    response: pb2.ServerMetadataResponse = grpc_stub.ServerMetadata(request)
    assert response.name == "max-grpc-kserve"
    assert response.version == "DEBUG"
    assert response.extensions == []


@pytest.mark.asyncio
async def test_model_metadata(grpc_stub):
    request = pb2.ModelMetadataRequest(name="perf-fake", version="0")
    response: pb2.ModelMetadataResponse = grpc_stub.ModelMetadata(request)
    assert response.name == "perf-fake" and response.versions == []


@pytest.mark.skip(
    reason="grpc-test doesn't handle async, but model-infer is async"
)
def test_model_infer(grpc_stub):
    request = pb2.ModelInferRequest(model_name="perf-fake", model_version="0")
    response: pb2.ModelInferResponse = grpc_stub.ModelInfer(request)
    assert response.name == "perf-fake" and response.versions == []
