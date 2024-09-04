# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import pytest
from fastapi.testclient import TestClient
from max.serve.api_server import fastapi_app
from max.serve.config import APIType, Settings
from max.serve.mocks.mock_api_requests import (
    simple_kserve_request,
    simple_kserve_response,
)


@pytest.fixture
def app():
    settings = Settings(api_types=[APIType.KSERVE])
    return fastapi_app(settings)


@pytest.mark.skip(reason="Implementing infer/ for real is a WIP.")
def test_kserve_basic_infer(app):
    with TestClient(app) as client:
        response = client.post(
            "/v2/models/Add/versions/0/infer",
            json=simple_kserve_request(),
        )
        assert response.json() == simple_kserve_response()
