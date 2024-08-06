# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import pytest
from fastapi.testclient import TestClient

from max.serve.config import APIType, get_settings

get_settings().api_types = [APIType.KSERVE]

from max.serve.api_server import create_app
from max.serve.mocks.mock_api_requests import (
    simple_kserve_request,
    simple_kserve_response,
)

app = create_app()
client = TestClient(app)


@pytest.mark.skip(reason="Implementing infer/ for real is a WIP.")
def test_kserve_basic_infer():
    response = client.post(
        "/v2/models/Add/versions/0/infer", json=simple_kserve_request()
    )
    assert response.json() == simple_kserve_response()
