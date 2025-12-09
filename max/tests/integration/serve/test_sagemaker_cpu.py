# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

import logging

logging.basicConfig(
    level=logging.DEBUG,
)
import pytest
from async_asgi_testclient import TestClient
from fastapi import FastAPI

request = {
    "model": "echo",
    "messages": [
        {
            "role": "user",
            "content": "I like to ride my bicycle, I like to ride my bike" * 10,
        },
    ],
    "stream": False,
}


@pytest.mark.asyncio
async def test_invocations(echo_app: FastAPI) -> None:
    async with TestClient(echo_app, timeout=720.0) as client:
        raw_response = await client.post(
            "/invocations",
            json=request,
        )

        assert raw_response.status_code == 200
