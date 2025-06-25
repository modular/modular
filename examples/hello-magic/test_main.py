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

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, magic!"}


def test_zero() -> None:
    mock_process = MagicMock()
    mock_process.stdout.readline.return_value = "tensor([0.])\n"
    mock_process.poll.return_value = 0

    with patch("subprocess.Popen", return_value=mock_process):
        response = client.get("/zero")
        assert response.status_code == 200
        assert response.json() == {"message": "answer is tensor([0.])\n"}


def test_zero_failure() -> None:
    mock_process = MagicMock()
    mock_process.stdout.readline.return_value = ""
    mock_process.poll.return_value = 1

    with patch("subprocess.Popen", return_value=mock_process):
        response = client.get("/zero")
        assert response.status_code == 500
        assert response.json() == {"detail": "Failed to produce zero"}
