# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
import socket

import pytest
from max.serve.api_server import (
    validate_port_is_free,
)


def test_setting_throws_occupied_port() -> None:
    # Use a free port from the OS. A fixed port can be held by another test
    # on a shared host, and then this test checks nothing.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        sock.listen()
        port = sock.getsockname()[1]
        with pytest.raises(ValueError):
            validate_port_is_free(port)
