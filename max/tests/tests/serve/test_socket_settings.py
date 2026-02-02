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
    # Ensure port occupied
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("", 8000))
        except OSError as e:
            # It's okay if it's occupied already for some reason.
            pass
        with pytest.raises(ValueError):
            validate_port_is_free(8000)
