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
from __future__ import annotations

import dataclasses
import uuid
from typing import TypeVar


@dataclasses.dataclass(frozen=True)
class RequestID:
    """A unique immutable identifier for a request.

    When instantiated without arguments, automatically generates a new
    UUID4-based ID.

    Configuration:
        value: The string identifier. If not provided, generates a UUID4 hex string.
    """

    value: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex)

    def __str__(self) -> str:
        return self.value


DUMMY_REQUEST_ID = RequestID("cuda_graph_dummy")


@dataclasses.dataclass(frozen=True)
class Request:
    """Base class representing a generic request within the MAX API.

    This class provides a unique identifier for each request, ensuring that
    all requests can be tracked and referenced consistently throughout the
    system. Subclasses can extend this class to include additional fields
    specific to their request types.

    """

    request_id: RequestID = dataclasses.field(
        metadata={
            "doc": (
                "A unique identifier for the request, automatically "
                "generated using UUID4 if not provided."
            )
        },
    )

    def __str__(self) -> str:
        return str(self.request_id)


RequestType = TypeVar("RequestType", bound=Request, contravariant=True)
"""
Type variable for request types.

This TypeVar is bound to the Request base class, ensuring that any type used
with this variable must inherit from Request. It is used for generic typing
in interfaces and implementations that operate on requests.
"""
