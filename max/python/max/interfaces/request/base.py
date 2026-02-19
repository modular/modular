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
from typing import Protocol, TypeVar, runtime_checkable


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


@runtime_checkable
class Request(Protocol):
    """Protocol representing a generic request within the MAX API.

    This protocol defines the interface for request types, ensuring that
    all requests can be tracked and referenced consistently throughout the
    system. Any class (dataclass, Pydantic model, etc.) that provides a
    ``request_id`` attribute satisfies this protocol.
    """

    @property
    def request_id(self) -> RequestID:
        """Returns the unique identifier for this request."""
        ...


RequestType = TypeVar("RequestType", bound=Request, contravariant=True)
"""Type variable for request types.

This TypeVar is bound to the Request protocol, ensuring that any type used
with this variable must satisfy the Request protocol interface (have a
request_id attribute and __str__ method). It is contravariant to allow
protocols that accept requests to work with more specific request types.
"""
