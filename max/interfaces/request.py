# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from __future__ import annotations

import dataclasses
import uuid
from typing import TypeVar


@dataclasses.dataclass(frozen=True)
class RequestID:
    """A unique identifier for a request within the MAX API.

    This class wraps a string value to provide a distinct type with stronger type safety
    than a simple alias. It ensures that RequestIDs must be explicitly constructed
    and cannot be accidentally interchanged with regular strings.

    When called without arguments, automatically generates a new UUID4-based ID.
    """

    value: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex)

    def __str__(self) -> str:
        return self.value


@dataclasses.dataclass(frozen=True)
class Request:
    """
    Base class representing a generic request within the MAX API.

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
