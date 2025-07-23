# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
import dataclasses
from typing import TypeVar

from typing_extensions import TypeAlias

RequestID: TypeAlias = str
"""A unique identifier for a request within the MAX API.

This type alias is used throughout the MAX stack to represent request IDs,
ensuring type clarity and consistency across interfaces and implementations.
"""


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
        }
    )


RequestType = TypeVar("RequestType", bound=Request, contravariant=True)
"""
Type variable for request types.

This TypeVar is bound to the Request base class, ensuring that any type used
with this variable must inherit from Request. It is used for generic typing
in interfaces and implementations that operate on requests.
"""
