# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Schemas for items appearing in a recording."""

from __future__ import annotations

import datetime
from collections.abc import Sequence
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

import msgspec

__all__ = [
    # Please keep this list alphabetized.
    "Dummy",
    "Recording",
    "RecordingItem",
    "Request",
    "Response",
    "ResponseChunk",
    "ResponseStart",
    "Transaction",
]


class Request(msgspec.Struct, kw_only=True):
    """Data provided by the client.

    The timestamps refer to the beginning and end of receiving data, not of the
    entire transaction.
    """

    start_timestamp: datetime.datetime | None = None
    end_timestamp: datetime.datetime | None = None
    method: str
    path: str
    headers: Sequence[tuple[bytes, bytes]] = []
    body: bytes | None = None


class ResponseStart(msgspec.Struct, kw_only=True):
    """Information provided at the beginning of the response."""

    timestamp: datetime.datetime | None = None
    status: int
    headers: Sequence[tuple[bytes, bytes]] = []


class ResponseChunk(msgspec.Struct, kw_only=True):
    """A discrete piece of a response provided by the server."""

    timestamp: datetime.datetime | None = None
    body: bytes


class Response(msgspec.Struct, kw_only=True):
    """The response provided by a server to a clients' request."""

    start: ResponseStart
    chunks: Sequence[ResponseChunk] = []


class Transaction(msgspec.Struct, kw_only=True, tag="transaction"):
    """A complete HTTP transaction.

    Responses may or may not be present, depending on recording settings.
    """

    request: Request
    response: Response | None = None


class Dummy(msgspec.Struct, kw_only=True, tag="dummy"):
    """Nothing of importance.

    Should not appear in real recordings, but serves as a union member in
    RecordingItem to ensure that isinstance is used appropriately, and
    RecordingItem is not used in code as a pure synonym for Transaction.
    """


# RecordingItem may include other items in the future.  Make sure to use
# isinstance as appropriate and do not assume this list is fixed.
# Note: Due to msgspec limitations, Union[] must be used, and the | syntax is
# unsupported in TypeAliases.
RecordingItem: TypeAlias = Union[Dummy, Transaction]

Recording: TypeAlias = Sequence[RecordingItem]
