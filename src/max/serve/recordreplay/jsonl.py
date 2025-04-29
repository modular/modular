# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""JSONL recording file format support.

In this format, recordings are stored with each recording item encoded as a
single line of JSON.  New lines separate recording items.  This matches the
common "JSON lines" or JSONL format.  See https://jsonlines.org for more
information.  Recording item schema is described by the schema module.
"""

from __future__ import annotations

import types
from collections.abc import Iterable
from pathlib import Path
from typing import Any, BinaryIO, TextIO

import msgspec.json

from .interfaces import Recorder
from .schema import RecordingItem

__all__ = [
    # Please keep this list alphabetized.
    "JSONLFileRecorder",
    "read_jsonl_recording",
]


class JSONLFileRecorder(Recorder):
    """A recorder that records items to a JSONL file."""

    _file: BinaryIO
    _owned: bool

    def __init__(
        self, f: str | Path | BinaryIO | TextIO, *, owned: bool = True
    ) -> None:
        """Create a recorder that records to a given JSONL file.

        The file can be provided as a path (as either a str or a pathlib.Path
        object), or as an already-open file-like object (both binary and text
        IO is OK).

        If 'owned' is true, this object takes ownership of the lifetime of the
        provided file-like object.  This object should be used as a context
        manager in that case, to ensure the owned file is properly closed.
        'owned' only matters for file-like objects; this object always owns the
        file if a path is provided.
        """
        if isinstance(f, (str, Path)):
            f = Path(f).open("wb")
            owned = True
        if hasattr(f, "buffer"):
            # If we were provided a text file-like object, dig in to get the
            # binary file-like object underneath it, so we can write bytes
            # directly (which is what msgspec gives us).
            f = f.buffer
        self._file = f
        self._owned = owned

    def close(self) -> None:
        """Close the underlying file, if owned.

        This is not required for correct behavior if the file is unowned.  This
        is only required if this recorder owns the underlying file.
        """
        if self._owned:
            self._file.close()

    def __enter__(self) -> JSONLFileRecorder:
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_value: Any,
        exc_tb: types.TracebackType,
    ) -> None:
        self.close()

    def record(self, item: RecordingItem) -> None:
        """Record an item."""
        encoded = msgspec.json.encode(item) + b"\n"
        self._file.write(encoded)


def read_jsonl_recording(
    f: str | Path | BinaryIO | TextIO,
) -> Iterable[RecordingItem]:
    """Read a recording in JSONL format.

    Items are streamed as required.
    """
    if isinstance(f, (str, Path)):
        with Path(f).open("rb") as fh:
            yield from read_jsonl_recording(fh)
        return
    for line in f:
        yield msgspec.json.decode(line, type=RecordingItem)
