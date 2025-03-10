# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Interfaces for interacting with record/replay structures."""

from __future__ import annotations

import abc

from .schema import RecordingItem

__all__ = [
    # Please keep this list alphabetized.
    "Recorder",
]


class Recorder(abc.ABC):
    """Something that can accept recording items to be persisted somewhere."""

    @abc.abstractmethod
    def record(self, item: RecordingItem) -> None:
        """Record an item."""
        ...
