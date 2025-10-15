# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Interfaces and status enums for generation processes in the MAX API."""

from enum import Enum


class GenerationStatus(str, Enum):
    """Enum representing the status of a generation process in the MAX API."""

    ACTIVE = "active"
    """The generation process is ongoing."""
    END_OF_SEQUENCE = "end_of_sequence"
    """The generation process has reached the end of the sequence."""
    MAXIMUM_LENGTH = "maximum_length"
    """The generation process has reached the maximum allowed length."""
    CANCELLED = "cancelled"
    """The generation process has been cancelled by the user."""

    @property
    def is_done(self) -> bool:
        """Returns True if the generation process is complete (not ACTIVE).

        Returns:
            bool: True if the status is not ACTIVE, indicating completion.
        """
        return self is not GenerationStatus.ACTIVE
