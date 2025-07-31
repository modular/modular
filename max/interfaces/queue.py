# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from __future__ import annotations

from typing import Generic, Protocol, TypeVar, runtime_checkable

ItemType = TypeVar("ItemType")
"""Type variable for items stored in the MAXQueue.

This allows the MAXQueue interface to be generic and type-safe for any item type.
"""


@runtime_checkable
class MAXQueue(Protocol, Generic[ItemType]):
    """
    Protocol for a minimal, non-blocking queue interface in MAX.

    This protocol defines the minimal contract for a queue that supports
    non-blocking put and get operations. It is generic over the item type.

    """

    def put_nowait(self, item: ItemType) -> None:
        """
        Attempt to put an item into the queue without blocking.

        This method is designed to immediately fail (typically by raising an exception)
        if the item cannot be added to the queue at the time of the call. Unlike the
        traditional 'put' method in many queue implementations—which may block until
        space becomes available or the transfer is completed—this method never waits.
        It is intended for use cases where the caller must be notified of failure to
        enqueue immediately, rather than waiting for space.

        Args:
            item (ItemType): The item to be added to the queue.
        """
        ...

    def get_nowait(self) -> ItemType:
        """
        Remove and return an item from the queue without blocking.

        This method is expected to raise `queue.Empty` if no item is available
        to retrieve from the queue.

        Returns:
            ItemType: The item removed from the queue.

        Raises:
            queue.Empty: If the queue is empty and no item can be retrieved.
        """
        ...
