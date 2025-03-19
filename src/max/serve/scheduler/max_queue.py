# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import multiprocessing
from abc import ABC, abstractmethod
from typing import Any


class AtomicInt:
    """A atomic integer counter that can be shared across processes.

    This counter is strictly non-negative.
    """

    def __init__(self, ctx: multiprocessing.context.BaseContext, x: int = 0):
        self.counter: Any = ctx.Value("i", x)

    def inc(self) -> int:
        """Increment the counter by 1 and returns the old value."""
        with self.counter.get_lock():
            x = self.counter.value
            self.counter.value += 1
            return x

    def dec(self) -> int:
        """Decrement the counter by 1 if it is greater than 0 and returns the old value.
        Returns None if the counter is 0."""
        with self.counter.get_lock():
            x = self.counter.value
            if x > 0:
                self.counter.value -= 1
            return x

    @property
    def value(self) -> int:
        """Return the value of the counter"""
        return self.counter.value


class MaxQueue(ABC):
    @abstractmethod
    def get_nowait(self) -> Any: ...

    @abstractmethod
    def put_nowait(self, item: Any) -> None: ...

    @abstractmethod
    def get(self, *args, **kwargs) -> Any: ...

    @abstractmethod
    def put(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def qsize(self) -> int: ...

    @abstractmethod
    def empty(self) -> bool: ...
