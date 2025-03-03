# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Wrapper around stdlib multiprocessing Queue with some additional niceties.
- Supports a new put_front_nowait method.
- Number of spurious queue.Empty exceptions is reduced. (https://bugs.python.org/issue43136)

Implementation taken from: https://github.com/keras-team/autokeras/issues/368#issuecomment-461625748
Also see: http://eli.thegreenplace.net/2012/01/04/shared-counter-with-pythons-multiprocessing/
"""

import queue
from multiprocessing import context, queues
from typing import Any


class _AtomicInt(object):
    """A atomic integer counter that can be shared across processes.

    This counter is strictly non-negative.
    """

    def __init__(self, ctx: context.BaseContext, x: int = 0):
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


class Queue:
    def __init__(self, *args, ctx: context.BaseContext, **kwargs):
        self.counter: _AtomicInt = _AtomicInt(ctx, 0)
        self.queue: queues.Queue = ctx.Queue(*args, **kwargs)

    def get_nowait(self):
        """Get an item from the queue without blocking for a long time."""
        timeout = 5
        ms = 1e-3
        return self.get(block=True, timeout=timeout * ms)

    def put_nowait(self, item):
        self.put(item, block=False)

    def get(self, *args, **kwargs):
        # dec the counter before getting an item from the queue
        count = self.counter.dec()
        if count == 0:
            raise queue.Empty()
        try:
            x = self.queue.get(*args, **kwargs)
            return x
        except queue.Empty:
            # inc the counter if we failed to get an item from the queue
            self.counter.inc()
            raise queue.Empty()

    def put(self, *args, **kwargs):
        try:
            self.queue.put(*args, **kwargs)
        except queue.Full:
            raise queue.Full()
        # only inc the counter if we successfully put an item in the queue
        self.counter.inc()

    def qsize(self) -> int:
        return self.counter.value

    def empty(self):
        return self.qsize() == 0
