# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Wrapper around stdlib multiprocessing Queue with some additional niceties.
- Number of spurious queue.Empty exceptions is reduced. (https://bugs.python.org/issue43136)
- We support qsize() on multiple platforms like MacOS.
Implementation taken from: https://github.com/keras-team/autokeras/issues/368#issuecomment-461625748
Also see: http://eli.thegreenplace.net/2012/01/04/shared-counter-with-pythons-multiprocessing/
"""

import queue
from multiprocessing import context, queues
from typing import Any

from max.serve.scheduler.max_queue import AtomicInt, MaxQueue


class MpQueue(MaxQueue):
    def __init__(self, ctx: context.BaseContext, *args, **kwargs):
        self.counter: AtomicInt = AtomicInt(ctx, 0)
        self.queue: queues.Queue = ctx.Queue(*args, **kwargs)

    def get_nowait(self) -> Any:
        """Get an item from the queue without blocking for a long time."""
        timeout = 5
        ms = 1e-3
        return self.get(block=True, timeout=timeout * ms)

    def put_nowait(self, item: Any) -> None:
        self.put(item, block=False)

    def get(self, *args, **kwargs) -> Any:
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

    def put(self, *args, **kwargs) -> None:
        try:
            self.queue.put(*args, **kwargs)
        except queue.Full:
            raise queue.Full()
        # only inc the counter if we successfully put an item in the queue
        self.counter.inc()

    def qsize(self) -> int:
        return self.counter.value

    def empty(self) -> bool:
        return self.qsize() == 0
