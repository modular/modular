# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import time
from memory.arc import Arc
from os import Atomic
from runtime.llcl import (
    Chain,
    _init_llcl_chain,
    _async_complete,
    _del_llcl_chain,
    _async_wait,
    _async_wait_timeout,
    Runtime,
)
from runtime.lock import BlockingSpinLock


trait Batchable(CollectionElement):
    pass


struct Batch[Input: Batchable, Output: Batchable]:
    var chain: Chain
    var pending: List[Input]
    var outputs: List[Output]
    var deadline: Int

    # state contains either 0 (unstarted) or 1 (started). The `_start` caller
    # that swaps the zero to the one is the caller that should execute the
    # batch and signal the chain.
    alias UNSTARTED = 0
    alias STARTED = 1
    var started: Atomic[DType.int64]

    fn __init__(inout self: Self, rt: Runtime, capacity: Int):
        var chain = Chain()
        _init_llcl_chain(rt, UnsafePointer[Chain].address_of(chain))
        self.chain = chain
        self.pending = List[Input](capacity=capacity)
        self.outputs = List[Output]()
        self.deadline = 0
        self.started = Int64(self.UNSTARTED)

    fn __moveinit__(inout self: Self, owned existing: Self):
        # N.B. The batch should not be moved presently this is required. See
        # the discussion in MSTDL-386. This should be removed when possible.
        self.chain = existing.chain
        self.pending = existing.pending^
        self.outputs = existing.outputs^
        self.deadline = existing.deadline
        self.started = existing.started.load()
        existing.chain.storage = Pointer[Int].get_null()

    fn _complete(inout self: Self):
        _async_complete(UnsafePointer[Chain].address_of(self.chain))

    fn _start(inout self: Self) -> Bool:
        var expected = Int64(self.UNSTARTED)
        var desired = Int64(self.STARTED)
        while expected == Int64(self.UNSTARTED):
            if self.started.compare_exchange_weak(expected, desired):
                # We have started, execute inline.
                return True
        _async_wait(UnsafePointer[Chain].address_of(self.chain))
        return False

    fn _wait(inout self, last: Bool) -> Bool:
        if self.deadline == 0:
            if last:
                # We will always start.
                return self._start()
            else:
                # Otherwise, wait for the last one always.
                _async_wait(UnsafePointer[Chain].address_of(self.chain))
                return False

        var timeout = self.deadline - time.now()
        if timeout < 0:
            # Execute immediately, if needed.
            return self._start()
        elif _async_wait_timeout(
            UnsafePointer[Chain].address_of(self.chain), timeout
        ):
            # We waited but it was completed before the timeout.
            return False
        else:
            # The timeout has expired, execute if needed.
            return self._start()

    fn __del__(owned self):
        if self.chain.storage != Pointer[Int].get_null():
            _del_llcl_chain(UnsafePointer[Chain].address_of(self.chain))


struct Batcher[
    Input: Batchable,
    Output: Batchable,
    func: fn (List[Input]) capturing -> List[Output],
]:
    var rt: Runtime
    var max: Int
    var timeout: Int
    var current: Arc[Batch[Input, Output]]
    var mu: BlockingSpinLock

    fn __init__(inout self, max: Int = 1, timeout: Int = 0):
        self.rt = Runtime()
        self.max = max
        self.timeout = timeout
        self.current = Batch[Input, Output](rt=self.rt, capacity=self.max)
        self.mu = BlockingSpinLock()

    fn submit(inout self, owned input: Input) -> Output:
        self.mu.lock(0)
        var batch = self.current
        var index = len(batch[].pending)
        batch[].pending.append(input^)
        var last = index + 1 >= self.max
        if index == 0 and self.timeout != 0:
            batch[].deadline = time.now() + self.timeout
        if last:
            self.current = Batch[Input, Output](rt=self.rt, capacity=self.max)
        _ = self.mu.unlock(0)

        # The batch will be executed by the first waiter past the deadline. We
        # wait until this is true, and only one caller will receive `True`,
        # which means we should execute.
        if batch[]._wait(last):
            # Check to see if we need to detach the local batch first. This
            # must happen before we go ahead and execute this batch. This may
            # race with other callers, but that's okay, their call to _wait
            # will return false.
            self.mu.lock(0)
            if self.current._inner == batch._inner:
                self.current = Batch[Input, Output](
                    rt=self.rt, capacity=self.max
                )
            _ = self.mu.unlock(0)
            batch[].outputs = func(batch[].pending)
            batch[]._complete()
        return batch[].outputs[index]

    fn __del__(owned self):
        self.current[]._complete()
