# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from time import sleep

from os import Atomic


struct BlockingSpinLock[SleepDuration: Float64 = 0.00001]:
    """A basic locking implementation that uses an integer to represent the
    owner of the lock."""

    alias UNLOCKED = -1
    # non-zero means locked, -1 means unlocked
    var counter: Atomic[DType.int64]

    fn __init__(inout self: Self):
        self.counter = Atomic[DType.int64](Self.UNLOCKED)

    fn lock(inout self: Self, owner: Int):
        var expected = Int64(Self.UNLOCKED)
        while not self.counter.compare_exchange_weak(expected, owner):
            # this should be yield
            sleep(self.SleepDuration)
            expected = Self.UNLOCKED

    fn unlock(inout self: Self, owner: Int) -> Bool:
        var expected = Int64(owner)
        if self.counter.load() != owner:
            # No one else can modify other than owner
            return False
        while not self.counter.compare_exchange_weak(expected, Self.UNLOCKED):
            expected = owner
        return True


struct BlockingScopedLock[SleepDuration: Float64 = 0.000001]:
    alias LockType = BlockingSpinLock[SleepDuration]
    var lock: Pointer[Self.LockType]

    fn __init__(
        inout self,
        lock: Pointer[Self.LockType],
    ):
        self.lock = lock

    @no_inline
    fn __enter__(inout self):
        """Acquire the lock on entry.
        This is done by setting the owner of the lock to own address."""
        var address = Pointer[Self].address_of(self)
        self.lock[].lock(int(address))

    @no_inline
    fn __exit__(inout self):
        """Release the lock on exit.
        Reset the address on the underlying lock."""
        var address = Pointer[Self].address_of(self)
        _ = self.lock[].unlock(int(address))
