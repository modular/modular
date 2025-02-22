# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""This module provides a device-wide semaphore implementation for NVIDIA GPUs.

The Semaphore struct enables inter-CTA (Cooperative Thread Array) synchronization
by providing atomic operations and memory barriers. It uses NVIDIA-specific intrinsics
to implement efficient thread synchronization.

Example:
    ```mojo
    var lock = UnsafePointer[Int32](...)
    var sem = Semaphore(lock, thread_id)

    # Wait for a specific state
    sem.wait(0)

    # Release the semaphore
    sem.release(1)
    ```
"""

from sys import is_nvidia_gpu, llvm_intrinsic
from sys._assembly import inlined_assembly

from memory import UnsafePointer

from .intrinsics import Scope, load_acquire, store_release
from .sync import barrier


@always_inline
fn _barrier_and(state: Int32) -> Int32:
    constrained[is_nvidia_gpu(), "target must be an nvidia GPU"]()
    return llvm_intrinsic["llvm.nvvm.barrier0.and", Int32](state)


@register_passable
struct Semaphore:
    """A device-wide semaphore implementation for GPUs.

    This struct provides atomic operations and memory barriers for inter-CTA synchronization.
    It uses a single thread per CTA to perform atomic operations on a shared lock variable.
    """

    var _lock: UnsafePointer[Int32]
    """Pointer to the shared lock variable in global memory that all CTAs synchronize on"""

    var _wait_thread: Bool
    """Flag indicating if this thread should perform atomic operations (true for thread 0)"""

    var _state: Int32
    """Current state of the semaphore, used to track synchronization status"""

    @always_inline
    fn __init__(out self, lock: UnsafePointer[Int32], thread_id: Int):
        """Initialize a new Semaphore instance.

        Args:
            lock: Pointer to shared lock variable in global memory.
            thread_id: Thread ID within the CTA, used to determine if this thread
                      should perform atomic operations.
        """
        constrained[is_nvidia_gpu(), "target must be cuda"]()
        self._lock = lock
        self._wait_thread = thread_id <= 0
        self._state = -1

    @always_inline
    fn fetch(mut self):
        """Fetch the current state of the semaphore from global memory.

        Only the designated wait thread (thread 0) performs the actual load,
        using an acquire memory ordering to ensure proper synchronization.
        """
        if self._wait_thread:
            self._state = load_acquire[scope = Scope.GPU](self._lock)

    @always_inline
    fn state(self) -> Int32:
        """Get the current state of the semaphore.

        Returns:
            The current state value of the semaphore.
        """
        return self._state

    @always_inline
    fn wait(mut self, status: Int = 0):
        """Wait until the semaphore reaches the specified state.

        Uses a barrier-based spin loop where all threads participate in checking
        the state. Only proceeds when the state matches the expected status.

        Args:
            status: The state value to wait for (defaults to 0).
        """
        while _barrier_and((self._state == status).select(Int32(0), Int32(1))):
            self.fetch()
        barrier()

    @always_inline
    fn release(mut self, status: Int32 = 0):
        """Release the semaphore by setting it to the specified state.

        Ensures all threads have reached this point via a barrier before
        the designated thread updates the semaphore state.

        Args:
            status: The new state value to set (defaults to 0).
        """
        barrier()
        if self._wait_thread:
            store_release[scope = Scope.GPU](self._lock, status)
