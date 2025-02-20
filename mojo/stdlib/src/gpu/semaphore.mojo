# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""
Implementation of a CTA-wide semaphore for inter-CTA synchronization.
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
    var _lock: UnsafePointer[Int32]
    var _wait_thread: Bool
    var _state: Int32

    @always_inline
    fn __init__(out self, lock: UnsafePointer[Int32], thread_id: Int):
        constrained[is_nvidia_gpu(), "target must be cuda"]()
        self._lock = lock
        self._wait_thread = thread_id <= 0
        self._state = -1

    @always_inline
    fn fetch(mut self):
        if self._wait_thread:
            self._state = load_acquire[scope = Scope.GPU](self._lock)

    @always_inline
    fn state(self) -> Int32:
        return self._state

    @always_inline
    fn wait(mut self, status: Int = 0):
        while _barrier_and((self._state == status).select(Int32(0), Int32(1))):
            self.fetch()
        barrier()

    @always_inline
    fn release(mut self, status: Int32 = 0):
        barrier()
        if self._wait_thread:
            store_release[scope = Scope.GPU](self._lock, status)
