# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""
Implementation of a CTA-wide semaphore for inter-CTA synchronization.
"""

from memory import UnsafePointer
from sys import triple_is_nvidia_cuda
from sys import llvm_intrinsic
from sys._assembly import inlined_assembly
from .sync import barrier


@register_passable
struct Semaphore:
    var _lock: UnsafePointer[Int32]
    var _wait_thread: Bool
    var _state: Int32

    @always_inline
    fn __init__(inout self, lock: UnsafePointer[Int32], thread_id: Int):
        constrained[triple_is_nvidia_cuda(), "target must be cuda"]()
        self._lock = lock
        self._wait_thread = thread_id <= 0
        self._state = -1

    @always_inline
    fn fetch(inout self):
        if self._wait_thread:
            self._state = inlined_assembly[
                "ld.global.acquire.gpu.b32 $0, [$1];", Int32, constraints="=r,l"
            ](self._lock)

    @always_inline
    fn state(self) -> Int32:
        return self._state

    @always_inline
    fn wait(inout self, status: Int = 0):
        while llvm_intrinsic["llvm.nvvm.barrier0.and", Int32](
            Int32(1) if self._state != status else Int32(0)
        ):
            self.fetch()
        barrier()

    @always_inline
    fn release(inout self, status: Int32 = 0):
        barrier()
        if self._wait_thread:
            inlined_assembly[
                "ld.global.release.gpu.b32 $0, [$1];",
                NoneType,
                constraints="=l,r",
            ](self._lock, status)
