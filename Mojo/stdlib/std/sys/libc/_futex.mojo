# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Futex API constants.

The values are transcribed from the Linux UAPI header `<linux/futex.h>`.
"""

from std.ffi import c_int

# ===----------------------------------------------------------------------=== #
# Operations
# ===----------------------------------------------------------------------=== #

comptime FUTEX_WAIT: c_int = 0
"""Sleep while the futex word holds the expected value."""
comptime FUTEX_WAKE: c_int = 1
"""Wake up to `val` waiters on the futex."""
comptime FUTEX_FD: c_int = 2
"""Create a file descriptor for the futex (removed in Linux 2.6.26)."""
comptime FUTEX_REQUEUE: c_int = 3
"""Wake waiters and requeue the rest onto a second futex."""
comptime FUTEX_CMP_REQUEUE: c_int = 4
"""Like `FUTEX_REQUEUE`, but first check the futex word against `val3`."""
comptime FUTEX_WAKE_OP: c_int = 5
"""Modify a second futex word and conditionally wake waiters on both."""
comptime FUTEX_LOCK_PI: c_int = 6
"""Acquire a priority-inheritance futex."""
comptime FUTEX_UNLOCK_PI: c_int = 7
"""Release a priority-inheritance futex."""
comptime FUTEX_TRYLOCK_PI: c_int = 8
"""Try to acquire a priority-inheritance futex without blocking."""
comptime FUTEX_WAIT_BITSET: c_int = 9
"""Like `FUTEX_WAIT`, but with an absolute timeout and a wake bitset."""
comptime FUTEX_WAKE_BITSET: c_int = 10
"""Like `FUTEX_WAKE`, but only waiters whose bitset intersects `val3`."""
comptime FUTEX_WAIT_REQUEUE_PI: c_int = 11
"""Wait on a non-PI futex that may be requeued onto a PI futex."""
comptime FUTEX_CMP_REQUEUE_PI: c_int = 12
"""Requeue waiters from a non-PI futex onto a PI futex."""
comptime FUTEX_LOCK_PI2: c_int = 13
"""Like `FUTEX_LOCK_PI`, but the timeout clock is selectable."""

# ===----------------------------------------------------------------------=== #
# Operation flags
# ===----------------------------------------------------------------------=== #

comptime FUTEX_PRIVATE_FLAG: c_int = 128
"""The futex is private to the process."""
comptime FUTEX_CLOCK_REALTIME: c_int = 256
"""Measure the timeout against `CLOCK_REALTIME` instead of `CLOCK_MONOTONIC`."""
comptime FUTEX_CMD_MASK: c_int = ~(FUTEX_PRIVATE_FLAG | FUTEX_CLOCK_REALTIME)
"""Mask that strips the operation flags, leaving the operation."""

comptime FUTEX_WAIT_PRIVATE: c_int = FUTEX_WAIT | FUTEX_PRIVATE_FLAG
"""`FUTEX_WAIT` on a process-private futex."""
comptime FUTEX_WAKE_PRIVATE: c_int = FUTEX_WAKE | FUTEX_PRIVATE_FLAG
"""`FUTEX_WAKE` on a process-private futex."""
comptime FUTEX_REQUEUE_PRIVATE: c_int = FUTEX_REQUEUE | FUTEX_PRIVATE_FLAG
"""`FUTEX_REQUEUE` on a process-private futex."""
comptime FUTEX_CMP_REQUEUE_PRIVATE: c_int = (
    FUTEX_CMP_REQUEUE | FUTEX_PRIVATE_FLAG
)
"""`FUTEX_CMP_REQUEUE` on a process-private futex."""
comptime FUTEX_WAKE_OP_PRIVATE: c_int = FUTEX_WAKE_OP | FUTEX_PRIVATE_FLAG
"""`FUTEX_WAKE_OP` on a process-private futex."""
comptime FUTEX_LOCK_PI_PRIVATE: c_int = FUTEX_LOCK_PI | FUTEX_PRIVATE_FLAG
"""`FUTEX_LOCK_PI` on a process-private futex."""
comptime FUTEX_LOCK_PI2_PRIVATE: c_int = FUTEX_LOCK_PI2 | FUTEX_PRIVATE_FLAG
"""`FUTEX_LOCK_PI2` on a process-private futex."""
comptime FUTEX_UNLOCK_PI_PRIVATE: c_int = FUTEX_UNLOCK_PI | FUTEX_PRIVATE_FLAG
"""`FUTEX_UNLOCK_PI` on a process-private futex."""
comptime FUTEX_TRYLOCK_PI_PRIVATE: c_int = (
    FUTEX_TRYLOCK_PI | FUTEX_PRIVATE_FLAG
)
"""`FUTEX_TRYLOCK_PI` on a process-private futex."""
comptime FUTEX_WAIT_BITSET_PRIVATE: c_int = (
    FUTEX_WAIT_BITSET | FUTEX_PRIVATE_FLAG
)
"""`FUTEX_WAIT_BITSET` on a process-private futex."""
comptime FUTEX_WAKE_BITSET_PRIVATE: c_int = (
    FUTEX_WAKE_BITSET | FUTEX_PRIVATE_FLAG
)
"""`FUTEX_WAKE_BITSET` on a process-private futex."""
comptime FUTEX_WAIT_REQUEUE_PI_PRIVATE: c_int = (
    FUTEX_WAIT_REQUEUE_PI | FUTEX_PRIVATE_FLAG
)
"""`FUTEX_WAIT_REQUEUE_PI` on a process-private futex."""
comptime FUTEX_CMP_REQUEUE_PI_PRIVATE: c_int = (
    FUTEX_CMP_REQUEUE_PI | FUTEX_PRIVATE_FLAG
)
"""`FUTEX_CMP_REQUEUE_PI` on a process-private futex."""

# ===----------------------------------------------------------------------=== #
# Robust and priority-inheritance futex word bits
# ===----------------------------------------------------------------------=== #

comptime FUTEX_WAITERS: UInt32 = 0x80000000
"""The futex has waiters blocked in the kernel."""
comptime FUTEX_OWNER_DIED: UInt32 = 0x40000000
"""The owning thread exited without releasing the futex."""
comptime FUTEX_TID_MASK: UInt32 = 0x3FFFFFFF
"""Mask that selects the owner's thread ID from the futex word."""
comptime ROBUST_LIST_LIMIT: Int = 2048
"""Maximum number of robust-list entries the kernel walks at thread exit."""
comptime FUTEX_BITSET_MATCH_ANY: UInt32 = 0xFFFFFFFF
"""Bitset that matches every waiter in the `*_BITSET` operations."""

# ===----------------------------------------------------------------------=== #
# FUTEX_WAKE_OP encoding
# ===----------------------------------------------------------------------=== #

comptime FUTEX_OP_SET: c_int = 0
"""`*uaddr2 = oparg`."""
comptime FUTEX_OP_ADD: c_int = 1
"""`*uaddr2 += oparg`."""
comptime FUTEX_OP_OR: c_int = 2
"""`*uaddr2 |= oparg`."""
comptime FUTEX_OP_ANDN: c_int = 3
"""`*uaddr2 &= ~oparg`."""
comptime FUTEX_OP_XOR: c_int = 4
"""`*uaddr2 ^= oparg`."""
comptime FUTEX_OP_OPARG_SHIFT: c_int = 8
"""Flag OR'd into the operation to use `1 << oparg` as the operand."""

comptime FUTEX_OP_CMP_EQ: c_int = 0
"""Wake on `uaddr2` if `oldval == cmparg`."""
comptime FUTEX_OP_CMP_NE: c_int = 1
"""Wake on `uaddr2` if `oldval != cmparg`."""
comptime FUTEX_OP_CMP_LT: c_int = 2
"""Wake on `uaddr2` if `oldval < cmparg`."""
comptime FUTEX_OP_CMP_LE: c_int = 3
"""Wake on `uaddr2` if `oldval <= cmparg`."""
comptime FUTEX_OP_CMP_GT: c_int = 4
"""Wake on `uaddr2` if `oldval > cmparg`."""
comptime FUTEX_OP_CMP_GE: c_int = 5
"""Wake on `uaddr2` if `oldval >= cmparg`."""


def FUTEX_OP(op: c_int, oparg: c_int, cmp: c_int, cmparg: c_int) -> c_int:
    """Encodes a `FUTEX_WAKE_OP` operation into its `val3` argument.

    Args:
        op: A `FUTEX_OP_*` operation, optionally OR'd with
            `FUTEX_OP_OPARG_SHIFT`.
        oparg: The 12-bit operand of `op`.
        cmp: A `FUTEX_OP_CMP_*` comparison.
        cmparg: The 12-bit operand of `cmp`.

    Returns:
        The encoded operation.
    """
    return (
        ((op & 0xF) << 28)
        | ((cmp & 0xF) << 24)
        | ((oparg & 0xFFF) << 12)
        | (cmparg & 0xFFF)
    )
