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
"""C ABI manifest: the `std.sys.libc._futex` mirrors of `<linux/futex.h>`."""

from test_utils.cabi_check import AbiConstant, AbiTypedefLike

from std.sys.libc._futex import (
    FUTEX_BITSET_MATCH_ANY,
    FUTEX_CLOCK_REALTIME,
    FUTEX_CMD_MASK,
    FUTEX_CMP_REQUEUE,
    FUTEX_CMP_REQUEUE_PI,
    FUTEX_CMP_REQUEUE_PI_PRIVATE,
    FUTEX_CMP_REQUEUE_PRIVATE,
    FUTEX_FD,
    FUTEX_LOCK_PI,
    FUTEX_LOCK_PI2,
    FUTEX_LOCK_PI2_PRIVATE,
    FUTEX_LOCK_PI_PRIVATE,
    FUTEX_OP_ADD,
    FUTEX_OP_ANDN,
    FUTEX_OP_CMP_EQ,
    FUTEX_OP_CMP_GE,
    FUTEX_OP_CMP_GT,
    FUTEX_OP_CMP_LE,
    FUTEX_OP_CMP_LT,
    FUTEX_OP_CMP_NE,
    FUTEX_OP_OPARG_SHIFT,
    FUTEX_OP_OR,
    FUTEX_OP_SET,
    FUTEX_OP_XOR,
    FUTEX_OWNER_DIED,
    FUTEX_PRIVATE_FLAG,
    FUTEX_REQUEUE,
    FUTEX_REQUEUE_PRIVATE,
    FUTEX_TID_MASK,
    FUTEX_TRYLOCK_PI,
    FUTEX_TRYLOCK_PI_PRIVATE,
    FUTEX_UNLOCK_PI,
    FUTEX_UNLOCK_PI_PRIVATE,
    FUTEX_WAIT,
    FUTEX_WAITERS,
    FUTEX_WAIT_BITSET,
    FUTEX_WAIT_BITSET_PRIVATE,
    FUTEX_WAIT_PRIVATE,
    FUTEX_WAIT_REQUEUE_PI,
    FUTEX_WAIT_REQUEUE_PI_PRIVATE,
    FUTEX_WAKE,
    FUTEX_WAKE_BITSET,
    FUTEX_WAKE_BITSET_PRIVATE,
    FUTEX_WAKE_OP,
    FUTEX_WAKE_OP_PRIVATE,
    FUTEX_WAKE_PRIVATE,
    ROBUST_LIST_LIMIT,
)

comptime CABI_INCLUDES: List[StaticString] = [
    "<linux/futex.h>",
]
"""The headers that declare the C side."""

comptime CABI_STRUCTS = TypeList.of[Trait=AnyType]()
"""The struct mirrors to check."""

comptime CABI_TYPEDEFS = TypeList.of[Trait=AbiTypedefLike]()
"""The type aliases to check."""

comptime CABI_CONSTANTS = [
    AbiConstant("FUTEX_WAIT", FUTEX_WAIT),
    AbiConstant("FUTEX_WAKE", FUTEX_WAKE),
    AbiConstant("FUTEX_FD", FUTEX_FD),
    AbiConstant("FUTEX_REQUEUE", FUTEX_REQUEUE),
    AbiConstant("FUTEX_CMP_REQUEUE", FUTEX_CMP_REQUEUE),
    AbiConstant("FUTEX_WAKE_OP", FUTEX_WAKE_OP),
    AbiConstant("FUTEX_LOCK_PI", FUTEX_LOCK_PI),
    AbiConstant("FUTEX_UNLOCK_PI", FUTEX_UNLOCK_PI),
    AbiConstant("FUTEX_TRYLOCK_PI", FUTEX_TRYLOCK_PI),
    AbiConstant("FUTEX_WAIT_BITSET", FUTEX_WAIT_BITSET),
    AbiConstant("FUTEX_WAKE_BITSET", FUTEX_WAKE_BITSET),
    AbiConstant("FUTEX_WAIT_REQUEUE_PI", FUTEX_WAIT_REQUEUE_PI),
    AbiConstant("FUTEX_CMP_REQUEUE_PI", FUTEX_CMP_REQUEUE_PI),
    AbiConstant("FUTEX_LOCK_PI2", FUTEX_LOCK_PI2),
    AbiConstant("FUTEX_PRIVATE_FLAG", FUTEX_PRIVATE_FLAG),
    AbiConstant("FUTEX_CLOCK_REALTIME", FUTEX_CLOCK_REALTIME),
    AbiConstant("FUTEX_CMD_MASK", FUTEX_CMD_MASK),
    AbiConstant("FUTEX_WAIT_PRIVATE", FUTEX_WAIT_PRIVATE),
    AbiConstant("FUTEX_WAKE_PRIVATE", FUTEX_WAKE_PRIVATE),
    AbiConstant("FUTEX_REQUEUE_PRIVATE", FUTEX_REQUEUE_PRIVATE),
    AbiConstant("FUTEX_CMP_REQUEUE_PRIVATE", FUTEX_CMP_REQUEUE_PRIVATE),
    AbiConstant("FUTEX_WAKE_OP_PRIVATE", FUTEX_WAKE_OP_PRIVATE),
    AbiConstant("FUTEX_LOCK_PI_PRIVATE", FUTEX_LOCK_PI_PRIVATE),
    AbiConstant("FUTEX_LOCK_PI2_PRIVATE", FUTEX_LOCK_PI2_PRIVATE),
    AbiConstant("FUTEX_UNLOCK_PI_PRIVATE", FUTEX_UNLOCK_PI_PRIVATE),
    AbiConstant("FUTEX_TRYLOCK_PI_PRIVATE", FUTEX_TRYLOCK_PI_PRIVATE),
    AbiConstant("FUTEX_WAIT_BITSET_PRIVATE", FUTEX_WAIT_BITSET_PRIVATE),
    AbiConstant("FUTEX_WAKE_BITSET_PRIVATE", FUTEX_WAKE_BITSET_PRIVATE),
    AbiConstant("FUTEX_WAIT_REQUEUE_PI_PRIVATE", FUTEX_WAIT_REQUEUE_PI_PRIVATE),
    AbiConstant("FUTEX_CMP_REQUEUE_PI_PRIVATE", FUTEX_CMP_REQUEUE_PI_PRIVATE),
    AbiConstant("FUTEX_WAITERS", FUTEX_WAITERS),
    AbiConstant("FUTEX_OWNER_DIED", FUTEX_OWNER_DIED),
    AbiConstant("FUTEX_TID_MASK", FUTEX_TID_MASK),
    AbiConstant("ROBUST_LIST_LIMIT", Int64(ROBUST_LIST_LIMIT)),
    AbiConstant("FUTEX_BITSET_MATCH_ANY", FUTEX_BITSET_MATCH_ANY),
    AbiConstant("FUTEX_OP_SET", FUTEX_OP_SET),
    AbiConstant("FUTEX_OP_ADD", FUTEX_OP_ADD),
    AbiConstant("FUTEX_OP_OR", FUTEX_OP_OR),
    AbiConstant("FUTEX_OP_ANDN", FUTEX_OP_ANDN),
    AbiConstant("FUTEX_OP_XOR", FUTEX_OP_XOR),
    AbiConstant("FUTEX_OP_OPARG_SHIFT", FUTEX_OP_OPARG_SHIFT),
    AbiConstant("FUTEX_OP_CMP_EQ", FUTEX_OP_CMP_EQ),
    AbiConstant("FUTEX_OP_CMP_NE", FUTEX_OP_CMP_NE),
    AbiConstant("FUTEX_OP_CMP_LT", FUTEX_OP_CMP_LT),
    AbiConstant("FUTEX_OP_CMP_LE", FUTEX_OP_CMP_LE),
    AbiConstant("FUTEX_OP_CMP_GT", FUTEX_OP_CMP_GT),
    AbiConstant("FUTEX_OP_CMP_GE", FUTEX_OP_CMP_GE),
]
"""The constants to check."""
