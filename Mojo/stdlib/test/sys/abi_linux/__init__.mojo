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
"""C ABI manifest: the Linux-specific `std.sys` mirrors."""

from test_utils.cabi_check import AbiConstant, AbiTypedefLike

from std.sys._io_uring import (
    IORING_CQ_EVENTFD_DISABLED,
    IORING_ENTER_EXT_ARG,
    IORING_ENTER_GETEVENTS,
    IORING_ENTER_SQ_WAIT,
    IORING_ENTER_SQ_WAKEUP,
    IORING_FEAT_CUR_PERSONALITY,
    IORING_FEAT_EXT_ARG,
    IORING_FEAT_FAST_POLL,
    IORING_FEAT_NATIVE_WORKERS,
    IORING_FEAT_NODROP,
    IORING_FEAT_POLL_32BITS,
    IORING_FEAT_RSRC_TAGS,
    IORING_FEAT_RW_CUR_POS,
    IORING_FEAT_SINGLE_MMAP,
    IORING_FEAT_SQPOLL_NONFIXED,
    IORING_FEAT_SUBMIT_STABLE,
    IORING_OFF_CQ_RING,
    IORING_OFF_SQES,
    IORING_OFF_SQ_RING,
    IORING_OP_NOP,
    IORING_SETUP_ATTACH_WQ,
    IORING_SETUP_CLAMP,
    IORING_SETUP_CQSIZE,
    IORING_SETUP_IOPOLL,
    IORING_SETUP_R_DISABLED,
    IORING_SETUP_SQPOLL,
    IORING_SETUP_SQ_AFF,
    IORING_SQ_CQ_OVERFLOW,
    IORING_SQ_NEED_WAKEUP,
    __NR_io_uring_enter,
    __NR_io_uring_setup,
    io_cqring_offsets,
    io_sqring_offsets,
    io_uring_cqe,
    io_uring_params,
    io_uring_sqe,
)
from std.sys._libc import MAP_POPULATE

comptime CABI_INCLUDES: List[StaticString] = [
    "<linux/io_uring.h>",
    "<sys/mman.h>",
    "<sys/syscall.h>",
]
"""The headers that declare the C side."""

comptime CABI_STRUCTS = TypeList.of[
    Trait=AnyType,
    io_uring_sqe,
    io_uring_cqe,
    io_sqring_offsets,
    io_cqring_offsets,
    io_uring_params,
]()
"""The struct mirrors to check."""

comptime CABI_TYPEDEFS = TypeList.of[Trait=AbiTypedefLike]()
"""The type aliases to check."""

comptime CABI_CONSTANTS = [
    AbiConstant("__NR_io_uring_setup", __NR_io_uring_setup),
    AbiConstant("__NR_io_uring_enter", __NR_io_uring_enter),
    AbiConstant("MAP_POPULATE", MAP_POPULATE),
    AbiConstant("IORING_OFF_SQ_RING", IORING_OFF_SQ_RING),
    AbiConstant("IORING_OFF_CQ_RING", IORING_OFF_CQ_RING),
    AbiConstant("IORING_OFF_SQES", IORING_OFF_SQES),
    AbiConstant("IORING_OP_NOP", IORING_OP_NOP),
    AbiConstant("IORING_SETUP_IOPOLL", IORING_SETUP_IOPOLL),
    AbiConstant("IORING_SETUP_SQPOLL", IORING_SETUP_SQPOLL),
    AbiConstant("IORING_SETUP_SQ_AFF", IORING_SETUP_SQ_AFF),
    AbiConstant("IORING_SETUP_CQSIZE", IORING_SETUP_CQSIZE),
    AbiConstant("IORING_SETUP_CLAMP", IORING_SETUP_CLAMP),
    AbiConstant("IORING_SETUP_ATTACH_WQ", IORING_SETUP_ATTACH_WQ),
    AbiConstant("IORING_SETUP_R_DISABLED", IORING_SETUP_R_DISABLED),
    AbiConstant("IORING_FEAT_SINGLE_MMAP", IORING_FEAT_SINGLE_MMAP),
    AbiConstant("IORING_FEAT_NODROP", IORING_FEAT_NODROP),
    AbiConstant("IORING_FEAT_SUBMIT_STABLE", IORING_FEAT_SUBMIT_STABLE),
    AbiConstant("IORING_FEAT_RW_CUR_POS", IORING_FEAT_RW_CUR_POS),
    AbiConstant("IORING_FEAT_CUR_PERSONALITY", IORING_FEAT_CUR_PERSONALITY),
    AbiConstant("IORING_FEAT_FAST_POLL", IORING_FEAT_FAST_POLL),
    AbiConstant("IORING_FEAT_POLL_32BITS", IORING_FEAT_POLL_32BITS),
    AbiConstant("IORING_FEAT_SQPOLL_NONFIXED", IORING_FEAT_SQPOLL_NONFIXED),
    AbiConstant("IORING_FEAT_EXT_ARG", IORING_FEAT_EXT_ARG),
    AbiConstant("IORING_FEAT_NATIVE_WORKERS", IORING_FEAT_NATIVE_WORKERS),
    AbiConstant("IORING_FEAT_RSRC_TAGS", IORING_FEAT_RSRC_TAGS),
    AbiConstant("IORING_SQ_NEED_WAKEUP", IORING_SQ_NEED_WAKEUP),
    AbiConstant("IORING_SQ_CQ_OVERFLOW", IORING_SQ_CQ_OVERFLOW),
    AbiConstant("IORING_CQ_EVENTFD_DISABLED", IORING_CQ_EVENTFD_DISABLED),
    AbiConstant("IORING_ENTER_GETEVENTS", IORING_ENTER_GETEVENTS),
    AbiConstant("IORING_ENTER_SQ_WAKEUP", IORING_ENTER_SQ_WAKEUP),
    AbiConstant("IORING_ENTER_SQ_WAIT", IORING_ENTER_SQ_WAIT),
    AbiConstant("IORING_ENTER_EXT_ARG", IORING_ENTER_EXT_ARG),
]
"""The constants to check."""
