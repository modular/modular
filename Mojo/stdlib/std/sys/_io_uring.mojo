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
"""Linux io_uring layouts, constants, and syscall bindings.

The layouts and values mirror `<linux/io_uring.h>` from the Linux 5.15.30
kernel headers, which the `abi_linux` C ABI suite compiles against (the jammy
sysroot). So this is the io_uring API of Linux 5.15.

C declares parts of `struct io_uring_sqe` as anonymous unions. Mojo has no
unions, so each union is one field named after its main member: `off` is also
`addr2`, `addr` is also `splice_off_in`, `op_flags` is the per-operation flags
union (`rw_flags`, `fsync_flags`, and the rest), `buf_index` is also
`buf_group`, and `splice_fd_in` is also `file_index`.

Moving to newer kernel headers: the kernel never moves or resizes an existing
field, so these layouts hold on every newer kernel, and newer constants and
structs are additions. Newer headers do rename some reserved fields, which the
`abi_linux` suite checks by name. Those fields are private here so they can be
renamed without breaking callers, and a `TODO` above each affected struct says
what changes. The `IORING_*_LAST` counts aren't bound, because their values
change with every kernel.
"""

from std.ffi import c_int, c_long, c_size_t, external_call
from std.memory import MaybeUninit
from std.sys.info import CompilationTarget
from std.testing._ffi import CField
from std.utils import StaticTuple

from ._libc import off_t

# ===----------------------------------------------------------------------=== #
# Syscall numbers
# ===----------------------------------------------------------------------=== #

comptime __NR_io_uring_setup: c_long = 425
"""The `io_uring_setup(2)` syscall number."""
comptime __NR_io_uring_enter: c_long = 426
"""The `io_uring_enter(2)` syscall number."""

# ===----------------------------------------------------------------------=== #
# Ring mappings
# ===----------------------------------------------------------------------=== #

comptime IORING_OFF_SQ_RING: off_t = 0
"""The `mmap` offset of the submission queue ring."""
comptime IORING_OFF_CQ_RING: off_t = 0x8000000
"""The `mmap` offset of the completion queue ring."""
comptime IORING_OFF_SQES: off_t = 0x10000000
"""The `mmap` offset of the submission queue entries."""

# ===----------------------------------------------------------------------=== #
# Submission and completion queue entries
# ===----------------------------------------------------------------------=== #


# TODO: Newer kernel headers replace 5.15's `__u64 __pad2[2]` at offset 48 with
# `addr3` at 48 and `__pad2[1]` at 56. Once `abi_linux` checks against them,
# add a public `addr3: UInt64` and shrink `_pad2` to one element. Their new
# union members at offsets 8, 16, 28, and 44 keep the 5.15 names, so those
# fields stay as they are.
# TODO: Conform to `Writable` once `StaticTuple` does.
struct io_uring_sqe(TrivialRegisterPassable):
    """C `struct io_uring_sqe`. All fields zero is an `IORING_OP_NOP`."""

    var opcode: UInt8
    var flags: UInt8
    var ioprio: UInt16
    var fd: Int32
    # TODO: Use a union of `off` and `addr2` once Mojo has unions.
    var off: UInt64
    # TODO: Use a union of `addr` and `splice_off_in` once Mojo has unions.
    var addr: UInt64
    var len: UInt32

    # TODO: Use a union of the per-operation flags once Mojo has unions, and
    # drop this rename. It names a member for the ABI check, which needs one
    # with this field's type, and `rw_flags` is a signed `int`.
    @__annotation(CField(rename="fsync_flags"))
    var op_flags: UInt32

    var user_data: UInt64
    # TODO: Use a union of `buf_index` and `buf_group` once Mojo has unions.
    var buf_index: UInt16
    var personality: UInt16
    # TODO: Use a union of `splice_fd_in` and `file_index` once Mojo has unions.
    var splice_fd_in: Int32

    @__annotation(CField(rename="__pad2", c_type=.array("__u64", 2)))
    var _pad2: StaticTuple[UInt64, 2]

    def __init__(out self):
        # SAFETY: Every field is an integer, so all-zero bytes are a valid
        # value.
        self = MaybeUninit[Self].zeroed().unsafe_assume_init()

    def __init__(
        out self,
        *,
        opcode: UInt8,
        flags: UInt8 = 0,
        ioprio: UInt16 = 0,
        fd: Int32 = 0,
        off: UInt64 = 0,
        addr: UInt64 = 0,
        len: UInt32 = 0,
        op_flags: UInt32 = 0,
        user_data: UInt64 = 0,
        buf_index: UInt16 = 0,
        personality: UInt16 = 0,
        splice_fd_in: Int32 = 0,
    ):
        self = Self()
        self.opcode = opcode
        self.flags = flags
        self.ioprio = ioprio
        self.fd = fd
        self.off = off
        self.addr = addr
        self.len = len
        self.op_flags = op_flags
        self.user_data = user_data
        self.buf_index = buf_index
        self.personality = personality
        self.splice_fd_in = splice_fd_in


@fieldwise_init
struct io_uring_cqe(TrivialRegisterPassable, Writable):
    """C `struct io_uring_cqe`."""

    var user_data: UInt64
    var res: Int32
    var flags: UInt32


comptime IORING_OP_NOP: UInt8 = 0
"""Complete immediately, doing nothing."""

# ===----------------------------------------------------------------------=== #
# Ring setup
# ===----------------------------------------------------------------------=== #


# TODO: Newer kernel headers rename `resv2` to `user_addr`, an input with
# `IORING_SETUP_NO_MMAP`. Once `abi_linux` checks against them, replace
# `_resv2` with a public `user_addr`.
struct io_sqring_offsets(TrivialRegisterPassable, Writable):
    """C `struct io_sqring_offsets`: where the submission ring's fields sit in
    its mapping."""

    var head: UInt32
    var tail: UInt32
    var ring_mask: UInt32
    var ring_entries: UInt32
    var flags: UInt32
    var dropped: UInt32
    var array: UInt32

    @__annotation(CField(rename="resv1"))
    var _resv1: UInt32

    @__annotation(CField(rename="resv2"))
    var _resv2: UInt64

    def __init__(out self):
        # SAFETY: Every field is an integer, so all-zero bytes are a valid
        # value.
        self = MaybeUninit[Self].zeroed().unsafe_assume_init()


# TODO: Newer kernel headers rename `resv2` to `user_addr`, an input with
# `IORING_SETUP_NO_MMAP`. Once `abi_linux` checks against them, replace
# `_resv2` with a public `user_addr`.
struct io_cqring_offsets(TrivialRegisterPassable, Writable):
    """C `struct io_cqring_offsets`: where the completion ring's fields sit in
    its mapping."""

    var head: UInt32
    var tail: UInt32
    var ring_mask: UInt32
    var ring_entries: UInt32
    var overflow: UInt32
    var cqes: UInt32
    var flags: UInt32

    @__annotation(CField(rename="resv1"))
    var _resv1: UInt32

    @__annotation(CField(rename="resv2"))
    var _resv2: UInt64

    def __init__(out self):
        # SAFETY: Every field is an integer, so all-zero bytes are a valid
        # value.
        self = MaybeUninit[Self].zeroed().unsafe_assume_init()


# TODO: Conform to `Writable` once `StaticTuple` does.
struct io_uring_params(TrivialRegisterPassable):
    """C `struct io_uring_params`: the `io_uring_setup` request, which the
    kernel fills in with the ring's sizes, features, and offsets."""

    var sq_entries: UInt32
    var cq_entries: UInt32
    var flags: UInt32
    var sq_thread_cpu: UInt32
    var sq_thread_idle: UInt32
    var features: UInt32
    var wq_fd: UInt32

    @__annotation(CField(rename="resv", c_type=.array("__u32", 3)))
    var _resv: StaticTuple[UInt32, 3]

    var sq_off: io_sqring_offsets
    var cq_off: io_cqring_offsets

    def __init__(out self):
        # SAFETY: Every field is an integer or a struct of integers, so
        # all-zero bytes are a valid value.
        self = MaybeUninit[Self].zeroed().unsafe_assume_init()

    def __init__(
        out self,
        *,
        flags: UInt32,
        cq_entries: UInt32 = 0,
        sq_thread_cpu: UInt32 = 0,
        sq_thread_idle: UInt32 = 0,
        wq_fd: UInt32 = 0,
    ):
        self = Self()
        self.flags = flags
        self.cq_entries = cq_entries
        self.sq_thread_cpu = sq_thread_cpu
        self.sq_thread_idle = sq_thread_idle
        self.wq_fd = wq_fd


comptime IORING_SETUP_IOPOLL: UInt32 = 1 << 0
"""Poll for completions instead of waiting for interrupts."""
comptime IORING_SETUP_SQPOLL: UInt32 = 1 << 1
"""Have a kernel thread poll the submission ring."""
comptime IORING_SETUP_SQ_AFF: UInt32 = 1 << 2
"""Pin the polling thread to `sq_thread_cpu`."""
comptime IORING_SETUP_CQSIZE: UInt32 = 1 << 3
"""Size the completion ring from `cq_entries`."""
comptime IORING_SETUP_CLAMP: UInt32 = 1 << 4
"""Clamp ring sizes to the maximum instead of failing."""
comptime IORING_SETUP_ATTACH_WQ: UInt32 = 1 << 5
"""Share the async workers of the ring in `wq_fd`."""
comptime IORING_SETUP_R_DISABLED: UInt32 = 1 << 6
"""Start the ring disabled, until `IORING_REGISTER_ENABLE_RINGS`."""

comptime IORING_FEAT_SINGLE_MMAP: UInt32 = 1 << 0
"""One `IORING_OFF_SQ_RING` mapping holds both rings."""
comptime IORING_FEAT_NODROP: UInt32 = 1 << 1
"""Completions that overflow the completion ring are kept, not dropped, unless
the kernel runs out of memory."""
comptime IORING_FEAT_SUBMIT_STABLE: UInt32 = 1 << 2
"""Data a request points to, such as an `iovec` array, can be reused once the
submission ring's head moves past its entry. Its I/O buffers must stay valid
until its completion."""
comptime IORING_FEAT_RW_CUR_POS: UInt32 = 1 << 3
"""Reads and writes at offset -1 use the file's current position."""
comptime IORING_FEAT_CUR_PERSONALITY: UInt32 = 1 << 4
"""Requests run with the submitter's credentials."""
comptime IORING_FEAT_FAST_POLL: UInt32 = 1 << 5
"""Requests on unready files are polled internally, not handed to a worker."""
comptime IORING_FEAT_POLL_32BITS: UInt32 = 1 << 6
"""Poll takes 32-bit event masks."""
comptime IORING_FEAT_SQPOLL_NONFIXED: UInt32 = 1 << 7
"""`IORING_SETUP_SQPOLL` works with unregistered files."""
comptime IORING_FEAT_EXT_ARG: UInt32 = 1 << 8
"""`io_uring_enter` takes `IORING_ENTER_EXT_ARG`."""
comptime IORING_FEAT_NATIVE_WORKERS: UInt32 = 1 << 9
"""Async workers are threads of the submitting task."""
comptime IORING_FEAT_RSRC_TAGS: UInt32 = 1 << 10
"""Registered files and buffers support tags and sparse updates."""

comptime IORING_SQ_NEED_WAKEUP: UInt32 = 1 << 0
"""The submission ring's polling thread needs `IORING_ENTER_SQ_WAKEUP`."""
comptime IORING_SQ_CQ_OVERFLOW: UInt32 = 1 << 1
"""The completion ring has overflowed."""

comptime IORING_CQ_EVENTFD_DISABLED: UInt32 = 1 << 0
"""Completions don't signal the registered eventfd."""

# ===----------------------------------------------------------------------=== #
# Entering the ring
# ===----------------------------------------------------------------------=== #

comptime IORING_ENTER_GETEVENTS: UInt32 = 1 << 0
"""Wait for `min_complete` completions."""
comptime IORING_ENTER_SQ_WAKEUP: UInt32 = 1 << 1
"""Wake the submission ring's polling thread."""
comptime IORING_ENTER_SQ_WAIT: UInt32 = 1 << 2
"""Wait for the `IORING_SETUP_SQPOLL` polling thread to free a submission ring
slot. Other rings ignore it."""
comptime IORING_ENTER_EXT_ARG: UInt32 = 1 << 3
"""`arg` is a `struct io_uring_getevents_arg`, not a signal mask."""

# ===----------------------------------------------------------------------=== #
# Syscalls
# ===----------------------------------------------------------------------=== #

# glibc has no io_uring wrappers, so these call `syscall(2)`. It reads every
# variadic argument as a `long`, so the integer arguments are widened to one.


def io_uring_setup(
    entries: UInt32, params: MutPointer[io_uring_params, _]
) -> c_int:
    """C `io_uring_setup(2)`: creates a ring and returns its descriptor."""
    comptime assert CompilationTarget.is_linux(), "io_uring is Linux-only"
    return c_int(
        external_call["syscall", c_long, num_fixed_args=1](
            __NR_io_uring_setup, c_long(entries), params
        )
    )


def io_uring_enter(
    fd: c_int,
    to_submit: UInt32,
    min_complete: UInt32,
    flags: UInt32,
    arg: Optional[ImmPointer[NoneType, _]],
    argsz: c_size_t,
) -> c_int:
    """C `io_uring_enter(2)`: submits entries and waits for completions."""
    comptime assert CompilationTarget.is_linux(), "io_uring is Linux-only"
    return c_int(
        external_call["syscall", c_long, num_fixed_args=1](
            __NR_io_uring_enter,
            c_long(fd),
            c_long(to_submit),
            c_long(min_complete),
            c_long(flags),
            arg,
            argsz,
        )
    )
