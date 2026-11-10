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

from std.ffi import ErrNo, _null, c_int, get_errno
from std.io import IOError, RawFileDescriptor
from std.io._syscall import check
from std.sys._io_uring import (
    IORING_FEAT_SINGLE_MMAP,
    io_uring_enter,
    io_uring_params,
    io_uring_setup,
)
from std.testing import TestSuite, assert_equal, assert_true

comptime ENTRIES: UInt32 = 4
comptime STDOUT: c_int = 1


# TODO: The Linux CI runners' sandbox blocks io_uring, so these tests skip
# there. Allow io_uring on every Linux runner, or have runners that allow it
# say so (for example with an environment variable) and fail here instead of
# skipping.
def io_uring_available() raises -> Bool:
    """Reports whether the kernel lets this process create a ring.

    Sandboxes, such as the remote test executors', block the io_uring
    syscalls.
    """
    var params = io_uring_params()
    var fd = io_uring_setup(1, Pointer(to=params))
    if fd >= 0:
        RawFileDescriptor(fd).close()
        return True
    var err = get_errno()
    if err == ErrNo.ENOSYS or err == ErrNo.EPERM:
        return False
    raise IOError.last_os_error("io_uring_setup")


def new_ring() raises -> c_int:
    var params = io_uring_params()
    return check(io_uring_setup(ENTRIES, Pointer(to=params)), "io_uring_setup")


def assert_fails_with(result: c_int, expected: ErrNo) raises:
    var errno = get_errno()
    assert_equal(result, -1)
    assert_equal(errno, expected)


def test_setup_fills_params() raises:
    var params = io_uring_params()
    var fd = check(
        io_uring_setup(ENTRIES, Pointer(to=params)), "io_uring_setup"
    )
    assert_equal(params.sq_entries, ENTRIES)
    assert_equal(params.cq_entries, 2 * ENTRIES)
    assert_true(params.features & IORING_FEAT_SINGLE_MMAP != 0)
    assert_true(params.sq_off.array != 0)
    assert_true(params.cq_off.cqes != 0)
    RawFileDescriptor(fd).close()


def test_setup_rejects_zero_entries() raises:
    var params = io_uring_params()
    assert_fails_with(io_uring_setup(0, Pointer(to=params)), ErrNo.EINVAL)


def test_setup_rejects_unknown_flags() raises:
    var params = io_uring_params(flags=UInt32.MAX)
    assert_fails_with(io_uring_setup(ENTRIES, Pointer(to=params)), ErrNo.EINVAL)


def test_enter_with_nothing_to_do() raises:
    var fd = new_ring()
    assert_equal(io_uring_enter(fd, 0, 0, 0, _null[NoneType], 0), 0)
    RawFileDescriptor(fd).close()


def test_enter_rejects_unknown_flags() raises:
    var fd = new_ring()
    assert_fails_with(
        io_uring_enter(fd, 0, 0, UInt32.MAX, _null[NoneType], 0), ErrNo.EINVAL
    )
    RawFileDescriptor(fd).close()


def test_enter_rejects_non_ring_fd() raises:
    assert_fails_with(
        io_uring_enter(STDOUT, 0, 0, 0, _null[NoneType], 0), ErrNo.EOPNOTSUPP
    )


def main() raises:
    var available = io_uring_available()
    TestSuite.discover_tests[__functions_in_module()]().run(
        skip_all=not available
    )
