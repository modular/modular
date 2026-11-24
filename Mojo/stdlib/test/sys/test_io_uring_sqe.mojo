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

# The union aliases of `io_uring_sqe` make no syscalls, so these tests run on
# every platform, unlike `test_io_uring.mojo`.

from std.sys._io_uring import IORING_OP_NOP, io_uring_sqe
from std.testing import TestSuite, assert_equal


def test_aliases_write_their_field() raises:
    var sqe = io_uring_sqe(opcode=IORING_OP_NOP)
    sqe.addr2 = 1
    sqe.splice_off_in = 2
    sqe.buf_group = 3
    assert_equal(sqe.off, 1)
    assert_equal(sqe.addr, 2)
    assert_equal(sqe.buf_index, 3)


def test_every_u32_flag_is_op_flags() raises:
    var sqe = io_uring_sqe(opcode=IORING_OP_NOP, op_flags=7)
    assert_equal(sqe.fsync_flags, 7)
    assert_equal(sqe.poll32_events, 7)
    assert_equal(sqe.sync_range_flags, 7)
    assert_equal(sqe.msg_flags, 7)
    assert_equal(sqe.timeout_flags, 7)
    assert_equal(sqe.accept_flags, 7)
    assert_equal(sqe.cancel_flags, 7)
    assert_equal(sqe.open_flags, 7)
    assert_equal(sqe.statx_flags, 7)
    assert_equal(sqe.fadvise_advice, 7)
    assert_equal(sqe.splice_flags, 7)
    assert_equal(sqe.rename_flags, 7)
    assert_equal(sqe.unlink_flags, 7)
    assert_equal(sqe.hardlink_flags, 7)


def test_signed_aliases_reinterpret_the_bits() raises:
    var sqe = io_uring_sqe(opcode=IORING_OP_NOP, splice_fd_in=-1)
    sqe.rw_flags = -1
    assert_equal(sqe.op_flags, UInt32.MAX)
    assert_equal(sqe.file_index, UInt32.MAX)


def test_poll_events_is_the_low_half_of_op_flags() raises:
    var sqe = io_uring_sqe(opcode=IORING_OP_NOP, op_flags=0xAAAABBBB)
    assert_equal(sqe.poll_events, 0xBBBB)
    sqe.poll_events = 0x1234
    assert_equal(sqe.op_flags, 0xAAAA1234)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
