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

from std.ffi import c_int
from std.io import IOError, RawFileDescriptor
from std.io._syscall import check
from std.sys._libc import timespec
from std.testing import TestSuite, assert_equal

from std.net._sys.unix import AF_UNIX, SOCK_STREAM, send, socketpair
from std.sys._kqueue.macos import EVFILT_READ, EV_ADD, kevent, kevent_s, kqueue


def poll(kq: c_int, changes: List[kevent_s]) raises IOError -> List[kevent_s]:
    """Applies `changes`, then returns the pending events without waiting."""
    var events = List[kevent_s](
        length=4, fill=kevent_s(ident=0, filter=0, flags=0)
    )
    var timeout = timespec(tv_sec=0, tv_nsec=0)
    var count = check(
        kevent(
            kq,
            changes.unsafe_ptr(),
            c_int(len(changes)),
            events.unsafe_ptr(),
            c_int(len(events)),
            Pointer(to=timeout).as_imm(),
        ),
        "kevent",
    )
    events.shrink(Int(count))
    return events^


def test_read_event_reports_readable_bytes() raises:
    # TODO: Replace with "real" `UnixSocket` types once we have them in the
    # stdlib.
    var pair = Array[c_int, 2](fill=-1)
    _ = check(
        socketpair(AF_UNIX, SOCK_STREAM, 0, Pointer(to=pair)), "socketpair"
    )
    var kq = check(kqueue(), "kqueue")
    var events: List[kevent_s]
    var message = "hello".as_bytes()
    try:
        _ = check(
            send(pair[1], message.unsafe_ptr(), UInt(len(message)), 0), "send"
        )
        events = poll(
            kq,
            [kevent_s(ident=UInt(pair[0]), filter=EVFILT_READ, flags=EV_ADD)],
        )
    finally:
        RawFileDescriptor(kq).close()
        RawFileDescriptor(pair[0]).close()
        RawFileDescriptor(pair[1]).close()
    assert_equal(len(events), 1)
    assert_equal(events[0].ident(), UInt(pair[0]))
    assert_equal(events[0].filter, EVFILT_READ)
    assert_equal(events[0].data(), len(message))


def test_zero_timeout_returns_without_events() raises:
    var kq = check(kqueue(), "kqueue")
    var events: List[kevent_s]
    try:
        events = poll(kq, [])
    finally:
        RawFileDescriptor(kq).close()
    assert_equal(len(events), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
