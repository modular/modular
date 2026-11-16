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
"""C ABI manifest: the macOS-specific `std.sys` mirrors."""

from test_utils.cabi_check import AbiConstant, AbiTypedefLike

from std.sys._kqueue.macos import (
    EVFILT_PROC,
    EVFILT_READ,
    EVFILT_SIGNAL,
    EVFILT_TIMER,
    EVFILT_USER,
    EVFILT_WRITE,
    EV_ADD,
    EV_CLEAR,
    EV_DELETE,
    EV_DISABLE,
    EV_DISPATCH,
    EV_ENABLE,
    EV_EOF,
    EV_ERROR,
    EV_ONESHOT,
    EV_RECEIPT,
    NOTE_ABSOLUTE,
    NOTE_EXIT,
    NOTE_FFAND,
    NOTE_FFCOPY,
    NOTE_FFCTRLMASK,
    NOTE_FFLAGSMASK,
    NOTE_FFNOP,
    NOTE_FFOR,
    NOTE_LOWAT,
    NOTE_NSECONDS,
    NOTE_SECONDS,
    NOTE_TRIGGER,
    NOTE_USECONDS,
    kevent_s,
)

comptime CABI_INCLUDES: List[StaticString] = ["<sys/event.h>"]
"""The headers that declare the C side."""

comptime CABI_STRUCTS = TypeList.of[Trait=AnyType, kevent_s]()
"""The struct mirrors to check."""

comptime CABI_TYPEDEFS = TypeList.of[Trait=AbiTypedefLike]()
"""The type aliases to check."""

comptime CABI_CONSTANTS = [
    AbiConstant("EVFILT_READ", EVFILT_READ),
    AbiConstant("EVFILT_WRITE", EVFILT_WRITE),
    AbiConstant("EVFILT_PROC", EVFILT_PROC),
    AbiConstant("EVFILT_SIGNAL", EVFILT_SIGNAL),
    AbiConstant("EVFILT_TIMER", EVFILT_TIMER),
    AbiConstant("EVFILT_USER", EVFILT_USER),
    AbiConstant("EV_ADD", EV_ADD),
    AbiConstant("EV_DELETE", EV_DELETE),
    AbiConstant("EV_ENABLE", EV_ENABLE),
    AbiConstant("EV_DISABLE", EV_DISABLE),
    AbiConstant("EV_ONESHOT", EV_ONESHOT),
    AbiConstant("EV_CLEAR", EV_CLEAR),
    AbiConstant("EV_RECEIPT", EV_RECEIPT),
    AbiConstant("EV_DISPATCH", EV_DISPATCH),
    AbiConstant("EV_ERROR", EV_ERROR),
    AbiConstant("EV_EOF", EV_EOF),
    AbiConstant("NOTE_TRIGGER", NOTE_TRIGGER),
    AbiConstant("NOTE_FFNOP", NOTE_FFNOP),
    AbiConstant("NOTE_FFAND", NOTE_FFAND),
    AbiConstant("NOTE_FFOR", NOTE_FFOR),
    AbiConstant("NOTE_FFCOPY", NOTE_FFCOPY),
    AbiConstant("NOTE_FFCTRLMASK", NOTE_FFCTRLMASK),
    AbiConstant("NOTE_FFLAGSMASK", NOTE_FFLAGSMASK),
    AbiConstant("NOTE_LOWAT", NOTE_LOWAT),
    AbiConstant("NOTE_EXIT", NOTE_EXIT),
    AbiConstant("NOTE_SECONDS", NOTE_SECONDS),
    AbiConstant("NOTE_USECONDS", NOTE_USECONDS),
    AbiConstant("NOTE_NSECONDS", NOTE_NSECONDS),
    AbiConstant("NOTE_ABSOLUTE", NOTE_ABSOLUTE),
]
"""The constants to check."""
