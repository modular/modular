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
"""macOS kqueue layouts, constants, and syscall bindings.

The layouts and values are transcribed from `<sys/event.h>` on macOS.
"""

from std.ffi import c_int, external_call
from std.sys import size_of
from std.sys.info import CompilationTarget
from std.testing._ffi import CField, CStruct

from .._libc import intptr_t, timespec, uintptr_t

# ===----------------------------------------------------------------------=== #
# Filters
# ===----------------------------------------------------------------------=== #

comptime EVFILT_READ: Int16 = -1
"""The descriptor has data to read."""
comptime EVFILT_WRITE: Int16 = -2
"""The descriptor can be written to."""
comptime EVFILT_PROC: Int16 = -5
"""The process `ident` did one of the `NOTE_*` actions in `fflags`."""
comptime EVFILT_SIGNAL: Int16 = -6
"""The signal `ident` was sent to the process, even if it's ignored."""
comptime EVFILT_TIMER: Int16 = -7
"""The timer `ident` fired."""
comptime EVFILT_USER: Int16 = -10
"""The user event `ident` was triggered with `NOTE_TRIGGER`."""

# ===----------------------------------------------------------------------=== #
# Action and status flags
# ===----------------------------------------------------------------------=== #

comptime EV_ADD: UInt16 = 0x0001
"""Add the event, or modify it if it's already added."""
comptime EV_DELETE: UInt16 = 0x0002
"""Remove the event."""
comptime EV_ENABLE: UInt16 = 0x0004
"""Let the event be returned."""
comptime EV_DISABLE: UInt16 = 0x0008
"""Keep the event from being returned, without removing it."""
comptime EV_ONESHOT: UInt16 = 0x0010
"""Remove the event after it's first returned."""
comptime EV_CLEAR: UInt16 = 0x0020
"""Reset the event's state after it's returned."""
comptime EV_RECEIPT: UInt16 = 0x0040
"""Report each change's result as an `EV_ERROR` event."""
comptime EV_DISPATCH: UInt16 = 0x0080
"""Disable the event after it's returned."""
comptime EV_ERROR: UInt16 = 0x4000
"""The event reports a change's result: `data` holds its `errno`, or 0."""
comptime EV_EOF: UInt16 = 0x8000
"""The filter hit end-of-file."""

# ===----------------------------------------------------------------------=== #
# Filter-specific flags
# ===----------------------------------------------------------------------=== #

comptime NOTE_TRIGGER: UInt32 = 0x01000000
"""Trigger an `EVFILT_USER` event."""
comptime NOTE_FFNOP: UInt32 = 0x00000000
"""Leave the `EVFILT_USER` event's user flags unchanged."""
comptime NOTE_FFAND: UInt32 = 0x40000000
"""AND the change's user flags into the `EVFILT_USER` event's."""
comptime NOTE_FFOR: UInt32 = 0x80000000
"""OR the change's user flags into the `EVFILT_USER` event's."""
comptime NOTE_FFCOPY: UInt32 = 0xC0000000
"""Copy the change's user flags to the `EVFILT_USER` event's."""
comptime NOTE_FFCTRLMASK: UInt32 = 0xC0000000
"""The bits of an `EVFILT_USER` change's `fflags` that pick its operation."""
comptime NOTE_FFLAGSMASK: UInt32 = 0x00FFFFFF
"""The bits of an `EVFILT_USER` event's `fflags` that hold its user flags."""
comptime NOTE_LOWAT: UInt32 = 0x00000001
"""Use a change's `data` as its `EVFILT_READ` or `EVFILT_WRITE` event's
low-water mark."""
comptime NOTE_EXIT: UInt32 = 0x80000000
"""Return an `EVFILT_PROC` event when the process exits."""
comptime NOTE_SECONDS: UInt32 = 0x00000001
"""Measure an `EVFILT_TIMER` event's `data` in seconds, not milliseconds."""
comptime NOTE_USECONDS: UInt32 = 0x00000002
"""Measure an `EVFILT_TIMER` event's `data` in microseconds."""
comptime NOTE_NSECONDS: UInt32 = 0x00000004
"""Measure an `EVFILT_TIMER` event's `data` in nanoseconds."""
comptime NOTE_ABSOLUTE: UInt32 = 0x00000008
"""Fire an `EVFILT_TIMER` event at the absolute time in `data`."""

# ===----------------------------------------------------------------------=== #
# Events
# ===----------------------------------------------------------------------=== #


# TODO(https://github.com/modular/modular/issues/7040): Remove this hack once
# we get support for packed types.
@align(4)
struct _Align4[dtype: DType](ImplicitlyCopyable, Writable):
    """A scalar held as bytes, so its alignment is 4 whatever its size."""

    var _bytes: Array[Byte, size_of[Scalar[Self.dtype]]()]

    def __init__(out self, value: Scalar[Self.dtype]):
        self._bytes = value.as_bytes()

    def __init__(out self, *, copy: Self):
        self._bytes = copy._bytes.copy()

    def get(self) -> Scalar[Self.dtype]:
        return .from_bytes(self._bytes)

    def write_to(self, mut writer: Some[Writer]):
        self.get().write_to(writer)

    def write_repr_to(self, mut writer: Some[Writer]):
        self.get().write_repr_to(writer)


# TODO(https://github.com/modular/modular/issues/7040): Declare `ident`, `data`,
# and `udata` as plain fields once Mojo can pack a struct.
# `<sys/event.h>` declares `struct kevent` under `#pragma pack(4)`,
# so its alignment is 4 rather than the 8 that its 8-byte fields would give
# it. `@align` can only raise a struct's alignment, so `_Align4` holds each of
# those fields as bytes and raises their alignment to 4.
@__annotation(CStruct(rename="kevent"))
struct kevent_s(ImplicitlyCopyable, Writable):
    """C `struct kevent`."""

    @__annotation(CField(rename="ident", c_type="uintptr_t"))
    var _ident: _Align4[uintptr_t.dtype]
    var filter: Int16
    var flags: UInt16
    var fflags: UInt32

    @__annotation(CField(rename="data", c_type="intptr_t"))
    var _data: _Align4[intptr_t.dtype]

    @__annotation(CField(rename="udata", c_type="void *"))
    var _udata: _Align4[uintptr_t.dtype]

    def __init__(
        out self,
        *,
        ident: uintptr_t,
        filter: Int16,
        flags: UInt16,
        fflags: UInt32 = 0,
        data: intptr_t = 0,
        udata: uintptr_t = 0,
    ):
        self._ident = _Align4(ident)
        self.filter = filter
        self.flags = flags
        self.fflags = fflags
        self._data = _Align4(data)
        self._udata = _Align4(udata)

    def ident(self) -> uintptr_t:
        return self._ident.get()

    def data(self) -> intptr_t:
        return self._data.get()

    def udata(self) -> uintptr_t:
        return self._udata.get()


# ===----------------------------------------------------------------------=== #
# Syscall bindings
# ===----------------------------------------------------------------------=== #


def kqueue() -> c_int:
    """C `kqueue(2)`."""
    return external_call["kqueue", c_int]()


def kevent(
    kq: c_int,
    changelist: Optional[ImmPointer[kevent_s, _]],
    nchanges: c_int,
    eventlist: Optional[MutPointer[kevent_s, _]],
    nevents: c_int,
    timeout: Optional[ImmPointer[timespec, _]],
) -> c_int:
    """C `kevent(2)`."""
    return external_call["kevent", c_int](
        kq, changelist, nchanges, eventlist, nevents, timeout
    )
