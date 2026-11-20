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
"""Provides `ZeroedMmap`, a large zero-filled buffer that costs memory only
for the pages a test touches."""

from std.ffi import _null, c_size_t, get_errno
from std.io import IOError
from std.sys._libc import (
    MAP_ANONYMOUS,
    MAP_PRIVATE,
    PROT_READ,
    PROT_WRITE,
    mmap,
    munmap,
)


struct ZeroedMmap[*, mut: Bool = False]:
    """A zero-filled anonymous memory mapping, unmapped on destruction.

    The kernel backs a page of the mapping with memory only when something
    touches it, so a test can pass a buffer of more than `c_int.MAX` bytes to
    a syscall that touches few of its pages.
    """

    var _ptr: MutOpaquePointer[MutUntrackedOrigin]
    var _length: Int

    comptime _InteriorOrigin[origin: Origin] = origin._get_owned_interior[
        "bytes"
    ]

    def __init__(out self, length: Int) raises:
        """Maps `length` bytes of readable and writable zeroed memory.

        Args:
            length: The length of the mapping, in bytes.

        Raises:
            An error if `length` is negative or `mmap` fails.
        """
        var protection_flags = PROT_READ
        if Self.mut:
            protection_flags |= PROT_WRITE
        var ptr = mmap(
            _null[NoneType],
            length.try_cast[c_size_t.dtype](),
            protection_flags,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1,
            0,
        )
        if Int(ptr) == -1:
            raise IOError.last_os_error("mmap")
        self._ptr = ptr
        self._length = length

    def __deinit__(deinit self):
        """Unmaps the memory."""
        _ = munmap(self._ptr, c_size_t(self._length))

    # Without this, a call through a mutable `ref self` counts as a write to
    # the mapping and invalidates the spans already formed from it. This is
    # safe because forming a span changes neither the mapping nor its bytes.
    @__unsafe_nested_origins_read_only
    def as_bytes[
        origin: Origin[mut=Self.mut], //
    ](ref[origin] self) -> Span[Byte, Self._InteriorOrigin[origin]]:
        """Returns the mapped bytes.

        Parameters:
            origin: The origin of `self`.

        Returns:
            A span over the whole mapping, mutable if `mut` is `True`.
        """
        return Span(
            unsafe_ptr=Pointer(
                to=self._ptr.unsafe_bitcast[
                    Byte
                ]()._get_ref_with_unsafe_interior_origin["bytes", origin]()
            ),
            length=self._length,
        )
