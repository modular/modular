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
"""Python projection of HAL ``Buffer``."""

from std.memory import ArcPointer, UnsafePointer
from std.os import abort
from std.python import PythonObject
from _hal.buffer import Buffer as HALBuffer
from _hal.context import Context as HALContext
from _hal.device import get_device_spec


@fieldwise_init
struct Buffer(ImplicitlyDeletable, Movable, Writable):
    """Python projection of HAL ``Buffer``.

    Owns a device (or host-pinned) memory allocation plus a strong
    ``ArcPointer`` to the parent ``Context``. The destructor frees
    via ``free_sync`` or ``free_host_pinned`` based on ``_is_pinned``,
    so the Mojo HAL's leak-unless-freed semantics never reach Python.

    Holding the context Arc — rather than relying on Python ref order —
    guarantees that if the user drops the ``Context`` before the
    ``Buffer``, ``free_*`` still has a valid context handle to call.
    """

    # TODO: generalize to multi-device — currently hardcoded to device 0.
    comptime device_spec = get_device_spec[0]()
    var _hal: HALBuffer[Self.device_spec]
    var _ctx: ArcPointer[HALContext[Self.device_spec]]
    var _is_pinned: Bool

    def __del__(deinit self):
        # Mojo destructors must be non-raising; aborting on a free
        # failure is too aggressive (the resource is leaked but
        # nothing else has gone wrong).
        try:
            if self._is_pinned:
                self._ctx[].free_host_pinned(self._hal^)
            else:
                self._ctx[].free_sync(self._hal^)
        except e:
            print("warning: buffer free failed:", e)

    @staticmethod
    def _self_ptr(
        py_self: PythonObject,
    ) -> UnsafePointer[Self, MutAnyOrigin]:
        try:
            return py_self.downcast_value_ptr[Self]()
        except e:
            abort(String("Buffer method receiver was not a Buffer: ", e))

    @staticmethod
    def get_byte_size(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._self_ptr(py_self)
        return PythonObject(Int(self_ptr[]._hal.byte_size))

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Buffer(byte_size=", self._hal.byte_size, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)
