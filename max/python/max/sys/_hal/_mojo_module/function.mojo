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
"""Python projection of HAL ``FunctionHandle``."""

from std.memory import ArcPointer, UnsafePointer
from std.os import abort
from std.python import PythonObject
from _hal.context import (
    Context as HALContext,
    RuntimeBundle as HALRuntimeBundle,
)
from _hal.device import get_device_spec
from _hal.plugin import FunctionHandle


@fieldwise_init
struct Function(ImplicitlyDeletable, Movable, Writable):
    """Python projection of HAL ``FunctionHandle``."""

    # TODO: generalize to multi-device — currently hardcoded to device 0.
    comptime device_spec = get_device_spec[0]()

    var _handle: FunctionHandle
    var _bundle: ArcPointer[HALRuntimeBundle]
    var _ctx: ArcPointer[HALContext[Self.device_spec]]

    def __del__(deinit self):
        # Mojo destructors must be non-raising; aborting on an unload
        # failure is too aggressive (the resource is leaked but
        # nothing else has gone wrong).
        try:
            self._ctx[].unload_function(self._handle)
        except e:
            print("warning: function unload failed:", e)

    @staticmethod
    def _self_ptr(
        py_self: PythonObject,
    ) -> UnsafePointer[Self, MutAnyOrigin]:
        try:
            return py_self.downcast_value_ptr[Self]()
        except e:
            abort(String("Function method receiver was not a Function: ", e))

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Function()")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("Function()")
