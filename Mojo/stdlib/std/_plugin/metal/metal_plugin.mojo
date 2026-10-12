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

from std._plugin._trait import (
    PluginHooks,
    StdPlugin,
    _PrintEmitPluginHookFnType,
)
from std.collections import OptionalReg
from std.sys.info import CompilationTarget
from std.collections.string.string_span import _get_kgen_string
from std.ffi import CStringSpan
from std.io.file_descriptor import FileDescriptor
from std.sys._metal_print import _metal_print_write


def _print_emit[O: Origin](cstr: CStringSpan[O], file: FileDescriptor):
    """Writes `print()` output from a Metal kernel to the os_log stream.

    Metal kernels have a single os_log stream, so `file` is ignored.
    """
    _metal_print_write(StringSlice(unsafe_from_utf8=cstr.as_bytes_with_nul()))


struct MetalPlugin(StdPlugin):
    """`PluginHooks` implementation for Apple Metal backends."""

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["metal"]()
    comptime HookFor[target: CompilationTarget] = MetalPluginHooks[target]


struct MetalPluginHooks[target: CompilationTarget](PluginHooks):
    comptime print_emit_fn: OptionalReg[
        _PrintEmitPluginHookFnType
    ] = _print_emit
