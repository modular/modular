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
"""Defines the compile-time hooks a backend implements to customize `layout`.
"""

from std._plugin.selector import Plugin
from std.collections.string.string_span import _get_kgen_string
from std.memory import AddressSpace

from ..tensor_engine import DefaultEngine, TensorEngine


trait LayoutPluginHooks(Plugin):
    """Defines compile-time hooks for target-specific `layout` behavior."""

    comptime name: __mlir_type.`!kgen.string`
    """The stable plugin identifier used by the target selector."""

    comptime TensorEngineFor[A: AddressSpace]: TensorEngine = DefaultEngine[
        element_width=1
    ]
    """The `TensorEngine` backing tiles in address space `A`.

    Some targets want one engine for every space; others need a different one
    per space -- an accelerator whose global memory is a memref but whose
    scratch is an ordinary pointer, say. Either way the backend and the space
    together pick out exactly one engine.
    """


struct DefaultLayoutPlugin(LayoutPluginHooks):
    """Provides the default `layout` behavior."""

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["default"]()
