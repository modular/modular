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
from ..tile_layout import TensorLayout


trait LayoutPluginHooks(Plugin):
    """Defines compile-time hooks for target-specific `layout` behavior."""

    comptime name: __mlir_type.`!kgen.string`
    """The stable plugin identifier used by the target selector."""

    comptime TensorEngineFor[
        L: TensorLayout, A: AddressSpace
    ]: TensorEngine = DefaultEngine[element_width=1]
    """The `TensorEngine` backing a tile of layout `L` in address space `A`.

    Some targets want one engine for every space; others need a different one
    per space -- an accelerator whose global memory is a memref but whose
    scratch is an ordinary pointer, say. A backend whose handle bakes in the
    tile's static extents reads them from `L`.
    """

    # TODO(GPUA-172): Delete once a backend's plugin defines what the built-in
    # address space names mean on it.
    comptime global_address_space: AddressSpace = AddressSpace.GENERIC
    """The address space a kernel sees device global memory in."""


struct DefaultLayoutPlugin(LayoutPluginHooks):
    """Provides the default `layout` behavior."""

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["default"]()
