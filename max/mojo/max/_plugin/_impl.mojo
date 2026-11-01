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
"""Selects the active `MaxPluginHooks` for a target.

A target's `opaque_plugin` payload is consulted first: the first type in it
that conforms to `MaxPlugin` is the target's MAX plugin. The NVIDIA, AMD and
Apple target tables beside their plugins name both the stdlib and the MAX
plugin there, as any backend that ships its own targets does, so no registry
has to list them.

A target whose payload names no MAX plugin is matched by the `name` of its
stdlib plugin, as every registry built on `PluginSelector` is:

- First against the builtin MAX plugins. This covers a target named only by
  its triple (`--target-triple`, or a `#kgen.target` written by hand), whose
  stdlib plugin the stdlib picks by vendor, so MAX inherits that choice
  instead of repeating it.
- Then against `MAX_PLUGINS` from `_overlay.mojo`, which serves overlay
  plugins whose target tables live in stdlib packages and so cannot name MAX
  types. A name neither lists gets `DefaultMaxPlugin`.
"""

from ._overlay import MAX_PLUGINS
from ._trait import (
    DefaultMaxPlugin,
    DefaultMaxPluginHooks,
    MaxPlugin,
    MaxPluginHooks,
)
from .cuda import CudaMaxPlugin
from .hip import HipMaxPlugin
from .metal import MetalMaxPlugin

from std.collections.string.string_span import _get_kgen_string
from std.sys.info import CompilationTarget
from std._plugin.selector import PluginSelector


struct _NoMaxPlugin(MaxPlugin):
    """Stands for a payload that names no MAX plugin.

    `DefaultMaxPlugin` can't play this role: a payload that names it keeps it,
    as a payload that names `DefaultPlugin` does in the stdlib.
    """

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["<none>"]()
    comptime HookFor[target: CompilationTarget] = DefaultMaxPluginHooks[target]


comptime _BUILTIN_MAX_PLUGINS = PluginSelector[
    Trait=MaxPlugin, CudaMaxPlugin, HipMaxPlugin, MetalMaxPlugin
]

comptime _payload_plugin[target: CompilationTarget]: MaxPlugin = (
    target._first_conforming_plugin[MaxPlugin, default=_NoMaxPlugin]
)
"""The MAX plugin named in `target`'s payload, or `_NoMaxPlugin`."""

comptime _fallback_plugin[target: CompilationTarget]: MaxPlugin = (
    _BUILTIN_MAX_PLUGINS.for_target_or_default[
        target._mlir_value,
        MAX_PLUGINS.for_target_or_default[target._mlir_value, DefaultMaxPlugin],
    ]
)
"""The MAX plugin for a target whose payload names none."""

comptime _max_plugin[target: CompilationTarget]: MaxPlugin = (
    _fallback_plugin[target] if _payload_plugin[target]
    == _NoMaxPlugin else _payload_plugin[target]
)
"""The MAX plugin selected for `target`, before specialization."""

comptime MaxPluginForTarget[target: CompilationTarget] = _max_plugin[
    target
].HookFor[target]
"""The MAX plugin selected for the specified target."""

comptime CurrentMaxPlugin: MaxPluginHooks = MaxPluginForTarget[
    CompilationTarget.current()
]
"""The active MAX plugin."""
