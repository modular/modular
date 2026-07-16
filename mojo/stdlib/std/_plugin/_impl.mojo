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
"""Selects the active `PluginHooks` used by the stdlib.

`CurrentPlugin` is resolved by the selector (`get_plugin_index`) from the set of
registered plugins (`PLUGINS`), keyed on the target's `stdlib_plugin` field.
The default build, whose field is `"default"`, resolves to `DefaultPlugin` and
leaves every hook at its default value.
"""

from ._trait import PluginHooks
from ._overlay import PLUGINS
from .selector import get_plugin_index
from std.sys.info import _TargetType

comptime CurrentPlugin: PluginHooks = PLUGINS._get_type_at_index[
    get_plugin_index()
]
"""The active `PluginHooks`."""

comptime PluginForTarget[Target: _TargetType] = PLUGINS._get_type_at_index[
    get_plugin_index[Target]()
]
"""The `PluginHooks` to use for the specified kgen.target."""
