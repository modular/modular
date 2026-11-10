# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Selects a target's HAL plugin hooks.

Every HAL type resolves its plugin through `HalPluginForTarget`, so this file
is the one place the selection rule lives. The plugin is matched from
`HAL_PLUGINS` by the `name` of the target's stdlib plugin, and a target that
no listed plugin names fails to compile.
"""

from ._trait import HalPluginHooks
from ._overlay import HAL_PLUGINS
from std.sys.info import CompilationTarget

comptime HalPluginForTarget[target: CompilationTarget]: HalPluginHooks = (
    HAL_PLUGINS.for_target[target._mlir_value].HookFor[target]
)
"""The HAL plugin hooks selected for `target`."""
