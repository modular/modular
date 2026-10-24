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
"""Selects the active HAL plugin hooks.

`CurrentHalPlugin` is resolved from the registered `HAL_PLUGINS`, keyed on the
target's `stdlib_plugin` field. The default build resolves to
`DefaultHalPlugin`.
"""

from ._trait import HalPluginHooks
from ._overlay import HAL_PLUGINS
from std.sys.info import CompilationTarget

comptime CurrentHalPlugin: HalPluginHooks = HAL_PLUGINS.current.HookFor[
    CompilationTarget.current()
]
"""The active HAL plugin."""
