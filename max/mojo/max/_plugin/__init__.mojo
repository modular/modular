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
"""Compile-time hooks for target-specific MAX behavior.

`CurrentMaxPlugin` is the `MaxPluginHooks` of the current compilation
target, and `MaxPluginForTarget[target]` that of any other. A target names
its MAX plugin in its `opaque_plugin` payload; see `_impl.mojo` for the
fallbacks when it doesn't.

Adding a plugin takes two structs: a `MaxPlugin` that names the backend, and
a `MaxPluginHooks` parameterized on the target that lowers the required
hooks (see the trait for which ones). Name the `MaxPlugin` in your targets'
`opaque_plugin` payload, next to your stdlib plugin, and no registry changes.
Each builtin vendor is a package here, `cuda`, `hip` and `metal`, whose
`__init__.mojo` holds the plugin and whose `targets.mojo` holds its target
table, built this way. The name-keyed list in `_overlay.mojo` is only for
overlay plugins whose target tables live in stdlib packages and so cannot
name MAX types.
"""

from ._impl import CurrentMaxPlugin, MaxPluginForTarget
from ._trait import DefaultMaxPlugin, MaxPlugin, MaxPluginHooks
