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

`CurrentPlugin` is selected from the current target's opaque plugin payload.
A GPU target without a payload gets its vendor's plugin, and any other target
without one gets `DefaultPlugin`, which leaves every hook at its default value.
"""

from std.sys.info import CompilationTarget

from ._trait import PluginHooks

comptime CurrentPlugin: PluginHooks = CompilationTarget.current().stdlib_plugin
"""The active `PluginHooks`."""
