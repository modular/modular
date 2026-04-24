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

This file is intentionally tiny. A vendor plugin overlay ships its own
`std/_plugin/_impl.mojo` that re-points `CurrentPlugin` at a vendor-specific
struct, so the stdlib call sites that look up `CurrentPlugin` transparently
pick up the plugin implementation.

Do not define new logic here — put it on `PluginHooks` in `std._plugin`.
"""

from ._trait import DefaultPlugin

comptime CurrentPlugin = DefaultPlugin
"""The active `PluginHooks`."""
