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
"""Pluggable backend hooks for vendor-specific stdlib behavior.

This package defines the `PluginHooks` trait — a compile-time interface that
can be used to override selected stdlib dispatch sites without
forking base stdlib files.

`CurrentPlugin` is the plugin the current compilation target selects;
`std._plugin._impl` documents how it is chosen. `DefaultPlugin` leaves every
hook at its default value and keeps the built-in code paths. A plugin
repository can instead supply its own `std/_plugin/_impl.mojo` that aliases
`CurrentPlugin` to a specific struct. Hooks are `comptime` members that call
sites resolve at compile time, so a hook left at its default costs nothing at
runtime or in code size.
"""

from ._impl import CurrentPlugin
from ._trait import (
    StdPlugin,
    DefaultPlugin,
    DefaultPluginHooks,
    PluginHooks,
    TargetDeviceKind,
)
