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
"""Picks the stdlib plugin for a target that carries no plugin payload."""

from std.sys.info import CompilationTarget

from ._trait import DefaultPlugin, StdPlugin
from .cuda import CUDAPlugin
from .hip import HIPPlugin
from .metal import MetalPlugin

# mblack joins this chain onto one long line.
# fmt: off
comptime _default_std_plugin[target: CompilationTarget]: StdPlugin = (
    CUDAPlugin if target.is_nvidia_gpu()
    else HIPPlugin if target.is_amd_gpu()
    else MetalPlugin if target.is_apple_gpu()
    else DefaultPlugin
)
"""The plugin for a target with no `opaque_plugin` payload, chosen by its
GPU vendor.

Targets from the built-in target table carry their plugin, but a target named
only by its triple (`--target-triple`, or a `#kgen.target` written by hand)
does not, because C++ cannot name Mojo plugin types. Without this, such a GPU
target would get `DefaultPlugin` and none of its vendor's lowerings.
"""
# fmt: on
