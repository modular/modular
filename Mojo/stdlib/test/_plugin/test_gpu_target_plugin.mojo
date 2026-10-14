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

from std._plugin import PluginForTarget

from std.testing import assert_equal, assert_false, TestSuite

from std.sys.info import CompilationTarget

from std._plugin import DefaultPluginHooks
from std._plugin.metal import MetalPluginHooks
from std._plugin.cuda import CUDAPluginHooks
from std._plugin.hip import HIPPluginHooks

from max.gpu.host.info import (
    _a100_target,
    _h100_target,
    _metal_m1_target,
    _mi300x_target,
)


comptime _host_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "x86_64-unknown-linux-gnu", arch = "x86-64"> : !kgen.target`
    ]
]()


def test_payload_plugin_selection() raises:
    # fmt: off
    comptime assert _host_target.stdlib_plugin == DefaultPluginHooks[_host_target]
    comptime assert _a100_target.stdlib_plugin == CUDAPluginHooks[_a100_target]
    comptime assert _h100_target.stdlib_plugin == CUDAPluginHooks[_h100_target]
    comptime assert _mi300x_target.stdlib_plugin == HIPPluginHooks[_mi300x_target]
    comptime assert _metal_m1_target.stdlib_plugin == MetalPluginHooks[_metal_m1_target]
    # fmt: on


def test_plugin_for_target_specializes_on_target() raises:
    assert_false(
        PluginForTarget[_h100_target._mlir_value]._handles_debug_assert
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
