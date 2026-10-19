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

from std.testing import assert_equal, assert_false, assert_true, TestSuite

from std.sys.info import CompilationTarget
from std.sys.intrinsics import PrefetchOptions

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


def test_gpu_plugin_hooks() raises:
    comptime cuda = _h100_target.stdlib_plugin
    comptime hip = _mi300x_target.stdlib_plugin

    comptime assert cuda.unary_math_fn[.sqrt, DType.float32, 4]
    comptime assert not cuda.unary_math_fn[.sqrt, DType.float64, 1]
    comptime assert cuda.unary_math_fn[.log, DType.float32, 1]
    comptime assert not cuda.unary_math_fn[.log, DType.float16, 1]
    comptime assert not CUDAPluginHooks[_a100_target].unary_math_fn[
        .exp2, DType.bfloat16, 1
    ]
    comptime assert cuda.unary_math_fn[.exp2, DType.bfloat16, 2]
    comptime assert hip.unary_math_fn[.rsqrt, DType.bfloat16, 1]
    comptime assert not hip.unary_math_fn[.sqrt, DType.float32, 1]
    comptime assert not hip.unary_math_fn[.log2, DType.float64, 1]
    comptime metal = _metal_m1_target.stdlib_plugin
    comptime assert metal.unary_math_fn[.sqrt, DType.float32, 1]
    comptime assert not metal.unary_math_fn[.exp2, DType.bfloat16, 1]
    comptime assert not metal.unary_math_fn[.log, DType.float32, 1]
    comptime assert metal.binary_math_fn[.pow, DType.float32, 1]

    comptime assert cuda.gpu_id_fn["thread_idx", "x"]
    comptime assert hip.gpu_id_fn["lane_id", ""]
    comptime assert not _host_target.stdlib_plugin.gpu_id_fn["block_idx", "x"]

    comptime assert cuda.print_emit_fn
    comptime assert hip.print_emit_fn
    comptime assert cuda.printf_emit_fn["%d", Int32]
    comptime assert not _host_target.stdlib_plugin.printf_emit_fn["%d", Int32]

    comptime assert cuda.global_perf_counter_ns_fn
    comptime assert hip.global_perf_counter_ns_fn
    comptime assert cuda.sleep_fn
    comptime assert hip.sleep_fn
    comptime assert not _host_target.stdlib_plugin.sleep_fn

    comptime assert cuda.prefetch_fn[.GENERIC, PrefetchOptions()]
    comptime assert not hip.prefetch_fn[.GENERIC, PrefetchOptions()]
    comptime assert _metal_m1_target.stdlib_plugin.prefetch_fn[
        .GENERIC, PrefetchOptions()
    ]

    comptime assert cuda.atomic_float_minmax_via_int
    comptime assert not hip.atomic_float_minmax_via_int


def test_payload_plugin_selection() raises:
    # fmt: off
    comptime assert _host_target.stdlib_plugin == DefaultPluginHooks[_host_target]
    comptime assert _a100_target.stdlib_plugin == CUDAPluginHooks[_a100_target]
    comptime assert _h100_target.stdlib_plugin == CUDAPluginHooks[_h100_target]
    comptime assert _mi300x_target.stdlib_plugin == HIPPluginHooks[_mi300x_target]
    comptime assert _metal_m1_target.stdlib_plugin == MetalPluginHooks[_metal_m1_target]
    # fmt: on


def test_plugin_for_target_specializes_on_target() raises:
    assert_true(_h100_target.stdlib_plugin._handles_debug_assert)
    assert_true(_mi300x_target.stdlib_plugin._handles_debug_assert)
    assert_false(_host_target.stdlib_plugin._handles_debug_assert)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
