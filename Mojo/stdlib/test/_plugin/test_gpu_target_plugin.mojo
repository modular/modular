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

from std.collections.string.string_span import _get_kgen_string
from std.sys.info import CompilationTarget
from std.sys.intrinsics import PrefetchOptions

from std.builtin.variadics import TypeList

from std._plugin import (
    DefaultPlugin,
    DefaultPluginHooks,
    PluginHooks,
    StdPlugin,
    TargetDeviceKind,
)
from std._plugin.metal import MetalPlugin, MetalPluginHooks
from std._plugin.cuda import CUDAPlugin, CUDAPluginHooks
from std._plugin.hip import HIPPlugin, HIPPluginHooks
from std._plugin.selector import PluginSelector

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

# Named only by their triple, as `--target-triple` builds them: no plugin payload.
comptime _bare_nvptx_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", arch = "sm_90"> : !kgen.target`
    ]
]()
comptime _bare_sm80_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", arch = "sm_80"> : !kgen.target`
    ]
]()
comptime _bare_amdgcn_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", arch = "gfx942"> : !kgen.target`
    ]
]()
comptime _bare_air_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "air64-apple-macosx", arch = "apple-m4"> : !kgen.target`
    ]
]()

# An NVIDIA triple whose payload names `DefaultPlugin`, as an out-of-tree plugin
# would name its own.
comptime _nvptx_default_payload_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", opaque_plugin = `,
        TypeList.of[DefaultPlugin]().values,
        `, arch = "sm_90"> : !kgen.target`,
    ]
]()

# An NVIDIA triple whose payload names no stdlib plugin at all, as a target
# that carries only another registry's plugin would.
comptime _nvptx_foreign_payload_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", opaque_plugin = `,
        TypeList.of[Int32]().values,
        `, arch = "sm_90"> : !kgen.target`,
    ]
]()


struct _GPUHooks[target: CompilationTarget](PluginHooks):
    comptime device_kind = TargetDeviceKind.gpu


struct _NPUHooks[target: CompilationTarget](PluginHooks):
    comptime device_kind = TargetDeviceKind.other("npu", True)


struct _FourthGPUPlugin(StdPlugin):
    """A GPU plugin for a vendor the stdlib doesn't name."""

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["fourth"]()
    comptime HookFor[target: CompilationTarget] = _GPUHooks[target]


# A triple the stdlib doesn't recognize as a GPU, carrying a GPU plugin.
comptime _fourth_gpu_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "x86_64-unknown-linux-gnu", opaque_plugin = `,
        TypeList.of[_FourthGPUPlugin]().values,
        `, arch = "x86-64"> : !kgen.target`,
    ]
]()


struct _NonGPUAcceleratorPlugin(StdPlugin):
    """A plugin for an accelerator that is not a SIMT GPU."""

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["npu"]()
    comptime HookFor[target: CompilationTarget] = _NPUHooks[target]


comptime _npu_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "x86_64-unknown-linux-gnu", opaque_plugin = `,
        TypeList.of[_NonGPUAcceleratorPlugin]().values,
        `, arch = "x86-64"> : !kgen.target`,
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
    comptime assert cuda.native_float8_cast[DType.float32, DType.float8_e4m3fn]
    comptime assert cuda.native_float8_cast[DType.float8_e5m2, DType.float16]
    comptime assert not cuda.native_float8_cast[
        DType.float8_e4m3fn, DType.float32
    ]
    comptime assert not CUDAPluginHooks[_a100_target].native_float8_cast[
        DType.float32, DType.float8_e4m3fn
    ]
    comptime assert hip.native_float8_cast[DType.float8_e4m3fnuz, DType.float32]
    comptime assert not hip.native_float8_cast[
        DType.float8_e4m3fn, DType.float32
    ]
    comptime assert cuda.has_native_float8
    comptime assert not CUDAPluginHooks[_a100_target].has_native_float8
    comptime assert hip.has_native_float8
    comptime metal = _metal_m1_target.stdlib_plugin
    comptime assert metal.unary_math_fn[.sqrt, DType.float32, 1]
    comptime assert not metal.unary_math_fn[.exp2, DType.bfloat16, 1]
    comptime assert not metal.unary_math_fn[.log, DType.float32, 1]
    comptime assert metal.binary_math_fn[.pow, DType.float32, 1]
    comptime assert not metal.has_native_float8

    comptime assert cuda.gpu_id_fn["thread_idx", "x"]
    comptime assert cuda.gpu_id_fn["cluster_idx", "x"]
    comptime assert not CUDAPluginHooks[_a100_target].gpu_id_fn[
        "cluster_idx", "x"
    ]
    comptime assert not hip.gpu_id_fn["cluster_dim", "y"]
    comptime assert not metal.gpu_id_fn["block_id_in_cluster", "z"]
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


def test_payloadless_target_gets_vendor_plugin() raises:
    # fmt: off
    comptime assert _bare_nvptx_target.stdlib_plugin == CUDAPluginHooks[_bare_nvptx_target]
    comptime assert _bare_amdgcn_target.stdlib_plugin == HIPPluginHooks[_bare_amdgcn_target]
    comptime assert _bare_air_target.stdlib_plugin == MetalPluginHooks[_bare_air_target]
    # fmt: on


def test_payload_wins_over_vendor_default() raises:
    # fmt: off
    comptime assert _nvptx_default_payload_target.stdlib_plugin == DefaultPluginHooks[_nvptx_default_payload_target]
    # fmt: on


def test_payload_without_stdlib_plugin_gets_vendor_default() raises:
    # fmt: off
    comptime assert _nvptx_foreign_payload_target.stdlib_plugin == CUDAPluginHooks[_nvptx_foreign_payload_target]
    # fmt: on


def test_vendor_default_specializes_on_target_arch() raises:
    # bfloat16 `exp2` has a native lowering only from SM90.
    comptime assert not _bare_sm80_target.stdlib_plugin.unary_math_fn[
        .exp2, DType.bfloat16, 1
    ]
    comptime assert _bare_nvptx_target.stdlib_plugin.unary_math_fn[
        .exp2, DType.bfloat16, 1
    ]


def test_registry_follows_stdlib_plugin() raises:
    comptime selector = PluginSelector[
        DefaultPlugin, MetalPlugin, CUDAPlugin, HIPPlugin
    ]
    # fmt: off
    comptime assert selector.for_target[_host_target._mlir_value] == DefaultPlugin
    comptime assert selector.for_target[_h100_target._mlir_value] == CUDAPlugin
    comptime assert selector.for_target[_bare_nvptx_target._mlir_value] == CUDAPlugin
    comptime assert selector.for_target[_bare_amdgcn_target._mlir_value] == HIPPlugin
    comptime assert selector.for_target[_bare_air_target._mlir_value] == MetalPlugin
    comptime assert selector.for_target[_nvptx_default_payload_target._mlir_value] == DefaultPlugin
    # fmt: on


def test_registry_falls_back_to_default() raises:
    comptime selector = PluginSelector[MetalPlugin, CUDAPlugin]
    # fmt: off
    comptime assert selector.for_target_or_default[_h100_target._mlir_value, HIPPlugin] == CUDAPlugin
    comptime assert selector.for_target_or_default[_bare_air_target._mlir_value, HIPPlugin] == MetalPlugin
    comptime assert selector.for_target_or_default[_host_target._mlir_value, HIPPlugin] == HIPPlugin
    comptime assert selector.for_target_or_default[_bare_amdgcn_target._mlir_value, DefaultPlugin] == DefaultPlugin
    # fmt: on


def test_plugin_classifies_target() raises:
    # The plugin decides, so a GPU the stdlib doesn't know by triple is still
    # a GPU, and a payload naming `DefaultPlugin` makes a GPU triple a plain
    # target.
    comptime assert _fourth_gpu_target.is_gpu()
    comptime assert _fourth_gpu_target.is_accelerator()
    comptime assert not _npu_target.is_gpu()
    comptime assert _npu_target.is_accelerator()
    comptime assert not _nvptx_default_payload_target.is_gpu()
    comptime assert not _nvptx_default_payload_target.is_accelerator()

    comptime assert not _host_target.is_gpu()
    comptime assert not _host_target.is_accelerator()
    comptime assert not CompilationTarget().is_gpu()

    comptime assert _h100_target.is_gpu()
    comptime assert _mi300x_target.is_gpu()
    comptime assert _metal_m1_target.is_gpu()
    comptime assert _h100_target.is_accelerator()
    comptime assert _mi300x_target.is_accelerator()
    comptime assert _metal_m1_target.is_accelerator()

    comptime assert _bare_nvptx_target.is_gpu()
    comptime assert _bare_amdgcn_target.is_gpu()
    comptime assert _bare_air_target.is_gpu()
    comptime assert _bare_air_target.is_accelerator()


def test_plugin_for_target_specializes_on_target() raises:
    assert_true(_h100_target.stdlib_plugin._handles_debug_assert)
    assert_true(_mi300x_target.stdlib_plugin._handles_debug_assert)
    assert_false(_host_target.stdlib_plugin._handles_debug_assert)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
