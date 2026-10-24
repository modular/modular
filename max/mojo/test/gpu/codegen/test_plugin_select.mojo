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
"""Checks how a target's MAX plugin is selected: from the target's plugin
payload first, then by the name of its stdlib plugin."""

from std.builtin.variadics import TypeList
from std.collections import OptionalReg
from std.collections.string.string_span import _get_kgen_string
from std.compile import compile_info
from std.sys import llvm_intrinsic
from std.sys.info import CompilationTarget
from std.testing import TestSuite, assert_false, assert_true

from std._gpu.host.info import TargetAcceleratorCollection

from std._plugin import (
    PluginHooks,
    StdPlugin,
    TargetDeviceKind,
)

from max._plugin import (
    DefaultMaxPlugin,
    MaxPlugin,
    MaxPluginForTarget,
    MaxPluginHooks,
)
from max._plugin._trait import DefaultMaxPluginHooks
from max._plugin.cuda import CudaMaxPlugin, CudaMaxPluginHooks
from max._plugin.cuda.targets import CudaTargets
from max._plugin.hip import HipMaxPlugin, HipMaxPluginHooks
from max._plugin.hip.targets import HipTargets
from max._plugin.metal import MetalMaxPlugin, MetalMaxPluginHooks
from max._plugin.metal.targets import MetalTargets
from max.gpu.host import get_gpu_target
from max.gpu.sync import barrier
from max.gpu.sync.sync import _BarrierPluginHookFnType


@inline(.always)
def _marker_barrier():
    # Not what any vendor's `barrier_fn` emits, so its presence in the IR
    # shows this plugin was selected.
    llvm_intrinsic["llvm.nvvm.membar.gl", NoneType]()


struct PayloadMaxPlugin(MaxPlugin):
    """A MAX plugin that reaches a target only through its payload."""

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["payload"]()
    comptime HookFor[target: CompilationTarget] = PayloadMaxPluginHooks[target]


struct PayloadMaxPluginHooks[target: CompilationTarget](MaxPluginHooks):
    comptime warp_size = 32
    comptime barrier_fn: OptionalReg[_BarrierPluginHookFnType] = _marker_barrier


struct PayloadStdPlugin(StdPlugin):
    """A stdlib plugin whose name no MAX registry lists."""

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["vendor"]()
    comptime HookFor[target: CompilationTarget] = PayloadStdPluginHooks[target]


struct PayloadStdPluginHooks[target: CompilationTarget](PluginHooks):
    comptime device_kind = TargetDeviceKind.gpu


# Named only by its triple, as `--target-triple` builds it: no
# `stdlib_plugin` field and no plugin payload.
comptime _bare_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_80", `,
        `features = "+ptx81", `,
        `data_layout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128,`,
        `index_bit_width = 64`,
        `> : !kgen.target`,
    ]
]()

# The same triple, with a payload naming a MAX plugin but no stdlib plugin,
# as a target table outside the stdlib would write it.
comptime _payload_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `opaque_plugin = `,
        TypeList.of[PayloadMaxPlugin]().values,
        `, `,
        `arch = "sm_80", `,
        `features = "+ptx81", `,
        `data_layout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128,`,
        `index_bit_width = 64`,
        `> : !kgen.target`,
    ]
]()

# A payload carrying both plugins, as a target table shipped with a backend
# would write it: neither registry has to list the backend.
comptime _vendor_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `opaque_plugin = `,
        TypeList.of[PayloadStdPlugin, PayloadMaxPlugin]().values,
        `, `,
        `arch = "sm_80", `,
        `features = "+ptx81", `,
        `data_layout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128,`,
        `index_bit_width = 64`,
        `> : !kgen.target`,
    ]
]()

# A payload naming `DefaultMaxPlugin`, as an out-of-tree backend would name its
# own: the payload wins over the plugin the name list would pick.
comptime _default_payload_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", opaque_plugin = `,
        TypeList.of[DefaultMaxPlugin]().values,
        `, arch = "sm_80"> : !kgen.target`,
    ]
]()


# A payload naming only a stdlib plugin that no MAX registry lists, as an
# out-of-tree backend with no MAX plugin would write it.
comptime _std_only_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `opaque_plugin = `,
        TypeList.of[PayloadStdPlugin]().values,
        `, `,
        `arch = "sm_80", `,
        `features = "+ptx81", `,
        `data_layout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128,`,
        `index_bit_width = 64`,
        `> : !kgen.target`,
    ]
]()


def _compile_barrier_kernel[target: CompilationTarget]() raises -> String:
    # Nested so `__functions_in_module()` doesn't compile it for the host,
    # where `barrier()` is rejected.
    def kernel():
        barrier()

    return String(compile_info[kernel, emission_kind="llvm", target=target]())


def test_payload_names_the_plugin() raises:
    comptime assert (
        MaxPluginForTarget[_payload_target]
        == PayloadMaxPluginHooks[_payload_target]
    )
    var llvm = _compile_barrier_kernel[_payload_target]()
    assert_true("@llvm.nvvm.membar.gl()" in llvm, llvm)
    assert_false("@llvm.nvvm.barrier" in llvm, llvm)


def test_payload_plugins_need_no_registry_entry() raises:
    comptime assert (
        _vendor_target.stdlib_plugin == PayloadStdPluginHooks[_vendor_target]
    )
    comptime assert (
        MaxPluginForTarget[_vendor_target]
        == PayloadMaxPluginHooks[_vendor_target]
    )
    assert_true(
        "@llvm.nvvm.membar.gl()" in _compile_barrier_kernel[_vendor_target]()
    )


def test_payload_default_wins_over_name_list() raises:
    comptime assert (
        MaxPluginForTarget[_default_payload_target]
        == DefaultMaxPluginHooks[_default_payload_target]
    )


def test_builtin_targets_carry_their_plugin() raises:
    comptime nvidia = get_gpu_target["sm_80"]()
    comptime amd = get_gpu_target["mi300x"]()
    comptime apple = get_gpu_target["apple-m4"]()
    # The plugin comes from the payload, not from a registry entry.
    comptime assert (
        nvidia._first_conforming_plugin[MaxPlugin, default=DefaultMaxPlugin]
        == CudaMaxPlugin
    )
    comptime assert (
        amd._first_conforming_plugin[MaxPlugin, default=DefaultMaxPlugin]
        == HipMaxPlugin
    )
    comptime assert (
        apple._first_conforming_plugin[MaxPlugin, default=DefaultMaxPlugin]
        == MetalMaxPlugin
    )
    comptime assert MaxPluginForTarget[nvidia] == CudaMaxPluginHooks[nvidia]
    comptime assert MaxPluginForTarget[amd] == HipMaxPluginHooks[amd]
    comptime assert MaxPluginForTarget[apple] == MetalMaxPluginHooks[apple]
    var llvm = _compile_barrier_kernel[nvidia]()
    assert_true("@llvm.nvvm.barrier" in llvm, llvm)
    assert_false("@llvm.nvvm.membar.gl()" in llvm, llvm)


def _assert_warp_sizes_match_table[C: TargetAcceleratorCollection]():
    comptime for idx in range(C.TARGETS.length):
        comptime entry = C.TARGETS[idx]
        comptime target = CompilationTarget[_mlir_value=entry.mlir_target]()
        comptime assert (
            MaxPluginForTarget[target].warp_size == entry.gpu_info.warp_size
        ), "a plugin's warp_size disagrees with its target table"


def test_builtin_warp_sizes_match_target_tables() raises:
    _assert_warp_sizes_match_table[CudaTargets]()
    _assert_warp_sizes_match_table[HipTargets]()
    _assert_warp_sizes_match_table[MetalTargets]()


def test_unlisted_stdlib_plugin_gets_default() raises:
    comptime assert (
        MaxPluginForTarget[_std_only_target]
        == DefaultMaxPluginHooks[_std_only_target]
    )


def test_bare_triple_follows_vendor() raises:
    comptime assert (
        MaxPluginForTarget[_bare_target] == CudaMaxPluginHooks[_bare_target]
    )
    assert_true("@llvm.nvvm.barrier" in _compile_barrier_kernel[_bare_target]())
    comptime amd = CompilationTarget[
        _mlir_value=__mlir_attr[
            `#kgen.target<triple = "amdgcn-amd-amdhsa", arch = "gfx942"> : !kgen.target`
        ]
    ]()
    comptime apple = CompilationTarget[
        _mlir_value=__mlir_attr[
            `#kgen.target<triple = "air64-apple-macosx", arch = "apple-m4"> : !kgen.target`
        ]
    ]()
    comptime assert MaxPluginForTarget[amd] == HipMaxPluginHooks[amd]
    comptime assert MaxPluginForTarget[apple] == MetalMaxPluginHooks[apple]


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
