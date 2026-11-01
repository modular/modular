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

"""Apple GPU targets, each carrying the Metal stdlib and MAX plugins."""

from std.sys import CompilationTarget

from std._gpu.host.info import (
    AcceleratorArchitectureFamily,
    GPUInfo,
    TargetAccelerator,
    TargetAcceleratorCollection,
    TargetAcceleratorType,
    _K,
    _KB,
    _supported_archs_line,
)
from std._plugin.metal import MetalPlugin

from . import MetalMaxPlugin

comptime METAL_PLUGINS = TypeList.of[MetalPlugin, MetalMaxPlugin]().values
"""The `opaque_plugin` payload of every Apple target."""


struct MetalTargets(TargetAcceleratorCollection):
    """Apple GPU targets that the Metal plugins contribute."""

    comptime vendor_name: String = "Apple"

    # fmt: off
    comptime RAW_TARGETS = TypeList.of[
        Trait=TargetAcceleratorType,
        MetalM1      ._with_cli_values[["apple-m1"       ]],
        MetalM1Metal4._with_cli_values[["apple-m1-metal4"]],
        MetalM2      ._with_cli_values[["apple-m2"       ]],
        MetalM2Metal4._with_cli_values[["apple-m2-metal4"]],
        MetalM3      ._with_cli_values[["apple-m3"       ]],
        MetalM3Metal4._with_cli_values[["apple-m3-metal4"]],
        MetalM4      ._with_cli_values[["apple-m4"       ]],
        MetalM4Metal4._with_cli_values[["apple-m4-metal4"]],
        MetalM5      ._with_cli_values[["apple-m5"       ]],
        MetalM5Metal4._with_cli_values[["apple-m5-metal4"]],
        MetalM6      ._with_cli_values[["apple-m6"       ]],
        MetalM6Metal4._with_cli_values[["apple-m6-metal4"]],
    ]().values
    # fmt: on

    @staticmethod
    def normalize_target_arch(target_arch0: StaticString) -> String:
        # "metal:4" -> "apple-m4". Idempotent: a canonical "apple-m4" has no
        # "metal:" substring to rewrite.
        return target_arch0.replace("metal:", "apple-m")

    @staticmethod
    def supported_archs_message() -> String:
        return String(
            "  ",
            _supported_archs_line[Self](),
            "\n  See:",
            " https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf",
            "\n\n",
        )


# ===-----------------------------------------------------------------------===#
# Apple Architecture Families
# ===-----------------------------------------------------------------------===#

comptime AppleMetalFamily = AcceleratorArchitectureFamily(
    warp_size=32,
    threads_per_multiprocessor=32 * 32,
    shared_memory_per_multiprocessor=32 * _KB,
    max_registers_per_block=64 * _K,
    max_thread_block_size=_K,
)
"""Apple Metal GPU architecture family."""


# ===-----------------------------------------------------------------------===#
# Apple Silicon
# ===-----------------------------------------------------------------------===#


comptime _metal_m1_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "air64-apple-macosx", `,
        `stdlib_plugin = "metal", `,
        `opaque_plugin = `,
        METAL_PLUGINS,
        `,`,
        `arch = "apple-m1", `,
        `features = "+metal3_2,+air2_7_0", `,
        `data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for M1 Metal GPU."""


comptime _metal_m2_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "air64-apple-macosx", `,
        `stdlib_plugin = "metal", `,
        `opaque_plugin = `,
        METAL_PLUGINS,
        `,`,
        `arch = "apple-m2", `,
        `features = "+metal3_2,+air2_7_0", `,
        `data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for M2 Metal GPU."""


comptime _metal_m3_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "air64-apple-macosx", `,
        `stdlib_plugin = "metal", `,
        `opaque_plugin = `,
        METAL_PLUGINS,
        `,`,
        `arch = "apple-m3", `,
        `features = "+metal3_2,+air2_7_0", `,
        `data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for M3 Metal GPU."""


comptime _metal_m4_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "air64-apple-macosx", `,
        `stdlib_plugin = "metal", `,
        `opaque_plugin = `,
        METAL_PLUGINS,
        `,`,
        `arch = "apple-m4", `,
        `features = "+metal3_2,+air2_7_0", `,
        `data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for M4 Metal GPU."""


comptime _metal_m5_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "air64-apple-macosx", `,
        `stdlib_plugin = "metal", `,
        `opaque_plugin = `,
        METAL_PLUGINS,
        `,`,
        `arch = "apple-m5", `,
        `features = "+metal3_2,+air2_7_0", `,
        `data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for M5 Metal GPU."""


comptime _metal_m6_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "air64-apple-macosx", `,
        `stdlib_plugin = "metal", `,
        `opaque_plugin = `,
        METAL_PLUGINS,
        `,`,
        `arch = "apple-m6", `,
        `features = "+metal3_2,+air2_7_0", `,
        `data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for M6 Metal GPU."""


comptime _metal_m1_metal4_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "air64-apple-macosx", `,
        `stdlib_plugin = "metal", `,
        `opaque_plugin = `,
        METAL_PLUGINS,
        `,`,
        `arch = "apple-m1", `,
        `features = "+metal4_0,+air2_8_0", `,
        `data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for M1 Metal GPU with Metal 4.0."""


comptime _metal_m2_metal4_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "air64-apple-macosx", `,
        `stdlib_plugin = "metal", `,
        `opaque_plugin = `,
        METAL_PLUGINS,
        `,`,
        `arch = "apple-m2", `,
        `features = "+metal4_0,+air2_8_0", `,
        `data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for M2 Metal GPU with Metal 4.0."""


comptime _metal_m3_metal4_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "air64-apple-macosx", `,
        `stdlib_plugin = "metal", `,
        `opaque_plugin = `,
        METAL_PLUGINS,
        `,`,
        `arch = "apple-m3", `,
        `features = "+metal4_0,+air2_8_0", `,
        `data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for M3 Metal GPU with Metal 4.0."""


comptime _metal_m4_metal4_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "air64-apple-macosx", `,
        `stdlib_plugin = "metal", `,
        `opaque_plugin = `,
        METAL_PLUGINS,
        `,`,
        `arch = "apple-m4", `,
        `features = "+metal4_0,+air2_8_0", `,
        `data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for M4 Metal GPU with Metal 4.0."""


comptime _metal_m5_metal4_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "air64-apple-macosx", `,
        `stdlib_plugin = "metal", `,
        `opaque_plugin = `,
        METAL_PLUGINS,
        `,`,
        `arch = "apple-m5", `,
        `features = "+metal4_0,+air2_8_0", `,
        `data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for M5 Metal GPU with Metal 4.0."""


comptime _metal_m6_metal4_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "air64-apple-macosx", `,
        `stdlib_plugin = "metal", `,
        `opaque_plugin = `,
        METAL_PLUGINS,
        `,`,
        `arch = "apple-m6", `,
        `features = "+metal4_0,+air2_8_0", `,
        `data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for M6 Metal GPU with Metal 4.0."""


comptime MetalM1 = TargetAccelerator[
    GPUInfo.from_family(
        family=AppleMetalFamily,
        name="M1",
        api="metal",
        arch_name="apple-m1",
        compute=3.0,  # Metal version 3.0
        version="metal_3",
        sm_count=8,  # M1 has 8 GPU cores
    ),
    _metal_m1_target,
]()
"""Apple M1 GPU configuration."""

comptime MetalM2 = TargetAccelerator[
    GPUInfo.from_family(
        family=AppleMetalFamily,
        name="M2",
        api="metal",
        arch_name="apple-m2",
        compute=3.0,  # Metal version 3.0
        version="metal_3",
        sm_count=10,  # M2 has 10 GPU cores
    ),
    _metal_m2_target,
]()
"""Apple M2 GPU configuration."""

comptime MetalM3 = TargetAccelerator[
    GPUInfo.from_family(
        family=AppleMetalFamily,
        name="M3",
        api="metal",
        arch_name="apple-m3",
        compute=3.0,  # Metal version 3.0 for M3
        version="metal_3",
        sm_count=10,  # M3 has 10 GPU cores
    ),
    _metal_m3_target,
]()
"""Apple M3 GPU configuration."""

comptime MetalM4 = TargetAccelerator[
    GPUInfo.from_family(
        family=AppleMetalFamily,
        name="M4",
        api="metal",
        arch_name="apple-m4",
        compute=3.0,  # Metal version 3.0 for M4
        version="metal_3",
        sm_count=10,  # M4 has 10 GPU cores
    ),
    _metal_m4_target,
]()
"""Apple M4 GPU configuration."""

comptime MetalM5 = TargetAccelerator[
    GPUInfo.from_family(
        family=AppleMetalFamily,
        name="M5",
        api="metal",
        arch_name="apple-m5",
        compute=3.0,  # Metal version 3.0 for M5
        version="metal_3",
        sm_count=10,  # M5 has 10 GPU cores
    ),
    _metal_m5_target,
]()
"""Apple M5 GPU configuration."""

comptime MetalM6 = TargetAccelerator[
    GPUInfo.from_family(
        family=AppleMetalFamily,
        name="M6",
        api="metal",
        arch_name="apple-m6",
        compute=3.0,  # Metal version 3.0 for M6
        version="metal_3",
        sm_count=10,  # Base M5 count; unverified on M6
    ),
    _metal_m6_target,
]()
"""Apple M6 GPU configuration."""

comptime MetalM1Metal4 = TargetAccelerator[
    GPUInfo.from_family(
        family=AppleMetalFamily,
        name="M1 Metal4",
        api="metal",
        arch_name="apple-m1-metal4",
        compute=4.0,  # Metal 4.0, requires macOS 26
        version="metal_4",
        sm_count=8,  # M1 has 8 GPU cores
    ),
    _metal_m1_metal4_target,
]()
"""Apple M1 GPU configuration for Metal 4."""

comptime MetalM2Metal4 = TargetAccelerator[
    GPUInfo.from_family(
        family=AppleMetalFamily,
        name="M2 Metal4",
        api="metal",
        arch_name="apple-m2-metal4",
        compute=4.0,  # Metal 4.0, requires macOS 26
        version="metal_4",
        sm_count=10,  # M2 has 10 GPU cores
    ),
    _metal_m2_metal4_target,
]()
"""Apple M2 GPU configuration for Metal 4."""

comptime MetalM3Metal4 = TargetAccelerator[
    GPUInfo.from_family(
        family=AppleMetalFamily,
        name="M3 Metal4",
        api="metal",
        arch_name="apple-m3-metal4",
        compute=4.0,  # Metal 4.0, requires macOS 26
        version="metal_4",
        sm_count=10,  # M3 has 10 GPU cores
    ),
    _metal_m3_metal4_target,
]()
"""Apple M3 GPU configuration for Metal 4."""

comptime MetalM4Metal4 = TargetAccelerator[
    GPUInfo.from_family(
        family=AppleMetalFamily,
        name="M4 Metal4",
        api="metal",
        arch_name="apple-m4-metal4",
        compute=4.0,  # Metal 4.0, requires macOS 26
        version="metal_4",
        sm_count=10,  # M4 has 10 GPU cores
    ),
    _metal_m4_metal4_target,
]()
"""Apple M4 GPU configuration for Metal 4."""

comptime MetalM5Metal4 = TargetAccelerator[
    GPUInfo.from_family(
        family=AppleMetalFamily,
        name="M5 Metal4",
        api="metal",
        arch_name="apple-m5-metal4",
        compute=4.0,  # Metal 4.0, requires macOS 26
        version="metal_4",
        sm_count=10,  # M5 has 10 GPU cores
    ),
    _metal_m5_metal4_target,
]()
"""Apple M5 GPU configuration for Metal 4."""

comptime MetalM6Metal4 = TargetAccelerator[
    GPUInfo.from_family(
        family=AppleMetalFamily,
        name="M6 Metal4",
        api="metal",
        arch_name="apple-m6-metal4",
        compute=4.0,  # Metal 4.0, requires macOS 26
        version="metal_4",
        sm_count=10,  # Base M5 count; unverified on M6
    ),
    _metal_m6_metal4_target,
]()
"""Apple M6 GPU configuration for Metal 4."""
