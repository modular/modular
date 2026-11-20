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
"""AMD GPU targets, each carrying the HIP stdlib and MAX plugins."""

from std.sys import CompilationTarget

from std._gpu.host.info import (
    AcceleratorArchitectureFamily,
    GPUInfo,
    TargetAccelerator,
    TargetAcceleratorCollection,
    TargetAcceleratorType,
    _K,
    _KB,
)
from std._plugin.hip import HIPPlugin

from . import HipMaxPlugin

comptime HIP_PLUGINS = TypeList.of[HIPPlugin, HipMaxPlugin]().values
"""The `opaque_plugin` payload of every AMD target."""


struct HipTargets(TargetAcceleratorCollection):
    """AMD GPU targets that the HIP plugins contribute."""

    comptime vendor_name: String = "AMD"

    # fmt: off
    comptime RAW_TARGETS = TypeList.of[
        Trait=TargetAcceleratorType,
        MI250X     ._with_cli_values[["gfx90a", "mi250x"]],
        MI300X     ._with_cli_values[["gfx942", "mi300x"]],
        # MI300A shares the gfx942 ISA with MI300X but has fewer CUs and
        # unified host/device memory. Reached via explicit "mi300a" opt-in
        # (e.g. `get_gpu_info["amdgpu:mi300a"]()`) since gfx942-only
        # detection cannot distinguish the two parts.
        MI300A     ._with_cli_values[["mi300a"]], # Could also include "gfx942", but `gfx942` resolves to MI300X.
        MI355X     ._with_cli_values[["gfx950", "mi355x"]],
        MI455X     ._with_cli_values[["gfx1250", "mi455x"]],
        Radeon6900 ._with_cli_values[["gfx1030"]],
        SteamDeck  ._with_cli_values[["gfx1033"]],
        Radeon7900 ._with_cli_values[["gfx1100"]],
        Radeon7800 ._with_cli_values[["gfx1101"]],
        Radeon7600 ._with_cli_values[["gfx1102"]],
        Radeon780m ._with_cli_values[["gfx1103"]],
        Radeon880m ._with_cli_values[["gfx1150"]],
        Radeon8060s._with_cli_values[["gfx1151"]],
        Radeon860m ._with_cli_values[["gfx1152"]],
        Radeon9060 ._with_cli_values[["gfx1200"]],
        Radeon9070 ._with_cli_values[["gfx1201"]],
    ]().values
    # fmt: on

    @staticmethod
    def normalize_target_arch(target_arch0: StaticString) -> String:
        # "amdgpu:gfx942" -> "gfx942", "amd:gfx942" -> "gfx942". Model names
        # such as "mi300x" are aliases in the table above. Both "amdgpu:"
        # (LLVM/ROCm target prefix) and "amd:" (vendor name) are accepted,
        # mirroring the "nvidia:" prefix that `CudaTargets` accepts.
        #
        # Keep normalization idempotent: applying it to an already normalized
        # target must not change it.
        return target_arch0.replace("amdgpu:", "").replace("amd:", "")

    @staticmethod
    def supported_archs_message() -> String:
        comptime amd_archs = (
            "gfx90a (MI250X), gfx942 (MI300X/MI300A), gfx950 (MI355X), gfx1030"
            " (Radeon 6900), gfx1033 (Van Gogh), gfx1100 (Radeon 7900), gfx1101"
            " (Radeon 7800), gfx1102 (Radeon 7600), gfx1103 (Radeon 780M),"
            " gfx1150/gfx1151/gfx1152 (Radeon 8xx), gfx1200 (Radeon 9060),"
            " gfx1201 (Radeon 9070), gfx1250 (MI455X)"
        )
        return String(
            "  AMD: ",
            amd_archs,
            (
                "\n  See:"
                " https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html"
                "\n\n"
            ),
        )


# ===----------------------------------------------------------------------=== #
# AMD Architecture Families
# ===----------------------------------------------------------------------=== #

comptime AMDCDNA2Family = AcceleratorArchitectureFamily(
    warp_size=64,
    threads_per_multiprocessor=64 * 32,
    shared_memory_per_multiprocessor=64 * _KB,
    max_registers_per_block=64 * _K,
    max_thread_block_size=_K,
)
"""AMD CDNA2 architecture family (gfx90a)."""

comptime AMDCDNA3Family = AcceleratorArchitectureFamily(
    warp_size=64,
    threads_per_multiprocessor=64 * 32,
    shared_memory_per_multiprocessor=64 * _KB,
    max_registers_per_block=64 * _K,
    max_thread_block_size=_K,
)
"""AMD CDNA3 architecture family (gfx94x)."""

comptime AMDCDNA4Family = AcceleratorArchitectureFamily(
    warp_size=64,
    threads_per_multiprocessor=64 * 32,
    shared_memory_per_multiprocessor=160 * _KB,
    max_registers_per_block=64 * _K,
    max_thread_block_size=_K,
)
"""AMD CDNA4 architecture family (gfx95x)."""

comptime AMDCDNA5Family = AcceleratorArchitectureFamily(
    warp_size=32,
    threads_per_multiprocessor=32 * 32,
    shared_memory_per_multiprocessor=320 * _KB,
    max_registers_per_block=32 * _K,
    max_thread_block_size=_K,
)
"""AMD CDNA5 architecture family (gfx125x)."""

comptime AMDRDNAFamily = AcceleratorArchitectureFamily(
    warp_size=32,
    threads_per_multiprocessor=32 * 32,
    shared_memory_per_multiprocessor=32 * _KB,
    max_registers_per_block=32 * _K,
    max_thread_block_size=_K,
)
"""AMD RDNA architecture family."""


# ===-----------------------------------------------------------------------===#
# MI250X
# ===-----------------------------------------------------------------------===#


comptime _mi250x_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "amdgpu9.0a-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `opaque_plugin = `,
        HIP_PLUGINS,
        `,`,
        `arch = "gfx90a", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-p10:32:32-p11:32:32-p12:32:32-p13:32:32-p14:32:32-p15:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for AMD MI250X GPU."""


comptime MI250X = TargetAccelerator[
    GPUInfo.from_family(
        family=AMDCDNA2Family,
        name="MI250X",
        api="hip",
        arch_name="gfx90a",
        compute=9.0,
        version="CDNA2",
        sm_count=220,
    ),
    _mi250x_target,
]()
"""AMD MI250X GPU configuration."""


# ===-----------------------------------------------------------------------===#
# MI300X
# ===-----------------------------------------------------------------------===#


comptime _mi300x_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "amdgpu9.42-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `opaque_plugin = `,
        HIP_PLUGINS,
        `,`,
        `arch = "gfx942", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-p10:32:32-p11:32:32-p12:32:32-p13:32:32-p14:32:32-p15:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for AMD MI300X GPU."""


comptime MI300X = TargetAccelerator[
    GPUInfo.from_family(
        family=AMDCDNA3Family,
        name="MI300X",
        api="hip",
        arch_name="gfx942",
        compute=9.4,
        version="CDNA3",
        sm_count=304,
    ),
    _mi300x_target,
]()
"""AMD MI300X GPU configuration."""


# ===-----------------------------------------------------------------------===#
# MI300A
# ===-----------------------------------------------------------------------===#


comptime _mi300a_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "amdgpu9.42-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `opaque_plugin = `,
        HIP_PLUGINS,
        `,`,
        `arch = "gfx942", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-p10:32:32-p11:32:32-p12:32:32-p13:32:32-p14:32:32-p15:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for AMD MI300A APU."""


comptime MI300A = TargetAccelerator[
    GPUInfo.from_family(
        family=AMDCDNA3Family,
        name="MI300A",
        api="hip",
        arch_name="gfx942",
        compute=9.4,
        version="CDNA3",
        sm_count=228,
    ),
    _mi300a_target,
]()
"""AMD MI300A APU configuration.

The MI300A is an Accelerated Processing Unit (APU) that integrates Zen 4 CPU
cores with CDNA 3 GPU compute units and unified HBM3 memory. It shares the
`gfx942` ISA with the MI300X but has fewer compute units (228 vs 304) and
unified host/device memory. Found in systems such as the CINES Adastra
supercomputer.
"""


# ===-----------------------------------------------------------------------===#
# MI355X
# ===-----------------------------------------------------------------------===#


comptime _mi355x_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "amdgpu9.50-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `opaque_plugin = `,
        HIP_PLUGINS,
        `,`,
        `arch = "gfx950", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-p10:32:32-p11:32:32-p12:32:32-p13:32:32-p14:32:32-p15:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for AMD MI355X GPU."""


comptime MI355X = TargetAccelerator[
    GPUInfo.from_family(
        family=AMDCDNA4Family,
        name="MI355X",
        api="hip",
        arch_name="gfx950",
        compute=9.5,
        version="CDNA4",
        sm_count=256,
    ),
    _mi355x_target,
]()
"""AMD MI355X GPU configuration."""


# ===-----------------------------------------------------------------------===#
# MI455X
# ===-----------------------------------------------------------------------===#


comptime _mi455x_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "amdgpu12.50-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `opaque_plugin = `,
        HIP_PLUGINS,
        `,`,
        `arch = "gfx1250", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-p10:32:32-p11:32:32-p12:32:32-p13:32:32-p14:32:32-p15:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for AMD MI455X GPU."""


comptime MI455X = TargetAccelerator[
    GPUInfo.from_family(
        family=AMDCDNA5Family,
        name="MI455X",
        api="hip",
        arch_name="gfx1250",
        compute=12.5,
        version="CDNA5",
        sm_count=256,
    ),
    _mi455x_target,
]()
"""AMD MI455X GPU configuration."""


# ===-----------------------------------------------------------------------===#
# Radeon 7xxx, 9xxx, 780m
# ===-----------------------------------------------------------------------===#


comptime _9070_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "amdgpu12.01-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `opaque_plugin = `,
        HIP_PLUGINS,
        `,`,
        `arch = "gfx1201", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-p10:32:32-p11:32:32-p12:32:32-p13:32:32-p14:32:32-p15:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for AMD Radeon 9070 GPU."""


comptime _9060_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "amdgpu12.00-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `opaque_plugin = `,
        HIP_PLUGINS,
        `,`,
        `arch = "gfx1200", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-p10:32:32-p11:32:32-p12:32:32-p13:32:32-p14:32:32-p15:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for AMD Radeon 9060 GPU."""


comptime _7900_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "amdgpu11.00-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `opaque_plugin = `,
        HIP_PLUGINS,
        `,`,
        `arch = "gfx1100", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-p10:32:32-p11:32:32-p12:32:32-p13:32:32-p14:32:32-p15:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for AMD Radeon 7900 GPU."""


comptime _7800_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "amdgpu11.01-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `opaque_plugin = `,
        HIP_PLUGINS,
        `,`,
        `arch = "gfx1101", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-p10:32:32-p11:32:32-p12:32:32-p13:32:32-p14:32:32-p15:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for AMD Radeon 7800/7700 GPU."""


comptime _7600_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "amdgpu11.02-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `opaque_plugin = `,
        HIP_PLUGINS,
        `,`,
        `arch = "gfx1102", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-p10:32:32-p11:32:32-p12:32:32-p13:32:32-p14:32:32-p15:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for AMD Radeon 7600 GPU."""


comptime _6900_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "amdgpu10.30-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `opaque_plugin = `,
        HIP_PLUGINS,
        `,`,
        `arch = "gfx1030", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-p10:32:32-p11:32:32-p12:32:32-p13:32:32-p14:32:32-p15:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for AMD Radeon 6900 GPU."""


comptime _780m_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "amdgpu11.03-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `opaque_plugin = `,
        HIP_PLUGINS,
        `,`,
        `arch = "gfx1103", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-p10:32:32-p11:32:32-p12:32:32-p13:32:32-p14:32:32-p15:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for AMD Radeon 780m GPU."""


comptime _880m_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "amdgpu11.50-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `opaque_plugin = `,
        HIP_PLUGINS,
        `,`,
        `arch = "gfx1150", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-p10:32:32-p11:32:32-p12:32:32-p13:32:32-p14:32:32-p15:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for AMD Radeon 880M GPU."""


comptime _8060s_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "amdgpu11.51-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `opaque_plugin = `,
        HIP_PLUGINS,
        `,`,
        `arch = "gfx1151", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-p10:32:32-p11:32:32-p12:32:32-p13:32:32-p14:32:32-p15:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for AMD Radeon 8060S GPU."""


comptime _860m_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "amdgpu11.52-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `opaque_plugin = `,
        HIP_PLUGINS,
        `,`,
        `arch = "gfx1152", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-p10:32:32-p11:32:32-p12:32:32-p13:32:32-p14:32:32-p15:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for AMD Radeon 860M GPU."""


comptime _steamdeck_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "amdgpu10.33-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `opaque_plugin = `,
        HIP_PLUGINS,
        `,`,
        `arch = "gfx1033", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-p10:32:32-p11:32:32-p12:32:32-p13:32:32-p14:32:32-p15:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for the Steam Deck's Van Gogh APU."""


comptime Radeon9070 = TargetAccelerator[
    GPUInfo.from_family(
        family=AMDRDNAFamily,
        name="Radeon 9070",
        api="hip",
        arch_name="gfx1201",
        compute=12.0,
        version="RDNA4",
        sm_count=64,
    ),
    _9070_target,
]()
"""AMD Radeon 9070 GPU configuration."""

comptime Radeon9060 = TargetAccelerator[
    GPUInfo.from_family(
        family=AMDRDNAFamily,
        name="Radeon 9060",
        api="hip",
        arch_name="gfx1200",
        compute=12.0,
        version="RDNA4",
        sm_count=32,
    ),
    _9060_target,
]()
"""AMD Radeon 9060 GPU configuration."""

comptime Radeon7900 = TargetAccelerator[
    GPUInfo.from_family(
        family=AMDRDNAFamily,
        name="Radeon 7900",
        api="hip",
        arch_name="gfx1100",
        compute=11.0,
        version="RDNA3",
        sm_count=96,
    ),
    _7900_target,
]()
"""AMD Radeon 7900 GPU configuration."""

comptime Radeon7800 = TargetAccelerator[
    GPUInfo.from_family(
        family=AMDRDNAFamily,
        name="Radeon 7800/7700",
        api="hip",
        arch_name="gfx1101",
        compute=11.0,
        version="RDNA3",
        sm_count=60,
    ),
    _7800_target,
]()
"""AMD Radeon 7800/7700 GPU configuration."""

comptime Radeon7600 = TargetAccelerator[
    GPUInfo.from_family(
        family=AMDRDNAFamily,
        name="Radeon 7600",
        api="hip",
        arch_name="gfx1102",
        compute=11.0,
        version="RDNA3",
        sm_count=32,
    ),
    _7600_target,
]()
"""AMD Radeon 7600 GPU configuration."""

comptime Radeon6900 = TargetAccelerator[
    GPUInfo.from_family(
        family=AMDRDNAFamily,
        name="Radeon 6900",
        api="hip",
        arch_name="gfx1030",
        compute=10.3,
        version="RDNA2",
        sm_count=60,
    ),
    _6900_target,
]()
"""AMD Radeon 6900 GPU configuration."""


comptime Radeon780m = TargetAccelerator[
    GPUInfo.from_family(
        family=AMDRDNAFamily,
        name="Radeon 780M",
        api="hip",
        arch_name="gfx1103",
        compute=11.0,
        version="RDNA3",
        sm_count=12,
    ),
    _780m_target,
]()
"""AMD Radeon 780M GPU configuration."""

comptime Radeon880m = TargetAccelerator[
    GPUInfo.from_family(
        family=AMDRDNAFamily,
        name="Radeon 880M",
        api="hip",
        arch_name="gfx1150",
        compute=11.5,
        version="RDNA3.5",
        sm_count=12,
    ),
    _880m_target,
]()
"""AMD Radeon 880M GPU configuration."""

comptime Radeon8060s = TargetAccelerator[
    GPUInfo.from_family(
        family=AMDRDNAFamily,
        name="Radeon 8060S",
        api="hip",
        arch_name="gfx1151",
        compute=11.5,
        version="RDNA3.5",
        sm_count=40,
    ),
    _8060s_target,
]()
"""AMD Radeon 8060S GPU configuration."""

comptime Radeon860m = TargetAccelerator[
    GPUInfo.from_family(
        family=AMDRDNAFamily,
        name="Radeon 860M",
        api="hip",
        arch_name="gfx1152",
        compute=11.5,
        version="RDNA3.5",
        sm_count=8,
    ),
    _860m_target,
]()
"""AMD Radeon 860M GPU configuration."""

comptime SteamDeck = TargetAccelerator[
    GPUInfo.from_family(
        family=AMDRDNAFamily,
        name="Steam Deck",
        api="hip",
        arch_name="gfx1033",
        compute=10.3,
        version="RDNA2",
        sm_count=8,
    ),
    _steamdeck_target,
]()
"""Steam Deck (Van Gogh) APU configuration."""
