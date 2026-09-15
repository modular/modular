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
"""Target descriptions for accelerators with builtin support in Mojo / MAX."""

from std.sys import CompilationTarget

from .info import (
    TargetAcceleratorCollection,
    AcceleratorArchitectureFamily,
    TargetAccelerator,
    TargetAcceleratorType,
    GPUInfo,
)


struct BuiltinTargets(TargetAcceleratorCollection):
    comptime vendor_name = "<builtin>"

    # fmt: off
    comptime RAW_TARGETS = TypeList.of[
        Trait=TargetAcceleratorType,
        # FIXME: GTX1060 and GTX1080Ti share `sm_61` but have different
        # `sm_count`; target resolution should differentiate them at compile time.
        TargetAccelerator[GTX1060   , _gtx1060_target    , []       ], # Could be ["sm_61"], but `sm_61` resolves to GTX1080Ti.
        TargetAccelerator[GTX1080Ti , _gtx1080ti_target  , ["sm_61"]],
        TargetAccelerator[TeslaP100 , _teslap100_target  , ["sm_60"]],
        TargetAccelerator[GTX970    , _gtx970_target     , ["sm_52"]],
        TargetAccelerator[RTX2060   , _rtx2060_target    , ["sm_75"]],
        # Note: `sm_86` is ambiguous; it resolves to A10 rather than RTX3090.
        TargetAccelerator[RTX3090   , _rtx3090_target    , []       ], # Could be ["sm_86"], but `sm_86` resolves to A10.
        TargetAccelerator[A10       , _a10_target        , ["sm_86"]],
        TargetAccelerator[A100      , _a100_target       , ["sm_80"]],
        TargetAccelerator[OrinNano  , _orin_nano_target  , ["sm_87"]],
        TargetAccelerator[L4        , _l4_target         , ["sm_89"]],
        TargetAccelerator[RTX4090m  , _rtx4090m_target   , []       ], # Could be ["sm_89"], but `sm_89` resolves to L4.
        TargetAccelerator[RTX4090   , _rtx4090_target    , []       ], # Could be ["sm_89"], but `sm_89` resolves to L4.
        # FIXME (KERN-1814): Unlike H100 and H200, blackwell devices (B100 vs B200)
        # architecture wise are different. We need to differentiate between them here.
        TargetAccelerator[B100      , _b100_target       , []                   ], # Could be ["sm_100", "sm_100a"], but both resolve to B200.
        TargetAccelerator[B200      , _b100_target       , ["sm_100", "sm_100a"]],
        TargetAccelerator[H100      , _h100_target       , ["sm_90" , "sm_90a" ]],
        TargetAccelerator[B300      , _b300_target       , ["sm_103", "sm_103a"]],
        TargetAccelerator[JetsonThor, _jetson_thor_target, ["sm_110", "sm_110a"]],
        TargetAccelerator[RTX5090   , _rtx5090_target    , ["sm_120", "sm_120a"]],
        TargetAccelerator[DGXSpark  , _dgx_spark_target  , ["sm_121", "sm_121a"]],
        #
        # Apple
        #
        TargetAccelerator[MetalM1      , _metal_m1_target       , ["apple-m1"       ]],
        TargetAccelerator[MetalM1Metal4, _metal_m1_metal4_target, ["apple-m1-metal4"]],
        TargetAccelerator[MetalM2      , _metal_m2_target       , ["apple-m2"       ]],
        TargetAccelerator[MetalM2Metal4, _metal_m2_metal4_target, ["apple-m2-metal4"]],
        TargetAccelerator[MetalM3      , _metal_m3_target       , ["apple-m3"       ]],
        TargetAccelerator[MetalM3Metal4, _metal_m3_metal4_target, ["apple-m3-metal4"]],
        TargetAccelerator[MetalM4      , _metal_m4_target       , ["apple-m4"       ]],
        TargetAccelerator[MetalM4Metal4, _metal_m4_metal4_target, ["apple-m4-metal4"]],
        TargetAccelerator[MetalM5      , _metal_m5_target       , ["apple-m5"       ]],
        TargetAccelerator[MetalM5Metal4, _metal_m5_metal4_target, ["apple-m5-metal4"]],
        #
        # AMD
        #
        TargetAccelerator[MI250X     , _mi250x_target  , ["gfx90a", "mi250x"]],
        TargetAccelerator[MI300X     , _mi300x_target  , ["gfx942", "mi300x"]],
        # MI300A shares the gfx942 ISA with MI300X but has fewer CUs and
        # unified host/device memory. Reached via explicit "mi300a" opt-in
        # (e.g. `GPUInfo.from_name["amdgpu:mi300a"]()`) since gfx942-only
        # detection cannot distinguish the two parts.
        TargetAccelerator[MI300A     , _mi300a_target   , ["mi300a"]], # Could also include "gfx942", but `gfx942` resolves to MI300X.
        TargetAccelerator[MI355X     , _mi355x_target   , ["gfx950", "mi355x"]],
        TargetAccelerator[Radeon6900 , _6900_target     , ["gfx1030"]],
        TargetAccelerator[SteamDeck  , _steamdeck_target, ["gfx1033"]],
        TargetAccelerator[Radeon7900 , _7900_target     , ["gfx1100"]],
        TargetAccelerator[Radeon7800 , _7800_target     , ["gfx1101"]],
        TargetAccelerator[Radeon7600 , _7600_target     , ["gfx1102"]],
        TargetAccelerator[Radeon780m , _780m_target     , ["gfx1103"]],
        TargetAccelerator[Radeon880m , _880m_target     , ["gfx1150"]],
        TargetAccelerator[Radeon8060s, _8060s_target    , ["gfx1151"]],
        TargetAccelerator[Radeon860m , _860m_target     , ["gfx1152"]],
        TargetAccelerator[Radeon9060 , _9060_target     , ["gfx1200"]],
        TargetAccelerator[Radeon9070 , _9070_target     , ["gfx1201"]],
    ]().values
    # fmt: on

    @staticmethod
    def normalize_target_arch(target_arch0: StaticString) -> String:
        # Normalize syntax-level target spellings before resolving table aliases.
        #
        # NVIDIA: "nvidia:sm_90a" -> "sm_90a", "nvidia:sm90" -> "sm_90",
        #         "nvidia:80" -> "sm_80", "sm80" -> "sm_80".
        # AMD: "amdgpu:gfx942" -> "gfx942", "amd:gfx942" -> "gfx942"
        #      (Model names such as "mi300x" are aliases in `BuiltinTargets`.)
        # Apple: "metal:4" -> "apple-m4".
        #
        # Keep normalization idempotent: applying it to an already normalized
        # target must not change it.
        #
        # These substring replacements must be ordered carefully, as a pattern
        # that is a prefix of an already canonical name corrupts it.
        return (
            target_arch0
            # NVIDIA normalization
            .replace("nvidia:sm_", "sm_")
            .replace("nvidia:sm", "sm_")
            .replace("nvidia:", "sm_")
            .replace("sm", "sm_")
            .replace("sm__", "sm_")
            # AMD normalization. Both "amdgpu:" (LLVM/ROCm target prefix) and
            # "amd:" (vendor name) are accepted, mirroring the "nvidia:" prefix
            # above.
            .replace("amdgpu:", "")
            .replace("amd:", "")
            # Apple normalization, general "metal:" → "apple-m" replacement.
            .replace("metal:", "apple-m")
        )


comptime _KB = 1024
comptime _K = 1024

# ===----------------------------------------------------------------------=== #
# Nvidia Architecture Families
# ===----------------------------------------------------------------------=== #

# NVIDIA Architecture Families
comptime NvidiaMaxwellFamily = AcceleratorArchitectureFamily(
    warp_size=32,
    threads_per_multiprocessor=64 * 32,
    shared_memory_per_multiprocessor=96 * _KB,
    max_registers_per_block=64 * _K,
    max_thread_block_size=_K,
)
"""NVIDIA Maxwell architecture family (sm_50-sm_53)."""

comptime NvidiaPascalFamily = AcceleratorArchitectureFamily(
    warp_size=32,
    threads_per_multiprocessor=64 * 32,
    shared_memory_per_multiprocessor=64 * _KB,
    max_registers_per_block=64 * _K,
    max_thread_block_size=_K,
)
"""NVIDIA Pascal architecture family (sm_60-sm_62)."""

comptime NvidiaTuringFamily = AcceleratorArchitectureFamily(
    warp_size=32,
    threads_per_multiprocessor=64 * 32,
    shared_memory_per_multiprocessor=64 * _KB,
    max_registers_per_block=32 * _K,
    max_thread_block_size=_K,
)
"""NVIDIA Turing architecture family (sm_75)."""

# Ampere architecture has three distinct variants based on compute capability:
# - sm_80: High-end datacenter (A100)
# - sm_86: Workstation/cloud (A10, RTX A-series)
# - sm_87: Embedded/edge (Jetson Orin)

comptime NvidiaAmpereDatacenterFamily = AcceleratorArchitectureFamily(
    warp_size=32,
    threads_per_multiprocessor=64 * 32,
    shared_memory_per_multiprocessor=164 * _KB,
    max_registers_per_block=64 * _K,
    max_thread_block_size=_K,
)
"""NVIDIA Ampere datacenter architecture family (sm_80)."""

comptime NvidiaAmpereWorkstationFamily = AcceleratorArchitectureFamily(
    warp_size=32,
    threads_per_multiprocessor=48 * 32,
    shared_memory_per_multiprocessor=100 * _KB,
    max_registers_per_block=64 * _K,
    max_thread_block_size=_K,
)
"""NVIDIA Ampere workstation architecture family (sm_86)."""

comptime NvidiaAmpereEmbeddedFamily = AcceleratorArchitectureFamily(
    warp_size=32,
    threads_per_multiprocessor=48 * 32,
    shared_memory_per_multiprocessor=164 * _KB,
    max_registers_per_block=64 * _K,
    max_thread_block_size=_K,
)
"""NVIDIA Ampere embedded architecture family (sm_87)."""

comptime NvidiaAdaFamily = AcceleratorArchitectureFamily(
    warp_size=32,
    threads_per_multiprocessor=48 * 32,
    shared_memory_per_multiprocessor=100 * _KB,
    max_registers_per_block=64 * _K,
    max_thread_block_size=_K,
)
"""NVIDIA Ada Lovelace architecture family (sm_89)."""

comptime NvidiaHopperFamily = AcceleratorArchitectureFamily(
    warp_size=32,
    threads_per_multiprocessor=64 * 32,
    shared_memory_per_multiprocessor=228 * _KB,
    max_registers_per_block=64 * _K,
    max_thread_block_size=_K,
)
"""NVIDIA Hopper architecture family (sm_90)."""

comptime NvidiaBlackwellFamily = AcceleratorArchitectureFamily(
    warp_size=32,
    threads_per_multiprocessor=64 * 32,
    shared_memory_per_multiprocessor=228 * _KB,
    max_registers_per_block=64 * _K,
    max_thread_block_size=_K,
)
"""NVIDIA Blackwell datacenter architecture family (sm_100)."""

comptime NvidiaBlackwellConsumerFamily = AcceleratorArchitectureFamily(
    warp_size=32,
    threads_per_multiprocessor=48 * 32,
    shared_memory_per_multiprocessor=100 * _KB,
    max_registers_per_block=64 * _K,
    max_thread_block_size=_K,
)
"""NVIDIA Blackwell consumer architecture family (sm_120)."""

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

comptime AMDRDNAFamily = AcceleratorArchitectureFamily(
    warp_size=32,
    threads_per_multiprocessor=32 * 32,
    shared_memory_per_multiprocessor=32 * _KB,
    max_registers_per_block=32 * _K,
    max_thread_block_size=_K,
)
"""AMD RDNA architecture family."""

# ===----------------------------------------------------------------------=== #
# Apple Architecture Families
# ===----------------------------------------------------------------------=== #

# Apple Architecture Families
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
        `arch = "apple-m5", `,
        `features = "+metal3_2,+air2_7_0", `,
        `data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for M5 Metal GPU."""


comptime _metal_m1_metal4_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "air64-apple-macosx", `,
        `stdlib_plugin = "metal", `,
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
        `arch = "apple-m5", `,
        `features = "+metal4_0,+air2_8_0", `,
        `data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for M5 Metal GPU with Metal 4.0."""


comptime MetalM1 = GPUInfo.from_family(
    family=AppleMetalFamily,
    name="M1",
    api="metal",
    arch_name="apple-m1",
    compute=3.0,  # Metal version 3.0
    version="metal_3",
    sm_count=8,  # M1 has 8 GPU cores
)
"""Apple M1 GPU configuration."""

comptime MetalM2 = GPUInfo.from_family(
    family=AppleMetalFamily,
    name="M2",
    api="metal",
    arch_name="apple-m2",
    compute=3.0,  # Metal version 3.0
    version="metal_3",
    sm_count=10,  # M2 has 10 GPU cores
)
"""Apple M2 GPU configuration."""

comptime MetalM3 = GPUInfo.from_family(
    family=AppleMetalFamily,
    name="M3",
    api="metal",
    arch_name="apple-m3",
    compute=3.0,  # Metal version 3.0 for M3
    version="metal_3",
    sm_count=10,  # M3 has 10 GPU cores
)
"""Apple M3 GPU configuration."""

comptime MetalM4 = GPUInfo.from_family(
    family=AppleMetalFamily,
    name="M4",
    api="metal",
    arch_name="apple-m4",
    compute=3.0,  # Metal version 3.0 for M4
    version="metal_3",
    sm_count=10,  # M4 has 10 GPU cores
)
"""Apple M4 GPU configuration."""

comptime MetalM5 = GPUInfo.from_family(
    family=AppleMetalFamily,
    name="M5",
    api="metal",
    arch_name="apple-m5",
    compute=3.0,  # Metal version 3.0 for M5
    version="metal_3",
    sm_count=10,  # M5 has 10 GPU cores
)
"""Apple M5 GPU configuration."""

comptime MetalM1Metal4 = GPUInfo.from_family(
    family=AppleMetalFamily,
    name="M1 Metal4",
    api="metal",
    arch_name="apple-m1-metal4",
    compute=4.0,  # Metal 4.0, requires macOS 26
    version="metal_4",
    sm_count=8,  # M1 has 8 GPU cores
)
"""Apple M1 GPU configuration for Metal 4."""

comptime MetalM2Metal4 = GPUInfo.from_family(
    family=AppleMetalFamily,
    name="M2 Metal4",
    api="metal",
    arch_name="apple-m2-metal4",
    compute=4.0,  # Metal 4.0, requires macOS 26
    version="metal_4",
    sm_count=10,  # M2 has 10 GPU cores
)
"""Apple M2 GPU configuration for Metal 4."""

comptime MetalM3Metal4 = GPUInfo.from_family(
    family=AppleMetalFamily,
    name="M3 Metal4",
    api="metal",
    arch_name="apple-m3-metal4",
    compute=4.0,  # Metal 4.0, requires macOS 26
    version="metal_4",
    sm_count=10,  # M3 has 10 GPU cores
)
"""Apple M3 GPU configuration for Metal 4."""

comptime MetalM4Metal4 = GPUInfo.from_family(
    family=AppleMetalFamily,
    name="M4 Metal4",
    api="metal",
    arch_name="apple-m4-metal4",
    compute=4.0,  # Metal 4.0, requires macOS 26
    version="metal_4",
    sm_count=10,  # M4 has 10 GPU cores
)
"""Apple M4 GPU configuration for Metal 4."""

comptime MetalM5Metal4 = GPUInfo.from_family(
    family=AppleMetalFamily,
    name="M5 Metal4",
    api="metal",
    arch_name="apple-m5-metal4",
    compute=4.0,  # Metal 4.0, requires macOS 26
    version="metal_4",
    sm_count=10,  # M5 has 10 GPU cores
)
"""Apple M5 GPU configuration for Metal 4."""

# ===-----------------------------------------------------------------------===#
# A100
# ===-----------------------------------------------------------------------===#

# Note: features = "+ptx81" means that the kernel should be compiled using
# PTX version 8.1. This must be less than or equal to the installed CUDA
# driver's maximum supported PTX version. Currently we hardcode this to
# PTX version 8.1 which means that you need to have a CUDA driver included with
# CUDA 12.5 toolkit. The mapping from CUDA Driver to PTX version can be found by
# looking at the PTX ISA in the versioned docs
# https://developer.nvidia.com/cuda-toolkit-archive.


comptime _a100_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `arch = "sm_80", `,
        `features = "+ptx81,+sm_80", `,
        `tune_cpu = "sm_80", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-p101:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for NVIDIA A100 GPU."""


comptime A100 = GPUInfo.from_family(
    family=NvidiaAmpereDatacenterFamily,
    name="A100",
    api="cuda",
    arch_name="ampere",
    compute=8.0,
    version="sm_80",
    sm_count=108,
)
"""NVIDIA A100 GPU configuration."""

# ===-----------------------------------------------------------------------===#
# A10
# ===-----------------------------------------------------------------------===#


comptime _a10_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `arch = "sm_86", `,
        `features = "+ptx81,+sm_86", `,
        `tune_cpu = "sm_86", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-p101:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for NVIDIA A10 GPU."""


comptime A10 = GPUInfo.from_family(
    family=NvidiaAmpereWorkstationFamily,
    name="A10",
    api="cuda",
    arch_name="ampere",
    compute=8.6,
    version="sm_86",
    sm_count=72,
)
"""NVIDIA A10 GPU configuration."""

# ===-----------------------------------------------------------------------===#
# Jetson Orin Nano
# ===-----------------------------------------------------------------------===#


comptime _orin_nano_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `arch = "sm_87", `,
        `features = "+ptx81,+sm_87", `,
        `tune_cpu = "sm_87", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-p101:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for NVIDIA Jetson Orin Nano GPU."""


comptime OrinNano = GPUInfo.from_family(
    family=NvidiaAmpereEmbeddedFamily,
    name="Orin Nano",
    api="cuda",
    arch_name="ampere",
    compute=8.7,
    version="sm_87",
    sm_count=8,
)
"""NVIDIA Orin Nano GPU configuration."""

# ===-----------------------------------------------------------------------===#
# Jetson Thor
# ===-----------------------------------------------------------------------===#


comptime _jetson_thor_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `arch = "sm_110", `,
        `features = "+ptx90,+sm_110", `,
        `tune_cpu = "sm_110", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-p101:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128,`,
        `index_bit_width = 64`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for NVIDIA Jetson Thor."""


comptime JetsonThor = GPUInfo.from_family(
    family=NvidiaBlackwellFamily,
    name="Jetson Thor",
    api="cuda",
    arch_name="blackwell",
    compute=11.0,
    version="sm_110",
    sm_count=20,
)
"""NVIDIA Jetson Thor GPU configuration."""

# ===-----------------------------------------------------------------------===#
# DGX Spark
# ===-----------------------------------------------------------------------===#


comptime _dgx_spark_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `arch = "sm_121a", `,
        `features = "+ptx88,+sm_121a", `,
        `tune_cpu = "sm_121a", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-p101:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for NVIDIA DGX Spark."""


comptime DGXSpark = GPUInfo.from_family(
    family=NvidiaBlackwellConsumerFamily,
    name="DGX Spark",
    api="cuda",
    arch_name="blackwell",
    compute=12.1,
    version="sm_121",
    sm_count=48,
)
"""NVIDIA DGX Spark GPU configuration."""

# ===-----------------------------------------------------------------------===#
# L4
# ===-----------------------------------------------------------------------===#


comptime _l4_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `arch = "sm_89", `,
        `features = "+ptx81,+sm_89", `,
        `tune_cpu = "sm_89", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-p101:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for NVIDIA L4 GPU."""


comptime L4 = GPUInfo.from_family(
    family=NvidiaAdaFamily,
    name="L4",
    api="cuda",
    arch_name="ada",
    compute=8.9,
    version="sm_89",
    sm_count=58,
)
"""NVIDIA L4 GPU configuration."""

# ===-----------------------------------------------------------------------===#
# RTX 4090 M
# ===-----------------------------------------------------------------------===#


comptime _rtx4090m_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `arch = "sm_89", `,
        `features = "+ptx81,+sm_89", `,
        `tune_cpu = "sm_90a", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-p101:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for NVIDIA RTX 4090 Mobile GPU."""


comptime RTX4090m = GPUInfo.from_family(
    family=NvidiaAdaFamily,
    name="RTX4090m",
    api="cuda",
    arch_name="ada lovelace",
    compute=8.9,
    version="sm_89",
    sm_count=76,
)
"""NVIDIA RTX 4090 Mobile GPU configuration."""

# ===-----------------------------------------------------------------------===#
# RTX 4090
# ===-----------------------------------------------------------------------===#


comptime _rtx4090_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `arch = "sm_89", `,
        `features = "+ptx81,+sm_89", `,
        `tune_cpu = "sm_90a", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-p101:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for NVIDIA RTX 4090."""


comptime RTX4090 = GPUInfo.from_family(
    family=NvidiaAdaFamily,
    name="RTX4090",
    api="cuda",
    arch_name="ada lovelace",
    compute=8.9,
    version="sm_89",
    sm_count=128,
)
"""NVIDIA RTX 4090 GPU configuration."""


# ===-----------------------------------------------------------------------===#
# H100
# ===-----------------------------------------------------------------------===#


comptime _h100_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `arch = "sm_90a", `,
        `features = "+ptx85,+sm_90a", `,
        `tune_cpu = "sm_90a", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-p101:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for NVIDIA H100 GPU."""


# https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper
comptime H100 = GPUInfo.from_family(
    family=NvidiaHopperFamily,
    name="H100",
    api="cuda",
    arch_name="hopper",
    compute=9.0,
    version="sm_90a",
    sm_count=132,
)
"""NVIDIA H100 GPU configuration."""

# ===-----------------------------------------------------------------------===#
# B100
# ===-----------------------------------------------------------------------===#


comptime _b100_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `arch = "sm_100a", `,
        `features = "+ptx88,+sm_100a", `,
        `tune_cpu = "sm_100a", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-p101:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for NVIDIA B100 GPU."""


# https://resources.nvidia.com/en-us-blackwell-architecture
# TODO: Update once we have B100 access.
comptime B100 = GPUInfo.from_family(
    family=NvidiaBlackwellFamily,
    name="B100",
    api="cuda",
    arch_name="blackwell",
    compute=10.0,
    version="sm_100a",
    sm_count=132,
)
"""NVIDIA B100 GPU configuration."""

comptime B200 = GPUInfo.from_family(
    family=NvidiaBlackwellFamily,
    name="B200",
    api="cuda",
    arch_name="blackwell",
    compute=10.0,
    version="sm_100a",
    sm_count=148,
)
"""NVIDIA B200 GPU configuration."""

# ===-----------------------------------------------------------------------===#
# B300
# ===-----------------------------------------------------------------------===#


comptime _b300_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `arch = "sm_103a", `,
        `features = "+ptx88,+sm_103a", `,
        `tune_cpu = "sm_103a", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-p101:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for NVIDIA B300 GPU."""


comptime B300 = GPUInfo.from_family(
    family=NvidiaBlackwellFamily,
    name="B300",
    api="cuda",
    arch_name="blackwell",
    compute=10.3,
    version="sm_103a",
    sm_count=160,
)
"""NVIDIA B300 GPU configuration."""


def _is_sm10x_gpu(info: GPUInfo) -> Bool:
    """Returns True for any Blackwell datacenter GPU (B100, B200, B300).

    Use this to check if the GPU supports SM100-class features. For
    architecture-specific tuning, compare against individual GPUInfo
    constants (e.g., `ctx.default_device_info == B300`).

    Args:
        info: GPU info to check.

    Returns:
        True if the GPU is a Blackwell datacenter GPU.
    """
    return (
        info == materialize[B100]()
        or info == materialize[B200]()
        or info == materialize[B300]()
    )


def _is_sm12x_gpu(info: GPUInfo) -> Bool:
    """Returns True for any Blackwell consumer GPU (sm_120 / sm_121).

    Covers the RTX 50-series / RTX PRO (sm_120) and GB10 / DGX Spark (sm_121),
    which have no SM100 warp-specialized path and route block-scaled / NVFP4
    work to the cuBLASLt vendor kernels. Mirrors `_is_sm10x_gpu`.

    Args:
        info: GPU info to check.

    Returns:
        True if the GPU is a Blackwell consumer (sm_12x) GPU.
    """
    return info.compute >= 12.0 and info.compute < 13.0


# ===-----------------------------------------------------------------------===#
# RTX5090
# ===-----------------------------------------------------------------------===#


comptime _rtx5090_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `arch = "sm_120a", `,
        `features = "+ptx87,+sm_120a", `,
        `tune_cpu = "sm_120a", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-p101:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for NVIDIA RTX5090 GPU."""


# https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/
comptime RTX5090 = GPUInfo.from_family(
    family=NvidiaBlackwellConsumerFamily,
    name="RTX5090",
    api="cuda",
    arch_name="blackwell",
    compute=12.0,
    version="sm_120a",
    sm_count=170,
)
"""NVIDIA RTX 5090 GPU configuration."""


# ===-----------------------------------------------------------------------===#
# RTX3090
# ===-----------------------------------------------------------------------===#


comptime _rtx3090_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `arch = "sm_86", `,
        `features = "+ptx63,+sm_86", `,
        `tune_cpu = "sm_86", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-p101:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for NVIDIA GeForce RTX 3090."""


# https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/
comptime RTX3090 = GPUInfo.from_family(
    family=NvidiaAmpereWorkstationFamily,
    name="NVIDIA GeForce RTX 3090",
    api="cuda",
    arch_name="ampere",
    compute=8.6,
    version="sm_86",
    sm_count=82,
)
"""NVIDIA GeForce RTX 3090 GPU configuration."""


# ===-----------------------------------------------------------------------===#
# GTX1080Ti
# ===-----------------------------------------------------------------------===#


comptime _gtx1080ti_target = CompilationTarget[
    _mlir_value=
    # Note: GTX 1080 Ti doesn't specify tune_cpu, data_layout, or index_bit_width
    __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `arch = "sm_61", `,
        `features = "+ptx50,+sm_61", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for NVIDIA GTX 1080 Ti GPU."""


comptime GTX1080Ti = GPUInfo.from_family(
    family=NvidiaPascalFamily,
    name="NVIDIA GeForce GTX 1080 Ti",
    api="cuda",
    arch_name="pascal",
    compute=6.1,
    version="sm_61",
    sm_count=28,
)
"""NVIDIA GeForce GTX 1080 Ti GPU configuration."""


# ===-----------------------------------------------------------------------===#
# GTX1060
# ===-----------------------------------------------------------------------===#


comptime _gtx1060_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `arch = "sm_61", `,
        `features = "+ptx50,+sm_61", `,
        `tune_cpu = "sm_61", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for NVIDIA GTX 1060 GPU."""


comptime GTX1060 = GPUInfo.from_family(
    family=NvidiaPascalFamily,
    name="NVIDIA GeForce GTX 1060",
    api="cuda",
    arch_name="pascal",
    compute=6.1,
    version="sm_61",
    sm_count=10,
)
"""NVIDIA GeForce GTX 1060 GPU configuration."""


# ===-----------------------------------------------------------------------===#
# GTX970
# ===-----------------------------------------------------------------------===#


comptime _gtx970_target = CompilationTarget[
    _mlir_value=
    # Note: GTX 970 doesn't specify tune_cpu, data_layout, or index_bit_width
    __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `arch = "sm_52", `,
        `features = "+ptx50,+sm_52", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for NVIDIA GTX 970 GPU."""


comptime GTX970 = GPUInfo.from_family(
    family=NvidiaMaxwellFamily,
    name="NVIDIA GeForce GTX 970",
    api="cuda",
    arch_name="maxwell",
    compute=5.2,
    version="sm_52",
    sm_count=13,
)
"""NVIDIA GeForce GTX 970 GPU configuration."""


# ===-----------------------------------------------------------------------===#
# Tesla P100
# ===-----------------------------------------------------------------------===#


comptime _teslap100_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `arch = "sm_60", `,
        `features = "+ptx50,+sm_60", `,
        `tune_cpu = "sm_60", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-p101:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for NVIDIA Tesla P100 GPU."""


comptime TeslaP100 = GPUInfo.from_family(
    family=NvidiaPascalFamily,
    name="NVIDIA Tesla P100",
    api="cuda",
    arch_name="pascal",
    compute=6.0,
    version="sm_60",
    sm_count=56,
)
"""NVIDIA Tesla P100 GPU configuration."""


# ===-----------------------------------------------------------------------===#
# RTX2060
# ===-----------------------------------------------------------------------===#


comptime _rtx2060_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `arch = "sm_75", `,
        `features = "+ptx63,+sm_75", `,
        `tune_cpu = "sm_75", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-p101:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for NVIDIA RTX 2060 GPU."""


comptime RTX2060 = GPUInfo.from_family(
    family=NvidiaTuringFamily,
    name="RTX2060",
    api="cuda",
    arch_name="turing",
    compute=7.5,
    version="sm_75",
    sm_count=30,
)
"""NVIDIA RTX 2060 GPU configuration."""


# ===-----------------------------------------------------------------------===#
# MI250X
# ===-----------------------------------------------------------------------===#


comptime _mi250x_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `arch = "gfx90a", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for AMD MI250X GPU."""


comptime MI250X = GPUInfo.from_family(
    family=AMDCDNA2Family,
    name="MI250X",
    api="hip",
    arch_name="gfx90a",
    compute=9.0,
    version="CDNA2",
    sm_count=220,
)
"""AMD MI250X GPU configuration."""


# ===-----------------------------------------------------------------------===#
# MI300X
# ===-----------------------------------------------------------------------===#


comptime _mi300x_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `arch = "gfx942", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for AMD MI300X GPU."""


comptime MI300X = GPUInfo.from_family(
    family=AMDCDNA3Family,
    name="MI300X",
    api="hip",
    arch_name="gfx942",
    compute=9.4,
    version="CDNA3",
    sm_count=304,
)
"""AMD MI300X GPU configuration."""


# ===-----------------------------------------------------------------------===#
# MI300A
# ===-----------------------------------------------------------------------===#


comptime _mi300a_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `arch = "gfx942", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for AMD MI300A APU."""


comptime MI300A = GPUInfo.from_family(
    family=AMDCDNA3Family,
    name="MI300A",
    api="hip",
    arch_name="gfx942",
    compute=9.4,
    version="CDNA3",
    sm_count=228,
)
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
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `arch = "gfx950", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for AMD MI355X GPU."""


comptime MI355X = GPUInfo.from_family(
    family=AMDCDNA4Family,
    name="MI355X",
    api="hip",
    arch_name="gfx950",
    compute=9.5,
    version="CDNA4",
    sm_count=256,
)
"""AMD MI355X GPU configuration."""


# ===-----------------------------------------------------------------------===#
# Radeon 7xxx, 9xxx, 780m
# ===-----------------------------------------------------------------------===#


comptime _9070_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `arch = "gfx1201", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for AMD Radeon 9070 GPU."""


comptime _9060_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `arch = "gfx1200", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for AMD Radeon 9060 GPU."""


comptime _7900_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `arch = "gfx1100", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for AMD Radeon 7900 GPU."""


comptime _7800_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `arch = "gfx1101", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for AMD Radeon 7800/7700 GPU."""


comptime _7600_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `arch = "gfx1102", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for AMD Radeon 7600 GPU."""


comptime _6900_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `arch = "gfx1030", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for AMD Radeon 6900 GPU."""


comptime _780m_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `arch = "gfx1103", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for AMD Radeon 780m GPU."""


comptime _880m_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `arch = "gfx1150", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for AMD Radeon 880M GPU."""


comptime _8060s_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `arch = "gfx1151", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for AMD Radeon 8060S GPU."""


comptime _860m_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `arch = "gfx1152", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for AMD Radeon 860M GPU."""


comptime _steamdeck_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `stdlib_plugin = "hip", `,
        `arch = "gfx1033", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for the Steam Deck's Van Gogh APU."""


comptime Radeon9070 = GPUInfo.from_family(
    family=AMDRDNAFamily,
    name="Radeon 9070",
    api="hip",
    arch_name="gfx1201",
    compute=12.0,
    version="RDNA4",
    sm_count=64,
)
"""AMD Radeon 9070 GPU configuration."""

comptime Radeon9060 = GPUInfo.from_family(
    family=AMDRDNAFamily,
    name="Radeon 9060",
    api="hip",
    arch_name="gfx1200",
    compute=12.0,
    version="RDNA4",
    sm_count=32,
)
"""AMD Radeon 9060 GPU configuration."""

comptime Radeon7900 = GPUInfo.from_family(
    family=AMDRDNAFamily,
    name="Radeon 7900",
    api="hip",
    arch_name="gfx1100",
    compute=11.0,
    version="RDNA3",
    sm_count=96,
)
"""AMD Radeon 7900 GPU configuration."""

comptime Radeon7800 = GPUInfo.from_family(
    family=AMDRDNAFamily,
    name="Radeon 7800/7700",
    api="hip",
    arch_name="gfx1101",
    compute=11.0,
    version="RDNA3",
    sm_count=60,
)
"""AMD Radeon 7800/7700 GPU configuration."""

comptime Radeon7600 = GPUInfo.from_family(
    family=AMDRDNAFamily,
    name="Radeon 7600",
    api="hip",
    arch_name="gfx1102",
    compute=11.0,
    version="RDNA3",
    sm_count=32,
)
"""AMD Radeon 7600 GPU configuration."""

comptime Radeon6900 = GPUInfo.from_family(
    family=AMDRDNAFamily,
    name="Radeon 6900",
    api="hip",
    arch_name="gfx1030",
    compute=10.3,
    version="RDNA2",
    sm_count=60,
)
"""AMD Radeon 6900 GPU configuration."""


comptime Radeon780m = GPUInfo.from_family(
    family=AMDRDNAFamily,
    name="Radeon 780M",
    api="hip",
    arch_name="gfx1103",
    compute=11.0,
    version="RDNA3",
    sm_count=12,
)
"""AMD Radeon 780M GPU configuration."""

comptime Radeon880m = GPUInfo.from_family(
    family=AMDRDNAFamily,
    name="Radeon 880M",
    api="hip",
    arch_name="gfx1150",
    compute=11.5,
    version="RDNA3.5",
    sm_count=12,
)
"""AMD Radeon 880M GPU configuration."""

comptime Radeon8060s = GPUInfo.from_family(
    family=AMDRDNAFamily,
    name="Radeon 8060S",
    api="hip",
    arch_name="gfx1151",
    compute=11.5,
    version="RDNA3.5",
    sm_count=40,
)
"""AMD Radeon 8060S GPU configuration."""

comptime Radeon860m = GPUInfo.from_family(
    family=AMDRDNAFamily,
    name="Radeon 860M",
    api="hip",
    arch_name="gfx1152",
    compute=11.5,
    version="RDNA3.5",
    sm_count=8,
)
"""AMD Radeon 860M GPU configuration."""

comptime SteamDeck = GPUInfo.from_family(
    family=AMDRDNAFamily,
    name="Steam Deck",
    api="hip",
    arch_name="gfx1033",
    compute=10.3,
    version="RDNA2",
    sm_count=8,
)
"""Steam Deck (Van Gogh) APU configuration."""
