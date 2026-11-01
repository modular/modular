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
"""NVIDIA GPU targets, each carrying the CUDA stdlib and MAX plugins."""

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
from std._plugin.cuda import CUDAPlugin

from . import CudaMaxPlugin

comptime CUDA_PLUGINS = TypeList.of[CUDAPlugin, CudaMaxPlugin]().values
"""The `opaque_plugin` payload of every NVIDIA target."""


struct CudaTargets(TargetAcceleratorCollection):
    """NVIDIA GPU targets that the CUDA plugins contribute."""

    comptime vendor_name: String = "NVIDIA"

    # fmt: off
    comptime RAW_TARGETS = TypeList.of[
        Trait=TargetAcceleratorType,
        # FIXME: GTX1060 and GTX1080Ti share `sm_61` but have different
        # `sm_count`; target resolution should differentiate them at compile time.
        GTX1060   ._with_cli_values[[]       ], # Could be ["sm_61"], but `sm_61` resolves to GTX1080Ti.
        GTX1080Ti ._with_cli_values[["sm_61"]],
        TeslaP100 ._with_cli_values[["sm_60"]],
        GTX970    ._with_cli_values[["sm_52"]],
        RTX2060   ._with_cli_values[["sm_75"]],
        # Note: `sm_86` is ambiguous; it resolves to A10 rather than RTX3090.
        RTX3090   ._with_cli_values[[]       ], # Could be ["sm_86"], but `sm_86` resolves to A10.
        A10       ._with_cli_values[["sm_86"]],
        A100      ._with_cli_values[["sm_80"]],
        OrinNano  ._with_cli_values[["sm_87"]],
        L4        ._with_cli_values[["sm_89"]],
        RTX4090m  ._with_cli_values[[]       ], # Could be ["sm_89"], but `sm_89` resolves to L4.
        RTX4090   ._with_cli_values[[]       ], # Could be ["sm_89"], but `sm_89` resolves to L4.
        # FIXME (KERN-1814): Unlike H100 and H200, blackwell devices (B100 vs B200)
        # architecture wise are different. We need to differentiate between them here.
        B100      ._with_cli_values[[]                   ], # Could be ["sm_100", "sm_100a"], but both resolve to B200.
        B200      ._with_cli_values[["sm_100", "sm_100a"]],
        H100      ._with_cli_values[["sm_90" , "sm_90a" ]],
        B300      ._with_cli_values[["sm_103", "sm_103a"]],
        JetsonThor._with_cli_values[["sm_110", "sm_110a"]],
        RTX5090   ._with_cli_values[["sm_120", "sm_120a"]],
        DGXSpark  ._with_cli_values[["sm_121", "sm_121a"]],
    ]().values
    # fmt: on

    @staticmethod
    def normalize_target_arch(target_arch0: StaticString) -> String:
        # "nvidia:sm_90a" -> "sm_90a", "nvidia:sm90" -> "sm_90",
        # "nvidia:80" -> "sm_80", "sm80" -> "sm_80".
        #
        # Keep normalization idempotent: applying it to an already normalized
        # target must not change it.
        #
        # These substring replacements must be ordered carefully, as a pattern
        # that is a prefix of an already canonical name corrupts it.
        return (
            target_arch0.replace("nvidia:sm_", "sm_")
            .replace("nvidia:sm", "sm_")
            .replace("nvidia:", "sm_")
            .replace("sm", "sm_")
            .replace("sm__", "sm_")
        )

    @staticmethod
    def supported_archs_message() -> String:
        comptime nvidia_archs = (
            "sm_52 (Maxwell), sm_60/sm_61 (Pascal), sm_75 (Turing), sm_80"
            " (Ampere A100), sm_86 (Ampere A10), sm_87 (Orin), sm_89 (Ada"
            " L4/RTX4090), sm_90/sm_90a (Hopper H100), sm_100/sm_100a"
            " (Blackwell B100/B200), sm_110 (Jetson Thor), sm_120/sm_120a"
            " (Blackwell RTX5090), sm_121 (DGX Spark)"
        )
        return String(
            "  NVIDIA: ",
            nvidia_archs,
            "\n  See: https://developer.nvidia.com/cuda-gpus\n\n",
        )


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
        `opaque_plugin = `,
        CUDA_PLUGINS,
        `,`,
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


comptime A100 = TargetAccelerator[
    GPUInfo.from_family(
        family=NvidiaAmpereDatacenterFamily,
        name="A100",
        api="cuda",
        arch_name="ampere",
        compute=8.0,
        version="sm_80",
        sm_count=108,
    ),
    _a100_target,
]()
"""NVIDIA A100 GPU configuration."""

# ===-----------------------------------------------------------------------===#
# A10
# ===-----------------------------------------------------------------------===#


comptime _a10_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `opaque_plugin = `,
        CUDA_PLUGINS,
        `,`,
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


comptime A10 = TargetAccelerator[
    GPUInfo.from_family(
        family=NvidiaAmpereWorkstationFamily,
        name="A10",
        api="cuda",
        arch_name="ampere",
        compute=8.6,
        version="sm_86",
        sm_count=72,
    ),
    _a10_target,
]()
"""NVIDIA A10 GPU configuration."""

# ===-----------------------------------------------------------------------===#
# Jetson Orin Nano
# ===-----------------------------------------------------------------------===#


comptime _orin_nano_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `opaque_plugin = `,
        CUDA_PLUGINS,
        `,`,
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


comptime OrinNano = TargetAccelerator[
    GPUInfo.from_family(
        family=NvidiaAmpereEmbeddedFamily,
        name="Orin Nano",
        api="cuda",
        arch_name="ampere",
        compute=8.7,
        version="sm_87",
        sm_count=8,
    ),
    _orin_nano_target,
]()
"""NVIDIA Orin Nano GPU configuration."""

# ===-----------------------------------------------------------------------===#
# Jetson Thor
# ===-----------------------------------------------------------------------===#


comptime _jetson_thor_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `opaque_plugin = `,
        CUDA_PLUGINS,
        `,`,
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


comptime JetsonThor = TargetAccelerator[
    GPUInfo.from_family(
        family=NvidiaBlackwellFamily,
        name="Jetson Thor",
        api="cuda",
        arch_name="blackwell",
        compute=11.0,
        version="sm_110",
        sm_count=20,
    ),
    _jetson_thor_target,
]()
"""NVIDIA Jetson Thor GPU configuration."""

# ===-----------------------------------------------------------------------===#
# DGX Spark
# ===-----------------------------------------------------------------------===#


comptime _dgx_spark_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `opaque_plugin = `,
        CUDA_PLUGINS,
        `,`,
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


comptime DGXSpark = TargetAccelerator[
    GPUInfo.from_family(
        family=NvidiaBlackwellConsumerFamily,
        name="DGX Spark",
        api="cuda",
        arch_name="blackwell",
        compute=12.1,
        version="sm_121",
        sm_count=48,
    ),
    _dgx_spark_target,
]()
"""NVIDIA DGX Spark GPU configuration."""

# ===-----------------------------------------------------------------------===#
# L4
# ===-----------------------------------------------------------------------===#


comptime _l4_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `opaque_plugin = `,
        CUDA_PLUGINS,
        `,`,
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


comptime L4 = TargetAccelerator[
    GPUInfo.from_family(
        family=NvidiaAdaFamily,
        name="L4",
        api="cuda",
        arch_name="ada",
        compute=8.9,
        version="sm_89",
        sm_count=58,
    ),
    _l4_target,
]()
"""NVIDIA L4 GPU configuration."""

# ===-----------------------------------------------------------------------===#
# RTX 4090 M
# ===-----------------------------------------------------------------------===#


comptime _rtx4090m_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `opaque_plugin = `,
        CUDA_PLUGINS,
        `,`,
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


comptime RTX4090m = TargetAccelerator[
    GPUInfo.from_family(
        family=NvidiaAdaFamily,
        name="RTX4090m",
        api="cuda",
        arch_name="ada lovelace",
        compute=8.9,
        version="sm_89",
        sm_count=76,
    ),
    _rtx4090m_target,
]()
"""NVIDIA RTX 4090 Mobile GPU configuration."""

# ===-----------------------------------------------------------------------===#
# RTX 4090
# ===-----------------------------------------------------------------------===#


comptime _rtx4090_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `opaque_plugin = `,
        CUDA_PLUGINS,
        `,`,
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


comptime RTX4090 = TargetAccelerator[
    GPUInfo.from_family(
        family=NvidiaAdaFamily,
        name="RTX4090",
        api="cuda",
        arch_name="ada lovelace",
        compute=8.9,
        version="sm_89",
        sm_count=128,
    ),
    _rtx4090_target,
]()
"""NVIDIA RTX 4090 GPU configuration."""


# ===-----------------------------------------------------------------------===#
# H100
# ===-----------------------------------------------------------------------===#


comptime _h100_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `opaque_plugin = `,
        CUDA_PLUGINS,
        `,`,
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
comptime H100 = TargetAccelerator[
    GPUInfo.from_family(
        family=NvidiaHopperFamily,
        name="H100",
        api="cuda",
        arch_name="hopper",
        compute=9.0,
        version="sm_90a",
        sm_count=132,
    ),
    _h100_target,
]()
"""NVIDIA H100 GPU configuration."""

# ===-----------------------------------------------------------------------===#
# B100
# ===-----------------------------------------------------------------------===#


comptime _b100_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `opaque_plugin = `,
        CUDA_PLUGINS,
        `,`,
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
comptime B100 = TargetAccelerator[
    GPUInfo.from_family(
        family=NvidiaBlackwellFamily,
        name="B100",
        api="cuda",
        arch_name="blackwell",
        compute=10.0,
        version="sm_100a",
        sm_count=132,
    ),
    _b100_target,
]()
"""NVIDIA B100 GPU configuration."""

comptime B200 = TargetAccelerator[
    GPUInfo.from_family(
        family=NvidiaBlackwellFamily,
        name="B200",
        api="cuda",
        arch_name="blackwell",
        compute=10.0,
        version="sm_100a",
        sm_count=148,
    ),
    # FIXME (KERN-1814): Unlike H100 and H200, blackwell devices (B100 vs B200)
    # architecture wise are different. We need to differentiate between them here.
    _b100_target,
]()
"""NVIDIA B200 GPU configuration."""

# ===-----------------------------------------------------------------------===#
# B300
# ===-----------------------------------------------------------------------===#


comptime _b300_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `opaque_plugin = `,
        CUDA_PLUGINS,
        `,`,
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


comptime B300 = TargetAccelerator[
    GPUInfo.from_family(
        family=NvidiaBlackwellFamily,
        name="B300",
        api="cuda",
        arch_name="blackwell",
        compute=10.3,
        version="sm_103a",
        sm_count=160,
    ),
    _b300_target,
]()
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
        info == materialize[B100.gpu_info]()
        or info == materialize[B200.gpu_info]()
        or info == materialize[B300.gpu_info]()
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
        `opaque_plugin = `,
        CUDA_PLUGINS,
        `,`,
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
comptime RTX5090 = TargetAccelerator[
    GPUInfo.from_family(
        family=NvidiaBlackwellConsumerFamily,
        name="RTX5090",
        api="cuda",
        arch_name="blackwell",
        compute=12.0,
        version="sm_120a",
        sm_count=170,
    ),
    _rtx5090_target,
]()
"""NVIDIA RTX 5090 GPU configuration."""


# ===-----------------------------------------------------------------------===#
# RTX3090
# ===-----------------------------------------------------------------------===#


comptime _rtx3090_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `opaque_plugin = `,
        CUDA_PLUGINS,
        `,`,
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
comptime RTX3090 = TargetAccelerator[
    GPUInfo.from_family(
        family=NvidiaAmpereWorkstationFamily,
        name="NVIDIA GeForce RTX 3090",
        api="cuda",
        arch_name="ampere",
        compute=8.6,
        version="sm_86",
        sm_count=82,
    ),
    _rtx3090_target,
]()
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
        `opaque_plugin = `,
        CUDA_PLUGINS,
        `,`,
        `arch = "sm_61", `,
        `features = "+ptx50,+sm_61", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for NVIDIA GTX 1080 Ti GPU."""


comptime GTX1080Ti = TargetAccelerator[
    GPUInfo.from_family(
        family=NvidiaPascalFamily,
        name="NVIDIA GeForce GTX 1080 Ti",
        api="cuda",
        arch_name="pascal",
        compute=6.1,
        version="sm_61",
        sm_count=28,
    ),
    _gtx1080ti_target,
]()
"""NVIDIA GeForce GTX 1080 Ti GPU configuration."""


# ===-----------------------------------------------------------------------===#
# GTX1060
# ===-----------------------------------------------------------------------===#


comptime _gtx1060_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `opaque_plugin = `,
        CUDA_PLUGINS,
        `,`,
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


comptime GTX1060 = TargetAccelerator[
    GPUInfo.from_family(
        family=NvidiaPascalFamily,
        name="NVIDIA GeForce GTX 1060",
        api="cuda",
        arch_name="pascal",
        compute=6.1,
        version="sm_61",
        sm_count=10,
    ),
    _gtx1060_target,
]()
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
        `opaque_plugin = `,
        CUDA_PLUGINS,
        `,`,
        `arch = "sm_52", `,
        `features = "+ptx50,+sm_52", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for NVIDIA GTX 970 GPU."""


comptime GTX970 = TargetAccelerator[
    GPUInfo.from_family(
        family=NvidiaMaxwellFamily,
        name="NVIDIA GeForce GTX 970",
        api="cuda",
        arch_name="maxwell",
        compute=5.2,
        version="sm_52",
        sm_count=13,
    ),
    _gtx970_target,
]()
"""NVIDIA GeForce GTX 970 GPU configuration."""


# ===-----------------------------------------------------------------------===#
# Tesla P100
# ===-----------------------------------------------------------------------===#


comptime _teslap100_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `opaque_plugin = `,
        CUDA_PLUGINS,
        `,`,
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


comptime TeslaP100 = TargetAccelerator[
    GPUInfo.from_family(
        family=NvidiaPascalFamily,
        name="NVIDIA Tesla P100",
        api="cuda",
        arch_name="pascal",
        compute=6.0,
        version="sm_60",
        sm_count=56,
    ),
    _teslap100_target,
]()
"""NVIDIA Tesla P100 GPU configuration."""


# ===-----------------------------------------------------------------------===#
# RTX2060
# ===-----------------------------------------------------------------------===#


comptime _rtx2060_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `stdlib_plugin = "cuda", `,
        `opaque_plugin = `,
        CUDA_PLUGINS,
        `,`,
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


comptime RTX2060 = TargetAccelerator[
    GPUInfo.from_family(
        family=NvidiaTuringFamily,
        name="RTX2060",
        api="cuda",
        arch_name="turing",
        compute=7.5,
        version="sm_75",
        sm_count=30,
    ),
    _rtx2060_target,
]()
"""NVIDIA RTX 2060 GPU configuration."""
