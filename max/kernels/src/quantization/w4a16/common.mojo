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

from std.gpu import WARP_SIZE
from std.sys import get_defined_bool, get_defined_int, is_defined

from quantization.w4a16.config_defaults import (
    DEFAULT_ASSUME_EVEN_K,
    DEFAULT_ASSUME_EVEN_MN,
    DEFAULT_ASSUME_EVEN_N,
    DEFAULT_BK,
    DEFAULT_BLOCK_SWIZZLE_SCALE,
    DEFAULT_BM,
    DEFAULT_BN,
    DEFAULT_DEQUANT_B_IN_BF16,
    DEFAULT_GROUP_SIZE,
    DEFAULT_GROUP_SIZE_M,
    DEFAULT_KERNEL_VARIANT,
    DEFAULT_LOAD_B_BY_QPACK,
    DEFAULT_NUM_STAGES,
    DEFAULT_QPACK_K_VECTOR_WIDTH,
    DEFAULT_RING_STARTUP_ALL_WARPS,
    DEFAULT_RING_PRODUCER_WARPS,
    DEFAULT_SCALE_AFTER_GROUP,
    DEFAULT_SMEM_PAD,
    DEFAULT_USE_LDS_SWIZZLE,
    DEFAULT_USE_FP16,
    DEFAULT_WARPS_M,
    DEFAULT_WARPS_N,
    DEFAULT_USE_QZEROS,
    DEFAULT_ZERO_OFFSET,
    DEFAULT_ZP_BIAS,
)


comptime USE_FP16 = (
    get_defined_bool["USE_FP16", False]() if is_defined[
        "USE_FP16"
    ]() else DEFAULT_USE_FP16
)

comptime BM = get_defined_int["BM"]() if is_defined["BM"]() else DEFAULT_BM
comptime BN = get_defined_int["BN"]() if is_defined["BN"]() else DEFAULT_BN
comptime BK = get_defined_int["BK"]() if is_defined["BK"]() else DEFAULT_BK

comptime GROUP_SIZE = (
    get_defined_int["GROUP_SIZE"]() if is_defined[
        "GROUP_SIZE"
    ]() else DEFAULT_GROUP_SIZE
)
comptime ZP_BIAS = (
    get_defined_int["ZP_BIAS"]() if is_defined["ZP_BIAS"]() else DEFAULT_ZP_BIAS
)
comptime USE_QZEROS = (
    get_defined_bool["USE_QZEROS", False]() if is_defined[
        "USE_QZEROS"
    ]() else DEFAULT_USE_QZEROS
)
comptime ZERO_OFFSET = (
    get_defined_int["ZERO_OFFSET"]() if is_defined[
        "ZERO_OFFSET"
    ]() else DEFAULT_ZERO_OFFSET
)

comptime MMA_M = 16
comptime MMA_N = 16
comptime MMA_K = 16
comptime AB = 8
comptime CD = 8
comptime SMEM_PAD = (
    get_defined_int["SMEM_PAD"]() if is_defined[
        "SMEM_PAD"
    ]() else DEFAULT_SMEM_PAD
)

comptime RING_PRODUCER_WARPS = (
    get_defined_int["RING_PRODUCER_WARPS"]() if is_defined[
        "RING_PRODUCER_WARPS"
    ]() else DEFAULT_RING_PRODUCER_WARPS
)
comptime NUM_STAGES = (
    get_defined_int["NUM_STAGES"]() if is_defined["NUM_STAGES"]() else (
        get_defined_int["RING_STAGES"]() if is_defined[
            "RING_STAGES"
        ]() else DEFAULT_NUM_STAGES
    )
)
comptime BLOCK_SWIZZLE_SCALE = (
    get_defined_int["BLOCK_SWIZZLE_SCALE"]() if is_defined[
        "BLOCK_SWIZZLE_SCALE"
    ]() else DEFAULT_BLOCK_SWIZZLE_SCALE
)
comptime GROUP_SIZE_M = (
    get_defined_int["GROUP_SIZE_M"]() if is_defined[
        "GROUP_SIZE_M"
    ]() else DEFAULT_GROUP_SIZE_M
)
comptime USE_LDS_SWIZZLE = (
    get_defined_bool["USE_LDS_SWIZZLE", False]() if is_defined[
        "USE_LDS_SWIZZLE"
    ]() else DEFAULT_USE_LDS_SWIZZLE
)
comptime RING_STARTUP_ALL_WARPS = (
    get_defined_bool["RING_STARTUP_ALL_WARPS", True]() if is_defined[
        "RING_STARTUP_ALL_WARPS"
    ]() else DEFAULT_RING_STARTUP_ALL_WARPS
)
comptime LOAD_B_BY_QPACK = (
    get_defined_bool["LOAD_B_BY_QPACK", True]() if is_defined[
        "LOAD_B_BY_QPACK"
    ]() else DEFAULT_LOAD_B_BY_QPACK
)
comptime QPACK_K_VECTOR_WIDTH = (
    get_defined_int["QPACK_K_VECTOR_WIDTH"]() if is_defined[
        "QPACK_K_VECTOR_WIDTH"
    ]() else DEFAULT_QPACK_K_VECTOR_WIDTH
)
comptime DEQUANT_B_IN_BF16 = (
    get_defined_bool["DEQUANT_B_IN_BF16", False]() if is_defined[
        "DEQUANT_B_IN_BF16"
    ]() else DEFAULT_DEQUANT_B_IN_BF16
)
comptime SCALE_AFTER_GROUP = (
    get_defined_bool["SCALE_AFTER_GROUP", True]() if is_defined[
        "SCALE_AFTER_GROUP"
    ]() else DEFAULT_SCALE_AFTER_GROUP
)
comptime ASSUME_EVEN_K = (
    get_defined_bool["ASSUME_EVEN_K", False]() if is_defined[
        "ASSUME_EVEN_K"
    ]() else DEFAULT_ASSUME_EVEN_K
)
comptime ASSUME_EVEN_MN = (
    get_defined_bool["ASSUME_EVEN_MN", False]() if is_defined[
        "ASSUME_EVEN_MN"
    ]() else DEFAULT_ASSUME_EVEN_MN
)
comptime ASSUME_EVEN_N = (
    get_defined_bool["ASSUME_EVEN_N", False]() if is_defined[
        "ASSUME_EVEN_N"
    ]() else DEFAULT_ASSUME_EVEN_N
)
comptime KERNEL_VARIANT = DEFAULT_KERNEL_VARIANT
comptime KERNEL_USES_FDOT2 = (
    get_defined_bool["USE_KPACKED_DOT2", False]() if is_defined[
        "USE_KPACKED_DOT2"
    ]() else KERNEL_VARIANT
    == "kpacked_dot2"
)
comptime WARPS_M = (
    get_defined_int["WARPS_M"]() if is_defined["WARPS_M"]() else DEFAULT_WARPS_M
)
comptime WARPS_N = (
    get_defined_int["WARPS_N"]() if is_defined["WARPS_N"]() else DEFAULT_WARPS_N
)
comptime COMPUTE_WARPS = WARPS_M * WARPS_N
comptime PRODUCTION_TOTAL_WARPS = RING_PRODUCER_WARPS + COMPUTE_WARPS
comptime PRODUCTION_TOTAL_THREADS = PRODUCTION_TOTAL_WARPS * WARP_SIZE

comptime dtype_in = DType.float16 if USE_FP16 else DType.bfloat16
comptime dtype_acc = DType.float32
comptime dtype_out = DType.float16 if USE_FP16 else DType.bfloat16
comptime dtype_q = DType.int32
