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
"""Compile-only register/LDS report for the blockwise FP8 grouped matmul.

Compiles `blockwise_scaled_fp8_grouped_matmul_amd_kernel` for gfx950 and
parses the AMD code-object metadata (`.vgpr_count`, `.sgpr_count`,
`.agpr_count`, `.group_segment_fixed_size` = static LDS bytes). Asserts
there are no register spills, which would silently wreck performance.
"""

from max.gpu.host import get_gpu_target
from max.gpu.host.compile import _compile_code
from layout import ComptimeInt, RowMajorLayout
from layout.tensor_engine import DefaultEngine
from std.testing import assert_true

from linalg.matmul.gpu.amd.blockwise_scaled_fp8_grouped_matmul_amd import (
    blockwise_scaled_fp8_grouped_matmul_amd_kernel,
)


def parse_directive_value(asm: String, directive: String) -> Int:
    if directive not in asm:
        return 0
    var directive_idx = asm.find(directive)
    var line_end = asm.find("\n", directive_idx)
    if line_end <= directive_idx:
        return 0
    var line = asm[byte=directive_idx:line_end]
    var colon_idx = line.find(":")
    if colon_idx < 0:
        return 0
    try:
        return Int(line[byte = colon_idx + 1 :].strip())
    except:
        return 0


def main() raises:
    comptime M = 16384
    comptime N = 4096
    comptime K = 6144
    comptime num_experts = 8
    comptime num_active = 8
    comptime c_type = DType.bfloat16
    comptime in_type = DType.float8_e4m3fn
    comptime scale_type = DType.float32

    comptime c_layout = RowMajorLayout[ComptimeInt[M], ComptimeInt[N]]
    comptime a_layout = RowMajorLayout[ComptimeInt[M], ComptimeInt[K]]
    comptime b_layout = RowMajorLayout[
        ComptimeInt[num_experts], ComptimeInt[N], ComptimeInt[K]
    ]
    comptime as_layout = RowMajorLayout[ComptimeInt[K // 128], ComptimeInt[M]]
    comptime bs_layout = RowMajorLayout[
        ComptimeInt[num_experts], ComptimeInt[N // 128], ComptimeInt[K // 128]
    ]
    comptime a_off_layout = RowMajorLayout[ComptimeInt[num_active + 1]]
    comptime eids_layout = RowMajorLayout[ComptimeInt[num_active]]
    comptime eng = DefaultEngine[element_width=1]

    comptime kernel = blockwise_scaled_fp8_grouped_matmul_amd_kernel[
        c_type,
        in_type,
        in_type,
        scale_type,
        scale_type,
        .float32,
        c_layout,
        a_layout,
        b_layout,
        as_layout,
        bs_layout,
        a_off_layout,
        eids_layout,
        eng,
        eng,
        eng,
        eng,
        eng,
        eng,
        eng,
        BM=128,
        BN=128,
        BK=128,
        WM=64,
        WN=64,
        MMA_M=16,
        MMA_N=16,
        MMA_K=128,
        N_SCALE=128,
        K_SCALE=128,
    ]

    var compiled = _compile_code[kernel, target=get_gpu_target["gfx950"]()]()
    var asm = compiled.asm

    var vgprs = parse_directive_value(asm, ".vgpr_count")
    var sgprs = parse_directive_value(asm, ".sgpr_count")
    var agprs = parse_directive_value(asm, ".agpr_count")
    var vspill = parse_directive_value(asm, ".vgpr_spill_count")
    var sspill = parse_directive_value(asm, ".sgpr_spill_count")
    var lds = parse_directive_value(asm, ".group_segment_fixed_size")

    print(
        "blockwise_scaled_fp8_grouped_matmul_amd (BM128 BN128 BK128 WM64 WN64):"
    )
    print("  VGPRs:", vgprs)
    print("  SGPRs:", sgprs)
    print("  AGPRs:", agprs)
    print("  VGPR spills:", vspill)
    print("  SGPR spills:", sspill)
    print("  LDS bytes:", lds)

    assert_true(vspill == 0, "VGPR spill count should be 0")
    assert_true(sspill == 0, "SGPR spill count should be 0")
    print("PASSED")
