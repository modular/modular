# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
from sys._assembly import inlined_assembly
from sys import is_nvidia_gpu
from sys.info import _is_sm_100x_or_newer


fn cast_to_e2m1[
    dtype: DType,
    width: Int, //,
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    constrained[
        dtype in (DType.float32, DType.bfloat16, DType.float16),
        "dtype must be float32, bfloat16 or float16",
    ]()
    # for float4_e2m1fn has only 16 values
    # (x >= 0.0) & (x <= 0.25)] => 0.0
    # (x > 0.25) & (x < 0.75)] => 0.5
    # (x >= 0.75) & (x <= 1.25)] => 1.0
    # (x > 1.25) & (x < 1.75)] => 1.5
    # (x >= 1.75) & (x <= 2.5)] => 2.0
    # (x > 2.5) & (x < 3.5)] => 3.0
    # (x >= 3.5) & (x <= 5.0)] => 4.0
    # (x > 5.0) => 6.0

    var sign = x.lt(0).select(-1.0, 1.0).cast[dtype]()
    var abs_x = abs(x)
    var result = SIMD[dtype, width]()

    @parameter
    for i in range(width):
        if abs_x[i] <= 0.25:
            result[i] = 0.0
        elif abs_x[i] < 0.75:
            result[i] = 0.5
        elif abs_x[i] <= 1.25:
            result[i] = 1.0
        elif abs_x[i] < 1.75:
            result[i] = 1.5
        elif abs_x[i] <= 2.5:
            result[i] = 2.0
        elif abs_x[i] < 3.5:
            result[i] = 3.0
        elif abs_x[i] <= 5.0:
            result[i] = 4.0
        else:
            result[i] = 6.0
    return result * sign


fn cast_fp32_to_e2m1[
    width: Int, //,
](x: SIMD[DType.float32, width]) -> UInt32:
    constrained[
        is_nvidia_gpu() and _is_sm_100x_or_newer(),
        "only supported on NVIDIA GPUs with SM 100 or newer",
    ]()
    constrained[width == 8, "width must be 8"]()

    comptime asm_code = """{
.reg .b8 byte0;
.reg .b8 byte1;
.reg .b8 byte2;
.reg .b8 byte3;
cvt.rn.satfinite.e2m1x2.f32   byte0, $2, $1;
cvt.rn.satfinite.e2m1x2.f32   byte1, $4, $3;
cvt.rn.satfinite.e2m1x2.f32   byte2, $6, $5;
cvt.rn.satfinite.e2m1x2.f32   byte3, $8, $7;
mov.b32 $0, {byte0, byte1, byte2, byte3};
}
"""
    return inlined_assembly[
        asm_code, UInt32, constraints="=r,f,f,f,f,f,f,f,f", has_side_effect=True
    ](x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7])


comptime E2M1_TO_FLOAT32 = SIMD[DType.float32, 16](
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)
