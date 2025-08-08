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

from gpu.host import DeviceContext
from memory import bitcast
from builtin.simd import _convert_f32_to_float8_ue8m0
from math import inf, nan


# CHECK-LABEL: test_simd_f32_to_ue8m0
# CHECK: 0
# CHECK: 0
# CHECK: 1
# CHECK: 255
# CHECK: 255
# CHECK: 129
# CHECK: 129
# CHECK: 129
# CHECK: 0
# CHECK: 127
# CHECK: 129
# CHECK: 139
# CHECK: 255
# CHECK: 0
# CHECK: 139
# CHECK: 129
# CHECK: 127
# CHECK: 254
fn test_simd_f32_to_ue8m0():
    print("== test_simd_f32_to_ue8m0")

    alias M = 32
    var mantissa = UInt32(0x00400000)
    var exp = UInt32(0x80)

    var f32_simd = SIMD[DType.float32, M](0.0)

    var i = 0
    f32_simd[i] = bitcast[DType.float32, 1](mantissa - 1)
    i += 1
    f32_simd[i] = bitcast[DType.float32, 1](mantissa)
    i += 1
    f32_simd[i] = bitcast[DType.float32, 1](mantissa + 1)
    i += 1
    f32_simd[i] = inf[DType.float32]()
    i += 1
    f32_simd[i] = nan[DType.float32]()
    i += 1
    f32_simd[i] = bitcast[DType.float32, 1]((exp << 23) + mantissa - 1)
    i += 1
    f32_simd[i] = bitcast[DType.float32, 1]((exp << 23) + mantissa)
    i += 1
    f32_simd[i] = bitcast[DType.float32, 1]((exp << 23) + mantissa + 1)
    i += 1
    f32_simd[i] = Scalar[DType.float32](0.0)
    i += 1
    f32_simd[i] = Scalar[DType.float32](1.0)
    i += 1
    f32_simd[i] = Scalar[DType.float32](4.0)
    i += 1
    f32_simd[i] = Scalar[DType.float32](4096.0)
    i += 1
    f32_simd[i] = -inf[DType.float32]()
    i += 1
    f32_simd[i] = Scalar[DType.float32](-0.0)
    i += 1
    f32_simd[i] = Scalar[DType.float32](-4096.0)
    i += 1
    f32_simd[i] = Scalar[DType.float32](-4.0)
    i += 1
    f32_simd[i] = Scalar[DType.float32](-1.0)
    i += 1
    f32_simd[i] = bitcast[DType.float32, 1](UInt32(0x7F000000))
    i += 1

    var f32_casted_ue8m0 = _convert_f32_to_float8_ue8m0[DType.uint8](f32_simd)

    for i in range(i):
        print(
            f32_casted_ue8m0[i],
        )


fn test_simd_f32_to_ue8m0_ptx_kernel[
    size: Int,
    target: DType,
](x: SIMD[DType.float32, size], last_idx: Int):
    var x_casted = _convert_f32_to_float8_ue8m0[target](x)

    for i in range(last_idx):
        print(
            x_casted[i],
        )


# CHECK-LABEL: test_simd_f32_to_ue8m0_ptx_path
# CHECK: 0
# CHECK: 0
# CHECK: 1
# CHECK: 254
# CHECK: 255
# CHECK: 129
# CHECK: 129
# CHECK: 129
# CHECK: 0
# CHECK: 127
# CHECK: 139
# CHECK: 254
# CHECK: 0
# CHECK: 139
# CHECK: 129
# CHECK: 127
# CHECK: 254
fn test_simd_f32_to_ue8m0_ptx_path(ctx: DeviceContext) raises:
    print("== test_simd_f32_to_ue8m0_ptx_path")

    alias M = 32
    var mantissa = UInt32(0x00400000)
    var exp = UInt32(0x80)

    var f32_simd = SIMD[DType.float32, M](0.0)

    var i = 0
    f32_simd[i] = bitcast[DType.float32, 1](mantissa - 1)
    i += 1
    f32_simd[i] = bitcast[DType.float32, 1](mantissa)
    i += 1
    f32_simd[i] = bitcast[DType.float32, 1](mantissa + 1)
    i += 1
    f32_simd[i] = inf[DType.float32]()
    i += 1
    f32_simd[i] = nan[DType.float32]()
    i += 1
    f32_simd[i] = bitcast[DType.float32, 1]((exp << 23) + mantissa - 1)
    i += 1
    f32_simd[i] = bitcast[DType.float32, 1]((exp << 23) + mantissa)
    i += 1
    f32_simd[i] = bitcast[DType.float32, 1]((exp << 23) + mantissa + 1)
    i += 1
    f32_simd[i] = Scalar[DType.float32](0.0)
    i += 1
    f32_simd[i] = Scalar[DType.float32](1.0)
    i += 1
    f32_simd[i] = Scalar[DType.float32](4.0)
    i += 1
    f32_simd[i] = Scalar[DType.float32](4096.0)
    i += 1
    f32_simd[i] = -inf[DType.float32]()
    i += 1
    f32_simd[i] = Scalar[DType.float32](-0.0)
    i += 1
    f32_simd[i] = Scalar[DType.float32](-4096.0)
    i += 1
    f32_simd[i] = Scalar[DType.float32](-4.0)
    i += 1
    f32_simd[i] = Scalar[DType.float32](-1.0)
    i += 1
    f32_simd[i] = bitcast[DType.float32, 1](UInt32(0x7F000000))
    i += 1

    ctx.enqueue_function[test_simd_f32_to_ue8m0_ptx_kernel[M, DType.uint8]](
        f32_simd, i, grid_dim=1, block_dim=1
    )
    ctx.synchronize()


fn main() raises:
    test_simd_f32_to_ue8m0()

    with DeviceContext() as ctx:
        test_simd_f32_to_ue8m0_ptx_path(ctx)
