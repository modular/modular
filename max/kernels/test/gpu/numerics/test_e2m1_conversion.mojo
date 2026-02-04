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

from linalg.fp4_utils import (
    cast_fp32_to_fp4e2m1,
    E2M1_TO_FLOAT32,
    cast_fp_to_fp4e2m1,
    cast_f4e2m1x2_to_fp16x2,
    cast_f4e2m1x8_to_fp16x8,
    cast_f4e2m1x16_to_fp16x16,
)
from gpu.host import DeviceContext
from math import nan, inf
from sys import bit_width_of
from memory import bitcast


# CHECK-LABEL: test_simd_f32_to_e2m1
# CHECK: [0.0, 0.0, 0.5, 1.0, 1.0, 1.5, 2.0, 2.0]
# CHECK: [2.0, 3.0, 4.0, 4.0, 4.0, 6.0, 6.0, 6.0]
# CHECK: [0.0, -0.0, -0.5, -1.0, -1.0, -1.5, -2.0, -2.0]
# CHECK: [-2.0, -3.0, -4.0, -4.0, -4.0, -6.0, 6.0, -6.0]
fn test_simd_f32_to_e2m1():
    print("== test_simd_f32_to_e2m1")

    comptime size = 32
    var f32_simd = SIMD[DType.float32, size](
        0.0,
        0.23,
        0.26,
        0.76,
        1.25,
        1.26,
        1.75,
        1.76,
        2.5,
        2.51,
        3.5,
        3.51,
        5.0,
        5.01,
        nan[DType.float32](),
        inf[DType.float32](),
        -0.0,
        -0.23,
        -0.26,
        -0.76,
        -1.25,
        -1.26,
        -1.75,
        -1.76,
        -2.5,
        -2.51,
        -3.5,
        -3.51,
        -5.0,
        -5.01,
        -nan[DType.float32](),
        -inf[DType.float32](),
    )

    var e2m1_simd = cast_fp_to_fp4e2m1(f32_simd)

    @parameter
    for iter in range(size // 8):
        var x_slice = e2m1_simd.slice[8, offset = iter * 8]()
        print(
            x_slice,
        )


fn test_simd_f32_to_e2m1_ptx_kernel[
    size: Int,
](x: SIMD[DType.float32, size]):
    comptime FP4_E2M1_WIDTH = 4
    comptime FP4_E2M1_MASK = pow(2, FP4_E2M1_WIDTH) - 1

    @parameter
    for iter in range(size // 8):
        var x_slice = x.slice[8, offset = iter * 8]()
        var x_casted = cast_fp32_to_fp4e2m1(x_slice)

        @parameter
        for shift in range(0, bit_width_of[DType.uint32](), FP4_E2M1_WIDTH):
            var x = (x_casted.to_bits() >> shift) & FP4_E2M1_MASK
            print(E2M1_TO_FLOAT32[Int(x)], end=" ")
        print("")


# CHECK-LABEL: test_simd_f32_to_e2m1_ptx_path
# CHECK: 0.0 0.0 0.5 1.0 1.0 1.5 2.0 2.0
# CHECK: 2.0 3.0 4.0 4.0 4.0 6.0 6.0 6.0
# CHECK: -0.0 -0.0 -0.5 -1.0 -1.0 -1.5 -2.0 -2.0
# CHECK: -2.0 -3.0 -4.0 -4.0 -4.0 -6.0 6.0 -6.0
fn test_simd_f32_to_e2m1_ptx_path(ctx: DeviceContext) raises:
    print("== test_simd_f32_to_e2m1_ptx_path")

    comptime size = 32
    var f32_simd = SIMD[DType.float32, size](
        0.0,
        0.23,
        0.26,
        0.76,
        1.25,
        1.26,
        1.75,
        1.76,
        2.5,
        2.51,
        3.5,
        3.51,
        5.0,
        5.01,
        nan[DType.float32](),
        inf[DType.float32](),
        -0.0,
        -0.23,
        -0.26,
        -0.76,
        -1.25,
        -1.26,
        -1.75,
        -1.76,
        -2.5,
        -2.51,
        -3.5,
        -3.51,
        -5.0,
        -5.01,
        -nan[DType.float32](),
        -inf[DType.float32](),
    )

    comptime kernel = test_simd_f32_to_e2m1_ptx_kernel[size,]
    ctx.enqueue_function_experimental[kernel](f32_simd, grid_dim=1, block_dim=1)
    ctx.synchronize()


fn test_simd_f4e2m1x2_to_fp16x2_ptx_kernel[
    size: Int,
](x: SIMD[DType.uint8, size]):
    @parameter
    for i in range(size // 4):
        for j in range(4):
            var x_casted = cast_f4e2m1x2_to_fp16x2(x[i * 4 + j])
            print(x_casted, end=" ")
        print("")


# CHECK-LABEL: test_simd_f4e2m1x2_to_fp16x2
# CHECK: [0.0, 0.0] [0.5, 0.0] [0.0, 0.5] [0.5, 0.5]
# CHECK: [4.0, -1.0] [1.0, 0.5] [-0.0, 1.5] [6.0, 1.0]
# CHECK: [0.0, 4.0] [1.5, -1.5] [0.5, 1.5] [1.5, 1.5]
# CHECK: [-6.0, 0.0] [3.0, 0.5] [-2.0, 2.0] [-4.0, 2.0]
fn test_simd_f4e2m1x2_to_fp16x2(ctx: DeviceContext) raises:
    print("== test_simd_f4e2m1x2_to_fp16x2")

    comptime size = 16
    var e4m21_simd = SIMD[DType.uint8, size](
        0x00,
        0x01,
        0x10,
        0x11,
        0xA6,
        0x12,
        0x38,
        0x27,
        0x60,
        0xB3,
        0x31,
        0x33,
        0x0F,
        0x15,
        0x4C,
        0x4E,
    )

    comptime kernel = test_simd_f4e2m1x2_to_fp16x2_ptx_kernel[size,]
    ctx.enqueue_function_experimental[kernel](
        e4m21_simd, grid_dim=1, block_dim=1
    )
    ctx.synchronize()


fn test_simd_f4e2m1x8_to_fp16x8_ptx_kernel[
    size: Int,
](x: SIMD[DType.uint32, size]):
    for i in range(size):
        var x_casted_0 = cast_f4e2m1x8_to_fp16x8(x[i])
        print(
            x_casted_0,
        )


# CHECK-LABEL: test_simd_f4e2m1x8_to_fp16x8
# CHECK: [0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5]
# CHECK: [4.0, -1.0, 1.0, 0.5, -0.0, 1.5, 6.0, 1.0]
# CHECK: [0.0, 4.0, 1.5, -1.5, 0.5, 1.5, 1.5, 1.5]
# CHECK: [-6.0, 0.0, 3.0, 0.5, -2.0, 2.0, -4.0, 2.0]
fn test_simd_f4e2m1x8_to_fp16x8(ctx: DeviceContext) raises:
    print("== test_simd_f4e2m1x8_to_fp16x8")

    comptime size = 4
    var e4m21_simd = SIMD[DType.uint32, size](
        0x00011011,
        0xA6123827,
        0x60B33133,
        0x0F154C4E,
    )

    comptime kernel = test_simd_f4e2m1x8_to_fp16x8_ptx_kernel[size,]
    ctx.enqueue_function_experimental[kernel](
        e4m21_simd, grid_dim=1, block_dim=1
    )
    ctx.synchronize()


fn test_simd_f4e2m1x16_to_fp16x16_ptx_kernel[
    size: Int,
](x: SIMD[DType.uint64, size]):
    for i in range(size):
        var x_casted_0 = cast_f4e2m1x16_to_fp16x16(x[i])
        print(
            x_casted_0,
        )


# CHECK-LABEL: test_simd_f4e2m1x16_to_fp16x16
# CHECK: [0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 4.0, -1.0, 1.0, 0.5, -0.0, 1.5, 6.0, 1.0]
# CHECK: [4.0, -1.0, 1.0, 0.5, -0.0, 1.5, 6.0, 1.0, 0.0, 4.0, 1.5, -1.5, 0.5, 1.5, 1.5, 1.5]
# CHECK: [0.0, 4.0, 1.5, -1.5, 0.5, 1.5, 1.5, 1.5, -6.0, 0.0, 3.0, 0.5, -2.0, 2.0, -4.0, 2.0]
# CHECK: [-6.0, 0.0, 3.0, 0.5, -2.0, 2.0, -4.0, 2.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5]
fn test_simd_f4e2m1x16_to_fp16x16(ctx: DeviceContext) raises:
    print("== test_simd_f4e2m1x16_to_fp16x16")

    comptime size = 4
    var e4m21_simd = SIMD[DType.uint64, size](
        0x00011011A6123827,
        0xA612382760B33133,
        0x60B331330F154C4E,
        0x0F154C4E00011011,
    )

    comptime kernel = test_simd_f4e2m1x16_to_fp16x16_ptx_kernel[size,]
    ctx.enqueue_function_experimental[kernel](
        e4m21_simd, grid_dim=1, block_dim=1
    )
    ctx.synchronize()


fn main() raises:
    test_simd_f32_to_e2m1()

    with DeviceContext() as ctx:
        test_simd_f32_to_e2m1_ptx_path(ctx)
        test_simd_f4e2m1x2_to_fp16x2(ctx)
        test_simd_f4e2m1x8_to_fp16x8(ctx)
        test_simd_f4e2m1x16_to_fp16x16(ctx)
