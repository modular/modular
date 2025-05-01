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
# REQUIRES: NVIDIA-GPU
# RUN: %mojo-no-debug %s

from sys.param_env import is_defined

from gpu.host._compile import _compile_code_asm, _get_gpu_target
from gpu.mma import mma
from testing import *


# CHECK-LABEL: SM80_16x8x8_F16F16F16F16_TN
fn SM80_16x8x8_F16F16F16F16_TN(
    a: SIMD[DType.float16, 4],
    b: SIMD[DType.float16, 2],
    c: SIMD[DType.float16, 4],
) -> SIMD[DType.float16, 4]:
    var d = SIMD[DType.float16, 4]()
    mma(d, a, b, c)
    return d


def test_SM80_16x8x8_F16F16F16F16_TN():
    var asm = _compile_code_asm[SM80_16x8x8_F16F16F16F16_TN]()
    assert_true("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16" in asm)
    assert_true("{%r6, %r7}," in asm)
    assert_true("{%r2, %r3}," in asm)
    assert_true("{%r1}," in asm)
    assert_true("{%r4, %r5};" in asm)


# CHECK-LABEL: SM80_m16n8k4_F32TF32TF32F32_TN
fn SM80_m16n8k4_F32TF32TF32F32_TN(
    a: SIMD[DType.float32, 2],
    b: Float32,
    c: SIMD[DType.float32, 4],
) -> SIMD[DType.float32, 4]:
    var d = SIMD[DType.float32, 4]()
    mma(d, a, b, c)
    return d


def test_SM80_m16n8k4_F32TF32TF32F32_TN():
    var asm = _compile_code_asm[SM80_m16n8k4_F32TF32TF32F32_TN]()
    assert_true("mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32" in asm)
    assert_true("{%f7, %f8, %f9, %f10}," in asm)
    assert_true("{%r2, %r1}," in asm)
    assert_true("{%r3}," in asm)
    assert_true("{%f3, %f4, %f5, %f6};" in asm)


fn SM80_m16n8k8_F32TF32TF32F32_TN(
    a: SIMD[DType.float32, 4],
    b: Float32,
    c: SIMD[DType.float32, 4],
) -> SIMD[DType.float32, 4]:
    var d = SIMD[DType.float32, 4]()
    mma(d, a, b.join(b), c)
    return d


def test_SM80_m16n8k8_F32TF32TF32F32_TN():
    var asm = _compile_code_asm[SM80_m16n8k8_F32TF32TF32F32_TN]()
    assert_true("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32" in asm)
    assert_true("{%f5, %f6, %f7, %f8}," in asm)
    assert_true("{%r1, %r2, %r3, %r4}," in asm)
    assert_true("{%r5, %r5}," in asm)
    assert_true("{%f1, %f2, %f3, %f4};" in asm)


fn SM80_m16n8k32_F8E4M3F8E4M_TN(
    a: SIMD[DType.float8_e4m3fn, 16],
    b: SIMD[DType.float8_e4m3fn, 8],
    c: SIMD[DType.float32, 4],
) -> SIMD[DType.float32, 4]:
    var d = SIMD[DType.float32, 4]()
    mma(d, a, b, c)
    return d


def test_SM80_m16n8k8_F8E4M3F8E4M3_TN():
    var asm = _compile_code_asm[
        SM80_m16n8k32_F8E4M3F8E4M_TN, target = _get_gpu_target["sm_90"]()
    ]()
    assert_true("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32" in asm)
    assert_true("{%f1, %f2, %f3, %f4}" in asm)
    assert_true("{%r1, %r2, %r3, %r4}" in asm)
    assert_true("{%r5, %r6}, {%r7, %r8, %r9, %r10}" in asm)


def main():
    test_SM80_16x8x8_F16F16F16F16_TN()
    test_SM80_m16n8k4_F32TF32TF32F32_TN()
    test_SM80_m16n8k8_F32TF32TF32F32_TN()
    test_SM80_m16n8k8_F8E4M3F8E4M3_TN()
