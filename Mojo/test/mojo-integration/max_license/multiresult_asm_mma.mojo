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
# RUN: kgen -emit=asm --target-triple=nvptx64-nvidia-cuda --target-cpu=sm_89 --target-features="" %s -o - | FileCheck %s

from std.memory.unsafe import bitcast
from std.utils import StaticTuple


@always_inline
def _SM80_16x8x8_F16F16F16F16_TN(
    mut d00: Float16,
    mut d01: Float16,
    mut d10: Float16,
    mut d11: Float16,
    a00: Float16,
    a01: Float16,
    a10: Float16,
    a11: Float16,
    b00: Float16,
    b01: Float16,
    c00: Float16,
    c01: Float16,
    c10: Float16,
    c11: Float16,
):
    var a0 = SIMD[.float16, 2](a00, a01)
    var a1 = SIMD[.float16, 2](a10, a11)
    var b0 = SIMD[.float16, 2](b00, b01)
    var c0 = SIMD[.float16, 2](c00, c01)
    var c1 = SIMD[.float16, 2](c10, c11)

    # CHECK: mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16
    var r = __mlir_op.`pop.inline_asm`[
        assembly=(
            "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {$0, $1},{$2,"
            " $3},{$4},{$5, $6};"
        ).value,
        _type=Tuple[UInt32, UInt32],
        constraints="=r,=r,r,r,r,r,r".value,
        hasSideEffects=True.__mlir_i1__(),
        isStackAligned=False.__mlir_i1__(),
    ](
        bitcast[.uint32, 1](a0),
        bitcast[.uint32, 1](a1),
        bitcast[.uint32, 1](b0),
        bitcast[.uint32, 1](c0),
        bitcast[.uint32, 1](c1),
    )
    var d0 = bitcast[.float16, 2](r[0])
    var d1 = bitcast[.float16, 2](r[0])

    d00 = d0[0]
    d01 = d0[1]

    d10 = d1[0]
    d11 = d1[1]


@export
def SM80_16x8x8_F16F16F16F16_TN(
    a00: Float16,
    a01: Float16,
    a10: Float16,
    a11: Float16,
    b00: Float16,
    b01: Float16,
    c00: Float16,
    c01: Float16,
    c10: Float16,
    c11: Float16,
) abi("Mojo") -> StaticTuple[Float16, 4]:
    var r00 = Float16(0)
    var r01 = Float16(0)
    var r02 = Float16(0)
    var r04 = Float16(0)
    _SM80_16x8x8_F16F16F16F16_TN(
        r00, r01, r02, r04, a00, a01, a10, a11, b00, b01, c00, c01, c10, c11
    )

    return StaticTuple[Float16, 4](r00, r01, r02, r04)
