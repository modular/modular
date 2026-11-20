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

# Check the RDNA WMMA A/B operand width per generation: gfx12 halves it vs
# gfx11, and a wrong width fails instruction selection. `compile_offload` pins
# the kernel target, so this cross-compiles without a GPU.

# UNSUPPORTED: asan

# RUN: %mojo %s | FileCheck %s

from std.collections.string.string_span import StaticString, _get_kgen_string
from max.gpu.compute.mma import mma
from std.sys.info import _TargetType
from max._plugin.hip.targets import _7900_target, _9070_target


@fieldwise_init
struct _Info(TrivialRegisterPassable):
    var kernel: __mlir_type.`!kgen.string`
    var name: __mlir_type.`!kgen.string`


def gfx1201_target() -> _TargetType:
    return _9070_target._mlir_value


def gfx1100_target() -> _TargetType:
    return _7900_target._mlir_value


@always_inline
def _emit[
    func_type: TrivialRegisterPassable,
    //,
    func: func_type,
    /,
    target: _TargetType,
]() -> StaticString:
    var info = __mlir_op.`kgen.compile_offload`[
        target_type=target,
        emission_kind=Int(1).__mlir_index__(),  # EmitAs::LLVM
        emission_option=_get_kgen_string[""](),
        emission_link_option=_get_kgen_string[""](),
        func=func,
        _type=_Info,
    ]()
    return StaticString(info.kernel)


def f16_kernel(dst: Pointer[Float32, MutAnyOrigin]):
    var acc = SIMD[.float32, 8](0.0)
    var a = SIMD[.float16, 16](1.0)
    var b = SIMD[.float16, 16](1.0)
    mma(acc, a, b, acc)
    dst[unsafe_offset=0] = acc[0]


def bf16_kernel(dst: Pointer[Float32, MutAnyOrigin]):
    var acc = SIMD[.float32, 8](0.0)
    var a = SIMD[.bfloat16, 16](1.0)
    var b = SIMD[.bfloat16, 16](1.0)
    mma(acc, a, b, acc)
    dst[unsafe_offset=0] = acc[0]


def int8_kernel(dst: Pointer[Int32, MutAnyOrigin]):
    var acc = SIMD[.int32, 8](0)
    var a = SIMD[.int8, 16](1)
    var b = SIMD[.int8, 16](1)
    mma(acc, a, b, acc)
    dst[unsafe_offset=0] = acc[0]


def uint4_kernel(dst: Pointer[Int32, MutAnyOrigin]):
    var acc = SIMD[.int32, 8](0)
    var a = SIMD[._uint4, 16](1)
    var b = SIMD[._uint4, 16](1)
    mma(acc, a, b, acc)
    dst[unsafe_offset=0] = acc[0]


def fp8_kernel(dst: Pointer[Float32, MutAnyOrigin]):
    var acc = SIMD[.float32, 8](0.0)
    var a = SIMD[.float8_e4m3fn, 16](1.0)
    var b = SIMD[.float8_e4m3fn, 16](1.0)
    mma(acc, a, b, acc)
    dst[unsafe_offset=0] = acc[0]


def main():
    # gfx12 (RDNA4) uses the narrow, non-replicated A/B fragments.
    # CHECK: wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half>
    print(_emit[f16_kernel, target=gfx1201_target()]())
    # CHECK: wmma.f32.16x16x16.bf16.v8f32.v8i16(<8 x i16>
    print(_emit[bf16_kernel, target=gfx1201_target()]())
    # CHECK: wmma.i32.16x16x16.iu8.v8i32.v2i32(i1 true, <2 x i32>
    print(_emit[int8_kernel, target=gfx1201_target()]())
    # CHECK: wmma.i32.16x16x16.iu4.v8i32.i32(i1 false, i32
    print(_emit[uint4_kernel, target=gfx1201_target()]())
    # CHECK: wmma.f32.16x16x16.fp8.fp8.v8f32.v2i32(<2 x i32>
    print(_emit[fp8_kernel, target=gfx1201_target()]())

    # gfx11 (RDNA3) uses the wide, replicated A/B fragments (no native fp8).
    # CHECK: wmma.f32.16x16x16.f16.v8f32.v16f16(<16 x half>
    print(_emit[f16_kernel, target=gfx1100_target()]())
    # CHECK: wmma.f32.16x16x16.bf16.v8f32.v16i16(<16 x i16>
    print(_emit[bf16_kernel, target=gfx1100_target()]())
    # CHECK: wmma.i32.16x16x16.iu8.v8i32.v4i32(i1 true, <4 x i32>
    print(_emit[int8_kernel, target=gfx1100_target()]())
    # CHECK: wmma.i32.16x16x16.iu4.v8i32.v2i32(i1 false, <2 x i32>
    print(_emit[uint4_kernel, target=gfx1100_target()]())
