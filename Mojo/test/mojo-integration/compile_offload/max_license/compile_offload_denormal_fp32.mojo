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

# Test that a `denormal-fp-math-f32=<mode>` item in the `kgen.compile_offload`
# op's `emission_option` sets the f32 denormal mode of that offload kernel.
# Cross-compiles to NVPTX, so no GPU is required.

# RUN: %mojo %s | FileCheck %s

from std.collections.string.string_span import StaticString, _get_kgen_string
from std.sys.info import _TargetType


@fieldwise_init
struct _Info(TrivialRegisterPassable):
    var kernel: __mlir_type.`!kgen.string`
    var name: __mlir_type.`!kgen.string`


@fieldwise_init
struct Info(TrivialRegisterPassable):
    var kernel: StaticString
    var name: StaticString


def a100_target() -> _TargetType:
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_80", `,
        `features = "+ptx81,+sm_80", `,
        `tune_cpu = "sm_80", `,
        `data_layout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128,`,
        `index_bit_width = 64`,
        `> : !kgen.target`,
    ]


@always_inline
def _compile_asm[
    func_type: TrivialRegisterPassable,
    //,
    func: func_type,
    /,
    compile_options: StaticString = "",
]() -> Info:
    var info = __mlir_op.`kgen.compile_offload`[
        target_type=a100_target(),
        emission_kind=Int(0).__mlir_index__(),  # EmitAs::ASM
        emission_option=_get_kgen_string[compile_options](),
        emission_link_option=_get_kgen_string[""](),
        func=func,
        _type=_Info,
    ]()
    return Info(
        StaticString(info.kernel),
        StaticString(info.name),
    )


def madd_kernel(a: Float32, b: Float32, c: Float32) -> Float32:
    return a * b + c


def main():
    # By default f32 denormals are kept, so the FMA does not flush them.
    # CHECK-LABEL: default:
    # CHECK: fma.rn.f32
    print("default:")
    print(_compile_asm[madd_kernel]().kernel)

    # `preserve-sign` flushes f32 denormals to zero.
    # CHECK-LABEL: preserve-sign:
    # CHECK: fma.rn.ftz.f32
    print("preserve-sign:")
    print(
        _compile_asm[
            madd_kernel, compile_options="denormal-fp-math-f32=preserve-sign"
        ]().kernel
    )
