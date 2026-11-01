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

# Test that a `contract=on|off` fp-mode item in the `kgen.compile_offload`
# op's `emission_option` controls FP contraction of that offload kernel,
# independently of the host. Cross-compiles to NVPTX, so no GPU is required.

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
def _compile_llvm[
    func_type: TrivialRegisterPassable,
    //,
    func: func_type,
    /,
    compile_options: StaticString = "",
]() -> Info:
    var info = __mlir_op.`kgen.compile_offload`[
        target_type=a100_target(),
        emission_kind=Int(1).__mlir_index__(),  # EmitAs::LLVM
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
    # By default contraction is on: the multiply-add keeps `contract` so it
    # can fuse into an FMA.
    # CHECK: fmul contract float
    # CHECK: fadd contract float
    print(_compile_llvm[madd_kernel]().kernel)

    # `contract=off` on this offload strips contraction from its ops.
    # CHECK: fmul float
    # CHECK: fadd float
    print(_compile_llvm[madd_kernel, compile_options="contract=off"]().kernel)
