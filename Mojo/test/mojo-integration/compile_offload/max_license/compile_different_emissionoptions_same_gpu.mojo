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

# RUN: kgen -elaborate -kgen-debug-only=kgen-compiler %s -o /dev/null 2>&1 | FileCheck %s

from std.sys.info import _TargetType
from std.collections.string.string_span import _get_kgen_string


@fieldwise_init
struct _Info(TrivialRegisterPassable):
    var kernel: __mlir_type.`!kgen.string`
    var name: __mlir_type.`!kgen.string`


@fieldwise_init
struct Info(TrivialRegisterPassable):
    var kernel: StaticString
    var name: StaticString


def get_a100_target_no_short_ptr() -> _TargetType:
    """
    Creates an MLIR target configuration for NVIDIA A100 GPU.

    Returns:
        MLIR target configuration for A100.
    """

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
def _compile_info[
    func_type: TrivialRegisterPassable,
    //,
    func: func_type,
    /,
    emission_kind: Int,
    compile_options: StaticString,
    target: _TargetType = get_a100_target_no_short_ptr(),
]() -> Info:
    var info = __mlir_op.`kgen.compile_offload`[
        target_type=target,
        emission_kind=emission_kind.__mlir_index__(),
        emission_option=_get_kgen_string[compile_options](),
        emission_link_option=_get_kgen_string[""](),
        func=func,
        _type=_Info,
    ]()

    return Info(
        kernel=StaticString(info.kernel),
        name=StaticString(info.name),
    )


def hello_nvptx():
    pass


def main() raises:
    var t1 = _compile_info[hello_nvptx, compile_options="", emission_kind=0]()
    print(t1.kernel)

    var t2 = _compile_info[
        hello_nvptx,
        compile_options="greedy-reverse-local-assignment=true",
        emission_kind=0,
    ]()
    print(t2.kernel)

    var t3 = _compile_info[
        hello_nvptx,
        compile_options="disable-machine-cse=true",
        emission_kind=0,
    ]()
    print(t3.kernel)


# CHECK: Emit offloads with options:
# CHECK: Emit offloads with options: disable-machine-cse=true
# CHECK: Emit offloads with options: greedy-reverse-local-assignment=true
