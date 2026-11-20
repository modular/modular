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

# REQUIRES: NVIDIA-GPU || AMD-GPU
# RUN: %mojo %s | FileCheck %s

from std.collections.string.string_span import StaticString, _get_kgen_string
from max.gpu.host.info import A100
from std.sys.info import CompilationTarget
from std.math import log2


@fieldwise_init
struct _Info(TrivialRegisterPassable):
    var kernel: __mlir_type.`!kgen.string`
    var name: __mlir_type.`!kgen.string`


@fieldwise_init
struct Info(TrivialRegisterPassable):
    var kernel: StaticString
    var name: StaticString


@always_inline
def _compile_info[
    func_type: TrivialRegisterPassable,
    //,
    func: func_type,
    /,
    emission_kind: Int = 0,
    target: CompilationTarget = A100.target,
    compile_options: StaticString = target.default_compile_options(),
]() -> Info:
    var info = __mlir_op.`kgen.compile_offload`[
        target_type=target._mlir_value,
        emission_kind=emission_kind.__mlir_index__(),
        emission_option=_get_kgen_string[compile_options](),
        emission_link_option=_get_kgen_string[""](),
        func=func,
        _type=_Info,
    ]()

    return Info(
        StaticString(info.kernel),
        StaticString(info.name),
    )


@no_inline
def test():
    # the following for loop is evaluated during compilation time
    # log2 should run the cpu version instead of gpu inline asm.
    comptime for i in range(Int(log2(Float32(8)))):
        print(i)


def main():
    var t0 = _compile_info[test, emission_kind=1]()
    print(t0.kernel)


# CHECK: ptx_kernel void @{{.*}}test
