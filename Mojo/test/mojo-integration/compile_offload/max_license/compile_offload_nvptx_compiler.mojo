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

# REQUIRES: NVIDIA-GPU
# UNSUPPORTED: asan, ubsan
# RUN: kgen --emit=object -kgen-debug-only=object-compiler %s -o %t.o 2>&1 | FileCheck %s --check-prefix=CHECK
# COM: run with env MODULAR_NVPTX_COMPILER_PATH should invalidate the cache generated without this env var.
# RUN: env MODULAR_NVPTX_COMPILER_PATH=/my/path not kgen --emit=object -kgen-debug-only=object-compiler %s -o %t.s 2>&1 | FileCheck %s --check-prefix=CHECK-PTXAS-PATH


from std.sys.info import CompilationTarget
from max.gpu.host import get_gpu_target
from std.collections.string.string_span import StaticString, _get_kgen_string


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
    target: CompilationTarget = get_gpu_target(),
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
        kernel=StaticString(info.kernel),
        name=StaticString(info.name),
    )


def hello_nvptx():
    pass


def main() raises:
    var t1 = _compile_info[hello_nvptx, emission_kind=3]()
    print(t1.kernel)


# CHECK: Using the NVPTXCompiler API to compile PTX to CUBIN
# CHECK-PTXAS-PATH-NOT: Using the NVPTXCompiler API to compile PTX to CUBIN
# CHECK-PTXAS-PATH: /my/path
