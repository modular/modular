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

# RUN: %mojo %s | FileCheck %s

from std.collections.string.string_span import StaticString, _get_kgen_string
from max.gpu.host.info import A100
from std.sys.info import CompilationTarget


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


# This is an exported function that is not kernel entry.
# This function should not compile into a ptx_kernel function.
@no_inline
@export
def world() abi("C"):
    pass


@no_inline
@export
def hello() abi("C"):
    world()
    pass


comptime a = 42


@no_inline
@__name("goodbye" + String(a))
def hello2():
    world()
    pass


@no_inline
@__name("ciao" + String(a) + "_" + String(n))
def hello3[n: Int]():
    world()
    pass


@no_inline
@__name("42.sanitize/this_" + String(n))
def hello4[n: Int]():
    world()
    pass


def main():
    var t0 = _compile_info[hello, emission_kind=1]()
    print(t0.kernel)
    var t1 = _compile_info[hello2, emission_kind=1]()
    print(t1.kernel)
    var t2 = _compile_info[hello3[100], emission_kind=1]()
    print(t2.kernel)
    var t3 = _compile_info[hello4[42], emission_kind=1]()
    print(t3.kernel)


# CHECK: internal void @world
# CHECK: ptx_kernel void @hello()
# CHECK: ptx_kernel void @goodbye42_{{[0-9a-f]+}}()
# CHECK: ptx_kernel void @ciao42_100_{{[0-9a-f]+}}()
# CHECK: ptx_kernel void @_42_sanitize_this_42_{{[0-9a-f]+}}()
