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

# ===----------------------------------------------------------------------=== #
#
# Integration test for linking an LLVM bitcode library that uses @llvm.compiler.used.
#
# ===----------------------------------------------------------------------=== #

# RUN: llvm-as %S/inputs/bitcode_llvm_compiler_used.ll -o %t.bc
# RUN: mojo --bitcode-libs=%t.bc %s | FileCheck %s


from std.compile import compile_info
from std.sys.info import CompilationTarget
from max.gpu.host.info import _a100_target


@extern("my_i32_inc")
def my_i32_inc(x: Pointer[Int32, MutAnyOrigin]) abi("Mojo"):
    ...


@extern("my_i32_dec")
def my_i32_dec(x: Pointer[Int32, MutAnyOrigin]) abi("Mojo"):
    ...


def test_inc():
    var x: Int32 = 10
    my_i32_inc(Pointer(to=x).as_unsafe_any_origin())
    print(x)


def test_dec():
    var x: Int32 = 10
    my_i32_dec(Pointer(to=x).as_unsafe_any_origin())
    print(x)


def test_no_extern_usage():
    pass


# Each kernel should land in a separate submodule. Each submodule should
# include a copy of @llvm.compiler.used.
def main():
    # CHECK: target triple = "nvptx64-nvidia-cuda"
    # CHECK: @llvm.compiler.used = appending global [2 x ptr]
    print(
        compile_info[
            test_inc,
            emission_kind="llvm",
            target=_a100_target,
        ]()
    )

    # CHECK: target triple = "nvptx64-nvidia-cuda"
    # CHECK: @llvm.compiler.used = appending global [2 x ptr]
    print(
        compile_info[
            test_dec,
            emission_kind="llvm",
            target=_a100_target,
        ]()
    )

    # CHECK: target triple = "nvptx64-nvidia-cuda"
    # CHECK-NOT: @llvm.compiler.used
    print(
        compile_info[
            test_no_extern_usage,
            emission_kind="llvm",
            target=_a100_target,
        ]()
    )
