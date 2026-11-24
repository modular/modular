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
#
# Test linking an external bitcode file.
#
# ===----------------------------------------------------------------------=== #

# Can only run with asan version of amd drivers installed.
# UNSUPPORTED: asan

# RUN: %mojo %S/inputs/define_extern_bc_func_offload.mojo > %t_offload.ll
# RUN: llvm-as %t_offload.ll -o %t_offload.bc
# RUN: kgen %S/../inputs/define_extern_bc_func.mojo -emit=llvm -o %t_native.ll
# RUN: llvm-as %t_native.ll -o %t_native.bc
# RUN: %mojo-build --bitcode-libs=%t_offload.bc --bitcode-libs=%t_native.bc %s -o %t
# RUN: %t | FileCheck %s

from std.compile import compile_info
from max.gpu.host.info import _mi355x_target


@extern("my_add_one")
def my_add_one(x: Pointer[Int32, MutAnyOrigin]) abi("Mojo"):
    ...


def my_add_two(y: Pointer[Int32, MutAnyOrigin]):
    my_add_one(y)
    my_add_one(y)


def main():
    # Use native.
    var ten: Int32 = 10
    my_add_one(Pointer(to=ten).as_unsafe_any_origin())
    print(ten)
    # CHECK: 11

    # Use offload.
    var asm = compile_info[
        my_add_two,
        emission_kind="llvm",
        target=_mi355x_target,
    ]()
    print(asm)
    # CHECK: call void @my_add_one
    # CHECK: call void @my_add_one
    # CHECK-LABEL: define {{.*}} @my_add_one
