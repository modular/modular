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
# C reference: c_abi_test_union.c
# RUN: mkdir -p %t.dir
# RUN: mojo build -Xlinker $(dirname %s)/libc_abi_reference.lo %s -o %t.dir/test_union_arguments
# RUN: %t.dir/test_union_arguments | FileCheck %s
#
# A __union crossing the C boundary by value, in argument and result
# position, in both directions: Mojo calls C through external_call, and C
# calls back into Mojo @export abi("C") functions through pointers.

from std.ffi import external_call


__union U(TrivialRegisterPassable):
    case i(Int)
    case f(Float64)


@export
def mojo_take_u(u: U) abi("C") -> Int:
    match u:
        case .i(v):
            return v * 10 + u._get_enum_discriminant()
        case .f(x):
            return 500


@export
def mojo_give_u() abi("C") -> U:
    return U.f(1.5)


def main():
    # Mojo builds a union, C reads it by value.
    var a = U.i(42)
    print("c_take_u:", external_call["c_take_u", Int](a))
    # CHECK: c_take_u: 420

    # C builds a union, Mojo reads it by value.
    var b = external_call["c_give_u", U]()
    match b:
        case .i(v):
            print("c_give_u: i", v)
        case .f(x):
            print("c_give_u: f", x)
    # CHECK: c_give_u: f 2.5

    # C calls a Mojo abi("C") function through a pointer, passing by value.
    print("c_invoke_take:", external_call["c_invoke_take", Int](mojo_take_u))
    # CHECK: c_invoke_take: 420

    # C calls a Mojo abi("C") function through a pointer, reading its
    # by-value result.
    print(
        "c_invoke_give:", external_call["c_invoke_give", Float64](mojo_give_u)
    )
    # CHECK: c_invoke_give: 1.5
