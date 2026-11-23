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
# C reference: c_abi_test_empty_struct.c
# RUN: mkdir -p %t.dir
# RUN: mojo build -Xlinker $(dirname %s)/libc_abi_reference.lo %s -o %t.dir/test_empty_struct_arguments
# RUN: %t.dir/test_empty_struct_arguments | FileCheck %s
#
# An empty struct between two arguments takes no register or stack slot, as in
# C, on each path across the C boundary (MOCO-5490).

from std.ffi import external_call


@fieldwise_init
struct Empty(Copyable, RegisterPassable):
    pass


comptime EmptyBetweenFn = def(Int32, Empty, Int32) thin abi("C") -> Int64


@extern("c_empty_between_extern")
def c_empty_between_extern(a: Int32, e: Empty, b: Int32) abi("C") -> Int64:
    ...


@export
def mojo_empty_between(a: Int32, e: Empty, b: Int32) abi("C") -> Int64:
    return Int64(a) * 1000 + Int64(b)


def main():
    print("extern:", c_empty_between_extern(1, Empty(), 2))
    # CHECK: extern: 1002

    print(
        "external_call:",
        external_call["c_empty_between", Int64](Int32(1), Empty(), Int32(2)),
    )
    # CHECK: external_call: 1002

    var fn_ptr = external_call["c_get_empty_between", EmptyBetweenFn]()
    print("pointer:", fn_ptr(1, Empty(), 2))
    # CHECK: pointer: 1002

    print(
        "c_invoke:",
        external_call["c_invoke_empty_between", Int64](mojo_empty_between),
    )
    # CHECK: c_invoke: 3004
