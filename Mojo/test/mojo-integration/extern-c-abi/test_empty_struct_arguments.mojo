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
# RUN: mojo build -Xlinker $(dirname %s)/libc_abi_reference.lo %s -o %t.dir/test_empty_struct
# RUN: %t.dir/test_empty_struct | FileCheck %s
#
# An empty struct occupies no register and no stack slot. All three ways of
# reaching C must drop it from the argument list: `@extern`, `external_call`,
# and a C function pointer returned by `external_call`. If a path keeps the
# argument, `b` is read from the wrong place and the result differs from 1002.

from std.ffi import external_call


@fieldwise_init
struct Empty(TrivialRegisterPassable):
    pass


comptime EmptyStructFn = def(Int32, Empty, Int32) thin abi("C") -> Int32


@extern("c_empty_struct_extern")
def c_empty_struct_extern(a: Int32, e: Empty, b: Int32) abi("C") -> Int32:
    ...


def test_extern():
    var result = c_empty_struct_extern(1, Empty(), 2)
    print("extern:", result)


# CHECK: extern: 1002


def test_external_call():
    var result = external_call["c_empty_struct_external_call", Int32](
        Int32(1), Empty(), Int32(2)
    )
    print("external_call:", result)


# CHECK: external_call: 1002


def test_function_pointer():
    var f = external_call["c_get_empty_struct_fn", EmptyStructFn]()
    var result = f(Int32(1), Empty(), Int32(2))
    print("function_pointer:", result)


# CHECK: function_pointer: 1002


def main():
    test_extern()
    test_external_call()
    test_function_pointer()
