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
# C reference: c_abi_test_rollback.c
# RUN: mkdir -p %t.dir
# RUN: mojo build -Xlinker $(dirname %s)/libc_abi_reference.lo %s -o %t.dir/test_rollback
# RUN: %t.dir/test_rollback | FileCheck %s
# RUN: mojo build --emit llvm --target-triple x86_64-unknown-linux-gnu --target-cpu x86-64 %s -o %t.x86.ll
# RUN: FileCheck %s --check-prefix=X86 --input-file=%t.x86.ll

from std.ffi import external_call
from std.memory import Pointer, alloc, dealloc


# 12-byte struct: two eightbytes, both INTEGER class (IntegerPair).
@fieldwise_init
struct Int3(TrivialRegisterPassable):
    var x: Int32
    var y: Int32
    var z: Int32


# Control: struct passed first, so it fits in registers.
def test_struct_early():
    var v = Int3(Int32(11), Int32(22), Int32(33))
    var status = external_call["check_struct_early", Int32](v)
    print("struct_early:", status)


# CHECK: struct_early: 0


# Struct passed after five integer-class args: must roll back to the stack.
# Under the old per-argument lowering the struct was split, corrupting it.
def test_struct_after_five():
    var p0_alloc = alloc[Int32]({length = 1})
    var p0 = p0_alloc.unsafe_ptr()
    var p1_alloc = alloc[Int32]({length = 1})
    var p1 = p1_alloc.unsafe_ptr()
    var v = Int3(Int32(11), Int32(22), Int32(33))
    var status = external_call["check_struct_after_five", Int32](
        p0, p1, Int32(101), Int32(202), Int32(303), v
    )
    print("struct_after_five:", status)
    dealloc(p0_alloc^)
    dealloc(p1_alloc^)


# CHECK: struct_after_five: 0


# Variadic args use the same registers as fixed args, so a variadic struct
# after five integer-class args rolls back to the stack too.
# X86: call i32 (i32, ...) @check_variadic_struct_after_five(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, ptr byval({{.*}}) {{.*}})
def test_variadic_struct_after_five():
    var v = Int3(Int32(11), Int32(22), Int32(33))
    var status = __mlir_op.`pop.external_call`[
        func="check_variadic_struct_after_five".value,
        numFixedArgs=__mlir_attr[`1 : index`],
        _type=Int32,
    ](Int32(1), Int32(2), Int32(3), Int32(4), Int32(5), v)
    print("variadic_struct_after_five:", status)


# CHECK: variadic_struct_after_five: 0


# Struct passed after seven integer-class args: on arm64 seven GPRs leave
# only X7, and on x86-64 none remain, so the struct must roll back to the
# stack on both. Under the old per-argument lowering the struct was split
# between a register and the stack, corrupting it.
def test_struct_after_seven():
    var v = Int3(Int32(11), Int32(22), Int32(33))
    var status = external_call["check_struct_after_seven", Int32](
        Int32(1),
        Int32(2),
        Int32(3),
        Int32(4),
        Int32(5),
        Int32(6),
        Int32(7),
        v,
    )
    print("struct_after_seven:", status)


# CHECK: struct_after_seven: 0


# Trailing argument after the rolled-back struct: AAPCS64 exhausts the
# register search, so the trailing int goes to the stack too. A lowering
# that left the last register for it (a byval pointer) corrupts it.
def test_struct_then_int_after_seven():
    var v = Int3(Int32(11), Int32(22), Int32(33))
    var status = external_call["check_struct_then_int_after_seven", Int32](
        Int32(1),
        Int32(2),
        Int32(3),
        Int32(4),
        Int32(5),
        Int32(6),
        Int32(7),
        v,
        Int32(44),
    )
    print("struct_then_int_after_seven:", status)


# CHECK: struct_then_int_after_seven: 0


def main():
    test_struct_early()
    test_struct_after_five()
    test_variadic_struct_after_five()
    test_struct_after_seven()
    test_struct_then_int_after_seven()
