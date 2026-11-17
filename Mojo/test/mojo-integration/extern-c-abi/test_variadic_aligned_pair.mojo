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
# C reference: c_abi_test_variadic_aligned_pair.c
# RUN: mojo build --emit llvm --target-triple arm64-apple-macosx --target-cpu apple-m1 %s -o %t.darwin.ll
# RUN: FileCheck %s --check-prefix=DARWIN --input-file=%t.darwin.ll
# RUN: mojo build --emit llvm --target-triple aarch64-unknown-linux-gnu --target-cpu generic %s -o %t.linux.ll
# RUN: FileCheck %s --check-prefix=LINUX --input-file=%t.linux.ll
# RUN: mojo build --emit llvm --target-triple x86_64-unknown-linux-gnu --target-cpu x86-64 %s -o %t.x86.ll
# RUN: FileCheck %s --check-prefix=X86 --input-file=%t.x86.ll
# RUN: mojo build -Xlinker $(dirname %s)/libc_abi_reference.lo %s -o %t
# RUN: %t | FileCheck %s --check-prefix=RUNTIME --match-full-lines

# Apple's arm64 ABI passes variadic arguments on the stack, and passes a
# 16-byte-aligned composite as an i128, so it starts at a 16-byte boundary
# after the 8-byte scalar slot. AAPCS64 uses the natural alignment, which does
# not include `@align`, so Linux passes the pair as two i64, or as a [2 x i64]
# on the stack. x86-64 passes it in two registers, or by value on the stack with
# the `@align` alignment.
# DARWIN-DAG: call i64 (i32, ...) @variadic_aligned_pair_early(i32 {{.*}}, i64 {{.*}}, i128 {{.*}})
# DARWIN-DAG: call i64 (i64, i64, i64, i64, i64, i64, i64, ...) @variadic_aligned_pair_late({{.*}}, i64 {{.*}}, i128 {{.*}})
# LINUX-DAG: call i64 (i32, ...) @variadic_aligned_pair_early(i32 {{.*}}, i64 {{.*}}, i64 {{.*}}, i64 {{.*}})
# LINUX-DAG: call i64 (i64, i64, i64, i64, i64, i64, i64, ...) @variadic_aligned_pair_late({{.*}}, i64 {{.*}}, [2 x i64] {{.*}})
# X86-DAG: call i64 (i32, ...) @variadic_aligned_pair_early(i32 {{.*}}, i64 {{.*}}, i64 {{.*}}, i64 {{.*}})
# X86-DAG: call i64 (i64, i64, i64, i64, i64, i64, i64, ...) @variadic_aligned_pair_late({{.*}}, i64 {{.*}}, ptr byval({{.*}}) align 16 {{.*}})

# A composite larger than 16 bytes goes by pointer on arm64 and by value on
# x86-64, where the stack slot has the `@align` alignment.
# X86-DAG: declare i64 @aligned32_after_seven(i64, i64, i64, i64, i64, i64, i64, ptr byval({{.*}}) align 32)

from std.ffi import external_call


@align(16)
@fieldwise_init
struct AlignedPair(TrivialRegisterPassable):
    var a: Int64
    var b: Int64


@align(32)
@fieldwise_init
struct Aligned32(TrivialRegisterPassable):
    var a: Int64
    var b: Int64
    var c: Int64


# RUNTIME: early: 10203
def test_early():
    var pair = AlignedPair(2, 3)
    var result = __mlir_op.`pop.external_call`[
        func="variadic_aligned_pair_early".value,
        numFixedArgs=__mlir_attr[`1 : index`],
        _type=Int64,
    ](Int32(0), Int64(1), pair)
    print("early:", result)


# RUNTIME: late: 10203
def test_late():
    var pair = AlignedPair(2, 3)
    var result = __mlir_op.`pop.external_call`[
        func="variadic_aligned_pair_late".value,
        numFixedArgs=__mlir_attr[`7 : index`],
        _type=Int64,
    ](
        Int64(0),
        Int64(0),
        Int64(0),
        Int64(0),
        Int64(0),
        Int64(0),
        Int64(0),
        Int64(1),
        pair,
    )
    print("late:", result)


# RUNTIME: aligned32: 7123
def test_aligned32():
    var s = Aligned32(1, 2, 3)
    var result = external_call["aligned32_after_seven", Int64](
        Int64(1),
        Int64(2),
        Int64(3),
        Int64(4),
        Int64(5),
        Int64(6),
        Int64(7),
        s,
    )
    print("aligned32:", result)


def main():
    test_early()
    test_late()
    test_aligned32()
