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
# RUN: mojo build --emit llvm --target-triple aarch64-unknown-linux-gnu --target-cpu generic %s -o %t.ll
# RUN: FileCheck %s --check-prefix=LINUX-IR --input-file=%t.ll
# RUN: mojo build --emit llvm --target-triple arm64-apple-macosx --target-cpu apple-m1 %s -o %t.ll
# RUN: FileCheck %s --check-prefix=DARWIN-IR --input-file=%t.ll
# RUN: mojo build -Xlinker $(dirname %s)/libc_abi_reference.lo %s -o %t
# RUN: %t | FileCheck %s --check-prefixes=RUNTIME,%if system-darwin || x86_64-linux %{NEST%} %else %{NEST-FLAT%} --match-full-lines
# RUN: mojo build -O3 -Xlinker $(dirname %s)/libc_abi_reference.lo %s -o %t
# RUN: %t | FileCheck %s --check-prefixes=RUNTIME,%if system-darwin || x86_64-linux %{NEST%} %else %{NEST-FLAT%} --match-full-lines

# Structs with an alignment of 16 bytes and a size of 16 bytes, passed between
# Mojo and C in both directions. See c_abi_test_align16.c for the C rules. Each
# function returns a checksum of the arguments that the alignment can move.
#
# Darwin passes all three structs as i128. Linux uses the natural alignment,
# so A16P16 and A16S16 stay as two i64 registers.
# TODO(MOCO-5129): LowerLIT flattens A16NestP16 into A16P16, so AArch64 Linux
# passes it as A16P16. Clang passes it as i128. Check the AArch64 Linux values
# when LowerLIT keeps the struct.
# LINUX-IR-DAG: define {{.*}}i64 @mojo_align16_reg_p16(i64 noundef %{{[0-9]+}}, i64 %{{[0-9]+}}, i64 %{{[0-9]+}}, i64 noundef %{{[0-9]+}},
# LINUX-IR-DAG: define {{.*}}i64 @mojo_align16_reg_s16(i64 noundef %{{[0-9]+}}, i64 %{{[0-9]+}}, i64 %{{[0-9]+}}, i64 noundef %{{[0-9]+}},
# LINUX-IR-DAG: define {{.*}}i64 @mojo_align16_reg_nestp16(i64 noundef %{{[0-9]+}}, i64 %{{[0-9]+}}, i64 %{{[0-9]+}}, i64 noundef %{{[0-9]+}},
# LINUX-IR-DAG: define {{.*}}i64 @mojo_align16_stack_p16({{.*}}, i64 noundef %{{[0-9]+}}, [2 x i64] %{{[0-9]+}}, i64 noundef %{{[0-9]+}})
# LINUX-IR-DAG: define {{.*}}i64 @mojo_align16_stack_s16({{.*}}, i64 noundef %{{[0-9]+}}, [2 x i64] %{{[0-9]+}}, i64 noundef %{{[0-9]+}})
# LINUX-IR-DAG: declare i64 @align16_reg_p16(i64, i64, i64, i64, i64, i64, i64, i64, i64)
# LINUX-IR-DAG: declare i64 @align16_stack_p16(i64, i64, i64, i64, i64, i64, i64, i64, i64, [2 x i64], i64)
# DARWIN-IR-DAG: define {{.*}}i64 @mojo_align16_reg_p16(i64 noundef %{{[0-9]+}}, i128 %{{[0-9]+}}, i64 noundef %{{[0-9]+}},
# DARWIN-IR-DAG: define {{.*}}i64 @mojo_align16_reg_s16(i64 noundef %{{[0-9]+}}, i128 %{{[0-9]+}}, i64 noundef %{{[0-9]+}},
# DARWIN-IR-DAG: define {{.*}}i64 @mojo_align16_reg_nestp16(i64 noundef %{{[0-9]+}}, i128 %{{[0-9]+}}, i64 noundef %{{[0-9]+}},
# DARWIN-IR-DAG: define {{.*}}i64 @mojo_align16_stack_p16({{.*}}, i64 noundef %{{[0-9]+}}, i128 %{{[0-9]+}}, i64 noundef %{{[0-9]+}})
# DARWIN-IR-DAG: define {{.*}}i64 @mojo_align16_stack_s16({{.*}}, i64 noundef %{{[0-9]+}}, i128 %{{[0-9]+}}, i64 noundef %{{[0-9]+}})
# DARWIN-IR-DAG: define {{.*}}i64 @mojo_align16_stack_nestp16({{.*}}, i64 noundef %{{[0-9]+}}, i128 %{{[0-9]+}}, i64 noundef %{{[0-9]+}})
# DARWIN-IR-DAG: declare i64 @align16_reg_p16(i64, i128, i64, i64, i64, i64, i64, i64)
# DARWIN-IR-DAG: declare i64 @align16_stack_p16(i64, i64, i64, i64, i64, i64, i64, i64, i64, i128, i64)

from std.ffi import external_call


@fieldwise_init
struct A16Pair(TrivialRegisterPassable):
    var a: Int
    var b: Int


@fieldwise_init
struct A16Wide(TrivialRegisterPassable):
    var x: UInt128


@align(16)
@fieldwise_init
struct A16P16(TrivialRegisterPassable):
    var a: Int
    var b: Int


@align(16)
@fieldwise_init
struct A16S16(TrivialRegisterPassable):
    var a: Int


@fieldwise_init
struct A16NestP16(TrivialRegisterPassable):
    var inner: A16P16


@export
def mojo_align16_reg_p16(
    a: Int, u: A16P16, b: Int, c: Int, d: Int, p: A16Pair, t: Int
) abi("C") -> Int:
    return u.a + 2 * u.b + 4 * p.a + 8 * p.b + 16 * t


@export
def mojo_align16_reg_s16(
    a: Int, u: A16S16, b: Int, c: Int, d: Int, p: A16Pair, t: Int
) abi("C") -> Int:
    return u.a + 4 * p.a + 8 * p.b + 16 * t


@export
def mojo_align16_reg_nestp16(
    a: Int, u: A16NestP16, b: Int, c: Int, d: Int, p: A16Pair, t: Int
) abi("C") -> Int:
    return u.inner.a + 2 * u.inner.b + 4 * p.a + 8 * p.b + 16 * t


@export
def mojo_align16_stack_wide(
    a0: Int,
    a1: Int,
    a2: Int,
    a3: Int,
    a4: Int,
    a5: Int,
    a6: Int,
    a7: Int,
    s: Int,
    u: A16Wide,
    t: Int,
) abi("C") -> Int:
    return s + 2 * Int(u.x) + 8 * t


@export
def mojo_align16_stack_p16(
    a0: Int,
    a1: Int,
    a2: Int,
    a3: Int,
    a4: Int,
    a5: Int,
    a6: Int,
    a7: Int,
    s: Int,
    u: A16P16,
    t: Int,
) abi("C") -> Int:
    return s + 2 * u.a + 4 * u.b + 8 * t


@export
def mojo_align16_stack_s16(
    a0: Int,
    a1: Int,
    a2: Int,
    a3: Int,
    a4: Int,
    a5: Int,
    a6: Int,
    a7: Int,
    s: Int,
    u: A16S16,
    t: Int,
) abi("C") -> Int:
    return s + 2 * u.a + 8 * t


@export
def mojo_align16_stack_nestp16(
    a0: Int,
    a1: Int,
    a2: Int,
    a3: Int,
    a4: Int,
    a5: Int,
    a6: Int,
    a7: Int,
    s: Int,
    u: A16NestP16,
    t: Int,
) abi("C") -> Int:
    return s + 2 * u.inner.a + 4 * u.inner.b + 8 * t


def main():
    var p = A16Pair(11, 22)
    var p16 = A16P16(3, 5)
    var s16 = A16S16(3)
    var nest = A16NestP16(p16)
    var wide = A16Wide(27)

    # RUNTIME: reg_p16 761 761
    print(
        "reg_p16",
        external_call["align16_reg_p16", Int](0, p16, 1, 2, 3, p, 33),
        external_call["invoke_align16_reg_p16", Int](mojo_align16_reg_p16),
    )
    # RUNTIME-NEXT: reg_s16 751 751
    print(
        "reg_s16",
        external_call["align16_reg_s16", Int](0, s16, 1, 2, 3, p, 33),
        external_call["invoke_align16_reg_s16", Int](mojo_align16_reg_s16),
    )
    # NEST-NEXT: reg_nestp16 761 761
    # NEST-FLAT-NEXT: reg_nestp16 {{-?[0-9]+ -?[0-9]+}}
    print(
        "reg_nestp16",
        external_call["align16_reg_nestp16", Int](0, nest, 1, 2, 3, p, 33),
        external_call["invoke_align16_reg_nestp16", Int](
            mojo_align16_reg_nestp16
        ),
    )
    # RUNTIME-NEXT: stack_wide 327 327
    print(
        "stack_wide",
        external_call["align16_stack_wide", Int](
            0, 1, 2, 3, 4, 5, 6, 7, 9, wide, 33
        ),
        external_call["invoke_align16_stack_wide", Int](
            mojo_align16_stack_wide
        ),
    )
    # RUNTIME-NEXT: stack_p16 299 299
    print(
        "stack_p16",
        external_call["align16_stack_p16", Int](
            0, 1, 2, 3, 4, 5, 6, 7, 9, p16, 33
        ),
        external_call["invoke_align16_stack_p16", Int](mojo_align16_stack_p16),
    )
    # RUNTIME-NEXT: stack_s16 279 279
    print(
        "stack_s16",
        external_call["align16_stack_s16", Int](
            0, 1, 2, 3, 4, 5, 6, 7, 9, s16, 33
        ),
        external_call["invoke_align16_stack_s16", Int](mojo_align16_stack_s16),
    )
    # NEST-NEXT: stack_nestp16 299 299
    # NEST-FLAT-NEXT: stack_nestp16 {{-?[0-9]+ -?[0-9]+}}
    print(
        "stack_nestp16",
        external_call["align16_stack_nestp16", Int](
            0, 1, 2, 3, 4, 5, 6, 7, 9, nest, 33
        ),
        external_call["invoke_align16_stack_nestp16", Int](
            mojo_align16_stack_nestp16
        ),
    )
