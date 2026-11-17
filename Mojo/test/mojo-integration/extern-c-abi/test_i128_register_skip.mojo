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
# RUN: FileCheck %s --check-prefixes=COMMON,LINUX --input-file=%t.ll
# RUN: mojo build --emit llvm --target-triple arm64-apple-macosx --target-cpu apple-m1 %s -o %t.ll
# RUN: FileCheck %s --check-prefixes=COMMON,DARWIN --input-file=%t.ll
# RUN: mojo build -Xlinker $(dirname %s)/libc_abi_reference.lo %s -o %t
# RUN: %t | FileCheck %s --check-prefix=RUNTIME --match-full-lines
# RUN: mojo build -O3 -Xlinker $(dirname %s)/libc_abi_reference.lo %s -o %t
# RUN: %t | FileCheck %s --check-prefix=RUNTIME --match-full-lines

from std.ffi import external_call


@fieldwise_init
struct SkipWide(TrivialRegisterPassable):
    var x: UInt128


@fieldwise_init
struct SkipSIMD(TrivialRegisterPassable):
    var x: SIMD[DType.int64, 2]


@fieldwise_init
struct SkipPair(TrivialRegisterPassable):
    var a: Int
    var b: Int


# Linux skips x1 before i128, leaving only x7 for the pair. Darwin uses x1/x2.
# LINUX-DAG: define {{.*}}i64 @mojo_i128_scalar_odd(i64 noundef %{{[0-9]+}}, i128 noundef %{{[0-9]+}}, i64 noundef %{{[0-9]+}}, i64 noundef %{{[0-9]+}}, i64 noundef %{{[0-9]+}}, [2 x i64] %{{[0-9]+}}, i64 noundef %{{[0-9]+}})
# LINUX-DAG: define {{.*}}i64 @mojo_i128_wide_odd(i64 noundef %{{[0-9]+}}, i128 noundef %{{[0-9]+}}, i64 noundef %{{[0-9]+}}, i64 noundef %{{[0-9]+}}, i64 noundef %{{[0-9]+}}, [2 x i64] %{{[0-9]+}}, i64 noundef %{{[0-9]+}})
# LINUX-DAG: declare i64 @i128_scalar_odd(i64, i128, i64, i64, i64, [2 x i64], i64)
# LINUX-DAG: declare i64 @i128_wide_odd(i64, i128, i64, i64, i64, [2 x i64], i64)
# DARWIN-DAG: define {{.*}}i64 @mojo_i128_scalar_odd(i64 noundef %{{[0-9]+}}, i128 noundef %{{[0-9]+}}, i64 noundef %{{[0-9]+}}, i64 noundef %{{[0-9]+}}, i64 noundef %{{[0-9]+}}, i64 %{{[0-9]+}}, i64 %{{[0-9]+}}, i64 noundef %{{[0-9]+}})
# DARWIN-DAG: define {{.*}}i64 @mojo_i128_wide_odd(i64 noundef %{{[0-9]+}}, i128 noundef %{{[0-9]+}}, i64 noundef %{{[0-9]+}}, i64 noundef %{{[0-9]+}}, i64 noundef %{{[0-9]+}}, i64 %{{[0-9]+}}, i64 %{{[0-9]+}}, i64 noundef %{{[0-9]+}})
# DARWIN-DAG: declare i64 @i128_scalar_odd(i64, i128, i64, i64, i64, i64, i64, i64)
# DARWIN-DAG: declare i64 @i128_wide_odd(i64, i128, i64, i64, i64, i64, i64, i64)
# COMMON-DAG: define {{.*}}i64 @mojo_i128_scalar_even(i128 noundef %{{[0-9]+}}, i64 noundef %{{[0-9]+}}, i64 noundef %{{[0-9]+}}, i64 noundef %{{[0-9]+}}, i64 %{{[0-9]+}}, i64 %{{[0-9]+}}, i64 noundef %{{[0-9]+}})
# COMMON-DAG: define {{.*}}i64 @mojo_i128_wide_even(i128 noundef %{{[0-9]+}}, i64 noundef %{{[0-9]+}}, i64 noundef %{{[0-9]+}}, i64 noundef %{{[0-9]+}}, i64 %{{[0-9]+}}, i64 %{{[0-9]+}}, i64 noundef %{{[0-9]+}})
# COMMON-DAG: declare i64 @i128_scalar_even(i128, i64, i64, i64, i64, i64, i64)
# COMMON-DAG: declare i64 @i128_wide_even(i128, i64, i64, i64, i64, i64, i64)
# COMMON-DAG: define {{.*}}i64 @mojo_i128_simd_even(<2 x i64> noundef %{{[0-9]+}}, i64 noundef %{{[0-9]+}}, i64 noundef %{{[0-9]+}}, i64 noundef %{{[0-9]+}}, i64 noundef %{{[0-9]+}}, i64 noundef %{{[0-9]+}}, i64 %{{[0-9]+}}, i64 %{{[0-9]+}}, i64 noundef %{{[0-9]+}})
# COMMON-DAG: define {{.*}}i64 @mojo_i128_simd_odd(i64 noundef %{{[0-9]+}}, <2 x i64> noundef %{{[0-9]+}}, i64 noundef %{{[0-9]+}}, i64 noundef %{{[0-9]+}}, i64 noundef %{{[0-9]+}}, i64 noundef %{{[0-9]+}}, i64 noundef %{{[0-9]+}}, i64 %{{[0-9]+}}, i64 %{{[0-9]+}}, i64 noundef %{{[0-9]+}})
# COMMON-DAG: declare i64 @i128_simd_even(<2 x i64>, i64, i64, i64, i64, i64, i64, i64, i64)
# COMMON-DAG: declare i64 @i128_simd_odd(i64, <2 x i64>, i64, i64, i64, i64, i64, i64, i64, i64)


@export
def mojo_i128_scalar_even(
    u: UInt128, b: Int, c: Int, d: Int, pair: SkipPair, tail: Int
) abi("C") -> Int:
    return pair.a + 2 * pair.b + 4 * tail


@export
def mojo_i128_scalar_odd(
    a: Int, u: UInt128, b: Int, c: Int, d: Int, pair: SkipPair, tail: Int
) abi("C") -> Int:
    return pair.a + 2 * pair.b + 4 * tail


@export
def mojo_i128_wide_even(
    u: SkipWide, b: Int, c: Int, d: Int, pair: SkipPair, tail: Int
) abi("C") -> Int:
    return pair.a + 2 * pair.b + 4 * tail


@export
def mojo_i128_wide_odd(
    a: Int, u: SkipWide, b: Int, c: Int, d: Int, pair: SkipPair, tail: Int
) abi("C") -> Int:
    return pair.a + 2 * pair.b + 4 * tail


@export
def mojo_i128_simd_even(
    u: SkipSIMD,
    b: Int,
    c: Int,
    d: Int,
    e: Int,
    f: Int,
    pair: SkipPair,
    tail: Int,
) abi("C") -> Int:
    return pair.a + 2 * pair.b + 4 * tail


@export
def mojo_i128_simd_odd(
    a: Int,
    u: SkipSIMD,
    b: Int,
    c: Int,
    d: Int,
    e: Int,
    f: Int,
    pair: SkipPair,
    tail: Int,
) abi("C") -> Int:
    return pair.a + 2 * pair.b + 4 * tail


def main():
    var u = UInt128(27)
    var wide = SkipWide(u)
    var simd = SkipSIMD(SIMD[DType.int64, 2](27, 28))
    var pair = SkipPair(11, 22)
    print(external_call["i128_scalar_even", Int](u, 1, 2, 3, pair, 33))
    print(external_call["invoke_i128_scalar_even", Int](mojo_i128_scalar_even))
    print(external_call["i128_scalar_odd", Int](0, u, 1, 2, 3, pair, 33))
    print(external_call["invoke_i128_scalar_odd", Int](mojo_i128_scalar_odd))
    print(external_call["i128_wide_even", Int](wide, 1, 2, 3, pair, 33))
    print(external_call["invoke_i128_wide_even", Int](mojo_i128_wide_even))
    print(external_call["i128_wide_odd", Int](0, wide, 1, 2, 3, pair, 33))
    print(external_call["invoke_i128_wide_odd", Int](mojo_i128_wide_odd))
    print(external_call["i128_simd_even", Int](simd, 1, 2, 3, 4, 5, pair, 33))
    print(external_call["invoke_i128_simd_even", Int](mojo_i128_simd_even))
    print(external_call["i128_simd_odd", Int](0, simd, 1, 2, 3, 4, 5, pair, 33))
    print(external_call["invoke_i128_simd_odd", Int](mojo_i128_simd_odd))
    # RUNTIME: 187
    # RUNTIME-NEXT: 187
    # RUNTIME-NEXT: 187
    # RUNTIME-NEXT: 187
    # RUNTIME-NEXT: 187
    # RUNTIME-NEXT: 187
    # RUNTIME-NEXT: 187
    # RUNTIME-NEXT: 187
    # RUNTIME-NEXT: 187
    # RUNTIME-NEXT: 187
    # RUNTIME-NEXT: 187
    # RUNTIME-NEXT: 187
