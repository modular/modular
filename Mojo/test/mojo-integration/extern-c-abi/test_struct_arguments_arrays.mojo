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
# C reference: c_abi_test_multi_args.c (MultiHfa3D {double a, b, c;}, the same
# layout)
# RUN: mkdir -p %t.dir
# RUN: mojo build -Xlinker $(dirname %s)/libc_abi_reference.lo %s -o %t.dir/test_arrays
# RUN: %t.dir/test_arrays | FileCheck %s

from std.collections import Array
from std.ffi import external_call


# ============================================================================
# Array member: Vec3d { Array[Float64, 3] } (24 bytes)
# ABI (ARM64): HFA of three doubles. The array's elements are its members, so
#              it is passed and returned in d0-d2 despite its size.
# ABI (x86-64): MEMORY class (>16 bytes).
# ============================================================================
@fieldwise_init
struct Vec3d(Copyable):
    var v: Array[Float64, 3]


# `external_call` cannot pass or return a struct that is not
# `RegisterPassable`.
@extern("c_func_hfa3d_sum")
def c_func_hfa3d_sum(s: Vec3d) abi("C") -> Float64:
    ...


@extern("c_func_hfa3d_make")
def c_func_hfa3d_make(base: Float64, step: Float64) abi("C") -> Vec3d:
    ...


def test_array_double_24byte_argument():
    var result = c_func_hfa3d_sum(Vec3d([1.0, 2.0, 3.0]))
    print("array_double_24byte_argument:", result)


# CHECK: array_double_24byte_argument: 3002001.0


def test_array_double_24byte_return():
    var result = c_func_hfa3d_make(10.0, 0.5)
    print("array_double_24byte_return:", result.v[0], result.v[1], result.v[2])


# CHECK: array_double_24byte_return: 10.5 11.0 11.5


def main():
    test_array_double_24byte_argument()
    test_array_double_24byte_return()
