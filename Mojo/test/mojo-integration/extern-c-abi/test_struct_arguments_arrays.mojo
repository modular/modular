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
# C reference: c_abi_test_array_structs.c
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


# `external_call` cannot return a struct that is not `RegisterPassable`.
@extern("c_func_array_double_24byte")
def c_func_array_double_24byte(s: Vec3d) abi("C") -> Vec3d:
    ...


def test_array_double_24byte():
    var result = c_func_array_double_24byte(Vec3d([1.0, 2.0, 3.0]))
    print("array_double_24byte:", result.v[0], result.v[1], result.v[2])


# CHECK: array_double_24byte: 2.0 3.0 4.0


def weigh_vec3d(s: Vec3d) abi("C") -> Float64:
    return s.v[0] + 10 * s.v[1] + 100 * s.v[2]


def test_array_double_24byte_callback():
    # C builds the Vec3d and passes it to an abi("C") Mojo function.
    var result = external_call["c_call_array_double_24byte", Float64](
        weigh_vec3d
    )
    print("array_double_24byte_callback:", result)


# CHECK: array_double_24byte_callback: 321.0


def main():
    test_array_double_24byte()
    test_array_double_24byte_callback()
