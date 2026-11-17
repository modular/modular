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
# C reference: c_abi_test_array_fields.c
# RUN: mkdir -p %t.dir
# RUN: mojo build -Xlinker $(dirname %s)/libc_abi_reference.lo %s -o %t.dir/test_array_fields
# RUN: %t.dir/test_array_fields | FileCheck %s

from std.ffi import external_call


# {float v[3]}: 12 bytes, an HFA on arm64 (s0-s2) and SSE on x86-64.
@fieldwise_init
struct FloatArr3:
    var v: Array[Float32, 3]


# {float v[5]}: 20 bytes, five members exceed the HFA limit.
@fieldwise_init
struct FloatArr5:
    var v: Array[Float32, 5]


# {double v[3]}: 24 bytes, still an HFA on arm64 (v0-v2).
@fieldwise_init
struct DoubleArr3:
    var v: Array[Float64, 3]


# {int32 v[2]}: 8 bytes, integer coercion on both targets.
@fieldwise_init
struct IntArr2:
    var v: Array[Int32, 2]


# {float v[2]; int32 w}: 12 bytes, integer-classified on both targets.
@fieldwise_init
struct MixedArr:
    var v: Array[Float32, 2]
    var w: Int32


def test_float_arr3():
    var s = FloatArr3([11.0, 22.0, 33.0])
    var status = external_call["check_float_arr3", Int32](s)
    print("float_arr3:", status)


# CHECK: float_arr3: 0


def test_float_arr5():
    var s = FloatArr5([1.0, 2.0, 3.0, 4.0, 5.0])
    var status = external_call["check_float_arr5", Int32](s)
    print("float_arr5:", status)


# CHECK: float_arr5: 0


def test_double_arr3():
    var s = DoubleArr3([1.5, 2.5, 3.5])
    var status = external_call["check_double_arr3", Int32](s)
    print("double_arr3:", status)


# CHECK: double_arr3: 0


def test_int_arr2():
    var s = IntArr2([7, 8])
    var status = external_call["check_int_arr2", Int32](s)
    print("int_arr2:", status)


# CHECK: int_arr2: 0


def test_mixed_arr():
    var s = MixedArr([1.0, 2.0], 42)
    var status = external_call["check_mixed_arr", Int32](s)
    print("mixed_arr:", status)


# CHECK: mixed_arr: 0


def main():
    test_float_arr3()
    test_float_arr5()
    test_double_arr3()
    test_int_arr2()
    test_mixed_arr()
