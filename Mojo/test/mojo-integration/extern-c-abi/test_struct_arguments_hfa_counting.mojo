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
# C reference: c_abi_test_hfa_counting.c
# RUN: mkdir -p %t.dir
# RUN: mojo build -Xlinker $(dirname %s)/libc_abi_reference.lo %s -o %t.dir/test_hfa_counting
# RUN: %t.dir/test_hfa_counting | FileCheck %s

from std.ffi import external_call


@fieldwise_init
struct Vec2(TrivialRegisterPassable):
    var x: Float32
    var y: Float32


@fieldwise_init
struct Vec3(TrivialRegisterPassable):
    var x: Float32
    var y: Float32
    var z: Float32


# Six flattened float members: not an HFA under AAPCS64, which counts members
# after flattening nested structs. Passed by pointer, returned through x8.
@fieldwise_init
struct Box(TrivialRegisterPassable):
    var min: Vec3
    var max: Vec3


# Four flattened members: still an HFA (v0-v3).
@fieldwise_init
struct Hfa4(TrivialRegisterPassable):
    var a: Vec2
    var b: Float32
    var c: Float32


# Five flattened members: not an HFA. Passed by pointer, returned through x8.
@fieldwise_init
struct Hfa5(TrivialRegisterPassable):
    var a: Vec2
    var b: Vec2
    var c: Float32


def test_box():
    var v = Box(Vec3(1.0, 2.0, 3.0), Vec3(4.0, 5.0, 6.0))
    var result = external_call["c_func_box", Box](v)
    print(
        "box:",
        result.min.x,
        result.min.y,
        result.min.z,
        result.max.x,
        result.max.y,
        result.max.z,
    )


# CHECK: box: 2.0 3.0 4.0 5.0 6.0 7.0


def test_hfa4():
    var v = Hfa4(Vec2(1.0, 2.0), 3.0, 4.0)
    var result = external_call["c_func_hfa4", Hfa4](v)
    print("hfa4:", result.a.x, result.a.y, result.b, result.c)


# CHECK: hfa4: 2.0 3.0 4.0 5.0


def test_hfa5():
    var v = Hfa5(Vec2(1.0, 2.0), Vec2(3.0, 4.0), 5.0)
    var result = external_call["c_func_hfa5", Hfa5](v)
    print("hfa5:", result.a.x, result.a.y, result.b.x, result.b.y, result.c)


# CHECK: hfa5: 2.0 3.0 4.0 5.0 6.0


def main():
    test_box()
    test_hfa4()
    test_hfa5()
