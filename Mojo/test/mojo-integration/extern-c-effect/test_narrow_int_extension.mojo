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
# Tests that abi("C") extends the narrow integers that Mojo produces: call
# arguments, and the results of Mojo definitions (MOCO-5051).
# Apple arm64 and x86-64 require the producer to extend, and the -O2 C side
# relies on it. Each value comes from a truncation, so its register keeps
# nonzero upper bits unless Mojo extends it.
#
# Also tests that Mojo does not trust the extension of a narrow integer it
# receives. C does not extend a struct of one narrow integer, which Mojo can
# receive as a bare integer or as a struct of one field.
#
# C reference: c_narrow_int.c
# RUN: mkdir -p %t.dir
# RUN: mojo build -Xlinker $(dirname %s)/libc_narrow_int_reference.lo %s -o %t.dir/test_narrow_int_extension
# RUN: %t.dir/test_narrow_int_extension | FileCheck %s

from std.ffi import external_call

comptime N = 1000


def mojo_trunc_u8(x: UInt32) abi("C") -> UInt8:
    return UInt8(x * 7)


def mojo_trunc_i8(x: UInt32) abi("C") -> Int8:
    return Int8(x * 7)


def test_arguments():
    var bad_u8 = 0
    var bad_i8 = 0
    var bad_u16 = 0
    var bad_i16 = 0
    for i in range(N):
        var x = UInt32(i) * 7
        var u8 = UInt8(x)
        var i8 = Int8(x)
        var u16 = UInt16(x * 97)
        var i16 = Int16(x * 97)
        if external_call["c_widen_u8", UInt32](u8) != UInt32(u8):
            bad_u8 += 1
        if external_call["c_widen_i8", Int32](i8) != Int32(i8):
            bad_i8 += 1
        if external_call["c_widen_u16", UInt32](u16) != UInt32(u16):
            bad_u16 += 1
        if external_call["c_widen_i16", Int32](i16) != Int32(i16):
            bad_i16 += 1
    print("u8 arg wrong:", bad_u8)
    print("i8 arg wrong:", bad_i8)
    print("u16 arg wrong:", bad_u16)
    print("i16 arg wrong:", bad_i16)


# CHECK: u8 arg wrong: 0
# CHECK: i8 arg wrong: 0
# CHECK: u16 arg wrong: 0
# CHECK: i16 arg wrong: 0


def test_indirect_argument():
    var widen = external_call[
        "c_get_widen_u8", def(UInt8) thin abi("C") -> UInt32
    ]()
    var bad = 0
    for i in range(N):
        var u8 = UInt8(UInt32(i) * 7)
        if widen(u8) != UInt32(u8):
            bad += 1
    print("indirect u8 arg wrong:", bad)


# CHECK: indirect u8 arg wrong: 0


def test_results():
    var bad_u8 = 0
    var bad_i8 = 0
    for i in range(N):
        var x = UInt32(i)
        var got_u8 = external_call["c_call_u8_fn", UInt32](mojo_trunc_u8, x)
        if got_u8 != UInt32(mojo_trunc_u8(x)):
            bad_u8 += 1
        var got_i8 = external_call["c_call_i8_fn", Int32](mojo_trunc_i8, x)
        if got_i8 != Int32(mojo_trunc_i8(x)):
            bad_i8 += 1
    print("u8 result wrong:", bad_u8)
    print("i8 result wrong:", bad_i8)


# CHECK: u8 result wrong: 0
# CHECK: i8 result wrong: 0


@fieldwise_init
struct U8Box(TrivialRegisterPassable):
    var x: UInt8


@fieldwise_init
struct I8Box(TrivialRegisterPassable):
    var x: Int8


def mojo_widen_u8(x: UInt8) abi("C") -> UInt32:
    return UInt32(x)


def mojo_widen_i8(x: Int8) abi("C") -> Int32:
    return Int32(x)


def mojo_widen_u8_box(b: U8Box) abi("C") -> UInt32:
    return UInt32(b.x)


def mojo_widen_i8_box(b: I8Box) abi("C") -> Int32:
    return Int32(b.x)


@fieldwise_init
struct U8Pair(TrivialRegisterPassable):
    var a: UInt8
    var b: UInt8


# The pair argument needs coercion, so this definition is rewritten to the C
# signature, unlike one whose arguments all pass unchanged. A struct of one
# field does not do that: Mojo passes it as its field.
def mojo_widen_second_u8(p: U8Pair, x: UInt8) abi("C") -> UInt32:
    return UInt32(x)


# The widened values are summed rather than compared one by one: LLVM narrows
# a compare of two extended values back to i8, which would hide upper bits
# that a wrong extension attribute let through.
def test_received_results():
    var want_u8 = 0
    var want_i8 = 0
    var got_u8 = 0
    var got_i8 = 0
    var got_u8_box = 0
    var got_i8_box = 0
    for i in range(N):
        var x = UInt32(i)
        want_u8 += Int(UInt32(UInt8(x * 7)))
        want_i8 += Int(Int32(Int8(x * 7)))
        got_u8 += Int(UInt32(external_call["c_box_u8", UInt8](x)))
        got_i8 += Int(Int32(external_call["c_box_i8", Int8](x)))
        got_u8_box += Int(UInt32(external_call["c_box_u8", U8Box](x).x))
        got_i8_box += Int(Int32(external_call["c_box_i8", I8Box](x).x))
    print("flattened u8 result wrong:", got_u8 != want_u8)
    print("flattened i8 result wrong:", got_i8 != want_i8)
    print("u8 box result wrong:", got_u8_box != want_u8)
    print("i8 box result wrong:", got_i8_box != want_i8)


# CHECK: flattened u8 result wrong: False
# CHECK: flattened i8 result wrong: False
# CHECK: u8 box result wrong: False
# CHECK: i8 box result wrong: False


def test_received_arguments():
    var want_u8 = 0
    var want_i8 = 0
    var got_u8 = 0
    var got_i8 = 0
    var got_u8_box = 0
    var got_i8_box = 0
    var got_u8_mixed = 0
    for i in range(N):
        var x = UInt32(i)
        want_u8 += Int(UInt32(UInt8(x * 7)))
        want_i8 += Int(Int32(Int8(x * 7)))
        got_u8 += Int(
            external_call["c_call_box_u8_fn", UInt32](mojo_widen_u8, x)
        )
        got_u8_mixed += Int(
            external_call["c_call_pair_box_u8_fn", UInt32](
                mojo_widen_second_u8, x
            )
        )
        got_i8 += Int(
            external_call["c_call_box_i8_fn", Int32](mojo_widen_i8, x)
        )
        got_u8_box += Int(
            external_call["c_call_box_u8_fn", UInt32](mojo_widen_u8_box, x)
        )
        got_i8_box += Int(
            external_call["c_call_box_i8_fn", Int32](mojo_widen_i8_box, x)
        )
    print("flattened u8 argument wrong:", got_u8 != want_u8)
    print("flattened i8 argument wrong:", got_i8 != want_i8)
    print("flattened u8 after pair wrong:", got_u8_mixed != want_u8)
    print("u8 box argument wrong:", got_u8_box != want_u8)
    print("i8 box argument wrong:", got_i8_box != want_i8)


# CHECK: flattened u8 argument wrong: False
# CHECK: flattened i8 argument wrong: False
# CHECK: flattened u8 after pair wrong: False
# CHECK: u8 box argument wrong: False
# CHECK: i8 box argument wrong: False


def main():
    test_arguments()
    test_indirect_argument()
    test_results()
    test_received_results()
    test_received_arguments()
