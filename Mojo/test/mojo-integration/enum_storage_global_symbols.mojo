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
# Union values in global constants:
# 1. Read Bool, Int, wide, SIMD and empty union globals.
# 2. Relocate function and string fragments at offsets zero and eight.
# 3. Preserve nested packed globals, typed-union layout and global alignment.
# RUN: %mojo %s | FileCheck %s
# RUN: %mojo-build --emit llvm %s -o - | FileCheck %s --check-prefixes=LLVM,TYPE

from std.builtin.globals import global_constant


__union Flag(ImplicitlyCopyable):
    case value(Bool)
    case none()


__union Word(ImplicitlyCopyable):
    case value(Int)
    case none()


__union Uneven(ImplicitlyCopyable):
    case aligned(SIMD[DType.int64, 4])
    case wide(Int64, Int64, Int64, Int64, Int64)


@export
def flag_address() abi("Mojo") -> Pointer[Flag, ImmStaticOrigin]:
    return Pointer(to=global_constant[Flag.value(True)]())


@export
def word_address() abi("Mojo") -> Pointer[Word, ImmStaticOrigin]:
    return Pointer(to=global_constant[Word.value(17)]())


@export
def wide_address() abi("Mojo") -> Pointer[Uneven, ImmStaticOrigin]:
    return Pointer(to=global_constant[Uneven.wide(1, 2, 3, 4, 5)]())


@export
def aligned_address() abi("Mojo") -> Pointer[Uneven, ImmStaticOrigin]:
    return Pointer(
        to=global_constant[
            Uneven.aligned(SIMD[DType.int64, 4](11, 22, 33, 44))
        ]()
    )


@export
def empty_address() abi("Mojo") -> Pointer[Word, ImmStaticOrigin]:
    return Pointer(to=global_constant[Word.none()]())


def print_flag(value: Flag):
    match value:
        case .value(payload):
            # CHECK-DAG: bool True
            print("bool", payload)
        case .none():
            print("none")


def print_word(value: Word):
    match value:
        case .value(payload):
            # CHECK-DAG: int 17
            print("int", payload)
        case .none():
            # CHECK-DAG: none
            print("none")


def print_uneven(value: Uneven):
    match value:
        case .aligned(payload):
            # CHECK-DAG: aligned [11, 22, 33, 44]
            print("aligned", payload)
        case .wide(a, b, c, d, e):
            # CHECK-DAG: wide 1 2 3 4 5
            print("wide", a, b, c, d, e)


__union NamedEvent(ImplicitlyCopyable):
    case aligned(SIMD[DType.float32, 4])
    case wide(UInt32, UInt32, UInt32, UInt32, UInt32)
    case named(StaticString)


@fieldwise_init
struct Outer(ImplicitlyCopyable):
    var event: NamedEvent
    var tail: UInt8


@export
def address() abi("Mojo") -> Pointer[Outer, ImmStaticOrigin]:
    return Pointer(to=global_constant[Outer(NamedEvent.named("hello"), 91)]())


@fieldwise_init
struct Callback(ImplicitlyCopyable):
    var function: def(Int) thin -> Int
    var a: Int
    var b: Int


__union Event(ImplicitlyCopyable):
    case aligned(SIMD[DType.int64, 2])
    case callback(Callback)
    case bytes(Int64, Int64, Int64)


__union Action(ImplicitlyCopyable):
    case number(Int)
    case named(Int, StaticString)
    case call(Int, Callback)


def add(value: Int) -> Int:
    return value + 7


# LLVM-DAG: @global_constant{{(_[0-9]+)?}} = internal constant <{{.*}}ptr @{{.*}}::add{{.*}}i8 1, [7 x i8] zeroinitializer{{.*}}align 16
@export
def read_event() abi("Mojo") -> Pointer[Event, ImmStaticOrigin]:
    return Pointer(to=global_constant[Event.callback(Callback(add, 11, 23))]())


# LLVM-DAG: @global_constant{{(_[0-9]+)?}} = internal constant <{{.*}}i8 5{{.*}}ptr @static_string_{{.*}}align 8
@export
def read_named() abi("Mojo") -> Pointer[Action, ImmStaticOrigin]:
    return Pointer(to=global_constant[Action.named(5, "hello")]())


# LLVM-DAG: @global_constant{{(_[0-9]+)?}} = internal constant <{{.*}}i8 5{{.*}}ptr @{{.*}}::add{{.*}}align 8
# TYPE-DAG: getelementptr { { i64, [24 x i8] }, i8 }
@export
def read_action() abi("Mojo") -> Pointer[Action, ImmStaticOrigin]:
    return Pointer(to=global_constant[Action.call(5, Callback(add, 11, 23))]())


def invoke_action(value: Action):
    match value:
        case .named(n, s):
            # CHECK-DAG: named 5 hello
            print("named", n, s)
        case .call(n, callback):
            # CHECK-DAG: action 5 48 11 23
            print("action", n, callback.function(41), callback.a, callback.b)
        case .number(n):
            print(n)


def invoke(value: Event):
    match value:
        case .callback(callback):
            # CHECK-DAG: callback 48 11 23
            print("callback", callback.function(41), callback.a, callback.b)
        case .bytes(a, b, c):
            print(a, b, c)
        case .aligned(payload):
            print(payload)


def main():
    print_flag(flag_address()[])
    print_word(word_address()[])
    print_uneven(wide_address()[])
    print_uneven(aligned_address()[])
    print_word(empty_address()[])
    var p = address()
    match p[].event:
        case .named(name):
            # CHECK-DAG: nested hello 91
            print("nested", name, p[].tail)
        case .aligned(value):
            print(value)
        case .wide(a, b, c, d, e):
            print(a, b, c, d, e)
    invoke(read_event()[])
    invoke_action(read_named()[])
    invoke_action(read_action()[])
