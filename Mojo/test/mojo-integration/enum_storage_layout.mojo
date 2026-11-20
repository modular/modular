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
# Check enum and union sizes, alignments and tag offsets, including tags in a
# union's tail padding.
# RUN: %mojo %s | FileCheck %s

from std.builtin._enum import __Union, __UnionStorage
from std.reflection import reflect
from std.sys import align_of, size_of


__enum Empty(Copyable):
    pass


__enum Flags(Movable):
    case a
    case b


__union Byte(ImplicitlyCopyable):
    case a(UInt8)


__union Word(ImplicitlyCopyable):
    case a(Int64)


__union Text(ImplicitlyCopyable):
    case a(String)
    case b()


__union Pair(ImplicitlyCopyable):
    case a(Int, Int)
    case b(UInt8)


__union Aligned(ImplicitlyCopyable):
    case a(SIMD[DType.int64, 8])
    case b()


__union Uneven(ImplicitlyCopyable):
    case a(SIMD[DType.int64, 4])
    case b(Int64, Int64, Int64, Int64, Int64)


__union Nested(ImplicitlyCopyable):
    case a(Text)
    case b(Word, Byte)


__union Generic[T: Copyable & Deinitable](Copyable):
    case a(Self.T)
    case b()


__enum Phantom[T: AnyType]:
    case only


def report[T: __Union](name: String):
    comptime Storage = __UnionStorage[T._enum_case_types]
    var storage = Storage(unsafe_uninitialized=())
    var tag_offset = Int(Pointer(to=storage._discriminant())) - Int(
        Pointer(to=storage)
    )
    __mlir_op.`lit.ownership.mark_destroyed`(__get_mvalue_as_litref(storage))
    print(name, size_of[T](), align_of[T](), tag_offset)


def main():
    comptime assert reflect[Empty].field_types()[0] == UInt8
    comptime assert reflect[Flags].field_types()[0] == UInt8
    comptime assert conforms_to(Empty, TrivialRegisterPassable)
    comptime assert conforms_to(Phantom[List[Int]], TrivialRegisterPassable)
    # CHECK: Empty 1 1 0
    print("Empty", size_of[Empty](), align_of[Empty](), 0)
    # CHECK-NEXT: Flags 1 1 0
    print("Flags", size_of[Flags](), align_of[Flags](), 0)
    # CHECK-NEXT: Byte 2 1 1
    report[Byte]("Byte")
    # CHECK-NEXT: Word 16 8 8
    report[Word]("Word")
    # CHECK-NEXT: Text 32 8 24
    report[Text]("Text")
    # CHECK-NEXT: Pair 24 8 16
    report[Pair]("Pair")
    # CHECK-NEXT: Aligned 128 64 64
    report[Aligned]("Aligned")
    # CHECK-NEXT: Uneven 64 32 40
    report[Uneven]("Uneven")
    # CHECK-NEXT: Nested 40 8 32
    report[Nested]("Nested")
    # CHECK-NEXT: Generic[Int] 16 8 8
    report[Generic[Int]]("Generic[Int]")
