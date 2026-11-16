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
# Check enum sizes, alignments and tag offsets, including tags in union tail padding.
# RUN: %mojo %s | FileCheck %s

from std.builtin._enum import __Enum, __EnumStorage
from std.sys import align_of, size_of


__enum Empty(Copyable):
    pass


__enum Flags(ImplicitlyCopyable):
    case a
    case b


__enum Byte(ImplicitlyCopyable):
    case a(UInt8)


__enum Word(ImplicitlyCopyable):
    case a(Int64)


__enum Text(ImplicitlyCopyable):
    case a(String)
    case b


__enum Pair(ImplicitlyCopyable):
    case a(Int, Int)
    case b(UInt8)


__enum Aligned(ImplicitlyCopyable):
    case a(SIMD[DType.int64, 8])
    case b


__enum Uneven(ImplicitlyCopyable):
    case a(SIMD[DType.int64, 4])
    case b(Int64, Int64, Int64, Int64, Int64)


__enum Nested(ImplicitlyCopyable):
    case a(Text)
    case b(Word, Byte)


__enum Generic[T: Copyable & Deinitable](Copyable):
    case a(Self.T)
    case b


def report[T: __Enum](name: String):
    comptime Storage = __EnumStorage[T._enum_case_types]
    var storage = Storage(unsafe_uninitialized=())
    var tag_offset = Int(Pointer(to=storage._discriminant())) - Int(
        Pointer(to=storage)
    )
    __mlir_op.`lit.ownership.mark_destroyed`(__get_mvalue_as_litref(storage))
    print(name, size_of[T](), align_of[T](), tag_offset)


def main():
    # CHECK: Empty 1 1 0
    report[Empty]("Empty")
    # CHECK-NEXT: Flags 1 1 0
    report[Flags]("Flags")
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
    # CHECK-NEXT: Generic[String] 32 8 24
    report[Generic[String]]("Generic[String]")
