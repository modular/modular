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

# RUN: not %mojo -D test=1 %s 2>&1 | FileCheck --check-prefix CHECK_1 %s
# RUN: not %mojo -D test=2 %s 2>&1 | FileCheck --check-prefix CHECK_2 %s
# RUN: %mojo -D test=3 %s | FileCheck --check-prefix CHECK_3 %s

# The defaults of `Equatable.__eq__` and `Hashable.__hash__` derive through
# `EnumLike`, so an enum whose payload does not conform gets an error about
# the payload, and never about the hidden storage.

from std.sys import get_defined_int


struct Opaque(Copyable, Movable):
    var x: Int

    def __init__(out self, x: Int):
        self.x = x


__enum Compared(Equatable, Hashable):
    case a
    case b(Opaque)


# The default `Writable` methods write a hand-written tuple payload too.
@fieldwise_init
struct HandWritten(EnumLike, ImplicitlyCopyable, Writable):
    var payload: Tuple[Int]

    comptime _enum_case_names = ParameterList.of["number".value].values
    comptime _enum_case_types = TypeList.of[Trait=AnyType, Tuple[Int]].values

    def _get_enum_discriminant(self) -> Int:
        return 0

    def _unsafe_get_enum_payload[
        id: Int
    ](ref self) -> ref[self] TypeList[Trait=AnyType, Self._enum_case_types]()[
        id
    ]:
        return rebind[TypeList[Trait=AnyType, Self._enum_case_types]()[id]](
            Pointer(to=self.payload).unsafe_origin_cast[origin_of(self)]()[]
        )


def main():
    comptime if get_defined_int["test"]() == 1:
        # CHECK_1: constraint failed: the payload of every case must conform to `Equatable` to use the default `Equatable` implementation for an `EnumLike` type
        # CHECK_1-NOT: storage
        print(Compared.b(Opaque(1)) == Compared.b(Opaque(1)))
    elif get_defined_int["test"]() == 2:
        # CHECK_2: constraint failed: the payload of every case must conform to `Hashable` to use the default `Hashable` implementation for an `EnumLike` type
        # CHECK_2-NOT: storage
        print(hash(Compared.b(Opaque(1))))
    elif get_defined_int["test"]() == 3:
        # CHECK_3: HandWritten.number(1)
        print(HandWritten((1,)))
        # CHECK_3-NEXT: HandWritten.number(Int(1))
        print(repr(HandWritten((1,))))
