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


def main():
    comptime if get_defined_int["test"]() == 1:
        # CHECK_1: constraint failed: the payload of every case must conform to `Equatable` to use the default `Equatable` implementation for an `EnumLike` type
        # CHECK_1-NOT: storage
        print(Compared.b(Opaque(1)) == Compared.b(Opaque(1)))
    elif get_defined_int["test"]() == 2:
        # CHECK_2: constraint failed: the payload of every case must conform to `Hashable` to use the default `Hashable` implementation for an `EnumLike` type
        # CHECK_2-NOT: storage
        print(hash(Compared.b(Opaque(1))))
