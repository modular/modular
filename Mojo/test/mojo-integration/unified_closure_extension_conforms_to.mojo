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
# RUN: %mojo %s | FileCheck %s


struct Concrete[F: def(Int) -> Int](Movable):
    var f: Self.F

    def __init__(out self, var f: Self.F):
        self.f = f^

    def needs_copyable(self) -> Int where conforms_to(Self.F, Copyable):
        return 1


struct Param[E: AnyType, F: def(E) -> Int](Movable):
    var f: Self.F

    def __init__(out self, var f: Self.F):
        self.f = f^

    def needs_copyable(self) -> Int where conforms_to(Self.F, Copyable):
        return 2


def ten(x: Int) -> Int:
    return x * 10


def main():
    var c = Concrete(ten)
    # CHECK: 1
    print(c.needs_copyable())

    # `F` is bound to a `#kgen.extension` of `ten`'s closure wrapper here.
    var p = Param[Int](ten)
    # CHECK-NEXT: True
    print(conforms_to(type_of(p).F, Copyable))
    # CHECK-NEXT: 2
    print(p.needs_copyable())
