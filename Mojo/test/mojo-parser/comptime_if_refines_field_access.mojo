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

# Reading a field of a value whose type is `A if c else B` inside a `comptime if
# c:` branch must work like a method call does: the value refines to the
# selected arm, so the field resolves on that arm.
#
# RUN: %parse-mojo-isolated %s -mlir-print-debuginfo | kgen-opt -lower-semantic-cf -check-lifetimes -verify-diagnostics


struct Foo(Movable):
    var other: Int
    var handle: Int

    def __init__(out self):
        self.other = 0
        self.handle = 1


struct Bar(Movable):
    var other: Int
    var handle: Int

    def __init__(out self):
        self.other = 0
        self.handle = 2


def use_legacy[t: Int]() -> Bool:
    return t == 0


struct Wrapper[t: Int = 0]:
    comptime _use = use_legacy[Self.t]()
    comptime _Impl = Foo if Self._use else Bar
    var _impl: Self._Impl

    def __init__(out self, var impl: Self._Impl):
        self._impl = impl^

    def handle(self) -> Int:
        comptime if Self._use:
            return self._impl.handle
        else:
            return self._impl.handle

    def sum(self) -> Int:
        comptime if Self._use:
            return self._impl.other + self._impl.handle
        else:
            return self._impl.other + self._impl.handle


def main():
    pass
