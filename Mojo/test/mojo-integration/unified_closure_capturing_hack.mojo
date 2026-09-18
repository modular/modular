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
# RUN: %mojo %s 3 1 4 | FileCheck %s

from std.sys import argv


trait ATrait(Movable):
    # In order for a struct that depends on a capturing closure
    # to conform to a trait, all the methods of that trait must be
    # marked as capturing. This is temporary until we remove the capturing
    # effect. Note that the legacy closures are responsible for this restriction.
    # In particular, the following is not supported:
    # trait ATrait(Movable):
    #     def my_method(self) -> Int:
    #         ...

    # struct ParamStruct[func: def (x: Int) capturing -> Int](ATrait):
    #     def my_method(self) -> Int:
    #         return func(2)
    def my_method(self) capturing -> Int:
        ...


struct AStruct[func: def(x: Int) -> Int](ATrait):
    var myFunc: Self.func

    def __init__(out self, var x: Self.func):
        self.myFunc = x^

    def my_method(self) -> Int:
        return self.myFunc(3)


def takeIt[T: ATrait](impl: T):
    print(impl.my_method())


def main() raises:
    var y: Int = atol(argv()[1])
    var one = atol(argv()[2])
    var four = atol(argv()[3])

    def myclosure(x: Int) {var y} -> Int:
        return y + x

    var s = AStruct(myclosure)
    # CHECK: 6
    takeIt(s)
