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


@no_inline
def aThing[f: def(Int) capturing -> Int](y: Int):
    def aClosure(z: Int) {var} -> Int:
        return f(y)

    takeIt(aClosure, y)


@no_inline
def itCaptures[THREE: Int](one: Int, four: Int):
    @__parameter
    def aParam(z: Int) -> Int:
        return THREE + four + z

    aThing[aParam](one)

    # COM: Ensure nesting in legacy closure does not corrupt the symbol calculation
    comptime if THREE == 3:

        @__copy_capture(one, four)
        @__parameter
        def aParam2(zz: Int) -> Int:
            def thing(z: Int) {var zz} -> Int:
                return zz

            takeIt(thing, four)
            return one + one

        aThing[aParam2](one)


def takeIt[f: ImplicitlyCopyable & def(z: Int) -> Int](impl: f, y: Int):
    print(impl(y))


def main() raises:
    var y: Int = atol(argv()[1])
    var one = atol(argv()[2])
    var four = atol(argv()[3])

    def myclosure(x: Int) {var y} -> Int:
        return y + x

    var s = AStruct(myclosure)
    # CHECK: 6
    takeIt(s)

    # CHECK: 8
    # CHECK: 1
    # CHECK: 2
    itCaptures[3](one, four)
