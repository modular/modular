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
# RUN: %mojo %s 3 | FileCheck %s

from std.sys import argv


trait ATrait(Movable):
    def my_method(self) -> Int:
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

    def myclosure(x: Int) {var y} -> Int:
        return y + x

    var s = AStruct(myclosure)
    # CHECK: 6
    takeIt(s)
