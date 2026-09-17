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
# RUN: env MOJO_ENABLE_PARAMETRIC_CLOSURE_TRAIT=1 %mojo -debug-level full %s \
# RUN:   | FileCheck %s


@fieldwise_init
struct Foo[T: Writable](def(T)):
    def __call__(mut self, arg: Self.T):
        print("via struct:", arg)


def call_int[T: def(Int)](mut closure: T):
    closure.__call__(1)


def main():
    var fi = Foo[Int]()
    # CHECK: via struct: 1
    call_int(fi)

    var y = 1

    def closure(x: Int) {mut y}:
        y += x

    call_int(closure)
    # CHECK: via closure: 2
    print("via closure:", y)
