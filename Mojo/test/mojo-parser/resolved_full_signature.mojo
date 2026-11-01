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

# RUN: %parse-mojo-isolated %s | FileCheck %s

# Functions inside a struct or trait record their signature with the parent's
# parameters prepended. Other functions do not.


# CHECK-LABEL: lit.fn @"free_fn
# CHECK-NOT: resolvedFullSignature
def free_fn(x: Int) -> Int:
    return x


struct Box[T: AnyType]:
    # CHECK-LABEL: lit.fn @"get
    # CHECK-SAME: resolvedFullSignature = !kgen.generator<!lit.generator<<"T": !AnyType
    def get(self, x: Int) -> Int:
        return x

    # CHECK-LABEL: lit.fn @"nested_owner
    # CHECK-SAME: resolvedFullSignature
    def nested_owner(self) -> Int:
        def inner(y: Int) -> Int:
            return y

        return inner(1)


trait Shape:
    # CHECK-LABEL: lit.fn @"area
    # CHECK-SAME: resolvedFullSignature = !kgen.generator<!lit.generator<<"_Self": !Shape
    def area(self) -> Int:
        ...


# A lambda at struct scope is resolved after its signature is first finalized
# and must still see the struct parameters.
struct Grid[cols: Int, rows: Int]:
    comptime to_index = lambda (x: Int, y: Int) -> Int: (y * Self.cols + x)

    def at(self, x: Int, y: Int) -> Int:
        return Self.to_index(x, y)

# The nested function is emitted last and has no parent parameters to prepend.
# CHECK-LABEL: lit.fn @"inner
# CHECK-NOT: resolvedFullSignature
# CHECK: lit.end_fn
