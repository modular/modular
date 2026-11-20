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

# RUN: %parse-mojo-isolated %s -o %t.mlir
# RUN: kgen-opt %t.mlir -lower-semantic-cf -check-lifetimes -verify-parameters -verify-diagnostics | FileCheck %s

# Reading a struct field whose type is an associated type of a trait
# parameter (e.g. `var v: Self.A.V`) must not be reported as a use of an
# uninitialized value when the parameter is concrete and the associated type
# is non-trivial.  CheckLifetimes used to compute the field's bit width in
# the object layout from the unresolved `get_witness` form (one bit) while
# the field access carried the resolved concrete type (several bits), so
# the access's ValueRef extended past the field's slot and overlapped bits
# that were never marked live.

trait Algebra:
    comptime V: ImplicitlyCopyable & Deinitable


struct Big(ImplicitlyCopyable, Deinitable):
    var a: Int
    var b: Int
    var c: Int

    def __init__(out self):
        self.a = 0
        self.b = 0
        self.c = 0


struct Text(Algebra):
    comptime V = Big


struct Holder[A: Algebra](ImplicitlyCopyable):
    var v: Self.A.V

    def __init__(out self, v: Self.A.V):
        self.v = v


def use_it(s: Big):
    pass


# CHECK-LABEL: lit.fn @"main()"
def main():
    var held = Holder[Text](Big())
    # CHECK-NOT: use of uninitialized value
    use_it(held.v)
