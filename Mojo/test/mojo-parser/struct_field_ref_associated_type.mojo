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

# RUN: %parse-mojo-isolated %s -mlir-print-debuginfo | kgen-opt -lower-semantic-cf -check-lifetimes -verify-parameters -verify-diagnostics

# `__struct_field_ref` on a field whose type is an associated type such as
# `Self.A.V` yields a reference to the concrete field type.


struct Inner(Movable):
    var p: String

    def __init__(out self, var p: String):
        self.p = p^


trait Algebra:
    comptime V: Movable & Deinitable


struct Text(Algebra):
    comptime V = Inner


struct Holder[A: Algebra](Movable):
    var x: Self.A.V

    def __init__(out self, var x: Self.A.V):
        self.x = x^


def struct_field_ref_associated_type():
    var h = Holder[Text](Inner(String("p")))
    __struct_field_ref(0, h) = Inner(String("p2"))
