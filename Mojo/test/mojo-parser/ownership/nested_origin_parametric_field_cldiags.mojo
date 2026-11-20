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

# A nested origin like `o.x.q` resolves `q` in the layout of the instantiated
# field type `Inner[String]`, where `p` spans several bits, not in the layout of
# the declared type `Inner[Self.T]`, where `p` is one bit. An associated type
# such as `Self.A.V` must also resolve to its concrete type.


struct Inner[U: Movable & Deinitable](Movable):
    var p: Self.U
    var q: String

    def __init__(out self, var p: Self.U, var q: String):
        self.p = p^
        self.q = q^


struct Outer[T: Movable & Deinitable](Movable):
    var x: Inner[Self.T]

    def __init__(out self, var x: Inner[Self.T]):
        self.x = x^


trait Algebra:
    comptime V: Movable & Deinitable


struct Text(Algebra):
    comptime V = Inner[String]


struct Holder[A: Algebra](Movable):
    var x: Self.A.V

    def __init__(out self, var x: Self.A.V):
        self.x = x^


def consume(var s: String):
    pass


def use(s: String):
    pass


def ref_to_sibling_survives_move():
    var o = Outer[String](Inner[String](String("p"), String("q")))
    ref r = o.x.q
    consume(o.x.p^)
    use(r)
    o.x.p = String("p2")


def ref_to_moved_field_is_uninitialized():
    var o = Outer[String](Inner[String](String("p"), String("q")))  # expected-note {{'o' declared here}}
    ref r = o.x.q
    consume(o.x.q^)
    use(r)  # expected-error {{use of uninitialized value 'o.x.q'}}
    o.x.q = String("q2")


def associated_type_ref_to_sibling_survives_move():
    var h = Holder[Text](Inner[String](String("p"), String("q")))
    ref r = h.x.q
    consume(h.x.p^)
    use(r)
    h.x.p = String("p2")
