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
# RUN: %parse-mojo-isolated %s -mlir-print-debuginfo | kgen-opt -lower-semantic-cf -check-lifetimes -verify-diagnostics

# A unified closure capturing a value whose type embeds an interior origin
# routes that origin through the closure's synthesized storage initializer,
# where it arrives as a function parameter.
#
# TODO(MOCO-4438): the diagnostic below is itself an over-rejection - the
# interior origin is initialized before the closure ever runs. Interior origins
# carried by a closure-storage field type are never seeded as live-in, so the
# lifted `__call__` reports them as never-initialized. Once that is fixed this
# should lower cleanly and the expected-error should go away; until then it pins
# down that we diagnose rather than crash.


struct MyList[T: AnyType](Movable where False):
    var data: UnsafePointer[Self.T, UntrackedOrigin[mut=True]]

    def __init__(out self):
        self.data = UnsafePointer[
            Self.T, UntrackedOrigin[mut=True]
        ].unsafe_dangling()

    def __deinit__(deinit self):
        pass

    @__unsafe_nested_origins_read_only
    def __getitem__(
        ref self,
    ) -> ref[self.data._get_ref_with_unsafe_interior_origin["element"](self)] Self.T:
        return self.data._get_ref_with_unsafe_interior_origin["element"](self)


def outer():
    var list = MyList[Int]()

    # `p`'s type embeds the interior origin `list["element"]`, so capturing it
    # carries that origin into the closure's storage type.
    var p = Pointer(to=list[])

    @always_inline
    # expected-error @+1 {{use of a never-initialized interior reference 'list["element"]'}}
    def closure() {imm}:
        _ = p[]

    closure()
