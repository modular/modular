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

# A Boolean `comptime __guard` on the condition of a conditional type
# `A if c else B` must refine that type to the selected arm in the else suite
# and in the continuation, like the two branches of a `comptime if`.
#
# RUN: %parse-mojo-isolated %s | FileCheck %s


@fieldwise_init
struct Foo(Movable):
    def foo(self):
        pass


@fieldwise_init
struct Bar(Movable):
    def bar(self):
        pass


# The continuation holds when `c` does, so it sees `Foo`. The else suite runs
# when `c` is false, so it sees `Bar`.
# CHECK-LABEL: lit.fn @"guard_condition
# CHECK: hlcf.comptime.if
# CHECK: kgen.rebind %arg : {{.*}} to !lit.ref<!Foo, imm
# CHECK: } else {
# CHECK: kgen.rebind %arg : {{.*}} to !lit.ref<!Bar, imm
def guard_condition[c: Bool](arg: Foo if c else Bar):
    comptime __guard c else:
        arg.bar()
        return
    arg.foo()


# CHECK-LABEL: lit.fn @"guard_negated
# CHECK: hlcf.comptime.if
# CHECK: kgen.rebind %arg : {{.*}} to !lit.ref<!Bar, imm
# CHECK: } else {
# CHECK: kgen.rebind %arg : {{.*}} to !lit.ref<!Foo, imm
def guard_negated[c: Bool](arg: Foo if c else Bar):
    comptime __guard not c else:
        arg.foo()
        return
    arg.bar()


def selects_legacy[target: Int]() -> Bool:
    return target == 0


struct Dispatch[target: Int](Movable):
    comptime _use_legacy = selects_legacy[Self.target]()
    comptime _Impl = Foo if Self._use_legacy else Bar

    var _impl: Self._Impl

    # CHECK-LABEL: lit.fn @"call(comptime_guard_refines_cond_type::Dispatch
    # CHECK: hlcf.comptime.if
    # CHECK: kgen.rebind {{.*}} to !lit.ref<!Bar, imm
    # CHECK: } else {
    # CHECK: kgen.rebind {{.*}} to !lit.ref<!Foo, imm
    def call(self):
        comptime __guard not Self._use_legacy else:
            self._impl.foo()
            return
        self._impl.bar()
