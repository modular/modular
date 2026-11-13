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

# A `comptime if` on the condition of a conditional type `A if c else B` must
# refine that type to the selected arm inside each branch.
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


@fieldwise_init
struct Baz(Movable):
    def baz(self):
        pass


# CHECK-LABEL: lit.fn @"then_and_else
# CHECK: hlcf.comptime.if
# CHECK: kgen.rebind %arg : {{.*}} to !lit.ref<!Foo, imm
# CHECK: } else {
# CHECK: kgen.rebind %arg : {{.*}} to !lit.ref<!Bar, imm
def then_and_else[c: Bool](arg: Foo if c else Bar):
    comptime if c:
        arg.foo()
    else:
        arg.bar()


comptime Alias[c: Bool] = Foo if c else Bar


# CHECK-LABEL: lit.fn @"through_alias
# CHECK: hlcf.comptime.if
# CHECK: kgen.rebind %arg : {{.*}} to !lit.ref<!Foo, imm
# CHECK: } else {
# CHECK: kgen.rebind %arg : {{.*}} to !lit.ref<!Bar, imm
def through_alias[c: Bool](arg: Alias[c]):
    comptime if c:
        arg.foo()
    else:
        arg.bar()


# CHECK-LABEL: lit.fn @"negated
# CHECK: hlcf.comptime.if
# CHECK: kgen.rebind %arg : {{.*}} to !lit.ref<!Bar, imm
# CHECK: } else {
# CHECK: kgen.rebind %arg : {{.*}} to !lit.ref<!Foo, imm
def negated[c: Bool](arg: Foo if c else Bar):
    comptime if not c:
        arg.bar()
    else:
        arg.foo()


# CHECK-LABEL: lit.fn @"nested
# CHECK: hlcf.comptime.if
# CHECK: kgen.rebind %arg : {{.*}} to !lit.ref<!Foo, imm
# CHECK: kgen.rebind %arg : {{.*}} to !lit.ref<!Bar, imm
# CHECK: kgen.rebind %arg : {{.*}} to !lit.ref<!Baz, imm
def nested[a: Bool, b: Bool](arg: Foo if a else (Bar if b else Baz)):
    comptime if a:
        arg.foo()
    elif b:
        arg.bar()
    else:
        arg.baz()


# CHECK-LABEL: lit.fn @"comptime_assert
# CHECK: kgen.rebind %arg : {{.*}} to !lit.ref<!Foo, imm
def comptime_assert[c: Bool](arg: Foo if c else Bar):
    comptime assert c
    arg.foo()


def take[c: Bool](var arg: Foo if c else Bar):
    pass


def take_foo(var arg: Foo):
    pass


# CHECK-LABEL: lit.fn @"introduce_by_argument
def introduce_by_argument[c: Bool]():
    comptime if c:
        take[c](Foo())
    else:
        take[c](Bar())


# CHECK-LABEL: lit.fn @"eliminate_by_argument
def eliminate_by_argument[c: Bool](var arg: Foo if c else Bar):
    comptime if c:
        take_foo(arg^)


# CHECK-LABEL: lit.fn @"introduce_by_return
def introduce_by_return[c: Bool]() -> Foo if c else Bar:
    comptime if c:
        return Foo()
    else:
        return Bar()


def selects_legacy[target: Int]() -> Bool:
    return target == 0


struct Dispatch[target: Int](Movable):
    comptime _use_legacy = selects_legacy[Self.target]()
    comptime _Impl = Foo if Self._use_legacy else Bar

    var _impl: Self._Impl

    # CHECK-LABEL: lit.fn @"call(comptime_if_refines_cond_type::Dispatch
    # CHECK: hlcf.comptime.if
    # CHECK: kgen.rebind {{.*}} to !lit.ref<!Foo, imm
    # CHECK: } else {
    # CHECK: kgen.rebind {{.*}} to !lit.ref<!Bar, imm
    def call(self):
        comptime if Self._use_legacy:
            self._impl.foo()
        else:
            self._impl.bar()

    # CHECK-LABEL: lit.fn @"call_after_assert(comptime_if_refines_cond_type::Dispatch
    # CHECK: kgen.rebind {{.*}} to !lit.ref<!Foo, imm
    def call_after_assert(self):
        comptime assert Self._use_legacy
        self._impl.foo()

    def __init__(out self):
        comptime if Self._use_legacy:
            self._impl = Foo()
        else:
            self._impl = Bar()

    def __init__(out self, *, var impl: Self._Impl):
        self._impl = impl^

    def reset(mut self):
        comptime if Self._use_legacy:
            self._impl = Foo()
        else:
            self._impl = Bar()

    # CHECK-LABEL: lit.fn @"make(comptime_if_refines_cond_type::Dispatch
    def make(self) -> Self:
        comptime if Self._use_legacy:
            return Self(impl=Foo())
        else:
            return Self(impl=Bar())


@fieldwise_init
struct TrivFoo(ImplicitlyCopyable, TrivialRegisterPassable):
    var x: Int

    def foo(self):
        pass


@fieldwise_init
struct TrivBar(ImplicitlyCopyable, TrivialRegisterPassable):
    var x: Int

    def bar(self):
        pass


# A conditional type field of a register-passable struct is read by value, not
# through a reference, and refines the same way.
struct TrivialDispatch[target: Int](ImplicitlyCopyable, TrivialRegisterPassable):
    comptime _use_legacy = selects_legacy[Self.target]()
    comptime _Impl = TrivFoo if Self._use_legacy else TrivBar

    var _impl: Self._Impl

    # CHECK-LABEL: lit.fn @"call(comptime_if_refines_cond_type::TrivialDispatch
    # CHECK: kgen.rebind {{.*}} to !TrivFoo
    # CHECK: } else {
    # CHECK: kgen.rebind {{.*}} to !TrivBar
    def call(self):
        comptime if Self._use_legacy:
            self._impl.foo()
        else:
            self._impl.bar()


def main():
    then_and_else[True](Foo())
    then_and_else[False](Bar())
