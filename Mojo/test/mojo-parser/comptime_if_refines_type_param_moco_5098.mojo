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

# Regression test for MOCO-5098: inside a scope guarded by
# `conforms_to(T, Deinitable)`, a struct that is `Deinitable` only when `T` is
# must be destructible, including in nested functions hoisted out of the scope.


struct Box[T: AnyType](
    Deinitable where conforms_to(T, Deinitable) else "Use Box.take",
    Movable,
):
    def __init__(out self):
        pass

    def use(self):
        pass

    def __deinit__(deinit self) where conforms_to(Self.T, Deinitable):
        pass


def take_box[U: AnyType](b: Box[U]):
    b.use()


def merge[U: AnyType](a: Box[U], b: Box[U]):
    a.use()
    b.use()


def destroy_under_guard[T: AnyType]() -> Int:
    comptime if conforms_to(T, Copyable & Deinitable):
        var b = Box[T]()
        b.use()
        return 1
    else:
        return -1


def destroy_in_elif_arm[T: AnyType]():
    comptime if conforms_to(T, Copyable):
        pass
    elif conforms_to(T, Deinitable):
        var b = Box[T]()
        b.use()


def destroy_after_comptime_assert[T: AnyType]():
    comptime assert conforms_to(T, Deinitable)
    var b = Box[T]()
    b.use()


def destroy_under_conjunction[T: AnyType, U: AnyType]():
    comptime if conforms_to(T, Deinitable) and (conforms_to(U, Deinitable)):
        var b = Box[T]()
        b.use()
        var c = Box[U]()
        c.use()


def destroy_under_repeated_subject[T: AnyType]():
    comptime if conforms_to(T, Copyable) and conforms_to(T, Deinitable):
        var b = Box[T]()
        b.use()


def destroy_in_nested_def_under_guard[T: AnyType]():
    comptime if conforms_to(T, Deinitable):

        def inner():
            var b = Box[T]()
            b.use()

        inner()


def mix_outer_and_guarded_values[T: AnyType](outer: Box[T]):
    comptime if conforms_to(T, Deinitable):
        var inner = Box[T]()
        merge(outer, inner)
        merge(inner, outer)
        take_box[T](outer)
        take_box[T](inner)


def return_guarded_value[T: AnyType]() -> Box[T]:
    comptime if conforms_to(T, Deinitable):
        var b = Box[T]()
        b.use()
        return b^
    else:
        return Box[T]()


def recurse_under_guard[T: AnyType](n: Int) -> Int:
    comptime if conforms_to(T, Deinitable):
        var b = Box[T]()
        b.use()
        if n > 0:
            return recurse_under_guard[T](n - 1)
        return 0
    else:
        return -1


def else_arm_stays_linear[T: AnyType]():
    comptime if conforms_to(T, Deinitable):
        pass
    else:
        var b = Box[T]()
        # expected-error @below {{'b' abandoned without being explicitly destroyed: Use Box.take}}
        b.use()


def negated_guard_stays_linear[T: AnyType]():
    comptime if not conforms_to(T, Deinitable):
        var b = Box[T]()
        # expected-error @below {{'b' abandoned without being explicitly destroyed: Use Box.take}}
        b.use()


def unrelated_guard_stays_linear[T: AnyType, U: AnyType]():
    comptime if conforms_to(U, Deinitable):
        var b = Box[T]()
        # expected-error @below {{'b' abandoned without being explicitly destroyed: Use Box.take}}
        b.use()
