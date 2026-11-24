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

# RUN: %parse-mojo-isolated %s -verify-diagnostics | FileCheck %s

##===----------------------------------------------------------------------===##
# Raise and Try
##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.fn @"simpleTryExcept
def simpleTryExcept():
    var a: Int
    # CHECK: lit.try
    try:
        # CHECK: lit.ref.store
        a = 0
        # CHECK-NEXT: lit.try.yield
    # CHECK-NEXT: except
    except:
        # CHECK-NEXT: lit.try.yield
        pass
    # CHECK-NEXT: else
    # CHECK-NEXT: lit.try.yield
    # CHECK: lit.end_fn


# CHECK-LABEL: lit.fn @"tryExceptElse
def tryExceptElse():
    var a: Int
    # CHECK: lit.try
    try:
        pass
    except:
        pass
    # CHECK: else
    else:
        # CHECK: lit.ref.store
        a = 0
        # CHECK-NEXT: lit.try.yield


def eatError(err: Error):
    pass


# CHECK-LABEL: lit.fn @"tryExceptArgDef
def tryExceptArgDef():
    try:
        pass
    # CHECK: } except {
    except err:
        # CHECK-NEXT: [[ERR:%.*]] = lit.ref.immut %err
        # CHECK-NEXT: eatError{{.*}}([[ERR]])
        eatError(err)


# CHECK-LABEL: lit.fn @"tryFinally
def tryFinally():
    # CHECK: lit.try
    try:
        # CHECK-NEXT: lit.try.yield
        pass
    # CHECK-NEXT: except
    # CHECK-NEXT: unreachable
    # CHECK: finally
    finally:
        # CHECK: lit.return
        return
    # CHECK: lit.try {
    try:
        # CHECK: lit.try {
        try:
            # CHECK-NEXT: lit.try.yield
            pass
        # CHECK-NEXT: except
        # CHECK-NEXT: lit.call {{.*}}Error::@"__init__{{.*}}"{{.*}}(%__try_error___0, %__try_error__){{.*}}*, "move"
        # CHECK-NEXT: lit.raise
        finally:
            pass
    except:
        pass


# CHECK-LABEL: lit.fn @"tryExceptInTryFinally
def tryExceptInTryFinally():
    # CHECK: lit.try {
    try:
        # CHECK: lit.try {
        try:
            # CHECK-NEXT: lit.try.yield
            pass
        # CHECK-NEXT: } except {
        # CHECK-NEXT: lit.try.yield
        # CHECK-NEXT: } else {
        # CHECK-NEXT: lit.try.yield
        # CHECK-NEXT: } finally {
        # CHECK-NEXT: lit.try.yield
        # CHECK-NEXT: }
        except:
            pass
    # CHECK-NEXT: lit.try.yield
    # CHECK-NEXT: } except {
    # CHECK-NEXT: hlcf.unreachable
    # CHECK-NEXT: } else {
    # CHECK-NEXT: lit.try.yield
    # CHECK-NEXT: } finally {
    # CHECK-NEXT: lit.try.yield
    # CHECK-NEXT: }
    finally:
        pass


def maybeRaises() raises -> Int:
    return 0


# CHECK-LABEL: lit.fn @"propagateErrorInDef
def propagateErrorInDef() raises:
    # CHECK: %a = lit.var.decl "a"
    # CHECK: %__call_result_tmp__ = lit.var.decl
    # CHECK: lit.call {{.*}}maybeRaises{{.*}}(%__error__, %__call_result_tmp__)
    var a = maybeRaises()


# CHECK-LABEL: lit.fn @"propagateErrorInRaisingFn
def propagateErrorInRaisingFn() raises:
    # CHECK:  %a = lit.var.decl {{.*}} : !lit.ref<:meta<!Int> #alias_Int,
    var a: Int
    # CHECK:  %__call_result_tmp__ = lit.var.decl
    # CHECK:  lit.call {{.*}}maybeRaises{{.*}}(%__error__, %__call_result_tmp__)
    a = maybeRaises()


# CHECK-LABEL: lit.fn @"propagateErrorInTry
def propagateErrorInTry():
    var a: Int
    # CHECK: lit.try
    try:
        # CHECK-NEXT: %__call_result_tmp__ = lit.var.decl
        # CHECK-NEXT: lit.call {{.*}}maybeRaises{{.*}}(%__try_error__, %__call_result_tmp__)
        # CHECK-NEXT: hlcf.if
        a = maybeRaises()
        # CHECK: lit.try.yield
    except:
        pass


# CHECK-LABEL: lit.fn @"raiseError
def raiseErrorInDef() raises:
    # CHECK: lit.call {{.*}}@Error::@"__init__{{.*}}(%__error__)
    # CHECK-NEXT: lit.raise
    raise Error()


# CHECK-LABEL: lit.fn @"raiseErrorInIf
def raiseErrorInIf(cond: Bool) raises:
    # CHECK: hlcf.if
    if cond:
        # CHECK: lit.call {{.*}}@Error::@"__init__{{.*}}(%__error__)
        # CHECK-NEXT: lit.raise
        raise Error()


# CHECK-LABEL: lit.fn @"raiseErrorInTry
def raiseErrorInTry():
    # CHECK: lit.try {
    try:
        # CHECK-NEXT: lit.call {{.*}}@Error::@"__init__{{.*}}(%__try_error__)
        # CHECK-NEXT: lit.raise
        raise Error()
    except:
        pass


# CHECK-LABEL: lit.fn @"rethrowsToRethrow
def rethrowsToRethrow():
    # CHECK: [[TRY_ERROR1:%.*]] = lit.var.decl "__try_error__"
    # CHECK-NEXT: lit.try {
    try:
        # CHECK: [[TRY_ERROR2:%.*]] = lit.var.decl "__try_error__"
        # CHECK-NEXT: lit.try {
        try:
            # CHECK: lit.call {{.*}}maybeRaises{{.*}}([[TRY_ERROR2]], %__call_result_tmp__
            maybeRaises()  # expected-warning {{'Int' value is unused; assign to '_' to discard the result}}
        # CHECK: } except {
        except:
            # CHECK-NEXT: lit.call {{.*}}Error::@"__init__{{.*}}"{{.*}}([[TRY_ERROR2]], [[TRY_ERROR1]]){{.*}}*, "move"
            # CHECK-NEXT: lit.raise
            raise
        # CHECK: }
    # CHECK: } except {
    except:
        # CHECK: lit.return %none
        return


struct S(ImplicitlyCopyable, Movable):
    var v: Int

    @implicit
    def __init__(out self, x: Int):
        self.v = x

    def __init__(out self) raises:
        self.v = 1


def fail() raises -> S:
    return 0


# CHECK-LABEL: lit.fn @"call_raising
def call_raising():
    # CHECK: lit.try {
    try:
        # CHECK: %__call_result_tmp__ = lit.var.decl
        # CHECK: [[ERR:%.*]] =  lit.call {{.*}}::@"fail{{.*}}(%e, %__call_result_tmp__)
        var x = fail()
        # CHECK: %y = lit.var.decl "y"
        # CHECK-NEXT: %__call_result_tmp___0 = lit.var.decl
        # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%e, %__call_result_tmp___0)
        var y = S()
        # CHECK: lit.try.yield
    except e:
        pass


def fail_raises() raises -> S:
    return fail()


def fail_register() raises -> Int:
    return 0


# CHECK-LABEL: lit.fn @"fail_register_raises
def fail_register_raises() raises -> Int:
    # CHECK-NEXT: %__call_result_tmp__ = lit.var.decl
    # CHECK-NEXT: call {{.*}}fail_register{{.*}}(%__error__, %__call_result_tmp__)
    # CHECK: [[FALSE:%.*]] = kgen.param.constant: scalar<bool> = <false>
    # CHECK-NEXT: lit.return [[FALSE]]
    return fail_register()


def raise_float32() raises Float32:
    raise 1.0


def raise_int() raises Int:
    raise 1


# CHECK-LABEL: lit.fn @"heterogeneous_try
def heterogeneous_try():
    # CHECK: %err = lit.var.decl {{.*}}Float32
    # CHECK-NEXT: %__try_error_tmp__ = lit.var.decl {{.*}}Int
    try:
        # CHECK: lit.call {{.*}}raise_float32{{.*}}(%err,
        raise_float32()
        # CHECK: lit.call {{.*}}raise_int{{.*}}(%__try_error_tmp__,
        # CHECK: lit.call {{.*}}SIMD::@"__init__{{.*}}
        # CHECK: lit.ref.store {{.*}}, %err
        # CHECK-NEXT: lit.raise
        raise_int()
    except err:
        var converted: Float32 = err


# CHECK-LABEL: lit.fn @"merged_numeric_try
def merged_numeric_try():
    # CHECK: %__try_error__ = lit.var.decl "__try_error__" synth : !lit.ref{{.*}}Int
    # CHECK-NEXT: %err = lit.var.decl "err" var : !lit.ref{{.*}}Float32
    try:
        # CHECK: %[[INT_ERROR:.*]] = lit.ref.load {{.*}} : {{.*}}Int
        # CHECK-NEXT: %[[FLOAT_ERROR:.*]] = lit.call tail {{.*}}SIMD::@"__init__{{.*}}(%[[INT_ERROR]])
        # CHECK-NEXT: %[[FLOAT_ERROR_ALIAS:.*]] = kgen.rebind %[[FLOAT_ERROR]]
        # CHECK-NEXT: lit.ref.store %[[FLOAT_ERROR_ALIAS]], %err : {{.*}}Float32
        raise_int()
        raise_float32()
        # CHECK: } except {
        # CHECK-NEXT: %converted = lit.var.decl "converted" var : !lit.ref{{.*}}Float32
        # CHECK-NEXT: %[[EXCEPT_ERROR:.*]] = lit.ref.load %err : {{.*}}Float32
        # CHECK-NEXT: lit.ref.store %[[EXCEPT_ERROR]], %converted : {{.*}}Float32
    except err:
        var converted: Float32 = err


# CHECK-LABEL: lit.fn @"typed_except
def typed_except():
    # CHECK: %__try_error__ = lit.var.decl "__try_error__" synth : !lit.ref{{.*}}Int
    # CHECK-NEXT: %err = lit.var.decl "err" var : !lit.ref{{.*}}Float32
    try:
        # CHECK: %[[INT_ERROR:.*]] = lit.ref.load {{.*}} : {{.*}}Int
        # CHECK-NEXT: %[[FLOAT_ERROR:.*]] = lit.call tail {{.*}}SIMD::@"__init__{{.*}}(%[[INT_ERROR]])
        # CHECK-NEXT: %[[FLOAT_ERROR_ALIAS:.*]] = kgen.rebind %[[FLOAT_ERROR]]
        # CHECK-NEXT: lit.ref.store %[[FLOAT_ERROR_ALIAS]], %err : {{.*}}Float32
        raise_int()
    except Float32 __as err:
        # CHECK: } except {
        # CHECK-NEXT: %converted = lit.var.decl "converted" var : !lit.ref{{.*}}Float32
        # CHECK-NEXT: %[[EXCEPT_ERROR:.*]] = lit.ref.load %err : {{.*}}Float32
        # CHECK-NEXT: lit.ref.store %[[EXCEPT_ERROR]], %converted : {{.*}}Float32
        var converted: Float32 = err


# CHECK-LABEL: lit.fn @"inferred_except_as
def inferred_except_as():
    # CHECK: %err = lit.var.decl "err" var : !lit.ref{{.*}}Int
    try:
        raise_int()
    except __as err:
        var converted: Int = err


struct FirstError(ErrorConversionTrait):
    def __init__(out self):
        pass


struct SecondError(ErrorConversionTrait):
    def __init__(out self):
        pass


def raise_first_error() raises FirstError:
    raise FirstError()


def raise_second_error() raises SecondError:
    raise SecondError()


# CHECK-LABEL: lit.fn @"multiple_typed_handlers
def multiple_typed_handlers(first: Bool):
    try:
        if first:
            # CHECK: lit.raise handler 0
            raise_first_error()
        else:
            # CHECK: lit.raise handler 1
            raise_second_error()
    except FirstError __as error:
        # CHECK: } except {
        # CHECK: lit.ownership.use %error : !lit.ref<!FirstError
        _ = error
    except SecondError __as error:
        # CHECK: } except {
        # CHECK: lit.ownership.use %error_0 : !lit.ref<!SecondError
        _ = error


# CHECK-LABEL: lit.fn @"typed_handler_with_catch_all
def typed_handler_with_catch_all(first: Bool):
    try:
        if first:
            raise_first_error()
        else:
            raise_int()
    except FirstError:
        pass
    except __as error:
        # CHECK: lit.ownership.use %error : {{.*}}Int
        _ = error


# CHECK-LABEL: lit.fn @"fallback_to_error_try
def fallback_to_error_try():
    # CHECK: %__try_error__ = lit.var.decl "__try_error__" synth : !lit.ref<!FirstError
    # CHECK-NEXT: %__try_error_tmp__ = lit.var.decl "__try_error_tmp__" synth : !lit.ref<!SecondError
    # CHECK-NEXT: %err = lit.var.decl "err" var : !lit.ref<!Error
    try:
        raise_first_error()
        raise_second_error()
    except err:
        # CHECK: } except {
        # CHECK-NEXT: %[[EXCEPT_ERROR:.*]] = lit.ref.immut %err : <!Error
        # CHECK-NEXT: {{.*}}lit.call {{.*}}Error::@"use{{.*}}(%[[EXCEPT_ERROR]])
        # This method is only valid on Error.
        err.use()
