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

# Referencing a struct by name binds every parameter to unbound. When a
# parameter's type applies an alias to an earlier parameter, the application
# gets a hole argument, which `bind_params` would read as leaving that
# parameter open. The alias can't spell the result, so only its expansion is
# kept. This used to crash the parameter evaluator.


struct Pair[A: AnyType, B: AnyType](Movable where False):
    pass


comptime Wrap[T: AnyType] = Pair[T, Int]


struct TakesWrapFn[T: AnyType, //, f: def(Int) thin -> Wrap[T]]:
    @staticmethod
    def run():
        pass


struct TakesNestedWrapFn[
    T: AnyType, //, f: def(Int) thin -> Wrap[Pair[T, T]]
]:
    @staticmethod
    def run():
        pass


# CHECK-LABEL: lit.fn @"bind_alias_typed_fn_param
def bind_alias_typed_fn_param[T: AnyType]():
    def f(i: Int) -> Wrap[T]:
        abort()

    def g(i: Int) -> Wrap[Pair[T, T]]:
        abort()

    # CHECK: TakesWrapFn{{.*}}::@"run()"
    TakesWrapFn[f].run()
    # CHECK: TakesNestedWrapFn{{.*}}::@"run()"
    TakesNestedWrapFn[g].run()


# CHECK-LABEL: lit.fn @"unbound_reference()"
def unbound_reference():
    # The hole for `f` gets the plain expansion `Pair[?, Int]` as its type;
    # the signature itself keeps the `Wrap[T]` sugar.
    # CHECK: alias.decl *"UnboundRef{{.*}}TakesWrapFn <:!AnyType ?, :!lit.generator<[1](!Int, |, ?, "__result__": !lit.ref<!lit.struct<#Pair <:!AnyType ?, :!AnyType !Int>>
    comptime UnboundRef = TakesWrapFn
