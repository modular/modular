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
# RUN: env MOJO_LAMBDA_PARTIAL_PARSING=1 %parse-mojo-isolated %s --kgen-print-inline-type-values | FileCheck %s

# A lambda whose signature is only partially typed infers its missing types
# from the contextual type. The closure is emitted only after a candidate is
# selected, so overload resolution does not emit one per candidate.


def take_lambda_param[t: def(Int) thin -> Int]():
    pass


def take_lambda_arg(T: Some[def(Int) -> Int]):
    pass


# A lambda that captures `y` becomes a closure struct that stores `y`. Its body
# adds the argument to the captured value.
# CHECK-LABEL: lit.struct.decl @"closure$main()::`lambda_1::__storage"
# CHECK: lit.struct.field y : !lit.ref<:meta<!Int> #alias_Int, imm *"y`">
# CHECK: lit.fn @"`lambda_1(::SIMD[DType.int, 1])`"{{.*}}(%0[*""]: {{.*}}, %1[*""]: !Int, |) -> !Int
# CHECK: lit.ref.struct.ger %0[y]
# CHECK: %[[Y:.*]] = lit.ref.load %{{.*}} : <!Int, imm *"y`">
# CHECK: lit.call tail @std::@builtin::@stubs::@SIMD::@"__add__(::SIMD[$0, $1],::SIMD[$0, $1])"{{.*}}(%1, %[[Y]])

# CHECK-LABEL: lit.fn @"main()"
# CHECK: lit.call tail @lambda_infer::@"take_lambda_param[def(::SIMD[DType.int, 1]) thin -> ::SIMD[DType.int, 1]]()"<:!lit.generator<(!Int, |) -> !alias_Int1> @lambda_infer::@"`lambda_0(::SIMD[DType.int, 1])`0x">()
# CHECK: %[[CLOSURE:.*]] = lit.var.decl "`lambda_1" var
# CHECK: lit.call @lambda_infer::@"closure$main()::`lambda_1::__storage"::@"__init__(::SIMD[DType.int, 1]%)"{{.*}}(%{{.*}}, %[[CLOSURE]])
# CHECK: lit.call @lambda_infer::@"take_lambda_arg[

# The lambda passed as a thin function parameter captures nothing, so it folds
# to a plain function.
# CHECK-LABEL: lit.fn @"`lambda_0(::SIMD[DType.int, 1])`0x"(%0[*""]: !Int, |) -> !alias_Int1
# CHECK-NEXT: %[[ONE:.*]] = kgen.param.constant: !Int = <{:scalar<index> 1}>
# CHECK-NEXT: %[[SUM:.*]] = lit.call tail @std::@builtin::@stubs::@SIMD::@"__add__(::SIMD[$0, $1],::SIMD[$0, $1])"{{.*}}(%0, %[[ONE]])
# CHECK-NEXT: %[[RET:.*]] = kgen.rebind %[[SUM]] : !Int to !alias_Int1
# CHECK-NEXT: lit.return %[[RET]] : !alias_Int1
def main():
    take_lambda_param[lambda (x): x + 1]()

    var y = 1
    take_lambda_arg(lambda (x): x + y)
