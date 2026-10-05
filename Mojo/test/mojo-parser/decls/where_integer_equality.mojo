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
# RUN: %parse-mojo-isolated %s -split-input-file | FileCheck %s

# COM: `==` between `Int` parameters compares their extracted `_mlir_value`
# COM: fields, and scalar int-like `eq` canonicalizes to `#kgen.param.identical`
# COM: -- the same form type-value `==` already uses. Both are identity
# COM: assertions, so both merge into one n-ary class when conjoined.


def needs_equal[N: Int, K: Int]() where N == K:
    pass


# COM: Neither assumption states `N == K` on its own, so discharging the callee's
# COM: constraint takes transitivity across both.
# CHECK-LABEL: lit.fn @"transitive_split
def transitive_split[N: Int, M: Int, K: Int]() where N == M where M == K:
    # CHECK: lit.call tail @{{.*}}needs_equal
    needs_equal[N, K]()


# // -----


def needs_equal[N: Int, K: Int]() where N == K:
    pass


# COM: The same two facts as one conjunction, which the saturation flattens
# COM: before relating them.
# CHECK-LABEL: lit.fn @"transitive_conj
def transitive_conj[N: Int, M: Int, K: Int]() where N == M and M == K:
    # CHECK: lit.call tail @{{.*}}needs_equal
    needs_equal[N, K]()


# // -----


def needs_same[A: AnyType, B: AnyType]() where A == B:
    pass


# COM: Type-value identities merge the same way integer `eq` does.
# CHECK-LABEL: lit.fn @"transitive_types
def transitive_types[A: AnyType, B: AnyType, C: AnyType]() where A == C where B == C:
    # CHECK: lit.call tail @{{.*}}needs_same
    needs_same[A, B]()


# // -----


def needs_same[A: AnyType, B: AnyType]() where A == B:
    pass


# COM: `A == Int` and `A == Bool` is unsat, so every goal discharges.
# CHECK-LABEL: lit.fn @"ex_falso_int_bool
def ex_falso_int_bool[A: AnyType]() where A == Int where A == Bool:
    # CHECK: lit.call tail @{{.*}}needs_same
    needs_same[A, String]()


# // -----


def needs_zero[x: Int]() where x == 0:
    pass


# COM: Restating `x == 0` is implied: Int `eq` is `param.identical` at
# COM: construction, so the callee constraint is the same attr as the caller.
# CHECK-LABEL: lit.fn @"has_zero
def has_zero[x: Int]() where x == 0:
    # CHECK: lit.call tail @{{.*}}needs_zero
    needs_zero[x]()


# // -----


def needs_fp[dtype: DType]() where dtype.is_floating_point():
    pass


# COM: `is_floating_point()` canonicalizes to `not (identical (and ui8, mask), 0)`.
# COM: Restating that `not` must not drop the stored clause.
# CHECK-LABEL: lit.fn @"has_fp
def has_fp[dtype: DType]() where dtype.is_floating_point():
    # CHECK: lit.call tail @{{.*}}needs_fp
    needs_fp[dtype]()


# // -----


trait HasElem:
    comptime Elem: Movable


struct IntHolder(HasElem):
    comptime Elem = Int

    def __init__(out self):
        pass


struct Keyed[K: KeyElement]:
    @staticmethod
    def from_holder[
        T: HasElem
    ](holder: T) where T.Elem == Self.K and conforms_to(Self.K, Deinitable):
        pass


# COM: Inside a conjunction the identity class rebinds `K` to the metatype of
# COM: `T.Elem`. Inference must still see `K` itself to bind it from `T.Elem`.
# CHECK-LABEL: lit.fn @"infer_parent_through_conjunction
def infer_parent_through_conjunction():
    # CHECK: lit.call @{{.*}}from_holder
    Keyed.from_holder(IntHolder())
