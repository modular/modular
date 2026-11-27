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

# RUN: not %parse-mojo-isolated %s 2>&1 | FileCheck %s


trait T:
    comptime n: Int
    comptime m: Int


comptime get_m[P: T]: Int = P.m


comptime get_n[P: T]: Int = P.n


# A requirement whose value reads its own witness is a real cycle.
# CHECK: error: attempt to resolve a recursive reference to declaration 'SelfRef.m'
# CHECK: note: by declaration 'SelfRef.m'
struct SelfRef(T):
    comptime n = 1
    comptime m = get_m[Self]


# Resolving one witness still reports a requirement it does not satisfy.
# CHECK: error: 'WrongType' does not implement all requirements for 'T'
# CHECK: note: comptime member 'n' type 'StringLiteral["x"]' does not conform to trait's required type 'Int'
# CHECK: note: trait 'T' declared here
struct WrongType(T):
    comptime n = "x"
    comptime m = get_n[Self]


struct Box[k: Int]:
    pass


trait U:
    comptime n: Int

    def f(self, x: Box[0]):
        ...


comptime get_n_u[P: U]: Int = P.n


# Listing the missing requirements must not resolve `f`: its signature reads
# `k`, which is in flight while `k` demands the failing witness `n`.
# CHECK: error: 'Transitive' does not implement all requirements for 'U'
# CHECK: note: comptime member 'n' type 'StringLiteral["x"]' does not conform to trait's required type 'Int'
# CHECK: note: trait 'U' declared here
# CHECK-NOT: recursive reference
struct Transitive(U):
    comptime n = "x"
    comptime k = get_n_u[Self]

    def f(self, x: Box[Self.k]):
        pass


# One error lists every unsatisfied requirement, and no member reports a
# second time.
# CHECK: error: 'Missing' does not implement all requirements for 'T'
# CHECK: note: required member 'n' is not specified
# CHECK: note: required member 'm' is not specified
# CHECK: note: trait 'T' declared here
# CHECK-NOT: error: 'Missing' does not implement all requirements for 'T'
struct Missing(T):
    pass


trait V:
    comptime n: Int
    comptime m: Int

    def g(self):
        ...


# A mismatch lists the requirements that are missing alongside it.
# CHECK: error: 'MismatchAndMissing' does not implement all requirements for 'V'
# CHECK-DAG: note: comptime member 'n' type 'StringLiteral["x"]' does not conform to trait's required type 'Int'
# CHECK-DAG: note: required member 'm' is not specified
# CHECK-DAG: note: required function 'g' is not implemented
# CHECK: note: trait 'V' declared here
# CHECK-NOT: error: 'MismatchAndMissing' does not implement all requirements for 'V'
struct MismatchAndMissing(V):
    comptime n = "x"
