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

# Resolving `S.m` asks for witness `n` only. Checking the whole conformance
# would reach `S.m` while it is being resolved and report a false cycle.


trait T:
    comptime n: Int
    comptime m: Int


comptime get_n[P: T]: Int = P.n


comptime get_m[P: T]: Int = P.m


# CHECK-LABEL: lit.struct.decl @S(
# CHECK: kgen.conformance @conformance_member_lazy::@T {
# CHECK-NEXT: kgen.witness "n" : !alias_Int1 = rebind(:!Int {:scalar<index> 1})
# CHECK-NEXT: kgen.witness "m" : !alias_Int1 = sugar_member_alias(!S, "n", rebind(:!Int {:scalar<index> 1}))
struct S(T):
    comptime n = 1
    comptime m = get_n[Self]


# Witness `m` is installed first, while `n` is still being resolved, and `n`
# is inserted before it to keep trait order.
# CHECK-LABEL: lit.struct.decl @R(
# CHECK: kgen.conformance @conformance_member_lazy::@T {
# CHECK-NEXT: kgen.witness "n" : !alias_Int1 = sugar_member_alias(!R, "m", rebind(:!Int {:scalar<index> 2}))
# CHECK-NEXT: kgen.witness "m" : !alias_Int1 = rebind(:!Int {:scalar<index> 2})
struct R(T):
    comptime n = get_m[Self]
    comptime m = 2
