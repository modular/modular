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

# A generator whose result type is an alias to an origin-parameterized struct
# reaches `bind_params` behind a rebind that only changes type sugar. It must
# still fold at parse time rather than when canonicalization strips the rebind.


@fieldwise_init
struct OBox[o: ImmOrigin](Copyable, Movable):
    var x: Int


comptime Aliased = OBox[ImmStaticOrigin]

comptime Mapper[id: Int]: Aliased = Aliased(1)

# CHECK-LABEL: lit.alias.decl *"Names`{{[^"]*}}": !lit.struct<#ParameterList
# CHECK-NOT: bind_params
# CHECK-SAME: rebind(:!Int {:scalar<index> 1}))]>)>
comptime Names = ParameterList.tabulate[2, Mapper]()
