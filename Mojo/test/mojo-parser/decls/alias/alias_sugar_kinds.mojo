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

# A direct alias reference is Alias sugar over the declaration reference. An
# application of parameters to an alias is Preserved sugar over the
# `bind_params` on that reference, with the expansion as the other side.


struct Tile[n: Int](ImplicitlyCopyable):
    var x: Int

    def __init__(out self):
        self.x = 0


comptime DoubledTile[n: Int] = Tile[n * 2]
comptime QuadTile = Tile[4]


# CHECK-LABEL: lit.fn @"sugar_kinds()"
def sugar_kinds():
    # CHECK: lit.var.decl "direct" {{.*}}#alias_QuadTile
    # CHECK-NOT: sugar_preserved(
    var direct: QuadTile = {}
    # CHECK: lit.var.decl "applied" {{.*}}sugar_preserved(
    var applied: DoubledTile[2] = {}
    _ = direct
    _ = applied
