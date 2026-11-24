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

# Regression test for MOCO-5357. The alias chains below expand to parameter
# expressions that share subexpressions at every level. InlineParametric must
# mangle each shared node once. Mangling it once per path takes time
# exponential in the chain depth.
#
# RUN: %mojo %s | FileCheck %s

from std.utils.coord import ComptimeInt, Coord, CoordLike, Idx


# Names `A` three times and cannot fold while `A` is symbolic.
comptime M[A: CoordLike]: CoordLike = ComptimeInt[
    A.static_value
] if A.is_static_value else Scalar[A.DTYPE]

comptime Acc0[A: CoordLike]: CoordLike = A
comptime Acc1[A: CoordLike]: CoordLike = Coord[Acc0[M[A]]]
comptime Acc2[A: CoordLike]: CoordLike = Coord[Acc1[M[A]]]
comptime Acc3[A: CoordLike]: CoordLike = Coord[Acc2[M[A]]]
comptime Acc4[A: CoordLike]: CoordLike = Coord[Acc3[M[A]]]
comptime Acc5[A: CoordLike]: CoordLike = Coord[Acc4[M[A]]]
comptime Acc6[A: CoordLike]: CoordLike = Coord[Acc5[M[A]]]
comptime Acc7[A: CoordLike]: CoordLike = Coord[Acc6[M[A]]]
comptime Acc8[A: CoordLike]: CoordLike = Coord[Acc7[M[A]]]
comptime Acc9[A: CoordLike]: CoordLike = Coord[Acc8[M[A]]]
comptime Acc10[A: CoordLike]: CoordLike = Coord[Acc9[M[A]]]
comptime Acc11[A: CoordLike]: CoordLike = Coord[Acc10[M[A]]]
comptime Acc12[A: CoordLike]: CoordLike = Coord[Acc11[M[A]]]
comptime Acc13[A: CoordLike]: CoordLike = Coord[Acc12[M[A]]]
comptime Acc14[A: CoordLike]: CoordLike = Coord[Acc13[M[A]]]
comptime Acc15[A: CoordLike]: CoordLike = Coord[Acc14[M[A]]]
comptime Acc16[A: CoordLike]: CoordLike = Coord[Acc15[M[A]]]

comptime Proj0[A: CoordLike]: CoordLike = A
comptime Proj1[A: CoordLike]: CoordLike = Coord[Proj0[A.ParamListType[0]]]
comptime Proj2[A: CoordLike]: CoordLike = Coord[Proj1[A.ParamListType[0]]]
comptime Proj3[A: CoordLike]: CoordLike = Coord[Proj2[A.ParamListType[0]]]
comptime Proj4[A: CoordLike]: CoordLike = Coord[Proj3[A.ParamListType[0]]]
comptime Proj5[A: CoordLike]: CoordLike = Coord[Proj4[A.ParamListType[0]]]
comptime Proj6[A: CoordLike]: CoordLike = Coord[Proj5[A.ParamListType[0]]]
comptime Proj7[A: CoordLike]: CoordLike = Coord[Proj6[A.ParamListType[0]]]
comptime Proj8[A: CoordLike]: CoordLike = Coord[Proj7[A.ParamListType[0]]]
comptime Proj9[A: CoordLike]: CoordLike = Coord[Proj8[A.ParamListType[0]]]
comptime Proj10[A: CoordLike]: CoordLike = Coord[Proj9[A.ParamListType[0]]]
comptime Proj11[A: CoordLike]: CoordLike = Coord[Proj10[A.ParamListType[0]]]
comptime Proj12[A: CoordLike]: CoordLike = Coord[Proj11[A.ParamListType[0]]]
comptime Proj13[A: CoordLike]: CoordLike = Coord[Proj12[A.ParamListType[0]]]
comptime Proj14[A: CoordLike]: CoordLike = Coord[Proj13[A.ParamListType[0]]]
comptime Proj15[A: CoordLike]: CoordLike = Coord[Proj14[A.ParamListType[0]]]
comptime Proj16[A: CoordLike]: CoordLike = Coord[Proj15[A.ParamListType[0]]]
comptime Proj17[A: CoordLike]: CoordLike = Coord[Proj16[A.ParamListType[0]]]
comptime Proj18[A: CoordLike]: CoordLike = Coord[Proj17[A.ParamListType[0]]]
comptime Proj19[A: CoordLike]: CoordLike = Coord[Proj18[A.ParamListType[0]]]
comptime Proj20[A: CoordLike]: CoordLike = Coord[Proj19[A.ParamListType[0]]]
comptime Proj21[A: CoordLike]: CoordLike = Coord[Proj20[A.ParamListType[0]]]
comptime Proj22[A: CoordLike]: CoordLike = Coord[Proj21[A.ParamListType[0]]]
comptime Proj23[A: CoordLike]: CoordLike = Coord[Proj22[A.ParamListType[0]]]
comptime Proj24[A: CoordLike]: CoordLike = Coord[Proj23[A.ParamListType[0]]]
comptime Proj25[A: CoordLike]: CoordLike = Coord[Proj24[A.ParamListType[0]]]
comptime Proj26[A: CoordLike]: CoordLike = Coord[Proj25[A.ParamListType[0]]]
comptime Proj27[A: CoordLike]: CoordLike = Coord[Proj26[A.ParamListType[0]]]
comptime Proj28[A: CoordLike]: CoordLike = Coord[Proj27[A.ParamListType[0]]]
comptime Proj29[A: CoordLike]: CoordLike = Coord[Proj28[A.ParamListType[0]]]
comptime Proj30[A: CoordLike]: CoordLike = Coord[Proj29[A.ParamListType[0]]]
comptime Proj31[A: CoordLike]: CoordLike = Coord[Proj30[A.ParamListType[0]]]
comptime Proj32[A: CoordLike]: CoordLike = Coord[Proj31[A.ParamListType[0]]]


struct H[T: CoordLike](TrivialRegisterPassable):
    var c: Self.T

    @inline(.always)
    def __init__(out self):
        self.c = {}


@inline(.always)
def acc[A: CoordLike](a: A) -> H[Acc16[A]]:
    return {}


@inline(.always)
def proj[A: CoordLike](a: A) -> H[Proj32[A]]:
    return {}


def main():
    _ = acc(Idx[1])
    _ = proj(Idx[1])
    # CHECK: done
    print("done")
