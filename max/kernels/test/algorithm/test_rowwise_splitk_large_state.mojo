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
"""A monoid whose width-1 state exceeds the split-K partial slot still
reduces correctly on shapes that pick the split-axis tier."""

from algorithm import rowwise
from algorithm.reduce_op import ReduceOp
from algorithm.rowwise_types import RowCoord
from layout import Coord
from std.sys import size_of
from std.testing import TestSuite, assert_equal, assert_true
from std.utils.index import Index

comptime dtype = DType.float32


struct BigSum[dtype: DType, W: Int](ReduceOp):
    """A sum carrying 256 bytes of unused internal state."""

    comptime Single = BigSum[Self.dtype, 1]
    comptime width = Self.W

    var acc: SIMD[Self.dtype, Self.W]
    var ballast: SIMD[DType.float32, 64]

    def __init__(out self):
        self.acc = 0
        self.ballast = 0

    def accumulate[
        val_dtype: DType, w: Int
    ](
        mut self,
        val: SIMD[val_dtype, w],
        idx: SIMD[.int64, w] = SIMD[.int64, w](0),
    ):
        self.acc += Self.pad[Self.W, Self.dtype, 0](val)

    def join(mut self, other: Self):
        self.acc += other.acc

    def __getitem__(self, j: Int) -> Self.Single:
        var r = Self.Single()
        r.acc[0] = self.acc[j]
        return r

    def __setitem__(mut self, j: Int, s: Self.Single):
        self.acc[j] = s.acc[0]


def _row_sums(rows: Int, cols: Int) raises -> List[Scalar[dtype]]:
    var x = List[Scalar[dtype]](length=rows * cols, fill=0)
    for i in range(rows * cols):
        x[i] = Scalar[dtype](i % 3)
    var out = List[Scalar[dtype]](length=rows, fill=-1)
    var x_ptr = x.unsafe_ptr()
    var out_ptr = out.unsafe_ptr()
    var shape = Coord(Index(rows, cols))

    @inline(.always)
    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut ctx: rowwise.Context[params]) {
        var cols, var x_ptr, var out_ptr
    }:
        comptime rank = row_coords.rank

        @inline(.always)
        def load[
            width: Int, alignment: Int
        ](idx: RowCoord[rank]) {var x_ptr, var cols} -> SIMD[dtype, width]:
            var i = Int(idx.coord[0]) * cols + Int(idx.coord[1])
            return x_ptr.unsafe_offset(i).unsafe_load[width=width]()

        var row = rowwise.Row[params, dtype, dtype, 1, rank, is_cached=False](
            row_coords, cols, ctx, load
        )

        @inline(.always)
        def ident[
            w: Int
        ](tile: SIMD[dtype, w], idx: RowCoord[rank]) {} -> SIMD[dtype, w]:
            return tile

        var acc = row.reduce[BigSum[dtype, params.simd_width]](ident, load).acc

        @inline(.always)
        def write(oc: RowCoord[rank]) {var acc, var out_ptr}:
            out_ptr.unsafe_offset(Int(oc.coord[0])).unsafe_store(acc[0])

        row.emit(write)

    rowwise.launch[axis=1, simd_width=1, target="cpu", num_phases=1](
        body, shape
    )
    _ = x^
    return out^


def _check(rows: Int, cols: Int) raises:
    var out = _row_sums(rows, cols)
    for r in range(rows):
        var expected = Scalar[dtype](0)
        for c in range(cols):
            expected += Scalar[dtype]((r * cols + c) % 3)
        assert_equal(out[r], expected)


def test_state_is_larger_than_the_slot() raises:
    assert_true(size_of[BigSum[dtype, 1]]() > 128)


def test_split_axis_shapes() raises:
    _check(1, 524288)
    _check(2, 524288)


def test_cooperative_shape() raises:
    _check(64, 1024)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
