# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""The module implements Transpose functions."""

from math import ceildiv
from sys.info import simd_width_of
from sys.intrinsics import strided_load, strided_store

from algorithm import parallel_memcpy, sync_parallelize, tile, vectorize
from buffer import NDBuffer
from buffer.dimlist import DimList
from layout import (
    LayoutTensor,
    Layout,
    RuntimeLayout,
    RuntimeTuple,
    UNKNOWN_VALUE,
)
from layout.int_tuple import fill_like
from layout.layout import is_row_major
from memory import memcpy
from runtime.asyncrt import parallelism_level

from utils.index import IndexList, StaticTuple


fn _transpose_inplace_4x4[
    rows: Int,
    cols: Int,
    dtype: DType,
](bufloat0: NDBuffer[mut=True, dtype, 2, _, DimList(rows, cols)]):
    constrained[rows == 4]()
    constrained[cols == 4]()
    var buf = rebind[
        NDBuffer[
            dtype,
            2,
            bufloat0.origin,
            DimList(4, 4),
        ],
    ](bufloat0)

    var row0 = buf.load[width=4](IndexList[2](0, 0))
    var row1 = buf.load[width=4](IndexList[2](1, 0))
    var row2 = buf.load[width=4](IndexList[2](2, 0))
    var row3 = buf.load[width=4](IndexList[2](3, 0))

    var tmp0 = row0.shuffle[0, 1, 4, 5](row1)
    var tmp1 = row2.shuffle[0, 1, 4, 5](row3)
    var tmp2 = row0.shuffle[2, 3, 6, 7](row1)
    var tmp3 = row2.shuffle[2, 3, 6, 7](row3)

    var r0 = tmp0.shuffle[0, 2, 4, 6](tmp1)
    var r1 = tmp0.shuffle[1, 3, 5, 7](tmp1)
    var r2 = tmp2.shuffle[0, 2, 4, 6](tmp3)
    var r3 = tmp2.shuffle[1, 3, 5, 7](tmp3)

    buf.store[width=4](IndexList[2](0, 0), r0)
    buf.store[width=4](IndexList[2](1, 0), r1)
    buf.store[width=4](IndexList[2](2, 0), r2)
    buf.store[width=4](IndexList[2](3, 0), r3)


fn _transpose_inplace_4x4[
    dtype: DType,
](bufloat0: LayoutTensor[mut=True, dtype, **_]):
    alias rows = Int(bufloat0.layout.shape[0])
    alias cols = Int(bufloat0.layout.shape[1])

    constrained[rows == 4]()
    constrained[cols == 4]()
    var buf = bufloat0.reshape[Layout.row_major(4, 4)]()

    var idx0 = buf.runtime_layout(
        RuntimeTuple[fill_like(buf.layout.shape, UNKNOWN_VALUE)](
            IndexList[2](0, 0)
        )
    )
    var row0 = buf.ptr.load[width=4](idx0)
    var idx1 = buf.runtime_layout(
        RuntimeTuple[fill_like(buf.layout.shape, UNKNOWN_VALUE)](
            IndexList[2](1, 0)
        )
    )
    var row1 = buf.ptr.load[width=4](idx1)
    var idx2 = buf.runtime_layout(
        RuntimeTuple[fill_like(buf.layout.shape, UNKNOWN_VALUE)](
            IndexList[2](2, 0)
        )
    )
    var row2 = buf.ptr.load[width=4](idx2)
    var idx3 = buf.runtime_layout(
        RuntimeTuple[fill_like(buf.layout.shape, UNKNOWN_VALUE)](
            IndexList[2](3, 0)
        )
    )
    var row3 = buf.ptr.load[width=4](idx3)

    var tmp0 = row0.shuffle[0, 1, 4, 5](row1)
    var tmp1 = row2.shuffle[0, 1, 4, 5](row3)
    var tmp2 = row0.shuffle[2, 3, 6, 7](row1)
    var tmp3 = row2.shuffle[2, 3, 6, 7](row3)

    var r0 = tmp0.shuffle[0, 2, 4, 6](tmp1)
    var r1 = tmp0.shuffle[1, 3, 5, 7](tmp1)
    var r2 = tmp2.shuffle[0, 2, 4, 6](tmp3)
    var r3 = tmp2.shuffle[1, 3, 5, 7](tmp3)

    buf.ptr.store[width=4](idx0, r0)
    buf.ptr.store[width=4](idx1, r1)
    buf.ptr.store[width=4](idx2, r2)
    buf.ptr.store[width=4](idx3, r3)


fn _transpose_inplace_8x8[
    rows: Int,
    cols: Int,
    dtype: DType,
](bufloat0: NDBuffer[mut=True, dtype, 2, _, DimList(rows, cols)]):
    constrained[rows == 8]()
    constrained[cols == 8]()
    var buf = rebind[
        NDBuffer[
            dtype,
            2,
            bufloat0.origin,
            DimList(8, 8),
        ],
    ](bufloat0)

    var row0 = buf.load[width=8](IndexList[2](0, 0))
    var row1 = buf.load[width=8](IndexList[2](1, 0))
    var row2 = buf.load[width=8](IndexList[2](2, 0))
    var row3 = buf.load[width=8](IndexList[2](3, 0))
    var row4 = buf.load[width=8](IndexList[2](4, 0))
    var row5 = buf.load[width=8](IndexList[2](5, 0))
    var row6 = buf.load[width=8](IndexList[2](6, 0))
    var row7 = buf.load[width=8](IndexList[2](7, 0))

    @parameter
    fn _apply_permute_0(
        vec: SIMD[dtype, 8], other: SIMD[dtype, 8]
    ) -> SIMD[dtype, 8]:
        return vec.shuffle[0, 8, 1, 9, 4, 12, 5, 13](other)

    @parameter
    fn _apply_permute_1(
        vec: SIMD[dtype, 8], other: SIMD[dtype, 8]
    ) -> SIMD[dtype, 8]:
        return vec.shuffle[2, 10, 3, 11, 6, 14, 7, 15](other)

    var k0 = _apply_permute_0(row0, row1)
    var k1 = _apply_permute_1(row0, row1)
    var k2 = _apply_permute_0(row2, row3)
    var k3 = _apply_permute_1(row2, row3)
    var k4 = _apply_permute_0(row4, row5)
    var k5 = _apply_permute_1(row4, row5)
    var k6 = _apply_permute_0(row6, row7)
    var k7 = _apply_permute_1(row6, row7)

    @parameter
    fn _apply_permute_2(
        vec: SIMD[dtype, 8], other: SIMD[dtype, 8]
    ) -> SIMD[dtype, 8]:
        return vec.shuffle[0, 1, 8, 9, 4, 5, 12, 13](other)

    @parameter
    fn _apply_permute_3(
        vec: SIMD[dtype, 8], other: SIMD[dtype, 8]
    ) -> SIMD[dtype, 8]:
        return vec.shuffle[2, 3, 10, 11, 6, 7, 14, 15](other)

    var k020 = _apply_permute_2(k0, k2)
    var k021 = _apply_permute_3(k0, k2)
    var k130 = _apply_permute_2(k1, k3)
    var k131 = _apply_permute_3(k1, k3)
    var k460 = _apply_permute_2(k4, k6)
    var k461 = _apply_permute_3(k4, k6)
    var k570 = _apply_permute_2(k5, k7)
    var k571 = _apply_permute_3(k5, k7)

    @parameter
    fn _apply_permute_4(
        vec: SIMD[dtype, 8], other: SIMD[dtype, 8]
    ) -> SIMD[dtype, 8]:
        return vec.shuffle[0, 1, 2, 3, 8, 9, 10, 11](other)

    @parameter
    fn _apply_permute_5(
        vec: SIMD[dtype, 8], other: SIMD[dtype, 8]
    ) -> SIMD[dtype, 8]:
        return vec.shuffle[4, 5, 6, 7, 12, 13, 14, 15](other)

    var r0 = _apply_permute_4(k020, k460)
    var r1 = _apply_permute_4(k021, k461)
    var r2 = _apply_permute_4(k130, k570)
    var r3 = _apply_permute_4(k131, k571)
    var r4 = _apply_permute_5(k020, k460)
    var r5 = _apply_permute_5(k021, k461)
    var r6 = _apply_permute_5(k130, k570)
    var r7 = _apply_permute_5(k131, k571)

    buf.store[width=8](IndexList[2](0, 0), r0)
    buf.store[width=8](IndexList[2](1, 0), r1)
    buf.store[width=8](IndexList[2](2, 0), r2)
    buf.store[width=8](IndexList[2](3, 0), r3)
    buf.store[width=8](IndexList[2](4, 0), r4)
    buf.store[width=8](IndexList[2](5, 0), r5)
    buf.store[width=8](IndexList[2](6, 0), r6)
    buf.store[width=8](IndexList[2](7, 0), r7)


fn _transpose_inplace_8x8[
    dtype: DType,
](bufloat0: LayoutTensor[mut=True, dtype, **_]):
    alias rows = Int(bufloat0.layout.shape[0])
    alias cols = Int(bufloat0.layout.shape[1])
    constrained[rows == 8]()
    constrained[cols == 8]()

    var buf = bufloat0.reshape[Layout.row_major(8, 8)]()

    var idx0 = buf.runtime_layout(
        RuntimeTuple[fill_like(buf.layout.shape, UNKNOWN_VALUE)](
            IndexList[2](0, 0)
        )
    )
    var row0 = buf.ptr.load[width=8](idx0)
    var idx1 = buf.runtime_layout(
        RuntimeTuple[fill_like(buf.layout.shape, UNKNOWN_VALUE)](
            IndexList[2](1, 0)
        )
    )
    var row1 = buf.ptr.load[width=8](idx1)
    var idx2 = buf.runtime_layout(
        RuntimeTuple[fill_like(buf.layout.shape, UNKNOWN_VALUE)](
            IndexList[2](2, 0)
        )
    )
    var row2 = buf.ptr.load[width=8](idx2)
    var idx3 = buf.runtime_layout(
        RuntimeTuple[fill_like(buf.layout.shape, UNKNOWN_VALUE)](
            IndexList[2](3, 0)
        )
    )
    var row3 = buf.ptr.load[width=8](idx3)
    var idx4 = buf.runtime_layout(
        RuntimeTuple[fill_like(buf.layout.shape, UNKNOWN_VALUE)](
            IndexList[2](4, 0)
        )
    )
    var row4 = buf.ptr.load[width=8](idx4)
    var idx5 = buf.runtime_layout(
        RuntimeTuple[fill_like(buf.layout.shape, UNKNOWN_VALUE)](
            IndexList[2](5, 0)
        )
    )
    var row5 = buf.ptr.load[width=8](idx5)
    var idx6 = buf.runtime_layout(
        RuntimeTuple[fill_like(buf.layout.shape, UNKNOWN_VALUE)](
            IndexList[2](6, 0)
        )
    )
    var row6 = buf.ptr.load[width=8](idx6)
    var idx7 = buf.runtime_layout(
        RuntimeTuple[fill_like(buf.layout.shape, UNKNOWN_VALUE)](
            IndexList[2](7, 0)
        )
    )
    var row7 = buf.ptr.load[width=8](idx7)

    @parameter
    fn _apply_permute_0(
        vec: SIMD[dtype, 8], other: SIMD[dtype, 8]
    ) -> SIMD[dtype, 8]:
        return vec.shuffle[0, 8, 1, 9, 4, 12, 5, 13](other)

    @parameter
    fn _apply_permute_1(
        vec: SIMD[dtype, 8], other: SIMD[dtype, 8]
    ) -> SIMD[dtype, 8]:
        return vec.shuffle[2, 10, 3, 11, 6, 14, 7, 15](other)

    var k0 = _apply_permute_0(row0, row1)
    var k1 = _apply_permute_1(row0, row1)
    var k2 = _apply_permute_0(row2, row3)
    var k3 = _apply_permute_1(row2, row3)
    var k4 = _apply_permute_0(row4, row5)
    var k5 = _apply_permute_1(row4, row5)
    var k6 = _apply_permute_0(row6, row7)
    var k7 = _apply_permute_1(row6, row7)

    @parameter
    fn _apply_permute_2(
        vec: SIMD[dtype, 8], other: SIMD[dtype, 8]
    ) -> SIMD[dtype, 8]:
        return vec.shuffle[0, 1, 8, 9, 4, 5, 12, 13](other)

    @parameter
    fn _apply_permute_3(
        vec: SIMD[dtype, 8], other: SIMD[dtype, 8]
    ) -> SIMD[dtype, 8]:
        return vec.shuffle[2, 3, 10, 11, 6, 7, 14, 15](other)

    var k020 = _apply_permute_2(k0, k2)
    var k021 = _apply_permute_3(k0, k2)
    var k130 = _apply_permute_2(k1, k3)
    var k131 = _apply_permute_3(k1, k3)
    var k460 = _apply_permute_2(k4, k6)
    var k461 = _apply_permute_3(k4, k6)
    var k570 = _apply_permute_2(k5, k7)
    var k571 = _apply_permute_3(k5, k7)

    @parameter
    fn _apply_permute_4(
        vec: SIMD[dtype, 8], other: SIMD[dtype, 8]
    ) -> SIMD[dtype, 8]:
        return vec.shuffle[0, 1, 2, 3, 8, 9, 10, 11](other)

    @parameter
    fn _apply_permute_5(
        vec: SIMD[dtype, 8], other: SIMD[dtype, 8]
    ) -> SIMD[dtype, 8]:
        return vec.shuffle[4, 5, 6, 7, 12, 13, 14, 15](other)

    var r0 = _apply_permute_4(k020, k460)
    var r1 = _apply_permute_4(k021, k461)
    var r2 = _apply_permute_4(k130, k570)
    var r3 = _apply_permute_4(k131, k571)
    var r4 = _apply_permute_5(k020, k460)
    var r5 = _apply_permute_5(k021, k461)
    var r6 = _apply_permute_5(k130, k570)
    var r7 = _apply_permute_5(k131, k571)

    buf.ptr.store[width=8](idx0, r0)
    buf.ptr.store[width=8](idx1, r1)
    buf.ptr.store[width=8](idx2, r2)
    buf.ptr.store[width=8](idx3, r3)
    buf.ptr.store[width=8](idx4, r4)
    buf.ptr.store[width=8](idx5, r5)
    buf.ptr.store[width=8](idx6, r6)
    buf.ptr.store[width=8](idx7, r7)


fn _transpose_inplace_16x16[
    rows: Int,
    cols: Int,
    dtype: DType,
](bufloat0: NDBuffer[mut=True, dtype, 2, _, DimList(rows, cols)]):
    constrained[rows == 16]()
    constrained[cols == 16]()
    var buf = rebind[
        NDBuffer[
            dtype,
            2,
            bufloat0.origin,
            DimList(16, 16),
        ],
    ](bufloat0)

    @parameter
    fn _apply_permute_0(
        vec: SIMD[dtype, 16], other: SIMD[dtype, 16]
    ) -> SIMD[dtype, 16]:
        return vec.shuffle[
            0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29
        ](other)

    @parameter
    fn _apply_permute_1(
        vec: SIMD[dtype, 16], other: SIMD[dtype, 16]
    ) -> SIMD[dtype, 16]:
        return vec.shuffle[
            2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31
        ](other)

    @parameter
    fn _apply_permute_2(
        vec: SIMD[dtype, 16], other: SIMD[dtype, 16]
    ) -> SIMD[dtype, 16]:
        return vec.shuffle[
            0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29
        ](other)

    @parameter
    fn _apply_permute_3(
        vec: SIMD[dtype, 16], other: SIMD[dtype, 16]
    ) -> SIMD[dtype, 16]:
        return vec.shuffle[
            2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31
        ](other)

    @parameter
    fn _apply_permute_4(
        vec: SIMD[dtype, 16], other: SIMD[dtype, 16]
    ) -> SIMD[dtype, 16]:
        return vec.shuffle[
            0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27
        ](other)

    @parameter
    fn _apply_permute_5(
        vec: SIMD[dtype, 16], other: SIMD[dtype, 16]
    ) -> SIMD[dtype, 16]:
        return vec.shuffle[
            4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31
        ](other)

    @parameter
    fn _apply_permute_6(
        vec: SIMD[dtype, 16], other: SIMD[dtype, 16]
    ) -> SIMD[dtype, 16]:
        return vec.shuffle[
            0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27
        ](other)

    @parameter
    fn _apply_permute_7(
        vec: SIMD[dtype, 16], other: SIMD[dtype, 16]
    ) -> SIMD[dtype, 16]:
        return vec.shuffle[
            4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31
        ](other)

    var row00 = buf.load[width=16](IndexList[2](0, 0))
    var row01 = buf.load[width=16](IndexList[2](1, 0))
    var row02 = buf.load[width=16](IndexList[2](2, 0))
    var row03 = buf.load[width=16](IndexList[2](3, 0))
    var row04 = buf.load[width=16](IndexList[2](4, 0))
    var row05 = buf.load[width=16](IndexList[2](5, 0))
    var row06 = buf.load[width=16](IndexList[2](6, 0))
    var row07 = buf.load[width=16](IndexList[2](7, 0))
    var row08 = buf.load[width=16](IndexList[2](8, 0))
    var row09 = buf.load[width=16](IndexList[2](9, 0))
    var row10 = buf.load[width=16](IndexList[2](10, 0))
    var row11 = buf.load[width=16](IndexList[2](11, 0))
    var row12 = buf.load[width=16](IndexList[2](12, 0))
    var row13 = buf.load[width=16](IndexList[2](13, 0))
    var row14 = buf.load[width=16](IndexList[2](14, 0))
    var row15 = buf.load[width=16](IndexList[2](15, 0))

    var k00 = _apply_permute_0(row00, row01)
    var k01 = _apply_permute_1(row00, row01)
    var k02 = _apply_permute_0(row02, row03)
    var k03 = _apply_permute_1(row02, row03)
    var k04 = _apply_permute_0(row04, row05)
    var k05 = _apply_permute_1(row04, row05)
    var k06 = _apply_permute_0(row06, row07)
    var k07 = _apply_permute_1(row06, row07)
    var k08 = _apply_permute_0(row08, row09)
    var k09 = _apply_permute_1(row08, row09)
    var k10 = _apply_permute_0(row10, row11)
    var k11 = _apply_permute_1(row10, row11)
    var k12 = _apply_permute_0(row12, row13)
    var k13 = _apply_permute_1(row12, row13)
    var k14 = _apply_permute_0(row14, row15)
    var k15 = _apply_permute_1(row14, row15)

    var j00 = _apply_permute_2(k00, k02)
    var j01 = _apply_permute_3(k00, k02)
    var j02 = _apply_permute_2(k01, k03)
    var j03 = _apply_permute_3(k01, k03)
    var j04 = _apply_permute_2(k04, k06)
    var j05 = _apply_permute_3(k04, k06)
    var j06 = _apply_permute_2(k05, k07)
    var j07 = _apply_permute_3(k05, k07)
    var j08 = _apply_permute_2(k08, k10)
    var j09 = _apply_permute_3(k08, k10)
    var j10 = _apply_permute_2(k09, k11)
    var j11 = _apply_permute_3(k09, k11)
    var j12 = _apply_permute_2(k12, k14)
    var j13 = _apply_permute_3(k12, k14)
    var j14 = _apply_permute_2(k13, k15)
    var j15 = _apply_permute_3(k13, k15)

    var t00 = _apply_permute_4(j00, j04)
    var t01 = _apply_permute_4(j01, j05)
    var t02 = _apply_permute_4(j02, j06)
    var t03 = _apply_permute_4(j03, j07)
    var t04 = _apply_permute_5(j00, j04)
    var t05 = _apply_permute_5(j01, j05)
    var t06 = _apply_permute_5(j02, j06)
    var t07 = _apply_permute_5(j03, j07)
    var t08 = _apply_permute_4(j08, j12)
    var t09 = _apply_permute_4(j09, j13)
    var t10 = _apply_permute_4(j10, j14)
    var t11 = _apply_permute_4(j11, j15)
    var t12 = _apply_permute_5(j08, j12)
    var t13 = _apply_permute_5(j09, j13)
    var t14 = _apply_permute_5(j10, j14)
    var t15 = _apply_permute_5(j11, j15)

    var r00 = _apply_permute_6(t00, t08)
    var r01 = _apply_permute_6(t01, t09)
    var r02 = _apply_permute_6(t02, t10)
    var r03 = _apply_permute_6(t03, t11)
    var r04 = _apply_permute_6(t04, t12)
    var r05 = _apply_permute_6(t05, t13)
    var r06 = _apply_permute_6(t06, t14)
    var r07 = _apply_permute_6(t07, t15)
    var r08 = _apply_permute_7(t00, t08)
    var r09 = _apply_permute_7(t01, t09)
    var r10 = _apply_permute_7(t02, t10)
    var r11 = _apply_permute_7(t03, t11)
    var r12 = _apply_permute_7(t04, t12)
    var r13 = _apply_permute_7(t05, t13)
    var r14 = _apply_permute_7(t06, t14)
    var r15 = _apply_permute_7(t07, t15)

    buf.store[width=16](IndexList[2](0, 0), r00)
    buf.store[width=16](IndexList[2](1, 0), r01)
    buf.store[width=16](IndexList[2](2, 0), r02)
    buf.store[width=16](IndexList[2](3, 0), r03)
    buf.store[width=16](IndexList[2](4, 0), r04)
    buf.store[width=16](IndexList[2](5, 0), r05)
    buf.store[width=16](IndexList[2](6, 0), r06)
    buf.store[width=16](IndexList[2](7, 0), r07)
    buf.store[width=16](IndexList[2](8, 0), r08)
    buf.store[width=16](IndexList[2](9, 0), r09)
    buf.store[width=16](IndexList[2](10, 0), r10)
    buf.store[width=16](IndexList[2](11, 0), r11)
    buf.store[width=16](IndexList[2](12, 0), r12)
    buf.store[width=16](IndexList[2](13, 0), r13)
    buf.store[width=16](IndexList[2](14, 0), r14)
    buf.store[width=16](IndexList[2](15, 0), r15)


fn _transpose_inplace_16x16[
    dtype: DType,
](bufloat0: LayoutTensor[mut=True, dtype, **_]):
    alias rows = Int(bufloat0.layout.shape[0])
    alias cols = Int(bufloat0.layout.shape[1])
    constrained[rows == 16]()
    constrained[cols == 16]()

    var buf = bufloat0.reshape[Layout.row_major(16, 16)]()

    @parameter
    fn _apply_permute_0(
        vec: SIMD[dtype, 16], other: SIMD[dtype, 16]
    ) -> SIMD[dtype, 16]:
        return vec.shuffle[
            0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29
        ](other)

    @parameter
    fn _apply_permute_1(
        vec: SIMD[dtype, 16], other: SIMD[dtype, 16]
    ) -> SIMD[dtype, 16]:
        return vec.shuffle[
            2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31
        ](other)

    @parameter
    fn _apply_permute_2(
        vec: SIMD[dtype, 16], other: SIMD[dtype, 16]
    ) -> SIMD[dtype, 16]:
        return vec.shuffle[
            0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29
        ](other)

    @parameter
    fn _apply_permute_3(
        vec: SIMD[dtype, 16], other: SIMD[dtype, 16]
    ) -> SIMD[dtype, 16]:
        return vec.shuffle[
            2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31
        ](other)

    @parameter
    fn _apply_permute_4(
        vec: SIMD[dtype, 16], other: SIMD[dtype, 16]
    ) -> SIMD[dtype, 16]:
        return vec.shuffle[
            0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27
        ](other)

    @parameter
    fn _apply_permute_5(
        vec: SIMD[dtype, 16], other: SIMD[dtype, 16]
    ) -> SIMD[dtype, 16]:
        return vec.shuffle[
            4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31
        ](other)

    @parameter
    fn _apply_permute_6(
        vec: SIMD[dtype, 16], other: SIMD[dtype, 16]
    ) -> SIMD[dtype, 16]:
        return vec.shuffle[
            0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27
        ](other)

    @parameter
    fn _apply_permute_7(
        vec: SIMD[dtype, 16], other: SIMD[dtype, 16]
    ) -> SIMD[dtype, 16]:
        return vec.shuffle[
            4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31
        ](other)

    var idx00 = buf.runtime_layout(
        RuntimeTuple[fill_like(buf.layout.shape, UNKNOWN_VALUE)](
            IndexList[2](0, 0)
        )
    )
    var row00 = buf.ptr.load[width=16](idx00)
    var idx01 = buf.runtime_layout(
        RuntimeTuple[fill_like(buf.layout.shape, UNKNOWN_VALUE)](
            IndexList[2](1, 0)
        )
    )
    var row01 = buf.ptr.load[width=16](idx01)
    var idx02 = buf.runtime_layout(
        RuntimeTuple[fill_like(buf.layout.shape, UNKNOWN_VALUE)](
            IndexList[2](2, 0)
        )
    )
    var row02 = buf.ptr.load[width=16](idx02)
    var idx03 = buf.runtime_layout(
        RuntimeTuple[fill_like(buf.layout.shape, UNKNOWN_VALUE)](
            IndexList[2](3, 0)
        )
    )
    var row03 = buf.ptr.load[width=16](idx03)
    var idx04 = buf.runtime_layout(
        RuntimeTuple[fill_like(buf.layout.shape, UNKNOWN_VALUE)](
            IndexList[2](4, 0)
        )
    )
    var row04 = buf.ptr.load[width=16](idx04)
    var idx05 = buf.runtime_layout(
        RuntimeTuple[fill_like(buf.layout.shape, UNKNOWN_VALUE)](
            IndexList[2](5, 0)
        )
    )
    var row05 = buf.ptr.load[width=16](idx05)
    var idx06 = buf.runtime_layout(
        RuntimeTuple[fill_like(buf.layout.shape, UNKNOWN_VALUE)](
            IndexList[2](6, 0)
        )
    )
    var row06 = buf.ptr.load[width=16](idx06)
    var idx07 = buf.runtime_layout(
        RuntimeTuple[fill_like(buf.layout.shape, UNKNOWN_VALUE)](
            IndexList[2](7, 0)
        )
    )
    var row07 = buf.ptr.load[width=16](idx07)
    var idx08 = buf.runtime_layout(
        RuntimeTuple[fill_like(buf.layout.shape, UNKNOWN_VALUE)](
            IndexList[2](8, 0)
        )
    )
    var row08 = buf.ptr.load[width=16](idx08)
    var idx09 = buf.runtime_layout(
        RuntimeTuple[fill_like(buf.layout.shape, UNKNOWN_VALUE)](
            IndexList[2](9, 0)
        )
    )
    var row09 = buf.ptr.load[width=16](idx09)
    var idx10 = buf.runtime_layout(
        RuntimeTuple[fill_like(buf.layout.shape, UNKNOWN_VALUE)](
            IndexList[2](10, 0)
        )
    )
    var row10 = buf.ptr.load[width=16](idx10)
    var idx11 = buf.runtime_layout(
        RuntimeTuple[fill_like(buf.layout.shape, UNKNOWN_VALUE)](
            IndexList[2](11, 0)
        )
    )
    var row11 = buf.ptr.load[width=16](idx11)
    var idx12 = buf.runtime_layout(
        RuntimeTuple[fill_like(buf.layout.shape, UNKNOWN_VALUE)](
            IndexList[2](12, 0)
        )
    )
    var row12 = buf.ptr.load[width=16](idx12)
    var idx13 = buf.runtime_layout(
        RuntimeTuple[fill_like(buf.layout.shape, UNKNOWN_VALUE)](
            IndexList[2](13, 0)
        )
    )
    var row13 = buf.ptr.load[width=16](idx13)
    var idx14 = buf.runtime_layout(
        RuntimeTuple[fill_like(buf.layout.shape, UNKNOWN_VALUE)](
            IndexList[2](14, 0)
        )
    )
    var row14 = buf.ptr.load[width=16](idx14)
    var idx15 = buf.runtime_layout(
        RuntimeTuple[fill_like(buf.layout.shape, UNKNOWN_VALUE)](
            IndexList[2](15, 0)
        )
    )
    var row15 = buf.ptr.load[width=16](idx15)

    var k00 = _apply_permute_0(row00, row01)
    var k01 = _apply_permute_1(row00, row01)
    var k02 = _apply_permute_0(row02, row03)
    var k03 = _apply_permute_1(row02, row03)
    var k04 = _apply_permute_0(row04, row05)
    var k05 = _apply_permute_1(row04, row05)
    var k06 = _apply_permute_0(row06, row07)
    var k07 = _apply_permute_1(row06, row07)
    var k08 = _apply_permute_0(row08, row09)
    var k09 = _apply_permute_1(row08, row09)
    var k10 = _apply_permute_0(row10, row11)
    var k11 = _apply_permute_1(row10, row11)
    var k12 = _apply_permute_0(row12, row13)
    var k13 = _apply_permute_1(row12, row13)
    var k14 = _apply_permute_0(row14, row15)
    var k15 = _apply_permute_1(row14, row15)

    var j00 = _apply_permute_2(k00, k02)
    var j01 = _apply_permute_3(k00, k02)
    var j02 = _apply_permute_2(k01, k03)
    var j03 = _apply_permute_3(k01, k03)
    var j04 = _apply_permute_2(k04, k06)
    var j05 = _apply_permute_3(k04, k06)
    var j06 = _apply_permute_2(k05, k07)
    var j07 = _apply_permute_3(k05, k07)
    var j08 = _apply_permute_2(k08, k10)
    var j09 = _apply_permute_3(k08, k10)
    var j10 = _apply_permute_2(k09, k11)
    var j11 = _apply_permute_3(k09, k11)
    var j12 = _apply_permute_2(k12, k14)
    var j13 = _apply_permute_3(k12, k14)
    var j14 = _apply_permute_2(k13, k15)
    var j15 = _apply_permute_3(k13, k15)

    var t00 = _apply_permute_4(j00, j04)
    var t01 = _apply_permute_4(j01, j05)
    var t02 = _apply_permute_4(j02, j06)
    var t03 = _apply_permute_4(j03, j07)
    var t04 = _apply_permute_5(j00, j04)
    var t05 = _apply_permute_5(j01, j05)
    var t06 = _apply_permute_5(j02, j06)
    var t07 = _apply_permute_5(j03, j07)
    var t08 = _apply_permute_4(j08, j12)
    var t09 = _apply_permute_4(j09, j13)
    var t10 = _apply_permute_4(j10, j14)
    var t11 = _apply_permute_4(j11, j15)
    var t12 = _apply_permute_5(j08, j12)
    var t13 = _apply_permute_5(j09, j13)
    var t14 = _apply_permute_5(j10, j14)
    var t15 = _apply_permute_5(j11, j15)

    var r00 = _apply_permute_6(t00, t08)
    var r01 = _apply_permute_6(t01, t09)
    var r02 = _apply_permute_6(t02, t10)
    var r03 = _apply_permute_6(t03, t11)
    var r04 = _apply_permute_6(t04, t12)
    var r05 = _apply_permute_6(t05, t13)
    var r06 = _apply_permute_6(t06, t14)
    var r07 = _apply_permute_6(t07, t15)
    var r08 = _apply_permute_7(t00, t08)
    var r09 = _apply_permute_7(t01, t09)
    var r10 = _apply_permute_7(t02, t10)
    var r11 = _apply_permute_7(t03, t11)
    var r12 = _apply_permute_7(t04, t12)
    var r13 = _apply_permute_7(t05, t13)
    var r14 = _apply_permute_7(t06, t14)
    var r15 = _apply_permute_7(t07, t15)

    buf.ptr.store[width=16](idx00, r00)
    buf.ptr.store[width=16](idx01, r01)
    buf.ptr.store[width=16](idx02, r02)
    buf.ptr.store[width=16](idx03, r03)
    buf.ptr.store[width=16](idx04, r04)
    buf.ptr.store[width=16](idx05, r05)
    buf.ptr.store[width=16](idx06, r06)
    buf.ptr.store[width=16](idx07, r07)
    buf.ptr.store[width=16](idx08, r08)
    buf.ptr.store[width=16](idx09, r09)
    buf.ptr.store[width=16](idx10, r10)
    buf.ptr.store[width=16](idx11, r11)
    buf.ptr.store[width=16](idx12, r12)
    buf.ptr.store[width=16](idx13, r13)
    buf.ptr.store[width=16](idx14, r14)
    buf.ptr.store[width=16](idx15, r15)


fn _transpose_inplace_naive[
    rows: Int,
    cols: Int,
    dtype: DType,
](buf: NDBuffer[mut=True, dtype, 2, _, DimList(rows, cols)]):
    for i in range(rows):
        for j in range(i + 1, cols):
            var tmp = buf[i, j]
            buf[IndexList[2](i, j)] = buf[j, i]
            buf[IndexList[2](j, i)] = tmp


fn _transpose_inplace_naive[
    dtype: DType,
](buf: LayoutTensor[mut=True, dtype, **_]):
    alias rows = Int(buf.layout.shape[0])
    alias cols = Int(buf.layout.shape[1])

    for i in range(rows):
        for j in range(i + 1, cols):
            var tmp = buf[i, j]
            buf[i, j] = buf[j, i]
            buf[j, i] = tmp


fn transpose_inplace[
    rows: Int,
    cols: Int,
    dtype: DType,
](buf: NDBuffer[mut=True, dtype, 2, _, DimList(rows, cols)]):
    # Reject sizes covered by specialized implementations
    constrained[rows == cols]()

    @parameter
    if rows == 4:
        _transpose_inplace_4x4[rows, cols, dtype](buf)
    elif rows == 8:
        _transpose_inplace_8x8[rows, cols, dtype](buf)
    elif rows == 16:
        _transpose_inplace_16x16[rows, cols, dtype](buf)
    else:
        _transpose_inplace_naive[rows, cols, dtype](buf)


fn transpose_inplace(buf: LayoutTensor[mut=True, **_]):
    # Reject sizes covered by specialized implementations
    constrained[buf.rank == 2]()
    alias rows = Int(buf.layout.shape[0])
    alias cols = Int(buf.layout.shape[1])
    constrained[rows == cols]()

    @parameter
    if rows == 4:
        _transpose_inplace_4x4(buf)
    elif rows == 8:
        _transpose_inplace_8x8(buf)
    elif rows == 16:
        _transpose_inplace_16x16(buf)
    else:
        _transpose_inplace_naive(buf)


fn _permute_data[
    size: Int,
    dtype: DType,
](
    input: UnsafePointer[Scalar[dtype]],
    output: UnsafePointer[Scalar[dtype]],
    perms: UnsafePointer[Scalar[DType.index]],
):
    """
    Ensures that output[i] = input[perms[i]] for i ∈ [0, size)
    """

    @parameter
    for idx in range(size):
        var perm_axis = perms.load(idx)[0]
        var perm_data = input.load(perm_axis)
        output[idx] = perm_data


fn _fill_strides[
    rank: Int,
    input_shape: DimList,
    dtype: DType,
](
    buf: NDBuffer[dtype, rank, _, input_shape],
    strides: UnsafePointer[Scalar[DType.index]],
):
    """
    Fill `strides`, which will be an array of strides indexed by axis, assuming
    `buf` contains contiguous buf.

    Note that `buf` is only used for querying its dimensions.
    """
    _fill_strides(buf, NDBuffer[DType.index, 1, _, rank](strides))


fn _fill_strides[
    rank: Int,
    input_shape: DimList,
    dtype: DType,
](
    buf: NDBuffer[dtype, rank, _, input_shape],
    strides: NDBuffer[mut=True, DType.index, 1, _, rank],
):
    """
    Fill `strides`, which will be an array of strides indexed by axis, assuming
    `buf` contains contiguous buf.

    Note that `buf` is only used for querying its dimensions.
    """
    constrained[rank > 0]()
    strides[rank - 1] = 1

    @parameter
    for idx in range(rank - 1):
        alias axis = rank - idx - 2
        var next_axis_stride = strides[axis + 1]
        var next_axis_dim = buf.dim[axis + 1]()
        var curr_axis_stride = next_axis_stride * next_axis_dim
        strides[axis] = curr_axis_stride


fn _fill_strides[
    input_layout: Layout,
    dtype: DType,
](
    buf: LayoutTensor[dtype, input_layout, **_],
    strides: LayoutTensor[
        mut=True, DType.index, Layout.row_major(buf.rank), **_
    ],
):
    """
    Fill `strides`, which will be an array of strides indexed by axis, assuming
    `buf` contains contiguous buf.

    Note that `buf` is only used for querying its dimensions.
    """
    constrained[buf.rank > 0]()
    strides[buf.rank - 1] = 1

    @parameter
    for idx in range(buf.rank - 1):
        alias axis = buf.rank - idx - 2
        var next_axis_stride = strides[axis + 1]
        var next_axis_dim = buf.dim[axis + 1]()
        var curr_axis_stride = next_axis_stride * next_axis_dim
        strides[axis] = curr_axis_stride


# ===------------------------------------------------------------------=== #
# Transpose Permutation simplification
# ===------------------------------------------------------------------=== #
@always_inline
fn _collapse_unpermuted_dims[
    rank: Int, tuple_size: Int
](
    mut simplified_shape: IndexList[tuple_size],
    mut simplified_perms: IndexList[tuple_size],
    dim: Int,
):
    var merged_dim = simplified_perms[dim]
    simplified_shape[merged_dim] = (
        simplified_shape[merged_dim] * simplified_shape[merged_dim + 1]
    )

    for j in range(merged_dim + 1, rank - 1):
        simplified_shape[j] = simplified_shape[j + 1]

    for i in range(rank):
        if simplified_perms[i] > merged_dim:
            simplified_perms[i] -= 1
    for k in range(dim + 1, rank - 1):
        simplified_perms[k] = simplified_perms[k + 1]
    simplified_shape[rank - 1] = 0
    simplified_perms[rank - 1] = 0


@always_inline
fn _devare_size_1_dim[
    rank: Int, tuple_size: Int
](
    mut simplified_shape: IndexList[tuple_size],
    mut simplified_perms: IndexList[tuple_size],
    dim: Int,
):
    for i in range(dim, rank - 1):
        simplified_shape[i] = simplified_shape[i + 1]

    var found_devared: Bool = False
    for i in range(rank - 1):
        if simplified_perms[i] == dim:
            found_devared = True
        if found_devared:
            simplified_perms[i] = simplified_perms[i + 1]
        if simplified_perms[i] > dim:
            simplified_perms[i] -= 1

    simplified_shape[rank - 1] = 0
    simplified_perms[rank - 1] = 0


@always_inline
fn _simplify_transpose_perms_impl[
    rank: Int, tuple_size: Int
](
    mut simplified_rank: Int,
    mut simplified_shape: IndexList[tuple_size],
    mut simplified_perms: IndexList[tuple_size],
):
    @parameter
    if rank < 2:
        return

    else:
        for i in range(rank - 1):
            if simplified_perms[i] + 1 == simplified_perms[i + 1]:
                _collapse_unpermuted_dims[rank](
                    simplified_shape, simplified_perms, i
                )
                simplified_rank -= 1
                _simplify_transpose_perms_impl[rank - 1, tuple_size](
                    simplified_rank, simplified_shape, simplified_perms
                )
                return
            if simplified_shape[i] == 1:
                _devare_size_1_dim[rank](simplified_shape, simplified_perms, i)
                simplified_rank -= 1
                _simplify_transpose_perms_impl[rank - 1, tuple_size](
                    simplified_rank, simplified_shape, simplified_perms
                )
                return


@always_inline
fn _simplify_transpose_perms[
    rank: Int
](
    mut simplified_rank: Int,
    mut simplified_shape: IndexList[rank],
    mut simplified_perms: IndexList[rank],
):
    """Simplify the given permutation pattern.

    In some cases a permutation can be modeled by another permutation of a smaller rank.
    For instance, if we have
        shape=[1,3,200,200], perm = [0, 2, 3, 1]
    Then it is equivalent to:
        shape=[1,3,40000], perm = [0, 2, 1]
    Which in its turn is equivalent to:
        shape=[3,40000], perm = [1, 0]

    This function takes the original shape, permutation, and rank by reference,
    and updates their values to simplified ones.
    """
    _simplify_transpose_perms_impl[rank, rank](
        simplified_rank, simplified_shape, simplified_perms
    )


@always_inline
fn _convert_transpose_perms_to_static_int_tuple[
    rank: Int
](perms: UnsafePointer[Scalar[DType.index]]) -> IndexList[rank]:
    var simplified_perms = IndexList[rank]()
    # TODO: unroll
    for j in range(rank):
        simplified_perms[j] = Int(perms.load(j)[0]._mlir_value)
    return simplified_perms


# ===------------------------------------------------------------------=== #
#  Transpose special cases
# ===------------------------------------------------------------------=== #
@always_inline
fn _process_tile[
    tile_size_m: Int, tile_size_n: Int, dtype: DType
](
    m: Int,
    n: Int,
    M: Int,
    N: Int,
    out_ptr: UnsafePointer[Scalar[dtype]],
    in_ptr: UnsafePointer[Scalar[dtype]],
):
    var input_tile_offset = M * n + m
    var output_tile_offset = N * m + n

    var input_vals = StaticTuple[SIMD[dtype, tile_size_m], tile_size_n]()
    var output_vals = StaticTuple[SIMD[dtype, tile_size_n], tile_size_m]()

    @parameter
    for i in range(tile_size_n):
        input_vals[i] = in_ptr.load[width=tile_size_m](
            input_tile_offset + M * i
        )

    @parameter
    for m in range(tile_size_m):

        @parameter
        for n in range(tile_size_n):
            output_vals[m][n] = input_vals[n][m]

    @parameter
    for i in range(tile_size_m):
        out_ptr.store(output_tile_offset + N * i, output_vals[i])


fn _transpose_2d_serial_tiled[
    rank: Int, dtype: DType, //
](
    output: NDBuffer[mut=True, dtype, rank, _, _],
    input: NDBuffer[dtype, rank, _, _],
    perms: UnsafePointer[Scalar[DType.index]],
    simplified_input_shape: IndexList[rank],
    simplified_rank: Int,
    offset: Int,
):
    alias simd_width = simd_width_of[dtype]()

    @parameter
    if rank < 2:
        return
    # The input tile is MxN, the output tile is NxM.
    # We want to do:
    #   output[m, n] = input[n, m]
    # This is equivalent to:
    #   output[n*M + m] = input[m*N + n]
    # And we also have a global offset which needs to be added to both output
    # and input pointers.
    var N = simplified_input_shape[simplified_rank - 2]
    var M = simplified_input_shape[simplified_rank - 1]

    @parameter
    @__copy_capture(N, M)
    @always_inline
    fn process_tile[tile_size_m: Int, tile_size_n: Int](m: Int, n: Int):
        _process_tile[tile_size_m, tile_size_n, dtype](
            m, n, M, N, output.data.offset(offset), input.data.offset(offset)
        )

    alias tile_size = simd_width if simd_width <= 16 else 1
    tile[
        process_tile,
        VariadicList[Int](tile_size, 1),
        VariadicList[Int](tile_size, 1),
    ](0, 0, M, N)


@always_inline
fn _should_run_parallel(
    M: Int, N: Int, simd_width: Int, min_work_per_task: Int
) -> Bool:
    if N == 1:
        return False

    # Check if we can tile the space evenly
    if (N % simd_width) != 0 or (M % simd_width) != 0:
        return False

    var work_per_row = M * simd_width
    if min_work_per_task > work_per_row:
        # We will have to process several rows in each thread
        if (min_work_per_task % work_per_row) != 0:
            return False
        var rows_per_worker = ceildiv(min_work_per_task, work_per_row)
        if N // rows_per_worker < 4:
            return False

    return True


fn _transpose_2d_parallel_tiled[
    rank: Int, dtype: DType, //
](
    output: NDBuffer[dtype, rank, _, _],
    input: NDBuffer[dtype, rank, _, _],
    perms: UnsafePointer[Scalar[DType.index]],
    simplified_input_shape: IndexList[rank],
    simplified_rank: Int,
    offset: Int,
):
    @parameter
    if rank < 2:
        return

    alias simd_width = simd_width_of[dtype]()
    var N = simplified_input_shape[simplified_rank - 2]
    var M = simplified_input_shape[simplified_rank - 1]
    alias min_work_per_task = 1024
    alias tile_size_m = simd_width if simd_width <= 16 else 1
    alias tile_size_n = simd_width if simd_width <= 16 else 1

    var n_unit_size = simd_width
    var m_unit_size = simd_width

    var n_tiles = N // n_unit_size
    var m_tiles = M // m_unit_size

    var rows_per_worker = (
        1  # Row in terms of tiles, i.e. we still take simd_width elements
    )
    if min_work_per_task > M * simd_width:
        rows_per_worker = min_work_per_task // (M * simd_width)

    var work = ceildiv(n_tiles, rows_per_worker)

    var num_threads = parallelism_level()

    var num_tasks = min(work, num_threads)

    var work_block_size = ceildiv(work, num_tasks)

    @parameter
    @__copy_capture(work_block_size, m_tiles, N, M)
    @always_inline
    fn _parallel_tile(thread_id: Int):
        var n_tile_begin = work_block_size * thread_id
        var n_tile_end = min(work_block_size * (thread_id + 1), work)

        for n_tile in range(n_tile_begin, n_tile_end):
            for m_tile in range(m_tiles):
                var m = tile_size_m * m_tile
                var n = tile_size_n * n_tile
                _process_tile[tile_size_m, tile_size_n, dtype](
                    m,
                    n,
                    M,
                    N,
                    output.data.offset(offset),
                    input.data.offset(offset),
                )

    sync_parallelize[_parallel_tile](num_tasks)


fn transpose_2d[
    rank: Int,
    output_shape: DimList,
    input_shape: DimList,
    dtype: DType,
](
    output: NDBuffer[mut=True, dtype, rank, _, output_shape],
    input: NDBuffer[dtype, rank, _, input_shape],
    perms: UnsafePointer[Scalar[DType.index]],
    simplified_input_shape: IndexList[rank],
    simplified_rank: Int,
    offset: Int,
):
    @parameter
    if rank < 2:
        return

    alias simd_width = simd_width_of[dtype]()
    var N = simplified_input_shape[simplified_rank - 2]
    var M = simplified_input_shape[simplified_rank - 1]
    alias min_work_per_task = 1024

    if _should_run_parallel(M, N, simd_width, min_work_per_task):
        _transpose_2d_parallel_tiled(
            output,
            input,
            perms,
            simplified_input_shape,
            simplified_rank,
            offset,
        )
    else:
        _transpose_2d_serial_tiled(
            output,
            input,
            perms,
            simplified_input_shape,
            simplified_rank,
            offset,
        )

        return


fn _transpose_4d_swap_middle_helper[
    dtype: DType, //
](
    dst_ptr: UnsafePointer[Scalar[dtype]],
    src_ptr: UnsafePointer[Scalar[dtype]],
    L: Int,
    M: Int,
    N: Int,
    K: Int,
):
    var work = L * M * N
    var total_size = L * M * N * K

    alias KB = 1024

    # TODO: These parameters might be tuned
    alias min_work_per_task = 1 * KB
    alias min_work_for_parallel = 4 * min_work_per_task

    # TODO: take into account dimension K for parallelization.
    #
    # E.g. if we're transposing 2x3x8192 -> 3x2x8192, then parallelizing just
    # on dimensions M and N is not enough.
    if total_size <= min_work_for_parallel:
        for l in range(L):
            for m in range(M):
                for n in range(N):
                    # We want to do:
                    #   output[l, n, m, k] = input[l, m, n, k]
                    var in_off = l * M * N * K + m * N * K + n * K
                    var out_off = l * M * N * K + n * M * K + m * K
                    memcpy(
                        dst_ptr.offset(out_off),
                        src_ptr.offset(in_off),
                        K,
                    )
        return
    else:
        var num_threads = parallelism_level()

        var num_tasks = min(work, num_threads)

        var work_block_size = ceildiv(work, num_tasks)

        @parameter
        @__copy_capture(work, work_block_size)
        @always_inline
        fn _parallel_copy(thread_id: Int):
            var begin = work_block_size * thread_id
            var end = min(work_block_size * (thread_id + 1), work)
            for block_idx in range(begin, end):
                var l = block_idx // (M * N)
                var block_idx_mn = block_idx % (M * N)
                var m = block_idx_mn // N
                var n = block_idx_mn % N

                var in_off = l * M * N * K + m * N * K + n * K
                var out_off = l * M * N * K + n * M * K + m * K
                memcpy(
                    dst_ptr.offset(out_off),
                    src_ptr.offset(in_off),
                    K,
                )

        sync_parallelize[_parallel_copy](num_tasks)


fn transpose_4d_swap_middle[
    rank: Int, dtype: DType, //
](
    output: NDBuffer[mut=True, dtype, rank, _, _],
    input: NDBuffer[dtype, rank, *_],
    perms: UnsafePointer[Scalar[DType.index]],
    simplified_input_shape: IndexList[rank],
    simplified_rank: Int,
):
    @parameter
    if rank < 4:
        return
    # The input tile is LxMxNxK, the output tile is LxNxMxK.
    # We want to do:
    #   output[l, n, m, k] = input[l, m, n, k]
    var L = simplified_input_shape[simplified_rank - 4]
    var M = simplified_input_shape[simplified_rank - 3]
    var N = simplified_input_shape[simplified_rank - 2]
    var K = simplified_input_shape[simplified_rank - 1]
    var src_ptr = input.data.offset(0)
    var dst_ptr = output.data.offset(0)
    _transpose_4d_swap_middle_helper(dst_ptr, src_ptr, L, M, N, K)


fn transpose_3d_swap_outer[
    rank: Int,
    output_shape: DimList,
    input_shape: DimList,
    dtype: DType,
](
    output: NDBuffer[mut=True, dtype, rank, _, output_shape],
    input: NDBuffer[dtype, rank, _, input_shape],
    perms: UnsafePointer[Scalar[DType.index]],
    simplified_input_shape: IndexList[rank],
    simplified_rank: Int,
):
    @parameter
    if rank < 3:
        return
    # The input tile is MxNxK, the output tile is NxMxK.
    # We want to do:
    #   output[n, m, k] = input[m, n, k]
    # We use a 4d helper function for this, pretending that we have an outer
    # dimensions L=1.
    var M = simplified_input_shape[simplified_rank - 3]
    var N = simplified_input_shape[simplified_rank - 2]
    var K = simplified_input_shape[simplified_rank - 1]
    var src_ptr = input.data.offset(0)
    var dst_ptr = output.data.offset(0)
    _transpose_4d_swap_middle_helper(dst_ptr, src_ptr, 1, M, N, K)


fn transpose_3d_swap_inner[
    rank: Int, dtype: DType, //
](
    output: NDBuffer[mut=True, dtype, rank, _, _],
    input: NDBuffer[dtype, rank, _, _],
    perms: UnsafePointer[Scalar[DType.index]],
    simplified_input_shape: IndexList[rank],
    simplified_rank: Int,
):
    @parameter
    if rank < 3:
        return
    # simplified perms must be 0, 2, 1
    var offset = 0
    var step = (
        simplified_input_shape[simplified_rank - 2]
        * simplified_input_shape[simplified_rank - 1]
    )
    # TODO: parallelize this loop
    for i in range(simplified_input_shape[0]):
        _transpose_2d_serial_tiled(
            output,
            input,
            perms,
            simplified_input_shape,
            simplified_rank,
            offset,
        )
        offset += step


fn transpose_trivial_memcpy[
    rank: Int,
    output_shape: DimList,
    input_shape: DimList,
    dtype: DType,
](
    output: NDBuffer[mut=True, dtype, rank, _, output_shape],
    input: NDBuffer[dtype, rank, _, input_shape],
):
    var src_ptr = input.data.offset(0)
    var dst_ptr = output.data.offset(0)

    alias KB = 1024
    alias min_work_per_task = 1 * KB
    alias min_work_for_parallel = 4 * min_work_per_task

    var total_size = output.size()

    if total_size <= min_work_for_parallel:
        memcpy(dst_ptr, src_ptr, total_size)

    else:
        var work_units = ceildiv(total_size, min_work_per_task)
        var num_tasks = min(work_units, parallelism_level())
        var work_block_size = ceildiv(work_units, num_tasks)

        parallel_memcpy(
            dst_ptr,
            src_ptr,
            total_size,
            work_block_size * min_work_per_task,
            num_tasks,
        )


# ===------------------------------------------------------------------=== #
#  Transpose generic strided implementation
# ===------------------------------------------------------------------=== #
fn _copy_with_strides[
    rank: Int, dtype: DType, //
](
    axis: Int,
    output: NDBuffer[mut=True, dtype, rank, _, _],
    input: UnsafePointer[Scalar[dtype]],
    input_strides: UnsafePointer[Scalar[DType.index]],
    output_strides: UnsafePointer[Scalar[DType.index]],
    input_offset: Int,
    output_offset: Int,
) raises:
    """
    Copy data from `input` to `output`, starting at corresponding offsets,
    based on given strides.

    Args:
        axis: The axis value.
        output: The output buffer.
        input: The input buffer.
        input_strides: The stride at each input axis.
        output_strides: The stride at each output axis.
        input_offset: The offset at which input data starts.
        output_offset: The offset at which output data starts.
    """
    if axis + 1 > rank:
        raise Error("out of range")

    var axis_dim = output.dim(axis)
    var input_axis_stride: Int = Int(input_strides.load(axis)[0]._mlir_value)
    var output_axis_stride: Int = Int(output_strides.load(axis)[0]._mlir_value)

    if axis + 1 == rank:
        var src_ptr = input.offset(input_offset)
        var dst_ptr = output.data.offset(output_offset)
        if input_axis_stride == 1 and output_axis_stride == 1:
            memcpy(dst_ptr, src_ptr, axis_dim)
        else:

            @always_inline
            @__copy_capture(input_axis_stride, output_axis_stride)
            @parameter
            fn _copy[simd_width: Int](offset: Int):
                strided_store(
                    strided_load[simd_width](src_ptr, input_axis_stride),
                    dst_ptr,
                    output_axis_stride,
                )
                src_ptr = src_ptr.offset(simd_width * input_axis_stride)
                dst_ptr = dst_ptr.offset(simd_width * output_axis_stride)

            vectorize[_copy, simd_width_of[dtype]()](axis_dim)

        return

    var next_axis = axis + 1

    alias KB = 1024

    # TODO: These parameters might be tuned
    alias min_work_per_task = 1 * KB
    alias min_work_for_parallel = 4 * min_work_per_task

    if output.bytecount() <= min_work_for_parallel or axis_dim == 1:
        var next_input_offset = input_offset
        var next_output_offset = output_offset
        for _ in range(axis_dim):
            _copy_with_strides(
                next_axis,
                output,
                input,
                input_strides,
                output_strides,
                next_input_offset,
                next_output_offset,
            )
            next_input_offset += input_axis_stride
            next_output_offset += output_axis_stride

    else:
        var num_threads = parallelism_level()
        var num_tasks = min(
            ceildiv(output.bytecount(), min_work_per_task), num_threads
        )

        var work = axis_dim
        var work_block_size = ceildiv(work, num_tasks)

        @always_inline
        @__copy_capture(
            work_block_size,
            work,
            next_axis,
            input_axis_stride,
            output_axis_stride,
        )
        @parameter
        fn _parallel_copy(thread_id: Int) raises:
            var next_input_offset = (
                thread_id * work_block_size * input_axis_stride + input_offset
            )
            var next_output_offset = (
                thread_id * work_block_size * output_axis_stride + output_offset
            )

            for _ in range(
                work_block_size * thread_id,
                min(work_block_size * (thread_id + 1), work),
            ):
                _copy_with_strides(
                    next_axis,
                    output,
                    input,
                    input_strides,
                    output_strides,
                    next_input_offset,
                    next_output_offset,
                )
                next_input_offset += input_axis_stride
                next_output_offset += output_axis_stride

        # TODO: transpose_strided is using stack allocated structueres and
        # so depends on us being synchronous. We need a better way to do this.
        sync_parallelize[_parallel_copy](num_tasks)


fn transpose_strided[
    rank: Int, dtype: DType, //
](
    output: NDBuffer[mut=True, dtype, rank, _, _],
    input: NDBuffer[dtype, rank, _, _],
    perms: UnsafePointer[Scalar[DType.index]],
) raises:
    # Compute `permuted_input_strides`
    var input_strides = UnsafePointer[Scalar[DType.index]].alloc(rank)
    var permuted_input_strides = UnsafePointer[Scalar[DType.index]].alloc(rank)
    _fill_strides(input, input_strides)
    _permute_data[rank, DType.index](
        input_strides, permuted_input_strides, perms
    )
    # Compute `output_strides`
    var output_strides = UnsafePointer[Scalar[DType.index]].alloc(rank)
    _fill_strides(output, output_strides)
    # Kickoff; for intuition on permuted input strides, note that
    #   transpose(output, input, [2, 0, 1])
    # guarantees
    #   (var isx denote input_stride_x, etc.)
    #   output[x, y, z] = input[z, x, y]
    # ~ output.at(offset(x*isx + y*isy + z*isz)) = input.at(offset(z*osx + x*osy + y*osz))
    # ~ output.at(offset(x*isx + y*isy + z*isz)) = input.at(offset(x*osy + y*osz + z*osx))
    # ~ output.at(offset([x, y, z], output_strides)) = input.at(offset([x, y, z], permuted_input_strides))
    # ~ output.at(offset(index, output_strides)) = input.at(offset(index, permuted_input_strides))
    alias init_axis = 0
    # NOTE: Synchronous, so the stack allocated input_strides, permuted_input_strings
    # and output_strides are safe to use.
    _copy_with_strides(
        init_axis,
        output,
        input.data,
        permuted_input_strides,
        output_strides,
        0,  # input_offset
        0,  # output_offset
    )
    input_strides.free()
    permuted_input_strides.free()
    output_strides.free()


# ===------------------------------------------------------------------=== #
#  Transpose entry points
# ===------------------------------------------------------------------=== #
fn transpose[
    rank: Int, dtype: DType, //
](
    output: NDBuffer[mut=True, dtype, rank, _, _],
    input: NDBuffer[dtype, rank, _, _],
    perms: UnsafePointer[Scalar[DType.index]],
) raises:
    """
    Permute the axis of `input` based on `perms`, and place the result in
    `output`.

    Example:
        ```mojo
        transpose(output, input, [2, 0, 1])
        # guarantees output[x, y, z] = input[z, x, y]
        ```

    Parameters:
        rank: The rank of input and output buffers.
        dtype: The dtype of buffer elements.

    Args:
        output: The output buffer.
        input: The input buffer.
        perms: Permutation of the input axes.
    """

    # If either input or output is not-contiguous, we need to use a general
    # strided implementation of transpose
    if not output.is_contiguous() or not input.is_contiguous():
        return transpose_strided(output, input, perms)

    # If they are contiguous, we can try to recognize common special cases in
    # the desired permutation.
    # E.g.
    #   shape=[1,3,200,200], perm = [0, 2, 3, 1]
    # is equivalent to
    #   shape=[1,3,40000], perm = [0, 2, 1]
    #
    # And that just swaps two inner dimensions.
    var simplified_perms = _convert_transpose_perms_to_static_int_tuple[rank](
        perms
    )
    var simplified_shape = input.get_shape()
    var simplified_rank = rank
    _simplify_transpose_perms[rank](
        simplified_rank, simplified_shape, simplified_perms
    )

    if simplified_rank == 1:
        # memcpy
        return transpose_trivial_memcpy(output, input)
    # TODO: Reenable once #15947 is fixed.
    # elif simplified_rank == 2:
    #     # tiled transpose
    #     return transpose_2d[rank, output_shape, input_shape, dtype](
    #         output,
    #         input,
    #         perms,
    #         simplified_shape,
    #         simplified_rank,
    #         0,
    #     )
    elif rank >= 3 and simplified_rank == 3:
        if (
            simplified_perms[0] == 0
            and simplified_perms[1] == 2
            and simplified_perms[2] == 1
        ):
            # batched tiled transpose
            return transpose_3d_swap_inner(
                output,
                input,
                perms,
                simplified_shape,
                simplified_rank,
            )
        elif (
            simplified_perms[0] == 1
            and simplified_perms[1] == 0
            and simplified_perms[2] == 2
        ):
            return transpose_3d_swap_outer(
                output,
                input,
                perms,
                simplified_shape,
                simplified_rank,
            )
    elif rank >= 4 and simplified_rank == 4:
        if (
            simplified_perms[0] == 0
            and simplified_perms[1] == 2
            and simplified_perms[2] == 1
            and simplified_perms[3] == 3
        ):
            return transpose_4d_swap_middle(
                output,
                input,
                perms,
                simplified_shape,
                simplified_rank,
            )
    transpose_strided(output, input, perms)
