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
"""GPU tests for scatter_nd_generator against host-computed references.

Covers row scatter, batched indices, sheet scatter (m=2, r_minus_m=1),
element scatter (r_minus_m=0), negative indices, the SKIP out-of-bounds
strategy, and the reduce path. The shape tests run under both OOB
strategies (with in-bounds indices they must agree), and dedicated tests
cover actual skipping for single- and multi-component indices.
"""

from std.gpu.host import DeviceContext
from std.testing import assert_equal
from layout import TileTensor, row_major
from nn.gather_scatter import ScatterOobIndexStrategy, scatter_nd_generator

comptime dtype = DType.float32
comptime itype = DType.int64


def test_row_scatter_2d[
    strategy: ScatterOobIndexStrategy
](ctx: DeviceContext) raises:
    """Data [64,16], indices [8,1] (with a negative index), updates [8,16]."""
    comptime rows = 64
    comptime cols = 16
    comptime n_idx = 8

    var data_dev = ctx.enqueue_create_buffer[dtype](rows * cols)
    var out_dev = ctx.enqueue_create_buffer[dtype](rows * cols)
    var upd_dev = ctx.enqueue_create_buffer[dtype](n_idx * cols)
    var idx_dev = ctx.enqueue_create_buffer[itype](n_idx)

    var data_host = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var upd_host = ctx.enqueue_create_host_buffer[dtype](n_idx * cols)
    var idx_host = ctx.enqueue_create_host_buffer[itype](n_idx)
    var out_host = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var expected = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    ctx.synchronize()

    for i in range(rows * cols):
        data_host[i] = Float32(i % 1000)
    for i in range(n_idx * cols):
        upd_host[i] = Float32(10000 + i)
    # Distinct rows, one negative (refers to rows - 2).
    for i in range(n_idx):
        idx_host[i] = Scalar[itype]((i * 7 + 3) % rows)
    idx_host[5] = Scalar[itype](-2)

    ctx.enqueue_copy(data_dev, data_host)
    ctx.enqueue_copy(upd_dev, upd_host)
    ctx.enqueue_copy(idx_dev, idx_host)
    ctx.synchronize()

    var data_tt = TileTensor(data_dev, row_major[rows, cols]())
    var out_tt = TileTensor(out_dev, row_major[rows, cols]())
    var upd_tt = TileTensor(upd_dev, row_major[n_idx, cols]())
    var idx_tt = TileTensor(idx_dev, row_major[n_idx, 1]())

    scatter_nd_generator[oob_index_strategy=strategy, target="gpu"](
        data_tt, idx_tt, upd_tt, out_tt, ctx
    )
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    for i in range(rows * cols):
        expected[i] = data_host[i]
    for i in range(n_idx):
        var r = Int(idx_host[i])
        if r < 0:
            r += rows
        for c in range(cols):
            expected[r * cols + c] = upd_host[i * cols + c]

    for i in range(rows * cols):
        assert_equal(out_host[i], expected[i])

    # Device buffers must outlive the async kernel and copies: TileTensor
    # holds only a raw pointer, so without these the buffers would be
    # destroyed at their last direct use, before the GPU work completes.
    _ = data_dev
    _ = out_dev
    _ = upd_dev
    _ = idx_dev


def test_batched_indices[
    strategy: ScatterOobIndexStrategy
](ctx: DeviceContext) raises:
    """Indices rank 3 [2,4,1], updates [2,4,16] — multi-dim row coords."""
    comptime rows = 64
    comptime cols = 16
    comptime a = 2
    comptime b = 4
    comptime n_idx = a * b

    var data_dev = ctx.enqueue_create_buffer[dtype](rows * cols)
    var out_dev = ctx.enqueue_create_buffer[dtype](rows * cols)
    var upd_dev = ctx.enqueue_create_buffer[dtype](n_idx * cols)
    var idx_dev = ctx.enqueue_create_buffer[itype](n_idx)

    var data_host = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var upd_host = ctx.enqueue_create_host_buffer[dtype](n_idx * cols)
    var idx_host = ctx.enqueue_create_host_buffer[itype](n_idx)
    var out_host = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var expected = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    ctx.synchronize()

    for i in range(rows * cols):
        data_host[i] = Float32(i % 1000)
    for i in range(n_idx * cols):
        upd_host[i] = Float32(20000 + i)
    for i in range(n_idx):
        idx_host[i] = Scalar[itype]((i * 11 + 1) % rows)

    ctx.enqueue_copy(data_dev, data_host)
    ctx.enqueue_copy(upd_dev, upd_host)
    ctx.enqueue_copy(idx_dev, idx_host)
    ctx.synchronize()

    var data_tt = TileTensor(data_dev, row_major[rows, cols]())
    var out_tt = TileTensor(out_dev, row_major[rows, cols]())
    var upd_tt = TileTensor(upd_dev, row_major[a, b, cols]())
    var idx_tt = TileTensor(idx_dev, row_major[a, b, 1]())

    scatter_nd_generator[oob_index_strategy=strategy, target="gpu"](
        data_tt, idx_tt, upd_tt, out_tt, ctx
    )
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    for i in range(rows * cols):
        expected[i] = data_host[i]
    for i in range(n_idx):
        var r = Int(idx_host[i])
        for c in range(cols):
            expected[r * cols + c] = upd_host[i * cols + c]

    for i in range(rows * cols):
        assert_equal(out_host[i], expected[i])

    _ = data_dev
    _ = out_dev
    _ = upd_dev
    _ = idx_dev


def test_sheet_scatter_3d[
    strategy: ScatterOobIndexStrategy
](ctx: DeviceContext) raises:
    """Data [16,8,12], indices [5,2], updates [5,12] — m=2, r_minus_m=1."""
    comptime d0 = 16
    comptime d1 = 8
    comptime d2 = 12
    comptime n_idx = 5
    comptime n_elems = d0 * d1 * d2

    var data_dev = ctx.enqueue_create_buffer[dtype](n_elems)
    var out_dev = ctx.enqueue_create_buffer[dtype](n_elems)
    var upd_dev = ctx.enqueue_create_buffer[dtype](n_idx * d2)
    var idx_dev = ctx.enqueue_create_buffer[itype](n_idx * 2)

    var data_host = ctx.enqueue_create_host_buffer[dtype](n_elems)
    var upd_host = ctx.enqueue_create_host_buffer[dtype](n_idx * d2)
    var idx_host = ctx.enqueue_create_host_buffer[itype](n_idx * 2)
    var out_host = ctx.enqueue_create_host_buffer[dtype](n_elems)
    var expected = ctx.enqueue_create_host_buffer[dtype](n_elems)
    ctx.synchronize()

    for i in range(n_elems):
        data_host[i] = Float32(i % 1000)
    for i in range(n_idx * d2):
        upd_host[i] = Float32(30000 + i)
    # Unique (i, j) pairs; the last one uses negative components.
    for i in range(n_idx):
        idx_host[2 * i] = Scalar[itype]((i * 3 + 1) % d0)
        idx_host[2 * i + 1] = Scalar[itype]((i * 5 + 2) % d1)
    idx_host[2 * 4] = Scalar[itype](-1)
    idx_host[2 * 4 + 1] = Scalar[itype](-8)

    ctx.enqueue_copy(data_dev, data_host)
    ctx.enqueue_copy(upd_dev, upd_host)
    ctx.enqueue_copy(idx_dev, idx_host)
    ctx.synchronize()

    var data_tt = TileTensor(data_dev, row_major[d0, d1, d2]())
    var out_tt = TileTensor(out_dev, row_major[d0, d1, d2]())
    var upd_tt = TileTensor(upd_dev, row_major[n_idx, d2]())
    var idx_tt = TileTensor(idx_dev, row_major[n_idx, 2]())

    scatter_nd_generator[oob_index_strategy=strategy, target="gpu"](
        data_tt, idx_tt, upd_tt, out_tt, ctx
    )
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    for i in range(n_elems):
        expected[i] = data_host[i]
    for i in range(n_idx):
        var i0 = Int(idx_host[2 * i])
        var i1 = Int(idx_host[2 * i + 1])
        if i0 < 0:
            i0 += d0
        if i1 < 0:
            i1 += d1
        for c in range(d2):
            expected[(i0 * d1 + i1) * d2 + c] = upd_host[i * d2 + c]

    for i in range(n_elems):
        assert_equal(out_host[i], expected[i])

    _ = data_dev
    _ = out_dev
    _ = upd_dev
    _ = idx_dev


def test_elem_scatter[
    strategy: ScatterOobIndexStrategy
](ctx: DeviceContext) raises:
    """Data [32,32], indices [32,2] unique pairs, updates [32] — r_minus_m=0."""
    comptime rows = 32
    comptime cols = 32
    comptime n_idx = 32
    comptime n_elems = rows * cols

    var data_dev = ctx.enqueue_create_buffer[dtype](n_elems)
    var out_dev = ctx.enqueue_create_buffer[dtype](n_elems)
    var upd_dev = ctx.enqueue_create_buffer[dtype](n_idx)
    var idx_dev = ctx.enqueue_create_buffer[itype](n_idx * 2)

    var data_host = ctx.enqueue_create_host_buffer[dtype](n_elems)
    var upd_host = ctx.enqueue_create_host_buffer[dtype](n_idx)
    var idx_host = ctx.enqueue_create_host_buffer[itype](n_idx * 2)
    var out_host = ctx.enqueue_create_host_buffer[dtype](n_elems)
    var expected = ctx.enqueue_create_host_buffer[dtype](n_elems)
    ctx.synchronize()

    for i in range(n_elems):
        data_host[i] = Float32(i % 1000)
    for i in range(n_idx):
        upd_host[i] = Float32(40000 + i)
        idx_host[2 * i] = Scalar[itype](i)
        idx_host[2 * i + 1] = Scalar[itype]((i * 5) % cols)

    ctx.enqueue_copy(data_dev, data_host)
    ctx.enqueue_copy(upd_dev, upd_host)
    ctx.enqueue_copy(idx_dev, idx_host)
    ctx.synchronize()

    var data_tt = TileTensor(data_dev, row_major[rows, cols]())
    var out_tt = TileTensor(out_dev, row_major[rows, cols]())
    var upd_tt = TileTensor(upd_dev, row_major[n_idx]())
    var idx_tt = TileTensor(idx_dev, row_major[n_idx, 2]())

    scatter_nd_generator[oob_index_strategy=strategy, target="gpu"](
        data_tt, idx_tt, upd_tt, out_tt, ctx
    )
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    for i in range(n_elems):
        expected[i] = data_host[i]
    for i in range(n_idx):
        var r = Int(idx_host[2 * i])
        var c = Int(idx_host[2 * i + 1])
        expected[r * cols + c] = upd_host[i]

    for i in range(n_elems):
        assert_equal(out_host[i], expected[i])

    _ = data_dev
    _ = out_dev
    _ = upd_dev
    _ = idx_dev


def test_skip_oob(ctx: DeviceContext) raises:
    """SKIP strategy: rows with indices outside [-rows, rows) are skipped."""
    comptime rows = 64
    comptime cols = 16
    comptime n_idx = 6

    var data_dev = ctx.enqueue_create_buffer[dtype](rows * cols)
    var out_dev = ctx.enqueue_create_buffer[dtype](rows * cols)
    var upd_dev = ctx.enqueue_create_buffer[dtype](n_idx * cols)
    var idx_dev = ctx.enqueue_create_buffer[itype](n_idx)

    var data_host = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var upd_host = ctx.enqueue_create_host_buffer[dtype](n_idx * cols)
    var idx_host = ctx.enqueue_create_host_buffer[itype](n_idx)
    var out_host = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var expected = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    ctx.synchronize()

    for i in range(rows * cols):
        data_host[i] = Float32(i % 1000)
    for i in range(n_idx * cols):
        upd_host[i] = Float32(50000 + i)
    idx_host[0] = Scalar[itype](2)
    idx_host[1] = Scalar[itype](rows + 6)  # OOB, skipped
    idx_host[2] = Scalar[itype](-3)
    idx_host[3] = Scalar[itype](-rows - 1)  # OOB, skipped
    idx_host[4] = Scalar[itype](5)
    idx_host[5] = Scalar[itype](0)

    ctx.enqueue_copy(data_dev, data_host)
    ctx.enqueue_copy(upd_dev, upd_host)
    ctx.enqueue_copy(idx_dev, idx_host)
    ctx.synchronize()

    var data_tt = TileTensor(data_dev, row_major[rows, cols]())
    var out_tt = TileTensor(out_dev, row_major[rows, cols]())
    var upd_tt = TileTensor(upd_dev, row_major[n_idx, cols]())
    var idx_tt = TileTensor(idx_dev, row_major[n_idx, 1]())

    scatter_nd_generator[
        oob_index_strategy=ScatterOobIndexStrategy.SKIP, target="gpu"
    ](data_tt, idx_tt, upd_tt, out_tt, ctx)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    for i in range(rows * cols):
        expected[i] = data_host[i]
    for i in range(n_idx):
        var r = Int(idx_host[i])
        if r < -rows or r >= rows:
            continue
        if r < 0:
            r += rows
        for c in range(cols):
            expected[r * cols + c] = upd_host[i * cols + c]

    for i in range(rows * cols):
        assert_equal(out_host[i], expected[i])

    _ = data_dev
    _ = out_dev
    _ = upd_dev
    _ = idx_dev


def test_skip_oob_multi_index(ctx: DeviceContext) raises:
    """SKIP strategy with m=2: a row is skipped if any index component is OOB.
    """
    comptime d0 = 16
    comptime d1 = 8
    comptime d2 = 12
    comptime n_idx = 4
    comptime n_elems = d0 * d1 * d2

    var data_dev = ctx.enqueue_create_buffer[dtype](n_elems)
    var out_dev = ctx.enqueue_create_buffer[dtype](n_elems)
    var upd_dev = ctx.enqueue_create_buffer[dtype](n_idx * d2)
    var idx_dev = ctx.enqueue_create_buffer[itype](n_idx * 2)

    var data_host = ctx.enqueue_create_host_buffer[dtype](n_elems)
    var upd_host = ctx.enqueue_create_host_buffer[dtype](n_idx * d2)
    var idx_host = ctx.enqueue_create_host_buffer[itype](n_idx * 2)
    var out_host = ctx.enqueue_create_host_buffer[dtype](n_elems)
    var expected = ctx.enqueue_create_host_buffer[dtype](n_elems)
    ctx.synchronize()

    for i in range(n_elems):
        data_host[i] = Float32(i % 1000)
    for i in range(n_idx * d2):
        upd_host[i] = Float32(60000 + i)
    # row 0: valid; row 1: first component OOB; row 2: second component OOB;
    # row 3: valid negative components.
    idx_host[0] = Scalar[itype](1)
    idx_host[1] = Scalar[itype](2)
    idx_host[2] = Scalar[itype](d0 + 4)  # OOB, skipped
    idx_host[3] = Scalar[itype](3)
    idx_host[4] = Scalar[itype](3)
    idx_host[5] = Scalar[itype](-d1 - 1)  # OOB, skipped
    idx_host[6] = Scalar[itype](-1)
    idx_host[7] = Scalar[itype](-d1)

    ctx.enqueue_copy(data_dev, data_host)
    ctx.enqueue_copy(upd_dev, upd_host)
    ctx.enqueue_copy(idx_dev, idx_host)
    ctx.synchronize()

    var data_tt = TileTensor(data_dev, row_major[d0, d1, d2]())
    var out_tt = TileTensor(out_dev, row_major[d0, d1, d2]())
    var upd_tt = TileTensor(upd_dev, row_major[n_idx, d2]())
    var idx_tt = TileTensor(idx_dev, row_major[n_idx, 2]())

    scatter_nd_generator[
        oob_index_strategy=ScatterOobIndexStrategy.SKIP, target="gpu"
    ](data_tt, idx_tt, upd_tt, out_tt, ctx)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    for i in range(n_elems):
        expected[i] = data_host[i]
    for i in range(n_idx):
        var i0 = Int(idx_host[2 * i])
        var i1 = Int(idx_host[2 * i + 1])
        if i0 < -d0 or i0 >= d0 or i1 < -d1 or i1 >= d1:
            continue
        if i0 < 0:
            i0 += d0
        if i1 < 0:
            i1 += d1
        for c in range(d2):
            expected[(i0 * d1 + i1) * d2 + c] = upd_host[i * d2 + c]

    for i in range(n_elems):
        assert_equal(out_host[i], expected[i])

    _ = data_dev
    _ = out_dev
    _ = upd_dev
    _ = idx_dev


def test_reduce_add[
    strategy: ScatterOobIndexStrategy
](ctx: DeviceContext) raises:
    """Reduce add with unique indices: out[idx] = data[idx] + updates."""
    comptime rows = 64
    comptime cols = 16
    comptime n_idx = 8

    var data_dev = ctx.enqueue_create_buffer[dtype](rows * cols)
    var out_dev = ctx.enqueue_create_buffer[dtype](rows * cols)
    var upd_dev = ctx.enqueue_create_buffer[dtype](n_idx * cols)
    var idx_dev = ctx.enqueue_create_buffer[itype](n_idx)

    var data_host = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var upd_host = ctx.enqueue_create_host_buffer[dtype](n_idx * cols)
    var idx_host = ctx.enqueue_create_host_buffer[itype](n_idx)
    var out_host = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var expected = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    ctx.synchronize()

    for i in range(rows * cols):
        data_host[i] = Float32(i % 1000)
    for i in range(n_idx * cols):
        upd_host[i] = Float32(i + 1)
    # Indices are deliberately unique. The reduce path applies reduce_fn as a
    # non-atomic read-modify-write, so duplicate indices targeting the same
    # output element race; that is a known limitation tracked separately, not
    # something this test should exercise (a duplicate-index case would be
    # nondeterministic, not a stable assertion).
    for i in range(n_idx):
        idx_host[i] = Scalar[itype]((i * 9 + 4) % rows)

    ctx.enqueue_copy(data_dev, data_host)
    ctx.enqueue_copy(upd_dev, upd_host)
    ctx.enqueue_copy(idx_dev, idx_host)
    ctx.synchronize()

    var data_tt = TileTensor(data_dev, row_major[rows, cols]())
    var out_tt = TileTensor(out_dev, row_major[rows, cols]())
    var upd_tt = TileTensor(upd_dev, row_major[n_idx, cols]())
    var idx_tt = TileTensor(idx_dev, row_major[n_idx, 1]())

    @always_inline
    def reduce_fn[
        dtype: DType, width: SIMDSize
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return lhs + rhs

    scatter_nd_generator[
        oob_index_strategy=strategy, target="gpu", reduce_fn=reduce_fn
    ](data_tt, idx_tt, upd_tt, out_tt, ctx)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    for i in range(rows * cols):
        expected[i] = data_host[i]
    for i in range(n_idx):
        var r = Int(idx_host[i])
        for c in range(cols):
            expected[r * cols + c] = (
                expected[r * cols + c] + upd_host[i * cols + c]
            )

    for i in range(rows * cols):
        assert_equal(out_host[i], expected[i])

    _ = data_dev
    _ = out_dev
    _ = upd_dev
    _ = idx_dev


def test_unaligned_slice(ctx: DeviceContext) raises:
    """Slice width not divisible by the vector pack (15 fp32 cols), so the
    launch must fall back to the scalar (simd_width=1) path. Verifies the
    fallback produces the same result as the vectorized path covered by the
    other tests."""
    comptime rows = 64
    comptime cols = 15
    comptime n_idx = 8

    var data_dev = ctx.enqueue_create_buffer[dtype](rows * cols)
    var out_dev = ctx.enqueue_create_buffer[dtype](rows * cols)
    var upd_dev = ctx.enqueue_create_buffer[dtype](n_idx * cols)
    var idx_dev = ctx.enqueue_create_buffer[itype](n_idx)

    var data_host = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var upd_host = ctx.enqueue_create_host_buffer[dtype](n_idx * cols)
    var idx_host = ctx.enqueue_create_host_buffer[itype](n_idx)
    var out_host = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var expected = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    ctx.synchronize()

    for i in range(rows * cols):
        data_host[i] = Float32(i % 1000)
    for i in range(n_idx * cols):
        upd_host[i] = Float32(10000 + i)
    for i in range(n_idx):
        idx_host[i] = Scalar[itype]((i * 7 + 3) % rows)

    ctx.enqueue_copy(data_dev, data_host)
    ctx.enqueue_copy(upd_dev, upd_host)
    ctx.enqueue_copy(idx_dev, idx_host)
    ctx.synchronize()

    var data_tt = TileTensor(data_dev, row_major[rows, cols]())
    var out_tt = TileTensor(out_dev, row_major[rows, cols]())
    var upd_tt = TileTensor(upd_dev, row_major[n_idx, cols]())
    var idx_tt = TileTensor(idx_dev, row_major[n_idx, 1]())

    scatter_nd_generator[target="gpu"](data_tt, idx_tt, upd_tt, out_tt, ctx)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    for i in range(rows * cols):
        expected[i] = data_host[i]
    for i in range(n_idx):
        var r = Int(idx_host[i])
        for c in range(cols):
            expected[r * cols + c] = upd_host[i * cols + c]

    for i in range(rows * cols):
        assert_equal(out_host[i], expected[i])

    _ = data_dev
    _ = out_dev
    _ = upd_dev
    _ = idx_dev


@always_inline
def _add[
    ty: DType, width: SIMDSize
](lhs: SIMD[ty, width], rhs: SIMD[ty, width]) -> SIMD[ty, width]:
    return lhs + rhs


@always_inline
def _max[
    ty: DType, width: SIMDSize
](lhs: SIMD[ty, width], rhs: SIMD[ty, width]) -> SIMD[ty, width]:
    return max(lhs, rhs)


@always_inline
def _min[
    ty: DType, width: SIMDSize
](lhs: SIMD[ty, width], rhs: SIMD[ty, width]) -> SIMD[ty, width]:
    return min(lhs, rhs)


def test_reduce_duplicate_indices[
    reduce_op: def[dtype: DType, width: SIMDSize](
        SIMD[dtype, width], SIMD[dtype, width]
    ) thin -> SIMD[dtype, width],
](ctx: DeviceContext) raises:
    """All index rows collide on one output row; the atomic path must apply
    every duplicate update. Integer-valued floats keep the reduction exact
    regardless of the (nondeterministic) atomic application order."""
    comptime rows = 64
    comptime cols = 16
    comptime n_idx = 8
    comptime target_row = 5

    var data_dev = ctx.enqueue_create_buffer[dtype](rows * cols)
    var out_dev = ctx.enqueue_create_buffer[dtype](rows * cols)
    var upd_dev = ctx.enqueue_create_buffer[dtype](n_idx * cols)
    var idx_dev = ctx.enqueue_create_buffer[itype](n_idx)

    var data_host = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var upd_host = ctx.enqueue_create_host_buffer[dtype](n_idx * cols)
    var idx_host = ctx.enqueue_create_host_buffer[itype](n_idx)
    var out_host = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var expected = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    ctx.synchronize()

    for i in range(rows * cols):
        data_host[i] = Float32(i % 7)
    for k in range(n_idx):
        for c in range(cols):
            upd_host[k * cols + c] = Float32(k + 1)
    for i in range(n_idx):
        idx_host[i] = Scalar[itype](target_row)

    ctx.enqueue_copy(data_dev, data_host)
    ctx.enqueue_copy(upd_dev, upd_host)
    ctx.enqueue_copy(idx_dev, idx_host)
    ctx.synchronize()

    var data_tt = TileTensor(data_dev, row_major[rows, cols]())
    var out_tt = TileTensor(out_dev, row_major[rows, cols]())
    var upd_tt = TileTensor(upd_dev, row_major[n_idx, cols]())
    var idx_tt = TileTensor(idx_dev, row_major[n_idx, 1]())

    scatter_nd_generator[target="gpu", reduce_fn=reduce_op](
        data_tt, idx_tt, upd_tt, out_tt, ctx
    )
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    for i in range(rows * cols):
        expected[i] = data_host[i]
    for c in range(cols):
        var acc = data_host[target_row * cols + c]
        for k in range(n_idx):
            acc = reduce_op[dtype, 1](acc, upd_host[k * cols + c])
        expected[target_row * cols + c] = acc

    for i in range(rows * cols):
        assert_equal(out_host[i], expected[i])

    _ = data_dev
    _ = out_dev
    _ = upd_dev
    _ = idx_dev


def main() raises:
    comptime UNDEFINED = ScatterOobIndexStrategy.UNDEFINED
    comptime SKIP = ScatterOobIndexStrategy.SKIP
    with DeviceContext() as ctx:
        test_row_scatter_2d[UNDEFINED](ctx)
        test_row_scatter_2d[SKIP](ctx)
        test_batched_indices[UNDEFINED](ctx)
        test_batched_indices[SKIP](ctx)
        test_sheet_scatter_3d[UNDEFINED](ctx)
        test_sheet_scatter_3d[SKIP](ctx)
        test_elem_scatter[UNDEFINED](ctx)
        test_elem_scatter[SKIP](ctx)
        test_skip_oob(ctx)
        test_skip_oob_multi_index(ctx)
        test_reduce_add[UNDEFINED](ctx)
        test_reduce_add[SKIP](ctx)
        test_unaligned_slice(ctx)
