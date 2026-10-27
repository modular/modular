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
"""Device generators match their CPU references bit for bit; driver logic."""

from std.sys import size_of
from std.testing import assert_equal, assert_raises, assert_true

from bench_sweep import (
    Config,
    Footprint,
    PageLayout,
    PagedBlocks,
    PageTable,
    Rotation,
    Sweep,
    affine_host,
    distinct_stride,
    dtype_arm_enabled,
    fill_affine,
    fill_linear_grade,
    fill_lut_view,
    fill_page_table,
    fill_sparse_indices,
    fill_uniform,
    linear_grade_host,
    lut_view_host,
    page_table_host,
    plan_groups,
    sparse_indices_host,
    uniform_host,
)
from max.gpu.host import DeviceBuffer, DeviceContext

# Not a multiple of any block size, and larger than one pass of the grid, so
# both the tail and the grid-stride loop are exercised.
comptime N = (1 << 21) + 7


def assert_bits_equal[
    dtype: DType
](
    ctx: DeviceContext,
    buf: DeviceBuffer[dtype, ...],
    expected: List[Scalar[dtype]],
    what: String,
) raises:
    ctx.synchronize()
    assert_true(len(buf) >= len(expected), what)
    var nbytes = len(expected) * size_of[dtype]()
    var want = expected.unsafe_ptr().unsafe_bitcast[UInt8]()
    with buf.map_to_host() as host:
        var got = host.unsafe_ptr().unsafe_bitcast[UInt8]()
        for b in range(nbytes):
            if got[unsafe_offset=b] != want[unsafe_offset=b]:
                raise Error(
                    what,
                    ": element ",
                    b // size_of[dtype](),
                    " differs from the host reference",
                )


def check_uniform[dtype: DType](ctx: DeviceContext) raises:
    var buf = ctx.enqueue_create_buffer[dtype](N)
    fill_uniform(ctx, buf, 7, -2.0, 3.0)
    assert_bits_equal(
        ctx,
        buf,
        uniform_host[dtype](N, 7, -2.0, 3.0),
        String("uniform ", dtype),
    )
    # A smaller fill is a prefix of the larger one.
    var small = ctx.enqueue_create_buffer[dtype](1000)
    fill_uniform(ctx, small, 7, -2.0, 3.0)
    assert_bits_equal(
        ctx,
        small,
        uniform_host[dtype](1000, 7, -2.0, 3.0),
        String("uniform prefix ", dtype),
    )


def check_affine[dtype: DType](ctx: DeviceContext) raises:
    var buf = ctx.enqueue_create_buffer[dtype](N)
    fill_affine(ctx, buf, 3, -5)
    assert_bits_equal(
        ctx, buf, affine_host[dtype](N, 3, -5), String("affine ", dtype)
    )


def check_page_table(ctx: DeviceContext, layout: PageLayout) raises:
    comptime n = 12345
    var lut = ctx.enqueue_create_buffer[DType.uint32](n)
    fill_page_table(ctx, lut, layout, seed=3)
    var expected = page_table_host(n, layout, seed=3)
    assert_bits_equal(ctx, lut, expected, String("page table ", layout))
    # Every block is used once, and block 0 never.
    var seen = List[Bool](length=n + 1, fill=False)
    for v in expected:
        var block = Int(v)
        assert_true(block >= 1 and block <= n, "block out of range")
        assert_true(not seen[block], "block used twice")
        seen[block] = True


def check_sparse_indices(
    ctx: DeviceContext, pad_invalid: Bool, index_stride: Int = 0
) raises:
    comptime page_size = 64
    comptime batch = 3
    comptime top_k = 256
    var seq_len = 5 if not pad_invalid else 300
    var cache_len = 1000
    var table = PageTable[page_size](
        ctx, batch, cache_len + seq_len, PageLayout.SHUFFLED, seed=9
    )
    var dst = ctx.enqueue_create_buffer[DType.int32](batch * seq_len * top_k)
    fill_sparse_indices(
        ctx,
        dst,
        table.lut,
        table.pages_per_seq,
        page_size,
        pad_invalid,
        batch,
        seq_len,
        cache_len,
        top_k,
        index_stride=index_stride,
    )
    var expected = sparse_indices_host(
        table.host_lut(),
        table.pages_per_seq,
        page_size,
        pad_invalid,
        batch,
        seq_len,
        cache_len,
        top_k,
        index_stride,
    )
    assert_bits_equal(
        ctx,
        dst,
        expected,
        String("sparse indices pad=", pad_invalid, " stride=", index_stride),
    )


def check_distinct_stride() raises:
    # Kept where `count` picks cannot wrap onto each other.
    assert_equal(distinct_stride(3, 10000, 2048), 3)
    # Kept when coprime, even though the picks wrap.
    assert_equal(distinct_stride(3, 2054, 2048), 3)
    # Bumped to the next stride coprime with the window.
    assert_equal(distinct_stride(4, 2054, 2048), 5)
    assert_equal(distinct_stride(3, 2049, 2048), 4)


def check_linear_grade(ctx: DeviceContext) raises:
    var buf = ctx.enqueue_create_buffer[DType.uint32](1000)
    fill_linear_grade(ctx, buf, 700, 1300)
    var expected = linear_grade_host(1000, 700, 1300)
    assert_equal(expected[0], 700)
    assert_equal(expected[999], 1300)
    assert_bits_equal(ctx, buf, expected, "linear grade")
    assert_equal(linear_grade_host(1, 700, 1300)[0], 1000)


def check_lut_view(ctx: DeviceContext) raises:
    comptime page_size = 64
    comptime batch = 5
    var table = PageTable[page_size](
        ctx, batch, 40 * page_size, PageLayout.FRAGMENTED, seed=4
    )
    comptime cols = 100
    comptime used = 30
    var view = ctx.enqueue_create_buffer[DType.uint32](batch * cols)
    table.write_view(ctx, view, batch, cols, used)
    var expected = lut_view_host(
        table.host_lut(), table.pages_per_seq, batch, cols, used
    )
    assert_equal(expected[cols + used - 1], table.host_lut()[40 + used - 1])
    assert_equal(expected[cols + used], 0)
    assert_bits_equal(ctx, view, expected, "lut view")
    with assert_raises():
        table.write_view(ctx, view, batch, cols, 41)


def check_sweep() raises:
    var sweep = Sweep()
    sweep.axis("seq_len", "6,8192")
    sweep.axis("cache_len", "32768, 0,1")
    assert_equal(len(sweep), 6)
    var configs = sweep.configs()
    assert_equal(len(configs), 6)
    assert_equal(configs[0]["seq_len"], 6)
    assert_equal(configs[0]["cache_len"], 32768)
    assert_equal(configs[1]["cache_len"], 0)
    assert_equal(configs[3]["seq_len"], 8192)
    assert_equal(String(configs[5]), "seq_len=8192, cache_len=1")
    with assert_raises():
        _ = configs[0]["heads"]

    def no_empty_cache(cfg: Config) raises -> Bool:
        return cfg["cache_len"] != 0

    assert_equal(len(sweep.configs(no_empty_cache)), 4)


def check_zipped_sweep() raises:
    var sweep = Sweep()
    sweep.zipped(["batch_size", "seq_len"], ["1,8,64", "8192"])
    sweep.axis("top_k", "16,32")
    assert_equal(len(sweep), 6)
    var configs = sweep.configs()
    assert_equal(String(configs[0]), "batch_size=1, seq_len=8192, top_k=16")
    assert_equal(String(configs[5]), "batch_size=64, seq_len=8192, top_k=32")
    var mismatched = Sweep()
    with assert_raises():
        mismatched.zipped(["batch_size", "seq_len"], ["1,8", "1,2,3"])


def check_plan_groups() raises:
    var sweep = Sweep()
    sweep.axis("batch", [8, 1])
    sweep.axis("tokens", [6, 8192])

    def fits_alone(cfg: Config) raises -> Bool:
        return cfg["batch"] * cfg["tokens"] < 10000

    var configs = sweep.configs(fits_alone)
    assert_equal(len(configs), 3)

    def require(cfg: Config, mut fp: Footprint) raises:
        fp.need("batch", cfg["batch"])
        fp.need("tokens", cfg["tokens"])

    def bytes(fp: Footprint) raises -> Int:
        return fp["batch"] * fp["tokens"]

    var one = plan_groups(configs, require, bytes)
    assert_equal(len(one), 1)
    assert_equal(one[0].bytes, 8 * 8192)

    # Decode and prefill fit apart but not at their joint maximum.
    var split = plan_groups(configs, require, bytes, budget=10000)
    assert_equal(len(split), 2)
    assert_equal(len(split[0].configs), 2)
    assert_equal(split[0].bytes, 8 * 6)
    assert_equal(split[1].bytes, 8192)

    with assert_raises():
        _ = plan_groups(configs, require, bytes, budget=40)


def check_fixture_bytes(ctx: DeviceContext) raises:
    var fp = Footprint()
    PageTable[128].require(fp, 2, 1000)
    PageTable[128].require(fp, 1, 5000)
    var table = PageTable[128].from_footprint(ctx, fp)
    assert_equal(table.max_batch, 2)
    assert_equal(table.pages_per_seq, 40)
    assert_equal(table.rows(), PageTable[128].rows_for(fp))
    var blocks = PagedBlocks[DType.bfloat16](ctx, table, 576, seed=1)
    assert_equal(
        blocks.bytes(), PagedBlocks[DType.bfloat16].bytes_for[128](fp, 576)
    )


def check_copy_luts(ctx: DeviceContext) raises:
    var fp = Footprint()
    # Three dense copies of a 2 x 3-page config, and four copies of a
    # 1 x 10-page config that touches 2 of its pages.
    PageTable[128].require(fp, 2, 300, copies=3)
    PageTable[128].require(fp, 1, 1200, copies=4, touched_pages=2)
    assert_equal(fp["pool_pages"], 18)
    assert_equal(fp["copy_lut_entries"], 40)
    var table = PageTable[128].from_footprint(
        ctx, fp, PageLayout.SHUFFLED, seed=5
    )
    assert_equal(table.rows(), PageTable[128].rows_for(fp))

    table.write_copies(ctx, 3, 2, 3)
    var seen = List[Bool](length=table.num_blocks, fill=False)
    for copy in range(3):
        with table.copy_lut(ctx, copy, 2, 3).map_to_host() as host:
            for i in range(6):
                var block = Int(host[i])
                assert_true(block >= 1 and block < table.num_blocks)
                assert_true(not seen[block], "dense copies share a block")
                seen[block] = True
    with assert_raises():
        _ = table.copy_lut(ctx, 3, 2, 3)

    var touched = List[Bool](length=10, fill=False)
    touched[2] = True
    touched[7] = True
    table.write_copies(ctx, 4, 1, 10, touched)
    var first = List[UInt32]()
    with table.copy_lut(ctx, 0, 1, 10).map_to_host() as host:
        for i in range(10):
            first.append(host[i])
    var fresh = List[Bool](length=table.num_blocks, fill=False)
    for copy in range(1, 4):
        with table.copy_lut(ctx, copy, 1, 10).map_to_host() as host:
            for i in range(10):
                if touched[i]:
                    var block = Int(host[i])
                    assert_true(block != Int(first[i]))
                    assert_true(not fresh[block], "touched pages repeat")
                    fresh[block] = True
                else:
                    assert_equal(host[i], first[i])
    with assert_raises():
        table.write_copies(ctx, 5, 1, 10, touched)


def check_host_fill(ctx: DeviceContext) raises:
    """The host fill and copy path: what a target without fill kernels gets,
    forced through `on_host`. Same bits as the device path, since both call
    the same formulas; the partial fills cover the shorter-copy view."""
    var buf = ctx.enqueue_create_buffer[DType.float32](1000)
    fill_uniform(ctx, buf, 11, -1.0, 5.0, on_host=True)
    assert_bits_equal(
        ctx,
        buf,
        uniform_host[DType.float32](1000, 11, -1.0, 5.0),
        "host uniform",
    )
    var small = ctx.enqueue_create_buffer[DType.float32](100)
    fill_uniform(ctx, small, 11, -1.0, 5.0, on_host=True)
    assert_bits_equal(
        ctx,
        small,
        uniform_host[DType.float32](100, 11, -1.0, 5.0),
        "host uniform prefix",
    )

    var affine = ctx.enqueue_create_buffer[DType.uint32](1000)
    fill_affine(ctx, affine, 3, -5, 40, on_host=True)
    assert_bits_equal(
        ctx, affine, affine_host[DType.uint32](40, 3, -5), "host affine"
    )

    var lut = ctx.enqueue_create_buffer[DType.uint32](777)
    fill_page_table(ctx, lut, PageLayout.SHUFFLED, seed=13, on_host=True)
    assert_bits_equal(
        ctx,
        lut,
        page_table_host(777, PageLayout.SHUFFLED, seed=13),
        "host page table",
    )

    var grade = ctx.enqueue_create_buffer[DType.uint32](100)
    fill_linear_grade(ctx, grade, 700, 1300, 50, on_host=True)
    assert_bits_equal(
        ctx, grade, linear_grade_host(50, 700, 1300), "host linear grade"
    )

    comptime page_size = 64
    comptime batch = 4
    var table = PageTable[page_size](
        ctx, batch, 40 * page_size, PageLayout.FRAGMENTED, seed=4
    )
    comptime cols = 50
    comptime used = 20
    var view = ctx.enqueue_create_buffer[DType.uint32](batch * cols)
    fill_lut_view(
        ctx,
        view,
        table.lut,
        table.pages_per_seq,
        batch,
        cols,
        used,
        on_host=True,
    )
    assert_bits_equal(
        ctx,
        view,
        lut_view_host(table.host_lut(), table.pages_per_seq, batch, cols, used),
        "host lut view",
    )

    var sparse = ctx.enqueue_create_buffer[DType.int32](batch * 32)
    fill_sparse_indices(
        ctx,
        sparse,
        table.lut,
        table.pages_per_seq,
        page_size,
        False,
        batch,
        1,
        1000,
        32,
        on_host=True,
    )
    assert_bits_equal(
        ctx,
        sparse,
        sparse_indices_host(
            table.host_lut(),
            table.pages_per_seq,
            page_size,
            False,
            batch,
            1,
            1000,
            32,
        ),
        "host sparse indices",
    )


def check_rotation() raises:
    var cold = Rotation(evict_bytes=1000, cold=True, max_bytes=10_000)
    assert_equal(cold.copies(1000), 1)
    assert_equal(cold.copies(999), 2)
    assert_equal(cold.copies(100, copy_bytes=1000), 10)
    assert_equal(cold.copies(10, copy_bytes=1000), 0)
    var warm = Rotation(evict_bytes=1000, cold=False, max_bytes=0)
    assert_equal(warm.copies(10, copy_bytes=1000), 1)


def main() raises:
    check_sweep()
    check_zipped_sweep()
    check_plan_groups()
    check_rotation()
    check_distinct_stride()
    assert_true(dtype_arm_enabled["sweep_test_dtype", DType.bfloat16]())
    with DeviceContext() as ctx:
        check_uniform[DType.float32](ctx)
        check_uniform[DType.bfloat16](ctx)
        check_uniform[DType.float8_e4m3fn](ctx)
        check_uniform[DType.uint8](ctx)
        check_affine[DType.int32](ctx)
        check_affine[DType.uint32](ctx)
        check_affine[DType.float32](ctx)
        check_page_table(ctx, PageLayout.IDENTITY)
        check_page_table(ctx, PageLayout.SHUFFLED)
        check_page_table(ctx, PageLayout.FRAGMENTED)
        check_sparse_indices(ctx, pad_invalid=False)
        check_sparse_indices(ctx, pad_invalid=True)
        check_sparse_indices(ctx, pad_invalid=False, index_stride=3)
        check_sparse_indices(ctx, pad_invalid=True, index_stride=7)
        check_linear_grade(ctx)
        check_lut_view(ctx)
        check_host_fill(ctx)
        check_fixture_bytes(ctx)
        check_copy_luts(ctx)
    print("PASS")
