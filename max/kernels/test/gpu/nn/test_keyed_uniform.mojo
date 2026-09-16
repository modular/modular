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
"""Correctness tests for `nn.rand_uniform.keyed_uniform`.

The kernel draws one uniform value in [0, 1) per row, keyed off that row's
own seed. Speculative decoding's accept coin needs exactly that: a draw a
request can reproduce from its own seed, whatever else shares the batch.

So the tests pin the two halves separately -- that the marginal really is
uniform on [0, 1), and that a row's value is a function of its seed alone
and not of its position or of the launch it rode in.
"""

from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from std.math import sqrt
from std.testing import assert_almost_equal, assert_equal, assert_true

from nn.rand_uniform import keyed_uniform


def _draw(ctx: DeviceContext, seeds: List[UInt64]) raises -> List[Float32]:
    """Runs one launch and returns the drawn value per row."""
    var rows = len(seeds)
    var seeds_host = ctx.enqueue_create_host_buffer[.uint64](rows)
    for row in range(rows):
        seeds_host[row] = seeds[row]

    var seeds_dev = ctx.enqueue_create_buffer[.uint64](rows)
    var out_dev = ctx.enqueue_create_buffer[.float32](rows)
    ctx.enqueue_copy(seeds_dev, seeds_host)

    keyed_uniform[target="gpu"](
        TileTensor(out_dev, row_major(rows)),
        TileTensor(seeds_dev, row_major(rows))
        .as_unsafe_any_origin()
        .as_immut(),
        ctx,
    )

    var out_host = ctx.enqueue_create_host_buffer[.float32](rows)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    var out = List[Float32]()
    for row in range(rows):
        out.append(out_host[row])

    _ = seeds_dev^
    _ = out_dev^
    return out^


def test_draw_is_uniform(ctx: DeviceContext) raises:
    """The marginal over distinct seeds is uniform on [0, 1).

    Sixteen equal-width buckets over 65536 rows, each checked against its
    expected occupancy at five binomial standard deviations, plus the mean
    and the bounds. A generator stuck on a subrange, biased toward either
    end, or quantized onto a coarse grid fails at least one of those. With
    fixed seeds the test is deterministic; the statistics only size the
    bounds.
    """
    comptime rows = 65536
    comptime buckets = 16

    var seeds = List[UInt64]()
    for row in range(rows):
        seeds.append(UInt64(0x51ED_C0DE + row))

    var out = _draw(ctx, seeds)

    var counts = List[Int]()
    for _ in range(buckets):
        counts.append(0)

    var total = Float64(0.0)
    for row in range(rows):
        var value = out[row]
        assert_true(
            value >= 0.0 and value < 1.0,
            String(t"row {row}: {value} is outside [0, 1)"),
        )
        total += Float64(value)
        counts[Int(Float64(value) * Float64(buckets))] += 1

    var mean = total / Float64(rows)
    # Five standard errors of the mean of a uniform: 5 * sqrt(1/12 / rows).
    var mean_tol = 5.0 * sqrt(1.0 / 12.0 / Float64(rows))
    assert_almost_equal(
        mean, 0.5, atol=mean_tol, msg=String(t"mean {mean} != 0.5")
    )

    var expected = 1.0 / Float64(buckets)
    var tol = 5.0 * sqrt(expected * (1.0 - expected) / Float64(rows))
    for bucket in range(buckets):
        var freq = Float64(counts[bucket]) / Float64(rows)
        assert_true(
            abs(freq - expected) < tol,
            String(t"bucket {bucket}: frequency {freq} != {expected}"),
        )


def test_equal_seeds_draw_equal_values(ctx: DeviceContext) raises:
    """A row's value is a function of its seed, and distinct seeds differ.

    The accept coin keys one seed per (request, draft position), so equal
    seeds must agree exactly -- and, as the anti-vacuity half, distinct
    seeds must not, or a generator that ignored its seed entirely would
    satisfy the first half alone.
    """
    var seeds: List[UInt64] = [42, 7, 42, 9, 7, 42]
    var out = _draw(ctx, seeds)

    assert_equal(out[0], out[2], "equal seeds must draw equal values")
    assert_equal(out[0], out[5], "equal seeds must draw equal values")
    assert_equal(out[1], out[4], "equal seeds must draw equal values")
    assert_true(out[0] != out[1], "seeds 42 and 7 drew the same value")
    assert_true(out[0] != out[3], "seeds 42 and 9 drew the same value")
    assert_true(out[1] != out[3], "seeds 7 and 9 drew the same value")


def test_draw_ignores_row_position(ctx: DeviceContext) raises:
    """A seed draws the same value wherever it sits, in any size of launch.

    This is the property `random_uniform` cannot offer: it keys the whole
    tensor off `seed_ptr[0]` and walks the flat element index as its Philox
    counter, so a value there is a function of position. A request's accept
    coin has to survive being rebatched next to different co-residents, so
    the draw may not read the row index at all.
    """
    comptime probe = UInt64(0xA5A5_1234_DEAD_BEEF)

    var alone = _draw(ctx, [probe])

    var padded = List[UInt64]()
    for row in range(257):
        padded.append(UInt64(row * 2654435761 + 1))
    padded[0] = probe
    padded[137] = probe
    padded[256] = probe
    var out = _draw(ctx, padded)

    assert_equal(out[0], alone[0], "row 0 of 257 must match a lone draw")
    assert_equal(out[137], alone[0], "an interior row must match")
    assert_equal(out[256], alone[0], "the last row must match")


def test_empty_batch(ctx: DeviceContext) raises:
    """Zero rows must be a no-op, not a zero-sized grid launch.

    The rejection sampler runs with no rows on every step that has no
    drafts to verify, so the empty batch is a normal input.
    """
    var seeds_dev = ctx.enqueue_create_buffer[.uint64](0)
    var out_dev = ctx.enqueue_create_buffer[.float32](0)
    keyed_uniform[target="gpu"](
        TileTensor(out_dev, row_major(0)),
        TileTensor(seeds_dev, row_major(0)).as_unsafe_any_origin().as_immut(),
        ctx,
    )
    ctx.synchronize()
    _ = seeds_dev^
    _ = out_dev^


def main() raises:
    with DeviceContext() as ctx:
        test_empty_batch(ctx)
        test_draw_is_uniform(ctx)
        test_equal_seeds_draw_equal_values(ctx)
        test_draw_ignores_row_position(ctx)
