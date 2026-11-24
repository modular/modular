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
"""Checks the GPU `mo.squeeze_shape` / `mo.unsqueeze_shape` kernels against a
host reference.

Covers every rank up to 130 (above every warp size and the launch block size),
randomized index sets with negative and unsorted entries, empty and full index
sets, and both index dtypes. Output buffers carry a canary tail that must stay
untouched.
"""

from std.testing import TestSuite, assert_equal

from builtin_kernels.gather_scatter import (
    squeeze_shape_gpu,
    unsqueeze_shape_gpu,
)
from max.gpu.host import DeviceContext

comptime shape_dtype = DType.int64
comptime MAX_RANK = 130
comptime CANARY = -7
comptime TAIL = 5


struct Rng:
    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def below(mut self, bound: Int) -> Int:
        """Returns a value in [0, bound) from a xorshift64 step."""
        self.state ^= self.state << 13
        self.state ^= self.state >> 7
        self.state ^= self.state << 17
        return Int((self.state >> 11) % UInt64(bound))


def _random_indices(mut rng: Rng, index_rank: Int, count: Int) -> List[Int]:
    """Returns `count` distinct dims of `index_rank`, unsorted, ~half negative.
    """
    var perm = List[Int]()
    for d in range(index_rank):
        perm.append(d)
    for k in range(count):
        var pick = k + rng.below(index_rank - k)
        var tmp = perm[k]
        perm[k] = perm[pick]
        perm[pick] = tmp
    var dims = List[Int]()
    for k in range(count):
        var d = perm[k]
        dims.append(d - index_rank if rng.below(2) == 0 else d)
    return dims^


def _reference(
    squeeze: Bool, shape: List[Int], indices: List[Int], index_rank: Int
) -> List[Int]:
    var listed = List[Bool](length=index_rank, fill=False)
    for i in indices:
        listed[i + index_rank if i < 0 else i] = True
    var out = List[Int]()
    if squeeze:
        for d in range(len(shape)):
            if not listed[d]:
                out.append(shape[d])
    else:
        var j = 0
        for d in range(index_rank):
            if listed[d]:
                out.append(1)
            else:
                out.append(shape[j])
                j += 1
    return out^


def run_case[
    squeeze: Bool, indices_type: DType
](ctx: DeviceContext, rank: Int, indices: List[Int], index_rank: Int) raises:
    var num_indices = len(indices)
    var out_rank = rank - num_indices if squeeze else rank + num_indices
    var shape = List[Int]()
    for d in range(rank):
        shape.append(10 + d)
    var expected = _reference(squeeze, shape, indices, index_rank)

    var in_host = ctx.enqueue_create_host_buffer[shape_dtype](rank + 1)
    var idx_host = ctx.enqueue_create_host_buffer[indices_type](num_indices + 1)
    var out_host = ctx.enqueue_create_host_buffer[shape_dtype](out_rank + TAIL)
    ctx.synchronize()
    for d in range(rank):
        in_host[d] = Scalar[shape_dtype](shape[d])
    for k in range(num_indices):
        idx_host[k] = Scalar[indices_type](indices[k])

    var in_dev = ctx.enqueue_create_buffer[shape_dtype](rank + 1)
    var idx_dev = ctx.enqueue_create_buffer[indices_type](num_indices + 1)
    var out_dev = ctx.enqueue_create_buffer[shape_dtype](out_rank + TAIL)
    ctx.enqueue_copy(in_dev, in_host)
    ctx.enqueue_copy(idx_dev, idx_host)
    out_dev.enqueue_fill(CANARY)

    var out_ptr = out_dev.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var in_ptr = in_dev.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var idx_ptr = idx_dev.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    comptime if squeeze:
        squeeze_shape_gpu(
            out_ptr,
            1,
            in_ptr,
            1,
            idx_ptr,
            1,
            rank,
            num_indices,
            ctx,
        )
    else:
        unsqueeze_shape_gpu(
            out_ptr,
            1,
            in_ptr,
            1,
            idx_ptr,
            1,
            rank,
            num_indices,
            ctx,
        )

    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()
    # The helpers take untracked pointers, so keep the device inputs alive
    # until the kernel has run.
    _ = in_dev^
    _ = idx_dev^
    for d in range(out_rank):
        assert_equal(
            Int(out_host[d]),
            expected[d],
            String(t"squeeze={squeeze} rank={rank} n={num_indices} out[{d}]"),
        )
    for t in range(TAIL):
        assert_equal(
            Int(out_host[out_rank + t]),
            CANARY,
            String(t"squeeze={squeeze} rank={rank} n={num_indices} tail[{t}]"),
        )


def run_rank[
    squeeze: Bool, indices_type: DType
](ctx: DeviceContext, mut rng: Rng, rank: Int) raises:
    # `index_rank` is the rank the indices address: the input rank for squeeze,
    # the output rank for unsqueeze.
    var max_count = rank if squeeze else MAX_RANK - rank
    var counts = List[Int]()
    counts.append(0)
    counts.append(min(max_count, 1))
    counts.append(min(max_count, 3))
    counts.append(max_count)
    counts.append(rng.below(max_count + 1))
    counts.append(rng.below(max_count + 1))
    for count in counts:
        var index_rank = rank if squeeze else rank + count
        var indices = _random_indices(rng, index_rank, count)
        run_case[squeeze, indices_type](ctx, rank, indices, index_rank)


def test_squeeze_shape_matches_reference() raises:
    var ctx = DeviceContext()
    var rng = Rng(0x9E3779B97F4A7C15)
    for rank in range(1, MAX_RANK + 1):
        run_rank[True, DType.int64](ctx, rng, rank)
        run_rank[True, DType.int32](ctx, rng, rank)


def test_unsqueeze_shape_matches_reference() raises:
    var ctx = DeviceContext()
    var rng = Rng(0xD1B54A32D192ED03)
    for rank in range(0, MAX_RANK):
        run_rank[False, DType.int64](ctx, rng, rank)
        run_rank[False, DType.int32](ctx, rng, rank)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
