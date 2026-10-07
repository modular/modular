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

"""Checks scalar register correction, including strided rows and padding."""

from std.math import exp, exp2
from std.testing import assert_almost_equal, assert_equal
from max.gpu import thread_idx
from max.gpu.host import DeviceContext
from layout import (
    Coord,
    Idx,
    MixedLayout,
    TensorLayout,
    TileTensor,
    row_major,
    stack_allocation,
)
from nn.softmax import _online_softmax_correction

comptime THREADS = 32
comptime GUARD = 2
comptime SENTINEL = Float32(-91)


@inline(.always)
def _previous(thread: Int, row: Int) -> Float32:
    return Float32(-1) + Float32(thread % 7) / 8 - Float32(row) / 16


@inline(.always)
def _next(thread: Int, row: Int) -> Float32:
    return (
        _previous(thread, row) + Float32(row + 1) / 4 + Float32(thread % 3) / 16
    )


def _kernel[
    N: Int, STRIDE: Int, USE_EXP2: Bool, OutputLayout: TensorLayout
](output: TileTensor[.float32, OutputLayout, MutAnyOrigin]) where (
    output.flat_rank == 2
):
    comptime extent = (N - 1) * STRIDE + 1
    comptime storage_size = extent + 2 * GUARD
    var previous_storage = stack_allocation[
        dtype=.float32, address_space=.LOCAL, alignment=16
    ](row_major[storage_size]()).fill(SENTINEL)
    var next_storage = stack_allocation[
        dtype=.float32, address_space=.LOCAL, alignment=16
    ](row_major[storage_size]()).fill(SENTINEL)
    var previous = TileTensor[address_space=.LOCAL, linear_idx_type=.int32](
        previous_storage.ptr + GUARD,
        MixedLayout(Coord(Idx[N]), Coord(Idx[STRIDE])),
    )
    var next_max = TileTensor[address_space=.LOCAL, linear_idx_type=.int32](
        next_storage.ptr + GUARD,
        MixedLayout(Coord(Idx[N]), Coord(Idx[STRIDE])),
    )
    comptime for row in range(N):
        previous[row] = _previous(thread_idx.x, row)
        next_max[row] = _next(thread_idx.x, row)
    _online_softmax_correction[use_exp2=USE_EXP2](
        previous.as_unsafe_any_origin(),
        next_max.as_unsafe_any_origin(),
    )
    comptime for i in range(storage_size):
        output[thread_idx.x, i] = previous_storage[i]
        output[thread_idx.x, storage_size + i] = next_storage[i]


def _check[N: Int, STRIDE: Int, USE_EXP2: Bool](ctx: DeviceContext) raises:
    comptime extent = (N - 1) * STRIDE + 1
    comptime storage_size = extent + 2 * GUARD
    comptime fields = 2 * storage_size
    comptime total = THREADS * fields + 2 * GUARD
    var host = ctx.enqueue_create_host_buffer[.float32](total)
    var device = ctx.enqueue_create_buffer[.float32](total)
    for i in range(total):
        host[i] = SENTINEL
    ctx.enqueue_copy(device, host)
    var output = TileTensor(
        device.unsafe_ptr() + GUARD, row_major[THREADS, fields]()
    ).as_unsafe_any_origin()
    ctx.enqueue_function[
        _kernel[N, STRIDE, USE_EXP2, type_of(output).LayoutType]
    ](output, grid_dim=1, block_dim=THREADS)
    ctx.enqueue_copy(host, device)
    ctx.synchronize()
    for i in range(GUARD):
        assert_equal(host[i], SENTINEL)
        assert_equal(host[GUARD + THREADS * fields + i], SENTINEL)
    for thread in range(THREADS):
        var base = GUARD + thread * fields
        for i in range(storage_size):
            var relative = i - GUARD
            if relative >= 0 and relative < extent and relative % STRIDE == 0:
                var row = relative // STRIDE
                var previous = Float64(_previous(thread, row))
                var next_max = Float64(_next(thread, row))
                var correction = exp2(previous - next_max) if USE_EXP2 else exp(
                    previous - next_max
                )
                assert_equal(host[base + i], Float32(next_max))
                assert_almost_equal(
                    host[base + storage_size + i],
                    Float32(correction),
                    atol=1e-6,
                    rtol=1e-5,
                )
            else:
                assert_equal(host[base + i], SENTINEL)
                assert_equal(host[base + storage_size + i], SENTINEL)


def main() raises:
    with DeviceContext() as ctx:
        _check[1, 1, False](ctx)
        _check[1, 1, True](ctx)
        _check[1, 3, False](ctx)
        _check[1, 3, True](ctx)
        _check[3, 1, False](ctx)
        _check[3, 1, True](ctx)
        _check[3, 3, False](ctx)
        _check[3, 3, True](ctx)
        _check[8, 1, False](ctx)
        _check[8, 1, True](ctx)
        _check[8, 3, False](ctx)
        _check[8, 3, True](ctx)
