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

"""Checks online-softmax register fragments against a scalar two-step oracle.

Two MMA tile axes and two communicating warps exercise every score and
output fragment. Scores vary across warps, tiles, and fragment lanes so
incorrect SIMD widths or tile indexing cannot hide behind uniform inputs.
"""

from std.math import exp2
from std.testing import assert_almost_equal, assert_equal
from max.gpu import WARP_SIZE, thread_idx, warp_id
from max.gpu.host import DeviceContext
from max.gpu.sync import barrier
from layout import Layout, TensorLayout, TileTensor, row_major, stack_allocation
from nn.softmax import _online_softmax_iter_for_mma_output

comptime NUM_THREADS = 64
comptime NUM_FIELDS = 32
comptime NUM_REGISTER_ELEMENTS = 8
comptime LOCAL_GUARD = 2
comptime DRAM_GUARD = 4
comptime SENTINEL = Float32(-91)


@inline(.always)
def _score(
    iteration: Int, m: Int, n: Int, warp: Int, lane_col: Int, element: Int
) -> Float32:
    return (
        Float32(2 * iteration + warp)
        + Float32(m) * 0.5
        + Float32(n) * 0.25
        + Float32(lane_col) * 0.125
        + Float32(element) * 0.0625
    )


def _kernel[
    OutputLayout: TensorLayout
](output: TileTensor[.float32, OutputLayout, MutAnyOrigin]) where (
    output.flat_rank == 2
):
    comptime assert WARP_SIZE == 32, "This fixture covers NVIDIA fragments"
    var score_storage = stack_allocation[
        dtype=.float32, address_space=.LOCAL, alignment=16
    ](row_major[NUM_REGISTER_ELEMENTS + 2 * LOCAL_GUARD]())
    var accum_storage = stack_allocation[
        dtype=.float32, address_space=.LOCAL, alignment=16
    ](row_major[NUM_REGISTER_ELEMENTS + 2 * LOCAL_GUARD]())
    var scratch_storage = stack_allocation[
        dtype=.float32, address_space=.SHARED, alignment=16
    ](row_major[64 + 2 * LOCAL_GUARD]())
    for i in range(NUM_REGISTER_ELEMENTS + 2 * LOCAL_GUARD):
        score_storage[i] = SENTINEL
        accum_storage[i] = SENTINEL
    for i in range(thread_idx.x, 64 + 2 * LOCAL_GUARD, NUM_THREADS):
        scratch_storage[i] = SENTINEL
    barrier()
    var scores = TileTensor[address_space=.LOCAL, linear_idx_type=.int32](
        score_storage.ptr + LOCAL_GUARD, row_major[4, 2]()
    )
    var accum = TileTensor[address_space=.LOCAL, linear_idx_type=.int32](
        accum_storage.ptr + LOCAL_GUARD, row_major[4, 2]()
    )
    var scratch = TileTensor[address_space=.SHARED, linear_idx_type=.int32](
        scratch_storage.ptr + LOCAL_GUARD, row_major[4, 16]()
    )
    var rowmax = stack_allocation[dtype=.float32](row_major[2, 1]())
    var rowsum = stack_allocation[dtype=.float32](row_major[2, 1]())
    comptime for m in range(2):
        rowmax[m, 0] = 0
        rowsum[m, 0] = 0
    comptime for tile in range(4):
        comptime for element in range(2):
            accum[tile, element] = Float32(2) + Float32(2 * tile + element) / 16
    var scores_fragment = scores.vectorize[1, 2]()
    var accum_fragment = accum.vectorize[1, 2]()
    comptime for iteration in range(2):
        comptime for n in range(2):
            comptime for m in range(2):
                comptime for element in range(2):
                    scores[m + 2 * n, element] = _score(
                        iteration, m, n, warp_id(), thread_idx.x % 4, element
                    )
        _online_softmax_iter_for_mma_output[
            .float32,
            Layout.row_major(2, 2),
            Layout.row_major(1, 2),
            Layout.row_major(8, 4),
            use_exp2=True,
            fragment_layout=Layout.row_major(1, 2),
        ](accum_fragment, scores_fragment, scratch, rowmax.ptr, rowsum.ptr)
    comptime for m in range(2):
        output[thread_idx.x, m] = rowmax[m, 0]
        output[thread_idx.x, 2 + m] = rowsum[m, 0]
    comptime for tile in range(4):
        comptime for element in range(2):
            output[thread_idx.x, 4 + 2 * tile + element] = accum[tile, element]
            output[thread_idx.x, 12 + 2 * tile + element] = scores[
                tile, element
            ]
    comptime for guard in range(LOCAL_GUARD):
        output[thread_idx.x, 20 + guard] = score_storage[guard]
        output[thread_idx.x, 22 + guard] = score_storage[
            LOCAL_GUARD + NUM_REGISTER_ELEMENTS + guard
        ]
        output[thread_idx.x, 24 + guard] = accum_storage[guard]
        output[thread_idx.x, 26 + guard] = accum_storage[
            LOCAL_GUARD + NUM_REGISTER_ELEMENTS + guard
        ]
        output[thread_idx.x, 28 + guard] = scratch_storage[guard]
        output[thread_idx.x, 30 + guard] = scratch_storage[
            LOCAL_GUARD + 64 + guard
        ]


def main() raises:
    with DeviceContext() as ctx:
        comptime count = NUM_THREADS * NUM_FIELDS + 2 * DRAM_GUARD
        var device = ctx.enqueue_create_buffer[.float32](count)
        var host = ctx.enqueue_create_host_buffer[.float32](count)
        for i in range(count):
            host[i] = SENTINEL
        ctx.enqueue_copy(device, host)
        var output = TileTensor(
            device.unsafe_ptr() + DRAM_GUARD,
            row_major[NUM_THREADS, NUM_FIELDS](),
        ).as_unsafe_any_origin()
        ctx.enqueue_function[_kernel[type_of(output).LayoutType]](
            output, grid_dim=1, block_dim=NUM_THREADS
        )
        ctx.enqueue_copy(host, device)
        ctx.synchronize()
        for guard in range(DRAM_GUARD):
            assert_equal(host[guard], SENTINEL)
            assert_equal(
                host[DRAM_GUARD + NUM_THREADS * NUM_FIELDS + guard], SENTINEL
            )
        for thread in range(NUM_THREADS):
            var base = DRAM_GUARD + thread * NUM_FIELDS
            for m in range(2):
                var final_max = Float64(_score(1, m, 1, 1, 3, 1))
                var sum = Float64(0)
                for iteration in range(2):
                    for warp in range(2):
                        for n in range(2):
                            for lane_col in range(4):
                                for element in range(2):
                                    sum += exp2(
                                        Float64(
                                            _score(
                                                iteration,
                                                m,
                                                n,
                                                warp,
                                                lane_col,
                                                element,
                                            )
                                        )
                                        - final_max
                                    )
                assert_almost_equal(host[base + m], Float32(final_max))
                assert_almost_equal(
                    host[base + 2 + m], Float32(sum), atol=1e-5, rtol=1e-5
                )
                for n in range(2):
                    var tile = m + 2 * n
                    for element in range(2):
                        var initial_accum = (
                            Float64(2) + Float64(2 * tile + element) / 16
                        )
                        var expected_accum = initial_accum * exp2(-final_max)
                        var final_score = Float64(
                            _score(
                                1,
                                m,
                                n,
                                thread // WARP_SIZE,
                                thread % 4,
                                element,
                            )
                        )
                        assert_almost_equal(
                            host[base + 4 + 2 * tile + element],
                            Float32(expected_accum),
                            atol=1e-6,
                            rtol=1e-5,
                        )
                        assert_almost_equal(
                            host[base + 12 + 2 * tile + element],
                            Float32(exp2(final_score - final_max)),
                            atol=1e-6,
                            rtol=1e-5,
                        )
            for field in range(20, NUM_FIELDS):
                assert_equal(host[base + field], SENTINEL)
