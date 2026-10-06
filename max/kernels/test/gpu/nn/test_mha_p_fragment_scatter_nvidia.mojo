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

"""Checks NVIDIA P-fragment stores against explicit scalar coordinates."""

from std.sys import simd_width_of
from std.testing import assert_equal
from max.gpu import thread_idx
from max.gpu.host import DeviceContext
from max.gpu.sync import barrier
from layout import TileTensor, row_major, stack_allocation
from layout.layout import Layout as LegacyLayout
from layout.layout_tensor import LayoutTensorIter
from layout.tensor_core import get_fragment_size, get_mma_shape
from nn.attention.mha_utils import _copy_frag_to_smem

comptime BM = 64
comptime BN = 128
comptime BK = 32
comptime WM = 32
comptime WN = 64
comptime MMA_M = 16
comptime MMA_N = 8
comptime FRAGMENT = 4
comptime REG_ROWS = (WM // MMA_M) * (WN // MMA_N)
comptime STAGES = BN // BK
comptime RING_SIZE = BM * BN
comptime GUARD = 16
comptime INITIAL_STAGE = 2
comptime LANES = 32
comptime TOTAL = RING_SIZE + 2 * GUARD + 2 * LANES


@inline(.always)
def _register_value(n_mma: Int, m_mma: Int, lane: Int, frag: Int) -> Float32:
    return Float32(n_mma * 10000 + m_mma * 1000 + lane * 16 + frag + 1) / 32


def _scatter[
    dtype: DType, WIDTH: Int
](output: UnsafePointer[Scalar[dtype], MutAnyOrigin]):
    comptime assert WIDTH == 1 or WIDTH == 2
    comptime mma = get_mma_shape[dtype, DType.float32]()
    comptime assert mma[0] == MMA_M and mma[1] == MMA_N
    comptime assert get_fragment_size[mma]()[2] == FRAGMENT
    comptime assert simd_width_of[dtype]() == (4 if dtype == .float32 else 8)
    var lane = Int(thread_idx.x)
    var storage = stack_allocation[dtype, address_space=.SHARED, alignment=16](
        row_major[RING_SIZE + 2 * GUARD]()
    )
    for i in range(lane, RING_SIZE + 2 * GUARD, LANES):
        storage[i] = -7
    var registers_storage = stack_allocation[.float32, address_space=.LOCAL](
        row_major[REG_ROWS * FRAGMENT + 2]()
    )
    registers_storage[0] = -29
    registers_storage[REG_ROWS * FRAGMENT + 1] = -30
    var registers = TileTensor[address_space=.LOCAL, linear_idx_type=.int32](
        registers_storage.unsafe_ptr() + 1,
        row_major[REG_ROWS, FRAGMENT](),
    )
    comptime for n_mma in range(WN // MMA_N):
        comptime for m_mma in range(WM // MMA_M):
            comptime for frag in range(FRAGMENT):
                registers[
                    n_mma * (WM // MMA_M) + m_mma, frag
                ] = _register_value(n_mma, m_mma, lane, frag)
    comptime Iterator = LayoutTensorIter[
        dtype,
        LegacyLayout.row_major(BM, BK),
        _,
        address_space=.SHARED,
        circular=True,
    ]
    var stages = Iterator(
        storage.unsafe_ptr() + GUARD,
        Iterator.linear_uint_type(RING_SIZE),
        offset=Iterator.linear_uint_type(INITIAL_STAGE * BM * BK),
    )
    barrier()
    _copy_frag_to_smem[BM, BN, BK, WM, WN, MMA_M, MMA_N, WIDTH](
        stages,
        registers.to_layout_tensor().as_unsafe_any_origin(),
        UInt32(1),
        UInt32(1),
    )
    barrier()
    for i in range(lane, RING_SIZE + 2 * GUARD, LANES):
        output[i] = storage[i]
    output[RING_SIZE + 2 * GUARD + 2 * lane] = registers_storage[0].cast[
        dtype
    ]()
    output[RING_SIZE + 2 * GUARD + 2 * lane + 1] = registers_storage[
        REG_ROWS * FRAGMENT + 1
    ].cast[dtype]()


def _check_scatter[dtype: DType, WIDTH: Int](ctx: DeviceContext) raises:
    comptime fragment_rows = MMA_M // 8
    comptime fragment_cols = (MMA_N // WIDTH) // 4
    comptime vectors = FRAGMENT // WIDTH
    comptime assert vectors == fragment_rows * fragment_cols
    comptime vector_group = 4 if dtype == .float32 else 8
    comptime xor_mask = 7 if dtype == .float32 else 3
    var expected = ctx.enqueue_create_host_buffer[dtype](TOTAL)
    var actual = ctx.enqueue_create_host_buffer[dtype](TOTAL)
    for i in range(TOTAL):
        expected[i] = Scalar[dtype](-7)
        actual[i] = Scalar[dtype](42)
    for n_mma in range(WN // MMA_N):
        for m_mma in range(WM // MMA_M):
            for lane in range(LANES):
                for vec in range(vectors):
                    # Legacy fragment ordinals advance the leading mode first.
                    var row = (
                        WM
                        + m_mma * MMA_M
                        + lane // 4
                        + (vec % fragment_rows) * 8
                    )
                    var col = (
                        WN
                        + n_mma * MMA_N
                        + (lane % 4) * WIDTH
                        + ((vec // fragment_rows) % fragment_cols) * (4 * WIDTH)
                    )
                    var scalar_offset = row * BK + col % BK
                    var vector_offset = scalar_offset // vector_group
                    # BK32 permutes vector-address bits3..4 (half) or3..5 (FP32).
                    var swizzled = vector_offset ^ (
                        (vector_offset >> 3) & xor_mask
                    )
                    var stage = (INITIAL_STAGE + col // BK) % STAGES
                    for element in range(WIDTH):
                        var offset = (
                            GUARD
                            + stage * BM * BK
                            + swizzled * vector_group
                            + scalar_offset % vector_group
                            + element
                        )
                        expected[offset] = Scalar[dtype](
                            _register_value(
                                n_mma, m_mma, lane, vec * WIDTH + element
                            )
                        )
    for lane in range(LANES):
        expected[RING_SIZE + 2 * GUARD + 2 * lane] = Scalar[dtype](-29)
        expected[RING_SIZE + 2 * GUARD + 2 * lane + 1] = Scalar[dtype](-30)
    var output = ctx.enqueue_create_buffer[dtype](TOTAL)
    ctx.enqueue_copy(output, actual)
    ctx.enqueue_function[_scatter[dtype, WIDTH]](
        output, grid_dim=1, block_dim=LANES
    )
    ctx.enqueue_copy(actual, output)
    ctx.synchronize()
    for i in range(TOTAL):
        assert_equal(actual[i], expected[i])


def main() raises:
    with DeviceContext() as ctx:
        _check_scatter[.bfloat16, 2](ctx)
        _check_scatter[.float16, 2](ctx)
        _check_scatter[.float32, 2](ctx)
        _check_scatter[.bfloat16, 1](ctx)
        _check_scatter[.float16, 1](ctx)
        _check_scatter[.float32, 1](ctx)
