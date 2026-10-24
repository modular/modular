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
"""Validates the W4A8 test harness's reference against an independent kernel.

The SM100 block-scaled W4A8 tests trust `w4a8_block_scaled_matmul_harness`
for operand generation, the scale-factor layout, nibble order, the sampled
comparison, and the tolerance. This test feeds harness-generated operands to
the grouped 1D1D W4A8 kernel, an existing implementation of the same
arithmetic, with a single expert, and requires the harness reference to
accept its output. It then requires the check to reject a 1% output error
and a single unwritten element, so an accepting reference is not vacuous.
"""

from std.math import ceildiv
from std.random import seed
from std.utils.numerics import nan
from std.testing import assert_raises

from max.gpu.host import DeviceContext
from layout import Coord, Idx, TileTensor, row_major
from linalg.fp4_utils import SF_ATOM_K, SF_ATOM_M, SF_MN_GROUP_SIZE
from linalg.matmul.gpu.sm100_structured.grouped_block_scaled_1d1d import (
    grouped_matmul_block_scaled_sm100_dispatch,
)
from w4a8_block_scaled_matmul_harness import (
    C_DTYPE,
    SF_DTYPE,
    SF_VECTOR_SIZE,
    W4A8_A_DTYPE,
    W4A8_B_DTYPE,
    assert_w4a8_output_matches_reference,
    fill_w4a8_operands,
)


def check_reference_against_grouped_kernel[
    N: Int, K: Int, realistic: Bool
](ctx: DeviceContext, M: Int) raises:
    """Runs one single-expert grouped W4A8 matmul and checks the reference."""
    print(t"reference check M={M} N={N} K={K} realistic={realistic}")
    seed(99)
    comptime k_groups = ceildiv(K, SF_VECTOR_SIZE * SF_ATOM_K)
    comptime sf_atom = SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K
    var m_groups = ceildiv(M, SF_MN_GROUP_SIZE)
    comptime n_groups = ceildiv(N, SF_MN_GROUP_SIZE)
    var a_host = ctx.enqueue_create_host_buffer[W4A8_A_DTYPE](M * K)
    var b_host = ctx.enqueue_create_host_buffer[W4A8_B_DTYPE](N * K // 2)
    var c_host = ctx.enqueue_create_host_buffer[C_DTYPE](M * N)
    var sfa_host = ctx.enqueue_create_host_buffer[SF_DTYPE](
        m_groups * k_groups * sf_atom
    )
    var sfb_host = ctx.enqueue_create_host_buffer[SF_DTYPE](
        n_groups * k_groups * sf_atom
    )
    var offsets_host = ctx.enqueue_create_host_buffer[.uint32](2)
    var sf_offsets_host = ctx.enqueue_create_host_buffer[.uint32](1)
    var ids_host = ctx.enqueue_create_host_buffer[.int32](1)
    var escale_host = ctx.enqueue_create_host_buffer[.float32](1)
    ctx.synchronize()
    fill_w4a8_operands[N, K, realistic](
        a_host, b_host, sfa_host, sfb_host, 1, M
    )
    offsets_host[0] = 0
    offsets_host[1] = UInt32(M)
    sf_offsets_host[0] = 0
    ids_host[0] = 0
    escale_host[0] = 1.0

    var a_dev = ctx.enqueue_create_buffer[W4A8_A_DTYPE](M * K)
    var b_dev = ctx.enqueue_create_buffer[W4A8_B_DTYPE](N * K // 2)
    var c_dev = ctx.enqueue_create_buffer[C_DTYPE](M * N)
    var sfa_dev = ctx.enqueue_create_buffer[SF_DTYPE](
        m_groups * k_groups * sf_atom
    )
    var sfb_dev = ctx.enqueue_create_buffer[SF_DTYPE](
        n_groups * k_groups * sf_atom
    )
    var offsets_dev = ctx.enqueue_create_buffer[.uint32](2)
    var sf_offsets_dev = ctx.enqueue_create_buffer[.uint32](1)
    var ids_dev = ctx.enqueue_create_buffer[.int32](1)
    var escale_dev = ctx.enqueue_create_buffer[.float32](1)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.enqueue_copy(sfa_dev, sfa_host)
    ctx.enqueue_copy(sfb_dev, sfb_host)
    ctx.enqueue_copy(offsets_dev, offsets_host)
    ctx.enqueue_copy(sf_offsets_dev, sf_offsets_host)
    ctx.enqueue_copy(ids_dev, ids_host)
    ctx.enqueue_copy(escale_dev, escale_host)
    ctx.enqueue_memset(c_dev, nan[C_DTYPE]())

    grouped_matmul_block_scaled_sm100_dispatch[transpose_b=True, target="gpu"](
        TileTensor(c_dev, row_major(Coord(M, Idx[N]))).as_unsafe_any_origin(),
        TileTensor(a_dev, row_major(Coord(M, Idx[K]))).as_unsafe_any_origin(),
        TileTensor(
            b_dev, row_major(Coord(Idx[1], Idx[N], Idx[K // 2]))
        ).as_unsafe_any_origin(),
        TileTensor(
            sfa_dev,
            row_major(
                Coord(
                    m_groups,
                    Idx[k_groups],
                    Idx[SF_ATOM_M[0]],
                    Idx[SF_ATOM_M[1]],
                    Idx[SF_ATOM_K],
                )
            ),
        ).as_unsafe_any_origin(),
        TileTensor(
            sfb_dev,
            row_major(
                Coord(
                    Idx[1],
                    Idx[n_groups],
                    Idx[k_groups],
                    Idx[SF_ATOM_M[0]],
                    Idx[SF_ATOM_M[1]],
                    Idx[SF_ATOM_K],
                )
            ),
        ).as_unsafe_any_origin(),
        TileTensor(
            offsets_dev, row_major(Coord(Int(2)))
        ).as_unsafe_any_origin(),
        TileTensor(
            sf_offsets_dev, row_major(Coord(Int(1)))
        ).as_unsafe_any_origin(),
        TileTensor(ids_dev, row_major(Coord(Int(1)))).as_unsafe_any_origin(),
        TileTensor(escale_dev, row_major(Coord(Idx[1]))).as_unsafe_any_origin(),
        1,
        M,
        ctx,
    )
    ctx.enqueue_copy(c_host, c_dev)
    ctx.synchronize()
    assert_w4a8_output_matches_reference[N, K, realistic](
        c_host, a_host, b_host, sfa_host, sfb_host, 1, M, 1.0
    )
    print("  grouped kernel output accepted")
    with assert_raises():
        assert_w4a8_output_matches_reference[N, K, realistic](
            c_host, a_host, b_host, sfa_host, sfb_host, 1, M, 1.01
        )
    c_host[M * N - 1] = nan[C_DTYPE]()
    with assert_raises(contains="never written"):
        assert_w4a8_output_matches_reference[N, K, realistic](
            c_host, a_host, b_host, sfa_host, sfb_host, 1, M, 1.0
        )
    print("  corrupted outputs rejected")
    _ = a_dev^
    _ = b_dev^
    _ = c_dev^
    _ = sfa_dev^
    _ = sfb_dev^
    _ = offsets_dev^
    _ = sf_offsets_dev^
    _ = ids_dev^
    _ = escale_dev^


def main() raises:
    with DeviceContext() as ctx:
        check_reference_against_grouped_kernel[N=256, K=512, realistic=False](
            ctx, 128
        )
        # M = 200 leaves the second 128-row scale group partly padding.
        check_reference_against_grouped_kernel[N=512, K=1152, realistic=False](
            ctx, 200
        )
        check_reference_against_grouped_kernel[N=512, K=1152, realistic=True](
            ctx, 200
        )
