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
"""Bitwise parity of the TileTensor SM90 attention output store against the
legacy `LayoutTensor` implementation it replaced."""

from std.sys import size_of
from max.gpu import thread_idx
from max.gpu.globals import WARPGROUP_SIZE
from max.gpu.host import DeviceContext
from max.gpu.sync import barrier
from layout import (
    IntTuple,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    row_major,
)
from layout.layout_tensor import (
    copy_local_to_shared as legacy_copy_local_to_shared,
    copy_sram_to_dram as legacy_copy_sram_to_dram,
)
from layout.swizzle import Swizzle, make_swizzle
from layout.tile_io import copy_sram_to_dram
from layout.tile_tensor import stack_allocation
from nn.attention.gpu.nvidia.mha_tile_scheduler import SeqInfo, WorkInfo
from nn.attention.gpu.nvidia.common import (
    MHAPosition,
    _LocalTT,
    _SharedMemTT,
    output_reg_to_smem_st_matrix,
)
from nn.attention.gpu.nvidia.sm90.attention import output_reg_to_smem
from std.testing import assert_equal

comptime SENTINEL = 7


def legacy_output_reg_to_smem[
    output_type: DType,
    accum_type: DType,
    num_m_mmas: Int,
    o_frag_size: Int,
    //,
    BM: Int,
    BN: Int,
    padded_depth: Int,
    swizzle: Swizzle,
    num_consumer: Int,
](
    tid: UInt32,
    local_warp_group_idx: UInt32,
    warp_y: UInt32,
    q_smem: UnsafePointer[
        Scalar[output_type], MutAnyOrigin, address_space=.SHARED
    ],
    output_reg_tile: LayoutTensor[
        accum_type,
        Layout.row_major(num_m_mmas, o_frag_size),
        MutAnyOrigin,
        address_space=.LOCAL,
    ],
) -> LayoutTensor[
    output_type,
    Layout.row_major(BM, padded_depth),
    MutAnyOrigin,
    address_space=.SHARED,
]:
    var accum_smem_tile = LayoutTensor[
        output_type,
        Layout.row_major(BM, padded_depth),
        address_space=.SHARED,
    ](q_smem)
    comptime use_stmatrix = accum_type == DType.float32 and padded_depth % 16 == 0 and size_of[
        output_type
    ]() == 2 and o_frag_size % 8 == 0

    comptime if use_stmatrix:
        var warp_group_thread_idx = tid % UInt32(WARPGROUP_SIZE)
        comptime reg_layout = row_major[num_m_mmas, o_frag_size]()
        comptime smem_layout = row_major[BM, padded_depth]()
        output_reg_to_smem_st_matrix[BM, swizzle, num_consumer](
            warp_group_thread_idx,
            local_warp_group_idx,
            _LocalTT[accum_type, reg_layout](output_reg_tile.ptr, reg_layout),
            _SharedMemTT[output_type, smem_layout](
                accum_smem_tile.ptr, smem_layout
            ),
        )
    else:
        comptime mma_thread_layout = Layout.row_major(8, 4)
        var accum_smem_warp_tile = accum_smem_tile.tile[16, BN](
            Int(warp_y), Int(0)
        )
        legacy_copy_local_to_shared[
            thread_layout=mma_thread_layout, swizzle=swizzle
        ](
            accum_smem_warp_tile.vectorize[1, 2](),
            output_reg_tile.vectorize[1, 2]().transpose(),
        )
    return accum_smem_tile


def kernel[
    output_type: DType,
    accum_type: DType,
    depth: Int,
    num_consumer: Int,
](
    new_out: UnsafePointer[Scalar[output_type], MutAnyOrigin],
    old_out: UnsafePointer[Scalar[output_type], MutAnyOrigin],
    valid_rows: Int32,
):
    comptime BM = 64 * num_consumer
    comptime num_consumer_threads = WARPGROUP_SIZE * num_consumer
    comptime MMA_M = 16
    comptime o_frag_size = MMA_M * depth // 32
    comptime stride = 2 * depth
    comptime simd_size = 16 // size_of[output_type]()
    comptime swizzle = make_swizzle[
        num_rows=MMA_M // 2, row_size=depth, access_size=8
    ]()

    var tid = UInt32(thread_idx.x)
    if tid < UInt32(WARPGROUP_SIZE):
        return
    var warp_group_idx = tid // UInt32(WARPGROUP_SIZE)
    var local_warp_group_idx = warp_group_idx - 1
    var warp_y = (tid - UInt32(WARPGROUP_SIZE)) // 32

    var reg_buffer = stack_allocation[
        dtype=accum_type, address_space=.LOCAL, alignment=16
    ](row_major[1, o_frag_size]())
    comptime for j in range(o_frag_size):
        reg_buffer[0, j] = (
            Scalar[accum_type]((Int(tid) * 131 + j * 7) % 251) / 8 - 10
        )
    var legacy_reg = LayoutTensor[
        accum_type,
        Layout.row_major(1, o_frag_size),
        MutAnyOrigin,
        address_space=.LOCAL,
    ](reg_buffer.ptr.unsafe_origin_cast[MutAnyOrigin]())

    var smem_new = stack_allocation[
        dtype=output_type, address_space=.SHARED, alignment=128
    ](row_major[BM, depth]())
    var smem_old = stack_allocation[
        dtype=output_type, address_space=.SHARED, alignment=128
    ](row_major[BM, depth]())

    var new_tile = output_reg_to_smem[BM, depth, depth, swizzle, num_consumer](
        tid,
        local_warp_group_idx,
        warp_y,
        smem_new.ptr.unsafe_origin_cast[MutAnyOrigin](),
        reg_buffer,
    )
    var old_tile = legacy_output_reg_to_smem[
        BM, depth, depth, swizzle, num_consumer
    ](
        tid,
        local_warp_group_idx,
        warp_y,
        smem_old.ptr.unsafe_origin_cast[MutAnyOrigin](),
        legacy_reg,
    )
    barrier()

    var position = MHAPosition[BM, depth, depth, depth, 2, 1, False](
        0,
        0,
        0,
        valid_rows.cast[DType.uint32](),
        0,
        SeqInfo(valid_rows.cast[DType.uint32](), 0, WorkInfo(0, 0, 0, True)),
    )
    var new_gmem = position.q_out_gmem_tensor(new_out)
    copy_sram_to_dram[
        thread_layout=row_major[
            num_consumer_threads * simd_size // depth, depth // simd_size
        ](),
        swizzle=swizzle,
        masked=True,
    ](
        new_gmem.vectorize[1, simd_size](),
        new_tile.vectorize[1, simd_size](),
        Int(position.q_tile_num_rows()),
    )

    comptime legacy_gmem_layout = Layout(
        IntTuple(BM, depth), IntTuple(stride, 1)
    )
    comptime LegacyRT = RuntimeLayout[
        legacy_gmem_layout,
        element_type=DType.int32,
        linear_idx_type=DType.int32,
    ]
    var old_gmem = LayoutTensor[
        output_type,
        legacy_gmem_layout,
        MutAnyOrigin,
        layout_int_type=DType.int32,
        linear_idx_type=DType.int32,
        masked=True,
    ](
        old_out,
        LegacyRT(
            LegacyRT.ShapeType(Int(valid_rows), depth),
            LegacyRT.StrideType(stride, 1),
        ),
    )
    legacy_copy_sram_to_dram[
        thread_layout=Layout.row_major(
            num_consumer_threads * simd_size // depth, depth // simd_size
        ),
        swizzle=swizzle,
    ](
        old_gmem.vectorize[1, simd_size](),
        old_tile.vectorize[1, simd_size](),
    )


def run[
    output_type: DType, accum_type: DType, depth: Int, num_consumer: Int
](valid_rows: Int, ctx: DeviceContext) raises:
    comptime BM = 64 * num_consumer
    comptime count = BM * 2 * depth
    print("== parity", output_type, accum_type, depth, num_consumer, valid_rows)
    var new_dev = ctx.enqueue_create_buffer[output_type](count)
    var old_dev = ctx.enqueue_create_buffer[output_type](count)
    new_dev.enqueue_fill(Scalar[output_type](SENTINEL))
    old_dev.enqueue_fill(Scalar[output_type](SENTINEL))
    ctx.enqueue_function[kernel[output_type, accum_type, depth, num_consumer]](
        new_dev,
        old_dev,
        Int32(valid_rows),
        grid_dim=(1),
        block_dim=(128 * (1 + num_consumer)),
    )
    var new_host = ctx.enqueue_create_host_buffer[output_type](count)
    var old_host = ctx.enqueue_create_host_buffer[output_type](count)
    ctx.enqueue_copy(new_host, new_dev)
    ctx.enqueue_copy(old_host, old_dev)
    ctx.synchronize()
    var written = 0
    for i in range(count):
        assert_equal(new_host[i], old_host[i])
        if new_host[i] != Scalar[output_type](SENTINEL):
            written += 1
    assert_equal(written > 0 or valid_rows == 0, True)


def main() raises:
    with DeviceContext() as ctx:
        for rows in [0, 1, 17, 64, 100, 128]:
            run[.bfloat16, .float32, 128, 2](rows, ctx)
            run[.float16, .float32, 64, 2](rows, ctx)
            run[.bfloat16, .float32, 128, 1](rows, ctx)
            run[.float32, .float32, 128, 2](rows, ctx)
