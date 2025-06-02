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

import linalg.vendor_blas
from buffer import DimList, NDBuffer
from gpu import barrier
from gpu.host import DeviceContext
from gpu.host._compile import _compile_code_asm, _get_gpu_target
from gpu.id import thread_idx
from gpu.memory import AddressSpace
from gpu.mma import (
    wgmma_async,
    wgmma_commit_group_sync,
    wgmma_fence_aligned,
    wgmma_wait_group_sync,
)
from layout import Layout, LayoutTensor
from layout._fillers import arange
from layout._utils import ManagedLayoutTensor
from layout.tensor_core_async import (
    _lhs_descriptor,
    _rhs_descriptor,
    tile_layout_k_major,
)
from testing import assert_almost_equal

from utils import StaticTuple
from utils.index import Index


fn wgmma_kernel_ss[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    WMMA_M: Int,
    WMMA_N: Int,
    WMMA_K: Int,
    a_smem_layout: Layout,
    b_smem_layout: Layout,
    transpose_b: Bool = False,
](
    a_gmem: LayoutTensor[a_type, a_layout, MutableAnyOrigin],
    b_gmem: LayoutTensor[b_type, b_layout, MutableAnyOrigin],
    c_gmem: LayoutTensor[c_type, c_layout, MutableAnyOrigin],
):
    var a_smem_tile = LayoutTensor[
        DType.bfloat16,
        a_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var b_smem_tile = LayoutTensor[
        DType.bfloat16,
        b_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    alias num_output_regs = WMMA_M * WMMA_N // 128
    var c_reg = StaticTuple[Scalar[DType.float32], num_output_regs](0)

    alias M = a_layout.shape[0].value()
    alias K = a_layout.shape[1].value()
    alias N = c_layout.shape[1].value()

    alias b_tile_dim0 = N if transpose_b else WMMA_K
    alias b_tile_dim1 = WMMA_K if transpose_b else N

    for k_i in range(K // WMMA_K):
        var a_gmem_tile = a_gmem.tile[M, WMMA_K](0, k_i)

        var b_tile_coord0 = 0 if transpose_b else k_i
        var b_tile_coord1 = k_i if transpose_b else 0
        var b_gmem_tile = b_gmem.tile[b_tile_dim0, b_tile_dim1](
            b_tile_coord0, b_tile_coord1
        )

        if thread_idx.x == 0:
            a_smem_tile.copy_from(a_gmem_tile)
            b_smem_tile.copy_from(b_gmem_tile)

        barrier()

        var mat_a_desc = _lhs_descriptor(a_smem_tile)
        var mat_b_desc = _rhs_descriptor[transpose_b](b_smem_tile)

        wgmma_fence_aligned()

        c_reg = wgmma_async[
            WMMA_M,
            WMMA_N,
            WMMA_K,
            a_type = DType.bfloat16,
            b_type = DType.bfloat16,
        ](mat_a_desc, mat_b_desc, c_reg)
        wgmma_commit_group_sync()
        wgmma_wait_group_sync()

    var warp_id = thread_idx.x // 32
    var lane_id = thread_idx.x % 32

    var th_local_res = (
        c_gmem.tile[16, WMMA_N](warp_id, 0)
        .vectorize[1, 2]()
        .distribute[Layout.row_major(8, 4)](lane_id)
    )

    for i in range(num_output_regs):
        th_local_res[(i // 2) % 2, i // 4][i % 2] = c_reg[i].cast[
            c_gmem.dtype
        ]()


fn wgmma_bf16_bf16_f32[
    M: Int, N: Int, K: Int, transpose_b: Bool = False, a_reg: Bool = False
](ctx: DeviceContext) raises:
    print(
        "== wgmma_bf16_bf16_f32_64xNx16(N, r/s) => ",
        N,
        ", r" if a_reg else ", s",
        sep="",
    )

    var a = ManagedLayoutTensor[DType.bfloat16, Layout.row_major(M, K)](ctx)
    arange(a.tensor[update=False]())

    var b = ManagedLayoutTensor[DType.bfloat16, Layout.row_major(N, K)](ctx)
    arange(b.tensor[update=False]())

    var c = ManagedLayoutTensor[DType.bfloat16, Layout.row_major(M, N)](ctx)
    var c_ref = ManagedLayoutTensor[DType.bfloat16, Layout.row_major(M, N)](ctx)

    alias a_smem_layout = tile_layout_k_major[DType.bfloat16, BM=M, BK=16]()

    alias b_smem_layout = tile_layout_k_major[DType.bfloat16, BM=N, BK=16]()

    alias kernel = wgmma_kernel_ss[
        DType.bfloat16,
        DType.bfloat16,
        DType.bfloat16,
        Layout.row_major(M, K),
        Layout.row_major(N, K),
        Layout.row_major(M, N),
        M,
        N,
        K,
        a_smem_layout,
        b_smem_layout,
        transpose_b=transpose_b,
    ]

    ctx.enqueue_function[kernel](
        a.device_tensor(),
        b.device_tensor(),
        c.device_tensor(),
        grid_dim=(1, 1),
        block_dim=(128),
    )
    ctx.synchronize()

    var a_buf = NDBuffer[DType.bfloat16, 2, _, DimList(M, K)](
        a.device_tensor().ptr
    )
    var b_buf = NDBuffer[DType.bfloat16, 2, _, DimList(N, K)](
        b.device_tensor().ptr
    )
    var c_ref_buf = NDBuffer[DType.bfloat16, 2, _, DimList(M, N)](
        c_ref.device_tensor().ptr
    )

    vendor_blas.matmul(
        ctx,
        c_ref_buf,
        a_buf,
        b_buf,
        c_row_major=True,
        transpose_b=transpose_b,
    )

    for m in range(M):
        for n in range(N):
            assert_almost_equal(
                c_ref.tensor()[m, n], c.tensor()[m, n], atol=1e-3, rtol=1e-3
            )

    _ = a^
    _ = b^
    _ = c^
    _ = c_ref^


def main():
    with DeviceContext() as ctx:

        @parameter
        for n in range(8, 264, 8):
            wgmma_bf16_bf16_f32[64, n, 16, True](ctx)
