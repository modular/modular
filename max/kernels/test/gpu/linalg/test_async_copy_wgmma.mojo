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

from sys import alignof

from gpu import barrier
from gpu.host import DeviceContext
from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.id import block_idx, thread_idx
from gpu.memory import (
    AddressSpace,
    async_copy_commit_group,
    async_copy_wait_group,
)
from layout import Layout, LayoutTensor
from layout._fillers import arange
from layout._utils import ManagedLayoutTensor
from layout.layout_tensor import (
    cp_async_k_major,
    cp_async_mn_major,
)
from layout.runtime_layout import RuntimeLayout
from layout.tensor_core_async import (
    TensorCoreAsync,
    tile_layout_mn_major,
    wgmma_c_layout,
)
from linalg import vendor_blas
from testing import assert_almost_equal

from utils.index import Index, IndexList
from utils.numerics import get_accum_type


fn cpasync_wgmma_kernel[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    block_tile_shape: IndexList[3],
    wgmma_shape: IndexList[3],
    transpose_b: Bool = False,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
](
    a: LayoutTensor[a_type, a_layout, MutableAnyOrigin],
    b: LayoutTensor[b_type, b_layout, MutableAnyOrigin],
    c: LayoutTensor[c_type, c_layout, MutableAnyOrigin],
    num_iters: UInt,
):
    """Test k_major @ mn_major with cp.async to simulate the 2nd matmul in mha.
    """
    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]

    alias a_smem_layout = Layout.row_major(BM, BK)
    var a_smem_tile = LayoutTensor[
        a_type,
        a_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ].stack_allocation()

    alias b_smem_layout = Layout.row_major(
        BN, BK
    ) if transpose_b else tile_layout_mn_major[b_type, BN, BK, b_swizzle]()
    var b_smem_tile = LayoutTensor[
        b_type,
        b_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ].stack_allocation()

    alias accum_type = get_accum_type[a_type]()
    wgmma_op = TensorCoreAsync[
        accum_type,
        a_type,
        b_type,
        wgmma_shape,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        transpose_b=transpose_b,
    ]()

    alias num_m_mmas = BM // wgmma_shape[0]
    alias num_n_mmas = BN // wgmma_shape[1]

    a_gmem_iter = a.tiled_iterator[BM, BK, axis=1](block_idx.y, 0)

    alias b_dim0 = BN if transpose_b else BK
    alias b_dim1 = BK if transpose_b else BN
    alias b_tile_axis = 1 if transpose_b else 0
    var b_tile_coords = (block_idx.x, UInt(0)) if transpose_b else (
        UInt(0),
        block_idx.y,
    )
    var b_gmem_iter = b.tiled_iterator[b_dim0, b_dim1, axis=b_tile_axis](
        b_tile_coords[0], b_tile_coords[1]
    )

    alias c_frag_size = wgmma_shape[0] * wgmma_shape[1] // 128
    var c_reg_tile = LayoutTensor[
        accum_type,
        Layout.row_major(num_m_mmas * num_n_mmas, c_frag_size),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ].stack_allocation()

    _ = c_reg_tile.fill(0.0)

    for i in range(num_iters):
        cp_async_k_major(a_smem_tile, a_gmem_iter[])

        if transpose_b:
            cp_async_k_major(b_smem_tile, b_gmem_iter[])
        else:
            cp_async_mn_major(b_smem_tile, b_gmem_iter[])

        async_copy_commit_group()
        async_copy_wait_group(0)

        barrier()

        wgmma_op.arrive()
        wgmma_op.wgmma(a_smem_tile, b_smem_tile, c_reg_tile)
        wgmma_op.commit_group()
        wgmma_op.wait_group()

        barrier()

        a_gmem_iter._incr()
        b_gmem_iter._incr()

    c_gmem_tile = c.tile[BM, BN](block_idx.y, block_idx.x)
    alias c_layouts = wgmma_c_layout[
        wgmma_shape[0], wgmma_shape[1], c_gmem_tile.layout
    ]()
    alias tv_tile_to_idx_const = c_layouts[2]
    alias tv_to_idx = tv_tile_to_idx_const[0]
    alias tile_to_idx = tv_tile_to_idx_const[1]
    alias t_to_idx_const = tv_to_idx[0]
    alias v_to_idx = tv_to_idx[1]
    t_to_idx = RuntimeLayout[t_to_idx_const]()
    t_idx = t_to_idx(thread_idx.x)

    c_reg_tile_vec2 = c_reg_tile.vectorize[1, 2]()
    alias T = c_reg_tile_vec2.element_type
    c_gmem_ptr = c_gmem_tile.ptr.offset(t_idx)

    @parameter
    for mma_id in range(tile_to_idx.size()):
        alias mma_idx = tile_to_idx(mma_id)

        @parameter
        for local_idx_v2 in range(c_reg_tile_vec2.layout[1].size()):
            alias local_idx = local_idx_v2 * 2
            alias v_idx = v_to_idx(local_idx)
            alias c_idx = v_idx + mma_idx
            casted_vec = c_reg_tile_vec2[mma_id, local_idx_v2].cast[c_type]()
            c_gmem_ptr.offset(c_idx).store[alignment = alignof[T]()](casted_vec)


def test_cpasync_wgmma[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    prob_shape: IndexList[3],
    block_tile_shape: IndexList[3],
    wgmma_shape: IndexList[3],
    transpose_b: Bool = True,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
](ctx: DeviceContext):
    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]

    alias WGMMA_M = wgmma_shape[0]
    alias WGMMA_N = wgmma_shape[1]
    alias WGMMA_K = wgmma_shape[2]

    print(
        "wgmma_ss_bf16_bf16_f32 block tile "
        + String(block_tile_shape)
        + " transb="
        + String(transpose_b)
        + "; inst shape "
        + String(wgmma_shape)
        + " A "
        + String(a_swizzle)
        + " B "
        + String(b_swizzle)
    )

    alias M = prob_shape[0]
    alias N = prob_shape[1]
    alias K = prob_shape[2]

    var a = ManagedLayoutTensor[
        a_type,
        Layout.row_major(M, K),
    ](ctx)
    arange(a.tensor[update=False]())

    alias b_layout = Layout.row_major(
        N, K
    ) if transpose_b else Layout.row_major(K, N)
    var b = ManagedLayoutTensor[b_type, b_layout](ctx)
    arange(b.tensor[update=False]())

    var c = ManagedLayoutTensor[
        c_type,
        Layout.row_major(M, N),
    ](ctx)

    var c_ref = ManagedLayoutTensor[
        c_type,
        Layout.row_major(M, N),
    ](ctx)

    alias kernel = cpasync_wgmma_kernel[
        a_type,
        b_type,
        c_type,
        __type_of(a).layout,
        __type_of(b).layout,
        Layout.row_major(M, N),
        block_tile_shape,
        wgmma_shape,
        transpose_b=transpose_b,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
    ]

    ctx.enqueue_function[kernel](
        a.device_tensor(),
        b.device_tensor(),
        c.device_tensor(),
        K // BK,
        grid_dim=(1, 1),
        block_dim=(128),
    )

    vendor_blas.matmul(
        ctx,
        c_ref.device_buffer(),
        a.device_buffer[update=False](),
        b.device_buffer[update=False](),
        c_row_major=True,
        transpose_b=transpose_b,
    )

    ctx.synchronize()

    c_host = c.tensor()
    c_host_ref = c_ref.tensor()

    for m in range(M):
        for n in range(N):
            assert_almost_equal(
                c_host[m, n], c_host_ref[m, n], atol=1e-3, rtol=1e-4
            )

    # print(c.tensor())
    _ = a^
    _ = b^
    _ = c^
    _ = c_ref^


def main():
    with DeviceContext() as ctx:
        test_cpasync_wgmma[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(64, 64, 64),
            Index(64, 64, 64),
            Index(64, 64, 16),
            a_swizzle = TensorMapSwizzle.SWIZZLE_128B,
            b_swizzle = TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=False,
        ](ctx)

        test_cpasync_wgmma[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(64, 128, 128),
            Index(64, 128, 128),
            Index(64, 128, 16),
            a_swizzle = TensorMapSwizzle.SWIZZLE_128B,
            b_swizzle = TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=False,
        ](ctx)

        test_cpasync_wgmma[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(64, 64, 64),
            Index(64, 64, 64),
            Index(64, 64, 16),
            a_swizzle = TensorMapSwizzle.SWIZZLE_128B,
            b_swizzle = TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=True,
        ](ctx)

        test_cpasync_wgmma[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(64, 128, 128),
            Index(64, 128, 128),
            Index(64, 128, 16),
            a_swizzle = TensorMapSwizzle.SWIZZLE_128B,
            b_swizzle = TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=True,
        ](ctx)

        test_cpasync_wgmma[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(128, 64, 128),
            Index(128, 64, 128),
            Index(64, 64, 16),
            a_swizzle = TensorMapSwizzle.SWIZZLE_128B,
            b_swizzle = TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=True,
        ](ctx)
