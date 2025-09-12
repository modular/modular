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
from math import ceildiv
from sys import simd_width_of, size_of
from gpu import thread_idx
from gpu.memory import AddressSpace, async_copy
from gpu.sync import async_copy_arrive
from layout import Layout, LayoutTensor
from layout.layout_tensor import LayoutTensorIter
from layout.swizzle import make_swizzle
from layout.tma_async import PipelineState, SharedMemBarrier, TMATensorTile
from memory.unsafe_pointer import UnsafePointer
from gpu.host._nvidia_cuda import TensorMapSwizzle
from utils.index import IndexList
from utils.static_tuple import StaticTuple

# TODO: We're defining this multiple times in different files.
# Move it to a common place.
alias WARPGROUP_SIZE = 128


# ===----------------------------------------------------------------------=== #
# Load A and B from using TMA
# ===----------------------------------------------------------------------=== #


@always_inline
fn async_load_AB[
    a_type: DType,
    b_type: DType,
    a_tile_layout: Layout,
    b_tile_layout: Layout,
    a_smem_layout: Layout,
    b_smem_layout: Layout,
    a_desc_layout: Layout,
    b_desc_layout: Layout,
    pipeline_stages: Int,
    /,
    *,
    num_k_iters: Int,
    block_tile_shape: IndexList[3],
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1, 1, 1),
    partitioned_multicast: Bool = False,
](
    a_tma_op: TMATensorTile[a_type, a_tile_layout, a_desc_layout],
    b_tma_op: TMATensorTile[b_type, b_tile_layout, b_desc_layout],
    a_smem_iter: LayoutTensorIter[
        a_type,
        a_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128, **_,
    ],
    b_smem_iter: LayoutTensorIter[
        b_type,
        b_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128, **_,
    ],
    m_coord: UInt,
    n_coord: UInt,
    k_coord: UInt,
    rank_n: UInt,
    rank_m: UInt,
    mut write_pipeline_states: PipelineState[pipeline_stages],
    empty_mbar: UnsafePointer[
        SharedMemBarrier, address_space = AddressSpace.SHARED
    ],
    full_mbar: UnsafePointer[
        SharedMemBarrier, address_space = AddressSpace.SHARED
    ],
):
    alias a_expected_bytes = a_smem_layout.size() * size_of[a_type]()
    alias b_expected_bytes = b_smem_layout.size() * size_of[b_type]()
    alias expected_bytes = a_expected_bytes + b_expected_bytes

    alias a_tma_load_size = a_desc_layout.size()
    alias b_tma_load_size = b_desc_layout.size()
    alias a_tma_rows = a_desc_layout.shape[0].value()
    alias b_tma_rows = b_desc_layout.shape[0].value()

    alias CLUSTER_N = UInt(cluster_shape[0])
    alias CLUSTER_M = UInt(cluster_shape[1])

    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]

    var multicast_column_mask = 0

    @parameter
    for i in range(CLUSTER_M):
        multicast_column_mask |= Int(1 << (i * CLUSTER_N))

    var multicast_row_mask = ((1 << CLUSTER_N) - 1) << (rank_m * CLUSTER_N)

    alias num_full_k_iters = ceildiv(num_k_iters, pipeline_stages)
    alias num_remaining_k_iters = num_k_iters % pipeline_stages

    # `num_pipeline_stages_to_unroll` determines how many pipeline stages should be unroll in the producer loop;
    # if num_k_iters % pipeline_stages != 0 then for the last loop, we only unroll (num_k_iters % pipeline_stages) pipeline stages
    @always_inline
    @parameter
    fn producer_loop[
        num_pipeline_stages_to_unroll: Int,
    ](k_iter: Int):
        @parameter
        for j in range(num_pipeline_stages_to_unroll):
            var write_idx = write_pipeline_states.index()

            empty_mbar[write_idx].wait(write_pipeline_states.phase())

            var a_smem_tile = a_smem_iter.next(write_idx)[]
            var b_smem_tile = b_smem_iter.next(write_idx)[]

            full_mbar[write_idx].expect_bytes(expected_bytes)

            @parameter
            if CLUSTER_N > 1:

                @parameter
                if partitioned_multicast:
                    var a_gmem_slice_coord = m_coord + Int(rank_n) * a_tma_rows
                    var a_smem_slice = __type_of(a_smem_tile)(
                        a_smem_tile.ptr + rank_n * a_tma_load_size
                    )

                    a_tma_op.async_multicast_load(
                        a_smem_slice,
                        full_mbar[write_idx],
                        (
                            UInt(k_coord + k_iter * pipeline_stages + j)
                            * UInt(BK),
                            UInt(a_gmem_slice_coord),
                        ),
                        UInt16(multicast_row_mask),
                    )

                else:
                    if rank_n == 0:
                        a_tma_op.async_multicast_load(
                            a_smem_tile,
                            full_mbar[write_idx],
                            (
                                UInt(k_coord + k_iter * pipeline_stages + j)
                                * UInt(BK),
                                m_coord,
                            ),
                            UInt16(multicast_row_mask),
                        )

            else:
                a_tma_op.async_copy(
                    a_smem_tile,
                    full_mbar[write_idx],
                    (
                        UInt(k_coord + k_iter * pipeline_stages + j) * UInt(BK),
                        m_coord,
                    ),
                )

            @parameter
            if CLUSTER_M > 1:

                @parameter
                if partitioned_multicast:
                    var b_gmem_slice_coord = n_coord + Int(rank_m) * b_tma_rows
                    var b_smem_slice = __type_of(b_smem_tile)(
                        b_smem_tile.ptr + rank_m * b_tma_load_size
                    )

                    b_tma_op.async_multicast_load(
                        b_smem_slice,
                        full_mbar[write_idx],
                        (
                            UInt(k_coord + k_iter * pipeline_stages + j)
                            * UInt(BK),
                            UInt(b_gmem_slice_coord),
                        ),
                        UInt16(multicast_column_mask << rank_n),
                    )

                else:
                    if rank_m == 0:
                        b_tma_op.async_multicast_load(
                            b_smem_tile,
                            full_mbar[write_idx],
                            (
                                UInt(k_coord + k_iter * pipeline_stages + j)
                                * UInt(BK),
                                n_coord,
                            ),
                            UInt16(multicast_column_mask << rank_n),
                        )

            else:
                b_tma_op.async_copy(
                    b_smem_tile,
                    full_mbar[write_idx],
                    (
                        UInt(k_coord + k_iter * pipeline_stages + j) * UInt(BK),
                        n_coord,
                    ),
                )

            write_pipeline_states.step()

    @parameter
    if num_remaining_k_iters == 0:
        for k_iter in range(num_full_k_iters):
            producer_loop[pipeline_stages](k_iter)
    else:
        for k_iter in range(num_full_k_iters - 1):
            producer_loop[pipeline_stages](k_iter)
        producer_loop[num_remaining_k_iters](num_full_k_iters - 1)


# ===----------------------------------------------------------------------=== #
# Load A and B from using cp.async
# ===----------------------------------------------------------------------=== #


@always_inline
fn async_load_AB[
    a_type: DType,
    b_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    a_smem_layout: Layout,
    b_smem_layout: Layout,
    pipeline_stages: Int, //,
    /,
    *,
    swizzle_mode: TensorMapSwizzle,
    cp_size: Int,
    num_k_iters: Int,
    block_tile_shape: IndexList[3],
](
    a: LayoutTensor[
        a_type,
        a_layout,
        MutableAnyOrigin,
    ],
    b: LayoutTensor[
        b_type,
        b_layout,
        MutableAnyOrigin,
    ],
    block_idx_m: UInt,
    block_idx_n: UInt,
    a_smem_iter: LayoutTensorIter[
        a_type,
        a_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128, **_,
    ],
    b_smem_iter: LayoutTensorIter[
        b_type,
        b_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128, **_,
    ],
    mut write_pipeline_states: PipelineState[pipeline_stages],
    empty_mbar: UnsafePointer[
        SharedMemBarrier, address_space = AddressSpace.SHARED
    ],
    full_mbar: UnsafePointer[
        SharedMemBarrier, address_space = AddressSpace.SHARED
    ],
):
    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]

    alias num_full_k_iters = ceildiv(num_k_iters, pipeline_stages)
    alias num_remaining_k_iters = num_k_iters % pipeline_stages

    alias num_threads_per_row = BK // cp_size
    alias thread_layout = Layout.row_major(
        WARPGROUP_SIZE // num_threads_per_row, num_threads_per_row
    )

    @always_inline
    @parameter
    fn producer_loop[
        num_pipeline_stages_to_unroll: Int,
    ](k_iter: Int):
        @parameter
        for j in range(num_pipeline_stages_to_unroll):
            var write_idx = write_pipeline_states.index()

            empty_mbar[write_idx].wait(write_pipeline_states.phase())

            var a_smem_tile = a_smem_iter.next(write_idx)[].vectorize[
                1, cp_size
            ]()
            var b_smem_tile = b_smem_iter.next(write_idx)[].vectorize[
                1, cp_size
            ]()

            var a_gmem_tile = a.tile[BM, BK](
                Int(block_idx_m), k_iter * pipeline_stages + j
            ).vectorize[1, cp_size]()
            var b_gmem_tile = b.tile[BN, BK](
                Int(block_idx_n), k_iter * pipeline_stages + j
            ).vectorize[1, cp_size]()

            async_copy_with_bound_check[thread_layout, swizzle_mode](
                a_gmem_tile, a_smem_tile
            )

            async_copy_with_bound_check[thread_layout, swizzle_mode](
                b_gmem_tile, b_smem_tile
            )

            async_copy_arrive(full_mbar[write_idx].unsafe_ptr())
            _ = full_mbar[write_idx].arrive()
            write_pipeline_states.step()

    @parameter
    if num_remaining_k_iters == 0:
        for k_iter in range(num_full_k_iters):
            producer_loop[pipeline_stages](k_iter)
    else:
        for k_iter in range(num_full_k_iters - 1):
            producer_loop[pipeline_stages](k_iter)
        producer_loop[num_remaining_k_iters](num_full_k_iters - 1)


@always_inline
fn async_copy_with_bound_check[
    dtype: DType, //,
    thread_layout: Layout,
    swizzle_mode: TensorMapSwizzle,
](
    src: LayoutTensor[
        dtype,
        _,
        MutableAnyOrigin,
        address_space = AddressSpace.GENERIC,
        *_, **_,
    ],
    dst: LayoutTensor[
        dtype, _, MutableAnyOrigin, address_space = AddressSpace.SHARED, *_, **_
    ],
):
    constrained[src.layout.rank() == 2, "Global memory tile must be rank 2."]()

    alias src_shape1 = src.layout.shape[1].value()
    alias swizzle_bytes = swizzle_mode.bytes()
    constrained[
        src_shape1 * src.element_size * size_of[src.dtype]() == swizzle_bytes,
        String(
            "Global memory tile shape-1 ",
            src_shape1 * src.element_size,
            "must match swizzle bytes.",
            swizzle_bytes,
        ),
    ]()

    var src_frag = src.distribute[thread_layout](thread_idx.x)
    var dst_frag = dst.distribute[thread_layout](thread_idx.x)

    alias src_stride0 = src.layout.stride[0].value()
    var src_bound0 = Int32(src.runtime_layout.shape.value[0])
    var src_bound1 = Int32(src.runtime_layout.shape.value[1]) * dst.element_size

    var dst_frag_offset = dst_frag.distance(dst.ptr)
    alias dst_stride0 = dst.layout.stride[0].value()
    var dst_frag_base_coord0 = Int32(dst_frag_offset // dst_stride0)
    var dst_frag_base_coord1 = Int32(dst_frag_offset % dst_stride0)
    alias swizzle = make_swizzle[
        8,
        Int(swizzle_bytes // size_of[dst.dtype]()),
        Int(simd_width_of[dst.dtype]()),
    ]()

    alias num_vecs = dst_frag.layout.size()

    @parameter
    for i in range(num_vecs):
        alias dst_idx = dst_frag.layout(i)
        alias dst_idx_base = dst_idx % swizzle.size()
        alias dst_idx_diff = dst_idx - dst_idx_base
        var dst_swizzled_idx = Int32(
            swizzle(dst_frag_offset + dst_idx_base) + dst_idx_diff
        )
        var dst_ptr = dst.ptr + Int(dst_swizzled_idx)

        # TODO: we should be able to use idx2crd for this.
        alias dst_shifted_coord0 = dst_idx // dst_stride0
        alias dst_shifted_coord1 = dst_idx % dst_stride0
        var dst_coord0 = dst_shifted_coord0 + dst_frag_base_coord0
        var dst_coord1 = dst_shifted_coord1 + dst_frag_base_coord1

        alias cp_size = dst.element_size * size_of[dst.dtype]()

        var src_ptr = (
            src.ptr.address_space_cast[AddressSpace.GLOBAL]()
            + dst_coord1
            + dst_coord0 * src_stride0
        )

        if dst_coord0 < src_bound0 and dst_coord1 < src_bound1:
            async_copy[
                cp_size,
                bypass_L1_16B=False,
                fill = Scalar[dst.dtype](0),
            ](src_ptr, dst_ptr, src_size=cp_size)
        else:
            # Zero-fill the OOB address
            async_copy[
                cp_size, bypass_L1_16B=False, fill = Scalar[dst.dtype](0)
            ](src_ptr, dst_ptr, src_size=0)
