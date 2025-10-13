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
from layout.tma_async import TMATensorTile
from layout.layout_tensor import LayoutTensor
from gpu.memory import (
    AddressSpace,
    async_copy,
)
from ....structuring import SharedMemBarrier
from layout.swizzle import make_swizzle
from gpu.id import thread_idx
from sys import simd_width_of
from gpu.host._nvidia_cuda import TensorMapSwizzle


@register_passable("trivial")
struct ScatterGather:
    @staticmethod
    @always_inline
    fn load_tile[
        dtype: DType,
        tile_layout: Layout,
        desc_layout: Layout,
        dst_layout: Layout, //,
        cluster_size: Int,
        use_partitioned_multicast: Bool,
    ](
        tma_op: TMATensorTile[dtype, tile_layout, desc_layout],
        dst: LayoutTensor[
            dtype,
            dst_layout,
            address_space = AddressSpace.SHARED,
            alignment=128,
            *_, **_,
        ],
        ref [AddressSpace.SHARED]mem_barrier: SharedMemBarrier,
        rank: UInt,
        coords: Tuple[UInt, UInt],
        multicast_mask: UInt16,
    ):
        alias tma_load_size = desc_layout.size()
        alias tma_rows = desc_layout.shape[0].value()

        @parameter
        if cluster_size > 1:

            @parameter
            if use_partitioned_multicast:
                tma_op.async_multicast_load_partitioned[
                    tma_rows, tma_load_size
                ](
                    dst,
                    mem_barrier,
                    rank,
                    coords,
                    multicast_mask,
                )

            else:
                if rank == 0:
                    tma_op.async_multicast_load(
                        dst,
                        mem_barrier,
                        coords,
                        multicast_mask,
                    )

        else:
            tma_op.async_copy(
                dst,
                mem_barrier,
                coords,
            )

    @staticmethod
    @always_inline("nodebug")
    fn load_tile[
        dtype: DType,
        src_layout: Layout,
        dst_layout: Layout, //,
        thread_layout: Layout,
        swizzle_mode: TensorMapSwizzle,
        vector_size: Int,
    ](
        src: LayoutTensor[
            dtype,
            src_layout,
            MutableAnyOrigin,
            address_space = AddressSpace.GENERIC,
            *_, **_,
        ],
        dst: LayoutTensor[
            dtype,
            dst_layout,
            MutableAnyOrigin,
            address_space = AddressSpace.SHARED,
            *_, **_,
        ],
        tile_idx_m: Int,
        tile_idx_n: Int,
    ):
        alias BM = dst_layout.shape[0].value()
        alias BN = dst_layout.shape[1].value()
        var a_gmem_tile = src.tile[BM, BN](
            tile_idx_m,
            tile_idx_n,
        ).vectorize[1, vector_size]()
        Self.async_copy_with_bound_check[
            thread_layout,
            swizzle_mode,
        ](a_gmem_tile, dst.vectorize[1, vector_size]())

    @staticmethod
    @always_inline
    fn async_copy_with_bound_check[
        dtype: DType,
        src_layout: Layout,
        dst_layout: Layout, //,
        thread_layout: Layout,
        swizzle_mode: TensorMapSwizzle,
    ](
        src: LayoutTensor[
            dtype,
            src_layout,
            MutableAnyOrigin,
            address_space = AddressSpace.GENERIC,
            *_, **_,
        ],
        dst: LayoutTensor[
            dtype,
            dst_layout,
            MutableAnyOrigin,
            address_space = AddressSpace.SHARED,
            *_, **_,
        ],
    ):
        """Helper function for cp.async with bound checking."""
        constrained[
            src.layout.rank() == 2, "Global memory tile must be rank 2."
        ]()

        constrained[
            src_layout.shape == dst_layout.shape,
            "Global memory tile must match source layout: "
            + String(src_layout)
            + " != "
            + String(dst_layout),
        ]()

        alias src_shape1 = src.layout.shape[1].value()
        alias swizzle_bytes = swizzle_mode.bytes()
        constrained[
            src_shape1 * src.element_size * size_of[src.dtype]()
            == swizzle_bytes,
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
        var src_bound1 = (
            Int32(src.runtime_layout.shape.value[1]) * dst.element_size
        )

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

            alias size_bytes = dst.element_size * size_of[dst.dtype]()

            var src_ptr = (
                src.ptr.address_space_cast[AddressSpace.GLOBAL]()
                + dst_coord1
                + dst_coord0 * src_stride0
            )

            if dst_coord0 < src_bound0 and dst_coord1 < src_bound1:
                async_copy[
                    size_bytes,
                    bypass_L1_16B=False,
                    fill = Scalar[dst.dtype](0),
                ](src_ptr, dst_ptr, src_size=size_bytes)
            else:
                # Zero-fill the OOB address
                async_copy[
                    size_bytes, bypass_L1_16B=False, fill = Scalar[dst.dtype](0)
                ](src_ptr, dst_ptr, src_size=0)
