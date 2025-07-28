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
from gpu.cluster import cluster_mask_base
from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.id import block_id_in_cluster
from gpu.memory import AddressSpace

from layout import IntTuple, Layout, LayoutTensor
from layout.layout import coalesce
from layout.tensor_core_async import tile_to_descriptor

from utils.index import Index, IndexList, product

from sys import sizeof

from gpu.mma_sm100 import *
from gpu.tcgen05 import *


# TODO: Add methods to conveniently extract specific modes from a layout.
fn extract_first_2_modes[l: Layout]() -> Layout:
    constrained[l.rank() >= 2]()

    return Layout(
        IntTuple(l.shape[0].value(), l.shape[1].value()),
        IntTuple(l.stride[0].value(), l.stride[1].value()),
    )


@fieldwise_init
@register_passable("trivial")
struct Major:
    var val: Int

    alias K = Major(0)
    alias MN = Major(1)

    fn __eq__(self, rhs: Major) -> Bool:
        return self.val == rhs.val


# TODO: add create method to mma_operand trait and unify this with
# SM90 counter part by abstracting the return type.
fn _create_mma_desc[
    type: DType, //, canonical_layout: Layout, swizzle_mode: TensorMapSwizzle
](
    ptr: UnsafePointer[
        Scalar[type], address_space = AddressSpace.SHARED, *_, **_
    ]
) -> MMASmemDescriptor:
    # Extract the stride values from the canonical layout
    # The canonical layout is expected to have at least 2 dimensions
    alias stride01 = canonical_layout[0].stride[1].value()
    alias stride11 = canonical_layout[1].stride[1].value()
    alias SBO = stride01 * sizeof[type]()
    alias LBO = stride11 * sizeof[type]()

    # Create and return the MMA shared memory descriptor
    # This will be used by the SM100 MMA operations to access shared memory
    return MMASmemDescriptor.create[SBO, LBO, swizzle_mode](ptr)


@register_passable("trivial")
struct MmaOpSM100_SS[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    /,
    *,
    accum_type: DType = DType.float32,
    cta_group: Int = 1,
    cluster_shape: IndexList[3] = Index(1, 1, 1),
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    transpose_b: Bool = False,
](Defaultable):
    var idesc: UMMAInsDescriptor[Self._get_umma_kind[a_type]()]
    var mask: UInt16

    @always_inline
    fn __init__(out self):
        constrained[transpose_b, "MmaOpSM100 only supports transposed B"]()
        constrained[
            cta_group in (1, 2), "MmaOpSM100 only supports cta_group 1 or 2"
        ]()
        constrained[
            a_type == b_type,
            "a_type and b_type must be the same",
        ]()

        self.idesc = UMMAInsDescriptor[Self._get_umma_kind[a_type]()].create[
            accum_type,
            a_type,
            b_type,
            Index[dtype = DType.uint32](mma_shape[0], mma_shape[1]),
            transpose_b=transpose_b,
        ]()

        self.mask = 0

        # Here we compute the mask inside mma object to hide the complexity.
        # We may get better asm if the mask if computed outside from TMA masks,
        # and passed to `commit`, need to verify.
        @parameter
        if product(cluster_shape) > 1:
            alias dim0_mask = cluster_mask_base[cluster_shape, 0]()
            alias dim1_mask = cluster_mask_base[cluster_shape, 1]()

            # The mask includes ctas on the same row and column in the cluster
            # Example mask for cta (0, 1) is cluster (4,4)
            #             x x x x
            #             o x o o
            #             o x o o
            #             o x o o
            self.mask = (
                dim0_mask << (block_id_in_cluster.y * cluster_shape[0])
            ) | (dim1_mask << block_id_in_cluster.x)

            # Include peer cta's row
            # Example mask for cta (0, 1) is cluster (4,4)
            #             x x x x
            #             x x x x
            #             o x o o
            #             o x o o
            @parameter
            if cta_group == 2:
                self.mask |= dim1_mask << (block_id_in_cluster.x ^ 1)

    @always_inline
    fn mma(
        self,
        a: LayoutTensor[address_space = AddressSpace.SHARED, *_, **_],
        b: LayoutTensor[address_space = AddressSpace.SHARED, *_, **_],
        c_tmem: UInt32,
        init_c: Bool,
    ):
        """MMA input tiles.

        The layout assumes that coalesce(A) has shape (bm, sw_k, num_sw_k), we currently
        assumes bm = mma_m. In future, we can tile it to (mma_m, sw_k, num_sw_k, num_mma_m)
        The same logic applies to matrix B.
        """

        # Coalesce a and b
        # A and B are coalesced to rank-2 if it's only one tile or rank-3 if it has
        # multiple canonical layouts in K dim.
        alias a_coalesced_layout = coalesce(a.layout)
        alias b_coalesced_layout = coalesce(b.layout)

        # Canonical layouts are tiled by core matrices.
        alias a_canonical_layout = tile_to_descriptor[
            a.dtype, extract_first_2_modes[a_coalesced_layout]()
        ]()
        alias b_canonical_layout = tile_to_descriptor[
            b.dtype, extract_first_2_modes[b_coalesced_layout]()
        ]()

        var a_desc = _create_mma_desc[a_canonical_layout, a_swizzle](a.ptr)
        var b_desc = _create_mma_desc[b_canonical_layout, b_swizzle](b.ptr)

        @parameter
        for k in range(0, block_tile_shape[2], mma_shape[2]):
            alias a_offset = a.layout(IntTuple(0, k)) * sizeof[a_type]()
            alias b_offset = b.layout(IntTuple(0, k)) * sizeof[b_type]()

            var c_scale: UInt32 = 0 if (init_c and k == 0) else 1

            mma[cta_group](
                a_desc + a_offset,
                b_desc + b_offset,
                c_tmem,
                self.idesc,
                c_scale=c_scale,
            )

    @always_inline
    fn commit(
        self,
        ptr_mbar: UnsafePointer[address_space = AddressSpace.SHARED, *_, **_],
    ):
        @parameter
        if product(cluster_shape) == 1:
            mma_arrive(ptr_mbar)
        else:
            mma_arrive_multicast[cta_group](ptr_mbar, self.mask)

    @always_inline
    fn wait(self):
        pass

    @staticmethod
    fn _get_umma_kind[dtype: DType]() -> UMMAKind:
        @parameter
        if dtype == DType.float32:
            return UMMAKind.KIND_TF32
        elif dtype in (DType.float16, DType.bfloat16):
            return UMMAKind.KIND_F16
        elif dtype in (DType.float8_e4m3fn, DType.float8_e5m2):
            return UMMAKind.KIND_F8F6F4
        else:
            constrained[
                False,
                "Unsupported/not implemented operand type for UMMA: ",
                String(dtype),
            ]()

        return UMMAKind(-1)
