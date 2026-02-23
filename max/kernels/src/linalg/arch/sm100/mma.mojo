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
from sys import size_of
from math import align_up

from gpu.primitives.cluster import cluster_mask_base
from gpu.host._tensormap import SwizzleMode
from gpu.memory import AddressSpace
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu import block_id_in_cluster
from gpu.compute.arch.mma_nvidia_sm100 import *
from gpu.compute.arch.tcgen05 import *
from gpu.compute.arch.mma_nvidia_sm100 import MMASmemDescriptorPair
from layout import IntTuple, Layout, LayoutTensor
from layout._coord import coord_to_int_tuple
from layout._tile_tensor import TileTensor
from layout.layout import coalesce
from layout.tensor_core_async import (
    tile_to_descriptor,
    tile_layout_k_major,
    tile_layout_mn_major,
)

from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from utils.index import Index, IndexList, product
from linalg.fp4_utils import SF_MN_GROUP_SIZE, SF_ATOM_M, SF_ATOM_K


# TODO: Add methods to conveniently extract specific modes from a layout.
fn extract_first_2_modes[l: Layout]() -> Layout:
    comptime assert l.rank() >= 2

    return Layout(
        IntTuple(l.shape[0].value(), l.shape[1].value()),
        IntTuple(l.stride[0].value(), l.stride[1].value()),
    )


@fieldwise_init("implicit")
struct Major(TrivialRegisterPassable):
    var val: Int

    comptime K = Major(0)
    comptime MN = Major(1)

    fn __eq__(self, rhs: Major) -> Bool:
        return self.val == rhs.val


fn max_contiguous_tile_shape[
    rank: Int,
    //,
    dtype: DType,
    tile_shape: IndexList[rank],
    /,
    *,
    major: Major = Major.K,
    swizzle_mode: SwizzleMode = SwizzleMode.NONE,
]() -> IntTuple:
    """Returns the maximum shape of a tile that's contiguous in memory for mma op. This is used to create TMA descriptor.
    """

    comptime assert rank == 2, "Only 2D tensors are supported!"

    comptime if major == Major.K:
        # Tile shape is (MN, K), max K is based on swizzle.
        return IntTuple(tile_shape[0], swizzle_mode.bytes() // size_of[dtype]())
    elif major == Major.MN:
        # Tile shape is (K, MN), max MN is based on swizzle, max K is 8 based on
        # canonical layout.
        # The following are rare in practice but worth checking.
        # TODO: this may not work for swizzle.NONE, need to double-check
        # TODO: for MN = swizzle_bytes // sizeof,  tile_shape[0] may be the max
        return IntTuple(8, swizzle_mode.bytes() // size_of[dtype]())
    else:
        comptime assert False, "Invalid major"
        return IntTuple()


# TODO: add create method to mma_operand trait and unify this with
# SM90 counter part by abstracting the return dtype.
fn _create_mma_desc[
    dtype: DType, //, canonical_layout: Layout, swizzle_mode: TensorMapSwizzle
](
    ptr: UnsafePointer[Scalar[dtype], address_space = AddressSpace.SHARED, ...]
) -> MMASmemDescriptor:
    # Extract the stride values from the canonical layout
    # The canonical layout is expected to have at least 2 dimensions
    comptime stride01 = canonical_layout[0].stride[1].value()
    comptime stride11 = canonical_layout[1].stride[1].value()
    comptime SBO = stride01 * size_of[dtype]()
    comptime LBO = stride11 * size_of[dtype]()

    # Create and return the MMA shared memory descriptor
    # This will be used by the SM100 MMA operations to access shared memory
    return MMASmemDescriptor.create[SBO, LBO, swizzle_mode](ptr)


fn _create_mma_desc_pair[
    dtype: DType, //, canonical_layout: Layout, swizzle_mode: TensorMapSwizzle
](
    ptr: UnsafePointer[Scalar[dtype], address_space = AddressSpace.SHARED, ...]
) -> MMASmemDescriptorPair:
    # Extract the stride values from the canonical layout
    # The canonical layout is expected to have at least 2 dimensions
    comptime stride01 = canonical_layout[0].stride[1].value()
    comptime stride11 = canonical_layout[1].stride[1].value()
    comptime SBO = stride01 * size_of[dtype]()
    comptime LBO = stride11 * size_of[dtype]()

    # Create and return the MMA shared memory descriptor
    # This will be used by the SM100 MMA operations to access shared memory
    return MMASmemDescriptorPair.create[SBO, LBO, swizzle_mode](ptr)


@always_inline
fn smem_descriptor[
    dtype: DType,
    //,
    *,
    BMN: Int,
    BK: Int,
    swizzle_mode: TensorMapSwizzle,
    is_k_major: Bool,
](
    ptr: UnsafePointer[Scalar[dtype], address_space = AddressSpace.SHARED, ...]
) -> MMASmemDescriptorPair:
    comptime smem_layout = tile_layout_k_major[
        dtype, BMN, BK, swizzle_mode
    ]() if is_k_major else tile_layout_mn_major[dtype, BMN, BK, swizzle_mode]()
    comptime canonical_layout = tile_to_descriptor[
        dtype, smem_layout, is_k_major=is_k_major
    ]()
    comptime cl = canonical_layout if is_k_major else canonical_layout.transpose()
    return _create_mma_desc_pair[
        canonical_layout=cl, swizzle_mode=swizzle_mode
    ](ptr)


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
](Defaultable, TrivialRegisterPassable):
    var idesc: UMMAInsDescriptor[Self._get_umma_kind[Self.a_type]()]
    var mask: UInt16

    @always_inline
    fn __init__(out self):
        comptime assert (
            Self.transpose_b
        ), "MmaOpSM100 only supports transposed B"
        comptime assert Self.cta_group in (
            1,
            2,
        ), "MmaOpSM100 only supports cta_group 1 or 2"
        comptime assert (
            Self.a_type == Self.b_type
        ), "a_type and b_type must be the same"

        self.idesc = UMMAInsDescriptor[
            Self._get_umma_kind[Self.a_type]()
        ].create[
            Self.accum_type,
            Self.a_type,
            Self.b_type,
            Index[dtype = DType.uint32](Self.mma_shape[0], Self.mma_shape[1]),
            transpose_b = Self.transpose_b,
        ]()

        self.mask = 0

        # Here we compute the mask inside mma object to hide the complexity.
        # We may get better asm if the mask if computed outside from TMA masks,
        # and passed to `commit`, need to verify.
        comptime if product(Self.cluster_shape) > 1:
            comptime dim0_mask = cluster_mask_base[Self.cluster_shape, 0]()
            comptime dim1_mask = cluster_mask_base[Self.cluster_shape, 1]()

            # The mask includes ctas on the same row and column in the cluster
            # Example mask for cta (0, 1) is cluster (4,4)
            #             x x x x
            #             o x o o
            #             o x o o
            #             o x o o
            self.mask = (
                dim0_mask
                << UInt16((block_id_in_cluster.y * UInt(Self.cluster_shape[0])))
            ) | (dim1_mask << UInt16(block_id_in_cluster.x))

            # Include peer cta's row
            # Example mask for cta (0, 1) is cluster (4,4)
            #             x x x x
            #             x x x x
            #             o x o o
            #             o x o o
            comptime if Self.cta_group == 2:
                self.mask |= dim1_mask << UInt16(block_id_in_cluster.x ^ 1)

    @always_inline
    fn mma(
        self,
        a: LayoutTensor[address_space = AddressSpace.SHARED, ...],
        b: LayoutTensor[address_space = AddressSpace.SHARED, ...],
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
        comptime a_coalesced_layout = coalesce(a.layout)
        comptime b_coalesced_layout = coalesce(b.layout)

        # Canonical layouts are tiled by core matrices.
        comptime a_canonical_layout = tile_to_descriptor[
            a.dtype, extract_first_2_modes[a_coalesced_layout]()
        ]()
        comptime b_canonical_layout = tile_to_descriptor[
            b.dtype, extract_first_2_modes[b_coalesced_layout]()
        ]()

        var a_desc = _create_mma_desc[a_canonical_layout, Self.a_swizzle](a.ptr)
        var b_desc = _create_mma_desc[b_canonical_layout, Self.b_swizzle](b.ptr)

        comptime for k in range(0, Self.block_tile_shape[2], Self.mma_shape[2]):
            comptime a_offset = a.layout(IntTuple(0, k)) * size_of[
                Self.a_type
            ]()
            comptime b_offset = b.layout(IntTuple(0, k)) * size_of[
                Self.b_type
            ]()

            var c_scale: UInt32 = UInt32(0) if (init_c and k == 0) else UInt32(
                1
            )

            mma[Self.cta_group](
                a_desc + a_offset,
                b_desc + b_offset,
                c_tmem,
                self.idesc,
                c_scale=c_scale,
            )

    @always_inline
    fn mma(
        self,
        a: TileTensor[address_space = AddressSpace.SHARED, ...],
        b: TileTensor[address_space = AddressSpace.SHARED, ...],
        c_tmem: UInt32,
        init_c: Bool,
    ):
        """TileTensor overload for MMA input tiles.

        This overload accepts TileTensor directly. The layout is extracted from
        TileTensor's compile-time type parameters (shape_types, stride_types).
        """

        # Extract Layout from TileTensor's compile-time type parameters
        # comptime a_layout = Layout(
        #     coord_to_int_tuple[*a.shape_types](),
        #     coord_to_int_tuple[*a.stride_types](),
        # )
        # comptime b_layout = Layout(
        #     coord_to_int_tuple[*b.shape_types](),
        #     coord_to_int_tuple[*b.stride_types](),
        # )

        comptime a_layout = Layout(
            IntTuple(
                IntTuple(
                    a.LayoutType._shape_types[0].VariadicType[0].static_value,
                    a.LayoutType._shape_types[0].VariadicType[1].static_value,
                ),
                IntTuple(
                    a.LayoutType._shape_types[1].VariadicType[0].static_value,
                    a.LayoutType._shape_types[1].VariadicType[1].static_value,
                ),
            ),
            IntTuple(
                IntTuple(
                    a.LayoutType._stride_types[0].VariadicType[0].static_value,
                    a.LayoutType._stride_types[0].VariadicType[1].static_value,
                ),
                IntTuple(
                    a.LayoutType._stride_types[1].VariadicType[0].static_value,
                    a.LayoutType._stride_types[1].VariadicType[1].static_value,
                ),
            ),
        )

        comptime b_layout = Layout(
            IntTuple(
                IntTuple(
                    b.LayoutType._shape_types[0].VariadicType[0].static_value,
                    b.LayoutType._shape_types[0].VariadicType[1].static_value,
                ),
                IntTuple(
                    b.LayoutType._shape_types[1].VariadicType[0].static_value,
                    b.LayoutType._shape_types[1].VariadicType[1].static_value,
                ),
            ),
            IntTuple(
                IntTuple(
                    b.LayoutType._stride_types[0].VariadicType[0].static_value,
                    b.LayoutType._stride_types[0].VariadicType[1].static_value,
                ),
                IntTuple(
                    b.LayoutType._stride_types[1].VariadicType[0].static_value,
                    b.LayoutType._stride_types[1].VariadicType[1].static_value,
                ),
            ),
        )

        # Coalesce using the extracted layouts
        comptime a_coalesced_layout = coalesce(a_layout)
        comptime b_coalesced_layout = coalesce(b_layout)

        # Canonical layouts are tiled by core matrices.
        comptime a_canonical_layout = tile_to_descriptor[
            a.dtype, extract_first_2_modes[a_coalesced_layout]()
        ]()
        comptime b_canonical_layout = tile_to_descriptor[
            b.dtype, extract_first_2_modes[b_coalesced_layout]()
        ]()

        var a_desc = _create_mma_desc[a_canonical_layout, Self.a_swizzle](a.ptr)
        var b_desc = _create_mma_desc[b_canonical_layout, Self.b_swizzle](b.ptr)

        comptime for k in range(0, Self.block_tile_shape[2], Self.mma_shape[2]):
            comptime a_offset = a_layout(IntTuple(0, k)) * size_of[
                Self.a_type
            ]()
            comptime b_offset = b_layout(IntTuple(0, k)) * size_of[
                Self.b_type
            ]()

            var c_scale: UInt32 = UInt32(0) if (init_c and k == 0) else UInt32(
                1
            )

            mma[Self.cta_group](
                a_desc + a_offset,
                b_desc + b_offset,
                c_tmem,
                self.idesc,
                c_scale=c_scale,
            )

    @always_inline
    fn commit(
        self,
        ptr_mbar: UnsafePointer[address_space = AddressSpace.SHARED, ...],
    ):
        comptime if product(Self.cluster_shape) == 1:
            mma_arrive[Self.cta_group](ptr_mbar)
        else:
            mma_arrive_multicast[Self.cta_group](ptr_mbar, self.mask)

    @always_inline
    fn wait(self):
        pass

    @staticmethod
    fn _get_umma_kind[dtype: DType]() -> UMMAKind:
        comptime if dtype == DType.float32:
            return UMMAKind.KIND_TF32
        elif dtype in (DType.float16, DType.bfloat16):
            return UMMAKind.KIND_F16
        elif dtype in (DType.float8_e4m3fn, DType.float8_e5m2):
            return UMMAKind.KIND_F8F6F4
        else:
            comptime assert False, String(
                "Unsupported/not implemented operand type for UMMA: ",
                String(dtype),
            )

        return UMMAKind(-1)


struct MmaOpSM100_BlockScaled_SS[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    sfa_dtype: DType,
    sfb_dtype: DType,
    scaling_kind: UMMAKind,
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
](Defaultable, TrivialRegisterPassable):
    var idesc: UMMAInsDescriptor[Self.scaling_kind]
    var mask: UInt16

    @always_inline
    fn __init__(out self):
        comptime assert (
            Self.scaling_kind == UMMAKind.KIND_MXF8F6F4
            or Self.scaling_kind == UMMAKind.KIND_MXF4NVF4
        ), (
            "Only support MXF8F6F4 or MXF4NVF4 scaling kind for block scaled"
            " matmul!"
        )
        comptime assert (
            Self.transpose_b
        ), "MmaOpSM100 only supports transposed B"
        comptime assert Self.cta_group in (
            1,
            2,
        ), "MmaOpSM100 only supports cta_group 1 or 2"
        comptime assert (
            Self.a_type == Self.b_type
        ), "a_type and b_type must be the same"
        comptime assert Self.a_type in (
            DType.float8_e4m3fn,
            DType.uint8,  # TODO: (KERN-2238) replace with FP4-E2M1
        ), "Only support float8_e4m3fn or uint8 (F4-E2M1) for input operands"
        comptime assert (
            Self.sfa_dtype == Self.sfb_dtype
        ), "sfa_dtype and sfb_dtype must be the same"
        comptime assert Self.sfa_dtype in (
            DType.float8_e4m3fn,
            DType.float8_e8m0fnu,
        ), "Only support float8_e4m3fn or float8_e8m0fnu for scales"

        self.idesc = UMMAInsDescriptor[Self.scaling_kind].create[
            Self.accum_type,
            Self.a_type,
            Self.b_type,
            Self.sfa_dtype,
            Index[dtype = DType.uint32](Self.mma_shape[0], Self.mma_shape[1]),
            transpose_b = Self.transpose_b,
        ]()

        self.mask = 0

        # Here we compute the mask inside mma object to hide the complexity.
        # We may get better asm if the mask if computed outside from TMA masks,
        # and passed to `commit`, need to verify.
        comptime if product(Self.cluster_shape) > 1:
            comptime dim0_mask = cluster_mask_base[Self.cluster_shape, 0]()
            comptime dim1_mask = cluster_mask_base[Self.cluster_shape, 1]()

            # The mask includes ctas on the same row and column in the cluster
            # Example mask for cta (0, 1) is cluster (4,4)
            #             x x x x
            #             o x o o
            #             o x o o
            #             o x o o
            self.mask = (
                dim0_mask
                << UInt16((block_id_in_cluster.y * UInt(Self.cluster_shape[0])))
            ) | (dim1_mask << UInt16(block_id_in_cluster.x))

            # Include peer cta's row
            # Example mask for cta (0, 1) is cluster (4,4)
            #             x x x x
            #             x x x x
            #             o x o o
            #             o x o o
            comptime if Self.cta_group == 2:
                self.mask |= dim1_mask << UInt16((block_id_in_cluster.x ^ 1))

    @always_inline
    fn mma(
        self,
        a: LayoutTensor[address_space = AddressSpace.SHARED, ...],
        b: LayoutTensor[address_space = AddressSpace.SHARED, ...],
        sfa_smem: LayoutTensor[address_space = AddressSpace.SHARED, ...],
        sfb_smem: LayoutTensor[address_space = AddressSpace.SHARED, ...],
        c_tmem: UInt32,
        sfa_tmem: UInt32,
        sfb_tmem: UInt32,
        init_c: Bool,
        work_tile_coord: Tuple[UInt, UInt],
    ):
        """MMA input tiles.

        The layout assumes that coalesce(A) has shape (bm, sw_k, num_sw_k), we currently
        assumes bm = mma_m. In future, we can tile it to (mma_m, sw_k, num_sw_k, num_mma_m)
        The same logic applies to matrix B.
        """

        # Coalesce a and b
        # A and B are coalesced to rank-2 if it's only one tile or rank-3 if it has
        # multiple canonical layouts in K dim.
        comptime a_coalesced_layout = coalesce(a.layout)
        comptime b_coalesced_layout = coalesce(b.layout)

        # Canonical layouts are tiled by core matrices.
        comptime a_canonical_layout = tile_to_descriptor[
            a.dtype, extract_first_2_modes[a_coalesced_layout]()
        ]()
        comptime b_canonical_layout = tile_to_descriptor[
            b.dtype, extract_first_2_modes[b_coalesced_layout]()
        ]()

        var a_desc = _create_mma_desc[a_canonical_layout, Self.a_swizzle](a.ptr)
        var b_desc = _create_mma_desc[b_canonical_layout, Self.b_swizzle](b.ptr)

        comptime assert (
            Self.block_tile_shape[2] == 128 and Self.mma_shape[2] == 32
        ), "block_tile_shape[2] must be 128 and mma_shape[2] must be 32"

        # when scaling kind is MXF8F6F4, one scale tile covers the whole [BM,BK] and [MMA_N,BK] tiles so we load it once.
        comptime if Self.scaling_kind == UMMAKind.KIND_MXF8F6F4:
            self.copy_sf_to_tmem[
                Self.sfa_dtype, sfa_smem.layout, Self.block_tile_shape[0], 0
            ](sfa_smem, sfa_tmem)
            self.copy_sf_to_tmem[
                Self.sfb_dtype,
                sfb_smem.layout,
                align_up(Self.mma_shape[1], SF_MN_GROUP_SIZE),
                0,
            ](sfb_smem, sfb_tmem)

        comptime for k in range(0, Self.block_tile_shape[2], Self.mma_shape[2]):
            comptime a_offset = a.layout(IntTuple(0, k)) * size_of[
                Self.a_type
            ]()
            comptime b_offset = b.layout(IntTuple(0, k)) * size_of[
                Self.b_type
            ]()

            var c_scale: UInt32 = UInt32(0) if (init_c and k == 0) else UInt32(
                1
            )

            comptime sf_idx = k // Self.mma_shape[2]

            @always_inline
            @parameter
            fn _get_sfb_tmem_offset[
                mma_n: Int,
            ](sfb_tmem: UInt32, work_tile_coord: Tuple[UInt, UInt],) -> UInt32:
                comptime if mma_n in (64, 192):
                    return sfb_tmem + UInt32(work_tile_coord[1] % 2) * 2
                else:
                    return sfb_tmem

            var sfb_tmem_offset = _get_sfb_tmem_offset[Self.mma_shape[1]](
                sfb_tmem, work_tile_coord
            )

            comptime if Self.scaling_kind == UMMAKind.KIND_MXF8F6F4:
                var runtime_desc = UMMAInsDescriptor[
                    Self.scaling_kind
                ].update_desc_with_sf_id[UInt32(sf_idx)](
                    self.idesc,
                )
                mma[Self.cta_group](
                    a_desc + a_offset,
                    b_desc + b_offset,
                    c_tmem,
                    runtime_desc,
                    sfa_tmem,
                    sfb_tmem_offset,
                    c_scale=c_scale,
                )
            else:
                self.copy_sf_to_tmem[
                    Self.sfa_dtype,
                    sfa_smem.layout,
                    Self.block_tile_shape[0],
                    sf_idx,
                ](sfa_smem, sfa_tmem)
                self.copy_sf_to_tmem[
                    Self.sfb_dtype,
                    sfb_smem.layout,
                    align_up(Self.mma_shape[1], SF_MN_GROUP_SIZE),
                    sf_idx,
                ](sfb_smem, sfb_tmem)

                mma[Self.cta_group](
                    a_desc + a_offset,
                    b_desc + b_offset,
                    c_tmem,
                    self.idesc,
                    sfa_tmem + UInt32(sf_idx * (SF_MN_GROUP_SIZE // 32)),
                    sfb_tmem_offset + UInt32(sf_idx * (SF_MN_GROUP_SIZE // 32)),
                    c_scale=c_scale,
                )

    @always_inline
    fn mma(
        self,
        a: TileTensor[address_space = AddressSpace.SHARED, ...],
        b: TileTensor[address_space = AddressSpace.SHARED, ...],
        sfa_smem: TileTensor[address_space = AddressSpace.SHARED, ...],
        sfb_smem: TileTensor[address_space = AddressSpace.SHARED, ...],
        c_tmem: UInt32,
        sfa_tmem: UInt32,
        sfb_tmem: UInt32,
        init_c: Bool,
    ):
        """TileTensor overload for block-scaled MMA input tiles.

        This overload accepts TileTensor directly for A, B, and scale factor
        tiles. The layout is extracted from TileTensor's compile-time type
        parameters (shape_types, stride_types) using direct VariadicType
        extraction for fast compile times.
        """

        # Extract Layout from TileTensor's compile-time type parameters
        # Use direct VariadicType extraction (fast) instead of coord_to_int_tuple (slow)
        # A and B tiles have nested Coord structure (swizzled internal_k_major layout)
        comptime a_layout = Layout(
            IntTuple(
                IntTuple(
                    a.LayoutType._shape_types[0].VariadicType[0].static_value,
                    a.LayoutType._shape_types[0].VariadicType[1].static_value,
                ),
                IntTuple(
                    a.LayoutType._shape_types[1].VariadicType[0].static_value,
                    a.LayoutType._shape_types[1].VariadicType[1].static_value,
                ),
            ),
            IntTuple(
                IntTuple(
                    a.LayoutType._stride_types[0].VariadicType[0].static_value,
                    a.LayoutType._stride_types[0].VariadicType[1].static_value,
                ),
                IntTuple(
                    a.LayoutType._stride_types[1].VariadicType[0].static_value,
                    a.LayoutType._stride_types[1].VariadicType[1].static_value,
                ),
            ),
        )
        comptime b_layout = Layout(
            IntTuple(
                IntTuple(
                    b.LayoutType._shape_types[0].VariadicType[0].static_value,
                    b.LayoutType._shape_types[0].VariadicType[1].static_value,
                ),
                IntTuple(
                    b.LayoutType._shape_types[1].VariadicType[0].static_value,
                    b.LayoutType._shape_types[1].VariadicType[1].static_value,
                ),
            ),
            IntTuple(
                IntTuple(
                    b.LayoutType._stride_types[0].VariadicType[0].static_value,
                    b.LayoutType._stride_types[0].VariadicType[1].static_value,
                ),
                IntTuple(
                    b.LayoutType._stride_types[1].VariadicType[0].static_value,
                    b.LayoutType._stride_types[1].VariadicType[1].static_value,
                ),
            ),
        )

        # Scale factor tiles: extract layout from TileTensor type parameters
        # SF layout has 3-level nesting: ((32, tiles_m), ((4, 4), tiles_k))
        comptime sfa_layout = Layout(
            IntTuple(
                IntTuple(
                    sfa_smem.LayoutType._shape_types[0]
                    .VariadicType[0]
                    .static_value,
                    sfa_smem.LayoutType._shape_types[0]
                    .VariadicType[1]
                    .static_value,
                ),
                IntTuple(
                    IntTuple(
                        sfa_smem.LayoutType._shape_types[1]
                        .VariadicType[0]
                        .VariadicType[0]
                        .static_value,
                        sfa_smem.LayoutType._shape_types[1]
                        .VariadicType[0]
                        .VariadicType[1]
                        .static_value,
                    ),
                    sfa_smem.LayoutType._shape_types[1]
                    .VariadicType[1]
                    .static_value,
                ),
            ),
            IntTuple(
                IntTuple(
                    sfa_smem.LayoutType._stride_types[0]
                    .VariadicType[0]
                    .static_value,
                    sfa_smem.LayoutType._stride_types[0]
                    .VariadicType[1]
                    .static_value,
                ),
                IntTuple(
                    IntTuple(
                        sfa_smem.LayoutType._stride_types[1]
                        .VariadicType[0]
                        .VariadicType[0]
                        .static_value,
                        sfa_smem.LayoutType._stride_types[1]
                        .VariadicType[0]
                        .VariadicType[1]
                        .static_value,
                    ),
                    sfa_smem.LayoutType._stride_types[1]
                    .VariadicType[1]
                    .static_value,
                ),
            ),
        )
        comptime sfb_layout = Layout(
            IntTuple(
                IntTuple(
                    sfb_smem.LayoutType._shape_types[0]
                    .VariadicType[0]
                    .static_value,
                    sfb_smem.LayoutType._shape_types[0]
                    .VariadicType[1]
                    .static_value,
                ),
                IntTuple(
                    IntTuple(
                        sfb_smem.LayoutType._shape_types[1]
                        .VariadicType[0]
                        .VariadicType[0]
                        .static_value,
                        sfb_smem.LayoutType._shape_types[1]
                        .VariadicType[0]
                        .VariadicType[1]
                        .static_value,
                    ),
                    sfb_smem.LayoutType._shape_types[1]
                    .VariadicType[1]
                    .static_value,
                ),
            ),
            IntTuple(
                IntTuple(
                    sfb_smem.LayoutType._stride_types[0]
                    .VariadicType[0]
                    .static_value,
                    sfb_smem.LayoutType._stride_types[0]
                    .VariadicType[1]
                    .static_value,
                ),
                IntTuple(
                    IntTuple(
                        sfb_smem.LayoutType._stride_types[1]
                        .VariadicType[0]
                        .VariadicType[0]
                        .static_value,
                        sfb_smem.LayoutType._stride_types[1]
                        .VariadicType[0]
                        .VariadicType[1]
                        .static_value,
                    ),
                    sfb_smem.LayoutType._stride_types[1]
                    .VariadicType[1]
                    .static_value,
                ),
            ),
        )

        # Coalesce using the extracted layouts
        comptime a_coalesced_layout = coalesce(a_layout)
        comptime b_coalesced_layout = coalesce(b_layout)

        # Canonical layouts are tiled by core matrices.
        comptime a_canonical_layout = tile_to_descriptor[
            a.dtype, extract_first_2_modes[a_coalesced_layout]()
        ]()
        comptime b_canonical_layout = tile_to_descriptor[
            b.dtype, extract_first_2_modes[b_coalesced_layout]()
        ]()

        var a_desc = _create_mma_desc[a_canonical_layout, Self.a_swizzle](a.ptr)
        var b_desc = _create_mma_desc[b_canonical_layout, Self.b_swizzle](b.ptr)

        comptime assert (
            Self.block_tile_shape[2] == 128 and Self.mma_shape[2] == 32
        ), "block_tile_shape[2] must be 128 and mma_shape[2] must be 32"

        # when scaling kind is MXF8F6F4, one scale tile covers the whole [BM,BK] and [MMA_N,BK] tiles so we load it once.
        comptime if Self.scaling_kind == UMMAKind.KIND_MXF8F6F4:
            self._copy_sf_to_tmem_tt[
                Self.sfa_dtype, sfa_layout, Self.block_tile_shape[0], 0
            ](sfa_smem, sfa_tmem)
            self._copy_sf_to_tmem_tt[
                Self.sfb_dtype, sfb_layout, Self.mma_shape[1], 0
            ](sfb_smem, sfb_tmem)

        comptime for k in range(0, Self.block_tile_shape[2], Self.mma_shape[2]):
            comptime a_offset = a_layout(IntTuple(0, k)) * size_of[
                Self.a_type
            ]()
            comptime b_offset = b_layout(IntTuple(0, k)) * size_of[
                Self.b_type
            ]()

            var c_scale: UInt32 = UInt32(0) if (init_c and k == 0) else UInt32(
                1
            )

            comptime if Self.scaling_kind == UMMAKind.KIND_MXF8F6F4:
                comptime sf_idx = k // Self.mma_shape[2]
                var runtime_desc = UMMAInsDescriptor[
                    Self.scaling_kind
                ].update_desc_with_sf_id[UInt32(sf_idx)](
                    self.idesc,
                )
                mma[Self.cta_group](
                    a_desc + a_offset,
                    b_desc + b_offset,
                    c_tmem,
                    runtime_desc,
                    sfa_tmem,
                    sfb_tmem,
                    c_scale=c_scale,
                )
            else:
                comptime sf_idx = k // Self.mma_shape[2]
                # when scaling kind is MXFP4NVF4, four scale tiles cover the whole [BM,BK] and [MMA_N,BK] tiles so we need to load one scale tile for each k iteration.
                self._copy_sf_to_tmem_tt[
                    Self.sfa_dtype,
                    sfa_layout,
                    Self.block_tile_shape[0],
                    sf_idx,
                ](sfa_smem, sfa_tmem)
                self._copy_sf_to_tmem_tt[
                    Self.sfb_dtype, sfb_layout, Self.mma_shape[1], sf_idx
                ](sfb_smem, sfb_tmem)

                mma[Self.cta_group](
                    a_desc + a_offset,
                    b_desc + b_offset,
                    c_tmem,
                    self.idesc,
                    sfa_tmem + UInt32(sf_idx * (SF_MN_GROUP_SIZE // 32)),
                    sfb_tmem + UInt32(sf_idx * (SF_MN_GROUP_SIZE // 32)),
                    c_scale=c_scale,
                )

    @always_inline
    fn commit(
        self,
        ptr_mbar: UnsafePointer[address_space = AddressSpace.SHARED, ...],
    ):
        comptime if product(Self.cluster_shape) == 1:
            mma_arrive[Self.cta_group](ptr_mbar)
        else:
            mma_arrive_multicast[Self.cta_group](ptr_mbar, self.mask)

    @always_inline
    fn wait(self):
        pass

    @always_inline
    fn copy_sf_to_tmem[
        sf_dtype: DType,
        sf_smem_layout: Layout,
        TILE_MN: Int,
        tile_k_idx: Int,
    ](
        self,
        sf_smem: LayoutTensor[address_space = AddressSpace.SHARED, ...],
        sf_tmem: UInt32,
    ):
        comptime sf_smem_size = sf_smem_layout.size()

        comptime for i in range(TILE_MN // SF_MN_GROUP_SIZE):
            comptime idx = IntTuple(
                i * SF_ATOM_M[0], tile_k_idx * SF_ATOM_M[1] * SF_ATOM_K
            )
            comptime sf_offset = sf_smem_layout(idx) * size_of[sf_dtype]()
            var sf_tmem_addr = (
                sf_tmem
                + UInt32(i * (SF_MN_GROUP_SIZE // 32))
                + UInt32(tile_k_idx * (SF_MN_GROUP_SIZE // 32))
            )
            var sf_desc = MMASmemDescriptor.create[
                8 * 16, 0, TensorMapSwizzle.SWIZZLE_NONE
            ](sf_smem.ptr + sf_offset)
            tcgen05_cp[
                cta_group = Int32(Self.cta_group),
                datapaths=32,
                bits=128,
                multicast="warpx4",
            ](sf_tmem_addr, sf_desc)

    @always_inline
    fn _copy_sf_to_tmem_tt[
        sf_dtype: DType,
        sf_smem_layout: Layout,
        TILE_MN: Int,
        tile_k_idx: Int,
    ](
        self,
        sf_smem: TileTensor[address_space = AddressSpace.SHARED, ...],
        sf_tmem: UInt32,
    ):
        """TileTensor overload for copying scale factors to TMEM."""
        comptime sf_smem_size = sf_smem_layout.size()

        comptime for i in range(TILE_MN // SF_MN_GROUP_SIZE):
            comptime idx = IntTuple(
                i * SF_ATOM_M[0], tile_k_idx * SF_ATOM_M[1] * SF_ATOM_K
            )
            comptime sf_offset = sf_smem_layout(idx) * size_of[sf_dtype]()
            var sf_tmem_addr = (
                sf_tmem
                + UInt32(i * (SF_MN_GROUP_SIZE // 32))
                + UInt32(tile_k_idx * (SF_MN_GROUP_SIZE // 32))
            )
            var sf_desc = MMASmemDescriptor.create[
                8 * 16, 0, TensorMapSwizzle.SWIZZLE_NONE
            ](sf_smem.ptr + sf_offset)
            tcgen05_cp[
                cta_group = Int32(Self.cta_group),
                datapaths=32,
                bits=128,
                multicast="warpx4",
            ](sf_tmem_addr, sf_desc)
