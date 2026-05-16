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
"""CPU entrypoint for grouped 1D-1D block-scaled SM100 matmul.

This module provides the public API for launching the grouped 1D-1D matmul
kernel for Mixture of Experts (MoE) layers.

Usage:
    grouped_matmul_block_scaled[transpose_b=True, config=config](
        c_tensor,  # Output: TileTensor (total_tokens, N)
        a_tensor,  # Input A: TileTensor (total_tokens, K)
        a_offsets,  # Per-expert offsets: TileTensor 1D
        a_scale_offsets,  # Per-expert scale offsets: TileTensor 1D
        b_tensor,  # Weights B: TileTensor (num_experts, N, K)
        expert_ids,  # Active expert IDs: TileTensor 1D
        a_scales,  # Scale factors for A: TileTensor 5D
        b_scales,  # Scale factors for B: TileTensor 6D
        expert_scales,  # Per-expert output scaling: TileTensor 1D
        num_active_experts,
        ctx,
    )
"""

from std.math import align_up, ceildiv
from std.sys import size_of

from std.gpu.host import DeviceContext, Dim, FuncAttribute
from std.gpu.host.info import B200
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from std.gpu.primitives.grid_controls import PDLLevel, pdl_launch_attributes
from layout import Coord, Idx, RuntimeInt, TileTensor, row_major
from layout.tma_async import create_tensor_tile
from structured_kernels.tile_types import create_tma_tile
from structured_kernels.kernel_common import WarpRole1D1D

from std.utils.index import Index
from std.utils.static_tuple import StaticTuple

from linalg.fp4_utils import (
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
    NVFP4_SF_DTYPE,
    MXFP4_SF_DTYPE,
    MXFP8_SF_DTYPE,
)
from structured_kernels.trace_buf import NullTrace, TraceBuf
from ..structured_kernels.config import BlockScaledMatmulConfig
from .grouped_1d1d_matmul_kernel import (
    Grouped1D1DMatmulKernel,
    NullSwiGLUOutput,
    SwiGLUOutput,
)
from std.memory import UnsafePointer


def grouped_matmul_block_scaled[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    sfa_dtype: DType,
    sfb_dtype: DType,
    transpose_b: Bool,
    *,
    config: BlockScaledMatmulConfig[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ],
    pdl_level: PDLLevel = PDLLevel.ON,
    # When True, the kernel emits packed NVFP4 + a 5D FP8-E4M3 scale tile
    # in place of the BF16 GMEM C store, fusing SwiGLU + per-block quant
    # into the matmul epilogue. Caller must:
    #   1) pre-permute W on the N axis with σ(2i)=i, σ(2i+1)=H+i,
    #   2) pass a `RealSwiGLUOutput[...]` instance via `swiglu_out`.
    # When False, swiglu_out=NullSwiGLUOutput() is used and the kernel
    # is bit-identical to the original BF16-output path.
    fuse_swiglu_nvfp4: Bool = False,
    SwiGLUOutputT: SwiGLUOutput = NullSwiGLUOutput,
    swiglu_match_bf16: Bool = True,
    swiglu_disable_compute: Bool = False,
    swiglu_enable_trace: Bool = False,
    TraceBufT: TraceBuf = NullTrace,
    swiglu_use_inplace: Bool = False,
](
    c_device: TileTensor,
    a_device: TileTensor,
    a_offsets: TileTensor,
    a_scale_offsets: TileTensor,
    _b_device: TileTensor,
    expert_ids: TileTensor,
    a_scales: TileTensor,
    _b_scales: TileTensor,
    expert_scales: TileTensor,
    num_active_experts: Int,
    ctx: DeviceContext,
    swiglu_out: SwiGLUOutputT = NullSwiGLUOutput(),
    trace_buf: TraceBufT = NullTrace(),
) raises:
    """Launch grouped 1D-1D block-scaled matmul kernel.

    This function sets up TMA descriptors and launches the kernel with the
    proper configuration for 1D-1D tensor layout.

    Args:
        c_device: Output tensor (total_tokens, N).
        a_device: Input A tensor (total_tokens, K).
        a_offsets: Per-expert offsets (num_active_experts + 1).
        a_scale_offsets: Per-expert scale offsets (num_active_experts).
        _b_device: Weight tensor B (num_experts, N, K).
        expert_ids: Active expert IDs (num_active_experts).
        a_scales: Scale factors for A (5D).
        _b_scales: Scale factors for B (6D).
        expert_scales: Per-expert output scaling (num_experts).
        num_active_experts: Number of active experts.
        ctx: Device context.
        swiglu_out: Sink carrier when `fuse_swiglu_nvfp4=True` (packed
            NVFP4 + E4M3 SF tile). `NullSwiGLUOutput()` otherwise.
        trace_buf: Per-CTA timestamp buffer when `swiglu_enable_trace=True`.
            `NullTrace()` otherwise.
    """
    comptime assert transpose_b, "Only support transposed B"

    comptime assert (
        sfa_dtype == sfb_dtype
    ), "Only support same scales dtype for A and B"
    comptime assert sfa_dtype in (
        NVFP4_SF_DTYPE,
        MXFP4_SF_DTYPE,
        MXFP8_SF_DTYPE,
    ), (
        "Only support NVFP4_SF_DTYPE, MXFP4_SF_DTYPE, or MXFP8_SF_DTYPE for"
        " scales"
    )

    comptime MMA_M = config.mma_shape[0]
    comptime MMA_N = config.mma_shape[1]
    comptime MMA_K = config.mma_shape[2]

    comptime BM = MMA_M // config.cta_group
    comptime BN = MMA_N // config.cta_group
    comptime BK = config.block_tile_shape[2]

    comptime assert config.cta_group in (
        1,
        2,
    ), "Only support cta_group == 1 or 2"

    comptime assert config.k_group_size == 1, "Only support k_group_size == 1"

    comptime assert config.num_split_k == 1, "Only support split_k == 1"

    comptime assert (
        config.num_pipeline_stages % config.k_group_size == 0
    ), "num_pipeline_stages must be a multiple of k_group_size"

    # Extract static dimensions from TileTensor types.
    # B is (num_experts, N, K), C is (M_dynamic, N), A is (M_dynamic, K).
    comptime num_experts = type_of(_b_device).static_shape[0]
    comptime N = type_of(c_device).static_shape[1]
    comptime expert_n = N
    comptime K = type_of(a_device).static_shape[1]
    comptime assert K % 16 == 0, (
        "Due to TMA limitations, K must be a multiple of 16 bytes"
        + " but got K = "
        + String(K)
    )

    # Reshape B from (num_experts, N, K) to (num_experts * N, K)
    var b_device = _b_device.reshape(
        row_major(Coord(Idx[num_experts * N](), Idx[K]()))
    )

    comptime if config.cta_group == 2:
        comptime assert MMA_M == 256 and MMA_N in (
            64,
            128,
            256,
        ), "Only support cta_group == 2 with MMA_M == 256"
        comptime assert (
            config.AB_swapped
        ), "cta_group == 2 requires AB_swapped for scheduler alignment"
    else:
        comptime assert MMA_M == 128 and MMA_N in (
            8,
            16,
            32,
            64,
            128,
            256,
        ), (
            "Only support MMA_M == 128 and MMA_N in (8, 16, 32, 64, 128,"
            " 256) when cta_group == 1"
        )

    comptime cluster_shape = config.cluster_shape

    comptime assert (
        ceildiv(K, BK) % config.k_group_size == 0
    ), "K iterations must be a multiple of k_group_size"

    # Instantiate kernel -- c_device_layout derived from caller's TileTensor
    # so types match by construction in enqueue_function.
    comptime c_device_tt_layout = type_of(c_device).LayoutType
    comptime matmul_kernel = Grouped1D1DMatmulKernel[
        a_type,
        b_type,
        c_type,
        sfa_dtype,
        sfb_dtype,
        c_device_tt_layout,
        transpose_b,
        config=config,
        static_N=expert_n,
        cluster_shape=StaticTuple[Int32, 3](
            Int32(config.cluster_shape[0]),
            Int32(config.cluster_shape[1]),
            Int32(config.cluster_shape[2]),
        ),
        pdl_level=pdl_level,
        fuse_swiglu_nvfp4=fuse_swiglu_nvfp4,
        SwiGLUOutputT=SwiGLUOutputT,
        swiglu_match_bf16=swiglu_match_bf16,
        swiglu_disable_compute=swiglu_disable_compute,
        swiglu_enable_trace=swiglu_enable_trace,
        TraceBufT=TraceBufT,
        swiglu_use_inplace=swiglu_use_inplace,
    ]
    comptime KernelType = type_of(matmul_kernel)

    # Shared memory size calculation
    comptime SmemType = KernelType.SmemType
    comptime smem_size = size_of[SmemType]()

    # B200 SMEM limit
    comptime b200_smem = B200.shared_memory_per_multiprocessor - 1024

    comptime c_tma_tile_shape_mma128 = Index(
        64, config.output_tile_shape[1]
    ) if not config.AB_swapped else Index(config.output_tile_shape[0], 64)
    comptime c_tma_tile_shape = config.output_tile_shape if (
        MMA_M == 256 or config.cta_group == 1
    ) else c_tma_tile_shape_mma128

    # When c_swizzle is SWIZZLE_NONE (MMA_N=8), c_swizzle.bytes() is 0.
    # The TMA descriptor dim1 must equal the tile dim1 in that case.
    comptime _c_swizzle_elems = config.c_swizzle.bytes() // size_of[c_type]()
    comptime c_tma_tile_shape_1 = c_tma_tile_shape[
        1
    ] if _c_swizzle_elems == 0 else _c_swizzle_elems

    # Scale factor TMA — use 4D uint16 with batch=1 to avoid 2× TMA overfetch.
    # SM100 TMA rounds boxDim[0] to 32B min; old innermost=16B caused 2× fetch.
    # Reinterpret as uint16, merge SF_ATOM_M[0] and SF_ATOM_M[1]*SF_ATOM_K into
    # sf_atom_u16 = 256 uint16 = 512B, well above 32B minimum.
    comptime sf_atom_u16 = (
        SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K
    ) // 2  # 256 uint16 = 512 bytes
    comptime sfb_atom_u16 = (
        KernelType.SFB_TMA_ROWS * SF_ATOM_M[1] * SF_ATOM_K
    ) // 2

    comptime sfa_tma_tile_shape = Index(
        1,  # batch dim
        BM // SF_MN_GROUP_SIZE,
        config.num_sf_k_tiles,
        sf_atom_u16,
    )

    # SFB TMA tile shape: for MMA_N < 64, reduced tile (1 k-atom, MMA_N rows)
    # loaded by the dedicated SfbTMALoad warp; for MMA_N >= 64, full atom.
    # Derive from kernel struct to keep a single source of truth.
    comptime sfb_tma_tile_shape = Index(
        1,  # batch dim
        align_up(MMA_N, SF_MN_GROUP_SIZE) // SF_MN_GROUP_SIZE,
        KernelType.SFB_TMA_K_ATOMS,
        sfb_atom_u16,
    )

    # Create 4D uint16 views of scale tensors (same memory, reinterpreted).
    # a_scales: 5D (M_groups, K_groups, SF_ATOM_M[0], SF_ATOM_M[1], SF_ATOM_K)
    #        -> 4D uint16 (1, M_groups, K_groups, sf_atom_u16)
    from std.memory import UnsafePointer as Ptr

    var sfa_4d_shape = Coord(
        Idx[1](),
        a_scales.layout.shape[0](),
        a_scales.layout.shape[1](),
        Idx[sf_atom_u16](),
    )
    var sfa_4d_layout = row_major(sfa_4d_shape)
    var sfa_4d = TileTensor[DType.uint16, type_of(sfa_4d_layout), MutAnyOrigin](
        rebind[Ptr[Scalar[DType.uint16], MutAnyOrigin]](a_scales.ptr),
        sfa_4d_layout,
    )

    # _b_scales: 6D -> 4D uint16 (1, num_experts*N_groups, K_groups, sf_atom_u16)
    # Global view always uses sf_atom_u16 (full atom); TMA tile may be smaller.
    var sfb_dim0 = Int(_b_scales.layout.shape[0]().value()) * Int(
        _b_scales.layout.shape[1]().value()
    )
    var sfb_4d_shape = Coord(
        Idx[1](),
        RuntimeInt[DType.int64](Scalar[DType.int64](sfb_dim0)),
        _b_scales.layout.shape[2](),
        Idx[sf_atom_u16](),
    )
    var sfb_4d_layout = row_major(sfb_4d_shape)
    var sfb_4d = TileTensor[DType.uint16, type_of(sfb_4d_layout), MutAnyOrigin](
        rebind[Ptr[Scalar[DType.uint16], MutAnyOrigin]](_b_scales.ptr),
        sfb_4d_layout,
    )

    # Raw SFB pointer and strides for cp.async path (MMA_N < 64).
    # When group_size < SF_MN_GROUP_SIZE, the SfbTMALoad warp uses cp.async
    # instead of TMA to avoid loading full 128-row atoms for small groups.
    comptime _sfb_K_TILE_ELEMS = SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K

    # Define kernel function
    comptime kernel = matmul_kernel.run

    var grid_dim = (
        B200.sm_count,
        1,
        1,
    )

    # Always launch with scheduler warp. SFB warps only on MMA_N < 64 decode
    # path, so MMA_N >= 64 (prefill / 2SM) shrinks from 384 → 224 threads and
    # frees ~7.5K registers per CTA. Source the launch dim from the kernel's
    # WarpRole so any per-config epilogue-warp count flows through.
    comptime block_threads = KernelType.WarpRole.TOTAL_THREADS_WITH_SCHED

    # Re-wrap 1D TileTensors with GMEMLayout1D to match the kernel's
    # expected types. The caller's TileTensors may have a different symbolic
    # LayoutType (from _IntTupleToCoordLike) than the kernel's GMEMLayout1D.
    from std.memory import UnsafePointer as Ptr
    from structured_kernels.tile_types import GMEMLayout1D

    def _to_1d[
        target_type: DType,
    ](t: TileTensor) -> TileTensor[target_type, GMEMLayout1D, MutAnyOrigin]:
        var shape = Coord(
            RuntimeInt[DType.int64](
                Scalar[DType.int64](t.layout.shape[0]().value())
            )
        )
        var stride = Coord(Idx[1]())
        return TileTensor[target_type, GMEMLayout1D, MutAnyOrigin](
            ptr=Ptr[Scalar[target_type], MutAnyOrigin](
                unsafe_from_address=Int(t.ptr)
            ),
            layout=GMEMLayout1D(shape, stride),
        )

    # When AB_swapped, swap A/B operands and their scale factors for TMA
    # and kernel launch. The @parameter if ensures compile-time branching.
    comptime if config.AB_swapped:
        var a_tma_op = create_tma_tile[
            KernelType.ATileLayout,
            KernelType.ADescLayout,
            Index(BM // cluster_shape[1], BK),
            swizzle_mode=config.a_swizzle,
        ](ctx, b_device)
        var b_tma_op = create_tma_tile[
            KernelType.BTileLayout,
            KernelType.BDescLayout,
            Index(
                BN // (cluster_shape[0] // config.cta_group), BK
            ) if transpose_b else Index(
                BK, BN // (cluster_shape[0] // config.cta_group)
            ),
            swizzle_mode=config.b_swizzle,
        ](ctx, a_device)
        var c_tma_op = create_tma_tile[
            KernelType.CTileLayout,
            KernelType.CDescLayout,
            Index(c_tma_tile_shape[0], c_tma_tile_shape_1),
            swizzle_mode=config.c_swizzle,
        ](ctx, c_device)
        # SF TMA: use create_tensor_tile directly with uint16 views.
        var sfa_tma_op = create_tensor_tile[
            sfa_tma_tile_shape,
            swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
            __tile_shape=sfa_tma_tile_shape,
            __desc_shape=sfa_tma_tile_shape,
        ](
            ctx, sfb_4d
        )  # AB_swapped: SFA uses sfb data
        var sfb_tma_op = create_tensor_tile[
            sfb_tma_tile_shape,
            swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
            __tile_shape=sfb_tma_tile_shape,
            __desc_shape=sfb_tma_tile_shape,
        ](
            ctx, sfa_4d
        )  # AB_swapped: SFB uses sfa data
        ctx.enqueue_function[kernel](
            a_tma_op,
            b_tma_op,
            c_tma_op,
            sfa_tma_op,
            sfb_tma_op,
            _to_1d[DType.uint32](a_offsets),
            _to_1d[DType.uint32](a_scale_offsets),
            _to_1d[DType.int32](expert_ids),
            _to_1d[DType.float32](expert_scales),
            c_device,
            num_active_experts,
            UInt32(K),
            # AB_swapped: SFB data comes from a_scales.
            rebind[UnsafePointer[Scalar[sfa_dtype], ImmutAnyOrigin]](
                a_scales.ptr
            ),
            Int(a_scales.layout.shape[1]().value()) * _sfb_K_TILE_ELEMS,
            Int(a_scales.layout.shape[1]().value()),
            swiglu_out,
            trace_buf,
            grid_dim=grid_dim,
            block_dim=block_threads,
            cluster_dim=Dim(
                cluster_shape[0], cluster_shape[1], cluster_shape[2]
            ),
            shared_mem_bytes=smem_size,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(b200_smem)
            ),
            attributes=pdl_launch_attributes(pdl_level),
        )
    else:
        var a_tma_op = create_tma_tile[
            KernelType.ATileLayout,
            KernelType.ADescLayout,
            Index(BM // cluster_shape[1], BK),
            swizzle_mode=config.a_swizzle,
        ](ctx, a_device)
        var b_tma_op = create_tma_tile[
            KernelType.BTileLayout,
            KernelType.BDescLayout,
            Index(
                BN // (cluster_shape[0] // config.cta_group), BK
            ) if transpose_b else Index(
                BK, BN // (cluster_shape[0] // config.cta_group)
            ),
            swizzle_mode=config.b_swizzle,
        ](ctx, b_device)
        var c_tma_op = create_tma_tile[
            KernelType.CTileLayout,
            KernelType.CDescLayout,
            c_tma_tile_shape,
            swizzle_mode=config.c_swizzle,
        ](ctx, c_device)
        # SF TMA: use create_tensor_tile directly with uint16 views.
        var sfa_tma_op = create_tensor_tile[
            sfa_tma_tile_shape,
            swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
            __tile_shape=sfa_tma_tile_shape,
            __desc_shape=sfa_tma_tile_shape,
        ](ctx, sfa_4d)
        var sfb_tma_op = create_tensor_tile[
            sfb_tma_tile_shape,
            swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
            __tile_shape=sfb_tma_tile_shape,
            __desc_shape=sfb_tma_tile_shape,
        ](ctx, sfb_4d)
        ctx.enqueue_function[kernel](
            a_tma_op,
            b_tma_op,
            c_tma_op,
            sfa_tma_op,
            sfb_tma_op,
            _to_1d[DType.uint32](a_offsets),
            _to_1d[DType.uint32](a_scale_offsets),
            _to_1d[DType.int32](expert_ids),
            _to_1d[DType.float32](expert_scales),
            c_device,
            num_active_experts,
            UInt32(K),
            # Non-swapped: SFB data comes from _b_scales.
            rebind[UnsafePointer[Scalar[sfb_dtype], ImmutAnyOrigin]](
                _b_scales.ptr
            ),
            Int(_b_scales.layout.shape[2]().value()) * _sfb_K_TILE_ELEMS,
            Int(_b_scales.layout.shape[2]().value()),
            swiglu_out,
            trace_buf,
            grid_dim=grid_dim,
            block_dim=block_threads,
            cluster_dim=Dim(
                cluster_shape[0], cluster_shape[1], cluster_shape[2]
            ),
            shared_mem_bytes=smem_size,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(b200_smem)
            ),
            attributes=pdl_launch_attributes(pdl_level),
        )
