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
from math import align_up
from sys import argv, size_of

import linalg.matmul.vendor.blas as vendor_blas
from buffer.buffer import NDBuffer
from buffer.dimlist import DimList, Dim
from gpu.host import DeviceContext
from gpu.host.nvidia.tma import TensorMapSwizzle
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from internal_utils import assert_almost_equal
from internal_utils._utils import ValOrDim, dynamic, static
from layout._ndbuffer_stub import from_ndbuffer_row_major
from linalg.grouped_matmul_sm100_1d1d import (
    blackwell_block_scaled_matmul_tma_umma_warp_specialized,
)
from linalg.matmul.gpu.sm100.config import BlockScaledMatmulConfig
from linalg.matmul.gpu.sm100_structured.grouped_block_scaled_1d1d import (
    grouped_matmul_1d1d_nvfp4,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    BlockScaledMatmulConfig as StructuredBlockScaledMatmulConfig,
)
from math import ceildiv, align_up
from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple
from linalg.fp4_utils import (
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
    NVFP4_SF_DTYPE,
    NVFP4_SF_VECTOR_SIZE,
    set_scale_factor,
)
from random import random_ui64, seed, rand
from builtin.simd import _convert_f32_to_float8_scalar
from layout import (
    LayoutTensor,
    Layout,
    RuntimeLayout,
    RuntimeTuple,
    IntTuple,
    UNKNOWN_VALUE,
)
from gpu.compute.arch.mma_nvidia_sm100 import UMMAKind
from layout._tile_tensor import TileTensor
from layout._coord import Coord, Idx, RuntimeInt
from layout._layout import row_major


fn simple_init() -> Bool:
    for arg in argv():
        if arg == "--simple-init":
            return True
    return False


def _test_kernel_impl[
    kernel_type: String,  # "old" or "new"
    a_type: DType,
    b_type: DType,
    c_type: DType,
    scales_dtype: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cluster_shape: StaticTuple[Int32, 3],
    cta_group: Int,
    num_experts: Int,
    expert_shape: IndexList[2],
    transpose_b: Bool = True,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    block_swizzle_size: Int = 0,
    benchmark: Bool = False,
    swapAB: Bool = False,
    k_group_size: Int = 1,
    SF_VECTOR_SIZE: Int = NVFP4_SF_VECTOR_SIZE,
](
    num_active_experts: Int,
    num_tokens_by_expert: List[Int],
    expert_ids: List[Int],
    ctx: DeviceContext,
):
    seed(1234)
    total_num_tokens = 0
    for i in range(len(num_tokens_by_expert)):
        total_num_tokens += num_tokens_by_expert[i]

    var M = total_num_tokens
    var N = expert_shape[0]
    var K = expert_shape[1]

    print(
        String(
            "[",
            kernel_type,
            " kernel] ",
            "in/out dtypes=(",
            a_type,
            ", ",
            b_type,
            ", ",
            c_type,
            ", ",
            scales_dtype,
            ") ",
            " problem shape=(",
            M,
            ", ",
            N,
            ", ",
            K,
            ") ",
            "mma_shape=",
            mma_shape,
            " block_tile_shape=",
            block_tile_shape,
            " cta_group=",
            cta_group,
            " cluster_shape=(",
            cluster_shape[0],
            ", ",
            cluster_shape[1],
            ", ",
            cluster_shape[2],
            ")",
            " swapAB=",
            swapAB,
            " k_group_size=",
            k_group_size,
            " SF_VECTOR_SIZE=",
            SF_VECTOR_SIZE,
        )
    )

    comptime static_a_shape = DimList(Dim(), expert_shape[1] // 2)
    comptime static_b_shape = DimList(
        num_experts, expert_shape[0], expert_shape[1] // 2
    )
    comptime static_c_shape = DimList(Dim(), expert_shape[0])
    var dynamic_a_shape = DimList(total_num_tokens, K // 2)
    var dynamic_b_shape = DimList(
        num_experts, expert_shape[0], expert_shape[1] // 2
    )
    var dynamic_c_shape = DimList(total_num_tokens, expert_shape[0])

    var a_size = total_num_tokens * K // 2
    var b_size = num_experts * expert_shape[0] * expert_shape[1] // 2
    var c_size = total_num_tokens * expert_shape[0]

    var a_host_ptr = UnsafePointer[Scalar[a_type]].alloc(a_size)
    var a_host = NDBuffer[a_type, 2, _, static_a_shape](
        a_host_ptr, dynamic_a_shape
    )
    var b_host_ptr = UnsafePointer[Scalar[b_type]].alloc(b_size)
    var b_host = NDBuffer[b_type, 3, _, static_b_shape](
        b_host_ptr, dynamic_b_shape
    )
    var c_host_ptr = UnsafePointer[Scalar[c_type]].alloc(c_size)
    var c_host = NDBuffer[c_type, 2, _, static_c_shape](
        c_host_ptr, dynamic_c_shape
    )
    var c_host_ref_ptr = UnsafePointer[Scalar[c_type]].alloc(c_size)
    var c_host_ref = NDBuffer[c_type, 2, _, static_c_shape](
        c_host_ref_ptr, dynamic_c_shape
    )

    var a_device = ctx.enqueue_create_buffer[a_type](a_size)
    var a_device_nd = NDBuffer[a_type, 2, _, static_a_shape](
        a_device.unsafe_ptr(), dynamic_a_shape
    )
    var a_offsets_device = ctx.enqueue_create_buffer[DType.uint32](
        num_active_experts + 1
    )
    var a_offsets_device_nd = NDBuffer[DType.uint32, 1](
        a_offsets_device.unsafe_ptr(), num_active_experts + 1
    )
    var b_device = ctx.enqueue_create_buffer[b_type](b_size)
    var b_device_nd = NDBuffer[b_type, 3, _, static_b_shape](
        b_device.unsafe_ptr(), dynamic_b_shape
    )
    var expert_ids_device = ctx.enqueue_create_buffer[DType.int32](
        num_active_experts
    )
    var expert_ids_device_nd = NDBuffer[DType.int32, 1](
        expert_ids_device.unsafe_ptr(), num_active_experts
    )
    var a_scale_offsets_device = ctx.enqueue_create_buffer[DType.uint32](
        num_active_experts
    )
    var a_scale_offsets_device_nd = NDBuffer[DType.uint32, 1](
        a_scale_offsets_device.unsafe_ptr(), num_active_experts
    )
    var expert_scales_device = ctx.enqueue_create_buffer[DType.float32](
        num_experts
    )
    var expert_scales_device_nd = NDBuffer[DType.float32, 1](
        expert_scales_device.unsafe_ptr(), num_experts
    )
    var c_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_device_nd = NDBuffer[c_type, 2, _, static_c_shape](
        c_device.unsafe_ptr(), dynamic_c_shape
    )
    var c_device_ref = ctx.enqueue_create_buffer[c_type](c_size)
    var c_device_ref_nd = NDBuffer[c_type, 2, _, static_c_shape](
        c_device_ref.unsafe_ptr(), dynamic_c_shape
    )

    var a_offsets_host_ptr = UnsafePointer[Scalar[DType.uint32]].alloc(
        num_active_experts + 1
    )
    var a_scale_offsets_ptr = UnsafePointer[Scalar[DType.uint32]].alloc(
        num_active_experts
    )
    var expert_ids_host_ptr = UnsafePointer[Scalar[DType.int32]].alloc(
        num_experts
    )
    var expert_scales_host_ptr = UnsafePointer[Scalar[DType.float32]].alloc(
        num_experts
    )
    # Initialize expert_scales to non-trivial values: 1 + (i+1)/num_experts
    for i in range(num_experts):
        expert_scales_host_ptr[i] = 1.0 + Float32(i + 1) / Float32(num_experts)

    a_scale_dim0 = 0
    a_offsets_host_ptr[0] = 0
    for i in range(num_active_experts):
        a_scale_offsets_ptr[i] = UInt32(
            a_scale_dim0
            - Int(a_offsets_host_ptr[i] // UInt32(SF_MN_GROUP_SIZE))
        )
        local_m = num_tokens_by_expert[i]
        a_offsets_host_ptr[i + 1] = a_offsets_host_ptr[i] + UInt32(local_m)
        a_scale_dim0 += ceildiv(local_m, SF_MN_GROUP_SIZE)
        expert_ids_host_ptr[i] = Int32(expert_ids[i])

    comptime static_a_scales_shape = DimList(
        # ceildiv(total_num_tokens, SF_MN_GROUP_SIZE),
        Dim(),
        ceildiv(expert_shape[1], SF_VECTOR_SIZE * SF_ATOM_K),
        Dim(SF_ATOM_M[0]),
        Dim(SF_ATOM_M[1]),
        Dim(SF_ATOM_K),
    )
    comptime static_b_scales_shape = DimList(
        num_experts,
        ceildiv(expert_shape[0], SF_MN_GROUP_SIZE),
        ceildiv(expert_shape[1], SF_VECTOR_SIZE * SF_ATOM_K),
        Dim(SF_ATOM_M[0]),
        Dim(SF_ATOM_M[1]),
        Dim(SF_ATOM_K),
    )

    var dynamic_a_scales_shape = DimList(
        a_scale_dim0,
        ceildiv(expert_shape[1], SF_VECTOR_SIZE * SF_ATOM_K),
        Dim(SF_ATOM_M[0]),
        Dim(SF_ATOM_M[1]),
        Dim(SF_ATOM_K),
    )
    var dynamic_b_scales_shape = DimList(
        num_experts,
        ceildiv(expert_shape[0], SF_MN_GROUP_SIZE),
        ceildiv(expert_shape[1], SF_VECTOR_SIZE * SF_ATOM_K),
        Dim(SF_ATOM_M[0]),
        Dim(SF_ATOM_M[1]),
        Dim(SF_ATOM_K),
    )

    var a_scales_total = (
        a_scale_dim0
        * ceildiv(expert_shape[1], SF_VECTOR_SIZE * SF_ATOM_K)
        * SF_ATOM_M[0]
        * SF_ATOM_M[1]
        * SF_ATOM_K
    )
    var b_scales_total = (
        num_experts
        * ceildiv(expert_shape[0], SF_MN_GROUP_SIZE)
        * ceildiv(expert_shape[1], SF_VECTOR_SIZE * SF_ATOM_K)
        * SF_ATOM_M[0]
        * SF_ATOM_M[1]
        * SF_ATOM_K
    )

    var a_scales_host_ptr = UnsafePointer[Scalar[scales_dtype]].alloc(
        a_scales_total
    )
    var a_scales_host = NDBuffer[scales_dtype, 5, _, static_a_scales_shape](
        a_scales_host_ptr, dynamic_a_scales_shape
    )
    var b_scales_host_ptr = UnsafePointer[Scalar[scales_dtype]].alloc(
        b_scales_total
    )
    var b_scales_host = NDBuffer[scales_dtype, 6, _, static_b_scales_shape](
        b_scales_host_ptr, dynamic_b_scales_shape
    )

    var a_scales_device = ctx.enqueue_create_buffer[scales_dtype](
        a_scales_total
    )
    var a_scales_device_nd = NDBuffer[
        scales_dtype, 5, _, static_a_scales_shape
    ](a_scales_device.unsafe_ptr(), dynamic_a_scales_shape)
    var b_scales_device = ctx.enqueue_create_buffer[scales_dtype](
        b_scales_total
    )
    var b_scales_device_nd = NDBuffer[
        scales_dtype, 6, _, static_b_scales_shape
    ](b_scales_device.unsafe_ptr(), dynamic_b_scales_shape)

    var a_tensor = from_ndbuffer_row_major(a_device_nd)
    var a_offsets_tensor = from_ndbuffer_row_major(a_offsets_device_nd)
    var a_scale_offsets_tensor = from_ndbuffer_row_major(
        a_scale_offsets_device_nd
    )
    var b_tensor = from_ndbuffer_row_major(b_device_nd)
    var expert_ids_tensor = from_ndbuffer_row_major(expert_ids_device_nd)
    var c_tensor = from_ndbuffer_row_major(c_device_nd)
    var a_scales_tensor = from_ndbuffer_row_major(a_scales_device_nd)
    var b_scales_tensor = from_ndbuffer_row_major(b_scales_device_nd)
    var c_ref_tensor = from_ndbuffer_row_major(c_device_ref_nd)

    # Initialize matmul operands
    if simple_init():
        for m in range(M):
            for k in range(K // 2):
                a_host[m, k] = UInt8(m).cast[a_type]()
        for e in range(num_experts):
            for n in range(N):
                for k in range(K // 2):
                    b_host[e, n, k] = UInt8(n).cast[b_type]()
    else:
        rand(a_host.data, a_host.num_elements(), min=0, max=255)
        rand(b_host.data, b_host.num_elements(), min=0, max=255)

    comptime scales_5d_layout[layout: Layout] = Layout.row_major(
        layout.shape[0].value(),
        layout.shape[1].value(),
        SF_ATOM_M[0],
        SF_ATOM_M[1],
        SF_ATOM_K,
    )
    comptime a_scales_5d_layout = scales_5d_layout[a_scales_tensor.layout]
    comptime b_scales_6d_layout = Layout.row_major(
        b_scales_tensor.layout.shape[0].value(),
        b_scales_tensor.layout.shape[1].value(),
        b_scales_tensor.layout.shape[2].value(),
        SF_ATOM_M[0],
        SF_ATOM_M[1],
        SF_ATOM_K,
    )

    var a_scales_tensor_host = LayoutTensor[
        scales_dtype, a_scales_5d_layout, MutAnyOrigin
    ](
        a_scales_host_ptr,
        RuntimeLayout[a_scales_5d_layout].row_major(
            IndexList[5](
                a_scales_host.dim(0),
                a_scales_host.dim(1),
                a_scales_host.dim(2),
                a_scales_host.dim(3),
                a_scales_host.dim(4),
            ),
        ),
    )

    var b_scales_tensor_host = LayoutTensor[
        scales_dtype, b_scales_6d_layout, MutAnyOrigin
    ](
        b_scales_host_ptr,
        RuntimeLayout[b_scales_6d_layout].row_major(
            IndexList[6](
                b_scales_host.dim(0),
                b_scales_host.dim(1),
                b_scales_host.dim(2),
                b_scales_host.dim(3),
                b_scales_host.dim(4),
                b_scales_host.dim(5),
            ),
        ),
    )

    for i in range(a_scales_host.num_elements()):
        a_scales_host.data[i] = Scalar[scales_dtype](0.0)
    rand(b_scales_host.data, b_scales_host.num_elements())
    # NOTE: It is very important that we set unused scales to 0.0 otherwise we will hit accuracy issues
    effective_n = expert_shape[0]
    effective_k = expert_shape[1]

    for i in range(num_active_experts):
        start = Int(a_offsets_host_ptr[i])
        end = Int(a_offsets_host_ptr[i + 1])
        local_m = end - start
        actual_start = (
            start // SF_MN_GROUP_SIZE + Int(a_scale_offsets_ptr[i])
        ) * SF_MN_GROUP_SIZE
        actual_end = actual_start + local_m
        for idx0 in range(actual_start, actual_end):
            for idx1 in range(
                0,
                align_up(effective_k, SF_VECTOR_SIZE * SF_ATOM_K),
                SF_VECTOR_SIZE,
            ):
                if idx1 < effective_k:
                    var scale_value = _convert_f32_to_float8_scalar[
                        scales_dtype
                    ]((1 << random_ui64(0, 2)).cast[DType.float32]())
                    set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                        a_scales_tensor_host, idx0, idx1, scale_value
                    )

    for e in range(num_experts):
        expert_slice_size = (
            b_scales_host.dim(1)
            * b_scales_host.dim(2)
            * b_scales_host.dim(3)
            * b_scales_host.dim(4)
            * b_scales_host.dim(5)
        )
        comptime b_scales_5d_layout = Layout.row_major(
            b_scales_tensor.layout.shape[1].value(),
            b_scales_tensor.layout.shape[2].value(),
            SF_ATOM_M[0],
            SF_ATOM_M[1],
            SF_ATOM_K,
        )
        var b_scales_tensor_expert_slice = LayoutTensor[
            scales_dtype, b_scales_5d_layout, MutAnyOrigin
        ](
            b_scales_host_ptr + e * expert_slice_size,
            RuntimeLayout[b_scales_5d_layout].row_major(
                IndexList[5](
                    b_scales_host.dim(1),
                    b_scales_host.dim(2),
                    b_scales_host.dim(3),
                    b_scales_host.dim(4),
                    b_scales_host.dim(5),
                ),
            ),
        )
        for idx0 in range(align_up(effective_n, SF_MN_GROUP_SIZE)):
            for idx1 in range(
                0,
                align_up(effective_k, SF_VECTOR_SIZE * SF_ATOM_K),
                SF_VECTOR_SIZE,
            ):
                if idx0 >= effective_n or idx1 >= effective_k:
                    set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                        b_scales_tensor_expert_slice,
                        idx0,
                        idx1,
                        Scalar[scales_dtype](0.0),
                    )

    # Move operands to the Device
    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(a_offsets_device, a_offsets_host_ptr)
    ctx.enqueue_copy(a_scale_offsets_device, a_scale_offsets_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(expert_ids_device, expert_ids_host_ptr)
    ctx.enqueue_copy(a_scales_device, a_scales_host_ptr)
    ctx.enqueue_copy(b_scales_device, b_scales_host_ptr)
    ctx.enqueue_copy(expert_scales_device, expert_scales_host_ptr)

    var expert_scales_tensor = from_ndbuffer_row_major(expert_scales_device_nd)

    # Call appropriate kernel based on kernel_type parameter
    @parameter
    if kernel_type == "old":
        # Old kernel using linalg.grouped_matmul_sm100_1d1d
        comptime matmul_config = BlockScaledMatmulConfig[
            a_type, b_type, c_type, scales_dtype, scales_dtype, transpose_b
        ](
            scaling_kind=UMMAKind.KIND_MXF4NVF4,
            cluster_shape=Index(
                cluster_shape[0], cluster_shape[1], cluster_shape[2]
            ),
            mma_shape=mma_shape,
            block_swizzle_size=block_swizzle_size,
            cta_group=cta_group,
            AB_swapped=swapAB,
            k_group_size=k_group_size,
            num_accum_pipeline_stages=1 if mma_shape[1] == 256 else 2,
        )

        blackwell_block_scaled_matmul_tma_umma_warp_specialized[
            transpose_b=transpose_b,
            config=matmul_config,
        ](
            c_tensor,
            a_tensor,
            a_offsets_tensor,
            a_scale_offsets_tensor,
            b_tensor,
            expert_ids_tensor,
            a_scales_tensor,
            b_scales_tensor,
            expert_scales_tensor,
            num_active_experts,
            ctx,
        )
    elif kernel_type == "new":
        # New structured kernel using grouped_matmul_1d1d_nvfp4
        comptime new_matmul_config = StructuredBlockScaledMatmulConfig[
            a_type, b_type, c_type, scales_dtype, scales_dtype, transpose_b
        ](
            scaling_kind=UMMAKind.KIND_MXF4NVF4,
            cluster_shape=Index(
                cluster_shape[0], cluster_shape[1], cluster_shape[2]
            ),
            mma_shape=mma_shape,
            block_swizzle_size=block_swizzle_size,
            cta_group=cta_group,
            AB_swapped=swapAB,
            k_group_size=k_group_size,
            num_accum_pipeline_stages=1 if mma_shape[1] == 256 else 2,
            c_swizzle_for_AB_swapped=TensorMapSwizzle.SWIZZLE_32B,
        )

        # Construct scale TileTensors from raw pointers with explicit
        # row_major layouts (NDBuffer→TileTensor doesn't work for 5D/6D).
        comptime k_groups = ceildiv(expert_shape[1], SF_VECTOR_SIZE * SF_ATOM_K)
        comptime n_groups = ceildiv(expert_shape[0], SF_MN_GROUP_SIZE)
        var a_scales_tt = TileTensor(
            a_scales_device.unsafe_ptr().bitcast[Scalar[scales_dtype]](),
            row_major(
                Coord(
                    RuntimeInt[DType.int64](Scalar[DType.int64](a_scale_dim0)),
                    Idx[k_groups](),
                    Idx[SF_ATOM_M[0]](),
                    Idx[SF_ATOM_M[1]](),
                    Idx[SF_ATOM_K](),
                )
            ),
        ).as_any_origin()
        var b_scales_tt = TileTensor(
            b_scales_device.unsafe_ptr().bitcast[Scalar[scales_dtype]](),
            row_major(
                Coord(
                    Idx[num_experts](),
                    Idx[n_groups](),
                    Idx[k_groups](),
                    Idx[SF_ATOM_M[0]](),
                    Idx[SF_ATOM_M[1]](),
                    Idx[SF_ATOM_K](),
                )
            ),
        ).as_any_origin()
        var expert_scales_tt = TileTensor(
            expert_scales_device.unsafe_ptr().bitcast[Scalar[DType.float32]](),
            row_major(
                Coord(
                    RuntimeInt[DType.int64](Scalar[DType.int64](num_experts)),
                )
            ),
        ).as_any_origin()

        grouped_matmul_1d1d_nvfp4[
            transpose_b=transpose_b,
            config=new_matmul_config,
        ](
            TileTensor(c_device_nd),
            TileTensor(a_device_nd),
            TileTensor(a_offsets_device_nd),
            TileTensor(a_scale_offsets_device_nd),
            TileTensor(b_device_nd),
            TileTensor(expert_ids_device_nd),
            a_scales_tt,
            b_scales_tt,
            expert_scales_tt,
            num_active_experts,
            ctx,
        )
    else:
        constrained[False, "kernel_type must be 'old' or 'new'"]()
        pass

    constrained[
        a_type != DType.float8_e4m3fn or transpose_b,
        (
            "Testing is only supported for transposed_b==True when"
            " a_type==float8_e4m3fn. Add the non-transposed case if needed."
        ),
    ]()

    comptime new_c_layout = Layout.row_major(UNKNOWN_VALUE, expert_shape[0])
    comptime new_a_layout = Layout.row_major(
        UNKNOWN_VALUE, expert_shape[1] // 2
    )
    comptime new_b_layout = Layout.row_major(
        expert_shape[0], expert_shape[1] // 2
    )
    comptime new_b_scales_layout = Layout.row_major(
        b_scales_tensor.layout.shape[1].value(),
        b_scales_tensor.layout.shape[2].value(),
        SF_ATOM_M[0],
        SF_ATOM_M[1],
        SF_ATOM_K,
    )
    comptime new_a_scales_layout = a_scales_tensor.layout

    for i in range(num_active_experts):
        start = Int(a_offsets_host_ptr[i])
        end = Int(a_offsets_host_ptr[i + 1])
        expert_id = expert_ids_host_ptr[i]

        if expert_id < 0 or end - start == 0:
            continue

        var c_stride = c_ref_tensor.runtime_layout.stride[0].get_int()
        var c_slice = LayoutTensor[c_type, new_c_layout, MutAnyOrigin](
            c_ref_tensor.ptr + start * Int(c_stride),
            RuntimeLayout[new_c_layout].row_major(
                IndexList[2](
                    end - start,
                    expert_shape[0],
                ),
            ),
        )

        var a_stride = a_tensor.runtime_layout.stride[0].get_int()
        var new_a_tensor = LayoutTensor[a_type, new_a_layout, MutAnyOrigin](
            a_tensor.ptr + start * Int(a_stride),
            RuntimeLayout[new_a_layout].row_major(
                IndexList[2](
                    end - start,
                    expert_shape[1] // 2,
                ),
            ),
        )

        comptime b_stride = b_tensor.layout.stride[0].value()
        var new_b_tensor = LayoutTensor[b_type, new_b_layout, MutAnyOrigin](
            b_tensor.ptr + expert_id * Int32(b_stride),
            RuntimeLayout[new_b_layout].row_major(
                IndexList[2](
                    b_tensor.dim(1),
                    b_tensor.dim(2),
                ),
            ),
        )

        comptime b_scales_stride = b_scales_tensor.layout.stride[0].value()
        var new_b_scales_tensor = LayoutTensor[
            scales_dtype, new_b_scales_layout, MutAnyOrigin
        ](
            b_scales_tensor.ptr + expert_id * Int32(b_scales_stride),
            RuntimeLayout[new_b_scales_layout].row_major(
                IndexList[5](
                    b_scales_host.dim(1),
                    b_scales_host.dim(2),
                    b_scales_host.dim(3),
                    b_scales_host.dim(4),
                    b_scales_host.dim(5),
                ),
            ),
        )

        var a_scales_start = start // SF_MN_GROUP_SIZE + Int(
            a_scale_offsets_ptr[i]
        )
        comptime a_scales_stride = a_scales_tensor.layout.stride[0].value()
        var new_a_scales_tensor = LayoutTensor[
            scales_dtype, new_a_scales_layout, MutAnyOrigin
        ](
            a_scales_tensor.ptr + a_scales_start * a_scales_stride,
            RuntimeLayout[new_a_scales_layout].row_major(
                IndexList[5](
                    ceildiv(end - start, SF_MN_GROUP_SIZE),
                    a_scales_host.dim(1),
                    a_scales_host.dim(2),
                    a_scales_host.dim(3),
                    a_scales_host.dim(4),
                ),
            ),
        )

        var expert_scale = expert_scales_host_ptr[Int(expert_id)]
        vendor_blas.matmul(
            ctx,
            c_slice,
            new_a_tensor,
            new_b_tensor,
            a_scales=new_a_scales_tensor.get_immutable(),
            b_scales=new_b_scales_tensor.get_immutable(),
            transpose_b=transpose_b,
            c_row_major=True,
            alpha=expert_scale,
        )

    ctx.synchronize()

    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)
    ctx.synchronize()

    # Zero output regions for skipped experts (expert_id == -1 or 0 tokens)
    # so both kernel and reference outputs match in those regions.
    for i in range(num_active_experts):
        start = Int(a_offsets_host_ptr[i])
        end = Int(a_offsets_host_ptr[i + 1])
        if expert_ids_host_ptr[i] < 0 or end - start == 0:
            for j in range(start * N, end * N):
                c_host_ptr[j] = Scalar[c_type](0)
                c_host_ref_ptr[j] = Scalar[c_type](0)

    assert_almost_equal(
        c_host.data,
        c_host_ref.data,
        c_host.num_elements(),
        atol=1e-2,
        rtol=1e-2,
    )
    print("\n=== TEST PASSED ===\n")

    # Cleanup
    a_host_ptr.free()
    b_host_ptr.free()
    c_host_ptr.free()
    c_host_ref_ptr.free()
    a_scales_host_ptr.free()
    b_scales_host_ptr.free()
    a_offsets_host_ptr.free()
    a_scale_offsets_ptr.free()
    expert_ids_host_ptr.free()
    expert_scales_host_ptr.free()
    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_device_ref^
    _ = a_scales_device^
    _ = b_scales_device^
    _ = a_offsets_device^
    _ = a_scale_offsets_device^
    _ = expert_ids_device^
    _ = expert_scales_device^


# Backward-compatible wrapper that maintains the original function name
def test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    scales_dtype: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cluster_shape: StaticTuple[Int32, 3],
    cta_group: Int,
    num_experts: Int,
    expert_shape: IndexList[2],
    transpose_b: Bool = True,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    block_swizzle_size: Int = 0,
    benchmark: Bool = False,
    swapAB: Bool = False,
    k_group_size: Int = 1,
    SF_VECTOR_SIZE: Int = NVFP4_SF_VECTOR_SIZE,
](
    num_active_experts: Int,
    num_tokens_by_expert: List[Int],
    expert_ids: List[Int],
    ctx: DeviceContext,
):
    """Test old kernel - backward compatible wrapper."""
    _test_kernel_impl[
        "old",
        a_type,
        b_type,
        c_type,
        scales_dtype,
        block_tile_shape,
        mma_shape,
        cluster_shape,
        cta_group,
        num_experts,
        expert_shape,
        transpose_b,
        a_swizzle,
        b_swizzle,
        c_swizzle,
        block_swizzle_size,
        benchmark,
        swapAB,
        k_group_size,
        SF_VECTOR_SIZE,
    ](num_active_experts, num_tokens_by_expert, expert_ids, ctx)


def main():
    with DeviceContext() as ctx:
        comptime dtype = DType.uint8  # TODO: (KERN-2238): Replace with float4-e2m1fn
        comptime out_dtype = DType.bfloat16
        comptime scale_dtype = NVFP4_SF_DTYPE
        comptime swizzle = TensorMapSwizzle.SWIZZLE_128B
        comptime BK = (swizzle.bytes() // size_of[dtype]())
        comptime MMA_K = 32
        comptime bm = 128
        comptime bn = 128
        comptime block_tile_shape = Index(bm, bn, BK)
        comptime umma_shape = Index(bm, bn, MMA_K)

        @parameter
        for structured in [False, True]:

            @parameter
            if structured:
                print("\n========================================")
                print("Testing NEW kernel (grouped_matmul_1d1d_nvfp4)")
                print("========================================\n")
            else:
                print("\n========================================")
                print(
                    "Testing OLD kernel"
                    " (blackwell_block_scaled_matmul_tma_umma_warp_specialized)"
                )
                print("========================================\n")

            comptime kernel_type = "new" if structured else "old"

            @parameter
            for swapAB in [False, True]:
                # Large token counts
                _test_kernel_impl[
                    kernel_type,
                    dtype,
                    dtype,
                    out_dtype,
                    scale_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape = StaticTuple[Int32, 3](1, 1, 1),
                    cta_group=1,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=8,
                    num_experts=6,
                    expert_shape = Index(2048, 1024),
                    swapAB=swapAB,
                ](
                    4,
                    [512, 1000, 2000, 3000],
                    [0, 3, 2, 4],
                    ctx,
                )

                # Unaligned token counts
                _test_kernel_impl[
                    kernel_type,
                    dtype,
                    dtype,
                    out_dtype,
                    scale_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape = StaticTuple[Int32, 3](1, 1, 1),
                    cta_group=1,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=8,
                    num_experts=4,
                    expert_shape = Index(2048, 1024),
                    swapAB=swapAB,
                ](
                    3,
                    [64 + 1, 1024 + 3, 128 * 3 + 2],
                    [2, 0, 1],
                    ctx,
                )

                # Aligned token counts
                _test_kernel_impl[
                    kernel_type,
                    dtype,
                    dtype,
                    out_dtype,
                    scale_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape = StaticTuple[Int32, 3](1, 1, 1),
                    cta_group=1,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=8,
                    num_experts=4,
                    expert_shape = Index(2048, 1024),
                    swapAB=swapAB,
                ](
                    3,
                    [128, 256, 1024],
                    [2, 0, 1],
                    ctx,
                )

                # Mixed aligned/unaligned per-expert token counts
                _test_kernel_impl[
                    kernel_type,
                    dtype,
                    dtype,
                    out_dtype,
                    scale_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape = StaticTuple[Int32, 3](1, 1, 1),
                    cta_group=1,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=8,
                    num_experts=6,
                    expert_shape = Index(2048, 1024),
                    swapAB=swapAB,
                ](
                    4,
                    [256, 512 + 7, 1024 + 13, 128 + 1],
                    [0, 3, 2, 4],
                    ctx,
                )

                # Just-off-alignment: 128-1, 256+1, 512+1, 1024+1
                _test_kernel_impl[
                    kernel_type,
                    dtype,
                    dtype,
                    out_dtype,
                    scale_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape = StaticTuple[Int32, 3](1, 1, 1),
                    cta_group=1,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=8,
                    num_experts=6,
                    expert_shape = Index(2048, 1024),
                    swapAB=swapAB,
                ](
                    4,
                    [127, 257, 513, 1025],
                    [0, 3, 2, 4],
                    ctx,
                )

                # Small token counts (total tiles < SM count)
                _test_kernel_impl[
                    kernel_type,
                    dtype,
                    dtype,
                    out_dtype,
                    scale_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape = StaticTuple[Int32, 3](1, 1, 1),
                    cta_group=1,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=8,
                    num_experts=4,
                    expert_shape = Index(2048, 1024),
                    swapAB=swapAB,
                ](
                    3,
                    [31, 97, 63],
                    [2, 0, 1],
                    ctx,
                )

                # Very small token counts (common MoE case)
                _test_kernel_impl[
                    kernel_type,
                    dtype,
                    dtype,
                    out_dtype,
                    scale_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape = StaticTuple[Int32, 3](1, 1, 1),
                    cta_group=1,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=8,
                    num_experts=6,
                    expert_shape = Index(2048, 1024),
                    swapAB=swapAB,
                ](
                    4,
                    [0, 1, 2, 3],
                    [0, 3, 2, 4],
                    ctx,
                )

                # -1 expert_id (invalid expert skipped by kernel)
                _test_kernel_impl[
                    kernel_type,
                    dtype,
                    dtype,
                    out_dtype,
                    scale_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape = StaticTuple[Int32, 3](1, 1, 1),
                    cta_group=1,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=8,
                    num_experts=4,
                    expert_shape = Index(2048, 1024),
                    swapAB=swapAB,
                ](
                    3,
                    [128, 256, 512],
                    [-1, 0, 2],
                    ctx,
                )

                # -1 expert_id with very small token counts
                _test_kernel_impl[
                    kernel_type,
                    dtype,
                    dtype,
                    out_dtype,
                    scale_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape = StaticTuple[Int32, 3](1, 1, 1),
                    cta_group=1,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=8,
                    num_experts=6,
                    expert_shape = Index(2048, 1024),
                    swapAB=swapAB,
                ](
                    4,
                    [0, 3, 1, 2],
                    [-1, 2, -1, 0],
                    ctx,
                )

        print("\n========================================")
        print("ALL TESTS PASSED!")
        print("========================================")
