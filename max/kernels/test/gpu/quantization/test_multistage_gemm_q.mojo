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

from collections.optional import OptionalReg
from math import ceildiv, isclose
from pathlib import Path
from random import rand, randint, random_float64, seed
from sys import alignof, argv, simdwidthof, sizeof
from sys._assembly import inlined_assembly

from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    barrier,
    block_idx,
    grid_dim,
    lane_id,
    thread_idx,
)
from gpu.host import DeviceContext, FuncAttribute
from gpu.host.info import DEFAULT_GPU_ARCH
from gpu.intrinsics import lop
from gpu.memory import (
    AddressSpace,
    async_copy_commit_group,
    async_copy_wait_group,
    external_memory,
)
from gpu.mma import ld_matrix, mma
from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    arange,
    assert_almost_equal,
    assert_equal,
    fill,
    random,
    zero,
)
from internal_utils._utils import ValOrDim, dynamic, static
from layout import RuntimeLayout
from layout._ndbuffer_stub import from_ndbuffer_row_major
from layout.int_tuple import IntTuple
from layout.layout import *
from layout.layout_tensor import (
    LayoutTensor,
    LayoutTensorIter,
    _swizzle_signature,
    copy,
    copy_dram_to_sram,
    copy_dram_to_sram_async,
    copy_local_to_dram,
    copy_local_to_local,
    copy_sram_to_dram,
)
from layout.runtime_tuple import RuntimeTuple
from layout.swizzle import Swizzle, make_swizzle
from layout.tensor_builder import LayoutTensorBuild as tb
from layout.tensor_core import TensorCore, get_fragment_size, get_mma_shape
from linalg.matmul_gpu import _matmul_gpu, multistage_gemm
from linalg.utils_gpu import (
    MatmulConfig,
    MatmulKernels,
    block_swizzle,
    select_config,
)
from memory import UnsafePointer
from memory.unsafe import bitcast
from quantization import Q4sym
from quantization.qmatmul_gpu import (
    multistage_gemm_q,
    pack_Q_tile,
    q_smem_usage,
)

from utils import StaticTuple
from utils.index import Index, IndexList
from utils.numerics import get_accum_type


fn is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark" or arg == "-benchmark":
            return True
    return False


@always_inline
fn args_to_tuple[swap: Bool](arg_0: Int, arg_1: Int) -> Tuple[Int, Int]:
    @parameter
    if swap:
        return Tuple(arg_1, arg_0)
    else:
        return Tuple(arg_0, arg_1)


fn repack_Q4_0_for_sm8x[
    q_layout: Layout,
    repack_layout: Layout,
    scales_type: DType,
](
    q_weight: LayoutTensor[DType.uint8, q_layout, MutableAnyOrigin],
    q_packed_weight: LayoutTensor[DType.uint8, repack_layout, MutableAnyOrigin],
):
    alias group_size = 32
    alias group_bytes = sizeof[DType.float16]() + (group_size // 2)
    alias pack_factor = 8
    alias repack_tile = Index(64, 16)
    alias WARP_SIZE = 32
    alias BN = 128
    alias BK = 1024

    var tid: UInt = thread_idx.x
    var warp_id: UInt = tid // WARP_SIZE
    alias num_warps_x = BN // repack_tile[0]
    var warp_x: UInt = warp_id % num_warps_x
    var warp_y: UInt = warp_id // num_warps_x
    var lane_id: Int = tid % WARP_SIZE
    var block_idx = Index(Int(block_idx.x), Int(block_idx.y))

    alias N = Int(q_layout.shape[0])
    alias K = Int(q_layout.shape[1]) // group_bytes * group_size

    alias K_groups = K // group_size
    alias BK_groups = BK // group_size

    alias uint_K = K // pack_factor
    alias uint_BK = BK // pack_factor

    @always_inline
    @parameter
    fn convert_bytes_to_bf16[
        scales_type: DType
    ](input_bytes: SIMD[DType.uint8, _]) -> SIMD[scales_type, 1]:
        var f32_values = bitcast[DType.float16, 1](input_bytes).cast[
            DType.float32
        ]()
        return bitcast[scales_type, 2](f32_values)[1]

    alias repacked_b_layout = Layout(
        IntTuple(
            IntTuple(64, N // 64),
            IntTuple(2, uint_K // 2),
        ),
        IntTuple(
            IntTuple(2, 128 * (uint_K // 2)),
            IntTuple(1, 128),
        ),
    )
    var repack_weights = LayoutTensor[DType.uint32, repacked_b_layout](
        q_packed_weight.ptr.bitcast[UInt32](),
    )

    alias b_scales_layout = Layout.row_major(K_groups, N)
    var b_scales_ptr = q_packed_weight.ptr + N * K // 2
    var repack_scales = LayoutTensor[scales_type, b_scales_layout](
        b_scales_ptr.bitcast[Scalar[scales_type]](),
    )

    # We keep 128x2 Q4_0 GGUF blocks in smem
    var smem = external_memory[
        UInt8,
        address_space = AddressSpace.SHARED,
        alignment = alignof[SIMD[DType.uint8, 1]](),
    ]()
    var qb_smem = LayoutTensor[
        DType.uint8,
        Layout.row_major(BN, 2 * group_bytes),
        address_space = AddressSpace.SHARED,
    ](smem.bitcast[UInt8]())

    var q_gmem_tile = q_weight.tile[BN, BK_groups * group_bytes](
        block_idx[0], block_idx[1]
    )
    var q_gmem_iter = q_gmem_tile.tiled_iterator[BN, 2 * group_bytes, axis=1](
        0, 0
    )

    var repacked_gmem_tile = repack_weights.tile[BN, uint_BK](
        block_idx[0], block_idx[1]
    )
    var repacked_gemm_iter = repacked_gmem_tile.tiled_iterator[
        BN, 2 * group_size // pack_factor, axis=1
    ](0, 0)

    var scales_gmem_tile = repack_scales.tile[BK_groups, BN](
        block_idx[1], block_idx[0]
    )
    var scales_gmem_iter = scales_gmem_tile.tiled_iterator[2, BN, axis=0](0, 0)

    # We load 128x2 Q4_0 GGUF blocks to smem.
    # Each warp repacks 64x1 Q4_0 GGUF blocks, which are
    # 64x32 4-bit weights. We repack weights into 64x16
    # tiles for our quantized matmul kernel, so there are
    # two tile for each warp.
    # frag_0 stores frags of the first 64x16 tile,
    # frag_1 stores frags of the second,
    for i in range(ceildiv(BK_groups, 2)):
        barrier()
        copy_dram_to_sram[thread_layout = Layout.row_major(128, 1)](
            qb_smem.vectorize[1, 4](),
            q_gmem_iter[]
            .bitcast[DType.uint8, address_space = AddressSpace.GENERIC]()
            .vectorize[1, 4](),
        )
        q_gmem_iter._incr()
        barrier()
        q_warp_tile = qb_smem.tile[repack_tile[0], group_bytes](warp_x, warp_y)

        if (BK_groups * block_idx[1] + i * 2 + warp_y) < K_groups:
            var frag_0: SIMD[DType.uint8, 16] = 0
            var frag_1: SIMD[DType.uint8, 16] = 0
            var raw_Q_tile = q_warp_tile.tile[repack_tile[0], group_bytes]()
            alias thd_layout = Layout.row_major(8, 4)
            var thread_tile = raw_Q_tile.slice[:, 2:]().vectorize[
                1, 2
            ]().distribute[thd_layout](lane_id)

            @parameter
            for i_ele in range(16):
                var val = thread_tile.load[2](i_ele // 2, i_ele % 2)
                frag_0[i_ele] = (val[0] & 0x0F) | ((val[1] & 0x0F) << 4)
                frag_1[i_ele] = ((val[0] & 0xF0) >> 4) | (val[1] & 0xF0)

            var repack_warp_tile = repacked_gemm_iter[].tile[
                64, group_size // pack_factor
            ](warp_x, warp_y)
            # The repack_warp_tile is of shape [64, (2, 2)]. In this case,
            # elements [0, 0], [0, 1], [1, 0] and [1, 1] are stored continuously
            # in the memory. We need to use a element shape of [2, 2] to
            # correctly vectorize this tensor.
            repack_warp_tile.vectorize[2, 2]().store(
                lane_id, 0, pack_Q_tile(frag_0)
            )
            repack_warp_tile.vectorize[2, 2]().store(
                lane_id, 1, pack_Q_tile(frag_1)
            )
            repacked_gemm_iter._incr()

            alias scales_thread_layout = Layout(
                IntTuple(4, 8),
                IntTuple(16, 1),
            )
            var rt_scales_thread_layout = RuntimeLayout[scales_thread_layout]()

            # cast scales to bf16 before storing back
            var scales_warp_tile = scales_gmem_iter[].tile[1, 64](
                warp_y, warp_x
            )

            scales_warp_tile[0, 2 * lane_id] = convert_bytes_to_bf16[
                scales_type
            ](
                q_warp_tile.vectorize[1, 2]()[
                    Int(rt_scales_thread_layout(lane_id)), 0
                ]
            )

            scales_warp_tile[0, 2 * lane_id + 1] = convert_bytes_to_bf16[
                scales_type
            ](
                q_warp_tile.vectorize[1, 2]()[
                    Int(rt_scales_thread_layout(lane_id)) + 8, 0
                ]
            )

            scales_gmem_iter._incr()


# this kernel dequantizes a repacked INT4 matrix into bf16 format
# Assuming a 64x16 (nxk) packing scheme
# Tile [i, j] stores part of the original matrix [i*64:(i+1)*64, j*16:(j+1)*16]
# Within each tile, weights are repacked similarly to the Marlin kernel.
# The memory address for tile [i, j] is (i * (N//64) + j) * tile_size,
# where tile_size is 64 * 16 * 4 / pack_factor = 512 Bytes.
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](128))
fn create_ref_b[
    type_q: DType,
    type_b: DType,
    b_q_layout: Layout,
    b_layout: Layout,
    group_size: Int,
    pack_factor: Int,
](
    b_packed: LayoutTensor[type_q, b_q_layout, MutableAnyOrigin],
    b_out: LayoutTensor[type_b, b_layout, MutableAnyOrigin],
):
    alias WARP_SIZE = 32
    alias BLOCK_N = 128
    alias BLOCK_K = 32
    alias repack_tile = Index(64, 16)
    alias TILE_N = 64
    alias TILE_K = 16
    alias num_k_warps = BLOCK_K // repack_tile[1]

    var tid: UInt = thread_idx.x
    var warp_id: UInt = tid // WARP_SIZE
    var lane_id: UInt = tid % WARP_SIZE
    var block_idx = Index(Int(block_idx.x), Int(block_idx.y))
    var warp_x: UInt = warp_id // num_k_warps
    var warp_y: UInt = warp_id % num_k_warps

    alias group_bytes = group_size // 2 + 2
    alias N = Int(b_q_layout.shape[0])
    alias K = Int(b_q_layout.shape[1]) // group_bytes * group_size

    # Unpack quantized weights
    alias scales_type = DType.bfloat16
    alias b_type = DType.uint32
    alias b_weight_layout = Layout.row_major(N // 64, K * 64 // pack_factor)
    var b_q = LayoutTensor[b_type, b_weight_layout](
        b_packed.ptr.bitcast[Scalar[b_type]](),
    )

    alias b_scales_layout = Layout.row_major(K // group_size, N)
    var b_scales_ptr = b_packed.ptr + N * K // 2
    var scales = LayoutTensor[scales_type, b_scales_layout](
        b_scales_ptr.bitcast[Scalar[scales_type]](),
    )

    var b_q_gmem_tile = b_q.tile[
        BLOCK_N // repack_tile[0], (BLOCK_K * repack_tile[0]) // pack_factor
    ](block_idx[0], block_idx[1])
    var warp_q_tile = b_q_gmem_tile.tile[
        1, (repack_tile[0] * repack_tile[1]) // pack_factor
    ](warp_x, warp_y)

    var scales_tile = scales.tile[ceildiv(BLOCK_K, group_size), BLOCK_N](
        (block_idx[1] * BLOCK_K) // group_size, block_idx[0]
    )
    var warp_scales_tile = scales_tile.tile[
        ceildiv(BLOCK_K, group_size), repack_tile[0]
    ](0, warp_x)
    alias smem_reg_scales_layout = Layout.row_major(8, 4)
    var scales_reg_tiles = tb[scales_type]().row_major[
        repack_tile[0] // 8, 1
    ]().local().alloc().vectorize[1, 1]()
    # load scales
    scales_reg_tiles.vectorize[8, 1]().copy_from(
        warp_scales_tile.vectorize[1, 8]().distribute[
            smem_reg_scales_layout, axis=0
        ](Int(lane_id))
    )

    var b_out_tile = b_out.tile[BLOCK_N, BLOCK_K](block_idx[0], block_idx[1])
    var warp_out_tile = b_out_tile.tile[repack_tile[0], repack_tile[1]](
        warp_x, warp_y
    )
    var mma_tile_iter_1 = warp_out_tile.tiled_iterator[8, 8, axis=0](0, 0)
    var mma_tile_iter_2 = warp_out_tile.tiled_iterator[8, 8, axis=0](0, 1)

    var vec = bitcast[DType.int32, 4](warp_q_tile.vectorize[1, 4]()[lane_id])

    @always_inline
    fn int4tobf16(
        i4: Int32, scale: SIMD[DType.bfloat16, 1]
    ) -> SIMD[DType.bfloat16, 2]:
        alias MASK: Int32 = 0x000F000F
        alias I4s_TO_BF16s_MAGIC_NUM: Int32 = 0x43004300
        alias lut: Int32 = (0xF0 & 0xCC) | 0xAA
        var BF16_BIAS = SIMD[DType.bfloat16, 2](-136, -136)
        var BF16_SCALE = SIMD[DType.bfloat16, 2](scale, scale)
        var BF16_ZERO = SIMD[DType.bfloat16, 2](0, 0)
        var BF16_ONE = SIMD[DType.bfloat16, 2](1, 1)

        var t = lop[lut](i4, MASK, I4s_TO_BF16s_MAGIC_NUM)

        var v = bitcast[DType.bfloat16, 2](t).fma(BF16_ONE, BF16_BIAS).fma(
            BF16_SCALE, BF16_ZERO
        )
        return v

    alias write_back_layout = Layout.row_major(1, 32)
    alias write_back_type = __type_of(mma_tile_iter_1[].vectorize[1, 2]()[0])

    @parameter
    for i in range(0, TILE_N // 8, 2):
        var q_int = vec[i // 2]

        var v1 = int4tobf16(
            q_int, bitcast[DType.bfloat16, 1](scales_reg_tiles[i])
        )
        mma_tile_iter_1[].vectorize[1, 2]()[lane_id // 4, lane_id % 4] = rebind[
            write_back_type
        ](v1)
        q_int >>= 4
        var v2 = int4tobf16(
            q_int, bitcast[DType.bfloat16, 1](scales_reg_tiles[i])
        )
        mma_tile_iter_2[].vectorize[1, 2]()[lane_id // 4, lane_id % 4] = rebind[
            write_back_type
        ](v2)
        q_int >>= 4
        mma_tile_iter_1._incr()
        mma_tile_iter_2._incr()

        v1 = int4tobf16(
            q_int, bitcast[DType.bfloat16, 1](scales_reg_tiles[i + 1])
        )
        mma_tile_iter_1[].vectorize[1, 2]()[lane_id // 4, lane_id % 4] = rebind[
            write_back_type
        ](v1)
        q_int >>= 4
        v2 = int4tobf16(
            q_int, bitcast[DType.bfloat16, 1](scales_reg_tiles[i + 1])
        )
        mma_tile_iter_2[].vectorize[1, 2]()[lane_id // 4, lane_id % 4] = rebind[
            write_back_type
        ](v2)
        mma_tile_iter_1._incr()
        mma_tile_iter_2._incr()


fn random_float16(min: Float64 = 0, max: Float64 = 1) -> Float16:
    # Avoid pulling in a __truncdfhf2 dependency for a float64->float16
    # conversion by casting through float32 first.
    return (
        random_float64(min=min, max=max)
        .cast[DType.float32]()
        .cast[DType.float16]()
    )


struct _block_Q4_0:
    alias group_size = 32

    var base_scale: Float16
    var q_bits: InlineArray[UInt8, Self.group_size // 2]


fn test_repack_Q4_0_for_sm8x(
    ctx: DeviceContext, n: ValOrDim, k: ValOrDim
) raises:
    print("test repack_Q4_0_for_sm8x")

    fn fill_random[type: DType](array: InlineArray[Scalar[type]]):
        rand(array.unsafe_ptr(), len(array), min=0, max=255)

    fn build_b_buffer(N: Int, K: Int, b_ptr: UnsafePointer[UInt8]):
        var k_groups = ceildiv(K, 32)
        var block_ptr = b_ptr.bitcast[_block_Q4_0]()

        for n in range(N):
            for k in range(k_groups):
                block_ptr[].base_scale = random_float16()
                fill_random(block_ptr[].q_bits)
                block_ptr += 1

    alias group_size = 32
    alias pack_factor = 8
    var N = n.value
    var K = k.value
    alias BN = 128
    alias BK = 1024
    alias group_bytes = 2 + (group_size // 2)

    alias static_gguf_b_shape = DimList(
        n.dim, (k.dim // group_size) * group_bytes
    )
    alias static_repacked_b_shape = DimList(
        n.dim, (k.dim // group_size) * group_bytes
    )
    alias static_dequan_shape = DimList(k.dim, n.dim)

    var dynamic_gguf_b_shape = DimList(
        n.value, (k.value // group_size) * group_bytes
    )
    var dynamic_repacked_b_shape = DimList(
        n.value, (k.value // group_size) * group_bytes
    )
    var dynamic_dequan_shape = DimList(k.value, n.value)

    var gguf_b_host = HostNDBuffer[DType.uint8, 2](dynamic_gguf_b_shape)
    var repacked_b_host = HostNDBuffer[DType.uint8, 2, static_repacked_b_shape](
        dynamic_repacked_b_shape
    )
    var gguf_dequan_ref_host = HostNDBuffer[DType.bfloat16, 2](
        dynamic_dequan_shape
    )
    var repacked_dequan_host = HostNDBuffer[DType.bfloat16, 2](
        dynamic_dequan_shape
    )

    zero(repacked_b_host.tensor)
    build_b_buffer(N, K, gguf_b_host.tensor.data)
    Q4sym[group_size, DType.bfloat16].dequantize_and_write_to_tensor(
        gguf_b_host.tensor,
        gguf_dequan_ref_host.tensor,
        gguf_dequan_ref_host.tensor.get_shape(),
    )
    zero(repacked_dequan_host.tensor)

    var gguf_b_device = DeviceNDBuffer[DType.uint8, 2, static_gguf_b_shape](
        dynamic_gguf_b_shape, ctx=ctx
    )
    var repacked_b_device = DeviceNDBuffer[
        DType.uint8, 2, static_repacked_b_shape
    ](dynamic_repacked_b_shape, ctx=ctx)
    var repacked_dequan_device = DeviceNDBuffer[
        DType.bfloat16, 2, static_dequan_shape
    ](dynamic_dequan_shape, ctx=ctx)

    ctx.enqueue_copy(gguf_b_device.buffer, gguf_b_host.tensor.data)
    ctx.enqueue_copy(repacked_b_device.buffer, repacked_b_host.tensor.data)

    alias gguf_b_layout = Layout.row_major[gguf_b_device.rank](
        gguf_b_device.shape
    )
    alias repacked_b_layout = Layout.row_major[repacked_b_device.rank](
        repacked_b_device.shape
    )
    alias repack_dequan_layout = Layout.row_major[repacked_dequan_device.rank](
        repacked_dequan_device.shape
    )
    alias repacked_b_old_layout = Layout.row_major(
        Int(n.dim) // 64,
        Int(k.dim) * 64 // pack_factor,
    )
    alias gguf_b_tensor_type = LayoutTensor[DType.uint8, gguf_b_layout]
    alias repacked_dequan_tensor_type = LayoutTensor[
        DType.bfloat16,
        repack_dequan_layout,
    ]

    var gguf_b_tensor = gguf_b_tensor_type(
        gguf_b_device.buffer,
        RuntimeLayout[
            gguf_b_layout,
            element_type = gguf_b_tensor_type.layout_int_type,
            linear_idx_type = gguf_b_tensor_type.linear_idx_type,
        ].row_major(
            gguf_b_device.tensor.dynamic_shape.cast[
                gguf_b_tensor_type.layout_int_type
            ]()
        ),
    )
    var repacked_b_tensor = LayoutTensor[DType.uint8, repacked_b_layout](
        repacked_b_device.buffer,
    )
    var repacked_dequan_tensor = repacked_dequan_tensor_type(
        repacked_dequan_device.buffer._unsafe_ptr(),
        RuntimeLayout[
            repack_dequan_layout,
            element_type = repacked_dequan_tensor_type.layout_int_type,
            linear_idx_type = repacked_dequan_tensor_type.linear_idx_type,
        ].row_major(
            repacked_dequan_device.tensor.dynamic_shape.cast[
                repacked_dequan_tensor_type.layout_int_type
            ]()
        ),
    )

    var smem_usage: Int = BN * 2 * group_bytes

    alias repack = repack_Q4_0_for_sm8x[
        gguf_b_tensor.layout,
        repacked_b_tensor.layout,
        DType.bfloat16,
    ]

    ctx.enqueue_function[repack](
        gguf_b_tensor,
        repacked_b_tensor,
        grid_dim=(ceildiv(N, BN), ceildiv(K, BK), 1),
        block_dim=(128, 1, 1),
        shared_mem_bytes=smem_usage,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(smem_usage),
    )

    alias dequan = create_ref_b[
        DType.uint8,
        DType.bfloat16,
        repacked_b_tensor.layout,
        repacked_dequan_tensor.layout,
        group_size,
        pack_factor,
    ]

    ctx.enqueue_function[dequan](
        repacked_b_tensor,
        repacked_dequan_tensor,
        grid_dim=(ceildiv(N, 128), ceildiv(K, 32), 1),
        block_dim=(128, 1, 1),
        shared_mem_bytes=smem_usage,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(smem_usage),
    )

    ctx.enqueue_copy(repacked_b_host.tensor.data, repacked_b_device.buffer)
    ctx.enqueue_copy(
        repacked_dequan_host.tensor.data, repacked_dequan_device.buffer
    )

    ctx.synchronize()

    alias rtol = 2e-2
    assert_almost_equal(
        gguf_dequan_ref_host.tensor,
        repacked_dequan_host.tensor,
        atol=0.0001,
        rtol=rtol,
    )

    _ = gguf_dequan_ref_host
    _ = repacked_dequan_tensor
    _ = repacked_dequan_host
    _ = repacked_dequan_device
    _ = gguf_b_tensor
    _ = gguf_b_device
    _ = gguf_b_host
    _ = repacked_b_tensor
    _ = repacked_b_device
    _ = repacked_b_host


fn test_quantized[
    type: DType
](ctx: DeviceContext, m: ValOrDim, n: ValOrDim, k: ValOrDim) raises:
    # quantization configs
    alias group_size = 128
    alias has_zero_point = False
    alias pack_factor = 8
    alias group_bytes = group_size // 2 + 2

    alias repack_tile = Index(64, 16)

    print("test multistage matmul")
    alias static_M = m.dim.get()
    alias static_N = n.dim.get()
    alias static_K = k.dim.get()
    alias a_type = DType.bfloat16

    var M = m.value
    var N = n.value
    var K = k.value

    alias static_a_shape = DimList(m.dim, k.dim)
    alias static_b_shape = DimList(n.dim, (k.dim // group_size) * group_bytes)
    alias static_b_ref_shape = DimList(n.dim, k.dim)
    alias static_c_shape = DimList(m.dim, n.dim)

    var dynamic_a_shape = DimList(m.value, k.value)
    var dynamic_b_shape = DimList(
        n.value, (k.value // group_size) * group_bytes
    )
    var dynamic_b_ref_shape = DimList(n.value, k.value)
    var dynamic_c_shape = DimList(m.value, n.value)

    var a_host = HostNDBuffer[a_type, 2, static_a_shape](dynamic_a_shape)
    var b_host = HostNDBuffer[type, 2, static_b_shape](dynamic_b_shape)
    var c_host = HostNDBuffer[a_type, 2, static_c_shape](dynamic_c_shape)
    var c_host_ref = HostNDBuffer[a_type, 2, static_c_shape](dynamic_c_shape)

    zero(c_host.tensor)
    random(a_host.tensor)

    var b_scales_ptr = (b_host.tensor.data + N * K // 2).bitcast[
        Scalar[a_type]
    ]()
    var b_scales_view = NDBuffer[
        a_type, 2, _, DimList(k.dim // group_size, n.dim)
    ](b_scales_ptr)
    # elements of b matrix is between [-1, 1]
    random(b_scales_view, 0, 0.125)
    randint(
        b_host.tensor.data.bitcast[UInt32](),
        n.value * (k.value // pack_factor),
        UInt.MIN,
        UInt.MAX,
    )

    var a_device = DeviceNDBuffer[a_type, 2, static_a_shape](
        dynamic_a_shape, ctx=ctx
    )
    var b_device = DeviceNDBuffer[type, 2, shape=static_b_shape](
        dynamic_b_shape, ctx=ctx
    )
    var b_device_ref = DeviceNDBuffer[a_type, 2, static_b_ref_shape](
        dynamic_b_ref_shape, ctx=ctx
    )
    var c_device = DeviceNDBuffer[a_type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )

    ctx.enqueue_copy(a_device.buffer, a_host.tensor.data)
    ctx.enqueue_copy(b_device.buffer, b_host.tensor.data)

    alias b_layout = Layout.row_major[c_device.rank](b_device.shape)
    alias b_ref_layout = Layout.row_major[b_device_ref.rank](b_device_ref.shape)
    alias b_tensor_type = LayoutTensor[type, b_layout]
    alias b_ref_tensor_type = LayoutTensor[a_type, b_ref_layout]

    var b_tensor = b_tensor_type(
        b_device.buffer,
        RuntimeLayout[
            b_layout,
            element_type = b_tensor_type.layout_int_type,
            linear_idx_type = b_tensor_type.linear_idx_type,
        ].row_major(
            b_device.tensor.dynamic_shape.cast[b_tensor_type.layout_int_type]()
        ),
    )
    var b_ref_tensor = b_ref_tensor_type(
        b_device_ref.buffer,
        RuntimeLayout[
            b_ref_layout,
            element_type = b_ref_tensor_type.layout_int_type,
            linear_idx_type = b_ref_tensor_type.linear_idx_type,
        ].row_major(
            b_device_ref.tensor.dynamic_shape.cast[
                b_ref_tensor_type.layout_int_type
            ]()
        ),
    )

    var c_device_ref = DeviceNDBuffer[a_type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )

    alias kernels = MatmulKernels[a_type, type, a_type, True]()
    alias config = kernels.ampere_128x128_4
    alias BM = config.block_tile_shape[0]
    alias BN = config.block_tile_shape[1]

    if is_benchmark():
        alias nrun = 200
        alias nwarmup = 2

        @always_inline
        @parameter
        fn run_func(ctx: DeviceContext) raises:
            multistage_gemm_q[
                group_size=group_size, pack_factor=pack_factor, config=config
            ](c_device.tensor, a_device.tensor, b_device.tensor, config, ctx)

        # Warmup
        for _ in range(nwarmup):
            multistage_gemm_q[
                group_size=group_size, pack_factor=pack_factor, config=config
            ](c_device.tensor, a_device.tensor, b_device.tensor, config, ctx)

        var nstime = ctx.execution_time[run_func](nrun) / nrun
        var sectime = nstime * 1e-9
        var TFlop = 2.0 * M * N * K * 1e-12
        print(
            "Tranpose B ",
            "True",
            nrun,
            " runs avg(s)",
            sectime,
            "TFlops/s",
            TFlop / sectime,
        )

    multistage_gemm_q[
        group_size=group_size, pack_factor=pack_factor, config=config
    ](c_device.tensor, a_device.tensor, b_device.tensor, config, ctx)

    alias dequan = create_ref_b[
        type,
        a_type,
        b_tensor.layout,
        b_ref_tensor.layout,
        group_size,
        pack_factor,
    ]

    ctx.enqueue_function[dequan](
        b_tensor,
        b_ref_tensor,
        grid_dim=(ceildiv(N, 128), ceildiv(K, 32), 1),
        block_dim=(128, 1, 1),
        # dump_llvm=Path("./pipeline-gemm.ir"),
        # dump_asm=Path("./pipeline-gemm-2.ptx"),
    )

    ctx.enqueue_copy(c_host.tensor.data, c_device.buffer)

    alias kernels_ref = MatmulKernels[a_type, a_type, a_type, True]()
    alias config_ref = kernels_ref.ampere_128x128_4
    _matmul_gpu[use_tensor_core=True, transpose_b=True, config=config_ref](
        c_device_ref.tensor,
        a_device.tensor,
        b_device_ref.tensor,
        ctx,
    )

    ctx.enqueue_copy(c_host_ref.tensor.data, c_device_ref.buffer)

    ctx.synchronize()

    alias rtol = 1e-2
    assert_almost_equal(
        c_host.tensor,
        c_host_ref.tensor,
        atol=0.0001,
        rtol=rtol,
    )

    _ = c_device
    _ = c_device_ref
    _ = a_host
    _ = b_host
    _ = c_host
    _ = c_host_ref
    _ = a_device
    _ = b_device

    _ = b_tensor


def main():
    with DeviceContext() as ctx:
        test_repack_Q4_0_for_sm8x(
            ctx,
            static[4096](),
            static[4096](),
        )
        test_quantized[DType.uint8](
            ctx, static[482](), static[6144](), static[4096]()
        )
        test_quantized[DType.uint8](
            ctx, static[482](), static[4096](), static[4096]()
        )
        test_quantized[DType.uint8](
            ctx, static[482](), static[28672](), static[4096]()
        )
        test_quantized[DType.uint8](
            ctx, static[482](), static[4096](), static[14336]()
        )
        test_quantized[DType.uint8](
            ctx, static[482](), static[128256](), static[4096]()
        )
        test_quantized[DType.uint8](
            ctx, dynamic(482), static[6144](), static[4096]()
        )
        test_quantized[DType.uint8](
            ctx, dynamic(482), static[4096](), static[4096]()
        )
        test_quantized[DType.uint8](
            ctx, dynamic(482), static[28672](), static[4096]()
        )
        test_quantized[DType.uint8](
            ctx, dynamic(482), static[4096](), static[14336]()
        )
        test_quantized[DType.uint8](
            ctx, dynamic(482), static[128256](), static[4096]()
        )
