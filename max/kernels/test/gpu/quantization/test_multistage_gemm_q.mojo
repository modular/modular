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

from std.math import ceildiv
from std.math.uutils import udivmod
from std.random import rand, randint, random_float64
from std.sys import argv, get_defined_bool

from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_dim,
    block_idx,
    global_idx,
    grid_dim,
    thread_idx,
)
from max.gpu.host import DeviceContext
from max.gpu.host.compile import _compile_code
from max.gpu.intrinsics import lop
from max.gpu.primitives.grid_controls import (
    _SUPPORT_PDL_LAUNCH as SUPPORT_PDL_LAUNCH,
)
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)

from internal_utils import assert_almost_equal
from layout import (
    Coord,
    CoordLike,
    Idx,
    TileTensor,
    TensorLayout,
    row_major,
)
from linalg.matmul.gpu import multistage_gemm
from linalg.utils import no_epilogue_fn
from linalg.utils_gpu import MatmulKernels
from nn.kv_cache_ragged import _qmatmul_common
from std.memory.unsafe import bitcast
from std.sys.intrinsics import llvm_intrinsic
from std.testing import assert_true
from quantization import Q4sym
from quantization.qmatmul_gpu import (
    matmul_gpu_qint4,
    multistage_gemm_q,
    multistage_qgemm_kernel,
    repack_Q4_0_for_sm8x,
)

from std.utils import StaticTuple
from std.utils.index import Index


def is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark" or arg == "-benchmark":
            return True
    return False


@inline(.always)
def args_to_tuple[swap: Bool](arg_0: Int, arg_1: Int) -> Tuple[Int, Int]:
    comptime if swap:
        return Tuple(arg_1, arg_0)
    else:
        return Tuple(arg_0, arg_1)


# This kernel dequantizes a repacked INT4 matrix into bf16 format.
# Assuming a 64x16 (nxk) packing scheme
# Tile [i, j] stores part of the original matrix [i*64:(i+1)*64, j*16:(j+1)*16]
# Within each tile, weights are repacked similarly to the Marlin kernel.
# The memory address for tile [i, j] is (i * (K//16) + j) * tile_size,
# where tile_size is 64 * 16 * 4 / pack_factor = 512 Bytes.
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](128))
def create_ref_b[
    type_q: DType,
    type_b: DType,
    b_q_layout: TensorLayout,
    b_layout: TensorLayout,
    group_size: Int,
    pack_factor: Int,
    b_packed_origin: ImmOrigin,
    b_out_origin: MutOrigin,
](
    b_packed: TileTensor[type_q, b_q_layout, b_packed_origin],
    b_out: TileTensor[type_b, b_layout, b_out_origin],
):
    comptime WARP_SIZE = 32
    comptime BLOCK_N = 128
    comptime BLOCK_K = 32
    comptime repack_tile = (64, 16)
    comptime TILE_N = 64
    comptime TILE_K = 16
    comptime num_k_warps = BLOCK_K // repack_tile[1]

    var tid = thread_idx.x
    var warp_id, lane_id = udivmod(tid, WARP_SIZE)
    var block_idx = Index(block_idx.x, block_idx.y)
    var warp_x, warp_y = udivmod(warp_id, num_k_warps)

    comptime group_bytes = group_size // 2 + 2
    comptime N = b_packed.static_shape[0]
    comptime K = b_packed.static_shape[1] // group_bytes * group_size

    # Unpack quantized weights
    comptime scales_type = DType.bfloat16
    comptime b_type = DType.uint32
    var b_q = b_packed.bitcast[b_type]().reshape(
        row_major[N // 64, K * 64 // pack_factor]()
    )

    var b_scales_ptr = b_packed.ptr.unsafe_offset(N * K // 2)
    var scales = TileTensor(
        b_scales_ptr.unsafe_bitcast[Scalar[scales_type]](),
        row_major[K // group_size, N](),
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
    # Groups of four lanes share the eight scales for one output row group.
    var scales_reg_tiles = warp_scales_tile.load[8](
        Coord(0, (lane_id // 4) * 8)
    )

    var b_out_tile = b_out.tile[BLOCK_N, BLOCK_K](block_idx[0], block_idx[1])
    var warp_out_tile = b_out_tile.tile[repack_tile[0], repack_tile[1]](
        warp_x, warp_y
    )

    var vec = bitcast[.int32, 4](warp_q_tile.vectorize[1, 4]()[0, lane_id])

    @inline(.always)
    def int4tobf16(i4: Int32, scale: BFloat16) -> SIMD[.bfloat16, 2]:
        comptime MASK: Int32 = 0x000F000F
        comptime I4s_TO_BF16s_MAGIC_NUM: Int32 = 0x43004300
        comptime lut: Int32 = (0xF0 & 0xCC) | 0xAA
        var BF16_BIAS = SIMD[.bfloat16, 2](-136, -136)
        var BF16_SCALE = SIMD[.bfloat16, 2](scale, scale)
        var BF16_ZERO = SIMD[.bfloat16, 2](0, 0)
        var BF16_ONE = SIMD[.bfloat16, 2](1, 1)

        var t = lop[lut](i4, MASK, I4s_TO_BF16s_MAGIC_NUM)

        var v = (
            bitcast[.bfloat16, 2](t)
            .fma(BF16_ONE, BF16_BIAS)
            .fma(BF16_SCALE, BF16_ZERO)
        )
        return v

    var lane_row, lane_col = udivmod(lane_id, 4)

    comptime for i in range(0, TILE_N // 8, 2):
        var q_int = vec[i // 2]

        var v1 = int4tobf16(q_int, bitcast[.bfloat16, 1](scales_reg_tiles[i]))
        warp_out_tile.store(
            ((i) * 8 + lane_row, 0 * 8 + lane_col * 2),
            v1.cast[type_b](),
        )
        q_int >>= 4
        var v2 = int4tobf16(q_int, bitcast[.bfloat16, 1](scales_reg_tiles[i]))
        warp_out_tile.store(
            ((i) * 8 + lane_row, 1 * 8 + lane_col * 2),
            v2.cast[type_b](),
        )
        q_int >>= 4

        v1 = int4tobf16(q_int, bitcast[.bfloat16, 1](scales_reg_tiles[i + 1]))
        warp_out_tile.store(
            ((i + 1) * 8 + lane_row, 0 * 8 + lane_col * 2),
            v1.cast[type_b](),
        )
        q_int >>= 4
        v2 = int4tobf16(q_int, bitcast[.bfloat16, 1](scales_reg_tiles[i + 1]))
        warp_out_tile.store(
            ((i + 1) * 8 + lane_row, 1 * 8 + lane_col * 2),
            v2.cast[type_b](),
        )


def random_float16(min: Float64 = 0, max: Float64 = 1) -> Float16:
    # Avoid pulling in a __truncdfhf2 dependency for a float64->float16
    # conversion by casting through float32 first.
    return random_float64(min=min, max=max).cast[.float32]().cast[.float16]()


struct _block_Q4_0:
    comptime group_size = 32

    var base_scale: Float16
    var q_bits: Array[UInt8, Self.group_size // 2]


def test_repack_Q4_0_for_sm8x[
    NType: CoordLike, KType: CoordLike, //
](ctx: DeviceContext[], n: NType, k: KType) raises:
    print("test repack_Q4_0_for_sm8x")

    def fill_random[dtype: DType](mut array: Array[Scalar[dtype], ...]):
        rand(array, min=0, max=255)

    def build_b_buffer(N: Int, K: Int, b_ptr: MutPointer[UInt8, _]):
        var k_groups = ceildiv(K, 32)
        var block_ptr = b_ptr.unsafe_bitcast[_block_Q4_0]()

        for _ in range(N):
            for _ in range(k_groups):
                block_ptr[].base_scale = random_float16()
                fill_random(block_ptr[].q_bits)
                block_ptr = block_ptr.unsafe_offset(1)

    comptime group_size = 32
    comptime pack_factor = 8
    var N = Int(n)
    var K = Int(k)
    comptime BN = 128
    comptime BK = 1024
    comptime group_bytes = 2 + (group_size // 2)

    var gguf_b_size = N * ((K // group_size) * group_bytes)
    var repacked_b_size = N * ((K // group_size) * group_bytes)
    var dequan_size = K * N

    var gguf_b_host_ptr = ctx.enqueue_create_host_buffer[.uint8](gguf_b_size)
    var repacked_b_host_ptr = ctx.enqueue_create_host_buffer[.uint8](
        repacked_b_size
    )
    var gguf_dequan_ref_host_ptr = ctx.enqueue_create_host_buffer[
        DType.bfloat16
    ](dequan_size)
    var repacked_dequan_host_ptr = ctx.enqueue_create_host_buffer[
        DType.bfloat16
    ](dequan_size)

    var gguf_shape = row_major(
        Coord(n, Idx[(KType.static_value // group_size) * group_bytes])
    )
    var dequan_shape = row_major(Coord(k, n))
    var gguf_b_host = TileTensor(gguf_b_host_ptr, gguf_shape)
    var gguf_dequan_ref_host = TileTensor(
        gguf_dequan_ref_host_ptr, dequan_shape
    )

    build_b_buffer(N, K, gguf_b_host_ptr.unsafe_ptr())
    Q4sym[group_size, DType.bfloat16].dequantize_and_write_to_tensor(
        gguf_b_host.as_imm(), gguf_dequan_ref_host, (K, N)
    )

    var gguf_b_device = ctx.enqueue_create_buffer[.uint8](gguf_b_size)
    var repacked_b_device = ctx.enqueue_create_buffer[.uint8](repacked_b_size)
    var repacked_dequan_device = ctx.enqueue_create_buffer[.bfloat16](
        dequan_size
    )

    ctx.enqueue_copy(gguf_b_device, gguf_b_host_ptr)
    ctx.enqueue_copy(repacked_b_device, repacked_b_host_ptr)

    var gguf_b_tensor = TileTensor(gguf_b_device.unsafe_ptr(), gguf_shape)
    var repacked_b_tensor = TileTensor(
        repacked_b_device.unsafe_ptr(), gguf_shape
    )
    var repacked_dequan_tensor = TileTensor(
        repacked_dequan_device.unsafe_ptr(), dequan_shape
    )

    var smem_usage: Int = BN * 2 * group_bytes

    comptime repack = repack_Q4_0_for_sm8x[
        gguf_b_tensor.LayoutType, repacked_b_tensor.LayoutType, DType.bfloat16
    ]

    ctx.enqueue_function[repack](
        gguf_b_tensor,
        repacked_b_tensor,
        grid_dim=(ceildiv(N, BN), ceildiv(K, BK), 1),
        block_dim=(128, 1, 1),
        shared_mem_bytes=smem_usage,
    )

    comptime dequan = create_ref_b[
        DType.uint8,
        DType.bfloat16,
        repacked_b_tensor.LayoutType,
        repacked_dequan_tensor.LayoutType,
        group_size,
        pack_factor,
        ImmOrigin(repacked_b_tensor.origin),
        repacked_dequan_tensor.origin,
    ]

    ctx.enqueue_function[dequan](
        repacked_b_tensor,
        repacked_dequan_tensor,
        grid_dim=(ceildiv(N, 128), ceildiv(K, 32), 1),
        block_dim=(128, 1, 1),
        shared_mem_bytes=smem_usage,
    )

    ctx.enqueue_copy(repacked_b_host_ptr, repacked_b_device)
    ctx.enqueue_copy(repacked_dequan_host_ptr, repacked_dequan_device)

    ctx.synchronize()

    comptime rtol = 2e-2
    assert_almost_equal(
        gguf_dequan_ref_host_ptr.unsafe_ptr(),
        repacked_dequan_host_ptr.unsafe_ptr(),
        dequan_size,
        atol=0.0001,
        rtol=rtol,
    )

    _ = repacked_dequan_tensor
    _ = gguf_b_tensor
    _ = repacked_b_tensor
    _ = gguf_b_device^
    _ = repacked_b_device^
    _ = repacked_dequan_device^


# --- PDL chain helpers -------------------------------------------------
# A producer that stages the activation operand on the device and a consumer
# that waits for the GEMM before reading its output. Chaining them around the
# int4 GEMM with no host synchronisation in between is what exercises the
# GEMM's own `wait_on_dependent_grids` / `launch_dependent_grids` placement.
#
# The producer releases its dependent on *entry* and only writes the
# activations after a spin, so the write is still pending when a dependent
# starts. That turns a missing GEMM wait into a wrong result instead of a race
# the scheduler happens to win: the activation buffer holds the stale
# `_PDL_STALE_A` fill until the producer's delayed write lands, and a GEMM that
# multiplies the sentinel cannot match the reference.

# Nanoseconds the producer spins between releasing its dependents and writing.
comptime _PDL_PRODUCER_DELAY_NS = 1_000_000

# Activation value visible to a dependent that reads before its own wait. Outside
# the `[0, 1)` range `rand` fills the reference activations with.
comptime _PDL_STALE_A = 4.0


def _pdl_fill_a_kernel(
    dst: MutPointer[Scalar[.bfloat16], MutAnyOrigin],
    n: Int32,
    value: Scalar[.bfloat16],
):
    """Fills `dst` with `value`: the stale activation a producer replaces."""
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < Int(n):
        dst[i] = value
        i += stride


def _pdl_stage_a_kernel(
    src: ImmPointer[Scalar[.bfloat16], ImmutAnyOrigin],
    dst: MutPointer[Scalar[.bfloat16], MutAnyOrigin],
    n: Int32,
    delay_ns: Int32,
):
    """Releases its dependents on entry, spins, then writes `src` to `dst`.

    The release comes before the write, so a GEMM that skips its own
    `wait_on_dependent_grids` reads the stale `dst` fill, not `src`.
    """
    # A producer normally only releases, but `-D MAX_CHECK_PDL_WAIT=True` asks
    # every PDL-launched kernel to carry a wait, so make it explicit here.
    comptime if get_defined_bool["MAX_CHECK_PDL_WAIT", False]():
        wait_on_dependent_grids()
    launch_dependent_grids()
    # `nanosleep` takes nanoseconds and the hardware caps one call near 1 ms.
    var spins = Int(delay_ns)
    while spins > 0:
        llvm_intrinsic["llvm.nvvm.nanosleep", NoneType](Int32(1_000_000))
        spins -= 1_000_000
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < Int(n):
        dst[i] = src[i]
        i += stride


def _pdl_consume_kernel(
    src: ImmPointer[Scalar[.bfloat16], ImmutAnyOrigin],
    dst: MutPointer[Scalar[.bfloat16], MutAnyOrigin],
    n: Int32,
):
    """Waits for the producing grid, then copies its output to `dst`."""
    wait_on_dependent_grids()
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < Int(n):
        dst[i] = src[i]
        i += stride


def test_quantized[
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    //,
    dtype: DType,
    use_dispatch: Bool = False,
    use_ragged_epilogue: Bool = False,
    join_pdl_chain: Bool = False,
    pdl_level: PDLLevel = PDLLevel(),
](ctx: DeviceContext[], m: MType, n: NType, k: KType) raises:
    # quantization configs
    comptime group_size = 128
    comptime has_zero_point = False
    comptime pack_factor = 8
    comptime group_bytes = group_size // 2 + 2

    comptime repack_tile = (64, 16)

    print("test multistage matmul")
    comptime static_M = MType.static_value
    comptime static_N = NType.static_value
    comptime static_K = KType.static_value
    comptime a_type = DType.bfloat16

    var M = Int(m)
    var N = Int(n)
    var K = Int(k)

    comptime _b_dim0 = NType.static_value
    comptime _b_dim1 = (KType.static_value // group_size) * group_bytes

    var a_size = M * K
    var b_size = N * ((K // group_size) * group_bytes)
    var b_ref_size = N * K
    var c_size = M * N

    var a_host_ptr = ctx.enqueue_create_host_buffer[a_type](a_size)
    var b_host_ptr = ctx.enqueue_create_host_buffer[dtype](b_size)
    var c_host_ptr = ctx.enqueue_create_host_buffer[a_type](c_size)
    var c_host_ref_ptr = ctx.enqueue_create_host_buffer[a_type](c_size)

    rand(a_host_ptr.unsafe_ptr(), a_size)

    var b_scales_ptr = (
        b_host_ptr.unsafe_ptr()
        .unsafe_offset(N * K // 2)
        .unsafe_bitcast[Scalar[a_type]]()
    )
    var b_scales_size = (K // group_size) * N
    # elements of b matrix is between [-1, 1]
    rand(b_scales_ptr, b_scales_size, min=0, max=0.125)
    randint(
        b_host_ptr.unsafe_ptr().unsafe_bitcast[UInt32](),
        N * (K // pack_factor),
        Int(UInt32.MIN),
        Int(UInt32.MAX),
    )

    var a_device = ctx.enqueue_create_buffer[a_type](a_size)
    var b_device = ctx.enqueue_create_buffer[dtype](b_size)
    var b_device_ref = ctx.enqueue_create_buffer[a_type](b_ref_size)
    var c_device = ctx.enqueue_create_buffer[a_type](c_size)
    # Everything the PDL chain touches is allocated before its launches: an
    # allocation or a copy enqueued between them can serialise the stream and
    # hide a missing wait. The staging buffer stays alive for the whole
    # function so it outlives the grids that write it, and is empty outside the
    # chain.
    var a_src_device = ctx.enqueue_create_buffer[a_type](a_size)
    # Allocated before any launch: an allocation between the producer and the
    # GEMM would serialise the stream and mask a missing wait.
    var c_device_ref = ctx.enqueue_create_buffer[a_type](c_size)
    var c_stage_device = ctx.enqueue_create_buffer[a_type](
        c_size if join_pdl_chain else 0
    )

    # The weights are staged before the activation producer so the producer and
    # the GEMM are adjacent dependent launches.
    ctx.enqueue_copy(b_device, b_host_ptr)
    comptime if join_pdl_chain:
        ctx.enqueue_copy(a_src_device, a_host_ptr)
        # The activations hold a stale sentinel until the producer's delayed
        # write lands, so a GEMM that reads them before its own wait computes
        # something the reference comparison cannot match.
        ctx.enqueue_function[_pdl_fill_a_kernel](
            a_device.unsafe_ptr(),
            Int32(a_size),
            Scalar[a_type](_PDL_STALE_A),
            grid_dim=ceildiv(a_size, 256),
            block_dim=256,
        )
        # A producer grid writes the activations immediately before the GEMM,
        # with no host synchronisation between them: the GEMM's own wait is
        # what keeps it from reading the sentinel.
        ctx.enqueue_function[_pdl_stage_a_kernel](
            a_src_device.unsafe_ptr(),
            a_device.unsafe_ptr(),
            Int32(a_size),
            Int32(_PDL_PRODUCER_DELAY_NS),
            grid_dim=ceildiv(a_size, 256),
            block_dim=256,
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )
    else:
        ctx.enqueue_copy(a_device, a_host_ptr)

    var b_tensor = TileTensor(
        b_device.unsafe_ptr(), row_major[_b_dim0, _b_dim1]()
    )
    var b_ref_tensor = TileTensor(
        b_device_ref.unsafe_ptr(), row_major(Coord(n, k))
    )

    comptime kernels = MatmulKernels[a_type, dtype, a_type, True]()
    comptime config = kernels.ampere_128x128_4
    comptime BM = config.block_tile_shape[0]
    comptime BN = config.block_tile_shape[1]

    # Create TileTensors for the matmul operands
    var a_tt_shape = row_major(m, Idx[KType.static_value])
    var b_tt_shape = row_major(
        Idx[NType.static_value],
        Idx[(KType.static_value // group_size) * group_bytes],
    )
    var c_tt_shape = row_major(m, Idx[NType.static_value])

    var c_dev_tt = TileTensor(c_device, c_tt_shape)
    var a_dev_tt = TileTensor(a_device, a_tt_shape)
    var b_dev_tt = TileTensor(b_device, b_tt_shape)

    if is_benchmark():
        comptime nrun = 200
        comptime nwarmup = 2

        @inline(.always)
        def run_func(ctx: DeviceContext[]) raises {imm}:
            multistage_gemm_q[
                group_size=group_size, pack_factor=pack_factor, config=config
            ](
                c_dev_tt,
                a_dev_tt,
                b_dev_tt,
                config,
                ctx,
            )

        # Warmup
        for _ in range(nwarmup):
            multistage_gemm_q[
                group_size=group_size, pack_factor=pack_factor, config=config
            ](
                c_dev_tt,
                a_dev_tt,
                b_dev_tt,
                config,
                ctx,
            )

        var nstime = Float64(ctx.execution_time(run_func, nrun)) / Float64(nrun)
        var sectime = nstime * 1e-9
        var TFlop = 2.0 * Float64(M) * Float64(N) * Float64(K) * 1e-12
        print(
            t"Transpose B  True {nrun}  runs avg(s) {sectime} TFlops/s"
            t" {TFlop / sectime}"
        )

    comptime if use_ragged_epilogue:
        comptime assert dtype == .uint8

        @inline(.always)
        def epilogue_fn[
            value_type: DType,
            width: SIMDLength,
            *,
            alignment: Int,
        ](idx: Coord, value: SIMD[value_type, width]) {var c_dev_tt}:
            comptime assert (
                idx.rank == 2
            ), "the matmul epilogue contract is rank 2"
            c_dev_tt.store[alignment=alignment](idx, value.cast[a_type]())

        # The helper supplies a null output view; only the epilogue may write.
        _qmatmul_common[group_size=group_size](
            a_dev_tt.as_imm(),
            b_dev_tt.bitcast[.uint8]().as_imm(),
            ctx,
            epilogue_fn,
        )
    elif use_dispatch:
        comptime assert dtype == .uint8
        matmul_gpu_qint4[group_size](
            c_dev_tt, a_dev_tt, b_dev_tt.bitcast[.uint8](), ctx
        )
    else:
        multistage_gemm_q[
            group_size=group_size,
            pack_factor=pack_factor,
            config=config,
            pdl_level=pdl_level,
        ](
            c_dev_tt,
            a_dev_tt,
            b_dev_tt,
            config,
            ctx,
        )

    comptime if join_pdl_chain:
        # Dependent consumer: it must not read the GEMM's output until the
        # producing grid has finished.
        ctx.enqueue_function[_pdl_consume_kernel](
            c_device.unsafe_ptr(),
            c_stage_device.unsafe_ptr(),
            Int32(c_size),
            grid_dim=ceildiv(c_size, 256),
            block_dim=256,
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )
        ctx.enqueue_copy(c_host_ptr, c_stage_device)
    else:
        ctx.enqueue_copy(c_host_ptr, c_device)

    comptime dequan = create_ref_b[
        dtype,
        a_type,
        b_tensor.LayoutType,
        b_ref_tensor.LayoutType,
        group_size,
        pack_factor,
        ImmOrigin(b_tensor.origin),
        b_ref_tensor.origin,
    ]

    ctx.enqueue_function[dequan](
        b_tensor,
        b_ref_tensor,
        grid_dim=(ceildiv(N, 128), ceildiv(K, 32), 1),
        block_dim=(128, 1, 1),
        # dump_llvm=Path("./pipeline-gemm.ir"),
        # dump_asm=Path("./pipeline-gemm-2.ptx"),
    )

    comptime kernels_ref = MatmulKernels[a_type, a_type, a_type, True]()
    comptime config_ref = kernels_ref.ampere_128x128_4
    var c_ref_tt_shape = row_major(m, Idx[NType.static_value])
    var b_ref_tt_shape = row_major(
        Idx[NType.static_value], Idx[KType.static_value]
    )
    var c_ref_tt = TileTensor(c_device_ref, c_ref_tt_shape)
    var b_ref_tt = TileTensor(b_device_ref, b_ref_tt_shape)
    multistage_gemm[transpose_b=True, config=config_ref](
        c_ref_tt,
        a_dev_tt,
        b_ref_tt,
        ctx,
    )

    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)

    ctx.synchronize()

    comptime rtol = 1e-2
    assert_almost_equal(
        c_host_ptr.unsafe_ptr(),
        c_host_ref_ptr.unsafe_ptr(),
        c_size,
        atol=0.0001,
        rtol=rtol,
    )

    _ = a_device^
    _ = a_src_device^
    _ = b_device^
    _ = b_device_ref^
    _ = c_device^
    _ = c_stage_device^
    _ = c_device_ref^

    _ = b_tensor


def test_pdl_release_sites() raises:
    """The GEMM releases a PDL dependent at the site its level names.

    `OVERLAP_AT_BEGINNING` must release on entry, right after the wait, as
    `PDL[overlap_at_beginning=True]` does; `OVERLAP_AT_END` must release only
    once the MMA loop has consumed its inputs. Both levels release through the
    same `griddepcontrol.launch_dependents` instruction, so the site is checked
    by where it lands relative to the loop's `mma.sync`: before the first one
    at the beginning level, after the last one at the end level. A level that
    leaves PDL off emits neither the wait nor the release, which is what lets
    the chain in `test_quantized` detect a missing wait.

    This target is `//:nvidia_gpu`, so it also runs on cards below SM90, where
    `launch_dependent_grids` and `wait_on_dependent_grids` compile to nothing
    and `pdl_launch_attributes` returns an empty list. There the check flips:
    no level may emit either instruction, and `mma.sync` still has to be in the
    assembly, so the instruction-absence checks are not vacuous.
    """
    comptime group_size = 128
    comptime group_bytes = group_size // 2 + 2
    comptime pack_factor = 8
    comptime a_type = DType.bfloat16
    comptime M = 16
    comptime N = 1024
    comptime K = 1024
    comptime c_lay = row_major[M, N]()
    comptime a_lay = row_major[M, K]()
    comptime b_lay = row_major[N, (K // group_size) * group_bytes]()
    comptime kernels = MatmulKernels[a_type, DType.uint8, a_type, True]()
    comptime config = kernels.ampere_128x128_4
    comptime release = "griddepcontrol.launch_dependents"
    comptime wait = "griddepcontrol.wait"
    comptime mma = "mma.sync"

    comptime kernel_off = multistage_qgemm_kernel[
        a_type,
        type_of(c_lay),
        a_type,
        type_of(a_lay),
        DType.uint8,
        type_of(b_lay),
        group_size,
        pack_factor,
        True,
        config,
        type_of(no_epilogue_fn),
        False,
        PDLLevel.OFF,
    ]
    var off_asm = _compile_code[kernel_off, emission_kind="asm"]().asm
    assert_true(wait not in off_asm)
    assert_true(release not in off_asm)

    comptime kernel_end = multistage_qgemm_kernel[
        a_type,
        type_of(c_lay),
        a_type,
        type_of(a_lay),
        DType.uint8,
        type_of(b_lay),
        group_size,
        pack_factor,
        True,
        config,
        type_of(no_epilogue_fn),
        False,
        PDLLevel.OVERLAP_AT_END,
    ]
    var end_asm = _compile_code[kernel_end, emission_kind="asm"]().asm
    assert_true(mma in end_asm)

    comptime kernel_beginning = multistage_qgemm_kernel[
        a_type,
        type_of(c_lay),
        a_type,
        type_of(a_lay),
        DType.uint8,
        type_of(b_lay),
        group_size,
        pack_factor,
        True,
        config,
        type_of(no_epilogue_fn),
        False,
        PDLLevel.OVERLAP_AT_BEGINNING,
    ]
    var beginning_asm = _compile_code[
        kernel_beginning, emission_kind="asm"
    ]().asm
    assert_true(mma in beginning_asm)

    # The helpers that emit these instructions gate on `SUPPORT_PDL_LAUNCH`,
    # so the assembly this test compiles for the default accelerator carries
    # them only where the target supports PDL. Check the target's own answer
    # rather than assuming SM90+.
    comptime if SUPPORT_PDL_LAUNCH:
        assert_true(wait in end_asm)
        assert_true(release in end_asm)
        # After the MMA loop: nothing in the loop reads an input from below the
        # release, so the release cannot be hoisted above every `mma.sync`.
        assert_true(end_asm.rfind(mma) < end_asm.find(release))

        assert_true(wait in beginning_asm)
        assert_true(release in beginning_asm)
        var wait_at = beginning_asm.find(wait)
        var release_at = beginning_asm.find(release)
        assert_true(wait_at < release_at)
        # On entry, before the MMA loop starts.
        assert_true(release_at < beginning_asm.find(mma))
    else:
        # Below SM90 the levels change nothing in the emitted code, so neither
        # instruction may appear at any level. `mma.sync` above still has to be
        # there, which is what makes these absence checks meaningful.
        assert_true(wait not in end_asm)
        assert_true(release not in end_asm)
        assert_true(wait not in beginning_asm)
        assert_true(release not in beginning_asm)


def main() raises:
    with DeviceContext() as ctx:
        test_quantized[.uint8, True](ctx, Int(16), Idx[4096], Idx[4096])
        test_quantized[.uint8, True](ctx, Int(65), Idx[1024], Idx[1024])
        # The launch attributes are part of the kernel instantiation, so the
        # same shape has to match the reference with programmatic dependent
        # launch disabled as well as at the default level.
        test_quantized[.uint8, pdl_level=PDLLevel.OFF](
            ctx, Int(16), Idx[4096], Idx[4096]
        )
        test_quantized[.uint8, pdl_level=PDLLevel.OVERLAP_AT_END](
            ctx, Int(65), Idx[1024], Idx[1024]
        )
        # The producer -> GEMM -> consumer chain: the GEMM must wait for the
        # producer and the consumer must wait for the GEMM, with no host
        # synchronisation in between.
        test_quantized[
            .uint8, join_pdl_chain=True, pdl_level=PDLLevel.OVERLAP_AT_END
        ](ctx, Int(16), Idx[1024], Idx[1024])
        test_quantized[
            .uint8,
            join_pdl_chain=True,
            pdl_level=PDLLevel.OVERLAP_AT_BEGINNING,
        ](ctx, Int(16), Idx[1024], Idx[1024])
        test_quantized[.uint8, join_pdl_chain=True, pdl_level=PDLLevel.OFF](
            ctx, Int(16), Idx[1024], Idx[1024]
        )
        test_quantized[.uint8, use_ragged_epilogue=True](
            ctx, Int(16), Idx[4096], Idx[4096]
        )
        test_quantized[.uint8, use_ragged_epilogue=True](
            ctx, Int(65), Idx[1024], Idx[1024]
        )
        test_repack_Q4_0_for_sm8x(
            ctx,
            Idx[4096],
            Idx[4096],
        )
        test_quantized[.uint8](ctx, Idx[482], Idx[6144], Idx[4096])
        test_quantized[.uint8](ctx, Idx[482], Idx[4096], Idx[4096])
        test_quantized[.uint8](ctx, Idx[482], Idx[28672], Idx[4096])
        test_quantized[.uint8](ctx, Idx[482], Idx[4096], Idx[14336])
        test_quantized[.uint8](ctx, Idx[482], Idx[128256], Idx[4096])
        test_quantized[.uint8](ctx, Int(482), Idx[6144], Idx[4096])
        test_quantized[.uint8](ctx, Int(482), Idx[4096], Idx[4096])
        test_quantized[.uint8](ctx, Int(482), Idx[28672], Idx[4096])
        test_quantized[.uint8](ctx, Int(482), Idx[4096], Idx[14336])
        test_quantized[.uint8](ctx, Int(482), Idx[128256], Idx[4096])
        # Where each level puts its release, in the emitted PTX.
        test_pdl_release_sites()
