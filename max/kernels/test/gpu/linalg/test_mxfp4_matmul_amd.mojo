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
"""Tests for the native MXFP4 block-scaled matmul kernel on AMD CDNA4.

Validates MXFP4MatmulAMD against a per-element GPU reference
that uses the llvm.amdgcn.cvt.scalef32.pk.f32.fp4 intrinsic for FP4→FP32
dequantization and scalar accumulation.

Usage:
  mojo test_mxfp4_matmul_amd.mojo
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.math import ceildiv
from std.memory import bitcast
from std.random import random_ui64
from std.sys.intrinsics import llvm_intrinsic

from internal_utils import assert_almost_equal
from layout import Coord, Idx, TileTensor, row_major
from linalg.fp4_utils import MXFP4_SF_VECTOR_SIZE
from linalg.matmul.gpu.amd.mxfp4_matmul_amd import MXFP4MatmulAMD


# ===----------------------------------------------------------------------=== #
# Reference kernel: scalar FP4 dequant matmul on GPU
# ===----------------------------------------------------------------------=== #


def block_scaled_matmul_ref(
    a_ptr: UnsafePointer[Scalar[DType.uint8], ImmutAnyOrigin],
    b_ptr: UnsafePointer[Scalar[DType.uint8], ImmutAnyOrigin],
    a_scales_ptr: UnsafePointer[Scalar[DType.float8_e8m0fnu], ImmutAnyOrigin],
    b_scales_ptr: UnsafePointer[Scalar[DType.float8_e8m0fnu], ImmutAnyOrigin],
    c_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    M: Int,
    N: Int,
    K: Int,
):
    """Per-element GPU reference for MXFP4 block-scaled matmul.

    Each thread computes one (m, n) output element by dequantizing
    packed FP4 data via the CDNA4 cvt.scalef32.pk.f32.fp4 intrinsic,
    multiplying with E8M0 scales, and accumulating in FP32.
    """

    @always_inline
    def cast_fp4x2_to_fp32x2[
        byte_select: Int
    ](packed: Int32, scale: Float32) -> SIMD[DType.float32, 2]:
        return llvm_intrinsic[
            "llvm.amdgcn.cvt.scalef32.pk.f32.fp4",
            SIMD[DType.float32, 2],
        ](packed, scale, Int32(byte_select))

    var m = global_idx.x
    var n = global_idx.y

    if m >= M or n >= N:
        return

    var k_groups = K // MXFP4_SF_VECTOR_SIZE

    var am_scales_ptr = a_scales_ptr + m * k_groups
    var bn_scales_ptr = b_scales_ptr + n * k_groups

    var am_ptr = a_ptr + m * (K // 2)
    var bn_ptr = b_ptr + n * (K // 2)

    var accum = SIMD[DType.float32, 2](0)

    for ko in range(k_groups):
        var a_scale = am_scales_ptr[ko].cast[DType.float32]()
        var b_scale = bn_scales_ptr[ko].cast[DType.float32]()

        for ki in range(0, MXFP4_SF_VECTOR_SIZE // 2, 4):
            var a_data = bitcast[DType.int32, 1](am_ptr.load[width=4](ki))
            var b_data = bitcast[DType.int32, 1](bn_ptr.load[width=4](ki))

            comptime for byte_select in range(4):
                accum += cast_fp4x2_to_fp32x2[byte_select](
                    a_data, a_scale
                ) * cast_fp4x2_to_fp32x2[byte_select](b_data, b_scale)

        am_ptr += MXFP4_SF_VECTOR_SIZE // 2
        bn_ptr += MXFP4_SF_VECTOR_SIZE // 2

    c_ptr[m * N + n] = accum.reduce_add()


# ===----------------------------------------------------------------------=== #
# Test harness
# ===----------------------------------------------------------------------=== #


def test_mxfp4_matmul[
    M_static: Int,
    N_static: Int,
    K_static: Int,
    BM: Int = 128,
    BN: Int = 128,
    BK_ELEMS: Int = 128,
    WM: Int = 64,
    WN: Int = 64,
](ctx: DeviceContext) raises:
    """Test MXFP4MatmulAMD against a GPU reference kernel.

    Launches MXFP4MatmulAMD directly with the provided BM/BN/BK_ELEMS/WM/WN.
    Defaults match the current production tile config.

    Parameters:
        M_static: Number of rows in A / C.
        N_static: Number of rows in B (transposed) / cols in C.
        K_static: Logical K dimension (FP4 elements, must be multiple of 128).
        BM: Block tile rows.
        BN: Block tile cols.
        BK_ELEMS: Block tile K in logical FP4 elements.
        WM: Warp tile rows.
        WN: Warp tile cols.
    """
    comptime assert (
        K_static % 128 == 0
    ), "K must be a multiple of 128 (MFMA K dimension)"
    comptime assert (
        K_static % MXFP4_SF_VECTOR_SIZE == 0
    ), "K must be a multiple of MXFP4_SF_VECTOR_SIZE (32)"
    comptime assert BK_ELEMS % 128 == 0, "BK_ELEMS must be a multiple of 128"
    comptime assert BM % WM == 0, "BM must be divisible by WM"
    comptime assert BN % WN == 0, "BN must be divisible by WN"

    print(
        M_static,
        "x",
        N_static,
        "x",
        K_static,
        " [BM=",
        BM,
        " BN=",
        BN,
        " BK=",
        BK_ELEMS,
        " WM=",
        WM,
        " WN=",
        WN,
        "]",
    )

    comptime input_dtype = DType.uint8
    comptime scales_dtype = DType.float8_e8m0fnu
    comptime output_dtype = DType.float32

    comptime K_PACKED = K_static // 2
    comptime K_SCALES = K_static // MXFP4_SF_VECTOR_SIZE

    comptime a_size = M_static * K_PACKED
    comptime b_size = N_static * K_PACKED
    comptime c_size = M_static * N_static
    comptime a_scales_size = M_static * K_SCALES
    comptime b_scales_size = N_static * K_SCALES

    comptime a_shape = row_major[M_static, K_PACKED]()
    comptime b_shape = row_major[N_static, K_PACKED]()
    comptime c_shape = row_major[M_static, N_static]()
    comptime a_scales_shape = row_major[M_static, K_SCALES]()
    comptime b_scales_shape = row_major[N_static, K_SCALES]()

    var a_host = alloc[Scalar[input_dtype]](a_size)
    var b_host = alloc[Scalar[input_dtype]](b_size)
    var a_scales_host = alloc[Scalar[scales_dtype]](a_scales_size)
    var b_scales_host = alloc[Scalar[scales_dtype]](b_scales_size)
    var c_host = alloc[Scalar[output_dtype]](c_size)
    var c_host_ref = alloc[Scalar[output_dtype]](c_size)

    for i in range(a_size):
        a_host[i] = UInt8(random_ui64(0, 255))
    for i in range(b_size):
        b_host[i] = UInt8(random_ui64(0, 255))
    for i in range(a_scales_size):
        a_scales_host[i] = bitcast[scales_dtype](UInt8(random_ui64(125, 129)))
    for i in range(b_scales_size):
        b_scales_host[i] = bitcast[scales_dtype](UInt8(random_ui64(125, 129)))

    var a_dev = ctx.enqueue_create_buffer[input_dtype](a_size)
    var b_dev = ctx.enqueue_create_buffer[input_dtype](b_size)
    var a_scales_dev = ctx.enqueue_create_buffer[scales_dtype](a_scales_size)
    var b_scales_dev = ctx.enqueue_create_buffer[scales_dtype](b_scales_size)
    var c_dev = ctx.enqueue_create_buffer[output_dtype](c_size)
    var c_ref_dev = ctx.enqueue_create_buffer[output_dtype](c_size)

    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.enqueue_copy(a_scales_dev, a_scales_host)
    ctx.enqueue_copy(b_scales_dev, b_scales_host)

    var a_tt = TileTensor[mut=False](a_dev, a_shape)
    var b_tt = TileTensor[mut=False](b_dev, b_shape)
    var c_tt = TileTensor[mut=True](c_dev, c_shape)
    var a_scales_tt = TileTensor[mut=False](a_scales_dev, a_scales_shape)
    var b_scales_tt = TileTensor[mut=False](b_scales_dev, b_scales_shape)

    # --- Direct launch with explicit tile params ---
    comptime Kernel = MXFP4MatmulAMD[
        BM=BM,
        BN=BN,
        BK_ELEMS=BK_ELEMS,
        WM=WM,
        WN=WN,
    ]
    comptime kernel = Kernel.run[
        DType.float32,
        type_of(c_tt).LayoutType,
        type_of(a_tt).LayoutType,
        type_of(b_tt).LayoutType,
        type_of(a_scales_tt).LayoutType,
        type_of(b_scales_tt).LayoutType,
    ]
    ctx.enqueue_function[kernel, kernel](
        c_tt,
        a_tt,
        b_tt,
        a_scales_tt,
        b_scales_tt,
        grid_dim=(ceildiv(N_static, BN), ceildiv(M_static, BM)),
        block_dim=Kernel.num_threads,
    )

    # --- Reference ---
    comptime BLOCK_DIM = 32
    ctx.enqueue_function_experimental[block_scaled_matmul_ref](
        a_dev,
        b_dev,
        a_scales_dev,
        b_scales_dev,
        c_ref_dev,
        M_static,
        N_static,
        K_static,
        grid_dim=(ceildiv(M_static, BLOCK_DIM), ceildiv(N_static, BLOCK_DIM)),
        block_dim=(BLOCK_DIM, BLOCK_DIM),
    )

    ctx.enqueue_copy(c_host, c_dev)
    ctx.enqueue_copy(c_host_ref, c_ref_dev)
    ctx.synchronize()

    assert_almost_equal(
        c_host,
        c_host_ref,
        c_size,
        atol=0.05,
        rtol=0.05,
    )

    print("  PASSED")

    a_host.free()
    b_host.free()
    a_scales_host.free()
    b_scales_host.free()
    c_host.free()
    c_host_ref.free()


def main() raises:
    with DeviceContext() as ctx:
        print("===> MXFP4 block-scaled matmul (native CDNA4 MFMA)")

        # === Bucket A: baseline aligned correctness ===
        print("\n--- A: baseline aligned shapes ---")

        test_mxfp4_matmul[128, 128, 128](ctx)
        test_mxfp4_matmul[128, 128, 256](ctx)
        test_mxfp4_matmul[128, 128, 512](ctx)
        test_mxfp4_matmul[256, 128, 256](ctx)
        test_mxfp4_matmul[128, 256, 256](ctx)
        test_mxfp4_matmul[256, 256, 256](ctx)
        test_mxfp4_matmul[256, 256, 512](ctx)
        test_mxfp4_matmul[128, 128, 1024](ctx)

        # === Bucket B: Kimi K2.5 unaligned-M OOB stress matrix ===
        print("\n--- B: Kimi K2.5 unaligned-M OOB stress ---")

        # M=1 (decode, single row) across all projections.
        test_mxfp4_matmul[1, 7168, 2048](ctx)
        test_mxfp4_matmul[1, 2048, 7168](ctx)
        test_mxfp4_matmul[1, 4096, 7168](ctx)
        test_mxfp4_matmul[1, 7168, 18432](ctx)
        test_mxfp4_matmul[1, 18432, 7168](ctx)
        test_mxfp4_matmul[1, 36864, 7168](ctx)

        # M=17 (short prefill)
        test_mxfp4_matmul[17, 7168, 2048](ctx)
        test_mxfp4_matmul[17, 2048, 7168](ctx)
        test_mxfp4_matmul[17, 4096, 7168](ctx)
        test_mxfp4_matmul[17, 18432, 7168](ctx)

        # M=53 (mid-range unaligned)
        test_mxfp4_matmul[53, 7168, 2048](ctx)
        test_mxfp4_matmul[53, 7168, 18432](ctx)

        # M=73 (mid-range unaligned)
        test_mxfp4_matmul[73, 4096, 7168](ctx)
        test_mxfp4_matmul[73, 7168, 18432](ctx)
        test_mxfp4_matmul[73, 36864, 7168](ctx)

        # M=111 (near BM=128 boundary — last block is 111 rows, 17-row short)
        test_mxfp4_matmul[111, 7168, 2048](ctx)
        test_mxfp4_matmul[111, 2048, 7168](ctx)
        test_mxfp4_matmul[111, 18432, 7168](ctx)

        # M=129 (crosses 1 full block + 1-row partial)
        test_mxfp4_matmul[129, 7168, 2048](ctx)
        test_mxfp4_matmul[129, 4096, 7168](ctx)

        # M=257 (crosses 2 full blocks + 1-row partial)
        test_mxfp4_matmul[257, 7168, 2048](ctx)
        test_mxfp4_matmul[257, 18432, 7168](ctx)

        print("\n--- B': exaggerated OOB stress ---")

        # M = BM - 1 — last block is one row short of full.
        # Maximum partial-block footprint (127 real rows, 1 OOB).
        test_mxfp4_matmul[127, 7168, 2048](ctx)
        test_mxfp4_matmul[127, 36864, 7168](ctx)

        # M = 2*BM - 1 — one full block + 127-row partial.
        test_mxfp4_matmul[255, 7168, 8192](ctx)

        # M=1 + huge N + deepest K — maximum DRAM/scale volume with 1
        # real row and 127 OOB rows per block.
        test_mxfp4_matmul[1, 36864, 18432](ctx)
        print("\n--- T: tile-shape parameter sweep ---")

        # Baseline (same as default Kernel) — sanity check.
        test_mxfp4_matmul[128, 128, 512, BM=128, BN=128, BK_ELEMS=128](ctx)

        # Deeper BK: num_k_tiles=2, enables Level 1 intra-BK pipelining.
        test_mxfp4_matmul[128, 128, 512, BM=128, BN=128, BK_ELEMS=256](ctx)
        test_mxfp4_matmul[256, 256, 1024, BM=128, BN=128, BK_ELEMS=256](ctx)

        # Wider M block: 8 warps/block, same warp tile.
        test_mxfp4_matmul[256, 128, 512, BM=256, BN=128, BK_ELEMS=128](ctx)
        test_mxfp4_matmul[512, 128, 1024, BM=256, BN=128, BK_ELEMS=128](ctx)

        # Wider N block.
        test_mxfp4_matmul[128, 256, 512, BM=128, BN=256, BK_ELEMS=128](ctx)

        # Biggest block we can run at 1024 threads/workgroup: 256×256
        # with WM=WN=64 = 16 warps = 1024 threads (at the limit).
        test_mxfp4_matmul[256, 256, 512, BM=256, BN=256, BK_ELEMS=128](ctx)

        # Combined: bigger block + deeper BK (the most-likely-fastest
        # config for Kimi medium-M shapes).
        test_mxfp4_matmul[256, 128, 1024, BM=256, BN=128, BK_ELEMS=256](ctx)

        # Partial-block with non-default tile: makes sure OOB handling
        # scales with BM.
        test_mxfp4_matmul[73, 4096, 7168, BM=256, BN=128, BK_ELEMS=128](ctx)

        print("\n--- T2: small-M tuning configs (BM=64, BN=32, WN=32) ---")

        # K=2048 → K_BYTES=1024. Verify at each BK_ELEMS that K divides.
        # 128 → 1024/64 = 16 iters (÷) ✓
        test_mxfp4_matmul[
            32,
            7168,
            2048,
            BM=64,
            BN=32,
            BK_ELEMS=128,
            WN=32,
        ](ctx)
        # 256 → 1024/128 = 8 iters ✓
        test_mxfp4_matmul[
            32,
            7168,
            2048,
            BM=64,
            BN=32,
            BK_ELEMS=256,
            WN=32,
        ](ctx)
        # 512 → 1024/256 = 4 iters ✓
        test_mxfp4_matmul[
            32,
            7168,
            2048,
            BM=64,
            BN=32,
            BK_ELEMS=512,
            WN=32,
        ](ctx)

        test_mxfp4_matmul[
            32,
            7168,
            2048,
            BM=64,
            BN=32,
            BK_ELEMS=1024,
            WN=32,
        ](ctx)

        print("\n==== All MXFP4 block-scaled matmul tests passed ====")
