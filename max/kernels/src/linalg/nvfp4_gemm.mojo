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
"""Fused NVFP4 dequant-GEMM for pre-Blackwell NVIDIA GPUs (SM80/86/89).

Computes ``C[M, N] = A[M, K] @ W.T`` where ``W`` is ``[N, K // 2]`` packed FP4
(uint8, two E2M1 values per byte) with float32 block scales ``[N, K // 16]``
(modelopt ``weight_scale_2`` pre-multiplied by the caller, group size 16).

This is the batched / prefill sibling of ``nvfp4_gemv``. Unlike the per-token
GEMV, the packed weight is decoded to bf16 ONCE per threadblock K-stage into a
tile in shared memory, then consumed by the synchronous Ampere/Ada tensor
cores (HMMA 16x8x16) exactly like a plain bf16 GEMM. The bf16 weight is never
materialized in DRAM, so weight DRAM traffic is the packed bytes (~0.5
B/element) plus tiny f32 scales -- but the decode is amortized across all
``BM`` activation rows, so throughput scales with M.

Design notes
------------
* Synchronous ``TensorCore`` only (HMMA 16x8x16). NO WGMMA/TMA/tcgen05, so it
  is valid on SM89 (L40S).
* ``transpose_b=True``: the logical weight is ``[N, K]`` so the contraction is
  over K; the decoded B tile is laid out ``row_major(BN, BK)`` and consumed
  transposed by the standard ldmatrix path.
* The standard (non-quant) ``load_b`` for NVIDIA reads the B SMEM tile through
  an ldmatrix swizzle derived from the tile's leading-dim stride
  (``tensor_core.mojo`` ``_load_b_nvidia``). The decode therefore scatters the
  decoded bf16 at the matching swizzled 8-element-vector offsets so the
  swizzled read lands on them.
* The block scale is folded into the bf16 weight at decode time, so the inner
  MMA loop is plain bf16 x bf16 -> f32. All NVFP4 specificity is confined to
  the producer side.
"""

from std.bit import log2_floor
from std.math import ceildiv
from std.sys import align_of, is_nvidia_gpu, simd_width_of, size_of

from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    barrier,
    block_dim,
    block_idx,
    lane_id,
    thread_idx,
)
from std.gpu.host import DeviceContext, FuncAttribute
from std.gpu.memory import (
    AddressSpace,
    async_copy,
    async_copy_commit_group,
    async_copy_wait_group,
    external_memory,
)
from layout import LayoutTensor, TileTensor
from layout.layout import Layout
from layout.layout_tensor import (
    LayoutTensorIter,
    copy_dram_to_sram_async,
)
from layout.swizzle import make_ldmatrix_swizzle
from layout.tensor_core import TensorCore, get_fragment_size, get_mma_shape

from std.utils import StaticTuple
from std.utils.numerics import get_accum_type

from .fp4_utils import cast_uint_to_fp4e2m1

comptime NVFP4_GEMM_SF_VECTOR_SIZE = 16
"""Elements covered by one NVFP4 block scale (group size)."""


# ===----------------------------------------------------------------------=== #
# Kernel
# ===----------------------------------------------------------------------=== #
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32((BM // WM) * (BN // WN) * WARP_SIZE)
    )
)
@__name(t"nvfp4_gemm_{c_type}_{a_type}_BM{BM}_BN{BN}_BK{BK}")
def _nvfp4_gemm_kernel[
    c_type: DType,
    a_type: DType,
    c_layout: Layout,
    a_layout: Layout,
    w_layout: Layout,
    s_layout: Layout,
    *,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    num_pipeline_stages: Int,
    stage_w: Bool = False,
    split_k: Int = 1,
](
    c: LayoutTensor[mut=True, c_type, c_layout, MutAnyOrigin],
    a: LayoutTensor[mut=False, a_type, a_layout, ImmutAnyOrigin],
    w_packed: LayoutTensor[mut=False, DType.uint8, w_layout, ImmutAnyOrigin],
    scales: LayoutTensor[mut=False, DType.float32, s_layout, ImmutAnyOrigin],
    workspace: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
):
    comptime assert is_nvidia_gpu(), "nvfp4_gemm only supports NVIDIA GPUs"
    comptime assert a_type == DType.bfloat16, "activations must be bfloat16"
    comptime GROUP = NVFP4_GEMM_SF_VECTOR_SIZE  # 16
    comptime assert BK % GROUP == 0, "BK must be a multiple of the group size"
    comptime assert BK % 2 == 0, "BK must be even (2 FP4 per packed byte)"

    comptime accum_type = get_accum_type[a_type]()  # float32
    comptime simd_a = simd_width_of[a_type]()  # 8 for bf16

    comptime num_warps_m = BM // WM
    comptime num_warps_n = BN // WN
    comptime num_threads = num_warps_m * num_warps_n * WARP_SIZE

    comptime byte_cols = BK // 2  # packed-byte columns per K-tile

    var tid = Int(thread_idx.x)
    var warp_id = tid // WARP_SIZE
    var warp_y = warp_id // num_warps_n
    var warp_x = warp_id % num_warps_n

    var block_n = Int(block_idx.x)  # tiles along N (BN)
    var block_m = Int(block_idx.y)  # tiles along M (BM)

    # Split-K: each z-block owns a contiguous range of K-tiles. The K range is
    # partitioned in units of whole BK tiles so scale-group / packed-byte
    # indexing stays aligned. The grid z dimension is `split_k`.
    var total_k_iters = ceildiv(k, BK)
    var iters_per_slice = ceildiv(total_k_iters, split_k)
    var slice_id = Int(block_idx.z)
    var k_tile_start = slice_id * iters_per_slice
    var num_k_iters = min(k_tile_start + iters_per_slice, total_k_iters) - (
        k_tile_start
    )

    # ---- Shared memory: circular multistage A (bf16) + one decoded B tile. -- #
    # Layout: [ A (num_pipeline_stages * BM * BK bf16) | B (BN * BK bf16) ]
    comptime alignment = align_of[SIMD[a_type, simd_a]]()
    var smem = external_memory[
        Scalar[a_type],
        address_space=AddressSpace.SHARED,
        alignment=alignment,
    ]()

    comptime a_smem_size = num_pipeline_stages * BM * BK
    comptime AIter = LayoutTensorIter[
        a_type,
        Layout.row_major(BM, BK),
        _,
        address_space=AddressSpace.SHARED,
        alignment=alignment,
        circular=True,
    ]
    var a_smem_iter = AIter(smem, AIter.linear_uint_type(a_smem_size))

    # B (decoded bf16) only needs 2 live slots: the tile being consumed by the
    # MMAs and the next tile being decoded one iteration ahead. Using fewer B
    # stages than the A/W pipeline depth shrinks the SMEM footprint and raises
    # block occupancy (the dominant lever at M=64).
    comptime b_stages = min(2, num_pipeline_stages)
    comptime b_smem_size = b_stages * BN * BK
    comptime BIter = LayoutTensorIter[
        a_type,
        Layout.row_major(BN, BK),
        _,
        address_space=AddressSpace.SHARED,
        alignment=alignment,
        circular=True,
    ]
    var b_smem_iter = BIter(
        smem + a_smem_size, BIter.linear_uint_type(b_smem_size)
    )

    # ---- Packed weight access. Two modes (comptime `stage_w`):
    #   * stage_w=False (small M, e.g. M<=64): decode reads packed uint8
    #     straight from DRAM into registers. ncu shows the M<=64 kernel is
    #     bound by the shared-memory/LSU pipe (~75% of peak) while DRAM sits at
    #     ~27%, so a W->SMEM->W round-trip is pure SMEM-pipe overhead; decoding
    #     from registers cuts SMEM write traffic and is faster here.
    #   * stage_w=True (larger M): cp.async the packed bytes into a circular
    #     SMEM staging buffer one+ stages ahead, hiding the DRAM latency behind
    #     the MMAs. At larger M the MMA work dominates and DRAM latency (not the
    #     SMEM pipe) is the risk, so staging wins.
    var w_gmem = w_packed.ptr  # global uint8 base, row-major [N, K//2]
    var w_packed_cols = w_packed.dim[1]()  # K // 2

    comptime W_BYTES = 16  # one cp.async transfer
    comptime w_stage_bytes = BN * byte_cols
    comptime num_w_chunks = w_stage_bytes // W_BYTES
    var w_smem = (smem + a_smem_size + b_smem_size).bitcast[
        Scalar[DType.uint8]
    ]()

    @always_inline
    @parameter
    def _copy_w_stage(k_iter: Int, stage: Int):
        comptime if stage_w:
            comptime assert byte_cols % W_BYTES == 0 or W_BYTES % byte_cols == 0
            var dst_base = w_smem + stage * w_stage_bytes
            var k0_byte = k_iter * byte_cols
            var c = tid
            while c < num_w_chunks:
                var byte_lin = c * W_BYTES
                var n_local = byte_lin // byte_cols
                var c_byte = byte_lin % byte_cols
                var n_glob = block_n * BN + n_local
                var src = w_gmem + n_glob * w_packed_cols + (k0_byte + c_byte)
                var pred = n_glob < n and (k0_byte + c_byte) < (k // 2)
                async_copy[W_BYTES](
                    src.address_space_cast[AddressSpace.GLOBAL](),
                    dst_base + byte_lin,
                    src_size=Int32(W_BYTES) if pred else Int32(0),
                )
                c += num_threads

    # ---- A global iterator (cp.async, ldmatrix-swizzled). ------------------ #
    var a_gmem_iter = a.tiled_iterator[BM, BK, axis=1](block_m, k_tile_start)

    comptime async_copy_a_layout = Layout.row_major(
        num_threads * simd_a // BK, BK // simd_a
    )

    @always_inline
    @parameter
    def _copy_a_stage(stage: Int):
        var dst = a_smem_iter.next_unsafe(a_smem_iter.linear_uint_type(stage))[]
        copy_dram_to_sram_async[
            src_thread_layout=async_copy_a_layout,
            dst_thread_layout=async_copy_a_layout,
            swizzle=True,
        ](
            dst.vectorize[1, simd_a](),
            a_gmem_iter[]
            .bitcast[a_type, target_address_space=AddressSpace.GENERIC]()
            .vectorize[1, simd_a](),
        )
        a_gmem_iter._incr()

    # ---- B decode: packed uint8 (DRAM) -> bf16 SMEM (scale folded). -------- #
    # Each thread decodes a contiguous run of bytes, 4 bytes (= 8 FP4 = half a
    # scale group) per vectorized step. cast_uint_to_fp4e2m1 with a 4-wide uint8
    # input produces 8 decoded values, all within one scale group (group size 16
    # = 8 bytes), so they share a single group scale. A 4-byte vector (vs 8) was
    # the key occupancy win: it halves the f32 decode transients
    # (man/exp/pow2/mag SIMD vectors), cutting register pressure enough to fit a
    # third resident block per SM. DECODE_W == simd_a (8), so the swizzled
    # scatter is a single 16-byte vector store per vec.
    comptime BYTES_PER_VEC = 4
    comptime DECODE_W = BYTES_PER_VEC * 2  # 2 FP4 per byte
    comptime total_bytes = BN * byte_cols
    comptime assert total_bytes % BYTES_PER_VEC == 0
    comptime num_vecs = total_bytes // BYTES_PER_VEC

    # The standard NVIDIA `load_b` reads the B SMEM tile through an ldmatrix
    # swizzle derived from the tile's leading-dim stride (BK). We must store
    # the decoded weights at the SAME swizzled offsets so the swizzled read
    # lands on them. `make_ldmatrix_swizzle[a_type, BK]` (2-arg, default
    # log2_vector_width=0) matches `_load_b_nvidia`'s derivation exactly.
    comptime b_swizzle = make_ldmatrix_swizzle[a_type, BK]()

    @always_inline
    @parameter
    def _decode_b_stage(k_iter: Int, w_stage: Int):
        # Decode the packed weights for K-tile `k_iter` straight from DRAM into
        # the swizzled bf16 B tile at the current circular slot. The packed
        # bytes are read into registers (the DRAM latency overlaps the prior
        # tile's MMAs since the decode runs one stage ahead), decoded, scaled,
        # and scattered to SMEM at the ldmatrix-swizzled offsets.
        var b_ptr = b_smem_iter[].ptr
        var k0 = k_iter * BK
        var k0_byte = k_iter * byte_cols
        var v = tid
        while v < num_vecs:
            var byte_lin = v * BYTES_PER_VEC
            var n_local = byte_lin // byte_cols
            var c_byte = byte_lin % byte_cols
            var k_local = c_byte * 2  # decoded element column within tile
            var n_glob = block_n * BN + n_local
            var gk = k0 + k_local
            var scaled = SIMD[a_type, DECODE_W](0)
            if n_glob < n and gk < k:
                var packed: SIMD[DType.uint8, BYTES_PER_VEC]
                comptime if stage_w:
                    # Packed bytes already cp.async'd into SMEM at `w_stage`.
                    var w_ptr = w_smem + w_stage * w_stage_bytes
                    packed = w_ptr.load[width=BYTES_PER_VEC](byte_lin)
                else:
                    var src = (
                        w_gmem + n_glob * w_packed_cols + (k0_byte + c_byte)
                    )
                    packed = src.load[width=BYTES_PER_VEC]()
                var vals = cast_uint_to_fp4e2m1[
                    out_dtype=DType.float32, out_width=DECODE_W
                ](packed)
                var s = rebind[Scalar[DType.float32]](
                    scales[n_glob, gk // GROUP]
                )
                scaled = (vals * s).cast[a_type]()
            # Store at swizzled 8-element-vector offsets so the ldmatrix
            # swizzle used by `load_b` reads them back correctly. The swizzle
            # operates on vector indices (units of simd_a).
            var vec_base = (n_local * BK + k_local) // simd_a
            comptime for h in range(DECODE_W // simd_a):
                var sw_vec = Int(b_swizzle(vec_base + h))
                b_ptr.store(
                    sw_vec * simd_a, scaled.slice[simd_a, offset=h * simd_a]()
                )
            v += num_threads

    # ---- MMA setup (standard bf16, transpose_b=True). ---------------------- #
    comptime mma_shape = get_mma_shape[a_type, accum_type]()  # 16x8x16
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]
    comptime num_k_mmas = BK // MMA_K
    comptime num_m_mmas = WM // MMA_M
    comptime num_n_mmas = WN // MMA_N

    comptime frag = get_fragment_size[mma_shape]()
    comptime a_frag = frag[0]
    comptime b_frag = frag[1]
    comptime c_frag = frag[2]

    var mma_op = TensorCore[accum_type, a_type, mma_shape, transpose_b=True]()

    comptime a_reg_layout = Layout.row_major(2 * num_m_mmas, a_frag)
    var a_reg_tiles = (
        LayoutTensor[
            mut=True,
            a_type,
            a_reg_layout,
            MutAnyOrigin,
            address_space=AddressSpace.LOCAL,
        ]
        .stack_allocation()
        .split[2]()
    )
    comptime b_reg_layout = Layout.row_major(2 * num_n_mmas, b_frag)
    var b_reg_tiles = (
        LayoutTensor[
            mut=True,
            a_type,
            b_reg_layout,
            MutAnyOrigin,
            address_space=AddressSpace.LOCAL,
        ]
        .stack_allocation()
        .vectorize[1, b_frag]()
        .split[2]()
    )
    comptime c_reg_layout = Layout.row_major(num_m_mmas * num_n_mmas, c_frag)
    var c_reg_tile = (
        LayoutTensor[
            mut=True,
            accum_type,
            c_reg_layout,
            MutAnyOrigin,
            address_space=AddressSpace.LOCAL,
        ]
        .stack_allocation()
        .fill(0)
    )

    # ---- Prologue: cp.async (num_pipeline_stages - 1) stages of A (and, when
    # staging, the raw packed weight bytes W). When not staging, the decode
    # reads packed weight straight from DRAM, so only A is prefetched. ------- #
    comptime for stage in range(num_pipeline_stages - 1):
        _copy_a_stage(stage)
        _copy_w_stage(k_tile_start + stage, stage)
        async_copy_commit_group()
    async_copy_wait_group(Int32(num_pipeline_stages - 2))
    barrier()

    var a_warp0 = a_smem_iter[].tile[WM, BK](warp_y, 0)
    comptime swizzle_a = make_ldmatrix_swizzle[a_type, a_warp0.stride[0]()]()

    # Decode the first K-tile's weights into the B slot the first main-loop
    # iteration will consume. Subsequent tiles are decoded one iteration ahead
    # (below) so the decode overlaps the MMAs.
    _decode_b_stage(k_tile_start, 0)
    barrier()

    # ---- Main loop. -------------------------------------------------------- #
    for k_iter in range(num_k_iters):
        var a_wt = a_smem_iter[].tile[WM, BK](warp_y, 0)
        var b_wt = b_smem_iter[].tile[WN, BK](warp_x, 0)

        # Single-buffered fragments: with only `num_k_mmas` (=2) MMAs per tile
        # the double-buffer prefetch saved little but doubled the B fragment
        # register footprint, capping occupancy at 2 blocks/SM. Loading each
        # k-mma's fragments just-in-time frees those registers for more
        # resident warps, which is what hides the SMEM-pipe latency here.
        comptime for k_mma in range(num_k_mmas):
            mma_op.load_a[swizzle_a](
                a_wt, a_reg_tiles[0].vectorize[1, a_frag](), k_mma
            )
            mma_op.load_b(b_wt, b_reg_tiles[0], k_mma)
            mma_op.mma(
                a_reg_tiles[0].vectorize[1, a_frag](),
                b_reg_tiles[0],
                c_reg_tile.vectorize[1, c_frag](),
            )

        # Prefetch the next K-tile's A (and, when staging, packed-W bytes) via
        # cp.async, and DECODE the next K-tile's B into the slot the
        # just-consumed stage frees. Both happen while the current tile's MMAs
        # are still in flight, so the FP4 decode overlaps the tensor-core work
        # instead of blocking it. The trailing barrier separates these writes
        # from next iteration's reads.
        var prefetch = k_iter + num_pipeline_stages - 1
        if prefetch < num_k_iters:
            _copy_a_stage(num_pipeline_stages - 1)
            _copy_w_stage(
                k_tile_start + prefetch, prefetch % num_pipeline_stages
            )
        async_copy_commit_group()
        a_smem_iter._incr()
        b_smem_iter._incr()
        async_copy_wait_group(Int32(num_pipeline_stages - 2))
        var next_k = k_iter + 1
        if next_k < num_k_iters:
            _decode_b_stage(k_tile_start + next_k, next_k % num_pipeline_stages)
        barrier()

    # ---- Epilogue: write the f32 accumulators to C with explicit bounds
    # masking so partial N/M tiles (e.g. N not a multiple of BN) are safe.
    #
    # For HMMA m16n8k16, c_frag == 4 (= MMA_M*MMA_N/WARP_SIZE). Lane L owns the
    # four accumulator elements e in 0..4 of MMA (m_mma, n_mma) at:
    #   row = m_mma*MMA_M + (L // 4) + (e // 2) * 8
    #   col = n_mma*MMA_N + (L %  4) * 2 + (e %  2)
    var ln = Int(lane_id())
    var warp_m0 = block_m * BM + warp_y * WM
    var warp_n0 = block_n * BN + warp_x * WN

    comptime for n_mma in range(num_n_mmas):
        comptime for m_mma in range(num_m_mmas):
            var frag = c_reg_tile.vectorize[1, c_frag]()[
                n_mma * num_m_mmas + m_mma, 0
            ]
            var row_base = warp_m0 + m_mma * MMA_M + (ln // 4)
            var col_base = warp_n0 + n_mma * MMA_N + (ln % 4) * 2
            comptime for e in range(c_frag):
                var row = row_base + (e // 2) * 8
                var col = col_base + (e % 2)
                if row < m and col < n:
                    comptime if split_k > 1:
                        # Write this K-slice's partial into its own plane of the
                        # f32 workspace [split_k, M, N] -- no atomics, no
                        # contention. A finalize kernel sums the planes into C.
                        workspace[slice_id * (m * n) + row * n + col] = frag[
                            e
                        ].cast[DType.float32]()
                    else:
                        c[row, col] = frag[e].cast[c_type]()


# ===----------------------------------------------------------------------=== #
# Split-K finalize: cast the f32 workspace accumulator into the output C.
# ===----------------------------------------------------------------------=== #
@__name(t"nvfp4_gemm_finalize_{c_type}")
def _nvfp4_gemm_finalize_kernel[
    c_type: DType,
    c_layout: Layout,
    split_k: Int,
](
    c: LayoutTensor[mut=True, c_type, c_layout, MutAnyOrigin],
    workspace: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    m: Int,
    n: Int,
):
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var total = m * n
    if idx < total:
        var row = idx // n
        var col = idx % n
        var acc = Float32(0)
        comptime for s in range(split_k):
            acc += workspace[s * total + idx]
        c[row, col] = acc.cast[c_type]()


# ===----------------------------------------------------------------------=== #
# Host launcher
# ===----------------------------------------------------------------------=== #
@always_inline
def nvfp4_gemm(
    ctx: DeviceContext,
    c: TileTensor,
    a: TileTensor,
    w_packed: TileTensor,
    scales: TileTensor,
    m: Int,
    n: Int,
    k: Int,
) raises:
    """Launches the fused NVFP4 dequant-GEMM (M>1 / batched path).

    Args:
        ctx: Device context for the launch.
        c: Output [m, n], bfloat16 or float32.
        a: Activations [m, k], bfloat16.
        w_packed: Packed FP4 weight [n, k // 2], uint8.
        scales: Float32 block scales [n, k // 16] (per-tensor scale
            pre-multiplied).
        m: Number of activation rows.
        n: Output features (weight rows).
        k: Inner dimension (unpacked element count, multiple of 32).
    """
    comptime assert scales.dtype == DType.float32, "scales must be float32"
    comptime assert (
        w_packed.dtype == DType.uint8
    ), "packed weights must be uint8"
    comptime assert a.dtype == DType.bfloat16, "activations must be bfloat16"
    if k % 32 != 0:
        raise Error("nvfp4_gemm requires k to be a multiple of 32")

    comptime WM = 16
    comptime WN = 64

    var c_lt = c.to_layout_tensor()
    var a_lt = a.to_layout_tensor()
    var w_lt = w_packed.to_layout_tensor()
    var s_lt = scales.to_layout_tensor()

    @always_inline
    @parameter
    def _launch[
        BM: Int,
        BN: Int,
        NS: Int,
        SK: Int = 1,
        SW: Bool = False,
        BK: Int = 64,
    ]() raises:
        comptime num_warps = (BM // WM) * (BN // WN)
        comptime num_threads = num_warps * WARP_SIZE
        comptime b_stages = min(2, NS)  # B needs only 2 live slots
        comptime a_bytes = NS * BM * BK * size_of[a.dtype]()
        comptime b_bytes = b_stages * BN * BK * size_of[a.dtype]()
        comptime w_bytes = NS * BN * (BK // 2) if SW else 0  # packed staging
        comptime c_bytes = BM * BN * size_of[c.dtype]()
        comptime smem = max(a_bytes + b_bytes + w_bytes, c_bytes)
        comptime kernel = _nvfp4_gemm_kernel[
            c.dtype,
            a.dtype,
            c_lt.layout,
            a_lt.layout,
            w_lt.layout,
            s_lt.layout,
            BM=BM,
            BN=BN,
            BK=BK,
            WM=WM,
            WN=WN,
            num_pipeline_stages=NS,
            split_k=SK,
            stage_w=SW,
        ]

        comptime if SK > 1:
            # f32 workspace of SK independent [M, N] planes; each K-slice writes
            # its own plane (no atomics), then a finalize kernel sums the planes
            # into C. No memset needed -- every plane element is fully written.
            var ws = ctx.enqueue_create_buffer[DType.float32](SK * m * n)
            ctx.enqueue_function[kernel](
                c_lt,
                a_lt,
                w_lt,
                s_lt,
                ws.unsafe_ptr(),
                m,
                n,
                k,
                grid_dim=(ceildiv(n, BN), ceildiv(m, BM), SK),
                block_dim=(num_threads, 1, 1),
                shared_mem_bytes=smem,
                func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                    UInt32(smem)
                ),
            )
            comptime FINAL_BLK = 256
            comptime finalize = _nvfp4_gemm_finalize_kernel[
                c.dtype, c_lt.layout, SK
            ]
            ctx.enqueue_function[finalize](
                c_lt,
                ws.unsafe_ptr(),
                m,
                n,
                grid_dim=(ceildiv(m * n, FINAL_BLK), 1, 1),
                block_dim=(FINAL_BLK, 1, 1),
            )
            _ = ws^
        else:
            # Unused dummy workspace (the non-split path writes C directly).
            var dummy_ws = ctx.enqueue_create_buffer[DType.float32](1)
            ctx.enqueue_function[kernel](
                c_lt,
                a_lt,
                w_lt,
                s_lt,
                dummy_ws.unsafe_ptr(),
                m,
                n,
                k,
                grid_dim=(ceildiv(n, BN), ceildiv(m, BM), 1),
                block_dim=(num_threads, 1, 1),
                shared_mem_bytes=smem,
                func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                    UInt32(smem)
                ),
            )
            _ = dummy_ws^

    # M-bucketed dispatch. WN=64 avoids the WN==32 load_b special case.
    # At small M the grid is N-tiles only (~240 blocks), under-filling the SMs
    # and serializing K per block. Split-K multiplies block parallelism without
    # re-decoding more weight per block, the dominant cost at M<=64.
    #
    # The decode runs an 8-wide (4-byte) packed vector per step rather than a
    # 16-wide (8-byte) one: the narrower decode roughly halves the f32 transient
    # working set, which lifts the M<=64 kernel from 2 to 3 blocks/SM (the prior
    # occupancy ceiling) and ~doubles throughput. With that headroom the M<=64
    # tile is BM=64/BN=64/BK=64: the small BN keeps the grid wide (more blocks)
    # while BK=64 gives 4 MMAs/stage to hide the decode + memory latency.
    if m <= 64:
        _launch[64, 64, 2, 4, SW=True, BK=64]()
    else:
        _launch[128, 64, 3, 2, SW=True, BK=32]()
