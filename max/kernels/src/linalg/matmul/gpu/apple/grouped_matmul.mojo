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
"""Apple M5 grouped (MoE) matmul for bf16 or fp16 operands: a grouped GEMV for
decode and a grouped simdgroup-MMA GEMM for prefill.

Target: Apple M5 (`compute_capability == 5`, Metal 4).

For each expert group `z`, `C[a_offsets[z]:a_offsets[z+1]] =
A[a_offsets[z]:a_offsets[z+1]] @ b[expert_ids[z]]^T`, with `b` the weight
stack `[num_experts, N, K]` and fp32 accumulation. `A` and `b` share one
16-bit float type (bf16 or fp16). One dispatch covers every
group (`block_idx.z`), as in `matmul2d_fp8.Matmul2dFp8.run_grouped`.

Two kernels, picked on the host by `max_num_tokens_per_expert`:

- Decode (few tokens per expert): `grouped_gemv_kernel`. At 1-2 tokens per
  expert there is no matrix to feed the 16x16 MMA, and a 64-row MMA tile wastes
  most of its rows, so this is a register-resident GEMV that streams each
  active expert's weight slab once per pass. One simdgroup owns `rows_per_sg`
  consecutive weight rows; its 32 lanes stride down K with 16-byte loads, so a
  simdgroup reads `rows_per_sg` contiguous 512-byte runs per step. The
  `tokens_per_pass` tokens of a pass reuse the same weight registers, so a
  pass reads the weight once. The ceiling is weight-read bandwidth.
- Prefill: `grouped_matmul_mma_kernel` drives the dense
  `AppleM5MatMul._run_gemm_body` (the NT 16-bit `use_x2` configuration
  `enqueue_apple_matmul` picks) on per-group views, so the grouped path
  shares the dense kernel's MMA, Morton tile order, bounded edges and epilogue.
  The host dispatches a few groups at a time (see
  `enqueue_apple_grouped_mma`).
"""

from std.bit import log2_ceil
from std.math import ceildiv
from std.sys import align_of, size_of

from max.gpu import WARP_SIZE, block_idx, lane_id, thread_idx
from max.gpu.host import DeviceContext
import max.gpu.primitives.warp as warp

from layout import Coord, Idx, TensorEngine, TileTensor
from layout.tile_layout import Layout, TensorLayout, row_major
from std.utils import IndexList

from linalg.arch.apple.mma import ConvIm2colParams
from linalg.matmul.gpu.apple.matmul2d_fp4 import _require_apple_m5
from linalg.matmul.gpu.apple.matmul_kernel import (
    AppleM5MatMul,
    DenseALoader,
    DenseWeightLoader,
)
from linalg.utils import (
    ElementwiseComputeFn,
    ElementwiseEpilogueFn,
    no_compute_fn,
    no_epilogue_fn,
)


# Largest per-expert token count routed to the GEMV. At 3072-6144-wide expert
# dims on M5 Max the GEMV holds ~560 GB/s through 8 tokens and still beats the
# MMA at 16 (two passes, 420-450 vs 370-380 GB/s); the MMA wins from 32 up.
comptime APPLE_GROUPED_GEMV_MAX_TOKENS = 16


@__name(
    t"apple_grouped_gemv_{a_type}_{b_type}_{c_type}_n{N}_k{K}_r{rows_per_sg}"
    t"_t{tokens_per_pass}_a{acc_width}"
)
def grouped_gemv_kernel[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    N: Int,
    K: Int,
    rows_per_sg: Int,
    tokens_per_pass: Int,
    num_sg: Int,
    acc_width: Int,
    c_layout: TensorLayout,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    ao_layout: TensorLayout,
    ei_layout: TensorLayout,
    c_engine: TensorEngine,
    a_engine: TensorEngine,
    b_engine: TensorEngine,
    ao_engine: TensorEngine,
    ei_engine: TensorEngine,
    EpilogueFnType: ElementwiseEpilogueFn,
    has_epilogue_fn: Bool,
    ComputeFnType: ElementwiseComputeFn,
    has_compute_fn: Bool,
](
    c: TileTensor[c_type, c_layout, MutAnyOrigin, Engine=c_engine],
    a: TileTensor[a_type, a_layout, ImmutAnyOrigin, Engine=a_engine],
    b: TileTensor[b_type, b_layout, ImmutAnyOrigin, Engine=b_engine],
    a_offsets: TileTensor[
        mut=False, .uint32, ao_layout, MutAnyOrigin, Engine=ao_engine
    ],
    expert_ids: TileTensor[
        mut=False, .int32, ei_layout, MutAnyOrigin, Engine=ei_engine
    ],
    epilogue_fn: EpilogueFnType,
    compute_fn: ComputeFnType,
):
    """Grouped GEMV: one simdgroup per `rows_per_sg` output columns of a
    group.

    Grid `(ceil(N / (rows_per_sg * num_sg)), 1, num_active_experts)`. Tokens of
    the group are processed `tokens_per_pass` at a time; the weight chunk is
    loaded once per step and reused by every token of the pass.
    `expert_ids[group] == -1` writes zeros, matching
    `naive_grouped_matmul_kernel`. With `has_epilogue_fn`, each output goes
    to `epilogue_fn` at its `(row, col)` in `c` instead. With
    `has_compute_fn`, `compute_fn` maps each output before it is stored.
    """
    # 8 16-bit elements = 16 bytes per lane per load. The vector load needs
    # its 16-byte alignment stated: `<8 x half>` at 2-byte alignment does not
    # vectorize on M5.
    comptime VW = 8
    comptime STEP = WARP_SIZE * VW
    comptime K_STEPS = K // STEP
    comptime K_VEC = K_STEPS * STEP
    # Every group slab starts at a multiple of K elements, so 16-byte
    # alignment holds exactly when K is a multiple of 8.
    comptime A_ALIGN = align_of[Scalar[a_type]]() * (VW if K % VW == 0 else 1)
    comptime B_ALIGN = align_of[Scalar[b_type]]() * (VW if K % VW == 0 else 1)

    var group = Int(block_idx.z)
    var a_start = Int(a_offsets[group])
    var m = Int(a_offsets[group + 1]) - a_start
    if m <= 0:
        return

    var sg = Int(thread_idx.x) // WARP_SIZE
    var lid = Int(lane_id())
    var n0 = (Int(block_idx.x) * num_sg + sg) * rows_per_sg
    if n0 >= N:
        return

    var expert = Int(expert_ids[group])
    var active = expert != -1

    # Static views keep the hot loop's index math int32; only the tile bases
    # need `b`'s and `a`'s wider index types.
    var b_e = b.tile[1, N, K](max(expert, 0), 0, 0).reshape(row_major[N, K]())

    # The last simdgroup can run past N; clamping keeps its loads in bounds
    # and the store below drops those rows.
    var w_rows = Array[Int, rows_per_sg](fill=0)
    comptime for r in range(rows_per_sg):
        w_rows[r] = min(n0 + r, N - 1)

    var t0 = 0
    while t0 < m:
        # A pass's view can extend past the group; those rows are clamped
        # away on load and skipped on store.
        var a_pass = a.tile[1, K](a_start + t0, 0).reshape(
            row_major[tokens_per_pass, K]()
        )
        var c_pass = c.tile[1, N](a_start + t0, 0).reshape(
            row_major[tokens_per_pass, N]()
        )
        var pass_rows = min(tokens_per_pass, m - t0)
        var t_rows = Array[Int, tokens_per_pass](fill=0)
        comptime for t in range(tokens_per_pass):
            t_rows[t] = min(t, pass_rows - 1)

        var acc = Array[
            SIMD[.float32, acc_width], rows_per_sg * tokens_per_pass
        ](fill=0)
        if active:
            for s in range(K_STEPS):
                var k0 = s * STEP + lid * VW
                var x = Array[SIMD[.float32, VW], tokens_per_pass](fill=0)
                comptime for t in range(tokens_per_pass):
                    x[t] = a_pass.load[width=VW, alignment=A_ALIGN](
                        Coord(t_rows[t], k0)
                    ).cast[.float32]()
                comptime for r in range(rows_per_sg):
                    var w = b_e.load[width=VW, alignment=B_ALIGN](
                        Coord(w_rows[r], k0)
                    ).cast[.float32]()
                    comptime for t in range(tokens_per_pass):
                        acc[r * tokens_per_pass + t] += (w * x[t]).reduce_add[
                            acc_width
                        ]()

            comptime if K_VEC < K:
                var kt = K_VEC + lid
                while kt < K:
                    comptime for r in range(rows_per_sg):
                        var w = b_e.load[width=1](Coord(w_rows[r], kt)).cast[
                            .float32
                        ]()
                        comptime for t in range(tokens_per_pass):
                            var xv = a_pass.load[width=1](
                                Coord(t_rows[t], kt)
                            ).cast[.float32]()
                            acc[r * tokens_per_pass + t][0] += (w * xv)[0]
                    kt += WARP_SIZE

        # Every lane gets every sum; lane `r` stores output column `n0 + r`,
        # so the `rows_per_sg` outputs of a token go out in one pass.
        comptime for t in range(tokens_per_pass):
            var sums = SIMD[.float32, rows_per_sg](0)
            comptime for r in range(rows_per_sg):
                sums[r] = warp.sum(acc[r * tokens_per_pass + t].reduce_add())
            if lid < rows_per_sg and n0 + lid < N and t < pass_rows:
                var out = SIMD[c_type, 1](sums[lid].cast[c_type]())
                var idx: IndexList[2] = (a_start + t0 + t, n0 + lid)
                comptime if has_epilogue_fn:
                    epilogue_fn[c_type, 1, alignment=1](idx, out)
                else:
                    comptime if has_compute_fn:
                        out = compute_fn[c_type, 1, alignment=1](idx, out)
                    c_pass.store(Coord(t, n0 + lid), out)
        t0 += tokens_per_pass


@__name(
    t"apple_grouped_matmul_mma_{a_type}_{b_type}_{c_type}_{linear_idx_type}"
)
def grouped_matmul_mma_kernel[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    N: Int,
    K: Int,
    linear_idx_type: DType,
    c_layout: TensorLayout,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    ao_layout: TensorLayout,
    ei_layout: TensorLayout,
    c_engine: TensorEngine,
    a_engine: TensorEngine,
    b_engine: TensorEngine,
    ao_engine: TensorEngine,
    ei_engine: TensorEngine,
    EpilogueFnType: ElementwiseEpilogueFn,
    has_epilogue_fn: Bool,
    ComputeFnType: ElementwiseComputeFn,
    has_compute_fn: Bool,
](
    c: TileTensor[c_type, c_layout, MutAnyOrigin, Engine=c_engine],
    a: TileTensor[a_type, a_layout, ImmutAnyOrigin, Engine=a_engine],
    b: TileTensor[b_type, b_layout, ImmutAnyOrigin, Engine=b_engine],
    a_offsets: TileTensor[
        mut=False, .uint32, ao_layout, MutAnyOrigin, Engine=ao_engine
    ],
    expert_ids: TileTensor[
        mut=False, .int32, ei_layout, MutAnyOrigin, Engine=ei_engine
    ],
    first_group: UInt32,
    log2_grid_m: UInt32,
    log2_grid_n: UInt32,
    epilogue_fn: EpilogueFnType,
    compute_fn: ComputeFnType,
):
    """Grouped GEMM: the dense `AppleM5MatMul` body on group
    `first_group + block_idx.z`.

    Grid `((1 << log2_grid_m) * (1 << log2_grid_n), 1, groups)`, with the M
    extent sized for the largest group; tiles past a smaller group's rows
    early-return in the body. This is `AppleM5MatMul.run` with per-group views
    in place of the whole-matrix operands. With `has_epilogue_fn`, each output
    goes to `epilogue_fn` at its `(row, col)` in `c` instead. With
    `has_compute_fn`, `compute_fn` maps each output before it is stored.
    """
    comptime MM = _GroupedMma[a_type, b_type, c_type, linear_idx_type]

    var group = Int(first_group) + Int(block_idx.z)
    var a_start = Int(a_offsets[group])
    var m = Int(a_offsets[group + 1]) - a_start
    if m <= 0:
        return
    var expert = Int(expert_ids[group])
    # An inactive group (`expert_ids == -1`) runs with K = 0: no K strips, so
    # the zero accumulator is stored (or passed to the epilogue), matching
    # `naive_grouped_matmul_kernel`.
    var k = K if expert != -1 else 0

    var c_g = c.tile[1, N](a_start, 0).reshape(row_major(Coord(m, Idx[N])))
    var b_e = b.tile[1, N, K](max(expert, 0), 0, 0).reshape(
        row_major(Coord(Idx[N], k))
    )

    # Pre-tile this simdgroup's A slab outside the K-loop, as `run` does,
    # including its untracked origin, which `DenseALoader` stores.
    var row_base = MM._sg_row_base(log2_grid_m, log2_grid_n)
    var sg_row_idx = row_base // Int32(MM.SG_M)
    var a_mat = TileTensor[linear_idx_type=MM.linear_idx_type](
        a.tile[1, K](a_start, 0).ptr.unsafe_origin_cast[ImmUntrackedOrigin](),
        Layout(Coord(m, k), Coord(Idx[K], Idx[1])),
    )
    var a_slab = a_mat.tile(Coord(Idx[MM.SG_M], k), Coord(Int(sg_row_idx), 0))
    var loader = DenseALoader[
        a_type,
        type_of(a_slab).LayoutType,
        BK=MM.BK,
        SG_M=MM.SG_M,
        use_x2=MM.use_x2,
        b_dtype=b_type,
    ](a_slab)

    # The body indexes this group's rows from 0; the epilogues take rows of
    # all of `c`.
    var c_ptr = c.ptr

    @inline(.always)
    def group_epilogue_fn[
        dtype: DType, width: SIMDLength, *, alignment: Int
    ](idx: IndexList[2], val: SIMD[dtype, width]) {
        var epilogue_fn, var compute_fn, var c_ptr, var a_start
    }:
        var row_idx: IndexList[2] = (a_start + idx[0], idx[1])
        comptime if has_compute_fn:
            (c_ptr + row_idx[0] * N + idx[1]).store[alignment=alignment](
                rebind[SIMD[c_type, width]](
                    compute_fn[dtype, width, alignment=alignment](row_idx, val)
                )
            )
        else:
            epilogue_fn[dtype, width, alignment=alignment](row_idx, val)

    MM._run_gemm_body[
        W=DenseWeightLoader[b_type, DType.float32],
        has_epilogue_fn=has_epilogue_fn or has_compute_fn,
    ](
        loader,
        c_g,
        b_e.as_imm(),
        k,
        ConvIm2colParams(),
        log2_grid_m,
        log2_grid_n,
        epilogue_fn=group_epilogue_fn,
    )


# The NT 16-bit configuration `enqueue_apple_matmul` dispatches: BK=32 double
# strip (`use_x2`), offsets relative to one group's slab.
comptime _GroupedMma[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    linear_idx_type: DType = .int32,
] = AppleM5MatMul[
    a_type,
    c_type,
    transpose_b=True,
    block_k=32,
    use_x2=True,
    linear_idx_type=linear_idx_type,
    b_type=b_type,
]


def _assert_supported_types[c_type: DType, a_type: DType, b_type: DType]():
    # `b_type` is separate so a weight-only quantized B can join later; the
    # `use_x2` MMA configuration and the GEMV are validated for A == B only.
    comptime assert a_type == b_type and (
        a_type == .bfloat16 or a_type == .float16
    ), "Apple grouped matmul: A and B must both be bf16 or both be fp16"
    comptime assert (
        c_type == .float16 or c_type == .bfloat16 or c_type == .float32
    ), "Apple grouped matmul: c_type must be one of {fp16, bf16, fp32}"


@inline(.always)
def enqueue_apple_grouped_gemv[
    EpilogueFnType: ElementwiseEpilogueFn = type_of(no_epilogue_fn),
    ComputeFnType: ElementwiseComputeFn = type_of(no_compute_fn),
    //,
    *,
    rows_per_sg: Int = 2,
    tokens_per_pass: Int = 1,
    num_sg: Int = 8,
    acc_width: Int = 8,
    has_epilogue_fn: Bool = False,
    has_compute_fn: Bool = False,
](
    c: TileTensor[mut=True, ...],
    a: TileTensor[...],
    b: TileTensor[...],
    a_offsets: TileTensor[mut=False, .uint32, ...],
    expert_ids: TileTensor[mut=False, .int32, ...],
    num_active_experts: Int,
    ctx: DeviceContext,
    epilogue_fn: EpilogueFnType = no_epilogue_fn,
    compute_fn: ComputeFnType = no_compute_fn,
) raises:
    """Enqueues the grouped GEMV.

    `b` is the weight stack `[num_experts, N, K]` with static `N` and `K`;
    `a` is `[total_M, K]` and `c` is `[total_M, N]`, both token-major. `a`
    and `b` are both bf16 or both fp16; `c` is fp16, bf16 or fp32, and
    accumulation is fp32. Correct for any group size, but it re-reads the
    weight every `tokens_per_pass` rows, so the dispatch only uses it for
    small groups.

    Parameters:
        rows_per_sg: Weight rows (output columns) per simdgroup.
        tokens_per_pass: Tokens of a group accumulated per pass over the
            weight.
        num_sg: Simdgroups per threadgroup.
        acc_width: Fp32 lanes kept per accumulator (1 to 8); narrower trades
            adds for registers when `rows_per_sg * tokens_per_pass` is large.
        has_epilogue_fn: Whether `epilogue_fn` stores each output at its
            `(row, col)` in `c`.
        has_compute_fn: Whether `compute_fn` maps each output before it is
            stored.
    """
    _assert_supported_types[c.dtype, a.dtype, b.dtype]()
    comptime N = type_of(b).static_shape[1]
    comptime K = type_of(b).static_shape[2]
    if num_active_experts == 0:
        return

    comptime kernel = grouped_gemv_kernel[
        c.dtype,
        a.dtype,
        b.dtype,
        N,
        K,
        rows_per_sg,
        tokens_per_pass,
        num_sg,
        acc_width,
        type_of(c).LayoutType,
        type_of(a).LayoutType,
        type_of(b).LayoutType,
        type_of(a_offsets).LayoutType,
        type_of(expert_ids).LayoutType,
        type_of(c).Engine,
        type_of(a).Engine,
        type_of(b).Engine,
        type_of(a_offsets).Engine,
        type_of(expert_ids).Engine,
        EpilogueFnType,
        has_epilogue_fn,
        ComputeFnType,
        has_compute_fn,
    ]
    ctx.enqueue_function[kernel](
        c,
        a.as_imm(),
        b.as_imm(),
        a_offsets,
        expert_ids,
        host_arg=epilogue_fn,
        host_arg2=compute_fn,
        grid_dim=(ceildiv(N, rows_per_sg * num_sg), 1, num_active_experts),
        block_dim=(num_sg * WARP_SIZE),
    )


@inline(.always)
def enqueue_apple_grouped_mma[
    EpilogueFnType: ElementwiseEpilogueFn = type_of(no_epilogue_fn),
    ComputeFnType: ElementwiseComputeFn = type_of(no_compute_fn),
    //,
    *,
    groups_per_launch: Int = 4,
    has_epilogue_fn: Bool = False,
    has_compute_fn: Bool = False,
](
    c: TileTensor[mut=True, ...],
    a: TileTensor[...],
    b: TileTensor[...],
    a_offsets: TileTensor[mut=False, .uint32, ...],
    expert_ids: TileTensor[mut=False, .int32, ...],
    max_num_tokens_per_expert: Int,
    num_active_experts: Int,
    ctx: DeviceContext,
    epilogue_fn: EpilogueFnType = no_epilogue_fn,
    compute_fn: ComputeFnType = no_compute_fn,
) raises:
    """Enqueues the grouped simdgroup-MMA GEMM (any token count).

    Same operand contract as `enqueue_apple_grouped_gemv`.

    One dispatch covering every group runs an MoE prefill at ~20 TF/s
    against ~50 for one dispatch per group: threadgroups from many groups
    run at once and evict each other's weight tiles from cache. Dispatching
    a few groups at a time keeps the concurrent working set small while
    still filling the GPU when groups are short (measured, M5 Max, 32 x
    1024 tokens: 4 groups per dispatch 57-59 TF/s, 1 per dispatch 50, all
    32 at once 20).

    Parameters:
        groups_per_launch: Expert groups per kernel dispatch.
        has_epilogue_fn: Whether `epilogue_fn` stores each output at its
            `(row, col)` in `c`.
        has_compute_fn: Whether `compute_fn` maps each output before it is
            stored.
    """
    comptime c_type = c.dtype
    comptime a_type = a.dtype
    comptime b_type = b.dtype
    _assert_supported_types[c_type, a_type, b_type]()
    comptime N = type_of(b).static_shape[1]
    comptime K = type_of(b).static_shape[2]
    comptime MM = _GroupedMma[a_type, b_type, c_type]
    # The MMA fragment loaders narrow the B row stride (K) to UInt16.
    comptime assert K <= 65535, "Apple grouped matmul: K must fit in UInt16"

    if num_active_experts == 0 or max_num_tokens_per_expert == 0:
        return

    var log2_m = log2_ceil(UInt32(ceildiv(max_num_tokens_per_expert, MM.BM)))
    var log2_n = log2_ceil(UInt32(ceildiv(N, MM.BN)))
    # Same int32 gate as `enqueue_apple_matmul`, on byte extents per group.
    comptime a_bytes = size_of[Scalar[a_type]]()
    comptime b_bytes = size_of[Scalar[b_type]]()
    comptime c_bytes = size_of[Scalar[c_type]]()
    var fits_i32 = (
        max_num_tokens_per_expert * N * c_bytes <= Int(Int32.MAX)
        and max_num_tokens_per_expert * K * a_bytes <= Int(Int32.MAX)
        and N * K * b_bytes <= Int(Int32.MAX)
    )

    var g0 = 0
    while g0 < num_active_experts:
        var ng = min(groups_per_launch, num_active_experts - g0)
        comptime for idx_type in [DType.int32, DType.int64]:
            if fits_i32 == (idx_type == DType.int32):
                comptime kernel = grouped_matmul_mma_kernel[
                    c_type,
                    a_type,
                    b_type,
                    N,
                    K,
                    idx_type,
                    type_of(c).LayoutType,
                    type_of(a).LayoutType,
                    type_of(b).LayoutType,
                    type_of(a_offsets).LayoutType,
                    type_of(expert_ids).LayoutType,
                    type_of(c).Engine,
                    type_of(a).Engine,
                    type_of(b).Engine,
                    type_of(a_offsets).Engine,
                    type_of(expert_ids).Engine,
                    EpilogueFnType,
                    has_epilogue_fn,
                    ComputeFnType,
                    has_compute_fn,
                ]
                ctx.enqueue_function[kernel](
                    c,
                    a.as_imm(),
                    b.as_imm(),
                    a_offsets,
                    expert_ids,
                    UInt32(g0),
                    log2_m,
                    log2_n,
                    host_arg=epilogue_fn,
                    host_arg2=compute_fn,
                    grid_dim=((1 << Int(log2_m)) * (1 << Int(log2_n)), 1, ng),
                    block_dim=(MM.THREADS_PER_BLOCK),
                )
        g0 += ng


@inline(.always)
def enqueue_apple_grouped_matmul[
    EpilogueFnType: ElementwiseEpilogueFn = type_of(no_epilogue_fn),
    ComputeFnType: ElementwiseComputeFn = type_of(no_compute_fn),
    //,
    *,
    has_epilogue_fn: Bool = False,
    has_compute_fn: Bool = False,
](
    c: TileTensor[mut=True, ...],
    a: TileTensor[...],
    b: TileTensor[...],
    a_offsets: TileTensor[mut=False, .uint32, ...],
    expert_ids: TileTensor[mut=False, .int32, ...],
    max_num_tokens_per_expert: Int,
    num_active_experts: Int,
    ctx: DeviceContext,
    epilogue_fn: EpilogueFnType = no_epilogue_fn,
    compute_fn: ComputeFnType = no_compute_fn,
) raises:
    """Enqueues the Apple M5 grouped matmul `C = A @ b[expert]^T`.

    Routes on the largest group: up to `APPLE_GROUPED_GEMV_MAX_TOKENS` tokens
    per expert takes the GEMV (decode), anything larger the MMA GEMM
    (prefill). `b` is `[num_experts, N, K]` with static `N` and `K`; the
    operand types and epilogues are those of `enqueue_apple_grouped_gemv`.

    Raises:
        If the attached GPU is not Apple M5 (`compute_capability == 5`).
    """
    _require_apple_m5(ctx)

    # GEMV tilings measured best at 3072-6144-wide expert dims (M5 Max):
    # the tokens of a group share one read of the weight; more rows per
    # simdgroup only pay off when the accumulators are narrow enough to
    # stay in registers.
    if max_num_tokens_per_expert <= 1:
        enqueue_apple_grouped_gemv[
            rows_per_sg=2,
            tokens_per_pass=1,
            has_epilogue_fn=has_epilogue_fn,
            has_compute_fn=has_compute_fn,
        ](
            c,
            a,
            b,
            a_offsets,
            expert_ids,
            num_active_experts,
            ctx,
            epilogue_fn,
            compute_fn,
        )
    elif max_num_tokens_per_expert <= 4:
        enqueue_apple_grouped_gemv[
            rows_per_sg=1,
            tokens_per_pass=4,
            has_epilogue_fn=has_epilogue_fn,
            has_compute_fn=has_compute_fn,
        ](
            c,
            a,
            b,
            a_offsets,
            expert_ids,
            num_active_experts,
            ctx,
            epilogue_fn,
            compute_fn,
        )
    elif max_num_tokens_per_expert <= APPLE_GROUPED_GEMV_MAX_TOKENS:
        enqueue_apple_grouped_gemv[
            rows_per_sg=2,
            tokens_per_pass=8,
            acc_width=1,
            has_epilogue_fn=has_epilogue_fn,
            has_compute_fn=has_compute_fn,
        ](
            c,
            a,
            b,
            a_offsets,
            expert_ids,
            num_active_experts,
            ctx,
            epilogue_fn,
            compute_fn,
        )
    else:
        enqueue_apple_grouped_mma[
            has_epilogue_fn=has_epilogue_fn,
            has_compute_fn=has_compute_fn,
        ](
            c,
            a,
            b,
            a_offsets,
            expert_ids,
            max_num_tokens_per_expert,
            num_active_experts,
            ctx,
            epilogue_fn,
            compute_fn,
        )
