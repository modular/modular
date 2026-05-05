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
"""4-wave FP8 matmul for AMD MI355X (CDNA4).

Entry point: AMD4WaveMatmul.run()
Host launcher: amd_4wave_matmul()

Hand-written line-by-line port of HipKittens FP8_4wave's
`matmul_device_1024` (BM=64) and `matmul_device_2048` (BM=128).
Mirrors the source kernel's exact structure:

  - 4 mini-iters per loop iter, each with `G_load + frag_load + mma_ABt`
  - Cross-stage register rotation: a[0]/b[0] are reloaded mid-iter from
    `next` stage so iter k+1's first MMA can fire without waiting on LDS
  - Explicit s_waitcnt vmcnt(N) values matching the source's empirical
    tuning (vmcnt(7) / vmcnt(6) / vmcnt(4) / vmcnt(2) / vmcnt(0))
  - Mid-iter s_barrier between mini-iters 2 and 3
  - 2-iter epilogue drain

The SMEM organization, TileLoaderLDS, and QuadrantMmaOp are reused
across the hand-written and framework-driven body strategies; the
two paths share all scaffolding except the loop body itself.
"""

from std.bit import log2_floor
from std.math import ceildiv
from std.sys import size_of, llvm_intrinsic
from std.sys.intrinsics import readfirstlane
from std.utils import Index, IndexList, StaticTuple
from std.utils.numerics import get_accum_type

from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    lane_id,
    warp_id,
)
from std.gpu.host import DeviceContext
from std.gpu.sync import schedule_barrier, s_waitcnt

from layout import TensorLayout, TileTensor
from layout.swizzle import Swizzle
from layout.tile_layout import row_major
from layout.tile_tensor import stack_allocation

from pipeline.config import ScheduleConfig, SchedulingStrategy
from pipeline.geometry import KernelGeometry
from pipeline.pipeline_dsl import ScheduleEntry
from pipeline.program_builder import derive_safe_max_globals
from pipeline.types import _Ops

from structured_kernels.amd_tile_io import RegTileEpilogue, TileLoaderLDS

from ....utils import elementwise_epilogue_type
from .amd_target import mi355x_target
from .amd_4wave_schedule import (
    LOAD_A,
    LOAD_B,
    MMA_LOAD_A,
    MMA_LOAD_B,
    MMA,
    build_schedule,
)
from .matmul_mma import QuadrantMmaOp


# ===----------------------------------------------------------------------=== #
# HK chiplet / L2 swizzle (XCD-stride remap + WGM=4 column-major grouping)
# ===----------------------------------------------------------------------=== #


@always_inline
def _xcd_wgm_swizzle(
    wgid_raw: Int, num_pid_m: Int, num_pid_n: Int
) -> Tuple[Int, Int]:
    """HipKittens chiplet + L2 swizzle for MI355X (CDNA4).

    Stage 1 (XCD remap): MI355X has 8 chiplets (XCDs) with their own
    L2 slices. Raw block_idx maps adjacent wgids to adjacent CUs which
    typically share an XCD; this remap spreads adjacent wgids across
    XCDs.

    Stage 2 (WGM column-major group): walks WGM=4 row-blocks together
    before advancing N. Improves L2 reuse on the shared operand when
    each CU processes multiple WGs of a row-group.

    Mirrors `4_wave.cu` lines 117-136 / 353-372 in HipKittens.

    The swizzle's L2 reuse only pays off when the WG count is large
    enough that each CU runs multiple WGs in one kernel launch — i.e.
    `num_wgs > num_CUs`. For our decode/prefill shapes (typically
    32-1024 WGs vs 304 CUs), the swizzle math would just add overhead
    without any L2 benefit. We gate on `num_wgs > 4 * num_CUs` to
    skip the swizzle on those shapes; only at the largest shapes
    (e.g. 16k³ at BM=BN=128 = 16384 WGs) does it actually help.
    """
    comptime NUM_XCDS = 8
    comptime WGM = 4
    comptime NUM_CUS = 304  # MI355X
    comptime SWIZZLE_THRESHOLD = 4 * NUM_CUS

    var num_wgs = num_pid_m * num_pid_n

    # Trivial row-major decode for small grids: skip the XCD remap +
    # WGM grouping (one divmod here, three on the full path).
    if num_wgs <= SWIZZLE_THRESHOLD or num_wgs % NUM_XCDS != 0:
        var pid_m, pid_n = divmod(wgid_raw, num_pid_n)
        return (pid_m, pid_n)

    # Full HK swizzle: XCD remap + WGM=4 column-major group.
    # intra_xcd is the wg's position within the XCD's allocation
    # (consecutive wgs every NUM_XCDS apart); xcd is which XCD the wg
    # lands on (adjacent raw wgids spread across XCDs).
    var intra_xcd, xcd = divmod(wgid_raw, NUM_XCDS)
    var wgid = xcd * (num_wgs // NUM_XCDS) + intra_xcd
    var num_wgid_in_group = WGM * num_pid_n
    var group_id, intra_group = divmod(wgid, num_wgid_in_group)
    var first_pid_m = group_id * WGM
    var group_size_m = min(num_pid_m - first_pid_m, WGM)
    var pid_n, intra_group_m = divmod(intra_group, group_size_m)
    var pid_m = first_pid_m + intra_group_m
    return (pid_m, pid_n)


# ===----------------------------------------------------------------------=== #
# KernelConfig
# ===----------------------------------------------------------------------=== #


struct KernelConfig(ImplicitlyCopyable, Movable, Writable):
    """Block/warp/MMA shape configuration for 4-wave-simple kernels."""

    var block_shape: IndexList[3]
    """Workgroup tile shape `(BM, BN, BK)`."""
    var warp_shape: IndexList[3]
    """Per-warp tile shape `(WM, WN, WK)`."""
    var mma_shape: IndexList[3]
    """Single-MMA instruction shape `(MMA_M, MMA_N, MMA_K)`."""

    def __init__(
        out self,
        *,
        block_shape: IndexList[3],
        warp_shape: IndexList[3],
        mma_shape: IndexList[3],
    ):
        """Constructs a `KernelConfig` from the three tile shapes.

        Args:
            block_shape: Workgroup tile shape `(BM, BN, BK)`.
            warp_shape: Per-warp tile shape `(WM, WN, WK)`.
            mma_shape: Single-MMA instruction shape `(MMA_M, MMA_N, MMA_K)`.
        """
        self.block_shape = block_shape
        self.warp_shape = warp_shape
        self.mma_shape = mma_shape

    @staticmethod
    def _write_index_list(
        mut writer: Some[Writer], list: IndexList, sep: StaticString
    ):
        comptime for i in range(list.size):
            if i != 0:
                writer.write(sep)
            writer.write(list[i])

    @always_inline
    def num_threads(self) -> Int:
        """Returns the total threads per workgroup (warps x `WARP_SIZE`).

        Returns:
            The number of threads in one workgroup.
        """
        var num_warps = self.block_shape // self.warp_shape
        return num_warps.flattened_length() * WARP_SIZE

    def write_to(self, mut writer: Some[Writer]):
        """Writes a human-readable shape tag to `writer`.

        Args:
            writer: Sink for the rendered config tag.
        """
        writer.write("4wave_config_")
        Self._write_index_list(writer, self.block_shape, "x")
        writer.write("_")
        Self._write_index_list(writer, self.warp_shape, "x")
        writer.write("_")
        Self._write_index_list(writer, self.mma_shape, "x")

    def write_repr_to(self, mut writer: Some[Writer]):
        """Writes a debug representation of this config to `writer`.

        Args:
            writer: Sink for the rendered config tag.
        """
        self.write_to(writer)


struct AMD4WaveMatmul[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    config: KernelConfig,
    /,
    enable_swizzle: Bool,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
]:
    """Hand-written 4-warp 2x2 inline-MMA matmul for AMD MI355X.

    Line-by-line port of HipKittens FP8_4wave's `matmul_device_1024`
    (BM=64) and `matmul_device_2048` (BM=128). No declarative pipeline
    framework — explicit waits, explicit barriers, explicit register
    rotation matching the source's k+2 prefetch pattern.

    Parameters:
        a_type: Input A element type.
        b_type: Input B element type.
        c_type: Output C element type.
        config: KernelConfig with block/warp/mma shapes.
        enable_swizzle: Enable LDS bank conflict avoidance.
        elementwise_lambda_fn: Optional epilogue.
    """

    comptime BM = Self.config.block_shape[0]
    """Workgroup M-tile size."""
    comptime BN = Self.config.block_shape[1]
    """Workgroup N-tile size."""
    comptime BK = Self.config.block_shape[2]
    """Workgroup K-tile size."""

    comptime WM = Self.config.warp_shape[0]
    """Per-warp M-tile size."""
    comptime WN = Self.config.warp_shape[1]
    """Per-warp N-tile size."""

    comptime MMA_M = Self.config.mma_shape[0]
    """Single-MMA M dimension."""
    comptime MMA_N = Self.config.mma_shape[1]
    """Single-MMA N dimension."""
    comptime MMA_K = Self.config.mma_shape[2]
    """Single-MMA K dimension."""

    comptime num_warps_m = Self.BM // Self.WM
    """Number of warps in the M dimension of the workgroup grid."""
    comptime num_warps_n = Self.BN // Self.WN
    """Number of warps in the N dimension of the workgroup grid."""
    comptime total_warps = Self.num_warps_m * Self.num_warps_n
    """Total warps per workgroup (must be 4 for this kernel)."""

    comptime num_m_mmas = Self.WM // Self.MMA_M
    """Number of MMAs per warp in the M dimension."""
    comptime num_n_mmas = Self.WN // Self.MMA_N
    """Number of MMAs per warp in the N dimension."""
    comptime num_k_mmas = Self.BK // Self.MMA_K
    """Number of MMAs per warp in the K dimension."""

    comptime in_type = Self.a_type
    """Input element type (A and B share a type for FP8)."""

    comptime simd_width = 16 // size_of[Self.in_type]()
    """SIMD lane width matched to AMD `buffer_load_lds`'s 16-byte transaction.

    Target-independent (unlike `simd_width_of[in_type]()`, which would
    return the host's SIMD width when this struct is instantiated from a
    comptime context running on the CPU host). `validate_config` asserts
    that `size_of[in_type]` divides 16 evenly; for any dtype that does
    (FP8/BF16/FP16/FP32) this gives the same value as `simd_width_of`
    inside an AMDGPU kernel.
    """
    comptime accum_dtype = get_accum_type[Self.c_type]()
    """Accumulator dtype derived from `c_type`."""

    comptime c_frag_size = Self.MMA_M * Self.MMA_N // WARP_SIZE
    """Per-lane output-fragment width for one MMA."""

    # Half-tile dimensions: each SMEM tile covers one M-subtile (or N-subtile)
    # of the WG-level BM x BK (or BN x BK) block.
    comptime half_BM = Self.BM // 2
    """Half of `BM` — one M-subtile per SMEM stage."""
    comptime half_BN = Self.BN // 2
    """Half of `BN` — one N-subtile per SMEM stage."""
    comptime mma_tile_m = Self.WM // 2
    """Per-quadrant M-tile size consumed by an MMA load."""
    comptime mma_tile_n = Self.WN // 2
    """Per-quadrant N-tile size consumed by an MMA load."""

    # Quadrant counts (pass-through to QuadrantMmaOp).
    comptime quadrant_m_mmas = Self.num_m_mmas // 2
    """Number of MMAs per quadrant in the M dimension."""
    comptime quadrant_n_mmas = Self.num_n_mmas // 2
    """Number of MMAs per quadrant in the N dimension."""

    # Producer thread layout for TileLoaderLDS.
    comptime _thread_cols = Self.BK // Self.simd_width

    # Swizzle geometry (shared between producer-side byte_swizzle and
    # consumer-side mma_swizzle). MI355X has 64 LDS banks x 4 bytes;
    # without swizzling the MMA thread access pattern causes 4-way bank
    # conflicts. FP8 16x16x128 uses a split-LDS path with 16-byte
    # fragments; everything else uses full MMA-fragment loads.
    #   FP8 16x16x128: log_tile=3, frag=16B -> Swizzle(3, 4, 4)
    comptime _elem_size = size_of[Self.in_type]()
    comptime _mma_frag_w = (Self.MMA_M * Self.MMA_K) // WARP_SIZE
    comptime _use_split_lds = (
        Self.in_type.is_float8() and Self.MMA_M == 16 and Self.MMA_K == 128
    )
    comptime _lds_frag_w = 16 if Self._use_split_lds else Self._mma_frag_w
    comptime _swizzle_log_tile = log2_floor(Self.MMA_K // 32) + 1
    comptime _frag_bytes = Self._lds_frag_w * Self._elem_size
    comptime _swizzle_subtile_cols = 4 * Self.simd_width

    # mma_swizzle (consumer side, element-space): base is always
    # log2_floor(_frag_bytes) regardless of dtype.
    comptime mma_swizzle = Optional(
        Swizzle(Self._swizzle_log_tile, log2_floor(Self._frag_bytes), 4)
    ) if Self.enable_swizzle else Optional[Swizzle]()
    """Optional consumer-side MMA swizzle, populated when swizzling is on."""

    # byte_swizzle (producer side, byte-space, for load_to_lds writes):
    # FP8 base matches mma_swizzle; non-FP8 base uses subtile-col math.
    comptime _byte_swizzle_base = (
        log2_floor(Self._frag_bytes) if Self.in_type.is_float8() else (
            log2_floor(Self._swizzle_subtile_cols // 2)
            + log2_floor(Self._elem_size)
        )
    )
    comptime byte_swizzle = Optional(
        Swizzle(Self._swizzle_log_tile, Self._byte_swizzle_base, 4)
    ) if Self.enable_swizzle else Optional[Swizzle]()
    """Optional producer-side (byte-space) SMEM-store swizzle."""

    # Scheduling-relevant geometry (vmcnt-per-prefetch and lgkm-per-
    # frag-load) derived from the kernel's tile shape, dtype, and
    # 4-warp 2x2 quadrant layout. Consumed by the framework arm via
    # `PipelineConfig.vm_per_load_*` / `ScheduleConfig.lgkm_per_load_*`.
    comptime _geometry = Self._build_geometry()
    comptime VMCNT_PER_LOAD_A = Self._geometry.vm_per_load_a
    """Global-load vmcnt cost per A prefetch (from `KernelGeometry`)."""
    comptime VMCNT_PER_LOAD_B = Self._geometry.vm_per_load_b
    """Global-load vmcnt cost per B prefetch (from `KernelGeometry`)."""

    @staticmethod
    def _build_geometry() -> KernelGeometry:
        """Builds the `KernelGeometry` for this kernel's 4-warp 2x2 layout.

        Hard-codes the 4-wave kernel's structural assumptions:
          - 4 loading warps (= total_warps).
          - 2x2 warp grid in (M, N), so WM = BM/2 and WN = BN/2.
          - QuadrantMmaOp pattern: `(num_m_mmas, num_n_mmas)` split
            into 2x2 quadrants.
          - AMD `buffer_load_lds` issues 16-byte loads regardless of
            dtype, so `simd_width` is `16 / elem_bytes` rather than
            `simd_width_of[in_type]()` (the latter is host-target-
            dependent and gives wrong values when this factory is
            comptime-called from a CPU host).
        """
        comptime in_type = Self.in_type
        comptime BM = Self.BM
        comptime BN = Self.BN
        comptime BK = Self.BK
        comptime MMA_M = Self.MMA_M
        comptime MMA_N = Self.MMA_N
        comptime MMA_K = Self.MMA_K
        comptime elem_bytes = size_of[in_type]()
        comptime simd_width = 16 // elem_bytes
        comptime is_fp8 = in_type.is_float8()
        comptime half_BM = BM // 2
        comptime half_BN = BN // 2
        comptime WM = BM // 2
        comptime WN = BN // 2
        comptime num_m_mmas = WM // MMA_M
        comptime num_n_mmas = WN // MMA_N
        comptime num_k_mmas = BK // MMA_K
        comptime quadrant_m_mmas = num_m_mmas // 2
        comptime quadrant_n_mmas = num_n_mmas // 2

        # vm_per_load: number of distinct buffer_load_lds transactions
        # per prefetch tile, given 4 loading warps.
        comptime loads_per_row = BK // simd_width
        comptime rows_per_iter_4warp = (4 * WARP_SIZE) // loads_per_row
        comptime vm_per_load_a = half_BM // rows_per_iter_4warp
        comptime vm_per_load_b = half_BN // rows_per_iter_4warp

        # lgkm_per_load: ds_reads per frag-load, accounting for FP8's
        # split-LDS path (16-byte fragments) vs BF16's full-fragment
        # path.
        comptime mma_frag_w = (MMA_M * MMA_K) // WARP_SIZE
        comptime use_split_lds = is_fp8 and MMA_M == 16 and MMA_K == 128
        comptime lds_frag_w = 16 if use_split_lds else mma_frag_w
        comptime k_loads_per_mma = mma_frag_w // lds_frag_w
        comptime ds_reads_per_frag = ceildiv(lds_frag_w * elem_bytes, 16)
        comptime lgkm_per_load_a = (
            quadrant_m_mmas * num_k_mmas * k_loads_per_mma * ds_reads_per_frag
        )
        comptime lgkm_per_load_b = (
            quadrant_n_mmas * num_k_mmas * k_loads_per_mma * ds_reads_per_frag
        )

        return KernelGeometry(
            BM=BM,
            BN=BN,
            BK=BK,
            MMA_M=MMA_M,
            MMA_N=MMA_N,
            MMA_K=MMA_K,
            elem_bytes=elem_bytes,
            simd_width=simd_width,
            is_fp8=is_fp8,
            vm_per_load_a=vm_per_load_a,
            vm_per_load_b=vm_per_load_b,
            lgkm_per_load_a=lgkm_per_load_a,
            lgkm_per_load_b=lgkm_per_load_b,
        )

    # Total SMEM the kernel allocates: 2 stages x 2 subtiles for each of
    # A and B, where A subtiles are (BM/2)*BK and B subtiles are
    # (BN/2)*BK in `in_type`. Collapses to 2*(BM+BN)*BK*size_of[in_type].
    comptime SMEM_BYTES = (
        2 * (Self.BM + Self.BN) * Self.BK * size_of[Self.in_type]()
    )
    """SMEM footprint per workgroup (bytes), derived from BM/BN/BK/in_type."""

    # CDNA4 default LDS budget per workgroup is 64 KB without
    # `MAX_DYNAMIC_SHARED_SIZE_BYTES`, which the 4-wave kernel does not
    # request. If you raise this, also wire the FuncAttribute through.
    comptime _SMEM_LIMIT_BYTES = 65536

    @staticmethod
    def is_valid_config() -> Bool:
        """Returns whether the kernel's tile shapes are a viable config.

        Pure predicate — does not raise. Use this from autotune drivers
        and dispatcher fallbacks to filter out impossible (BM, BN, BK,
        dtype) combinations before instantiating the kernel. The full
        per-check set is in `validate_config`; this is its non-throwing
        counterpart.

        Returns:
            True if every structural and resource invariant holds.
        """
        return (
            Self.BM % Self.WM == 0
            and Self.BN % Self.WN == 0
            and Self.BK % Self.MMA_K == 0
            and Self.WM % Self.MMA_M == 0
            and Self.WN % Self.MMA_N == 0
            and Self.total_warps == 4
            and Self.num_warps_m == 2
            and Self.num_warps_n == 2
            and Self.num_m_mmas % 2 == 0
            and Self.num_n_mmas % 2 == 0
            and Self.a_type == Self.b_type
            and 16 % size_of[Self.in_type]() == 0
            and Self.SMEM_BYTES <= Self._SMEM_LIMIT_BYTES
        )

    @staticmethod
    def validate_config():
        """Asserts that the kernel's tile shapes meet 4-wave invariants.

        Throws (via `comptime assert`) on the first failing invariant,
        with a check-specific message. Called from `run` so any kernel
        instantiation that compiles is guaranteed valid. For non-
        throwing tests (autotune sweeps, dispatcher pre-flight), use
        `is_valid_config` instead.
        """
        comptime assert (
            Self.BM % Self.WM == 0
        ), "Block M must be divisible by Warp M"
        comptime assert (
            Self.BN % Self.WN == 0
        ), "Block N must be divisible by Warp N"
        comptime assert (
            Self.BK % Self.MMA_K == 0
        ), "Block K must be divisible by MMA K"
        comptime assert (
            Self.WM % Self.MMA_M == 0
        ), "Warp M must be divisible by MMA M"
        comptime assert (
            Self.WN % Self.MMA_N == 0
        ), "Warp N must be divisible by MMA N"
        comptime assert (
            Self.total_warps == 4
        ), "4-wave-simple kernel requires exactly 4 warps"
        comptime assert (
            Self.num_warps_m == 2
        ), "4-wave-simple requires 2 warps in M dimension"
        comptime assert (
            Self.num_warps_n == 2
        ), "4-wave-simple requires 2 warps in N dimension"
        comptime assert (
            Self.num_m_mmas % 2 == 0
        ), "4-wave-simple needs num_m_mmas even (QuadrantMmaOp constraint)"
        comptime assert (
            Self.num_n_mmas % 2 == 0
        ), "4-wave-simple needs num_n_mmas even (QuadrantMmaOp constraint)"
        comptime assert Self.a_type == Self.b_type, "A/B must match"
        comptime assert 16 % size_of[Self.in_type]() == 0, (
            "in_type element size must divide 16 (AMD buffer_load_lds"
            " transaction)"
        )
        comptime assert Self.SMEM_BYTES <= Self._SMEM_LIMIT_BYTES, (
            "SMEM budget exceeded: 2*(BM+BN)*BK*size_of[in_type] must fit in"
            " 64 KB"
        )

    @__llvm_metadata(`rocdl.waves_per_eu`=SIMDSize(1))
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.config.num_threads())
        )
    )
    @staticmethod
    def run[
        a_layout: TensorLayout,
        b_layout: TensorLayout,
        c_layout: TensorLayout,
        *,
        use_framework_schedule: Bool = False,
        num_splits: Int = 1,
    ](
        a: TileTensor[Self.a_type, a_layout, ImmutAnyOrigin],
        b: TileTensor[Self.b_type, b_layout, ImmutAnyOrigin],
        c: TileTensor[Self.c_type, c_layout, MutAnyOrigin],
    ):
        """Runs the 4-wave GEMM kernel for one workgroup tile.

        Body strategy is selected at comptime via `use_framework_schedule`:
          - False (default): hand-written `_run_iter` body. Explicit waits,
            barriers, and cross-stage register rotation matching HipKittens'
            FP8_4wave reference. Currently the perf champion (~516 TFLOPS at
            FP8 M=256 N=K=4096 on MI355X).
          - True: framework-driven body via `Pipeline4Wave` under
            `SchedulingStrategy.IDENTITY` + `minimal_barriers` +
            `omit_mma_set_prio`. The framework consumes the 24-op
            cross-stage-rotation body verbatim (no CSP/double-buffer
            reorder). ~3% slower than the hand-written body on FP8 BM=128;
            gap is structurally tied to op ordering inside each mini-iter
            (pre-barrier ds_reads vs post-barrier).

        SMEM/loader/MMA/output-store scaffolding is shared between both
        paths; only the prologue/main-loop/epilogue body differs.

        Parameters:
            a_layout: Logical layout of `a`.
            b_layout: Logical layout of `b`.
            c_layout: Logical layout of `c`.
            use_framework_schedule: When True, uses the framework-driven
                schedule body; otherwise emits the hand-written body.
            num_splits: Split-K factor (1 means no split).

        Args:
            a: Input tile-tensor for A.
            b: Input tile-tensor for B.
            c: Output tile-tensor for C (or workspace for split-K).
        """
        Self.validate_config()

        comptime BM = Self.BM
        comptime BN = Self.BN
        comptime BK = Self.BK
        comptime WM = Self.WM
        comptime WN = Self.WN
        comptime MMA_M = Self.MMA_M
        comptime MMA_N = Self.MMA_N
        comptime num_m_mmas = Self.num_m_mmas
        comptime num_n_mmas = Self.num_n_mmas
        comptime c_frag_size = Self.c_frag_size
        comptime simd_width = Self.simd_width
        comptime consumer_swizzle = Self.mma_swizzle
        comptime half_BM = Self.half_BM
        comptime half_BN = Self.half_BN
        comptime mma_tile_m = Self.mma_tile_m
        comptime mma_tile_n = Self.mma_tile_n

        var M = Int(a.dim[0]())
        comptime N = type_of(b).static_shape[0]
        comptime K = type_of(a).static_shape[1]
        comptime K_per_split = K // num_splits
        comptime assert (
            K_per_split * num_splits == K
        ), "num_splits must evenly divide K"
        comptime assert (
            K_per_split % (2 * Self.BK) == 0
        ), "K_per_split must be a multiple of 2*BK"

        # split_id from grid_dim.z; always 0 when num_splits=1
        # (grid_dim.z=1) so reading is harmless.
        var split_id = Int(block_idx.z)

        var _lane_id = lane_id()
        var _warp_id = readfirstlane(warp_id())
        # HK chiplet+L2 swizzle: 1D launch, derive (pid_m, pid_n) from
        # block_idx.x via XCD-stride remap + WGM=4 column-major group.
        # `num_pid_n` must match the launcher's `ceildiv(N, BN)` so that
        # the partial last column block (when N % BN != 0) decodes to its
        # own (pid_m, pid_n) instead of aliasing with another block. With
        # `N // BN` the partial block ID gets mis-mapped to the next
        # row's blocks, and the in-bounds tail of the partial block (cols
        # `(N//BN)*BN .. N-1`) silently goes unwritten.
        var num_pid_m = ceildiv(M, BM)
        comptime num_pid_n_static = ceildiv(N, BN)
        var pid_m_pid_n = _xcd_wgm_swizzle(
            Int(block_idx.x), num_pid_m, num_pid_n_static
        )
        var pid_m = pid_m_pid_n[0]
        var pid_n = pid_m_pid_n[1]
        var m = pid_m * BM
        var n = pid_n * BN
        var warp_id_m, warp_id_n = divmod(_warp_id, Self.num_warps_n)

        # === Unified-dtype GMEM views ===
        var a_gmem = a.bitcast[Self.a_type]()
        var b_gmem = b.bitcast[Self.in_type]()

        # === SMEM: 2 stages x 2 M-subtiles for A, 2 stages x 2 N-subtiles for B
        comptime a_half_layout = row_major[half_BM, BK]()
        var a_s0_g0 = stack_allocation[Self.in_type, AddressSpace.SHARED](
            a_half_layout
        )
        var a_s0_g1 = stack_allocation[Self.in_type, AddressSpace.SHARED](
            a_half_layout
        )
        var a_s1_g0 = stack_allocation[Self.in_type, AddressSpace.SHARED](
            a_half_layout
        )
        var a_s1_g1 = stack_allocation[Self.in_type, AddressSpace.SHARED](
            a_half_layout
        )

        comptime b_half_layout = row_major[half_BN, BK]()
        var b_s0_h0 = stack_allocation[Self.in_type, AddressSpace.SHARED](
            b_half_layout
        )
        var b_s0_h1 = stack_allocation[Self.in_type, AddressSpace.SHARED](
            b_half_layout
        )
        var b_s1_h0 = stack_allocation[Self.in_type, AddressSpace.SHARED](
            b_half_layout
        )
        var b_s1_h1 = stack_allocation[Self.in_type, AddressSpace.SHARED](
            b_half_layout
        )

        # MMA sub-tiles: each warp reads its mma_tile-sized slice from
        # BOTH M-halves (g0 and g1) for A, and BOTH N-halves (h0 and h1)
        # for B. The schedule's `subtile` axis indexes which M-half (for
        # A) or N-half (for B). Within a half, the warp picks its slice
        # via warp_id_m (or warp_id_n).
        var a_mma_s0_0 = a_s0_g0.tile[mma_tile_m, BK](warp_id_m, 0)
        var a_mma_s0_1 = a_s0_g1.tile[mma_tile_m, BK](warp_id_m, 0)
        var a_mma_s1_0 = a_s1_g0.tile[mma_tile_m, BK](warp_id_m, 0)
        var a_mma_s1_1 = a_s1_g1.tile[mma_tile_m, BK](warp_id_m, 0)

        var b_mma_s0_0 = b_s0_h0.tile[mma_tile_n, BK](warp_id_n, 0)
        var b_mma_s0_1 = b_s0_h1.tile[mma_tile_n, BK](warp_id_n, 0)
        var b_mma_s1_0 = b_s1_h0.tile[mma_tile_n, BK](warp_id_n, 0)
        var b_mma_s1_1 = b_s1_h1.tile[mma_tile_n, BK](warp_id_n, 0)

        # === DRAM->LDS loaders ===
        comptime _is_fp8 = Self.a_type.is_float8()
        comptime use_fp8_row_major = _is_fp8
        comptime byte_swizzle = Self.byte_swizzle

        # K-band selected by split_id; for num_splits=1 this is the
        # whole K range (split_id=0, K_per_split=K).
        var a_block_gmem = a_gmem.tile[BM, K_per_split](pid_m, split_id)
        var b_block_gmem = b_gmem.tile[BN, K_per_split](pid_n, split_id)

        var a_loader = TileLoaderLDS[
            Self.in_type,
            half_BM,
            BK,
            stride=type_of(a_gmem).static_stride[0],
            num_loading_warps=4,
            swizzle=byte_swizzle,
            load_width=simd_width,
            use_full_tile_width=use_fp8_row_major,
        ](a_block_gmem, _warp_id, Int(_lane_id))
        var b_loader = TileLoaderLDS[
            Self.in_type,
            half_BN,
            BK,
            stride=type_of(b_gmem).static_stride[0],
            num_loading_warps=4,
            swizzle=byte_swizzle,
            load_width=simd_width,
            use_full_tile_width=use_fp8_row_major,
        ](b_block_gmem, _warp_id, Int(_lane_id))

        # === MMA operator (full warp tile, quadrant access via methods) ===
        var mma_op = QuadrantMmaOp[
            out_type=Self.accum_dtype,
            in_type=Self.in_type,
            shape=Self.config.mma_shape,
            k_group_size=1,
            num_k_groups=Self.num_k_mmas,
            num_m_mmas=num_m_mmas,
            num_n_mmas=num_n_mmas,
            swizzle=consumer_swizzle,
        ]()

        @always_inline
        def s_barrier():
            llvm_intrinsic["llvm.amdgcn.s.barrier", NoneType]()

        @always_inline
        def s_setprio[priority: Int16]():
            llvm_intrinsic["llvm.amdgcn.s.setprio", NoneType](priority)

        # === Load helpers ===
        var a_load_tiles = (
            (a_s0_g0, a_s0_g1),
            (a_s1_g0, a_s1_g1),
        )
        var b_load_tiles = (
            (b_s0_h0, b_s0_h1),
            (b_s1_h0, b_s1_h1),
        )

        @always_inline
        @parameter
        def load_a[stage: Int, which: Int](k: Int):
            a_loader.load_tile(
                a_load_tiles[stage][which],
                src_row=which * half_BM,
                src_col=k,
            )

        @always_inline
        @parameter
        def load_b[stage: Int, which: Int](k: Int):
            b_loader.load_tile(
                b_load_tiles[stage][which],
                src_row=which * half_BN,
                src_col=k,
            )

        # === MMA tile lookup: [stage][subtile] -> SMEM tile ===
        var a_mma_tiles = (
            (a_mma_s0_0, a_mma_s0_1),
            (a_mma_s1_0, a_mma_s1_1),
        )
        var b_mma_tiles = (
            (b_mma_s0_0, b_mma_s0_1),
            (b_mma_s1_0, b_mma_s1_1),
        )

        # ====================================================
        # Body: hand-written prologue + _run_iter + epilogue.
        # Mirrors HipKittens FP8_4wave's `matmul_device_*`.
        # ====================================================
        @parameter
        @always_inline
        def _emit_handwritten_body():
            # vmcnt budget: 8 G-loads in flight per steady iter, each
            # consuming VMCNT_PER_LOAD_A (or _B; equal for square BM=BN).
            # Wait values are parameterized so BM=64/128 share the same
            # code path.
            comptime VMCNT = Self.VMCNT_PER_LOAD_A
            comptime TOTAL_VM = 8 * VMCNT
            comptime PROLOGUE_WAIT_1 = TOTAL_VM - VMCNT  # 7 / 14 / 28
            comptime PROLOGUE_WAIT_2 = TOTAL_VM - 2 * VMCNT  # 6 / 12 / 24
            comptime MAIN_WAIT = 4 * VMCNT  # 4 / 8 / 16
            comptime EPILOGUE_MID_WAIT = 2 * VMCNT  # 2 / 4 / 8

            comptime assert (
                K_per_split % (2 * BK) == 0
            ), "4-wave requires K_per_split divisible by 2*BK"

            # Schedule-barrier hint (LLVM machine scheduler, mask=0 = no
            # movement). BM=64 has too few instructions per mini-iter for
            # the fences to help — LLVM's default scheduler already does
            # well, and rigid barriers regress ~4%. BM=128 (4 MFMAs/quadrant)
            # benefits ~13%.
            @always_inline
            @parameter
            def _sched_barrier():
                comptime if BM >= 128:
                    schedule_barrier()

            @parameter
            @always_inline
            def _run_iter[curr: Int, next: Int](k_iter_elem: Int):
                """One source K-tile iteration (4 mini-iters)."""
                # Top wait
                _sched_barrier()
                s_waitcnt[vmcnt=UInt32(MAIN_WAIT)]()
                s_waitcnt[lgkmcnt=UInt32(0)]()
                s_barrier()
                _sched_barrier()

                # Mini-iter 1: prefetch As[curr][0], frag b[1] from curr,
                # mma c[0][0]
                load_a[curr, 0](k_iter_elem + 2 * BK)
                mma_op.load_b_quadrant[1](b_mma_tiles[curr][1])
                mma_op.mma_quadrant[0, 0]()

                _sched_barrier()
                s_waitcnt[lgkmcnt=UInt32(0)]()
                _sched_barrier()

                # Mini-iter 2: prefetch Bs[curr][0], frag a[1] from curr,
                # mma c[0][1]
                load_b[curr, 0](k_iter_elem + 2 * BK)
                mma_op.load_a_quadrant[1](a_mma_tiles[curr][1])
                mma_op.mma_quadrant[0, 1]()

                # Mid wait
                _sched_barrier()
                s_waitcnt[vmcnt=UInt32(MAIN_WAIT)]()
                s_waitcnt[lgkmcnt=UInt32(0)]()
                s_barrier()
                _sched_barrier()

                # Mini-iter 3: cross-stage frag a[0] from next stage,
                # mma c[1][0]
                load_b[curr, 1](k_iter_elem + 2 * BK)
                mma_op.load_a_quadrant[0](a_mma_tiles[next][0])
                mma_op.mma_quadrant[1, 0]()

                # Mini-iter 4: cross-stage frag b[0] from next stage,
                # mma c[1][1]
                load_a[curr, 1](k_iter_elem + 2 * BK)
                mma_op.load_b_quadrant[0](b_mma_tiles[next][0])
                mma_op.mma_quadrant[1, 1]()

            # === PROLOGUE: load K[0] into stage 0, K[1] into stage 1 ===
            # 8 G-loads total. After both prologue waits, 2 of the 8 have
            # completed (the first 2 stage-0 loads' partial completion via
            # vmcnt structure), leaving 6 outstanding; the first 2
            # frag-loads (a[0], b[0]) come from the now-ready As[0][0] /
            # Bs[0][0].
            _sched_barrier()
            load_a[0, 0](0)
            load_b[0, 0](0)
            load_b[0, 1](0)
            load_a[0, 1](0)

            load_a[1, 0](BK)
            load_b[1, 0](BK)
            load_b[1, 1](BK)
            load_a[1, 1](BK)

            _sched_barrier()
            s_waitcnt[vmcnt=UInt32(PROLOGUE_WAIT_1)]()
            s_barrier()
            _sched_barrier()
            mma_op.load_a_quadrant[0](a_mma_tiles[0][0])

            _sched_barrier()
            s_waitcnt[vmcnt=UInt32(PROLOGUE_WAIT_2)]()
            s_barrier()
            _sched_barrier()
            mma_op.load_b_quadrant[0](b_mma_tiles[0][0])

            # === MAIN LOOP: 2 source iters per framework iter ===
            # Source for-loop runs k = 0..k_iters-3 (k_iters-2 iters
            # total). Two epilogue iters cover k_iters-2 and k_iters-1.
            # Group 2 source iters into one framework iter (curr alternates
            # 0/1 → known at comptime within each pair).
            for k_loop in range(0, K_per_split - 2 * BK, 2 * BK):
                _run_iter[0, 1](k_loop)  # half 0: source iter 2L  (curr=0)
                _run_iter[1, 0](k_loop + BK)  # half 1: 2L+1 (curr=1)

            # === EPILOGUE iter 1 (k_iters-2, curr=0, no further G-loads) ===
            _sched_barrier()
            s_waitcnt[vmcnt=UInt32(MAIN_WAIT)]()
            s_barrier()
            _sched_barrier()

            _sched_barrier()
            s_waitcnt[lgkmcnt=UInt32(0)]()
            _sched_barrier()
            mma_op.load_b_quadrant[1](b_mma_tiles[0][1])
            _sched_barrier()
            mma_op.mma_quadrant[0, 0]()
            _sched_barrier()

            _sched_barrier()
            s_waitcnt[lgkmcnt=UInt32(0)]()
            _sched_barrier()
            mma_op.load_a_quadrant[1](a_mma_tiles[0][1])
            _sched_barrier()
            mma_op.mma_quadrant[0, 1]()
            _sched_barrier()

            _sched_barrier()
            s_waitcnt[vmcnt=UInt32(EPILOGUE_MID_WAIT)]()
            s_barrier()
            _sched_barrier()

            _sched_barrier()
            s_waitcnt[lgkmcnt=UInt32(0)]()
            _sched_barrier()
            mma_op.load_a_quadrant[0](a_mma_tiles[1][0])
            _sched_barrier()
            mma_op.mma_quadrant[1, 0]()
            _sched_barrier()

            mma_op.load_b_quadrant[0](b_mma_tiles[1][0])
            _sched_barrier()
            mma_op.mma_quadrant[1, 1]()
            _sched_barrier()

            # === EPILOGUE iter 2 (k_iters-1, curr=1, drain only) ===
            _sched_barrier()
            s_waitcnt[vmcnt=UInt32(0)]()
            s_barrier()
            _sched_barrier()

            _sched_barrier()
            s_waitcnt[lgkmcnt=UInt32(0)]()
            _sched_barrier()
            mma_op.load_b_quadrant[1](b_mma_tiles[1][1])
            _sched_barrier()
            mma_op.mma_quadrant[0, 0]()
            _sched_barrier()

            _sched_barrier()
            s_waitcnt[lgkmcnt=UInt32(0)]()
            _sched_barrier()
            mma_op.load_a_quadrant[1](a_mma_tiles[1][1])
            _sched_barrier()
            mma_op.mma_quadrant[0, 1]()
            _sched_barrier()

            _sched_barrier()
            s_waitcnt[lgkmcnt=UInt32(0)]()
            _sched_barrier()
            mma_op.mma_quadrant[1, 0]()
            _sched_barrier()
            mma_op.mma_quadrant[1, 1]()
            _sched_barrier()

        # ====================================================
        # Body: framework-driven via Pipeline4Wave.
        # ====================================================
        @parameter
        @always_inline
        def _emit_framework_body():
            # `Pipeline4Wave.__init__` forces IDENTITY +
            # minimal_barriers + omit_mma_set_prio; the ScheduleConfig
            # we pass here only contributes `sched_barrier_mask` and
            # auto-waits opt-in (the rest is overridden by the schedule
            # struct).
            #
            # `sched_barrier_mask` is `0xFF` at BM>=128 (8 fences per
            # loop iter at MMA-block boundaries, matching
            # `_sched_barrier()`'s density) and `0` at BM<128 — fences
            # regress ~4% on BM=64 because the mini-iter is too small
            # to benefit.
            #
            # `wrap_waits_with_sched_barrier` is enabled at BM>=128 so
            # every wait/barrier group inside a block gets wrapped with
            # `schedule_barrier` fences — matching the density of
            # `_sched_barrier()` calls in the hand-tuned body. At
            # BM=64 the mini-iter is too small for the fences to help
            # (LLVM's default scheduler already does well there).
            comptime SCHED_MASK = 0xFF if Self.BM >= 128 else 0
            comptime sched_config = ScheduleConfig(
                scheduling=SchedulingStrategy.IDENTITY,
                auto_waits=True,
                sched_barrier_mask=SCHED_MASK,
                wrap_waits_with_sched_barrier=Self.BM >= 128,
            )
            comptime target = mi355x_target(
                vm_per_load_a=Self.VMCNT_PER_LOAD_A,
                vm_per_load_b=Self.VMCNT_PER_LOAD_B,
                max_globals=0 if (
                    Self.num_k_mmas
                    * Self.quadrant_m_mmas
                    * Self.quadrant_n_mmas
                    < 8
                ) else derive_safe_max_globals(Self.num_k_mmas),
            )
            # `KernelGeometry` bundles `is_fp8` + `lgkm_per_load_*` into
            # one comptime arg so they don't have to be threaded as
            # individual template params.
            comptime schedule = build_schedule[geometry=Self._geometry](
                sched_config, target
            )

            @parameter
            @always_inline
            def _bind[entry: ScheduleEntry](k_base: Int):
                # Framework-level infrastructure tags (BARRIER / WAIT_* /
                # SET_PRIO / SCHEDULE_BARRIER) emit AMD intrinsics
                # directly; kernel-specific data tags (LOAD_A / LOAD_B /
                # MMA_LOAD_* / MMA) call the kernel's own loaders.
                comptime if entry.op.tag == _Ops.BARRIER.value:
                    llvm_intrinsic["llvm.amdgcn.s.barrier", NoneType]()
                elif entry.op.tag == _Ops.WAIT_VM.value:
                    s_waitcnt[vmcnt=UInt32(entry.op.wait_value)]()
                elif entry.op.tag == _Ops.WAIT_LGKM.value:
                    s_waitcnt[lgkmcnt=UInt32(entry.op.wait_value)]()
                elif entry.op.tag == _Ops.SET_PRIO.value:
                    llvm_intrinsic["llvm.amdgcn.s.setprio", NoneType](
                        Int16(entry.op.wait_value)
                    )
                elif entry.op.tag == _Ops.SCHEDULE_BARRIER.value:
                    schedule_barrier()
                else:
                    comptime k_off = entry.op.k_offset.signed_bk_multiple()
                    var k = k_base + k_off * BK
                    comptime if entry.op.tag == LOAD_A:
                        load_a[entry.op.stage, entry.op.subtile](k)
                    elif entry.op.tag == LOAD_B:
                        load_b[entry.op.stage, entry.op.subtile](k)
                    elif entry.op.tag == MMA_LOAD_A:
                        mma_op.load_a_quadrant[entry.op.subtile](
                            a_mma_tiles[entry.op.stage][entry.op.subtile]
                        )
                    elif entry.op.tag == MMA_LOAD_B:
                        mma_op.load_b_quadrant[entry.op.subtile](
                            b_mma_tiles[entry.op.stage][entry.op.subtile]
                        )
                    elif entry.op.tag == MMA:
                        mma_op.mma_quadrant[entry.op.stage, entry.op.subtile]()

            # Prologue. The schedule overrides `warp_stagger=none()`
            # (4-wave has only 4 warps, so warp-group staggering is
            # meaningless), sets `partial_prologue_drain=True`, and
            # supplies `bootstrap_frags()` for the two same-stage
            # sub=0 frag-loads needed for the first main iter. The
            # framework appends them to the prologue paired with
            # `wait_vm(N) + barrier` partial drains computed from
            # cumulative prefetch vm_cost. A single comptime for over
            # the full prologue emits everything correctly.
            comptime for i in range(len(schedule.prologue)):
                _bind[schedule.prologue[i]](0)

            # Main loop: step 2*BK because each schedule iter unrolls 2
            # source K-iters (matches ping-pong's stepping).
            for k in range(BK * 2, K_per_split, BK * 2):
                comptime for i in range(len(schedule.kernel)):
                    _bind[schedule.kernel[i]](k)

            # Epilogue.
            comptime for i in range(len(schedule.epilogue)):
                _bind[schedule.epilogue[i]](K_per_split)

        # ====================================================
        # Dispatch: comptime-select which body to emit.
        # ====================================================
        comptime if use_framework_schedule:
            _emit_framework_body()
        else:
            _emit_handwritten_body()

        # === Output Store (shared) ===
        # 4-wave's warp output is interleaved: each warp covers 2
        # disjoint M-row ranges (one per M-half) × 2 disjoint N-col
        # ranges. Per (m_mma, n_mma), each lane writes a SIMD-of-
        # c_frag_size at a per-lane (m_global, n_global) coordinate
        # in the (BM × BN) block:
        #
        #   m_global = pid_m*BM + m_quad*half_BM
        #            + warp_id_m*mma_tile_m + m_within*MMA_M + thread_m
        #   n_global = pid_n*BN + n_quad*half_BN
        #            + warp_id_n*mma_tile_n + n_within*MMA_N
        #            + lane_group*c_frag_size
        #
        # `RegTileEpilogue` handles the per-lane chunk store and the
        # partial-chunk fallback at the column boundary; the kernel
        # only computes (m, n) and feeds the SIMD chunk. For split-K
        # the matmul kernel writes f32 partials to a stacked workspace
        # at row `split_id * M + pid_m * BM + m_global_base` while the
        # in-bounds gate uses the LOGICAL row `m + m_global_base` — the
        # two coincide for non-split-K. Lambda mode requires the lambda
        # to fire on the FINAL value, so split-K + lambda is rejected
        # at comptime; the reduce kernel applies the lambda there.
        comptime quad_m_mmas = Self.quadrant_m_mmas
        comptime quad_n_mmas = Self.quadrant_n_mmas

        comptime if Bool(Self.elementwise_lambda_fn):
            comptime assert num_splits == 1, (
                "elementwise epilogue is not supported with split-K"
                " (would fire on each partial, not the reduced output)"
            )

        var c_writer = RegTileEpilogue[
            Self.c_type,
            c_frag_size,
            elementwise_lambda_fn=Self.elementwise_lambda_fn,
        ](c)

        if m < M and n < N:
            var c_reg = mma_op.accum_tile()
            var lane_group, thread_m = divmod(Int(_lane_id), MMA_M)

            comptime for m_mma in range(num_m_mmas):
                comptime m_quad, m_within = divmod(m_mma, quad_m_mmas)
                var m_global_base = (
                    m_quad * half_BM
                    + Int(warp_id_m) * mma_tile_m
                    + m_within * MMA_M
                    + thread_m
                )
                var m_dram = split_id * M + pid_m * BM + m_global_base
                var m_logical = m + m_global_base
                if m_logical < M:
                    comptime for n_mma in range(num_n_mmas):
                        comptime n_quad, n_within = divmod(n_mma, quad_n_mmas)
                        var n_global = (
                            n
                            + n_quad * half_BN
                            + Int(warp_id_n) * mma_tile_n
                            + n_within * MMA_N
                            + lane_group * c_frag_size
                        )
                        var v = (
                            c_reg.tile[1, c_frag_size](m_mma, n_mma)
                            .raw_load[width=c_frag_size](0)
                            .cast[Self.c_type]()
                        )
                        c_writer.store(v, m=m_dram, n=n_global)


@always_inline
def amd_4wave_matmul[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    //,
    enable_swizzle: Bool = True,
    block_m_override: Int = 0,
    block_n_override: Int = 0,
    dump_asm_path: StaticString = "",
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    a: TileTensor[mut=False, a_type, ...],
    b: TileTensor[mut=False, b_type, ...],
    c: TileTensor[mut=True, c_type, ...],
    ctx: DeviceContext,
) raises:
    """Launches the hand-written 4-wave matmul on the device.

    Parameters:
        a_type: Element type of `a`.
        b_type: Element type of `b`.
        c_type: Element type of `c`.
        enable_swizzle: Enable LDS bank-conflict avoidance.
        block_m_override: If > 0, force BM to this value (must be 64 or
            128 — the kernel's QuadrantMmaOp + 2x2 warp grid is baked
            into its SMEM/half partitioning, so smaller BM requires a
            kernel rewrite, not just a config knob).
        block_n_override: If > 0, force BN to this value (must be 64,
            128, or 256). Default 0 uses BM=BN.
        dump_asm_path: If non-empty, dumps the compiled GCN assembly to
            the given file path. Only used for ASM-level diff-debugging.
        elementwise_lambda_fn: Optional fused epilogue. When set, the
            kernel's output store calls the lambda with global
            `(m_global, n_global)` coords and a SIMD-of-`c_frag_size`
            instead of writing to `c` directly.

    Args:
        a: Input tile-tensor for A.
        b: Input tile-tensor for B.
        c: Output tile-tensor for C.
        ctx: Device context used to enqueue the kernel.

    Raises:
        An error if device enqueue fails.
    """
    comptime assert a_type == b_type, "A and B must have the same type"
    comptime assert (
        a_type.is_float8()
    ), "4-wave currently only supports float8_e4m3fn"

    var N = Int(c.dim[1]())
    var M = Int(c.dim[0]())

    @always_inline
    @parameter
    def run_kernel[config: KernelConfig]() raises:
        comptime kernel = AMD4WaveMatmul[
            a_type,
            b_type,
            c_type,
            config,
            enable_swizzle,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ].run[a.LayoutType, b.LayoutType, c.LayoutType]

        # 1D launch grid for the HK chiplet/L2 swizzle. The kernel
        # decodes (pid_m, pid_n) from block_idx.x.
        var num_blocks_n = ceildiv(N, config.block_shape[1])
        var num_blocks_m = ceildiv(M, config.block_shape[0])
        comptime if dump_asm_path != "":
            ctx.enqueue_function[kernel, dump_asm=dump_asm_path](
                a,
                b,
                c,
                grid_dim=(num_blocks_n * num_blocks_m,),
                block_dim=config.num_threads(),
            )
        else:
            ctx.enqueue_function[kernel](
                a,
                b,
                c,
                grid_dim=(num_blocks_n * num_blocks_m,),
                block_dim=config.num_threads(),
            )

    # FP8: MMA shape 16x16x128, BK=128. 4 warps in 2x2 grid.
    # WM = BM/2, WN = BN/2 (2x2 grid), so:
    #   BM in {64, 128} (smaller would fail QuadrantMmaOp's even-mma assert)
    #   BN in {64, 128, 256} (smaller would fail same)

    comptime if block_m_override > 0 and block_n_override > 0:
        # Caller-specified config. The kernel's directional asymmetric
        # bug: BN < BM produces wrong results (~33% of elements off);
        # BN >= BM is correct. Until the kernel-level assumption is
        # fixed, we require BN >= BM.
        comptime BM_o = block_m_override
        comptime BN_o = block_n_override
        comptime assert (
            BM_o == 64 or BM_o == 128
        ), "block_m_override must be 64 or 128"
        comptime assert (
            BN_o == 64 or BN_o == 128 or BN_o == 256
        ), "block_n_override must be 64, 128, or 256"
        comptime assert (
            BN_o >= BM_o
        ), "block_n_override must be >= block_m_override (BN < BM is broken)"
        comptime config_override = KernelConfig(
            block_shape=Index(BM_o, BN_o, 128),
            warp_shape=Index(BM_o // 2, BN_o // 2, 128),
            mma_shape=Index(16, 16, 128),
        )
        run_kernel[config_override]()
        return

    # Auto-pick. Heuristic from the symmetric (BN==BM) sweep at
    # N=K=4096 (working doc bench_amd_matmul_results.md):
    #
    #   M     | BM=BN=64 | BM=BN=128
    #   512   | 837      | 728      <- BM=64 wins
    #   1024  | 914      | 1382     <- BM=128 wins
    #
    # Cutoff sits between 512 and 1024.
    #
    #   M ≤ 512 : BM=64,  BN=64
    #   M > 512 : BM=128, BN=128
    comptime config_64 = KernelConfig(
        block_shape=Index(64, 64, 128),
        warp_shape=Index(32, 32, 128),
        mma_shape=Index(16, 16, 128),
    )
    comptime config_128 = KernelConfig(
        block_shape=Index(128, 128, 128),
        warp_shape=Index(64, 64, 128),
        mma_shape=Index(16, 16, 128),
    )

    if M <= 512:
        run_kernel[config_64]()
    else:
        run_kernel[config_128]()


@always_inline
def amd_4wave_scheduled_matmul[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    //,
    enable_swizzle: Bool = True,
    block_m_override: Int = 0,
    block_n_override: Int = 0,
    dump_asm_path: StaticString = "",
](
    a: TileTensor[mut=False, a_type, ...],
    b: TileTensor[mut=False, b_type, ...],
    c: TileTensor[mut=True, c_type, ...],
    ctx: DeviceContext,
) raises:
    """Launches the schedule-compiler-driven 4-wave matmul on the device.

    Identical dispatch to `amd_4wave_matmul` (same auto-pick
    heuristic, same override gates, same chiplet/L2 swizzle, 1D launch
    grid), but invokes `AMD4WaveMatmul.run` with the
    `use_framework_schedule=True` comptime flag. Use this as the
    framework arm of an A/B against the inline arm to attribute perf
    gaps to op ordering vs scaffolding.

    Parameters:
        a_type: Element type of `a`.
        b_type: Element type of `b`.
        c_type: Element type of `c`.
        enable_swizzle: Enable LDS bank-conflict avoidance.
        block_m_override: If > 0, force BM to this value (must be 64 or
            128).
        block_n_override: If > 0, force BN to this value (must be 64,
            128, or 256). Default 0 uses BM=BN.
        dump_asm_path: If non-empty, dumps the compiled GCN assembly to
            the given file path. Only used for ASM-level diff-debugging.

    Args:
        a: Input tile-tensor for A.
        b: Input tile-tensor for B.
        c: Output tile-tensor for C.
        ctx: Device context used to enqueue the kernel.

    Raises:
        An error if device enqueue fails.
    """
    comptime assert a_type == b_type, "A and B must have the same type"
    comptime assert (
        a_type.is_float8()
    ), "4-wave-scheduled currently only supports float8_e4m3fn"

    var N = Int(c.dim[1]())
    var M = Int(c.dim[0]())

    @always_inline
    @parameter
    def run_kernel[config: KernelConfig]() raises:
        comptime kernel = AMD4WaveMatmul[
            a_type,
            b_type,
            c_type,
            config,
            enable_swizzle,
        ].run[
            a.LayoutType,
            b.LayoutType,
            c.LayoutType,
            use_framework_schedule=True,
        ]

        var num_blocks_n = ceildiv(N, config.block_shape[1])
        var num_blocks_m = ceildiv(M, config.block_shape[0])
        comptime if dump_asm_path != "":
            ctx.enqueue_function[kernel, dump_asm=dump_asm_path](
                a,
                b,
                c,
                grid_dim=(num_blocks_n * num_blocks_m,),
                block_dim=config.num_threads(),
            )
        else:
            ctx.enqueue_function[kernel](
                a,
                b,
                c,
                grid_dim=(num_blocks_n * num_blocks_m,),
                block_dim=config.num_threads(),
            )

    comptime if block_m_override > 0 and block_n_override > 0:
        comptime BM_o = block_m_override
        comptime BN_o = block_n_override
        comptime assert (
            BM_o == 64 or BM_o == 128
        ), "block_m_override must be 64 or 128"
        comptime assert (
            BN_o == 64 or BN_o == 128 or BN_o == 256
        ), "block_n_override must be 64, 128, or 256"
        comptime assert (
            BN_o >= BM_o
        ), "block_n_override must be >= block_m_override (BN < BM is broken)"
        comptime config_override = KernelConfig(
            block_shape=Index(BM_o, BN_o, 128),
            warp_shape=Index(BM_o // 2, BN_o // 2, 128),
            mma_shape=Index(16, 16, 128),
        )
        run_kernel[config_override]()
        return

    comptime config_64 = KernelConfig(
        block_shape=Index(64, 64, 128),
        warp_shape=Index(32, 32, 128),
        mma_shape=Index(16, 16, 128),
    )
    comptime config_128 = KernelConfig(
        block_shape=Index(128, 128, 128),
        warp_shape=Index(64, 64, 128),
        mma_shape=Index(16, 16, 128),
    )

    if M <= 512:
        run_kernel[config_64]()
    else:
        run_kernel[config_128]()
