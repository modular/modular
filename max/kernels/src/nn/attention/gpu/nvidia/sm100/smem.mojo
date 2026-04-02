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
"""Shared memory layout for SM100 attention kernels.

Encapsulates the smem offset calculations used by all FA4 warp-specialized
functions (kernel, softmax, correction, load, mma) so that each consumer
derives pointers from a single source of truth instead of duplicating the
arithmetic.

Split-mode memory layout (low to high address):
    [Q: q_nope_bytes + q_rope_bytes]
    [K: num_kv_stages * (padded_ov_depth*BN*qkv_dt + rope_depth*BN*rope_dt)]
    [V: num_kv_stages * padded_ov_depth * BN elements of qkv_dtype]
    [correction: BM elements of Float32]
    [q_scale: BM * scale_dtype (0 when scale_dtype is invalid)]
    [k_scale: num_k_scale_bufs * BN * scale_dtype (0 when invalid)]
    [mbars: FA4MiscMBars.size SharedMemBarriers]
    [tmem_addr: 1 UInt32]

All K stages are contiguous, followed by all V stages contiguous.

Fused-mode memory layout (low to high address):
    [Q: BM * padded_qk_depth elements of qkv_dtype]
    [KV_fused: num_kv_stages * padded_ov_depth * BN elements of qkv_dtype]
    [Rope: ceil(num_kv_stages/2) * BN * rope_depth elements of qkv_dtype]
    [correction: BM elements of Float32]
    [q_scale: BM * scale_dtype (0 when scale_dtype is invalid)]
    [k_scale: num_k_scale_bufs * BN * scale_dtype (0 when invalid)]
    [mbars: FA4MiscMBars.size SharedMemBarriers]
    [tmem_addr: 1 UInt32]

In fused mode, K_nope and V alternate in the same buffer (padded_ov_depth
wide), and rope data is stored separately at half the staging rate.
k_smem_base() and v_smem_base() return the same pointer.
"""

from std.sys import size_of
from std.gpu.memory import AddressSpace, external_memory
from layout.tma_async import SharedMemBarrier
from nn.attention.gpu.nvidia.sm100.attention import (
    FA4Config,
    EnableForcedOrdering,
)
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    SharedMemPointer,
    FA4MiscMBars,
)


struct SM100AttentionSMem[
    qkv_dtype: DType,
    rope_dtype: DType,
    scale_dtype: DType,
    //,
    config: FA4Config[
        qkv_dtype, rope_dtype=rope_dtype, scale_dtype=scale_dtype
    ],
    *,
    use_order_barriers: Bool = EnableForcedOrdering,
](TrivialRegisterPassable):
    """Shared memory layout manager for SM100 Flash Attention kernels.

    Stores a base pointer into dynamic shared memory and provides accessor
    methods for each region (Q, K, V, correction, mbarriers, tmem address).
    All byte-offset arithmetic is comptime so the accessors compile down to a
    single pointer add + bitcast.

    Parameters:
        qkv_dtype: Element type of Q/K/V data in shared memory.
        rope_dtype: Element type of Q and K rope.
        scale_dtype: Element type of the per-token scale used for Q and K.
        config: FA4 configuration (tile sizes, depths, staging counts, etc.).
        use_order_barriers: Whether forced-ordering barriers are allocated.
    """

    # ---- comptime byte offsets ------------------------------------------------
    # Every offset is relative to the beginning of dynamic shared memory.

    comptime _qkv_dt_size: Int = size_of[Self.qkv_dtype]()

    comptime rope_dt_size: Int = (
        size_of[Self.rope_dtype]() if Self.rope_dtype != DType.invalid else 0
    )
    comptime q_byte_offset: Int = 0
    comptime q_nope_bytes: Int = (
        Self.config.BM * Self.config.padded_ov_depth * Self._qkv_dt_size
    )
    comptime q_rope_byte_offset: Int = Self.q_nope_bytes
    comptime q_rope_bytes: Int = (
        Self.config.BM * Self.config.rope_depth() * Self.rope_dt_size
    )
    comptime q_bytes: Int = Self.q_nope_bytes + Self.q_rope_bytes

    # KV region.
    # Split mode: [K_stage0]...[K_stageN][V_stage0]...[V_stageN]
    # Fused mode: [KV_fused_stage0]...[KV_fused_stageN][Rope0]...[RopeM]
    comptime kv_byte_offset: Int = Self.q_bytes

    # Per-stage sizes in bytes.
    # In fused mode K and V share the same padded_ov_depth-wide buffer.
    # In split mode K_nope and K_rope may have different dtypes.
    comptime k_stage_bytes: Int = (
        Self.config.padded_ov_depth
        * Self.config.BN
        * Self._qkv_dt_size if Self.config.use_fused_kv else (
            Self.config.padded_ov_depth * Self.config.BN * Self._qkv_dt_size
            + Self.rope_depth * Self.config.BN * Self.rope_dt_size
        )
    )
    comptime v_stage_bytes: Int = (
        Self.config.padded_ov_depth * Self.config.BN * Self._qkv_dt_size
    )

    # Total K bytes across all stages.
    comptime k_total_bytes: Int = (
        Self.config.num_kv_stages * Self.k_stage_bytes
    )

    # V region starts after all K stages.
    # In fused mode, V shares the same buffer as K (same offset).
    comptime v_byte_offset: Int = (
        Self.kv_byte_offset if Self.config.use_fused_kv else Self.kv_byte_offset
        + Self.k_total_bytes
    )

    # Rope region (fused mode only): ceildiv(num_kv_stages, 2) buffers
    # of BN * rope_depth elements, placed after the fused KV stages.
    comptime rope_depth: Int = Self.config.rope_depth()
    comptime rope_stage_elems: Int = Self.config.BN * Self.rope_depth
    comptime num_rope_bufs: Int = Self.config.num_rope_buffers()
    comptime rope_byte_offset: Int = Self.kv_byte_offset + Self.k_total_bytes
    comptime rope_bytes: Int = (
        Self.num_rope_bufs * Self.rope_stage_elems * Self.rope_dt_size
    )

    # Total KV bytes (including rope in fused mode).
    # Split: num_kv_stages * (k_stage_bytes + v_stage_bytes)
    # Fused: num_kv_stages * padded_ov_depth * BN * qkv_dt + rope_bytes
    comptime kv_stages_bytes: Int = (
        Self.config.num_kv_stages * (Self.k_stage_bytes + Self.v_stage_bytes)
    )
    comptime kv_bytes: Int = (
        Self.k_total_bytes
        + Self.rope_bytes if Self.config.use_fused_kv else Self.kv_stages_bytes
    )

    # Correction region: BM elements of Float32.
    comptime correction_byte_offset: Int = Self.kv_byte_offset + Self.kv_bytes
    comptime correction_bytes: Int = Self.config.BM * size_of[DType.float32]()

    # Scale regions (per-token scale only; zero-sized when scale_dtype is invalid).
    comptime _scale_dt_size: Int = (
        size_of[Self.scale_dtype]() if Self.scale_dtype != DType.invalid else 0
    )
    comptime q_scale_bytes: Int = Self.config.BM * Self._scale_dt_size
    comptime q_scale_byte_offset: Int = (
        Self.correction_byte_offset + Self.correction_bytes
    )

    comptime num_k_scale_bufs: Int = Self.config.num_k_scale_bufs()
    comptime k_scale_stride_bytes: Int = Self.config.BN * Self._scale_dt_size
    comptime k_scale_bytes: Int = (
        Self.num_k_scale_bufs * Self.k_scale_stride_bytes
    )
    comptime k_scale_byte_offset: Int = (
        Self.q_scale_byte_offset + Self.q_scale_bytes
    )

    # Mbarrier region.
    comptime mbar_byte_offset: Int = (
        Self.k_scale_byte_offset + Self.k_scale_bytes
    )

    comptime MiscMBarsType = FA4MiscMBars[
        num_qk_stages=Self.config.num_qk_stages,
        num_pv_stages=Self.config.num_pv_stages,
        num_kv_stages=Self.config.num_kv_stages,
        use_order_barriers=Self.use_order_barriers,
        use_fused_kv=Self.config.use_fused_kv,
    ]

    comptime mbar_bytes: Int = Int(Self.MiscMBarsType.num_mbars()) * size_of[
        SharedMemBarrier
    ]()

    # tmem_addr: 1 UInt32, immediately after the barriers.
    comptime tmem_addr_byte_offset: Int = (
        Self.mbar_byte_offset + Self.mbar_bytes
    )

    # ---- element-count offsets (for compatibility with existing callers) ------

    # Q offset in elements of qkv_dtype.
    comptime q_offset: Int32 = 0

    # KV offset in elements of qkv_dtype.
    comptime kv_offset: Int32 = Int32(Self.kv_byte_offset // Self._qkv_dt_size)

    # Correction offset in elements of Float32 (derived from byte offset).
    comptime correction_offset: Int32 = Int32(
        Self.correction_byte_offset // size_of[DType.float32]()
    )

    # Mbarrier offset in elements of SharedMemBarrier.
    comptime mbar_offset: Int32 = Int32(
        Self.mbar_byte_offset // size_of[SharedMemBarrier]()
    )

    # ---- storage -------------------------------------------------------------
    var base: SharedMemPointer[Scalar[DType.uint8]]

    # ---- construction --------------------------------------------------------

    @always_inline
    def __init__(out self):
        """Obtain the base pointer from the kernel's dynamic shared memory."""

        comptime assert Self.rope_dtype != DType.invalid or Self.rope_depth == 0
        self.base = external_memory[
            Scalar[DType.uint8],
            address_space=AddressSpace.SHARED,
            alignment=128,
            name="mha_dynamic_shared_memory",
        ]()

    # ---- accessors -----------------------------------------------------------

    @always_inline
    def misc_mbars(self) -> Self.MiscMBarsType:
        """Return the FA4MiscMBars wrapper over the mbarrier region."""
        return Self.MiscMBarsType(
            (self.base + Self.mbar_byte_offset).bitcast[SharedMemBarrier]()
        )

    @always_inline
    def q_smem(self) -> SharedMemPointer[Scalar[Self.qkv_dtype]]:
        """Base of the Q region (offset 0)."""
        return (self.base + Self.q_byte_offset).bitcast[
            Scalar[Self.qkv_dtype]
        ]()

    @always_inline
    def q_rope_smem(self) -> SharedMemPointer[Scalar[Self.rope_dtype]]:
        """Base of the Q rope region (after Q nope in smem)."""
        return (self.base + Self.q_rope_byte_offset).bitcast[
            Scalar[Self.rope_dtype]
        ]()

    @always_inline
    def o_smem[
        output_type: DType
    ](self) -> SharedMemPointer[Scalar[output_type]]:
        """Same physical memory as Q, bitcast to the output element type."""
        return (self.base + Self.q_byte_offset).bitcast[Scalar[output_type]]()

    @always_inline
    def k_smem_base(self) -> SharedMemPointer[Scalar[Self.qkv_dtype]]:
        """Base of the K region (first stage, offset = kv_byte_offset)."""
        return (self.base + Self.kv_byte_offset).bitcast[
            Scalar[Self.qkv_dtype]
        ]()

    @always_inline
    def v_smem_base(self) -> SharedMemPointer[Scalar[Self.qkv_dtype]]:
        """Base of the V region (stage 0).

        Split mode: V stage 0 starts after all K stages at
        kv_byte_offset + num_kv_stages * padded_qk_depth * BN * sizeof.
        Fused mode: Returns the same pointer as k_smem_base() since
        K_nope and V share the same buffer.
        """
        return (self.base + Self.v_byte_offset).bitcast[
            Scalar[Self.qkv_dtype]
        ]()

    @always_inline
    def rope_smem_base(self) -> SharedMemPointer[Scalar[Self.rope_dtype]]:
        """Base of the rope region (fused mode only)."""
        return (self.base + Self.rope_byte_offset).bitcast[
            Scalar[Self.rope_dtype]
        ]()

    @always_inline
    def correction_smem(self) -> SharedMemPointer[Float32]:
        """Base of the correction region (BM Float32 elements)."""
        return (self.base + Self.correction_byte_offset).bitcast[Float32]()

    @always_inline
    def q_scale_smem(self) -> SharedMemPointer[Scalar[Self.scale_dtype]]:
        """Base of the q_scale region (BM elements)."""
        return (self.base + Self.q_scale_byte_offset).bitcast[
            Scalar[Self.scale_dtype]
        ]()

    @always_inline
    def k_scale_smem(self) -> SharedMemPointer[Scalar[Self.scale_dtype]]:
        """Base of the k_scale region (num_k_scale_bufs * BN elements)."""
        return (self.base + Self.k_scale_byte_offset).bitcast[
            Scalar[Self.scale_dtype]
        ]()

    @always_inline
    def tmem_addr_ptr(self) -> SharedMemPointer[UInt32]:
        """Pointer to the single UInt32 storing the TMEM address."""
        return (self.base + Self.tmem_addr_byte_offset).bitcast[UInt32]()

    @staticmethod
    @always_inline
    def smem_size() -> Int:
        """Total dynamic shared memory bytes required."""
        return Self.tmem_addr_byte_offset + size_of[UInt32]()
