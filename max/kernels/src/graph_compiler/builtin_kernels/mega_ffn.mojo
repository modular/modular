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
"""Graph-op binding for the fused MegaFFN MoE FFN composite op (NVFP4 + MXFP8).

Target: NVIDIA SM100 (B200). Registers `mo.composite.mega_ffn_nvfp4` and
DTYPE-BRANCHES on the operand element type: packed `uint8` activations ->
`mega_ffn_nvfp4_dispatch` (NVFP4), `float8_e4m3fn` activations ->
`mega_ffn_mxfp8_dispatch` (MXFP8).  Either way it is the single-launch kernel
that fuses the MoE gate/up grouped matmul + SwiGLU + re-quant (L1) and the down
projection (L2) into one launch (one phase pipelined across SMs).  The graph
compiler emits this composite when the MegaFFN fusion fires (SM100,
`num_experts <= 64` per device, single-use intermediate, shared routing, bf16
out); the fusion + composite op are dtype-agnostic, so the same op carries
NVFP4 (uint8) or MXFP8 (e4m3) data and this binding selects the dispatch.

Clamped-SwiGLU (`swigluoai`) status: the activation SELECTOR
`clamp_activation` is plumbed end-to-end as a `Bool` op attribute (the
emitter sets it, the MegaFFN fusion copies it from the L1 leg, and MOGG
binds it to the comptime `clamp_activation` param here).  When the
selector is set, this binding supplies `swigluoai`'s canonical clamp
constants (alpha=1.702, limit=7.0) to `mega_ffn_{nvfp4,mxfp8}_dispatch`,
so the standard `swigluoai` activation is numerically CORRECT through the
fusion; `clamp_activation=False` (plain SwiGLU, the default) is also
correct.  The alpha/limit VALUES are supplied here rather than carried on
the op because MOGG cannot bind an f32 kernel arg from an op attribute
(comptime params forward only Integer/Bool/String attributes; runtime
kernel args bind only from operands).  A model whose clamp uses
NON-standard alpha/limit would need those values carried as host f32
scalar-tensor OPERANDS on the composite ops (mirroring the
`masked_flash_attention` `scale` operand) -- a `.td` operand addition
tracked as a follow-up.

Per-expert scaling is applied PER LEG: the composite op exposes two
scale arrays -- `gate_up_expert_scales` (L1) and `down_expert_scales`
(L2) -- and both are forwarded to the fused kernel, which applies the L1
scale in the SwiGLU store and the L2 scale in the final-output store
(`grouped_1d1d_matmul_kernel.execute_epilogue`).  The kernel's scheduler
resolves each tile's single per-slot scale to the leg-matching array by
tile phase before publishing, so an MoE that emits distinct L1 and L2
expert scales is reproduced exactly (matching the two-launch chain).
See KERN-3085.

Modeled on the sibling `msa.mojo` / `linalg.mojo` registrations (private
`//Kernels` kernels registered in dedicated builtin_kernels files). MegaFFN is
internal-only: unlike `msa` / `matmul_rs` it is NOT shipped in the OSS wheel, so
open-source builds drop the `//Kernels/lib/mega_ffn` dep (in `api.bzl`) and
exclude this file from the public export (copybara).

On-chip scratch (the composite op does not expose these): `c_packed` and
`c_swiglu_scales` are allocated per call via `ctx.enqueue_create_buffer`
-- the capture-safe MOGG workspace pattern used throughout
`builtin_kernels` (`msa.mojo`, `attention.mojo`, `ep.mojo`,
`linalg.mojo`).  `DeviceBuffer` frees are stream-ordered, so the buffers
outlive the launch that uses them.  They are write-then-read within the
launch, so they need no initialization.

`arrival_count` (the cross-CTA pool-slot counters) is instead a
PERSISTENT graph buffer OPERAND: the MegaFFN fusion mints it once
(`mo.buffer.create`) and zeroes it once at setup.  Under the dispatch
default `POST_SELF_CLEAN_UP` each launch claims one 16-bit half of every
slot as its arrival count -- chosen by a generation parity the kernel
keeps in the buffer itself -- and clears the half its predecessor used.
So the buffer is NOT all-zero between launches: it carries the last
launch's counts in that launch's half, and a reader has to know the
generation to interpret it.  What the caller needs is unchanged: no
per-launch memset (this replaces the per-launch allocate + memset this
binding used to do, which sat on the launch-bound decode critical path),
and one zeroing at setup.  Slot 0's padding carries the generation, the
CTA entry tally and a high-water mark over every launch's pool count; a
comptime assert keeps those words inside that padding rather than
aliasing a pool slot.  The fusion sizes the buffer to a static upper
bound on `total_m_blocks`.  A launch touches its own pools and clears
the stale half across the high-water span, which can exceed its own
pool count but not that bound, so over-allocation is safe.
"""

import extensibility as compiler

from comm.sync import is_p2p_enabled
from max.gpu.compute.arch.mma_nvidia_sm100 import UMMAKind
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.host.info import B200, is_gpu
from max.gpu.primitives.grid_controls import PDLLevel
from std.collections import Array
from std.ffi import _get_global_or_null
from std.math import align_up, ceildiv
from std.memory import UnsafePointer
from std.memory.alloc import Layout as AllocLayout
from std.sys import size_of
from std.utils.index import Index
from std.utils.static_tuple import StaticTuple

from layout import Coord, Idx, TileTensor, row_major

from extensibility import InputTensor, OutputTensor
from extensibility import _MutableInputTensor as MutableInputTensor

from linalg.fp4_utils import (
    MXFP8_SF_DTYPE,
    MXFP8_SF_VECTOR_SIZE,
    NVFP4_SF_DTYPE,
    NVFP4_SF_VECTOR_SIZE,
    SF_ATOM_K,
    SF_ATOM_M,
    SF_MN_GROUP_SIZE,
)
from linalg.matmul.gpu.sm100_structured.grouped_block_scaled_1d1d.grouped_1d1d_matmul_kernel import (
    RealSwiGLUOutput,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    BlockScaledMatmulConfig,
    GEMMKind,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.output_writer import (
    P3_MAX_RANKS,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.row_scales import (
    RealRowScales,
)

from shmem import shmem_my_pe
from shmem.ep import global_cache_insert, pack_ptrs_array
from shmem.ep_comm import (
    EPLocalSyncCounters,
    NVBlockScaledTokenFormat,
)

from mega_ffn.mega_ffn_ep_workspace import (
    MegaMoECounters,
    fused_ep_token_block,
    fused_ep_unsupported_reason,
)
from mega_ffn.mega_ffn_kernel import MODE_MEGAFFN
from mega_ffn.mega_ffn_scheduler import (
    ATOMIC_PAD,
    POST_SELF_CLEAN_UP,
    fused_l2_pool_slots,
)
from mega_ffn.mega_ffn_matmul import (
    EPCombineSendOperands,
    FusedEPBankGeometry,
    mega_ffn_block_scaled_ep_fused,
    mega_ffn_mxfp8_dispatch,
    mega_ffn_nvfp4_dispatch,
)


@compiler.register("mo.composite.mega_ffn_nvfp4")
struct Struct_mega_ffn_nvfp4:
    """MOGG wrapper for the fused single-launch MegaFFN NVFP4 MoE FFN.

    Lowers the `mo.composite.mega_ffn_nvfp4` composite op (gate/up GMM +
    SwiGLU + NVFP4 re-quant fused with the down GMM, all in one launch)
    to `mega_ffn_nvfp4_dispatch` on SM100 GPUs.  The clamped-SwiGLU
    activation selector `clamp_activation` is bound from the same-named
    `Bool` op attribute; the alpha/limit values are a follow-up (see the
    module docstring).
    """

    @inline(.always)
    @staticmethod
    def execute[
        c_type: DType,
        a_type: DType,
        b_type: DType,
        scales_type: DType,
        row_scales_type: DType,
        //,
        clamp_activation: Bool,
        has_gate_up_a_row_scales: Bool,
        target: StaticString,
    ](
        output: OutputTensor[dtype=c_type, rank=2, ...],
        hidden_states: InputTensor[dtype=a_type, rank=2, ...],
        gate_up_weight: InputTensor[dtype=b_type, rank=3, ...],
        gate_up_a_scales: InputTensor[dtype=scales_type, rank=5, ...],
        gate_up_b_scales: InputTensor[dtype=scales_type, rank=6, ...],
        down_weight: InputTensor[dtype=b_type, rank=3, ...],
        down_b_scales: InputTensor[dtype=scales_type, rank=6, ...],
        expert_start_indices: InputTensor[dtype=.uint32, rank=1, ...],
        expert_ids: InputTensor[dtype=.int32, rank=1, ...],
        a_scale_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        gate_up_expert_scales: InputTensor[dtype=.float32, rank=1, ...],
        gate_up_a_row_scales: InputTensor[dtype=row_scales_type, rank=1, ...],
        down_expert_scales: InputTensor[dtype=.float32, rank=1, ...],
        c_input_scales: InputTensor[dtype=.float32, rank=1, ...],
        estimated_total_m: UInt32,
        gate_up_num_active_experts: UInt32,
        down_num_active_experts: UInt32,
        arrival_count: MutableInputTensor[dtype=.uint32, rank=1, ...],
        context: DeviceContext,
    ) raises:
        """Executes the fused single-launch MegaFFN NVFP4 MoE FFN.

        Computes `out = down((swiglu(gate_up(hidden_states))) re-quantized
        to NVFP4)` for the active MoE experts in one kernel launch.  `a`,
        both weights, and the down intermediate are NVFP4 (4-bit packed as
        uint8); scales are `float8_e4m3fn` in tcgen05 layout.

        The clamped-SwiGLU selector `clamp_activation` is forwarded to the
        dispatch (its alpha/limit values are a follow-up; see the module
        docstring).  Both per-leg scale arrays are forwarded:
        `gate_up_expert_scales` drives the L1 SwiGLU store and
        `down_expert_scales` drives the L2 final-output store.  The
        on-chip `c_packed` / `c_swiglu_scales` scratch is allocated per call
        (capture-safe `enqueue_create_buffer`); the `arrival_count` pool-slot
        counters are a persistent graph buffer operand (zeroed once at setup;
        thereafter `POST_SELF_CLEAN_UP` rotates them per launch rather than
        returning them to zero).

        Parameters:
            c_type: The output tensor data type (`bfloat16`).
            a_type: The input A / activation element type. `uint8` (packed
                NVFP4) selects the NVFP4 dispatch; `float8_e4m3fn` selects MXFP8.
            b_type: The weight element type (matches `a_type`: `uint8` for
                NVFP4 or `float8_e4m3fn` for MXFP8).
            scales_type: The block scale-factor dtype (`float8_e4m3fn` for
                NVFP4, `float8_e8m0fnu` for MXFP8).
            row_scales_type: The L1 per-row input scale dtype (`bfloat16`
                when `has_gate_up_a_row_scales`).
            clamp_activation: Activation flavor for the fused L1 SwiGLU
                epilogue. `False` = plain SwiGLU; `True` = clamped
                (`swigluoai`). Bound from the op's `clamp_activation`
                attribute.
            has_gate_up_a_row_scales: Whether `gate_up_a_row_scales` holds
                per-row input scales for the L1 leg. Constraints: NVFP4
                only.
            target: The target GPU device.

        Args:
            output: Final output `(M_total, N2)` bf16.
            hidden_states: Token activations `(M_total, K1 // 2)` uint8
                (packed NVFP4).
            gate_up_weight: Pre-permuted gate/up weights
                `(E, N1, K1 // 2)` uint8; `N1 == 2 * moe_dim`.
            gate_up_a_scales: Token (A) 5D E4M3 scale tile for L1; its
                leading dim is the scale-block count reused by the SwiGLU
                intermediate scratch.
            gate_up_b_scales: Pre-permuted W13 6D E4M3 scale tile.
            down_weight: Down-projection weights `(E, N2, moe_dim // 2)`
                uint8 (packed NVFP4).
            down_b_scales: W2 6D E4M3 scale tile.
            expert_start_indices: Per-expert prefix-sum token offsets
                `(E + 1,)` uint32.
            expert_ids: Active expert IDs `(E,)` int32 (`-1` = masked).
            a_scale_offsets: Per-expert 128-row scale-block offsets
                `(E,)` uint32.
            gate_up_expert_scales: L1 (gate+up) per-expert scaling
                `(E,)` f32. Applied in the fused SwiGLU store.
            gate_up_a_row_scales: L1 per-row input scales `(M_total,)`,
                applied with `gate_up_expert_scales` before the SwiGLU.
                Unread unless `has_gate_up_a_row_scales`.
            down_expert_scales: L2 / final-output per-expert scaling
                `(E,)` f32. Applied in the down store. May differ from
                `gate_up_expert_scales`.
            c_input_scales: L1 SwiGLU per-expert input scale (`tensor_sf`)
                `(E,)` f32, used for the NVFP4 re-quant of the
                intermediate.
            estimated_total_m: Estimated total non-padded token count (the
                avg_m gate numerator).
            gate_up_num_active_experts: Active expert slots for L1.
            down_num_active_experts: Active expert slots for L2 (must equal
                `gate_up_num_active_experts`; the kernel walks one shared
                expert list).
            arrival_count: Persistent cross-CTA pool-slot counters (`uint32`,
                rank 1), minted + zeroed once by the MegaFFN fusion. Under
                `POST_SELF_CLEAN_UP` a launch accumulates into the slot half
                its generation parity selects and clears the half the launch
                before it used, so this buffer holds the last launch's counts
                rather than zeros between launches -- do not assume an
                all-zero buffer or reset it per launch, which would pin the
                generation. Sized to a static upper bound on `total_m_blocks`,
                which also bounds the high-water span a launch clears.
            context: The device context.
        """
        comptime assert is_gpu[
            target
        ](), "fused MegaFFN NVFP4 only supports GPUs"

        # L1 and L2 share one expert walk; the two host active-expert counts
        # agree, so honor L1 and discard the L2 copy (satisfies `-Werror`).
        var num_active = Int(gate_up_num_active_experts)
        _ = down_num_active_experts
        if num_active == 0:
            return

        # Two per-expert scale arrays, one per leg: `gate_up_expert_scales`
        # drives the L1 (gate+up) SwiGLU+quant epilogue and
        # `down_expert_scales` drives the L2 (down) store. In MoE these
        # legitimately differ (different weights / input scales), which is
        # why the fused kernel takes both (the scheduler picks the
        # phase-matching array per tile).

        # Element-format branch. The composite op is dtype-agnostic (all
        # MO_Tensor) and the SAME fusion produces it for both formats; the
        # operand element type selects the dispatch + on-chip scratch geometry.
        # NVFP4: `a`/weights packed uint8 (2 elems/byte), 16-elem scale blocks,
        # a separate `c_input_scales` (`tensor_sf`) re-quant tensor. MXFP8:
        # `a`/weights `float8_e4m3fn` (1 elem/byte), 32-elem scale blocks, no
        # `tensor_sf`. The intermediate SF dtype follows the A-scale operand
        # (`scales_type`: E4M3 for NVFP4, E8M0 for MXFP8).
        comptime is_mxfp8 = a_type == DType.float8_e4m3fn
        comptime sf_vector_size = MXFP8_SF_VECTOR_SIZE if is_mxfp8 else NVFP4_SF_VECTOR_SIZE

        # Total expert count is the kernel comptime `num_experts` (the
        # weights' expert dim). Both weights carry it on axis 0.
        comptime num_experts = Int(gate_up_weight.static_spec.shape_tuple[0])
        comptime down_experts = Int(down_weight.static_spec.shape_tuple[0])
        comptime assert (
            num_experts == down_experts
        ), "gate_up and down weights must have the same expert count"

        # MoE intermediate width (SwiGLU output): N1 = 2 * moe_dim, so
        # moe_dim = gate_up_weight.shape[1] // 2 (the N axis is never packed).
        comptime moe_dim = Int(gate_up_weight.static_spec.shape_tuple[1]) // 2
        # c_packed / down-K storage width: down_weight.shape[2] is moe_dim // 2
        # for NVFP4 (2 elems/byte) and moe_dim for MXFP8 (1 elem/byte); c_packed
        # has exactly this many columns on either path.
        comptime packed_K2 = Int(down_weight.static_spec.shape_tuple[2])

        comptime if is_mxfp8:
            comptime assert (
                packed_K2 == moe_dim
            ), "down_weight K dim must equal moe_dim (MXFP8, 1 elem/byte)"
        else:
            comptime assert (
                packed_K2 == moe_dim // 2
            ), "down_weight K dim must equal moe_dim // 2 (packed NVFP4)"

        # SwiGLU intermediate scale tile k-group count, over moe_dim.
        comptime k_groups_swiglu = ceildiv(moe_dim, sf_vector_size * SF_ATOM_K)

        # Total non-padded tokens; `c_packed` rows key off this runtime dim.
        # (The `arrival_count` buffer is sized in the fusion pattern off a
        # static M upper bound, not here.)
        var m_total = Int(hidden_states.dim_size[0]())

        # The intermediate scale tile shares its leading (scale-block)
        # dim with the L1 A-scale tile, so read it exactly off the
        # `gate_up_a_scales` operand rather than re-deriving from device
        # token counts.
        var a_scale_dim0 = Int(gate_up_a_scales.dim_size[0]())

        # ---- On-chip scratch (capture-safe per-call allocation). ----
        # Packed intermediate `(M_total, packed_K2)`: NVFP4 -> uint8, MXFP8 ->
        # float8_e4m3fn. Write-then-read within the launch, no init needed.
        comptime CPackedType = DType.float8_e4m3fn if is_mxfp8 else DType.uint8
        var c_packed_buf = context.enqueue_create_buffer[CPackedType](
            m_total * packed_K2
        )
        var c_packed = TileTensor(
            c_packed_buf.unsafe_ptr(),
            row_major(Int64(m_total), Idx[packed_K2]),
        )

        # Intermediate 5D SwiGLU scale tile, same SF dtype as the A-scale
        # operand (`scales_type`: E4M3 for NVFP4, E8M0 for MXFP8). Shape
        # `(a_scale_dim0, k_groups_swiglu, SF_ATOM_M[0], SF_ATOM_M[1],
        # SF_ATOM_K)`; write-then-read within the launch, no init needed.
        var s_size = (
            a_scale_dim0
            * k_groups_swiglu
            * SF_ATOM_M[0]
            * SF_ATOM_M[1]
            * SF_ATOM_K
        )
        var c_swiglu_scales_buf = context.enqueue_create_buffer[scales_type](
            s_size
        )
        var c_swiglu_scales = TileTensor(
            c_swiglu_scales_buf.unsafe_ptr(),
            row_major(
                Int64(a_scale_dim0),
                Idx[k_groups_swiglu],
                Idx[SF_ATOM_M[0]],
                Idx[SF_ATOM_M[1]],
                Idx[SF_ATOM_K],
            ),
        )

        # Cross-CTA pool-slot counters (strided by ATOMIC_PAD) come in as the
        # PERSISTENT `arrival_count` buffer operand: the MegaFFN fusion mints it
        # once (`mo.buffer.create`) and zeroes it once at setup. Under the
        # `POST_SELF_CLEAN_UP` default each launch claims one half of every slot
        # and clears the half its predecessor used, so this binding must not
        # memset it per launch -- that pins the generation and disarms the
        # rotation. This replaces the per-launch allocate + memset this binding
        # used to do (a launch-bound decode cost). The fusion sizes the buffer
        # to a static upper bound on `total_m_blocks`, which bounds both this
        # launch's pools and the high-water span it clears, so over-allocation
        # is safe.
        var arrival_count_ptr = arrival_count.unsafe_ptr()

        # The clamped-SwiGLU (`swigluoai`) runtime alpha/limit cannot ride as op
        # attributes (MOGG binds f32 kernel args only from operands), so for the
        # standard `swigluoai` activation this binding supplies its canonical
        # constants directly when the `clamp_activation` selector is set. A model
        # whose clamp uses non-standard alpha/limit would need host f32 scalar
        # operands on the composite op (a follow-up; see the module docstring).
        comptime swiglu_alpha = Float32(1.702) if clamp_activation else Float32(
            0.0
        )
        comptime swiglu_limit = Float32(7.0) if clamp_activation else Float32(
            0.0
        )

        comptime if is_mxfp8:
            comptime assert (
                not has_gate_up_a_row_scales
            ), "per-row input scales are only supported for NVFP4"
            # MXFP8 carries no `tensor_sf`; the `c_input_scales` op operand is
            # unused on this path (consume for `-Werror`).
            _ = c_input_scales
            _ = gate_up_a_row_scales
            mega_ffn_mxfp8_dispatch[
                num_experts=num_experts,
                transpose_b=True,
                clamp_activation=clamp_activation,
            ](
                output.to_tile_tensor[.int64](),
                c_packed,
                c_swiglu_scales,
                hidden_states.to_tile_tensor[.int64](),
                gate_up_weight.to_tile_tensor[.int64](),
                down_weight.to_tile_tensor[.int64](),
                gate_up_a_scales.to_tile_tensor[.int64](),
                gate_up_b_scales.to_tile_tensor[.int64](),
                down_b_scales.to_tile_tensor[.int64](),
                expert_start_indices.to_tile_tensor[.int64](),
                a_scale_offsets.to_tile_tensor[.int64](),
                expert_ids.to_tile_tensor[.int64](),
                gate_up_expert_scales.to_tile_tensor[.int64](),
                down_expert_scales.to_tile_tensor[.int64](),
                num_active,
                Int(estimated_total_m),
                context,
                arrival_count_ptr,
                swiglu_alpha=swiglu_alpha,
                swiglu_limit=swiglu_limit,
            )
        elif has_gate_up_a_row_scales:
            comptime assert (
                row_scales_type == DType.bfloat16
            ), "per-row input scales must be bfloat16"
            mega_ffn_nvfp4_dispatch[
                num_experts=num_experts,
                transpose_b=True,
                clamp_activation=clamp_activation,
                RowScalesT=RealRowScales,
            ](
                output.to_tile_tensor[.int64](),
                c_packed,
                c_swiglu_scales,
                hidden_states.to_tile_tensor[.int64](),
                gate_up_weight.to_tile_tensor[.int64](),
                down_weight.to_tile_tensor[.int64](),
                gate_up_a_scales.to_tile_tensor[.int64](),
                gate_up_b_scales.to_tile_tensor[.int64](),
                down_b_scales.to_tile_tensor[.int64](),
                expert_start_indices.to_tile_tensor[.int64](),
                a_scale_offsets.to_tile_tensor[.int64](),
                expert_ids.to_tile_tensor[.int64](),
                gate_up_expert_scales.to_tile_tensor[.int64](),
                down_expert_scales.to_tile_tensor[.int64](),
                c_input_scales.to_tile_tensor[.int64](),
                num_active,
                Int(estimated_total_m),
                context,
                arrival_count_ptr,
                swiglu_alpha=swiglu_alpha,
                swiglu_limit=swiglu_limit,
                a_row_scales=RealRowScales(
                    rebind[Pointer[BFloat16, ImmutAnyOrigin]](
                        gate_up_a_row_scales.unsafe_ptr()
                    )
                ),
            )
        else:
            _ = gate_up_a_row_scales
            mega_ffn_nvfp4_dispatch[
                num_experts=num_experts,
                transpose_b=True,
                clamp_activation=clamp_activation,
            ](
                output.to_tile_tensor[.int64](),
                c_packed,
                c_swiglu_scales,
                hidden_states.to_tile_tensor[.int64](),
                gate_up_weight.to_tile_tensor[.int64](),
                down_weight.to_tile_tensor[.int64](),
                gate_up_a_scales.to_tile_tensor[.int64](),
                gate_up_b_scales.to_tile_tensor[.int64](),
                down_b_scales.to_tile_tensor[.int64](),
                expert_start_indices.to_tile_tensor[.int64](),
                a_scale_offsets.to_tile_tensor[.int64](),
                expert_ids.to_tile_tensor[.int64](),
                gate_up_expert_scales.to_tile_tensor[.int64](),
                down_expert_scales.to_tile_tensor[.int64](),
                c_input_scales.to_tile_tensor[.int64](),
                num_active,
                Int(estimated_total_m),
                context,
                arrival_count_ptr,
                swiglu_alpha=swiglu_alpha,
                swiglu_limit=swiglu_limit,
            )

        # Keep the on-chip scratch buffers alive until the launch is enqueued
        # (stream-ordered free schedules after the kernel completes).
        # `arrival_count` is a persistent graph buffer operand (owned by the
        # runtime, not allocated here), so it needs no keep-alive.
        _ = c_packed_buf^
        _ = c_swiglu_scales_buf^


@compiler.register("mega_ffn.ep_combine_send")
struct Struct_mega_ffn_ep_combine_send:
    """MOGG wrapper for the MegaFFN MoE FFN with the EP combine send fused in.

    Same kernel as `mo.composite.mega_ffn_nvfp4`, with the L2 epilogue's peer
    scatter-send and its pair-space arrival signal switched on. That makes the
    send the only consumer of the FFN's output, so the local store and its TMA
    descriptor are both elided and this op has NO tensor result: the down
    projection's `(max_recv_tokens, hidden_size)` staging buffer -- the largest
    single activation in the MoE region -- is never allocated.

    The model emits this op directly rather than a graph-compiler pattern
    rewriting into it. The send only pays for itself at prefill-scale tokens
    per expert, so it is a per-model choice rather than a universal fusion
    rule, and the caller here is the one place that knows a combine WAIT is
    downstream and therefore that the arrival signal must be published.

    A consumer still has to run `ep.combine_wait` afterwards to drain the peer
    buffers and reduce; that op reads only the symmetric receive buffers and
    never took the FFN's output, which is what makes dropping it possible.

    NVFP4 only, unlike its sibling: the send needs a per-expert input scale,
    and E8M0 cannot carry one.
    """

    @inline(.always)
    @staticmethod
    def execute[
        a_type: DType,
        b_type: DType,
        scales_type: DType,
        //,
        combine_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        clamp_activation: Bool,
        target: StaticString,
    ](
        arrival_count: MutableInputTensor[dtype=DType.uint32, rank=1, ...],
        atomic_counters: MutableInputTensor[dtype=DType.int32, rank=1, ...],
        hidden_states: InputTensor[dtype=a_type, rank=2, ...],
        gate_up_weight: InputTensor[dtype=b_type, rank=3, ...],
        gate_up_a_scales: InputTensor[dtype=scales_type, rank=5, ...],
        gate_up_b_scales: InputTensor[dtype=scales_type, rank=6, ...],
        down_weight: InputTensor[dtype=b_type, rank=3, ...],
        down_b_scales: InputTensor[dtype=scales_type, rank=6, ...],
        expert_start_indices: InputTensor[dtype=DType.uint32, rank=1, ...],
        expert_ids: InputTensor[dtype=DType.int32, rank=1, ...],
        a_scale_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        gate_up_expert_scales: InputTensor[dtype=DType.float32, rank=1, ...],
        down_expert_scales: InputTensor[dtype=DType.float32, rank=1, ...],
        c_input_scales: InputTensor[dtype=DType.float32, rank=1, ...],
        src_info: InputTensor[dtype=DType.int32, rank=2, ...],
        recv_ptrs: InputTensor[dtype=DType.uint64, rank=1, ...],
        recv_count_ptrs: InputTensor[dtype=DType.uint64, rank=1, ...],
        estimated_total_m: UInt32,
        num_active_experts: UInt32,
        context: DeviceContext,
    ) raises:
        """Runs the fused MoE FFN and sends its output straight to the peers.

        Parameters:
            a_type: Token / activation element type (inferred).
            b_type: Weight element type (inferred).
            scales_type: Block scale-factor dtype (inferred).
            combine_dtype: Payload dtype of the combine phase; also the dtype
                of the output tensor this op declines to allocate.
            hidden_size: Model hidden dimension, the send's row width.
            top_k: Experts each token routes to; the receive buffer's second
                dimension.
            n_experts: GLOBAL expert count across ranks. Distinct from the
                kernel's `num_experts`, which is this rank's LOCAL count and
                comes from the weight shape; the sync-counter layout is keyed
                on the global count.
            max_token_per_rank: Receive-buffer capacity per rank, in tokens.
            n_gpus_per_node: GPUs per node; the number of live pointer-table
                entries.
            n_nodes: Physical node count.
            clamp_activation: `True` selects the clamped (`swigluoai`) L1
                activation; the dispatch supplies its canonical constants.
            target: Target GPU device.

        Args:
            arrival_count: Persistent cross-CTA pool-slot buffer, zeroed once
                at setup. Also holds the send's own per-expert state -- the
                arrival-signal election marker and the L2 completion counts --
                in the padding of the slots the pool protocol never touches,
                which is why this op needs no per-launch-zeroed operand of its
                own.
            atomic_counters: EP sync counters for this device. The send reads
                the combine-async region for its destination resolve and the
                rank-completion counter that follows it.
            hidden_states: Dispatched tokens `(max_recv_tokens, K1 // 2)`.
            gate_up_weight: Pre-permuted gate/up weights `(E, N1, K1 // 2)`.
            gate_up_a_scales: Token A-scale tile for L1.
            gate_up_b_scales: Pre-permuted W13 scale tile.
            down_weight: Down-projection weights `(E, N2, moe_dim // 2)`.
            down_b_scales: W2 scale tile.
            expert_start_indices: Per-expert prefix-sum token offsets.
            expert_ids: Active local expert IDs (`-1` = masked).
            a_scale_offsets: Per-expert scale-block offsets.
            gate_up_expert_scales: L1 per-expert output scaling.
            down_expert_scales: L2 per-expert output scaling.
            c_input_scales: L1 per-expert input scale for the NVFP4 re-quant.
            src_info: Per-row dispatch metadata the send resolves destinations
                from; the high bits carry the one-based source rank.
            recv_ptrs: Peer combine receive-buffer addresses, one per rank.
            recv_count_ptrs: Peer receive-count addresses the arrival signal
                releases, one per rank.
            estimated_total_m: Estimated total non-padded tokens; the tile
                regime gate's numerator.
            num_active_experts: Active expert slots; the kernel walks one
                shared expert list for both legs.
            context: Device context.
        """
        comptime assert is_gpu[
            target
        ](), "the fused MegaFFN combine send only supports GPUs"

        comptime assert (
            n_gpus_per_node * n_nodes <= P3_MAX_RANKS
        ), "the send resolves destinations into fixed P3_MAX_RANKS-wide tables"

        # The graph sizes `arrival_count` from constants restated on the Python
        # side, because a graph cannot read a Mojo comptime. Nothing keeps the
        # two in step, and a stride that disagrees silently corrupts the pool
        # rather than failing, so check the shape the kernel actually needs.
        var arrival_words = Int(arrival_count.dim_size[0]())
        if (
            arrival_words % ATOMIC_PAD != 0
            or arrival_words < (n_experts + 1) * ATOMIC_PAD
        ):
            raise Error(
                "arrival_count must be a whole number of ATOMIC_PAD-word pool"
                " slots and hold at least n_experts + 1 of them, since the"
                " send keeps its per-expert words in slot padding"
            )

        var num_active = Int(num_active_experts)
        if num_active == 0:
            return

        # Enforced here too, so the binding cannot grow a second dispatch
        # arm that nothing reaches.
        comptime assert (
            a_type != DType.float8_e4m3fn
        ), "the fused combine send is NVFP4-only"
        comptime sf_vector_size = NVFP4_SF_VECTOR_SIZE

        comptime num_experts = Int(gate_up_weight.static_spec.shape_tuple[0])
        comptime down_experts = Int(down_weight.static_spec.shape_tuple[0])
        comptime assert (
            num_experts == down_experts
        ), "gate_up and down weights must have the same expert count"

        comptime moe_dim = Int(gate_up_weight.static_spec.shape_tuple[1]) // 2
        comptime packed_K2 = Int(down_weight.static_spec.shape_tuple[2])
        comptime N2 = Int(down_weight.static_spec.shape_tuple[1])
        comptime assert (
            N2 == hidden_size
        ), "the send's row width must equal the down projection's N"

        comptime assert (
            packed_K2 == moe_dim // 2
        ), "down_weight K dim must equal moe_dim // 2 (packed NVFP4)"

        comptime k_groups_swiglu = ceildiv(moe_dim, sf_vector_size * SF_ATOM_K)

        var m_total = Int(hidden_states.dim_size[0]())
        var a_scale_dim0 = Int(gate_up_a_scales.dim_size[0]())

        # On-chip scratch, same capture-safe per-call pattern as the sibling
        # registration: write-then-read inside the launch, so no init.
        comptime CPackedType = DType.uint8
        var c_packed_buf = context.enqueue_create_buffer[CPackedType](
            m_total * packed_K2
        )
        var c_packed = TileTensor(
            c_packed_buf.unsafe_ptr(),
            row_major(Int64(m_total), Idx[packed_K2]),
        )

        var s_size = (
            a_scale_dim0
            * k_groups_swiglu
            * SF_ATOM_M[0]
            * SF_ATOM_M[1]
            * SF_ATOM_K
        )
        var c_swiglu_scales_buf = context.enqueue_create_buffer[scales_type](
            s_size
        )
        var c_swiglu_scales = TileTensor(
            c_swiglu_scales_buf.unsafe_ptr(),
            row_major(
                Int64(a_scale_dim0),
                Idx[k_groups_swiglu],
                Idx[SF_ATOM_M[0]],
                Idx[SF_ATOM_M[1]],
                Idx[SF_ATOM_K],
            ),
        )

        # The point of this op. With the peer send on, the epilogue's local
        # store and its TMA encode are both gone, so C needs no allocation at
        # all -- not even an empty one, which would still cost a stream-ordered
        # alloc and free per layer per launch. It stays 2-D with a STATIC
        # trailing dim because the launcher asserts `c_device`'s N equals the
        # down projection's; only the row extent goes to zero.
        var c_device = TileTensor(
            UnsafePointer[
                Scalar[combine_dtype], MutAnyOrigin
            ].unsafe_dangling(),
            row_major(Int(0), Idx[N2]),
        )

        # Destination resolve reads the combine-async counter region; the
        # arrival signal's last-arriver election reads the rank-completion
        # counter that follows it, keyed on the GLOBAL expert count.
        var ep_counters = EPLocalSyncCounters[n_experts](
            atomic_counters.unsafe_ptr().unsafe_origin_cast[
                MutUntrackedOrigin
            ]()
        )
        var combine_counter = ep_counters.get_combine_async_ptr()

        # The rank is a property of the device this launch lands on, not of the
        # graph, so derive it the way the EP host APIs do rather than baking it
        # into a parameter and specializing the op per device.
        var my_rank = Int32(context.id())
        comptime if n_nodes > 1:
            my_rank = Int32(shmem_my_pe())

        # Peer address tables, widened from `n_gpus_per_node` live entries to
        # the fixed width the kernel's stack arrays use. Entries past the world
        # size are never indexed: the resolve walks `range(p3_n_ranks)`.
        var recv_arr = pack_ptrs_array[DType.uint8](
            recv_ptrs.to_tile_tensor[.int64](), my_rank
        )
        var recv_bufs = StaticTuple[
            UnsafePointer[UInt8, MutUntrackedOrigin], P3_MAX_RANKS
        ](UnsafePointer[UInt8, MutUntrackedOrigin].unsafe_dangling())
        var count_arr = pack_ptrs_array[DType.uint64](
            recv_count_ptrs.to_tile_tensor[.int64](), my_rank
        )
        var recv_counts = StaticTuple[
            UnsafePointer[UInt64, MutUntrackedOrigin], P3_MAX_RANKS
        ](UnsafePointer[UInt64, MutUntrackedOrigin].unsafe_dangling())
        for r in range(n_gpus_per_node):
            recv_bufs[r] = recv_arr[r]
            recv_counts[r] = count_arr[r]

        comptime n_ranks = n_gpus_per_node * n_nodes
        comptime msg_bytes = size_of[combine_dtype]() * hidden_size

        comptime swiglu_alpha = Float32(1.702) if clamp_activation else Float32(
            0.0
        )
        comptime swiglu_limit = Float32(7.0) if clamp_activation else Float32(
            0.0
        )

        var ep_send = EPCombineSendOperands(
            control=0,
            atomic_counter=combine_counter,
            src_info_ptr=src_info.unsafe_ptr()
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmUntrackedOrigin](),
            row_base_ptr=UnsafePointer[
                UInt64, MutUntrackedOrigin
            ].unsafe_dangling(),
            recv_buf_ptrs=recv_bufs,
            n_ranks=n_ranks,
            p2p_world_size=n_gpus_per_node,
            top_k=top_k,
            msg_bytes=msg_bytes,
            max_tokens_per_rank=max_token_per_rank,
            signal_control=0,
            recv_count_ptrs=recv_counts,
            rank_completion_counter=combine_counter.unsafe_offset(
                2 * n_experts
            ),
            my_rank=my_rank,
        )

        mega_ffn_nvfp4_dispatch[
            num_experts=num_experts,
            transpose_b=True,
            clamp_activation=clamp_activation,
            p5_direct_scatter=True,
            p4_signal=True,
            emit_ffn_done=True,
        ](
            c_device,
            c_packed,
            c_swiglu_scales,
            hidden_states.to_tile_tensor[.int64](),
            gate_up_weight.to_tile_tensor[.int64](),
            down_weight.to_tile_tensor[.int64](),
            gate_up_a_scales.to_tile_tensor[.int64](),
            gate_up_b_scales.to_tile_tensor[.int64](),
            down_b_scales.to_tile_tensor[.int64](),
            expert_start_indices.to_tile_tensor[.int64](),
            a_scale_offsets.to_tile_tensor[.int64](),
            expert_ids.to_tile_tensor[.int64](),
            gate_up_expert_scales.to_tile_tensor[.int64](),
            down_expert_scales.to_tile_tensor[.int64](),
            c_input_scales.to_tile_tensor[.int64](),
            num_active,
            Int(estimated_total_m),
            context,
            arrival_count.unsafe_ptr(),
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
            ep_send=ep_send,
        )

        _ = c_packed_buf^
        _ = c_swiglu_scales_buf^


# ===----------------------------------------------------------------------=== #
# One-launch fused EP MoE: `mega_ffn.ep_fused_init` + `mega_ffn.ep_fused`
# ===----------------------------------------------------------------------=== #
#
# `mega_ffn.ep_fused` runs dispatch -> MegaFFN (L1 + SwiGLU + requant + L2) ->
# weighted combine in ONE launch per rank (`mega_ffn_block_scaled_ep_fused` with
# `ep_fused_full=True`), in the flag set the EP8 correctness gates qualified
# (`test_mega_ffn_ep_fused_combine_send_full_ep8{,_nvfp4}`) plus the
# device-derived EP generation (EP2): ready-pool L2 scheduling on dynamic L1
# tile claiming, final contiguous rows with the production block reservation,
# the in-epilogue send, the one-signal publication with poll-then-fence, and
# the in-kernel combine wait and reduce.
# NVFP4 or MXFP8 (`nvfp4`; unit input scale at MXFP8); PDL OFF (asserted).
#
# Workspace. `mega_ffn.ep_fused_init` runs once per device in the EP init graph,
# right after `ep.init`, and decides whether the fused path can serve the EP
# configuration (`fused_ep_unsupported_reason`). If it can, it allocates ONE
# arena per rank, the `_FusedEPGeometry.OFF_*` fields at 128-byte offsets,
# and records it in the process-wide global cache under this rank's dispatch
# receive-count address. `ep.init` allocated that buffer for this EP instance
# and never frees it, so the key is unique per EP instance and rank, and
# `mega_ffn.ep_fused` finds every rank's arena through the receive-count table
# the EP layer already passes. The init op records an arena only after it holds
# its initial values. Arenas and records live as long as the process, like
# `ep.init`'s buffers: nothing frees them at model teardown, so a captured
# launch keeps valid addresses. The fused counters live in the words
# `EPLocalSyncCounters` reserves past its regions (`MegaMoECounters`, in the
# device's group-0 EP counter buffer). The init op takes that buffer in place
# in the EP init graph, after `ep.init`, and synchronizes before it records the
# arena; that graph finishes before any model graph runs `mega_ffn.ep_fused`,
# whose launches take the same buffer in place, which orders them.
#
# Nothing is reset per launch: the kernel's own generation protocol
# (parity-banked ingress state reset at the next monitor entry, sentinel re-arm
# by the FULL waiter, POST arrival rotation) carries every word from launch to
# launch. So an EP instance serves one execution at a time, as its shipping
# buffers do: two graphs running concurrently on one instance would interleave
# that state.
#
# EP generation: the kernel reads the device generation word
# `arrival_count[GEN_WORD]` at entry and derives the ingress bank parity and
# the block-base mailbox tag from it (`ep_device_generation=True`). Nothing
# per-launch comes from the host, so a captured launch replays with the right
# bank and tag; this binding passes BANK-0 bases of two-bank fields whose
# strides are the launcher's (`FusedEPBankGeometry`), and the device-shared
# `arrival_count` (one per EP instance and device, shared by every fused
# layer of that instance) is what counts launches.

comptime _FUSED_EP_WS_VERSION = 2
"""Recorded with every arena; bump it when the arena or counter layout
changes."""

comptime _FUSED_EP_WS_LAYOUT_WORDS = 10
"""Words of `_FusedEPGeometry.layout()`."""


@inline(.always)
def _arena_end(offset: Int, nbytes: Int) -> Int:
    """Returns the next field's offset: the first multiple of 128 bytes at or
    past `nbytes` bytes from `offset`.

    A field needs 16-byte alignment (TMA bases, 128-bit accesses), which the
    init op checks for the arena base; a field starts an L2 line only when
    the allocator's base does."""
    return align_up(offset + nbytes, 128)


struct _FusedEPGeometry[
    hidden_size: Int,
    top_k: Int,
    n_experts: Int,
    max_tpr: Int,
    n_ranks: Int,
    moe_dim: Int,
    nvfp4: Bool,
    token_block: Int,
]:
    """Comptime geometry shared by `mega_ffn.ep_fused_init` and
    `mega_ffn.ep_fused`, so the allocation and the launch cannot disagree.

    Every size is the fused gate harness's (`test_mega_ffn_ep_fused.mojo`):
    NVFP4 or MXFP8 (`nvfp4`) token format at 16-byte alignment, the MMA
    config (mma (128, token_block, 32), cta_group 1; token block 8 = decode,
    32 = the dense specialization, the widest block the row-gather scale
    transport serves without an SF-atom arena; k_group 4, except MXFP8 at
    token block 32, which runs its 8 input-ring K tiles as 4 groups of 2),
    staging rows
    `n_local * n_ranks * max_tpr`, and the pool-slot bound from the
    `mega_ffn_scheduler` helpers, which also sizes the fused counters'
    Region H.
    """

    comptime n_local = Self.n_experts // Self.n_ranks
    comptime n1 = 2 * Self.moe_dim
    # Element format: NVFP4 (uint8 E2M1 pairs, E4M3 scales per 16 elements)
    # or MXFP8 (float8_e4m3fn, E8M0 scales per 32). The K extents are BYTES.
    comptime qdt = DType.uint8 if Self.nvfp4 else DType.float8_e4m3fn
    comptime sfdt = NVFP4_SF_DTYPE if Self.nvfp4 else MXFP8_SF_DTYPE
    comptime sfv = (
        NVFP4_SF_VECTOR_SIZE if Self.nvfp4 else MXFP8_SF_VECTOR_SIZE
    )
    comptime k1 = Self.hidden_size // 2 if Self.nvfp4 else Self.hidden_size
    comptime k2 = Self.moe_dim // 2 if Self.nvfp4 else Self.moe_dim
    comptime k1g = Self.hidden_size // (Self.sfv * SF_ATOM_K)
    comptime k2g = Self.moe_dim // (Self.sfv * SF_ATOM_K)
    comptime n1g = Self.n1 // 128
    comptime n2g = Self.hidden_size // 128
    comptime rows = Self.n_local * Self.n_ranks * Self.max_tpr
    comptime sf_blocks = Self.rows // SF_MN_GROUP_SIZE + Self.n_local + 1
    comptime sf_atom = SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K

    comptime OutLayout = type_of(row_major[Self.rows, Self.k1]())
    comptime OffLayout = type_of(row_major[Self.n_local + 1]())
    comptime Fmt = NVBlockScaledTokenFormat[
        quant_dtype=Self.qdt,
        scales_dtype=Self.sfdt,
        output_layout=Self.OutLayout,
        scales_offset_layout=Self.OffLayout,
        Self.hidden_size,
        Self.top_k,
        16,
    ]
    comptime msg = Self.Fmt.msg_size()
    comptime scales_off = Self.Fmt.scales_offset()

    # Receive counts: the (n_local, n_ranks) grid, the ORD-R rank flags, then
    # the per-local-expert reservation cursors (`recv_count_size()`).
    comptime rc_words = Self.n_experts + Self.n_ranks + Self.n_local
    comptime rc_cursor_base = Self.n_experts + Self.n_ranks

    # Bank strides: the kernel selects bank `gen & 1` with the launcher's
    # EXACT strides (`FusedEPBankGeometry`, in elements of each buffer's
    # type; bytes for the staging buffer), so each parity-banked field below
    # holds two banks of exactly these sizes.
    comptime banks = FusedEPBankGeometry[
        Self.n_local, Self.n_ranks, Self.max_tpr, Self.msg
    ]
    comptime staging_bank = Self.banks.staging_bank_bytes
    comptime rc_bank_words = Self.banks.rc_bank_words
    comptime ec_bank_words = Self.banks.ec_bank_words
    comptime ef_bank_words = Self.banks.ef_bank_words
    comptime eir_bank_words = Self.banks.eir_bank_words

    comptime pool_slots = fused_l2_pool_slots(
        Self.n_local, Self.n_ranks, Self.max_tpr, Self.top_k, Self.token_block
    )
    # The fused counters and `arrival_count`, in the EP counters' reserve.
    comptime Counters = MegaMoECounters[Self.n_experts, Self.pool_slots]
    comptime wait_ctr_base = EPLocalSyncCounters[
        Self.n_experts
    ].dispatch_async_size()
    comptime rank_prefix_off = 2 * Self.n_experts

    # The arena: byte offsets from a rank's base, in field order.
    # Parity-banked fields hold two banks of the launcher's exact stride.
    # Peers write the staging, receive and early counts, early flags,
    # combine receive buffer and its counts.
    comptime OFF_STAGING = 0
    comptime OFF_RC = _arena_end(Self.OFF_STAGING, 2 * Self.staging_bank)
    comptime OFF_EC = _arena_end(Self.OFF_RC, 2 * Self.rc_bank_words * 8)
    comptime OFF_EF = _arena_end(Self.OFF_EC, 2 * Self.ec_bank_words * 8)
    comptime OFF_EIR = _arena_end(Self.OFF_EF, 2 * Self.ef_bank_words * 4)
    # Row offsets, expert ids and scale offsets: one copy, written and read
    # inside one launch.
    comptime OFF_RO = _arena_end(Self.OFF_EIR, 2 * Self.eir_bank_words * 4)
    comptime OFF_EI = _arena_end(Self.OFF_RO, (Self.n_local + 1) * 4)
    comptime OFF_OFF = _arena_end(Self.OFF_EI, Self.n_local * 4)
    comptime OFF_SEND = _arena_end(Self.OFF_OFF, (Self.n_local + 1) * 4)
    # Early-publication election counters, then the block-base mailbox
    # (bases, then generation tags), then the dedicated rank-completion words
    # (not Region B).
    comptime OFF_EPC = _arena_end(Self.OFF_SEND, Self.max_tpr * Self.msg)
    comptime OFF_MAILBOX = _arena_end(Self.OFF_EPC, Self.n_ranks * 4)
    comptime OFF_RCC = _arena_end(Self.OFF_MAILBOX, 2 * Self.n_experts * 4)
    comptime OFF_SRC_INFO = _arena_end(Self.OFF_RCC, Self.n_ranks * 4)
    comptime OFF_CRECV = _arena_end(Self.OFF_SRC_INFO, Self.rows * 2 * 4)
    comptime OFF_CRC = _arena_end(
        Self.OFF_CRECV, Self.max_tpr * Self.top_k * Self.hidden_size * 2
    )
    # The packed L1 -> L2 intermediate and its 5D block scales, then the L1
    # token and token-scale placeholders the fused path never reads.
    comptime OFF_INTER = _arena_end(Self.OFF_CRC, Self.n_experts * 8)
    comptime OFF_INTER_SF = _arena_end(Self.OFF_INTER, Self.rows * Self.k2)
    comptime OFF_PH_A = _arena_end(
        Self.OFF_INTER_SF, Self.sf_blocks * Self.k2g * Self.sf_atom
    )
    comptime OFF_PH_ASC = _arena_end(Self.OFF_PH_A, Self.rows * Self.k1)
    comptime ARENA_BYTES = _arena_end(
        Self.OFF_PH_ASC, Self.sf_blocks * Self.k1g * Self.sf_atom
    )

    @staticmethod
    def layout() -> StaticTuple[Int, _FUSED_EP_WS_LAYOUT_WORDS]:
        """Returns what an arena of this geometry is laid out for."""
        return StaticTuple[Int, _FUSED_EP_WS_LAYOUT_WORDS](
            _FUSED_EP_WS_VERSION,
            Self.hidden_size,
            Self.top_k,
            Self.n_experts,
            Self.max_tpr,
            Self.n_ranks,
            Self.moe_dim,
            Int(Self.nvfp4),
            Self.token_block,
            Self.ARENA_BYTES,
        )

    # MXFP8 at token block 32: 4 input-ring group stages of 2 K tiles instead
    # of 2 of 4, the same 8 K tiles pinned. Left to the config, k_group 2
    # sizes the ring to 10 K tiles, which exceeds the fused SMEM budget.
    comptime kg2 = not Self.nvfp4 and Self.token_block == 32

    @staticmethod
    def _pipe_stages() -> Optional[Int]:
        if Self.kg2:
            return 8
        return None

    comptime config = BlockScaledMatmulConfig[
        Self.qdt,
        Self.qdt,
        DType.bfloat16,
        Self.sfdt,
        Self.sfdt,
        True,
    ](
        scaling_kind=(
            UMMAKind.KIND_MXF4NVF4 if Self.nvfp4 else UMMAKind.KIND_MXF8F6F4
        ),
        cluster_shape=Index(1, 1, 1),
        mma_shape=Index(128, Self.token_block, 32),
        block_swizzle_size=8,
        cta_group=1,
        AB_swapped=True,
        k_group_size=2 if Self.kg2 else 4,
        num_pipeline_stages=Self._pipe_stages(),
        num_accum_pipeline_stages=2,
        is_gmm=True,
        gemm_kind=GEMMKind.GMM,
    )


@fieldwise_init
struct _FusedEPWorkspaceRecord(Copyable, Movable):
    """One rank's fused workspace as recorded in the global cache."""

    var layout: StaticTuple[Int, _FUSED_EP_WS_LAYOUT_WORDS]
    """`_FusedEPGeometry.layout()` of the arena."""

    var device: Int
    """The device holding the arena."""

    var arena: Int
    """The arena's device address."""


def _fused_ep_ws_key(ep_recv_count_addr: UInt64) -> String:
    """Returns the global-cache key of the arena of the rank whose dispatch
    receive-count buffer (from `ep.init`) is at `ep_recv_count_addr`."""
    return String(t"MEGA_FFN_EP_FUSED_WS_{ep_recv_count_addr}")


def _fused_counters_reserve[
    n_experts: Int
](
    atomic_counters: MutableInputTensor[dtype=DType.int32, rank=1, ...]
) raises -> UnsafePointer[Int32, MutUntrackedOrigin]:
    """Returns the words one device's EP counter buffer reserves past its
    regions, where the fused counters (`MegaMoECounters`) live."""
    if (
        Int(atomic_counters.dim_size[0]())
        != EPLocalSyncCounters[n_experts].allocation_size()
    ):
        raise Error("atomic_counters must be one EP counter buffer")
    return EPLocalSyncCounters[n_experts](
        atomic_counters.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    ).get_reserved_ptr()


@inline(.always)
def _fill[
    dtype: DType
](
    context: DeviceContext, address: Int, count: Int, value: Scalar[dtype]
) raises:
    """Enqueues setting `count` elements at device `address` to `value`."""
    context.enqueue_memset(
        DeviceBuffer(
            context,
            UnsafePointer[Scalar[dtype], MutUntrackedOrigin](
                unsafe_from_address=address
            ),
            count,
            owning=False,
        ),
        value,
    )


struct _FusedEPPlan[
    dispatch_dtype: DType,
    hidden_size: Int,
    top_k: Int,
    n_experts: Int,
    max_token_per_rank: Int,
    n_gpus_per_node: Int,
    n_nodes: Int,
    dispatch_scale_dtype: DType,
    dispatch_fmt_str: StaticString,
    moe_dim: Int,
    fused_shared_expert: Bool,
    eplb: Bool,
    nvfp4_dyn_global_scales: Bool,
]:
    """What the fused path does for an EP configuration, shared by
    `mega_ffn.ep_fused_init` (which then allocates) and `mega_ffn.ep_fused_plan`
    (which only reports): why the backend refuses it, or the geometry it
    serves it with. Read `Geometry` only when `reason` is empty."""

    comptime reason = fused_ep_unsupported_reason(
        hidden_size=Self.hidden_size,
        moe_dim=Self.moe_dim,
        top_k=Self.top_k,
        n_experts=Self.n_experts,
        max_tokens_per_rank=Self.max_token_per_rank,
        n_ranks=Self.n_gpus_per_node,
        n_nodes=Self.n_nodes,
        token_fmt=Self.dispatch_fmt_str,
        quant_dtype=Self.dispatch_dtype,
        scales_dtype=Self.dispatch_scale_dtype,
        dyn_global_scales=Self.nvfp4_dyn_global_scales,
        fused_shared_expert=Self.fused_shared_expert,
        eplb=Self.eplb,
    )
    """Why the backend does not serve the configuration; empty if it does."""

    comptime Geometry = _FusedEPGeometry[
        Self.hidden_size,
        Self.top_k,
        Self.n_experts,
        Self.max_token_per_rank,
        Self.n_gpus_per_node,
        Self.moe_dim,
        Self.dispatch_dtype == DType.uint8,
        fused_ep_token_block(Self.max_token_per_rank),
    ]
    """The arena and kernel geometry of a served configuration."""

    @staticmethod
    def write(
        workspace: OutputTensor[dtype=DType.uint64, rank=1, ...],
        refusal: OutputTensor[dtype=DType.uint8, rank=1, ...],
    ) raises:
        """Writes the answer: the arena's bytes and token block, or zeros and
        the NUL-padded reason the backend refuses."""
        comptime n_reason = Self.reason.byte_length()
        if (
            Int(workspace.dim_size[0]()) != 2
            or Int(refusal.dim_size[0]()) <= n_reason
        ):
            raise Error("workspace must be [2]; refusal must fit the reason")
        comptime if n_reason > 0:
            workspace[0] = 0
            workspace[1] = 0
        else:
            workspace[0] = UInt64(Self.Geometry.ARENA_BYTES)
            workspace[1] = UInt64(Self.Geometry.token_block)
        for i in range(Int(refusal.dim_size[0]())):
            refusal[i] = Self.reason.as_bytes()[i] if i < n_reason else 0


@compiler.register("mega_ffn.ep_fused_plan")
struct Struct_mega_ffn_ep_fused_plan:
    """MOGG wrapper that reports, without allocating, what
    `mega_ffn.ep_fused_init` does for the same parameters: the arena's bytes
    and token block, or why the backend refuses the configuration. It runs
    on the host, so a memory plan can ask before any device memory exists.

    With `exact=False` it reports an upper bound for a caller that knows the
    dispatch element type but not the format, its global-scale mode, the
    shared-expert rows or EPLB (a plan made before the checkpoint is
    parsed): the answer for the one format served with that element type,
    with none of those features, since EP init refuses and allocates nothing
    for every other variant.
    """

    @inline(.always)
    @staticmethod
    def execute[
        dispatch_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        dispatch_scale_dtype: DType,
        dispatch_fmt_str: StaticString,
        moe_dim: Int,
        fused_shared_expert: Bool,
        eplb: Bool,
        target: StaticString,
        *,
        nvfp4_dyn_global_scales: Bool = False,
        exact: Bool = True,
    ](
        workspace: OutputTensor[dtype=DType.uint64, rank=1, ...],
        refusal: OutputTensor[dtype=DType.uint8, rank=1, ...],
    ) raises:
        """Reports the fused path's answer for an EP configuration.

        Parameters:
            dispatch_dtype: The EP dispatch element type.
            hidden_size: Model hidden dimension.
            top_k: Routing top-k.
            n_experts: GLOBAL routed expert count.
            max_token_per_rank: The EP per-rank token capacity.
            n_gpus_per_node: GPUs per node (the EP world).
            n_nodes: Nodes.
            dispatch_scale_dtype: The EP dispatch block-scale type.
            dispatch_fmt_str: The EP dispatch format (`ep.init`'s).
            moe_dim: Routed experts' intermediate width (0 when unknown).
            fused_shared_expert: Whether the dispatch carries shared-expert
                rows.
            eplb: Whether EPLB remaps experts to replicas.
            target: Target device.
            nvfp4_dyn_global_scales: Whether NVFP4 tokens carry their own
                global scale.
            exact: False for the upper bound described above; the format
                parameters are then ignored.

        Args:
            workspace: `[2]` (host): the arena's bytes and token block, 0
                and 0 when refused.
            refusal: Host bytes of why the backend refuses the
                configuration, NUL-padded; all zero when it does not.
        """
        comptime if not exact:
            # Packed FP4 elements are served as NVFP4 and FP8 ones as MXFP8;
            # MXFP4, MXFP6 and 128x128 block FP8 share those element types
            # and are refused.
            comptime fp4 = (
                dispatch_dtype == DType.uint8
                or dispatch_dtype == DType.float4_e2m1fn
            )
            _FusedEPPlan[
                DType.uint8 if fp4 else dispatch_dtype,
                hidden_size,
                top_k,
                n_experts,
                max_token_per_rank,
                n_gpus_per_node,
                n_nodes,
                NVFP4_SF_DTYPE if fp4 else MXFP8_SF_DTYPE,
                "BLOCK_SCALED_NV",
                moe_dim,
                False,
                False,
                False,
            ].write(workspace, refusal)
        else:
            _FusedEPPlan[
                dispatch_dtype,
                hidden_size,
                top_k,
                n_experts,
                max_token_per_rank,
                n_gpus_per_node,
                n_nodes,
                dispatch_scale_dtype,
                dispatch_fmt_str,
                moe_dim,
                fused_shared_expert,
                eplb,
                nvfp4_dyn_global_scales,
            ].write(workspace, refusal)


@compiler.register("mega_ffn.ep_fused_init")
struct Struct_mega_ffn_ep_fused_init:
    """MOGG wrapper that sets up one device's fused EP MoE workspace.

    Runs once per device in the EP init graph, after `ep.init`. Writes why the
    fused path cannot serve the configuration, or allocates this rank's arena,
    gives it its allocation-time values, zeroes the fused counters and records
    the arena in the global cache.
    """

    @inline(.always)
    @staticmethod
    def execute[
        dispatch_dtype: DType,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        dispatch_scale_dtype: DType,
        dispatch_fmt_str: StaticString,
        moe_dim: Int,
        fused_shared_expert: Bool,
        eplb: Bool,
        target: StaticString,
        *,
        nvfp4_dyn_global_scales: Bool = False,
    ](
        workspace: OutputTensor[dtype=DType.uint64, rank=1, ...],
        refusal: OutputTensor[dtype=DType.uint8, rank=1, ...],
        atomic_counters: MutableInputTensor[dtype=DType.int32, rank=1, ...],
        ep_dev_ptrs: InputTensor[dtype=DType.uint64, rank=2, ...],
        context: DeviceContext,
    ) raises:
        """Sets up this device's fused workspace, or says why it cannot.

        Parameters:
            dispatch_dtype: The EP dispatch element type.
            hidden_size: Model hidden dimension.
            top_k: Routing top-k.
            n_experts: GLOBAL routed expert count.
            max_token_per_rank: The EP per-rank token capacity.
            n_gpus_per_node: GPUs per node (the EP world).
            n_nodes: Nodes.
            dispatch_scale_dtype: The EP dispatch block-scale type.
            dispatch_fmt_str: The EP dispatch format (`ep.init`'s).
            moe_dim: Routed experts' intermediate width (0 when unknown).
            fused_shared_expert: Whether the dispatch carries shared-expert
                rows.
            eplb: Whether EPLB remaps experts to replicas.
            target: Target GPU device.
            nvfp4_dyn_global_scales: Whether NVFP4 tokens carry their own
                global scale.

        Args:
            workspace: `[2]` (host): the arena's bytes and token block, 0
                and 0 when refused.
            refusal: Host bytes of why the fused path cannot serve the
                configuration, NUL-padded; all zero when it can.
            atomic_counters: This device's group-0 EP sync counters, whose
                reserved words hold the fused counters (zeroed here).
            ep_dev_ptrs: This device's `ep.init` pointers `[2, 3]` (host).
            context: Device context.
        """
        comptime assert is_gpu[target](), "the fused EP MoE only supports GPUs"
        comptime n_ranks = n_gpus_per_node
        comptime P = _FusedEPPlan[
            dispatch_dtype,
            hidden_size,
            top_k,
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
            dispatch_scale_dtype,
            dispatch_fmt_str,
            moe_dim,
            fused_shared_expert,
            eplb,
            nvfp4_dyn_global_scales,
        ]
        comptime if P.reason.byte_length() > 0:
            P.write(workspace, refusal)
        else:
            comptime G = P.Geometry
            comptime assert G.rc_words == G.rc_bank_words, (
                "the receive-count bank must be exactly the dispatch view's"
                " words"
            )
            if not is_p2p_enabled():
                raise Error("P2P is not supported on this system.")
            var counters = G.Counters(
                _fused_counters_reserve[n_experts](atomic_counters)
            )

            # `ep.init` just allocated this rank's buffers, so nothing is
            # recorded under their address yet.
            var key = _fused_ep_ws_key(ep_dev_ptrs[0, 2])
            if _get_global_or_null(key):
                raise Error(
                    "a fused EP MoE workspace is already recorded for this EP"
                    " instance"
                )
            var arena = Int(
                context.enqueue_create_buffer[DType.uint8](
                    G.ARENA_BYTES
                ).take_ptr()
            )
            if arena % 16 != 0:
                # TMA bases and 128-bit accesses need 16 bytes; the device
                # allocator guarantees at least 64.
                raise Error("the fused EP MoE arena must be 16-byte aligned")

            # Allocation-time values; the kernel owns them afterwards. A
            # parity-banked field holds two banks (the kernel picks `gen & 1`).
            _fill(
                context,
                arena + G.OFF_RC,
                2 * G.rc_bank_words,
                UInt64.MAX_FINITE,
            )
            for b in range(2):
                # Reservation cursors are counts (start at 0), not flags.
                _fill(
                    context,
                    arena
                    + G.OFF_RC
                    + (b * G.rc_bank_words + G.rc_cursor_base) * 8,
                    G.n_local,
                    UInt64(0),
                )
            _fill(
                context,
                arena + G.OFF_EC,
                2 * G.ec_bank_words,
                UInt64.MAX_FINITE,
            )
            _fill(context, arena + G.OFF_EF, 2 * G.ef_bank_words, Int32(0))
            _fill(context, arena + G.OFF_EIR, 2 * G.eir_bank_words, Int32(0))
            _fill(context, arena + G.OFF_RO, G.n_local + 1, UInt32(0))
            _fill(context, arena + G.OFF_EI, G.n_local, Int32(-1))
            _fill(context, arena + G.OFF_OFF, G.n_local + 1, UInt32(0))
            _fill(context, arena + G.OFF_EPC, n_ranks, Int32(0))
            # Tags start at 0 and every launch publishes a strictly larger
            # one, so a stale base can never satisfy a later launch's wait.
            _fill(context, arena + G.OFF_MAILBOX, 2 * n_experts, Int32(0))
            _fill(context, arena + G.OFF_RCC, n_ranks, Int32(0))
            _fill(context, arena + G.OFF_SRC_INFO, G.rows * 2, Int32(-1))
            # MAX_FINITE = "no count published yet"; the FULL waiter re-arms it.
            _fill(context, arena + G.OFF_CRC, n_experts, UInt64.MAX_FINITE)
            # The staging, send, combine receive, intermediate and placeholder
            # fields need no initial value.

            _fill(context, Int(counters.base), G.Counters.words, Int32(0))

            # Publish only an initialized arena: a launch needs every rank's
            # record, so a cache hit must mean that rank's initial values are
            # in place. An error before this point leaves no record. The cache
            # never replaces an entry (a second insert is dropped silently),
            # so read the record back.
            context.synchronize()
            var record = alloc(
                AllocLayout[_FusedEPWorkspaceRecord].single()
            ).unsafe_leak()
            record.unsafe_write(
                _FusedEPWorkspaceRecord(G.layout(), Int(context.id()), arena)
            )
            global_cache_insert(key, record.bitcast[NoneType]())
            var stored = _get_global_or_null(key)
            if not stored or Int(stored.value()) != Int(record):
                raise Error("could not record the fused EP MoE workspace")
            P.write(workspace, refusal)


@compiler.register("mega_ffn.ep_fused")
struct Struct_mega_ffn_ep_fused:
    """MOGG wrapper for the one-launch fused EP MoE (dispatch -> MegaFFN ->
    weighted combine) on NVFP4 or MXFP8 routed experts.

    Replaces `ep.dispatch_async` + `ep.dispatch_wait`, the local FFN and
    `ep.combine_async` + `ep.combine_wait` for one MoE layer on one device and
    returns the token-indexed routed output. Every rank must launch it for
    every layer, including a rank that holds zero tokens (its peers wait for
    its early counts and flags), so this binding never skips the launch.
    """

    @inline(.always)
    @staticmethod
    @__parameter
    def execute[
        b_type: DType,
        scales_type: DType,
        //,
        hidden_size: Int,
        top_k: Int,
        n_experts: Int,
        max_token_per_rank: Int,
        n_gpus_per_node: Int,
        n_nodes: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=DType.bfloat16, rank=2, ...],
        atomic_counters: MutableInputTensor[dtype=DType.int32, rank=1, ...],
        input_tokens: InputTensor[dtype=DType.bfloat16, rank=2, ...],
        topk_ids: InputTensor[dtype=DType.int32, rank=2, ...],
        router_weights: InputTensor[dtype=DType.float32, rank=2, ...],
        input_scales: InputTensor[dtype=DType.float32, rank=1, ...],
        gate_up_weight: InputTensor[dtype=b_type, rank=3, ...],
        gate_up_b_scales: InputTensor[dtype=scales_type, rank=6, ...],
        down_weight: InputTensor[dtype=b_type, rank=3, ...],
        down_b_scales: InputTensor[dtype=scales_type, rank=6, ...],
        gate_up_expert_scales: InputTensor[dtype=DType.float32, rank=1, ...],
        down_expert_scales: InputTensor[dtype=DType.float32, rank=1, ...],
        c_input_scales: InputTensor[dtype=DType.float32, rank=1, ...],
        ep_recv_count_ptrs: InputTensor[dtype=DType.uint64, rank=1, ...],
        context: DeviceContext,
    ) raises:
        """Runs one fused EP MoE layer on this device.

        Parameters:
            b_type: Weight element type (inferred): `uint8` packed E2M1 at
                NVFP4, `float8_e4m3fn` at MXFP8.
            scales_type: Weight block-scale type (inferred): E4M3 at NVFP4,
                E8M0 at MXFP8.
            hidden_size: Model hidden dimension.
            top_k: Routing top-k.
            n_experts: GLOBAL routed expert count.
            max_token_per_rank: The EP per-rank token capacity.
            n_gpus_per_node: GPUs per node (the EP world).
            n_nodes: Must be 1 (P2P only).
            target: Target GPU device.

        Args:
            output: Token-indexed routed output `(T, hidden)` BF16.
            atomic_counters: This device's group-0 EP sync counters
                (in place); their reserved words hold the fused counters.
            input_tokens: Pre-dispatch BF16 tokens `(T, hidden)`.
            topk_ids: Routed global expert ids `(T, top_k)`.
            router_weights: Router weights `(T, top_k)`, applied once.
            input_scales: Inverted NVFP4 dispatch input scale; `[0]` is read.
                At MXFP8 pass 1.0: the quantization ignores it.
            gate_up_weight: Sigma-permuted gate/up weights `(E, N1, K1 bytes)`.
            gate_up_b_scales: Gate/up weight scales `(E, N1/128, K1g, 32, 4, 4)`.
            down_weight: Down weights `(E, hidden, moe bytes)`.
            down_b_scales: Down weight scales `(E, hidden/128, K2g, 32, 4, 4)`.
            gate_up_expert_scales: Per-local-expert L1 alpha.
            down_expert_scales: Per-local-expert L2 alpha.
            c_input_scales: Per-local-expert L1 -> L2 requant scale.
            ep_recv_count_ptrs: The EP dispatch receive-count table (host),
                which names every rank's arena.
            context: Device context.
        """
        comptime assert is_gpu[target](), "the fused EP MoE only supports GPUs"
        comptime n_ranks = n_gpus_per_node
        # The format and MoE width follow from the weights; the decision and
        # geometry come from the same plan `mega_ffn.ep_fused_init` used.
        comptime nvfp4 = b_type == DType.uint8
        comptime P = _FusedEPPlan[
            b_type,
            hidden_size,
            top_k,
            n_experts,
            max_token_per_rank,
            n_gpus_per_node,
            n_nodes,
            scales_type,
            "BLOCK_SCALED_NV",
            Int(gate_up_weight.static_spec.shape_tuple[1]) // 2,
            False,  # fused_shared_expert
            False,  # eplb
            False,  # nvfp4_dyn_global_scales
        ]
        comptime assert P.reason.byte_length() == 0, (
            "the fused EP MoE does not serve this configuration;"
            " `mega_ffn.ep_fused_init` reports why"
        )
        comptime G = P.Geometry
        comptime assert (
            Int(gate_up_weight.static_spec.shape_tuple[0]) == G.n_local
            and Int(down_weight.static_spec.shape_tuple[0]) == G.n_local
        ), "the weights must carry exactly this rank's local experts"
        comptime assert (
            Int(gate_up_weight.static_spec.shape_tuple[1]) == G.n1
            and Int(gate_up_weight.static_spec.shape_tuple[2]) == G.k1
        ), "gate/up weights must be (E, 2 * moe_dim, K1 bytes)"
        comptime assert (
            Int(down_weight.static_spec.shape_tuple[1]) == hidden_size
            and Int(down_weight.static_spec.shape_tuple[2]) == G.k2
        ), "down weights must be (E, hidden, K2 bytes)"

        var counters = G.Counters(
            _fused_counters_reserve[n_experts](atomic_counters)
        )
        if Int(ep_recv_count_ptrs.dim_size[0]()) != n_ranks:
            raise Error("ep_recv_count_ptrs must hold one address per rank")
        var n_tok = Int(input_tokens.dim_size[0]())
        if n_tok > max_token_per_rank:
            raise Error(
                "the fused EP MoE's staging holds max_token_per_rank source"
                " tokens per rank; this batch puts more on this rank"
            )
        if (
            Int(topk_ids.dim_size[1]()) != top_k
            or Int(router_weights.dim_size[1]()) != top_k
        ):
            raise Error("topk_ids / router_weights must be (T, top_k)")
        # NOTE: no early return at n_tok == 0. A rank with no tokens still
        # publishes its (empty) early counts and flags, and still receives and
        # reduces nothing; skipping it would hang every peer.

        var my_rank = Int(context.id())
        # Every rank's arena, set up by `mega_ffn.ep_fused_init` before any
        # launch. Fail closed rather than launch into a missing or different
        # workspace: peers write into these arenas.
        var bases = Array[Int, n_ranks](fill=0)
        for r in range(n_ranks):
            var found = _get_global_or_null(
                _fused_ep_ws_key(ep_recv_count_ptrs[r])
            )
            if not found:
                raise Error(
                    "no fused EP MoE workspace for rank ",
                    r,
                    ": the EP init graph has not set it up",
                )
            ref record = found.value().unsafe_bitcast[
                _FusedEPWorkspaceRecord
            ]()[]
            if record.layout != G.layout() or record.device != r:
                raise Error(
                    "the fused EP MoE workspace of rank ",
                    r,
                    " was set up for another configuration",
                )
            bases[r] = record.arena

        # ---- peer tables: BANK-0 bases (the kernel selects the bank) ----
        var recv_buf_ptrs = Array[
            UnsafePointer[UInt8, MutUntrackedOrigin], n_ranks
        ](fill=UnsafePointer[UInt8, MutUntrackedOrigin].unsafe_dangling())
        var recv_count_ptrs = Array[
            UnsafePointer[UInt64, MutUntrackedOrigin], n_ranks
        ](fill=UnsafePointer[UInt64, MutUntrackedOrigin].unsafe_dangling())
        var early_count_ptrs = Array[
            UnsafePointer[UInt64, MutUntrackedOrigin], n_ranks
        ](fill=UnsafePointer[UInt64, MutUntrackedOrigin].unsafe_dangling())
        var early_flag_ptrs = Array[
            UnsafePointer[Int32, MutUntrackedOrigin], n_ranks
        ](fill=UnsafePointer[Int32, MutUntrackedOrigin].unsafe_dangling())
        var p3_recv = StaticTuple[
            UnsafePointer[UInt8, MutUntrackedOrigin], P3_MAX_RANKS
        ](UnsafePointer[UInt8, MutUntrackedOrigin].unsafe_dangling())
        var p4_rc = Array[
            UnsafePointer[UInt64, MutUntrackedOrigin], P3_MAX_RANKS
        ](fill=UnsafePointer[UInt64, MutUntrackedOrigin].unsafe_dangling())
        for r in range(n_ranks):
            recv_buf_ptrs[r] = UnsafePointer[UInt8, MutUntrackedOrigin](
                unsafe_from_address=bases[r] + G.OFF_STAGING
            )
            recv_count_ptrs[r] = UnsafePointer[UInt64, MutUntrackedOrigin](
                unsafe_from_address=bases[r] + G.OFF_RC
            )
            early_count_ptrs[r] = UnsafePointer[UInt64, MutUntrackedOrigin](
                unsafe_from_address=bases[r] + G.OFF_EC
            )
            early_flag_ptrs[r] = UnsafePointer[Int32, MutUntrackedOrigin](
                unsafe_from_address=bases[r] + G.OFF_EF
            )
            p3_recv[r] = UnsafePointer[UInt8, MutUntrackedOrigin](
                unsafe_from_address=bases[r] + G.OFF_CRECV
            )
            p4_rc[r] = UnsafePointer[UInt64, MutUntrackedOrigin](
                unsafe_from_address=bases[r] + G.OFF_CRC
            )

        # ---- this rank's buffers (bank-0 bases; the "prev" operands are
        # ignored under `ep_device_generation` and get the same base) ----
        var me = my_rank
        var staging_me = bases[me] + G.OFF_STAGING
        var rc_prev = UnsafePointer[UInt64, MutUntrackedOrigin](
            unsafe_from_address=bases[me] + G.OFF_RC
        )
        var eir_cur = UnsafePointer[Int32, MutUntrackedOrigin](
            unsafe_from_address=bases[me] + G.OFF_EIR
        )
        var eir_prev = eir_cur
        var ro_tt = TileTensor(
            UnsafePointer[Scalar[DType.uint32], MutUntrackedOrigin](
                unsafe_from_address=bases[me] + G.OFF_RO
            ),
            row_major[G.n_local + 1](),
        )
        var off_tt = TileTensor(
            UnsafePointer[Scalar[DType.uint32], MutUntrackedOrigin](
                unsafe_from_address=bases[me] + G.OFF_OFF
            ),
            row_major[G.n_local + 1](),
        )
        var ei_tt = TileTensor(
            UnsafePointer[Scalar[DType.int32], MutUntrackedOrigin](
                unsafe_from_address=bases[me] + G.OFF_EI
            ),
            row_major[G.n_local](),
        )

        # ---- MegaFFN operands (the gate harness's shapes) ----
        # C is dead (`c_store_dead`: the rows leave through the send), so it
        # needs no backing, as in `mega_ffn.ep_combine_send`.
        var c_tt = TileTensor(
            UnsafePointer[
                Scalar[DType.bfloat16], MutAnyOrigin
            ].unsafe_dangling(),
            row_major(Coord(Int(0), Idx[hidden_size])),
        )
        var a_tt = TileTensor(
            UnsafePointer[Scalar[G.qdt], ImmUntrackedOrigin](
                unsafe_from_address=bases[me] + G.OFF_PH_A
            ),
            row_major[G.rows, G.k1](),
        )
        var cp_tt = TileTensor(
            UnsafePointer[Scalar[G.qdt], MutUntrackedOrigin](
                unsafe_from_address=bases[me] + G.OFF_INTER
            ),
            row_major[G.rows, G.k2](),
        )
        var w13_tt = TileTensor(
            UnsafePointer[Scalar[G.qdt], ImmUntrackedOrigin](
                unsafe_from_address=Int(gate_up_weight.unsafe_ptr())
            ),
            row_major[G.n_local, G.n1, G.k1](),
        )
        var w2_tt = TileTensor(
            UnsafePointer[Scalar[G.qdt], ImmUntrackedOrigin](
                unsafe_from_address=Int(down_weight.unsafe_ptr())
            ),
            row_major[G.n_local, hidden_size, G.k2](),
        )
        var asc_tt = TileTensor(
            UnsafePointer[Scalar[G.sfdt], MutUntrackedOrigin](
                unsafe_from_address=bases[me] + G.OFF_PH_ASC
            ),
            row_major((Idx[G.sf_blocks], Idx[G.k1g], Idx[32], Idx[4], Idx[4])),
        )
        var csw_tt = TileTensor(
            UnsafePointer[Scalar[G.sfdt], MutUntrackedOrigin](
                unsafe_from_address=bases[me] + G.OFF_INTER_SF
            ),
            row_major((Idx[G.sf_blocks], Idx[G.k2g], Idx[32], Idx[4], Idx[4])),
        )
        var w13s_tt = TileTensor(
            UnsafePointer[Scalar[G.sfdt], MutUntrackedOrigin](
                unsafe_from_address=Int(gate_up_b_scales.unsafe_ptr())
            ),
            row_major(
                (
                    Idx[G.n_local],
                    Idx[G.n1g],
                    Idx[G.k1g],
                    Idx[32],
                    Idx[4],
                    Idx[4],
                )
            ),
        )
        var w2s_tt = TileTensor(
            UnsafePointer[Scalar[G.sfdt], MutUntrackedOrigin](
                unsafe_from_address=Int(down_b_scales.unsafe_ptr())
            ),
            row_major(
                (
                    Idx[G.n_local],
                    Idx[G.n2g],
                    Idx[G.k2g],
                    Idx[32],
                    Idx[4],
                    Idx[4],
                )
            ),
        )
        var e1_tt = TileTensor(
            UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin](
                unsafe_from_address=Int(gate_up_expert_scales.unsafe_ptr())
            ),
            row_major[G.n_local](),
        )
        var e2_tt = TileTensor(
            UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin](
                unsafe_from_address=Int(down_expert_scales.unsafe_ptr())
            ),
            row_major[G.n_local](),
        )
        var swiglu_out = RealSwiGLUOutput[G.k2, G.k2g, G.sfdt, G.sfv, False](
            UnsafePointer[UInt8, MutAnyOrigin](
                unsafe_from_address=bases[me] + G.OFF_INTER
            ),
            UnsafePointer[Scalar[G.sfdt], MutAnyOrigin](
                unsafe_from_address=bases[me] + G.OFF_INTER_SF
            ),
            UnsafePointer[Float32, ImmutAnyOrigin](
                unsafe_from_address=Int(c_input_scales.unsafe_ptr())
            ),
        )
        # The format's compaction targets are never written on the fused path
        # (the tokens are consumed straight from staging); they alias the L1
        # placeholders, which are never read either.
        var handler = G.Fmt(
            TileTensor(
                UnsafePointer[Scalar[G.qdt], MutUntrackedOrigin](
                    unsafe_from_address=bases[me] + G.OFF_PH_A
                ),
                row_major[G.rows, G.k1](),
            ),
            TileTensor(
                UnsafePointer[Scalar[G.sfdt], MutUntrackedOrigin](
                    unsafe_from_address=bases[me] + G.OFF_PH_ASC
                ),
                row_major((Idx[G.sf_blocks], Idx[G.k1g], Idx[32], Idx[16])),
            ),
            off_tt,
            context,
        )
        var in_tt = TileTensor(
            UnsafePointer[Scalar[DType.bfloat16], ImmUntrackedOrigin](
                unsafe_from_address=Int(input_tokens.unsafe_ptr())
            ),
            row_major((n_tok, Idx[hidden_size])),
        )
        var tk_tt = TileTensor(
            UnsafePointer[Scalar[DType.int32], ImmUntrackedOrigin](
                unsafe_from_address=Int(topk_ids.unsafe_ptr())
            ),
            row_major((n_tok, Idx[top_k])),
        )
        # Spelled exactly as the launcher's operand type
        # (`EPLocalSyncCounters[num_experts * ep_n_ranks]`): Mojo unifies type
        # parameters structurally, so `n_experts` would not match
        # `(n_experts // n_ranks) * n_ranks` here.
        comptime assert G.n_local * n_ranks == n_experts
        var ep_ctrs = EPLocalSyncCounters[G.n_local * n_ranks](counters.base)
        # The dispatch's global input scale: one device value read at index 0
        # for every token (production passes 1 / gate_up_input).
        var isc_p = UnsafePointer[Float32, ImmUntrackedOrigin](
            unsafe_from_address=Int(input_scales.unsafe_ptr())
        )

        @__parameter
        @inline(.always)
        @__copy_capture(isc_p)
        def in_scale_fn[dtype: DType](expert_id: Int) -> Scalar[dtype]:
            return isc_p.load(0).cast[dtype]()

        mega_ffn_block_scaled_ep_fused[
            config=G.config,
            num_experts=G.n_local,
            SwiGLUOutputT=type_of(swiglu_out),
            swiglu_match_bf16=True,
            swiglu_use_inplace=True,
            mode=MODE_MEGAFFN,
            clean_up=POST_SELF_CLEAN_UP,
            # FULL is specified with PDL off (asserted in-kernel).
            pdl_level=PDLLevel(),
            ep_dynamic_tile_claim=True,
            ep_eligible_pool_queue=True,
            ep_final_layout=True,
            ep_prod_reserve=True,
            ep_n_sms=B200.sm_count,
            ep_n_ranks=n_ranks,
            ep_max_tokens_per_rank=max_token_per_rank,
            ep_p2p_world_size=n_ranks,
            staging_rows=G.rows,
            staging_n_ranks=n_ranks,
            staging_max_tpr=max_token_per_rank,
            staging_msg_bytes=G.msg,
            staging_scales_offset=G.scales_off,
            staging_n_topk=top_k,
            emit_ffn_done=True,
            p5_direct_scatter=True,
            p5_row_cache=False,
            p4_signal=True,
            c_store_dead=True,
            ep_emit_src_info=True,
            p4_sweep_lane_parallel=True,
            p4_release_every=True,
            p4_direct_pair_release=True,
            p4_one_signal=True,
            p4_poll_fence=True,
            # MXFP8 at token block 32 only, the measured configuration and the
            # one its gates cover; NVFP4 and token block 8 keep the release at
            # the first k-group.
            ep_late_full_release=(not nvfp4 and G.token_block == 32),
            ep_coop_src_info=(not nvfp4 and G.token_block == 32),
            ep_fused_full=True,
            ep_fused_full_verify=False,
            ep_input_scales_wrapper=in_scale_fn,
            # Bank and tag from the device generation word; replay-safe.
            ep_device_generation=True,
        ](
            c_tt,
            a_tt,
            cp_tt,
            w13_tt,
            w2_tt,
            ro_tt,
            off_tt,
            ei_tt,
            e1_tt,
            e2_tt,
            asc_tt,
            csw_tt,
            w13s_tt,
            w2s_tt,
            G.n_local,
            context,
            swiglu_out,
            UnsafePointer[Scalar[DType.uint32], MutAnyOrigin](
                unsafe_from_address=Int(counters.arrival_count())
            ),
            UnsafePointer[UInt8, ImmutAnyOrigin](
                unsafe_from_address=staging_me
            ),
            # Within-expert rank prefixes: Region B of the fused counters.
            UnsafePointer[Scalar[DType.int32], ImmutAnyOrigin](
                unsafe_from_address=Int(counters.base)
                + (G.wait_ctr_base + G.rank_prefix_off) * size_of[Int32]()
            ),
            in_tt,
            tk_tt,
            ro_tt,
            ei_tt,
            handler,
            UnsafePointer[UInt8, MutUntrackedOrigin](
                unsafe_from_address=bases[me] + G.OFF_SEND
            ),
            recv_buf_ptrs,
            recv_count_ptrs,
            early_count_ptrs,
            rc_prev,
            early_flag_ptrs,
            UnsafePointer[Int32, MutUntrackedOrigin](
                unsafe_from_address=bases[me] + G.OFF_EPC
            ),
            eir_cur,
            eir_prev,
            ep_ctrs,
            Int32(my_rank),
            ep_prod_base_p=UnsafePointer[Int32, MutUntrackedOrigin](
                unsafe_from_address=bases[me] + G.OFF_MAILBOX
            ),
            # Ignored under `ep_device_generation` (the tag is `gen + 1`).
            ep_prod_gen=Int32(0),
            src_info_ptr=UnsafePointer[Int32, MutUntrackedOrigin](
                unsafe_from_address=bases[me] + G.OFF_SRC_INFO
            ),
            p3_control=0,
            p3_atomic_counter=ep_ctrs.get_combine_async_ptr(),
            p3_recv_buf_ptrs=p3_recv,
            p3_n_ranks=n_ranks,
            p3_p2p_world_size=n_ranks,
            p3_top_k=top_k,
            p3_msg_bytes=hidden_size * size_of[DType.bfloat16](),
            p3_max_tokens_per_rank=max_token_per_rank,
            p4_control=0,
            p4_recv_count_ptrs=p4_rc,
            # A DEDICATED word per destination: Region B (the combine's usual
            # convention) is read live by this launch's L1 loader.
            p4_rank_completion_counter=UnsafePointer[Int32, MutUntrackedOrigin](
                unsafe_from_address=bases[me] + G.OFF_RCC
            ),
            p4_my_rank=Int32(my_rank),
            ff_out_p=UnsafePointer[Scalar[DType.bfloat16], MutUntrackedOrigin](
                unsafe_from_address=Int(output.unsafe_ptr())
            ),
            ff_rw_p=UnsafePointer[Float32, ImmUntrackedOrigin](
                unsafe_from_address=Int(router_weights.unsafe_ptr())
            ),
        )
