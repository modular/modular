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
"""The chained two-leg MoE FFN graphs the MegaFFN fusion rewrites.

Shared by the GPU test that runs them and the CPU test that checks the fusion
fires in the IR a virtual-device compile dumps.
"""

from __future__ import annotations

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kernels import (
    block_scales_interleave,
    grouped_matmul_block_scaled,
    grouped_matmul_blocked_swiglu,
)

# NVFP4 scale-tile geometry (mirrors kernels.py).
NVFP4_SF_VECTOR_SIZE = 16
SF_MN_GROUP_SIZE = 128  # SF_ATOM_M[0](32) * SF_ATOM_M[1](4)
# MXFP8 scale-tile geometry: E8M0 scales over 32-element blocks.
MXFP8_SF_VECTOR_SIZE = 32


def _random_uint8(
    shape: tuple[int, ...], rng: np.random.Generator
) -> np.ndarray:
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def _random_e4m3fn_safe(
    shape: tuple[int, ...], rng: np.random.Generator
) -> np.ndarray:
    """Random float8_e4m3fn bytes with the single NaN encoding masked to +0."""
    arr = rng.integers(0, 256, size=shape, dtype=np.uint8)
    arr[(arr & 0x7F) == 0x7F] = 0
    return arr


def _sigma_permute_n(x: np.ndarray, d: int) -> np.ndarray:
    """Apply sigma(2i)=i, sigma(2i+1)=D+i on axis 1 (the gate/up N axis)."""
    assert x.shape[1] == 2 * d
    out = np.empty_like(x)
    out[:, 0::2] = x[:, :d]
    out[:, 1::2] = x[:, d:]
    return out


def build_np_inputs(
    E: int, M: int, D: int, K1: int, N2: int, rng: np.random.Generator
) -> tuple[dict[str, np.ndarray], int]:
    """Synthesize all per-tensor inputs for the chained L1 -> L2 MoE FFN.

    All ``E`` experts active; tokens distributed evenly. The gate-up leg
    contracts over ``K1`` and produces a ``2D``-wide pre-SwiGLU output whose
    SwiGLU result is ``D``-wide; the down leg contracts over ``D`` and produces
    ``N2``-wide bf16 output.

    Returns ``(arrays, sf_dim_0)`` where ``sf_dim_0`` is the shared first dim
    of the L1 a_scales / L1 SwiGLU scale tile (re-used as L2 a_scales).
    """
    K1_groups = K1 // NVFP4_SF_VECTOR_SIZE
    D_groups = D // NVFP4_SF_VECTOR_SIZE  # down-leg K-group count
    sf_dim_0 = M // SF_MN_GROUP_SIZE + E  # per-expert tail-pad slots

    # ---- L1 (gate-up) inputs ----
    hidden = _random_uint8((M, K1 // 2), rng)
    gate_packed = _random_uint8((E, D, K1 // 2), rng)
    up_packed = _random_uint8((E, D, K1 // 2), rng)
    # sigma-permuted gate/up weight (path-B layout).
    w13 = _sigma_permute_n(np.concatenate([gate_packed, up_packed], axis=1), D)

    # Pre-interleave per-expert b_scales (rank 3); the in-graph
    # block_scales_interleave lifts to the rank-5 tcgen05 layout per expert.
    gate_b_scales = _random_e4m3fn_safe((E, D, K1_groups), rng)
    up_b_scales = _random_e4m3fn_safe((E, D, K1_groups), rng)
    b_scales13_pre = _sigma_permute_n(
        np.concatenate([gate_b_scales, up_b_scales], axis=1), D
    )

    # a_scales already in rank-5 tcgen05 layout (shared by L1 and L2).
    a_scales = _random_e4m3fn_safe((sf_dim_0, K1_groups // 4, 32, 4, 4), rng)

    # ---- L2 (down) inputs ----
    # Down weight: (E, N2, D/2) packed NVFP4; contracts over D, outputs N2.
    down_w = _random_uint8((E, N2, D // 2), rng)
    # Down b_scales: per-expert rank-2 (N2, D_groups) -> block_scales_interleave
    # in-graph -> rank-5 per expert -> stacked to rank-6.
    down_b_scales_pre = _random_e4m3fn_safe((E, N2, D_groups), rng)

    # ---- shared routing / scalars ----
    tokens_per = M // E
    expert_start = np.array(
        [tokens_per * i for i in range(E + 1)], dtype=np.uint32
    )
    a_scale_offsets = np.arange(E, dtype=np.uint32)
    expert_ids = np.arange(E, dtype=np.int32)
    usage_stats = np.array([tokens_per, E], dtype=np.uint32)

    # Per-expert scales (values irrelevant for this shape/dtype validation).
    es13 = np.ones(E, dtype=np.float32)
    es_down = np.ones(E, dtype=np.float32)
    raw_input_scales = np.full(E, 0.5, dtype=np.float32)

    arrays = {
        "hidden": hidden,
        "w13": w13,
        "a_scales": a_scales,
        "b_scales13_pre": b_scales13_pre,
        "down_w": down_w,
        "down_b_scales_pre": down_b_scales_pre,
        "expert_start": expert_start,
        "a_scale_offsets": a_scale_offsets,
        "expert_ids": expert_ids,
        "es13": es13,
        "es_down": es_down,
        "usage_stats": usage_stats,
        "raw_input_scales": raw_input_scales,
    }
    return arrays, sf_dim_0


def build_graph(
    E: int,
    M: int,
    D: int,
    K1: int,
    N2: int,
    sf_dim_0: int,
    device_ref: DeviceRef,
    cpu_ref: DeviceRef,
    clamp: bool = False,
    row_scales: bool = False,
    keep_intermediate: bool = False,
) -> Graph:
    """Build the chained L1 -> L2 NVFP4 MoE FFN graph.

    The fusion fires only if both legs share the routing SSA values, L1's two
    outputs feed ONLY L2, and ``estimated_total_m`` defaults to the same
    ``usage_stats[0]`` SSA on both legs. The `arrival_count` scratch is NOT a
    graph input here: the fusion mints it (`mo.buffer.create`) when it fires.

    ``row_scales`` adds an ``(M,)`` bf16 input passed to L1 as ``a_row_scales``.
    ``keep_intermediate`` also returns L1's packed output, which blocks the
    fusion and leaves the two-launch chain.
    """
    K1_groups = K1 // NVFP4_SF_VECTOR_SIZE
    D_groups = D // NVFP4_SF_VECTOR_SIZE

    input_types: list[TensorType] = [
        TensorType(DType.uint8, (M, K1 // 2), device=device_ref),  # hidden
        TensorType(DType.uint8, (E, 2 * D, K1 // 2), device=device_ref),  # w13
        TensorType(
            DType.float8_e4m3fn,
            (sf_dim_0, K1_groups // 4, 32, 4, 4),
            device=device_ref,
        ),  # a_scales (shared L1/L2)
        TensorType(
            DType.float8_e4m3fn, (E, 2 * D, K1_groups), device=device_ref
        ),  # b_scales13_pre
        TensorType(DType.uint8, (E, N2, D // 2), device=device_ref),  # down_w
        TensorType(
            DType.float8_e4m3fn, (E, N2, D_groups), device=device_ref
        ),  # down_b_scales_pre
        TensorType(DType.uint32, (E + 1,), device=device_ref),  # expert_start
        TensorType(DType.uint32, (E,), device=device_ref),  # a_scale_offsets
        TensorType(DType.int32, (E,), device=device_ref),  # expert_ids
        TensorType(DType.float32, (E,), device=device_ref),  # es13
        TensorType(DType.float32, (E,), device=device_ref),  # es_down
        TensorType(DType.uint32, (2,), device=cpu_ref),  # usage_stats (host)
        TensorType(DType.float32, (E,), device=device_ref),  # raw_input_scales
    ]
    if row_scales:
        input_types.append(TensorType(DType.bfloat16, (M,), device=device_ref))

    with Graph("mega_ffn_nvfp4_fusion", input_types=input_types) as graph:
        (
            hidden_t,
            w13_t,
            a_scales_t,
            b_scales13_pre_t,
            down_w_t,
            down_b_scales_pre_t,
            expert_start_t,
            a_scale_offsets_t,
            expert_ids_t,
            es13_t,
            es_down_t,
            usage_stats_t,
            raw_input_scales_t,
        ) = (inp.tensor for inp in graph.inputs[:13])
        row_scales_t = graph.inputs[13].tensor if row_scales else None

        # Lift per-expert L1 b_scales (rank 3) to rank-6 tcgen05.
        b_scales13 = ops.stack(
            [
                block_scales_interleave(s.reshape([2 * D, K1_groups]))
                for s in ops.split(b_scales13_pre_t, [1] * E, axis=0)
            ],
            axis=0,
        )

        # Lift per-expert down b_scales (rank 3) to rank-6 tcgen05.
        down_b_scales = ops.stack(
            [
                block_scales_interleave(s.reshape([N2, D_groups]))
                for s in ops.split(down_b_scales_pre_t, [1] * E, axis=0)
            ],
            axis=0,
        )

        inv_input_scales = (
            ops.constant(1.0, DType.float32, device=device_ref)
            / raw_input_scales_t
        )

        # L1: gate-up + SwiGLU + bf16->nvfp4 quant. Emits
        # mo.composite.grouped_matmul_swiglu_nvfp4. estimated_total_m defaults
        # to usage_stats[0] (the SAME SSA the down leg defaults to).
        packed_b, sf_b = grouped_matmul_blocked_swiglu(
            hidden_t,
            w13_t,
            a_scales_t,
            b_scales13,
            expert_start_t,
            a_scale_offsets_t,
            expert_ids_t,
            usage_stats_t,
            expert_scales=es13_t,
            c_input_scales=inv_input_scales,
            # The emitter sets `clamp_activation` as a Bool op attribute; the
            # fusion copies it onto the fused op and the registration binds it
            # (supplying swigluoai's canonical alpha/limit to the dispatch). The
            # alpha/limit passed here are dropped by the emitter (carried only as
            # the selector), but the leg API requires them non-zero when clamping.
            clamp_activation=clamp,
            swiglu_alpha=1.702,
            swiglu_limit=7.0,
            a_row_scales=row_scales_t,
        )

        # L2: down GEMM. Emits mo.composite.grouped_matmul_block_scaled.
        # CRITICAL for the fusion to fire:
        #   - hidden_states = packed_b (L1 output #0), a_scales = sf_b (#1),
        #   - SAME SSA routing (expert_start, a_scale_offsets, expert_ids),
        #   - estimated_total_m defaults to usage_stats[0] (same SSA),
        #   - packed_b + sf_b consumed ONLY here (not graph.output).
        out = grouped_matmul_block_scaled(
            packed_b,
            down_w_t,
            sf_b,
            down_b_scales,
            expert_start_t,
            a_scale_offsets_t,
            expert_ids_t,
            es_down_t,
            usage_stats_t,
            out_type=DType.bfloat16,
        )

        if keep_intermediate:
            graph.output(out, packed_b)
        else:
            graph.output(out)

    return graph


def build_graph_mxfp8(
    E: int,
    M: int,
    D: int,
    K1: int,
    N2: int,
    sf_dim_0: int,
    device_ref: DeviceRef,
    cpu_ref: DeviceRef,
    clamp: bool = False,
) -> Graph:
    """Build the chained L1 -> L2 MXFP8 MoE FFN graph (MiniMax-M3 shape).

    MXFP8 vs NVFP4: elements are ``float8_e4m3fn`` (unpacked -- A's K stride is
    ``K1`` not ``K1/2``, the down weight's K is full ``D``); block scales are
    ``float8_e8m0fnu`` over 32-element blocks. It drives the SAME dtype-agnostic
    ``grouped_matmul_blocked_swiglu`` / ``grouped_matmul_block_scaled`` emitters,
    so the graph presents the same two leg composites to the fusion.
    """
    K1_groups = K1 // MXFP8_SF_VECTOR_SIZE
    D_groups = D // MXFP8_SF_VECTOR_SIZE

    input_types: list[TensorType] = [
        TensorType(DType.float8_e4m3fn, (M, K1), device=device_ref),  # hidden
        TensorType(
            DType.float8_e4m3fn, (E, 2 * D, K1), device=device_ref
        ),  # w13
        TensorType(
            DType.float8_e8m0fnu,
            (sf_dim_0, K1_groups // 4, 32, 4, 4),
            device=device_ref,
        ),  # a_scales (shared L1/L2)
        TensorType(
            DType.float8_e8m0fnu, (E, 2 * D, K1_groups), device=device_ref
        ),  # b_scales13_pre
        TensorType(
            DType.float8_e4m3fn, (E, N2, D), device=device_ref
        ),  # down_w
        TensorType(
            DType.float8_e8m0fnu, (E, N2, D_groups), device=device_ref
        ),  # down_b_scales_pre
        TensorType(DType.uint32, (E + 1,), device=device_ref),  # expert_start
        TensorType(DType.uint32, (E,), device=device_ref),  # a_scale_offsets
        TensorType(DType.int32, (E,), device=device_ref),  # expert_ids
        TensorType(DType.float32, (E,), device=device_ref),  # es13
        TensorType(DType.float32, (E,), device=device_ref),  # es_down
        TensorType(DType.uint32, (2,), device=cpu_ref),  # usage_stats (host)
        TensorType(DType.float32, (E,), device=device_ref),  # raw_input_scales
    ]

    with Graph("mega_ffn_mxfp8_fusion", input_types=input_types) as graph:
        (
            hidden_t,
            w13_t,
            a_scales_t,
            b_scales13_pre_t,
            down_w_t,
            down_b_scales_pre_t,
            expert_start_t,
            a_scale_offsets_t,
            expert_ids_t,
            es13_t,
            es_down_t,
            usage_stats_t,
            raw_input_scales_t,
        ) = (inp.tensor for inp in graph.inputs)

        # E8M0 -> block_scales_interleave needs SF_VECTOR_SIZE=32.
        b_scales13 = ops.stack(
            [
                block_scales_interleave(
                    s.reshape([2 * D, K1_groups]),
                    sf_vector_size=MXFP8_SF_VECTOR_SIZE,
                )
                for s in ops.split(b_scales13_pre_t, [1] * E, axis=0)
            ],
            axis=0,
        )
        down_b_scales = ops.stack(
            [
                block_scales_interleave(
                    s.reshape([N2, D_groups]),
                    sf_vector_size=MXFP8_SF_VECTOR_SIZE,
                )
                for s in ops.split(down_b_scales_pre_t, [1] * E, axis=0)
            ],
            axis=0,
        )

        inv_input_scales = (
            ops.constant(1.0, DType.float32, device=device_ref)
            / raw_input_scales_t
        )

        # L1: the emitter's e4m3 (MXFP8) branch produces the e4m3 intermediate +
        # an E8M0 scale tile. clamp_activation rides as a Bool op attribute ->
        # fusion -> registration (which supplies swigluoai's canonical
        # alpha/limit to mega_ffn_mxfp8_dispatch).
        packed_b, sf_b = grouped_matmul_blocked_swiglu(
            hidden_t,
            w13_t,
            a_scales_t,
            b_scales13,
            expert_start_t,
            a_scale_offsets_t,
            expert_ids_t,
            usage_stats_t,
            expert_scales=es13_t,
            c_input_scales=inv_input_scales,
            clamp_activation=clamp,
            swiglu_alpha=1.702,
            swiglu_limit=7.0,
        )

        out = grouped_matmul_block_scaled(
            packed_b,
            down_w_t,
            sf_b,
            down_b_scales,
            expert_start_t,
            a_scale_offsets_t,
            expert_ids_t,
            es_down_t,
            usage_stats_t,
            out_type=DType.bfloat16,
        )

        graph.output(out)

    return graph
