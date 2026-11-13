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
"""The fused W4A8 MoE FFN op and the two-launch chain it replaces, as graphs.

Shared by the surface test that checks the op's contract without compiling and
the GPU test that runs both against each other. The shapes are
MiMo-V2.6-Flash's per-rank expert slice (``K1 = 4096`` over ``N1 = 1024``
gate-up rows, ``N2 = 4096`` over ``K2 = 512``), the geometry
``max/kernels/src``'s ``mega_ffn.mega_ffn_mimo`` is locked to.

The op is the graph-level entry to the TileTensor port of the W4A8 Moka
component ``Kernels/test/gpu/Moka/kernels/ffn/megaffn_plan.mojo``; the chain is
the fused SwiGLU grouped matmul followed by the grouped block-scaled matmul,
which is what a routed MoE layer calls today and what the port reproduces byte
for byte.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from max.dtype import DType
from max.graph import BufferType, DeviceRef, Graph, TensorType, Type, ops
from max.nn._graph_ops import (
    MEGAFFN_MIMO_COUNTER_WORDS,
    MEGAFFN_MIMO_K1,
    MEGAFFN_MIMO_K2,
    MEGAFFN_MIMO_N1,
    MEGAFFN_MIMO_N2,
    grouped_matmul_block_scaled,
    grouped_matmul_blocked_swiglu,
    megaffn_mimo_w4a8,
)
from max.nn.activation import GatedActivation

K1 = MEGAFFN_MIMO_K1
N1 = MEGAFFN_MIMO_N1
N2 = MEGAFFN_MIMO_N2
K2 = MEGAFFN_MIMO_K2
SF_MN_GROUP_SIZE = 128
"""Token rows one input-scale group covers."""
SF_ATOM_M = (32, 4)
SF_ATOM_K = 4
"""The tcgen05 scale tile: 32 rows, 4 groups of 4, 4 scales of 32 elements."""
E8M0 = DType.float8_e8m0fnu
E4M3 = DType.float8_e4m3fn

BAND_ROWS = 512
FLOOR_ROWS = 64
"""The band MiMo-V2.6-Flash ships: the op takes the one launch for
``FLOOR_ROWS < rows <= BAND_ROWS`` and runs the two-launch chain outside it.
Deliberately restated here rather than read from the layer, so a change to the
layer's constants cannot quietly move what these graphs compare."""

FUSED_ACTIVATION = GatedActivation(round_to_bf16=True)
"""MiMo's fused activation, which both arms are given: plain ``silu`` with its
f32 result rounded through ``bfloat16`` before the requantize."""


def scale_rows(num_active_experts: int, rows_per_expert: int) -> int:
    """Returns the input-scale rows a front-packed routing of that shape takes.

    One row a 128-token-row group of an expert, which is what
    `a_scale_offsets` indexes.

    Args:
        num_active_experts: The number of experts with routed rows.
        rows_per_expert: The rows each of them carries.

    Returns:
        The scale tile's first dimension.
    """
    return num_active_experts * (
        (rows_per_expert + SF_MN_GROUP_SIZE - 1) // SF_MN_GROUP_SIZE
    )


def input_specs(
    num_experts: int,
    rows: int | str,
    groups: int | str,
    device_ref: DeviceRef,
    cpu_ref: DeviceRef,
) -> tuple[list[tuple[str, Type[Any]]], list[tuple[str, Type[Any]]]]:
    """Returns the (name, type) inputs of the op's graph and of the chain's.

    The op also takes the counter workspace, as a buffer it mutates; the chain
    ignores it, and takes the host usage stats the op reads off the routing
    arrays instead.

    Args:
        num_experts: The routing arrays' expert count.
        rows: The routed token row count, or a symbolic dimension's name.
        groups: The input-scale rows, or a symbolic dimension's name.
        device_ref: The accelerator device.
        cpu_ref: The host device, for the chain's usage stats.

    Returns:
        The op's input list and the chain's, each ordered as its graph takes
        them.
    """
    routing = [
        (
            "expert_start_indices",
            TensorType(DType.uint32, (num_experts + 1,), device=device_ref),
        ),
        (
            "a_scale_offsets",
            TensorType(DType.uint32, (num_experts,), device=device_ref),
        ),
        (
            "expert_ids",
            TensorType(DType.int32, (num_experts,), device=device_ref),
        ),
        (
            "expert_scales_l1",
            TensorType(DType.float32, (num_experts,), device=device_ref),
        ),
        (
            "expert_scales_l2",
            TensorType(DType.float32, (num_experts,), device=device_ref),
        ),
    ]
    shared = [
        ("hidden_states", TensorType(E4M3, (rows, K1), device=device_ref)),
        (
            "w13",
            TensorType(
                DType.uint8, (num_experts, N1, K1 // 2), device=device_ref
            ),
        ),
        (
            "w2",
            TensorType(
                DType.uint8, (num_experts, N2, K2 // 2), device=device_ref
            ),
        ),
        (
            "a_scales",
            TensorType(
                E8M0,
                (groups, K1 // 128, *SF_ATOM_M, SF_ATOM_K),
                device=device_ref,
            ),
        ),
        (
            "w13_scales",
            TensorType(
                E8M0,
                (num_experts, N1 // 128, K1 // 128, *SF_ATOM_M, SF_ATOM_K),
                device=device_ref,
            ),
        ),
        (
            "w2_scales",
            TensorType(
                E8M0,
                (num_experts, N2 // 128, K2 // 128, *SF_ATOM_M, SF_ATOM_K),
                device=device_ref,
            ),
        ),
        *routing,
    ]
    port: list[tuple[str, Type[Any]]] = [
        *shared,
        (
            "arrival_count",
            BufferType(
                DType.uint32, (MEGAFFN_MIMO_COUNTER_WORDS,), device=device_ref
            ),
        ),
    ]
    chain: list[tuple[str, Type[Any]]] = [
        *shared,
        ("usage_stats", TensorType(DType.uint32, (2,), device=cpu_ref)),
    ]
    return port, chain


def np_inputs(
    num_experts: int,
    num_active_experts: int,
    rows_per_expert: int,
    seed: int = 20261105,
    *,
    sparse: bool = False,
) -> dict[str, np.ndarray]:
    """Synthesizes one routing's worth of arrays.

    By default the first `num_active_experts` experts carry rows, so the chain's
    front-packed contract holds, and every slot keeps an identity weight index
    the way `moe_create_indices` writes them: an expert with no rows is then a
    zero-width range that both arms skip.

    `sparse` moves that routing onto slots spread across the array instead, one
    active slot every `num_experts // num_active_experts`, and gives the empty
    slots between them an id of -1. That is the other layout a routing takes: an
    empty slot between two active ones, and one the walk has to skip by its id
    rather than by a measured zero-width range. Both arms have to agree on it,
    and both kernels' schedulers fast-skip a negative id.

    Weights are uniform bytes, the gate and up rows stacked and then
    sigma-permuted on the N axis (`sigma(2i) = i, sigma(2i+1) = N1 // 2 + i`,
    the permutation the fused SwiGLU epilogue pairs on), and the scales stay
    near 2^-7, so neither leg saturates E4M3.

    Args:
        num_experts: The routing arrays' expert count.
        num_active_experts: The experts with routed rows.
        rows_per_expert: The rows each active expert carries.
        seed: The generator's seed.
        sparse: Whether to spread the active experts over the slots, leaving
            -1 ids in the empty ones.

    Returns:
        The arrays, keyed by the names `input_specs` uses.

    Raises:
        ValueError: If `sparse` is asked for an expert count the number of
            active experts does not divide.
    """
    rng = np.random.default_rng(seed)
    rows = num_active_experts * rows_per_expert
    groups = scale_rows(num_active_experts, rows_per_expert)
    if sparse:
        if num_experts % num_active_experts:
            raise ValueError(
                f"{num_active_experts} active experts do not spread over"
                f" {num_experts} slots"
            )
        stride = num_experts // num_active_experts
        slots = [slot * stride for slot in range(num_active_experts)]
    else:
        slots = list(range(num_active_experts))

    hidden = rng.integers(0, 256, size=(rows, K1), dtype=np.uint8)
    hidden[(hidden & 0x7F) == 0x7F] = 0

    gate = rng.integers(
        0, 256, size=(num_experts, N1 // 2, K1 // 2), dtype=np.uint8
    )
    up = rng.integers(
        0, 256, size=(num_experts, N1 // 2, K1 // 2), dtype=np.uint8
    )
    w13 = np.empty((num_experts, N1, K1 // 2), dtype=np.uint8)
    w13[:, 0::2] = gate
    w13[:, 1::2] = up
    w2 = rng.integers(0, 256, size=(num_experts, N2, K2 // 2), dtype=np.uint8)

    def _scales(shape: tuple[int, ...]) -> np.ndarray:
        return (118 + rng.integers(0, 4, size=shape)).astype(np.uint8)

    a_scales = _scales((groups, K1 // 128, *SF_ATOM_M, SF_ATOM_K))
    w13_scales = _scales(
        (num_experts, N1 // 128, K1 // 128, *SF_ATOM_M, SF_ATOM_K)
    )
    w2_scales = _scales(
        (num_experts, N2 // 128, K2 // 128, *SF_ATOM_M, SF_ATOM_K)
    )

    counts = np.zeros(num_experts, dtype=np.uint32)
    counts[slots] = rows_per_expert
    expert_start_indices = np.zeros(num_experts + 1, dtype=np.uint32)
    expert_start_indices[1:] = np.cumsum(counts)
    # Each expert's first scale group minus its first row's, the offset the
    # per-expert scale tiles are addressed by.
    group_of_expert = np.zeros(num_experts + 1, dtype=np.uint32)
    for e in range(num_experts):
        group_of_expert[e + 1] = group_of_expert[e] + (
            (int(counts[e]) + SF_MN_GROUP_SIZE - 1) // SF_MN_GROUP_SIZE
        )
    a_scale_offsets = (
        group_of_expert[:num_experts]
        - expert_start_indices[:num_experts] // SF_MN_GROUP_SIZE
    ).astype(np.uint32)
    expert_ids = np.arange(num_experts, dtype=np.int32)
    if sparse:
        expert_ids[counts == 0] = -1

    return {
        "hidden_states": hidden,
        "w13": w13,
        "w2": w2,
        "a_scales": a_scales,
        "w13_scales": w13_scales,
        "w2_scales": w2_scales,
        "expert_start_indices": expert_start_indices,
        "a_scale_offsets": a_scale_offsets,
        "expert_ids": expert_ids,
        # Per-leg expert scales, distinct so an arm that swapped the operands
        # would show it in the bytes.
        "expert_scales_l1": (
            1.0 + np.arange(num_experts, dtype=np.float32) + 1.0
        )
        / num_experts,
        "expert_scales_l2": (
            1.0 + np.arange(num_experts, dtype=np.float32) + 2.0
        )
        / num_experts,
        "usage_stats": np.array(
            [rows_per_expert, num_experts], dtype=np.uint32
        ),
    }


def build_port_graph(
    specs: list[tuple[str, Type[Any]]],
    *,
    band_rows: int = BAND_ROWS,
    floor_rows: int = FLOOR_ROWS,
) -> Graph:
    """Builds the graph of one fused op call.

    Args:
        specs: The op's input (name, type) list from `input_specs`.
        band_rows: The most routed rows that take the one launch; 0 keeps every
            step on the chain.
        floor_rows: The fewest routed rows that take the one launch.

    Returns:
        The graph, whose outputs are the bf16 result, the packed intermediate
        and that intermediate's scale tiles.
    """
    with Graph("megaffn_mimo_w4a8", input_types=[t for _, t in specs]) as graph:
        args = {
            name: value
            for (name, _), value in zip(specs, graph.inputs, strict=True)
        }
        result, packed, scales = megaffn_mimo_w4a8(
            args["hidden_states"].tensor,
            args["w13"].tensor,
            args["w2"].tensor,
            args["a_scales"].tensor,
            args["w13_scales"].tensor,
            args["w2_scales"].tensor,
            args["expert_start_indices"].tensor,
            args["a_scale_offsets"].tensor,
            args["expert_ids"].tensor,
            args["expert_scales_l1"].tensor,
            args["expert_scales_l2"].tensor,
            args["arrival_count"].buffer,
            band_rows=band_rows,
            floor_rows=floor_rows,
            activation=FUSED_ACTIVATION,
        )
        graph.output(result, packed, scales)
    return graph


def build_chain_graph(specs: list[tuple[str, Type[Any]]]) -> Graph:
    """Builds the graph of the two-launch chain the op replaces.

    The intermediate is a graph output, which is what keeps the MegaFFN
    fusion from collapsing the legs: the reference has to stay two launches.

    The row estimate the legs are launched with is the routed row count, as the
    model passes it, rather than the fixture's rows-per-expert: the dispatch
    picks its config from rows per active expert, and a test that compared an
    arm launched under one config against a reference launched under another
    would be comparing configurations, not arms.

    Args:
        specs: The chain's input (name, type) list from `input_specs`.

    Returns:
        The graph, with the op's three outputs in the same order.
    """
    with Graph(
        "megaffn_mimo_chain", input_types=[t for _, t in specs]
    ) as graph:
        args = {
            name: value
            for (name, _), value in zip(specs, graph.inputs, strict=True)
        }
        estimated_total_m = ops.shape_to_tensor(
            args["hidden_states"].tensor.shape
        )[0].cast(DType.uint32)
        packed, scales = grouped_matmul_blocked_swiglu(
            args["hidden_states"].tensor,
            args["w13"].tensor,
            args["a_scales"].tensor,
            args["w13_scales"].tensor,
            args["expert_start_indices"].tensor,
            args["a_scale_offsets"].tensor,
            args["expert_ids"].tensor,
            args["usage_stats"].tensor,
            expert_scales=args["expert_scales_l1"].tensor,
            estimated_total_m=estimated_total_m,
            activation=FUSED_ACTIVATION,
        )
        result = grouped_matmul_block_scaled(
            packed,
            args["w2"].tensor,
            scales,
            args["w2_scales"].tensor,
            args["expert_start_indices"].tensor,
            args["a_scale_offsets"].tensor,
            args["expert_ids"].tensor,
            args["expert_scales_l2"].tensor,
            args["usage_stats"].tensor,
            out_type=DType.bfloat16,
            estimated_total_m=estimated_total_m,
        )
        graph.output(result, packed, scales)
    return graph
