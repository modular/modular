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

"""Each row of an expert grouped GEMM is computed from that row alone.

EP dispatch fixes which expert a routed row belongs to but not its offset
inside the (expert, source rank) group, so the order inside a group changes
between runs. Bitwise emulation needs a grouped GEMM whose output row depends
only on the row: permuting the rows of a group must permute the output rows and
change no byte of them.

These tests run the two grouped matmuls the GLM-5.3 MoE runs after dispatch,
permute the real rows inside every expert group, run again, and require every
output row -- padding rows included -- to match its first-run bytes once the
permutation is undone.

- ``grouped_dynamic_scaled_fp8_matmul``: the blockwise-FP8 leg.
- ``grouped_matmul_block_scaled``: the NVFP4 leg.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
import torch
from max.driver import Accelerator, Buffer, accelerator_count
from max.driver.driver import accelerator_architecture_name
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn._graph_ops import (
    block_scales_interleave,
    grouped_dynamic_scaled_fp8_matmul,
    grouped_matmul_block_scaled,
)
from max.nn.quant_config import (
    InputScaleSpec,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)

pytestmark = pytest.mark.skipif(
    accelerator_count() == 0
    or not accelerator_architecture_name().startswith("sm_10"),
    reason="both grouped matmuls take their SM100 path here",
)

_E = 3
_K = 512
_N = 384
_SEED = 20260518

# Rows per expert, in dispatch order. 5/1/9 all sit under one MMA tile; 129
# crosses the 128-row NVFP4 scale/MMA tile and the FP8 64-row MMA tile.
_ROW_CASES: dict[str, list[int]] = {
    "5_1_9": [5, 1, 9],
    "1_17_129": [1, 17, 129],
}

# Prefill-sized FP8 groups: with the step's row estimate passed, rows that
# average more than 128 a group take the SM100 128-row tile, and 257 rows also
# leave a partial last tile.
_FP8_PREFILL_ROW_CASES: dict[str, list[int]] = {
    "130_257_200": [130, 257, 200],
}

# Decode-sized FP8 groups at an expert GEMM shape the SM100 swapped-AB tile
# serves (N 4096, K 2048): a row estimate of at most 8 rows a group selects
# it. The estimate only selects the tile, so the 41-row group still gets
# two 32-row token tiles.
_FP8_DECODE_N = 4096
_FP8_DECODE_K = 2048
_FP8_DECODE_ROW_CASES: dict[str, tuple[list[int], int]] = {
    "5_1_9": ([5, 1, 9], 15),
    "1_41_2": ([1, 41, 2], 24),
}

# Dispatch pads every expert's row range up to these counts: 4 rows for
# blockwise FP8, 16 for NVFP4 (`ep_comm.mojo:1307,1406-1429`).
_FP8_ROW_PAD = 4
_NVFP4_ROW_PAD = 16

# NVFP4 block-scale tile geometry, from `linalg/fp4_utils.mojo:610-617`.
_SF_VECTOR = 16
_SF_ATOM_M0 = 32
_SF_ATOM_M1 = 4
_SF_ATOM_K = 4
_SF_MN_GROUP = _SF_ATOM_M0 * _SF_ATOM_M1

_FP8_INPUT_SCALE = InputScaleSpec(
    granularity=ScaleGranularity.BLOCK,
    origin=ScaleOrigin.DYNAMIC,
    dtype=DType.float32,
    block_size=(1, 128),
)
_FP8_WEIGHT_SCALE = WeightScaleSpec(
    granularity=ScaleGranularity.BLOCK,
    dtype=DType.float32,
    block_size=(128, 128),
)


def _pad_to(counts: list[int], multiple: int) -> list[int]:
    return [-(-count // multiple) * multiple for count in counts]


def _starts(counts: list[int]) -> np.ndarray:
    return np.array([0, *np.cumsum(counts)], dtype=np.uint32)


def _row_permutation(
    real: list[int], counts: list[int], seed: int = _SEED
) -> npt.NDArray[np.int64]:
    """Permutes the real rows inside each group, leaving padding rows alone.

    ``order[i]`` is the original index of the row that moves to position ``i``,
    so a permuted run's output row ``i`` must equal the first run's ``order[i]``.
    """
    order = np.arange(sum(counts))
    rng = np.random.default_rng(seed)
    offset = 0
    for n_real, count in zip(real, counts, strict=True):
        if n_real > 1:
            perm = rng.permutation(n_real)
            if np.array_equal(perm, np.arange(n_real)):
                perm = perm[::-1].copy()
            order[offset : offset + n_real] = offset + perm
        offset += count
    return order


def _fp8_bytes(
    rng: np.random.Generator, shape: tuple[int, ...], scale: float
) -> np.ndarray:
    """Finite ``float8_e4m3fn`` bytes, so no NaN survives into the compare."""
    values = (rng.standard_normal(shape) * scale).astype(np.float32)
    return (
        torch.from_numpy(values)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .numpy()
    )


def _e4m3_bytes(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    values = rng.uniform(0.25, 0.5, size=shape).astype(np.float32)
    return (
        torch.from_numpy(values)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .numpy()
    )


def _buffer(array: np.ndarray, device: Accelerator) -> Buffer:
    return Buffer.from_numpy(np.ascontiguousarray(array)).to(device)


def _fp8_buffer(array: np.ndarray, device: Accelerator) -> Buffer:
    """Wraps ``float8_e4m3fn`` bytes without letting torch reinterpret them."""
    return (
        Buffer.from_dlpack(torch.from_numpy(np.ascontiguousarray(array)))
        .view(DType.float8_e4m3fn)
        .to(device)
    )


def _run(model: Model, inputs: list[Buffer]) -> torch.Tensor:
    return torch.from_dlpack(model.execute(*inputs)[0]).cpu().clone()


def _assert_rows_permute(
    got: torch.Tensor,
    permuted: torch.Tensor,
    order: npt.NDArray[np.int64],
) -> None:
    """Requires the permuted run to be the first run's rows, byte for byte."""
    assert torch.isfinite(got.float()).all()
    # An empty or row-replicated output would make the comparison vacuous.
    assert got.float().abs().max() > 0
    assert (got != got[0]).any()
    undone = torch.empty_like(permuted)
    undone[torch.from_numpy(order)] = permuted
    # Padding rows never move, so every row -- not just the real ones -- has to
    # come back unchanged.
    assert torch.equal(undone, got)


def _build_fp8(
    session: InferenceSession,
    device: Accelerator,
    total_m: int,
    estimated_total_m: int | None = None,
    n: int = _N,
    k: int = _K,
) -> Model:
    gpu = DeviceRef(device.label, device.id)
    cpu = DeviceRef.CPU()
    with Graph(
        "fp8_grouped_matmul_row_invariance",
        input_types=[
            TensorType(DType.float8_e4m3fn, (total_m, k), gpu),
            TensorType(DType.float8_e4m3fn, (_E, n, k), gpu),
            TensorType(DType.float32, (k // 128, total_m), gpu),
            TensorType(DType.float32, (_E, n // 128, k // 128), gpu),
            TensorType(DType.uint32, (_E + 1,), gpu),
            TensorType(DType.int32, (_E,), gpu),
            TensorType(DType.uint32, (2,), cpu),
        ],
    ) as graph:
        hidden, weight, a_scales, b_scales, starts, ids, stats = (
            value.tensor for value in graph.inputs
        )
        graph.output(
            grouped_dynamic_scaled_fp8_matmul(
                hidden,
                weight,
                a_scales,
                b_scales,
                starts,
                ids,
                stats,
                _FP8_INPUT_SCALE,
                _FP8_WEIGHT_SCALE,
                estimated_total_m=None
                if estimated_total_m is None
                else ops.constant(estimated_total_m, DType.uint32, device=cpu),
            )
        )
    return session.load(graph)


def _nvfp4_scale_tiles(
    per_row: np.ndarray, starts: np.ndarray, counts: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Packs per-row ``[M, K/16]`` NVFP4 scales into the tcgen05 5D tile.

    A row's scale for K vector ``v`` lands at
    ``[tile, v // 4, row % 32, (row % 128) // 32, v % 4]`` with ``row`` the
    row's offset inside its expert's own 128-row block, matching
    ``set_scale_factor``. Each expert owns ``ceildiv(count, 128)`` tiles, and
    ``a_scale_offsets`` rebases the kernel's absolute-row tile index onto them.
    """
    tiles_per_group = [-(-count // _SF_MN_GROUP) for count in counts]
    tile_base = np.cumsum([0, *tiles_per_group[:-1]]).astype(np.int64)
    tiles = np.zeros(
        (
            int(tile_base[-1] + tiles_per_group[-1]),
            per_row.shape[1] // _SF_ATOM_K,
            _SF_ATOM_M0,
            _SF_ATOM_M1,
            _SF_ATOM_K,
        ),
        dtype=np.uint8,
    )
    for group, count in enumerate(counts):
        for row in range(count):
            scale_row = per_row[int(starts[group]) + row].reshape(
                -1, _SF_ATOM_K
            )
            tiles[
                tile_base[group] + row // _SF_MN_GROUP,
                :,
                row % _SF_ATOM_M0,
                (row % _SF_MN_GROUP) // _SF_ATOM_M0,
                :,
            ] = scale_row
    offsets = tile_base - starts[:-1].astype(np.int64) // _SF_MN_GROUP
    return tiles, offsets.astype(np.uint32)


def _build_nvfp4(
    session: InferenceSession,
    device: Accelerator,
    total_m: int,
    num_tiles: int,
) -> Model:
    gpu = DeviceRef(device.label, device.id)
    cpu = DeviceRef.CPU()
    with Graph(
        "nvfp4_grouped_matmul_row_invariance",
        input_types=[
            TensorType(DType.uint8, (total_m, _K // 2), gpu),
            TensorType(DType.uint8, (_E, _N, _K // 2), gpu),
            TensorType(
                DType.float8_e4m3fn,
                (
                    num_tiles,
                    _K // (4 * _SF_VECTOR),
                    _SF_ATOM_M0,
                    _SF_ATOM_M1,
                    _SF_ATOM_K,
                ),
                gpu,
            ),
            TensorType(DType.float8_e4m3fn, (_E, _N, _K // _SF_VECTOR), gpu),
            TensorType(DType.uint32, (_E + 1,), gpu),
            TensorType(DType.uint32, (_E,), gpu),
            TensorType(DType.int32, (_E,), gpu),
            TensorType(DType.float32, (_E,), gpu),
            TensorType(DType.uint32, (2,), cpu),
        ],
    ) as graph:
        (
            hidden,
            weight,
            a_scales,
            b_scales_rows,
            starts,
            scale_offsets,
            ids,
            expert_scales,
            stats,
        ) = (value.tensor for value in graph.inputs)
        # Per-expert rank-3 scales lift to the rank-6 tcgen05 layout, the same
        # way `NvMxf4f8Strategy.prepare_weight_scales` builds them.
        b_scales = ops.stack(
            [
                block_scales_interleave(slab.reshape([_N, _K // _SF_VECTOR]))
                for slab in ops.split(b_scales_rows, [1] * _E, axis=0)
            ],
            axis=0,
        )
        graph.output(
            grouped_matmul_block_scaled(
                hidden,
                weight,
                a_scales,
                b_scales,
                starts,
                scale_offsets,
                ids,
                expert_scales,
                stats,
            )
        )
    return session.load(graph)


@pytest.mark.parametrize(
    "case",
    list(_ROW_CASES)
    + list(_FP8_PREFILL_ROW_CASES)
    + [f"decode_{name}" for name in _FP8_DECODE_ROW_CASES],
)
def test_fp8_blockwise_grouped_matmul_rows_are_position_invariant(
    case: str,
) -> None:
    rng = np.random.default_rng(_SEED)
    n, k = _N, _K
    estimated_total_m: int | None = None
    if case.startswith("decode_"):
        real, estimated_total_m = _FP8_DECODE_ROW_CASES[
            case.removeprefix("decode_")
        ]
        n, k = _FP8_DECODE_N, _FP8_DECODE_K
    elif case in _FP8_PREFILL_ROW_CASES:
        real = _FP8_PREFILL_ROW_CASES[case]
    else:
        real = _ROW_CASES[case]
    counts = _pad_to(real, _FP8_ROW_PAD)
    starts = _starts(counts)
    total_m = int(starts[-1])
    if case in _FP8_PREFILL_ROW_CASES:
        estimated_total_m = total_m

    hidden = _fp8_bytes(rng, (total_m, k), 1.0)
    weight = _fp8_bytes(rng, (_E, n, k), 0.5)
    # One A scale per (128-element K block, row), so a row's scale has to move
    # with it.
    a_scales = rng.uniform(0.005, 0.02, size=(k // 128, total_m)).astype(
        np.float32
    )
    b_scales = rng.uniform(0.005, 0.02, size=(_E, n // 128, k // 128)).astype(
        np.float32
    )
    ids = np.array([2, 0, 1], dtype=np.int32)
    stats = np.array([max(counts), _E], dtype=np.uint32)

    device = Accelerator()
    session = InferenceSession(devices=[device])
    model = _build_fp8(
        session, device, total_m, estimated_total_m=estimated_total_m, n=n, k=k
    )

    def inputs(rows: np.ndarray, scales: np.ndarray) -> list[Buffer]:
        return [
            _fp8_buffer(rows, device),
            _fp8_buffer(weight, device),
            _buffer(scales, device),
            _buffer(b_scales, device),
            _buffer(starts, device),
            _buffer(ids, device),
            Buffer.from_numpy(stats),
        ]

    got = _run(model, inputs(hidden, a_scales))
    order = _row_permutation(real, counts)
    permuted = _run(model, inputs(hidden[order], a_scales[:, order]))
    _assert_rows_permute(got, permuted, order)


@pytest.mark.parametrize("case", list(_ROW_CASES))
def test_nvfp4_grouped_matmul_rows_are_position_invariant(case: str) -> None:
    rng = np.random.default_rng(_SEED + 1)
    real = _ROW_CASES[case]
    counts = _pad_to(real, _NVFP4_ROW_PAD)
    starts = _starts(counts)
    total_m = int(starts[-1])

    # Any nibble pair is a finite E2M1 value, so random bytes are safe here.
    hidden = rng.integers(0, 256, size=(total_m, _K // 2), dtype=np.uint8)
    weight = rng.integers(0, 256, size=(_E, _N, _K // 2), dtype=np.uint8)
    # NVFP4 carries one A scale per (16-element K block, row), so the scales
    # permute with the rows or the arithmetic would change.
    per_row_scales = _e4m3_bytes(rng, (total_m, _K // _SF_VECTOR))
    b_scales_rows = _e4m3_bytes(rng, (_E, _N, _K // _SF_VECTOR))
    ids = np.array([2, 0, 1], dtype=np.int32)
    expert_scales = np.array([1.0, 0.5, 2.0], dtype=np.float32)
    stats = np.array([max(counts), _E], dtype=np.uint32)
    a_scales, scale_offsets = _nvfp4_scale_tiles(per_row_scales, starts, counts)

    device = Accelerator()
    session = InferenceSession(devices=[device])
    model = _build_nvfp4(session, device, total_m, len(a_scales))

    def inputs(rows: np.ndarray, scales: np.ndarray) -> list[Buffer]:
        return [
            _buffer(rows, device),
            _buffer(weight, device),
            _fp8_buffer(scales, device),
            _fp8_buffer(b_scales_rows, device),
            _buffer(starts, device),
            _buffer(scale_offsets, device),
            _buffer(ids, device),
            _buffer(expert_scales, device),
            Buffer.from_numpy(stats),
        ]

    got = _run(model, inputs(hidden, a_scales))
    order = _row_permutation(real, counts)
    # `a_scale_offsets` follows the padded row counts, not the order inside a
    # group, so the permuted run reuses it.
    permuted_scales, _ = _nvfp4_scale_tiles(
        per_row_scales[order], starts, counts
    )
    permuted = _run(model, inputs(hidden[order], permuted_scales))
    _assert_rows_permute(got, permuted, order)
