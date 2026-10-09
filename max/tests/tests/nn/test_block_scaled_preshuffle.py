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
"""Pins the in-graph AMD preb permutations to the layouts the kernel reads.

A disagreement between these permutations and
``max/kernels/src/linalg/matmul/gpu/amd/block_scaled_preshuffle_layouts.mojo``
runs, reads fluently and scores zero, so each layout is checked against an
independent reference rather than against itself. CPU only: the graphs are
pure reshapes and transposes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.nn.kernels import (
    MXFP6_LANE_BYTES,
    preshuffle_block_scaled_b,
    preshuffle_block_scaled_b_scales,
)

_MFMA_MN_LANES = 16
_MFMA_K_LANES = 4
_MFMA_LANE_BYTES = 16


@pytest.fixture(scope="module")
def session() -> InferenceSession:
    return InferenceSession(devices=[CPU()])


def _bytes(seed: int, shape: tuple[int, ...]) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


def _run(
    session: InferenceSession,
    permute: Callable[[TensorValue], TensorValue],
    src: np.ndarray,
) -> np.ndarray:
    cpu = DeviceRef.CPU()
    with Graph(
        "preshuffle", input_types=[TensorType(DType.uint8, src.shape, cpu)]
    ) as graph:
        graph.output(permute(graph.inputs[0].tensor))
    (out,) = session.load(graph).execute(Buffer.from_numpy(src))
    return cast(Buffer, out).to_numpy()


def _expected_b_5d(src: np.ndarray) -> np.ndarray:
    n, k_bytes = src.shape
    return (
        src.reshape(n // 16, 16, k_bytes // 64, 4, 16)
        .transpose(0, 2, 3, 1, 4)
        .reshape(n, k_bytes)
    )


def _expected_scale_4d(src: np.ndarray) -> np.ndarray:
    mn, k_scales = src.shape
    return (
        src.reshape(mn // 32, 2, 16, k_scales // 8, 2, 4)
        .transpose(0, 3, 5, 2, 4, 1)
        .reshape(mn, k_scales)
    )


def _b_plane_byte_off(n: int, k_byte: int, plane: int, *, k_bytes: int) -> int:
    """Scalar transcription of ``Shuffler.b_plane_byte_off`` for one expert."""
    lane_bytes = MXFP6_LANE_BYTES
    mfma_k_bytes = _MFMA_K_LANES * lane_bytes
    tile_bytes = _MFMA_MN_LANES * mfma_k_bytes
    plane_width = min(_MFMA_LANE_BYTES, lane_bytes - plane * _MFMA_LANE_BYTES)
    plane_base = (
        _MFMA_MN_LANES
        * _MFMA_K_LANES
        * (lane_bytes - max(0, lane_bytes - plane * _MFMA_LANE_BYTES))
    )
    k0_count = k_bytes // mfma_k_bytes

    n0, nlane = divmod(n, _MFMA_MN_LANES)
    k0, k_in_tile = divmod(k_byte, mfma_k_bytes)
    klane = k_in_tile // lane_bytes

    return (
        n0 * (k0_count * tile_bytes)
        + k0 * tile_bytes
        + plane_base
        + klane * (_MFMA_MN_LANES * plane_width)
        + nlane * plane_width
    )


def _expected_planes(src: np.ndarray) -> np.ndarray:
    """Places ``[N, K_BYTES]`` one lane fragment at a time, via the formula."""
    n_rows, k_bytes = src.shape
    dst = np.zeros(n_rows * k_bytes, dtype=np.uint8)
    for n in range(n_rows):
        for k_byte in range(0, k_bytes, MXFP6_LANE_BYTES):
            for plane in range(2):
                width = min(
                    _MFMA_LANE_BYTES,
                    MXFP6_LANE_BYTES - plane * _MFMA_LANE_BYTES,
                )
                offset = _b_plane_byte_off(n, k_byte, plane, k_bytes=k_bytes)
                start = k_byte + plane * _MFMA_LANE_BYTES
                dst[offset : offset + width] = src[n, start : start + width]
    return dst.reshape(n_rows, k_bytes)


def test_stacked_gate_up_matches_per_half_layout(
    session: InferenceSession,
) -> None:
    """Permuting the stacked ``[E, 2D, K]`` gate/up equals permuting each half.

    The graph stacks gate and up on N before permuting once; the layout keeps
    16-row ``N0`` tiles whole, so the two halves must land exactly where
    permuting them separately would put them.
    """
    experts, d, k_bytes = 2, 32, 128
    src = _bytes(0, (experts, 2 * d, k_bytes))
    got = _run(session, preshuffle_block_scaled_b, src)
    for e in range(experts):
        for half in (slice(0, d), slice(d, 2 * d)):
            np.testing.assert_array_equal(
                got[e, half], _expected_b_5d(src[e, half])
            )


def test_scales_match_cell_layout(session: InferenceSession) -> None:
    src = _bytes(1, (2, 64, 16))
    got = _run(session, preshuffle_block_scaled_b_scales, src)
    for e in range(src.shape[0]):
        np.testing.assert_array_equal(got[e], _expected_scale_4d(src[e]))


# K_BYTES must be a whole number of MFMA K tiles: 4 lanes * 24 bytes = 96.
@pytest.mark.parametrize(("n_rows", "k_bytes"), [(16, 384), (48, 288)])
def test_mxfp6_planes_match_kernel_offsets(
    session: InferenceSession, n_rows: int, k_bytes: int
) -> None:
    """MXFP6 matches the kernel's plane offsets, and plane 0 is the FP4 layout.

    A 16-byte plane is exactly one FP4 lane fragment, so plane 0 of each tile
    must equal the FP4 layout of the fragments truncated to 16 bytes; that
    catches a plane-ordering mistake the offset formula alone could share.
    """
    src = _bytes(n_rows * k_bytes, (1, n_rows, k_bytes))
    got = _run(
        session,
        lambda w: preshuffle_block_scaled_b(w, lane_bytes=MXFP6_LANE_BYTES),
        src,
    )[0]
    np.testing.assert_array_equal(got, _expected_planes(src[0]))

    truncated = src[0].reshape(n_rows, -1, MXFP6_LANE_BYTES)[
        ..., :_MFMA_LANE_BYTES
    ]
    tile_bytes = _MFMA_MN_LANES * _MFMA_K_LANES * MXFP6_LANE_BYTES
    plane0_bytes = _MFMA_MN_LANES * _MFMA_K_LANES * _MFMA_LANE_BYTES
    np.testing.assert_array_equal(
        got.reshape(-1, tile_bytes)[:, :plane0_bytes].reshape(-1),
        _expected_b_5d(truncated.reshape(n_rows, -1)).reshape(-1),
    )


@pytest.mark.parametrize(
    ("permute", "shape", "match"),
    [
        (preshuffle_block_scaled_b, [2, 24, 128], "N % 16"),
        (preshuffle_block_scaled_b, [2, 32, 96], "K_BYTES % 64"),
        (preshuffle_block_scaled_b_scales, [2, 48, 16], "MN % 32"),
        (preshuffle_block_scaled_b_scales, [2, 64, 12], "K_SCALES % 8"),
    ],
)
def test_unaligned_shapes_are_refused(
    permute: Callable[[TensorValue], TensorValue],
    shape: list[int],
    match: str,
) -> None:
    """A shape the kernel cannot address fails at graph build, not as noise."""
    with Graph(
        "unaligned",
        input_types=[TensorType(DType.uint8, shape, DeviceRef.CPU())],
    ) as graph:
        with pytest.raises(ValueError, match=match):
            permute(graph.inputs[0].tensor)
