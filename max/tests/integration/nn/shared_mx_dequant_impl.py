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
"""Shared checks for the MXFP4 / MXFP8 dequant ops.

Run on CPU by ``test_mx_dequant.py`` and on a GPU by ``test_mx_dequant_gpu.py``.

The reference is independent of the ops under test: an E2M1 lookup table, a
from-the-definition E4M3 decode, and ``2 ** (code - 127)`` for the E8M0 scale.
The scale is a power of two, so ``element * scale`` is exact in float32 and
float32 output is expected to match bit for bit. Scale codes stay in a
moderate range so no product overflows or goes subnormal.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType
from max.nn.legacy.kernels import mxfp4_dequant, mxfp8_dequant
from shared_nvfp4_dequant_impl import (
    E2M1_TABLE,
    device_ref,
    e4m3_to_f32,
    interleave_scales,
    round_to_bfloat16,
    to_buffer,
    to_numpy,
)

Format = Literal["mxfp4", "mxfp8"]

BLOCK = 32
E8M0_BIAS = 127

# E8M0 scale codes: 2**-27 .. 2**23, with the extremes of that range first.
SCALE_CODE_RANGE = (100, 151)
EDGE_SCALE_CODES = [100, 127, 150, 126, 128]

# E4M3 element codes that probe the corners: zero, smallest subnormal, largest
# subnormal, smallest normal, largest finite value, and the negative of each.
# 0x7F / 0xFF are NaN and excluded.
EDGE_ELEMENT_CODES = [0x00, 0x01, 0x07, 0x08, 0x7E, 0x80, 0x81, 0x87, 0xFE]


# --------------------------------------------------------------------------- #
# Independent numpy reference
# --------------------------------------------------------------------------- #
def e8m0_to_f32(codes: np.ndarray) -> np.ndarray:
    """Decodes E8M0 codes (excluding the NaN code 0xFF) as 2 ** (code - 127)."""
    return np.exp2(codes.astype(np.float32) - E8M0_BIAS).astype(np.float32)


def dequant_reference(
    data: np.ndarray, scales: np.ndarray, fmt: Format
) -> np.ndarray:
    """``element * 2 ** (scale - 127)`` in float32, for any leading dims.

    ``data`` is ``[..., K // 2]`` packed bytes for MXFP4 or ``[..., K]``
    E4M3 codes for MXFP8; ``scales`` is ``[..., K // 32]`` E8M0 codes.
    """
    if fmt == "mxfp4":
        codes = np.stack([data & 0x0F, data >> 4], axis=-1)
        codes = codes.reshape(*data.shape[:-1], data.shape[-1] * 2)
        values = E2M1_TABLE[codes]
    else:
        values = e4m3_to_f32(data)
    blocks = values.reshape(
        *values.shape[:-1], values.shape[-1] // BLOCK, BLOCK
    )
    out = blocks * e8m0_to_f32(scales)[..., None]
    return out.reshape(values.shape).astype(np.float32)


# --------------------------------------------------------------------------- #
# Test data
# --------------------------------------------------------------------------- #
def make_case(
    rng: np.random.Generator, shape: tuple[int, ...], fmt: Format
) -> tuple[np.ndarray, np.ndarray]:
    """Random weights and valid E8M0 scale codes for ``[..., N, K]``."""
    *lead, k = shape
    assert k % BLOCK == 0
    if fmt == "mxfp4":
        # All 256 byte values: every pair of E2M1 codes.
        data = rng.integers(0, 256, (*lead, k // 2), dtype=np.uint8)
    else:
        magnitude = rng.integers(0, 0x7F, (*lead, k), dtype=np.uint8)
        negative = rng.random(magnitude.shape) < 0.5
        data = (magnitude | (negative.astype(np.uint8) << 7)).astype(np.uint8)
        edges = EDGE_ELEMENT_CODES[:k]
        data[..., 0, : len(edges)] = edges
    scales = rng.integers(
        *SCALE_CODE_RANGE, (*lead, k // BLOCK), dtype=np.uint8
    )
    edges = EDGE_SCALE_CODES[: scales.shape[-1]]
    scales[..., 0, : len(edges)] = edges
    return data, scales


# --------------------------------------------------------------------------- #
# Execution plumbing
# --------------------------------------------------------------------------- #
def data_type(fmt: Format) -> DType:
    return DType.uint8 if fmt == "mxfp4" else DType.float8_e4m3fn


def _data_buffer(data: np.ndarray, device: Device, fmt: Format) -> Buffer:
    return to_buffer(
        data, device, None if fmt == "mxfp4" else DType.float8_e4m3fn
    )


def run_dequant(
    device: Device,
    fmt: Format,
    data: np.ndarray,
    scales: np.ndarray,
    out_dtype: DType,
) -> np.ndarray:
    """Executes the dequant op for ``fmt`` on ``device``; returns float32."""
    ref = device_ref(device)
    op = mxfp4_dequant if fmt == "mxfp4" else mxfp8_dequant
    with Graph(
        f"{fmt}_dequant",
        input_types=[
            TensorType(data_type(fmt), data.shape, device=ref),
            TensorType(DType.float8_e8m0fnu, scales.shape, device=ref),
        ],
    ) as graph:
        d, s = (value.tensor for value in graph.inputs)
        graph.output(op(d, s, out_type=out_dtype))
    model = InferenceSession(devices=[device]).load(graph)
    (result,) = model.execute(
        _data_buffer(data, device, fmt),
        to_buffer(scales, device, DType.float8_e8m0fnu),
    )
    return to_numpy(result)


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def check_dequant(
    device: Device,
    fmt: Format,
    shape: tuple[int, ...],
    out_dtype: DType,
    *,
    interleaved: bool = False,
    seed: int = 0,
) -> None:
    """The op matches the reference exactly, for rank-2 or rank-3 weights."""
    rng = np.random.default_rng(seed)
    data, scales = make_case(rng, shape, fmt)
    expected = dequant_reference(data, scales, fmt)
    if out_dtype == DType.bfloat16:
        expected = round_to_bfloat16(expected)

    op_scales = interleave_scales(scales) if interleaved else scales
    got = run_dequant(device, fmt, data, op_scales, out_dtype)

    assert got.shape == expected.shape
    # Exact: the product with a power of two is exact in float32, and the
    # bfloat16 cast is the same round-to-nearest-even on both sides.
    np.testing.assert_array_equal(got, expected)
