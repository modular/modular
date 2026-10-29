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
"""Shared checks for the NVFP4 dequant op.

Run on CPU by ``test_nvfp4_dequant.py`` and on a GPU by
``test_nvfp4_dequant_gpu.py``.

The reference here is deliberately independent of both the op under test and
``max.pipelines.weights.fp4_quantization``: a lookup table for E2M1 and a
from-the-definition E4M3 decode. For float32 output the op is expected to
match it bit for bit, because ``e2m1 * |scale|`` is exact in float32 and only
the final multiply by ``tensor_sf`` rounds.
"""

from __future__ import annotations

import numpy as np
import torch
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn.legacy.kernels import nvfp4_dequant

BLOCK = 16

# E2M1 code (sign, 2 exponent bits, 1 mantissa bit) -> value.
E2M1_TABLE = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    + [-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)

# Scale codes that probe the corners of E4M3: zero, the smallest and largest
# subnormal, the smallest normal, the largest finite value, and negative
# codes (the op takes abs of the scale). 0x7F / 0xFF are NaN and excluded.
EDGE_SCALE_CODES = [0x00, 0x01, 0x07, 0x08, 0x7E, 0x80, 0x81, 0xFE]


# --------------------------------------------------------------------------- #
# Independent numpy reference
# --------------------------------------------------------------------------- #
def e4m3_to_f32(codes: np.ndarray) -> np.ndarray:
    """Decodes finite E4M3 codes from the definition (bias 7, 3 mantissa)."""
    c = codes.astype(np.int32)
    sign = np.where(c & 0x80, -1.0, 1.0).astype(np.float32)
    exp = (c >> 3) & 0xF
    mant = (c & 0x7).astype(np.float32)
    normal = (1.0 + mant / 8.0) * np.exp2((exp - 7).astype(np.float32))
    subnormal = mant * np.float32(2.0**-9)  # exp == 0: mant * 2^(1 - 7 - 3)
    return (sign * np.where(exp == 0, subnormal, normal)).astype(np.float32)


def dequant_reference(
    packed: np.ndarray, scales: np.ndarray, tensor_sf: float
) -> np.ndarray:
    """``(e2m1 * |scale|) * tensor_sf`` in float32, for any leading dims.

    ``packed`` is ``[..., K // 2]`` and ``scales`` (E4M3 codes) is
    ``[..., K // 16]``.
    """
    codes = np.stack([packed & 0x0F, packed >> 4], axis=-1)
    codes = codes.reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    values = E2M1_TABLE[codes]
    blocks = values.reshape(*codes.shape[:-1], codes.shape[-1] // BLOCK, BLOCK)
    block_scale = np.abs(e4m3_to_f32(scales))[..., None]
    out = (blocks * block_scale) * np.float32(tensor_sf)
    return out.reshape(codes.shape).astype(np.float32)


def interleave_scales(scales: np.ndarray) -> np.ndarray:
    """``[N, K // 16]`` -> the SM100 ``[ceil(N/128), ceil(K/64), 32, 4, 4]``.

    Written out from the layout's definition (a 128-row by 4-block atom, rows
    split into a 32-wide and a 4-wide index), not from the kernel's helper.
    Padding stays zero.
    """
    n, blocks = scales.shape
    out = np.zeros((-(-n // 128), -(-blocks // 4), 32, 4, 4), dtype=np.uint8)
    r = np.arange(n)[:, None]
    b = np.arange(blocks)[None, :]
    out[r // 128, b // 4, r % 32, (r % 128) // 32, b % 4] = scales
    return out


# --------------------------------------------------------------------------- #
# Test data
# --------------------------------------------------------------------------- #
def make_case(
    rng: np.random.Generator, shape: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Random packed weights and valid E4M3 scale codes for ``[..., N, K]``.

    Packed bytes cover all 256 values. Scale codes are every finite magnitude
    with a random sign, and the first row carries the edge codes.
    """
    *lead, k = shape
    assert k % BLOCK == 0
    packed = rng.integers(0, 256, (*lead, k // 2), dtype=np.uint8)
    magnitude = rng.integers(0, 0x7F, (*lead, k // BLOCK), dtype=np.uint8)
    negative = rng.random(magnitude.shape) < 0.2
    scales = magnitude | (negative.astype(np.uint8) << 7)
    edges = EDGE_SCALE_CODES[: scales.shape[-1]]
    scales[..., 0, : len(edges)] = edges
    return packed, scales


# --------------------------------------------------------------------------- #
# Execution plumbing
# --------------------------------------------------------------------------- #
def device_ref(device: Device) -> DeviceRef:
    return DeviceRef(device.label, device.id)


def to_buffer(
    array: np.ndarray, device: Device, dtype: DType | None = None
) -> Buffer:
    buffer = Buffer.from_dlpack(torch.from_numpy(np.ascontiguousarray(array)))
    if device.label != "cpu":
        buffer = buffer.to(device)
    return buffer.view(dtype) if dtype is not None else buffer


def to_numpy(buffer: Buffer) -> np.ndarray:
    return torch.from_dlpack(buffer).cpu().float().numpy()


def round_to_bfloat16(x: np.ndarray) -> np.ndarray:
    return torch.from_numpy(x).to(torch.bfloat16).float().numpy()


def run_dequant(
    device: Device,
    packed: np.ndarray,
    scales: np.ndarray,
    tensor_sf: float,
    out_dtype: DType,
) -> np.ndarray:
    """Executes ``nvfp4_dequant`` on ``device``; returns float32 values."""
    ref = device_ref(device)
    with Graph(
        "nvfp4_dequant",
        input_types=[
            TensorType(DType.uint8, packed.shape, device=ref),
            TensorType(DType.float8_e4m3fn, scales.shape, device=ref),
        ],
    ) as graph:
        p, s = (value.tensor for value in graph.inputs)
        graph.output(nvfp4_dequant(p, s, tensor_sf, out_type=out_dtype))
    model = InferenceSession(devices=[device]).load(graph)
    (result,) = model.execute(
        to_buffer(packed, device),
        to_buffer(scales, device, DType.float8_e4m3fn),
    )
    return to_numpy(result)


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def check_dequant(
    device: Device,
    shape: tuple[int, ...],
    out_dtype: DType,
    tensor_sf: float,
    *,
    interleaved: bool = False,
    seed: int = 0,
) -> None:
    """The op matches the reference exactly, for rank-2 or rank-3 weights."""
    rng = np.random.default_rng(seed)
    packed, scales = make_case(rng, shape)
    expected = dequant_reference(packed, scales, tensor_sf)
    if out_dtype == DType.bfloat16:
        expected = round_to_bfloat16(expected)

    op_scales = interleave_scales(scales) if interleaved else scales
    got = run_dequant(device, packed, op_scales, tensor_sf, out_dtype)

    assert got.shape == expected.shape
    # Exact: the same IEEE operations in the same order, then the same
    # round-to-nearest-even cast for bfloat16.
    np.testing.assert_array_equal(got, expected)
