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

"""Graph-level W4A8 grouped matmul on AMD against a NumPy dequantized dot.

The Mojo kernel tests call the kernel entry point directly, which skips the
Python operand validation and the custom op's dtype-to-format dispatch. This
builds ``grouped_dynamic_block_scaled_matmul_amd`` with float8_e4m3fn
activations and packed uint8 MXFP4 weights and runs it, so a wrong K extent,
operand order, or format selection in either layer shows up as wrong values.

NOTE: AMD CDNA4-only (the mixed E4M3 x E2M1 MFMA targets MI355).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
import torch
from max.driver import Accelerator, Buffer, accelerator_api
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn.kernels import grouped_dynamic_block_scaled_matmul_amd

_E2M1_VALUES = np.array(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=np.float64,
)


def _e8m0(scales: npt.NDArray[np.uint8]) -> npt.NDArray[np.float64]:
    """Expands one E8M0 scale per 32 elements along the last axis."""
    return np.exp2(scales.astype(np.float64) - 127).repeat(32, axis=-1)


def _buffer(array: npt.NDArray[np.generic]) -> Buffer:
    return Buffer.from_dlpack(torch.from_numpy(array))


@pytest.mark.skipif(
    accelerator_api() != "hip",
    reason="mixed E4M3 x E2M1 block-scaled MFMA is AMD CDNA4-only",
)
@pytest.mark.parametrize(
    "tokens_per_expert, expert_ids, n, k, out_type",
    [
        # Ragged groups with an empty one and a permuted expert order, so a
        # group reading another expert's weights shows up as wrong values.
        ([5, 0, 37], [2, 0, 1], 128, 384, DType.float32),
        ([1, 64, 65], [1, 2, 0], 256, 512, DType.bfloat16),
    ],
)
def test_w4a8_grouped_matmul_amd(
    tokens_per_expert: list[int],
    expert_ids: list[int],
    n: int,
    k: int,
    out_type: DType,
) -> None:
    num_experts = len(tokens_per_expert)
    total_tokens = sum(tokens_per_expert)
    rng = np.random.default_rng(13)

    a_bytes = rng.integers(0, 256, size=(total_tokens, k), dtype=np.uint8)
    # 0x7F and 0xFF are the E4M3 NaN encodings.
    a_bytes[(a_bytes & 0x7F) == 0x7F] = 0x38
    b_bytes = rng.integers(
        0, 256, size=(num_experts, n, k // 2), dtype=np.uint8
    )
    a_scales = rng.integers(125, 130, size=(total_tokens, k // 32)).astype(
        np.uint8
    )
    b_scales = rng.integers(125, 130, size=(num_experts, n, k // 32)).astype(
        np.uint8
    )
    offsets = np.zeros(num_experts + 1, dtype=np.uint32)
    offsets[1:] = np.cumsum(tokens_per_expert)
    ids = np.array(expert_ids, dtype=np.int32)
    usage = np.array([max(tokens_per_expert), num_experts], dtype=np.uint32)

    a = (
        torch.from_numpy(a_bytes)
        .view(torch.float8_e4m3fn)
        .to(torch.float64)
        .numpy()
    ) * _e8m0(a_scales)
    b_values = np.empty((num_experts, n, k), dtype=np.float64)
    # E2M1 packs the even K element in the low nibble.
    b_values[..., 0::2] = _E2M1_VALUES[b_bytes & 0xF]
    b_values[..., 1::2] = _E2M1_VALUES[b_bytes >> 4]
    b = b_values * _e8m0(b_scales)
    expected = np.zeros((total_tokens, n), dtype=np.float64)
    magnitude = np.zeros((total_tokens, n), dtype=np.float64)
    for slot, expert in enumerate(expert_ids):
        rows = slice(offsets[slot], offsets[slot + 1])
        expected[rows] = a[rows] @ b[expert].T
        magnitude[rows] = np.abs(a[rows]) @ np.abs(b[expert]).T

    device = Accelerator()
    device_ref = DeviceRef(device.label, device.id)
    input_types = [
        TensorType(DType.float8_e4m3fn, ("tokens", k), device=device_ref),
        TensorType(DType.uint8, (num_experts, n, k // 2), device=device_ref),
        TensorType(
            DType.float8_e8m0fnu, ("tokens", k // 32), device=device_ref
        ),
        TensorType(
            DType.float8_e8m0fnu, (num_experts, n, k // 32), device=device_ref
        ),
        TensorType(DType.uint32, (num_experts + 1,), device=device_ref),
        TensorType(DType.int32, (num_experts,), device=device_ref),
        TensorType(DType.uint32, (2,), device=DeviceRef.CPU()),
    ]
    with Graph("w4a8_grouped_matmul_amd", input_types=input_types) as graph:
        hidden_states, weights, a_scale_t, b_scale_t, starts, ids_t, stats = (
            value.tensor for value in graph.inputs
        )
        out = grouped_dynamic_block_scaled_matmul_amd(
            hidden_states,
            weights,
            a_scale_t,
            b_scale_t,
            starts,
            ids_t,
            stats,
            out_type=out_type,
        )
        # numpy from_dlpack can't read bfloat16.
        graph.output(out.cast(DType.float32))

    compiled = InferenceSession(devices=[device]).load(graph)
    (result,) = compiled.execute(
        _buffer(a_bytes).view(DType.float8_e4m3fn).to(device),
        _buffer(b_bytes).to(device),
        _buffer(a_scales).view(DType.float8_e8m0fnu).to(device),
        _buffer(b_scales).view(DType.float8_e8m0fnu).to(device),
        _buffer(offsets).to(device),
        _buffer(ids).to(device),
        _buffer(usage),
    )
    actual = np.from_dlpack(result.to(Accelerator.cpu())).astype(np.float64)

    # FP32 accumulation order differs from the reference; bfloat16 output adds
    # one rounding of the result.
    tolerance = 6e-5 * magnitude + 1e-5
    if out_type == DType.bfloat16:
        tolerance += 0.004 * np.abs(expected)
    error = np.abs(actual - expected)
    assert np.all(error <= tolerance), (
        f"max error {error.max()} exceeds tolerance at"
        f" {np.unravel_index(np.argmax(error - tolerance), error.shape)}"
    )
