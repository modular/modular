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
"""Tests ``block_diagonal_block_scaled_matmul`` against per-block matmuls.

One graph quantizes ``x [t, groups * k]`` once and runs the block-diagonal
op. The reference quantizes each block's columns on its own and runs the
dense ``dynamic_block_scaled_matmul`` with that block's weight rows and
scale tiles. The quantize blocks of 32 never straddle two groups and both
paths accumulate K in the same order, so the outputs must match bit for bit.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer, accelerator_api, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.nn.kernels import (
    block_diagonal_block_scaled_matmul,
    dynamic_block_scaled_matmul,
    quantize_dynamic_block_scaled,
)
from test_common.graph_utils import is_b100_b200

GROUPS = 8
N = 256
K = 512


def _quantize(x: TensorValue) -> tuple[TensorValue, TensorValue]:
    return quantize_dynamic_block_scaled(
        x,
        sf_vector_size=32,
        scales_type=DType.float8_e8m0fnu,
        out_type=DType.float8_e4m3fn,
    )


def test_block_diagonal_matches_per_block() -> None:
    if accelerator_count() == 0 or accelerator_api() == "hip":
        pytest.skip("needs an NVIDIA GPU")
    if not is_b100_b200():
        pytest.skip("MXFP8 block-scaled MMA requires SM100")

    device = Accelerator()
    device_ref = DeviceRef(device.label, device.id)
    session = InferenceSession(devices=[device])

    with Graph(
        "block_diagonal_block_scaled_matmul",
        input_types=[
            TensorType(DType.bfloat16, ("M", GROUPS * K), device=device_ref),
            TensorType(DType.float8_e4m3fn, (GROUPS * N, K), device=device_ref),
            TensorType(
                DType.float8_e8m0fnu,
                (GROUPS * N // 128, K // 128, 32, 4, 4),
                device=device_ref,
            ),
        ],
    ) as graph:
        x, w, w_scales = (inp.tensor for inp in graph.inputs)
        x_q, x_scales = _quantize(x)
        fused = block_diagonal_block_scaled_matmul(
            x_q, w, x_scales, w_scales, GROUPS
        )
        tiles = N // 128
        per_block = []
        for g in range(GROUPS):
            xg_q, xg_scales = _quantize(x[:, g * K : (g + 1) * K])
            per_block.append(
                dynamic_block_scaled_matmul(
                    xg_q,
                    w[g * N : (g + 1) * N, :],
                    xg_scales,
                    w_scales[g * tiles : (g + 1) * tiles],
                    sf_vector_size=32,
                    out_type=DType.bfloat16,
                )
            )
        graph.output(fused, *per_block)
    model = session.load(graph)

    rng = np.random.default_rng(0)
    w_t = torch.from_numpy(
        (rng.standard_normal((GROUPS * N, K)) * 0.5).astype(np.float32)
    ).to(torch.float8_e4m3fn)
    # Random exponents near 2^0 in every scale slot, so each block reads its
    # own scale tiles.
    ws_t = torch.from_numpy(
        rng.integers(124, 131, (GROUPS * N // 128, K // 128, 32, 4, 4)).astype(
            np.uint8
        )
    )
    w_buf = Buffer.from_dlpack(w_t.view(torch.uint8)).to(device)
    w_buf = w_buf.view(DType.float8_e4m3fn)
    ws_buf = Buffer.from_dlpack(ws_t).to(device).view(DType.float8_e8m0fnu)
    for m in [0, 1, 7, 96, 129, 200]:
        x_t = torch.from_numpy(
            rng.standard_normal((m, GROUPS * K)).astype(np.float32)
        ).to(torch.bfloat16)
        got_buf, *block_bufs = model.execute(
            Buffer.from_dlpack(x_t).to(device), w_buf, ws_buf
        )
        got = torch.from_dlpack(got_buf).cpu()
        want = torch.cat([torch.from_dlpack(b).cpu() for b in block_bufs], 1)
        assert got.shape == (m, GROUPS * N)
        assert torch.equal(got.view(torch.int16), want.view(torch.int16)), (
            f"M={m}: block-diagonal output differs from the per-block matmuls"
        )
        if m:
            assert got.float().abs().sum() > 0, f"M={m}: all-zero output"
