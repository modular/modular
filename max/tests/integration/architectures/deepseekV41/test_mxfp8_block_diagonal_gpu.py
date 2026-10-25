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
"""Tests ``mxfp8_block_diagonal`` against the production block-diagonal
matmul at the ``attn.wo_a`` shape.

At decode sizes the cluster split-K matmul adds each output's K slices in
another order, so the outputs match to a bf16 rounding, not bit for bit.
A planted fault, the reference with its groups' output columns rotated by
one group, must fail the same comparison.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn.kernels import (
    block_diagonal_block_scaled_matmul,
    quantize_dynamic_block_scaled,
)
from max.pipelines.architectures.deepseekV41.layers.mxfp8_matmul import (
    mxfp8_block_diagonal,
)
from max.pipelines.architectures.deepseekV41.model_config import (
    MXFP8_SPLITK_ENV,
)

GROUPS = 8
N = 1024
K = 4096


def _close(got: torch.Tensor, want: torch.Tensor) -> bool:
    return torch.allclose(got.float(), want.float(), rtol=1e-2, atol=1e-2)


def test_block_diagonal_split_k_matches_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MXFP8_SPLITK_ENV, "1")
    device = Accelerator()
    device_ref = DeviceRef(device.label, device.id)
    session = InferenceSession(devices=[device])

    with Graph(
        "mxfp8_block_diagonal",
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
        x_q, x_scales = quantize_dynamic_block_scaled(
            x,
            sf_vector_size=32,
            scales_type=DType.float8_e8m0fnu,
            out_type=DType.float8_e4m3fn,
        )
        graph.output(
            mxfp8_block_diagonal(x_q, w, x_scales, w_scales, GROUPS),
            block_diagonal_block_scaled_matmul(
                x_q, w, x_scales, w_scales, GROUPS
            ),
        )
    assert "mo.matmul.mxfp8.cluster_splitk.block_diagonal" in str(graph)
    model = session.load(graph)

    rng = np.random.default_rng(0)
    w_t = torch.from_numpy(
        (rng.standard_normal((GROUPS * N, K)) * 0.5).astype(np.float32)
    ).to(torch.float8_e4m3fn)
    # Random exponents near 2^0 in every scale slot, so each group reads its
    # own scale tiles.
    ws_t = torch.from_numpy(
        rng.integers(124, 131, (GROUPS * N // 128, K // 128, 32, 4, 4)).astype(
            np.uint8
        )
    )
    w_buf = Buffer.from_dlpack(w_t.view(torch.uint8)).to(device)
    w_buf = w_buf.view(DType.float8_e4m3fn)
    ws_buf = Buffer.from_dlpack(ws_t).to(device).view(DType.float8_e8m0fnu)
    # The split-K token tiles (BN 8, 16, 32) at full and ragged sizes, and a
    # size past them that takes the production matmul.
    for m in [1, 7, 16, 31, 97]:
        x_t = torch.from_numpy(
            rng.standard_normal((m, GROUPS * K)).astype(np.float32)
        ).to(torch.bfloat16)
        got_buf, want_buf = model.execute(
            Buffer.from_dlpack(x_t).to(device), w_buf, ws_buf
        )
        got = torch.from_dlpack(got_buf).cpu()
        want = torch.from_dlpack(want_buf).cpu()
        assert got.shape == (m, GROUPS * N)
        assert got.float().abs().sum() > 0, f"M={m}: all-zero output"
        assert _close(got, want), (
            f"M={m}: split-K block-diagonal output differs from production,"
            f" max |diff| {(got.float() - want.float()).abs().max()}"
        )
        assert not _close(got, torch.roll(want, N, dims=1)), (
            f"M={m}: the comparison does not catch a group shift"
        )
