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
"""Tests ``swiglu_quantize_dynamic_block_scaled`` feeding an MXFP8 matmul.

The kernel tests check the fused op's bytes against the unfused kernels. This
test checks the graph op: its parameters, its symbolic output shapes and its
scale layout, by running its output through ``dynamic_block_scaled_matmul``
and comparing with an fp32 reference of ``activation(gate, up) @ w.T``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer, accelerator_api, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn.activation import GatedActivation
from max.nn.kernels import (
    dynamic_block_scaled_matmul,
    quantize_dynamic_block_scaled,
    swiglu_quantize_dynamic_block_scaled,
)
from test_common.graph_utils import is_b100_b200

K = 2304
N = 1024


def _gated_reference(
    gate_up: np.ndarray, activation: GatedActivation
) -> np.ndarray:
    g, u = np.split(gate_up, 2, axis=1)
    if activation.limit:
        g = np.minimum(g, activation.limit)
        u = np.clip(u, -activation.limit, activation.limit)
    if activation.fused_activation == "swigluoai":
        return g / (1 + np.exp(-activation.alpha * g)) * (u + 1)
    return g / (1 + np.exp(-g)) * u


# A limit of 1 clamps most inputs, so a dropped or misread clamp fails.
@pytest.mark.parametrize(
    "activation",
    [
        GatedActivation(),
        GatedActivation("silu", limit=1.0),
        GatedActivation("swigluoai", alpha=1.702, limit=1.0),
    ],
)
def test_swiglu_quantize_mxfp8_matmul(activation: GatedActivation) -> None:
    if accelerator_count() == 0 or accelerator_api() == "hip":
        pytest.skip("needs an NVIDIA GPU")
    if not is_b100_b200():
        pytest.skip("MXFP8 block-scaled MMA requires SM100")

    device = Accelerator()
    device_ref = DeviceRef(device.label, device.id)
    session = InferenceSession(devices=[device])

    with Graph(
        "swiglu_quantize_mxfp8_matmul",
        input_types=[
            TensorType(DType.bfloat16, shape=("M", 2 * K), device=device_ref),
            TensorType(DType.bfloat16, shape=(N, K), device=device_ref),
        ],
    ) as graph:
        gate_up, w = (inp.tensor for inp in graph.inputs)
        h, h_scales = swiglu_quantize_dynamic_block_scaled(
            gate_up,
            activation,
            sf_vector_size=32,
            scales_type=DType.float8_e8m0fnu,
            out_type=DType.float8_e4m3fn,
        )
        w_q, w_scales = quantize_dynamic_block_scaled(
            w,
            sf_vector_size=32,
            scales_type=DType.float8_e8m0fnu,
            out_type=DType.float8_e4m3fn,
        )
        graph.output(
            dynamic_block_scaled_matmul(
                h,
                w_q,
                h_scales,
                w_scales,
                sf_vector_size=32,
                out_type=DType.bfloat16,
            )
        )
    model = session.load(graph)

    rng = np.random.default_rng(0)
    w_t = torch.from_numpy(
        (rng.standard_normal((N, K)) * 0.1).astype(np.float32)
    ).to(torch.bfloat16)
    w_buf = Buffer.from_dlpack(w_t).to(device)
    w_np = w_t.to(torch.float32).numpy()
    for m in [1, 96, 129]:
        gate_up_t = torch.from_numpy(
            (rng.standard_normal((m, 2 * K)) * 2).astype(np.float32)
        ).to(torch.bfloat16)
        (out_buf,) = model.execute(
            Buffer.from_dlpack(gate_up_t).to(device), w_buf
        )
        out = torch.from_dlpack(out_buf).to(torch.float32).cpu().numpy()
        ref = (
            _gated_reference(gate_up_t.to(torch.float32).numpy(), activation)
            @ w_np.T
        )

        out_flat, ref_flat = out.reshape(-1), ref.reshape(-1)
        cos = float(
            np.dot(out_flat, ref_flat)
            / (np.linalg.norm(out_flat) * np.linalg.norm(ref_flat) + 1e-12)
        )
        rel_err = float(
            np.linalg.norm(out_flat - ref_flat)
            / (np.linalg.norm(ref_flat) + 1e-12)
        )
        assert cos > 0.99, f"M={m} {activation}: cosine {cos:.5f}"
        assert rel_err < 0.1, f"M={m} {activation}: rel L2 {rel_err:.5f}"


def test_swiglu_quantize_rejects_partial_scale_block() -> None:
    """The wrapper rejects halves that end in a partial scale block."""
    if accelerator_count() == 0 or accelerator_api() == "hip":
        pytest.skip("needs an NVIDIA GPU")
    if not is_b100_b200():
        pytest.skip("MXFP8 block-scaled MMA requires SM100")

    with Graph(
        "swiglu_quantize_partial_block",
        input_types=[
            TensorType(
                DType.bfloat16, shape=("M", 2 * 48), device=DeviceRef.GPU()
            )
        ],
    ) as graph:
        gate_up = graph.inputs[0].tensor
        with pytest.raises(ValueError, match="multiple of sf_vector_size"):
            swiglu_quantize_dynamic_block_scaled(
                gate_up,
                GatedActivation(),
                sf_vector_size=32,
                scales_type=DType.float8_e8m0fnu,
                out_type=DType.float8_e4m3fn,
            )
        graph.output(gate_up)
