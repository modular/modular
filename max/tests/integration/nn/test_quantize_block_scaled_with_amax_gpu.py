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
"""Tests the ``quantize_block_scaled_with_amax`` graph op.

The reference clips each block of the input to its given amax and runs
``quantize_dynamic_block_scaled``, whose own block abs-max is then that
amax. Both paths encode the same scales and saturate the same values, so
their bytes must match.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer, accelerator_api, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn._graph_ops import (
    quantize_block_scaled_with_amax,
    quantize_dynamic_block_scaled,
)
from test_common.graph_utils import is_b100_b200

K = 512
BLOCK = 16


def _bytes(buf: Buffer) -> torch.Tensor:
    return torch.from_dlpack(buf.view(DType.uint8)).cpu()


def _unsigned_zeros(packed: torch.Tensor) -> torch.Tensor:
    """``packed`` with each -0 E2M1 nibble (``0x8``) written as +0.

    A zero-amax block scales every value to a zero that keeps its sign,
    while the clipped reference holds only -0.
    """
    lo, hi = packed & 0xF, packed >> 4
    lo = torch.where(lo == 0x8, 0, lo)
    hi = torch.where(hi == 0x8, 0, hi)
    return lo | (hi << 4)


def test_given_amax_matches_clipping_the_input() -> None:
    if accelerator_count() == 0 or accelerator_api() == "hip":
        pytest.skip("needs an NVIDIA GPU")
    if not is_b100_b200():
        pytest.skip("the given-amax quantize runs on SM100")

    device = Accelerator()
    device_ref = DeviceRef(device.label, device.id)
    session = InferenceSession(devices=[device])
    tensor_sf = 448.0 * 6.0 / 4.0

    with Graph(
        "quantize_block_scaled_with_amax",
        input_types=[
            TensorType(DType.bfloat16, ("M", K), device_ref),
            TensorType(DType.float32, ("M", K // BLOCK), device_ref),
            TensorType(DType.bfloat16, ("M", K), device_ref),
        ],
    ) as graph:
        x, block_amax, clipped = (inp.tensor for inp in graph.inputs)
        q, scales = quantize_block_scaled_with_amax(
            x, block_amax, tensor_sf=tensor_sf
        )
        ref_q, ref_scales = quantize_dynamic_block_scaled(
            clipped, tensor_sf=tensor_sf
        )
        graph.output(q, scales, ref_q, ref_scales)
    model = session.load(graph)

    rng = np.random.default_rng(0)
    for m in [1, 128, 129]:
        x_t = torch.from_numpy(
            rng.standard_normal((m, K)).astype(np.float32)
        ).to(torch.bfloat16)
        blocks = x_t.float().reshape(m, K // BLOCK, BLOCK)
        own_amax = blocks.abs().amax(-1)
        # 0 kills a block, 1 keeps its own amax, the rest clip it.
        clip = torch.from_numpy(
            rng.choice([0.0, 0.5, 0.7, 1.0], own_amax.shape).astype(np.float32)
        )
        # The kernel reads the bf16 clip values as their abs-max.
        amax = (own_amax * clip).to(torch.bfloat16).float()
        clipped = torch.maximum(
            torch.minimum(blocks, amax[..., None]), -amax[..., None]
        ).reshape(m, K)

        outs = model.execute(
            Buffer.from_dlpack(x_t).to(device),
            Buffer.from_dlpack(amax.contiguous()).to(device),
            Buffer.from_dlpack(clipped.to(torch.bfloat16)).to(device),
        )
        q_b, s_b, ref_q_b, ref_s_b = (_bytes(b) for b in outs)
        assert q_b.shape == ref_q_b.shape
        assert torch.equal(_unsigned_zeros(q_b), _unsigned_zeros(ref_q_b)), (
            f"M={m}: quantized bytes"
        )
        assert torch.equal(s_b, ref_s_b), f"M={m}: scale bytes"
