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
"""Tests the ``rope_quantize_dynamic_block_scaled`` graph op.

One graph runs the fused op and, as the reference, ``rope_ragged`` with one
ragged sequence (token ``i`` reads table row ``i``) followed by
``quantize_dynamic_block_scaled``. An inverse rotation is a forward one by
the conjugated table. Both paths do the same fp32 arithmetic, so their
bytes must match. This checks the op's parameters, its rank-3 table reshape
and its output shapes; the kernel test covers the edge values.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer, accelerator_api, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kernels import (
    quantize_dynamic_block_scaled,
    rope_quantize_dynamic_block_scaled,
    rope_ragged,
)
from test_common.graph_utils import is_b100_b200

HEADS = 8
HEAD_DIM = 512
ROPE_DIM = 64

_FORMATS = {
    "mxfp8": (32, DType.float8_e8m0fnu, DType.float8_e4m3fn),
    "nvfp4": (16, DType.float8_e4m3fn, DType.uint8),
}


def _bytes(buf: Buffer) -> torch.Tensor:
    return torch.from_dlpack(buf.view(DType.uint8)).cpu()


@pytest.mark.parametrize("fmt", sorted(_FORMATS))
@pytest.mark.parametrize("inverse", [False, True])
def test_rope_quantize_matches_rope_then_quantize(
    fmt: str, inverse: bool
) -> None:
    if accelerator_count() == 0 or accelerator_api() == "hip":
        pytest.skip("needs an NVIDIA GPU")
    if not is_b100_b200():
        pytest.skip("the fused RoPE quantize runs on SM100")

    sf_vector_size, scales_type, out_type = _FORMATS[fmt]
    device = Accelerator()
    device_ref = DeviceRef(device.label, device.id)
    session = InferenceSession(devices=[device])

    with Graph(
        "rope_quantize_block_scaled",
        input_types=[
            TensorType(DType.bfloat16, ("M", HEADS * HEAD_DIM), device_ref),
            TensorType(DType.float32, ("M", ROPE_DIM // 2, 2), device_ref),
            TensorType(DType.float32, ("M", ROPE_DIM), device_ref),
            TensorType(DType.uint32, (2,), device_ref),
            TensorType(DType.uint32, (1,), device_ref),
        ],
    ) as graph:
        x, freqs, ref_freqs, row_offsets, start_pos = (
            inp.tensor for inp in graph.inputs
        )
        fused_q, fused_scales = rope_quantize_dynamic_block_scaled(
            x,
            freqs,
            HEAD_DIM,
            inverse=inverse,
            sf_vector_size=sf_vector_size,
            scales_type=scales_type,
            out_type=out_type,
        )
        rotated = rope_ragged(
            ops.reshape(x, [x.shape[0], HEADS, HEAD_DIM]),
            row_offsets,
            start_pos,
            ref_freqs,
            interleaved=True,
        )
        ref_q, ref_scales = quantize_dynamic_block_scaled(
            ops.reshape(rotated, x.shape),
            sf_vector_size=sf_vector_size,
            scales_type=scales_type,
            out_type=out_type,
        )
        graph.output(fused_q, fused_scales, ref_q, ref_scales)
    model = session.load(graph)

    rng = np.random.default_rng(0)
    for m in [1, 96, 129]:
        x_t = torch.from_numpy(
            rng.standard_normal((m, HEADS * HEAD_DIM)).astype(np.float32)
        ).to(torch.bfloat16)
        theta = rng.uniform(-np.pi, np.pi, (m, ROPE_DIM // 2))
        table = np.stack([np.cos(theta), np.sin(theta)], -1).astype(np.float32)
        ref_table = table.copy()
        if inverse:
            ref_table[..., 1] = -ref_table[..., 1]
        outs = model.execute(
            Buffer.from_dlpack(x_t).to(device),
            Buffer.from_numpy(table).to(device),
            Buffer.from_numpy(ref_table.reshape(m, ROPE_DIM)).to(device),
            Buffer.from_numpy(np.array([0, m], np.uint32)).to(device),
            Buffer.from_numpy(np.zeros(1, np.uint32)).to(device),
        )
        fused_q_b, fused_s_b, ref_q_b, ref_s_b = (_bytes(b) for b in outs)
        assert fused_q_b.shape == ref_q_b.shape
        assert torch.equal(fused_q_b, ref_q_b), f"M={m}: quantized bytes"
        assert torch.equal(fused_s_b, ref_s_b), f"M={m}: scale bytes"
