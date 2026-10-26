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
"""Tests ``mo.matmul.mxfp8.rope_tail`` against the matmul then RoPE."""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kernels import (
    block_scales_interleave,
    quantize_dynamic_block_scaled,
)
from max.nn.quant_ops import _matmul_float8_mxfp8
from max.pipelines.architectures.deepseekV4.layers.rope import apply_rope_tail

HEAD_DIM = 512
ROPE_DIM = 64
K = 1280


def _bf16_bits(x: np.ndarray) -> np.ndarray:
    """Rounds float32 to bf16 (nearest even), as raw bits."""
    bits = x.astype(np.float32).view(np.uint32)
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


def _bf16_values(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _input_types(n: int, device: DeviceRef) -> list[TensorType]:
    return [
        TensorType(DType.bfloat16, ["m", K], device),
        TensorType(DType.float8_e4m3fn, [n, K], device),
        TensorType(DType.float8_e8m0fnu, [n, K // 32], device),
        TensorType(DType.float32, ["m", ROPE_DIM // 2, 2], device),
    ]


def _build_reference(n: int, device: DeviceRef) -> Graph:
    """The product, and the product followed by ``apply_rope_tail``."""
    with Graph(
        f"rope_tail_ref_{n}", input_types=_input_types(n, device)
    ) as graph:
        x, w, w_scale, freqs = (v.tensor for v in graph.inputs)
        plain = _matmul_float8_mxfp8(x, w, w_scale)
        heads = ops.reshape(plain, [1, x.shape[0], n // HEAD_DIM, HEAD_DIM])
        unfused = ops.reshape(
            apply_rope_tail(heads, freqs, ROPE_DIM), [x.shape[0], n]
        )
        graph.output(plain, unfused)
    return graph


def _build_fused(n: int, device: DeviceRef) -> Graph:
    with Graph(
        f"rope_tail_fused_{n}", input_types=_input_types(n, device)
    ) as graph:
        x, w, w_scale, freqs = (v.tensor for v in graph.inputs)
        x_fp8, x_scales = quantize_dynamic_block_scaled(
            x,
            sf_vector_size=32,
            scales_type=DType.float8_e8m0fnu,
            out_type=DType.float8_e4m3fn,
        )
        fused = ops.custom(
            "mo.matmul.mxfp8.rope_tail",
            device=device,
            values=[
                x_fp8,
                w,
                x_scales,
                block_scales_interleave(w_scale, sf_vector_size=32),
                ops.reshape(freqs, [x.shape[0], ROPE_DIM]),
            ],
            out_types=[TensorType(DType.bfloat16, [x.shape[0], n], device)],
            parameters={"head_dim": HEAD_DIM, "rope_dim": ROPE_DIM},
        )[0].tensor
        graph.output(fused)
    return graph


@pytest.fixture(scope="module")
def accel() -> Accelerator:
    return Accelerator()


@pytest.fixture(scope="module")
def models(accel: Accelerator) -> dict[int, tuple[Model, Model]]:
    """The reference and fused models at each N, compiled once for every M."""
    session = InferenceSession(devices=[accel])
    device = DeviceRef(accel.label, accel.id)
    return {
        n: (
            session.load(_build_reference(n, device)),
            session.load(_build_fused(n, device)),
        )
        for n in (64 * HEAD_DIM, 4 * HEAD_DIM)
    }


# Rows on both sides of the matmul's 512-row split between a separate RoPE
# pass and the fused epilogue, ragged against every tile size. 513 runs a
# config that swaps A and B, so its RoPE takes the separate pass too. N is
# wq_b's local width at 64 heads and a narrower 4-head one. 48 and 144 are
# decode at 8 and 24 requests of 6 rows; 577 is prime past the split.
@pytest.mark.parametrize("m", [1, 7, 48, 144, 512, 513, 577, 2053, 8168])
@pytest.mark.parametrize("n", [64 * HEAD_DIM, 4 * HEAD_DIM])
def test_rope_tail_matches_matmul_then_rope(
    models: dict[int, tuple[Model, Model]], accel: Accelerator, m: int, n: int
) -> None:
    rng = np.random.default_rng(m * 131 + n)
    x = _bf16_bits(rng.standard_normal((m, K)).astype(np.float32))
    # E4M3 bytes with the NaN encodings (0x7F, 0xFF) cleared.
    w = rng.integers(0, 256, (n, K), dtype=np.uint8)
    w[(w & 0x7F) == 0x7F] = 0x38
    scales = rng.integers(120, 128, (n, K // 32), dtype=np.uint8)
    angles = rng.uniform(-np.pi, np.pi, (m, ROPE_DIM // 2)).astype(np.float32)
    freqs = np.stack([np.cos(angles), np.sin(angles)], axis=-1).astype(
        np.float32
    )

    inputs = [
        Buffer.from_numpy(x).view(DType.bfloat16).to(accel),
        Buffer.from_numpy(w).view(DType.float8_e4m3fn).to(accel),
        Buffer.from_numpy(scales).view(DType.float8_e8m0fnu).to(accel),
        Buffer.from_numpy(freqs).to(accel),
    ]

    # Separate models, each read back before the next runs, so a fault is
    # reported against the model that raised it.
    def run(model: Model) -> list[np.ndarray]:
        outs = model.execute(*inputs)
        return [out.to(CPU()).view(DType.uint16).to_numpy() for out in outs]

    reference, fused_model = models[n]
    plain, unfused = run(reference)
    (fused,) = run(fused_model)

    np.testing.assert_array_equal(fused, unfused)

    plain_heads = plain.reshape(m, n // HEAD_DIM, HEAD_DIM)
    fused_heads = fused.reshape(m, n // HEAD_DIM, HEAD_DIM)
    nope = HEAD_DIM - ROPE_DIM
    np.testing.assert_array_equal(
        fused_heads[..., :nope], plain_heads[..., :nope]
    )

    # An independent rotation of the rounded product, so the check fails if
    # neither path rotates.
    tail = _bf16_values(plain_heads[..., nope:]).reshape(
        m, -1, ROPE_DIM // 2, 2
    )
    cos = freqs[:, None, :, 0]
    sin = freqs[:, None, :, 1]
    want = np.stack(
        [
            tail[..., 0] * cos - tail[..., 1] * sin,
            tail[..., 0] * sin + tail[..., 1] * cos,
        ],
        axis=-1,
    ).reshape(m, -1, ROPE_DIM)
    got = _bf16_values(fused_heads[..., nope:])
    np.testing.assert_allclose(
        got, want, rtol=2**-6, atol=1e-5 * float(np.abs(tail).max())
    )
    assert not np.array_equal(fused_heads[..., nope:], plain_heads[..., nope:])
