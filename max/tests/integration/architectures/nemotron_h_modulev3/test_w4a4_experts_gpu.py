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
"""Checks the W4A4 routed-expert matmul against a dequantized reference."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kernels import moe_create_indices
from max.pipelines.architectures.nemotron_h_modulev3.layers.moe import (
    _nvfp4_expert_matmul,
)
from max.pipelines.architectures.nemotron_h_modulev3.weight_adapters import (
    interleave_nvfp4_scales,
)
from max.pipelines.weights._fp8 import e4m3fn_lut
from max.pipelines.weights.fp4_quantization import (
    FP4Format,
    e2m1_decode_table,
)

NUM_EXPERTS = 128
TOP_K = 6


def _dequantize(
    codes: npt.NDArray[np.uint8],
    scales: npt.NDArray[np.uint8],
    global_scale: float,
) -> npt.NDArray[np.float32]:
    table = e2m1_decode_table(FP4Format.NVFP4)
    values = np.stack([table[codes & 0xF], table[codes >> 4]], axis=-1)
    values = values.reshape(codes.shape[0], -1, 16)
    values = values * e4m3fn_lut()[scales][..., None] * global_scale
    return values.reshape(codes.shape[0], -1).astype(np.float32)


@pytest.mark.parametrize(
    "n, k, gather_in_kernel",
    [
        # The up projection; N is not a whole 128-row scale granule.
        (1856, 2688, True),
        # The down projection, on activations already in routed order.
        (2688, 1856, False),
    ],
)
@pytest.mark.parametrize("tokens", [1, 64, 700])
def test_w4a4_matches_dequantized_reference(
    n: int, k: int, gather_in_kernel: bool, tokens: int
) -> None:
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 256, (NUM_EXPERTS, n, k // 2), dtype=np.uint8)
    # E4M3 codes 0x28..0x48 span 2^-2..2^2.
    scales = rng.integers(0x28, 0x48, (NUM_EXPERTS, n, k // 16), dtype=np.uint8)
    global_scales = rng.uniform(0.005, 0.02, NUM_EXPERTS).astype(np.float32)
    experts = np.stack(
        [rng.choice(NUM_EXPERTS, TOP_K, replace=False) for _ in range(tokens)]
    ).astype(np.int32)
    x = rng.standard_normal((tokens, k)).astype(np.float32)
    # Rows whose magnitudes differ by orders of magnitude, and a few
    # outliers, exercise the per-row scale.
    x *= np.logspace(-2, 2, tokens, dtype=np.float32)[:, None]
    x[:, rng.integers(0, k, 4)] *= 20
    x_bf16 = torch.from_numpy(x).to(torch.bfloat16)

    device = Accelerator()
    dev = DeviceRef.GPU()
    with Graph(
        "w4a4_experts",
        input_types=[
            TensorType(DType.bfloat16, ["tokens", k], dev),
            TensorType(DType.int32, ["tokens", TOP_K], dev),
            TensorType(DType.uint8, [NUM_EXPERTS, n, k // 2], dev),
            TensorType(
                DType.float8_e4m3fn,
                [NUM_EXPERTS, (n + 127) // 128, k // 64, 32, 4, 4],
                dev,
            ),
            TensorType(DType.float32, [NUM_EXPERTS], dev),
        ],
    ) as graph:
        gx, gexperts, gweight, gblock, gglobal = (
            v.tensor for v in graph.inputs
        )
        order, starts, restore, expert_ids, _, offsets = moe_create_indices(
            ops.reshape(gexperts, [-1]), NUM_EXPERTS, needs_scales_offset=True
        )
        rows = ops.cast(order // TOP_K, DType.int32)
        out = _nvfp4_expert_matmul(
            gx if gather_in_kernel else ops.gather(gx, rows, axis=0),
            gweight,
            gblock,
            gglobal,
            starts,
            offsets,
            expert_ids,
            rows=rows if gather_in_kernel else None,
        )
        graph.output(ops.gather(out, restore, axis=0))

    model = InferenceSession(devices=[device]).load(graph)
    block = np.stack([interleave_nvfp4_scales(s) for s in scales])
    (result,) = model.execute(
        Buffer.from_dlpack(x_bf16).to(device),
        Buffer.from_numpy(experts).to(device),
        Buffer.from_numpy(codes).to(device),
        Buffer.from_numpy(block)
        .view(DType.float8_e4m3fn, block.shape)
        .to(device),
        Buffer.from_numpy(global_scales).to(device),
    )
    assert isinstance(result, Buffer)
    got = torch.from_dlpack(result).float().cpu().numpy()
    got = got.reshape(tokens, TOP_K, n)

    x_ref = x_bf16.float().numpy()
    want = np.empty_like(got)
    for e in np.unique(experts):
        w = _dequantize(codes[e], scales[e], float(global_scales[e]))
        t, j = np.nonzero(experts == e)
        want[t, j] = x_ref[t] @ w.T

    # FP4 activations cost a few percent of the output norm; a wrong layout
    # or scale costs all of it.
    err = np.linalg.norm(got - want, axis=-1) / np.linalg.norm(want, axis=-1)
    assert err.max() < 0.15, f"worst row relative error {err.max():.3f}"
    if n % 128:
        # The rows past the last whole scale granule.
        tail = slice(n // 128 * 128, n)
        tail_err = np.linalg.norm(
            got[..., tail] - want[..., tail]
        ) / np.linalg.norm(want[..., tail])
        assert tail_err < 0.15, f"padded-granule error {tail_err:.3f}"
