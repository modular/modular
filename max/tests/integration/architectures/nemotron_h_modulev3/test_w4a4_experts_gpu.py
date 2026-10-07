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
"""Checks the routed experts against a dequantized reference: the W4A4
matmul, and the devices' shares of the experts' channels summed."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, Graph, Shape, TensorType, ops
from max.graph.weights import WeightData
from max.nn.kernels import moe_create_indices
from max.pipelines.architectures.nemotron_h_modulev3.layers.moe import (
    _nvfp4_expert_matmul,
    _routed_experts,
)
from max.pipelines.architectures.nemotron_h_modulev3.quantization import (
    ModuleFormat,
)
from max.pipelines.architectures.nemotron_h_modulev3.weight_adapters import (
    interleave_nvfp4_scales,
    stack_bf16_experts,
    stack_nvfp4_experts,
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


def _weight(array: npt.NDArray[np.generic], dtype: DType) -> WeightData:
    return WeightData(
        data=Buffer.from_numpy(np.ascontiguousarray(array)).view(
            dtype, array.shape
        ),
        name="",
        dtype=dtype,
        shape=Shape(array.shape),
    )


def _shares(weight: WeightData, axis: int | None, n: int) -> list[Buffer]:
    """Splits a stacked weight evenly on ``axis``, as the loader does."""
    host = {DType.bfloat16: DType.uint16, DType.float8_e4m3fn: DType.uint8}
    array = np.from_dlpack(
        weight.to_buffer().view(host.get(weight.dtype, weight.dtype))
    )
    parts = [array] * n if axis is None else np.split(array, n, axis=axis)
    return [
        Buffer.from_numpy(np.ascontiguousarray(part)).view(
            weight.dtype, part.shape
        )
        for part in parts
    ]


@pytest.mark.parametrize("w4a4", [True, False])
def test_device_shares_of_the_routed_experts_sum_to_the_whole(
    w4a4: bool,
) -> None:
    """Two devices' shares of every expert's channels sum to the whole.

    Each device runs every routed row through its 96 of the 192 channels,
    padded with zeros to 128. No token picks expert 3, so its group is
    empty.
    """
    experts, top_k, hidden, inner, tokens, n = 8, 2, 256, 192, 37, 2
    mixer = "backbone.layers.1.mixer"
    shapes = {"up_proj": (inner, hidden), "down_proj": (hidden, inner)}
    rng = np.random.default_rng(0)
    checkpoint: dict[str, WeightData] = {}
    modules: dict[str, ModuleFormat] = {}
    dense: dict[tuple[int, str], npt.NDArray[np.float32]] = {}
    for e in range(experts):
        for proj, (rows, cols) in shapes.items():
            module = f"{mixer}.experts.{e}.{proj}"
            if w4a4:
                codes = rng.integers(0, 256, (rows, cols // 2), dtype=np.uint8)
                scales = rng.integers(
                    0x28, 0x48, (rows, cols // 16), dtype=np.uint8
                )
                global_scale = np.float32(rng.uniform(0.005, 0.02))
                checkpoint[f"{module}.weight"] = _weight(codes, DType.uint8)
                checkpoint[f"{module}.weight_scale"] = _weight(
                    scales, DType.float8_e4m3fn
                )
                checkpoint[f"{module}.weight_scale_2"] = _weight(
                    np.array(global_scale), DType.float32
                )
                modules[module] = ModuleFormat.NVFP4_WEIGHT_ONLY
                dense[e, proj] = _dequantize(codes, scales, float(global_scale))
            else:
                values = torch.from_numpy(
                    rng.standard_normal((rows, cols)) * 0.05
                ).to(torch.bfloat16)
                checkpoint[f"{module}.weight"] = _weight(
                    values.view(torch.int16).numpy().view(np.uint16),
                    DType.bfloat16,
                )
                dense[e, proj] = values.float().numpy()
    stack = stack_nvfp4_experts if w4a4 else stack_bf16_experts
    whole, _ = stack(checkpoint, modules, {mixer})
    split, _ = stack(checkpoint, modules, {mixer}, num_devices=n)

    # The block scales split with the rows of up and the columns of down;
    # the global scales are whole on every device.
    params: dict[str, dict[str, int | None]] = {
        "up": {"weight": 1},
        "down": {"weight": 2},
    }
    if w4a4:
        params["up"] |= {"block_scale": 1, "scale": None}
        params["down"] |= {"block_scale": 2, "scale": None}
    device = Accelerator()

    def on_device(buffer: Buffer) -> Tensor:
        return Tensor(storage=buffer.to(device))

    def projection(
        stacked: dict[str, WeightData], proj: str, shares: int, d: int
    ) -> list[Tensor]:
        return [
            on_device(
                _shares(stacked[f"{mixer}.{proj}_{name}"], axis, shares)[d]
            )
            for name, axis in params[proj].items()
        ]

    ids = np.stack(
        [
            rng.choice([e for e in range(experts) if e != 3], top_k, False)
            for _ in range(tokens)
        ]
    ).astype(np.int32)
    weights = rng.uniform(0.1, 1.0, (tokens, top_k)).astype(np.float32)
    x = torch.from_numpy(rng.standard_normal((tokens, hidden))).to(
        torch.bfloat16
    )
    with F.lazy():
        args = (
            on_device(Buffer.from_dlpack(x)),
            on_device(Buffer.from_numpy(ids)),
            on_device(Buffer.from_numpy(weights)),
        )
        outputs = [
            F.cast(
                _routed_experts(
                    *args,
                    projection(stacked, "up", shares, d),
                    projection(stacked, "down", shares, d),
                ),
                DType.float32,
            )
            for stacked, shares in ((whole, 1), (split, n))
            for d in range(shares)
        ]
    got_whole, *got_shares = (o.to_numpy() for o in outputs)

    x_ref = x.float().numpy()
    want = np.zeros((tokens, hidden), dtype=np.float32)
    for t in range(tokens):
        for e, w in zip(ids[t], weights[t], strict=True):
            up = np.maximum(x_ref[t] @ dense[e, "up_proj"].T, 0) ** 2
            want[t] += w * (up @ dense[e, "down_proj"].T)

    assert all(np.isfinite(o).all() for o in (got_whole, *got_shares))
    # FP4 activations cost a few percent of the output norm per matmul, and
    # the second quantizes relu2's wide-ranging outputs: the whole experts
    # land near 0.2 for the worst token. A wrong share costs all of it.
    tolerance = 0.3 if w4a4 else 0.02
    for name, got in (("whole", got_whole), ("shares", sum(got_shares))):
        err = np.linalg.norm(got - want, axis=-1) / np.linalg.norm(
            want, axis=-1
        )
        assert err.max() < tolerance, (
            f"{name}: worst token relative error {err.max():.3f}"
        )
