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
"""Measures activation precision in MiMo's production-width expert path.

Uses synthetic MXFP4 weights, fixed routing, and reduced expert count. This
isolates arithmetic; it does not measure the real checkpoint's accuracy.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, Shape, TensorType, ops
from max.graph.weights import WeightData
from max.nn.moe import MoEQuantized
from max.nn.quant_config import QuantConfig
from max.pipelines.architectures.mimo_v2.layers.moe import MiMoV2MoE

HIDDEN, WIDTH, EXPERTS, TOP_K = 4096, 2048, 16, 8
FP4_VALUES = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=torch.float32,
)


def make_weights() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(674)
    weights = {
        "gate.gate_score.weight": torch.zeros(
            EXPERTS, HIDDEN, dtype=torch.bfloat16
        )
    }
    for expert in range(EXPERTS):
        for projection, rows, cols in (
            ("gate_proj", WIDTH, HIDDEN),
            ("up_proj", WIDTH, HIDDEN),
            ("down_proj", HIDDEN, WIDTH),
        ):
            prefix = f"experts.{expert}.{projection}"
            weights[f"{prefix}.weight"] = torch.randint(
                256, (rows, cols // 2), dtype=torch.uint8, generator=generator
            )
            weights[f"{prefix}.weight_scale"] = torch.randint(
                118,
                120,
                (rows, cols // 32),
                dtype=torch.uint8,
                generator=generator,
            )
    return weights


def dequantize(weights: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    packed = weights[f"{prefix}.weight"].cuda()
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    scales = weights[f"{prefix}.weight_scale"].cuda().to(torch.float32)
    scales = torch.exp2(scales - 127).repeat_interleave(32, dim=-1)
    return FP4_VALUES.cuda()[codes.long()] * scales


def reference(
    x: torch.Tensor, routing: torch.Tensor, weights: dict[str, torch.Tensor]
) -> np.ndarray:
    result = torch.empty(
        (x.shape[0], TOP_K, HIDDEN), dtype=torch.float32, device="cuda"
    )
    for expert in range(EXPERTS):
        rows, slots = torch.where(routing == expert)
        if not rows.numel():
            continue
        selected = x[rows].float()
        prefix = f"experts.{expert}"
        gate = selected @ dequantize(weights, f"{prefix}.gate_proj").T
        up = selected @ dequantize(weights, f"{prefix}.up_proj").T
        hidden = torch.nn.functional.silu(gate) * up
        result[rows, slots] = (
            hidden @ dequantize(weights, f"{prefix}.down_proj").T
        )
    return result.flatten(0, 1).cpu().numpy()


def compile_experts(
    session: InferenceSession,
    weights: dict[str, torch.Tensor],
    activation_bits: int,
) -> Model:
    moe_cls = MoEQuantized if activation_bits == 4 else MiMoV2MoE
    moe = moe_cls(
        devices=[DeviceRef.GPU()],
        hidden_dim=HIDDEN,
        num_experts=EXPERTS,
        num_experts_per_token=TOP_K,
        moe_dim=WIDTH,
        dtype=DType.uint8,
        quant_config=QuantConfig.mxfp4(mlp_quantized_layers={0}),
    )
    state: dict[str, torch.Tensor | WeightData] = {}
    for name, value in weights.items():
        if name.endswith("weight_scale"):
            state[name] = WeightData(
                Buffer.from_dlpack(value).view(DType.float8_e8m0fnu),
                name,
                DType.float8_e8m0fnu,
                Shape(value.shape),
            )
        else:
            state[name] = value
    moe.load_state_dict(state, strict=True)
    with Graph(
        f"mimo_experts_a{activation_bits}",
        input_types=(
            TensorType(
                DType.bfloat16, ["tokens", HIDDEN], device=DeviceRef.GPU()
            ),
            TensorType(DType.int64, ["tokens", TOP_K], device=DeviceRef.GPU()),
        ),
    ) as graph:
        down, order = moe._expert_matmuls(
            graph.inputs[0].tensor, ops.reshape(graph.inputs[1].tensor, [-1])
        )
        graph.output(ops.cast(down, DType.float32), order)
    return session.load(graph, weights_registry=moe.state_dict())


def test_expert_activation_precision() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    weights = make_weights()
    device = Accelerator()
    session = InferenceSession(devices=[device])
    models = {bits: compile_experts(session, weights, bits) for bits in (4, 8)}
    reports = []
    for tokens in (1, 11, 17, 65):
        generator = torch.Generator().manual_seed(674 + tokens)
        x = torch.randn(tokens, HIDDEN, generator=generator).to(
            device="cuda", dtype=torch.bfloat16
        )
        routing = (
            torch.arange(tokens * TOP_K, device="cuda").reshape(tokens, TOP_K)
            * 5
        ) % EXPERTS
        expected = reference(x, routing, weights)
        errors = {}
        for bits, model in models.items():
            down, order = model.execute(
                Buffer.from_dlpack(x).to(device),
                Buffer.from_dlpack(routing).to(device),
            )
            actual = down.to(CPU()).to_numpy()[
                order.to(CPU()).to_numpy().astype(np.int64)
            ]
            assert actual.shape == expected.shape
            assert np.isfinite(actual).all()
            errors[bits] = float(
                np.linalg.norm(actual - expected) / np.linalg.norm(expected)
            )
        reports.append({"tokens": tokens, "relative_rmse": errors})
        print(json.dumps(reports[-1]), flush=True)
        assert errors[8] < errors[4]
        assert errors[8] < 0.1
