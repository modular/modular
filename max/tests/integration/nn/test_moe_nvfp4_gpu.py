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

"""Minimal smoke test for single-GPU NVFP4 MoEQuantized layer."""

import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, Shape, TensorType
from max.graph.weights import WeightData
from max.nn.moe import MoEQuantized
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from torch.utils.dlpack import from_dlpack

HIDDEN_DIM = 256
MOE_DIM = 128
NUM_EXPERTS = 4
NUM_EXPERTS_PER_TOKEN = 2
SEQ_LEN = 16


def _add_fp4_proj(
    weights: dict[str, torch.Tensor],
    prefix: str,
    out_dim: int,
    in_dim: int,
) -> None:
    weight = torch.randint(0, 256, (out_dim, in_dim // 2), dtype=torch.uint8)
    weight_scale = (
        torch.rand(out_dim, weight.shape[1] // 8, dtype=torch.float32) * 100
    ).to(torch.float8_e4m3fn)
    weights[f"{prefix}.weight"] = weight
    weights[f"{prefix}.weight_scale"] = weight_scale
    weights[f"{prefix}.weight_scale_2"] = (
        torch.rand((), dtype=torch.float32) * 1e-4
    )
    weights[f"{prefix}.input_scale"] = (
        torch.rand((), dtype=torch.float32) * 1e-3
    )


def test_moe_nvfp4_single_gpu() -> None:
    """Verify single-GPU NVFP4 MoEQuantized builds and produces finite output."""
    torch.manual_seed(42)

    fp4_config = QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.STATIC,
            dtype=DType.float32,
            block_size=(1, 16),
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float8_e4m3fn,
            block_size=(1, 16),
        ),
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(),
        embedding_output_dtype=None,
        format=QuantFormat.NVFP4,
    )

    moe = MoEQuantized(
        devices=[DeviceRef.GPU()],
        hidden_dim=HIDDEN_DIM,
        num_experts=NUM_EXPERTS,
        num_experts_per_token=NUM_EXPERTS_PER_TOKEN,
        moe_dim=MOE_DIM,
        dtype=DType.uint8,
        quant_config=fp4_config,
    )

    # Build FP4 weights inline
    raw_weights: dict[str, torch.Tensor] = {}
    raw_weights["gate.gate_score.weight"] = torch.randn(
        NUM_EXPERTS, HIDDEN_DIM, dtype=torch.bfloat16
    )
    for i in range(NUM_EXPERTS):
        _add_fp4_proj(
            raw_weights, f"experts.{i}.gate_proj", MOE_DIM, HIDDEN_DIM
        )
        _add_fp4_proj(raw_weights, f"experts.{i}.up_proj", MOE_DIM, HIDDEN_DIM)
        _add_fp4_proj(
            raw_weights, f"experts.{i}.down_proj", HIDDEN_DIM, MOE_DIM
        )

    # Wrap float8_e4m3fn tensors as WeightData
    wrapped: dict[str, WeightData | torch.Tensor] = {}
    for key, value in raw_weights.items():
        if value.dtype == torch.float8_e4m3fn:
            wrapped[key] = WeightData(
                Buffer.from_dlpack(value.view(torch.uint8)).view(
                    DType.float8_e4m3fn
                ),
                key,
                DType.float8_e4m3fn,
                Shape(value.shape),
            )
        else:
            wrapped[key] = value

    moe.load_state_dict(wrapped)

    device = Accelerator()
    session = InferenceSession(devices=[device])
    input_type = TensorType(
        DType.bfloat16, [SEQ_LEN, HIDDEN_DIM], device=DeviceRef.GPU()
    )

    with Graph("MoEQuantized_NVFP4_test", input_types=(input_type,)) as graph:
        x = graph.inputs[0]
        output = moe(x.tensor)
        graph.output(output)

    compiled = session.load(graph, weights_registry=moe.state_dict())

    hidden_states = torch.randn(
        SEQ_LEN, HIDDEN_DIM, dtype=torch.bfloat16, device="cuda"
    )
    result = compiled.execute(Buffer.from_dlpack(hidden_states).to(device))
    output_tensor = from_dlpack(result[0])

    assert output_tensor.shape == (SEQ_LEN, HIDDEN_DIM)
    assert torch.all(torch.isfinite(output_tensor))
