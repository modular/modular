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

"""Minimal smoke test for StackedMoE layer."""

import pytest
import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, Shape, TensorType
from max.graph.weights import WeightData
from max.nn.moe import MoEGate, StackedMoE
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from test_common.graph_utils import is_nvidia_gpu
from torch.utils.dlpack import from_dlpack

HIDDEN_DIM = 256
NUM_EXPERTS = 4
NUM_EXPERTS_PER_TOKEN = 2
MOE_DIM = 512
SEQ_LEN = 16
DTYPE = DType.bfloat16


def test_stacked_moe_basic() -> None:
    """Verify StackedMoE compiles and produces finite outputs."""
    torch.manual_seed(42)

    moe = StackedMoE(
        devices=[DeviceRef.GPU()],
        hidden_dim=HIDDEN_DIM,
        num_experts=NUM_EXPERTS,
        num_experts_per_token=NUM_EXPERTS_PER_TOKEN,
        moe_dim=MOE_DIM,
        gate_cls=MoEGate,
        dtype=DTYPE,
    )
    moe.load_state_dict(
        {
            "gate.gate_score.weight": torch.randn(
                NUM_EXPERTS, HIDDEN_DIM, dtype=torch.bfloat16
            ),
            "experts.gate_up_proj": torch.randn(
                NUM_EXPERTS, HIDDEN_DIM, 2 * MOE_DIM, dtype=torch.bfloat16
            )
            * 0.02,
            "experts.down_proj": torch.randn(
                NUM_EXPERTS, MOE_DIM, HIDDEN_DIM, dtype=torch.bfloat16
            )
            * 0.02,
        },
        strict=True,
    )

    device = Accelerator()
    session = InferenceSession(devices=[device])
    input_type = TensorType(
        DTYPE, [SEQ_LEN, HIDDEN_DIM], device=DeviceRef.GPU()
    )

    with Graph("StackedMoE_test", input_types=(input_type,)) as graph:
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


def _wrap_float8_weight(
    name: str, value: torch.Tensor, dtype: DType
) -> WeightData:
    return WeightData(
        Buffer.from_dlpack(value.view(torch.uint8)).view(dtype),
        name,
        dtype,
        Shape(value.shape),
    )


@pytest.mark.skipif(not is_nvidia_gpu(), reason="MXFP4 MoE requires NVIDIA GPU")
def test_stacked_moe_mxfp4_basic() -> None:
    """Verify MXFP4 StackedMoE compiles and produces finite outputs."""
    torch.manual_seed(123)

    quant_config = QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.DYNAMIC,
            dtype=DType.float32,
            block_size=(1, 32),
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float8_e8m0fnu,
            block_size=(1, 32),
        ),
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(),
        embedding_output_dtype=DType.bfloat16,
        format=QuantFormat.MXFP4,
        can_use_fused_mlp=False,
    )

    moe = StackedMoE(
        devices=[DeviceRef.GPU()],
        hidden_dim=HIDDEN_DIM,
        num_experts=NUM_EXPERTS,
        num_experts_per_token=NUM_EXPERTS_PER_TOKEN,
        moe_dim=MOE_DIM,
        gate_cls=MoEGate,
        dtype=DTYPE,
        quant_config=quant_config,
    )

    gate_up_weight = torch.randint(
        0,
        256,
        (NUM_EXPERTS, 2 * MOE_DIM, HIDDEN_DIM // 2),
        dtype=torch.uint8,
    )
    down_weight = torch.randint(
        0,
        256,
        (NUM_EXPERTS, HIDDEN_DIM, MOE_DIM // 2),
        dtype=torch.uint8,
    )
    gate_up_scale = torch.rand(
        NUM_EXPERTS,
        2 * MOE_DIM,
        (HIDDEN_DIM + 31) // 32,
        dtype=torch.float32,
    ).to(torch.float8_e8m0fnu)
    down_scale = torch.rand(
        NUM_EXPERTS,
        HIDDEN_DIM,
        (MOE_DIM + 31) // 32,
        dtype=torch.float32,
    ).to(torch.float8_e8m0fnu)

    moe.load_state_dict(
        {
            "gate.gate_score.weight": torch.randn(
                NUM_EXPERTS, HIDDEN_DIM, dtype=torch.bfloat16
            ),
            "experts.gate_up_proj": gate_up_weight,
            "experts.down_proj": down_weight,
            "experts.gate_up_proj_scale": _wrap_float8_weight(
                "experts.gate_up_proj_scale",
                gate_up_scale,
                DType.float8_e8m0fnu,
            ),
            "experts.down_proj_scale": _wrap_float8_weight(
                "experts.down_proj_scale",
                down_scale,
                DType.float8_e8m0fnu,
            ),
        },
        strict=True,
    )

    device = Accelerator()
    session = InferenceSession(devices=[device])
    input_type = TensorType(
        DTYPE, [SEQ_LEN, HIDDEN_DIM], device=DeviceRef.GPU()
    )

    with Graph("StackedMoE_mxfp4_test", input_types=(input_type,)) as graph:
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
