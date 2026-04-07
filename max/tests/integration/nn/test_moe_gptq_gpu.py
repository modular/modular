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

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer, DLPackArray
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, Shape, TensorType
from max.graph.quantization import QuantizationConfig
from max.graph.weights import WeightData
from max.nn.moe import MoEGate, MoEGPTQ
from test_common.graph_utils import is_nvidia_gpu
from torch.utils.dlpack import from_dlpack

HIDDEN_DIM = 2048
NUM_EXPERTS = 8
NUM_EXPERTS_PER_TOKEN = 2
MOE_DIM = 768
SEQ_LEN = 2
DESC_ACT = False
GROUP_SIZE = 128


def _weight_data_from_numpy(name: str, value: np.ndarray) -> WeightData:
    return WeightData(
        Buffer.from_numpy(value),
        name,
        DType.from_numpy(value.dtype),
        Shape(value.shape),
    )


def _gptq_scales_shape(in_dim: int, out_dim: int) -> tuple[int, int]:
    return (in_dim // GROUP_SIZE, out_dim)


def _make_gptq_state_dict() -> dict[str, object]:
    rng = np.random.default_rng(123)
    state_dict: dict[str, object] = {
        "gate.gate_score.weight": torch.randn(
            NUM_EXPERTS, HIDDEN_DIM, dtype=torch.bfloat16
        )
        * 0.02
    }

    for expert_idx in range(NUM_EXPERTS):
        prefix = f"experts.{expert_idx}"
        state_dict[f"{prefix}.gate_proj.qweight"] = _weight_data_from_numpy(
            f"{prefix}.gate_proj.qweight",
            rng.integers(
                0,
                np.iinfo(np.uint32).max,
                size=(HIDDEN_DIM // 8, MOE_DIM),
                dtype=np.uint32,
            ),
        )
        state_dict[f"{prefix}.gate_proj.scales"] = _weight_data_from_numpy(
            f"{prefix}.gate_proj.scales",
            rng.standard_normal(
                size=_gptq_scales_shape(HIDDEN_DIM, MOE_DIM)
            ).astype(np.float16),
        )
        if DESC_ACT:
            state_dict[f"{prefix}.gate_proj.perm_idx"] = np.arange(
                HIDDEN_DIM, dtype=np.int32
            )[::-1].copy()

        state_dict[f"{prefix}.up_proj.qweight"] = _weight_data_from_numpy(
            f"{prefix}.up_proj.qweight",
            rng.integers(
                0,
                np.iinfo(np.uint32).max,
                size=(HIDDEN_DIM // 8, MOE_DIM),
                dtype=np.uint32,
            ),
        )
        state_dict[f"{prefix}.up_proj.scales"] = _weight_data_from_numpy(
            f"{prefix}.up_proj.scales",
            rng.standard_normal(
                size=_gptq_scales_shape(HIDDEN_DIM, MOE_DIM)
            ).astype(np.float16),
        )
        if DESC_ACT:
            state_dict[f"{prefix}.up_proj.perm_idx"] = np.arange(
                HIDDEN_DIM, dtype=np.int32
            )[::-1].copy()

        state_dict[f"{prefix}.down_proj.qweight"] = _weight_data_from_numpy(
            f"{prefix}.down_proj.qweight",
            rng.integers(
                0,
                np.iinfo(np.uint32).max,
                size=(MOE_DIM // 8, HIDDEN_DIM),
                dtype=np.uint32,
            ),
        )
        state_dict[f"{prefix}.down_proj.scales"] = _weight_data_from_numpy(
            f"{prefix}.down_proj.scales",
            rng.standard_normal(
                size=_gptq_scales_shape(MOE_DIM, HIDDEN_DIM)
            ).astype(np.float16),
        )
        if DESC_ACT:
            state_dict[f"{prefix}.down_proj.perm_idx"] = np.arange(
                MOE_DIM, dtype=np.int32
            )[::-1].copy()

    return state_dict


@pytest.mark.skipif(not is_nvidia_gpu(), reason="GPTQ MoE requires NVIDIA GPU")
def test_moe_gptq_smoke() -> None:
    quantization_config = QuantizationConfig(
        quant_method="gptq",
        bits=4,
        group_size=GROUP_SIZE,
        desc_act=DESC_ACT,
        sym=True,
    )
    moe = MoEGPTQ(
        devices=[DeviceRef.GPU()],
        hidden_dim=HIDDEN_DIM,
        num_experts=NUM_EXPERTS,
        num_experts_per_token=NUM_EXPERTS_PER_TOKEN,
        moe_dim=MOE_DIM,
        gate_cls=MoEGate,
        dtype=DType.bfloat16,
        quantization_config=quantization_config,
    )
    moe.load_state_dict(
        cast(Mapping[str, DLPackArray | WeightData], _make_gptq_state_dict()),
        strict=True,
    )
    expert = cast(Any, moe.experts[0])
    assert hasattr(expert.gate_proj, "packed_weight_tensor")
    assert not hasattr(expert.gate_proj, "weight")
    assert hasattr(expert.up_proj, "packed_weight_tensor")
    assert not hasattr(expert.up_proj, "weight")
    assert hasattr(expert.down_proj, "packed_weight_tensor")
    assert not hasattr(expert.down_proj, "weight")
    device = Accelerator()
    session = InferenceSession(devices=[device])
    input_type = TensorType(
        DType.bfloat16, [SEQ_LEN, HIDDEN_DIM], device=DeviceRef.GPU()
    )

    with Graph("MoEGPTQ_test", input_types=(input_type,)) as graph:
        x = graph.inputs[0].tensor
        graph.output(moe(x))

    compiled = session.load(graph, weights_registry=moe.state_dict())

    hidden_states = torch.randn(
        SEQ_LEN, HIDDEN_DIM, dtype=torch.bfloat16, device="cuda"
    )
    output = compiled.execute(Buffer.from_dlpack(hidden_states).to(device))[0]
    result = from_dlpack(output)

    assert result.shape == (SEQ_LEN, HIDDEN_DIM)
    assert torch.all(torch.isfinite(result))
