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

import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn.moe import MoEGate, StackedMoE
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
