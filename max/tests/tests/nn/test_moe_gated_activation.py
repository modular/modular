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
"""Checks that an MoE layer's GatedActivation picks the activation it emits.

Builds the graph only, so it runs on CPU: which activation reaches the graph is
visible in the IR without compiling the GPU kernels.
"""

from __future__ import annotations

import torch
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.nn.moe import GatedActivation, MoE

_HIDDEN_DIM = 64
_MOE_DIM = 32
_FUSED_SILU = 'symbol = "ep.fused_silu"'


def _build_moe_graph(
    activation: GatedActivation | None, interleaved_gate_up: bool = False
) -> str:
    device = DeviceRef.CPU()
    moe = MoE(
        devices=[device],
        hidden_dim=_HIDDEN_DIM,
        num_experts=4,
        num_experts_per_token=2,
        moe_dim=_MOE_DIM,
        activation=activation,
        interleaved_gate_up=interleaved_gate_up,
    )
    # Loading names the weights; the values never reach a kernel.
    moe.load_state_dict(
        {
            name: torch.zeros(
                [int(d) for d in weight.shape], dtype=torch.bfloat16
            )
            for name, weight in moe.raw_state_dict().items()
        }
    )
    with Graph(
        "moe",
        input_types=[
            TensorType(DType.bfloat16, [8, _HIDDEN_DIM], device=device)
        ],
    ) as graph:
        graph.output(moe(graph.inputs[0].tensor))
    return str(graph)


def test_default_fuses_silu() -> None:
    ir = _build_moe_graph(None)
    assert ir.count(_FUSED_SILU) == 1, ir
    assert 'activation = "silu"' in ir, ir
    assert "interleaved = false" in ir, ir


def test_interleaved_layout_reaches_the_kernel() -> None:
    ir = _build_moe_graph(None, interleaved_gate_up=True)
    assert ir.count(_FUSED_SILU) == 1, ir
    assert "interleaved = true" in ir, ir


def test_graph_fn_replaces_the_fused_kernel() -> None:
    calls: list[int] = []

    def gated(gate_up: TensorValue, moe_dim: int) -> TensorValue:
        calls.append(moe_dim)
        return gate_up[:, :moe_dim] * gate_up[:, moe_dim:]

    ir = _build_moe_graph(GatedActivation(gated_activation_fn=gated))
    assert calls == [_MOE_DIM]
    assert _FUSED_SILU not in ir, ir


def test_swigluoai_fuses_with_its_constants() -> None:
    ir = _build_moe_graph(GatedActivation("swigluoai", alpha=1.702, limit=7.0))
    assert ir.count(_FUSED_SILU) == 1, ir
    assert 'activation = "swigluoai"' in ir, ir
    assert "swiglu_alpha = #M.dense_array<1.702000e+00> : tensor<f32>" in ir, ir
    assert "swiglu_limit = #M.dense_array<7.000000e+00> : tensor<f32>" in ir, ir
    assert "mo.sigmoid" not in ir, ir
